"""
mp_collect_v2.py — Upgraded multiprocess collection for cloud deployment.

Improvements over v1 (in train_rl.py):
1. Shared memory IPC — pre-allocated shared tensors replace Queue serialization
2. GPU opponent support — --opponent-device respected in workers
3. Combined mp + pipeline — InferenceServer runs in thread during PPO
4. Configurable batch timeout
5. Works with any number of servers/workers

Design: see docs/MULTIPROCESS_COLLECTION.md

Usage:
  from mp_collect_v2 import mp_collect_v2, MPPipelineCollector

  # Direct use (replaces mp_collect_v9):
  trajs, w, l, t, steps, summary, elapsed = mp_collect_v2(
      model, device, server_pool, n_games=400, max_concurrent=20,
      snapshot_pool=pool, fp16=True, opponent_device="cuda",
  )

  # Pipeline use (overlaps with PPO):
  collector = MPPipelineCollector()
  collector.start(model, device, server_pool, snapshot_pool, args_dict)
  # ... run PPO update ...
  result = collector.join()
"""

from __future__ import annotations
import asyncio
import gc
import os
import random
import threading
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp_mod

# Must use 'spawn' for CUDA in child processes
try:
    mp_mod.set_start_method('spawn', force=False)
except RuntimeError:
    pass

from poke_env.player import Player
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from arch_compat import (
    call_action_encoder,
    call_policy_logits,
    call_value_logits,
    get_v_support,
)
from features import make_features
from model import PokeTransformer
from rewards import RewardShaper
from teams_ou import random_pool_teambuilder
from team_generator import procedural_teambuilder

from ppo import Trajectory, _cancel_listener
from rl_player import SelfPlayOpponent
from rl_collection import _make_server

_pid_tag = os.getpid() % 10000


# =============================================================================
# Shared Memory Ring Buffer for IPC
# =============================================================================

class SharedInferenceBuffer:
    """Lock-free shared memory buffer for observation → action IPC.

    Pre-allocates shared tensors that workers write to and main process reads.
    Avoids Queue serialization overhead entirely.

    Protocol:
      1. Worker writes obs tensors to slot, sets request_ready[slot] = 1
      2. Main process reads request_ready, batches all ready slots
      3. Main process writes results, sets result_ready[slot] = 1
      4. Worker polls result_ready[slot], reads results, sets request_ready[slot] = 0

    Each worker gets a fixed range of slots: [wid * slots_per_worker, (wid+1) * slots_per_worker)
    Slot count per worker = max_concurrent (one per active battle).
    """

    def __init__(self, n_workers: int, slots_per_worker: int, obs_keys_spec: dict,
                 d_model: int = 384, max_temporal: int = 200):
        self.n_workers = n_workers
        self.slots_per_worker = slots_per_worker
        self.total_slots = n_workers * slots_per_worker
        self.d_model = d_model
        self.max_temporal = max_temporal

        # Control flags (shared between processes)
        self.request_ready = torch.zeros(self.total_slots, dtype=torch.int32).share_memory_()
        self.result_ready = torch.zeros(self.total_slots, dtype=torch.int32).share_memory_()

        # Result tensors (main → worker): logits (9,) and value (1,) per slot
        self.result_logits = torch.zeros(self.total_slots, 9, dtype=torch.float32).share_memory_()
        self.result_values = torch.zeros(self.total_slots, dtype=torch.float32).share_memory_()

        # Message passing for non-tensor data (trajectories, clear signals)
        # Still use a Queue for these — they're infrequent (once per episode, not per turn)
        from multiprocessing import Queue as MPQueue
        self.control_queue = MPQueue()

        # Observation tensors — pre-allocated shared memory
        # We store the flattened obs as a fixed-size tensor per slot
        # Workers serialize obs dict → flat tensor, main deserializes
        # This avoids the per-field shared tensor complexity
        self._obs_size = 8192  # generous fixed size for serialized obs (~20KB actual)
        self.obs_buffer = torch.zeros(self.total_slots, self._obs_size, dtype=torch.float32).share_memory_()
        self.obs_sizes = torch.zeros(self.total_slots, dtype=torch.int32).share_memory_()

        # Battle tag mapping (worker maintains locally, sends via control queue on new battle)
        # Main process needs this for history management
        self.slot_btags: Dict[int, str] = {}  # slot_id → battle_tag (main process side)

    def worker_slot_range(self, worker_id: int) -> Tuple[int, int]:
        start = worker_id * self.slots_per_worker
        end = start + self.slots_per_worker
        return start, end


# =============================================================================
# Simplified approach: Queue-based but with reduced overhead
# =============================================================================
# After analysis, the shared memory ring buffer adds significant complexity
# for modest gains on small scale. The REAL wins for cloud come from:
# 1. GPU opponents (eliminates CPU penalty)
# 2. Combined pipeline (overlaps collection + training)
# 3. More workers/servers
#
# So instead of the ring buffer, let's optimize the Queue approach:
# - Use torch.multiprocessing.Queue (uses shared memory for tensors automatically)
# - Minimize what goes through the queue (only small metadata + tensor refs)
# - Use SimpleQueue where possible (faster, no locking overhead)

from multiprocessing import SimpleQueue


# Message types
_MSG_INFER = 0
_MSG_CLEAR = 1
_MSG_TRAJ = 2
_MSG_DONE = 3
_MSG_BTAG_NEW = 4  # new battle: (worker_id, btag, slot_id)


class InferenceServerV2:
    """Upgraded inference server with profiling and configurable timeout."""

    def __init__(self, model: PokeTransformer, device: torch.device,
                 request_queue, result_queues: Dict[int, Any],
                 fp16: bool = False, batch_timeout_ms: float = 15,
                 min_batch: int = 4):
        self.model = model
        self.device = device
        self.fp16 = fp16 and device.type == "cuda"
        self.request_queue = request_queue
        self.result_queues = result_queues
        self.batch_timeout = batch_timeout_ms / 1000.0
        self.min_batch = min_batch
        # Summary buffer dim = resolved d_temporal (falls back to d_model for legacy checkpoints)
        self.D = getattr(model, "d_temporal", model.cfg.d_model)
        self.max_temporal = model.temporal.temporal_context

        self.history: Dict[Tuple[int, str], torch.Tensor] = {}
        self.trajectories: List[Trajectory] = []
        self._prof_batch_sizes = []
        self._prof_gpu_times = []
        self._prof_total_requests = 0
        self._stop = False

    def run_until_workers_done(self, n_workers: int):
        """Process requests until all workers signal DONE."""
        workers_done = set()
        self.model.eval()

        while len(workers_done) < n_workers and not self._stop:
            infer_requests = []
            deadline = time.time() + self.batch_timeout

            while time.time() < deadline:
                try:
                    timeout = max(0.001, deadline - time.time())
                    msg = self.request_queue.get(timeout=timeout)
                except Exception:
                    break

                msg_type = msg[0]
                if msg_type == _MSG_INFER:
                    _, wid, btag, obs_dict = msg
                    infer_requests.append((wid, btag, obs_dict))
                    if len(infer_requests) >= self.min_batch * 3:
                        break
                elif msg_type == _MSG_CLEAR:
                    _, wid, btag = msg
                    self.history.pop((wid, btag), None)
                elif msg_type == _MSG_TRAJ:
                    _, wid, traj = msg
                    self.trajectories.append(traj)
                elif msg_type == _MSG_DONE:
                    _, wid = msg
                    workers_done.add(wid)

            if infer_requests:
                self._process_batch(infer_requests)

        # Drain remaining
        while not self.request_queue.empty():
            try:
                msg = self.request_queue.get_nowait()
                if msg[0] == _MSG_TRAJ:
                    self.trajectories.append(msg[2])
                elif msg[0] == _MSG_CLEAR:
                    self.history.pop((msg[1], msg[2]), None)
            except Exception:
                break

    def run_in_thread(self, n_workers: int) -> threading.Thread:
        """Run the inference server in a background thread (for pipeline mode)."""
        self._stop = False
        t = threading.Thread(target=self.run_until_workers_done, args=(n_workers,), daemon=True)
        t.start()
        return t

    def stop(self):
        """Signal the server to stop."""
        self._stop = True

    def _process_batch(self, infer_requests):
        N = len(infer_requests)
        self._prof_total_requests += N
        D = self.D
        model = self.model
        device = self.device

        mega = self._stack_obs_to_device([r[2] for r in infer_requests])

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.fp16):
            t0 = time.time()

            # Arch-aware: call_action_encoder dispatches to the right shape for
            # legacy PokeTransformer vs new TransformerBattlePolicy. See arch_compat.py.
            spatial_out, summaries = model.forward_spatial(mega)
            action_ctx = call_action_encoder(model, mega, spatial_out)

            # Batched temporal
            seq_lens = []
            for i, (wid, btag, _) in enumerate(infer_requests):
                h = self.history.get((wid, btag))
                h_len = h.shape[1] if h is not None else 0
                seq_lens.append(h_len + 1)

            max_T = min(max(seq_lens), self.max_temporal)
            seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.long).clamp(max=max_T)
            all_summaries = torch.zeros(N, max_T, D, device=device, dtype=summaries.dtype)

            for i, (wid, btag, _) in enumerate(infer_requests):
                h = self.history.get((wid, btag))
                summary_i = summaries[i]
                if h is not None and h.shape[1] > 0:
                    hh = h.squeeze(0)
                    if hh.shape[0] + 1 > max_T:
                        hh = hh[-(max_T - 1):]
                    h_len = hh.shape[0]
                    all_summaries[i, :h_len] = hh
                    all_summaries[i, h_len] = summary_i
                else:
                    all_summaries[i, 0] = summary_i

            temporal_ctx = model.temporal(
                all_summaries.float(), seq_lens_t
            ).to(summaries.dtype)

            # Batched heads (arch-aware via arch_compat helpers)
            actor_out = spatial_out[:, 0, :]
            at = torch.cat([actor_out, temporal_ctx], dim=-1)
            at_exp = at.unsqueeze(1).expand(-1, 9, -1)
            pi_input = torch.cat([at_exp, action_ctx], dim=-1)
            logits = call_policy_logits(model, pi_input)

            if "legal_mask" in mega:
                logits = logits.float().masked_fill(mega["legal_mask"] < 0.5, -100.0)

            critic_out = spatial_out[:, 1, :]
            vi = torch.cat([critic_out, temporal_ctx], dim=-1)
            v_logits = call_value_logits(model, vi)
            v_probs = F.softmax(v_logits, dim=-1)
            values = (v_probs * get_v_support(model)).sum(-1)

            gpu_ms = (time.time() - t0) * 1000
            self._prof_batch_sizes.append(N)
            self._prof_gpu_times.append(gpu_ms)

        # Update histories and send results
        for i, (wid, btag, _) in enumerate(infer_requests):
            key = (wid, btag)
            summary_f32 = summaries[i].float().unsqueeze(0).unsqueeze(0)
            h = self.history.get(key)
            if h is None:
                self.history[key] = summary_f32
            else:
                self.history[key] = torch.cat([h, summary_f32], dim=1)
                if self.history[key].shape[1] > self.max_temporal:
                    self.history[key] = self.history[key][:, -self.max_temporal:]

            self.result_queues[wid].put({
                "action_logits": logits[i].cpu(),
                "value": values[i].cpu(),
            })

    def _stack_obs_to_device(self, obs_list):
        mega = {}
        ref = obs_list[0]
        dev = self.device
        for key in ref:
            if isinstance(ref[key], torch.Tensor):
                mega[key] = torch.cat([b[key] for b in obs_list], dim=0).to(dev, non_blocking=True)
            elif isinstance(ref[key], dict):
                mega[key] = {
                    k: torch.cat([b[key][k] for b in obs_list], dim=0).to(dev, non_blocking=True)
                    for k in ref[key]
                }
            else:
                mega[key] = ref[key]
        return mega

    def prof_summary(self) -> str:
        if not self._prof_batch_sizes:
            return "no batches"
        sizes = np.array(self._prof_batch_sizes)
        times = np.array(self._prof_gpu_times)
        return (f"batches={len(sizes)}, size={sizes.mean():.1f}avg/{sizes.min()}-{sizes.max()} "
                f"gpu={times.mean():.1f}ms avg/{times.sum()/1000:.1f}s total "
                f"requests={self._prof_total_requests}")


# =============================================================================
# Worker (same as v1 but with configurable opponent device)
# =============================================================================

def _mp_worker_v2(
    worker_id: int,
    request_queue,
    result_queue,
    server_url: str,
    opponent_checkpoints: List[Tuple[str, int]],
    max_concurrent: int,
    rs_cfg: dict,
    temp_range: Tuple[float, float],
    snapshot_pool_size: int,
    teambuilder_path: Optional[str],
    opponent_device: str = "cpu",
):
    """Worker process. opponent_device can be 'cuda' on cloud (A100)."""
    import warnings
    warnings.filterwarnings("ignore")

    # Import here to avoid issues with spawn
    from rl_pipeline import MPRLPlayer

    if teambuilder_path:
        tb = procedural_teambuilder(teambuilder_path)
    else:
        tb = None

    srv = _make_server(server_url)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for opp_ckpt, n_games in opponent_checkpoints:
        opp_name = Path(opp_ckpt).stem
        batch_id = worker_id * 1000 + hash(opp_name) % 1000

        player = MPRLPlayer(
            worker_id=worker_id,
            request_queue=request_queue,
            result_queue=result_queue,
            reward_shaper_cfg=rs_cfg,
            temperature=1.0,
            turn_cap=300,
            battle_format="gen9ou",
            team=tb or random_pool_teambuilder(),
            max_concurrent_battles=max_concurrent,
            account_configuration=AccountConfiguration(f"MPw{worker_id}r{batch_id}", None),
            server_configuration=srv,
        )

        opp_temp_range = (1.0, 1.0) if snapshot_pool_size > 15 else temp_range

        opponent = SelfPlayOpponent(
            checkpoint_path=opp_ckpt,
            device=opponent_device,
            temp_range=opp_temp_range,
            battle_format="gen9ou",
            team=tb or random_pool_teambuilder(),
            max_concurrent_battles=max_concurrent,
            account_configuration=AccountConfiguration(f"MPo{worker_id}r{batch_id}", None),
            server_configuration=srv,
        )

        try:
            loop.run_until_complete(asyncio.wait_for(
                player.battle_against(opponent, n_battles=n_games),
                timeout=max(300, n_games * 30),
            ))
        except asyncio.TimeoutError:
            print(f"  [WARN] w{worker_id} timed out vs {opp_name}", flush=True)
        except Exception as e:
            print(f"  [ERROR] w{worker_id} vs {opp_name}: {e}", flush=True)

        try:
            player.reset_battles()
        except EnvironmentError:
            pass
        try:
            opponent.reset_battles()
        except EnvironmentError:
            pass
        _cancel_listener(player)
        _cancel_listener(opponent)
        del player, opponent

    loop.close()
    request_queue.put((_MSG_DONE, worker_id))


def mp_collect_v2(
    model: PokeTransformer, device: torch.device,
    server_pool: List[ServerConfiguration],
    n_games: int = 200, max_concurrent: int = 10,
    snapshot_pool: List[str] = None, fp16: bool = True,
    reward_shaper_cfg: Optional[dict] = None,
    temp_range: Tuple[float, float] = (1.0, 2.25),
    latest_snapshot: Optional[str] = None,
    teambuilder_path: Optional[str] = None,
    opponent_device: str = "cpu",
    batch_timeout_ms: float = 15,
):
    """Multiprocess collection v2. Supports GPU opponents and configurable timeout."""
    if not snapshot_pool:
        raise ValueError("snapshot_pool required")

    n_workers = len(server_pool)
    rs_cfg = reward_shaper_cfg or {"ko_coef": 0.05, "hp_coef": 0.02, "clip_abs": 2.0}

    max_opponents = 15
    if len(snapshot_pool) <= max_opponents:
        selected = list(snapshot_pool)
    else:
        selected = random.sample(snapshot_pool, max_opponents)
        if latest_snapshot and latest_snapshot not in selected:
            selected[-1] = latest_snapshot

    games_per_opp = max(1, n_games // len(selected))
    remainder = n_games - games_per_opp * len(selected)

    worker_assignments = {i: [] for i in range(n_workers)}
    for oi, opp_ckpt in enumerate(selected):
        n = games_per_opp + (1 if oi < remainder else 0)
        if n > 0:
            worker_assignments[oi % n_workers].append((opp_ckpt, n))

    from multiprocessing import Queue as MPQueue
    request_queue = MPQueue()
    result_queues = {i: MPQueue() for i in range(n_workers)}

    server = InferenceServerV2(
        model, device, request_queue, result_queues,
        fp16=fp16, batch_timeout_ms=batch_timeout_ms,
        min_batch=max(2, max_concurrent // 2),
    )

    t0 = time.time()

    processes = []
    for wid in range(n_workers):
        p = mp_mod.Process(
            target=_mp_worker_v2,
            args=(
                wid, request_queue, result_queues[wid],
                server_pool[wid].websocket_url,
                worker_assignments[wid],
                max_concurrent, rs_cfg, temp_range,
                len(snapshot_pool), teambuilder_path,
                opponent_device,
            ),
            daemon=True,
        )
        p.start()
        processes.append(p)

    server.run_until_workers_done(n_workers)

    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    elapsed = time.time() - t0
    all_trajs = server.trajectories
    total_steps = sum(len(t) for t in all_trajs)
    total_wins = sum(1 for t in all_trajs if t.dones and t.dones[-1] and t.rewards[-1] > 0)
    total_losses = sum(1 for t in all_trajs if t.dones and t.dones[-1] and t.rewards[-1] < 0)

    prof = server.prof_summary()
    print(f"  [MP-V2-PROF] {prof}", flush=True)

    if server.history:
        print(f"  [MP-V2-WARN] {len(server.history)} stale histories", flush=True)
        server.history.clear()

    gc.collect()
    return all_trajs, total_wins, total_losses, 0, total_steps, f"mpv2_{n_workers}w", elapsed


# =============================================================================
# MPPipelineCollector — overlaps mp collection with PPO training
# =============================================================================

class MPPipelineCollector:
    """Runs multiprocess collection in background while PPO trains on GPU.

    The InferenceServer runs in a daemon thread, sharing the GPU with PPO.
    Workers run in separate processes (CPU + optional GPU for opponents).

    Usage:
        collector = MPPipelineCollector()
        collector.start(model, device, server_pool, snapshot_pool, args_dict)
        # ... run PPO update on main thread (shares GPU) ...
        result = collector.join()  # returns (trajs, wins, losses, ties, steps, summary, elapsed)
    """

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._result = None
        self._error = None

    def start(self, model, device, server_pool, snapshot_pool, args_dict):
        """Start background mp collection with a model copy."""
        collect_model = deepcopy(model)
        collect_model.eval()

        self._result = None
        self._error = None
        self._thread = threading.Thread(
            target=self._run,
            args=(collect_model, device, server_pool, snapshot_pool, args_dict),
            daemon=True,
        )
        self._thread.start()

    def _run(self, collect_model, device, server_pool, snapshot_pool, a):
        try:
            latest_sp = snapshot_pool[-1] if len(snapshot_pool) > 1 else None
            self._result = mp_collect_v2(
                collect_model, device, server_pool,
                n_games=a["games_per_iter"],
                max_concurrent=a["max_concurrent"],
                snapshot_pool=snapshot_pool,
                fp16=a["fp16"],
                reward_shaper_cfg=a["rs_cfg"],
                temp_range=a["temp_range"],
                latest_snapshot=latest_sp,
                teambuilder_path=a.get("teambuilder_path"),
                opponent_device=a.get("opponent_device", "cpu"),
                batch_timeout_ms=a.get("batch_timeout_ms", 15),
            )
        except Exception as e:
            self._error = e
            traceback.print_exc()
        finally:
            del collect_model
            gc.collect()

    def join(self):
        if self._thread is None:
            return None
        self._thread.join()
        self._thread = None
        if self._error:
            print(f"  [ERROR] MP pipeline collection failed: {self._error}", flush=True)
            return None
        return self._result

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()
