"""
mp_collect_v3.py — ps-ppo style collection with InferenceServer as a SEPARATE PROCESS.

Architecture (matching ps-ppo's proven pattern):
  Process 1: InferenceServer — owns model copy on GPU, handles all inference requests
  Process 2: Main/Learner — owns training model on GPU, does PPO updates
  Process 3-N: Workers — CPU only, play battles via poke-env websockets

Key difference from v2: InferenceServer is a PROCESS (own CUDA context), not a thread.
CUDA multiplexes two contexts on the same GPU without Python-level contention.
This fixes the all-zero PPO bug from v2's thread-based approach.

Weight sync: after each PPO update, Main sends state_dict to InferenceServer via queue.
InferenceServer uses slightly stale weights during collection (at most 1 iter behind).
This is the same staleness ps-ppo uses — proven safe.

Usage:
  collector = PSPPOCollector(model, device, server_pool, args)
  collector.start()

  for iter in range(n_iters):
      trajs = collector.get_trajectories()  # blocks until ready
      do_ppo_update(model, trajs)
      collector.update_weights(model.state_dict())  # sync to inference server
      collector.start_next()  # begin collecting next iter (overlaps with PPO)

  collector.stop()
"""

from __future__ import annotations
import asyncio
import gc
import os
import random
import time
import traceback
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

try:
    mp.set_start_method('spawn', force=False)
except RuntimeError:
    pass

# See mp_collect_v2.py for rationale (avoid vm.max_map_count exhaustion under
# high-volume tensor IPC). file_system strategy uses ~100x fewer mmaps.
try:
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

from multiprocessing import Queue as MPQueue, Event as MPEvent

from precision_config import autocast_ctx

from poke_env.ps_client.account_configuration import AccountConfiguration

from model import PokeTransformer, PokeTransformerConfig
from rewards import RewardShaper
from teams_ou import random_pool_teambuilder
from team_generator import procedural_teambuilder

from ppo import Trajectory, _cancel_listener
from rl_player import SelfPlayOpponent, make_self_play_opponent
from rl_pipeline import MPRLPlayer
from rl_collection import _make_server

# Message types
_MSG_INFER = 0
_MSG_CLEAR = 1
_MSG_TRAJ = 2
_MSG_DONE = 3


# =============================================================================
# InferenceServer Process
# =============================================================================

def _inference_server_process(
    model_config: dict,
    model_state_dict: dict,
    device_str: str,
    fp16: bool,
    request_queue: MPQueue,
    result_queues: Dict[int, MPQueue],
    weight_queue: MPQueue,
    traj_queue: MPQueue,
    stop_event,
    batch_timeout_ms: float = 15,
    min_batch: int = 4,
):
    """Inference server — runs in its own process with its own CUDA context.

    Handles all neural inference for workers. Manages temporal histories.
    Receives weight updates from main process after each PPO update.
    """
    import warnings
    warnings.filterwarnings("ignore")

    device = torch.device(device_str)
    cfg = PokeTransformerConfig(**model_config)
    model = PokeTransformer(cfg).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    # Summary buffer dim = resolved d_temporal (falls back to d_model for legacy configs)
    D = cfg.d_temporal if cfg.d_temporal is not None else cfg.d_model
    max_temporal = cfg.temporal_context
    batch_timeout = batch_timeout_ms / 1000.0
    history: Dict[Tuple[int, str], torch.Tensor] = {}

    # Profiling
    prof_batch_sizes = []
    prof_gpu_times = []
    prof_total = 0
    weight_version = 0

    while not stop_event.is_set():
        # --- Check for weight updates (non-blocking) ---
        try:
            new_state_dict = weight_queue.get_nowait()
            model.load_state_dict(new_state_dict)
            model.eval()
            weight_version += 1
            # Clear histories on weight update (new model = new representations)
            history.clear()
        except Empty:
            pass

        # --- Drain inference requests ---
        infer_requests = []
        deadline = time.time() + batch_timeout

        while time.time() < deadline:
            try:
                timeout = max(0.001, deadline - time.time())
                msg = request_queue.get(timeout=timeout)
            except Exception:
                break

            msg_type = msg[0]
            if msg_type == _MSG_INFER:
                _, wid, btag, obs_dict = msg
                infer_requests.append((wid, btag, obs_dict))
                if len(infer_requests) >= min_batch * 3:
                    break
            elif msg_type == _MSG_CLEAR:
                _, wid, btag = msg
                history.pop((wid, btag), None)
            elif msg_type == _MSG_TRAJ:
                # Forward trajectory to main process
                traj_queue.put(msg)
            elif msg_type == _MSG_DONE:
                # Forward done signal to main
                traj_queue.put(msg)

        if not infer_requests:
            continue

        # --- Process batch ---
        N = len(infer_requests)
        prof_total += N

        mega = _stack_obs_to_device([r[2] for r in infer_requests], device)

        with torch.no_grad(), autocast_ctx(fp16):
            t0 = time.time()

            spatial_out, summaries = model.forward_spatial(mega)
            action_ctx = model.action_encoder(
                mega["active_move_ids"], mega["active_move_banks"],
                mega["active_move_cont"], mega["switch_ids"], mega["switch_cont"],
            )

            # Batched temporal
            seq_lens = []
            for i, (wid, btag, _) in enumerate(infer_requests):
                h = history.get((wid, btag))
                h_len = h.shape[1] if h is not None else 0
                seq_lens.append(h_len + 1)

            max_T = min(max(seq_lens), max_temporal)
            seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.long).clamp(max=max_T)
            all_summaries = torch.zeros(N, max_T, D, device=device, dtype=summaries.dtype)

            for i, (wid, btag, _) in enumerate(infer_requests):
                h = history.get((wid, btag))
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

            actor_out = spatial_out[:, 0, :]
            at = torch.cat([actor_out, temporal_ctx], dim=-1)
            at_exp = at.unsqueeze(1).expand(-1, 9, -1)
            pi_input = torch.cat([at_exp, action_ctx], dim=-1)
            logits = model.policy_head(pi_input).squeeze(-1)

            if "legal_mask" in mega:
                logits = logits.float().masked_fill(mega["legal_mask"] < 0.5, -100.0)

            critic_out = spatial_out[:, 1, :]
            vi = torch.cat([critic_out, temporal_ctx], dim=-1)
            v_logits = model.value_head(vi)
            v_probs = F.softmax(v_logits, dim=-1)
            values = (v_probs * model.v_support).sum(-1)

            gpu_ms = (time.time() - t0) * 1000
            prof_batch_sizes.append(N)
            prof_gpu_times.append(gpu_ms)

        # Update histories and send results
        for i, (wid, btag, _) in enumerate(infer_requests):
            key = (wid, btag)
            summary_f32 = summaries[i].float().unsqueeze(0).unsqueeze(0)
            h = history.get(key)
            if h is None:
                history[key] = summary_f32
            else:
                history[key] = torch.cat([h, summary_f32], dim=1)
                if history[key].shape[1] > max_temporal:
                    history[key] = history[key][:, -max_temporal:]

            result_queues[wid].put({
                "action_logits": logits[i].cpu(),
                "value": values[i].cpu(),
            })

    # Print profiling on exit
    if prof_batch_sizes:
        sizes = np.array(prof_batch_sizes)
        times = np.array(prof_gpu_times)
        print(f"  [INF-SERVER] weights_v{weight_version}, "
              f"batches={len(sizes)}, size={sizes.mean():.1f}avg/{sizes.min()}-{sizes.max()} "
              f"gpu={times.mean():.1f}ms, total_requests={prof_total}", flush=True)


def _stack_obs_to_device(obs_list, device):
    mega = {}
    ref = obs_list[0]
    for key in ref:
        if isinstance(ref[key], torch.Tensor):
            mega[key] = torch.cat([b[key] for b in obs_list], dim=0).to(device, non_blocking=True)
        elif isinstance(ref[key], dict):
            mega[key] = {
                k: torch.cat([b[key][k] for b in obs_list], dim=0).to(device, non_blocking=True)
                for k in ref[key]
            }
        else:
            mega[key] = ref[key]
    return mega


# =============================================================================
# Worker Process (reuses MPRLPlayer from rl_train_v9)
# =============================================================================

def _mp_worker_v3(
    worker_id: int,
    request_queue: MPQueue,
    result_queue: MPQueue,
    server_url: str,
    opponent_checkpoints: List[Tuple[str, int]],
    max_concurrent: int,
    rs_cfg: dict,
    temp_range: Tuple[float, float],
    snapshot_pool_size: int,
    teambuilder_path: Optional[str],
    opponent_device: str = "cpu",
):
    """Worker process — same as v2 but sends everything to InferenceServer process."""
    import warnings
    warnings.filterwarnings("ignore")

    if teambuilder_path:
        tb = procedural_teambuilder(teambuilder_path)
    else:
        tb = None

    from poke_env.ps_client.server_configuration import ServerConfiguration
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

        opponent = make_self_play_opponent(
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


# =============================================================================
# PSPPOCollector — orchestrates everything
# =============================================================================

class PSPPOCollector:
    """ps-ppo style collector with persistent InferenceServer process.

    The InferenceServer stays alive across iterations. Workers are spawned
    per-iteration. Weights are synced after each PPO update.

    Usage:
        collector = PSPPOCollector(model, device, server_pool, ...)
        collector.start_inference_server()

        for iter in range(...):
            # Start workers for this iteration
            collector.start_collection(snapshot_pool, ...)
            # Optionally do PPO on previous data while collection runs
            if prev_trajs:
                do_ppo(model, prev_trajs)
                collector.update_weights(model.state_dict())
            # Get this iteration's trajectories
            prev_trajs = collector.wait_for_collection()

        collector.stop()
    """

    def __init__(self, model: PokeTransformer, device: torch.device,
                 server_pool, fp16: bool = True,
                 max_concurrent: int = 10,
                 batch_timeout_ms: float = 15,
                 opponent_device: str = "cpu"):
        self.device = device
        self.server_pool = server_pool
        self.fp16 = fp16
        self.max_concurrent = max_concurrent
        self.batch_timeout_ms = batch_timeout_ms
        self.opponent_device = opponent_device
        self.n_workers = len(server_pool)

        # Queues (persistent across iterations)
        self.request_queue = MPQueue()
        self.result_queues = {i: MPQueue() for i in range(self.n_workers)}
        self.weight_queue = MPQueue()
        self.traj_queue = MPQueue()  # InferenceServer forwards trajs here
        self.stop_event = MPEvent()

        # Store model info for inference server
        self._model_config = model.cfg.to_dict()
        self._model_state = {k: v.cpu() for k, v in model.state_dict().items()}

        self._inf_proc = None
        self._worker_procs = []

    def start_inference_server(self):
        """Start the persistent InferenceServer process."""
        self._inf_proc = mp.Process(
            target=_inference_server_process,
            args=(
                self._model_config,
                self._model_state,
                str(self.device),
                self.fp16,
                self.request_queue,
                self.result_queues,
                self.weight_queue,
                self.traj_queue,
                self.stop_event,
                self.batch_timeout_ms,
                max(2, self.max_concurrent // 2),
            ),
            daemon=True,
        )
        self._inf_proc.start()
        print(f"  [PSPPO] InferenceServer started (PID {self._inf_proc.pid})", flush=True)

    def start_collection(self, snapshot_pool: List[str], n_games: int = 200,
                         rs_cfg: dict = None, temp_range=(1.0, 2.25),
                         teambuilder_path: Optional[str] = None):
        """Spawn worker processes for one iteration of collection."""
        rs_cfg = rs_cfg or {"ko_coef": 0.05, "hp_coef": 0.02, "clip_abs": 2.0, "immune_penalty": 0.01}

        # Select opponents
        max_opponents = 10  # S67 2026-05-22: was 15, capped per Phase 2 design
        if len(snapshot_pool) <= max_opponents:
            selected = list(snapshot_pool)
        else:
            selected = random.sample(snapshot_pool, max_opponents)

        games_per_opp = max(1, n_games // len(selected))
        remainder = n_games - games_per_opp * len(selected)

        worker_assignments = {i: [] for i in range(self.n_workers)}
        for oi, opp_ckpt in enumerate(selected):
            n = games_per_opp + (1 if oi < remainder else 0)
            if n > 0:
                worker_assignments[oi % self.n_workers].append((opp_ckpt, n))

        self._worker_procs = []
        for wid in range(self.n_workers):
            p = mp.Process(
                target=_mp_worker_v3,
                args=(
                    wid, self.request_queue, self.result_queues[wid],
                    self.server_pool[wid].websocket_url,
                    worker_assignments[wid],
                    self.max_concurrent, rs_cfg, temp_range,
                    len(snapshot_pool), teambuilder_path,
                    self.opponent_device,
                ),
                daemon=True,
            )
            p.start()
            self._worker_procs.append(p)

    def wait_for_collection(self, timeout: float = 600) -> Tuple:
        """Wait for all workers to finish. Returns (trajs, wins, losses, steps, elapsed)."""
        t0 = time.time()
        workers_done = set()
        trajs = []

        while len(workers_done) < self.n_workers:
            if time.time() - t0 > timeout:
                print(f"  [PSPPO-WARN] Collection timeout after {timeout}s", flush=True)
                break
            try:
                msg = self.traj_queue.get(timeout=1.0)
                if msg[0] == _MSG_TRAJ:
                    trajs.append(msg[2])
                elif msg[0] == _MSG_DONE:
                    workers_done.add(msg[1])
            except Empty:
                pass

        # Wait for worker processes to exit
        for p in self._worker_procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self._worker_procs = []

        # Drain any remaining
        while not self.traj_queue.empty():
            try:
                msg = self.traj_queue.get_nowait()
                if msg[0] == _MSG_TRAJ:
                    trajs.append(msg[2])
            except Exception:
                break

        elapsed = time.time() - t0
        steps = sum(len(t) for t in trajs)
        wins = sum(1 for t in trajs if t.dones and t.dones[-1] and t.rewards[-1] > 0)
        losses = sum(1 for t in trajs if t.dones and t.dones[-1] and t.rewards[-1] < 0)

        return trajs, wins, losses, steps, elapsed

    def update_weights(self, state_dict: dict):
        """Send new weights to InferenceServer after PPO update."""
        cpu_state = {k: v.cpu() for k, v in state_dict.items()}
        self.weight_queue.put(cpu_state)

    def stop(self):
        """Shut down InferenceServer and clean up."""
        self.stop_event.set()
        if self._inf_proc and self._inf_proc.is_alive():
            self._inf_proc.join(timeout=10)
            if self._inf_proc.is_alive():
                self._inf_proc.terminate()
        for p in self._worker_procs:
            if p.is_alive():
                p.terminate()
        print(f"  [PSPPO] Stopped", flush=True)
