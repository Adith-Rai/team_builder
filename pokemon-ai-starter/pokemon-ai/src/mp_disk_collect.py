"""
mp_disk_collect.py — disk-backed multi-process PPO collection for cloud.

Replaces mp_collect_v2.py for the --mp flag. Designed for RunPod A100
80GB and other linux containers where vm.max_map_count is read-only.

Architecture (see docs/MP_DISK_REDESIGN.md for full spec):
- N forkserver-spawned workers, each owning its own GPU model copy
  + own InferenceBatcher + own asyncio loop driving conc=200 battles
- Cross-process IPC is JSON-only (str/dict/int/float). NEVER torch.tensor.
  This is the root-cause fix for the mmap explosion in v2.
- Trajectories written to /tmp/traj_w{id}_iter{N}.pkl.gz at iter end;
  main reads from disk + runs ppo_update unchanged.
- Weight sync: main writes /tmp/weights_iter{N}.pt atomically + signals
  workers to reload via small ctrl msg (path string only, no tensor).

Constraints:
- Cloud-only. Errors out fast on device='cpu'.
- Transformer arch only (validated via is_transformer_checkpoint in worker).
- All training guardrails preserved: forfeit filter (V9RLPlayer
  unchanged), NaN guards (ppo_update unchanged), KL early stop, adaptive
  entropy, EMA win-rate, perm aug, reward clipping, turn cap.

Memory hygiene (Session 50 cont., 2026-05-07):
After Phase 1 v3 cloud run showed iter time growing 41 -> 51 min over 7
warmup iters, an audit (docs/diag/mp_memory_audit.md) found three
patterns present in local code paths but missing from this file. All
three are now applied in `_play_vs_opp`:
  3.5a opp ckpt strip after load   -> _run_collect_in_worker (~line 528)
  3.5b cancel listener + del       -> _run_collect_in_worker (~line 594)
  3.5c per-opp empty_cache         -> _run_collect_in_worker (~line 658)

CRITICAL — fix 3.5b implementation note:
PSClient._listening_coroutine is a concurrent.futures.Future running in
POKE_LOOP (a separate thread), NOT an asyncio.Task. cancel() is
fire-and-forget across threads. rl_collection.py uses asyncio.gather
which gives POKE_LOOP natural yield points; mp_disk_collect runs opps
sequentially, so we MUST `await asyncio.sleep(...)` after the cancel
for POKE_LOOP to drain. First implementation without the sleep caused
worker hangs at iter 10 (POKE_LOOP backed up). ANY new code path here
that creates poke-env Players MUST cancel their listeners AND yield
wall-clock time before creating new ones, or the workers will hang.

Public API:
- mp_disk_collect_sync(...): synchronous one-iter collect. Drop-in for
  collect_v9 in train_rl.py:_collect_data when args.mp is set.
- MPDiskBgCollector: background mode for --mp + --pipeline. Mirrors
  BackgroundCollector's start/join interface.
- shutdown_workers(): clean shutdown at end of run.
"""

from __future__ import annotations

import asyncio
import gc
import gzip
import os
import pickle
import random
import signal
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp_torch

# file_system strategy is defensive only — we don't send tensors through
# queues in this design. Set anyway in case some legacy path leaks one.
try:
    mp_torch.set_sharing_strategy('file_system')
except Exception:
    pass

# Spawn chosen over forkserver after Session 50 testing: forkserver hits a
# CPython 3.11 multiprocessing.resource_tracker race when spawning N>=4
# workers (SemLock files unlinked before children open them →
# FileNotFoundError in SemLock._rebuild). Spawn is slightly slower per
# child (~3-5s for fresh python init) but bypasses the shared
# resource_tracker entirely. With persistent workers across iters, the
# spawn cost is paid ONCE per run (~25-40s for N=8).
try:
    _MP_CTX = mp_torch.get_context('spawn')
except (ValueError, OSError):
    # Last resort fallback (shouldn't happen — spawn is universal)
    _MP_CTX = mp_torch.get_context('fork')


# =============================
# Module-level singleton manager
# =============================
# Single global manager so multiple iters re-use workers (no respawn each iter).

_GLOBAL_MANAGER: Optional["WorkerManager"] = None


def shutdown_workers():
    """Call at end of run / on cleanup. Safe to call multiple times."""
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is not None:
        _GLOBAL_MANAGER.kill_all()
        _GLOBAL_MANAGER = None


# =============================
# Worker manager + health
# =============================

class WorkerManager:
    """Owns the N spawn workers. Tracks heartbeats. Respawns on
    death/hang. Caps consecutive respawns to avoid toxic-state loops.

    Uses mp.Pipe (no SemLock) for ctrl direction and a single shared
    mp.Queue for results. This avoids the CPython 3.11 + RunPod-container
    SemLock unlink race that fires on N>=4 with per-worker mp.Queue."""

    # Heartbeat tolerance. Bumped to 600s in Session 51 cont. after Phase 1 v3
    # iter 17 had ALL 8 workers go stale-heartbeat simultaneously when the
    # PFSP pool grew to 3 ckpts at once (snapshot_0015 just saved). With 8
    # workers each loading a NEW 240MB opp ckpt from disk simultaneously
    # (3.8 GB concurrent reads), the blocking torch.load() call inside
    # _play_vs_opp blocked the asyncio loop for 5+ minutes per worker;
    # heartbeats inside the async coroutine couldn't fire and ALL workers
    # got respawned despite being alive and making progress. Iter 17 ended
    # with 0 trajectories - PPO update FATAL - run hung in shutdown
    # cleanup. 600s tolerance + the new heartbeat-from-liveness-thread fix
    # (see _liveness_loop below) together prevent the catastrophic mass
    # respawn while still detecting actually-dead workers within 10 min.
    HEARTBEAT_TIMEOUT_S = 600.0
    RESPAWN_CAP = 3                 # >this respawns in 5 iters → mark dead
    RESPAWN_WINDOW_ITERS = 5

    def __init__(self, n_workers: int):
        self.n = n_workers
        self.workers: Dict[int, mp_torch.Process] = {}
        # Per-worker pipes both directions: ctrl (parent→worker) + result (worker→parent).
        # All Pipes are FD-based, NO SemLock anywhere → no resource_tracker race
        # at high N. mp.connection.wait() in main multiplexes across all result pipes.
        self.ctrl_pipes: Dict[int, Any] = {}    # parent end of ctrl direction
        self.result_pipes: Dict[int, Any] = {}  # parent end of result direction
        self.last_heartbeat: Dict[int, float] = {}
        self.respawn_history: Dict[int, List[int]] = {}
        self.dead_workers: set = set()

    def spawn(self, worker_id: int):
        """Fresh spawn worker. Communicates exclusively via Pipes (no SemLock).

        Pipe(duplex=False) returns (reader, writer):
        - ctrl direction (parent→worker): parent writes, worker reads
        - result direction (worker→parent): worker writes, parent reads
        """
        # ctrl: parent → worker. parent_end=writer, worker_end=reader.
        ctrl_reader, ctrl_writer = _MP_CTX.Pipe(duplex=False)
        # result: worker → parent. parent_end=reader, worker_end=writer.
        result_reader, result_writer = _MP_CTX.Pipe(duplex=False)
        proc = _MP_CTX.Process(
            target=_worker_main,
            args=(worker_id, ctrl_reader, result_writer),
            daemon=False,
        )
        proc.start()
        # Close worker's ends in parent process (held only by worker)
        ctrl_reader.close()
        result_writer.close()
        self.workers[worker_id] = proc
        self.ctrl_pipes[worker_id] = ctrl_writer       # parent writes here
        self.result_pipes[worker_id] = result_reader   # parent reads here
        self.last_heartbeat[worker_id] = time.time()

    def respawn(self, worker_id: int, iter_n: int) -> bool:
        """Returns True if worker is back; False if it exceeded the respawn cap."""
        if worker_id in self.dead_workers:
            return False

        # Track for cap
        history = self.respawn_history.setdefault(worker_id, [])
        history.append(iter_n)
        # Prune entries older than RESPAWN_WINDOW_ITERS
        history[:] = [it for it in history if it > iter_n - self.RESPAWN_WINDOW_ITERS]
        if len(history) > self.RESPAWN_CAP:
            print(f"[mp-disk] Worker {worker_id} exceeded respawn cap "
                  f"({self.RESPAWN_CAP} respawns in {self.RESPAWN_WINDOW_ITERS} iters); "
                  f"marking dead.", flush=True)
            self.dead_workers.add(worker_id)
            return False

        # Kill if alive
        if worker_id in self.workers:
            p = self.workers[worker_id]
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=2.0)
        # Close the old pipes (worker is dead; nothing else reads them).
        try:
            self.ctrl_pipes[worker_id].close()
        except Exception:
            pass
        try:
            self.result_pipes[worker_id].close()
        except Exception:
            pass

        self.spawn(worker_id)
        print(f"[mp-disk] Worker {worker_id} respawned at iter {iter_n}.", flush=True)
        return True

    def health_check(self) -> List[Tuple[int, str]]:
        """Returns [(worker_id, reason), ...] for workers needing respawn."""
        unhealthy = []
        now = time.time()
        for wid, p in self.workers.items():
            if wid in self.dead_workers:
                continue
            if not p.is_alive():
                unhealthy.append((wid, "dead"))
            elif now - self.last_heartbeat.get(wid, now) > self.HEARTBEAT_TIMEOUT_S:
                unhealthy.append((wid, "stale_heartbeat"))
        return unhealthy

    def alive_worker_ids(self) -> List[int]:
        return [wid for wid in self.workers
                if wid not in self.dead_workers and self.workers[wid].is_alive()]

    def kill_all(self):
        for wid, p in list(self.workers.items()):
            if p.is_alive():
                p.terminate()
        time.sleep(2.0)
        for wid, p in list(self.workers.items()):
            if p.is_alive():
                p.kill()
            p.join(timeout=2.0)
        self.workers.clear()


# =============================
# Worker entry point (forkserver child)
# =============================

def _worker_main(worker_id: int, ctrl_pipe, result_pipe):
    """Spawn child entrypoint.

    Both `ctrl_pipe` and `result_pipe` are child ends of mp.Pipes.
    All IPC is FD-based — NO SemLock anywhere. This sidesteps the CPython
    multiprocessing resource_tracker race at N>=4.

    Worker stays alive across iters; reloads weights on demand. Sends
    heartbeats every 30s, posts done/error per iter to result_pipe.
    """
    import warnings
    warnings.filterwarnings("ignore")

    # Defensive: set sharing strategy in child too. Doesn't matter (we
    # don't send tensors), but harmless.
    try:
        import torch.multiprocessing as _mp_child
        _mp_child.set_sharing_strategy('file_system')
    except Exception:
        pass

    # Local imports — keep init light.
    import torch as _torch
    import logging
    logging.basicConfig(level=logging.WARNING)

    # Worker-local state — survives across iters
    state = {
        "model": None,                   # PokeTransformer (transformer arch only)
        "cfg": None,                     # PokeTransformerConfig
        "device": None,                  # torch.device
        "current_weights_path": None,    # last loaded weights file
        "opp_cache": {},                 # LRU: opp_path -> (loaded_obj, last_used_t)
        "opp_cache_max": 3,
    }

    def _send_heartbeat(iter_n: int, n_done: int, n_total: int):
        try:
            result_pipe.send({
                "status": "heartbeat",
                "worker_id": worker_id,
                "iter_n": iter_n,
                "n_games_done": n_done,
                "n_games_total": n_total,
                "ts": time.time(),
            })
        except (BrokenPipeError, OSError):
            pass

    while True:
        # Wait for next ctrl msg via Pipe (with periodic heartbeat during idle).
        # Pipe.poll(timeout) blocks up to timeout for incoming data; recv()
        # then returns the actual message.
        try:
            if not ctrl_pipe.poll(timeout=30.0):
                _send_heartbeat(iter_n=-1, n_done=0, n_total=0)
                continue
            cmd = ctrl_pipe.recv()
        except (EOFError, BrokenPipeError) as e:
            print(f"[worker {worker_id}] ctrl_pipe closed: {e}", flush=True)
            break
        except Exception as e:
            print(f"[worker {worker_id}] ctrl_pipe.recv failed: {e}", flush=True)
            break

        if cmd.get("cmd") == "shutdown":
            break

        if cmd.get("cmd") != "collect_iter":
            continue

        iter_n = cmd["iter_n"]
        # Immediate ack-heartbeat so main knows we got the cmd; model load
        # below can take 30-60s with N>=4 concurrent disk reads.
        _send_heartbeat(iter_n=iter_n, n_done=0, n_total=cmd.get("n_games", 0))
        try:
            _do_collect_iter(state, worker_id, cmd, result_pipe, _send_heartbeat)
        except Exception as e:
            tb = traceback.format_exc()
            try:
                result_pipe.send({
                    "status": "error",
                    "worker_id": worker_id,
                    "iter_n": iter_n,
                    "exc_type": type(e).__name__,
                    "exc_msg": str(e),
                    "traceback": tb,
                })
            except Exception:
                pass
            # Worker exits on error → manager respawns next iter
            print(f"[worker {worker_id}] iter {iter_n} crashed:\n{tb}", flush=True)
            return


def _do_collect_iter(state, worker_id, cmd, result_pipe, heartbeat_fn):
    """Execute one iter's collect inside a worker process.

    Reuses existing collect_v9 single-process logic — workers ARE
    single-process from their POV. The only worker-side novelty is:
    weight reloading on iter boundary, traj serialization at end.
    """
    iter_n = cmd["iter_n"]
    weights_path = cmd["weights_path"]
    n_games = cmd["n_games"]
    max_concurrent = cmd["max_concurrent"]
    server_url = cmd["server_url"]
    rs_cfg = cmd["rs_cfg"]
    fp16 = cmd.get("fp16", True)
    # bf16/fp16/fp32 selection: each worker is a separate Python interpreter, so
    # the global amp dtype must be set on the worker side too. Main passes a
    # string ("fp16"/"bf16"/"fp32") in the cmd dict; we translate + set here
    # BEFORE any forward call. See precision_config.py docstring.
    try:
        from precision_config import set_amp_dtype, parse_amp_dtype
        set_amp_dtype(parse_amp_dtype(cmd.get("amp_dtype")))
    except Exception:
        pass  # legacy callers without amp_dtype field fall back to fp16 bool
    turn_cap = cmd.get("turn_cap", 300)
    battle_format = cmd.get("battle_format", "gen9ou")
    procedural_teams_path = cmd.get("procedural_teams_path")
    rng_seed = cmd.get("rng_seed", 0)
    opp_pool = cmd["opp_pool"]  # list of dicts: {path, wr, weight}
    temp_range = tuple(cmd.get("temp_range", (1.0, 2.25)))
    opp_temp_range = tuple(cmd.get("opp_temp_range", temp_range))
    opponent_device = cmd.get("opponent_device", "cuda")  # NEW

    # Seed RNG (per-worker offset to vary games across workers)
    random.seed(rng_seed + worker_id * 1000 + iter_n)

    # Load model on first iter or when weights path changes
    if state["model"] is None or state["current_weights_path"] != weights_path:
        from ppo import load_checkpoint
        # load_checkpoint dispatches on arch; will return TransformerBattlePolicy
        # for transformer ckpts. We require transformer arch in --mp.
        device_str = cmd.get("device", "cuda")
        device = torch.device(device_str)
        model, cfg, _ = load_checkpoint(weights_path, device)
        if not _is_transformer_arch(model):
            raise RuntimeError(
                "--mp requires transformer architecture; got legacy ckpt at "
                f"{weights_path}. Use --pipeline (without --mp) for legacy."
            )
        model.eval()
        state["model"] = model
        state["cfg"] = cfg
        state["device"] = device
        state["current_weights_path"] = weights_path

    model = state["model"]
    device = state["device"]

    # Liveness probe thread: prints worker status every 5s INDEPENDENTLY
    # of asyncio loop. If asyncio is dead, this thread keeps printing,
    # which lets us distinguish "asyncio dead" from "worker fully dead".
    #
    # MITIGATION (Session 51 cont., 2026-05-07 iter 17 hang): this thread
    # ALSO fires heartbeats now, not just stdout prints. When workers hit a
    # blocking torch.load() on a new opp ckpt at iter boundary (5+ min in
    # the iter 17 incident), the asyncio-loop heartbeat coroutine can't run.
    # This thread runs in its own OS thread, releases the GIL during sleep,
    # and reaches result_pipe.send (which is thread-safe via internal lock).
    # So heartbeats fire even when asyncio is fully blocked. The async
    # heartbeat coroutine still fires too; this thread is the safety net.
    import threading
    _liveness_state = {"alive": True, "n_done": 0, "n_total": n_games}
    def _liveness_loop():
        i = 0
        while _liveness_state["alive"]:
            try:
                print(f"[w{worker_id} LIVE +{i*5}s iter={iter_n}] "
                      f"n_done={_liveness_state['n_done']}/{_liveness_state['n_total']}",
                      flush=True)
                # Heartbeat from THIS THREAD - decoupled from asyncio.
                # Pipe.send is thread-safe in CPython (internal lock).
                heartbeat_fn(iter_n=iter_n,
                             n_done=_liveness_state["n_done"],
                             n_total=_liveness_state["n_total"])
            except Exception:
                pass
            i += 1
            time.sleep(5.0)
    _liveness_thread = threading.Thread(target=_liveness_loop, daemon=True)
    _liveness_thread.start()

    # Run collection. Reuse collect_v9 single-process via a slim adapter
    # that builds the local server pool + InferenceBatcher.
    heartbeat_fn(iter_n=iter_n, n_done=0, n_total=n_games)
    t0 = time.time()
    trajs, w, l, ties, summary, wr_per_opp, n_fft_w, n_fft_l = _run_collect_in_worker(
        model=model,
        device=device,
        worker_id=worker_id,
        iter_n=iter_n,
        n_games=n_games,
        max_concurrent=max_concurrent,
        server_url=server_url,
        opp_pool=opp_pool,
        temp_range=temp_range,
        opp_temp_range=opp_temp_range,
        fp16=fp16,
        rs_cfg=rs_cfg,
        turn_cap=turn_cap,
        battle_format=battle_format,
        procedural_teams_path=procedural_teams_path,
        heartbeat_fn=heartbeat_fn,
        opp_cache=state["opp_cache"],
        opp_cache_max=state["opp_cache_max"],
        liveness_state=_liveness_state,
        opponent_device=opponent_device,
    )
    elapsed_s = time.time() - t0
    _liveness_state["alive"] = False  # stop probe thread

    # Write traj file atomically.
    traj_path = f"/tmp/traj_w{worker_id}_iter{iter_n}.pkl.gz"
    tmp_path = traj_path + ".tmp"
    bundle = {
        "trajectories": trajs,
        "iter_n": iter_n,
        "worker_id": worker_id,
        "n_games": w + l + ties,
        "wr_per_opp": wr_per_opp,
        "elapsed_s": elapsed_s,
    }
    with gzip.open(tmp_path, "wb", compresslevel=1) as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, traj_path)

    # Post done.
    result_pipe.send({
        "status": "done",
        "worker_id": worker_id,
        "iter_n": iter_n,
        "traj_path": traj_path,
        "n_games_played": w + l + ties,
        "wins": w,
        "losses": l,
        "ties": ties,
        "n_forfeit_wins": n_fft_w,
        "n_forfeit_losses": n_fft_l,
        "wr_per_opp": wr_per_opp,
        "elapsed_s": elapsed_s,
    })


def _is_transformer_arch(model) -> bool:
    """Same discriminator as ppo.is_transformer_checkpoint, but on a model."""
    return (hasattr(model, "tokenizer") and
            hasattr(model, "_per_action_context") and
            hasattr(model, "action_head"))


def _run_collect_in_worker(*, model, device, worker_id, iter_n, n_games,
                            max_concurrent, server_url, opp_pool,
                            temp_range, opp_temp_range, fp16, rs_cfg,
                            turn_cap, battle_format, procedural_teams_path,
                            heartbeat_fn, opp_cache, opp_cache_max,
                            liveness_state=None, opponent_device="cuda"):
    """Single-process collect inside a worker. Picks one PFSP-sampled
    opponent per game (weighted by `weight` from main's PFSP calc),
    runs n_games battles via V9RLPlayer.battle_against. Aggregates
    trajectories + W/L per opp.
    """
    from poke_env.ps_client.account_configuration import AccountConfiguration
    from poke_env.ps_client.server_configuration import ServerConfiguration
    from inference_batcher import InferenceBatcher
    from rl_player import V9RLPlayer, make_self_play_opponent
    from rl_collection import _make_server
    from teams_ou import random_pool_teambuilder
    from team_generator import procedural_teambuilder

    srv = _make_server(server_url)

    # PFSP weights → sample opponent assignments.
    weights_arr = [max(o.get("weight", 1.0), 1e-6) for o in opp_pool]
    total_w = sum(weights_arr)
    fractions = [w_ / total_w for w_ in weights_arr]
    n_per_opp = [max(1, int(round(n_games * fr))) for fr in fractions]
    # Adjust rounding
    diff = n_games - sum(n_per_opp)
    if diff != 0:
        n_per_opp[0] += diff

    # Build teambuilder once (shared across opps in this worker).
    if procedural_teams_path:
        train_tb = procedural_teambuilder(procedural_teams_path)
    else:
        train_tb = None

    # Local InferenceBatcher serves this worker's V9RLPlayer.
    batcher = InferenceBatcher(
        model, device, fp16=fp16,
        min_batch=min(8, max_concurrent),
        timeout_ms=15,
    )

    all_trajs = []
    total_w_count = 0
    total_l = 0
    total_ties = 0
    total_fft_w = 0
    total_fft_l = 0
    wr_per_opp: Dict[str, Dict[str, int]] = {}

    n_done = 0

    async def _play_vs_opp(opp_entry: dict, n_for_opp: int):
        nonlocal n_done, total_w_count, total_l, total_ties, total_fft_w, total_fft_l
        opp_path = opp_entry["path"]
        batch_id = (worker_id * 100000) + (iter_n * 1000) + (hash(opp_path) % 1000)

        opp_tb = train_tb or random_pool_teambuilder()
        player = V9RLPlayer(
            batcher=batcher, device=device,
            reward_shaper_cfg=rs_cfg,
            temperature=1.0,
            turn_cap=turn_cap,
            battle_format=battle_format,
            team=train_tb or random_pool_teambuilder(),
            max_concurrent_battles=min(max_concurrent, n_for_opp),
            account_configuration=AccountConfiguration(
                f"MPDw{worker_id}r{batch_id}", None),
            server_configuration=srv,
        )

        # LRU opp ckpt cache: keep last opp_cache_max ckpts loaded
        if opp_path in opp_cache:
            opp_cache[opp_path]["last_used"] = time.time()
            cached_ckpt = opp_cache[opp_path]["ckpt"]
        else:
            # Evict LRU if over cap
            if len(opp_cache) >= opp_cache_max:
                lru = min(opp_cache.items(), key=lambda kv: kv[1]["last_used"])[0]
                del opp_cache[lru]
            # Load opp ckpt onto opponent_device (cpu reduces GPU contention
            # when N workers + main all share one A100).
            #
            # MEMORY HYGIENE (Session 50 audit, docs/diag/mp_memory_audit.md
            # fix #3.5a): the saved ckpt dict contains AdamW optimizer state
            # (~480 MB for our 240 MB transformer), scheduler state, snapshot
            # pool list, and other training metadata that the opponent does
            # not need at inference. Strip to the three fields read by
            # BattleAgentTransformer.__init__ (model_state_dict, model_config,
            # arch) BEFORE caching. Pattern mirrors eval_elo_ladder.py:201-216
            # (load_ckpt_cached). Saves ~480 MB per cached ckpt ×
            # opp_cache_max=3 × N workers (~11 GB at N=8). Without this,
            # opp_cache pins optimizer state across iters until LRU eviction.
            full = torch.load(opp_path, map_location=opponent_device, weights_only=False)
            cached_ckpt = {
                "model_state_dict": full["model_state_dict"],
                "model_config": full.get("model_config", {}),
                "arch": full.get("arch", "transformer"),
            }
            del full
            gc.collect()
            opp_cache[opp_path] = {"ckpt": cached_ckpt, "last_used": time.time()}

        opponent = make_self_play_opponent(
            checkpoint_path=opp_path,
            device=opponent_device,    # use opp_device, not worker's main device
            temp_range=opp_temp_range,
            battle_format=battle_format,
            team=opp_tb,
            max_concurrent_battles=min(max_concurrent, n_for_opp),
            account_configuration=AccountConfiguration(
                f"MPDo{worker_id}r{batch_id}", None),
            server_configuration=srv,
            _cached_ckpt=cached_ckpt,
        )

        try:
            await asyncio.wait_for(
                player.battle_against(opponent, n_battles=n_for_opp),
                timeout=max(300, n_for_opp * 30),
            )
        except asyncio.TimeoutError:
            print(f"[worker {worker_id}] timeout vs {Path(opp_path).stem}", flush=True)
        except Exception as e:
            print(f"[worker {worker_id}] error vs {Path(opp_path).stem}: {e}", flush=True)

        # Pull trajectories + counts off player.
        opp_w = sum(1 for b in player.battles.values() if b.won is True
                    and player._finish_looks_real(b))
        opp_l = sum(1 for b in player.battles.values() if b.won is False
                    and player._finish_looks_real(b))
        opp_t = sum(1 for b in player.battles.values() if b.won is None)

        # Forfeit-finish accounting: matches V9RLPlayer's logic
        opp_fft_w = getattr(player, 'n_forfeit_wins', 0)
        opp_fft_l = getattr(player, 'n_forfeit_losses', 0)

        all_trajs.extend(player.completed_trajectories)
        total_w_count += opp_w
        total_l += opp_l
        total_ties += opp_t
        total_fft_w += opp_fft_w
        total_fft_l += opp_fft_l

        wr_per_opp[opp_path] = {
            "w": opp_w, "g": opp_w + opp_l,
            "fft_w": opp_fft_w, "fft_l": opp_fft_l,
        }
        n_done += n_for_opp
        if liveness_state is not None:
            liveness_state["n_done"] = n_done

        try:
            player.reset_battles()
            opponent.reset_battles()
        except EnvironmentError:
            pass

        # MEMORY HYGIENE (Session 50 audit fix #3.5b, REVISED 2026-05-07):
        # Cancel poke-env websocket listeners + del player/opponent so
        # asyncio loop doesn't pin them via the still-running listener
        # task (without this, 40-100 stale tasks accumulate over 7
        # iters -> asyncio scheduler tax = the 41->51 min iter-time
        # creep observed in Phase 1 v3 warmup).
        #
        # CRITICAL: PSClient._listening_coroutine is a
        # concurrent.futures.Future running in POKE_LOOP (a separate
        # thread), NOT an asyncio.Task. cancel() is fire-and-forget
        # across threads. The first attempt at this fix (commit
        # 997fa32 / 2026-05-07 ~05:00 UTC) cancelled and immediately
        # started the next matchup -> POKE_LOOP backed up on async
        # listener cleanup + websocket.close() while we were creating
        # new ws connections for the next opp -> worker hung at 99%
        # CPU at iter 10 of post-snapshot smoke. rl_collection.py uses
        # the same cancel pattern but in asyncio.gather (parallel) which
        # provides natural yield points; mp_disk_collect runs opps
        # sequentially, so we MUST manually yield wall-clock time after
        # each cancel for POKE_LOOP to drain. The 1.5s sleep is
        # negligible overhead (~15s/iter total at N=8 workers) vs the
        # ~$30-50 leak cost it prevents.
        # Mirrors ppo.py:_cancel_listener helper used by
        # rl_collection.py:458-463 and eval_elo_ladder.py:320-327.
        try:
            from ppo import _cancel_listener
            _cancel_listener(player)
            _cancel_listener(opponent)
        except Exception:
            pass
        del player, opponent

        # Wall-clock yield: lets POKE_LOOP (another thread) finish
        # propagating the cancellation + closing the underlying
        # websocket before the next matchup creates a new connection.
        # Without this, sequential per-matchup pattern overwhelms
        # POKE_LOOP and worker hangs.
        await asyncio.sleep(1.5)

        # MEMORY HYGIENE (Session 50 audit fix #3.5c): per-opp gc +
        # cuda empty_cache reduces cudaMalloc fragmentation that
        # accumulates across 6-15 opp matchups per iter. Placed AFTER
        # the asyncio.sleep so any objects whose cleanup is pending in
        # POKE_LOOP have settled before we GC. Mirrors
        # eval_diag.py:101-102 and eval_elo_ladder.py:488-490.
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    async def _heartbeat_during_collect():
        # Periodic in-worker status: how many battles active/finished. Helps
        # diagnose pipeline+mp hangs (vs slow forward vs zero progress).
        loop_counter = 0
        while True:
            heartbeat_fn(iter_n=iter_n, n_done=n_done, n_total=n_games)
            # Probe player(s) created so far for live battle counts
            try:
                # Workers create players inside _play_vs_opp; we can't reach them
                # here. Instead, log prof state of the InferenceBatcher.
                if hasattr(batcher, '_prof_total_requests'):
                    print(f"[worker {worker_id} t+{loop_counter*15}s] "
                          f"infer_reqs={batcher._prof_total_requests} "
                          f"n_done={n_done}/{n_games}", flush=True)
            except Exception:
                pass
            loop_counter += 1
            await asyncio.sleep(15.0)

    async def _main():
        hb_task = asyncio.create_task(_heartbeat_during_collect())
        try:
            opp_coros = []
            for opp_entry, n_for in zip(opp_pool, n_per_opp):
                if n_for <= 0:
                    continue
                opp_coros.append(_play_vs_opp(opp_entry, n_for))
            print(f"[worker {worker_id}] gather start: {len(opp_coros)} opp coros, "
                  f"n_per_opp={n_per_opp}, opp_pool={[o['path'] for o in opp_pool]}",
                  flush=True)
            results = await asyncio.gather(*opp_coros, return_exceptions=True)
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    import traceback as _tb
                    print(f"[worker {worker_id}] opp {i} EXCEPTION: "
                          f"{type(r).__name__}: {r}\n"
                          f"{''.join(_tb.format_exception(type(r), r, r.__traceback__))}",
                          flush=True)
        finally:
            hb_task.cancel()
            try:
                await hb_task
            except asyncio.CancelledError:
                pass

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_main())
    finally:
        loop.close()

    # Free batcher
    del batcher
    gc.collect()
    torch.cuda.empty_cache() if device.type == "cuda" else None

    summary = {"worker_id": worker_id, "iter_n": iter_n}
    return (all_trajs, total_w_count, total_l, total_ties, summary,
            wr_per_opp, total_fft_w, total_fft_l)


# =============================
# Main-side public API
# =============================

def _save_worker_weights_atomic(model, iter_n: int) -> str:
    """Atomic write of worker-only weights file. Returns path."""
    final = f"/tmp/weights_iter{iter_n}.pt"
    tmp = final + ".tmp"
    cfg_dict = model.cfg.to_dict() if hasattr(model.cfg, 'to_dict') else None
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": cfg_dict,
        "arch": "transformer",  # mp is transformer-only
    }, tmp)
    # Best-effort fsync — make sure rename sees a complete file
    try:
        with open(tmp, 'rb') as f:
            os.fsync(f.fileno())
    except Exception:
        pass
    os.replace(tmp, final)
    return final


def _cleanup_old_files(iter_n: int, keep_last: int = 2):
    """Remove weights/traj files older than (iter_n - keep_last)."""
    threshold = iter_n - keep_last
    for f in Path("/tmp").glob("weights_iter*.pt"):
        try:
            n = int(str(f.stem).replace("weights_iter", ""))
            if n < threshold:
                f.unlink()
        except (ValueError, OSError):
            pass
    for f in Path("/tmp").glob("traj_w*_iter*.pkl.gz"):
        try:
            stem = f.name.replace(".pkl.gz", "")
            n = int(stem.split("_iter")[-1])
            if n < threshold:
                f.unlink()
        except (ValueError, OSError):
            pass


def _build_opp_pool_msg(snapshot_pool, win_rates) -> List[dict]:
    """Convert pool + win_rates into the JSON-safe opp_pool list.
    Each entry: {path, wr, weight} where weight = (1-wr)^2 (PFSP)."""
    out = []
    for path in snapshot_pool:
        wr_entry = (win_rates or {}).get(path, {})
        if isinstance(wr_entry, dict):
            w = wr_entry.get("w", 0)
            g = max(wr_entry.get("g", 0), 1)
            wr = w / g
        else:
            wr = 0.5  # neutral if no record
        weight = (1.0 - wr) ** 2
        out.append({"path": path, "wr": wr, "weight": weight})
    return out


def mp_disk_collect_sync(
    model, device, server_pool, *,
    n_games: int,
    max_concurrent: int,
    snapshot_pool: List[str],
    fp16: bool,
    reward_shaper_cfg: dict,
    temp_range: Tuple[float, float],
    opponent_device: str,
    win_rates: Optional[dict],
    turn_cap: int,
    battle_format: str,
    procedural_teams_path: Optional[str],
    iter_n: int,
    n_workers: int = 8,
    rng_seed: Optional[int] = None,
    amp_dtype: Optional[str] = None,  # "fp16"|"bf16"|"fp32"|None
) -> Tuple[List, int, int, int, Dict, float, dict]:
    """Synchronous one-iter collect using N workers + disk traj.

    Drop-in replacement for collect_v9() when args.mp is set (no pipeline).
    Returns: (trajs, wins, losses, ties, summary, elapsed, opp_records)
    where opp_records aggregates wins/games per opp for PFSP wr update.
    """
    if device.type == "cpu":
        raise ValueError("--mp not supported on CPU; use --pipeline or sync collect.")
    if rng_seed is None:
        rng_seed = random.randint(0, 1_000_000)

    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = WorkerManager(n_workers=n_workers)
        # Pace the spawns: at N>=4, simultaneous forkserver spawns race the
        # resource_tracker → SemLock._rebuild FileNotFoundError in children.
        # 0.5s between spawns avoids the race; total startup overhead is
        # ~4-8s for N=8 workers, amortized over the whole run.
        for wid in range(n_workers):
            _GLOBAL_MANAGER.spawn(wid)
            time.sleep(0.5)
        time.sleep(2.0)

    t_start = time.time()

    # 1. Save weights for workers to load
    weights_path = _save_worker_weights_atomic(model, iter_n)

    # 2. Build opp_pool message (PFSP-weighted)
    opp_pool_msg = _build_opp_pool_msg(snapshot_pool, win_rates)

    # 3. Distribute work across alive workers
    alive = _GLOBAL_MANAGER.alive_worker_ids()
    if not alive:
        # All workers dead — try respawn
        for wid in range(n_workers):
            _GLOBAL_MANAGER.respawn(wid, iter_n)
        alive = _GLOBAL_MANAGER.alive_worker_ids()
    if not alive:
        raise RuntimeError("[mp-disk] All workers dead, cannot proceed.")

    # Round-robin server assignment per worker
    n_alive = len(alive)
    games_per_worker = n_games // n_alive
    remainder = n_games % n_alive
    cmds_sent = []
    for i, wid in enumerate(alive):
        worker_n = games_per_worker + (1 if i < remainder else 0)
        srv_idx = i % len(server_pool)
        # Reconstruct server URL from ServerConfiguration object
        srv_obj = server_pool[srv_idx]
        srv_url = getattr(srv_obj, 'websocket_url', None) or str(srv_obj)
        cmd = {
            "cmd": "collect_iter",
            "iter_n": iter_n,
            "weights_path": weights_path,
            "n_games": worker_n,
            "max_concurrent": max_concurrent,
            "server_url": srv_url,
            "opp_pool": opp_pool_msg,
            "temp_range": list(temp_range),
            "opp_temp_range": list(temp_range),
            "fp16": fp16,
            "rs_cfg": reward_shaper_cfg,
            "turn_cap": turn_cap,
            "battle_format": battle_format,
            "procedural_teams_path": procedural_teams_path,
            "device": str(device),
            "opponent_device": opponent_device,  # workers load opp ckpt on this device
            "rng_seed": rng_seed,
            "amp_dtype": amp_dtype,  # picked up by worker via precision_config.set_amp_dtype
        }
        # Pipe.send is blocking-but-fast (no timeout API); fine for small msgs.
        # MITIGATION (Session 51 cont.): stagger by 0.25s per worker so that
        # all 8 workers don't hit disk simultaneously when loading new opp
        # ckpts at iter boundaries. Without staggering, 8 concurrent 240MB
        # disk reads + page-cache contention can stall workers for 5+ min,
        # which triggered the iter 17 mass-stale-heartbeat hang. Total added
        # latency: ~2s for N=8 workers, negligible vs ~16 min/iter collect.
        _GLOBAL_MANAGER.ctrl_pipes[wid].send(cmd)
        cmds_sent.append(wid)
        if i + 1 < len(alive):
            time.sleep(0.25)

    # 4. Watchdog loop: multiplex result_pipes via mp.connection.wait.
    # All Pipes are FD-backed; wait() does select/poll on the FDs directly.
    from multiprocessing.connection import wait as mp_wait
    expected_responses = len(cmds_sent)
    received_responses = 0
    results: List[dict] = []
    expected_collect_s = max(300.0, n_games / max(n_alive, 1) * 2.0)
    deadline = time.time() + 4.0 * expected_collect_s

    # Map from connection object back to worker_id for fast lookup
    pipes_to_wid = {_GLOBAL_MANAGER.result_pipes[wid]: wid for wid in cmds_sent}

    while received_responses < expected_responses and time.time() < deadline:
        ready = mp_wait(list(pipes_to_wid.keys()), timeout=10.0)
        if not ready:
            # Health check on idle tick
            unhealthy = _GLOBAL_MANAGER.health_check()
            for wid, reason in unhealthy:
                if wid in cmds_sent:
                    print(f"[mp-disk] iter {iter_n}: worker {wid} {reason}; "
                          f"respawning, dropping its slice.", flush=True)
                    _GLOBAL_MANAGER.respawn(wid, iter_n)
                    expected_responses -= 1
                    # Old pipe is invalidated; replace map entry
                    pipes_to_wid = {_GLOBAL_MANAGER.result_pipes[w]: w
                                     for w in cmds_sent
                                     if w not in _GLOBAL_MANAGER.dead_workers}
            continue

        for conn in ready:
            try:
                msg = conn.recv()
            except (EOFError, OSError) as e:
                # Worker closed pipe (likely crashed). Respawn.
                wid = pipes_to_wid.get(conn)
                print(f"[mp-disk] iter {iter_n}: worker {wid} pipe closed: {e}; "
                      f"respawning.", flush=True)
                if wid is not None and wid in cmds_sent:
                    _GLOBAL_MANAGER.respawn(wid, iter_n)
                    expected_responses -= 1
                continue

            wid = msg.get("worker_id")
            status = msg.get("status")

            if status == "heartbeat":
                _GLOBAL_MANAGER.last_heartbeat[wid] = msg.get("ts", time.time())
                continue

            received_responses += 1

            if status == "done":
                if msg.get("iter_n") != iter_n:
                    print(f"[mp-disk] WARN: stale msg from worker {wid} for "
                          f"iter {msg.get('iter_n')} (current={iter_n}); ignoring.",
                          flush=True)
                    continue
                results.append(msg)
            elif status == "error":
                print(f"[mp-disk] iter {iter_n}: worker {wid} error: "
                      f"{msg.get('exc_msg')}\n{msg.get('traceback', '')}", flush=True)
                _GLOBAL_MANAGER.respawn(wid, iter_n)

    # If we hit the deadline with stragglers, log + continue with what we have
    if received_responses < expected_responses:
        print(f"[mp-disk] iter {iter_n}: deadline hit. "
              f"{received_responses}/{expected_responses} workers responded; "
              f"continuing with partial sample.", flush=True)
        # Respawn any workers that didn't respond
        unresponsive = set(cmds_sent) - {r["worker_id"] for r in results}
        for wid in unresponsive:
            _GLOBAL_MANAGER.respawn(wid, iter_n)

    # 5. Aggregate: read traj files, sum stats
    # opp_records format MUST match collect_v9's: {path: [wins, games]} (2-list).
    # Forfeit counters tracked separately in main if needed.
    all_trajs = []
    total_w = 0
    total_l = 0
    total_ties = 0
    aggregated_wr: Dict[str, List[int]] = {}
    for r in results:
        traj_path = r.get("traj_path")
        if not traj_path or not Path(traj_path).exists():
            print(f"[mp-disk] worker {r['worker_id']}: traj file missing "
                  f"({traj_path}); dropping slice.", flush=True)
            continue
        try:
            with gzip.open(traj_path, "rb") as f:
                bundle = pickle.load(f)
        except Exception as e:
            print(f"[mp-disk] failed to read {traj_path}: {e}", flush=True)
            continue

        all_trajs.extend(bundle["trajectories"])
        total_w += r["wins"]
        total_l += r["losses"]
        total_ties += r["ties"]
        for opp_path, stats in r.get("wr_per_opp", {}).items():
            rec = aggregated_wr.setdefault(opp_path, [0, 0])
            rec[0] += stats.get("w", 0)
            rec[1] += stats.get("g", 0)

    # 6. Cleanup old files (keep last 2)
    _cleanup_old_files(iter_n, keep_last=2)

    elapsed = time.time() - t_start
    total_steps = sum(len(t) for t in all_trajs)
    opp_name = f"mp-disk(N={len(alive)},responded={len(results)})"
    # Return signature matches collect_v9: (trajs, w, l, t, steps, opp_name,
    # collect_time, opp_records).
    return (all_trajs, total_w, total_l, total_ties, total_steps, opp_name,
            elapsed, aggregated_wr)


# =============================
# Background mode (--mp + --pipeline)
# =============================

class MPDiskBgCollector:
    """Background-mode mp-disk collect. Mirrors BackgroundCollector
    interface (start, join, running) so train_rl.py's pipeline path
    can use it as drop-in.

    Semantics: workers always use the LAST FINALIZED weights at iter
    boundary. That means iter K's collect uses weights from iter K-1's
    update (off-by-1 stale, same as BackgroundCollector).
    """

    def __init__(self):
        self._kicked_off = False
        self._iter_n: Optional[int] = None
        self._collect_args: Optional[dict] = None
        self._weights_path: Optional[str] = None
        self._result_cached: Optional[Tuple] = None

    @property
    def running(self) -> bool:
        return self._kicked_off and self._result_cached is None

    def start(self, model, device, server_pool, snapshot_pool, args_dict,
              win_rates=None, iter_n: int = 0):
        """Kick off mp-disk collect for next iter. Returns immediately."""
        global _GLOBAL_MANAGER
        if _GLOBAL_MANAGER is None:
            _GLOBAL_MANAGER = WorkerManager(
                n_workers=args_dict.get("n_workers", 8))
            # Paced spawns to avoid resource_tracker race (see mp_disk_collect_sync).
            for wid in range(_GLOBAL_MANAGER.n):
                _GLOBAL_MANAGER.spawn(wid)
                time.sleep(0.5)
            time.sleep(2.0)

        # Save weights
        weights_path = _save_worker_weights_atomic(model, iter_n)
        opp_pool_msg = _build_opp_pool_msg(snapshot_pool, win_rates)

        alive = _GLOBAL_MANAGER.alive_worker_ids()
        if not alive:
            raise RuntimeError("[mp-disk-bg] All workers dead.")

        n_games = args_dict["games_per_iter"]
        n_alive = len(alive)
        games_per_worker = n_games // n_alive
        remainder = n_games % n_alive

        srv_url = (getattr(server_pool[0], 'websocket_url', None)
                   or str(server_pool[0]))

        for i, wid in enumerate(alive):
            worker_n = games_per_worker + (1 if i < remainder else 0)
            srv_idx = i % len(server_pool)
            srv = server_pool[srv_idx]
            cmd = {
                "cmd": "collect_iter",
                "iter_n": iter_n,
                "weights_path": weights_path,
                "n_games": worker_n,
                "max_concurrent": args_dict["max_concurrent"],
                "server_url": (getattr(srv, 'websocket_url', None) or str(srv)),
                "opp_pool": opp_pool_msg,
                "temp_range": list(args_dict["temp_range"]),
                "opp_temp_range": list(args_dict.get("opp_temp_range",
                                                     args_dict["temp_range"])),
                "fp16": args_dict["fp16"],
                "rs_cfg": args_dict["rs_cfg"],
                "turn_cap": args_dict.get("turn_cap", 300),
                "battle_format": args_dict.get("battle_format", "gen9ou"),
                "procedural_teams_path": args_dict.get("teambuilder_path"),
                "device": str(device),
                "rng_seed": random.randint(0, 1_000_000),
                "amp_dtype": args_dict.get("amp_dtype"),  # bf16 plumbing for bg path
            }
            # Stagger same as sync path - see mp_disk_collect_sync for rationale.
            _GLOBAL_MANAGER.ctrl_pipes[wid].send(cmd)
            if i + 1 < len(alive):
                time.sleep(0.25)

        self._kicked_off = True
        self._iter_n = iter_n
        self._n_alive_at_start = len(alive)
        self._cmds_sent = list(alive)

    def join(self) -> Tuple:
        """Wait for all workers, return collect result tuple."""
        if not self._kicked_off:
            return None
        if self._result_cached is not None:
            res = self._result_cached
            self._result_cached = None
            self._kicked_off = False
            return res

        # Re-use sync watchdog loop semantics
        result = _wait_for_iter_results(
            iter_n=self._iter_n,
            cmds_sent=self._cmds_sent,
        )
        self._kicked_off = False
        return result


def _wait_for_iter_results(iter_n: int, cmds_sent: List[int]) -> Tuple:
    """Multiplex result_pipes for an iter. Same logic as mp_disk_collect_sync
    step 4-6, factored out for use by both sync + bg paths."""
    from multiprocessing.connection import wait as mp_wait
    expected = len(cmds_sent)
    received = 0
    results: List[dict] = []
    deadline = time.time() + 60.0 * 30  # 30 min hard ceiling per iter
    pipes_to_wid = {_GLOBAL_MANAGER.result_pipes[wid]: wid for wid in cmds_sent}

    while received < expected and time.time() < deadline:
        ready = mp_wait(list(pipes_to_wid.keys()), timeout=10.0)
        if not ready:
            unhealthy = _GLOBAL_MANAGER.health_check()
            for wid, reason in unhealthy:
                if wid in cmds_sent:
                    print(f"[mp-disk] iter {iter_n}: worker {wid} {reason}; "
                          f"respawning.", flush=True)
                    _GLOBAL_MANAGER.respawn(wid, iter_n)
                    expected -= 1
                    pipes_to_wid = {_GLOBAL_MANAGER.result_pipes[w]: w
                                     for w in cmds_sent
                                     if w not in _GLOBAL_MANAGER.dead_workers}
            continue

        for conn in ready:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                wid = pipes_to_wid.get(conn)
                if wid is not None and wid in cmds_sent:
                    _GLOBAL_MANAGER.respawn(wid, iter_n)
                    expected -= 1
                continue

            if msg.get("status") == "heartbeat":
                _GLOBAL_MANAGER.last_heartbeat[msg.get("worker_id")] = time.time()
                continue

            received += 1
            if msg.get("status") == "done" and msg.get("iter_n") == iter_n:
                results.append(msg)
            elif msg.get("status") == "error":
                print(f"[mp-disk] iter {iter_n}: worker error: "
                      f"{msg.get('exc_msg')}", flush=True)
            _GLOBAL_MANAGER.respawn(msg.get("worker_id"), iter_n)

    # Read traj files. opp_records format = {path: [wins, games]} per collect_v9.
    all_trajs = []
    total_w, total_l, total_ties = 0, 0, 0
    aggregated_wr: Dict[str, List[int]] = {}
    for r in results:
        traj_path = r.get("traj_path")
        if not traj_path or not Path(traj_path).exists():
            continue
        try:
            with gzip.open(traj_path, "rb") as f:
                bundle = pickle.load(f)
            all_trajs.extend(bundle["trajectories"])
            total_w += r["wins"]
            total_l += r["losses"]
            total_ties += r["ties"]
            for opp_path, stats in r.get("wr_per_opp", {}).items():
                rec = aggregated_wr.setdefault(opp_path, [0, 0])
                rec[0] += stats.get("w", 0)
                rec[1] += stats.get("g", 0)
        except Exception as e:
            print(f"[mp-disk] read failure for {traj_path}: {e}", flush=True)

    _cleanup_old_files(iter_n, keep_last=2)
    total_steps = sum(len(t) for t in all_trajs)
    opp_name = f"mp-disk-bg(responded={len(results)}/{len(cmds_sent)})"
    elapsed = 0.0  # bg mode doesn't track its own elapsed cleanly
    return (all_trajs, total_w, total_l, total_ties, total_steps, opp_name,
            elapsed, aggregated_wr)
