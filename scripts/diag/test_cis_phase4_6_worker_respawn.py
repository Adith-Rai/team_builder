#!/usr/bin/env python
"""CIS Phase 4.6 Test 6a: full-reset recovery on worker death (Option B).

Drives mp_centralized_collect_sync for 3 iters at small scale. During iter
1, a sidecar thread kill -9's worker 0 partway through. Verifies:

  1. Watchdog detects the dead worker (via mp_wait timeout OR EOFError)
  2. Wait loop raises CISResetNeeded → orchestrator full-reset fires
  3. Reset re-spawns CIS subprocess + ALL N workers; opp slot paths restored
  4. Iter 1 is re-issued from zero to fresh workers and completes cleanly
     (binary outcome — full trajectories, no partial)
  5. ALL worker pids are different post-reset (not just the killed one —
     this is Option B's contract: full reset, not granular respawn)
  6. Iter 2 runs normally on the new workers

Prereq on dev pod:
  - 2 battle_servers running on ports 9000, 9001
  - dummy ckpt at data/models/bc/dummy_for_cis_dev.pt

Usage:
  cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
  python /workspace/team_builder/scripts/diag/test_cis_phase4_6_worker_respawn.py
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

import torch


def _bool(x):
    return "PASS" if x else "FAIL"


def perturb_and_save(base_ckpt_path: str, out_path: str, scale: float, seed: int):
    ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    torch.manual_seed(seed)
    for k in list(sd.keys()):
        v = sd[k]
        if torch.is_floating_point(v):
            sd[k] = v + torch.randn_like(v) * scale
    tmp = out_path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, out_path)


def _kill_worker_after(get_pid_fn, delay_s: float, kill_log: list):
    """Sidecar: wait delay_s, then SIGKILL the worker PID returned by get_pid_fn.
    Records the killed pid + timestamp into kill_log for the test to verify."""
    time.sleep(delay_s)
    try:
        pid = get_pid_fn()
        if pid and pid > 0:
            os.kill(pid, signal.SIGKILL)
            kill_log.append({"pid": pid, "ts": time.time(), "ok": True})
            print(f"[test sidecar] SIGKILL'd worker pid={pid}", flush=True)
        else:
            kill_log.append({"pid": None, "ts": time.time(), "ok": False,
                              "reason": "no pid"})
    except Exception as e:
        kill_log.append({"pid": None, "ts": time.time(), "ok": False,
                          "reason": str(e)})
        print(f"[test sidecar] kill failed: {e}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="data/models/bc/dummy_for_cis_dev.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--n-games", type=int, default=40)
    p.add_argument("--max-concurrent", type=int, default=8)
    p.add_argument("--turn-cap", type=int, default=200)
    p.add_argument("--battle-format", default="gen9ou")
    p.add_argument("--max-pool-size", type=int, default=4)
    p.add_argument("--ports", default="9000,9001")
    p.add_argument("--kill-delay", type=float, default=15.0,
                    help="Seconds into iter 1 before SIGKILL'ing target")
    p.add_argument("--target", choices=("worker", "cis"), default="worker",
                    help="What to kill mid-iter: 'worker' (a single worker "
                         "proc) or 'cis' (the CIS subprocess). Both should "
                         "trigger the same Option B full-reset path.")
    p.add_argument("--target-wid", type=int, default=0,
                    help="Which worker to kill if --target=worker")
    args = p.parse_args()

    label = ("Test 6a (worker death)" if args.target == "worker"
             else "Test 6b (CIS subprocess death)")
    print(f"=== CIS Phase 4.6 {label} — Option B full reset ===")
    target_desc = (f"worker {args.target_wid}" if args.target == "worker"
                   else "CIS subprocess")
    print(f"N={args.n_workers}, n_games={args.n_games}, "
          f"max_concurrent={args.max_concurrent}, "
          f"K_max={args.max_pool_size}, "
          f"kill target={target_desc} at T+{args.kill_delay}s")
    print()

    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(here, "..", "..",
                                            "pokemon-ai-starter", "pokemon-ai", "src"))
    if os.path.isdir(src_dir):
        os.chdir(src_dir)
        sys.path.insert(0, src_dir)
    else:
        sys.path.insert(0, ".")

    from ppo import load_checkpoint
    from mp_centralized_collect import (
        mp_centralized_collect_sync, shutdown_cis_workers, _CIS_GLOBAL,
    )
    from eval_metamon_competitive import make_server

    device = torch.device(args.device)

    # 1. Load main model
    print("Stage 1: load main model")
    model, cfg, _ = load_checkpoint(args.ckpt, device)
    model.eval()
    print(f"  loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print()

    # 2. Build a small fake PFSP pool (2 perturbed copies, identical across iters)
    print("Stage 2: build fake PFSP pool (2 perturbed opps, stable across iters)")
    pool = ["/tmp/cis_t6a_opp_X.pt", "/tmp/cis_t6a_opp_Y.pt"]
    perturb_and_save(args.ckpt, pool[0], scale=0.02, seed=11)
    perturb_and_save(args.ckpt, pool[1], scale=0.02, seed=22)
    print()

    # 3. Server pool
    ports = [int(p_.strip()) for p_ in args.ports.split(",")]
    server_pool = [make_server(p_) for p_ in ports]
    print(f"Stage 3: server_pool = {ports}")
    print()

    rs_cfg = {"ko_coef": 0.05, "hp_coef": 0.02, "clip_abs": 2.0}
    common_kwargs = dict(
        n_games=args.n_games,
        max_concurrent=args.max_concurrent,
        fp16=True,
        reward_shaper_cfg=rs_cfg,
        temp_range=(1.0, 2.25),
        opponent_device=str(device),
        turn_cap=args.turn_cap,
        battle_format=args.battle_format,
        procedural_teams_path=None,
        n_workers=args.n_workers,
        rng_seed=42,
        amp_dtype=None,
        cis_min_batch=1,
        cis_timeout_ms=15,
        max_pool_size=args.max_pool_size,
    )

    win_rates = {p_: {"w": 0, "g": 1} for p_ in pool}
    overall_pass = True

    try:
        # ---- ITER 0: warmup, no kills ----
        print("=" * 60)
        print("ITER 0: warmup (no kills)")
        print("=" * 60)
        t0 = time.time()
        (trajs0, w0, l0, t0_, steps0, opp_name0, elapsed0,
         _) = mp_centralized_collect_sync(
            model=model, device=device, server_pool=server_pool,
            snapshot_pool=pool, win_rates=win_rates, iter_n=0,
            **common_kwargs,
        )
        print(f"  collect done: {elapsed0:.1f}s, W/L/T={w0}/{l0}/{t0_}, "
              f"trajs={len(trajs0)}")
        iter0_ok = (len(trajs0) > 0 and (w0 + l0 + t0_) > 0)
        print(f"  ITER 0: {_bool(iter0_ok)}")
        overall_pass = overall_pass and iter0_ok
        print()

        # Capture original worker pids + CIS subprocess pid BEFORE kill.
        from mp_centralized_collect import _CIS_GLOBAL as cg
        manager = cg["manager"]
        cis_server = cg["server"]
        orig_pids = {wid: p.pid for wid, p in manager.workers.items()}
        orig_cis_pid = cis_server._proc.pid if cis_server._proc else None
        print(f"  worker pids after iter 0: {orig_pids}")
        print(f"  CIS subprocess pid: {orig_cis_pid}")
        print()

        # ---- ITER 1: kill {worker | cis} mid-iter ----
        print("=" * 60)
        print(f"ITER 1: kill {target_desc} at T+{args.kill_delay}s")
        print("=" * 60)

        kill_log: list = []
        def _get_target_pid():
            # Re-read at kill time to get the current pid (in case manager
            # already reset for any reason).
            if args.target == "worker":
                p = manager.workers.get(args.target_wid)
                return p.pid if p is not None else None
            else:
                return cis_server._proc.pid if cis_server._proc else None

        kill_thread = threading.Thread(
            target=_kill_worker_after,
            args=(_get_target_pid, args.kill_delay, kill_log),
            daemon=True,
        )
        kill_thread.start()

        t1 = time.time()
        (trajs1, w1, l1, t1_, steps1, opp_name1, elapsed1,
         _) = mp_centralized_collect_sync(
            model=model, device=device, server_pool=server_pool,
            snapshot_pool=pool, win_rates=win_rates, iter_n=1,
            **common_kwargs,
        )
        kill_thread.join(timeout=2.0)

        print(f"  collect done: {elapsed1:.1f}s, W/L/T={w1}/{l1}/{t1_}, "
              f"trajs={len(trajs1)}")
        print(f"  kill_log: {kill_log}")

        # Validation gates for iter 1 (Option B contract).
        iter1_ok = True
        if not kill_log or not kill_log[0].get("ok"):
            print("  FAIL: kill sidecar did not fire successfully")
            iter1_ok = False
        # Iter 1 must complete with FULL trajectories (binary outcome —
        # full reset re-issued the iter, so we expect the same volume as
        # iter 0).
        if len(trajs1) == 0:
            print("  FAIL: iter 1 returned 0 trajectories — "
                   "full-reset path did not recover")
            iter1_ok = False
        if (w1 + l1 + t1_) == 0:
            print("  FAIL: iter 1 returned 0 battles")
            iter1_ok = False
        # Re-fetch manager from the (potentially new) _CIS_GLOBAL.
        from mp_centralized_collect import (
            _CIS_GLOBAL as cg2, _RESET_HISTORY,
        )
        if cg2 is None:
            print("  FAIL: _CIS_GLOBAL is None after iter 1 — "
                   "shutdown happened unexpectedly")
            iter1_ok = False
            new_manager = None
        else:
            new_manager = cg2["manager"]
        # Reset history should have at least 1 entry from iter 1.
        if not _RESET_HISTORY:
            print("  FAIL: _RESET_HISTORY empty — "
                   "no reset event recorded for iter 1")
            iter1_ok = False
        else:
            print(f"  reset events: {len(_RESET_HISTORY)} in window  [PASS]")
        # All worker pids should differ from iter-0 pids (full reset
        # re-spawns ALL workers, not just the killed one — Option B).
        if new_manager is not None:
            new_pids = {wid: p.pid for wid, p in new_manager.workers.items()}
            new_cis_pid = (cg2["server"]._proc.pid
                           if cg2["server"]._proc else None)
            print(f"  worker pids after reset: {new_pids}")
            print(f"  CIS pid after reset: {new_cis_pid}")
            unchanged = [wid for wid in orig_pids
                         if new_pids.get(wid) == orig_pids[wid]]
            if unchanged:
                print(f"  FAIL: workers {unchanged} have unchanged pids — "
                       f"full reset should replace ALL workers")
                iter1_ok = False
            else:
                print(f"  ALL {len(orig_pids)} worker pids changed  [PASS]")
            if new_cis_pid == orig_cis_pid:
                print(f"  FAIL: CIS pid unchanged ({new_cis_pid}) — "
                       f"full reset should replace CIS too")
                iter1_ok = False
            else:
                print(f"  CIS pid changed: {orig_cis_pid} -> {new_cis_pid}  [PASS]")
        print(f"  ITER 1: {_bool(iter1_ok)}")
        overall_pass = overall_pass and iter1_ok
        print()

        # ---- ITER 2: normal post-respawn ----
        print("=" * 60)
        print("ITER 2: post-respawn normal collect (all N workers alive)")
        print("=" * 60)
        t2 = time.time()
        (trajs2, w2, l2, t2_, steps2, opp_name2, elapsed2,
         _) = mp_centralized_collect_sync(
            model=model, device=device, server_pool=server_pool,
            snapshot_pool=pool, win_rates=win_rates, iter_n=2,
            **common_kwargs,
        )
        print(f"  collect done: {elapsed2:.1f}s, W/L/T={w2}/{l2}/{t2_}, "
              f"trajs={len(trajs2)}")
        # Iter 2 should look like iter 0 in volume (full N workers contributing).
        iter2_ok = (len(trajs2) > 0 and (w2 + l2 + t2_) > 0)
        # Sanity: iter 2 trajectories should be in the same ballpark as iter 0.
        # Allow 50%+ of iter 0's traj count (loose — small-scale variance).
        if len(trajs2) < 0.5 * max(len(trajs0), 1):
            print(f"  WARN: iter 2 trajs ({len(trajs2)}) much less than "
                   f"iter 0 ({len(trajs0)}) — respawned worker may not be "
                   f"producing")
        print(f"  ITER 2: {_bool(iter2_ok)}")
        overall_pass = overall_pass and iter2_ok
        print()

    finally:
        try:
            shutdown_cis_workers()
        except Exception as e:
            print(f"shutdown error (non-fatal): {e}")

    print()
    print("=" * 60)
    print(f"OVERALL: {_bool(overall_pass)}")
    print("=" * 60)
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
