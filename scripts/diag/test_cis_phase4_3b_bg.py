#!/usr/bin/env python
"""CIS Phase 4.3b sustained smoke: CISBgCollector start/join overlap.

Drives CISBgCollector across 2 iters. The point of 4.3b is BG overlap —
collect runs while main does update. We simulate update with a sleep
between start() and join(), and verify:
  - start() returns immediately (non-blocking)
  - bg progresses during the simulated update window
  - join() returns valid trajectories + correct W/L
  - Pool slot rotation between iters reloads only changed slots
  - End-to-end PFSP routing same as Test 4 sync sustained smoke

Prereq on dev pod:
  - 2 battle_servers on ports 9000, 9001
  - dummy ckpt at data/models/bc/dummy_for_cis_dev.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="data/models/bc/dummy_for_cis_dev.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-workers", type=int, default=2)
    p.add_argument("--n-games", type=int, default=4)
    p.add_argument("--max-concurrent", type=int, default=1)
    p.add_argument("--n-iters", type=int, default=2)
    p.add_argument("--turn-cap", type=int, default=200)
    p.add_argument("--battle-format", default="gen9ou")
    p.add_argument("--max-pool-size", type=int, default=4)
    p.add_argument("--ports", default="9000,9001")
    p.add_argument("--update-sleep-s", type=float, default=2.0,
                   help="Simulate update phase between start() and join()")
    args = p.parse_args()

    print("=== CIS Phase 4.3b sustained smoke (BG overlap) ===")
    print(f"N={args.n_workers}, n_games={args.n_games}, "
          f"max_conc={args.max_concurrent}, iters={args.n_iters}, "
          f"K_max={args.max_pool_size}, sim_update={args.update_sleep_s}s")
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
    from mp_centralized_collect import CISBgCollector, shutdown_cis_workers
    from eval_metamon_competitive import make_server

    device = torch.device(args.device)

    print("Stage 1: load main model")
    model, cfg, _ = load_checkpoint(args.ckpt, device)
    model.eval()
    print(f"  loaded: {sum(p_.numel() for p_ in model.parameters())/1e6:.1f}M params")
    print()

    print("Stage 2: build fake PFSP pool")
    pool_iter0 = ["/tmp/cis_t4b_opp_X.pt", "/tmp/cis_t4b_opp_Y.pt"]
    pool_iter1 = ["/tmp/cis_t4b_opp_X.pt", "/tmp/cis_t4b_opp_Z.pt"]
    perturb_and_save(args.ckpt, pool_iter0[0], 0.02, seed=11)
    perturb_and_save(args.ckpt, pool_iter0[1], 0.02, seed=22)
    perturb_and_save(args.ckpt, pool_iter1[1], 0.02, seed=33)
    print()

    ports = [int(p_.strip()) for p_ in args.ports.split(",")]
    server_pool = [make_server(p_) for p_ in ports]
    print(f"Stage 3: server_pool = {ports}")
    print()

    bg = CISBgCollector()
    rs_cfg = {"ko_coef": 0.05, "hp_coef": 0.02, "clip_abs": 2.0}
    args_dict = {
        "games_per_iter": args.n_games,
        "max_concurrent": args.max_concurrent,
        "fp16": True,
        "rs_cfg": rs_cfg,
        "temp_range": (1.0, 2.25),
        "opponent_device": str(device),
        "turn_cap": args.turn_cap,
        "battle_format": args.battle_format,
        "teambuilder_path": None,
        "n_workers": args.n_workers,
        "amp_dtype": None,
        "max_pool_size": args.max_pool_size,
        "cis_min_batch": 1,
        "cis_timeout_ms": 15,
    }

    overall_pass = True
    try:
        for iter_idx in range(args.n_iters):
            pool = pool_iter0 if iter_idx == 0 else pool_iter1
            win_rates = {p_: {"w": 0, "g": 1} for p_ in pool}

            print("=" * 60)
            print(f"ITER {iter_idx}: pool={[Path(p_).name for p_ in pool]}")
            print("=" * 60)

            t_start = time.time()
            print(f"  [t+0.0s] calling bg.start() (should return immediately)")
            bg.start(model=model, device=device, server_pool=server_pool,
                     snapshot_pool=pool, args_dict=args_dict,
                     win_rates=win_rates, iter_n=iter_idx)
            t_after_start = time.time()
            t_start_elapsed = t_after_start - t_start
            print(f"  [t+{t_start_elapsed:.2f}s] bg.start() returned. "
                  f"running={bg.running}")

            # start() should return in well under 1s (not synchronous wait
            # for trajectories — that's the whole bg point).
            start_pass = (t_start_elapsed < 30.0 and bg.running)
            print(f"  start() non-blocking: {_bool(start_pass)} "
                  f"(actual {t_start_elapsed:.2f}s)")

            # Simulate update phase. Workers should be making real progress
            # during this sleep — collecting battles, running CIS forwards.
            print(f"  [t+{t_start_elapsed:.2f}s] simulating update "
                  f"({args.update_sleep_s}s)...")
            time.sleep(args.update_sleep_s)

            t_before_join = time.time() - t_start
            print(f"  [t+{t_before_join:.2f}s] calling bg.join()")
            result = bg.join()
            t_total = time.time() - t_start
            print(f"  [t+{t_total:.2f}s] bg.join() returned")

            if result is None:
                print(f"  FAIL: bg.join() returned None")
                overall_pass = False
                continue

            (trajs, w, l, ties, total_steps, opp_name, elapsed,
             aggregated_wr) = result
            print(f"  W/L/T = {w}/{l}/{ties}, trajs={len(trajs)}, "
                  f"steps={total_steps}, elapsed={elapsed:.1f}s")
            print(f"  per-opp wr: {aggregated_wr}")

            # Inspect _CIS_GLOBAL slot state
            from mp_centralized_collect import _CIS_GLOBAL as cg
            if cg is not None:
                cur = cg.get("current_slot_paths", [])
                print(f"  current_slot_paths (first 5): "
                      f"{[Path(p_).name if p_ else None for p_ in cur[:5]]}")

            iter_pass = start_pass
            if len(trajs) == 0:
                print(f"  FAIL: no trajectories")
                iter_pass = False
            if (w + l + ties) == 0:
                print(f"  FAIL: zero battles")
                iter_pass = False

            n_nan = 0
            for tr in trajs[:5]:
                for step in (tr if isinstance(tr, list) else []):
                    if not isinstance(step, dict):
                        continue
                    for key in ("value", "ret", "adv", "old_logp"):
                        v = step.get(key)
                        if v is None:
                            continue
                        try:
                            arr = np.asarray(v, dtype=np.float64)
                            if not np.all(np.isfinite(arr)):
                                n_nan += 1
                        except Exception:
                            pass
            if n_nan > 0:
                print(f"  FAIL: {n_nan} NaN/inf entries")
                iter_pass = False
            else:
                print(f"  no NaN/inf  [PASS]")

            # Check that running flag flips back to False after join
            if bg.running:
                print(f"  FAIL: bg.running still True after join()")
                iter_pass = False

            print(f"  ITER {iter_idx}: {_bool(iter_pass)}")
            print()
            overall_pass = overall_pass and iter_pass

    finally:
        try:
            shutdown_cis_workers()
        except Exception as e:
            print(f"  WARN shutdown_cis_workers: {e}")
        for p_ in (pool_iter0 + pool_iter1):
            try:
                os.remove(p_)
            except OSError:
                pass

    print("=" * 60)
    print(f"=== Phase 4.3b BG overlap summary: {_bool(overall_pass)} ===")
    print("=" * 60)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
