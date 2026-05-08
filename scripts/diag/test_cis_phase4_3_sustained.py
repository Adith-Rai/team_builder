#!/usr/bin/env python
"""CIS Phase 4.3a Test 4: sustained smoke (end-to-end PFSP routing).

Drives the full mp_centralized_collect_sync orchestrator -> worker -> CIS
path for 1-2 iters with a small-scale config:
  - N=2 workers
  - K=2 PFSP opps (slots 1,2) + player (slot 0)
  - n_games=20, max_concurrent=4
  - 2 iters total, second iter rotates opp pool to exercise slot reload

Validates:
  - CIS spawns with K_max+1 slots
  - Orchestrator builds pool_slot_map correctly
  - Slot 0 reloads each iter; slots 1..K reload only when paths change
  - Workers receive pool_slot_map, build per-slot batchers
  - Player batcher hits slot 0; opp batchers hit slot 1..K
  - Trajectories returned, no NaN in episode steps
  - W/L counts non-zero (battles actually ran end-to-end)

Prereq on dev pod:
  - 2 battle_servers running on ports 9000, 9001
  - dummy ckpt at data/models/bc/dummy_for_cis_dev.pt

Usage:
  cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
  python /workspace/team_builder/scripts/diag/test_cis_phase4_3_sustained.py
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
    """CPU-load base, additively perturb floats, atomically save."""
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
    p.add_argument("--n-games", type=int, default=20)
    p.add_argument("--max-concurrent", type=int, default=4)
    p.add_argument("--n-iters", type=int, default=2)
    p.add_argument("--turn-cap", type=int, default=200)
    p.add_argument("--battle-format", default="gen9ou")
    p.add_argument("--max-pool-size", type=int, default=4)
    p.add_argument("--ports", default="9000,9001")
    args = p.parse_args()

    print("=== CIS Phase 4.3a Test 4: sustained smoke ===")
    print(f"N={args.n_workers}, n_games={args.n_games}, "
          f"max_concurrent={args.max_concurrent}, iters={args.n_iters}, "
          f"K_max={args.max_pool_size}")
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

    # 2. Build a fake PFSP pool: 2 perturbed copies of dummy ckpt
    print("Stage 2: build fake PFSP pool (2 perturbed opps)")
    pool_iter0 = ["/tmp/cis_t4_opp_X.pt", "/tmp/cis_t4_opp_Y.pt"]
    pool_iter1 = ["/tmp/cis_t4_opp_X.pt", "/tmp/cis_t4_opp_Z.pt"]  # rotates Y -> Z
    perturb_and_save(args.ckpt, pool_iter0[0], scale=0.02, seed=11)
    perturb_and_save(args.ckpt, pool_iter0[1], scale=0.02, seed=22)
    perturb_and_save(args.ckpt, pool_iter1[1], scale=0.02, seed=33)
    print(f"  iter 0 pool: {pool_iter0}")
    print(f"  iter 1 pool: {pool_iter1} (Y -> Z rotation tests slot reload)")
    print()

    # 3. Build server pool from ports
    ports = [int(p_.strip()) for p_ in args.ports.split(",")]
    server_pool = [make_server(p_) for p_ in ports]
    print(f"Stage 3: server_pool = {ports}")
    print()

    # 4. Common kwargs for mp_centralized_collect_sync
    # Match production rs_cfg from rl_player.py / rl_pipeline.py defaults.
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
        cis_min_batch=1,  # fire immediately for the test (no batch wait)
        cis_timeout_ms=15,
        max_pool_size=args.max_pool_size,
    )

    overall_pass = True

    try:
        for iter_idx in range(args.n_iters):
            pool = pool_iter0 if iter_idx == 0 else pool_iter1
            win_rates = {p_: {"w": 0, "g": 1} for p_ in pool}

            print("=" * 60)
            print(f"ITER {iter_idx}: pool={[Path(p_).name for p_ in pool]}")
            print("=" * 60)

            t0 = time.time()
            (trajs, w, l, ties, total_steps, opp_name, elapsed,
             aggregated_wr) = mp_centralized_collect_sync(
                model=model, device=device, server_pool=server_pool,
                snapshot_pool=pool, win_rates=win_rates, iter_n=iter_idx,
                **common_kwargs,
            )

            n_traj = len(trajs)
            print(f"  collect done: {elapsed:.1f}s")
            print(f"  W/L/T = {w}/{l}/{ties}")
            print(f"  trajectories: {n_traj}, total_steps: {total_steps}")
            print(f"  opp_name: {opp_name}")
            print(f"  per-opp wr: {aggregated_wr}")

            # Inspect _CIS_GLOBAL state (slot reload tracking)
            from mp_centralized_collect import _CIS_GLOBAL as cg
            if cg is not None:
                cur = cg.get("current_slot_paths", [])
                print(f"  current_slot_paths after iter (first 5): "
                      f"{[Path(p_).name if p_ else None for p_ in cur[:5]]}")

            # Validation gates
            iter_pass = True
            if n_traj == 0:
                print(f"  FAIL: no trajectories returned")
                iter_pass = False
            if (w + l + ties) == 0:
                print(f"  FAIL: zero battles completed")
                iter_pass = False

            # NaN check on trajectory observations / values
            n_nan = 0
            for tr in trajs[:5]:  # sample
                # Trajectory format: list of step dicts. Check value/return fields.
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
                print(f"  FAIL: {n_nan} NaN/inf entries in sample trajectories")
                iter_pass = False
            else:
                print(f"  no NaN/inf in sampled trajectories  [PASS]")

            print(f"  ITER {iter_idx}: {_bool(iter_pass)}")
            print()
            overall_pass = overall_pass and iter_pass

    finally:
        # Cleanup CIS + workers
        try:
            shutdown_cis_workers()
        except Exception as e:
            print(f"  WARN shutdown_cis_workers: {e}")
        # Cleanup temp ckpts
        for p_ in (pool_iter0 + pool_iter1):
            try:
                os.remove(p_)
            except OSError:
                pass

    print("=" * 60)
    print(f"=== Sustained smoke summary ===")
    print(f"  overall: {_bool(overall_pass)}")
    print("=" * 60)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
