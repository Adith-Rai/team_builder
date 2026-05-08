#!/usr/bin/env python
"""CIS Phase 4.3c small-scale wall-time A/B.

Demonstrates that --cis --pipeline (bg overlap) actually saves wall time
vs --cis only (sync). Two variants run sequentially; total wall-time
compared.

Setup (per iter):
  - Collect ~4 games via CIS, observe collect_time T_c
  - Simulate update with sleep(--update-sleep-s)
  - Variant A (sync): time = T_c + sleep
  - Variant B (bg overlap): start bg, sleep simultaneously, join.
    time = max(T_c, sleep) + small overhead

Acceptance:
  - Variant B total < Variant A total for the same N iters
  - Savings ≥ ~30% of (N-1) * min(T_c, sleep)  (iter 0 has no prev bg
    to overlap, so savings only kick in iters 1..N-1)

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
    p.add_argument("--n-iters", type=int, default=3)
    p.add_argument("--turn-cap", type=int, default=200)
    p.add_argument("--battle-format", default="gen9ou")
    p.add_argument("--max-pool-size", type=int, default=4)
    p.add_argument("--ports", default="9000,9001")
    p.add_argument("--update-sleep-s", type=float, default=7.0,
                   help="Simulated update phase duration (matches collect time)")
    args = p.parse_args()

    print("=== CIS Phase 4.3c small-scale wall-time A/B ===")
    print(f"N={args.n_workers}, n_games={args.n_games}, "
          f"max_conc={args.max_concurrent}, n_iters={args.n_iters}, "
          f"sim_update={args.update_sleep_s}s")
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
        CISBgCollector, mp_centralized_collect_sync, shutdown_cis_workers,
    )
    from eval_metamon_competitive import make_server

    device = torch.device(args.device)

    # 1. Load model + build pool
    model, _, _ = load_checkpoint(args.ckpt, device)
    model.eval()
    pool = ["/tmp/cis_t4c_opp_X.pt", "/tmp/cis_t4c_opp_Y.pt"]
    perturb_and_save(args.ckpt, pool[0], 0.02, seed=11)
    perturb_and_save(args.ckpt, pool[1], 0.02, seed=22)
    win_rates = {p_: {"w": 0, "g": 1} for p_ in pool}
    ports = [int(p_.strip()) for p_ in args.ports.split(",")]
    server_pool = [make_server(p_) for p_ in ports]

    rs_cfg = {"ko_coef": 0.05, "hp_coef": 0.02, "clip_abs": 2.0}
    common = dict(
        n_games=args.n_games,
        max_concurrent=args.max_concurrent,
        snapshot_pool=pool,
        fp16=True,
        reward_shaper_cfg=rs_cfg,
        temp_range=(1.0, 2.25),
        opponent_device=str(device),
        win_rates=win_rates,
        turn_cap=args.turn_cap,
        battle_format=args.battle_format,
        procedural_teams_path=None,
        n_workers=args.n_workers,
        amp_dtype=None,
        cis_min_batch=1,
        cis_timeout_ms=15,
        max_pool_size=args.max_pool_size,
    )

    # ---------- VARIANT A: sync (no bg overlap) ----------
    print("=" * 60)
    print(f"VARIANT A: --cis only (sync), {args.n_iters} iters")
    print("=" * 60)
    a_collect_times = []
    a_total_t0 = time.time()
    for i in range(args.n_iters):
        t0 = time.time()
        result = mp_centralized_collect_sync(
            model=model, device=device, server_pool=server_pool,
            iter_n=i, **common,
        )
        t_collect = time.time() - t0
        a_collect_times.append(t_collect)
        print(f"  iter {i}: collect={t_collect:.1f}s "
              f"(W/L/T={result[1]}/{result[2]}/{result[3]}, "
              f"trajs={len(result[0])})")
        # Simulate update phase
        time.sleep(args.update_sleep_s)
    a_total = time.time() - a_total_t0
    print(f"  Variant A total: {a_total:.1f}s  "
          f"(sum collect={sum(a_collect_times):.1f}s, "
          f"sleeps={args.n_iters * args.update_sleep_s:.1f}s)")
    print()

    # Reset CIS global so variant B starts fresh.
    shutdown_cis_workers()
    time.sleep(2.0)

    # ---------- VARIANT B: bg overlap ----------
    print("=" * 60)
    print(f"VARIANT B: --cis --pipeline (bg overlap), {args.n_iters} iters")
    print("=" * 60)
    bg = CISBgCollector()
    args_dict = dict(
        games_per_iter=args.n_games,
        max_concurrent=args.max_concurrent,
        fp16=True,
        rs_cfg=rs_cfg,
        temp_range=(1.0, 2.25),
        opponent_device=str(device),
        turn_cap=args.turn_cap,
        battle_format=args.battle_format,
        teambuilder_path=None,
        n_workers=args.n_workers,
        amp_dtype=None,
        max_pool_size=args.max_pool_size,
        cis_min_batch=1,
        cis_timeout_ms=15,
    )

    b_join_waits = []
    b_total_t0 = time.time()
    pending_result = None
    for i in range(args.n_iters):
        # First iter: no prior bg, do sync collect and start bg for next.
        if i == 0:
            t0 = time.time()
            result = mp_centralized_collect_sync(
                model=model, device=device, server_pool=server_pool,
                iter_n=i, **common,
            )
            t_collect = time.time() - t0
            print(f"  iter {i} (sync): collect={t_collect:.1f}s "
                  f"(W/L/T={result[1]}/{result[2]}/{result[3]})")
            # Kick off bg for iter 1
            if i + 1 < args.n_iters:
                bg.start(model=model, device=device, server_pool=server_pool,
                         snapshot_pool=pool, args_dict=args_dict,
                         win_rates=win_rates, iter_n=i + 1)
                print(f"    [bg kicked off for iter {i+1}]")
            # Simulate update
            time.sleep(args.update_sleep_s)
        else:
            # Use bg result kicked off in prev iter
            t_join0 = time.time()
            result = bg.join()
            t_join = time.time() - t_join0
            b_join_waits.append(t_join)
            print(f"  iter {i} (bg-join): join_wait={t_join:.2f}s "
                  f"(W/L/T={result[1]}/{result[2]}/{result[3]})")
            # Kick off bg for next iter
            if i + 1 < args.n_iters:
                bg.start(model=model, device=device, server_pool=server_pool,
                         snapshot_pool=pool, args_dict=args_dict,
                         win_rates=win_rates, iter_n=i + 1)
                print(f"    [bg kicked off for iter {i+1}]")
            # Simulate update
            time.sleep(args.update_sleep_s)

    b_total = time.time() - b_total_t0
    print(f"  Variant B total: {b_total:.1f}s  "
          f"(join_waits={[f'{w:.2f}' for w in b_join_waits]})")
    print()

    # ---------- Verdict ----------
    print("=" * 60)
    print("=== Wall-time A/B verdict ===")
    print(f"  Variant A (sync):       {a_total:.1f}s")
    print(f"  Variant B (bg overlap): {b_total:.1f}s")
    saving = a_total - b_total
    saving_pct = 100.0 * saving / a_total if a_total > 0 else 0.0
    print(f"  Saved: {saving:.1f}s ({saving_pct:+.1f}%)")
    print()

    # Expected savings: with N iters, iters 1..N-1 overlap collect with
    # update-sleep. Ideal savings = (N-1) * min(collect, sleep).
    avg_collect = float(np.mean(a_collect_times)) if a_collect_times else 0.0
    ideal_overlap_per_iter = min(avg_collect, args.update_sleep_s)
    ideal_savings = (args.n_iters - 1) * ideal_overlap_per_iter
    print(f"  Theoretical ideal saving: ~{ideal_savings:.1f}s "
          f"((N-1) × min(avg_collect={avg_collect:.1f}s, "
          f"sleep={args.update_sleep_s:.1f}s))")
    overlap_efficiency = (saving / ideal_savings * 100.0
                          if ideal_savings > 0 else 0.0)
    print(f"  Overlap efficiency: {overlap_efficiency:.1f}%")

    # Pass criteria: positive saving AND ≥30% of theoretical ideal.
    overall_pass = saving > 0 and overlap_efficiency >= 30.0
    print(f"  Overall: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 60)

    # Cleanup
    try:
        shutdown_cis_workers()
    except Exception:
        pass
    for p_ in pool:
        try:
            os.remove(p_)
        except OSError:
            pass

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
