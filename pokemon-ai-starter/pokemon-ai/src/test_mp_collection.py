"""Verification tests for multiprocess collection.
See docs/MULTIPROCESS_COLLECTION.md for the full test plan.

Test 2: Single worker smoke test — 1 worker, 1 server, 10 games.
Test 3: Multi-worker correctness — 3 workers, 3 servers, 30 games.
"""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np

from model import PokeTransformer, PokeTransformerConfig
from ppo import load_checkpoint, Trajectory
from rl_pipeline import mp_collect_v9, InferenceServer, MSG_INFER as _MSG_INFER, MSG_CLEAR as _MSG_CLEAR, MSG_TRAJ as _MSG_TRAJ, MSG_DONE as _MSG_DONE
from rl_collection import _make_server


CKPT = "data/models/rl_v9/selfplay_v9_20260404_192922/snapshot_1164.pt"
INIT = "data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt"


def load_model(device="cuda"):
    model, cfg, ckpt = load_checkpoint(CKPT, device)
    model.eval()
    return model


def get_snapshot_pool():
    """Get a small pool of snapshots for testing."""
    import glob
    pool = sorted(glob.glob("data/models/rl_v9/selfplay_v9_*/snapshot_*.pt"))
    if len(pool) > 20:
        pool = pool[-20:]  # last 20
    return pool


def verify_trajectory(traj: Trajectory, idx: int) -> list:
    """Check a trajectory for integrity. Returns list of issues."""
    issues = []
    n = len(traj)
    if n == 0:
        issues.append(f"traj[{idx}]: empty trajectory")
        return issues

    # Must have matching lengths
    if len(traj.actions) != n:
        issues.append(f"traj[{idx}]: actions len {len(traj.actions)} != {n}")
    if len(traj.log_probs) != n:
        issues.append(f"traj[{idx}]: log_probs len {len(traj.log_probs)} != {n}")
    if len(traj.values) != n:
        issues.append(f"traj[{idx}]: values len {len(traj.values)} != {n}")
    if len(traj.rewards) != n:
        issues.append(f"traj[{idx}]: rewards len {len(traj.rewards)} != {n}")
    if len(traj.dones) != n:
        issues.append(f"traj[{idx}]: dones len {len(traj.dones)} != {n}")
    if len(traj.action_masks) != n:
        issues.append(f"traj[{idx}]: action_masks len {len(traj.action_masks)} != {n}")
    if len(traj.feat_batches) != n:
        issues.append(f"traj[{idx}]: feat_batches len {len(traj.feat_batches)} != {n}")

    # Last step must be done
    if traj.dones and not traj.dones[-1]:
        issues.append(f"traj[{idx}]: last step not done")

    # Only last step should be done
    for t, d in enumerate(traj.dones[:-1]):
        if d:
            issues.append(f"traj[{idx}]: done=True at step {t} (not last)")

    # Actions must be valid (0-8)
    for t, a in enumerate(traj.actions):
        if not (0 <= a <= 8):
            issues.append(f"traj[{idx}]: invalid action {a} at step {t}")

    # Log probs must be finite
    for t, lp in enumerate(traj.log_probs):
        if not np.isfinite(lp):
            issues.append(f"traj[{idx}]: non-finite log_prob {lp} at step {t}")

    # Values must be finite
    for t, v in enumerate(traj.values):
        if not np.isfinite(v):
            issues.append(f"traj[{idx}]: non-finite value {v} at step {t}")

    # Feat batches must have tensors
    for t, fb in enumerate(traj.feat_batches):
        if not isinstance(fb, dict):
            issues.append(f"traj[{idx}]: feat_batch[{t}] is not dict")
        elif "legal_mask" not in fb:
            issues.append(f"traj[{idx}]: feat_batch[{t}] missing legal_mask")

    return issues


def test_single_worker(n_games=10):
    """Test 2: Single worker smoke test."""
    print("\n" + "=" * 60)
    print("  TEST 2: Single Worker Smoke Test (%d games)" % n_games)
    print("=" * 60)

    model = load_model()
    pool = get_snapshot_pool()
    server_pool = [_make_server("9000")]

    t0 = time.time()
    trajs, wins, losses, ties, steps, summary, elapsed = mp_collect_v9(
        model, torch.device("cuda"), server_pool,
        n_games=n_games, max_concurrent=5,
        snapshot_pool=pool, fp16=True,
        reward_shaper_cfg={"ko_coef": 0.05, "hp_coef": 0.02, "clip_abs": 2.0},
        temp_range=(1.0, 2.25),
    )
    wall = time.time() - t0

    print(f"\n  Results: {len(trajs)} trajectories, {steps} steps, {elapsed:.0f}s collect, {wall:.0f}s wall")
    print(f"  W/L/T: {wins}/{losses}/{ties}")
    print(f"  Summary: {summary}")

    # Verify trajectories
    all_issues = []
    for i, t in enumerate(trajs):
        all_issues.extend(verify_trajectory(t, i))

    if all_issues:
        print(f"\n  TRAJECTORY ISSUES ({len(all_issues)}):")
        for issue in all_issues[:20]:
            print(f"    {issue}")
    else:
        print(f"\n  All {len(trajs)} trajectories passed integrity check")

    # Check expected count
    ok = True
    if len(trajs) < n_games * 0.5:
        print(f"  FAIL: Only {len(trajs)} trajectories for {n_games} games (expected ~{n_games})")
        ok = False
    if all_issues:
        print(f"  FAIL: {len(all_issues)} trajectory integrity issues")
        ok = False
    if steps == 0:
        print(f"  FAIL: 0 steps collected")
        ok = False

    print(f"\n  {'PASS' if ok else 'FAIL'}: Single worker test")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return ok


def test_multi_worker(n_games=30, n_workers=3):
    """Test 3: Multi-worker correctness."""
    print("\n" + "=" * 60)
    print("  TEST 3: Multi-Worker Correctness (%d games, %d workers)" % (n_games, n_workers))
    print("=" * 60)

    model = load_model()
    pool = get_snapshot_pool()
    server_pool = [_make_server(str(9000 + i)) for i in range(n_workers)]

    t0 = time.time()
    trajs, wins, losses, ties, steps, summary, elapsed = mp_collect_v9(
        model, torch.device("cuda"), server_pool,
        n_games=n_games, max_concurrent=5,
        snapshot_pool=pool, fp16=True,
        reward_shaper_cfg={"ko_coef": 0.05, "hp_coef": 0.02, "clip_abs": 2.0},
        temp_range=(1.0, 2.25),
    )
    wall = time.time() - t0

    print(f"\n  Results: {len(trajs)} trajectories, {steps} steps, {elapsed:.0f}s collect, {wall:.0f}s wall")
    print(f"  W/L/T: {wins}/{losses}/{ties}")

    # Verify trajectories
    all_issues = []
    for i, t in enumerate(trajs):
        all_issues.extend(verify_trajectory(t, i))

    if all_issues:
        print(f"\n  TRAJECTORY ISSUES ({len(all_issues)}):")
        for issue in all_issues[:20]:
            print(f"    {issue}")
    else:
        print(f"\n  All {len(trajs)} trajectories passed integrity check")

    # Check expected count
    ok = True
    if len(trajs) < n_games * 0.5:
        print(f"  FAIL: Only {len(trajs)} trajectories for {n_games} games")
        ok = False
    if all_issues:
        print(f"  FAIL: {len(all_issues)} trajectory integrity issues")
        ok = False
    if steps == 0:
        print(f"  FAIL: 0 steps collected")
        ok = False

    # Speed check: should be faster than single-process baseline
    games_per_sec = len(trajs) / max(1, elapsed)
    print(f"  Throughput: {games_per_sec:.2f} games/sec ({elapsed:.0f}s for {len(trajs)} games)")

    print(f"\n  {'PASS' if ok else 'FAIL'}: Multi-worker test")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return ok


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--test", choices=["single", "multi", "all"], default="all")
    p.add_argument("--games", type=int, default=10)
    args = p.parse_args()

    if args.test in ("single", "all"):
        test_single_worker(args.games)

    if args.test in ("multi", "all"):
        test_multi_worker(args.games * 3, 3)
