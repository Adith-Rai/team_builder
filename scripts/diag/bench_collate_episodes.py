#!/usr/bin/env python
"""Microbench: vectorized collate_episodes vs V1 reference (S60 Fix #3).

Measures wall-time on prod-scale inputs to confirm the ~10% update savings
projection from PROFILE_BOTTLENECKS_REPORT.md §4.1 (~55s/iter at prod scale).

The S59 profile measured 55s in collate_episodes across the full 506s update
phase at 800g (8 chunks × 100 episodes per chunk via minibatch_size=16 ×
multiple PPO epochs). Per single collate call: 55s / (8 chunks × ~4 epochs)
≈ 1.7s/call at chunk_size=16, L_max=200.

We bench at chunk_size=16 (production --tier3-minibatch-size 16) and at
chunk_size=50 (validation scale) to characterize scaling.

CPU-only (where collation actually runs in prod — only the final device move
goes to GPU). Reports median over N trials to avoid GC noise.

Usage:
  python scripts/diag/bench_collate_episodes.py
"""

from __future__ import annotations

import os
import sys
import time
import gc
import statistics

import numpy as np
import torch


def _setup():
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(here, "..", "..",
                                            "pokemon-ai-starter", "pokemon-ai", "src"))
    sys.path.insert(0, src_dir)


_setup()
from ppo import collate_episodes as collate_vec  # noqa: E402

# Import the frozen V1 from the bit-exact test module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_collate_episodes_vec_bitexact import _collate_episodes_v1 as collate_v1  # noqa: E402


def _make_realistic_episode(T: int, A: int = 9, feat_dim: int = 64,
                            n_leaves: int = 6, seed: int = 0):
    """Build a more realistic episode: 6 tensor leaves (mimicking the actual
    build_turn_batch schema — Pokemon state, types, hp, moves, items, abilities,
    each ~64 floats per turn) plus nested categorical embedding fields.
    """
    rng = np.random.RandomState(seed)
    feat_batches = []
    g = torch.Generator().manual_seed(seed)
    for t in range(T):
        turn = {}
        for k in range(n_leaves):
            turn[f"leaf_{k}"] = torch.randn(1, feat_dim, generator=g)
        # One nested dict to mimic battle_state.{spatial, temporal, etc}
        turn["nested"] = {
            "embed_a": torch.randint(0, 1000, (1, 12), generator=g),
            "embed_b": torch.randn(1, 32, generator=g),
        }
        feat_batches.append(turn)
    actions = rng.randint(0, A, size=T).astype(int).tolist()
    old_logp = rng.randn(T).astype(np.float32).tolist()
    advantages = rng.randn(T).astype(np.float32).tolist()
    returns = rng.randn(T).astype(np.float32).tolist()
    action_masks = [rng.uniform(0, 1, size=A).astype(np.float32).tolist()
                    for _ in range(T)]
    return {
        "feat_batches": feat_batches,
        "actions": actions,
        "old_logp": old_logp,
        "advantages": advantages,
        "returns": returns,
        "action_masks": action_masks,
    }


def _bench(label: str, fn, episodes, kwargs, n_warmup: int = 2, n_trials: int = 10):
    # Warmup
    for _ in range(n_warmup):
        fn(episodes, **kwargs)
    gc.collect()
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        out = fn(episodes, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        del out
        gc.collect()
    median = statistics.median(times)
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if n_trials > 1 else 0.0
    print(f"  {label}: median={median*1000:.2f} ms  mean={mean*1000:.2f} ms  stdev={stdev*1000:.2f} ms  (n={n_trials})")
    return median


def main():
    print("=" * 72)
    print("Microbench: collate_episodes V1 vs vectorized (S60 Fix #3)")
    print("=" * 72)
    print(f"  torch: {torch.__version__}")
    print(f"  CPU-only collation (matches prod — device move is separate)")
    print()

    # Production call pattern: --tier3-minibatch-size 16 means chunk_size=16
    # episodes per collate call. L_max=temporal_context=200, tail=True.
    # Realistic T distribution: 150-250 turns; we use uniform-ish for benching.
    for label, B, T_range in [
        ("Prod chunk: B=16, T~200, L_max=200, tail",        16, (180, 220)),
        ("Validation chunk: B=50, T~150, L_max=200, tail",  50, (130, 170)),
        ("Stress: B=64, T~250, L_max=200, tail",            64, (230, 270)),
    ]:
        print(f"--- {label} ---")
        rng = np.random.RandomState(42)
        episodes = [_make_realistic_episode(int(rng.randint(*T_range)),
                                            seed=1000 + b)
                    for b in range(B)]
        kwargs = {"L_max": 200, "device": None, "tail": True}

        t_v1 = _bench("V1 (current)", collate_v1, episodes, kwargs)
        t_vec = _bench("Vectorized  ", collate_vec, episodes, kwargs)
        speedup = t_v1 / t_vec if t_vec > 0 else float("inf")
        savings = (t_v1 - t_vec) * 1000  # ms per call
        print(f"  speedup: {speedup:.2f}x  savings: {savings:.2f} ms/call")
        print()

    # Per-PPO-epoch projection at production scale.
    # Prod: 800g (collect_games) × ~4 PPO epochs × (800 / 16 = 50 chunks per epoch)
    #     = 200 collate calls per iter.
    # If we save X ms/call: total saved per iter = X * 200 / 1000 sec.
    print("=" * 72)
    print("Production iter savings projection (800g × 4 epochs × 50 chunks):")
    print("  At chunk=16, savings = (V1 - vec) ms/call × 200 calls/iter")
    print("  Reported above as 'savings' line. Scale by 200 for total iter savings.")
    print("=" * 72)


if __name__ == "__main__":
    main()
