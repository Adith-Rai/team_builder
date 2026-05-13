#!/usr/bin/env python
"""Bench multiple collate_episodes variants to diagnose where the cost is.

Variants tested:
  V1:       current production (per-episode pad + cat + stack)
  V2:       pre-alloc + per-episode cat + slice-assign (no pad alloc, no pad cat)
  V3:       advanced-index scatter (the version I tried; SLOWER than V1)
  V4:       pre-alloc, slice-assign, AND bulk-convert scalar fields

Goal: figure out which (if any) variant actually beats V1 at prod scale.
"""

from __future__ import annotations

import os
import sys
import time
import gc
import statistics
import numpy as np
import torch as _t


def _setup():
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(here, "..", "..",
                                            "pokemon-ai-starter", "pokemon-ai", "src"))
    sys.path.insert(0, src_dir)
    sys.path.insert(0, here)


_setup()
from test_collate_episodes_vec_bitexact import (
    _collate_episodes_v1 as collate_v1,
    _make_synth_episode,
)
from ppo import collate_episodes as collate_v3_advindex  # noqa: E402


# ---------- Variant V2: pre-alloc + per-episode slice-assign ----------

def collate_v2(episodes, L_max=None, device=None, tail=False):
    if not episodes:
        raise ValueError("collate_episodes: empty episode list")
    full_lens_list = [len(ep["actions"]) for ep in episodes]
    if L_max is None:
        L_max = max(full_lens_list)
    if tail:
        start_idx_list = [max(0, T - L_max) for T in full_lens_list]
    else:
        start_idx_list = [0] * len(episodes)
    seq_lens_list = [min(T, L_max) for T in full_lens_list]
    B = len(episodes)
    if any(s == 0 for s in seq_lens_list):
        raise ValueError("T_actual==0")

    seq_lens = _t.tensor(seq_lens_list, dtype=_t.long)
    pad_mask = _t.arange(L_max).unsqueeze(0) < seq_lens.unsqueeze(1)

    actions = _t.zeros(B, L_max, dtype=_t.long)
    old_logp = _t.zeros(B, L_max, dtype=_t.float32)
    advantages = _t.zeros(B, L_max, dtype=_t.float32)
    returns = _t.zeros(B, L_max, dtype=_t.float32)
    for b, (ep, st, T) in enumerate(zip(episodes, start_idx_list, seq_lens_list)):
        actions[b, :T] = _t.as_tensor(ep["actions"][st:st+T], dtype=_t.long)
        old_logp[b, :T] = _t.as_tensor(ep["old_logp"][st:st+T], dtype=_t.float32)
        advantages[b, :T] = _t.as_tensor(ep["advantages"][st:st+T], dtype=_t.float32)
        returns[b, :T] = _t.as_tensor(ep["returns"][st:st+T], dtype=_t.float32)

    A = None
    for ep in episodes:
        if ep["action_masks"]:
            A = len(ep["action_masks"][0])
            break
    action_masks = _t.zeros(B, L_max, A, dtype=_t.float32)
    for b, (ep, st, T) in enumerate(zip(episodes, start_idx_list, seq_lens_list)):
        action_masks[b, :T] = _t.as_tensor(ep["action_masks"][st:st+T], dtype=_t.float32)

    sample_turn = episodes[0]["feat_batches"][start_idx_list[0]]

    def _leaf(path, sample):
        leaf_shape = tuple(sample.shape[1:])
        out = _t.zeros(B, L_max, *leaf_shape, dtype=sample.dtype)
        for b, (ep, st, T) in enumerate(zip(episodes, start_idx_list, seq_lens_list)):
            per_ep = [ep["feat_batches"][st+t] for t in range(T)]
            for k in path:
                per_ep = [d[k] for d in per_ep]
            stacked = _t.cat(per_ep, dim=0)
            out[b, :T] = stacked
        return out

    def _walk(sample, path):
        r = {}
        for k, v in sample.items():
            if isinstance(v, _t.Tensor):
                r[k] = _leaf(path + [k], v)
            elif isinstance(v, dict):
                r[k] = _walk(v, path + [k])
        return r

    feat_batches = _walk(sample_turn, [])
    return {"feat_batches": feat_batches, "actions": actions, "old_logp": old_logp,
            "advantages": advantages, "returns": returns, "action_masks": action_masks,
            "pad_mask": pad_mask, "seq_lens": seq_lens, "B": B, "L_max": L_max}


def _bench(label, fn, eps, kwargs, n_warmup=2, n_trials=10):
    for _ in range(n_warmup):
        fn(eps, **kwargs)
    gc.collect()
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        out = fn(eps, **kwargs)
        times.append(time.perf_counter() - t0)
        del out
        gc.collect()
    m = statistics.median(times)
    print(f"  {label:18s}: median={m*1000:6.2f} ms  stdev={statistics.stdev(times)*1000:5.2f} ms")
    return m


def _realistic_episode(T, A=9, feat_dim=64, n_leaves=6, seed=0):
    rng = np.random.RandomState(seed)
    feat_batches = []
    g = _t.Generator().manual_seed(seed)
    for t in range(T):
        turn = {}
        for k in range(n_leaves):
            turn[f"leaf_{k}"] = _t.randn(1, feat_dim, generator=g)
        turn["nested"] = {
            "embed_a": _t.randint(0, 1000, (1, 12), generator=g),
            "embed_b": _t.randn(1, 32, generator=g),
        }
        feat_batches.append(turn)
    return {"feat_batches": feat_batches,
            "actions": rng.randint(0, A, size=T).astype(int).tolist(),
            "old_logp": rng.randn(T).astype(np.float32).tolist(),
            "advantages": rng.randn(T).astype(np.float32).tolist(),
            "returns": rng.randn(T).astype(np.float32).tolist(),
            "action_masks": [rng.uniform(0, 1, size=A).astype(np.float32).tolist()
                             for _ in range(T)]}


def main():
    print(f"torch: {_t.__version__}")
    for label, B, Trange in [
        ("Prod chunk B=16 T~200",   16, (180, 220)),
        ("Validation B=50 T~150",   50, (130, 170)),
        ("Stress B=64 T~250",       64, (230, 270)),
    ]:
        print(f"\n--- {label} ---")
        rng = np.random.RandomState(42)
        eps = [_realistic_episode(int(rng.randint(*Trange)), seed=1000+b)
               for b in range(B)]
        kw = {"L_max": 200, "device": None, "tail": True}
        v1 = _bench("V1 current", collate_v1, eps, kw)
        v2 = _bench("V2 prealloc-slice", collate_v2, eps, kw)
        v3 = _bench("V3 adv-index", collate_v3_advindex, eps, kw)
        print(f"  V2 vs V1: {v1/v2:.2f}x   V3 vs V1: {v1/v3:.2f}x")


if __name__ == "__main__":
    main()
