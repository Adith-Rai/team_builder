#!/usr/bin/env python
"""Bit-exact A/B test for vectorized collate_episodes (S60 Fix #3).

Validates that the production `ppo.collate_episodes` produces bit-IDENTICAL
output to the frozen V1 (S55 C1) reference implementation embedded below,
across a battery of inputs covering the production call patterns.

Inputs covered:
  1. B=1, T=10 (single-episode minimum case)
  2. B=3, mixed T (3, 7, 12), L_max=default
  3. B=3, mixed T, L_max=10 (head truncation)
  4. B=3, mixed T, L_max=10, tail=True (tail truncation — S57 d1b101bb path)
  5. B=4, mixed T, L_max=200, tail=True (Tier 3 production call pattern)
  6. B=large (32), uniform T (Phase 2 chunk size)
  7. B=4, one episode at exactly L_max (no truncation, no padding)
  8. Nested dict feat_batches (the production schema)

Acceptance gate: torch.equal on every output tensor (bool, long, float).
For float tensors we use torch.equal not allclose — vectorization MUST be
bit-equivalent. Any difference triggers investigation before merge.

CPU-only; ~2-3s.

Usage:
  python scripts/diag/test_collate_episodes_vec_bitexact.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch


def _setup():
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(here, "..", "..",
                                            "pokemon-ai-starter", "pokemon-ai", "src"))
    if os.path.isdir(src_dir):
        sys.path.insert(0, src_dir)
    else:
        sys.path.insert(0, ".")


_setup()
from ppo import collate_episodes as collate_episodes_prod  # noqa: E402


# ============================================================================
# V1 reference (frozen copy of S55 C1 collate_episodes; do NOT edit)
#
# Source: f79ea477 (Tier 3 C1), updated to S57 d1b101bb with `tail=True`.
# This is the bit-exact reference. The production function MUST match this
# on every output tensor across the test battery.
# ============================================================================

def _collate_episodes_v1(episodes, L_max=None, device=None, tail: bool = False) -> dict:
    import torch as _t

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

    seq_lens = _t.tensor(seq_lens_list, dtype=_t.long)
    arange_L = _t.arange(L_max).unsqueeze(0)
    pad_mask = arange_L < seq_lens.unsqueeze(1)

    def _pad_1d(ep_list, start, T_actual, dtype, fill=0.0):
        x = _t.as_tensor(list(ep_list)[start:start + T_actual], dtype=dtype)
        if T_actual < L_max:
            pad = _t.full((L_max - T_actual,), fill, dtype=dtype)
            x = _t.cat([x, pad], dim=0)
        return x

    actions = _t.stack([_pad_1d(ep["actions"], st, s, _t.long, fill=0)
                         for ep, st, s in zip(episodes, start_idx_list, seq_lens_list)], dim=0)
    old_logp = _t.stack([_pad_1d(ep["old_logp"], st, s, _t.float32, fill=0.0)
                          for ep, st, s in zip(episodes, start_idx_list, seq_lens_list)], dim=0)
    advantages = _t.stack([_pad_1d(ep["advantages"], st, s, _t.float32, fill=0.0)
                            for ep, st, s in zip(episodes, start_idx_list, seq_lens_list)], dim=0)
    returns = _t.stack([_pad_1d(ep["returns"], st, s, _t.float32, fill=0.0)
                         for ep, st, s in zip(episodes, start_idx_list, seq_lens_list)], dim=0)

    A = None
    for ep in episodes:
        if ep["action_masks"]:
            first_m = ep["action_masks"][0]
            A = (first_m.shape[0] if hasattr(first_m, "shape")
                 else len(first_m))
            break
    if A is None:
        raise ValueError("collate_episodes: no action_masks found")

    def _pad_2d(am_list, start, T_actual, A):
        if T_actual == 0:
            stacked = _t.zeros(0, A, dtype=_t.float32)
        else:
            stacked = _t.stack([_t.as_tensor(m, dtype=_t.float32)
                                for m in am_list[start:start + T_actual]], dim=0)
        if T_actual < L_max:
            pad = _t.zeros(L_max - T_actual, A, dtype=_t.float32)
            stacked = _t.cat([stacked, pad], dim=0)
        return stacked

    action_masks = _t.stack([_pad_2d(ep["action_masks"], st, s, A)
                              for ep, st, s in zip(episodes, start_idx_list, seq_lens_list)], dim=0)

    def _stack_pad_one_episode(turn_dicts, start, T_actual):
        if T_actual == 0:
            raise ValueError("collate_episodes: T_actual==0 episode")
        sample = turn_dicts[start]
        out = {}
        for k, v in sample.items():
            if isinstance(v, _t.Tensor):
                stacked = _t.cat([turn_dicts[start + t][k]
                                  for t in range(T_actual)], dim=0)
                if T_actual < L_max:
                    pad_shape = (L_max - T_actual,) + tuple(stacked.shape[1:])
                    pad = _t.zeros(pad_shape, dtype=stacked.dtype,
                                    device=stacked.device)
                    stacked = _t.cat([stacked, pad], dim=0)
                out[k] = stacked
            elif isinstance(v, dict):
                inner_out = {}
                for inner_k, inner_v in v.items():
                    if isinstance(inner_v, _t.Tensor):
                        inner_stacked = _t.cat(
                            [turn_dicts[start + t][k][inner_k]
                             for t in range(T_actual)], dim=0)
                        if T_actual < L_max:
                            pad_shape = ((L_max - T_actual,)
                                         + tuple(inner_stacked.shape[1:]))
                            pad = _t.zeros(pad_shape, dtype=inner_stacked.dtype,
                                            device=inner_stacked.device)
                            inner_stacked = _t.cat([inner_stacked, pad], dim=0)
                        inner_out[inner_k] = inner_stacked
                out[k] = inner_out
        return out

    per_episode_collated = [
        _stack_pad_one_episode(ep["feat_batches"], st, s)
        for ep, st, s in zip(episodes, start_idx_list, seq_lens_list)
    ]

    def _stack_batch_dim(per_ep_list):
        sample = per_ep_list[0]
        out = {}
        for k, v in sample.items():
            if isinstance(v, _t.Tensor):
                out[k] = _t.stack([d[k] for d in per_ep_list], dim=0)
            elif isinstance(v, dict):
                inner_out = {}
                for inner_k in v:
                    inner_out[inner_k] = _t.stack(
                        [d[k][inner_k] for d in per_ep_list], dim=0)
                out[k] = inner_out
        return out

    feat_batches = _stack_batch_dim(per_episode_collated)

    if device is not None:
        def _to_device(d):
            r = {}
            for k, v in d.items():
                if isinstance(v, _t.Tensor):
                    r[k] = v.to(device, non_blocking=True)
                elif isinstance(v, dict):
                    r[k] = _to_device(v)
                else:
                    r[k] = v
            return r
        feat_batches = _to_device(feat_batches)
        actions = actions.to(device, non_blocking=True)
        old_logp = old_logp.to(device, non_blocking=True)
        advantages = advantages.to(device, non_blocking=True)
        returns = returns.to(device, non_blocking=True)
        action_masks = action_masks.to(device, non_blocking=True)
        pad_mask = pad_mask.to(device, non_blocking=True)
        seq_lens = seq_lens.to(device, non_blocking=True)

    return {
        "feat_batches": feat_batches,
        "actions": actions,
        "old_logp": old_logp,
        "advantages": advantages,
        "returns": returns,
        "action_masks": action_masks,
        "pad_mask": pad_mask,
        "seq_lens": seq_lens,
        "B": B,
        "L_max": L_max,
    }


# ============================================================================
# Synthetic episode generator — matches build_ppo_episodes output schema
# ============================================================================

def _make_synth_episode(T: int, A: int = 9, feat_dim: int = 4, seed: int = 0,
                       nested: bool = True):
    """Build a synthetic episode dict matching build_ppo_episodes' output:
      actions, old_logp, advantages, returns: Python lists
      action_masks: list of length-A Python lists (from m.tolist())
      feat_batches: list of T per-turn dicts with (1, *) tensor leaves
    """
    rng = np.random.RandomState(seed)
    feat_batches = []
    for t in range(T):
        turn = {"x": torch.randn(1, feat_dim, generator=torch.Generator().manual_seed(seed * 1000 + t))}
        if nested:
            turn["nested"] = {"y": torch.randn(1, 2, 3,
                                                generator=torch.Generator().manual_seed(seed * 1000 + t + 500))}
        feat_batches.append(turn)
    actions = list(rng.randint(0, A, size=T).astype(int).tolist())
    old_logp = list(rng.randn(T).astype(np.float32).tolist())
    advantages = list(rng.randn(T).astype(np.float32).tolist())
    returns = list(rng.randn(T).astype(np.float32).tolist())
    action_masks = [rng.uniform(0, 1, size=A).astype(np.float32).tolist() for _ in range(T)]
    return {
        "feat_batches": feat_batches,
        "actions": actions,
        "old_logp": old_logp,
        "advantages": advantages,
        "returns": returns,
        "action_masks": action_masks,
    }


# ============================================================================
# Bit-exact comparison helpers
# ============================================================================

def _assert_tensor_bit_equal(a: torch.Tensor, b: torch.Tensor, name: str):
    assert a.shape == b.shape, f"{name}: shape mismatch {a.shape} vs {b.shape}"
    assert a.dtype == b.dtype, f"{name}: dtype mismatch {a.dtype} vs {b.dtype}"
    assert a.device == b.device, f"{name}: device mismatch {a.device} vs {b.device}"
    if not torch.equal(a, b):
        diff = (a.float() - b.float()).abs()
        max_diff = diff.max().item() if diff.numel() > 0 else 0.0
        n_diff = (a != b).sum().item() if a.numel() > 0 else 0
        raise AssertionError(
            f"{name}: NOT bit-equal. n_diff={n_diff}/{a.numel()}, max_abs_diff={max_diff}")


def _assert_collated_bit_equal(a: dict, b: dict, case: str):
    # Scalar fields
    assert a["B"] == b["B"], f"{case}: B mismatch"
    assert a["L_max"] == b["L_max"], f"{case}: L_max mismatch"
    # Tensor fields at top level
    for k in ("actions", "old_logp", "advantages", "returns",
              "action_masks", "pad_mask", "seq_lens"):
        _assert_tensor_bit_equal(a[k], b[k], f"{case}.{k}")
    # feat_batches: recurse
    def _recurse(da, db, path):
        assert set(da.keys()) == set(db.keys()), f"{path}: key mismatch {da.keys()} vs {db.keys()}"
        for k, va in da.items():
            vb = db[k]
            if isinstance(va, torch.Tensor):
                _assert_tensor_bit_equal(va, vb, f"{path}.{k}")
            elif isinstance(va, dict):
                _recurse(va, vb, f"{path}.{k}")
            else:
                assert va == vb, f"{path}.{k}: scalar mismatch"
    _recurse(a["feat_batches"], b["feat_batches"], f"{case}.feat_batches")


# ============================================================================
# Test cases
# ============================================================================

def _run_case(label: str, episodes, **kwargs):
    v1 = _collate_episodes_v1(episodes, **kwargs)
    prod = collate_episodes_prod(episodes, **kwargs)
    _assert_collated_bit_equal(v1, prod, label)
    print(f"  {label}: bit-equal across all fields  [PASS]")


def test_case_1_single_episode():
    eps = [_make_synth_episode(10, seed=100)]
    _run_case("Case 1 (B=1, T=10)", eps)


def test_case_2_mixed_T_default_Lmax():
    eps = [_make_synth_episode(3, seed=200),
           _make_synth_episode(7, seed=201),
           _make_synth_episode(12, seed=202)]
    _run_case("Case 2 (B=3, mixed T, default L_max)", eps)


def test_case_3_head_truncation():
    eps = [_make_synth_episode(15, seed=300),
           _make_synth_episode(7, seed=301),
           _make_synth_episode(20, seed=302)]
    _run_case("Case 3 (B=3, L_max=10, head trunc)", eps, L_max=10)


def test_case_4_tail_truncation():
    """S57 d1b101bb path: tail=True is load-bearing for forward_ppo_sequence."""
    eps = [_make_synth_episode(15, seed=400),
           _make_synth_episode(7, seed=401),
           _make_synth_episode(20, seed=402)]
    _run_case("Case 4 (B=3, L_max=10, tail=True)", eps, L_max=10, tail=True)


def test_case_5_production_call_pattern():
    """Mimics the Tier 3 production call: L_max=temporal_context (200), tail=True.
    Some episodes shorter than 200 (no trunc), some equal (T=L_max), some longer (tail trunc)."""
    eps = [_make_synth_episode(150, seed=500),
           _make_synth_episode(200, seed=501),
           _make_synth_episode(250, seed=502),
           _make_synth_episode(50, seed=503)]
    _run_case("Case 5 (B=4, L_max=200, tail=True — prod pattern)", eps,
              L_max=200, tail=True)


def test_case_6_large_B_uniform_T():
    """Phase 2 chunk size — 32 episodes of moderate T."""
    eps = [_make_synth_episode(80, seed=600 + b) for b in range(32)]
    _run_case("Case 6 (B=32, uniform T=80)", eps)


def test_case_7_T_exactly_Lmax():
    """No padding, no truncation — all episodes exactly at L_max."""
    eps = [_make_synth_episode(50, seed=700 + b) for b in range(4)]
    _run_case("Case 7 (B=4, T=L_max=50 exactly)", eps, L_max=50)


def test_case_8_no_nested():
    """feat_batches with only flat tensor leaves (no nested dict). Validates
    the non-nested code path."""
    eps = [_make_synth_episode(8, seed=800, nested=False),
           _make_synth_episode(12, seed=801, nested=False)]
    _run_case("Case 8 (B=2, flat feat_batches only)", eps)


def test_case_9_device_cpu_passthrough():
    """device=None should leave tensors on CPU. Test device-arg path explicitly."""
    eps = [_make_synth_episode(6, seed=900),
           _make_synth_episode(10, seed=901)]
    v1 = _collate_episodes_v1(eps, device=None)
    prod = collate_episodes_prod(eps, device=None)
    _assert_collated_bit_equal(v1, prod, "Case 9 (device=None)")
    print("  Case 9 (device=None): bit-equal  [PASS]")


def test_case_10_all_tail_truncated():
    """Every episode is longer than L_max — every one tail-truncated."""
    eps = [_make_synth_episode(300, seed=1000 + b) for b in range(4)]
    _run_case("Case 10 (B=4, all T > L_max=200, tail)", eps, L_max=200, tail=True)


def main():
    print("=" * 64)
    print("Bit-exact A/B: collate_episodes V1 (frozen) vs production")
    print("=" * 64)
    test_case_1_single_episode()
    test_case_2_mixed_T_default_Lmax()
    test_case_3_head_truncation()
    test_case_4_tail_truncation()
    test_case_5_production_call_pattern()
    test_case_6_large_B_uniform_T()
    test_case_7_T_exactly_Lmax()
    test_case_8_no_nested()
    test_case_9_device_cpu_passthrough()
    test_case_10_all_tail_truncated()
    print("=" * 64)
    print("ALL BIT-EQUAL — production collate_episodes matches V1 reference")
    print("=" * 64)


if __name__ == "__main__":
    main()
