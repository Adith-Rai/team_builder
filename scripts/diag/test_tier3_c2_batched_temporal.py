#!/usr/bin/env python
"""Tier 3 C2: TemporalTransformer batched-vs-sequential equivalence test.

Validates the new `return_all_positions=True` mode of TemporalTransformer.
This is the ONLY new algorithmic component in Tier 3 C2 — `forward_ppo_sequence`
itself is just composition of (existing forward_spatial + this batched temporal
+ existing action_head + existing value_head). If batched temporal is
bit-equivalent to sequential at valid positions, then forward_ppo_sequence is
correct by composition.

Acceptance gate (the load-bearing one):
  For all valid (b, t) where t < seq_lens[b]:
    TemporalTransformer(summaries, seq_lens, return_all_positions=True)[b, t]
    == TemporalTransformer(summaries[b:b+1, :t+1])[0]
  within fp32 precision tolerance.

Why this matters: C4's PPO update will compute losses at every (b, t) using
the BATCHED temporal forward. If batched outputs differ from per-t outputs at
valid positions, gradients would be wrong → training corrupted. This test
proves the batched path is mathematically equivalent.

Padding positions are NOT tested (they're fillers — caller masks them in loss).

CPU-only. Runs in ~3-5s. No GPU needed.

Usage:
  python scripts/diag/test_tier3_c2_batched_temporal.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _setup():
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(here, "..", "..",
                                            "pokemon-ai-starter", "pokemon-ai", "src"))
    if os.path.isdir(src_dir):
        sys.path.insert(0, src_dir)
    else:
        sys.path.insert(0, ".")


_setup()
from model_transformer import TemporalTransformer, TransformerConfig  # noqa: E402


def _make_temporal(d_temporal: int = 32, n_layers: int = 2, n_heads: int = 4,
                    temporal_context: int = 100) -> TemporalTransformer:
    """Build a minimal TemporalTransformer for testing. Small dims for speed."""
    # TransformerConfig has many required fields; use a minimal subclass-compatible
    # construction by passing the attributes the temporal stack reads.
    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.d_temporal = d_temporal
    cfg.n_heads = n_heads
    cfg.ff_mult = 4
    cfg.dropout = 0.0
    cfg.n_temporal_layers = n_layers
    cfg.temporal_context = temporal_context
    cfg.init_std = 0.02
    cfg.gradient_checkpoint = False
    return TemporalTransformer(cfg)


def test_batched_equals_sequential_no_seqlens():
    """Without seq_lens, batched output at every t should equal sequential
    forward on summaries[:b+1, :t+1] returning last position.

    NB: 'no seq_lens' means all positions valid. The batched forward returns
    (B, T, D) — output[b, t] should equal sequential forward on the prefix
    summaries[b:b+1, :t+1] taking its LAST position output.
    """
    print("=== Test 1: batched equiv sequential (no seq_lens, all-valid) ===")
    torch.manual_seed(0)
    temporal = _make_temporal(d_temporal=32, n_layers=2)
    temporal.eval()

    B, T, D = 3, 8, 32
    summaries = torch.randn(B, T, D)

    # Batched forward: (B, T, D) outputs
    with torch.no_grad():
        all_pos = temporal(summaries, seq_lens=None, return_all_positions=True)
    assert all_pos.shape == (B, T, D), f"shape {all_pos.shape}"

    # Sequential forward: for each (b, t), run on summaries[b:b+1, :t+1]
    # and compare to all_pos[b, t].
    n_compared = 0
    max_diff = 0.0
    for b in range(B):
        for t in range(T):
            with torch.no_grad():
                seq_out = temporal(
                    summaries[b:b+1, :t+1], seq_lens=None
                )  # (1, D) — last position
            diff = (all_pos[b, t] - seq_out[0]).abs().max().item()
            max_diff = max(max_diff, diff)
            n_compared += 1
    print(f"  compared {n_compared} (b, t) positions; max abs diff = {max_diff:.2e}")
    assert max_diff < 1e-5, f"max diff {max_diff} exceeds 1e-5"
    print("  batched matches sequential at all positions  [PASS]")


def test_batched_equals_sequential_with_seqlens():
    """With seq_lens, batched output at valid positions (t < seq_lens[b])
    should equal sequential forward on the valid prefix.

    Padding positions are NOT tested (fillers; padding_mask in temporal
    attention prevents them from polluting valid positions, but their own
    output is meaningless and depends on init noise of the padding inputs).
    """
    print("=== Test 2: batched equiv sequential (with seq_lens, padded) ===")
    torch.manual_seed(1)
    temporal = _make_temporal(d_temporal=32, n_layers=2)
    temporal.eval()

    B, L_max, D = 4, 12, 32
    seq_lens_list = [5, 8, 12, 3]  # episode lengths
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.long)
    summaries = torch.randn(B, L_max, D)

    # Batched forward
    with torch.no_grad():
        all_pos = temporal(summaries, seq_lens=seq_lens, return_all_positions=True)
    assert all_pos.shape == (B, L_max, D)

    # Sequential at valid positions only
    n_compared = 0
    max_diff = 0.0
    failed_positions = []
    for b in range(B):
        T_b = seq_lens_list[b]
        for t in range(T_b):
            with torch.no_grad():
                seq_out = temporal(summaries[b:b+1, :t+1], seq_lens=None)
            diff = (all_pos[b, t] - seq_out[0]).abs().max().item()
            max_diff = max(max_diff, diff)
            if diff > 1e-5:
                failed_positions.append((b, t, diff))
            n_compared += 1
    print(f"  compared {n_compared} valid (b, t) positions; "
          f"max abs diff = {max_diff:.2e}")
    if failed_positions:
        print(f"  FAIL: {len(failed_positions)} positions exceed 1e-5:")
        for b, t, d in failed_positions[:5]:
            print(f"    (b={b}, t={t}): diff={d:.2e}")
    assert not failed_positions, \
        f"{len(failed_positions)} valid positions exceed tolerance"
    print("  batched matches sequential at all valid positions  [PASS]")


def test_default_mode_unchanged():
    """Without return_all_positions=True, behavior must be unchanged
    (returns last valid position only). Backward compat for existing callers
    (inference, BC training per-t loop)."""
    print("=== Test 3: default mode (last position only) unchanged ===")
    torch.manual_seed(2)
    temporal = _make_temporal(d_temporal=32, n_layers=2)
    temporal.eval()

    B, L_max, D = 3, 6, 32
    seq_lens = torch.tensor([6, 4, 2], dtype=torch.long)
    summaries = torch.randn(B, L_max, D)

    with torch.no_grad():
        last_pos = temporal(summaries, seq_lens=seq_lens)  # default
        all_pos = temporal(summaries, seq_lens=seq_lens, return_all_positions=True)

    assert last_pos.shape == (B, D), f"default shape {last_pos.shape}"
    assert all_pos.shape == (B, L_max, D), f"all-pos shape {all_pos.shape}"

    # Default last_pos[b] should equal all_pos[b, seq_lens[b]-1]
    for b in range(B):
        idx = seq_lens[b].item() - 1
        diff = (last_pos[b] - all_pos[b, idx]).abs().max().item()
        assert diff < 1e-6, f"row {b}: default vs all-pos[seq_lens-1] diff={diff}"
    print("  default mode = all_pos[b, seq_lens[b]-1]  [PASS]")


def test_causal_mask_no_future_leak():
    """Sanity: changing future positions must NOT change current position's
    output. Tests the causal mask is actually causal."""
    print("=== Test 4: causal mask — no future-info leak ===")
    torch.manual_seed(3)
    temporal = _make_temporal(d_temporal=32, n_layers=2)
    temporal.eval()

    B, T, D = 1, 8, 32
    summaries = torch.randn(B, T, D)

    with torch.no_grad():
        out_orig = temporal(summaries, seq_lens=None, return_all_positions=True)

    # Mutate position T-1 (the future) and re-run; positions 0..T-2 outputs
    # must be unchanged (causal mask blocks future-attention).
    summaries_mod = summaries.clone()
    summaries_mod[0, -1] = torch.randn(D) * 100  # large perturbation at t=T-1

    with torch.no_grad():
        out_mod = temporal(summaries_mod, seq_lens=None, return_all_positions=True)

    # Positions 0..T-2 should be identical
    diff = (out_orig[0, :-1] - out_mod[0, :-1]).abs().max().item()
    print(f"  positions [0..{T-2}] max abs diff after future perturbation: "
          f"{diff:.2e}")
    assert diff < 1e-6, f"causal mask LEAKS — positions 0..{T-2} changed by {diff}"
    # Position T-1 SHOULD change (it sees its own input)
    diff_last = (out_orig[0, -1] - out_mod[0, -1]).abs().max().item()
    print(f"  position {T-1} changed (as expected): {diff_last:.2e}")
    assert diff_last > 0.01, \
        f"position {T-1} did not change after self-input perturbation"
    print("  causal mask is causal — no future leak, current input affects output  [PASS]")


def test_temporal_context_truncation():
    """If T > temporal_context, summaries are right-truncated to keep the
    last `temporal_context` positions."""
    print("=== Test 5: temporal_context truncation ===")
    torch.manual_seed(4)
    temporal = _make_temporal(d_temporal=16, n_layers=1, temporal_context=5)
    temporal.eval()

    B, T, D = 2, 10, 16  # T > temporal_context=5
    summaries = torch.randn(B, T, D)

    with torch.no_grad():
        out = temporal(summaries, seq_lens=None, return_all_positions=True)
    # After truncation, T effectively = 5
    assert out.shape == (B, 5, D), f"truncated shape {out.shape}"
    print(f"  T=10 truncated to {out.shape[1]} (cap=5)  [PASS]")


def main():
    print("Tier 3 C2 unit test: TemporalTransformer batched temporal forward")
    print("=" * 70)
    test_batched_equals_sequential_no_seqlens()
    print()
    test_batched_equals_sequential_with_seqlens()
    print()
    test_default_mode_unchanged()
    print()
    test_causal_mask_no_future_leak()
    print()
    test_temporal_context_truncation()
    print()
    print("=" * 70)
    print("ALL C2 TEMPORAL EQUIVALENCE TESTS PASS")
    print("=" * 70)
    print()
    print("forward_ppo_sequence (composition of forward_spatial + batched")
    print("temporal + heads) is correct by composition. Full end-to-end")
    print("validation will happen in C4 + C6 (20-iter A/B vs Phase 1 v3).")


if __name__ == "__main__":
    main()
