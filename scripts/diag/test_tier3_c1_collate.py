#!/usr/bin/env python
"""Tier 3 C1: collate_episodes() unit test.

Validates the new `ppo.collate_episodes()` foundational function for
sequence-batched PPO. Does NOT touch model code — purely tests the
shape/mask machinery that C2/C3/C4 will build on.

Acceptance gates:
  1. Shapes correct: (B, L_max, *) for all leaves; (B, L_max) bool pad_mask
  2. pad_mask matches expected per-episode lengths (True at valid, False
     at padded positions)
  3. seq_lens matches input episode lengths
  4. Reduce-sum equivalence for ALL fields:
     sum(collated[field] * pad_mask) == sum(per-episode field)
     This proves padding contributes zero when masked.
  5. Round-trip equivalence: per-row valid prefix of collated[field][b, :T_b]
     equals the original episode's field
  6. Truncation: when L_max < max_T, episodes are right-truncated correctly
  7. Empty edge case: graceful error on empty input

CPU-only. Runs in ~1s. No GPU needed.

Usage:
  python scripts/diag/test_tier3_c1_collate.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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
from ppo import collate_episodes  # noqa: E402


def _make_synth_episode(T: int, A: int = 9, feat_dim: int = 4, seed: int = 0):
    """Build a synthetic episode dict matching build_ppo_episodes' output schema.
    feat_batches[t] is a per-turn dict with one tensor leaf "x" of shape (1, feat_dim)
    AND one nested dict "nested" with leaf "y" of shape (1, 2, 3) — exercise both
    flat and nested paths.
    """
    rng = np.random.RandomState(seed)
    feat_batches = []
    for t in range(T):
        feat_batches.append({
            "x": torch.randn(1, feat_dim),
            "nested": {"y": torch.randn(1, 2, 3)},
        })
    actions = list(rng.randint(0, A, size=T).astype(int))
    old_logp = list(rng.randn(T).astype(np.float32))
    advantages = list(rng.randn(T).astype(np.float32))
    returns = list(rng.randn(T).astype(np.float32))
    action_masks = [rng.uniform(0, 1, size=A).astype(np.float32) for _ in range(T)]
    return {
        "feat_batches": feat_batches,
        "actions": actions,
        "old_logp": old_logp,
        "advantages": advantages,
        "returns": returns,
        "action_masks": action_masks,
    }


def test_shapes_and_pad_mask():
    print("=== Test 1: shapes + pad_mask correctness ===")
    eps = [_make_synth_episode(5, seed=1),
           _make_synth_episode(8, seed=2),
           _make_synth_episode(12, seed=3)]
    out = collate_episodes(eps)
    B, L_max = out["B"], out["L_max"]
    A = 9
    feat_dim = 4
    assert B == 3, f"B={B}, expected 3"
    assert L_max == 12, f"L_max={L_max}, expected 12"
    # pad_mask checks
    pm = out["pad_mask"]
    assert pm.shape == (3, 12), f"pad_mask shape={pm.shape}"
    assert pm.dtype == torch.bool, f"pad_mask dtype={pm.dtype}"
    expected_pm = torch.zeros(3, 12, dtype=torch.bool)
    expected_pm[0, :5] = True
    expected_pm[1, :8] = True
    expected_pm[2, :12] = True
    assert torch.equal(pm, expected_pm), "pad_mask layout mismatch"
    # seq_lens
    assert torch.equal(out["seq_lens"], torch.tensor([5, 8, 12], dtype=torch.long))
    # Scalar-per-turn fields
    assert out["actions"].shape == (3, 12) and out["actions"].dtype == torch.long
    assert out["old_logp"].shape == (3, 12) and out["old_logp"].dtype == torch.float32
    assert out["advantages"].shape == (3, 12) and out["advantages"].dtype == torch.float32
    assert out["returns"].shape == (3, 12) and out["returns"].dtype == torch.float32
    # action_masks
    assert out["action_masks"].shape == (3, 12, A)
    # feat_batches: flat tensor leaf "x" + nested dict "nested.y"
    assert out["feat_batches"]["x"].shape == (3, 12, feat_dim)
    assert out["feat_batches"]["nested"]["y"].shape == (3, 12, 2, 3)
    print("  shapes correct, pad_mask correct  [PASS]")


def test_reduce_sum_equivalence():
    print("=== Test 2: reduce-sum equivalence on valid positions ===")
    eps = [_make_synth_episode(7, seed=10),
           _make_synth_episode(11, seed=11),
           _make_synth_episode(4, seed=12)]
    out = collate_episodes(eps)
    pm = out["pad_mask"]  # (B, L_max) bool
    pm_f = pm.float()

    # Sum per-episode for actions (sum over int values)
    expected_actions_sum = sum(sum(ep["actions"]) for ep in eps)
    got_actions_sum = (out["actions"].float() * pm_f).sum().item()
    assert abs(got_actions_sum - expected_actions_sum) < 1e-3, \
        f"actions sum mismatch: got {got_actions_sum}, expected {expected_actions_sum}"
    print(f"  actions:    {got_actions_sum:.4f} == {expected_actions_sum:.4f}  [PASS]")

    # Float fields: numerical equality up to fp32 precision
    for field in ("old_logp", "advantages", "returns"):
        expected = sum(sum(ep[field]) for ep in eps)
        got = (out[field] * pm_f).sum().item()
        assert abs(got - expected) < 1e-4, \
            f"{field} sum mismatch: got {got}, expected {expected}"
        print(f"  {field}: {got:.4f} == {expected:.4f}  [PASS]")

    # action_masks: (B, L_max, A) — sum over (B, L_max) at valid positions
    expected_am_sum = sum(sum(np.sum(m) for m in ep["action_masks"]) for ep in eps)
    pm_3d = pm_f.unsqueeze(-1)  # (B, L_max, 1)
    got_am_sum = (out["action_masks"] * pm_3d).sum().item()
    assert abs(got_am_sum - expected_am_sum) < 1e-3, \
        f"action_masks sum mismatch: got {got_am_sum}, expected {expected_am_sum}"
    print(f"  action_masks: {got_am_sum:.4f} == {expected_am_sum:.4f}  [PASS]")

    # feat_batches "x" leaf: (B, L_max, feat_dim) — reduce-sum over (B, L_max)
    expected_x_sum = sum(
        sum(ep["feat_batches"][t]["x"].sum().item() for t in range(len(ep["actions"])))
        for ep in eps)
    got_x_sum = (out["feat_batches"]["x"] * pm_3d).sum().item()
    assert abs(got_x_sum - expected_x_sum) < 1e-2, \
        f"feat_batches.x sum mismatch: got {got_x_sum}, expected {expected_x_sum}"
    print(f"  feat.x:     {got_x_sum:.4f} == {expected_x_sum:.4f}  [PASS]")

    # Nested feat_batches.nested.y: (B, L_max, 2, 3)
    expected_y_sum = sum(
        sum(ep["feat_batches"][t]["nested"]["y"].sum().item()
            for t in range(len(ep["actions"])))
        for ep in eps)
    pm_4d = pm_f.unsqueeze(-1).unsqueeze(-1)  # (B, L_max, 1, 1)
    got_y_sum = (out["feat_batches"]["nested"]["y"] * pm_4d).sum().item()
    assert abs(got_y_sum - expected_y_sum) < 1e-2, \
        f"feat_batches.nested.y sum mismatch"
    print(f"  feat.y:     {got_y_sum:.4f} == {expected_y_sum:.4f}  [PASS]")


def test_round_trip_per_row_prefix():
    print("=== Test 3: per-row valid prefix == original episode ===")
    eps = [_make_synth_episode(3, seed=20),
           _make_synth_episode(7, seed=21)]
    out = collate_episodes(eps)
    for b, ep in enumerate(eps):
        T = len(ep["actions"])
        # Actions
        assert torch.equal(
            out["actions"][b, :T],
            torch.tensor(ep["actions"], dtype=torch.long)
        ), f"actions row {b} prefix mismatch"
        # old_logp
        assert torch.allclose(
            out["old_logp"][b, :T],
            torch.tensor(ep["old_logp"], dtype=torch.float32),
            atol=1e-6
        ), f"old_logp row {b} prefix mismatch"
        # feat_batches.x
        expected_x = torch.cat([ep["feat_batches"][t]["x"] for t in range(T)], dim=0)
        assert torch.allclose(out["feat_batches"]["x"][b, :T], expected_x), \
            f"feat.x row {b} prefix mismatch"
        # feat_batches.nested.y
        expected_y = torch.cat([ep["feat_batches"][t]["nested"]["y"] for t in range(T)], dim=0)
        assert torch.allclose(out["feat_batches"]["nested"]["y"][b, :T], expected_y), \
            f"feat.y row {b} prefix mismatch"
        # Padding region is zero
        if T < out["L_max"]:
            assert (out["actions"][b, T:] == 0).all(), \
                f"row {b} actions padding not zero"
            assert (out["feat_batches"]["x"][b, T:] == 0).all(), \
                f"row {b} feat.x padding not zero"
    print("  per-row prefix matches; padding region is zero  [PASS]")


def test_truncation():
    print("=== Test 4: L_max truncation ===")
    eps = [_make_synth_episode(15, seed=30), _make_synth_episode(8, seed=31)]
    out = collate_episodes(eps, L_max=10)
    assert out["L_max"] == 10
    # First episode (T=15) gets truncated to 10; second (T=8) stays at 8
    assert torch.equal(out["seq_lens"], torch.tensor([10, 8], dtype=torch.long))
    pm = out["pad_mask"]
    assert pm[0].all(), "row 0 should be all valid (truncated to L_max)"
    assert pm[1, :8].all() and not pm[1, 8:].any()
    # Truncated row's actions should equal original episode's first 10
    assert torch.equal(
        out["actions"][0, :10],
        torch.tensor(eps[0]["actions"][:10], dtype=torch.long)
    )
    print("  truncation correct  [PASS]")


def test_empty_input():
    print("=== Test 5: empty input edge case ===")
    try:
        collate_episodes([])
        print("  FAIL: empty input should raise")
        return False
    except ValueError as e:
        print(f"  empty input raised ValueError as expected: {e}  [PASS]")
        return True


def test_device_move_cpu_passthrough():
    print("=== Test 6: device=None keeps on CPU ===")
    eps = [_make_synth_episode(5, seed=40)]
    out = collate_episodes(eps, device=None)
    assert out["actions"].device.type == "cpu"
    assert out["feat_batches"]["x"].device.type == "cpu"
    print("  CPU stays CPU  [PASS]")


def main():
    print("Tier 3 C1 unit test: collate_episodes()")
    print("=" * 60)
    test_shapes_and_pad_mask()
    print()
    test_reduce_sum_equivalence()
    print()
    test_round_trip_per_row_prefix()
    print()
    test_truncation()
    print()
    test_empty_input()
    print()
    test_device_move_cpu_passthrough()
    print()
    print("=" * 60)
    print("ALL C1 TESTS PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
