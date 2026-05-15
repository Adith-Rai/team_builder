"""Equivalence tests for collate_episodes_packed vs legacy collate_episodes.

S64 Phase A acceptance gate: for any input `episodes` and matching
(max_seqlen, tail, device) args, the packed output unpacked via
cu_seqlens equals the unpadded prefix of legacy collate_episodes.

Cases covered:
  - Uniform episode lengths (no padding in legacy path)
  - Varied episode lengths (padding in legacy path)
  - tail=True truncation (max_seqlen < T → keep last max_seqlen)
  - tail=False no-truncation (max_seqlen ≥ max(T) → identity)
  - max_seqlen=None default (caps at max(T))
  - Nested-dict feat_batches (recursion path)
  - Single-episode B=1
  - Empty list raises ValueError
  - T_actual=0 raises ValueError

Run: python test_collate_packed.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from ppo import collate_episodes, collate_episodes_packed


# ---------------------------------------------------------------- helpers

def _make_episode(T: int, A: int = 10, feat_dims=(6, 5), seed: int = 0,
                   nested: bool = False) -> dict:
    """Construct a minimal episode dict matching the build_ppo_episodes output.

    Mirrors the actual production contract:
      - feat_batches: list of T per-turn dicts. Each leaf tensor is shape (1, *).
        When `nested=True`, includes one level of nested dict to exercise
        the recursion path.
      - actions/old_logp/advantages/returns: python lists of length T
        (build_ppo_episodes converts numpy → .tolist() before returning).
      - action_masks: list of T python lists of length A (build_ppo_episodes
        does `[m.tolist() for m in traj.action_masks]`).
    """
    rng = np.random.RandomState(seed)
    g = torch.Generator().manual_seed(seed)

    feat_batches = []
    for t in range(T):
        turn = {"obs": torch.randn(1, *feat_dims, generator=g)}
        if nested:
            turn["inner"] = {
                "ids": torch.randint(0, 100, (1, 4), generator=g),
                "weights": torch.randn(1, 3, generator=g),
            }
        feat_batches.append(turn)

    return {
        "feat_batches":  feat_batches,
        "actions":       rng.randint(0, A, size=T).tolist(),
        "old_logp":      rng.randn(T).astype(np.float32).tolist(),
        "advantages":    rng.randn(T).astype(np.float32).tolist(),
        "returns":       rng.randn(T).astype(np.float32).tolist(),
        "action_masks":  [rng.choice([0.0, 1.0], size=A).tolist()
                          for _ in range(T)],
    }


def _check_equivalence(legacy: dict, packed: dict, nested: bool = False) -> None:
    """Assert packed[k][cu[b]:cu[b+1]] == legacy[k][b, :seq_lens[b]] for every k, b.

    Covers all returned scalar/2d/feat-batch fields. Raises AssertionError on
    any mismatch.
    """
    B = packed["B"]
    assert B == legacy["B"], f"B mismatch: packed={B} legacy={legacy['B']}"

    cu = packed["cu_seqlens"]
    seq_lens_p = packed["seq_lens"]
    seq_lens_l = legacy["seq_lens"]

    # cu_seqlens must be int32, monotone non-decreasing, start at 0,
    # end at sum_T = sum(seq_lens).
    assert cu.dtype == torch.int32, f"cu_seqlens dtype {cu.dtype} != int32"
    assert cu.shape == (B + 1,), f"cu_seqlens shape {cu.shape} != ({B+1},)"
    assert int(cu[0]) == 0
    for b in range(B):
        seg = int(cu[b + 1]) - int(cu[b])
        assert seg == int(seq_lens_p[b]), \
            f"cu_seqlens segment {b}: cu_diff={seg} seq_lens={int(seq_lens_p[b])}"

    # max_seqlen agrees with legacy L_max only when both bundles have at
    # least one episode hitting the cap. When max_seqlen is None on both
    # sides, both should equal max(seq_lens).
    assert packed["max_seqlen"] == int(seq_lens_p.max())

    # seq_lens equality across legacy + packed
    assert torch.equal(seq_lens_p, seq_lens_l), \
        f"seq_lens diverged: packed={seq_lens_p.tolist()} legacy={seq_lens_l.tolist()}"

    # Scalar-per-turn fields
    for key in ("actions", "old_logp", "advantages", "returns"):
        for b in range(B):
            sl = int(seq_lens_p[b])
            start, stop = int(cu[b]), int(cu[b + 1])
            packed_slice = packed[key][start:stop]
            legacy_slice = legacy[key][b, :sl]
            assert torch.equal(packed_slice, legacy_slice), (
                f"{key} mismatch at b={b}: "
                f"packed[{start}:{stop}]={packed_slice.tolist()} "
                f"legacy[{b},:{sl}]={legacy_slice.tolist()}"
            )

    # action_masks (sum_T, A)
    for b in range(B):
        sl = int(seq_lens_p[b])
        start, stop = int(cu[b]), int(cu[b + 1])
        packed_slice = packed["action_masks"][start:stop]
        legacy_slice = legacy["action_masks"][b, :sl]
        assert torch.equal(packed_slice, legacy_slice), \
            f"action_masks mismatch at b={b}"

    # feat_batches (top-level)
    for k, v in packed["flat_feat_batches"].items():
        if isinstance(v, torch.Tensor):
            for b in range(B):
                sl = int(seq_lens_p[b])
                start, stop = int(cu[b]), int(cu[b + 1])
                packed_slice = v[start:stop]
                legacy_slice = legacy["feat_batches"][k][b, :sl]
                assert torch.equal(packed_slice, legacy_slice), \
                    f"feat_batches[{k}] mismatch at b={b}"
        elif isinstance(v, dict):
            assert nested, f"unexpected nested dict at key {k}"
            for inner_k, inner_v in v.items():
                for b in range(B):
                    sl = int(seq_lens_p[b])
                    start, stop = int(cu[b]), int(cu[b + 1])
                    packed_slice = inner_v[start:stop]
                    legacy_slice = legacy["feat_batches"][k][inner_k][b, :sl]
                    assert torch.equal(packed_slice, legacy_slice), \
                        f"feat_batches[{k}][{inner_k}] mismatch at b={b}"


# ---------------------------------------------------------------- tests

def test_uniform_lengths():
    """All episodes same T → no padding in legacy, packed should match exactly."""
    eps = [_make_episode(T=10, seed=i) for i in range(3)]
    legacy = collate_episodes(eps)
    packed = collate_episodes_packed(eps)
    _check_equivalence(legacy, packed)
    # Sum of packed turns equals B*T
    assert int(packed["cu_seqlens"][-1]) == 3 * 10


def test_varied_lengths():
    """Mixed episode lengths → legacy pads, packed doesn't.
    Unpacked-prefix equivalence must still hold."""
    Ts = [5, 12, 8, 3, 12]
    eps = [_make_episode(T=T, seed=i) for i, T in enumerate(Ts)]
    legacy = collate_episodes(eps)
    packed = collate_episodes_packed(eps)
    _check_equivalence(legacy, packed)
    assert int(packed["cu_seqlens"][-1]) == sum(Ts)
    assert packed["max_seqlen"] == max(Ts) == legacy["L_max"]


def test_tail_truncation():
    """Episodes longer than max_seqlen with tail=True → keep last max_seqlen turns.

    Both legacy and packed must produce the same TAIL slice. (Legacy stores
    that tail at positions [b, :seq_lens[b]] after tail-shifting, which is
    [b, :max_seqlen] when T > max_seqlen.)
    """
    eps = [_make_episode(T=20, seed=0), _make_episode(T=15, seed=1)]
    legacy = collate_episodes(eps, L_max=10, tail=True)
    packed = collate_episodes_packed(eps, max_seqlen=10, tail=True)
    _check_equivalence(legacy, packed)
    assert int(packed["seq_lens"][0]) == 10
    assert int(packed["seq_lens"][1]) == 10
    assert int(packed["cu_seqlens"][-1]) == 20  # 10 + 10


def test_tail_no_truncation_needed():
    """tail=True but max_seqlen >= all T → no truncation, identity behavior."""
    eps = [_make_episode(T=5, seed=0), _make_episode(T=8, seed=1)]
    legacy = collate_episodes(eps, L_max=20, tail=True)
    packed = collate_episodes_packed(eps, max_seqlen=20, tail=True)
    _check_equivalence(legacy, packed)


def test_no_tail_default():
    """tail=False → take first max_seqlen turns of overlong episodes."""
    eps = [_make_episode(T=15, seed=0), _make_episode(T=10, seed=1)]
    legacy = collate_episodes(eps, L_max=8, tail=False)
    packed = collate_episodes_packed(eps, max_seqlen=8, tail=False)
    _check_equivalence(legacy, packed)


def test_nested_dict_feat_batches():
    """feat_batches with one level of nested dict — exercises recursion path."""
    eps = [_make_episode(T=T, seed=i, nested=True)
           for i, T in enumerate([7, 11, 4])]
    legacy = collate_episodes(eps)
    packed = collate_episodes_packed(eps)
    _check_equivalence(legacy, packed, nested=True)
    # Sanity: nested keys present in packed output
    assert "inner" in packed["flat_feat_batches"]
    assert "ids" in packed["flat_feat_batches"]["inner"]
    assert "weights" in packed["flat_feat_batches"]["inner"]


def test_single_episode():
    """B=1 — both BC anchor reference forward and degenerate-batch path."""
    eps = [_make_episode(T=17, seed=42)]
    legacy = collate_episodes(eps)
    packed = collate_episodes_packed(eps)
    _check_equivalence(legacy, packed)
    assert packed["B"] == 1
    assert int(packed["cu_seqlens"][-1]) == 17
    # cu_seqlens for B=1: [0, T]
    assert packed["cu_seqlens"].tolist() == [0, 17]


def test_empty_list_raises():
    try:
        collate_episodes_packed([])
    except ValueError as e:
        assert "empty" in str(e)
        return
    raise AssertionError("expected ValueError on empty episode list")


def test_zero_length_episode_raises():
    """T=0 episode after truncation should raise (matches legacy semantics)."""
    eps = [_make_episode(T=5, seed=0)]
    # Force T_actual = 0 by setting actions list to empty
    eps[0]["actions"] = []
    eps[0]["old_logp"] = []
    eps[0]["advantages"] = []
    eps[0]["returns"] = []
    eps[0]["action_masks"] = []
    eps[0]["feat_batches"] = []
    try:
        collate_episodes_packed(eps)
    except ValueError as e:
        assert "T_actual==0" in str(e)
        return
    raise AssertionError("expected ValueError on zero-length episode")


def test_max_seqlen_int_correctness():
    """max_seqlen field in return should equal max(seq_lens), not the cap."""
    eps = [_make_episode(T=T, seed=i) for i, T in enumerate([5, 8, 3])]
    # max_seqlen arg = 20 (no truncation happens since all T < 20)
    packed = collate_episodes_packed(eps, max_seqlen=20)
    assert packed["max_seqlen"] == 8  # max actual T after no-op truncation


def test_device_move_cpu_noop():
    """device=cpu should be a no-op move (tensors stay on cpu)."""
    eps = [_make_episode(T=6, seed=0), _make_episode(T=9, seed=1)]
    legacy = collate_episodes(eps, device=torch.device("cpu"))
    packed = collate_episodes_packed(eps, device=torch.device("cpu"))
    _check_equivalence(legacy, packed)
    assert packed["cu_seqlens"].device.type == "cpu"
    assert packed["seq_lens"].device.type == "cpu"


# ---------------------------------------------------------------- driver

def _run_all():
    tests = [
        test_uniform_lengths,
        test_varied_lengths,
        test_tail_truncation,
        test_tail_no_truncation_needed,
        test_no_tail_default,
        test_nested_dict_feat_batches,
        test_single_episode,
        test_empty_list_raises,
        test_zero_length_episode_raises,
        test_max_seqlen_int_correctness,
        test_device_move_cpu_noop,
    ]
    failures = []
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  PASS  {name}")
        except Exception as e:
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            failures.append((name, e))
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(_run_all())
