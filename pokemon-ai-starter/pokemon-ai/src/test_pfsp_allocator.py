"""Unit tests for the PFSP allocator (legacy + balanced).

The balanced allocator is the S64 polish of S58 narrative §6's documented
imbalance bug. These tests cover:
  - The S58 imbalance reproduction (legacy IS imbalanced; balanced is NOT)
  - Standard cases at various (n_workers, n_opps) configurations
  - Edge cases (0 workers, 0 opps, 1 worker)
  - Game-count distribution invariants for the balanced version

Run with:
  python test_pfsp_allocator.py
"""
from __future__ import annotations

import sys

from mp_centralized_collect import (
    _allocate_opps_to_workers_legacy,
    _allocate_opps_to_workers_balanced,
)


def _opps(weights):
    return [
        {"key": f"opp{i}", "path": f"opp{i}.pt", "weight": w}
        for i, w in enumerate(weights)
    ]


def _per_worker_games(assignments):
    return [a["n_games"] for a in assignments]


def _per_opp_games(assignments):
    by_opp = {}
    for a in assignments:
        opp = a.get("opp")
        if opp is None:
            continue
        by_opp.setdefault(opp["key"], 0)
        by_opp[opp["key"]] += a["n_games"]
    return by_opp


# ----------------------------------------------------------------------------
# Core property tests for the balanced allocator
# ----------------------------------------------------------------------------

def test_balanced_perworker_max_minus_min_le_1():
    """Core invariant: max(per_worker) - min(per_worker) <= 1."""
    cases = [
        # (n_workers, n_opps_with_weight, total_games)
        (8, [1, 1, 1, 1, 1], 1600),    # the S58 imbalance case
        (15, [1, 1, 1, 1, 1], 1600),   # 15w uniform 5 opps
        (8, [1, 1, 1, 1, 1, 1, 1, 1], 1600),  # 8w/8opp natural balance
        (8, [1, 1, 1], 1600),          # 8w/3opp
        (16, [1] * 8, 1600),           # 16w/8opp
        (8, [5, 3, 2, 1, 1], 1600),    # 8w/5opp skewed
        (15, [10, 1, 1, 1], 1500),     # 15w/4opp extreme skew
        (8, [1, 1, 1, 1, 1], 1601),    # non-divisible games
    ]
    for n_workers, weights, total in cases:
        assignments = _allocate_opps_to_workers_balanced(
            _opps(weights), n_workers, total
        )
        assert len(assignments) == n_workers, (
            f"n_workers={n_workers}: got {len(assignments)} assignments"
        )
        pwg = _per_worker_games(assignments)
        spread = max(pwg) - min(pwg)
        assert spread <= 1, (
            f"n_workers={n_workers}, weights={weights}, total={total}: "
            f"per-worker games {pwg} has spread {spread} > 1"
        )
        # Total games match
        assert sum(pwg) == total, (
            f"n_workers={n_workers}, weights={weights}, total={total}: "
            f"sum(per-worker) = {sum(pwg)} != total {total}"
        )


def test_balanced_s58_imbalance_repro():
    """The S58 narrative §6 case: 8w/5opp uniform/1600g.
    Legacy gives [160x6, 320x2] (max=320). Balanced gives [200x8] (max=200).
    """
    opp_pool = _opps([1, 1, 1, 1, 1])
    legacy = _allocate_opps_to_workers_legacy(opp_pool, 8, 1600)
    balanced = _allocate_opps_to_workers_balanced(opp_pool, 8, 1600)

    legacy_pwg = sorted(_per_worker_games(legacy))
    balanced_pwg = sorted(_per_worker_games(balanced))

    # Legacy IS imbalanced (max-min > 1 in this regime).
    assert max(legacy_pwg) - min(legacy_pwg) > 1, (
        f"legacy should be imbalanced, got {legacy_pwg}"
    )
    # Specifically the S58 documented case:
    assert legacy_pwg == [160, 160, 160, 160, 160, 160, 320, 320], (
        f"legacy unexpected: {legacy_pwg}"
    )
    # Balanced is balanced (max-min == 0 for 1600/8 = 200 exact).
    assert balanced_pwg == [200] * 8, (
        f"balanced unexpected: {balanced_pwg}"
    )
    # Both still cover all 5 opps (S58 F constraint).
    assert len(_per_opp_games(legacy)) == 5
    assert len(_per_opp_games(balanced)) == 5


def test_balanced_natural_balance_8w8opp():
    """8w/8opp/1600g — each worker gets one opp, 200 games.
    Both legacy and balanced should produce identical assignments here.
    """
    opp_pool = _opps([1, 1, 1, 1, 1, 1, 1, 1])
    legacy = _allocate_opps_to_workers_legacy(opp_pool, 8, 1600)
    balanced = _allocate_opps_to_workers_balanced(opp_pool, 8, 1600)

    legacy_pwg = sorted(_per_worker_games(legacy))
    balanced_pwg = sorted(_per_worker_games(balanced))
    assert legacy_pwg == [200] * 8
    assert balanced_pwg == [200] * 8


def test_balanced_15w5opp_uniform():
    """15w/5opp/1600g — natural split 3 workers per opp, 1600/15 = 106 rem 10."""
    opp_pool = _opps([1, 1, 1, 1, 1])
    balanced = _allocate_opps_to_workers_balanced(opp_pool, 15, 1600)
    pwg = sorted(_per_worker_games(balanced))
    # 1600 = 15 * 106 + 10 → 10 workers get 107, 5 workers get 106.
    assert pwg == [106] * 5 + [107] * 10, f"got {pwg}"
    # Each opp gets 3 workers, 3 × (mix of 106 and 107) games.
    poo = _per_opp_games(balanced)
    assert len(poo) == 5
    # Each opp's total is in range [3*106, 3*107] = [318, 321]
    for opp, games in poo.items():
        assert 318 <= games <= 321, f"opp {opp} got {games} games"


def test_balanced_n_workers_lt_n_opps():
    """8w/12opp/1600g — top-8 by weight, each plays 200 games. 4 opps skipped.
    Weight order matters; uniform weights → ties broken by index.
    """
    opp_pool = _opps([1] * 12)
    balanced = _allocate_opps_to_workers_balanced(opp_pool, 8, 1600)
    pwg = _per_worker_games(balanced)
    # All workers exactly 200 games (1600 / 8 = 200 exact).
    assert sorted(pwg) == [200] * 8
    # Only 8 opps are played; 4 skipped.
    poo = _per_opp_games(balanced)
    assert len(poo) == 8


def test_balanced_n_workers_lt_n_opps_skewed():
    """Skewed weights — top-N picks highest weights."""
    weights = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]  # 10 opps
    opp_pool = _opps(weights)
    balanced = _allocate_opps_to_workers_balanced(opp_pool, 4, 1600)
    poo = _per_opp_games(balanced)
    # Top 4 by weight = opp0, opp1, opp2, opp3 (weights 10, 9, 8, 7).
    expected_keys = {"opp0", "opp1", "opp2", "opp3"}
    assert set(poo.keys()) == expected_keys, f"got {set(poo.keys())}"
    # Each plays 1600/4 = 400 games.
    for key in expected_keys:
        assert poo[key] == 400


def test_balanced_extreme_weight_skew():
    """Weight [100, 1, 1, 1] with 8w — opp 0 should get ~most workers."""
    opp_pool = _opps([100, 1, 1, 1])
    balanced = _allocate_opps_to_workers_balanced(opp_pool, 8, 1600)
    poo = _per_opp_games(balanced)
    # All 4 opps still represented (n_workers >= n_opps + min=1 invariant).
    assert len(poo) == 4
    # Opp 0 should have the most workers (probably 5-6) but everyone is >= 1.
    # workers_per_opp = max(1, int(8 * w / sum_w)) initially
    # int(8*100/103) = 7; int(8*1/103) = 0 → max(1, 0) = 1 for others
    # workers_per_opp = [7, 1, 1, 1] (sum=10? no — 7+1+1+1=10, but n_workers=8)
    # Need to remove 2. Remove from smallest fractional remainder.
    # target_w_float = [7.767, 0.0776, 0.0776, 0.0776]
    # fractional remainders = [0.767, 0.0776, 0.0776, 0.0776]
    # Remove from smallest first (tied at 0.0776), but min=1 so can't go below
    # So remove from opp 1 → workers=[7,0,1,1]? Wait min=1...
    # Actually because of min=1 we can't reduce below 1 for any opp.
    # Let me trace again. After max(1, int(t)): [7, 1, 1, 1] sum=10, delta=8-10=-2
    # Need to remove 2 workers. Each opp has workers_per_opp > 1 only at opp 0 (7).
    # Reduce opp 0: 7→6, 6→5. workers_per_opp = [5, 1, 1, 1]. sum=8 ✓.
    # So opp 0 gets 5 workers, opps 1-3 get 1 each.
    # Per worker = 1600/8 = 200. Opp 0 gets 5*200 = 1000 games; others 200.
    assert poo["opp0"] == 1000, f"opp0 got {poo['opp0']}"
    for k in ("opp1", "opp2", "opp3"):
        assert poo[k] == 200, f"{k} got {poo[k]}"


def test_balanced_edge_zero_workers():
    """0 workers → empty list."""
    assignments = _allocate_opps_to_workers_balanced(_opps([1, 1]), 0, 100)
    assert assignments == []


def test_balanced_edge_zero_opps():
    """0 opps → all workers get None opp, 0 games."""
    assignments = _allocate_opps_to_workers_balanced([], 4, 100)
    assert len(assignments) == 4
    for a in assignments:
        assert a["opp"] is None
        assert a["n_games"] == 0


def test_balanced_edge_single_worker():
    """1 worker/1 opp/100 games — worker plays all 100."""
    assignments = _allocate_opps_to_workers_balanced(_opps([1]), 1, 100)
    assert len(assignments) == 1
    assert assignments[0]["n_games"] == 100
    assert assignments[0]["opp"]["key"] == "opp0"


def test_balanced_round_off_distribution():
    """1601 games / 8 workers → 7 workers get 200, 1 worker gets 201."""
    opp_pool = _opps([1, 1, 1, 1, 1])
    balanced = _allocate_opps_to_workers_balanced(opp_pool, 8, 1601)
    pwg = sorted(_per_worker_games(balanced))
    assert pwg == [200] * 7 + [201], f"got {pwg}"
    assert sum(pwg) == 1601


def test_balanced_15w_full_pool_realistic():
    """15w/5opp non-uniform weights — realistic Phase 2 mid-iter scenario."""
    weights = [3.0, 2.5, 2.0, 1.5, 1.0]  # PFSP-ish weights
    opp_pool = _opps(weights)
    balanced = _allocate_opps_to_workers_balanced(opp_pool, 15, 1600)
    pwg = _per_worker_games(balanced)
    # Max-min spread <= 1
    assert max(pwg) - min(pwg) <= 1
    # Total == 1600
    assert sum(pwg) == 1600
    # All 5 opps represented (n_workers >= n_opps)
    poo = _per_opp_games(balanced)
    assert len(poo) == 5
    # Higher-weight opps get more workers (and thus more games)
    assert poo["opp0"] >= poo["opp4"]


# ----------------------------------------------------------------------------
# Dispatcher tests
# ----------------------------------------------------------------------------

def test_dispatcher_legacy_default():
    """Without calling set_pfsp_allocator_mode, dispatcher should use legacy."""
    from mp_centralized_collect import _allocate_opps_to_workers, _PFSP_ALLOCATOR_MODE
    # Confirm module-level default
    assert _PFSP_ALLOCATOR_MODE == "legacy", (
        f"default mode should be 'legacy', got {_PFSP_ALLOCATOR_MODE!r}"
    )
    # Dispatcher should match legacy output
    opp_pool = _opps([1, 1, 1, 1, 1])
    via_dispatcher = _allocate_opps_to_workers(opp_pool, 8, 1600)
    direct_legacy = _allocate_opps_to_workers_legacy(opp_pool, 8, 1600)
    assert _per_worker_games(via_dispatcher) == _per_worker_games(direct_legacy)


def test_dispatcher_balanced_after_setter():
    """After set_pfsp_allocator_mode('balanced'), dispatcher uses balanced."""
    from mp_centralized_collect import (
        _allocate_opps_to_workers,
        set_pfsp_allocator_mode,
    )
    opp_pool = _opps([1, 1, 1, 1, 1])

    set_pfsp_allocator_mode("balanced")
    via_dispatcher = _allocate_opps_to_workers(opp_pool, 8, 1600)
    direct_balanced = _allocate_opps_to_workers_balanced(opp_pool, 8, 1600)
    assert _per_worker_games(via_dispatcher) == _per_worker_games(direct_balanced)

    # Reset for hygiene (other tests may run after)
    set_pfsp_allocator_mode("legacy")


def test_dispatcher_rejects_invalid_mode():
    from mp_centralized_collect import set_pfsp_allocator_mode
    try:
        set_pfsp_allocator_mode("nonsense")
        assert False, "should have raised"
    except ValueError:
        pass


# ----------------------------------------------------------------------------
# Diagnostic helper — print a realistic scenario side-by-side
# ----------------------------------------------------------------------------

def _print_comparison(label, opp_pool, n_workers, total_games):
    legacy = _allocate_opps_to_workers_legacy(opp_pool, n_workers, total_games)
    balanced = _allocate_opps_to_workers_balanced(opp_pool, n_workers, total_games)
    legacy_pwg = sorted(_per_worker_games(legacy))
    balanced_pwg = sorted(_per_worker_games(balanced))
    legacy_poo = _per_opp_games(legacy)
    balanced_poo = _per_opp_games(balanced)

    legacy_max = max(legacy_pwg)
    balanced_max = max(balanced_pwg)
    savings_pct = (legacy_max - balanced_max) / legacy_max * 100 if legacy_max else 0

    print(f"\n=== {label} ===")
    print(f"  n_workers={n_workers}, total_games={total_games}, "
          f"weights={[o['weight'] for o in opp_pool]}")
    print(f"  legacy   per-worker: {legacy_pwg} max={legacy_max}")
    print(f"  balanced per-worker: {balanced_pwg} max={balanced_max}")
    print(f"  theoretical wall savings vs legacy max-worker: {savings_pct:.1f}%")
    print(f"  legacy   per-opp games: {dict(sorted(legacy_poo.items()))}")
    print(f"  balanced per-opp games: {dict(sorted(balanced_poo.items()))}")


def main():
    tests = [
        test_balanced_perworker_max_minus_min_le_1,
        test_balanced_s58_imbalance_repro,
        test_balanced_natural_balance_8w8opp,
        test_balanced_15w5opp_uniform,
        test_balanced_n_workers_lt_n_opps,
        test_balanced_n_workers_lt_n_opps_skewed,
        test_balanced_extreme_weight_skew,
        test_balanced_edge_zero_workers,
        test_balanced_edge_zero_opps,
        test_balanced_edge_single_worker,
        test_balanced_round_off_distribution,
        test_balanced_15w_full_pool_realistic,
        test_dispatcher_legacy_default,
        test_dispatcher_balanced_after_setter,
        test_dispatcher_rejects_invalid_mode,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")

    # Diagnostic comparisons (always print, not pass/fail)
    print("\n" + "=" * 70)
    print("Diagnostic side-by-side: legacy vs balanced per-worker distributions")
    print("=" * 70)
    _print_comparison(
        "S58 imbalance case (8w / 5opp uniform / 1600g)",
        _opps([1, 1, 1, 1, 1]), 8, 1600
    )
    _print_comparison(
        "8w / 8opp uniform / 1600g (natural balance)",
        _opps([1] * 8), 8, 1600
    )
    _print_comparison(
        "15w / 5opp uniform / 1600g",
        _opps([1, 1, 1, 1, 1]), 15, 1600
    )
    _print_comparison(
        "8w / 3opp uniform / 1600g",
        _opps([1, 1, 1]), 8, 1600
    )
    _print_comparison(
        "15w / 7opp PFSP-skewed / 1600g",
        _opps([5, 3, 2, 1, 1, 1, 1]), 15, 1600
    )
    _print_comparison(
        "8w / 16opp uniform / 1600g (n_workers < n_opps)",
        _opps([1] * 16), 8, 1600
    )

    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
