"""Unit tests for the adaptive CIS batching formula (S64 task #46 Stage A).

Tests _compute_adaptive_cis_params against:
  - Empirical calibration from Run A's CIS-STATS
  - Edge cases (pool=0, pool>>n_workers, smoke scale)
  - Monotonicity (min_batch is non-increasing as pool grows)
  - Floor (min_batch >= 2, timeout >= 1)

Run with:
  python test_cis_adaptive.py
"""
from __future__ import annotations

import sys

from mp_centralized_collect import _compute_adaptive_cis_params


def test_prod_config_15w():
    """Our canonical 15w config: 1600g, max_conc=200."""
    cases = {
        1: (10, 25),
        2: (5, 25),
        3: (3, 25),
        4: (2, 25),
        5: (2, 25),  # floor
        10: (2, 25),  # floor
        16: (2, 25),  # floor
    }
    for pool, expected in cases.items():
        got = _compute_adaptive_cis_params(
            pool_size=pool, games_per_iter=1600,
            n_workers=15, max_concurrent=200,
        )
        assert got == expected, (
            f"pool={pool}: expected {expected}, got {got}"
        )


def test_monotonic_min_batch():
    """min_batch should be non-increasing as pool grows (or floor)."""
    prev = None
    for pool in range(1, 20):
        mb, _ = _compute_adaptive_cis_params(
            pool_size=pool, games_per_iter=1600,
            n_workers=15, max_concurrent=200,
        )
        if prev is not None:
            assert mb <= prev, (
                f"pool={pool}: min_batch={mb} > prev={prev} (non-monotonic)"
            )
        prev = mb


def test_floor():
    """min_batch >= 2 always; timeout_ms >= 1."""
    for pool in [1, 5, 100, 1000]:
        mb, to = _compute_adaptive_cis_params(
            pool_size=pool, games_per_iter=1600,
            n_workers=15, max_concurrent=200,
        )
        assert mb >= 2, f"pool={pool}: min_batch={mb} below floor"
        assert to >= 1, f"pool={pool}: timeout_ms={to} below 1"


def test_pool_zero_degenerate():
    """pool=0 should behave like pool=1 (max(1, pool_size) clamp)."""
    a = _compute_adaptive_cis_params(
        pool_size=0, games_per_iter=1600,
        n_workers=15, max_concurrent=200,
    )
    b = _compute_adaptive_cis_params(
        pool_size=1, games_per_iter=1600,
        n_workers=15, max_concurrent=200,
    )
    assert a == b, f"pool=0 should be clamped to pool=1: got {a} vs {b}"


def test_n_workers_bounds_concurrent():
    """When n_workers * max_concurrent < games_per_iter, the smaller value
    determines arrival_rate (real concurrent games = n_workers × max_conc)."""
    # 2w × 50 conc = 100 concurrent < 1600 games_per_iter
    mb_small, _ = _compute_adaptive_cis_params(
        pool_size=1, games_per_iter=1600,
        n_workers=2, max_concurrent=50,
    )
    # 100 × 0.3 / 1 × 0.025 × 0.8 = 0.6 → floor 2
    assert mb_small == 2, f"got {mb_small}"


def test_smoke_scale():
    """Smoke scale: 200g, 4w, max_conc=50 = 200 concurrent.
    Arrival 60/sec. Per-slot at pool=1: 60. min_batch = 60×0.025×0.8 = 1.2 → 2."""
    mb, to = _compute_adaptive_cis_params(
        pool_size=1, games_per_iter=200,
        n_workers=4, max_concurrent=50,
    )
    assert mb == 2 and to == 25


def test_calibration_param():
    """Higher per_game_inf_rate → bigger min_batch (more arrivals per ms)."""
    mb_low, _ = _compute_adaptive_cis_params(
        pool_size=3, games_per_iter=1600,
        n_workers=15, max_concurrent=200,
        per_game_inf_rate=0.3,
    )
    mb_high, _ = _compute_adaptive_cis_params(
        pool_size=3, games_per_iter=1600,
        n_workers=15, max_concurrent=200,
        per_game_inf_rate=0.6,  # double
    )
    assert mb_high > mb_low, f"low={mb_low}, high={mb_high}"


def test_timeout_param():
    """Higher timeout → bigger min_batch (more time to accumulate)."""
    mb_short, _ = _compute_adaptive_cis_params(
        pool_size=3, games_per_iter=1600,
        n_workers=15, max_concurrent=200,
        timeout_ms=15,
    )
    mb_long, _ = _compute_adaptive_cis_params(
        pool_size=3, games_per_iter=1600,
        n_workers=15, max_concurrent=200,
        timeout_ms=50,
    )
    assert mb_long > mb_short, f"short={mb_short}, long={mb_long}"


def main():
    tests = [
        test_prod_config_15w,
        test_monotonic_min_batch,
        test_floor,
        test_pool_zero_degenerate,
        test_n_workers_bounds_concurrent,
        test_smoke_scale,
        test_calibration_param,
        test_timeout_param,
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
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
