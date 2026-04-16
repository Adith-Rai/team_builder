"""Tests for EMA PFSP win-rate mode.

Run: python test_ema_pfsp.py
"""
import sys, os
from pathlib import Path
os.chdir(Path(__file__).parent)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def ema_update(old_rec, batch_w, batch_g, alpha=0.3, window=50):
    """Mirror of the EMA update logic in train_rl.py — for testing."""
    old_rate = (old_rec[0] / old_rec[1]) if old_rec[1] > 0 else 0.5
    batch_rate = (batch_w / batch_g) if batch_g > 0 else 0.5
    new_rate = (1.0 - alpha) * old_rate + alpha * batch_rate
    eff_games = min(old_rec[1] + batch_g, window)
    return [new_rate * eff_games, eff_games]


def cum_update(old_rec, batch_w, batch_g):
    """Mirror of cumulative mode."""
    return [old_rec[0] + batch_w, old_rec[1] + batch_g]


def rate_of(rec):
    return rec[0] / rec[1] if rec[1] > 0 else 0.5


# ── EMA correctness tests ──

def test_ema_first_encounter():
    """First encounter with unknown opponent uses batch rate directly (not blended)."""
    # Unknown opponent: rec=[0,0], default rate=0.5
    rec = ema_update([0, 0], batch_w=10, batch_g=14)
    # new_rate = 0.7 * 0.5 + 0.3 * (10/14) = 0.35 + 0.214 = 0.564
    expected_rate = 0.7 * 0.5 + 0.3 * (10 / 14)
    assert abs(rate_of(rec) - expected_rate) < 1e-6, f"Got {rate_of(rec)}, expected {expected_rate}"
    # eff_games capped at min(0+14, 50) = 14
    assert rec[1] == 14, f"Expected 14 eff games, got {rec[1]}"
    print("  PASS: first encounter computes EMA with default 0.5 prior")


def test_ema_forgets_old_data():
    """EMA forgets old data — a stale high rate should converge to recent reality."""
    # Start with 100 games at 90% WR (old stale data)
    rec = [90, 100]
    # Now we repeatedly encounter this opponent losing badly (30% win rate, 14 games each)
    for i in range(20):
        rec = ema_update(rec, batch_w=4, batch_g=14)  # 28.5% batch
    # After 20 encounters, EMA should be much closer to 30% than 90%
    final_rate = rate_of(rec)
    assert final_rate < 0.35, f"EMA didn't forget: rate still {final_rate:.1%}, expected <35%"
    # Effective games capped at 50
    assert rec[1] == 50, f"Expected eff_games=50 cap, got {rec[1]}"
    print(f"  PASS: EMA forgets old data (90%→{final_rate:.1%} after 20 batches of 28.5%)")


def test_ema_cap_at_window():
    """Effective games should cap at window size, preventing unbounded growth."""
    rec = [0, 0]
    for _ in range(100):  # 100 encounters
        rec = ema_update(rec, batch_w=7, batch_g=14, window=50)
    assert rec[1] == 50, f"Expected eff_games=50 (capped), got {rec[1]}"
    # Rate should be ~50% (all batches at 50%)
    assert abs(rate_of(rec) - 0.5) < 0.01, f"Expected ~50%, got {rate_of(rec):.1%}"
    print(f"  PASS: eff_games capped at window (stayed at 50)")


def test_ema_alpha_effect():
    """Higher alpha = faster forgetting."""
    # Start at 90%
    rec_slow = [90, 100]
    rec_fast = [90, 100]
    # Lose consistently at 30%
    for _ in range(10):
        rec_slow = ema_update(rec_slow, batch_w=4, batch_g=14, alpha=0.1)
        rec_fast = ema_update(rec_fast, batch_w=4, batch_g=14, alpha=0.5)
    slow_rate = rate_of(rec_slow)
    fast_rate = rate_of(rec_fast)
    assert fast_rate < slow_rate, f"Higher alpha should converge faster: fast={fast_rate:.1%}, slow={slow_rate:.1%}"
    print(f"  PASS: higher alpha converges faster (alpha=0.5: {fast_rate:.1%}, alpha=0.1: {slow_rate:.1%})")


def test_cumulative_preserved():
    """Cumulative mode should work exactly as before."""
    rec = [0, 0]
    rec = cum_update(rec, 10, 14)
    assert rec == [10, 14], f"Cum: got {rec}"
    rec = cum_update(rec, 20, 28)
    assert rec == [30, 42], f"Cum: got {rec}"
    # Rate: 30/42 = 0.714
    assert abs(rate_of(rec) - 30/42) < 1e-6
    print("  PASS: cumulative mode preserves counts exactly")


def test_ema_reaches_steady_state():
    """Training against consistent 60% win rate should converge EMA to ~60%."""
    rec = [0, 0]
    for _ in range(50):
        rec = ema_update(rec, batch_w=8, batch_g=13)  # 61.5%
    final = rate_of(rec)
    assert abs(final - 8/13) < 0.02, f"Expected ~61.5%, got {final:.1%}"
    print(f"  PASS: EMA converges to steady-state rate ({final:.1%} ≈ 61.5%)")


def test_pfsp_weight_responds_to_ema():
    """PFSP weight should respond to EMA rate, not stale cumulative."""
    # Simulated scenario: opponent was hard historically (30% WR over 100 games)
    # Cumulative would say: weight = (1-0.3)^2 = 0.49
    # But we've caught up — recent encounters show we win 80%

    # In cumulative mode, after 10 more encounters at 80%:
    cum_rec = [30, 100]
    for _ in range(10):
        cum_rec = cum_update(cum_rec, batch_w=11, batch_g=14)  # 78.6%
    cum_rate = rate_of(cum_rec)  # (30 + 110) / (100 + 140) = 140/240 = 0.583
    cum_weight = (1 - cum_rate) ** 2

    # In EMA mode (alpha=0.3):
    ema_rec = [30, 100]  # Same start
    for _ in range(10):
        ema_rec = ema_update(ema_rec, batch_w=11, batch_g=14, alpha=0.3)
    ema_rate = rate_of(ema_rec)
    ema_weight = (1 - ema_rate) ** 2

    # EMA should show rate closer to recent (80%) → weight closer to 0.04
    # Cumulative stuck closer to historical (30%) → weight closer to 0.49
    assert ema_rate > cum_rate + 0.1, f"EMA should reflect recent better: ema={ema_rate:.1%}, cum={cum_rate:.1%}"
    assert ema_weight < cum_weight / 2, f"EMA weight should be much lower: ema={ema_weight:.3f}, cum={cum_weight:.3f}"
    print(f"  PASS: EMA weight responds to recent reality (ema_rate={ema_rate:.1%} vs cum_rate={cum_rate:.1%})")
    print(f"    → PFSP would deprioritize this mastered opponent under EMA, keep oversampling under cumulative")


def test_pfsp_sample_works_with_ema_data():
    """Verify pfsp_sample works correctly with EMA-stored win_rates."""
    from rl_collection import pfsp_sample
    # Create a pool with mix of opponents
    pool = [f"snap_{i}.pt" for i in range(50)]
    # Build win_rates via EMA updates
    win_rates = {}
    for i, p in enumerate(pool):
        # Simulate some encounters
        rec = [0, 0]
        if i < 10:
            # Make these hard
            for _ in range(5):
                rec = ema_update(rec, batch_w=3, batch_g=14)  # 21% WR
        elif i < 40:
            # Moderate
            for _ in range(5):
                rec = ema_update(rec, batch_w=7, batch_g=14)  # 50% WR
        else:
            # Easy
            for _ in range(5):
                rec = ema_update(rec, batch_w=12, batch_g=14)  # 86% WR
        win_rates[p] = rec

    # Sample should prefer hard opponents (first 10)
    from collections import Counter
    counts = Counter()
    for _ in range(500):
        selected = pfsp_sample(pool, win_rates, n_opponents=10, uniform_frac=0.0)
        for s in selected:
            counts[s] += 1

    hard_count = sum(counts[pool[i]] for i in range(10))
    easy_count = sum(counts[pool[i]] for i in range(40, 50))
    # Per-opponent rates
    hard_per = hard_count / 10
    easy_per = easy_count / 10
    ratio = hard_per / max(1, easy_per)
    assert ratio > 2.0, f"PFSP should prefer hard (21%) over easy (86%): ratio={ratio:.1f}"
    print(f"  PASS: pfsp_sample works correctly with EMA data (hard/easy ratio: {ratio:.1f}x)")


if __name__ == "__main__":
    print("=" * 60)
    print("EMA PFSP TESTS")
    print("=" * 60)
    test_ema_first_encounter()
    test_ema_forgets_old_data()
    test_ema_cap_at_window()
    test_ema_alpha_effect()
    test_cumulative_preserved()
    test_ema_reaches_steady_state()
    test_pfsp_weight_responds_to_ema()
    test_pfsp_sample_works_with_ema_data()
    print()
    print("=" * 60)
    print("ALL EMA TESTS PASSED")
    print("=" * 60)
