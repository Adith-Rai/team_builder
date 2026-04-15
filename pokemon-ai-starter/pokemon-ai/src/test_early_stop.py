"""Tests for composite early stopping logic.

Run: python test_early_stop.py
"""
import sys, os
from pathlib import Path
os.chdir(Path(__file__).parent)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


class FakeArgs:
    """Minimal argparse-like namespace for testing."""
    early_stop = True
    early_stop_patience = 3
    early_stop_savg_threshold = 2.0
    early_stop_bot_threshold = 3.0
    early_stop_bot_count = 3
    early_stop_min_evals = 5


def make_eval(it, savg, sh=None, smd=None, tac=None, stra=None):
    """Build an eval dict. If bot values are None, use savg."""
    return {
        "iter": it,
        "savg": savg,
        "SH": sh if sh is not None else savg,
        "SmartDmg": smd if smd is not None else savg,
        "Tactical": tac if tac is not None else savg,
        "Strategic": stra if stra is not None else savg,
    }


def test_not_enough_data():
    """Below min_evals threshold → don't stop."""
    from train_rl import _check_early_stop
    args = FakeArgs()
    history = [make_eval(100 + i*20, 55) for i in range(4)]
    assert _check_early_stop(history, args) == False, "Should not stop with only 4 evals"
    print("  PASS: not enough data prevents stop")


def test_stable_performance_no_stop():
    """Stable savg + bots hovering → don't stop."""
    from train_rl import _check_early_stop
    args = FakeArgs()
    history = [make_eval(100 + i*20, 55 + (i % 3 - 1)) for i in range(10)]  # oscillation
    assert _check_early_stop(history, args) == False, "Should not stop on noise"
    print("  PASS: stable performance doesn't trigger stop")


def test_single_iter_dip_no_stop():
    """Single bad eval in middle of good ones → don't stop."""
    from train_rl import _check_early_stop
    args = FakeArgs()
    history = [
        make_eval(20, 55), make_eval(40, 56), make_eval(60, 55),
        make_eval(80, 57), make_eval(100, 45),  # single bad
        make_eval(120, 56), make_eval(140, 55),
    ]
    assert _check_early_stop(history, args) == False, "Single bad eval shouldn't stop"
    print("  PASS: single-iter dip doesn't trigger stop")


def test_sustained_regression_stops():
    """3 consecutive evals below best with multi-bot consensus → STOP."""
    from train_rl import _check_early_stop
    args = FakeArgs()
    # Good history, then sustained degradation across all metrics
    history = [
        make_eval(20, 57, sh=58, smd=58, tac=55, stra=57),  # good
        make_eval(40, 58, sh=60, smd=58, tac=56, stra=58),  # best
        make_eval(60, 57, sh=58, smd=57, tac=56, stra=57),
        make_eval(80, 56, sh=57, smd=56, tac=56, stra=55),
        make_eval(100, 56, sh=56, smd=56, tac=55, stra=56),  # min 5 evals
        make_eval(120, 53, sh=53, smd=53, tac=53, stra=53),  # degraded
        make_eval(140, 52, sh=52, smd=52, tac=52, stra=52),  # degraded
        make_eval(160, 51, sh=51, smd=51, tac=51, stra=51),  # degraded
    ]
    assert _check_early_stop(history, args) == True, "Sustained regression should stop"
    print("  PASS: sustained multi-bot regression triggers stop")


def test_savg_drops_but_bots_ok_no_stop():
    """savg drops but <3 bots degrade → don't stop (missing consensus)."""
    from train_rl import _check_early_stop
    args = FakeArgs()
    # Only 2 of 4 bots degrade, but savg drops on them
    history = [
        make_eval(20, 58, sh=60, smd=60, tac=56, stra=56),
        make_eval(40, 58, sh=60, smd=60, tac=56, stra=56),  # best
        make_eval(60, 57, sh=59, smd=59, tac=55, stra=55),
        make_eval(80, 56, sh=58, smd=58, tac=55, stra=55),
        make_eval(100, 56, sh=58, smd=58, tac=54, stra=54),
        make_eval(120, 54, sh=48, smd=48, tac=55, stra=65),  # 2 bots much worse, 2 better
        make_eval(140, 54, sh=48, smd=48, tac=55, stra=65),
        make_eval(160, 54, sh=48, smd=48, tac=55, stra=65),
    ]
    # Bot regression: SH (60→48, -12) and SmD (60→48, -12), Tac (56→55, -1), Str (56→65, +9)
    # Only 2 bots degrade by >3%, need 3. So should NOT stop.
    assert _check_early_stop(history, args) == False, "Only 2 bots degrading shouldn't stop"
    print("  PASS: only 2 bots degrading doesn't trigger (needs 3)")


def test_bots_degrade_but_savg_stable_no_stop():
    """Most bots drop slightly but savg stays stable → don't stop."""
    from train_rl import _check_early_stop
    args = FakeArgs()
    history = [
        make_eval(20, 56, sh=56, smd=56, tac=56, stra=56),
        make_eval(40, 56, sh=56, smd=56, tac=56, stra=56),
        make_eval(60, 56, sh=56, smd=56, tac=56, stra=56),
        make_eval(80, 56, sh=56, smd=56, tac=56, stra=56),  # best
        make_eval(100, 56, sh=56, smd=56, tac=56, stra=56),
        make_eval(120, 55, sh=53, smd=53, tac=53, stra=61),  # 3 bots down, 1 up, savg -1
        make_eval(140, 55, sh=53, smd=53, tac=53, stra=61),
        make_eval(160, 55, sh=53, smd=53, tac=53, stra=61),
    ]
    # savg dropped only 1 (< 2.0 threshold) → no trigger
    assert _check_early_stop(history, args) == False, "Stable savg shouldn't trigger"
    print("  PASS: stable savg prevents stop even with bot regression")


def test_patience_prevents_early_stop():
    """Only 2 bad evals (not 3) → don't stop yet."""
    from train_rl import _check_early_stop
    args = FakeArgs()
    history = [
        make_eval(20, 58, sh=60, smd=60, tac=56, stra=56),
        make_eval(40, 58, sh=60, smd=60, tac=56, stra=56),
        make_eval(60, 57, sh=59, smd=59, tac=55, stra=55),
        make_eval(80, 57, sh=59, smd=59, tac=55, stra=55),
        make_eval(100, 57, sh=59, smd=59, tac=55, stra=55),
        make_eval(120, 57, sh=58, smd=58, tac=56, stra=56),  # good
        make_eval(140, 53, sh=53, smd=53, tac=52, stra=54),  # bad
        make_eval(160, 52, sh=53, smd=53, tac=52, stra=50),  # bad
        # only 2 consecutive bad, patience=3
    ]
    assert _check_early_stop(history, args) == False, "2 consecutive bad doesn't meet patience=3"
    print("  PASS: patience requires 3 consecutive bad evals")


def test_recovers_from_dip_no_stop():
    """3 bad evals but then recovers → current state is fine."""
    from train_rl import _check_early_stop
    args = FakeArgs()
    history = [
        make_eval(20, 58, sh=60, smd=60, tac=56, stra=56),
        make_eval(40, 58, sh=60, smd=60, tac=56, stra=56),
        make_eval(60, 57, sh=59, smd=59, tac=55, stra=55),
        make_eval(80, 57, sh=59, smd=59, tac=55, stra=55),
        make_eval(100, 57, sh=59, smd=59, tac=55, stra=55),
        make_eval(120, 53, sh=53, smd=53, tac=52, stra=54),  # bad
        make_eval(140, 52, sh=53, smd=53, tac=52, stra=50),  # bad
        make_eval(160, 52, sh=53, smd=53, tac=52, stra=50),  # bad
        make_eval(180, 57, sh=59, smd=59, tac=55, stra=55),  # recovered
    ]
    # Last 3 evals: 52/53, 52/53, 57/59 — last one not bad
    assert _check_early_stop(history, args) == False, "Recovery in last eval shouldn't stop"
    print("  PASS: recovery from dip prevents stop")


def test_exp4_scenario():
    """Simulate the Exp 4 collapse — should have triggered early stop."""
    from train_rl import _check_early_stop
    args = FakeArgs()
    # Based on actual exp 4 data:
    # Eval 1 (iter 2999): SH=64, SmD=65, Tac=58, Str=56, savg=61 (BEST)
    # Eval 2 (iter 3019): SH=62, SmD=53, Tac=50, Str=50, savg=54
    # Eval 3 (iter 3039): SH=49, SmD=44, Tac=60, Str=57, savg=52
    # Pad with fake pre-history to get above min_evals
    history = [
        make_eval(2919, 55, sh=56, smd=55, tac=54, stra=55),
        make_eval(2939, 57, sh=58, smd=57, tac=57, stra=57),
        make_eval(2959, 58, sh=60, smd=59, tac=57, stra=56),
        make_eval(2979, 59, sh=62, smd=60, tac=58, stra=56),
        make_eval(2999, 61, sh=64, smd=65, tac=58, stra=56),  # PEAK
        make_eval(3019, 54, sh=62, smd=53, tac=50, stra=50),
        make_eval(3039, 52, sh=49, smd=44, tac=60, stra=57),
    ]
    # After eval 3039: only 2 bad evals (3019, 3039). Need a 3rd.
    # So should NOT stop yet — real experiment had limited evals before FATAL.
    # But if we had one more bad eval, it should trigger.
    history.append(make_eval(3055, 50, sh=45, smd=40, tac=58, stra=57))
    result = _check_early_stop(history, args)
    assert result == True, f"Exp 4 collapse with 3 bad evals should trigger stop, got {result}"
    print("  PASS: Exp 4 scenario with 3 bad evals triggers stop (would have saved compute!)")


if __name__ == "__main__":
    print("=" * 60)
    print("EARLY STOP UNIT TESTS")
    print("=" * 60)
    test_not_enough_data()
    test_stable_performance_no_stop()
    test_single_iter_dip_no_stop()
    test_sustained_regression_stops()
    test_savg_drops_but_bots_ok_no_stop()
    test_bots_degrade_but_savg_stable_no_stop()
    test_patience_prevents_early_stop()
    test_recovers_from_dip_no_stop()
    test_exp4_scenario()
    print()
    print("=" * 60)
    print("ALL EARLY STOP TESTS PASSED")
    print("=" * 60)
