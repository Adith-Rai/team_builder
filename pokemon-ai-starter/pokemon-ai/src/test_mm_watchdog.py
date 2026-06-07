"""Unit tests for mm_watchdog — unified trigger+confirm design.

Design: any CRITICAL log record → arm a 30s confirmation timer →
exit if no battle event in the window, else recover.

Tests monkey-patch os._exit so they don't kill the test process.

Run with: python -m pytest test_mm_watchdog.py -v
"""
from __future__ import annotations
import logging
import time
import pytest

import mm_watchdog as wd


@pytest.fixture
def captured_exits(monkeypatch):
    """Capture os._exit calls instead of letting them kill the test process."""
    calls = []

    def fake_exit(code):
        calls.append(code)
        # Raise to abort the calling function — mirrors os._exit's
        # process-terminating effect without actually exiting.
        raise SystemExit(code)

    monkeypatch.setattr(wd.os, "_exit", fake_exit)
    return calls


@pytest.fixture
def short_confirm_window(monkeypatch):
    """Replace CONFIRM_WINDOW_S with a tiny value so tests run fast."""
    monkeypatch.setattr(wd, "CONFIRM_WINDOW_S", 0.2)
    return 0.2


# --- mark_battle_event basic mechanics ---

def test_mark_battle_event_updates_timestamp():
    """mark_battle_event resets the module-level timestamp to ~now."""
    before = time.time()
    wd._last_battle_event_at = before - 9999  # simulate stale
    wd.mark_battle_event()
    after = time.time()
    assert before - 1 <= wd._last_battle_event_at <= after + 1


def test_seconds_since_last_event_increases():
    wd.mark_battle_event()
    initial = wd._seconds_since_last_event()
    time.sleep(0.05)
    later = wd._seconds_since_last_event()
    assert later > initial


# --- Confirmation: hang case (no battle event in window) ---

def test_confirm_hang_fires_when_no_progress(captured_exits):
    """If _last_battle_event_at hasn't moved past baseline_t,
    _confirm_hang must call os._exit(1).
    """
    wd.mark_battle_event()
    baseline_t = wd._last_battle_event_at
    # _confirm_hang reads _last_battle_event_at — it's still baseline_t
    # (no event called since), so it should fire.
    with pytest.raises(SystemExit):
        wd._confirm_hang("test trigger msg", baseline_t)
    assert captured_exits == [1]


def test_confirm_hang_skips_when_progress_happens(captured_exits):
    """If a battle event happens between baseline_t and the confirmation
    check, _confirm_hang must NOT exit.
    """
    baseline_t = time.time() - 5  # baseline in the past
    time.sleep(0.01)
    wd.mark_battle_event()  # progress happened
    # Confirm time > baseline → recovery, no exit
    wd._confirm_hang("test trigger msg", baseline_t)
    assert captured_exits == [], (
        "confirm should not exit if battle event happened after baseline"
    )


# --- HangDetector handler: trigger + timer + confirm flow ---

def test_handler_fires_on_critical_no_recovery(captured_exits, short_confirm_window):
    """End-to-end: emit a CRITICAL → handler arms timer → no battle event
    happens → timer fires → exit. Verifies the full pipeline.
    """
    handler = wd.HangDetector()
    wd.mark_battle_event()  # set baseline
    record = logging.LogRecord(
        name="poke-env",
        level=logging.CRITICAL,
        pathname=__file__,
        lineno=1,
        msg="Unexpected error: [Invalid choice] Can't switch: pass to fainted",
        args=None,
        exc_info=None,
    )
    handler.emit(record)
    # Wait for the timer to fire. Timer thread will call _confirm_hang →
    # SystemExit raised inside the thread (silently swallowed). Just check
    # that os._exit got called by polling captured_exits.
    deadline = time.time() + 2.0
    while time.time() < deadline and not captured_exits:
        time.sleep(0.05)
    assert captured_exits == [1], (
        f"handler should have triggered exit within 2s (window={wd.CONFIRM_WINDOW_S}s)"
    )


def test_handler_does_not_fire_if_battle_event_in_window(captured_exits, short_confirm_window):
    """If a battle event happens after the CRITICAL but before the
    confirmation window expires, no exit.
    """
    handler = wd.HangDetector()
    wd.mark_battle_event()  # baseline = now
    record = logging.LogRecord(
        name="poke-env",
        level=logging.CRITICAL,
        pathname=__file__,
        lineno=1,
        msg="Recoverable critical event",
        args=None,
        exc_info=None,
    )
    handler.emit(record)
    # Battle progresses immediately (within confirmation window):
    time.sleep(0.05)
    wd.mark_battle_event()
    # Wait for timer to fire and call _confirm_hang. With progress, it
    # should NOT call os._exit.
    time.sleep(short_confirm_window + 0.3)
    assert captured_exits == [], (
        "should NOT exit if battle event happened in the confirmation window"
    )


def test_handler_ignores_non_critical(captured_exits, short_confirm_window):
    """ERROR/WARNING/INFO must not arm the timer."""
    handler = wd.HangDetector()
    for level in (logging.ERROR, logging.WARNING, logging.INFO):
        record = logging.LogRecord(
            name="x",
            level=level,
            pathname=__file__,
            lineno=1,
            msg="Can't switch: pass to fainted",  # matches old pattern, but level too low
            args=None,
            exc_info=None,
        )
        handler.emit(record)
    # Wait past the confirmation window to be sure nothing fires
    time.sleep(short_confirm_window + 0.3)
    assert captured_exits == [], "non-CRITICAL records must not arm the timer"


def test_handler_pipeline_via_logger(captured_exits, short_confirm_window):
    """Sanity check: handler attached to a logger triggers on real log call."""
    handler = wd.HangDetector()
    logger = logging.getLogger("test_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        wd.mark_battle_event()
        logger.critical("Some critical message")
        deadline = time.time() + 2.0
        while time.time() < deadline and not captured_exits:
            time.sleep(0.05)
        assert captured_exits == [1]
    finally:
        logger.removeHandler(handler)


# --- install() wiring ---

def test_install_attaches_handler():
    """install() should add exactly one HangDetector to the root logger."""
    root = logging.getLogger()
    n_before = sum(1 for h in root.handlers if isinstance(h, wd.HangDetector))
    wd.install()
    n_after = sum(1 for h in root.handlers if isinstance(h, wd.HangDetector))
    assert n_after == n_before + 1
    # Cleanup
    for h in list(root.handlers):
        if isinstance(h, wd.HangDetector):
            root.removeHandler(h)
            break


# --- Regression guards ---

def test_confirm_window_reasonable():
    """CONFIRM_WINDOW_S should be > 0 and well below the trainer-side
    300s allocator timeout (otherwise the watchdog provides no benefit).
    """
    assert 0 < wd.CONFIRM_WINDOW_S < 300, (
        f"CONFIRM_WINDOW_S={wd.CONFIRM_WINDOW_S} is outside the sensible range"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
