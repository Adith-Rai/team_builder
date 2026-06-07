"""Unit tests for mm_watchdog (Layer A + Layer B).

Layer A: log handler that exits on known poke-env CRITICAL hang patterns.
Layer B: daemon thread that exits on stall (no battle event for X sec).

Both layers use os._exit. Tests monkey-patch os._exit to capture calls
instead of killing the test process.

Run with: python -m pytest test_mm_watchdog.py -v
"""
from __future__ import annotations
import logging
import os
import time
import pytest

import mm_watchdog as wd


@pytest.fixture
def captured_exits(monkeypatch):
    """Capture os._exit calls so tests don't actually kill the process."""
    calls = []

    def fake_exit(code):
        calls.append(code)
        # Raise to abort the handler's emit() — mirrors os._exit's
        # process-terminating effect without actually exiting.
        raise SystemExit(code)

    monkeypatch.setattr(wd.os, "_exit", fake_exit)
    return calls


# --- Layer A: known-pattern CRITICAL handler ---

def test_layer_a_fires_on_known_pattern(captured_exits):
    """The handler should exit when a CRITICAL record matches a known pattern."""
    handler = wd.HangPatternHandler(wd.KNOWN_HANG_PATTERNS)
    record = logging.LogRecord(
        name="poke-env",
        level=logging.CRITICAL,
        pathname=__file__,
        lineno=1,
        msg="Unexpected error: [Invalid choice] Can't switch: You have to pass to a fainted Pokémon",
        args=None,
        exc_info=None,
    )
    with pytest.raises(SystemExit):
        handler.emit(record)
    assert captured_exits == [1], "should have called os._exit(1) exactly once"


def test_layer_a_ignores_unrelated_critical(captured_exits):
    """CRITICAL records that don't match any known pattern should NOT exit."""
    handler = wd.HangPatternHandler(wd.KNOWN_HANG_PATTERNS)
    record = logging.LogRecord(
        name="poke-env",
        level=logging.CRITICAL,
        pathname=__file__,
        lineno=1,
        msg="Some other unrelated critical message about something",
        args=None,
        exc_info=None,
    )
    # Should NOT raise
    handler.emit(record)
    assert captured_exits == [], "should not have exited on unrelated CRITICAL"


def test_layer_a_ignores_lower_levels(captured_exits):
    """ERROR/WARNING/INFO records should not trigger the handler, even if
    they contain the hang pattern (the handler is CRITICAL-only).
    """
    handler = wd.HangPatternHandler(wd.KNOWN_HANG_PATTERNS)
    for level in (logging.ERROR, logging.WARNING, logging.INFO):
        record = logging.LogRecord(
            name="poke-env",
            level=level,
            pathname=__file__,
            lineno=1,
            msg="Can't switch: You have to pass to a fainted Pokémon",
            args=None,
            exc_info=None,
        )
        handler.emit(record)
    assert captured_exits == [], "non-CRITICAL records must not trigger exit"


def test_layer_a_handler_via_logging_pipeline(captured_exits):
    """Sanity check: when handler is attached to a logger and that logger
    emits a matching CRITICAL, the handler triggers exit. Catches install
    bugs (wrong level, wrong attachment point).
    """
    handler = wd.HangPatternHandler(wd.KNOWN_HANG_PATTERNS)
    logger = logging.getLogger("test_layer_a_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        with pytest.raises(SystemExit):
            logger.critical("Issue: Can't switch: You have to pass to a fainted")
        assert captured_exits == [1]
    finally:
        logger.removeHandler(handler)


def test_layer_a_multiple_patterns(captured_exits):
    """Multiple patterns: each one independently triggers."""
    handler = wd.HangPatternHandler([
        "pattern_alpha",
        "pattern_beta",
    ])
    record = logging.LogRecord(
        name="x",
        level=logging.CRITICAL,
        pathname=__file__,
        lineno=1,
        msg="something pattern_beta something",
        args=None,
        exc_info=None,
    )
    with pytest.raises(SystemExit):
        handler.emit(record)
    assert captured_exits == [1]


# --- Layer B: mark_battle_event progress tracking ---

def test_mark_battle_event_updates_timestamp():
    """mark_battle_event should update the module-level timestamp to ~now."""
    before = time.time()
    wd._last_battle_event_at = before - 9999  # simulate stale
    wd.mark_battle_event()
    after = time.time()
    assert before - 1 <= wd._last_battle_event_at <= after + 1, (
        "mark_battle_event must reset timestamp to ~now"
    )


def test_seconds_since_last_event_increases():
    """Verify _seconds_since_last_event reflects elapsed time."""
    wd.mark_battle_event()
    initial = wd._seconds_since_last_event()
    time.sleep(0.05)
    later = wd._seconds_since_last_event()
    assert later > initial, "elapsed should increase as time passes"
    assert later < 1.0, "should be sub-second for this test"


def test_start_progress_watchdog_fires_on_stall(captured_exits, tmp_path):
    """Layer B watchdog with tiny threshold fires when no battle events
    happen. Uses captured_exits fixture so the test process doesn't die.
    """
    # Write a sample log file to verify the tail dump works
    log_file = tmp_path / "sample.log"
    log_file.write_text("line A\nline B\nCRITICAL bad-pattern\nline D\n")

    wd.mark_battle_event()  # set t=0
    # Start watchdog with threshold 0.1s and check interval 0.05s — should
    # fire almost immediately because no further mark_battle_event calls.
    wd.start_progress_watchdog(
        log_path_for_tail=str(log_file),
        threshold_s=0.1,
        check_interval_s=0.05,
    )
    # Give the daemon thread time to wake up + fire. captured_exits
    # raises SystemExit inside the thread, which is silently swallowed
    # (threads can't propagate exceptions to main), so we just wait and
    # check the exit list.
    deadline = time.time() + 3.0
    while time.time() < deadline and not captured_exits:
        time.sleep(0.05)
    assert captured_exits, (
        "Layer B should have fired within 3s with threshold 0.1s"
    )
    assert captured_exits[0] == 1


def test_start_progress_watchdog_does_not_fire_when_events_happen(captured_exits):
    """If mark_battle_event keeps being called, Layer B never fires."""
    wd.mark_battle_event()
    wd.start_progress_watchdog(
        log_path_for_tail=None,
        threshold_s=0.5,
        check_interval_s=0.05,
    )
    # Keep marking events frequently for 1 second — should outpace the
    # 0.5s threshold.
    end = time.time() + 1.0
    while time.time() < end:
        wd.mark_battle_event()
        time.sleep(0.05)
    assert captured_exits == [], (
        "Layer B should NOT fire when battle events keep happening"
    )


# --- install() wiring ---

def test_install_attaches_handler():
    """install() should add a HangPatternHandler to root logger."""
    root = logging.getLogger()
    n_before = sum(
        1 for h in root.handlers if isinstance(h, wd.HangPatternHandler)
    )
    wd.install(log_path_for_tail=None, threshold_s=10_000)  # huge threshold so Layer B doesn't fire mid-test
    n_after = sum(
        1 for h in root.handlers if isinstance(h, wd.HangPatternHandler)
    )
    assert n_after == n_before + 1, (
        "install should add exactly one HangPatternHandler"
    )
    # Cleanup
    for h in list(root.handlers):
        if isinstance(h, wd.HangPatternHandler):
            root.removeHandler(h)
            break


def test_known_patterns_includes_observed_trigger():
    """Regression guard: ensure the observed 'Can't switch' pattern stays
    in KNOWN_HANG_PATTERNS (don't accidentally delete it).
    """
    assert any(
        "Can't switch" in p for p in wd.KNOWN_HANG_PATTERNS
    ), "The observed 'Can't switch' hang trigger MUST stay in KNOWN_HANG_PATTERNS"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
