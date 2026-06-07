"""MM watchdog (Layer A + Layer B) — pure-stdlib, no metamon/poke-env deps.

Used by metamon_accept_serve.py (and potentially foul_play_accept_serve.py)
to detect + recover from poke-env CRITICAL errors that hang the MM bot's
main thread.

Layer A (fast path): log handler watches for KNOWN poke-env CRITICAL
patterns that have been observed to hang the bot, and exits the process
within ~1 second of the trigger. Manager respawns within ~5-10s. Total
downtime per known-pattern stall: ~5-10s vs ~300s with absorption-only.

Layer B (fallback): daemon thread tracks "last battle event" timestamp.
If no battle event for HANG_THRESHOLD_S, exits with a log tail so we can
grow the KNOWN_HANG_PATTERNS list. Catches new/unknown hang triggers.

Both use os._exit(1) (not sys.exit) because sys.exit raises SystemExit
in the calling thread only — won't kill a hung asyncio loop. os._exit
is immediate, no cleanup, no async coordination needed.

Observed history:
  2026-06-01 (mm-largerl-0 fishbowl iter 152) — "Can't switch" hang 5+ min
  2026-06-07 (mm-largerl-2 smoke v5 iter 1) — same pattern, hung 33+ min
"""
from __future__ import annotations
import logging
import os
import threading
import time


KNOWN_HANG_PATTERNS = [
    # The specific game-state error confirmed to freeze the MM main thread
    # twice (Jun 1 and Jun 7, 2026). Likely caused by a Pokemon being
    # forced to switch out (Sacred Fire / Volt Tackle faint, etc.) when
    # there's no valid switch target — poke-env's choice generator doesn't
    # recover and the asyncio loop blocks forever.
    "Can't switch: You have to pass to a fainted",
    # Add new patterns here as discovered. Layer B will print the log tail
    # when it catches an unknown pattern → identify the CRITICAL line →
    # add the offending substring here so next time Layer A catches it in
    # ~1s instead of ~180s.
]

# Threshold for Layer B. ~3× the typical max battle wall (~60s on slow
# MM models) but well below the trainer-side 300s allocator timeout so
# we get faster recovery for unknown-pattern hangs.
HANG_THRESHOLD_S = 180

# Module-level state updated by mark_battle_event(). Initialized to module
# import time so a slow first-battle startup doesn't false-positive.
_last_battle_event_at = time.time()


def mark_battle_event() -> None:
    """Reset the Layer B stall timer. Call this each time a battle event
    happens (awaiting challenge, battle started, battle ended) — anything
    that confirms the main thread is still making progress.
    """
    global _last_battle_event_at
    _last_battle_event_at = time.time()


def _seconds_since_last_event() -> float:
    """Test helper: how long since the last battle event."""
    return time.time() - _last_battle_event_at


class HangPatternHandler(logging.Handler):
    """Layer A: catches known poke-env CRITICAL hang patterns within ~1s
    and exits the process so the orchestrator respawns.
    """

    def __init__(self, patterns: list):
        super().__init__(level=logging.CRITICAL)
        self.patterns = list(patterns)

    def emit(self, record: logging.LogRecord) -> None:
        # CRITICAL-only filter; be defensive in case someone reuses this.
        if record.levelno < logging.CRITICAL:
            return
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        for pat in self.patterns:
            if pat in msg:
                print(
                    f"[watchdog A] known hang pattern '{pat}' detected — "
                    f"exiting process so manager respawns",
                    flush=True,
                )
                os._exit(1)


def _dump_log_tail(log_path: str, n_lines: int = 50) -> None:
    """Best-effort tail dump for Layer B diagnostics. Helps identify the
    CRITICAL pattern that caused an unknown-pattern hang so it can be
    added to KNOWN_HANG_PATTERNS.
    """
    try:
        with open(log_path, 'r', errors='ignore') as f:
            lines = f.readlines()
        print(
            f"[watchdog B] tail of {log_path} (last {n_lines} lines, look "
            f"for CRITICAL to add to KNOWN_HANG_PATTERNS):",
            flush=True,
        )
        for line in lines[-n_lines:]:
            print(f"  | {line.rstrip()}", flush=True)
    except Exception as e:
        print(f"[watchdog B] couldn't tail log {log_path}: {e}", flush=True)


def start_progress_watchdog(log_path_for_tail: str = None,
                            threshold_s: float = HANG_THRESHOLD_S,
                            check_interval_s: float = 30.0) -> None:
    """Layer B: daemon thread that exits the process if no battle event
    happens for threshold_s. Catches unknown hang patterns that Layer A
    misses.

    Args:
        log_path_for_tail: optional path to subprocess log; on trigger,
                           the last 50 lines are printed.
        threshold_s: how long with no battle event before exit.
        check_interval_s: how often the watchdog checks (finer than threshold).
    """

    def _watchdog():
        while True:
            try:
                time.sleep(check_interval_s)
                elapsed = _seconds_since_last_event()
                if elapsed > threshold_s:
                    print(
                        f"[watchdog B] no battle event for {elapsed:.0f}s "
                        f"(threshold {threshold_s:.0f}s) — exiting so "
                        f"manager respawns",
                        flush=True,
                    )
                    if log_path_for_tail:
                        _dump_log_tail(log_path_for_tail)
                    os._exit(1)
            except Exception:
                # Watchdog must never die; swallow and keep checking.
                pass

    t = threading.Thread(target=_watchdog, daemon=True,
                         name="mm-watchdog-B")
    t.start()


def install(log_path_for_tail: str = None,
            patterns: list = None,
            threshold_s: float = HANG_THRESHOLD_S) -> None:
    """One-call setup: install Layer A (log handler) + start Layer B
    (daemon thread).
    """
    if patterns is None:
        patterns = KNOWN_HANG_PATTERNS
    logging.getLogger().addHandler(HangPatternHandler(patterns))
    start_progress_watchdog(
        log_path_for_tail=log_path_for_tail,
        threshold_s=threshold_s,
    )
    print(
        f"[watchdog] installed Layer A ({len(patterns)} known patterns) "
        f"+ Layer B ({threshold_s:.0f}s heartbeat threshold)",
        flush=True,
    )
