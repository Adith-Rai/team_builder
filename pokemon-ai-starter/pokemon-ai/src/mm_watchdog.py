"""MM watchdog — unified trigger+confirm hang detection for poke-env CRITICALs.

DESIGN (verified empirically 2026-06-07):
  71/71 CRITICAL events across all MM-subprocess logs (~10 days production)
  are the same poke-env hang trigger:
      "Unexpected error message: ['', 'error',
       '[Invalid choice] Can't switch: You have to pass to a fainted Pokémon']"

  In MM-subprocess scope, NO other CRITICAL pattern has ever appeared.
  So a broad CRITICAL trigger is safe: we don't need to enumerate hang
  subtypes — anything that reaches CRITICAL in the MM process is either
  (a) the known hang, or (b) some new unknown CRITICAL.

  Confirmation step protects against new/recoverable CRITICALs: after a
  CRITICAL fires we wait CONFIRM_WINDOW_S seconds — if a battle event
  happens in that window, MM recovered, no exit. If not, real hang.

WARNING — scope:
  This handler is SAFE only in MM-subprocess Python interpreters where
  CRITICAL fires only on poke-env hang triggers. If this is ever installed
  in the trainer process (e.g., for in-process MCTS / FP adapters),
  you MUST filter by logger name — trainer-side CIS workers log
  "CRITICAL - Listen interrupted by..." every iter boundary, which is
  normal cleanup, NOT a hang. Such CRITICALs would false-positive this
  handler.

WHY os._exit (not sys.exit):
  sys.exit raises SystemExit in the calling thread — won't break out of
  a hung asyncio event loop running on the main thread. os._exit is
  immediate process termination, no async coordination needed. The
  manager (external_opponent_manager._monitor_loop) polls every 5s and
  respawns the subprocess on exit.

Observed history:
  2026-06-01 (mm-largerl-0 fishbowl iter 152) — 5+ min hang
  2026-06-07 (mm-largerl-2 smoke v5 iter 1)  — 33+ min hang
  ...both same "Can't switch" trigger
"""
from __future__ import annotations
import logging
import os
import threading
import time


# After a CRITICAL fires, wait this long. If a battle event happens within
# the window (mark_battle_event called), it was recoverable and we don't
# exit. If not, MM is hung and we exit.
#
# 30s is well below the trainer-side 300s allocator timeout (so we
# recover faster) and well above the worst recoverable-CRITICAL latency
# we've observed (which is ~0s — recoverable poke-env errors don't even
# block accept_challenges). Tunable here if needed.
CONFIRM_WINDOW_S = 30


# Module-level battle-progress timestamp. Updated by mark_battle_event()
# from the accept_loop. The confirmation step reads it to decide whether
# a CRITICAL was followed by real progress.
_last_battle_event_at = time.time()


def mark_battle_event() -> None:
    """Reset the progress marker. Call this each time a real battle event
    happens (awaiting challenge, battle started, battle ended) — anything
    that confirms the main thread is making progress.
    """
    global _last_battle_event_at
    _last_battle_event_at = time.time()


def _seconds_since_last_event() -> float:
    """Test helper."""
    return time.time() - _last_battle_event_at


def _confirm_hang(triggering_msg: str, baseline_t: float) -> None:
    """Called via threading.Timer CONFIRM_WINDOW_S after a CRITICAL.

    If no battle event has updated _last_battle_event_at since the trigger
    (baseline_t), MM is hung — exit. Otherwise, MM recovered, do nothing.
    """
    if _last_battle_event_at <= baseline_t:
        print(
            f"[watchdog] CONFIRMED hang ({CONFIRM_WINDOW_S}s after CRITICAL, "
            f"no battle event) — exiting so manager respawns. "
            f"Trigger: {triggering_msg}",
            flush=True,
        )
        os._exit(1)
    else:
        # Recovered — log it so we can see false-positive triggers if any
        # turn up in production (they shouldn't, per the data, but if they
        # do we want to know).
        print(
            f"[watchdog] CRITICAL recovered within {CONFIRM_WINDOW_S}s "
            f"(battle progressed). No exit. Trigger: {triggering_msg}",
            flush=True,
        )


class HangDetector(logging.Handler):
    """ANY CRITICAL → start CONFIRM_WINDOW_S timer → exit if no progress.

    Trigger is broad (any CRITICAL log record); confirmation is narrow
    (battle progress must happen within the window). Net: catches all
    observed hangs AND any future unknown CRITICAL pattern, with zero
    false-positive risk on idle MMs / long battles / recoverable
    CRITICAL events.
    """

    def __init__(self):
        super().__init__(level=logging.CRITICAL)

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.CRITICAL:
            return
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        # Truncate for log readability — full message is in MM log already.
        triggering_msg = msg[:200]
        baseline_t = _last_battle_event_at
        print(
            f"[watchdog] CRITICAL detected — armed confirmation timer "
            f"({CONFIRM_WINDOW_S}s). Trigger: {triggering_msg}",
            flush=True,
        )
        threading.Timer(
            CONFIRM_WINDOW_S,
            _confirm_hang,
            args=(triggering_msg, baseline_t),
        ).start()


def install() -> None:
    """One-call setup: attach the HangDetector to the root logger so any
    CRITICAL from poke-env / metamon / showdown subsystems is caught.
    """
    logging.getLogger().addHandler(HangDetector())
    print(
        f"[watchdog] installed: any CRITICAL → {CONFIRM_WINDOW_S}s "
        f"confirmation timer → exit if no battle event",
        flush=True,
    )
