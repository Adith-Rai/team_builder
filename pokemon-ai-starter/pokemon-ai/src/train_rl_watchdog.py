#!/usr/bin/env python3
"""External watchdog for train_rl.py.

Runs as a separate process. Monitors the train_rl log file's mtime. If the
log hasn't been updated in --stall-min minutes, SIGKILLs the entire
train_rl process tree (the main PID and all its descendants).

S68 (2026-06-13): created after Run #9 v4 silently froze for 7 hours.
Main process was in wait4() inside the cis-orch reset path; no internal
watchdog could fire because they all run inside main itself. This external
watchdog is structurally isolated — it can detect main hangs because it
runs in a separate process tree.

Why external (vs in-process watchdog thread):
  - In-process watchdog threads share main's failure modes (GIL deadlock,
    signal handler blocked, etc.). When main itself hangs, the watchdog
    thread often hangs too.
  - External watchdog runs as its own kernel-level process. Its scheduler
    slot is independent of main's. Catches the supercategory "main has
    stopped making progress" rather than specific subcategories.

Detection strategy:
  - We use log mtime, NOT log content. mtime advances on every write.
  - train_rl prints heartbeats every 5s (cis-w LIVE messages, FLOW lines,
    etc.). Even an idle main process has these. If mtime hasn't advanced
    for several minutes, main is genuinely stuck.

Kill strategy:
  - Walk the process tree (pgrep -P recursively, or psutil.children).
  - SIGKILL leaf-first, then root, so children die before parent and don't
    re-parent to init creating orphans.
  - We do NOT try graceful shutdown — the entire point is that graceful
    paths have already failed. SIGKILL is the right tool here.

Usage:
  # As sibling of train_rl launch:
  setsid nohup python train_rl_watchdog.py \\
      --pid 12345 \\
      --log /tmp/run9_heur_diversity_v1.log \\
      --stall-min 15 \\
      </dev/null >>/tmp/run9_watchdog.log 2>&1 &
  disown

  # --pid: PID of train_rl main process to monitor + kill on stall
  # --log: log file whose mtime indicates main's liveness
  # --stall-min: kill after this many minutes without log update (default 15)
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _log(msg):
    """Stamp-prefixed print so the watchdog log is readable."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[watchdog {ts}] {msg}", flush=True)


def _pid_alive(pid: int) -> bool:
    """True iff process with this PID exists and is not a zombie."""
    try:
        os.kill(pid, 0)
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("State:"):
                    state = line.split()[1]
                    return state != "Z"  # Z = zombie
        return True
    except (OSError, FileNotFoundError):
        return False


def _all_descendants(root_pid: int) -> list[int]:
    """Return all descendants of root_pid (children, grandchildren, ...) in
    leaf-first order. Returns [] if root has no descendants or doesn't exist."""
    result = []
    frontier = [root_pid]
    while frontier:
        next_frontier = []
        for parent in frontier:
            try:
                out = subprocess.run(
                    ["pgrep", "-P", str(parent)],
                    capture_output=True, text=True, timeout=5,
                )
                children = [int(line) for line in out.stdout.splitlines() if line.strip()]
            except (subprocess.TimeoutExpired, ValueError):
                children = []
            next_frontier.extend(children)
        result.extend(next_frontier)
        frontier = next_frontier
    # Reverse so leaves come first (kill deepest first so we don't orphan)
    return list(reversed(result))


def _sigkill_tree(root_pid: int):
    """SIGKILL root_pid and all descendants, leaf-first."""
    descendants = _all_descendants(root_pid)
    _log(f"killing tree: root={root_pid}, {len(descendants)} descendants")
    # Leaf-first so children don't re-parent to init before we kill the parent
    for pid in descendants:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    # Then the root
    try:
        os.kill(root_pid, signal.SIGKILL)
        _log(f"sent SIGKILL to root pid {root_pid}")
    except OSError as e:
        _log(f"could not SIGKILL root {root_pid}: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pid", type=int, required=True,
                   help="PID of train_rl main process to monitor + kill on stall")
    p.add_argument("--log", required=True,
                   help="Path to train_rl log file. Watchdog uses its mtime "
                        "as the liveness signal.")
    p.add_argument("--stall-min", type=float, default=15.0,
                   help="Kill the process tree if log hasn't been written to "
                        "in this many minutes. Default 15.")
    p.add_argument("--check-interval-s", type=float, default=30.0,
                   help="Seconds between mtime checks. Default 30s.")
    args = p.parse_args()

    log_path = Path(args.log)
    pid = args.pid
    stall_seconds = args.stall_min * 60

    _log(f"monitoring pid={pid} log={log_path} stall_min={args.stall_min}")

    if not log_path.exists():
        _log(f"FATAL: log file {log_path} does not exist; exiting")
        sys.exit(1)
    if not _pid_alive(pid):
        _log(f"FATAL: pid {pid} not alive at start; exiting")
        sys.exit(1)

    while True:
        time.sleep(args.check_interval_s)

        if not _pid_alive(pid):
            _log(f"pid {pid} no longer alive (clean exit or external kill); "
                 f"watchdog shutting down")
            sys.exit(0)

        try:
            mtime = log_path.stat().st_mtime
        except OSError as e:
            _log(f"could not stat log file: {e}; sleeping + retrying")
            continue

        stall = time.time() - mtime
        if stall >= stall_seconds:
            _log(f"STALL DETECTED: log mtime is {stall:.0f}s old "
                 f"(threshold {stall_seconds:.0f}s). Killing process tree.")
            _sigkill_tree(pid)
            # Give kernel a moment to reap, then verify
            time.sleep(5)
            if _pid_alive(pid):
                _log(f"WARNING: pid {pid} still alive after SIGKILL — "
                     f"may be uninterruptible (D state). Cannot recover.")
            else:
                _log(f"pid {pid} confirmed dead. Watchdog exiting.")
            sys.exit(0)
        elif stall >= stall_seconds * 0.5:
            # Halfway warning — useful for tuning the stall threshold
            _log(f"log mtime is {stall:.0f}s old (warn at {stall_seconds * 0.5:.0f}s, "
                 f"kill at {stall_seconds:.0f}s)")


if __name__ == "__main__":
    main()
