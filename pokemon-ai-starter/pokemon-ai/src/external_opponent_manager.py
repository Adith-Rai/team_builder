"""external_opponent_manager.py — spawn + supervise external Showdown bots.

Used by train_rl.py when --external-opponents is set. Each entry in the YAML
config spawns a subprocess (Foul Play, Metamon variants, etc.) that connects
to one of our local battle servers as a unique Showdown username. PPO
collection then challenges those usernames as additional PFSP opponents.

Design choices:
- Subprocess (not Python import) — avoids dependency conflicts between e.g.
  Metamon's amago stack and our torch/poke-env. Each bot has its own venv.
- Auto-restart with exponential backoff (per-opponent) — so a flaky external
  bot doesn't kill the training run.
- Logs go to per-opponent files, not stdout, so we don't drown the train log.

Not in this file:
- The PFSP integration — pool entries that target external usernames vs .pt
  paths is wired in train_rl.py / rl_collection.py.
- Live team-builder calls (e.g. Metamon's TeamPredictor) — that's handled by
  MultiSourceTeambuilder in team_generator.py, separately.
"""
from __future__ import annotations

import os
import sys
import time
import yaml
import signal
import logging
import threading
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExternalOpponent:
    name: str
    showdown_username: str
    command: List[str]
    cwd: Optional[str] = None
    venv: Optional[str] = None              # path to venv root (Scripts/python.exe used)
    auto_restart: bool = True
    log_file: Optional[str] = None
    description: str = ""

    # runtime state
    proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    n_restarts: int = 0
    started_at: float = 0.0


class ExternalOpponentManager:
    """Spawn + supervise N external Showdown bot subprocesses."""

    def __init__(self, config_path: str, base_dir: Optional[Path] = None):
        self.config_path = Path(config_path)
        # Paths in YAML are relative to the YAML file's directory by default.
        self.base_dir = base_dir or self.config_path.parent
        self.opponents: List[ExternalOpponent] = []
        self.servers: List[str] = []
        self.default_team_folder: Optional[str] = None
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._load_config()

    # ── Config loading ────────────────────────────────────────────
    def _load_config(self):
        with open(self.config_path) as f:
            cfg = yaml.safe_load(f)
        self.servers = cfg.get('servers', [])
        self.default_team_folder = cfg.get('default_team_folder')
        for entry in cfg.get('opponents', []):
            # Filter to known fields to avoid TypeError on extras.
            allowed = {'name', 'showdown_username', 'command', 'cwd', 'venv',
                       'auto_restart', 'log_file', 'description'}
            kwargs = {k: v for k, v in entry.items() if k in allowed}
            self.opponents.append(ExternalOpponent(**kwargs))
        logger.info(f"Loaded {len(self.opponents)} external opponents from {self.config_path}")

    # ── Process spawning ──────────────────────────────────────────
    def _resolve_path(self, p: Optional[str]) -> Optional[Path]:
        if p is None:
            return None
        path = Path(p)
        if not path.is_absolute():
            path = (self.base_dir / path).resolve()
        return path

    def _venv_python(self, venv_path: Path) -> Path:
        if os.name == 'nt':
            return venv_path / 'Scripts' / 'python.exe'
        return venv_path / 'bin' / 'python'

    def _spawn(self, opp: ExternalOpponent):
        cmd = list(opp.command)

        # Substitute python interpreter from venv if specified.
        if opp.venv:
            venv = self._resolve_path(opp.venv)
            python = self._venv_python(venv) if venv else None
            if python and python.exists() and cmd and cmd[0] == 'python':
                cmd[0] = str(python)
            elif cmd and cmd[0] == 'python':
                logger.warning(f"{opp.name}: venv python not found at {python}; "
                               f"falling back to system python")

        # On respawn (n_restarts > 0), tell the subprocess NOT to wipe its team
        # queue on startup. The trainer pre-enqueues all n_battles teams for the
        # iter; if the subprocess crashed mid-iter and we let the restarted one
        # wipe the queue, the iter's remaining teams are lost and the subprocess
        # sits idle until the trainer's per-opponent wait_for fires (~5 min),
        # corrupting throughput. First start keeps the default (clean=true) so
        # leftover .team files from prior runs don't confuse the new iter. If
        # the user set --clean-on-init in the YAML cmd, we strip it before
        # appending the override so there's only one value on the cmdline.
        if opp.n_restarts > 0 and any(
            cmd_part.endswith("foul_play_accept_serve.py")
            or cmd_part.endswith("metamon_accept_serve.py")
            for cmd_part in cmd
        ):
            stripped = []
            i = 0
            while i < len(cmd):
                if cmd[i] == "--clean-on-init":
                    i += 2  # skip flag + its value
                    continue
                stripped.append(cmd[i])
                i += 1
            cmd = stripped + ["--clean-on-init", "false"]

        cwd = self._resolve_path(opp.cwd)

        # Open log file (append mode so restarts share a log).
        log_path = self._resolve_path(opp.log_file) if opp.log_file else None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = open(log_path, 'a', buffering=1)
        else:
            log_fh = subprocess.DEVNULL

        # logger.warning (not info) so it survives Python's default lastResort
        # filter — we want every (re)spawn to be visible in training.log,
        # otherwise the user sees "<opp> exited" without the matching restart
        # acknowledgement and may think the subprocess is dead permanently.
        logger.warning(f"Spawning {opp.name} (user={opp.showdown_username}, restarts={opp.n_restarts})")
        opp.proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        opp.started_at = time.time()

    # ── Lifecycle ─────────────────────────────────────────────────
    def start_all(self):
        for opp in self.opponents:
            self._spawn(opp)
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name='ExtOppMonitor'
        )
        self._monitor_thread.start()

    # Per-launcher "ready" markers we tail the log file for. Each launcher
    # prints these once it has logged into Showdown and entered its accept
    # loop. metamon's marker fires after the env constructor's `Laddering
    # for N battles`. foul_play prints "iter 1/N — waiting for team in queue".
    _READY_MARKERS = (
        "iter 1/",                  # foul_play_accept_serve.py
        "metamon-accept] iter 1/",  # metamon_accept_serve.py
        "Laddering for",            # metamon parent QueueOnLocalLadder fallback
    )

    def wait_until_ready(self, per_opp_timeout_s: float = 120.0) -> bool:
        """Block until every spawned opponent's log file contains a "ready"
        marker, or per_opp_timeout_s elapses for any. Returns True if all
        opponents reported ready, False if any timed out (caller's choice
        whether to abort or proceed).

        Without this, train_rl can dive into a collection wave before the
        subprocesses have logged into Showdown, then sit in send_challenges
        timeouts for many minutes. Metamon's model-load + amago env
        construction takes ~30s; Foul Play's data-load ~10s.
        """
        all_ready = True
        for opp in self.opponents:
            if not opp.log_file:
                # No log file — can't wait, just sleep a little to let it spawn.
                time.sleep(2)
                continue
            log_path = self._resolve_path(opp.log_file)
            deadline = time.time() + per_opp_timeout_s
            ready = False
            while time.time() < deadline:
                # Subprocess might have died without ever writing to the log.
                if opp.proc and opp.proc.poll() is not None:
                    logger.warning(
                        f"{opp.name} exited rc={opp.proc.returncode} before signaling ready"
                    )
                    all_ready = False
                    break
                try:
                    if log_path and log_path.exists():
                        with open(log_path, 'r', errors='ignore') as f:
                            text = f.read()
                        if any(m in text for m in self._READY_MARKERS):
                            ready = True
                            break
                except OSError:
                    pass
                time.sleep(1)
            if ready:
                logger.info(f"{opp.name} ready (after {time.time() - (deadline - per_opp_timeout_s):.1f}s)")
            else:
                logger.warning(f"{opp.name} did NOT signal ready in {per_opp_timeout_s:.0f}s")
                all_ready = False
        return all_ready

    def stop_all(self, timeout: float = 10.0):
        self._stop_event.set()
        for opp in self.opponents:
            if opp.proc and opp.proc.poll() is None:
                logger.info(f"Stopping {opp.name}")
                if os.name == 'nt':
                    opp.proc.terminate()
                else:
                    opp.proc.send_signal(signal.SIGINT)
        # Wait for graceful exit, then kill stragglers.
        deadline = time.time() + timeout
        for opp in self.opponents:
            if opp.proc:
                remaining = max(0.0, deadline - time.time())
                try:
                    opp.proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    logger.warning(f"{opp.name} did not exit; killing")
                    opp.proc.kill()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

    def _tail_log(self, opp: ExternalOpponent, n_lines: int = 30) -> str:
        """Best-effort tail of the subprocess's log file, for crash post-mortems.

        Logged alongside the exit warning so the next session has the last
        words of the dying process without needing to grep through gigabyte
        log files. Silent on any error (we never want monitor failures to
        mask the underlying crash).
        """
        log_path = self._resolve_path(opp.log_file) if opp.log_file else None
        if not log_path or not log_path.exists():
            return "(no log file)"
        try:
            with open(log_path, "r", errors="ignore") as f:
                lines = f.readlines()
            tail = lines[-n_lines:] if len(lines) > n_lines else lines
            return "".join(tail).rstrip()
        except OSError:
            return "(log unreadable)"

    # If a subprocess's log file hasn't been touched in this many seconds AND
    # `Popen.poll()` still returns None (process technically alive), treat it
    # as a zombie and force-kill+respawn. Threshold needs to be longer than the
    # longest legitimate quiet window (= QueueTeambuilder's --queue-wait-timeout-s
    # of 14400s = 4 hours, after which the subprocess SHOULD raise+exit on its
    # own). In S43 production, MMs idled out and amago's env loop swallowed the
    # RuntimeError, leaving Popen.poll() == None forever — manager never
    # respawned. Log-mtime catches that case: even a healthy idle FP/MM prints
    # *some* line per iter (the "iter N — waiting for team" line), so a stale
    # mtime past this threshold is a strong dead-zombie signal.
    _LIVENESS_MTIME_THRESHOLD_S = 5400.0  # 90 min

    def _is_zombie(self, opp: ExternalOpponent) -> bool:
        """True iff Popen says alive but log mtime says dead."""
        if opp.proc is None or opp.proc.poll() is not None:
            return False  # not alive, regular exit-detection path handles it
        if not opp.log_file:
            return False  # no log file — can't check mtime
        log_path = self._resolve_path(opp.log_file)
        if not log_path or not log_path.exists():
            return False
        try:
            stale_s = time.time() - log_path.stat().st_mtime
        except OSError:
            return False
        # Also require the subprocess to have been up for at least the threshold;
        # otherwise an old log file from a prior run masquerades as stale.
        if (time.time() - opp.started_at) < self._LIVENESS_MTIME_THRESHOLD_S:
            return False
        return stale_s > self._LIVENESS_MTIME_THRESHOLD_S

    def _monitor_loop(self):
        """Background thread: detect exited (or zombie) processes and restart."""
        while not self._stop_event.is_set():
            for opp in self.opponents:
                if opp.proc is None:
                    continue
                rc = opp.proc.poll()
                if rc is None:
                    # Process technically alive — but is it actually doing work?
                    # Layered check for the S43 zombie pattern (amago swallowed
                    # the QueueTeambuilder RuntimeError, MM stayed alive but
                    # silent, manager never detected).
                    if self._is_zombie(opp):
                        log_path = self._resolve_path(opp.log_file) if opp.log_file else None
                        try:
                            stale_min = (time.time() - log_path.stat().st_mtime) / 60 if log_path else -1
                        except OSError:
                            stale_min = -1
                        logger.warning(
                            f"{opp.name} ZOMBIE detected (Popen alive but log "
                            f"stale {stale_min:.0f} min). Killing + respawning."
                        )
                        try:
                            opp.proc.kill()
                            opp.proc.wait(timeout=10)
                        except Exception as e:
                            logger.warning(f"  (kill failed: {e})")
                        rc = opp.proc.returncode if opp.proc else -1
                        # Fall through to the exit-handling block below.
                    else:
                        continue  # still running normally
                uptime = time.time() - opp.started_at
                # Log the exit AND a tail of the subprocess log. Without the
                # tail, post-mortem requires grepping a giant rolling log; with
                # it, the proximate cause (traceback / WS error) is visible
                # alongside the manager's restart line. This is the only data
                # source we have for diagnosing the not-yet-root-caused FP/MM
                # `ConnectionClosedError` crash mode at 6+ slots.
                tail = self._tail_log(opp, n_lines=30)
                logger.warning(
                    f"{opp.name} exited rc={rc} after {uptime:.0f}s "
                    f"(total restarts={opp.n_restarts})\n"
                    f"  --- last 30 lines of {opp.log_file or '(no log)'} ---\n"
                    f"{tail}\n"
                    f"  --- end tail ---"
                )
                if not opp.auto_restart:
                    opp.proc = None
                    continue
                opp.n_restarts += 1
                # Exponential backoff capped at 60s. Crash loops won't melt CPU.
                backoff = min(60, 2 ** min(opp.n_restarts, 6))
                if self._stop_event.wait(backoff):
                    return
                self._spawn(opp)
            # Poll every 5s; cheaper than per-process tight loops.
            self._stop_event.wait(5)

    # ── Pool integration ──────────────────────────────────────────
    def get_usernames(self) -> List[str]:
        """Showdown usernames the PFSP pool should challenge."""
        return [opp.showdown_username for opp in self.opponents]

    def is_alive(self, name: str) -> bool:
        """True if the named opponent's process is currently running."""
        for opp in self.opponents:
            if opp.name == name:
                return opp.proc is not None and opp.proc.poll() is None
        return False

    def status(self) -> List[dict]:
        """Snapshot of all opponents — for logging during training."""
        out = []
        for opp in self.opponents:
            alive = opp.proc is not None and opp.proc.poll() is None
            out.append({
                'name': opp.name,
                'username': opp.showdown_username,
                'alive': alive,
                'restarts': opp.n_restarts,
                'uptime_s': (time.time() - opp.started_at) if alive else 0,
            })
        return out


if __name__ == '__main__':
    # Smoke test: launch from a config and print status every 5s.
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='Path to external_opponents.yaml')
    p.add_argument('--duration', type=int, default=60,
                   help='Seconds to run before stopping (smoke test)')
    args = p.parse_args()

    mgr = ExternalOpponentManager(args.config)
    mgr.start_all()
    try:
        deadline = time.time() + args.duration
        while time.time() < deadline:
            time.sleep(5)
            for s in mgr.status():
                print(f"  {s['name']:30s} alive={s['alive']} restarts={s['restarts']} uptime={s['uptime_s']:.0f}s")
    finally:
        mgr.stop_all()
