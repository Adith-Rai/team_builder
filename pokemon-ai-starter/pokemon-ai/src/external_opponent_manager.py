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

        cwd = self._resolve_path(opp.cwd)

        # Open log file (append mode so restarts share a log).
        log_path = self._resolve_path(opp.log_file) if opp.log_file else None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = open(log_path, 'a', buffering=1)
        else:
            log_fh = subprocess.DEVNULL

        logger.info(f"Spawning {opp.name} (user={opp.showdown_username}, restarts={opp.n_restarts})")
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

    def _monitor_loop(self):
        """Background thread: detect exited processes and restart if configured."""
        while not self._stop_event.is_set():
            for opp in self.opponents:
                if opp.proc is None:
                    continue
                rc = opp.proc.poll()
                if rc is None:
                    continue  # still running
                uptime = time.time() - opp.started_at
                logger.warning(
                    f"{opp.name} exited rc={rc} after {uptime:.0f}s "
                    f"(total restarts={opp.n_restarts})"
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
