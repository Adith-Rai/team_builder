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
    # S67-ext per-iter spawn (2026-05-28): the LOGICAL opp name this instance
    # belongs to. For single-instance opps it's the same as `name`. For
    # multi-instance, name = "{logical}-{i}" but logical_name stays at the
    # base (e.g., "mm-minikazam"). Used by spawn_active(logical_names) to
    # find all instances of a given logical opp without prefix-string parsing.
    logical_name: Optional[str] = None

    # runtime state
    proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    n_restarts: int = 0
    started_at: float = 0.0
    # S67-ext (2026-05-28): byte offset in log_file at spawn time. Used by
    # _wait_one_ready to scan ONLY this spawn's output, not stale markers
    # from prior spawns (log file is append-mode for crash-debug preservation;
    # without this offset the marker scan finds OLD "iter 1/" entries from
    # an earlier spawn and returns ready immediately, causing the MM race).
    spawn_log_pos: int = 0


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
        # S67-ext per-iter spawn (2026-05-28): set of logical opp names
        # currently "active" (spawned for this iter). Empty when between
        # iters (release_all clears it). Monitor loop only auto-restarts
        # opps whose logical_name is in this set — prevents respawn of
        # opps that were deliberately killed by release_all.
        self._active_logical: set = set()
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
                       'auto_restart', 'log_file', 'description', 'logical_name'}
            kwargs = {k: v for k, v in entry.items() if k in allowed}
            opp = ExternalOpponent(**kwargs)
            if opp.logical_name is None:
                opp.logical_name = opp.name  # single-instance default
            self.opponents.append(opp)
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
        # Record byte offset BEFORE opening so _wait_one_ready can scan
        # only this spawn's output (prior spawns wrote ready markers to
        # the same file and would falsely satisfy the check).
        log_path = self._resolve_path(opp.log_file) if opp.log_file else None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                opp.spawn_log_pos = log_path.stat().st_size if log_path.exists() else 0
            except OSError:
                opp.spawn_log_pos = 0
            log_fh = open(log_path, 'a', buffering=1)
        else:
            opp.spawn_log_pos = 0
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
                        # Seek past pre-spawn content so stale markers from
                        # prior restarts don't satisfy this check.
                        with open(log_path, 'rb') as f:
                            f.seek(opp.spawn_log_pos)
                            text = f.read().decode('utf-8', errors='ignore')
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

    # ── S67-ext per-iter spawn architecture (2026-05-28) ──────────
    # Unlike start_all (spawn all at boot, keep alive forever), per-iter
    # spawn matches MM subprocess lifecycle to the iter that uses them:
    #
    #   - iter start: spawn_active(active_logical_names) — spawns ONLY the
    #     MMs the composition picked for THIS iter, in parallel. SP/MCTS
    #     workers can fire immediately (battle_server pendingChallenges
    #     queues their /challenges; MMs serve them after login).
    #   - end of collect: release_all() — kills active MMs, waits clean
    #     shutdown. Frees GPU (model weights + CUDA contexts) before main
    #     process's PPO update peak.
    #   - update runs with full GPU.
    #   - next iter: spawn_active() with NEW composition's active set.
    #
    # Why this fits the architectural ceiling: only ACTIVE MMs consume GPU
    # during collect (e.g., 3 of 5 picked → 3 × N instances alive, not all
    # 5 × N). Pool size becomes virtually unbounded — can add Kakuna,
    # Superkazam, etc., without GPU pressure, since only picked-active
    # ones load. Cost: ~60-90s parallel MM startup per iter (overlapped
    # with SP/MCTS collect — net iter wall +1-2 min).

    def start_monitor_only(self):
        """Start the monitor thread WITHOUT spawning anything. Use with
        per-iter spawn architecture where spawn_active() controls lifecycle.

        Alternative entry to start_all(): start_all spawns all opponents
        immediately + starts monitor (legacy spawn-at-boot pattern);
        start_monitor_only just runs the monitor (waits for spawn_active
        to fire spawns)."""
        if self._monitor_thread is not None:
            return
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name='ExtOppMonitor'
        )
        self._monitor_thread.start()

    def _wait_one_ready(self, opp: ExternalOpponent, timeout_s: float) -> bool:
        """Wait for a single opponent's log to contain a ready marker emitted
        by THIS spawn (not a previous spawn's marker — the log file is
        append-mode across restarts, so we seek past the pre-spawn offset
        recorded in `_spawn`).

        Extracted from wait_until_ready loop so spawn_active can poll
        individual opps (e.g., to report readiness incrementally)."""
        if not opp.log_file:
            time.sleep(2)
            return True
        log_path = self._resolve_path(opp.log_file)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if opp.proc and opp.proc.poll() is not None:
                logger.warning(
                    f"{opp.name} exited rc={opp.proc.returncode} before ready"
                )
                return False
            try:
                if log_path and log_path.exists():
                    with open(log_path, 'rb') as f:
                        f.seek(opp.spawn_log_pos)
                        new_bytes = f.read()
                    text = new_bytes.decode('utf-8', errors='ignore')
                    if any(m in text for m in self._READY_MARKERS):
                        return True
            except OSError:
                pass
            time.sleep(1)
        logger.warning(f"{opp.name} did NOT signal ready in {timeout_s:.0f}s")
        return False

    def spawn_active(self, active_logical: set,
                     wait_ready: bool = True,
                     per_opp_timeout_s: float = 180.0) -> dict:
        """Spawn all instances of the listed logical opp names in parallel.

        Idempotent: instances already alive are skipped (no double-spawn).
        Returns a map of {instance_name: is_ready_bool}.

        Set the active_logical set as authoritative: monitor loop will
        only auto-restart opps whose logical_name is in this set, so any
        opps NOT in active_logical that are alive will NOT be respawned
        on death this iter.

        For per-iter spawn architecture:
          - SP/MCTS workers can fire collect immediately (don't wait).
          - MM workers' /challenges queue in battle_server pendingChallenges
            until their assigned MM finishes login (bug 10 fix handles the
            resend on /trn login).
        """
        self._active_logical = set(active_logical)
        spawned: List[ExternalOpponent] = []
        results: dict = {}
        for opp in self.opponents:
            if opp.logical_name not in active_logical:
                results[opp.name] = False
                continue
            # Already alive — skip (idempotent)
            if opp.proc is not None and opp.proc.poll() is None:
                results[opp.name] = True
                continue
            # Spawn fresh
            opp.n_restarts = 0  # reset restart counter (fresh iter)
            self._spawn(opp)
            spawned.append(opp)

        # Wait for spawned opps to signal ready (in parallel — each opp's
        # readiness is independent of others; we poll sequentially but the
        # spawning happens in parallel because each spawn returns immediately
        # after Popen, before the subprocess finishes initialization).
        if wait_ready and spawned:
            for opp in spawned:
                results[opp.name] = self._wait_one_ready(opp, per_opp_timeout_s)
        logger.info(
            f"spawn_active: {len(active_logical)} logical opps requested, "
            f"{len(spawned)} new instances spawned, "
            f"{sum(1 for k, v in results.items() if v)}/{len(results)} ready"
        )
        return results

    def release_all(self, timeout_s: float = 15.0) -> int:
        """Kill all currently-alive opponents, wait for clean shutdown.

        Clears active_logical set so monitor thread won't auto-restart
        the killed opps. Returns count of opponents killed.

        Used between collect and update in per-iter spawn architecture:
        frees GPU before main process's PPO update memory peak. Kills
        in parallel (SIGKILL all, then wait for each).
        """
        # Clear active set FIRST so monitor doesn't race to respawn during kill
        self._active_logical = set()
        # Phase 1: SIGKILL all alive opps in parallel (don't wait yet)
        killed = 0
        for opp in self.opponents:
            if opp.proc is None or opp.proc.poll() is not None:
                continue
            try:
                opp.proc.kill()
                killed += 1
            except Exception as e:
                logger.warning(f"release_all: kill of {opp.name} failed: {e}")
        # Phase 2: wait for all to actually die. Since SIGKILL was sent in
        # parallel above, the deaths happen concurrently — this sequential
        # wait just collects the results. Total wall ≈ slowest single kill.
        t0 = time.time()
        deadline = t0 + timeout_s
        for opp in self.opponents:
            if opp.proc is None:
                continue
            remaining = max(0.5, deadline - time.time())
            try:
                opp.proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                logger.warning(f"release_all: {opp.name} did not exit in {timeout_s:.0f}s")
            opp.proc = None  # clear so next spawn_active knows to (re)spawn
        elapsed = time.time() - t0
        logger.info(f"release_all: killed {killed} opponents in {elapsed:.1f}s")
        return killed

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
    # as a zombie and force-kill+respawn. Designed in concert with the heartbeat
    # threads in foul_play_accept_serve.py and metamon_accept_serve.py: a healthy
    # subprocess prints a `[heartbeat HH:MM:SS]` line every 60s regardless of
    # what its main loop is doing, so log mtime stays fresh whenever the process
    # is actually running. This means a stale mtime past this threshold is a
    # very strong "the process scheduler isn't running" signal — much more
    # specific than the prior 90-min "I haven't been sampled in a while" check
    # which falsely tripped on legitimately-idle MMs (PFSP correctly under-
    # samples mastered opponents — see S43 attempt 3 cascade).
    #
    # 10 min is generous enough to absorb a slow heartbeat (e.g. heavy PPO
    # update on the trainer side temporarily starving subprocess scheduling)
    # but tight enough to catch true hangs quickly.
    _LIVENESS_MTIME_THRESHOLD_S = 600.0  # 10 min

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
                # S67-ext per-iter spawn (2026-05-28): only respawn if this
                # opp's logical_name is in the active set. Otherwise it was
                # deliberately killed by release_all (between collect and
                # update) — don't fight that by respawning.
                # When active_logical is empty (no iter in progress), nothing
                # respawns regardless of auto_restart.
                if (self._active_logical
                        and opp.logical_name not in self._active_logical):
                    opp.proc = None
                    continue
                if not self._active_logical:
                    # Between iters — no respawning. Clear and wait.
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

    def restart_subprocess(self, name: str) -> bool:
        """Force-kill a named subprocess; the monitor thread will respawn it.

        Used by the trainer's dispatch watchdog (Layer 4) when an opponent
        gets stuck in the silent "logged in but _challenge_queue not bound"
        state — Popen still alive, heartbeats firing, but never accepting
        challenges. Layer 2's exit-detection won't fire (the proc didn't
        crash); Layer 2's zombie-detection won't fire (heartbeats keep log
        mtime fresh). The watchdog has the only signal that something's
        wrong (no battles finishing) and uses this to escalate.

        Returns True if a process was killed, False if not found / not alive.
        """
        for opp in self.opponents:
            if opp.name != name:
                continue
            if opp.proc is None or opp.proc.poll() is not None:
                logger.warning(f"restart_subprocess({name}): not currently alive")
                return False
            logger.warning(
                f"restart_subprocess({name}): force-killing for stall recovery; "
                f"monitor thread will respawn"
            )
            try:
                opp.proc.kill()
            except Exception as e:
                logger.warning(f"  (kill failed for {name}: {e})")
                return False
            return True
        logger.warning(f"restart_subprocess({name}): no opponent with that name")
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
