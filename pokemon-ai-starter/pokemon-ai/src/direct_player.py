"""
direct_player.py - Drop-in replacement for poke-env's websocket transport.

Uses battle_worker.js subprocess instead of a Showdown server websocket.
The key insight: poke-env's websocket boundary is only 2 methods in PSClient
(listen + send_message). We replace those and everything else works.

Architecture:
    battle_worker.js  <-stdin/stdout->  DirectClient (replaces PSClient)
                                          | calls _handle_message()
                                        Player._handle_battle_message() (unchanged)
                                          | calls choose_move()
                                        User's AI (unchanged)
                                          | returns BattleOrder
                                        DirectClient.send_message() -> stdin write
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from asyncio import Condition, Event, Lock, Queue, Semaphore
from logging import Logger
from pathlib import Path
from time import perf_counter
from typing import Any, Awaitable, Dict, List, Optional, Set, Tuple, Union

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.battle import Battle
from poke_env.battle.double_battle import DoubleBattle
from poke_env.concurrency import POKE_LOOP, create_in_poke_loop, handle_threaded_coroutines
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
    SingleBattleOrder,
)
from poke_env.player.player import Player
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder
from poke_env.teambuilder.teambuilder import Teambuilder

# Path to battle_worker.js relative to this file
WORKER_JS = str(Path(__file__).parent / "battle_worker.js")

# Restart the Node subprocess every N battles to prevent memory buildup
WORKER_RESTART_INTERVAL = 20


# ---------------------------------------------------------------------------
# WorkerProcess — manages the Node.js subprocess lifecycle
# ---------------------------------------------------------------------------

class WorkerProcess:
    """Manages a single battle_worker.js Node subprocess.

    Handles starting, restarting, sending JSON commands, and reading JSON
    responses. Thread-safe for use from the POKE_LOOP asyncio event loop.
    """

    def __init__(self, worker_path: str = WORKER_JS):
        self._worker_path = worker_path
        self._proc: Optional[subprocess.Popen] = None
        self._battle_count = 0
        self._active_battles = 0
        self._lock = create_in_poke_loop(Lock)
        self._reader_task: Optional[asyncio.Task] = None
        self._handlers: Dict[str, asyncio.Queue] = {}  # battle_id -> queue of messages
        self._global_handlers: List[asyncio.Queue] = []
        self._started = False
        self._logger = logging.getLogger("WorkerProcess")

    def _start_process(self):
        """Start the Node.js subprocess."""
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass

        self._proc = subprocess.Popen(
            ["node", self._worker_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            # Prevent Windows from opening a console window
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        self._logger.debug("Started worker process pid=%d", self._proc.pid)

    async def ensure_started(self):
        """Ensure the worker process is running, start if needed."""
        async with self._lock:
            if self._started and self._proc is not None and self._proc.poll() is None:
                return
            self._start_process()
            self._started = True

            # Start the reader coroutine on POKE_LOOP
            if self._reader_task is not None and not self._reader_task.done():
                self._reader_task.cancel()
            self._reader_task = asyncio.ensure_future(
                self._read_loop(), loop=POKE_LOOP
            )

            # Wait for the "ready" message
            ready_q: asyncio.Queue = asyncio.Queue()
            self._global_handlers.append(ready_q)
            try:
                msg = await asyncio.wait_for(ready_q.get(), timeout=15.0)
                if msg.get("type") != "ready":
                    self._logger.warning("Expected 'ready', got: %s", msg)
            except asyncio.TimeoutError:
                raise RuntimeError("battle_worker.js did not send 'ready' within 15s")
            finally:
                self._global_handlers.remove(ready_q)

    async def _read_loop(self):
        """Continuously read JSON lines from stdout and dispatch them."""
        loop = asyncio.get_event_loop()
        proc = self._proc
        assert proc is not None and proc.stdout is not None

        try:
            while proc.poll() is None:
                # Read a line in a thread to avoid blocking the event loop
                line = await loop.run_in_executor(None, proc.stdout.readline)
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    self._logger.warning("Non-JSON from worker: %s", line[:200])
                    continue

                battle_id = msg.get("id")

                # Dispatch to global handlers first (for ready, etc.)
                for q in self._global_handlers:
                    await q.put(msg)

                # Dispatch to battle-specific handler
                if battle_id and battle_id in self._handlers:
                    await self._handlers[battle_id].put(msg)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._logger.error("Worker read loop error: %s", e)

    async def send(self, data: dict):
        """Send a JSON message to the worker's stdin."""
        await self.ensure_started()
        proc = self._proc
        assert proc is not None and proc.stdin is not None
        line = json.dumps(data) + "\n"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, proc.stdin.write, line.encode())
        await loop.run_in_executor(None, proc.stdin.flush)

    def register_battle(self, battle_id: str) -> asyncio.Queue:
        """Register a handler queue for a battle ID. Returns the queue."""
        q: asyncio.Queue = asyncio.Queue()
        self._handlers[battle_id] = q
        self._active_battles += 1
        return q

    def unregister_battle(self, battle_id: str):
        """Remove the handler for a battle ID."""
        if battle_id in self._handlers:
            self._active_battles = max(0, self._active_battles - 1)
        self._handlers.pop(battle_id, None)

    async def maybe_restart(self, n_battles: int = 1):
        """Restart the worker if battle count exceeds the threshold.

        Only restarts when no battles are currently active to avoid killing
        the subprocess while other battles are in-flight.
        """
        self._battle_count += n_battles
        if self._battle_count >= WORKER_RESTART_INTERVAL and self._active_battles == 0:
            self._logger.debug(
                "Restarting worker after %d battles", self._battle_count
            )
            self._battle_count = 0
            await self.shutdown()
            # Will be restarted on next ensure_started()

    async def shutdown(self):
        """Kill the worker process."""
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            self._reader_task = None
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None
        self._started = False
        self._handlers.clear()


# Global shared worker instance
_global_worker: Optional[WorkerProcess] = None
_global_worker_lock = create_in_poke_loop(Lock)


async def get_global_worker() -> WorkerProcess:
    """Get or create the global WorkerProcess singleton."""
    global _global_worker
    async with _global_worker_lock:
        if _global_worker is None:
            _global_worker = WorkerProcess()
        await _global_worker.ensure_started()
        return _global_worker


# ---------------------------------------------------------------------------
# DirectClient — replaces PSClient
# ---------------------------------------------------------------------------

class DirectClient:
    """Drop-in replacement for PSClient that communicates via battle_worker.js.

    Supports the same interface that Player.__init__ monkey-patches onto:
      - _handle_battle_message (set by Player)
      - _update_challenges (set by Player)
      - _handle_challenge_request (set by Player)

    And the methods Player calls:
      - send_message(message, room)
      - logged_in (Event)
      - username (str)
      - logger (Logger)
    """

    def __init__(
        self,
        account_configuration: AccountConfiguration,
        *,
        log_level: Optional[int] = None,
    ):
        self._account_configuration = account_configuration
        self._logged_in: Event = create_in_poke_loop(Event)
        self._sending_lock = create_in_poke_loop(Lock)
        self._active_tasks: Set[Any] = set()
        self._logger = self._create_logger(log_level)

        # These will be monkey-patched by Player.__init__
        self._handle_battle_message = None
        self._update_challenges = None
        self._handle_challenge_request = None

        # Battle tag -> internal battle_id mapping
        self._battle_tag_to_id: Dict[str, str] = {}
        self._battle_tag_to_player: Dict[str, str] = {}  # battle_tag -> "p1" or "p2"

        # Immediately mark as logged in since there's no auth
        self._logged_in.set()

    def _create_logger(self, log_level: Optional[int]) -> Logger:
        logger = logging.getLogger(f"DirectClient-{self.username}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        if log_level is not None:
            logger.setLevel(log_level)
        return logger

    async def _handle_message(self, message: str):
        """Handle a received message, mimicking PSClient._handle_message.

        The message should be in the same format as a websocket message from
        Showdown: first line is the battle tag (>battle-xxx), remaining lines
        are pipe-separated protocol messages.
        """
        try:
            split_messages = [m.split("|") for m in message.split("\n")]
            if split_messages[0][0].startswith(">battle"):
                if self._handle_battle_message is not None:
                    await self._handle_battle_message(split_messages)
            else:
                self._logger.debug("Ignoring non-battle message: %s", message[:100])
        except asyncio.CancelledError as e:
            self._logger.critical("CancelledError intercepted: %s", e)
        except Exception as e:
            self._logger.debug(
                "Error handling message (non-fatal): %s\n%s", e, message[:200]
            )

    async def send_message(
        self, message: str, room: str = "", message_2: Optional[str] = None
    ):
        """Send a message, translating poke-env format to worker protocol.

        poke-env calls this as: send_message("/choose move 1", "battle-gen9ou-1")
        We need to translate to: {"type":"choose","id":"b1","player":"p1","choice":"move 1"}
        """
        if message_2:
            full = "|".join([room, message, message_2])
        else:
            full = "|".join([room, message])
        self._logger.debug(">>> %s", full)

        # Handle battle room messages
        if room and room.startswith("battle-"):
            if message.startswith("/choose "):
                choice = message[len("/choose "):]
                battle_id = self._battle_tag_to_id.get(room)
                player = self._battle_tag_to_player.get(room)
                if battle_id and player:
                    worker = await get_global_worker()
                    await worker.send({
                        "type": "choose",
                        "id": battle_id,
                        "player": player,
                        "choice": choice,
                    })
                else:
                    self._logger.warning(
                        "send_message: no mapping for room %s", room
                    )
            elif message.startswith("/team "):
                # Team preview — send as a choice
                choice = "team " + message[len("/team "):]
                battle_id = self._battle_tag_to_id.get(room)
                player = self._battle_tag_to_player.get(room)
                if battle_id and player:
                    worker = await get_global_worker()
                    await worker.send({
                        "type": "choose",
                        "id": battle_id,
                        "player": player,
                        "choice": choice,
                    })
            elif message.startswith("/leave "):
                # poke-env sends /leave after battle ends — ignore
                pass
            elif message.startswith("/timer "):
                # Timer control — not applicable for direct battles
                pass
            else:
                self._logger.debug("Ignoring battle message: %s", message)
        elif message.startswith("/utm "):
            # Team setting — handled externally before battle start
            pass
        else:
            self._logger.debug("Ignoring global message: %s", message)

    def register_battle(self, battle_tag: str, battle_id: str, player: str):
        """Register the mapping from poke-env battle tag to worker battle ID."""
        self._battle_tag_to_id[battle_tag] = battle_id
        self._battle_tag_to_player[battle_tag] = player

    def unregister_battle(self, battle_tag: str):
        """Remove the mapping for a finished battle to prevent unbounded growth."""
        self._battle_tag_to_id.pop(battle_tag, None)
        self._battle_tag_to_player.pop(battle_tag, None)

    @property
    def account_configuration(self) -> AccountConfiguration:
        return self._account_configuration

    @property
    def logged_in(self) -> Event:
        return self._logged_in

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def username(self) -> str:
        return self._account_configuration.username


# ---------------------------------------------------------------------------
# DirectPlayer — base class for AI players using direct transport
# ---------------------------------------------------------------------------

class DirectPlayer(Player):
    """A Player subclass that uses DirectClient instead of PSClient.

    Subclass this instead of Player to use direct subprocess transport.
    All Player functionality (choose_move, battle handling, etc.) works
    unchanged.
    """

    def __init__(
        self,
        account_configuration: Optional[AccountConfiguration] = None,
        *,
        battle_format: str = "gen9randombattle",
        log_level: Optional[int] = None,
        max_concurrent_battles: int = 1,
        accept_open_team_sheet: bool = False,
        save_replays: Union[bool, str] = False,
        start_timer_on_battle_start: bool = False,
        team: Optional[Union[str, Teambuilder]] = None,
    ):
        # We need to bypass Player.__init__ because it creates a PSClient
        # that tries to connect to a websocket. Instead, we replicate the
        # init logic with our DirectClient.

        # DO NOT call super().__init__() — it creates PSClient with websocket
        # Instead, replicate Player.__init__ with DirectClient

        self.ps_client = DirectClient(
            account_configuration=account_configuration
            or AccountConfiguration.generate(self.__class__.__name__),
            log_level=log_level,
        )

        # Monkey-patch the same 3 methods Player normally patches onto PSClient
        self.ps_client._handle_battle_message = self._handle_battle_message
        self.ps_client._update_challenges = self._update_challenges
        self.ps_client._handle_challenge_request = self._handle_challenge_request

        self._format: str = battle_format
        self._max_concurrent_battles: int = max_concurrent_battles
        self._save_replays = save_replays
        self._start_timer_on_battle_start: bool = start_timer_on_battle_start
        self._accept_open_team_sheet: bool = accept_open_team_sheet

        self._battles: Dict[str, AbstractBattle] = {}
        self._battle_semaphore: Semaphore = create_in_poke_loop(Semaphore, 0)
        self._battle_start_condition: Condition = create_in_poke_loop(Condition)
        self._battle_count_queue: Queue[Any] = create_in_poke_loop(
            Queue, max_concurrent_battles
        )
        self._battle_end_condition: Condition = create_in_poke_loop(Condition)
        self._challenge_queue: Queue[Any] = create_in_poke_loop(Queue)
        self._waiting: Event = create_in_poke_loop(Event)
        self._trying_again: Event = create_in_poke_loop(Event)
        self._team: Optional[Teambuilder] = None

        if isinstance(team, Teambuilder):
            self._team = team
        elif isinstance(team, str):
            self._team = ConstantTeambuilder(team)

        self.logger.debug("DirectPlayer initialisation finished")


class RandomDirectPlayer(DirectPlayer):
    """A DirectPlayer that makes random moves. Useful for testing."""

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        return self.choose_random_move(battle)


# ---------------------------------------------------------------------------
# direct_battle_against — orchestrator for direct subprocess battles
# ---------------------------------------------------------------------------

_battle_counter = 0


async def _run_single_battle(
    p1: Player,
    p2: Player,
    battle_format: str,
    battle_id: str,
    worker: WorkerProcess,
    timeout: float = 120.0,
) -> Optional[str]:
    """Run a single battle between two DirectPlayers.

    Returns the winner's username, or None for a tie.
    """
    # Get teams
    p1_team_packed = p1.next_team or ""
    p2_team_packed = p2.next_team or ""

    # Construct the poke-env battle tag that will appear in messages
    # The worker uses >{battle_id} as the tag prefix, which becomes the tag
    # "battle-{format}-{counter}" style that poke-env expects
    battle_tag = battle_id  # Worker sends >battle_id, so tag = battle_id

    # Register battle mappings on each player's client
    assert isinstance(p1.ps_client, DirectClient)
    assert isinstance(p2.ps_client, DirectClient)
    p1.ps_client.register_battle(battle_tag, battle_id, "p1")
    p2.ps_client.register_battle(battle_tag, battle_id, "p2")

    # Register message queue for this battle
    msg_queue = worker.register_battle(battle_id)

    # Start the battle in the worker
    await worker.send({
        "type": "start",
        "id": battle_id,
        "format": battle_format,
        "p1_team": p1_team_packed,
        "p2_team": p2_team_packed,
        "p1_name": p1.username,
        "p2_name": p2.username,
    })

    # Process messages until the battle ends
    winner = None
    battle_ended = False
    start_time = perf_counter()
    # Track whether we've injected |init|battle for each player.
    # BattleStream's player streams don't include |init|, but poke-env
    # requires it to create the Battle object.
    init_sent = {"p1": False, "p2": False}

    while not battle_ended:
        elapsed = perf_counter() - start_time
        if elapsed > timeout:
            p1.logger.warning("Battle %s timed out after %.1fs", battle_id, elapsed)
            break

        try:
            msg = await asyncio.wait_for(
                msg_queue.get(), timeout=min(30.0, timeout - elapsed)
            )
        except asyncio.TimeoutError:
            continue

        msg_type = msg.get("type")

        if msg_type == "sideupdate":
            # Route the raw message to the appropriate player's client
            player_key = msg.get("player")
            raw = msg.get("raw", "")
            if not raw:
                continue

            # The raw message from battle_worker has format:
            #   >{battle_id}\n|request|{...}\n|switch|...
            # This is exactly what poke-env expects from a websocket.
            # We need to rewrite the battle tag line to use the poke-env
            # format: >battle-{format}-{N}
            # Actually, the worker already sends >{battle_id} and we use
            # battle_id as the tag, so it works directly.

            if player_key == "p1":
                client = p1.ps_client
            elif player_key == "p2":
                client = p2.ps_client
            else:
                continue

            # Inject |init|battle on the first message for each player.
            # BattleStream player streams don't include this, but poke-env
            # needs it to create the Battle object in _handle_battle_message.
            if not init_sent[player_key]:
                # Insert |init|battle right after the battle tag line
                lines = raw.split("\n", 1)
                if len(lines) == 2:
                    raw = f"{lines[0]}\n|init|battle\n{lines[1]}"
                else:
                    raw = f"{raw}\n|init|battle"
                init_sent[player_key] = True

            # Fire-and-forget — same as PSClient.listen() does with
            # create_task. The _safe_handle_battle_message wrapper
            # (installed by patch_to_direct) prevents double |win|
            # deadlocks, so fire-and-forget is safe.
            asyncio.create_task(client._handle_message(raw))

        elif msg_type == "end":
            winner = msg.get("winner")
            battle_ended = True

            # Send |win|/|tie| to both players. If the sideupdate
            # already contained |win|, the wrapper will skip this one.
            if winner:
                end_line = f"|win|{winner}"
            else:
                end_line = "|tie"

            end_msg = f">{battle_tag}\n{end_line}"
            for client in [p1.ps_client, p2.ps_client]:
                assert isinstance(client, DirectClient)
                asyncio.create_task(client._handle_message(end_msg))
            await asyncio.sleep(0.05)  # let end tasks process

        elif msg_type == "error":
            p1.logger.error("Worker error for %s: %s", battle_id, msg.get("message"))

    # Cleanup worker and client mappings to prevent unbounded dict growth
    worker.unregister_battle(battle_id)
    for client in [p1.ps_client, p2.ps_client]:
        if isinstance(client, DirectClient):
            client.unregister_battle(battle_tag)
    return winner


async def direct_battle_against(
    p1: Player,
    p2: Player,
    n_battles: int = 1,
    battle_format: Optional[str] = None,
    timeout_per_battle: float = 120.0,
) -> Dict[str, int]:
    """Run n_battles between two Players using battle_worker.js.

    Players must have been patched with patch_to_direct() (or be DirectPlayer
    instances) so that their ps_client is a DirectClient.

    Concurrency is controlled by the players' max_concurrent_battles setting
    (same as poke-env's websocket mode). Battles run concurrently up to that
    limit via asyncio.Semaphore, matching poke-env's _send_challenges pattern.

    Args:
        p1: First player (patched or DirectPlayer).
        p2: Second player (patched or DirectPlayer).
        n_battles: Number of battles to play.
        battle_format: Battle format string (e.g. "gen9ou"). Defaults to p1's format.
        timeout_per_battle: Maximum seconds per battle before timing out.

    Returns:
        Dict with keys "p1_wins", "p2_wins", "ties", "errors".
    """
    global _battle_counter

    fmt = battle_format or p1._format
    results = {"p1_wins": 0, "p2_wins": 0, "ties": 0, "errors": 0}

    worker = await get_global_worker()

    # Use the lower of the two players' max_concurrent_battles as the limit.
    # We manage concurrency ourselves via asyncio.Semaphore rather than relying
    # on poke-env's _battle_count_queue (which can deadlock in direct mode
    # since _battle_finished_callback may not drain it synchronously).
    max_concurrent = min(
        p1._max_concurrent_battles or 1,
        p2._max_concurrent_battles or 1,
    )
    # Replace poke-env's bounded _battle_count_queue with unbounded.
    # Bounded queue deadlocks in direct mode when _create_battle.put()
    # blocks the event loop, preventing win handlers from calling get().
    for player in (p1, p2):
        player._battle_count_queue = asyncio.Queue(0)

    def classify_winner(winner):
        if winner == p1.username:
            return "p1_wins"
        elif winner == p2.username:
            return "p2_wins"
        elif winner is None:
            return "ties"
        elif winner in p1.username or p1.username in winner:
            return "p1_wins"
        elif winner in p2.username or p2.username in winner:
            return "p2_wins"
        return "ties"

    # Rolling pipeline with semaphore — matches poke-env's _send_challenges
    # pattern. As soon as one battle finishes, the next starts immediately.
    # No batch synchronization barriers. Uses gather for reliable completion
    # tracking (Event-based approach can miss if tasks raise exceptions).
    sem = asyncio.Semaphore(max_concurrent)

    async def run_one(battle_idx: int):
        nonlocal worker
        async with sem:
            bid = _next_battle_id(fmt)
            try:
                winner = await _run_single_battle(
                    p1, p2, fmt, bid, worker, timeout=timeout_per_battle
                )
                results[classify_winner(winner)] += 1
            except Exception as e:
                p1.logger.error("Battle %d error: %s", battle_idx + 1, e)
                results["errors"] += 1

            # Periodic worker restart
            await worker.maybe_restart()
            if not worker._started:
                worker = await get_global_worker()

    # Launch all battles as tasks (rolling pipeline via semaphore),
    # then gather to wait for completion. gather + semaphore gives us
    # rolling concurrency with reliable completion tracking.
    tasks = [asyncio.ensure_future(run_one(i)) for i in range(n_battles)]
    await asyncio.gather(*tasks, return_exceptions=True)

    return results


def _next_battle_id(fmt: str) -> str:
    """Generate a unique battle ID."""
    global _battle_counter
    _battle_counter += 1
    return f"battle-{fmt}-{_battle_counter}"


def patch_to_direct(player: Player) -> None:
    """Convert an existing Player instance to use direct transport.

    Call this AFTER Player.__init__ has completed (with start_listening=False
    to prevent websocket connections). Replaces the player's PSClient with a
    DirectClient so that direct_battle_against() can drive battles through
    the battle_worker.js subprocess.

    Example usage in orchestration files::

        player = SomePlayer(..., start_listening=False)
        patch_to_direct(player)
        # Now use direct_battle_against(player, opponent, n_battles=N)
    """
    dc = DirectClient(
        account_configuration=player.ps_client.account_configuration,
        log_level=None,
    )
    # Replace the client
    player.ps_client = dc

    # Wrap _handle_battle_message to prevent double |win|/|tie| processing.
    # In direct mode, player streams already contain |win|, AND _run_single_battle
    # sends an explicit |win| via the end handler. poke-env's _handle_battle_message
    # calls _battle_count_queue.get() on each |win|, so double |win| = deadlock.
    # This wrapper tracks which battles have had |win| processed and strips duplicates.
    original_hbm = player._handle_battle_message
    _processed_wins: set = set()

    async def _safe_handle_battle_message(split_messages):
        battle_tag = split_messages[0][0].lstrip(">") if split_messages else ""
        has_win = any(
            len(m) > 1 and m[1] in ("win", "tie")
            for m in split_messages[1:]
        )

        if has_win and battle_tag in _processed_wins:
            # Duplicate |win| — process everything EXCEPT win/tie lines
            filtered = [split_messages[0]] + [
                m for m in split_messages[1:]
                if len(m) <= 1 or m[1] not in ("win", "tie")
            ]
            if len(filtered) > 1:
                await original_hbm(filtered)
            return

        if has_win:
            _processed_wins.add(battle_tag)
            # Prevent unbounded growth: clear when set gets too large
            if len(_processed_wins) > 10000:
                _processed_wins.clear()

        await original_hbm(split_messages)

    # Apply wrapped version
    dc._handle_battle_message = _safe_handle_battle_message
    dc._update_challenges = player._update_challenges
    dc._handle_challenge_request = player._handle_challenge_request


async def shutdown_worker():
    """Shut down the global worker process."""
    global _global_worker
    if _global_worker is not None:
        await _global_worker.shutdown()
        _global_worker = None


def _atexit_cleanup():
    """Synchronously kill the worker process on interpreter exit."""
    global _global_worker
    if _global_worker is not None and _global_worker._proc is not None:
        try:
            _global_worker._proc.kill()
            _global_worker._proc.wait(timeout=3)
        except Exception:
            pass
        _global_worker = None


import atexit
atexit.register(_atexit_cleanup)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    # Suppress excessive logging during tests
    logging.basicConfig(level=logging.WARNING)

    # Import teams for non-random formats
    try:
        from teams_ou import TEAM_A, TEAM_B, TEAM_C, TEAM_D
        HAS_TEAMS = True
    except ImportError:
        HAS_TEAMS = False

    async def test_1_random_battles():
        """Test 1: Two random players play 5 random battles."""
        print("\n=== Test 1: 5 random battles ===")
        p1 = RandomDirectPlayer(
            account_configuration=AccountConfiguration("Alice", None),
            battle_format="gen9randombattle",
            max_concurrent_battles=1,
        )
        p2 = RandomDirectPlayer(
            account_configuration=AccountConfiguration("Bob", None),
            battle_format="gen9randombattle",
            max_concurrent_battles=1,
        )

        t0 = perf_counter()
        results = await direct_battle_against(p1, p2, n_battles=5)
        elapsed = perf_counter() - t0

        print(f"  Results: {results}")
        print(f"  Time: {elapsed:.2f}s ({5 / elapsed:.1f} games/sec)")

        # Verify battle objects have meaningful state
        ok = True
        for tag, battle in p1.battles.items():
            if not battle.finished:
                print(f"  WARN: Battle {tag} not finished")
                ok = False
            if not battle.team:
                print(f"  WARN: Battle {tag} has no team")
                ok = False

        total_finished = results["p1_wins"] + results["p2_wins"] + results["ties"]
        if total_finished == 5 and results["errors"] == 0:
            print("  PASS")
        else:
            print(f"  FAIL: expected 5 finished battles, got {total_finished} ({results['errors']} errors)")
            ok = False
        return ok

    async def test_2_speed_benchmark():
        """Test 2: 50 sequential random battles for speed benchmark."""
        print("\n=== Test 2: 50 battles speed benchmark ===")
        p1 = RandomDirectPlayer(
            account_configuration=AccountConfiguration("Speed1", None),
            battle_format="gen9randombattle",
            max_concurrent_battles=1,
        )
        p2 = RandomDirectPlayer(
            account_configuration=AccountConfiguration("Speed2", None),
            battle_format="gen9randombattle",
            max_concurrent_battles=1,
        )

        t0 = perf_counter()
        results = await direct_battle_against(p1, p2, n_battles=50)
        elapsed = perf_counter() - t0

        total = results["p1_wins"] + results["p2_wins"] + results["ties"]
        print(f"  Results: {results}")
        print(f"  Time: {elapsed:.2f}s ({total / elapsed:.1f} games/sec)")
        print(f"  P1 win rate: {results['p1_wins'] / max(total, 1) * 100:.1f}%")

        ok = total >= 45 and results["errors"] <= 5
        if ok:
            print("  PASS")
        else:
            print(f"  FAIL: only {total}/50 completed, {results['errors']} errors")
        return ok

    async def test_3_bc_policy():
        """Test 3: Import bc_policy_player and play vs a random bot."""
        print("\n=== Test 3: BC policy vs random (if checkpoint exists) ===")

        # Check if bc_policy_player and a checkpoint exist
        try:
            # This is a basic structural test — we try to import but don't
            # actually run if no checkpoint is available
            from bc_policy_player import BCPolicyPlayer
            checkpoint_dir = Path(__file__).parent.parent / "data" / "models" / "bc"
            if not checkpoint_dir.exists():
                print("  SKIP: No BC checkpoint directory found")
                return True
            # Find the best.pt in the latest v5 model
            best_pts = list(checkpoint_dir.glob("*/best.pt"))
            if not best_pts:
                print("  SKIP: No best.pt checkpoint found")
                return True

            # We'd need to create a DirectPlayer-based BC player here
            # For now, just verify the import works
            print("  bc_policy_player imported successfully")
            print("  SKIP: Full BC test requires custom DirectPlayer integration")
            return True
        except ImportError as e:
            print(f"  SKIP: Could not import bc_policy_player: {e}")
            return True

    async def test_4_stress_test():
        """Test 4: Stress test - 100 battles with worker restart every 20."""
        print("\n=== Test 4: 100 battles stress test ===")
        p1 = RandomDirectPlayer(
            account_configuration=AccountConfiguration("Stress1", None),
            battle_format="gen9randombattle",
            max_concurrent_battles=1,
        )
        p2 = RandomDirectPlayer(
            account_configuration=AccountConfiguration("Stress2", None),
            battle_format="gen9randombattle",
            max_concurrent_battles=1,
        )

        t0 = perf_counter()
        results = await direct_battle_against(p1, p2, n_battles=100)
        elapsed = perf_counter() - t0

        total = results["p1_wins"] + results["p2_wins"] + results["ties"]
        print(f"  Results: {results}")
        print(f"  Time: {elapsed:.2f}s ({total / elapsed:.1f} games/sec)")
        print(f"  Worker restarts: {100 // WORKER_RESTART_INTERVAL}")

        # Verify battle objects
        finished_battles = [b for b in p1.battles.values() if b.finished]
        battles_with_team = [b for b in finished_battles if b.team]
        battles_with_moves = [
            b for b in finished_battles
            if any(
                len(mon.moves) > 0
                for mon in b.team.values()
            )
        ]
        print(f"  Battles with team data: {len(battles_with_team)}/{len(finished_battles)}")
        print(f"  Battles with move data: {len(battles_with_moves)}/{len(finished_battles)}")

        ok = total >= 90 and results["errors"] <= 10
        if ok:
            print("  PASS")
        else:
            print(f"  FAIL: only {total}/100 completed, {results['errors']} errors")
        return ok

    async def test_5_ou_format():
        """Test 5: OU format battles with real teams (if available)."""
        if not HAS_TEAMS:
            print("\n=== Test 5: OU format (SKIP - no teams) ===")
            return True

        print("\n=== Test 5: 5 OU format battles with real teams ===")
        p1 = RandomDirectPlayer(
            account_configuration=AccountConfiguration("OUPlayer1", None),
            battle_format="gen9ou",
            max_concurrent_battles=1,
            team=TEAM_A,
        )
        p2 = RandomDirectPlayer(
            account_configuration=AccountConfiguration("OUPlayer2", None),
            battle_format="gen9ou",
            max_concurrent_battles=1,
            team=TEAM_B,
        )

        t0 = perf_counter()
        results = await direct_battle_against(p1, p2, n_battles=5, battle_format="gen9ou")
        elapsed = perf_counter() - t0

        total = results["p1_wins"] + results["p2_wins"] + results["ties"]
        print(f"  Results: {results}")
        print(f"  Time: {elapsed:.2f}s")

        # Verify OU-specific battle state
        for tag, battle in p1.battles.items():
            if battle.finished and battle.team:
                team_species = [mon.species for mon in battle.team.values()]
                print(f"  Battle {tag}: team = {team_species[:3]}...")
                break

        ok = total == 5 and results["errors"] == 0
        if ok:
            print("  PASS")
        else:
            print(f"  FAIL: {total}/5 completed, {results['errors']} errors")
        return ok

    async def run_all_tests():
        """Run all tests sequentially."""
        print("=" * 60)
        print("direct_player.py self-test")
        print("=" * 60)

        all_pass = True
        for test_fn in [test_1_random_battles, test_2_speed_benchmark,
                        test_3_bc_policy, test_4_stress_test, test_5_ou_format]:
            try:
                ok = await test_fn()
                if not ok:
                    all_pass = False
            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()
                all_pass = False

        # Cleanup
        await shutdown_worker()

        print("\n" + "=" * 60)
        if all_pass:
            print("ALL TESTS PASSED")
        else:
            print("SOME TESTS FAILED")
        print("=" * 60)

    # Run on the POKE_LOOP
    future = asyncio.run_coroutine_threadsafe(run_all_tests(), POKE_LOOP)
    try:
        future.result(timeout=600)
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
