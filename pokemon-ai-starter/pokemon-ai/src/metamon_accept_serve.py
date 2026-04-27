"""metamon_accept_serve.py — long-running Metamon subprocess in accept_challenges mode.

Runs inside `metamon_venv` (NOT the main project venv — Metamon's deps are
incompatible with our torch/poke-env). Loads a pretrained Metamon agent and
sits accepting challenges from any user against `--port` for `--num-battles`
battles.

Used by external_adapters.py + external_opponent_manager.py to spawn one
subprocess per Metamon variant in the PFSP pool. Our V9RLPlayer challenges
each subprocess's Showdown username via send_challenges (Phase 2 hybrid
adapter path — see docs/EXTERNAL_OPPONENTS_PHASE2.md).

Why this is needed (vs metamon's own serve_model.py): their ladder mode
(`QueueOnLocalLadder`) couples to Showdown's matchmaker, which doesn't give
explicit pairing. PFSP needs explicit pairing — sender targets a specific
username, that bot accepts. So we add an `AcceptChallengesOnLocal` wrapper
sibling to `QueueOnLocalLadder`.

Setup (one-time):
    python -m venv metamon_venv
    metamon_venv/Scripts/pip install -e metamon_ref/   # pulls torch+cpu by default
    # Reinstall torch with CUDA — metamon's transformer is CPU-bound otherwise
    # (~5-15 min per battle vs seconds on GPU):
    metamon_venv/Scripts/pip install --index-url https://download.pytorch.org/whl/cu121 \\
                                     --force-reinstall torch
    # The `torch>=2.6` requirement in metamon's pyproject is too strict for what's
    # actually on PyPI's CUDA index — torch 2.5.1+cu121 works fine in practice.

Example invocation (from main project — external_opponent_manager.py runs this):

    metamon_venv/Scripts/python.exe metamon_accept_serve.py \\
        --model Minikazam --username MM-Minikazam \\
        --server-port 9000 --num-battles 10000 \\
        --format gen9ou --team-set competitive --temperature 1.0
"""
from __future__ import annotations

import argparse
import functools
import logging
import os
import random
import warnings
from pathlib import Path

# Bypass Metamon's strict poke-env version pin (installed 0.8.3.3 vs metamon's
# 0.8.3.2 expectation; the diff is a couple of unrelated commits).
os.environ.setdefault("METAMON_ALLOW_ANY_POKE_ENV", "True")

import amago  # noqa: F401  (force-import for the gin registry)
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.concurrency import POKE_LOOP

from metamon.env.wrappers import (
    PokeEnvWrapper,
    QueueOnLocalLadder,
    get_metamon_teams,
)
from metamon.interface import ObservationSpace, RewardFunction, ActionSpace
from metamon.rl.pretrained import get_pretrained_model
from metamon.rl.metamon_to_amago import PSLadderAMAGOWrapper

warnings.filterwarnings("ignore")


def _local_server(port: int) -> ServerConfiguration:
    return ServerConfiguration(
        f"ws://127.0.0.1:{port}/showdown/websocket",
        f"http://127.0.0.1:{port}/action.php?",
    )


class AcceptChallengesOnLocal(QueueOnLocalLadder):
    """Accept incoming challenges from any user on a local Showdown server.

    Subclasses `QueueOnLocalLadder` so it passes `PSLadderAMAGOWrapper`'s
    isinstance check (the wrapper has nothing ladder-specific in its body —
    just an obs-mask and an auto-reset guard, both useful here too).

    Differences from the parent:
    - We override `handle_ladder_start` (the explicit hook QueueOnLocalLadder
      provides for this) to schedule an accept_challenges loop on POKE_LOOP
      instead of `start_laddering`.
    - We accept a `server_port` so the metamon subprocess can target our
      battle server, rather than poke-env's default LocalhostServerConfiguration.
    """

    # poke-env's `OpenAIGymEnv.reset` polls `agent.current_battle` up to
    # `_INIT_RETRIES * _TIME_BETWEEN_RETRIES` seconds (default 100*0.5=50s)
    # before raising `RuntimeError("Agent is not challenging")`. amago's
    # `evaluate_test` calls reset() right after every battle ends, expecting
    # the next battle to already be in flight. In our PPO loop, PFSP can
    # leave Metamon idle for several minutes between waves while the
    # trainer plays self-play / Foul Play. Bump to ~1 hour total wait so MM
    # sits patiently across PFSP gaps. Inherited via MRO; openai_api uses
    # `self._INIT_RETRIES`, so our override wins.
    _INIT_RETRIES = 7200          # × _TIME_BETWEEN_RETRIES = 60 minutes
    _TIME_BETWEEN_RETRIES = 0.5

    def __init__(
        self,
        battle_format: str,
        num_battles: int,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        reward_function: RewardFunction,
        player_team_set,
        player_username: str,
        server_port: int,
        opponent_username: str | None = None,
        save_trajectories_to: str | None = None,
        save_team_results_to: str | None = None,
        battle_backend: str = "metamon",
        team_preview_model=None,
    ):
        # Stash before super().__init__ — server_configuration is read during base init.
        # Note: PokeEnvWrapper.__init__ unconditionally sets self._accept_opponent_filter
        # to a random "MM-XXXXXXXXXX" (used as the Showdown account name for an
        # in-process opponent_type Player, which we don't have). We use a distinct
        # attribute name to avoid that overwrite — this filter is what the
        # accept_challenges loop matches against (None = accept anyone).
        self._server_configuration = _local_server(server_port)
        self._accept_opponent_filter = opponent_username
        super().__init__(
            battle_format=battle_format,
            num_battles=num_battles,
            observation_space=observation_space,
            action_space=action_space,
            reward_function=reward_function,
            player_team_set=player_team_set,
            player_username=player_username,
            save_trajectories_to=save_trajectories_to,
            save_team_results_to=save_team_results_to,
            battle_backend=battle_backend,
            team_preview_model=team_preview_model,
            print_battle_bar=False,
        )

    @property
    def server_configuration(self):
        # Override default LocalhostServerConfiguration so we can pick a custom port
        return self._server_configuration

    def start_laddering(self, *args, **kwargs):
        # Belt-and-suspenders: parent QueueOnLocalLadder calls handle_ladder_start
        # which we override; but if anything calls start_laddering directly
        # (some upstream code does in error paths), redirect to accept-loop too.
        return self.handle_ladder_start(*args, **kwargs)

    def handle_ladder_start(self, n_challenges: int):
        """Hook from QueueOnLocalLadder — replace the laddering loop with
        an accept_challenges loop, scheduled on POKE_LOOP just like
        start_laddering does."""
        import asyncio
        import time

        if self._challenge_task and not self._challenge_task.done():
            count = self._SWITCH_CHALLENGE_TASK_RETRIES
            while not self._challenge_task.done():
                if count == 0:
                    raise RuntimeError("Agent is already challenging")
                count -= 1
                time.sleep(self._TIME_BETWEEN_SWITCH_RETIRES)
        if not n_challenges:
            self._keep_challenging = True
        self._challenge_task = asyncio.run_coroutine_threadsafe(
            self._accept_loop(n_challenges), POKE_LOOP
        )

    async def _accept_loop(self, n_battles: int):
        # The future returned by run_coroutine_threadsafe is never awaited, so
        # any exception in this loop is silently swallowed. Log explicitly.
        import traceback
        try:
            if n_battles and n_battles > 0:
                for i in range(n_battles):
                    print(f"[metamon-accept] iter {i+1}/{n_battles} — awaiting challenge", flush=True)
                    await self.agent.accept_challenges(self._accept_opponent_filter, 1)
                    print(f"[metamon-accept] iter {i+1}/{n_battles} — battle ended", flush=True)
            else:
                while self._keep_challenging:
                    await self.agent.accept_challenges(self._accept_opponent_filter, 1)
        except Exception as e:
            print(f"[metamon-accept] FATAL in accept_loop: {e}", flush=True)
            traceback.print_exc()
            raise


def make_accept_env(
    battle_format: str,
    num_battles: int,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    reward_function: RewardFunction,
    player_team_set,
    player_username: str,
    server_port: int,
    opponent_username: str | None = None,
    save_trajectories_to: str | None = None,
    battle_backend: str = "metamon",
):
    print(f"[metamon] make_accept_env: building AcceptChallengesOnLocal for {player_username} on port {server_port}", flush=True)
    menv = AcceptChallengesOnLocal(
        battle_format=battle_format,
        num_battles=num_battles,
        observation_space=observation_space,
        action_space=action_space,
        reward_function=reward_function,
        player_team_set=player_team_set,
        player_username=player_username,
        server_port=server_port,
        opponent_username=opponent_username,
        save_trajectories_to=save_trajectories_to,
        battle_backend=battle_backend,
    )
    # PSLadderAMAGOWrapper is the right wrapper here (despite the name):
    # it just adds an illegal-action mask to the obs and handles a quirk
    # with parallel-actor auto-resets. Nothing ladder-specific in it for our purposes.
    return PSLadderAMAGOWrapper(menv)


def main():
    # Enable poke-env DEBUG so we see challenge receipt + accept dispatch in the log
    if os.environ.get("METAMON_DEBUG_POKEENV"):
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(name)s][%(levelname)s] %(message)s",
        )

    p = argparse.ArgumentParser(description="Metamon accept-challenges subprocess")
    p.add_argument("--model", required=True, help="Pretrained model name (e.g. Minikazam, SmallRL)")
    p.add_argument("--username", required=True, help="Showdown username for this Metamon agent")
    p.add_argument("--server-port", type=int, default=9000, help="Local Showdown server port")
    p.add_argument("--format", default="gen9ou")
    p.add_argument("--num-battles", type=int, default=10000,
                   help="Total battles to accept before exit (set high for long-running)")
    p.add_argument("--team-set", default="competitive",
                   help="Metamon team set name (competitive / modern_replays / etc.). "
                        "Ignored if --team-queue is given.")
    p.add_argument("--team-queue", default=None,
                   help="Path to a coordinator-controlled team queue dir. When set, "
                        "we use QueueTeambuilder to pop one packed team per battle "
                        "from this dir, instead of metamon's own static team set. "
                        "Coordinator (in main project venv) must call enqueue_team() "
                        "before each challenge. This is how the coordinator hands "
                        "Metamon procedural Smogon teams matching what V9RLPlayer is using.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature for MetamonDiscrete")
    p.add_argument("--checkpoint", type=int, default=None,
                   help="Override default checkpoint epoch")
    p.add_argument("--opponent-username", default=None,
                   help="If set, only accept challenges from this user (default: anyone)")
    p.add_argument("--queue-wait-timeout-s", type=float, default=3600.0,
                   help="Seconds QueueTeambuilder waits for a team before crashing "
                        "(default 1 hour, used only when --team-queue is set).")
    p.add_argument("--clean-on-init", default="true",
                   help="Whether QueueTeambuilder wipes stale .team files on startup. "
                        "Default true. ExternalOpponentManager overrides to false on "
                        "respawn so teams the trainer enqueued during a mid-iter crash "
                        "survive the restart.")
    args = p.parse_args()

    if "METAMON_CACHE_DIR" not in os.environ:
        raise SystemExit("METAMON_CACHE_DIR must be set (used for HF model downloads + teams)")

    print(f"[metamon] model={args.model} user={args.username} "
          f"port={args.server_port} format={args.format} temp={args.temperature}",
          flush=True)

    agent_maker = get_pretrained_model(args.model)
    if args.team_queue:
        # Pop one team from a coordinator-managed queue per battle. Lets the
        # main process hand us its own procedural Smogon teams so both sides
        # play matched-source teams without us shipping the procedural builder
        # into metamon_venv.
        # team_generator.py is plain stdlib (the procedural code only needs
        # poke-env's Teambuilder base class which exists in this fork too),
        # so we can import it from the main src/ via path injection.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from team_generator import QueueTeambuilder
        print(f"[metamon] team source: queue dir {args.team_queue}", flush=True)
        # `clean_on_init=False` is set by ExternalOpponentManager on respawn so
        # the trainer's already-enqueued teams from the crash window survive —
        # without this, the restarted subprocess wipes them and sits idle until
        # the trainer's per-opponent wait_for fires ~5 min later.
        clean_on_init = str(args.clean_on_init).strip().lower() in ("true", "1", "yes")
        print(f"[metamon] queue clean_on_init={clean_on_init}", flush=True)
        # Big timeout — subprocess should sit waiting between PFSP waves, not crash.
        team_set = QueueTeambuilder(args.team_queue,
                                    wait_timeout_s=float(args.queue_wait_timeout_s),
                                    clean_on_init=clean_on_init)
    else:
        print(f"[metamon] team source: metamon's '{args.team_set}' set", flush=True)
        team_set = get_metamon_teams(args.format, args.team_set)
    agent = agent_maker.initialize_agent(
        checkpoint=args.checkpoint, log=False, action_temperature=args.temperature
    )
    agent.env_mode = "sync"
    agent.verbose = False
    agent.parallel_actors = 1

    make_envs = [
        functools.partial(
            make_accept_env,
            battle_format=args.format,
            num_battles=args.num_battles,
            observation_space=agent_maker.observation_space,
            action_space=agent_maker.action_space,
            reward_function=agent_maker.reward_function,
            player_team_set=team_set,
            player_username=args.username,
            server_port=args.server_port,
            opponent_username=args.opponent_username,
            battle_backend=agent_maker.battle_backend,
        )
    ]

    print(f"[metamon] starting evaluate_test for {args.num_battles} battles", flush=True)
    results = agent.evaluate_test(
        make_envs,
        timesteps=args.num_battles * 350,
        episodes=args.num_battles,
    )
    print(f"[metamon] done. results: {results}", flush=True)


if __name__ == "__main__":
    main()
