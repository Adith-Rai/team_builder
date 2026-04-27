"""foul_play_accept_serve.py — long-running real-Foul-Play subprocess in
accept_challenges mode, using our QueueTeambuilder for matched teams.

Runs inside `foul_play_venv` (not the main project venv — Foul Play's deps
are minimal but we keep it isolated for parity with metamon_venv). Spins up
Foul Play's MCTS strategy via foul_play_ref/, configured to accept any
incoming challenge from our V9RLPlayer.

Why not just `foul_play_ref/run.py --bot-mode=accept_challenge`? Two
reasons:
  1. Foul Play's run.py uses TeamListIterator / load_team to read teams
     from `foul_play_ref/teams/teams/...` in Showdown EXPORT format. Our
     coordinator pushes packed-format teams via QueueTeambuilder, which is
     a different format. Bridging that on Foul Play's side would require
     converting packed→export.
  2. We want to bypass `load_team`'s file-on-disk indirection and pump the
     packed team straight into `ps_websocket_client.update_team()`.

So this script imports Foul Play's modules and re-implements its main loop
with two changes:
  - Pop a packed team from the queue (instead of load_team(--team-name))
  - team_dict=None (Foul Play's strategy mostly degrades gracefully without
    it; the validation it gates on is non-critical for our PFSP use case)

Setup (one-time):
    python -m venv foul_play_venv
    foul_play_venv/Scripts/pip install -r foul_play_ref/requirements.txt
    # poke-engine builds via Rust during this; needs cargo on PATH or the
    # cached cp311 wheel.

Example invocation (external_opponent_manager.py runs this):

    foul_play_venv/Scripts/python.exe foul_play_accept_serve.py \\
        --username FoulPlayBot --server-port 9000 \\
        --num-battles 10000 --search-time-ms 200 \\
        --team-queue data/external_team_queue/foulplay
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import traceback
from copy import deepcopy
from pathlib import Path

# Make foul_play_ref/ importable so we can pull in fp/, teams/, config.py, data/.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_FOUL_PLAY_REF = _PROJECT_ROOT / "foul_play_ref"
if not _FOUL_PLAY_REF.exists():
    raise SystemExit(f"foul_play_ref not found at {_FOUL_PLAY_REF}")
sys.path.insert(0, str(_FOUL_PLAY_REF))
# team_generator.py (in main src/) is plain stdlib + uses poke-env's
# Teambuilder base class only optionally — Foul Play doesn't ship poke-env,
# so QueueTeambuilder will inherit from `object`, which is fine since we
# only ever call .yield_team() on it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from team_generator import QueueTeambuilder  # noqa: E402

from config import FoulPlayConfig, init_logging, BotModes, SaveReplay  # noqa: E402
from fp.run_battle import pokemon_battle  # noqa: E402
from fp.websocket_client import PSWebsocketClient  # noqa: E402
from data import all_move_json, pokedex  # noqa: E402
from data.mods.apply_mods import apply_mods  # noqa: E402

logger = logging.getLogger(__name__)


def _set_foul_play_config(args):
    """Populate FoulPlayConfig directly without going through its argparse."""
    FoulPlayConfig.websocket_uri = f"ws://127.0.0.1:{args.server_port}/showdown/websocket"
    FoulPlayConfig.username = args.username
    FoulPlayConfig.password = args.password
    FoulPlayConfig.avatar = None
    FoulPlayConfig.bot_mode = BotModes.accept_challenge
    FoulPlayConfig.pokemon_format = args.format
    FoulPlayConfig.smogon_stats = None
    FoulPlayConfig.search_time_ms = int(args.search_time_ms)
    FoulPlayConfig.parallelism = int(args.search_parallelism)
    FoulPlayConfig.run_count = int(args.num_battles)
    FoulPlayConfig.team_name = None        # we feed teams via the queue
    FoulPlayConfig.team_list = None
    FoulPlayConfig.user_to_challenge = None
    FoulPlayConfig.save_replay = SaveReplay.never
    FoulPlayConfig.room_name = None
    FoulPlayConfig.log_level = args.log_level
    FoulPlayConfig.log_to_file = False
    # Internal init that .configure() normally does:
    FoulPlayConfig.user_id = None
    FoulPlayConfig.file_log_handler = None


async def run_loop(args):
    _set_foul_play_config(args)
    init_logging(FoulPlayConfig.log_level, FoulPlayConfig.log_to_file)
    apply_mods(FoulPlayConfig.pokemon_format)

    original_pokedex = deepcopy(pokedex)
    original_move_json = deepcopy(all_move_json)

    # 1-hour timeout: between PFSP waves there can be multi-minute idle gaps
    # (PPO update + eval, etc.). The subprocess should sit waiting, not crash.
    # `clean_on_init=False` is set by ExternalOpponentManager on respawn so the
    # trainer's already-enqueued teams from the crash window survive — without
    # this, the restarted subprocess wipes them and sits idle until the
    # trainer's per-opponent wait_for fires ~5 min later.
    clean_on_init = str(args.clean_on_init).strip().lower() in ("true", "1", "yes")
    queue_tb = QueueTeambuilder(
        args.team_queue,
        wait_timeout_s=float(args.queue_wait_timeout_s),
        clean_on_init=clean_on_init,
    )
    print(f"[foulplay] queue clean_on_init={clean_on_init}", flush=True)
    print(f"[foulplay] using team queue: {args.team_queue}", flush=True)

    ws_client = await PSWebsocketClient.create(
        FoulPlayConfig.username, FoulPlayConfig.password, FoulPlayConfig.websocket_uri
    )
    FoulPlayConfig.user_id = await ws_client.login()
    print(f"[foulplay] logged in as {FoulPlayConfig.username}", flush=True)

    wins = losses = 0
    for i in range(args.num_battles):
        print(f"[foulplay] iter {i+1}/{args.num_battles} — waiting for team in queue", flush=True)
        # Pop one packed team from the coordinator's queue. Blocks up to
        # wait_timeout_s. If this raises, the coordinator isn't writing —
        # raise out and let our process manager restart us.
        packed_team = queue_tb.yield_team()
        await ws_client.update_team(packed_team)
        print(f"[foulplay] iter {i+1}/{args.num_battles} — got team, awaiting challenge", flush=True)

        await ws_client.accept_challenge(FoulPlayConfig.pokemon_format, FoulPlayConfig.room_name)
        # team_dict=None — Foul Play's own validation gated on `if team_dict
        # is not None` is the only place this goes; the in-battle MCTS path
        # doesn't actually need team_dict, just the packed team that's
        # already been sent via update_team().
        winner = await pokemon_battle(ws_client, FoulPlayConfig.pokemon_format, None)
        if winner == FoulPlayConfig.username:
            wins += 1
        else:
            losses += 1
        print(f"[foulplay] iter {i+1}/{args.num_battles} — done. W={wins} L={losses}", flush=True)

        if pokedex != original_pokedex or all_move_json != original_move_json:
            print("[foulplay] FATAL: data dictionaries mutated mid-run, aborting", flush=True)
            break

    await ws_client.close()


def main():
    p = argparse.ArgumentParser(description="Foul Play accept-challenges subprocess")
    p.add_argument("--username", required=True, help="Showdown username (e.g. FoulPlayBot)")
    p.add_argument("--password", default=None, help="Optional password (local server: leave unset)")
    p.add_argument("--server-port", type=int, default=9000)
    p.add_argument("--format", default="gen9ou")
    p.add_argument("--num-battles", type=int, default=10000)
    p.add_argument("--search-time-ms", type=int, default=200,
                   help="MCTS time per move. Higher = stronger but slower.")
    p.add_argument("--search-parallelism", type=int, default=1,
                   help="Foul Play's internal multi-MCTS fan-out (uses ProcessPoolExecutor "
                        "across N opponent-set guesses; orthogonal to running multiple "
                        "instances of this whole subprocess for concurrency).")
    p.add_argument("--team-queue", required=True,
                   help="Coordinator-controlled directory we pop packed Showdown teams from "
                        "(one per battle). The main project's rl_collection.py writes one "
                        "procedural Smogon team per upcoming challenge.")
    p.add_argument("--queue-wait-timeout-s", type=float, default=3600.0,
                   help="Seconds to wait for a team file before crashing. Default 1 hour "
                        "to absorb idle gaps between PFSP waves; manager restarts the "
                        "subprocess if this fires.")
    p.add_argument("--clean-on-init", default="true",
                   help="Whether QueueTeambuilder wipes stale .team files on startup. "
                        "Default true. ExternalOpponentManager overrides to false on "
                        "respawn so teams the trainer enqueued during a mid-iter crash "
                        "survive the restart.")
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()

    print(f"[foulplay] user={args.username} port={args.server_port} "
          f"format={args.format} search_time={args.search_time_ms}ms",
          flush=True)
    try:
        asyncio.run(run_loop(args))
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
