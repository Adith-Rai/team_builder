"""smoke_pokeengine.py — verify PokeEnginePlayer plays a battle end-to-end.

Run from src/ with at least one battle server (default 9000) up:

    python smoke_pokeengine.py --n-games 3 --search-time-ms 100

Pits PokeEnginePlayer (Foul Play MCTS via poke_engine) against a heuristic
opponent. Reports wins/losses and the per-game outcomes. If any battle hangs
we want to know — that's the failure mode the subprocess approach hit.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time

from poke_env.player import RandomPlayer, MaxBasePowerPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from pokeengine_player import PokeEnginePlayer
from teams_ou import random_pool_teambuilder


def _make_server(port: int) -> ServerConfiguration:
    ws = f"ws://127.0.0.1:{port}/showdown/websocket"
    http = f"http://127.0.0.1:{port}/action.php?"
    return ServerConfiguration(ws, http)


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--n-games", type=int, default=3)
    p.add_argument("--search-time-ms", type=int, default=100,
                   help="poke-engine MCTS time per move (default 100ms — fast smoke)")
    p.add_argument("--opponent", choices=["random", "maxbp"], default="maxbp")
    p.add_argument("--timeout-s", type=int, default=240,
                   help="hard timeout for the whole batch")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    server = _make_server(args.port)
    tb1 = random_pool_teambuilder()
    tb2 = random_pool_teambuilder()

    pe_player = PokeEnginePlayer(
        search_time_ms=args.search_time_ms,
        battle_format="gen9ou",
        team=tb1,
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration("PEsmoke1", None),
        server_configuration=server,
    )

    if args.opponent == "random":
        opp = RandomPlayer(
            battle_format="gen9ou",
            team=tb2,
            max_concurrent_battles=1,
            account_configuration=AccountConfiguration("PEsmokeOpp", None),
            server_configuration=server,
        )
    else:
        opp = MaxBasePowerPlayer(
            battle_format="gen9ou",
            team=tb2,
            max_concurrent_battles=1,
            account_configuration=AccountConfiguration("PEsmokeOpp", None),
            server_configuration=server,
        )

    print(f"[smoke] {args.n_games} games of PokeEnginePlayer "
          f"(MCTS {args.search_time_ms}ms) vs {args.opponent}")
    t0 = time.time()
    try:
        await asyncio.wait_for(
            pe_player.battle_against(opp, n_battles=args.n_games),
            timeout=args.timeout_s,
        )
    except asyncio.TimeoutError:
        print(f"[smoke] TIMEOUT after {args.timeout_s}s")
        return 2

    elapsed = time.time() - t0
    won = pe_player.n_won_battles
    lost = pe_player.n_lost_battles
    tied = pe_player.n_tied_battles
    finished = won + lost + tied
    print(f"[smoke] finished {finished}/{args.n_games} "
          f"({won}W/{lost}L/{tied}T) in {elapsed:.1f}s")
    if finished == 0:
        print("[smoke] NO BATTLES COMPLETED — adapter is failing to dispatch moves")
        return 1
    if finished < args.n_games:
        print(f"[smoke] WARNING: {args.n_games - finished} games did not complete")
        return 1
    print("[smoke] OK — all games played through")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    raise SystemExit(exit_code)
