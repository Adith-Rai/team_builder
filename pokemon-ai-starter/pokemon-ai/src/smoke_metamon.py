"""smoke_metamon.py — verify MetamonPlayer subprocess answers challenges.

Sends N challenges from our main venv to a Metamon subprocess (running
metamon_accept_serve.py), confirming the accept-mode wrapper works end-to-end.

Usage (subprocess must already be running on the same port):

    python smoke_metamon.py --opponent MM-Minikazam --port 9000 --n-games 2
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time

from poke_env.player import MaxBasePowerPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from teams_ou import random_pool_teambuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s][%(levelname)s] %(message)s")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--n-games", type=int, default=2)
    p.add_argument("--opponent", default="MM-Minikazam",
                   help="Showdown username of the metamon subprocess")
    p.add_argument("--timeout-s", type=int, default=300)
    args = p.parse_args()

    server = ServerConfiguration(
        f"ws://127.0.0.1:{args.port}/showdown/websocket",
        f"http://127.0.0.1:{args.port}/action.php?",
    )

    challenger = MaxBasePowerPlayer(
        battle_format="gen9ou",
        team=random_pool_teambuilder(),
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration("MetSmokeUs", None),
        server_configuration=server,
    )

    print(f"[smoke] challenging {args.opponent} for {args.n_games} games")
    t0 = time.time()
    try:
        await asyncio.wait_for(
            challenger.send_challenges(args.opponent, n_challenges=args.n_games),
            timeout=args.timeout_s,
        )
    except asyncio.TimeoutError:
        print(f"[smoke] TIMEOUT after {args.timeout_s}s")
        return 2

    elapsed = time.time() - t0
    w, l, t = challenger.n_won_battles, challenger.n_lost_battles, challenger.n_tied_battles
    print(f"[smoke] finished {w + l + t}/{args.n_games} ({w}W/{l}L/{t}T) in {elapsed:.1f}s")
    return 0 if (w + l + t) == args.n_games else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
