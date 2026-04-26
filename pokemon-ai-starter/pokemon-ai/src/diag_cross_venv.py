"""diag_cross_venv.py — pinpoint the cross-venv challenge disconnect.

Runs in main venv. Sends ONE challenge from a poke-env 0.10 client to a
foul_play_venv subprocess (FoulPlayBot, already running in
accept_challenges mode with QueueTeambuilder). Captures every poke-env
DEBUG line so we can see:

  - the exact bytes sent on /challenge
  - the exact bytes received back from server (battle init etc.)
  - any traceback in poke-env's _handle_message that kills the listener

Pair with battle_server.js BS_TRACE_USER=<sender id> for full
sender-side trace.

Usage (foulplay subprocess + battle server already up):
    python diag_cross_venv.py --opponent FoulPlayBot --port 9000

Output: full DEBUG log to stdout. Pipe to tee.
"""
import argparse
import asyncio
import logging
import sys
import time
import traceback

# Engage every poke-env logger at DEBUG. basicConfig must come BEFORE the
# poke_env import or the loggers it creates inherit a higher level.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s][%(levelname)s] %(message)s",
    stream=sys.stdout,
    force=True,
)
# Suppress websockets DEBUG which is mostly bytes-on-wire noise — we want
# the protocol-layer events from poke_env.player and poke_env.ps_client.
logging.getLogger("websockets.client").setLevel(logging.INFO)
logging.getLogger("websockets.protocol").setLevel(logging.INFO)

from poke_env.player import Player
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from teams_ou import random_pool_teambuilder


class _Sender(Player):
    """Smallest possible Player — random moves, no model, no batcher.

    If the disconnect happens with this minimal Player, the bug is purely
    at the poke-env protocol layer; not in our V9RLPlayer or BattleAgent code.
    """

    async def _on_battle_message(self, *args, **kwargs):
        # Log entry to the protected message handler. If poke-env's parser
        # raises here, the listening task dies and the websocket closes.
        return await super()._on_battle_message(*args, **kwargs)

    def choose_move(self, battle):
        print(f"[sender] choose_move FIRED — battle={battle.battle_tag} turn={battle.turn}",
              flush=True)
        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        print(f"[sender] battle finished — won={battle.won} lost={battle.lost}",
              flush=True)
        super()._battle_finished_callback(battle)


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--opponent", default="FoulPlayBot")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--n-games", type=int, default=1)
    p.add_argument("--account", default="DiagSender")
    p.add_argument("--timeout-s", type=int, default=120)
    args = p.parse_args()

    server = ServerConfiguration(
        f"ws://127.0.0.1:{args.port}/showdown/websocket",
        f"http://127.0.0.1:{args.port}/action.php?",
    )
    sender = _Sender(
        battle_format="gen9ou",
        team=random_pool_teambuilder(),
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration(args.account, None),
        server_configuration=server,
    )
    print(f"[diag] sender={args.account} -> {args.opponent} port={args.port}",
          flush=True)

    t0 = time.time()
    try:
        await asyncio.wait_for(
            sender.send_challenges(args.opponent, n_challenges=args.n_games),
            timeout=args.timeout_s,
        )
        print(f"[diag] OK — battles {sender.n_finished_battles} in {time.time()-t0:.1f}s",
              flush=True)
    except asyncio.TimeoutError:
        print(f"[diag] TIMEOUT after {args.timeout_s}s "
              f"(finished={sender.n_finished_battles})", flush=True)
    except Exception as e:
        print(f"[diag] EXCEPTION after {time.time()-t0:.1f}s: "
              f"{type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
