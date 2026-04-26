"""Like smoke_metamon.py but without asyncio.wait_for, with full traceback."""
import asyncio
import logging
import time
import traceback

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(name)s][%(levelname)s] %(message)s")

from poke_env.player import MaxBasePowerPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from teams_ou import random_pool_teambuilder


async def main():
    server = ServerConfiguration(
        "ws://127.0.0.1:9000/showdown/websocket",
        "http://127.0.0.1:9000/action.php?",
    )
    p = MaxBasePowerPlayer(
        battle_format="gen9ou",
        team=random_pool_teambuilder(),
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration("MetSmokeUs", None),
        server_configuration=server,
    )
    print("[naked] sending 1 challenge to MM-Minikazam, no timeout")
    t0 = time.time()
    try:
        await p.send_challenges("MM-Minikazam", n_challenges=1)
        print(f"[naked] OK — battles {p.n_finished_battles} in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"[naked] EXCEPTION after {time.time()-t0:.1f}s: {type(e).__name__}: {e}")
        traceback.print_exc()


asyncio.run(main())
