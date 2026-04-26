"""Two heuristics fighting on our local server — sanity check for the smoke harness."""
import asyncio, time
from poke_env.player import RandomPlayer, MaxBasePowerPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration
from teams_ou import random_pool_teambuilder

async def main():
    s = ServerConfiguration("ws://127.0.0.1:9000/showdown/websocket",
                            "http://127.0.0.1:9000/action.php?")
    a = MaxBasePowerPlayer(battle_format="gen9ou", team=random_pool_teambuilder(),
                           max_concurrent_battles=1,
                           account_configuration=AccountConfiguration("BLA", None),
                           server_configuration=s)
    b = RandomPlayer(battle_format="gen9ou", team=random_pool_teambuilder(),
                     max_concurrent_battles=1,
                     account_configuration=AccountConfiguration("BLB", None),
                     server_configuration=s)
    t0 = time.time()
    await asyncio.wait_for(a.battle_against(b, n_battles=2), timeout=60)
    print(f"baseline: {a.n_won_battles}W/{a.n_lost_battles}L in {time.time()-t0:.1f}s")

asyncio.run(main())
