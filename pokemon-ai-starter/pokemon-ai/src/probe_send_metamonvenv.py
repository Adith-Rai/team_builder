"""Same-venv challenge sender — has a real team for gen9ou."""
import asyncio
import os
import random
import string
import sys

os.environ.setdefault("METAMON_ALLOW_ANY_POKE_ENV", "True")

from poke_env.player import Player
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.teambuilder.teambuilder import Teambuilder


PACKED_GEN9OU = (
    "Dragapult||choicespecs|infiltrator|shadowball,dracometeor,uturn,thunderbolt|"
    "Timid|0,0,0,252,4,252||,0,,,,||50|]"
    "Garchomp||rockyhelmet|roughskin|stealthrock,earthquake,fire-fang,dragontail|"
    "Jolly|0,252,0,0,4,252|||50|]"
    "Heatran||leftovers|flashfire|magmastorm,earthpower,toxic,protect|"
    "Calm|252,0,0,0,236,20|||50|]"
    "Cinderace||heavydutyboots|libero|pyroball,uturn,courtchange,willowisp|"
    "Jolly|0,252,0,0,4,252|||50|]"
    "GreatTusk||booster-energy|protosynthesis|earthquake,headlongrush,iceshard,closecombat|"
    "Adamant|0,252,0,0,4,252|||50|]"
    "Toxapex||blacksludge|regenerator|surf,toxic,recover,haze|"
    "Bold|252,0,252,0,4,0||,0,,,,||50|"
)


class FixedTeamBuilder(Teambuilder):
    def __init__(self, packed):
        super().__init__()
        self._packed = packed
    def yield_team(self):
        return self._packed


class _Sender(Player):
    def choose_move(self, battle):
        return self.choose_random_move(battle)


async def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "MM-Minikazam"
    suffix = "".join(random.choices(string.ascii_lowercase, k=3))
    server = ServerConfiguration(
        "ws://127.0.0.1:9000/showdown/websocket",
        "http://127.0.0.1:9000/action.php?",
    )
    sender = _Sender(
        battle_format="gen9ou",
        team=FixedTeamBuilder(PACKED_GEN9OU),
        max_concurrent_battles=1,
        account_configuration=AccountConfiguration(f"FvSend{suffix}", None),
        server_configuration=server,
    )
    print(f"[probe-send] sender={sender.username} -> {target}")
    try:
        await asyncio.wait_for(sender.send_challenges(target, 1), timeout=120)
        print(f"[probe-send] OK — battles {sender.n_finished_battles}")
    except asyncio.TimeoutError:
        print("[probe-send] TIMEOUT")


asyncio.run(main())
