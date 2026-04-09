import asyncio
import os
import random
from pathlib import Path
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder
from policy_random import RandomPolicy
from teams_ou import TEAM_POOL, random_teambuilder

REPLAY_DIR = Path(os.environ.get("REPLAY_DIR", "../data/replays"))
REPLAY_DIR.mkdir(parents=True, exist_ok=True)

# Showdown server connection — configurable via environment variables.
# Defaults to localhost:8000 for local dev; set SHOWDOWN_HOST for Docker (e.g. "showdown").
_SD_HOST = os.environ.get("SHOWDOWN_HOST", "127.0.0.1")
_SD_PORT = os.environ.get("SHOWDOWN_PORT", "8000")
SERVER = ServerConfiguration(
    f"ws://{_SD_HOST}:{_SD_PORT}/showdown/websocket",
    f"http://{_SD_HOST}:{_SD_PORT}/action.php?"
)

# Back-compat for code that still uses TEAM1/TEAM2:
TEAM1 = TEAM_POOL[0]
TEAM2 = TEAM_POOL[1]

TB1 = ConstantTeambuilder(TEAM1)
TB2 = ConstantTeambuilder(TEAM2)

class SelfPlaySession:
    def __init__(self, battle_format="gen9ou"):
        self.format = battle_format

    async def play(self, n_battles:int=10):
        p1 = RandomPolicy(
            battle_format=self.format,
            server_configuration=SERVER,
            max_concurrent_battles=10,
            team=random_teambuilder(),
            #save_replays=True,
            #replay_folder=str(REPLAY_DIR),
        )
        p2 = RandomPolicy(
            battle_format=self.format,
            server_configuration=SERVER,
            max_concurrent_battles=10,
            team=random_teambuilder(),
            #save_replays=True,
            #replay_folder=str(REPLAY_DIR),
        )
        await p1.battle_against(p2, n_battles=n_battles)
        return p1, p2
