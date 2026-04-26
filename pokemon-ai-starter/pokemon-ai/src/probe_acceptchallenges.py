"""Minimal repro: bare poke-env Player (from metamon_venv's fork) accepts challenges.

If this works but metamon_accept_serve.py doesn't, the bug is amago + PokeEnvWrapper
interaction. If this also hangs, the bug is at the poke-env layer (port mismatch,
username conflict, etc.).
"""
import asyncio
import logging
import sys

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(name)s][%(levelname)s] %(message)s")

from poke_env.player import Player
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration


class _SimplePlayer(Player):
    def choose_move(self, battle):
        return self.choose_random_move(battle)


async def main():
    server = ServerConfiguration(
        "ws://127.0.0.1:9000/showdown/websocket",
        "http://127.0.0.1:9000/action.php?",
    )
    p = _SimplePlayer(
        battle_format="gen9ou",
        account_configuration=AccountConfiguration("MM-Probe", None),
        server_configuration=server,
        max_concurrent_battles=1,
    )
    print("[probe] player created, accepting up to 1 challenge for 60s")
    try:
        await asyncio.wait_for(p.accept_challenges(None, 1), timeout=60)
        print(f"[probe] OK — accepted, battle records: {p.n_finished_battles}")
    except asyncio.TimeoutError:
        print("[probe] TIMEOUT — no challenge accepted")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
