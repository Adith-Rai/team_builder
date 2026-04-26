"""Within metamon_venv: send a challenge to a username, see if it lands.

Pair with probe_acceptchallenges.py (also in metamon_venv) to test bare
challenge dispatch end-to-end inside the older fork's protocol layer.
"""
import asyncio
import sys

from poke_env.player import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration


async def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "MM-Probe"
    server = ServerConfiguration(
        "ws://127.0.0.1:9000/showdown/websocket",
        "http://127.0.0.1:9000/action.php?",
    )
    sender = RandomPlayer(
        battle_format="gen9ou",
        account_configuration=AccountConfiguration("ProbeSender", None),
        server_configuration=server,
        max_concurrent_battles=1,
    )
    print(f"[probe-send] challenging {target} from ProbeSender")
    try:
        await asyncio.wait_for(sender.send_challenges(target, 1), timeout=45)
        print(f"[probe-send] OK — battles {sender.n_finished_battles}")
    except asyncio.TimeoutError:
        print("[probe-send] TIMEOUT")


asyncio.run(main())
