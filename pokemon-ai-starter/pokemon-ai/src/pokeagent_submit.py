#!/usr/bin/env python
"""Submit our model to the PokeAgent Challenge ladder.

Usage:
  # First: create an account at battling.pokeagentchallenge.com
  # Username should start with "PAC"

  python pokeagent_submit.py \
      --checkpoint data/models/rl_v9/selfplay_v9_20260413_061236/snapshot_2979.pt \
      --username PAC_YourTeamName \
      --password YourPassword \
      --team TEAM_AU \
      --n-games 50 \
      --device cuda

This connects to the PokeAgent Showdown server, plays rated ladder games
in Gen9 OU, and your rating updates automatically on their leaderboard.
"""
import argparse
import asyncio
import sys
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from battle_agent import BattleAgent
from teams_ou import get_team, list_teams, TEAMS, random_pool_teambuilder


# PokeAgent Challenge server
POKEAGENT_SERVER = ServerConfiguration(
    "wss://pokeagentshowdown.com/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)


def main():
    parser = argparse.ArgumentParser(description="Submit to PokeAgent Challenge")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--username", required=True, help="PokeAgent account username (start with PAC)")
    parser.add_argument("--password", required=True, help="PokeAgent account password")
    parser.add_argument("--team", default="TEAM_AU",
                        help=f"Team to use (default: TEAM_AU, best from selection). "
                             f"Use 'random' for random from top 10. Available: {', '.join(list_teams()[:5])}...")
    parser.add_argument("--n-games", type=int, default=50,
                        help="Number of ladder games to play (default: 50)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--format", default="gen9ou", help="Battle format (default: gen9ou)")
    args = parser.parse_args()

    # Team setup
    if args.team == "random":
        # Use top 10 teams from selection results
        top_teams = ["TEAM_AU", "TEAM_T", "TEAM_G", "TEAM_B", "TEAM_C",
                      "TEAM_O", "TEAM_AZ", "TEAM_BK", "TEAM_BJ", "TEAM_BE"]
        import random
        team_str = get_team(random.choice(top_teams))
        teambuilder = ConstantTeambuilder(team_str)
        print(f"Using random top-10 team")
    elif args.team.startswith("TEAM_"):
        team_str = get_team(args.team)
        teambuilder = ConstantTeambuilder(team_str)
        print(f"Using {args.team}")
    else:
        print(f"Unknown team: {args.team}. Use a TEAM_XX name or 'random'.")
        sys.exit(1)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Username: {args.username}")
    print(f"Format: {args.format}")
    print(f"Games: {args.n_games}")
    print(f"Server: {POKEAGENT_SERVER.websocket_url}")
    print()

    # Pre-load checkpoint for efficiency
    cached_ckpt = torch.load(args.checkpoint, map_location=torch.device(args.device),
                              weights_only=False)

    async def run():
        player = BattleAgent(
            args.checkpoint,
            device=args.device,
            _cached_ckpt=cached_ckpt,
            account_configuration=AccountConfiguration(args.username, args.password),
            battle_format=args.format,
            server_configuration=POKEAGENT_SERVER,
            team=teambuilder,
            max_concurrent_battles=1,  # ladder plays one at a time
        )

        print(f"Connected. Starting {args.n_games} ladder games...\n", flush=True)

        # Play ladder games
        for i in range(args.n_games):
            try:
                await player.ladder(1)
                w, l, t = player.n_won_battles, player.n_lost_battles, player.n_tied_battles
                print(f"  Game {i+1}/{args.n_games}: W={w} L={l} T={t} "
                      f"(WR={w/max(1,w+l+t)*100:.0f}%)", flush=True)
            except Exception as e:
                print(f"  Game {i+1}/{args.n_games}: ERROR - {e}", flush=True)

        total = player.n_won_battles + player.n_lost_battles + player.n_tied_battles
        print(f"\nDone! {player.n_won_battles}/{total} wins "
              f"({player.n_won_battles/max(1,total)*100:.0f}%)")
        print(f"Check your rating at: battling.pokeagentchallenge.com/ladder")

    asyncio.run(run())


if __name__ == "__main__":
    main()
