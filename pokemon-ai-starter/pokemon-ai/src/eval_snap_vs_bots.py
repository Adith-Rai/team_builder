"""Quick snap-vs-bots eval with replay save.

Plays snapshot vs each requested bot for N games, saves replays per matchup.
Useful for inspecting how bots actually play against the model in battle.

Usage:
  python eval_snap_vs_bots.py --snapshot path/to/snap.pt \
      --bots GreedySEv2,SetupThenSweepv2,SwitchAwareEscapev3,AntiSetupBot,HazardSensev2 \
      --n-games 20 --replay-root /tmp/snap_vs_bots_replays \
      --server ws://127.0.0.1:9000/showdown/websocket
"""

import argparse
import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.ps_client.account_configuration import AccountConfiguration

from battle_agent_transformer import BattleAgentTransformer, is_transformer_checkpoint
from battle_agent import BattleAgent
from teams_ou import random_pool_teambuilder
from eval_elo_ladder import ALL_BOTS, resolve_server
import torch


def snapshot_replays():
    root = Path("replays")
    if not root.exists():
        return set()
    return set(root.iterdir())


def move_new_replays(before, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    after = snapshot_replays()
    for p in after - before:
        try:
            shutil.move(str(p), str(dest_dir / p.name))
            n += 1
        except Exception:
            pass
    return n


async def run_matchup(snap_path, snap_cached, bot_name, n_games, server_cfg, teambuilder, concurrency, match_idx=0):
    """Play snap vs one bot for n_games, save replays."""
    # Unique account names per matchup to avoid showdown username collision
    suffix = f"{os.getpid() % 999}m{match_idx}"
    snap_account = AccountConfiguration(f"Sm{suffix}", None)
    bot_account = AccountConfiguration(f"B{bot_name[:5]}{suffix}", None)

    common = dict(
        battle_format="gen9ou",
        max_concurrent_battles=concurrency,
        server_configuration=server_cfg,
        team=teambuilder,
    )

    # Snap player
    AgentClass = BattleAgentTransformer if is_transformer_checkpoint(snap_cached) else BattleAgent
    snap = AgentClass(
        checkpoint_path=snap_path,
        _cached_ckpt=snap_cached,
        device="cuda",
        save_replays=True,
        account_configuration=snap_account,
        **common,
    )

    bot_cls = ALL_BOTS[bot_name]
    bot = bot_cls(save_replays=False, account_configuration=bot_account, **common)

    before = snapshot_replays()
    t0 = time.time()
    await snap.battle_against(bot, n_battles=n_games)
    elapsed = time.time() - t0

    snap_wins = snap.n_won_battles
    bot_wins = bot.n_won_battles
    ties = snap.n_tied_battles
    total = snap_wins + bot_wins + ties

    dest = Path("/tmp/snap_vs_bots_replays") / bot_name
    n_replays = move_new_replays(before, dest)

    return {
        "bot": bot_name,
        "snap_wins": snap_wins,
        "bot_wins": bot_wins,
        "ties": ties,
        "total": total,
        "snap_wr": snap_wins / max(1, total),
        "elapsed_s": round(elapsed, 1),
        "n_replays": n_replays,
        "replay_dir": str(dest),
    }


async def main_async(snap_path, bots, n_games, server_url, concurrency):
    print(f"Loading snap: {snap_path}", flush=True)
    snap_cached = torch.load(snap_path, map_location="cuda", weights_only=False)
    print(f"Snap loaded ({sum(v.numel() for v in snap_cached['model_state_dict'].values() if hasattr(v, 'numel')):,} params)", flush=True)

    server_cfg = resolve_server(server_url)
    teambuilder = random_pool_teambuilder()

    results = []
    for idx, bot_name in enumerate(bots):
        print(f"\n=== {bot_name} ===", flush=True)
        r = await run_matchup(snap_path, snap_cached, bot_name, n_games, server_cfg, teambuilder, concurrency, match_idx=idx)
        results.append(r)
        print(f"  snap {r['snap_wins']}-{r['bot_wins']} ({r['snap_wr']*100:.0f}% snap WR, {r['elapsed_s']}s, {r['n_replays']} replays @ {r['replay_dir']})", flush=True)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'Bot':<25} {'Snap WR':>10} {'W-L':>10} {'Replays':>10}")
    for r in results:
        print(f"{r['bot']:<25} {r['snap_wr']*100:>9.0f}% {r['snap_wins']}-{r['bot_wins']:<5} {r['n_replays']:>10}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--bots", required=True, help="Comma-separated bot names")
    p.add_argument("--n-games", type=int, default=20)
    p.add_argument("--server", default="ws://127.0.0.1:9000/showdown/websocket")
    p.add_argument("--concurrency", type=int, default=10)
    args = p.parse_args()

    bot_list = [b.strip() for b in args.bots.split(",")]
    for b in bot_list:
        if b not in ALL_BOTS:
            raise SystemExit(f"Unknown bot: {b}. Available: {sorted(ALL_BOTS.keys())}")

    asyncio.run(main_async(args.snapshot, bot_list, args.n_games, args.server, args.concurrency))


if __name__ == "__main__":
    main()
