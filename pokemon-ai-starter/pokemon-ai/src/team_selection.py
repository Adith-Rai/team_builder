#!/usr/bin/env python
"""Team Selection: test our model with each of the 70 teams against 4 smart bots.

Ranks teams by overall win rate to find which teams our model plays best with.
High concurrency since opponents are CPU bots (no GPU needed for them).

Usage:
  python team_selection.py --checkpoint <path> --servers 9000,9001,9002 \
      --concurrency 300 --games-per-bot 50 --device cuda

Output: ranked table of all 70 teams with per-bot and overall win rates.
"""
import argparse
import asyncio
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.player.baselines import SimpleHeuristicsPlayer
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from policy_smartbots import SmartDamagePlayer, TacticalPlayer, StrategicPlayer
from battle_agent import BattleAgent
from teams_ou import TEAMS, list_teams


def make_server(port_or_url):
    if isinstance(port_or_url, int) or port_or_url.isdigit():
        ws = f"ws://127.0.0.1:{port_or_url}/showdown/websocket"
    else:
        ws = port_or_url
    http = ws.replace("ws://", "http://").replace("/showdown/websocket", "/action.php?")
    return ServerConfiguration(ws, http)


OPPONENTS = [
    (SimpleHeuristicsPlayer, "SH"),
    (SmartDamagePlayer, "SmartDmg"),
    (TacticalPlayer, "Tactical"),
    (StrategicPlayer, "Strategic"),
]


async def eval_team_vs_bot(cached_ckpt, checkpoint_path, device, team_str, team_name,
                            opp_cls, opp_name, n_games, concurrency, server):
    """Play n_games with a specific team against a specific bot.
    Uses cached_ckpt to avoid reloading model from disk each time."""
    from teams_ou import random_pool_teambuilder
    tb = ConstantTeambuilder(team_str)
    p1 = BattleAgent(
        checkpoint_path, device=device, _cached_ckpt=cached_ckpt,
        account_configuration=AccountConfiguration.generate(f"T{team_name[:6]}", rand=True),
        battle_format="gen9ou",
        max_concurrent_battles=concurrency,
        server_configuration=server,
        team=tb,
    )
    p2 = opp_cls(
        account_configuration=AccountConfiguration.generate(f"B{opp_name[:4]}", rand=True),
        battle_format="gen9ou",
        max_concurrent_battles=concurrency,
        server_configuration=server,
        team=random_pool_teambuilder(),
    )

    try:
        await asyncio.wait_for(
            p1.battle_against(p2, n_battles=n_games),
            timeout=max(180, n_games * 30),
        )
    except asyncio.TimeoutError:
        print(f"  [WARN] Timeout: {team_name} vs {opp_name}", flush=True)
    except Exception as e:
        print(f"  [ERROR] {team_name} vs {opp_name}: {e}", flush=True)

    wins = p1.n_won_battles
    total = p1.n_won_battles + p1.n_lost_battles + p1.n_tied_battles
    wr = wins / max(1, total) * 100

    try:
        p1.reset_battles()
    except Exception:
        pass
    try:
        p2.reset_battles()
    except Exception:
        pass
    del p1, p2

    return wr, wins, total


async def eval_one_team(cached_ckpt, checkpoint_path, device, team_str, team_name,
                         n_games_per_bot, concurrency, servers):
    """Evaluate one team against all 4 bots, using round-robin across servers."""
    results = {}
    for i, (opp_cls, opp_name) in enumerate(OPPONENTS):
        server = servers[i % len(servers)]
        wr, wins, total = await eval_team_vs_bot(
            cached_ckpt, checkpoint_path, device, team_str, team_name,
            opp_cls, opp_name, n_games_per_bot, concurrency, server,
        )
        results[opp_name] = wr
    results["savg"] = sum(results[n] for _, n in OPPONENTS) / len(OPPONENTS)
    return results


def main():
    parser = argparse.ArgumentParser(description="Team selection: rank 70 teams by win rate")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--servers", default="9000,9001,9002",
                        help="Comma-separated server ports or URLs")
    parser.add_argument("--concurrency", type=int, default=300,
                        help="Concurrent battles per matchup (default 300, high for bot eval)")
    parser.add_argument("--games-per-bot", type=int, default=50,
                        help="Games per team per bot (default 50, total = 70*4*50 = 14000)")
    parser.add_argument("--out-json", default="team_selection_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    servers = [make_server(s.strip()) for s in args.servers.split(",")]
    team_names = list_teams()

    # Load checkpoint ONCE (avoids VRAM leak from repeated loads)
    print(f"Loading checkpoint once: {args.checkpoint}", flush=True)
    cached_ckpt = torch.load(args.checkpoint, map_location=torch.device(args.device),
                              weights_only=False)
    print(f"Checkpoint loaded. Model params: {sum(p.numel() for p in cached_ckpt.get('model_state_dict', {}).values() if hasattr(p, 'numel')):,}", flush=True)

    # Resume: load existing results if any
    all_results = {}
    if os.path.exists(args.out_json):
        try:
            with open(args.out_json) as f:
                saved = json.load(f)
            if "raw" in saved:
                all_results = saved["raw"]
            else:
                all_results = saved
            print(f"Resumed: {len(all_results)} teams already done", flush=True)
        except Exception:
            pass

    remaining = [(ti, tn) for ti, tn in enumerate(team_names) if tn not in all_results]
    print(f"\nTeam Selection: {len(remaining)} remaining of {len(team_names)} teams "
          f"x {len(OPPONENTS)} bots x {args.games_per_bot} games")
    print(f"Total games remaining: {len(remaining) * len(OPPONENTS) * args.games_per_bot}")
    print(f"Servers: {len(servers)}, Concurrency: {args.concurrency}")
    print(f"Device: {args.device}")
    print()

    t0 = time.time()
    done_count = 0

    for ti, tname in remaining:
        team_str = TEAMS[ti]
        tt0 = time.time()

        results = asyncio.run(eval_one_team(
            cached_ckpt, args.checkpoint, args.device, team_str, tname,
            args.games_per_bot, args.concurrency, servers,
        ))
        all_results[tname] = results
        done_count += 1
        elapsed = time.time() - tt0
        total_elapsed = time.time() - t0
        eta = (total_elapsed / done_count) * (len(remaining) - done_count)

        print(f"[{ti+1:2d}/{len(team_names)}] {tname:8s}: "
              f"SH={results['SH']:.0f}% SmD={results['SmartDmg']:.0f}% "
              f"Tac={results['Tactical']:.0f}% Str={results['Strategic']:.0f}% "
              f"savg={results['savg']:.1f}%  ({elapsed:.0f}s, ETA {eta/60:.0f}m)",
              flush=True)

        # Save incrementally
        if done_count % 5 == 0 or done_count == len(remaining):
            with open(args.out_json, "w") as f:
                json.dump(all_results, f, indent=2)

        gc.collect()
        torch.cuda.empty_cache()

    # Final ranking
    print()
    print("=" * 80)
    print("TEAM RANKING (by smart_avg)")
    print("=" * 80)
    ranked = sorted(all_results.items(), key=lambda x: -x[1]["savg"])
    print(f"\n{'Rank':>4s}  {'Team':>8s}  {'savg':>6s}  {'SH':>5s}  {'SmD':>5s}  {'Tac':>5s}  {'Str':>5s}")
    print("-" * 55)
    for rank, (tname, r) in enumerate(ranked, 1):
        print(f"  {rank:2d}   {tname:>8s}  {r['savg']:5.1f}%  {r['SH']:4.0f}%  {r['SmartDmg']:4.0f}%  "
              f"{r['Tactical']:4.0f}%  {r['Strategic']:4.0f}%")

    # Save final
    with open(args.out_json, "w") as f:
        json.dump({"ranking": [{"rank": i+1, "team": t, **r}
                                for i, (t, r) in enumerate(ranked)],
                   "raw": all_results}, f, indent=2)
    print(f"\nSaved results to {args.out_json}")

    # Summary
    top5 = ranked[:5]
    bot5 = ranked[-5:]
    print(f"\nTop 5: {', '.join(t for t, _ in top5)} (mean savg {sum(r['savg'] for _, r in top5)/5:.1f}%)")
    print(f"Bot 5: {', '.join(t for t, _ in bot5)} (mean savg {sum(r['savg'] for _, r in bot5)/5:.1f}%)")
    spread = ranked[0][1]["savg"] - ranked[-1][1]["savg"]
    print(f"Spread: {spread:.1f}% (top - bottom)")
    print(f"Total time: {(time.time() - t0)/60:.0f} min")


if __name__ == "__main__":
    main()
