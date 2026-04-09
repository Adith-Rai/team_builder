#!/usr/bin/env python3
"""
Round-robin bot-vs-bot evaluation.
Usage:
    SHOWDOWN_HOST=127.0.0.1 python eval_bots_roundrobin.py --n-battles 50
"""
from __future__ import annotations
import argparse, os, time, asyncio, csv, json
from pathlib import Path
from itertools import combinations

from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer

from policy_rulebots import (
    GreedySEPlayer, HazardSensePlayer,
    SwitchAwareEscapePlayer, SetupThenSweepPlayer,
)
from policy_smartbots import SmartDamagePlayer, TacticalPlayer, StrategicPlayer
from policy_random import RandomPolicy
from teams_ou import random_teambuilder, random_pool_teambuilder

BOTS = {
    "Random": RandomPolicy,
    "MaxBasePower": MaxBasePowerPlayer,
    "GreedySE": GreedySEPlayer,
    "HazardSense": HazardSensePlayer,
    "SwitchAwareEscape": SwitchAwareEscapePlayer,
    "SetupThenSweep": SetupThenSweepPlayer,
    "SimpleHeuristics": SimpleHeuristicsPlayer,
    "SmartDamage": SmartDamagePlayer,
    "Tactical": TacticalPlayer,
    "Strategic": StrategicPlayer,
}


def resolve_server(ws_url: str) -> ServerConfiguration:
    ws = ws_url.rstrip("/")
    http = ws.replace("wss://", "https://").replace("ws://", "http://")
    if ws.endswith("/showdown/websocket"):
        http = http[: http.rfind("/showdown/websocket")] + "/action.php?"
    return ServerConfiguration(ws, http)


async def run_matchup(cls_a, cls_b, name_a, name_b, n_battles, server, battle_format):
    a = cls_a(
        battle_format=battle_format,
        server_configuration=server,
        team=random_pool_teambuilder(),
        max_concurrent_battles=1,
    )
    b = cls_b(
        battle_format=battle_format,
        server_configuration=server,
        team=random_pool_teambuilder(),
        max_concurrent_battles=1,
    )
    await a.battle_against(b, n_battles=n_battles)
    return {
        "bot_a": name_a, "bot_b": name_b,
        "a_wins": a.n_won_battles, "b_wins": a.n_lost_battles,
        "ties": a.n_tied_battles,
        "a_winrate": a.n_won_battles / max(1, a.n_won_battles + a.n_lost_battles + a.n_tied_battles),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-battles", type=int, default=50)
    p.add_argument("--format", default="gen9ou")
    p.add_argument("--bots", default=None, help="Comma list; default=all")
    _host = os.environ.get("SHOWDOWN_HOST", "127.0.0.1")
    _port = os.environ.get("SHOWDOWN_PORT", "8000")
    p.add_argument("--server", default=f"ws://{_host}:{_port}/showdown/websocket")
    args = p.parse_args()

    server = resolve_server(args.server)

    if args.bots:
        names = [b.strip() for b in args.bots.split(",")]
    else:
        names = list(BOTS.keys())

    results = []
    wins_table = {n: 0 for n in names}
    games_table = {n: 0 for n in names}

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    pairs = list(combinations(names, 2))
    print(f"Running {len(pairs)} matchups × {args.n_battles} battles each...")

    for i, (a, b) in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] {a} vs {b} ...", end=" ", flush=True)
        t0 = time.time()
        res = loop.run_until_complete(
            run_matchup(BOTS[a], BOTS[b], a, b, args.n_battles, server, args.format)
        )
        dt = time.time() - t0
        print(f"{res['a_wins']}-{res['b_wins']}-{res['ties']} ({dt:.1f}s)")
        results.append(res)
        wins_table[a] += res["a_wins"]
        wins_table[b] += res["b_wins"]
        games_table[a] += res["a_wins"] + res["b_wins"] + res["ties"]
        games_table[b] += res["a_wins"] + res["b_wins"] + res["ties"]

    # Print rankings
    print("\n=== Bot Rankings (total wins / total games) ===")
    ranked = sorted(names, key=lambda n: wins_table[n] / max(1, games_table[n]), reverse=True)
    for rank, n in enumerate(ranked, 1):
        wr = wins_table[n] / max(1, games_table[n])
        print(f"  {rank}. {n:20s}  {wins_table[n]:4d}W / {games_table[n]:4d}G  ({wr:.1%})")

    # Print detailed results matrix
    print("\n=== Head-to-Head Results ===")
    for r in results:
        print(f"  {r['bot_a']:20s} vs {r['bot_b']:20s}: {r['a_wins']}-{r['b_wins']}-{r['ties']} ({r['a_winrate']:.1%})")

    # Save CSV
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data/evaluations/bot_roundrobin") / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bot_a", "bot_b", "a_wins", "b_wins", "ties", "a_winrate"])
        w.writeheader()
        w.writerows(results)

    rankings_path = out_dir / "rankings.json"
    with open(rankings_path, "w") as f:
        json.dump({
            "rankings": [{"rank": i+1, "bot": n,
                          "wins": wins_table[n], "games": games_table[n],
                          "winrate": wins_table[n] / max(1, games_table[n])}
                         for i, n in enumerate(ranked)],
            "n_battles_per_matchup": args.n_battles,
            "timestamp": ts,
        }, f, indent=2)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
