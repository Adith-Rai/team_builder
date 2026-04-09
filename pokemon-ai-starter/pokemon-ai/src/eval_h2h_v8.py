#!/usr/bin/env python3
"""Head-to-head tournament between v8 PokeTransformer checkpoints.

Round-robin: every pair plays N battles, random team per battle.
Outputs win matrix + overall rankings.

Usage:
  python -u eval_h2h_v8.py \
    --checkpoints ckpt1.pt ckpt2.pt ckpt3.pt \
    --names BC iter30 iter120 \
    --n-battles 200 --server ws://127.0.0.1:9000/showdown/websocket \
    --device cuda --concurrency 10
"""

import argparse, asyncio, json, os, shutil, sys, time
from pathlib import Path
from itertools import combinations

import torch
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.ps_client.account_configuration import AccountConfiguration

from battle_agent import BattleAgent
from teams_ou import random_pool_teambuilder


_pid = os.getpid() % 10000
_match_id = 0


def resolve_server(ws_url: str) -> ServerConfiguration:
    ws = ws_url.rstrip("/")
    http = ws.replace("wss://", "https://").replace("ws://", "http://")
    if ws.endswith("/showdown/websocket"):
        http = http[: http.rfind("/showdown/websocket")] + "/action.php?"
    return ServerConfiguration(ws, http)


async def battle_pair(p1, p2, n_battles, timeout=600):
    await asyncio.wait_for(
        p1.battle_against(p2, n_battles=n_battles),
        timeout=timeout,
    )


def snapshot_replays() -> set:
    root = Path("replays")
    if not root.exists():
        return set()
    return set(str(p) for p in root.rglob("*.html"))


def move_new_replays(before_set: set, dest_dir: Path):
    root = Path("replays")
    if not root.exists():
        return 0
    added = [p for p in root.rglob("*.html") if str(p) not in before_set]
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(added):
        dst = dest_dir / src.name
        try:
            shutil.move(str(src), str(dst))
            count += 1
        except Exception:
            pass
    return count


def run_match(ckpt_a, name_a, ckpt_b, name_b, n_battles, server, device,
              battle_format="gen9ou", concurrency=10,
              save_replays=False, replay_root="data/replays/h2h_v8"):
    """Run n_battles between two v8 checkpoints. Returns result dict."""
    global _match_id
    _match_id += 1
    mid = _match_id

    server_cfg = resolve_server(server)
    before = snapshot_replays() if save_replays else set()

    p1 = BattleAgent(
        checkpoint_path=ckpt_a,
        device=device,
        battle_format=battle_format,
        team=random_pool_teambuilder(),
        save_replays=save_replays,
        max_concurrent_battles=concurrency,
        account_configuration=AccountConfiguration(f"H2H{_pid}m{mid}a", None),
        server_configuration=server_cfg,
    )
    p2 = BattleAgent(
        checkpoint_path=ckpt_b,
        device=device,
        battle_format=battle_format,
        team=random_pool_teambuilder(),
        save_replays=save_replays,
        max_concurrent_battles=concurrency,
        account_configuration=AccountConfiguration(f"H2H{_pid}m{mid}b", None),
        server_configuration=server_cfg,
    )

    t0 = time.time()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(battle_pair(p1, p2, n_battles,
                                            timeout=max(600, n_battles * 10)))
    finally:
        loop.close()
    elapsed = time.time() - t0

    w1, w2 = p1.n_won_battles, p2.n_won_battles
    ties = p1.n_tied_battles
    total = w1 + w2 + ties

    # Move replays
    n_replays = 0
    if save_replays:
        dest = Path(replay_root) / f"{name_a}_vs_{name_b}"
        n_replays = move_new_replays(before, dest)

    result = {
        "p1": name_a, "p2": name_b,
        "p1_wins": w1, "p2_wins": w2, "ties": ties,
        "total": total,
        "p1_wr": w1 / max(1, total),
        "p2_wr": w2 / max(1, total),
        "elapsed": round(elapsed, 1),
        "replays": n_replays,
    }

    print(f"  {name_a} vs {name_b}: {w1}-{w2} (ties:{ties}) | "
          f"{result['p1_wr']:.0%} vs {result['p2_wr']:.0%} | {elapsed:.0f}s"
          f"{f' | {n_replays} replays' if save_replays else ''}",
          flush=True)

    return result


def main():
    p = argparse.ArgumentParser(description="V8 H2H tournament")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--names", nargs="+", default=None)
    p.add_argument("--n-battles", type=int, default=200)
    p.add_argument("--format", default="gen9ou")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--server", default="ws://127.0.0.1:9000/showdown/websocket")
    p.add_argument("--save-replays", action="store_true")
    p.add_argument("--replay-root", default="data/replays/h2h_v8")
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    names = args.names or [Path(c).stem for c in args.checkpoints]
    assert len(names) == len(args.checkpoints)

    print(f"\n=== V8 H2H Tournament ===")
    print(f"Players: {names}")
    print(f"Battles per matchup: {args.n_battles}")
    print(f"Device: {args.device}, Server: {args.server}")
    print(f"Random team per battle: YES\n", flush=True)

    # Round-robin
    results = []
    win_matrix = {n: {n2: 0 for n2 in names} for n in names}
    game_matrix = {n: {n2: 0 for n2 in names} for n in names}

    pairs = list(combinations(range(len(names)), 2))
    print(f"{len(pairs)} matchups to play\n", flush=True)

    for pi, (i, j) in enumerate(pairs):
        print(f"[{pi+1}/{len(pairs)}] {names[i]} vs {names[j]}:", flush=True)
        r = run_match(
            args.checkpoints[i], names[i],
            args.checkpoints[j], names[j],
            args.n_battles, args.server, args.device,
            battle_format=args.format, concurrency=args.concurrency,
            save_replays=args.save_replays, replay_root=args.replay_root,
        )
        results.append(r)
        win_matrix[names[i]][names[j]] = r["p1_wins"]
        win_matrix[names[j]][names[i]] = r["p2_wins"]
        game_matrix[names[i]][names[j]] = r["total"]
        game_matrix[names[j]][names[i]] = r["total"]

    # Print results
    print(f"\n{'='*60}")
    print(f"  RESULTS ({args.n_battles} games per matchup)")
    print(f"{'='*60}\n")

    # Win rate matrix
    header = f"{'':>12s}" + "".join(f"{n:>10s}" for n in names) + "     Avg WR"
    print(header)
    print("-" * len(header))

    rankings = []
    for n in names:
        total_w = sum(win_matrix[n][n2] for n2 in names if n2 != n)
        total_g = sum(game_matrix[n][n2] for n2 in names if n2 != n)
        avg_wr = total_w / max(1, total_g)

        row = f"{n:>12s}"
        for n2 in names:
            if n == n2:
                row += f"{'---':>10s}"
            else:
                g = game_matrix[n][n2]
                w = win_matrix[n][n2]
                wr = w / max(1, g)
                row += f"{wr:>9.0%} "
        row += f"    {avg_wr:.1%}"
        print(row)
        rankings.append((n, avg_wr, total_w, total_g))

    print()
    rankings.sort(key=lambda x: -x[1])
    print("Rankings:")
    for rank, (n, wr, w, g) in enumerate(rankings, 1):
        print(f"  #{rank} {n}: {wr:.1%} ({w}/{g})")

    # Save JSON
    if args.out_json:
        out = {
            "config": {
                "n_battles": args.n_battles,
                "format": args.format,
                "checkpoints": {n: c for n, c in zip(names, args.checkpoints)},
            },
            "matchups": results,
            "rankings": [{"name": n, "avg_wr": round(wr, 3), "wins": w, "games": g}
                         for n, wr, w, g in rankings],
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved results to {args.out_json}")


if __name__ == "__main__":
    main()
