#!/usr/bin/env python3
"""Head-to-head tournament between multiple BC/IQL checkpoints.

Runs a round-robin of all checkpoint pairs, plus optionally vs heuristic bots.
Saves replays for playstyle analysis.
"""

import argparse, asyncio, json, os, sys, shutil, time
from pathlib import Path
from itertools import combinations

import torch
from poke_env.ps_client.server_configuration import ServerConfiguration

from bc_policy_player import BCPolicyPlayer
from teams_ou import random_pool_teambuilder


def resolve_server(ws_url: str) -> ServerConfiguration:
    ws = ws_url.rstrip("/")
    http = ws.replace("wss://", "https://").replace("ws://", "http://")
    if ws.endswith("/showdown/websocket"):
        http = http[: http.rfind("/showdown/websocket")] + "/action.php?"
    return ServerConfiguration(ws, http)


def snapshot_replays() -> set:
    root = Path("replays")
    if not root.exists():
        return set()
    return set(str(p) for p in root.rglob("*.html"))


def move_new_replays(before_set: set, dest_dir: Path, tag: str):
    root = Path("replays")
    if not root.exists():
        return []
    added = [p for p in root.rglob("*.html") if str(p) not in before_set]
    moved = []
    for i, src in enumerate(sorted(added)):
        dst = dest_dir / f"{tag}_{i:04d}.html"
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(src), str(dst))
        except Exception:
            try:
                shutil.copy2(str(src), str(dst))
            except Exception:
                continue
        moved.append(str(dst))
    return moved


async def battle_pair(p1, p2, n_battles, timeout=300):
    """Run n_battles between p1 and p2 with a per-match timeout (default 5 min)."""
    await asyncio.wait_for(
        p1.battle_against(p2, n_battles=n_battles),
        timeout=timeout,
    )


def run_match(ckpt_a, name_a, ckpt_b, name_b, n_battles, server, device,
              save_replays, replay_root, use_direct=False, battle_format="gen9ou",
              concurrency=1):
    """Run n_battles between two checkpoints. Returns result dict."""

    before = snapshot_replays() if save_replays else set()

    # Common kwargs; when --direct, skip websocket listener
    extra_kwargs = {}
    if use_direct:
        extra_kwargs["start_listening"] = False
    else:
        server_cfg = resolve_server(server)
        extra_kwargs["server_configuration"] = server_cfg

    p1 = BCPolicyPlayer(
        checkpoint_path=ckpt_a,
        device=device,
        battle_format=battle_format,
        team=random_pool_teambuilder(),
        save_replays=save_replays,
        max_concurrent_battles=concurrency,
        **extra_kwargs,
    )
    p2 = BCPolicyPlayer(
        checkpoint_path=ckpt_b,
        device=device,
        battle_format=battle_format,
        team=random_pool_teambuilder(),
        save_replays=save_replays,
        max_concurrent_battles=concurrency,
        **extra_kwargs,
    )

    # Patch players for direct transport if --direct
    if use_direct:
        from direct_player import patch_to_direct, direct_battle_against
        patch_to_direct(p1)
        patch_to_direct(p2)

    t0 = time.time()
    if use_direct:
        from poke_env.concurrency import POKE_LOOP
        future = asyncio.run_coroutine_threadsafe(
            direct_battle_against(p1, p2, n_battles=n_battles),
            POKE_LOOP,
        )
        future.result(timeout=n_battles * 60)
    else:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(battle_pair(p1, p2, n_battles))
        finally:
            loop.close()
    elapsed = time.time() - t0

    w1, w2 = p1.n_won_battles, p2.n_won_battles
    ties = p1.n_tied_battles

    # Move replays
    moved = []
    if save_replays:
        tag = f"{name_a}_vs_{name_b}"
        dest = Path(replay_root) / tag
        dest.mkdir(parents=True, exist_ok=True)
        moved = move_new_replays(before, dest, tag)

    result = {
        "p1": name_a, "p2": name_b,
        "p1_wins": w1, "p2_wins": w2, "ties": ties,
        "total": w1 + w2 + ties,
        "p1_wr": w1 / max(1, w1 + w2 + ties),
        "p2_wr": w2 / max(1, w1 + w2 + ties),
        "elapsed": round(elapsed, 1),
        "replays": len(moved),
    }

    print(f"  {name_a} vs {name_b}: {w1}-{w2} (ties:{ties}) | "
          f"{result['p1_wr']:.0%} vs {result['p2_wr']:.0%} | {elapsed:.0f}s | "
          f"{len(moved)} replays", flush=True)

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Checkpoint paths")
    p.add_argument("--names", nargs="+", default=None,
                   help="Names for each checkpoint (default: filenames)")
    p.add_argument("--n-battles", type=int, default=30,
                   help="Games per matchup")
    p.add_argument("--format", default="gen9ou",
                   help="Battle format (default: gen9ou)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Max concurrent battles per matchup")
    p.add_argument("--device", default="cpu")
    p.add_argument("--server", default="ws://127.0.0.1:8000/showdown/websocket")
    p.add_argument("--save-replays", action="store_true")
    p.add_argument("--replay-root", default="data/replays/replays_h2h")
    p.add_argument("--out-json", default=None,
                   help="Save results JSON to this path")
    p.add_argument("--direct", action="store_true",
                   help="Use direct BattleStream transport (no websockets/Docker)")
    args = p.parse_args()

    names = args.names or [Path(c).stem for c in args.checkpoints]
    assert len(names) == len(args.checkpoints), "Need same number of names and checkpoints"

    # Verify all checkpoints have LSTM
    for ckpt, name in zip(args.checkpoints, names):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = ck.get("policy_cfg", ck.get("cfg", {}))
        lstm = cfg.get("use_lstm", False)
        print(f"  {name}: use_lstm={lstm}", flush=True)
        if not lstm:
            print(f"  WARNING: {name} has use_lstm=False!")

    n = len(args.checkpoints)
    pairs = list(combinations(range(n), 2))
    print(f"\nRound-robin: {n} models, {len(pairs)} matchups, "
          f"{args.n_battles} games each = {len(pairs) * args.n_battles} total games\n",
          flush=True)

    all_results = []
    # Win tracking for standings
    wins = {name: 0 for name in names}
    losses = {name: 0 for name in names}
    ties_count = {name: 0 for name in names}
    games = {name: 0 for name in names}

    for i, (a, b) in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}]", end=" ", flush=True)
        result = run_match(
            args.checkpoints[a], names[a],
            args.checkpoints[b], names[b],
            args.n_battles, args.server, args.device,
            args.save_replays, args.replay_root,
            use_direct=getattr(args, "direct", False),
            battle_format=args.format,
            concurrency=args.concurrency,
        )
        all_results.append(result)

        wins[names[a]] += result["p1_wins"]
        wins[names[b]] += result["p2_wins"]
        losses[names[a]] += result["p2_wins"]
        losses[names[b]] += result["p1_wins"]
        ties_count[names[a]] += result["ties"]
        ties_count[names[b]] += result["ties"]
        games[names[a]] += result["total"]
        games[names[b]] += result["total"]

    # Print standings
    print(f"\n{'='*60}")
    print(f"  ROUND-ROBIN STANDINGS")
    print(f"{'='*60}")
    print(f"  {'Model':<20s} {'W':>5s} {'L':>5s} {'T':>5s} {'Games':>6s} {'WR':>7s}")
    print(f"  {'-'*20} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*7}")

    standings = sorted(names, key=lambda n: wins[n] / max(1, games[n]), reverse=True)
    for name in standings:
        wr = wins[name] / max(1, games[name])
        print(f"  {name:<20s} {wins[name]:5d} {losses[name]:5d} {ties_count[name]:5d} {games[name]:6d} {wr:6.1%}")

    # Head-to-head matrix
    print(f"\n  HEAD-TO-HEAD MATRIX (row win% vs column)")
    print(f"  {'':20s}", end="")
    for name in standings:
        print(f" {name[:8]:>8s}", end="")
    print()

    h2h = {}
    for r in all_results:
        h2h[(r["p1"], r["p2"])] = r["p1_wr"]
        h2h[(r["p2"], r["p1"])] = r["p2_wr"]

    for row in standings:
        print(f"  {row:<20s}", end="")
        for col in standings:
            if row == col:
                print(f" {'---':>8s}", end="")
            elif (row, col) in h2h:
                print(f" {h2h[(row, col)]:>7.0%}", end=" ")
            else:
                print(f" {'?':>8s}", end="")
        print()

    # Save results
    output = {
        "matchups": all_results,
        "standings": {name: {"wins": wins[name], "losses": losses[name],
                             "ties": ties_count[name], "games": games[name],
                             "win_rate": wins[name] / max(1, games[name])}
                      for name in names},
        "n_battles_per_matchup": args.n_battles,
    }

    out_path = args.out_json or f"data/evaluations/h2h_{time.strftime('%Y%m%d_%H%M%S')}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
    if "--direct" in sys.argv:
        import os
        os._exit(0)
