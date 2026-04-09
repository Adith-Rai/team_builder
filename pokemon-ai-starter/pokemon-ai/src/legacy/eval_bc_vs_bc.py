#!/usr/bin/env python3
"""Battle BC checkpoints against each other (round-robin)."""

from __future__ import annotations
import asyncio, argparse, os, sys, time
from itertools import combinations
from pathlib import Path

from poke_env import AccountConfiguration, ServerConfiguration

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bc_policy_player import BCPolicyPlayer
from teams_ou import random_pool_teambuilder


def resolve_server(ws_url: str) -> ServerConfiguration:
    ws = ws_url.rstrip("/")
    http = ws.replace("wss://", "https://").replace("ws://", "http://")
    if ws.endswith("/showdown/websocket"):
        http = http[: http.rfind("/showdown/websocket")] + "/action.php?"
    return ServerConfiguration(ws, http)


_round = 0

async def battle_pair(ckpt_a, name_a, ckpt_b, name_b, n_battles, server_cfg, fmt, device):
    """Run n_battles between two BC checkpoints. Returns (wins_a, wins_b, draws)."""
    global _round
    _round += 1
    player_a = BCPolicyPlayer(
        checkpoint_path=ckpt_a,
        device=device,
        battle_format=fmt,
        server_configuration=server_cfg,
        team=random_pool_teambuilder(),
        max_concurrent_battles=2,
        account_configuration=AccountConfiguration(f"BC{name_a}r{_round}", None),
    )
    player_b = BCPolicyPlayer(
        checkpoint_path=ckpt_b,
        device=device,
        battle_format=fmt,
        server_configuration=server_cfg,
        team=random_pool_teambuilder(),
        max_concurrent_battles=2,
        account_configuration=AccountConfiguration(f"BC{name_b}r{_round}", None),
    )

    await player_a.battle_against(player_b, n_battles=n_battles)

    wins_a = player_a.n_won_battles
    wins_b = player_b.n_won_battles
    draws = n_battles - wins_a - wins_b

    return wins_a, wins_b, draws


def parse_args():
    p = argparse.ArgumentParser(description="BC vs BC round-robin evaluation")
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Paths to best.pt files")
    p.add_argument("--names", nargs="+", default=None,
                   help="Short names for each checkpoint (same order)")
    p.add_argument("--n-battles", type=int, default=50)
    p.add_argument("--format", default="gen9ou")
    p.add_argument("--device", default="cuda")
    _host = os.environ.get("SHOWDOWN_HOST", "127.0.0.1")
    _port = os.environ.get("SHOWDOWN_PORT", "8000")
    p.add_argument("--server", dest="server_url",
                   default=f"ws://{_host}:{_port}/showdown/websocket")
    return p.parse_args()


def main():
    args = parse_args()
    checkpoints = args.checkpoints
    names = args.names or [Path(c).parent.name for c in checkpoints]
    assert len(names) == len(checkpoints), "Must have same number of names and checkpoints"

    server_cfg = resolve_server(args.server_url)
    pairs = list(combinations(range(len(checkpoints)), 2))

    print(f"[BC-vs-BC] {len(checkpoints)} models, {len(pairs)} matchups, {args.n_battles} battles each")
    for i, n in enumerate(names):
        print(f"  [{i}] {n}: {checkpoints[i]}")
    print()

    results = {}  # (i, j) -> (wins_i, wins_j, draws)

    for idx_a, idx_b in pairs:
        na, nb = names[idx_a], names[idx_b]
        ca, cb = checkpoints[idx_a], checkpoints[idx_b]
        print(f"[BC-vs-BC] {na} vs {nb} ({args.n_battles} battles)...", flush=True)
        t0 = time.time()

        wins_a, wins_b, draws = asyncio.run(
            battle_pair(ca, na, cb, nb, args.n_battles, server_cfg, args.format, args.device)
        )

        elapsed = time.time() - t0
        results[(idx_a, idx_b)] = (wins_a, wins_b, draws)
        print(f"  {na}: {wins_a}W  |  {nb}: {wins_b}W  |  draws: {draws}  ({elapsed:.1f}s)")

    # Summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Win totals
    total_wins = {i: 0 for i in range(len(checkpoints))}
    total_games = {i: 0 for i in range(len(checkpoints))}
    for (i, j), (wi, wj, d) in results.items():
        total_wins[i] += wi
        total_wins[j] += wj
        total_games[i] += wi + wj + d
        total_games[j] += wi + wj + d

    for i in range(len(checkpoints)):
        g = total_games[i]
        w = total_wins[i]
        wr = w / g * 100 if g > 0 else 0
        print(f"  {names[i]:20s}: {w:3d}W / {g:3d}G = {wr:.1f}%")

    print("\nHead-to-head:")
    for (i, j), (wi, wj, d) in results.items():
        print(f"  {names[i]} vs {names[j]}: {wi}-{wj} (draws: {d})")


if __name__ == "__main__":
    main()
