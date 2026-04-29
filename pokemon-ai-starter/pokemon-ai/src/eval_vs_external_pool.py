"""eval_vs_external_pool.py — eval one or more checkpoints against the full
external opponent pool (FP + MM + mcts) defined in a YAML.

Why this exists: eval_elo_ladder.py only supports snapshot+heuristic-bot
matchups. After S43 we need a way to measure how each candidate checkpoint
fares against the SAME opponents we trained against — FP MCTS, Metamon
variants, in-process mcts. This is also the most relevant signal for
deciding which checkpoint to ship to the live PokeAgent ladder.

Usage:
    python -u eval_vs_external_pool.py \
      --checkpoints \
        data/models/rl_v9_full_pool/.../snapshot_0114.pt \
        data/models/rl_v9/selfplay_v9_20260425_062416/snapshot_0229.pt \
        data/models/bc/v8_bc_20260423_195603/best.pt \
      --names sp_0114 sp_0229 bc_base \
      --external-adapters external_adapters_full_pool.yaml \
      --n-games 50 \
      --concurrency 4 \
      --device cuda \
      --server ws://127.0.0.1:9000/showdown/websocket \
      --out-json data/eval/vs_external_pool.json

Output: per-checkpoint × per-opponent W/L table, both as a JSON dump and
a human-readable table on stdout.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List

from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from battle_agent import BattleAgent
from external_adapters import load_pool_entries
from team_generator import enqueue_team, procedural_teambuilder


def _make_server(ws_url: str) -> ServerConfiguration:
    ws = ws_url.strip().rstrip("/")
    if not ws.endswith("/showdown/websocket"):
        ws += "/showdown/websocket"
    if not ws.startswith("ws://"):
        ws = "ws://" + ws
    http = ws.replace("ws://", "http://").replace("/showdown/websocket", "/action.php?")
    return ServerConfiguration(ws, http)


async def _eval_one_matchup(player: BattleAgent, entry, teambuilder,
                            n_games: int, server, timeout_s: int) -> tuple:
    """Run player vs one external entry. Returns (wins, losses, ties)."""
    player.reset_battles()
    if entry.factory is not None:
        # In-process adapter (mcts-fast)
        opp = entry.factory(
            server_configuration=server,
            account_configuration=AccountConfiguration(f"EvalOpp{int(time.time())%10000}", None),
            team=teambuilder,
            battle_format="gen9ou",
            max_concurrent_battles=4,
            **(entry.factory_kwargs or {}),
        )
        try:
            await asyncio.wait_for(
                player.battle_against(opp, n_battles=n_games),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            print(f"  [TIMEOUT] {entry.key} after {timeout_s}s")
        try: opp.reset_battles()
        except Exception: pass
    elif entry.showdown_username is not None:
        # Subprocess opponent (FP / MM)
        if entry.team_queue_dir:
            for _ in range(n_games):
                try:
                    enqueue_team(entry.team_queue_dir, teambuilder.yield_team())
                except Exception as e:
                    print(f"  [WARN] enqueue failed for {entry.key}: {e}")
        try:
            await asyncio.wait_for(
                player.send_challenges(entry.showdown_username, n_challenges=n_games),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            print(f"  [TIMEOUT] {entry.key} after {timeout_s}s")
    return player.n_won_battles, player.n_lost_battles, player.n_tied_battles


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--external-adapters", required=True)
    p.add_argument("--procedural-teams", required=True)
    p.add_argument("--n-games", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--server", default="ws://127.0.0.1:9000/showdown/websocket")
    p.add_argument("--out-json", required=True)
    p.add_argument("--per-game-timeout-s", type=int, default=600,
                   help="Per-matchup timeout cap; n_games × 30s default.")
    args = p.parse_args()

    if len(args.checkpoints) != len(args.names):
        raise SystemExit("--checkpoints and --names must have the same length")

    # Load external adapters + spawn any subprocess opponents.
    print(f"[eval] loading external adapters from {args.external_adapters}")
    entries, manager = load_pool_entries(args.external_adapters, default_server_port=9000)
    print(f"[eval] loaded {len(entries)} adapter entries")

    if manager is not None:
        print(f"[eval] starting {len(manager.opponents)} subprocess adapter(s)...")
        manager.start_all()
        print(f"[eval] waiting for subprocesses to be ready (max 180s/each)...")
        ready = manager.wait_until_ready(per_opp_timeout_s=180.0)
        if not ready:
            print(f"[eval] WARN — one or more adapters not ready; proceeding anyway")

    server = _make_server(args.server)
    teambuilder = procedural_teambuilder(args.procedural_teams)

    timeout_s = max(args.per_game_timeout_s, args.n_games * 30)
    results: Dict[str, Dict[str, List[int]]] = {}

    try:
        for ckpt, name in zip(args.checkpoints, args.names):
            print(f"\n[eval] === {name} ({ckpt}) ===")
            results[name] = {}
            for entry in entries:
                player = BattleAgent(
                    checkpoint_path=ckpt,
                    device=args.device,
                    battle_format="gen9ou",
                    team=teambuilder,
                    max_concurrent_battles=args.concurrency,
                    account_configuration=AccountConfiguration(f"Eval{name[:8]}{int(time.time())%1000}", None),
                    server_configuration=server,
                )
                t0 = time.time()
                w, l, t = await _eval_one_matchup(player, entry, teambuilder,
                                                   args.n_games, server, timeout_s)
                dt = time.time() - t0
                pct = 100 * w / max(1, w + l + t)
                print(f"  vs {entry.key:25s}  {w:>3}W/{l:<3}L  ({pct:>5.1f}%) [{dt:.0f}s]")
                results[name][entry.key] = [w, l, t]
                # Free the player before next matchup
                try:
                    player.reset_battles()
                except Exception:
                    pass
                del player

        # Save results
        out = {
            "checkpoints": dict(zip(args.names, args.checkpoints)),
            "external_adapters_yaml": args.external_adapters,
            "n_games_per_matchup": args.n_games,
            "results": results,
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"\n[eval] saved results to {args.out_json}")

        # Print summary table
        all_opps = sorted({k for r in results.values() for k in r})
        print("\n" + "=" * 80)
        print(f"{'opponent':<25}", end="")
        for n in args.names:
            print(f"{n:>14}", end="")
        print()
        print("-" * 80)
        for opp in all_opps:
            print(f"{opp:<25}", end="")
            for n in args.names:
                rec = results[n].get(opp, [0, 0, 0])
                w, l, t = rec
                g = w + l + t
                if g > 0:
                    print(f" {w:>3}/{g:<3} ({100*w/g:>3.0f}%)", end="")
                else:
                    print(f"     -      ", end="")
            print()
    finally:
        if manager is not None:
            print("\n[eval] stopping subprocess adapters...")
            manager.stop_all()


if __name__ == "__main__":
    asyncio.run(main())
