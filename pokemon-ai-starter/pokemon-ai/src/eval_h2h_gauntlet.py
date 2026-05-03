"""eval_h2h_gauntlet.py — head-to-head tournament between checkpoints.

Mixes legacy MLP-arch (`BattleAgent`) and new transformer-arch
(`BattleAgentTransformer`) via arch dispatch. Both sides sample from the
same MetamonCompetitiveTeambuilder so team-quality variance is bounded
(matching the smart_avg eval methodology).

Usage:
  python eval_h2h_gauntlet.py \
    --champion v10_cloud_e1=path/to/epoch_001.pt \
    --opponents sp_0229=path/to/sp_0229.pt iter_0119=path/to/iter_0119.pt ... \
    --n-battles 100 --concurrency 100 --device cuda \
    --out-json data/eval/registry/h2h_v10_gauntlet.json

Champion fights each opponent. Reports win rate per matchup + JSON dump.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from battle_agent import BattleAgent
from battle_agent_transformer import BattleAgentTransformer, is_transformer_checkpoint
from eval_metamon_competitive import MetamonCompetitiveTeambuilder, _default_metamon_teams_dir


def make_server(port_or_url):
    s = str(port_or_url)
    ws = f"ws://127.0.0.1:{s}/showdown/websocket" if s.isdigit() else s
    http = ws.replace("ws://", "http://").replace("/showdown/websocket", "/action.php?")
    return ServerConfiguration(ws, http)


def parse_ckpt_arg(s: str):
    if "=" in s:
        label, path = s.split("=", 1)
    else:
        label = Path(s).stem
        path = s
    return label.strip(), path.strip()


def make_player(label: str, path: str, ckpt_dict: dict, device: str, server, tb,
                concurrency: int, role: str):
    """Spawn the right Player class for the checkpoint's arch."""
    AgentClass = BattleAgentTransformer if is_transformer_checkpoint(ckpt_dict) else BattleAgent
    arch = "tx" if AgentClass is BattleAgentTransformer else "mlp"
    return AgentClass(
        path, device=device, _cached_ckpt=ckpt_dict,
        # Account name max 16 chars; squeeze label + role.
        account_configuration=AccountConfiguration.generate(
            f"H{role[0]}{label[:6]}", rand=True
        ),
        battle_format="gen9ou",
        max_concurrent_battles=concurrency,
        server_configuration=server,
        team=tb,
    ), arch


async def run_matchup(p1_label, p1_path, p1_ckpt,
                     p2_label, p2_path, p2_ckpt,
                     n_battles, concurrency, device, server, tb):
    """Run n_battles between two checkpoints. Returns dict."""
    p1, p1_arch = make_player(p1_label, p1_path, p1_ckpt, device, server, tb, concurrency, "a")
    p2, p2_arch = make_player(p2_label, p2_path, p2_ckpt, device, server, tb, concurrency, "b")

    try:
        await asyncio.wait_for(
            p1.battle_against(p2, n_battles=n_battles),
            timeout=max(180, n_battles * 30),
        )
    except asyncio.TimeoutError:
        print(f"  [WARN] Timeout: {p1_label} vs {p2_label}", flush=True)
    except Exception as e:
        print(f"  [ERROR] {p1_label} vs {p2_label}: {e}", flush=True)

    w1 = p1.n_won_battles
    w2 = p2.n_won_battles
    ties = p1.n_tied_battles
    total = w1 + w2 + ties
    wr1 = 100 * w1 / max(1, total)
    wr2 = 100 * w2 / max(1, total)

    try: p1.reset_battles()
    except Exception: pass
    try: p2.reset_battles()
    except Exception: pass
    del p1, p2

    return {
        "p1": p1_label, "p2": p2_label,
        "p1_arch": p1_arch, "p2_arch": p2_arch,
        "wins": w1, "losses": w2, "ties": ties, "total": total,
        "wr_pct": wr1, "opp_wr_pct": wr2,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--champion", required=True,
                   help="'label=path' for the contender (gauntlet runs champion vs each opponent).")
    p.add_argument("--opponents", nargs="+", required=True,
                   help="One or more 'label=path' for opponents.")
    p.add_argument("--n-battles", type=int, default=100)
    p.add_argument("--concurrency", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--servers", default="9000")
    p.add_argument("--out-json", default="data/eval/registry/h2h_gauntlet.json")
    p.add_argument("--team-dir", default=str(_default_metamon_teams_dir()))
    args = p.parse_args()

    server = make_server(args.servers.split(",")[0])
    tb = MetamonCompetitiveTeambuilder(Path(args.team_dir))

    champ_label, champ_path = parse_ckpt_arg(args.champion)
    opp_specs = [parse_ckpt_arg(o) for o in args.opponents]

    print(f"\n=== H2H Gauntlet ===")
    print(f"Champion: {champ_label} ({champ_path})")
    print(f"Opponents: {[lbl for lbl, _ in opp_specs]}")
    print(f"Battles per matchup: {args.n_battles} (Metamon competitive teams, 16 teams)")
    print(f"Concurrency: {args.concurrency}")
    print()

    # Pre-load all checkpoints to avoid disk I/O during matchup setup.
    print("[load] caching checkpoints...")
    if not Path(champ_path).exists():
        raise SystemExit(f"Champion not found: {champ_path}")
    champ_ckpt = torch.load(champ_path, map_location=args.device, weights_only=False)
    opps = []
    for lbl, path in opp_specs:
        if not Path(path).exists():
            print(f"  [SKIP] not found: {lbl} -> {path}")
            continue
        ck = torch.load(path, map_location=args.device, weights_only=False)
        opps.append((lbl, path, ck))
        print(f"  loaded {lbl}")

    print()
    print(f"=== Champion {champ_label} vs each opponent ===\n")

    results = []
    t_start = time.time()
    for opp_label, opp_path, opp_ckpt in opps:
        t0 = time.time()
        r = asyncio.run(run_matchup(
            champ_label, champ_path, champ_ckpt,
            opp_label, opp_path, opp_ckpt,
            args.n_battles, args.concurrency, args.device, server, tb,
        ))
        dt = time.time() - t0
        r["elapsed_s"] = round(dt, 1)
        results.append(r)
        print(f"  {champ_label} ({r['p1_arch']}) vs {opp_label} ({r['p2_arch']:>3}): "
              f"{r['wins']:>3}-{r['losses']:<3} ties:{r['ties']:<2} "
              f"({r['wr_pct']:>5.1f}%) [{dt:.0f}s]", flush=True)

    # Print summary table.
    print()
    print("=" * 78)
    print(f"{'opponent':<22}  arch   record   {'champion%':>9}  {'opp%':>5}  {'gap':>6}")
    print("-" * 78)
    for r in results:
        gap = r["wr_pct"] - r["opp_wr_pct"]
        sign = "+" if gap >= 0 else ""
        print(f"{r['p2']:<22}  {r['p2_arch']:<5}  {r['wins']:>3}-{r['losses']:<3}-{r['ties']:<2}  "
              f"{r['wr_pct']:>7.1f}%  {r['opp_wr_pct']:>4.1f}%  {sign}{gap:>5.1f}")
    print()

    # Aggregate
    total_w = sum(r["wins"] for r in results)
    total_l = sum(r["losses"] for r in results)
    total_t = sum(r["ties"] for r in results)
    total_g = total_w + total_l + total_t
    avg_wr = sum(r["wr_pct"] for r in results) / max(1, len(results))
    print(f"Champion overall: {total_w}W-{total_l}L-{total_t}T across {total_g} games")
    print(f"Avg WR vs opponents: {avg_wr:.1f}%")
    print(f"Total wall-clock: {(time.time() - t_start) / 60:.1f} min")

    # Save JSON.
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    out = {
        "champion": {"label": champ_label, "path": champ_path},
        "n_battles_per_matchup": args.n_battles,
        "concurrency": args.concurrency,
        "team_source": str(args.team_dir),
        "matchups": results,
        "summary": {
            "total_wins": total_w, "total_losses": total_l, "total_ties": total_t,
            "total_games": total_g, "avg_wr_pct": avg_wr,
        },
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
