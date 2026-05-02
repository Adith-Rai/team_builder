"""Eval N checkpoints against the 4 smart bots, using Metamon's curated
"competitive" 16-team set (downloaded to metamon_cache/teams/competitive/gen9ou/).

Why this exists: our 70-team eval pool has a 51pt smart_avg spread between
TEAM_AX (81.5%) and TEAM_AR (30.5%) on sp_0229. Random-team eval is
dominated by team-draw noise. Metamon's "competitive" set is 16 human-made
Smogon teams (Kakuna, Abra etc. claim 50% GXE on the human ladder with this
set), so team-quality variance is much lower → cleaner skill measurement.

Usage:
  python eval_metamon_competitive.py \
    --checkpoints sp_0229=path1 sp_0114=path2 sp_warmup=path3 ... \
    --servers 9000 \
    --n-games 200 \
    --concurrency 8 \
    --device cuda \
    --out-json data/eval/metamon_competitive_eval.json

Output: per-checkpoint × per-bot W/L table, JSON dump + stdout summary.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.player.baselines import SimpleHeuristicsPlayer
from poke_env.teambuilder.teambuilder import Teambuilder

from policy_smartbots import SmartDamagePlayer, TacticalPlayer, StrategicPlayer
from battle_agent import BattleAgent
from battle_agent_transformer import BattleAgentTransformer, is_transformer_checkpoint


METAMON_COMPETITIVE_DIR = Path(
    "C:/Users/raiad/OneDrive/Desktop/team_builder/metamon_cache/teams/competitive/gen9ou"
)


class MetamonCompetitiveTeambuilder(Teambuilder):
    """Random sampler over Metamon's curated `competitive` gen9ou team files.

    Reads the 16 .gen9ou_team text files at startup, parses each via poke-env's
    Teambuilder.parse_showdown_team, and returns a packed team string per call.
    """

    def __init__(self, team_dir: Path = METAMON_COMPETITIVE_DIR):
        self.team_dir = Path(team_dir)
        self.teams = []
        for f in sorted(self.team_dir.glob("*.gen9ou_team")):
            with open(f, "r", encoding="utf-8") as fh:
                team_data = fh.read().strip()  # strip trailing newlines that parse as empty 7th mon
            mons = self.parse_showdown_team(team_data)
            # parse_showdown_team puts the species name in `nickname` (if no
            # explicit "Nickname (Species)" syntax in the file). The species
            # attribute may be None. So filter on having ANY identifier:
            mons = [m for m in mons
                    if getattr(m, "species", None) or getattr(m, "nickname", None)]
            if len(mons) != 6:
                print(f"  [WARN] {f.name} has {len(mons)} mons (expected 6) — skipping",
                      flush=True)
                continue
            packed = self.join_team(mons)
            self.teams.append(packed)
        if not self.teams:
            raise RuntimeError(
                f"No .gen9ou_team files found in {self.team_dir}. "
                "Run metamon's download_teams() first."
            )
        print(f"[MetamonCompetitive] loaded {len(self.teams)} teams from {self.team_dir}",
              flush=True)

    def yield_team(self) -> str:
        return random.choice(self.teams)


def make_server(port_or_url):
    s = str(port_or_url)
    if s.isdigit():
        ws = f"ws://127.0.0.1:{s}/showdown/websocket"
    else:
        ws = s
    http = ws.replace("ws://", "http://").replace("/showdown/websocket", "/action.php?")
    return ServerConfiguration(ws, http)


OPPONENTS = [
    (SimpleHeuristicsPlayer, "SH"),
    (SmartDamagePlayer, "SmartDmg"),
    (TacticalPlayer, "Tactical"),
    (StrategicPlayer, "Strategic"),
]


async def eval_ckpt_vs_bot(ckpt_label, ckpt_path, cached, device,
                            opp_cls, opp_name, n_games, concurrency, server, tb):
    """Play n_games for one checkpoint vs one heuristic bot, sampling teams
    from `tb` on each side. Returns (wins, losses, ties, wr_pct)."""
    # Arch dispatch: pick the right BattleAgent class for this ckpt.
    AgentClass = BattleAgentTransformer if is_transformer_checkpoint(cached) else BattleAgent
    p1 = AgentClass(
        ckpt_path, device=device, _cached_ckpt=cached,
        account_configuration=AccountConfiguration.generate(f"E{ckpt_label[:6]}", rand=True),
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
        team=tb,  # same teambuilder — both sides draw from the 16 teams
    )

    try:
        await asyncio.wait_for(
            p1.battle_against(p2, n_battles=n_games),
            timeout=max(180, n_games * 30),
        )
    except asyncio.TimeoutError:
        print(f"  [WARN] Timeout: {ckpt_label} vs {opp_name}", flush=True)
    except Exception as e:
        print(f"  [ERROR] {ckpt_label} vs {opp_name}: {e}", flush=True)

    w = p1.n_won_battles
    l = p1.n_lost_battles
    t = p1.n_tied_battles
    total = w + l + t
    wr = 100 * w / max(1, total)

    try: p1.reset_battles()
    except Exception: pass
    try: p2.reset_battles()
    except Exception: pass
    del p1, p2

    return w, l, t, wr


async def eval_ckpt(ckpt_label, ckpt_path, cached, device, n_games, concurrency, server, tb):
    """Run all 4 smart-bot matchups for one checkpoint. Returns dict of per-bot results."""
    results = {}
    for opp_cls, opp_name in OPPONENTS:
        t0 = time.time()
        w, l, t, wr = await eval_ckpt_vs_bot(
            ckpt_label, ckpt_path, cached, device,
            opp_cls, opp_name, n_games, concurrency, server, tb,
        )
        dt = time.time() - t0
        results[opp_name] = {"wins": w, "losses": l, "ties": t, "wr_pct": wr}
        print(f"  {ckpt_label} vs {opp_name:>10}: {w:>3}/{w+l+t:<3} ({wr:>5.1f}%) [{dt:.0f}s]",
              flush=True)
    smart_avg = sum(r["wr_pct"] for r in results.values()) / len(results)
    results["smart_avg"] = smart_avg
    return results


def parse_ckpt_arg(s: str):
    """Format: 'label=path' or just 'path' (label = stem)."""
    if "=" in s:
        label, path = s.split("=", 1)
    else:
        label = Path(s).stem
        path = s
    return label.strip(), path.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="One or more 'label=path' specs (or just paths). Each runs vs all 4 bots.")
    p.add_argument("--servers", default="9000",
                   help="Comma-separated battle_server ports (we use 1 per matchup, sequential).")
    p.add_argument("--n-games", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=100,
                   help="poke-env max_concurrent_battles (both sides). vs-smart-bot "
                        "eval is CPU-bound on the bot side; GPU sees serialized forwards. "
                        "100 is healthy at 20M params. Bump to 200 if no slack measured. "
                        "Drop to 70 on smaller VRAM if forward queue saturates.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-json", default="data/eval/metamon_competitive_eval.json")
    p.add_argument("--team-dir", default=str(METAMON_COMPETITIVE_DIR),
                   help="Metamon competitive team dir (default = metamon_cache).")
    args = p.parse_args()

    server = make_server(args.servers.split(",")[0])
    tb = MetamonCompetitiveTeambuilder(Path(args.team_dir))

    ckpt_specs = [parse_ckpt_arg(c) for c in args.checkpoints]
    print(f"[eval] {len(ckpt_specs)} checkpoints × {len(OPPONENTS)} bots × {args.n_games} games "
          f"= {len(ckpt_specs) * len(OPPONENTS) * args.n_games} total games", flush=True)
    print(f"[eval] team source: {args.team_dir} ({len(tb.teams)} teams)", flush=True)
    print()

    all_results = {}
    t_start = time.time()
    for label, path in ckpt_specs:
        print(f"=== {label} ({path}) ===", flush=True)
        if not Path(path).exists():
            print(f"  [SKIP] not found", flush=True)
            continue
        # Load checkpoint once per label (avoids reload-per-bot leak).
        cached = torch.load(path, map_location=torch.device(args.device), weights_only=False)
        n_params = sum(x.numel() for x in cached.get("model_state_dict", {}).values()
                       if hasattr(x, "numel"))
        print(f"  loaded ({n_params:,} params)", flush=True)
        t0 = time.time()
        per_bot = asyncio.run(eval_ckpt(
            label, path, cached, args.device, args.n_games, args.concurrency, server, tb,
        ))
        dt = time.time() - t0
        per_bot["__time_s"] = dt
        all_results[label] = {"path": path, **per_bot}
        print(f"  {label} smart_avg = {per_bot['smart_avg']:.1f}%  ({dt:.0f}s)", flush=True)
        print()

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    out = {
        "team_source": args.team_dir,
        "n_teams": len(tb.teams),
        "n_games_per_matchup": args.n_games,
        "concurrency": args.concurrency,
        "results": all_results,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] saved to {args.out_json}", flush=True)

    # Summary
    print()
    print("=" * 70)
    print(f"{'label':<14}  {'SH':>5}  {'SmD':>5}  {'Tac':>5}  {'Str':>5}  {'savg':>6}")
    print("-" * 70)
    for label, r in all_results.items():
        if "smart_avg" not in r:
            continue
        print(f"{label:<14}  "
              f"{r['SH']['wr_pct']:>4.0f}%  "
              f"{r['SmartDmg']['wr_pct']:>4.0f}%  "
              f"{r['Tactical']['wr_pct']:>4.0f}%  "
              f"{r['Strategic']['wr_pct']:>4.0f}%  "
              f"{r['smart_avg']:>5.1f}%")
    print()
    print(f"Total wall-clock: {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
