#!/usr/bin/env python
# eval_report_v8.py — Comprehensive evaluation report for v8 PokeTransformer checkpoints.
#
# Generates a full report including:
#   1. Win rates per bot with confidence intervals
#   2. Playstyle stats (action categories, type effectiveness, KO ratios)
#   3. Per-mon analysis (usage, moves, leads, KO contribution)
#   4. Type awareness metrics (STAB usage, SE exploitation, immune avoidance)
#   5. Team performance (win rate per team from pool)
#
# Usage:
#   python -u eval_report_v8.py --checkpoint path/to/model.pt --n-battles 200
#   python -u eval_report_v8.py --checkpoint path/to/model.pt --compare path/to/other.pt

from __future__ import annotations
import argparse
import asyncio
import gc
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.player.baselines import SimpleHeuristicsPlayer, MaxBasePowerPlayer
from poke_env.player.baselines import RandomPlayer as PokeRandomPlayer

from policy_smartbots import SmartDamagePlayer, TacticalPlayer, StrategicPlayer
from battle_agent import BattleAgent
from teams_ou import random_pool_teambuilder
from analyze_eval import parse_replay, analyze_battles

import glob


# =============================
# Stats helpers
# =============================

def binomial_ci(wins: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    z = 1.96 if confidence == 0.95 else 1.645  # z-score
    p = wins / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return max(0, center - spread), min(1, center + spread)


# =============================
# Eval runner
# =============================

SMART_BOTS = {
    "SH": SimpleHeuristicsPlayer,
    "SmartDmg": SmartDamagePlayer,
    "Tactical": TacticalPlayer,
    "Strategic": StrategicPlayer,
}

ALL_BOTS = {
    **SMART_BOTS,
    "Random": PokeRandomPlayer,
    "MaxBP": MaxBasePowerPlayer,
}


async def run_eval(checkpoint_path: str, device: str, server_url: str,
                   n_battles: int, replay_dir: str, bots: Dict[str, type],
                   concurrency: int = 10) -> Dict[str, Dict]:
    """Run eval against all bots, return per-bot results with replays."""
    SERVER = ServerConfiguration(server_url, None)
    results = {}

    for bot_name, bot_cls in bots.items():
        rd = os.path.join(replay_dir, bot_name)
        os.makedirs(rd, exist_ok=True)

        p1 = BattleAgent(
            checkpoint_path, device=device,
            account_configuration=AccountConfiguration.generate("EvalR", rand=True),
            battle_format="gen9ou", max_concurrent_battles=concurrency,
            server_configuration=SERVER, team=random_pool_teambuilder(),
            save_replays=rd,
        )
        p2 = bot_cls(
            account_configuration=AccountConfiguration.generate(bot_name, rand=True),
            battle_format="gen9ou", max_concurrent_battles=concurrency,
            server_configuration=SERVER, team=random_pool_teambuilder(),
        )

        await p1.battle_against(p2, n_battles=n_battles)
        wins = p1.n_won_battles
        results[bot_name] = {"wins": wins, "total": n_battles, "replay_dir": rd}
        p1.reset_battles(); p2.reset_battles()
        del p1, p2; gc.collect(); torch.cuda.empty_cache()
        print(f"  vs {bot_name:12s}: {wins}/{n_battles} = {wins/n_battles*100:.0f}%", flush=True)

    return results


# =============================
# Report generation
# =============================

def generate_report(eval_results: Dict[str, Dict], checkpoint_name: str) -> str:
    """Generate comprehensive text report from eval results + replay analysis."""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  COMPREHENSIVE EVAL REPORT: {checkpoint_name}")
    lines.append(f"{'='*70}\n")

    # ---- Section 1: Win Rates with CI ----
    lines.append("1. WIN RATES (95% confidence intervals)")
    lines.append("-" * 50)
    smart_wins, smart_total = 0, 0
    for bot_name in ["SH", "SmartDmg", "Tactical", "Strategic", "Random", "MaxBP"]:
        if bot_name not in eval_results:
            continue
        r = eval_results[bot_name]
        w, t = r["wins"], r["total"]
        wr = w / t * 100
        lo, hi = binomial_ci(w, t)
        lines.append(f"  vs {bot_name:12s}: {w:>3d}/{t} = {wr:5.1f}%  (CI: {lo*100:.1f}% - {hi*100:.1f}%)")
        if bot_name in SMART_BOTS:
            smart_wins += w
            smart_total += t

    if smart_total > 0:
        smart_avg = smart_wins / smart_total * 100
        lo, hi = binomial_ci(smart_wins, smart_total)
        lines.append(f"\n  Smart avg:      {smart_wins:>3d}/{smart_total} = {smart_avg:5.1f}%  (CI: {lo*100:.1f}% - {hi*100:.1f}%)")
    lines.append("")

    # ---- Sections 2-5: Per-bot playstyle analysis ----
    for bot_name in ["SH", "SmartDmg", "Tactical", "Strategic"]:
        if bot_name not in eval_results:
            continue
        rd = eval_results[bot_name].get("replay_dir", "")
        if not rd:
            continue

        files = sorted(glob.glob(os.path.join(rd, "*.html")))
        if not files:
            continue
        battles = [b for b in [parse_replay(f) for f in files] if b]
        if not battles:
            continue

        s = analyze_battles(battles, our_player_prefix="EvalR")
        t = s["our_total_actions"] or 1
        tb = s["total_battles"] or 1
        w = eval_results[bot_name]["wins"]
        ko = s["opp_faints"] / max(1, s["our_faints"])

        lines.append(f"2-4. vs {bot_name} ({w}/{tb} wins)")
        lines.append("-" * 50)

        # Action breakdown
        lines.append(f"  Actions: Atk={s['attacking_moves']/t*100:.0f}% Sw={s['our_switches']/t*100:.0f}% "
                     f"Haz={s['hazard_moves']/t*100:.0f}% Rec={s['recovery_moves']/t*100:.0f}% "
                     f"Pvt={s['pivot_moves']/t*100:.0f}% Set={s['setup_moves']/t*100:.0f}% "
                     f"Prot={s['protect_moves']/t*100:.0f}%")

        # Type effectiveness
        se_pct = s["our_se_moves"] / t * 100
        res_pct = s["our_resisted_moves"] / t * 100
        imm_pct = s["our_immune_moves"] / t * 100
        atk_total = s["attacking_moves"] or 1
        stab_pct = 0  # would need STAB tracking in analyze_eval
        lines.append(f"  Types: SE={se_pct:.0f}% Resisted={res_pct:.0f}% Immune={imm_pct:.0f}%")
        lines.append(f"  Combat: KO_ratio={ko:.2f} Turns/game={s['total_turns']/tb:.0f} "
                     f"VolSwitch={s['voluntary_switches']/t*100:.0f}%")

        # Top moves
        top_moves = s["our_moves"].most_common(8)
        lines.append(f"  Top moves: {' '.join(f'{m}({c})' for m,c in top_moves)}")

        # Top mons
        top_mons = s["our_pokemon_usage"].most_common(6)
        lines.append(f"  Top mons: {' '.join(f'{m}({c})' for m,c in top_mons)}")

        # Leads
        top_leads = s["our_lead"].most_common(5)
        if top_leads:
            lines.append(f"  Leads: {' '.join(f'{m}({c})' for m,c in top_leads)}")

        # Per-mon moves (top 4 mons)
        lines.append(f"  Per-mon breakdown:")
        for mon, moves in sorted(s["moves_per_mon"].items(),
                                  key=lambda x: -sum(x[1].values()))[:4]:
            top = moves.most_common(4)
            total_mon = sum(moves.values())
            move_str = " ".join(f"{m}({c})" for m, c in top)
            lines.append(f"    {mon:20s} ({total_mon:>3d} acts): {move_str}")

        # Type awareness: immune rate as indicator
        if imm_pct > 5:
            lines.append(f"  [!] High immune rate ({imm_pct:.0f}%) — type awareness needs work")
        elif imm_pct < 2:
            lines.append(f"  [+] Low immune rate ({imm_pct:.0f}%) — good type awareness")

        if res_pct > 20:
            lines.append(f"  [!] High resist rate ({res_pct:.0f}%) — hitting into resistances often")

        if ko > 1.0:
            lines.append(f"  [+] Positive KO ratio ({ko:.2f}) — winning the trade game")
        elif ko < 0.6:
            lines.append(f"  [!] Poor KO ratio ({ko:.2f}) — losing mons too fast")

        lines.append("")

    # ---- Section 6: Team Performance ----
    lines.append("5. TEAM PERFORMANCE (win rate by team drawn)")
    lines.append("-" * 50)

    # Aggregate across all bots
    team_wins = Counter()
    team_total = Counter()
    for bot_name, r in eval_results.items():
        rd = r.get("replay_dir", "")
        if not rd:
            continue
        files = sorted(glob.glob(os.path.join(rd, "*.html")))
        for f in files:
            b = parse_replay(f)
            if not b:
                continue
            our_key = "p1" if "EvalR" in (b.get("p1") or "") else "p2"
            team_mons = tuple(sorted(b.get(f"{our_key}_team", [])))
            won = b.get("winner") and "EvalR" in b.get("winner", "")
            team_total[team_mons] += 1
            if won:
                team_wins[team_mons] += 1

    if team_total:
        # Sort by games played
        sorted_teams = sorted(team_total.items(), key=lambda x: -x[1])
        lines.append(f"  {'Team (sorted by usage)':60s} {'W':>4s} {'T':>4s} {'WR':>6s}")
        for team, total in sorted_teams[:15]:
            wins = team_wins.get(team, 0)
            wr = wins / total * 100
            team_str = ", ".join(team)[:58]
            lines.append(f"  {team_str:60s} {wins:>4d} {total:>4d} {wr:5.0f}%")

        # Summary stats
        wrs = [team_wins.get(t, 0) / c * 100 for t, c in sorted_teams if c >= 3]
        if wrs:
            lines.append(f"\n  Teams with 3+ games: {len(wrs)}")
            lines.append(f"  Best team WR: {max(wrs):.0f}%  Worst: {min(wrs):.0f}%  "
                         f"Avg: {sum(wrs)/len(wrs):.0f}%  Std: {(sum((w-sum(wrs)/len(wrs))**2 for w in wrs)/len(wrs))**0.5:.0f}%")
    lines.append("")

    # ---- Summary ----
    lines.append("SUMMARY")
    lines.append("-" * 50)
    if smart_total > 0:
        lines.append(f"  Smart avg: {smart_avg:.1f}%")
    lines.append(f"  Strongest vs: {max(((bn, r['wins']/r['total']*100) for bn, r in eval_results.items() if bn in SMART_BOTS), key=lambda x: x[1])}")
    lines.append(f"  Weakest vs:  {min(((bn, r['wins']/r['total']*100) for bn, r in eval_results.items() if bn in SMART_BOTS), key=lambda x: x[1])}")
    lines.append("")

    return "\n".join(lines)


# =============================
# CLI
# =============================

def main():
    p = argparse.ArgumentParser(description="Comprehensive eval report for v8 checkpoints")
    p.add_argument("--checkpoint", required=True, nargs="+",
                   help="One or more checkpoint paths to evaluate")
    p.add_argument("--names", nargs="+", default=None,
                   help="Names for each checkpoint (default: filenames)")
    p.add_argument("--n-battles", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--server", default="ws://127.0.0.1:9000/showdown/websocket")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--smart-only", action="store_true",
                   help="Only eval vs 4 smart bots (skip Random, MaxBP)")
    p.add_argument("--output-dir", default="eval_reports")
    args = p.parse_args()

    names = args.names or [Path(c).stem for c in args.checkpoint]
    bots = SMART_BOTS if args.smart_only else ALL_BOTS

    os.makedirs(args.output_dir, exist_ok=True)

    for ckpt_path, name in zip(args.checkpoint, names):
        print(f"\n{'='*70}")
        print(f"Evaluating: {name} ({ckpt_path})")
        print(f"{'='*70}")

        replay_dir = os.path.join(args.output_dir, f"replays_{name}")

        results = asyncio.run(run_eval(
            ckpt_path, args.device, args.server,
            args.n_battles, replay_dir, bots, args.concurrency,
        ))

        report = generate_report(results, name)
        print(report)

        # Save report to file
        report_path = os.path.join(args.output_dir, f"report_{name}.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
