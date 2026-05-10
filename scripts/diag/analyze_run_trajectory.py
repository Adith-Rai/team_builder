#!/usr/bin/env python
"""Cloud-run trajectory analysis: high-level + deep playstyle metrics across
all eval points in a PPO run dir, dumped to CSV + summary table.

Wraps `analyze_eval.py` functions directly (does NOT shell out) so the
high-level + deep metrics land in ONE row per iter for trend diagnosis.

Reusable for any cloud run; defaults to the first one. Used initially
(S57) to investigate why pure self-play collapsed on the new transformer
arch (savg trajectory 69→60→53→49→35), given the historical context that
self-play with the same shape (procedural teams + perm augmentation, no
external opps) gave 45→65%+ gains on the OLD MLP arch.

Usage:
  python scripts/diag/analyze_run_trajectory.py
  python scripts/diag/analyze_run_trajectory.py data/cloud_runs/001_phase1_v3
  python scripts/diag/analyze_run_trajectory.py <run_dir> --by-opponent

Outputs:
  {run_dir}/trajectory.csv  — full metrics (one row per iter, optionally per opp)
  stdout: summary table with iter-to-iter deltas highlighted
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


def _setup():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(os.path.join(here, "..", "..",
                                        "pokemon-ai-starter", "pokemon-ai", "src"))
    if os.path.isdir(src):
        sys.path.insert(0, src)


_setup()
from analyze_eval import (  # noqa: E402
    _scan_iter_dirs, analyze_battles, compute_playstyle_profile,
    deep_analyze_battles, filter_by_opponent, load_replays,
)


# ----------------------------------------------------------------
# Metric extraction — flatten profile + deep into a single flat dict
# ----------------------------------------------------------------

# Order matters: defines CSV column order and summary table order.
# Each tuple: (csv_column, source, key, fmt) where:
#   source ∈ {"profile", "deep_calc", "stats"}
#   fmt    ∈ {"%", "f", "i"} for printing
_METRICS = [
    # ─── Outcome ───
    ("iter",                    None,        None,                       "i"),
    ("games",                   None,        None,                       "i"),
    ("wins",                    "stats",     "wins",                     "i"),
    ("losses",                  "stats",     "losses",                   "i"),
    ("ties",                    "stats",     "ties",                     "i"),
    ("win_rate",                "profile",   "win_rate",                 "%"),
    ("avg_turns",               "profile",   "avg_turns",                "f"),
    # ─── Playstyle (high-level — same as analyze_eval.py --iter-trajectory) ───
    ("attack_pct",              "profile",   "attack_pct",               "%"),
    ("setup_pct",               "profile",   "setup_pct",                "%"),
    ("pivot_pct",               "profile",   "pivot_pct",                "%"),
    ("hazard_pct",              "profile",   "hazard_pct",               "%"),
    ("status_pct",              "profile",   "status_pct",               "%"),
    ("recovery_pct",            "profile",   "recovery_pct",             "%"),
    ("protect_pct",             "profile",   "protect_pct",              "%"),
    ("switch_rate",             "profile",   "switch_rate",              "%"),
    ("voluntary_switch_rate",   "profile",   "voluntary_switch_rate",    "%"),
    # ─── Type / damage knowledge (THE forgetting signature watch) ───
    ("se_ratio",                "profile",   "se_ratio",                 "%"),
    ("immune_of_all",           "profile",   "immune_of_all",            "%"),
    ("ko_ratio",                "profile",   "ko_ratio",                 "f"),
    ("spam_streaks_per_game",   "profile",   "spam_streaks_per_game",    "f"),
    # ─── Deep: switch quality ───
    ("good_sw_pct",             "deep_calc", "good_sw_pct",              "%"),
    ("neutral_sw_pct",          "deep_calc", "neutral_sw_pct",           "%"),
    ("bad_sw_pct",              "deep_calc", "bad_sw_pct",               "%"),
    # ─── Deep: momentum ───
    ("our_momentum_runs",       "deep",      "our_momentum_runs",        "i"),
    ("opp_momentum_runs",       "deep",      "opp_momentum_runs",        "i"),
    ("momentum_dominance",      "deep_calc", "momentum_dominance",       "%"),
    ("our_avg_run_len",         "deep_calc", "our_avg_run_len",          "f"),
    ("opp_avg_run_len",         "deep_calc", "opp_avg_run_len",          "f"),
    # ─── Deep: move waste ───
    ("waste_rate",              "deep_calc", "waste_rate",               "%"),
    ("wasted_moves",            "deep",      "wasted_moves",             "i"),
    ("total_attacking",         "deep",      "total_attacking_moves",    "i"),
    # ─── Deep: HP management ───
    ("avg_switchout_hp",        "deep_calc", "avg_switchout_hp",         "f"),
    ("switchout_low_pct",       "deep_calc", "switchout_low_pct",        "%"),
    ("switchout_mid_pct",       "deep_calc", "switchout_mid_pct",        "%"),
    ("switchout_high_pct",      "deep_calc", "switchout_high_pct",       "%"),
    # ─── Deep: turn efficiency ───
    ("kos_per_turn_wins",       "deep_calc", "kos_per_turn_wins",        "f"),
    ("kos_per_turn_losses",     "deep_calc", "kos_per_turn_losses",      "f"),
    # ─── Deep: lead performance ───
    ("lead_stay_wr",            "deep_calc", "lead_stay_wr",             "%"),
    ("lead_switch_wr",          "deep_calc", "lead_switch_wr",           "%"),
    # ─── Deep: endgame ───
    ("endgame_wr",              "deep_calc", "endgame_wr",               "%"),
    ("endgame_count",           "deep",      "endgame_total",            "i"),
    # ─── Deep: prediction ───
    ("switch_prediction_pct",   "deep_calc", "switch_prediction_pct",    "%"),
    # ─── Deep: recovery ───
    ("recovery_useful_pct",     "deep_calc", "recovery_useful_pct",      "%"),
]


def _safe_div(n, d, default=0.0):
    return n / d if d else default


def _compute_derived(deep: dict) -> dict:
    """Derive the deep_calc metrics from raw deep counters."""
    total_vol = deep['good_switches'] + deep['bad_switches'] + deep['neutral_switches']
    total_swings = deep['our_momentum_runs'] + deep['opp_momentum_runs']
    n_switchouts = len(deep.get('switchout_hps', []) or [])
    total_rec = deep.get('recovery_useful', 0) + deep.get('recovery_wasted', 0)

    if n_switchouts:
        avg_hp = sum(deep['switchout_hps']) / n_switchouts
        low = sum(1 for h in deep['switchout_hps'] if h < 33) / n_switchouts
        mid = sum(1 for h in deep['switchout_hps'] if 33 <= h < 66) / n_switchouts
        high = sum(1 for h in deep['switchout_hps'] if h >= 66) / n_switchouts
    else:
        avg_hp = low = mid = high = 0.0

    return {
        "good_sw_pct":          _safe_div(deep['good_switches'], total_vol),
        "neutral_sw_pct":       _safe_div(deep['neutral_switches'], total_vol),
        "bad_sw_pct":           _safe_div(deep['bad_switches'], total_vol),
        "momentum_dominance":   _safe_div(deep['our_momentum_runs'], total_swings),
        "our_avg_run_len":      _safe_div(deep['our_momentum_total_len'],
                                          deep['our_momentum_runs']),
        "opp_avg_run_len":      _safe_div(deep['opp_momentum_total_len'],
                                          deep['opp_momentum_runs']),
        "waste_rate":           _safe_div(deep['wasted_moves'],
                                          deep['total_attacking_moves']),
        "avg_switchout_hp":     avg_hp,
        "switchout_low_pct":    low,
        "switchout_mid_pct":    mid,
        "switchout_high_pct":   high,
        "kos_per_turn_wins":    _safe_div(deep['kos_in_wins'], deep['turns_in_wins']),
        "kos_per_turn_losses":  _safe_div(deep['kos_in_losses'], deep['turns_in_losses']),
        "lead_stay_wr":         _safe_div(deep['lead_stay_wins'], deep['lead_stay_total']),
        "lead_switch_wr":       _safe_div(deep['lead_switch_wins'], deep['lead_switch_total']),
        "endgame_wr":           _safe_div(deep['endgame_wins'], deep['endgame_total']),
        "switch_prediction_pct": _safe_div(deep.get('predicted_switches', 0),
                                           deep.get('opp_switches_total', 0)),
        "recovery_useful_pct":  _safe_div(deep.get('recovery_useful', 0), total_rec),
    }


def _extract_row(it: int, n_games: int, stats: dict, profile: dict,
                 deep: dict) -> dict:
    """Flatten profile + deep + derived into one row keyed by csv-column name."""
    derived = _compute_derived(deep)
    row = {"iter": it, "games": n_games}
    for col, source, key, _fmt in _METRICS:
        if source is None:
            continue  # iter / games already set
        if source == "profile":
            row[col] = profile.get(key)
        elif source == "stats":
            row[col] = stats.get(key)
        elif source == "deep":
            row[col] = deep.get(key)
        elif source == "deep_calc":
            row[col] = derived.get(key)
    return row


def _fmt(val, kind: str) -> str:
    if val is None:
        return "    -"
    if kind == "%":
        return f"{val * 100:6.1f}%"
    if kind == "i":
        return f"{int(val):>6d}"
    return f"{val:6.2f}"


def _fmt_delta(prev, cur, kind: str) -> str:
    if prev is None or cur is None:
        return "      "
    delta = cur - prev
    if kind == "%":
        return f"{delta * 100:+5.1f}"
    if kind == "i":
        return f"{int(delta):+5d}"
    return f"{delta:+5.2f}"


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def _process_iter(it: int, dpath: Path, our_prefix: str,
                  by_opponent: bool, rows_out: list):
    battles = load_replays(str(dpath))
    if not battles:
        return
    if by_opponent:
        grouped = filter_by_opponent(battles, our_prefix=our_prefix)
        for opp_name in sorted(grouped.keys()):
            opp_battles = grouped[opp_name]
            stats = analyze_battles(opp_battles, our_player_prefix=our_prefix)
            profile = compute_playstyle_profile(stats)
            deep = deep_analyze_battles(opp_battles, our_player_prefix=our_prefix)
            row = _extract_row(it, len(opp_battles), stats, profile, deep)
            row["opp"] = opp_name
            rows_out.append(row)
    else:
        stats = analyze_battles(battles, our_player_prefix=our_prefix)
        profile = compute_playstyle_profile(stats)
        deep = deep_analyze_battles(battles, our_player_prefix=our_prefix)
        row = _extract_row(it, len(battles), stats, profile, deep)
        rows_out.append(row)


def _print_summary(rows: list, by_opponent: bool):
    """Print compact iter-by-iter table with per-row deltas vs prior."""
    if not rows:
        print("[WARN] no rows to summarize")
        return

    # Pick a curated subset for the screen view (full data in CSV)
    keys_for_screen = [
        ("iter",                  "iter",   "i"),
        ("games",                 "games",  "i"),
        ("win_rate",              "WR",     "%"),
        ("se_ratio",              "SE",     "%"),
        ("immune_of_all",         "Imm",    "%"),
        ("waste_rate",            "Wst",    "%"),
        ("ko_ratio",              "KO",     "f"),
        ("hazard_pct",            "Haz",    "%"),
        ("setup_pct",             "Set",    "%"),
        ("voluntary_switch_rate", "vSw",    "%"),
        ("good_sw_pct",           "GdSw",   "%"),
        ("bad_sw_pct",            "BdSw",   "%"),
        ("momentum_dominance",    "Mom",    "%"),
        ("lead_stay_wr",          "LdStWR", "%"),
        ("endgame_wr",            "EndWR",  "%"),
    ]

    header = " ".join(f"{label:>7}" for _key, label, _ in keys_for_screen)
    if by_opponent:
        header = f"{'opp':>10} " + header
    print()
    print(header)
    print("-" * len(header))

    prev_by_opp = {}
    for row in rows:
        opp_key = row.get("opp", "<all>")
        prev = prev_by_opp.get(opp_key)
        line_vals = " ".join(_fmt(row.get(k), kind) for k, _, kind in keys_for_screen)
        line = (f"{opp_key[:10]:>10} " if by_opponent else "") + line_vals
        print(line)
        # Delta line (skip for first row of each opp series)
        if prev is not None:
            delta_vals = " ".join(_fmt_delta(prev.get(k), row.get(k), kind)
                                   for k, _, kind in keys_for_screen)
            delta_line = (" " * 11 if by_opponent else "") + delta_vals
            print(delta_line)
        prev_by_opp[opp_key] = row


def _write_csv(out_path: Path, rows: list, by_opponent: bool):
    if not rows:
        return
    cols = (["iter", "games"] + (["opp"] if by_opponent else [])
            + [c for c, _src, _k, _f in _METRICS if c not in ("iter", "games")])
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in rows:
            w.writerow([row.get(c, "") for c in cols])
    print(f"\nCSV written: {out_path}  ({len(rows)} rows × {len(cols)} cols)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?",
                   default="data/cloud_runs/001_phase1_v3",
                   help="PPO run dir containing replays_iter*/ subdirs "
                        "(default: data/cloud_runs/001_phase1_v3)")
    p.add_argument("--by-opponent", action="store_true",
                   help="Split each iter row by opponent bot")
    p.add_argument("--player-prefix", default="Eval",
                   help="Player name prefix to identify 'our' side. Default 'Eval' "
                        "matches the eval_diag.py player class (used by production "
                        "self-play eval, names like 'Eval z31xm'). Use 'BCPolicyPlayer' "
                        "for legacy BC eval, or 'p1'/'p2' for H2H replays.")
    p.add_argument("--csv", default=None,
                   help="Output CSV path (default: <run_dir>/trajectory.csv)")
    args = p.parse_args()

    run = Path(args.run_dir)
    if not run.exists():
        print(f"[ERROR] run dir does not exist: {run}", file=sys.stderr)
        sys.exit(1)

    iter_dirs = _scan_iter_dirs(str(run))
    if not iter_dirs:
        print(f"[ERROR] no replays_iter*/ subdirs under {run}", file=sys.stderr)
        sys.exit(1)

    print(f"Cloud-run trajectory analysis: {run}")
    print(f"  {len(iter_dirs)} eval iters: "
          f"{iter_dirs[0][0]}..{iter_dirs[-1][0]}")
    print(f"  player prefix: {args.player_prefix}")
    if args.by_opponent:
        print(f"  by-opponent: split per (iter, opp)")
    print()

    rows = []
    for it, dpath in iter_dirs:
        print(f"  processing iter {it}...", flush=True)
        _process_iter(it, dpath, args.player_prefix, args.by_opponent, rows)

    _print_summary(rows, args.by_opponent)

    csv_path = Path(args.csv) if args.csv else run / "trajectory.csv"
    _write_csv(csv_path, rows, args.by_opponent)

    print()
    print("Reading the table:")
    print("  - WR drop + SE/Imm/Wst stable → exploration drift (recoverable)")
    print("  - WR drop + SE/Imm/Wst worsening → real type-knowledge degradation")
    print("  - Bad-switch rate climbing → noisier decision making")
    print("  - Momentum dominance dropping → losing pressure exchange")
    print("  - Lead-stay WR + Endgame WR dropping → opening + closing both worse")
    print("  - Setup % dropping + Hazard % rising → playstyle shift to grindy")


if __name__ == "__main__":
    main()
