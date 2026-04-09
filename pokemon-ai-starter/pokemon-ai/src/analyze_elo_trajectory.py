#!/usr/bin/env python3
"""Analyze the Elo trajectory across training eras.

Loads an Elo ladder result JSON (produced by eval_elo_ladder.py) plus an era
config (data/eval/eras.json) and produces:
  1. Per-snapshot iter -> Elo table (text)
  2. ASCII trajectory bar plot (text)
  3. Per-era aggregate stats (mean, range, delta-vs-prev)
  4. matplotlib scatter plot with era boundary lines + per-era means + bot anchors (PNG)
  5. Optional CSV dump of the iter -> Elo mapping for further analysis

Usage:
  # Default: read elo_session33_EXTENDED_FINAL.json + eras.json, print + save plot
  python analyze_elo_trajectory.py

  # Custom paths
  python analyze_elo_trajectory.py --input data/eval/elo_NEW.json \
                                    --eras data/eval/eras.json \
                                    --out-png data/eval/trajectory.png \
                                    --out-csv data/eval/trajectory.csv

  # Interactive plot (requires display)
  python analyze_elo_trajectory.py --show

The eras.json file is editable. Add new eras or update boundaries as the project
progresses without touching this script.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

DEFAULT_INPUT = "data/eval/elo_session33_EXTENDED_FINAL.json"
DEFAULT_ERAS = "data/eval/eras.json"
DEFAULT_OUT_PNG = "data/eval/elo_trajectory.png"

# Bot players are anchors, not snapshots — exclude from the iter-axis trajectory
BOTS = {
    "Random", "MaxBasePower", "GreedySE", "HazardSense", "SwitchAwareEscape",
    "SetupThenSweep", "SH", "SmartDmg", "Tactical", "Strategic",
}


def get_iter(name: str):
    """Extract training iter from a snapshot name. Returns None for non-snapshots."""
    if name == "BC_base":
        return 0
    if name in BOTS:
        return None
    # Match the first 4-digit number in the name (snapshot_NNNN, dip_NNNN, etc.)
    m = re.search(r"(\d{4})", name)
    return int(m.group(1)) if m else None


def load_eras(path: str):
    """Load eras config from JSON."""
    p = Path(path)
    if not p.exists():
        print(f"WARN: eras config not found at {path}, using minimal defaults", file=sys.stderr)
        return [{"id": "?", "name": "Unknown", "iter_lo": 0, "iter_hi": 99999, "description": ""}]
    with open(p) as f:
        data = json.load(f)
    return data["eras"]


def era_for(it: int, eras):
    """Return the era dict containing this iter, or None."""
    for era in eras:
        if era["iter_lo"] <= it < era["iter_hi"]:
            return era
    return None


def build_snapshot_list(elos: dict, cis: dict, eras: list):
    """Build sorted snapshot list with era annotations."""
    snaps = []
    for name, elo in elos.items():
        it = get_iter(name)
        if it is None:
            continue
        ci = cis.get(name, {"lo95": elo, "hi95": elo})
        era = era_for(it, eras)
        snaps.append({
            "iter": it,
            "name": name,
            "elo": elo,
            "ci_lo": ci["lo95"],
            "ci_hi": ci["hi95"],
            "era_id": era["id"] if era else "?",
            "era_name": era["name"] if era else "(unknown)",
        })
    snaps.sort(key=lambda s: s["iter"])
    return snaps


def print_table(snaps):
    print(f"\n{'Iter':>6}  {'Elo':>5}  {'95% CI':>14}  {'Era':<6}  {'Description':<32}  Snapshot")
    print("-" * 100)
    for s in snaps:
        print(
            f"{s['iter']:>6}  {s['elo']:>5.0f}  "
            f"[{s['ci_lo']:>4.0f}, {s['ci_hi']:>4.0f}]  "
            f"{s['era_id']:<6}  {s['era_name']:<32}  {s['name']}"
        )


def print_ascii_plot(snaps, elo_min: int = 700, elo_max: int = 1050, width: int = 60):
    """Print an ASCII bar plot of iter -> Elo, sorted by iter."""
    print(f"\nASCII trajectory ({elo_min} ... {elo_max}, width={width}):")
    print("-" * (40 + width))
    for s in snaps:
        pct = max(0.0, min(1.0, (s["elo"] - elo_min) / (elo_max - elo_min)))
        bar = "#" * int(pct * width)
        print(f"  iter {s['iter']:>4d}  Elo {s['elo']:>4.0f}  {s['era_id']:<5}  {bar}")


def print_era_aggregates(snaps, eras):
    print(f"\n{'='*100}\nERA AGGREGATES (mean, range, delta vs prev era):\n{'='*100}")
    by_era = {}
    for s in snaps:
        by_era.setdefault(s["era_id"], []).append(s)

    prev_mean = None
    rows = []
    for era in eras:
        members = by_era.get(era["id"], [])
        if not members:
            continue
        n = len(members)
        mean = sum(m["elo"] for m in members) / n
        rlo = min(m["elo"] for m in members)
        rhi = max(m["elo"] for m in members)
        delta = (mean - prev_mean) if prev_mean is not None else None
        rows.append((era, n, mean, rlo, rhi, delta))
        prev_mean = mean

    for era, n, mean, rlo, rhi, delta in rows:
        delta_str = f"  vs prev: {delta:+6.1f} Elo" if delta is not None else "  (first era)"
        iter_lo = era["iter_lo"]
        iter_hi_display = era["iter_hi"] - 1 if era["iter_hi"] < 99999 else "..."
        print(
            f"  {era['id']:<6}  iters {iter_lo:>4d}-{iter_hi_display:<6}  n={n:>2d}  "
            f"mean={mean:>6.1f}  range={rlo:>4.0f}-{rhi:<4.0f}  {delta_str}"
        )
        print(f"          '{era['name']}'")
    return rows


def save_csv(snaps, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iter", "name", "elo", "ci_lo", "ci_hi", "era_id", "era_name"])
        w.writeheader()
        for s in snaps:
            w.writerow(s)
    print(f"\nSaved CSV: {path}")


def make_plot(snaps, eras, era_rows, elo_data, ci_data, out_png: str = None, show: bool = False):
    """Render the matplotlib trajectory plot.

    Shows: scatter of iter -> Elo with CI error bars, era background bands,
    era boundary vertical lines, per-era mean horizontal lines, top bot
    anchor lines (SH, Tactical), title, axis labels.
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(15, 8))

    if not snaps:
        print("No snapshots to plot.", file=sys.stderr)
        return

    max_iter = max(s["iter"] for s in snaps)
    plot_xmax = max_iter + 60

    # ---- Era background bands ----
    for era in eras:
        lo = era["iter_lo"]
        hi = era["iter_hi"] if era["iter_hi"] < 99999 else plot_xmax
        color = era.get("color", "#dddddd")
        ax.axvspan(lo, hi, alpha=0.30, color=color, zorder=0)

    # ---- Era boundary vertical lines + top labels ----
    # Get y-range estimates first
    elos_for_range = [s["elo"] for s in snaps] + [s["ci_hi"] for s in snaps] + [s["ci_lo"] for s in snaps]
    bot_elos = [elo_data[b] for b in BOTS if b in elo_data]
    elos_for_range.extend(bot_elos)
    y_min = min(elos_for_range) - 30
    y_max = max(elos_for_range) + 30
    ax.set_ylim(y_min, y_max)

    for era in eras[1:]:  # skip the first (no boundary at start)
        ax.axvline(era["iter_lo"], color="black", linestyle="--", alpha=0.4, linewidth=1)

    # Era labels at top
    for era in eras:
        lo = era["iter_lo"]
        hi = era["iter_hi"] if era["iter_hi"] < 99999 else plot_xmax
        center = (lo + hi) / 2
        ax.text(
            center, y_max - 5, era["id"],
            ha="center", va="top", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
        )

    # ---- Per-era mean horizontal lines ----
    for era, n, mean, rlo, rhi, delta in era_rows:
        lo = era["iter_lo"]
        hi = era["iter_hi"] if era["iter_hi"] < 99999 else plot_xmax
        ax.hlines(
            mean, lo, hi,
            colors="red", linestyles="-", linewidth=2, alpha=0.75,
            label="Era mean" if era["id"] == era_rows[0][0]["id"] else None,
        )
        # Mean label
        ax.text(
            (lo + hi) / 2, mean + 4, f"{mean:.0f}",
            ha="center", va="bottom", fontsize=9, color="red", fontweight="bold",
        )

    # ---- Top bot anchor horizontal lines ----
    anchor_bots = {"SH": "SH (anchor)", "Tactical": "Tactical (top bot)"}
    for bot_name, label in anchor_bots.items():
        if bot_name in elo_data:
            bot_elo = elo_data[bot_name]
            ax.axhline(bot_elo, color="green", linestyle=":", alpha=0.6, linewidth=1.5)
            ax.text(
                plot_xmax - 30, bot_elo + 1, label,
                ha="right", va="bottom", fontsize=9, color="green",
            )

    # ---- Snapshot scatter with CI error bars ----
    iters = [s["iter"] for s in snaps]
    elos = [s["elo"] for s in snaps]
    ci_los = [s["elo"] - s["ci_lo"] for s in snaps]
    ci_his = [s["ci_hi"] - s["elo"] for s in snaps]
    ax.errorbar(
        iters, elos, yerr=[ci_los, ci_his],
        fmt="o", color="#1f77b4", markersize=7, capsize=4,
        ecolor="#1f77b4", elinewidth=1.2, alpha=0.85,
        label="Snapshot Elo (with 95% CI)",
        zorder=3,
    )

    # ---- Trajectory line connecting points ----
    ax.plot(iters, elos, color="#1f77b4", linewidth=1.0, alpha=0.4, zorder=2)

    # ---- Cosmetics ----
    ax.set_xlabel("Training iteration", fontsize=12)
    ax.set_ylabel("Elo (anchored to SH=1000)", fontsize=12)
    ax.set_title(
        "PPO Training Trajectory: Elo vs Iter, by Era\n"
        f"({len(snaps)} snapshots, anchored to SH=1000, "
        f"data from {Path(DEFAULT_INPUT).name})",
        fontsize=13,
    )
    ax.set_xlim(-30, plot_xmax)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()

    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=130)
        print(f"\nSaved plot: {out_png}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Analyze Elo trajectory across training eras")
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help=f"Elo ladder JSON to analyze (default: {DEFAULT_INPUT})")
    p.add_argument("--eras", default=DEFAULT_ERAS,
                   help=f"Era config JSON (default: {DEFAULT_ERAS})")
    p.add_argument("--out-png", default=DEFAULT_OUT_PNG,
                   help=f"Output PNG path for matplotlib plot (default: {DEFAULT_OUT_PNG})")
    p.add_argument("--out-csv", default=None,
                   help="Optional CSV dump of the iter -> Elo table")
    p.add_argument("--show", action="store_true",
                   help="Show interactive matplotlib plot (requires display)")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip plot generation entirely")
    p.add_argument("--no-ascii", action="store_true",
                   help="Skip ASCII trajectory plot")
    args = p.parse_args()

    # Load Elo data
    if not Path(args.input).exists():
        print(f"ERROR: Elo input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    with open(args.input) as f:
        data = json.load(f)
    elos = data["elos"]
    cis = data["cis"]

    # Load eras
    eras = load_eras(args.eras)

    # Build snapshot list
    snaps = build_snapshot_list(elos, cis, eras)
    if not snaps:
        print("No snapshots found in the input data.", file=sys.stderr)
        sys.exit(1)

    # Print outputs
    print_table(snaps)
    if not args.no_ascii:
        print_ascii_plot(snaps)
    era_rows = print_era_aggregates(snaps, eras)

    # Save CSV
    if args.out_csv:
        save_csv(snaps, args.out_csv)

    # Plot
    if not args.no_plot:
        make_plot(snaps, eras, era_rows, elos, cis,
                  out_png=args.out_png, show=args.show)


if __name__ == "__main__":
    main()
