#!/usr/bin/env python
"""Extract eval metrics from all TensorBoard logs into a single CSV.

Usage:
    python extract_eval_csv.py [--out eval_history.csv] [--plot]

Output CSV columns:
    iter, SH, SmartDmg, Tactical, Strategic, smart_avg, win_rate, entropy, pi_loss, v_loss, kl

Also optionally generates a plot (eval_history.png) with per-bot trends.
"""
import argparse
import csv
import glob
import os
import sys

def extract_all(tb_dirs, tags):
    """Extract scalar data from multiple TB dirs, merge by step (iter)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    all_data = {}  # iter -> {tag: value}

    for tb_dir in sorted(tb_dirs):
        ea = EventAccumulator(tb_dir)
        ea.Reload()
        available = ea.Tags().get("scalars", [])

        for tag in tags:
            if tag not in available:
                continue
            for event in ea.Scalars(tag):
                step = event.step
                if step not in all_data:
                    all_data[step] = {}
                # Use short name for CSV column
                col = tag.split("/")[-1]
                all_data[step][col] = event.value

    return all_data


def main():
    parser = argparse.ArgumentParser(description="Extract eval metrics to CSV")
    parser.add_argument("--tb-glob", default="data/models/rl_v9/selfplay_v9_*/tb",
                        help="Glob pattern for TensorBoard dirs")
    parser.add_argument("--out", default="data/eval/eval_history.csv",
                        help="Output CSV path")
    parser.add_argument("--plot", action="store_true",
                        help="Generate eval_history.png plot")
    args = parser.parse_args()

    tb_dirs = sorted(glob.glob(args.tb_glob))
    if not tb_dirs:
        print(f"No TB dirs found matching {args.tb_glob}")
        sys.exit(1)
    print(f"Found {len(tb_dirs)} TensorBoard dirs")

    tags = [
        "eval/SH", "eval/SmartDmg", "eval/Tactical", "eval/Strategic", "eval/smart_avg",
        "train/win_rate", "train/entropy", "train/pi_loss", "train/v_loss", "train/kl",
    ]

    all_data = extract_all(tb_dirs, tags)
    print(f"Extracted {len(all_data)} unique iterations")

    # Filter to iters that have at least one eval metric
    eval_iters = {k: v for k, v in all_data.items()
                  if any(col in v for col in ["SH", "SmartDmg", "Tactical", "Strategic", "smart_avg"])}
    print(f"  {len(eval_iters)} iterations with eval data")

    # Write CSV
    columns = ["iter", "SH", "SmartDmg", "Tactical", "Strategic", "smart_avg",
               "win_rate", "entropy", "pi_loss", "v_loss", "kl"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for step in sorted(eval_iters.keys()):
            row = [step]
            for col in columns[1:]:
                val = eval_iters[step].get(col, "")
                if isinstance(val, float):
                    row.append(f"{val:.4f}")
                else:
                    row.append(val)
            writer.writerow(row)

    print(f"Wrote {len(eval_iters)} rows to {args.out}")

    # Also write full training metrics (every iter, not just eval)
    full_out = args.out.replace(".csv", "_full.csv")
    full_columns = ["iter", "win_rate", "entropy", "pi_loss", "v_loss", "kl"]
    train_iters = {k: v for k, v in all_data.items()
                   if any(col in v for col in ["win_rate", "entropy"])}
    with open(full_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(full_columns)
        for step in sorted(train_iters.keys()):
            row = [step]
            for col in full_columns[1:]:
                val = train_iters[step].get(col, "")
                if isinstance(val, float):
                    row.append(f"{val:.4f}")
                else:
                    row.append(val)
            writer.writerow(row)
    print(f"Wrote {len(train_iters)} rows to {full_out}")

    if args.plot:
        _plot(args.out)


def _plot(csv_path):
    """Generate eval trend plot from CSV."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    iters, sh, sd, tac, strat, savg = [], [], [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["SH"]:
                continue
            iters.append(int(row["iter"]))
            sh.append(float(row["SH"]))
            sd.append(float(row["SmartDmg"]))
            tac.append(float(row["Tactical"]))
            strat.append(float(row["Strategic"]))
            savg.append(float(row["smart_avg"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Per-bot win rates
    ax1.plot(iters, sh, label="SH", alpha=0.7)
    ax1.plot(iters, sd, label="SmartDmg", alpha=0.7)
    ax1.plot(iters, tac, label="Tactical", alpha=0.7)
    ax1.plot(iters, strat, label="Strategic", alpha=0.7)
    ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50%")
    ax1.set_ylabel("Win Rate %")
    ax1.set_title("Per-Bot Win Rate Over Training")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # Smart average with rolling mean
    ax2.plot(iters, savg, label="smart_avg (raw)", alpha=0.4, color="blue")
    # Rolling average over 5 evals
    if len(savg) >= 5:
        import numpy as np
        kernel = np.ones(5) / 5
        rolling = np.convolve(savg, kernel, mode="valid")
        rolling_iters = iters[2:2 + len(rolling)]
        ax2.plot(rolling_iters, rolling, label="smart_avg (5-eval rolling)", color="blue", linewidth=2)
    ax2.axhline(y=50, color="gray", linestyle="--", alpha=0.5)

    # Mark Exp 1 start
    ax2.axvline(x=1785, color="red", linestyle="--", alpha=0.7, label="Exp 1 start (lam=0.95)")
    ax1.axvline(x=1785, color="red", linestyle="--", alpha=0.7, label="Exp 1 start")

    ax2.set_ylabel("Smart Avg %")
    ax2.set_xlabel("Iteration")
    ax2.set_title("Smart Average Over Training")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    png_path = csv_path.replace(".csv", ".png")
    plt.savefig(png_path, dpi=150)
    print(f"Plot saved to {png_path}")


if __name__ == "__main__":
    main()
