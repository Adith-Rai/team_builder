#!/usr/bin/env python3
"""Per-opp WR trajectory analysis from train_rl logs and/or win_rates snapshots.

Two data sources, both supported (mixed input also works):

1) Train log with [PFSP-ITER] lines (added in train_rl.py post-S67 fix).
   Format per iter:
     [PFSP-ITER] iter=N per-opp: opp_name1=Xw/Yg opp_name2=Xw/Yg ...
   Gives EXACT per-iter wins/games per opp.

2) Directory of win_rates snapshots (e.g., from the wr_poller.sh polling daemon).
   Each file is win_rates.json content with format:
     {"opp_key": [ema_wins, eff_games], ...}
   Filename convention: wr_iter<N>_<HHMMSS>.json — iter parsed from name.
   Gives EMA-smoothed WR (last ~50 games window), one point per save.

Usage:
  # From log
  python analyze_per_opp_trajectory.py --log /tmp/phase2_ext_v2_dense.log

  # From snapshot dir
  python analyze_per_opp_trajectory.py --snap-dir path/to/win_rates_history/

  # Both (combined; log overrides snap when same iter)
  python analyze_per_opp_trajectory.py --log foo.log --snap-dir bar/

  # Filter to externals only
  python analyze_per_opp_trajectory.py --log foo.log --externals-only

Output: per-opp trajectory table sorted by iter, plus per-opp summary
(first/last/delta/trend).
"""
import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


PFSP_LINE = re.compile(r"\[PFSP-ITER\] iter=(\d+) per-opp:\s*(.*)$")
OPP_WG = re.compile(r"([^\s=]+)=(\d+)w/(\d+)g")
SNAP_FILE = re.compile(r"wr_iter(\d+)(?:_\d+)?\.json$")


def parse_log(path):
    """Yield (iter, opp_name, wins, games) tuples. wins+games are RAW per-iter."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = PFSP_LINE.search(line)
            if not m:
                continue
            it = int(m.group(1))
            rest = m.group(2)
            for opp_m in OPP_WG.finditer(rest):
                opp = opp_m.group(1)
                w = int(opp_m.group(2))
                g = int(opp_m.group(3))
                yield it, opp, w, g, "per-iter"


def parse_snap_dir(dir_path):
    """Yield (iter, opp_name, ema_wins, eff_games) from win_rates snapshots.
    Note: ema_wins/eff_games here are EMA-smoothed, not raw per-iter."""
    for fp in sorted(glob.glob(str(Path(dir_path) / "wr_iter*.json"))):
        m = SNAP_FILE.search(Path(fp).name)
        if not m:
            continue
        it = int(m.group(1))
        try:
            data = json.load(open(fp))
        except Exception as e:
            print(f"  skip {fp}: {e}", file=sys.stderr)
            continue
        for opp, val in data.items():
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                continue
            ema_w, eff_g = val
            short = opp.replace("\\", "/").split("/")[-1]
            yield it, short, ema_w, eff_g, "ema"


def is_external(opp):
    return opp.startswith("mm-") or opp.startswith("mcts-")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", help="Train log path to scan for [PFSP-ITER] lines")
    p.add_argument("--snap-dir", help="Dir of wr_iter*.json snapshots")
    p.add_argument("--externals-only", action="store_true",
                   help="Show only mm-* and mcts-* opps")
    p.add_argument("--opp", action="append",
                   help="Show only specific opp(s); repeatable")
    args = p.parse_args()

    if not args.log and not args.snap_dir:
        p.error("need --log or --snap-dir (or both)")

    # opp -> list of (iter, wr, games, kind)
    records = defaultdict(list)

    if args.log:
        for it, opp, w, g, kind in parse_log(args.log):
            if g <= 0:
                continue
            records[opp].append((it, w / g, g, kind))

    if args.snap_dir:
        for it, opp, ew, eg, kind in parse_snap_dir(args.snap_dir):
            if eg <= 0:
                continue
            records[opp].append((it, ew / eg, eg, kind))

    if not records:
        print("No data found.", file=sys.stderr)
        sys.exit(1)

    # Filter
    opps = sorted(records.keys())
    if args.externals_only:
        opps = [o for o in opps if is_external(o)]
    if args.opp:
        opps = [o for o in opps if o in args.opp]

    if not opps:
        print("No opps after filter.", file=sys.stderr)
        sys.exit(1)

    # Trajectory table
    print("\n=== Per-opp WR trajectory ===")
    all_iters = sorted({it for opp in opps for it, *_ in records[opp]})
    if not all_iters:
        print("(no iters)")
        return

    # Header
    header = f"{'opp':30s}" + " ".join(f"i{it:>3d}" for it in all_iters)
    print(header)
    print("-" * len(header))

    for opp in opps:
        # Dedup per iter (prefer "per-iter" over "ema" if both present for same iter)
        per_iter = {}
        for it, wr, g, kind in sorted(records[opp]):
            existing = per_iter.get(it)
            if existing is None:
                per_iter[it] = (wr, g, kind)
            elif existing[2] == "ema" and kind == "per-iter":
                per_iter[it] = (wr, g, kind)
        row = f"{opp[:30]:30s}"
        for it in all_iters:
            if it in per_iter:
                wr, g, kind = per_iter[it]
                tag = "*" if kind == "ema" else " "
                row += f" {wr*100:4.1f}{tag}"
            else:
                row += "  --  "
        print(row)
    print("\n(* = EMA-smoothed snapshot value; blank = per-iter exact WR)")

    # Per-opp summary with movement direction classification
    print("\n=== Per-opp summary ===")
    print(f"{'opp':28s} {'first':>9s} {'last':>9s} {'range':>7s} {'slope':>8s} "
          f"{'rev':>4s} {'n':>3s}  pattern")
    print("-" * 88)
    for opp in opps:
        per_iter = {}
        for it, wr, g, kind in sorted(records[opp]):
            existing = per_iter.get(it)
            if existing is None or (existing[2] == "ema" and kind == "per-iter"):
                per_iter[it] = (wr, g, kind)
        if not per_iter or len(per_iter) < 2:
            continue
        items = sorted(per_iter.items())
        iters = [it for it, _ in items]
        wrs = [wr for _, (wr, _, _) in items]  # 0..1
        wrs_pp = [w * 100 for w in wrs]

        first_wr = wrs_pp[0]
        last_wr = wrs_pp[-1]
        vmin, vmax = min(wrs_pp), max(wrs_pp)
        rng = vmax - vmin

        # Linear regression slope (pp per 10 iters)
        n = len(iters)
        mx = sum(iters) / n
        my = sum(wrs_pp) / n
        num = sum((iters[i] - mx) * (wrs_pp[i] - my) for i in range(n))
        den = sum((iters[i] - mx) ** 2 for i in range(n))
        slope_per_iter = num / den if den > 0 else 0.0
        slope_per_10 = slope_per_iter * 10
        span = iters[-1] - iters[0]
        net = slope_per_iter * span  # implied net change over span (pp)

        # Direction reversals
        diffs = [wrs_pp[i+1] - wrs_pp[i] for i in range(n - 1)]
        reversals = sum(
            1 for i in range(1, len(diffs))
            if (diffs[i] > 0 and diffs[i-1] < 0) or (diffs[i] < 0 and diffs[i-1] > 0)
        )

        # Classification
        # FLAT: range < 1pp
        # TRENDING_*: |net| > 1.5x range/2 (slope explains > 75% of swing)
        # OSCILLATING: reversals >= (n-1)/2  AND  |net| < range/4
        # MIXED: reversals high but with slope
        if rng < 1.0:
            pattern = "FLAT"
        elif abs(net) > 0.75 * rng:
            pattern = "TREND_UP" if net > 0 else "TREND_DOWN"
        elif reversals >= max(1, (n - 1) // 2) and abs(net) < rng / 3:
            pattern = "OSCILLATING"
        elif abs(net) > rng / 3:
            pattern = "DRIFT_UP" if net > 0 else "DRIFT_DOWN"
        else:
            pattern = "BOUNCING"

        sign = "+" if slope_per_10 >= 0 else ""
        print(f"{opp[:28]:28s} {first_wr:6.1f}%@i{iters[0]:<3d} "
              f"{last_wr:6.1f}%@i{iters[-1]:<3d} "
              f"{rng:5.1f}pp {sign}{slope_per_10:+5.2f}/10it "
              f"{reversals:4d} {n:3d}  {pattern}")
    print("\nLegend: range=max-min, slope=pp per 10 iters via linreg, rev=direction reversals.")
    print("Patterns: FLAT=<1pp range; TREND_*=slope dominates; DRIFT_*=slope present but noisy;")
    print("          OSCILLATING=many reversals + small net; BOUNCING=noisy without direction.")


if __name__ == "__main__":
    main()
