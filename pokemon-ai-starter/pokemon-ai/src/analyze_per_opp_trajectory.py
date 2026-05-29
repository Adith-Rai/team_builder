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

    # Per-opp summary
    print("\n=== Per-opp summary ===")
    print(f"{'opp':30s} {'first':>10s} {'last':>10s} {'delta':>8s} {'n_pts':>6s}")
    print("-" * 72)
    for opp in opps:
        per_iter = {}
        for it, wr, g, kind in sorted(records[opp]):
            existing = per_iter.get(it)
            if existing is None or (existing[2] == "ema" and kind == "per-iter"):
                per_iter[it] = (wr, g, kind)
        if not per_iter:
            continue
        items = sorted(per_iter.items())
        first_it, (first_wr, _, _) = items[0]
        last_it, (last_wr, _, _) = items[-1]
        delta = (last_wr - first_wr) * 100
        sign = "+" if delta >= 0 else ""
        print(f"{opp[:30]:30s} {first_wr*100:5.1f}@i{first_it:<3d} "
              f"{last_wr*100:5.1f}@i{last_it:<3d} {sign}{delta:+5.1f}pp {len(items):6d}")


if __name__ == "__main__":
    main()
