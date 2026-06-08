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


def load_records(log_path=None, snap_dir=None):
    """Build per-opp records: opp -> list of (iter, wr, games, kind).

    Helper extracted from main() so --compare can re-use the same parsing
    for two logs.
    """
    records = defaultdict(list)
    if log_path:
        for it, opp, w, g, kind in parse_log(log_path):
            if g <= 0:
                continue
            records[opp].append((it, w / g, g, kind))
    if snap_dir:
        for it, opp, ew, eg, kind in parse_snap_dir(snap_dir):
            if eg <= 0:
                continue
            records[opp].append((it, ew / eg, eg, kind))
    return records


def bucket_records(records, window):
    """Aggregate per-iter records into N-iter windows.

    For each opp, sum wins and games within each bucket, re-derive WR.
    The bucket "iter" is the bucket center (e.g., 4.5 for 0-9 window) so
    linear-regression slope calculations remain meaningful in iter units.

    Returns a new records dict with the same shape (opp -> list of
    (iter, wr, games, kind)) but with bucketed entries. EMA snapshot
    records are excluded from bucketing (each is a single point already).
    """
    if window <= 1:
        return records
    out = defaultdict(list)
    for opp, entries in records.items():
        # Reconstruct per-iter (wins, games) from (wr, games) — lossy but
        # only used for re-aggregation, original wr=w/g so w = wr*g (rounded).
        per_iter_w = defaultdict(int)
        per_iter_g = defaultdict(int)
        ema_passthrough = []
        for it, wr, g, kind in entries:
            if kind == "ema":
                ema_passthrough.append((it, wr, g, kind))
                continue
            per_iter_w[it] += int(round(wr * g))
            per_iter_g[it] += g
        # Bucket
        buckets = defaultdict(lambda: [0, 0])  # bucket_start -> [w, g]
        for it, g in per_iter_g.items():
            b = (it // window) * window
            buckets[b][0] += per_iter_w[it]
            buckets[b][1] += g
        for b_start, (w, g) in sorted(buckets.items()):
            if g <= 0:
                continue
            center = b_start + (window - 1) / 2
            out[opp].append((center, w / g, g, "per-iter"))
        out[opp].extend(ema_passthrough)
        out[opp].sort()
    return out


def filter_opps(records, externals_only=False, opp_filter=None):
    """Apply opp filters; returns the sorted list of opps to display."""
    opps = sorted(records.keys())
    if externals_only:
        opps = [o for o in opps if is_external(o)]
    if opp_filter:
        opps = [o for o in opps if o in opp_filter]
    return opps


def compute_summary(records, opps):
    """Return list of (opp, first_wr, last_wr, range_pp, slope_per_10,
    reversals, n, pattern, first_iter, last_iter) tuples — extracted from
    main() so --compare can use it for both logs.
    """
    out = []
    for opp in opps:
        per_iter = {}
        for it, wr, g, kind in sorted(records[opp]):
            existing = per_iter.get(it)
            if existing is None or (existing[2] == "ema" and kind == "per-iter"):
                per_iter[it] = (wr, g, kind)
        if not per_iter or len(per_iter) < 2:
            out.append((opp, None, None, 0.0, 0.0, 0, len(per_iter), "INSUFFICIENT", None, None))
            continue
        items = sorted(per_iter.items())
        iters = [it for it, _ in items]
        wrs_pp = [wr * 100 for _, (wr, _, _) in items]
        first_wr, last_wr = wrs_pp[0], wrs_pp[-1]
        vmin, vmax = min(wrs_pp), max(wrs_pp)
        rng = vmax - vmin
        n = len(iters)
        mx = sum(iters) / n
        my = sum(wrs_pp) / n
        num = sum((iters[i] - mx) * (wrs_pp[i] - my) for i in range(n))
        den = sum((iters[i] - mx) ** 2 for i in range(n))
        slope_per_iter = num / den if den > 0 else 0.0
        slope_per_10 = slope_per_iter * 10
        span = iters[-1] - iters[0]
        net = slope_per_iter * span
        diffs = [wrs_pp[i + 1] - wrs_pp[i] for i in range(n - 1)]
        reversals = sum(
            1 for i in range(1, len(diffs))
            if (diffs[i] > 0 and diffs[i - 1] < 0) or (diffs[i] < 0 and diffs[i - 1] > 0)
        )
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
        out.append((opp, first_wr, last_wr, rng, slope_per_10, reversals, n,
                    pattern, iters[0], iters[-1]))
    return out


def print_trajectory(records, opps, label=""):
    """Print the iter-by-iter trajectory table for one set of records."""
    all_iters = sorted({it for opp in opps for it, *_ in records[opp]})
    if not all_iters:
        print(f"\n=== Per-opp WR trajectory{' ' + label if label else ''} ===")
        print("(no iters)")
        return
    print(f"\n=== Per-opp WR trajectory{' ' + label if label else ''} ===")
    # If iters are floats (bucketed), format with one decimal
    is_float = any(isinstance(it, float) for it in all_iters)
    fmt_it = (lambda it: f"i{it:>5.1f}") if is_float else (lambda it: f"i{it:>3d}")
    header = f"{'opp':30s}" + " ".join(fmt_it(it) for it in all_iters)
    print(header)
    print("-" * len(header))
    for opp in opps:
        per_iter = {}
        for it, wr, g, kind in sorted(records[opp]):
            existing = per_iter.get(it)
            if existing is None or (existing[2] == "ema" and kind == "per-iter"):
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


def print_summary(summary_rows, label=""):
    print(f"\n=== Per-opp summary{' ' + label if label else ''} ===")
    print(f"{'opp':28s} {'first':>11s} {'last':>11s} {'range':>7s} {'slope':>8s} "
          f"{'rev':>4s} {'n':>3s}  pattern")
    print("-" * 90)
    for opp, first_wr, last_wr, rng, slope10, reversals, n, pattern, i0, i1 in summary_rows:
        if first_wr is None:
            print(f"{opp[:28]:28s}   (insufficient data, n={n})")
            continue
        i0_s = f"i{i0:.1f}" if isinstance(i0, float) else f"i{i0}"
        i1_s = f"i{i1:.1f}" if isinstance(i1, float) else f"i{i1}"
        sign = "+" if slope10 >= 0 else ""
        print(f"{opp[:28]:28s} {first_wr:5.1f}%@{i0_s:>5s} "
              f"{last_wr:5.1f}%@{i1_s:>5s} "
              f"{rng:5.1f}pp {sign}{slope10:+5.2f}/10it "
              f"{reversals:4d} {n:3d}  {pattern}")


def print_compare(summary_a, summary_b, label_a, label_b):
    """Side-by-side summary table for two runs."""
    print(f"\n=== Per-opp slope comparison: [A]={label_a}  [B]={label_b} ===")
    by_opp_a = {row[0]: row for row in summary_a}
    by_opp_b = {row[0]: row for row in summary_b}
    all_opps = sorted(set(by_opp_a) | set(by_opp_b))
    print(f"{'opp':28s} "
          f"{'slope[A]':>9s} {'pat[A]':>11s} "
          f"{'slope[B]':>9s} {'pat[B]':>11s} "
          f"{'Δ slope':>8s}")
    print("-" * 82)
    for opp in all_opps:
        a = by_opp_a.get(opp)
        b = by_opp_b.get(opp)
        sa = a[4] if a and a[1] is not None else None
        sb = b[4] if b and b[1] is not None else None
        pa = a[7] if a else "—"
        pb = b[7] if b else "—"
        sa_s = f"{sa:+5.2f}" if sa is not None else "  —  "
        sb_s = f"{sb:+5.2f}" if sb is not None else "  —  "
        d_s = f"{sa - sb:+5.2f}" if sa is not None and sb is not None else "  —  "
        print(f"{opp[:28]:28s} {sa_s:>9s} {pa[:11]:>11s} "
              f"{sb_s:>9s} {pb[:11]:>11s} {d_s:>8s}")
    print("\nΔ slope = A − B (pp/10it). Positive = A growing faster than B on this opp.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", help="Train log path to scan for [PFSP-ITER] lines")
    p.add_argument("--snap-dir", help="Dir of wr_iter*.json snapshots")
    p.add_argument("--externals-only", action="store_true",
                   help="Show only mm-* and mcts-* opps")
    p.add_argument("--opp", action="append",
                   help="Show only specific opp(s); repeatable")
    p.add_argument("--window", type=int, default=1,
                   help="Aggregate per-iter records into N-iter windows. "
                        "Bucket iter = window center (e.g., 4.5 for 0-9 window). "
                        "Slope still in pp/10it units. Default 1 (no bucketing).")
    p.add_argument("--compare",
                   help="Path to a SECOND log; show side-by-side slope "
                        "comparison instead of single-run summary. Filters "
                        "(--externals-only / --opp / --window) apply to both.")
    args = p.parse_args()

    if not args.log and not args.snap_dir:
        p.error("need --log or --snap-dir (or both)")
    if args.window < 1:
        p.error("--window must be >= 1")

    records = load_records(args.log, args.snap_dir)

    if not records:
        print("No data found.", file=sys.stderr)
        sys.exit(1)

    if args.window > 1:
        records = bucket_records(records, args.window)

    opps = filter_opps(records, args.externals_only, set(args.opp) if args.opp else None)
    if not opps:
        print("No opps after filter.", file=sys.stderr)
        sys.exit(1)

    label_a = ""
    if args.window > 1:
        label_a = f"(window={args.window})"

    if args.compare:
        # Second log
        records_b = load_records(args.compare, None)
        if args.window > 1:
            records_b = bucket_records(records_b, args.window)
        opps_b = filter_opps(records_b, args.externals_only,
                             set(args.opp) if args.opp else None)
        union_opps = sorted(set(opps) | set(opps_b))
        summary_a = compute_summary(records, union_opps)
        summary_b = compute_summary(records_b, union_opps)
        # Use basenames for display labels
        name_a = Path(args.log).name if args.log else "A"
        name_b = Path(args.compare).name
        if args.window > 1:
            name_a += f" w={args.window}"
            name_b += f" w={args.window}"
        # Per-run summaries first (for context)
        print_summary(summary_a, label=f"[A]={name_a}")
        print_summary(summary_b, label=f"[B]={name_b}")
        # Side-by-side
        print_compare(summary_a, summary_b, name_a, name_b)
    else:
        print_trajectory(records, opps, label=label_a)
        summary = compute_summary(records, opps)
        print_summary(summary, label=label_a)

    print("\nLegend: range=max-min, slope=pp per 10 iters via linreg, rev=direction reversals.")
    print("Patterns: FLAT=<1pp range; TREND_*=slope dominates; DRIFT_*=slope present but noisy;")
    print("          OSCILLATING=many reversals + small net; BOUNCING=noisy without direction.")


if __name__ == "__main__":
    main()
