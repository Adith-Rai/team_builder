"""Parse a training_curated.log and trend per-opponent W/L over time.

Iter line format (in trainer log):
  [HH:MM:SS] Iter N: W/L/T=X/Y/Z (W%), ... vs=opp1=w/g opp2=w/g ... pool=K

Usage:
  python _analyze_run_log.py <log_path> [--bucket 10|20] [--start <iter>]

For each opponent, prints:
  - baseline: pooled win rate across all sampled iters (sum w / sum g)
  - per-bucket pooled WR: sum w / sum g within bucket (catches sample-size effects)
  - per-bucket mean / median of per-iter WRs (catches single-iter outliers vs sustained trend)

Also prints overall iter-line WR trend by bucket.
"""
import argparse
import re
import statistics as _stats
import sys
from collections import defaultdict


iter_re = re.compile(
    r"^\[\d+:\d+:\d+\] Iter (\d+): W/L/T=(\d+)/(\d+)/(\d+) \(([\d.]+)%\).*?vs=(.+?)(?:pool=(\d+))?$"
)
opp_re = re.compile(r"(\S+?)=(\d+)/(\d+)(?:\[\+\d+fft\])?")


def is_external(k):
    return any(p in k for p in ("mcts-", "mm-", "foulplay-"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log_path")
    p.add_argument("--bucket", type=int, default=10, choices=[5, 10, 20],
                   help="Bucket size in iters (default 10)")
    p.add_argument("--start", type=int, default=None,
                   help="Only include iters >= this number (default: all)")
    args = p.parse_args()

    per_opp = defaultdict(list)  # opp_key -> list of (iter, wins, games)
    overall = []  # (iter, total_wr, pool)

    with open(args.log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = iter_re.match(line.strip())
            if not m:
                continue
            it = int(m.group(1))
            if args.start is not None and it < args.start:
                continue
            total_wr = float(m.group(5))
            opp_str = m.group(6)
            pool = int(m.group(7)) if m.group(7) else 0
            overall.append((it, total_wr, pool))
            for om in opp_re.finditer(opp_str):
                key = om.group(1)
                w = int(om.group(2))
                g = int(om.group(3))
                per_opp[key].append((it, w, g))

    if not overall:
        print("No iter lines parsed.")
        sys.exit(1)

    print(f"\nParsed {len(overall)} iters (range {overall[0][0]}-{overall[-1][0]}) "
          f"from {args.log_path}")
    print(f"Bucket size: {args.bucket} iters")
    if args.start is not None:
        print(f"Filter: iters >= {args.start}")
    print()

    bucket_size = args.bucket
    max_iter = max(it for it, _, _ in overall)
    min_iter = min(it for it, _, _ in overall)
    bucket_starts = list(range((min_iter // bucket_size) * bucket_size,
                                max_iter + bucket_size, bucket_size))[:-1]

    def bucket_for(it):
        return (it // bucket_size) * bucket_size

    # === Per-opponent table: pooled, mean, median per bucket ===
    print("=" * 80)
    print("PER-OPPONENT WIN RATE (% w/g) BY BUCKET — pooled / mean-of-iters / median-of-iters")
    print("=" * 80)

    all_keys = sorted([k for k in per_opp if is_external(k)]) + sorted(
        [k for k in per_opp if not is_external(k)])

    # Header
    bucket_labels = [f"i{b}-{b+bucket_size-1}" for b in bucket_starts]
    print(f"\n{'opp':<20} | {'pooled':>7} | {'iters':>5} | "
          + " | ".join(f"{lab:>15}" for lab in bucket_labels))
    print("-" * (22 + 10 + 8 + len(bucket_starts) * 18))

    for k in all_keys:
        samples = per_opp[k]
        # Pooled baseline
        total_w = sum(w for _, w, _ in samples)
        total_g = sum(g for _, _, g in samples)
        baseline = (100 * total_w / total_g) if total_g > 0 else 0.0

        # Per-bucket aggregates
        bucket_pooled = {b: [0, 0] for b in bucket_starts}  # (sum_w, sum_g)
        bucket_iter_wrs = {b: [] for b in bucket_starts}    # list of per-iter WRs
        for it, w, g in samples:
            b = bucket_for(it)
            if b not in bucket_pooled:
                continue
            bucket_pooled[b][0] += w
            bucket_pooled[b][1] += g
            if g > 0:
                bucket_iter_wrs[b].append(100 * w / g)

        cells = []
        for b in bucket_starts:
            w, g = bucket_pooled[b]
            iter_wrs = bucket_iter_wrs[b]
            if g == 0:
                cells.append(f"{'-':>15}")
            else:
                pooled = 100 * w / g
                mean_wr = _stats.mean(iter_wrs)
                med_wr = _stats.median(iter_wrs)
                # Format: "p55/m54/d54" (pooled / mean / median)
                cells.append(f"{pooled:>4.0f}/{mean_wr:>4.0f}/{med_wr:>3.0f}({len(iter_wrs):>2})")

        print(f"{k:<20} | {baseline:>6.1f}% | {len(samples):>5} | " + " | ".join(cells))

    # === Overall iter-WR trend by bucket ===
    print("\n" + "=" * 80)
    print("OVERALL ITER-WR TREND BY BUCKET (per-iter WR, not pooled)")
    print("=" * 80)
    bucket_iter_wrs = {b: [] for b in bucket_starts}
    for it, wr, _ in overall:
        b = bucket_for(it)
        if b in bucket_iter_wrs:
            bucket_iter_wrs[b].append(wr)

    print(f"\n{'bucket':>12} | {'n_iters':>7} | {'mean':>6} | {'median':>7} | {'min':>5} | {'max':>5} | trend")
    print("-" * 70)
    for b in bucket_starts:
        wrs = bucket_iter_wrs[b]
        if not wrs:
            continue
        m = _stats.mean(wrs)
        md = _stats.median(wrs)
        bar = "#" * int(m / 2)
        print(f"i{b:>3}-{b+bucket_size-1:<3} | {len(wrs):>7} | "
              f"{m:>5.1f}% | {md:>6.1f}% | "
              f"{min(wrs):>4.0f}% | {max(wrs):>4.0f}% | {bar}")

    # === Headline: opponents with significant trend (mean drop > 5pt across run) ===
    print("\n" + "=" * 80)
    print("OPPONENTS WITH NOTEWORTHY TRENDS (first vs last bucket where sampled)")
    print("=" * 80)
    print(f"\n{'opp':<22} | {'first':>10} | {'last':>10} | {'delta':>7} | {'note'}")
    print("-" * 70)
    flagged = []
    for k in all_keys:
        bucket_means = {}
        bucket_pooled_dict = {}
        for it, w, g in per_opp[k]:
            b = bucket_for(it)
            if b not in bucket_starts: continue
            if g == 0: continue
            bucket_means.setdefault(b, []).append(100 * w / g)
            bucket_pooled_dict.setdefault(b, [0, 0])
            bucket_pooled_dict[b][0] += w
            bucket_pooled_dict[b][1] += g
        if len(bucket_pooled_dict) < 2:
            continue
        sorted_buckets = sorted(bucket_pooled_dict.keys())
        first_b = sorted_buckets[0]
        last_b = sorted_buckets[-1]
        first_w, first_g = bucket_pooled_dict[first_b]
        last_w, last_g = bucket_pooled_dict[last_b]
        first_wr = 100 * first_w / first_g
        last_wr = 100 * last_w / last_g
        delta = last_wr - first_wr
        if abs(delta) >= 5:
            tag = "DOWN" if delta < 0 else "UP"
            flagged.append((k, first_b, first_wr, first_g, last_b, last_wr, last_g, delta, tag))
    flagged.sort(key=lambda x: x[7])  # sort by delta (worst first)
    for (k, fb, fwr, fg, lb, lwr, lg, d, tag) in flagged:
        note = f"{tag}{abs(d):.0f}pt"
        print(f"{k:<22} | i{fb:>3}({fwr:>4.0f}%/{fg:>2}g) | "
              f"i{lb:>3}({lwr:>4.0f}%/{lg:>2}g) | {d:>+6.1f} | {note}")
    if not flagged:
        print("(no opponent showed >=5pt mean change first-to-last bucket)")


if __name__ == "__main__":
    main()
