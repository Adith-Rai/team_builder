"""Quick one-off: parse a training_curated.log and trend per-opponent W/L per iter.

Iter line format (in trainer log):
  [HH:MM:SS] Iter N: W/L/T=X/Y/Z (W%), ... vs=opp1=w/g opp2=w/g ... pool=K
"""
import re
import sys
from collections import defaultdict

if len(sys.argv) != 2:
    print("usage: python _analyze_run_log.py <training_curated.log>")
    sys.exit(2)

iter_re = re.compile(
    r"^\[\d+:\d+:\d+\] Iter (\d+): W/L/T=(\d+)/(\d+)/(\d+) \(([\d.]+)%\).*?vs=(.+?)(?:pool=(\d+))?$"
)
opp_re = re.compile(r"(\S+?)=(\d+)/(\d+)(?:\[\+\d+fft\])?")

per_opp = defaultdict(list)  # opp_key -> list of (iter, wins, games)
overall = []  # (iter, total_wr, pool)

with open(sys.argv[1], "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = iter_re.match(line.strip())
        if not m:
            continue
        it = int(m.group(1))
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
    print("No iter lines parsed. Format may have changed.")
    sys.exit(1)

# Bucket into deciles (every 10 iters)
print(f"\nParsed {len(overall)} iters from {sys.argv[1]}\n")

buckets = list(range(0, max(it for it, _, _ in overall) + 10, 10))
print(f"{'opp':<22} | {'baseline':>8} | " + " | ".join(f"i{b:>2}-{b+9:<2}" for b in buckets[:-1]))
print("-" * (24 + 12 + len(buckets) * 9))

# Categorise opps: externals first, then self-play
def is_external(k):
    return any(p in k for p in ("mcts-", "mm-", "foulplay-"))

ext_keys = sorted(per_opp.keys(), key=lambda k: (not is_external(k), k))
sp_keys = [k for k in per_opp if k.startswith("sp")]

# Deduped: externals + self-play, both subsets sorted
all_keys = sorted([k for k in per_opp if is_external(k)]) + sorted(sp_keys)

for k in all_keys:
    samples = per_opp[k]
    # Build per-decile aggregate
    bucket_wr = {b: [0, 0] for b in buckets[:-1]}
    for it, w, g in samples:
        b = (it // 10) * 10
        if b in bucket_wr:
            bucket_wr[b][0] += w
            bucket_wr[b][1] += g
    # Baseline = all-iter average
    total_w = sum(w for _, w, _ in samples)
    total_g = sum(g for _, _, g in samples)
    baseline = (100 * total_w / total_g) if total_g > 0 else 0.0
    cells = []
    for b in buckets[:-1]:
        w, g = bucket_wr[b]
        if g == 0:
            cells.append("    -   ")
        else:
            cells.append(f"{100*w/g:>5.1f}%({g:>2})")
    print(f"{k:<22} | {baseline:>7.1f}% | " + " | ".join(cells))

# Final summary
print("\n[overall trend by decile]")
total_buckets = {b: [0, 0] for b in buckets[:-1]}
for it, wr, _ in overall:
    b = (it // 10) * 10
    if b in total_buckets:
        total_buckets[b][0] += wr
        total_buckets[b][1] += 1
for b in buckets[:-1]:
    s, n = total_buckets[b]
    avg = s / n if n > 0 else 0
    bar = "#" * int(avg / 2)
    print(f"  i{b:>2}-{b+9:<2}: {avg:>5.1f}% {bar}")
