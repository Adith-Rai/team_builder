"""Analyze self-play win rates by opponent snapshot era."""
import re
from collections import defaultdict

def era(sp_num):
    if sp_num < 400: return 'early(<400)'
    if sp_num < 600: return 'mid(400-599)'
    if sp_num < 700: return 'peak(600-699)'
    if sp_num < 920: return 'post-lr(700-919)'
    if sp_num < 1000: return 'accum1(920-999)'
    return 'current(1000+)'

ERA_ORDER = ['early(<400)', 'mid(400-599)', 'peak(600-699)', 'post-lr(700-919)', 'accum1(920-999)', 'current(1000+)']

def parse_log(path):
    results = []
    for line in open(path):
        m = re.match(r'Iter (\d+):.*\(([0-9.]+)%\).*vs=(.*?)pool=', line)
        if not m: continue
        it = int(m.group(1))
        wr = float(m.group(2))
        vs_str = m.group(3).strip()
        era_wins = defaultdict(int)
        era_games = defaultdict(int)
        per_opp = {}
        for match in re.finditer(r'sp(\d+)=(\d+)/(\d+)', vs_str):
            sp = int(match.group(1))
            w = int(match.group(2))
            g = int(match.group(3))
            e = era(sp)
            era_wins[e] += w
            era_games[e] += g
            per_opp[sp] = (w, g)
        era_wr = {}
        for e in ERA_ORDER:
            if era_games[e] > 0:
                era_wr[e] = (era_wins[e], era_games[e])
        results.append((it, wr, era_wr, per_opp))
    return results

def print_phase(label, results, lo, hi):
    ew = defaultdict(int)
    eg = defaultdict(int)
    total_w = total_g = 0
    count = 0
    for it, wr, era_wr, _ in results:
        if lo <= it <= hi:
            count += 1
            for e, (w, g) in era_wr.items():
                ew[e] += w
                eg[e] += g
                total_w += w
                total_g += g
    print(label + " (%d iters)" % count)
    for e in ERA_ORDER:
        if eg[e] > 0:
            pct = 100.0 * ew[e] / eg[e]
            print("  vs %-20s: %5.1f%% (%3d/%3d)" % (e, pct, ew[e], eg[e]))
    if total_g > 0:
        print("  %-22s: %5.1f%% (%3d/%3d)" % ("OVERALL", 100.0*total_w/total_g, total_w, total_g))
    print()

def print_per_iter_detail(results, lo, hi):
    """Print per-iter WR against new vs old opponents."""
    print("  Iter  Overall  vs_old(<700)  vs_recent(700+)  vs_current(1000+)")
    print("  " + "-" * 65)
    for it, wr, era_wr, per_opp in results:
        if lo <= it <= hi:
            # Old = <700, recent = 700-999, current = 1000+
            old_w = old_g = rec_w = rec_g = cur_w = cur_g = 0
            for sp, (w, g) in per_opp.items():
                if sp < 700:
                    old_w += w; old_g += g
                elif sp < 1000:
                    rec_w += w; rec_g += g
                else:
                    cur_w += w; cur_g += g
            old_pct = "%.0f%%" % (100*old_w/old_g) if old_g else "  -  "
            rec_pct = "%.0f%%" % (100*rec_w/rec_g) if rec_g else "  -  "
            cur_pct = "%.0f%%" % (100*cur_w/cur_g) if cur_g else "  -  "
            print("  %4d  %5.1f%%     %-12s  %-15s  %-s" % (it, wr, old_pct, rec_pct, cur_pct))

# ============================================
print("=" * 80)
print("  SELF-PLAY WR BY OPPONENT ERA")
print("=" * 80)

# Current run
print("\n--- CURRENT RUN (training_3srv_test.log) ---\n")
r1 = parse_log('training_3srv_test.log')
print_phase('Stable (1000-1019, ent~0.76)', r1, 1000, 1019)
print_phase('Exploring (1020-1039, ent~0.78)', r1, 1020, 1039)
print_phase('Peak explore (1040-1059, ent~0.86)', r1, 1040, 1059)
print_phase('Settling (1060-1082, ent~0.79)', r1, 1060, 1082)

print("\n--- Per-iter detail (current run) ---\n")
print_per_iter_detail(r1, 1000, 1082)

# Previous run
print("\n\n--- PREVIOUS RUN (training_accum1_test2.log) ---\n")
r2 = parse_log('training_accum1_test2.log')
print_phase('Early accum1 (920-950)', r2, 920, 950)
print_phase('Mid accum1 (950-980)', r2, 950, 980)
print_phase('Late accum1 (980-1001)', r2, 980, 1001)

print("\n--- Per-iter detail (accum1 run, last 20) ---\n")
print_per_iter_detail(r2, 982, 1001)

# 699 run
print("\n\n--- 699 RUN (training_699_preserved.log) ---\n")
r3 = parse_log('training_699_preserved.log')
print_phase('Just after lr-restart (700-740)', r3, 700, 740)
print_phase('Rebuilding (740-800)', r3, 740, 800)
print_phase('Mid rebuild (800-860)', r3, 800, 860)
print_phase('Late rebuild (860-920)', r3, 860, 920)

print("\n--- Per-iter detail (699 run, last 20) ---\n")
start_699 = max(700, r3[-1][0] - 19) if r3 else 700
print_per_iter_detail(r3, start_699, 9999)

# Cross-run comparison
print("\n\n" + "=" * 80)
print("  CROSS-RUN COMPARISON: WR vs OLD (<700) OPPONENTS")
print("=" * 80 + "\n")

for label, results in [("699 run", r3), ("accum1 run", r2), ("current run", r1)]:
    old_w = old_g = 0
    for it, wr, era_wr, per_opp in results:
        for sp, (w, g) in per_opp.items():
            if sp < 700:
                old_w += w; old_g += g
    if old_g > 0:
        print("%-15s vs old(<700): %.1f%% (%d/%d)" % (label, 100*old_w/old_g, old_w, old_g))

print()
for label, results in [("699 run", r3), ("accum1 run", r2), ("current run", r1)]:
    rec_w = rec_g = 0
    for it, wr, era_wr, per_opp in results:
        for sp, (w, g) in per_opp.items():
            if 700 <= sp < 1000:
                rec_w += w; rec_g += g
    if rec_g > 0:
        print("%-15s vs post-lr(700-999): %.1f%% (%d/%d)" % (label, 100*rec_w/rec_g, rec_w, rec_g))
