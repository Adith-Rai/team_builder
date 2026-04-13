"""
Analyze PFSP timing: how often each opponent is seen, gaps between encounters,
and whether some opponents are oversampled while others are starved.
"""
import json, re, sys, os, glob
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

SRC = Path(__file__).parent


def parse_log(log_path):
    iters = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r'\] Iter\s+(\d+):', line)
            if not m:
                continue
            it = int(m.group(1))
            pairs = re.findall(r'(sp\d+)=(\d+)/(\d+)', line)
            opp_results = [(name, int(w), int(g)) for name, w, g in pairs]
            iters.append((it, opp_results))
    return iters


def main():
    log_path = SRC / "exp2_exp3_pfsp.log"
    iters = parse_log(log_path)
    n_iters = len(iters)
    first_iter = iters[0][0]
    last_iter = iters[-1][0]
    print(f"Parsed {n_iters} iters ({first_iter}-{last_iter})")

    # Build per-opponent encounter timeline
    encounters = defaultdict(list)  # name -> [(iter, wins, games), ...]
    for it, opp_results in iters:
        for name, w, g in opp_results:
            encounters[name].append((it, w, g))

    total_opps = len(encounters)

    # ── 1. Gap analysis: how long between encounters? ──
    print()
    print("=" * 70)
    print("SECTION 1: TIME BETWEEN ENCOUNTERS (iter gaps)")
    print("=" * 70)

    all_gaps = []
    opp_gap_stats = []
    for name, encs in encounters.items():
        if len(encs) < 2:
            # Only seen once — gap = "iters since last seen until now"
            gap_since_last = last_iter - encs[-1][0]
            opp_gap_stats.append((name, len(encs), gap_since_last, gap_since_last, gap_since_last))
            continue
        gaps = [encs[i+1][0] - encs[i][0] for i in range(len(encs)-1)]
        all_gaps.extend(gaps)
        avg_gap = sum(gaps) / len(gaps)
        # Also compute gap since last encounter to now
        gap_since_last = last_iter - encs[-1][0]
        opp_gap_stats.append((name, len(encs), avg_gap, max(gaps), gap_since_last))

    if all_gaps:
        all_gaps.sort()
        print(f"\nGap between consecutive encounters (all opponents, all pairs):")
        print(f"  N gaps: {len(all_gaps)}")
        print(f"  Mean: {sum(all_gaps)/len(all_gaps):.1f} iters")
        print(f"  Median: {all_gaps[len(all_gaps)//2]} iters")
        print(f"  Min: {min(all_gaps)}, Max: {max(all_gaps)}")
        print(f"  P25: {all_gaps[len(all_gaps)//4]}, P75: {all_gaps[3*len(all_gaps)//4]}, P90: {all_gaps[9*len(all_gaps)//10]}")

        # Distribution
        buckets = [0]*10
        for g in all_gaps:
            idx = min(9, g // 10)
            buckets[idx] += 1
        print(f"\n  Gap distribution:")
        for i, c in enumerate(buckets):
            lo = i * 10
            hi = str(lo + 9) if i < 9 else "+"
            bar = "#" * int(c / max(buckets) * 40)
            print(f"    {lo:3d}-{hi:>3s} iters: {c:4d} ({c/len(all_gaps)*100:4.1f}%) {bar}")

    # ── 2. Opponents not seen for a long time ──
    print()
    print("=" * 70)
    print("SECTION 2: STALE OPPONENTS (long time since last encounter)")
    print("=" * 70)

    opp_gap_stats.sort(key=lambda x: -x[4])  # sort by gap_since_last descending
    print(f"\nOpponents not seen for the longest (stale ratings):")
    print(f"  {'Opponent':>10s}  {'Encs':>4s}  {'Last seen':>10s}  {'Gap':>6s}  {'Cum WR':>7s}  {'PFSP wt':>8s}")
    for name, n_enc, avg_gap, max_gap, gap_since in opp_gap_stats[:25]:
        encs = encounters[name]
        cum_w = sum(w for _, w, _ in encs)
        cum_g = sum(g for _, _, g in encs)
        wr = cum_w / cum_g if cum_g > 0 else 0.5
        weight = (1 - wr) ** 2
        last_seen = encs[-1][0]
        print(f"  {name:>10s}  {n_enc:4d}  iter {last_seen:5d}  {gap_since:5d}  {wr:6.0%}  {weight:8.3f}")

    # How many haven't been seen in 50+ iters?
    stale_50 = sum(1 for *_, gap in opp_gap_stats if gap >= 50)
    stale_100 = sum(1 for *_, gap in opp_gap_stats if gap >= 100)
    stale_20 = sum(1 for *_, gap in opp_gap_stats if gap >= 20)
    print(f"\n  Not seen in 20+ iters: {stale_20}/{total_opps} ({stale_20/total_opps*100:.0f}%)")
    print(f"  Not seen in 50+ iters: {stale_50}/{total_opps} ({stale_50/total_opps*100:.0f}%)")
    print(f"  Not seen in 100+ iters: {stale_100}/{total_opps} ({stale_100/total_opps*100:.0f}%)")

    # What about pool members NEVER seen? (930 total pool - 475 encountered)
    never_seen = 930 - total_opps
    print(f"  NEVER seen (not in log at all): {never_seen}/{930} ({never_seen/930*100:.0f}%)")

    # ── 3. Oversampled opponents ──
    print()
    print("=" * 70)
    print("SECTION 3: OVERSAMPLED OPPONENTS")
    print("=" * 70)

    opp_by_freq = [(len(encs), name, encs) for name, encs in encounters.items()]
    opp_by_freq.sort(reverse=True)

    expected_per_opp = n_iters * 15 / 930  # with 15 opponents per iter from 930 pool
    print(f"\n  Expected encounters per opponent (uniform): {expected_per_opp:.1f}")
    print(f"  Actual mean: {sum(len(encs) for encs in encounters.values()) / total_opps:.1f}")

    print(f"\n  Most oversampled (>2x expected):")
    print(f"  {'Opponent':>10s}  {'Encs':>4s}  {'Ratio':>6s}  {'Cum WR':>7s}  {'Recent WR':>10s}  {'PFSP wt':>8s}")
    for n_enc, name, encs in opp_by_freq[:20]:
        ratio = n_enc / expected_per_opp
        cum_w = sum(w for _, w, _ in encs)
        cum_g = sum(g for _, _, g in encs)
        wr = cum_w / cum_g if cum_g > 0 else 0.5
        r2_w = sum(w for _, w, _ in encs[-2:])
        r2_g = sum(g for _, _, g in encs[-2:])
        recent = r2_w / r2_g if r2_g > 0 else 0.5
        weight = (1 - wr) ** 2
        flag = " <-- STALE" if abs(wr - recent) > 0.15 else ""
        print(f"  {name:>10s}  {n_enc:4d}  {ratio:5.1f}x  {wr:6.0%}  {recent:9.0%}  {weight:8.3f}{flag}")

    # ── 4. Are frequently-seen opponents the HARD ones? ──
    print()
    print("=" * 70)
    print("SECTION 4: DOES FREQUENCY CORRELATE WITH DIFFICULTY?")
    print("=" * 70)

    # Bin opponents by encounter frequency, show mean WR per bin
    freq_bins = defaultdict(list)
    for name, encs in encounters.items():
        n = len(encs)
        if n <= 1:
            freq_bins["1"].append(encs)
        elif n <= 3:
            freq_bins["2-3"].append(encs)
        elif n <= 5:
            freq_bins["4-5"].append(encs)
        elif n <= 8:
            freq_bins["6-8"].append(encs)
        else:
            freq_bins["9+"].append(encs)

    print(f"\n  {'Freq bin':>10s}  {'Count':>6s}  {'Mean WR':>8s}  {'Mean PFSP wt':>12s}")
    for label in ["1", "2-3", "4-5", "6-8", "9+"]:
        if label not in freq_bins:
            continue
        all_encs = freq_bins[label]
        wrs = []
        for encs in all_encs:
            cum_w = sum(w for _, w, _ in encs)
            cum_g = sum(g for _, _, g in encs)
            wrs.append(cum_w / cum_g if cum_g > 0 else 0.5)
        avg_wr = sum(wrs) / len(wrs)
        avg_wt = sum((1 - wr) ** 2 for wr in wrs) / len(wrs)
        print(f"  {label:>10s}  {len(all_encs):6d}  {avg_wr:7.0%}  {avg_wt:12.3f}")

    print(f"\n  If PFSP works: high-frequency opponents should have LOWER win rate (harder)")
    print(f"  If frequency is random: no correlation")

    # ── 5. Training time spent on easy vs hard opponents ──
    print()
    print("=" * 70)
    print("SECTION 5: TRAINING TIME ALLOCATION")
    print("=" * 70)

    total_games = sum(g for encs in encounters.values() for _, _, g in encs)
    total_games_hard = 0  # WR < 50%
    total_games_medium = 0  # 50-65%
    total_games_easy = 0  # > 65%

    for name, encs in encounters.items():
        cum_w = sum(w for _, w, _ in encs)
        cum_g = sum(g for _, _, g in encs)
        wr = cum_w / cum_g if cum_g > 0 else 0.5
        games = sum(g for _, _, g in encs)
        if wr < 0.50:
            total_games_hard += games
        elif wr < 0.65:
            total_games_medium += games
        else:
            total_games_easy += games

    print(f"\n  Total games played: {total_games}")
    print(f"  vs Hard opponents (WR<50%):   {total_games_hard:5d} ({total_games_hard/total_games*100:.0f}%)")
    print(f"  vs Medium opponents (50-65%): {total_games_medium:5d} ({total_games_medium/total_games*100:.0f}%)")
    print(f"  vs Easy opponents (WR>65%):   {total_games_easy:5d} ({total_games_easy/total_games*100:.0f}%)")
    print(f"\n  Ideal (maximal learning signal): most games vs Hard + Medium")
    print(f"  Uniform baseline: ~23% hard, ~62% medium, ~15% easy (from WR distribution)")

    # ── 6. Adaptation speed: how many encounters to stabilize? ──
    print()
    print("=" * 70)
    print("SECTION 6: ADAPTATION SPEED")
    print("=" * 70)

    # For opponents seen 6+ times, compute running cumulative WR after each encounter
    # and see when it stabilizes (changes < 3% between encounters)
    stable_at = []
    for name, encs in encounters.items():
        if len(encs) < 6:
            continue
        running_wr = []
        cum_w, cum_g = 0, 0
        for _, w, g in encs:
            cum_w += w
            cum_g += g
            running_wr.append(cum_w / cum_g)
        # Find when changes become < 3%
        for i in range(2, len(running_wr)):
            recent_change = abs(running_wr[i] - running_wr[i-1])
            if all(abs(running_wr[j] - running_wr[j-1]) < 0.03 for j in range(i, min(i+3, len(running_wr)))):
                stable_at.append(i + 1)  # encounters to stabilize
                break
        else:
            stable_at.append(len(encs))  # never fully stabilized

    if stable_at:
        print(f"\n  Opponents with 6+ encounters: {len(stable_at)}")
        print(f"  Encounters to stabilize (< 3% change): "
              f"mean={sum(stable_at)/len(stable_at):.1f}, "
              f"median={sorted(stable_at)[len(stable_at)//2]}")
        never = sum(1 for s in stable_at if s >= 6)
        print(f"  Still unstable after all encounters: {never}/{len(stable_at)} ({never/len(stable_at)*100:.0f}%)")

    print()
    print("=" * 70)
    print("END OF REPORT")
    print("=" * 70)


if __name__ == "__main__":
    os.chdir(SRC)
    main()
