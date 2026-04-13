"""
Analyze PFSP opponent dynamics during training.

Parses per-opponent results from every iter in the log, tracks how each
opponent's win rate evolves over encounters, and compares against the
saved win_rates.json that PFSP uses for sampling.

Answers:
- Are opponents being rated accurately by the cumulative tracker?
- Are "hard" opponents staying hard or does the model catch up quickly?
- Are cumulative rates stale vs recent performance?
- How fast does the model's win rate against each opponent drift?
"""
import json, re, sys, os, glob
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

SRC = Path(__file__).parent


def parse_log_opponents(log_path):
    """Parse per-opponent results from every iter line.

    Returns: list of (iter_num, [(opp_name, wins, games), ...])
    """
    iters = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r'\] Iter\s+(\d+):', line)
            if not m:
                continue
            it = int(m.group(1))
            # Extract sp####=W/G pairs
            pairs = re.findall(r'(sp\d+)=(\d+)/(\d+)', line)
            opp_results = [(name, int(w), int(g)) for name, w, g in pairs]
            iters.append((it, opp_results))
    return iters


def load_win_rates_json():
    """Load the latest win_rates.json from the active run."""
    files = sorted(glob.glob(str(SRC / "data/models/rl_v9/selfplay_v9_20260412_*/win_rates.json")))
    if not files:
        return None, None
    path = files[-1]
    with open(path) as f:
        wr = json.load(f)
    # Convert full paths to short names for matching
    short_wr = {}
    for k, v in wr.items():
        m = re.search(r'snapshot_(\d+)\.pt', k)
        if m:
            short_wr[f"sp{m.group(1)}"] = v
        elif 'BEST_PPO' in k:
            short_wr["spinit"] = v
    return short_wr, path


def main():
    log_path = SRC / "exp2_exp3_pfsp.log"
    if not log_path.exists():
        print(f"Log not found: {log_path}")
        return

    iters = parse_log_opponents(log_path)
    print(f"Parsed {len(iters)} iters from log")
    print()

    # Build per-opponent encounter history
    # opp_name -> [(iter, wins, games), ...]
    encounters = defaultdict(list)
    for it, opp_results in iters:
        for name, w, g in opp_results:
            encounters[name].append((it, w, g))

    # ── 1. Opponent encounter frequency ──
    print("=" * 70)
    print("SECTION 1: ENCOUNTER FREQUENCY")
    print("=" * 70)
    freq = [(len(encs), name) for name, encs in encounters.items()]
    freq.sort(reverse=True)
    print(f"\nTotal unique opponents faced: {len(encounters)}")
    print(f"Total encounters: {sum(f for f, _ in freq)}")
    print(f"Mean encounters per opponent: {sum(f for f,_ in freq)/len(freq):.1f}")
    print(f"\nMost frequently faced (PFSP should select these more):")
    for count, name in freq[:15]:
        total_w = sum(w for _, w, _ in encounters[name])
        total_g = sum(g for _, _, g in encounters[name])
        wr = total_w / total_g if total_g > 0 else 0
        print(f"  {name}: {count} encounters, cumulative {total_w}/{total_g} = {wr:.0%}")
    print(f"\nLeast frequently faced:")
    for count, name in freq[-10:]:
        total_w = sum(w for _, w, _ in encounters[name])
        total_g = sum(g for _, _, g in encounters[name])
        wr = total_w / total_g if total_g > 0 else 0
        print(f"  {name}: {count} encounters, cumulative {total_w}/{total_g} = {wr:.0%}")

    # ── 2. Win rate drift: first encounter vs last encounter ──
    print()
    print("=" * 70)
    print("SECTION 2: WIN RATE DRIFT (first encounter vs last encounter)")
    print("=" * 70)
    print("\nOpponents with 3+ encounters — does the model catch up or fall behind?")

    drifters = []
    for name, encs in encounters.items():
        if len(encs) < 3:
            continue
        first_wr = encs[0][1] / encs[0][2] if encs[0][2] > 0 else 0.5
        last_wr = encs[-1][1] / encs[-1][2] if encs[-1][2] > 0 else 0.5
        # Also compute first-half vs second-half
        mid = len(encs) // 2
        h1_w = sum(w for _, w, _ in encs[:mid])
        h1_g = sum(g for _, _, g in encs[:mid])
        h2_w = sum(w for _, w, _ in encs[mid:])
        h2_g = sum(g for _, _, g in encs[mid:])
        h1_wr = h1_w / h1_g if h1_g > 0 else 0.5
        h2_wr = h2_w / h2_g if h2_g > 0 else 0.5
        drift = h2_wr - h1_wr
        drifters.append((drift, name, len(encs), h1_wr, h2_wr, first_wr, last_wr))

    drifters.sort()
    print(f"\n{len(drifters)} opponents with 3+ encounters")

    print(f"\nGetting HARDER (model losing ground — 1st half WR > 2nd half WR):")
    for drift, name, n, h1, h2, first, last in drifters[:10]:
        print(f"  {name}: {n} enc, 1st={h1:.0%} -> 2nd={h2:.0%} (drift {drift:+.0%})")

    print(f"\nGetting EASIER (model catching up — 1st half WR < 2nd half WR):")
    for drift, name, n, h1, h2, first, last in drifters[-10:]:
        print(f"  {name}: {n} enc, 1st={h1:.0%} -> 2nd={h2:.0%} (drift {drift:+.0%})")

    # Aggregate drift
    pos_drift = [d for d, *_ in drifters if d > 0.05]
    neg_drift = [d for d, *_ in drifters if d < -0.05]
    flat = [d for d, *_ in drifters if abs(d) <= 0.05]
    print(f"\nAggregate: {len(pos_drift)} getting easier, {len(neg_drift)} getting harder, {len(flat)} flat")
    if drifters:
        mean_drift = sum(d for d, *_ in drifters) / len(drifters)
        print(f"Mean drift: {mean_drift:+.1%} ({'model improving' if mean_drift > 0 else 'model declining'})")

    # ── 3. Staleness check: cumulative vs recent win rate ──
    print()
    print("=" * 70)
    print("SECTION 3: STALENESS — cumulative rate vs last encounter rate")
    print("=" * 70)

    stale_cases = []
    for name, encs in encounters.items():
        if len(encs) < 2:
            continue
        cum_w = sum(w for _, w, _ in encs)
        cum_g = sum(g for _, _, g in encs)
        cum_wr = cum_w / cum_g if cum_g > 0 else 0.5

        # Last encounter
        last_w, last_g = encs[-1][1], encs[-1][2]
        last_wr = last_w / last_g if last_g > 0 else 0.5

        # Last 2 encounters if available
        if len(encs) >= 2:
            r2_w = sum(w for _, w, _ in encs[-2:])
            r2_g = sum(g for _, _, g in encs[-2:])
            recent_wr = r2_w / r2_g if r2_g > 0 else 0.5
        else:
            recent_wr = last_wr

        gap = cum_wr - recent_wr  # positive = cumulative says easier than recent reality
        stale_cases.append((gap, name, len(encs), cum_wr, recent_wr, cum_w, cum_g))

    stale_cases.sort(reverse=True)

    print(f"\nCumulative OVERESTIMATES win rate (thinks easy, actually hard now):")
    print(f"  These opponents should get MORE weight but cumulative is stale-high")
    for gap, name, n, cum, recent, cw, cg in stale_cases[:10]:
        pfsp_weight_cum = (1 - cum) ** 2
        pfsp_weight_real = (1 - recent) ** 2
        print(f"  {name}: cum={cum:.0%} ({cw}/{cg}), recent={recent:.0%}, gap={gap:+.0%}  "
              f"PFSP weight: {pfsp_weight_cum:.3f} -> should be {pfsp_weight_real:.3f} ({pfsp_weight_real/max(0.001,pfsp_weight_cum):.1f}x)")

    print(f"\nCumulative UNDERESTIMATES win rate (thinks hard, actually easy now):")
    print(f"  These opponents get too much weight — model already caught up")
    for gap, name, n, cum, recent, cw, cg in stale_cases[-10:]:
        pfsp_weight_cum = (1 - cum) ** 2
        pfsp_weight_real = (1 - recent) ** 2
        print(f"  {name}: cum={cum:.0%} ({cw}/{cg}), recent={recent:.0%}, gap={gap:+.0%}  "
              f"PFSP weight: {pfsp_weight_cum:.3f} -> should be {pfsp_weight_real:.3f}")

    # Overall staleness
    if stale_cases:
        gaps = [abs(g) for g, *_ in stale_cases]
        print(f"\nOverall staleness: mean |gap| = {sum(gaps)/len(gaps):.1%}")
        big_gaps = [g for g in gaps if g > 0.15]
        print(f"Opponents with >15% staleness: {len(big_gaps)} / {len(stale_cases)} ({len(big_gaps)/len(stale_cases)*100:.0f}%)")

    # ── 4. Compare against saved JSON ──
    print()
    print("=" * 70)
    print("SECTION 4: JSON vs LOG — does saved file match actual data?")
    print("=" * 70)

    json_wr, json_path = load_win_rates_json()
    if json_wr is None:
        print("  No win_rates.json found")
    else:
        print(f"  JSON file: {json_path}")
        print(f"  JSON entries: {len(json_wr)}")

        mismatches = []
        matches = 0
        for name, encs in encounters.items():
            cum_w = sum(w for _, w, _ in encs)
            cum_g = sum(g for _, _, g in encs)
            if name in json_wr:
                jw, jg = json_wr[name]
                if jw == cum_w and jg == cum_g:
                    matches += 1
                else:
                    mismatches.append((name, cum_w, cum_g, jw, jg))

        print(f"  Matching: {matches}")
        print(f"  Mismatched: {len(mismatches)}")
        if mismatches:
            print(f"  (Mismatches expected — JSON saves every 5 iters, log has latest)")
            for name, lw, lg, jw, jg in mismatches[:5]:
                print(f"    {name}: log={lw}/{lg}, json={jw}/{jg}")

    # ── 5. PFSP weight distribution — is it actually doing anything? ──
    print()
    print("=" * 70)
    print("SECTION 5: EFFECTIVE PFSP WEIGHT DISTRIBUTION")
    print("=" * 70)

    all_weights = []
    for name, encs in encounters.items():
        cum_w = sum(w for _, w, _ in encs)
        cum_g = sum(g for _, _, g in encs)
        cum_wr = cum_w / cum_g if cum_g > 0 else 0.5
        weight = (1 - cum_wr) ** 2
        all_weights.append((weight, cum_wr, name, len(encs)))

    all_weights.sort(reverse=True)
    total_w = sum(w for w, *_ in all_weights)

    # How concentrated is the weight?
    cumsum = 0
    for i, (w, *_) in enumerate(all_weights):
        cumsum += w
        if cumsum >= total_w * 0.5:
            print(f"Top {i+1} opponents ({i+1}/{len(all_weights)} = {(i+1)/len(all_weights)*100:.0f}%) hold 50% of PFSP weight")
            break

    cumsum = 0
    for i, (w, *_) in enumerate(all_weights):
        cumsum += w
        if cumsum >= total_w * 0.8:
            print(f"Top {i+1} opponents ({i+1}/{len(all_weights)} = {(i+1)/len(all_weights)*100:.0f}%) hold 80% of PFSP weight")
            break

    print(f"\nWeight quintiles (20% chunks of opponents by weight):")
    chunk = len(all_weights) // 5
    for qi in range(5):
        start = qi * chunk
        end = start + chunk if qi < 4 else len(all_weights)
        chunk_w = sum(w for w, *_ in all_weights[start:end])
        chunk_wr = [wr for _, wr, *_ in all_weights[start:end]]
        avg_wr = sum(chunk_wr) / len(chunk_wr)
        avg_enc = sum(n for _, _, _, n in all_weights[start:end]) / (end - start)
        print(f"  Q{qi+1} (rank {start+1}-{end}): {chunk_w/total_w*100:.0f}% of weight, "
              f"mean WR={avg_wr:.0%}, mean encounters={avg_enc:.1f}")

    # ── 6. Timeline: how fast are ratings building? ──
    print()
    print("=" * 70)
    print("SECTION 6: RATING CONVERGENCE OVER TIME")
    print("=" * 70)

    # For opponents with 4+ encounters, how stable is their rating?
    # Compare rolling windows
    stable_opps = [(name, encs) for name, encs in encounters.items() if len(encs) >= 4]
    print(f"\n{len(stable_opps)} opponents with 4+ encounters")

    if stable_opps:
        volatilities = []
        for name, encs in stable_opps:
            wrs = [w/g if g > 0 else 0.5 for _, w, g in encs]
            mean_wr = sum(wrs) / len(wrs)
            var = sum((wr - mean_wr)**2 for wr in wrs) / len(wrs)
            volatilities.append((var**0.5, name, len(encs), mean_wr, wrs))

        volatilities.sort(reverse=True)
        print(f"\nMost VOLATILE opponents (win rate swings wildly — hard to rate):")
        for vol, name, n, mean, wrs in volatilities[:10]:
            wr_str = " ".join(f"{wr:.0%}" for wr in wrs)
            print(f"  {name}: std={vol:.0%}, mean={mean:.0%}, encounters: [{wr_str}]")

        print(f"\nMost STABLE opponents (consistent win rate — well rated):")
        for vol, name, n, mean, wrs in volatilities[-10:]:
            wr_str = " ".join(f"{wr:.0%}" for wr in wrs)
            print(f"  {name}: std={vol:.0%}, mean={mean:.0%}, encounters: [{wr_str}]")

        avg_vol = sum(v for v, *_ in volatilities) / len(volatilities)
        print(f"\nMean volatility: {avg_vol:.0%}")
        print(f"(Higher = per-encounter results are noisy, cumulative counting is appropriate)")
        print(f"(Lower = results are stable, cumulative counting works fine)")

    print()
    print("=" * 70)
    print("END OF REPORT")
    print("=" * 70)


if __name__ == "__main__":
    os.chdir(SRC)
    main()
