"""
Comprehensive analysis of Exp 5 (safeguards run).
Correlates eval metrics with training metrics, entropy, KL, pi_loss,
adaptive entropy triggers, and opponent pool dynamics.
"""
import json, re, sys, os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

SRC = Path(__file__).parent


def parse_iters(log_path):
    data = []
    with open(log_path) as f:
        for line in f:
            m = re.search(
                r'\] Iter\s+(\d+).*W/L/T=(\d+)/(\d+)/\d+\s+\(([0-9.]+)%\).*'
                r'collect=(\d+)s.*update=(\d+)s.*'
                r'pi=([0-9eE.+-]+).*v=([0-9eE.+-]+).*ent=([0-9eE.+-]+).*kl=([0-9eE.+-]+)', line)
            if m:
                data.append({
                    "iter": int(m.group(1)),
                    "wins": int(m.group(2)), "losses": int(m.group(3)),
                    "wp": float(m.group(4)),
                    "collect": int(m.group(5)), "update": int(m.group(6)),
                    "pi": float(m.group(7)), "v": float(m.group(8)),
                    "ent": float(m.group(9)), "kl": float(m.group(10)),
                })
    return data


def parse_evals(log_path):
    evals = []
    iter_idx = 0
    with open(log_path) as f:
        for line in f:
            m = re.search(r'\] Iter\s+(\d+)', line)
            if m:
                iter_idx = int(m.group(1))
            m2 = re.search(r'SH=(\d+)%.*SmartDmg=(\d+)%.*Tactical=(\d+)%.*Strategic=(\d+)%.*smart_avg=(\d+)%', line)
            if m2:
                evals.append({
                    "iter": iter_idx,
                    "SH": int(m2.group(1)), "SmD": int(m2.group(2)),
                    "Tac": int(m2.group(3)), "Str": int(m2.group(4)),
                    "savg": int(m2.group(5)),
                })
    return evals


def parse_ent_triggers(log_path):
    triggers = []
    with open(log_path) as f:
        for line in f:
            m = re.search(r'\[ENT\] (Low|High) \(([0-9.]+)', line)
            if m:
                triggers.append({"type": m.group(1), "ent": float(m.group(2))})
    return triggers


def correlate(xs, ys):
    if len(xs) < 3:
        return 0
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    sx = (sum((x-mx)**2 for x in xs)/n)**0.5
    sy = (sum((y-my)**2 for y in ys)/n)**0.5
    if sx < 1e-9 or sy < 1e-9:
        return 0
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / (n * sx * sy)


def trend_slope(xs, ys):
    if len(xs) < 2:
        return 0
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    den = sum((x-mx)**2 for x in xs)
    return num / den if den > 0 else 0


def main():
    log_path = SRC / "exp5_safeguards.log"
    iters = parse_iters(log_path)
    evals = parse_evals(log_path)
    ent_triggers = parse_ent_triggers(log_path)

    print(f"Parsed: {len(iters)} iters, {len(evals)} evals, {len(ent_triggers)} entropy triggers")
    print(f"Iter range: {iters[0]['iter']}-{iters[-1]['iter']}")
    print()

    # ── 1. Eval trend ──
    print("=" * 70)
    print("SECTION 1: EVAL TREND")
    print("=" * 70)
    savgs = [e["savg"] for e in evals]
    iters_at_eval = [e["iter"] for e in evals]
    slope = trend_slope(list(range(len(savgs))), savgs) * 10  # per 10 evals
    print(f"\n  savg timeline: {savgs}")
    print(f"  Mean: {sum(savgs)/len(savgs):.1f}%, Std: {(sum((s-sum(savgs)/len(savgs))**2 for s in savgs)/len(savgs))**0.5:.1f}%")
    print(f"  Trend slope: {slope:+.2f}% per 10 evals ({'improving' if slope > 0.3 else 'declining' if slope < -0.3 else 'FLAT'})")

    # Rolling-3
    print(f"\n  Rolling-3 savg:")
    for i in range(len(evals)):
        start = max(0, i-2)
        rm3 = sum(evals[j]["savg"] for j in range(start, i+1)) / (i - start + 1)
        print(f"    eval {i+1:2d} (iter {evals[i]['iter']:4d}): savg={evals[i]['savg']:2d}  rm3={rm3:.1f}")

    # ── 2. Per-bot trend ──
    print()
    print("=" * 70)
    print("SECTION 2: PER-BOT TRENDS")
    print("=" * 70)
    bots = ["SH", "SmD", "Tac", "Str"]
    for bot in bots:
        vals = [e[bot] for e in evals]
        slope = trend_slope(list(range(len(vals))), vals) * 10
        first5 = sum(vals[:5]) / min(5, len(vals))
        last5 = sum(vals[-5:]) / min(5, len(vals))
        print(f"\n  {bot}:")
        print(f"    Timeline: {vals}")
        print(f"    Mean: {sum(vals)/len(vals):.1f}%, First5: {first5:.1f}%, Last5: {last5:.1f}%, Delta: {last5-first5:+.1f}")
        print(f"    Trend: {slope:+.2f}% per 10 evals")

    # ── 3. Training metrics in 20-iter windows aligned with evals ──
    print()
    print("=" * 70)
    print("SECTION 3: TRAINING METRICS PER EVAL WINDOW")
    print("=" * 70)

    print(f"\n  {'Eval':>5s}  {'Iter':>5s}  {'savg':>5s}  {'SP WR':>6s}  {'Ent':>6s}  {'KL':>6s}  {'Pi':>7s}  {'Upd(s)':>6s}  {'Ep0%':>5s}")

    for ei, ev in enumerate(evals):
        # Find training iters in the 10-iter window before this eval
        window = [d for d in iters if ev["iter"] - 10 < d["iter"] <= ev["iter"]]
        if not window:
            window = [d for d in iters if ev["iter"] - 20 < d["iter"] <= ev["iter"]]
        if not window:
            continue
        avg_wp = sum(d["wp"] for d in window) / len(window)
        avg_ent = sum(d["ent"] for d in window) / len(window)
        avg_kl = sum(d["kl"] for d in window) / len(window)
        avg_pi = sum(d["pi"] for d in window) / len(window)
        avg_upd = sum(d["update"] for d in window) / len(window)
        ep0_pct = sum(1 for d in window if d["update"] < 80) / len(window) * 100
        print(f"  {ei+1:5d}  {ev['iter']:5d}  {ev['savg']:4d}%  {avg_wp:5.1f}%  {avg_ent:5.3f}  {avg_kl:.4f}  {avg_pi:+.4f}  {avg_upd:5.0f}  {ep0_pct:4.0f}%")

    # ── 4. Correlations ──
    print()
    print("=" * 70)
    print("SECTION 4: CORRELATIONS (eval savg vs training metrics)")
    print("=" * 70)

    # For each eval, compute avg training metrics in surrounding window
    eval_savgs = []
    eval_ents = []
    eval_kls = []
    eval_pis = []
    eval_upds = []
    eval_ep0s = []
    eval_wps = []

    for ev in evals:
        window = [d for d in iters if ev["iter"] - 10 < d["iter"] <= ev["iter"]]
        if not window:
            continue
        eval_savgs.append(ev["savg"])
        eval_ents.append(sum(d["ent"] for d in window) / len(window))
        eval_kls.append(sum(d["kl"] for d in window) / len(window))
        eval_pis.append(sum(d["pi"] for d in window) / len(window))
        eval_upds.append(sum(d["update"] for d in window) / len(window))
        eval_ep0s.append(sum(1 for d in window if d["update"] < 80) / len(window))
        eval_wps.append(sum(d["wp"] for d in window) / len(window))

    print(f"\n  Pearson r (savg vs ...):")
    for label, vals in [
        ("Entropy", eval_ents),
        ("KL", eval_kls),
        ("Pi loss", eval_pis),
        ("Update time", eval_upds),
        ("Epoch0 rate", eval_ep0s),
        ("SP Win%", eval_wps),
    ]:
        r = correlate(eval_savgs, vals)
        strength = "STRONG" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"
        print(f"    {label:>15s}: r={r:+.3f}  ({strength})")

    # Per-bot correlations with entropy
    print(f"\n  Pearson r (per-bot vs entropy):")
    for bot in bots:
        bot_vals = [evals[i][bot] for i in range(len(eval_ents))]
        r = correlate(bot_vals, eval_ents)
        strength = "STRONG" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"
        print(f"    {bot:>15s}: r={r:+.3f}  ({strength})")

    # ── 5. Entropy trajectory ──
    print()
    print("=" * 70)
    print("SECTION 5: ENTROPY + ENT_COEF TRAJECTORY")
    print("=" * 70)

    # Chunk entropy in 20-iter windows
    print(f"\n  Entropy over time (20-iter windows):")
    for i in range(0, len(iters), 20):
        chunk = iters[i:i+20]
        if len(chunk) < 5:
            continue
        avg_ent = sum(d["ent"] for d in chunk) / len(chunk)
        min_ent = min(d["ent"] for d in chunk)
        max_ent = max(d["ent"] for d in chunk)
        bar = "#" * int(avg_ent * 30)
        print(f"    iters {chunk[0]['iter']:4d}-{chunk[-1]['iter']:4d}: "
              f"mean={avg_ent:.3f} [{min_ent:.3f}-{max_ent:.3f}] {bar}")

    print(f"\n  Adaptive entropy triggers: {len(ent_triggers)}")
    low_count = sum(1 for t in ent_triggers if t["type"] == "Low")
    high_count = sum(1 for t in ent_triggers if t["type"] == "High")
    print(f"    Low (raised ent_coef): {low_count}")
    print(f"    High (lowered ent_coef): {high_count}")

    # ── 6. KL / update time trajectory ──
    print()
    print("=" * 70)
    print("SECTION 6: KL + UPDATE TIME TRAJECTORY")
    print("=" * 70)

    print(f"\n  KL and epoch-0 rate over time (20-iter windows):")
    for i in range(0, len(iters), 20):
        chunk = iters[i:i+20]
        if len(chunk) < 5:
            continue
        avg_kl = sum(d["kl"] for d in chunk) / len(chunk)
        avg_upd = sum(d["update"] for d in chunk) / len(chunk)
        ep0 = sum(1 for d in chunk if d["update"] < 80) / len(chunk) * 100
        avg_pi = sum(d["pi"] for d in chunk) / len(chunk)
        print(f"    iters {chunk[0]['iter']:4d}-{chunk[-1]['iter']:4d}: "
              f"KL={avg_kl:.4f}  update={avg_upd:5.0f}s  ep0={ep0:4.0f}%  pi={avg_pi:+.4f}")

    # ── 7. Opponent pool analysis ──
    print()
    print("=" * 70)
    print("SECTION 7: OPPONENT POOL DYNAMICS")
    print("=" * 70)

    opp_encounters = defaultdict(list)
    for d in iters:
        # Parse opp results from the line (need to re-read log for vs= data)
        pass

    # Re-parse log for opponent data
    with open(log_path) as f:
        for line in f:
            m = re.search(r'\] Iter\s+(\d+)', line)
            if not m:
                continue
            it = int(m.group(1))
            opps = re.findall(r'(sp\d+)=(\d+)/(\d+)', line)
            for name, w, g in opps:
                opp_encounters[name].append((it, int(w), int(g)))

    # First 50 iters vs last 50 iters opponent diversity
    first_opps = set()
    last_opps = set()
    first_iters = [d["iter"] for d in iters[:50]]
    last_iters = [d["iter"] for d in iters[-50:]]

    for name, encs in opp_encounters.items():
        for it, w, g in encs:
            if it in first_iters:
                first_opps.add(name)
            if it in last_iters:
                last_opps.add(name)

    print(f"\n  Unique opponents in first 50 iters: {len(first_opps)}")
    print(f"  Unique opponents in last 50 iters:  {len(last_opps)}")
    print(f"  Overlap: {len(first_opps & last_opps)}")

    # Per-opponent WR in first vs last half
    first_half_its = set(d["iter"] for d in iters[:len(iters)//2])
    last_half_its = set(d["iter"] for d in iters[len(iters)//2:])

    drifters = []
    for name, encs in opp_encounters.items():
        h1 = [(w, g) for it, w, g in encs if it in first_half_its]
        h2 = [(w, g) for it, w, g in encs if it in last_half_its]
        if len(h1) >= 2 and len(h2) >= 2:
            h1_wr = sum(w for w, g in h1) / sum(g for w, g in h1)
            h2_wr = sum(w for w, g in h2) / sum(g for w, g in h2)
            drifters.append((h2_wr - h1_wr, name, h1_wr, h2_wr))

    if drifters:
        drifters.sort()
        pos = sum(1 for d, *_ in drifters if d > 0.05)
        neg = sum(1 for d, *_ in drifters if d < -0.05)
        flat = len(drifters) - pos - neg
        mean_drift = sum(d for d, *_ in drifters) / len(drifters)
        print(f"\n  Opponent WR drift (first half -> last half):")
        print(f"    N opponents with 2+ encounters in each half: {len(drifters)}")
        print(f"    Getting easier: {pos}, Getting harder: {neg}, Flat: {flat}")
        print(f"    Mean drift: {mean_drift:+.1%}")

    # ── 8. Summary ──
    print()
    print("=" * 70)
    print("SECTION 8: SUMMARY")
    print("=" * 70)

    print(f"\n  Run: {iters[0]['iter']}-{iters[-1]['iter']} ({len(iters)} iters)")
    print(f"  Evals: {len(evals)}, mean savg={sum(savgs)/len(savgs):.1f}%")
    print(f"  Eval trend: {slope:+.2f}% per 10 evals ({'FLAT' if abs(slope) < 0.3 else 'improving' if slope > 0 else 'declining'})")
    print(f"  Entropy: mean={sum(d['ent'] for d in iters)/len(iters):.3f}, "
          f"min={min(d['ent'] for d in iters):.3f}, max={max(d['ent'] for d in iters):.3f}")
    print(f"  Adaptive entropy triggers: {low_count} low + {high_count} high")
    print(f"  KL: mean={sum(d['kl'] for d in iters)/len(iters):.4f}")
    ep0_total = sum(1 for d in iters if d["update"] < 80) / len(iters) * 100
    print(f"  Epoch-0 stop rate: {ep0_total:.0f}%")
    print(f"  Early stop triggered: {'YES' if any('EARLY STOP' in open(log_path).read() for _ in [1]) else 'NO'}")
    print(f"  Exp 1c baseline: 54.6%")
    print(f"  Exp 2+3 last-8 mean: 55.6%")

    print()
    print("=" * 70)
    print("END OF REPORT")
    print("=" * 70)


if __name__ == "__main__":
    os.chdir(SRC)
    main()
