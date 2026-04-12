"""
Experiment comparison analytics.
Pulls from evals.jsonl (bot evals every 20 iters) and log files (per-iter training metrics).
Compares Pre-Exp baseline, Exp 1 (ent=0.02), Exp 1b (ent=0.03), Exp 1c (ent=0.02 from 2119).
"""
import json, re, sys, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).parent

# ── Experiment definitions (iter ranges + log files) ──────────────────────
EXPERIMENTS = {
    "Pre-Exp (lam=0.75, ent=0.04)": {
        "iter_range": (1600, 1784),  # late baseline only (comparable training maturity)
        "run_dirs": [
            "data\\models\\rl_v9\\selfplay_v9_20260407_124041",
            "data\\models\\rl_v9\\selfplay_v9_20260408_042048",
        ],
        "log": None,  # no single log for this era
    },
    "Exp 1 (lam=0.95, ent=0.02)": {
        "iter_range": (1789, 2019),
        "run_dirs": [
            "data\\models\\rl_v9\\selfplay_v9_20260409_080620",
            "data\\models\\rl_v9\\selfplay_v9_20260410_001804",
            # also include the short extension run that went to iter 2019
            "data\\models\\rl_v9\\selfplay_v9_20260410_082514",
        ],
        "log": "exp1_lambda095.log",
    },
    "Exp 1b (lam=0.95, ent=0.03)": {
        "iter_range": (2020, 2230),
        "run_dirs": [
            "data\\models\\rl_v9\\selfplay_v9_20260410_201319",
        ],
        "log": "exp1b_ent03.log",
    },
    "Exp 1c (lam=0.95, ent=0.02)": {
        "iter_range": (2120, 2544),
        "run_dirs": [
            "data\\models\\rl_v9\\selfplay_v9_20260411_115905",
        ],
        "log": "exp1c_ent02_from2119.log",
    },
}

BOTS = ["SH", "SmartDmg", "Tactical", "Strategic"]


# ── Parse evals.jsonl ─────────────────────────────────────────────────────
def load_evals():
    path = SRC / "data" / "eval" / "registry" / "evals.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def assign_experiment(it, run_dir):
    """Assign an eval row to an experiment based on iter + run_dir."""
    for name, cfg in EXPERIMENTS.items():
        lo, hi = cfg["iter_range"]
        if lo <= it <= hi and run_dir in cfg["run_dirs"]:
            return name
    # No fallback — require run_dir match to avoid cross-experiment duplicates
    return None


# ── Parse log files for per-iter training metrics ─────────────────────────
ITER_RE = re.compile(
    r"\[[\d:]+\]\s+Iter\s+(\d+):\s+W/L/T=(\d+)/(\d+)/(\d+)\s+\(([0-9.]+)%\),\s+"
    r"(\d+)\s+steps,\s+collect=(\d+)s,\s+update=(\d+)s,\s+"
    r"pi=([0-9eE.+-]+)\s+v=([0-9eE.+-]+)\s+ent=([0-9eE.+-]+)\s+kl=([0-9eE.+-]+)"
)


def parse_log(log_name):
    """Parse a training log file, return list of dicts with per-iter metrics."""
    path = SRC / log_name
    if not path.exists():
        return []
    results = []
    with open(path) as f:
        for line in f:
            m = ITER_RE.search(line)
            if m:
                it = int(m.group(1))
                wins, losses, ties = int(m.group(2)), int(m.group(3)), int(m.group(4))
                results.append({
                    "iter": it,
                    "win_pct": float(m.group(5)),
                    "wins": wins, "losses": losses, "ties": ties,
                    "steps": int(m.group(6)),
                    "collect_s": int(m.group(7)),
                    "update_s": int(m.group(8)),
                    "pi_loss": float(m.group(9)),
                    "v_loss": float(m.group(10)),
                    "entropy": float(m.group(11)),
                    "kl": float(m.group(12)),
                })
    return results


# ── Statistics helpers ────────────────────────────────────────────────────
def stats(values):
    if not values:
        return {"n": 0, "mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0
    std = var ** 0.5
    sv = sorted(values)
    median = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2
    return {"n": n, "mean": mean, "std": std, "min": min(values), "max": max(values), "median": median}


def rolling_mean(values, window=5):
    """Simple rolling mean."""
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start:i+1]) / (i - start + 1))
    return out


def trend_slope(xs, ys):
    """Simple linear regression slope."""
    if len(xs) < 2:
        return 0.0
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


def floor_rate(values, threshold=50.0):
    """Fraction of evals below threshold."""
    if not values:
        return 0
    return sum(1 for v in values if v < threshold) / len(values)


# ── Main analysis ─────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("EXPERIMENT COMPARISON ANALYTICS")
    print("=" * 80)

    # ── 1. Bot eval analysis from evals.jsonl ──
    evals = load_evals()
    exp_evals = defaultdict(list)  # exp_name -> list of eval rows

    for row in evals:
        exp = assign_experiment(row["iter"], row.get("run_dir", ""))
        if exp:
            exp_evals[exp].append(row)

    print("\n" + "─" * 80)
    print("SECTION 1: BOT EVAL PERFORMANCE (from evals.jsonl, every 20 iters)")
    print("─" * 80)

    for exp_name in EXPERIMENTS:
        rows = exp_evals.get(exp_name, [])
        if not rows:
            print(f"\n  {exp_name}: NO EVAL DATA")
            continue

        print(f"\n  {exp_name}  ({len(rows)} eval points, iters {rows[0]['iter']}-{rows[-1]['iter']})")
        print(f"  {'Bot':>12s}  {'Mean':>6s}  {'Std':>5s}  {'Min':>5s}  {'Max':>5s}  {'Med':>5s}  {'<50%':>5s}  {'Trend':>7s}")
        print(f"  {'─'*12}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*7}")

        for bot in BOTS:
            vals = [r[bot] for r in rows]
            iters = [r["iter"] for r in rows]
            s = stats(vals)
            slope = trend_slope(iters, vals) * 100  # per 100 iters
            fr = floor_rate(vals)
            print(f"  {bot:>12s}  {s['mean']:6.1f}  {s['std']:5.1f}  {s['min']:5.1f}  {s['max']:5.1f}  {s['median']:5.1f}  {fr:5.1%}  {slope:+7.2f}")

        savgs = [r["smart_avg"] for r in rows]
        iters = [r["iter"] for r in rows]
        s = stats(savgs)
        slope = trend_slope(iters, savgs) * 100
        fr = floor_rate(savgs)
        print(f"  {'smart_avg':>12s}  {s['mean']:6.1f}  {s['std']:5.1f}  {s['min']:5.1f}  {s['max']:5.1f}  {s['median']:5.1f}  {fr:5.1%}  {slope:+7.2f}")

    # ── 2. Cross-experiment comparison table ──
    print("\n" + "─" * 80)
    print("SECTION 2: HEAD-TO-HEAD EXPERIMENT COMPARISON (smart_avg)")
    print("─" * 80)
    print(f"\n  {'Experiment':>35s}  {'N':>3s}  {'Mean':>6s}  {'Std':>5s}  {'Min':>5s}  {'Max':>5s}  {'<50%':>5s}  {'<52%':>5s}  {'≥56%':>5s}  {'Trend/100it':>11s}")
    print(f"  {'─'*35}  {'─'*3}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*11}")

    for exp_name in EXPERIMENTS:
        rows = exp_evals.get(exp_name, [])
        if not rows:
            continue
        savgs = [r["smart_avg"] for r in rows]
        iters = [r["iter"] for r in rows]
        s = stats(savgs)
        slope = trend_slope(iters, savgs) * 100
        fr50 = floor_rate(savgs, 50.0)
        fr52 = floor_rate(savgs, 52.0)
        high = sum(1 for v in savgs if v >= 56.0) / len(savgs)
        print(f"  {exp_name:>35s}  {s['n']:3d}  {s['mean']:6.1f}  {s['std']:5.1f}  {s['min']:5.1f}  {s['max']:5.1f}  {fr50:5.1%}  {fr52:5.1%}  {high:5.1%}  {slope:+11.3f}")

    # ── 3. Per-bot comparison across experiments ──
    print("\n" + "─" * 80)
    print("SECTION 3: PER-BOT MEAN ACROSS EXPERIMENTS")
    print("─" * 80)
    print(f"\n  {'Bot':>12s}", end="")
    exp_names_with_data = [n for n in EXPERIMENTS if exp_evals.get(n)]
    for exp_name in exp_names_with_data:
        short = exp_name.split("(")[0].strip()
        print(f"  {short:>14s}", end="")
    print()
    print(f"  {'─'*12}", end="")
    for _ in exp_names_with_data:
        print(f"  {'─'*14}", end="")
    print()

    for bot in BOTS + ["smart_avg"]:
        print(f"  {bot:>12s}", end="")
        for exp_name in exp_names_with_data:
            rows = exp_evals[exp_name]
            vals = [r[bot] for r in rows]
            s = stats(vals)
            print(f"  {s['mean']:6.1f}±{s['std']:4.1f}", end="")
        print()

    # ── 4. Training metrics from logs ──
    print("\n" + "─" * 80)
    print("SECTION 4: TRAINING METRICS (from log files, per-iter)")
    print("─" * 80)

    for exp_name, cfg in EXPERIMENTS.items():
        if not cfg["log"]:
            continue
        log_data = parse_log(cfg["log"])
        lo, hi = cfg["iter_range"]
        log_data = [d for d in log_data if lo <= d["iter"] <= hi]
        if not log_data:
            print(f"\n  {exp_name}: NO LOG DATA in range {lo}-{hi}")
            continue

        print(f"\n  {exp_name}  ({len(log_data)} iters from log)")

        # Self-play win rate
        wp = [d["win_pct"] for d in log_data]
        ent = [d["entropy"] for d in log_data]
        kl_vals = [d["kl"] for d in log_data]
        pi_vals = [d["pi_loss"] for d in log_data]
        v_vals = [d["v_loss"] for d in log_data]
        steps = [d["steps"] for d in log_data]
        iters = [d["iter"] for d in log_data]

        print(f"    {'Metric':>14s}  {'Mean':>7s}  {'Std':>6s}  {'Min':>7s}  {'Max':>7s}  {'Start→End':>16s}  {'Trend/100it':>11s}")
        print(f"    {'─'*14}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*16}  {'─'*11}")

        for label, vals in [
            ("SP Win%", wp),
            ("Entropy", ent),
            ("KL", kl_vals),
            ("Pi Loss", pi_vals),
            ("V Loss", v_vals),
            ("Steps/iter", steps),
        ]:
            s = stats(vals)
            slope = trend_slope(iters, vals) * 100
            # rolling start/end (5-iter window)
            rm = rolling_mean(vals, 5)
            start_val = rm[min(4, len(rm)-1)] if rm else 0
            end_val = rm[-1] if rm else 0
            print(f"    {label:>14s}  {s['mean']:7.3f}  {s['std']:6.3f}  {s['min']:7.3f}  {s['max']:7.3f}  {start_val:7.3f}→{end_val:7.3f}  {slope:+11.4f}")

    # ── 5. Entropy trajectory comparison ──
    print("\n" + "─" * 80)
    print("SECTION 5: ENTROPY TRAJECTORY (rolling 20-iter mean)")
    print("─" * 80)

    for exp_name, cfg in EXPERIMENTS.items():
        if not cfg["log"]:
            continue
        log_data = parse_log(cfg["log"])
        lo, hi = cfg["iter_range"]
        log_data = [d for d in log_data if lo <= d["iter"] <= hi]
        if not log_data:
            continue
        ent = [d["entropy"] for d in log_data]
        iters = [d["iter"] for d in log_data]
        rm = rolling_mean(ent, 20)

        print(f"\n  {exp_name}:")
        # sample every ~50 iters for compact display
        step = max(1, len(log_data) // 10)
        for i in range(0, len(log_data), step):
            bar_len = int(rm[i] * 30)  # scale: 1.0 entropy = 30 chars
            bar = "█" * bar_len
            print(f"    iter {iters[i]:4d}:  ent={rm[i]:.3f}  {bar}")
        # always show last
        if (len(log_data) - 1) % step != 0:
            i = len(log_data) - 1
            bar_len = int(rm[i] * 30)
            bar = "█" * bar_len
            print(f"    iter {iters[i]:4d}:  ent={rm[i]:.3f}  {bar}")

    # ── 6. Consistency analysis ──
    print("\n" + "─" * 80)
    print("SECTION 6: CONSISTENCY ANALYSIS (eval smart_avg)")
    print("─" * 80)
    print("\n  Consecutive dips below 52% smart_avg:")

    for exp_name in EXPERIMENTS:
        rows = exp_evals.get(exp_name, [])
        if not rows:
            continue
        savgs = [r["smart_avg"] for r in rows]
        iters_list = [r["iter"] for r in rows]

        # find runs of consecutive dips
        dip_runs = []
        current_run = []
        for i, (it, sv) in enumerate(zip(iters_list, savgs)):
            if sv < 52.0:
                current_run.append((it, sv))
            else:
                if current_run:
                    dip_runs.append(current_run)
                    current_run = []
        if current_run:
            dip_runs.append(current_run)

        max_dip = max((len(r) for r in dip_runs), default=0)
        total_dips = sum(1 for s in savgs if s < 52.0)
        print(f"\n  {exp_name}:")
        print(f"    Total dips <52%: {total_dips}/{len(savgs)} ({total_dips/len(savgs)*100:.0f}%)")
        print(f"    Max consecutive dips: {max_dip}")
        if dip_runs:
            for run in dip_runs:
                iters_str = f"{run[0][0]}-{run[-1][0]}" if len(run) > 1 else str(run[0][0])
                vals_str = ", ".join(f"{v:.1f}" for _, v in run)
                print(f"      dip at iters {iters_str}: [{vals_str}]")

    # ── 7. Rolling smart_avg over time ──
    print("\n" + "─" * 80)
    print("SECTION 7: ROLLING SMART_AVG (3-eval window) TIMELINE")
    print("─" * 80)

    for exp_name in EXPERIMENTS:
        rows = exp_evals.get(exp_name, [])
        if len(rows) < 3:
            continue
        savgs = [r["smart_avg"] for r in rows]
        iters_list = [r["iter"] for r in rows]
        rm = rolling_mean(savgs, 3)

        print(f"\n  {exp_name}:")
        for i in range(len(rows)):
            bar_len = max(0, int((rm[i] - 45) * 2))  # scale: 45-65% range
            bar = "█" * bar_len
            marker = " ←peak" if rm[i] == max(rm) else ""
            print(f"    iter {iters_list[i]:4d}: savg={savgs[i]:5.1f}  rm3={rm[i]:5.1f}  {bar}{marker}")

    # ── 8. Per-bot deep dive: rolling timeline per bot per experiment ──
    print("\n" + "=" * 80)
    print("SECTION 8: PER-BOT ROLLING TIMELINE (3-eval window)")
    print("=" * 80)

    for bot in BOTS:
        print(f"\n  ┌─ {bot} ─────────────────────────────────────────────────")
        for exp_name in EXPERIMENTS:
            rows = exp_evals.get(exp_name, [])
            if len(rows) < 3:
                continue
            vals = [r[bot] for r in rows]
            iters_list = [r["iter"] for r in rows]
            rm = rolling_mean(vals, 3)
            peak_rm = max(rm)
            trough_rm = min(rm[2:]) if len(rm) > 2 else min(rm)  # skip warmup

            short = exp_name.split("(")[0].strip()
            print(f"  │")
            print(f"  │ {short}  (N={len(rows)})")
            for i in range(len(rows)):
                bar_len = max(0, int((rm[i] - 40) * 1.5))  # scale: 40-70 range
                bar = "█" * bar_len
                markers = []
                if rm[i] == peak_rm:
                    markers.append("PEAK")
                if i >= 2 and rm[i] == trough_rm:
                    markers.append("LOW")
                if vals[i] < 48:
                    markers.append("!!")
                elif vals[i] >= 60:
                    markers.append("**")
                mark = f"  <{','.join(markers)}>" if markers else ""
                print(f"  │   iter {iters_list[i]:4d}: {vals[i]:5.1f}  rm3={rm[i]:5.1f}  {bar}{mark}")
        print(f"  └{'─'*60}")

    # ── 9. Per-bot: cross-experiment stats table (extended) ──
    print("\n" + "─" * 80)
    print("SECTION 9: PER-BOT EXTENDED STATS")
    print("─" * 80)

    for bot in BOTS:
        print(f"\n  === {bot} ===")
        print(f"  {'Experiment':>35s}  {'N':>3s}  {'Mean':>6s}  {'Std':>5s}  {'Min':>5s}  {'Max':>5s}  {'<48%':>5s}  {'<50%':>5s}  {'>=55':>5s}  {'>=60':>5s}  {'Trend':>7s}")
        print(f"  {'─'*35}  {'─'*3}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*7}")

        for exp_name in EXPERIMENTS:
            rows = exp_evals.get(exp_name, [])
            if not rows:
                continue
            vals = [r[bot] for r in rows]
            iters = [r["iter"] for r in rows]
            s = stats(vals)
            slope = trend_slope(iters, vals) * 100
            lt48 = sum(1 for v in vals if v < 48) / len(vals)
            lt50 = sum(1 for v in vals if v < 50) / len(vals)
            ge55 = sum(1 for v in vals if v >= 55) / len(vals)
            ge60 = sum(1 for v in vals if v >= 60) / len(vals)
            print(f"  {exp_name:>35s}  {s['n']:3d}  {s['mean']:6.1f}  {s['std']:5.1f}  {s['min']:5.1f}  {s['max']:5.1f}  {lt48:5.1%}  {lt50:5.1%}  {ge55:5.1%}  {ge60:5.1%}  {slope:+7.2f}")

    # ── 10. Per-bot consistency: dips and streaks ──
    print("\n" + "─" * 80)
    print("SECTION 10: PER-BOT CONSISTENCY (dips <50%, streaks >=55%)")
    print("─" * 80)

    for bot in BOTS:
        print(f"\n  === {bot} ===")
        for exp_name in EXPERIMENTS:
            rows = exp_evals.get(exp_name, [])
            if not rows:
                continue
            vals = [r[bot] for r in rows]
            iters_list = [r["iter"] for r in rows]

            # Dip runs (<50)
            dip_runs = []
            cur = []
            for it, v in zip(iters_list, vals):
                if v < 50:
                    cur.append((it, v))
                else:
                    if cur:
                        dip_runs.append(cur)
                    cur = []
            if cur:
                dip_runs.append(cur)

            # Hot streaks (>=55)
            hot_runs = []
            cur = []
            for it, v in zip(iters_list, vals):
                if v >= 55:
                    cur.append((it, v))
                else:
                    if cur:
                        hot_runs.append(cur)
                    cur = []
            if cur:
                hot_runs.append(cur)

            max_dip = max((len(r) for r in dip_runs), default=0)
            max_hot = max((len(r) for r in hot_runs), default=0)
            total_dips = sum(1 for v in vals if v < 50)
            total_hot = sum(1 for v in vals if v >= 55)

            short = exp_name.split("(")[0].strip()
            print(f"\n    {short}:")
            print(f"      Dips <50%: {total_dips}/{len(vals)} ({total_dips/len(vals)*100:.0f}%), max consecutive: {max_dip}")
            if dip_runs:
                for run in dip_runs:
                    istr = f"{run[0][0]}-{run[-1][0]}" if len(run) > 1 else str(run[0][0])
                    vstr = ", ".join(f"{v:.0f}" for _, v in run)
                    print(f"        [{istr}]: {vstr}")
            print(f"      Hot >=55%: {total_hot}/{len(vals)} ({total_hot/len(vals)*100:.0f}%), max consecutive: {max_hot}")
            if hot_runs:
                for run in hot_runs:
                    istr = f"{run[0][0]}-{run[-1][0]}" if len(run) > 1 else str(run[0][0])
                    vstr = ", ".join(f"{v:.0f}" for _, v in run)
                    print(f"        [{istr}]: {vstr}")

    # ── 11. Bot-pair correlation: which bots move together? ──
    print("\n" + "─" * 80)
    print("SECTION 11: BOT-PAIR CORRELATION (Exp 1c, N=21)")
    print("─" * 80)

    rows_1c = exp_evals.get("Exp 1c (lam=0.95, ent=0.02)", [])
    if len(rows_1c) >= 5:
        print(f"\n  Pearson r between bot win rates (higher = they move together):")
        print(f"  {'':>12s}", end="")
        for b in BOTS:
            print(f"  {b:>10s}", end="")
        print()

        for b1 in BOTS:
            print(f"  {b1:>12s}", end="")
            v1 = [r[b1] for r in rows_1c]
            s1 = stats(v1)
            for b2 in BOTS:
                v2 = [r[b2] for r in rows_1c]
                s2 = stats(v2)
                if s1["std"] == 0 or s2["std"] == 0:
                    print(f"  {'---':>10s}", end="")
                else:
                    n = len(v1)
                    cov = sum((a - s1["mean"]) * (b - s2["mean"]) for a, b in zip(v1, v2)) / n
                    r_val = cov / (s1["std"] * s2["std"])
                    print(f"  {r_val:10.3f}", end="")
            print()

    # ── 12. Best/worst eval points per bot ──
    print("\n" + "─" * 80)
    print("SECTION 12: BEST & WORST EVAL POINTS (Exp 1c)")
    print("─" * 80)

    if rows_1c:
        for bot in BOTS:
            vals_with_iter = [(r[bot], r["iter"]) for r in rows_1c]
            vals_with_iter.sort(reverse=True)
            best3 = vals_with_iter[:3]
            worst3 = vals_with_iter[-3:]

            print(f"\n  {bot}:")
            print(f"    Best:  ", end="")
            for v, it in best3:
                print(f"  iter {it}={v:.1f}", end="")
            print()
            print(f"    Worst: ", end="")
            for v, it in worst3:
                print(f"  iter {it}={v:.1f}", end="")
            print()

        # Also show: which iters were globally best/worst across all bots
        print(f"\n  Overall savg:")
        savg_with_iter = [(r["smart_avg"], r["iter"]) for r in rows_1c]
        savg_with_iter.sort(reverse=True)
        print(f"    Best 5:  ", end="")
        for v, it in savg_with_iter[:5]:
            print(f"  iter {it}={v:.1f}", end="")
        print()
        print(f"    Worst 5: ", end="")
        for v, it in savg_with_iter[-5:]:
            print(f"  iter {it}={v:.1f}", end="")
        print()

        # Show what all bots did at the best and worst overall evals
        print(f"\n  Bot breakdown at best/worst savg points:")
        print(f"    {'iter':>6s}  {'savg':>5s}  {'SH':>5s}  {'SmD':>5s}  {'Tac':>5s}  {'Str':>5s}  {'Note':>8s}")
        print(f"    {'─'*6}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*8}")
        for v, it in savg_with_iter[:3]:
            r = next(r for r in rows_1c if r["iter"] == it)
            print(f"    {it:6d}  {v:5.1f}  {r['SH']:5.1f}  {r['SmartDmg']:5.1f}  {r['Tactical']:5.1f}  {r['Strategic']:5.1f}  {'BEST':>8s}")
        for v, it in savg_with_iter[-3:]:
            r = next(r for r in rows_1c if r["iter"] == it)
            print(f"    {it:6d}  {v:5.1f}  {r['SH']:5.1f}  {r['SmartDmg']:5.1f}  {r['Tactical']:5.1f}  {r['Strategic']:5.1f}  {'WORST':>8s}")

    # ── 13. First half vs second half comparison (Exp 1c) ──
    print("\n" + "─" * 80)
    print("SECTION 13: EXP 1c FIRST HALF vs SECOND HALF")
    print("─" * 80)

    if len(rows_1c) >= 6:
        mid = len(rows_1c) // 2
        first_half = rows_1c[:mid]
        second_half = rows_1c[mid:]

        print(f"\n  First half: iters {first_half[0]['iter']}-{first_half[-1]['iter']} (N={len(first_half)})")
        print(f"  Second half: iters {second_half[0]['iter']}-{second_half[-1]['iter']} (N={len(second_half)})")
        print(f"\n  {'Bot':>12s}  {'1st Mean':>8s}  {'1st Std':>7s}  {'2nd Mean':>8s}  {'2nd Std':>7s}  {'Delta':>7s}")
        print(f"  {'─'*12}  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*7}")

        for bot in BOTS + ["smart_avg"]:
            v1 = [r[bot] for r in first_half]
            v2 = [r[bot] for r in second_half]
            s1 = stats(v1)
            s2 = stats(v2)
            delta = s2["mean"] - s1["mean"]
            print(f"  {bot:>12s}  {s1['mean']:8.1f}  {s1['std']:7.1f}  {s2['mean']:8.1f}  {s2['std']:7.1f}  {delta:+7.1f}")

    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)


if __name__ == "__main__":
    main()
