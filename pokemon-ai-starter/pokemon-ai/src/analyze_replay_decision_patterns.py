"""Replay-decision-pattern analyzer for training-time saved replays.

Parses HTML replay files saved by train_rl.py during smart-bot evals and
counts decision-pattern stats per outcome (W/L) per bot opponent:
  - setup move usage rate (% of moves)
  - setup-in-turns-1-5 frequency (the "early commit" failure pattern)
  - switches per battle (over-switching / flailing signal)
  - "sacrifice / desperation" moves (Healing Wish, Memento, etc.)
  - average moves and switches per battle

The script was created during S68 (2026-06-09) to diagnose why Run #7
(no BC anchor) was losing ground vs heuristic smart-bots (SH / SmartDmg /
Tactical / Strategic) while improving on SP-pool and MM-tier opponents.
The diagnostic finding: the no-anchor model develops a setup-heavy playstyle
that wins via opponent uncertainty (works on neural opps) but is hard-punished
by deterministic damage maximizers. See `memory/project_s68_run7_decision_
pattern_findings_2026_06_09.md` for the full evidence summary.

Usage:
  python analyze_replay_decision_patterns.py <run_dir>
  python analyze_replay_decision_patterns.py <run_dir> --iters 9,19,29,39

  where <run_dir> is a directory containing replays_iter{N} subdirs, each
  containing SH/, SmartDmg/, Tactical/, Strategic/ subdirs of HTML replays.

The model is assumed to be `p1` in the replays (this matches the smart-bot
eval setup in train_rl.py). If you change which side is "us" elsewhere,
update OUR_PREFIX below.
"""

import argparse
import os
import re
import sys

OUR_PREFIX = "p1"

# Standard gen9 OU setup moves. Add to this set as the metagame shifts.
SETUP_MOVES = {
    "Growth", "Swords Dance", "Calm Mind", "Bulk Up", "Nasty Plot",
    "Dragon Dance", "Quiver Dance", "Shell Smash", "Coil", "Curse",
    "Iron Defense", "Cosmic Power", "Cotton Guard", "Stockpile", "Acid Armor",
    "Tail Glow", "Geomancy", "Belly Drum", "Filet Away",
}

# "Sacrifice / desperation" moves — these end your own Pokemon. High use rate
# in losses (vs wins) signals desperation play.
SACRIFICE_MOVES = {
    "Healing Wish", "Memento", "Self-Destruct", "Explosion",
    "Final Gambit", "Lunar Dance",
}


def analyze_replay_dir(replay_dir, bots=("SH", "SmartDmg", "Tactical", "Strategic")):
    """Return per-outcome stats for one replays_iter{N} directory."""
    out = {
        "WIN": _empty_stats(),
        "LOSS": _empty_stats(),
    }
    for bot in bots:
        bot_dir = os.path.join(replay_dir, bot)
        if not os.path.isdir(bot_dir):
            continue
        for fn in os.listdir(bot_dir):
            if not fn.endswith(".html"):
                continue
            with open(os.path.join(bot_dir, fn), encoding="utf-8", errors="ignore") as fh:
                txt = fh.read()
            _accumulate_one_battle(out, txt)
    return out


def _empty_stats():
    return {"n": 0, "moves": 0, "switches": 0,
            "setup": 0, "setup_t1_5": 0, "sacrifice": 0}


def _accumulate_one_battle(out, txt):
    p1f = txt.count("|faint|p1a:")
    p2f = txt.count("|faint|p2a:")
    if p1f == 6:
        cat = "LOSS" if OUR_PREFIX == "p1" else "WIN"
    elif p2f == 6:
        cat = "WIN" if OUR_PREFIX == "p1" else "LOSS"
    else:
        return
    s = out[cat]
    s["n"] += 1
    our_moves = re.findall(rf"\|move\|{OUR_PREFIX}a: [^|]+\|([^|]+)", txt)
    s["moves"] += len(our_moves)
    s["switches"] += max(0, txt.count(f"|switch|{OUR_PREFIX}a:") - 1)
    for m in our_moves:
        mc = m.strip()
        if mc in SETUP_MOVES:
            s["setup"] += 1
        if mc in SACRIFICE_MOVES:
            s["sacrifice"] += 1
    # Setup in turns 1-5: scan segments delimited by |turn|N tokens.
    for turn in range(1, 6):
        idx = txt.find(f"|turn|{turn}\n")
        if idx < 0:
            continue
        end = txt.find(f"|turn|{turn+1}\n", idx)
        if end < 0:
            end = idx + 5000
        seg = txt[idx:end]
        for m in re.findall(rf"\|move\|{OUR_PREFIX}a: [^|]+\|([^|]+)", seg):
            if m.strip() in SETUP_MOVES:
                s["setup_t1_5"] += 1


def print_one_iter(tag, out):
    print(f"\n=== {tag} ===")
    for cat in ("WIN", "LOSS"):
        s = out[cat]
        n = s["n"]
        if n == 0:
            continue
        moves = s["moves"]
        print(f"  {cat} n={n}: moves/battle={moves/n:.1f}, switches/battle={s['switches']/n:.2f}")
        print(f"    setup_total={s['setup']} ({100*s['setup']/max(1,moves):.2f}% of moves)")
        print(f"    setup_in_t1_5={s['setup_t1_5']} (avg {s['setup_t1_5']/n:.2f}/battle)")
        print(f"    sacrifice_moves={s['sacrifice']} (avg {s['sacrifice']/n:.3f}/battle)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir", help="Run dir containing replays_iter{N} subdirs")
    p.add_argument("--iters", default=None,
                   help="Comma-separated list of iter numbers (zero-padded "
                        "to 4 digits). Default: auto-detect all replays_iter*.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.iters:
        iters = [f"iter{int(s.strip()):04d}" for s in args.iters.split(",")]
    else:
        iters = []
        for d in sorted(os.listdir(args.run_dir)):
            if d.startswith("replays_iter"):
                iters.append(d[len("replays_"):])
    for it in iters:
        rd = os.path.join(args.run_dir, f"replays_{it}")
        if not os.path.isdir(rd):
            print(f"[skip] {rd} not found", file=sys.stderr)
            continue
        out = analyze_replay_dir(rd)
        print_one_iter(it, out)


if __name__ == "__main__":
    main()
