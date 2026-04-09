#!/usr/bin/env python3
"""Recover an eval_elo_ladder shard JSON from its .log file.

eval_elo_ladder.py only writes the final JSON at the end of a shard's run, so if
a shard crashes mid-tournament we lose all its progress. This script parses the
shard's .log file (which records every completed matchup with timestamps) into a
shard-JSON compatible with `eval_elo_ladder.py --combine`.

Usage:
  python recover_elo_from_log.py data/eval/elo_shard0.log \
    --out data/eval/elo_session33_shard0_recovered.json

  # Then combine recovered + still-running shards normally:
  python eval_elo_ladder.py --combine \
    data/eval/elo_session33_shard0_recovered.json \
    data/eval/elo_session33_shard1.json \
    data/eval/elo_session33_shard2.json \
    data/eval/elo_session33_shard3.json \
    --out-json data/eval/elo_session33_final.json
"""

import argparse
import json
import re
import sys
from pathlib import Path


# Match the eval_elo_ladder log format:
#   [HH:MM:SS] [k/N] PLAYER_A vs PLAYER_B  (elapsed Xm, ETA Ym)
#   [HH:MM:SS]   -> W1-W2 (ties:T) | WR% | Es
MATCHUP_RE = re.compile(
    r'^\[\d\d:\d\d:\d\d\] \[(\d+)/\d+\] (\S+) vs (\S+)'
)
RESULT_RE = re.compile(
    r'^\[\d\d:\d\d:\d\d\]\s+->\s+(\d+)-(\d+)\s+\(ties:(\d+)\)\s+\|\s+\d+%\s+\|\s+(\d+)s'
)


# Bot names from eval_elo_ladder ALL_BOTS — used to label kinds in the recovered JSON
KNOWN_BOTS = {
    "Random", "MaxBasePower", "GreedySE", "HazardSense", "SwitchAwareEscape",
    "SetupThenSweep", "SH", "SmartDmg", "Tactical", "Strategic",
}


def parse_log(log_path: Path):
    """Parse a shard log into (matches, all_player_names)."""
    matches = []
    pending = None  # (p1_name, p2_name) of last opened matchup line
    seen_players: set = set()

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = MATCHUP_RE.match(line)
            if m:
                p1, p2 = m.group(2), m.group(3)
                pending = (p1, p2)
                seen_players.add(p1)
                seen_players.add(p2)
                continue
            r = RESULT_RE.match(line)
            if r and pending is not None:
                p1, p2 = pending
                p1w, p2w, ties, elapsed = (int(x) for x in r.groups())
                total = p1w + p2w + ties
                matches.append({
                    "p1": p1, "p2": p2,
                    "p1_kind": "bot" if p1 in KNOWN_BOTS else "snapshot",
                    "p2_kind": "bot" if p2 in KNOWN_BOTS else "snapshot",
                    "p1_wins": p1w, "p2_wins": p2w, "ties": ties,
                    "total": total,
                    "p1_wr": p1w / max(1, total),
                    "elapsed": float(elapsed),
                })
                pending = None

    return matches, sorted(seen_players)


def main():
    p = argparse.ArgumentParser(description="Recover eval_elo_ladder shard JSON from log")
    p.add_argument("log", help="Path to elo_shard*.log file")
    p.add_argument("--out", required=True, help="Output JSON path (combine-compatible)")
    args = p.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Log not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    matches, names = parse_log(log_path)
    print(f"Parsed {len(matches)} completed matchups from {log_path}")
    print(f"Found {len(names)} unique players: {names[:10]}{'...' if len(names)>10 else ''}")

    if not matches:
        print("WARN: no matches found — log format may differ or shard hadn't completed any matchup yet.")
        sys.exit(2)

    # Build a players list compatible with combine_shards()
    players = [
        {
            "name": n,
            "kind": "bot" if n in KNOWN_BOTS else "snapshot",
            "ckpt": None,  # We don't know paths from the log; not needed for BT fit
        }
        for n in names
    ]

    out = {
        "config": {
            "recovered_from": str(log_path),
            "n_matches_recovered": len(matches),
        },
        "players": players,
        "matches": matches,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path} ({len(matches)} matches, {len(players)} players)")


if __name__ == "__main__":
    main()
