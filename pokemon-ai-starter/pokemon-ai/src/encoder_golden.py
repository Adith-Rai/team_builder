# encoder_golden.py — bit-identity harness for the feature encoder.
#
# Purpose: the FormatConfig threading refactor (CURRENT_STATE.md section 2,
# step 1) replaces import-time FORMAT_SINGLES constants with config-derived
# values. For gen9ou the dynamic values EQUAL today's hardcoded ones, so the
# encoder output must be bit-identical before and after. Any difference is a
# bug, not a design choice.
#
# This harness captures a golden fingerprint of the full offline path:
#
#     replay log -> poke_env.Battle -> make_features() -> feature dict
#
# It reuses replay_to_memmap.py's parsing helpers rather than reimplementing
# them, so it exercises the real production path.
#
# make_features() is deterministic: the only randomness in features.py is
# _permute_team(), which is reached solely via build_turn_batch(training=True)
# and is not on this path. Verified 2026-09-21.
#
# Usage:
#   # once, before the refactor (needs network the first time to build fixture)
#   python encoder_golden.py capture
#
#   # after each refactor step (offline, fast)
#   python encoder_golden.py verify
#
# Exit code is 0 on match, 1 on any drift, so it can gate a commit.

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

GOLDEN_DIR = Path("data/golden")
FIXTURE_PATH = GOLDEN_DIR / "fixture_logs.json"
GOLDEN_PATH = GOLDEN_DIR / "encoder_golden.json"

DEFAULT_N = 15
DEFAULT_FORMAT = "gen9ou"
DEFAULT_MIN_RATING = 1500


# ---------------------------------------------------------------- fixtures

def build_fixture(n: int, fmt: str, min_rating: int) -> List[Dict[str, str]]:
    """Stream a handful of real replay logs from HuggingFace and cache them.

    Only needed once. Logs average ~8 KB, so the fixture is small enough to
    commit, which makes `verify` hermetic and offline.
    """
    import re
    from datasets import load_dataset

    print(f"Streaming {n} {fmt} replays (min_rating={min_rating}) from HuggingFace...")
    ds = load_dataset("jakegrigsby/metamon-raw-replays", split="train", streaming=True)

    fmt_clean = re.sub(r"[^a-z0-9]", "", fmt.lower())
    out: List[Dict[str, str]] = []
    for row in ds:
        if len(out) >= n:
            break
        fid = str(row.get("formatid") or row.get("format") or "")
        if fmt_clean not in re.sub(r"[^a-z0-9]", "", fid.lower()):
            continue
        try:
            rating = int(row.get("rating") or 0)
        except (ValueError, TypeError):
            rating = 0
        if rating < min_rating:
            continue
        log = row.get("log", "")
        if not log or len(log) < 50:
            continue
        out.append({
            "id": str(row.get("id", f"replay-{len(out)}")),
            "log": log,
            "rating": rating,
        })
        print(f"  [{len(out)}/{n}] {out[-1]['id']} (rating {rating}, {len(log)} chars)")

    if len(out) < n:
        print(f"WARNING: only found {len(out)} of {n} requested replays")
    return out


def load_fixture() -> List[Dict[str, str]]:
    if not FIXTURE_PATH.exists():
        raise SystemExit(
            f"No fixture at {FIXTURE_PATH}. Run `python encoder_golden.py capture` first "
            f"(needs network once)."
        )
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------ fingerprint

def _canon(obj: Any) -> Any:
    """Canonicalise for hashing. Floats via repr() (exact round-trip in py3),
    numpy arrays/scalars unwrapped, dict keys sorted by json.dumps."""
    import numpy as np

    if isinstance(obj, (np.ndarray,)):
        return [_canon(x) for x in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        return repr(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return repr(obj)
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(x) for x in obj]
    return obj


def _sha(obj: Any) -> str:
    blob = json.dumps(_canon(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def fingerprint_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-field hashes + per-field shapes across every encoded turn.

    Keys are derived from the records themselves rather than hardcoded, so the
    harness cannot silently hash nothing if the record schema changes.

    Per-field rather than one blob hash so a mismatch localises to the field
    that broke instead of just reporting 'differs'. Shapes are tracked
    separately so a dimension change is reported as a dimension change rather
    than as an opaque hash difference.
    """
    from features import DIMS

    if not records:
        raise SystemExit(
            "encode_fixture produced 0 records — refusing to write an empty golden. "
            "An empty golden would pass any refactor. Check that data/vocab/ exists "
            "(rebuild with `python vocab.py`) and that the fixture logs parse."
        )

    keys = sorted(records[0].keys())
    for i, r in enumerate(records):
        if sorted(r.keys()) != keys:
            raise SystemExit(f"record {i} has inconsistent keys vs record 0")

    per_field = {k: _sha([r[k] for r in records]) for k in keys}
    shapes = {
        k: (list(records[0][k].shape) if hasattr(records[0][k], "shape")
            else type(records[0][k]).__name__)
        for k in keys
    }
    return {
        "n_turns": len(records),
        "dims": {k: int(v) for k, v in sorted(DIMS.items())},
        "shapes": shapes,
        "per_field": per_field,
        "all": _sha([{k: r[k] for k in keys} for r in records]),
    }


# --------------------------------------------------------------- encoding

def encode_fixture(fixture: List[Dict[str, str]], fmt: str) -> List[Dict[str, Any]]:
    """Run the real offline pipeline over the fixture logs.

    Reuses replay_to_memmap helpers so this exercises the production path,
    not a parallel reimplementation.
    """
    # Seeded defensively. make_features is deterministic today; if that ever
    # changes, this keeps the harness stable rather than silently flaky.
    random.seed(0)

    from replay_parser import (
        _split_log_lines, _extract_players, _extract_winner, _is_tie,
        _prescan_moves, _find_turn_boundaries, _parse_gen_from_format,
    )
    from replay_to_memmap import _parse_perspective_v8

    gen = _parse_gen_from_format(fmt)
    all_feats: List[Dict[str, Any]] = []

    for entry in fixture:
        replay_id, log_text = entry["id"], entry["log"]
        lines = _split_log_lines(log_text)
        players = _extract_players(lines)
        winner_name = _extract_winner(lines)
        tie = _is_tie(lines)
        moves_map = _prescan_moves(lines)
        turn_bounds = _find_turn_boundaries(lines)
        if not turn_bounds or len(players) < 2:
            print(f"  skip {replay_id}: unparseable")
            continue

        winner_role = None
        if winner_name:
            for role, name in players.items():
                if name == winner_name:
                    winner_role = role
                    break

        for persp in sorted(players.keys()):  # sorted => deterministic order
            records = _parse_perspective_v8(
                replay_id, lines, persp, players, winner_role, tie,
                fmt, entry.get("rating", 0), moves_map, turn_bounds, gen,
            )
            if records:
                all_feats.extend(records)
        print(f"  {replay_id}: cumulative {len(all_feats)} records")

    return all_feats


# ------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["capture", "verify"])
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--format", default=DEFAULT_FORMAT)
    ap.add_argument("--min-rating", type=int, default=DEFAULT_MIN_RATING)
    args = ap.parse_args()

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "capture":
        if not FIXTURE_PATH.exists():
            fixture = build_fixture(args.n, args.format, args.min_rating)
            with open(FIXTURE_PATH, "w", encoding="utf-8") as f:
                json.dump(fixture, f)
            print(f"Fixture cached: {FIXTURE_PATH} ({FIXTURE_PATH.stat().st_size/1024:.0f} KB)")
        else:
            print(f"Reusing existing fixture: {FIXTURE_PATH}")
            fixture = load_fixture()

        feats = encode_fixture(fixture, args.format)
        fp = fingerprint_records(feats)
        with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
            json.dump(fp, f, indent=2, sort_keys=True)
        print(f"\nGolden written: {GOLDEN_PATH}")
        print(f"  turns={fp['n_turns']}  all={fp['all']}")
        for k, v in fp["per_field"].items():
            print(f"    {k:16s} {v}")
        return 0

    # verify
    if not GOLDEN_PATH.exists():
        raise SystemExit(f"No golden at {GOLDEN_PATH}. Run capture first.")
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        ref = json.load(f)

    feats = encode_fixture(load_fixture(), args.format)
    cur = fingerprint_records(feats)

    problems = []
    if cur["n_turns"] != ref["n_turns"]:
        problems.append(f"n_turns {ref['n_turns']} -> {cur['n_turns']}")
    for k in sorted(set(ref["dims"]) | set(cur["dims"])):
        a, b = ref["dims"].get(k), cur["dims"].get(k)
        if a != b:
            problems.append(f"DIM {k}: {a} -> {b}")
    for k in sorted(set(ref.get("shapes", {})) | set(cur["shapes"])):
        a, b = ref.get("shapes", {}).get(k), cur["shapes"].get(k)
        if a != b:
            problems.append(f"SHAPE {k}: {a} -> {b}")
    for k in sorted(set(ref["per_field"]) | set(cur["per_field"])):
        a, b = ref["per_field"].get(k), cur["per_field"].get(k)
        if a != b:
            problems.append(f"FIELD {k}: {a} -> {b}")

    print()
    if problems:
        print("DRIFT DETECTED - encoder output is NOT bit-identical:")
        for p in problems:
            print(f"  {p}")
        return 1

    print(f"MATCH - encoder output bit-identical ({cur['n_turns']} turns, all={cur['all']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
