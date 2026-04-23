#!/usr/bin/env python
# replay_to_memmap.py — Stream human replays from HuggingFace directly to v8 memmap.
#
# No JSONL intermediate. Saves ~150+ GB of disk.
#
# Flow:
#   1. Stream replays from HuggingFace (jakegrigsby/metamon-raw-replays)
#   2. Parse each replay through poke-env (reuses replay_parser.py logic)
#   3. Extract v8 features via features_v8.make_features()
#   4. Validate each episode (dims, NaN, completeness)
#   5. Write directly to pre-allocated memmap arrays
#   6. Trim + build episode_index at end
#
# Usage:
#   python -u replay_to_memmap.py --min-rating 1500 --max-replays 80000 \
#     --out-dir data/datasets/human_v8_memmap

from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from features import make_features, POKEMON_CONT_DIM, FIELD_CONT_DIM, TRANSITION_CONT_DIM
from features import MOVE_SLOT_CONT_DIM, SWITCH_SLOT_CONT_DIM

# Import replay parsing helpers from existing replay_parser
from replay_parser import (
    _split_log_lines, _extract_players, _extract_winner, _is_tie,
    _prescan_moves, _register_prescanned_moves, _preregister_team,
    _populate_available_actions, _find_move_index, _find_switch_index,
    _extract_actions_for_turn, _find_turn_boundaries, _safe_parse,
    _parse_gen_from_format,
)
from poke_env.battle.battle import Battle


# =============================
# V8 record extraction from a replay perspective
# =============================

def _parse_perspective_v8(
    replay_id: str, lines: List[List[str]], perspective: str,
    players: Dict[str, str], winner_role: Optional[str], is_tie: bool,
    fmt: str, rating: Optional[int], moves_map: Dict[str, List[str]],
    turn_bounds: List[Tuple[int, int]], gen: int,
) -> List[Dict[str, Any]]:
    """Parse one perspective of a replay into v8 feature records.

    Returns list of flat dicts ready for memmap writing, or empty list on failure.
    """
    username = players[perspective]
    episode_id = f"{replay_id}::{perspective}"

    battle = Battle(
        battle_tag=replay_id,
        username=username,
        logger=None,
        gen=gen,
        save_replays=False,
    )

    records = []

    # Feed pre-turn lines
    first_turn_idx = turn_bounds[0][0]
    for i in range(first_turn_idx):
        parts = lines[i]
        if len(parts) < 2 or parts[1] in ("", "t:"):
            continue
        _safe_parse(battle, parts)

    _preregister_team(battle, lines)
    _register_prescanned_moves(battle, moves_map)

    for turn_idx_pos in range(len(turn_bounds)):
        turn_line_idx, turn_num = turn_bounds[turn_idx_pos]
        _safe_parse(battle, lines[turn_line_idx])

        next_turn_line_idx = (turn_bounds[turn_idx_pos + 1][0]
                              if turn_idx_pos + 1 < len(turn_bounds) else len(lines))

        actions = _extract_actions_for_turn(lines, turn_line_idx + 1, next_turn_line_idx)

        if perspective not in actions or actions[perspective][0] == "cant":
            for i in range(turn_line_idx + 1, next_turn_line_idx):
                _safe_parse(battle, lines[i])
                _register_prescanned_moves(battle, moves_map)
            continue

        action_type, action_detail = actions[perspective]
        _populate_available_actions(battle)

        if action_type == "move":
            action_idx = _find_move_index(battle, action_detail)
        else:
            action_idx = _find_switch_index(battle, action_detail)

        if action_idx < 0:
            for i in range(turn_line_idx + 1, next_turn_line_idx):
                _safe_parse(battle, lines[i])
                _register_prescanned_moves(battle, moves_map)
            continue

        # Extract V8 features
        try:
            feat = make_features(battle)
        except Exception:
            for i in range(turn_line_idx + 1, next_turn_line_idx):
                _safe_parse(battle, lines[i])
                _register_prescanned_moves(battle, moves_map)
            continue

        # Ensure chosen action is legal
        legal = feat["legal_mask"]
        if action_idx < len(legal) and legal[action_idx] == 0.0:
            legal[action_idx] = 1.0
        if legal.sum() == 0:
            legal[action_idx] = 1.0

        # Build flat record for memmap
        rec = _feat_to_flat_record(feat, episode_id, len(records), action_idx, rating)
        records.append(rec)

        # Feed turn events to advance state
        for i in range(turn_line_idx + 1, next_turn_line_idx):
            _safe_parse(battle, lines[i])
            _register_prescanned_moves(battle, moves_map)

    # Stamp terminal
    if records:
        records[-1]["done"] = True
        if is_tie:
            records[-1]["result"] = 0.5
        elif winner_role == perspective:
            records[-1]["result"] = 1.0
        elif winner_role is not None:
            records[-1]["result"] = 0.0

    return records


def _feat_to_flat_record(feat: dict, episode_id: str, t: int,
                         action_idx: int, rating: Optional[int]) -> dict:
    """Convert features_v8 output to flat arrays for memmap writing."""

    def _poke_ids(p):
        ids = p["ids"]
        return [ids["species"], ids["item"], ids["ability"],
                ids["move0"], ids["move1"], ids["move2"], ids["move3"]]

    def _poke_banks(p):
        b = p["banks"]
        return [b["hp_pct"], b["level"], b["weight"], b["height"],
                b["stat_hp"], b["stat_atk"], b["stat_def"],
                b["stat_spa"], b["stat_spd"], b["stat_spe"]]

    def _poke_move_cont(p):
        from features import extract_move_cont
        return extract_move_cont(p["continuous"])

    def _move_slot(m):
        if m is None:
            return [0], [0, 0, 0, 6], [0.0] * MOVE_SLOT_CONT_DIM
        return ([m["move_id"]], [m["bp_int"], m["acc_int"], m["pp_int"], m["priority_int"]],
                m["continuous"])

    def _switch_slot(s):
        if s is None:
            return [0], [0.0] * SWITCH_SLOT_CONT_DIM
        return [s["species_id"]], s["continuous"]

    our = feat["our_pokemon"]
    opp = feat["opp_pokemon"]
    field = feat["field"]
    trans = feat["transition"]

    move_slots = [_move_slot(m) for m in feat["active_moves"]]
    switch_slots = [_switch_slot(s) for s in feat["switch_slots"]]

    return {
        "episode_id": episode_id,
        "t": t,
        "action": action_idx,
        "done": False,
        "result": -1.0,
        "our_poke_ids": np.array([_poke_ids(p) for p in our], dtype=np.int32),
        "our_poke_banks": np.array([_poke_banks(p) for p in our], dtype=np.int32),
        "our_poke_cont": np.array([p["continuous"] for p in our], dtype=np.float32),
        "our_poke_mcont": np.array([_poke_move_cont(p) for p in our], dtype=np.float32),
        "opp_poke_ids": np.array([_poke_ids(p) for p in opp], dtype=np.int32),
        "opp_poke_banks": np.array([_poke_banks(p) for p in opp], dtype=np.int32),
        "opp_poke_cont": np.array([p["continuous"] for p in opp], dtype=np.float32),
        "opp_poke_mcont": np.array([_poke_move_cont(p) for p in opp], dtype=np.float32),
        "field_banks": np.array([field["banks"]["turn"], field["banks"]["weather_dur"],
                                  field["banks"]["terrain_dur"], field["banks"]["tr_dur"]], dtype=np.int32),
        "field_cont": np.array(field["continuous"], dtype=np.float32),
        "trans_ids": np.array([trans["ids"]["our_action"], trans["ids"]["opp_action"]], dtype=np.int32),
        "trans_cont": np.array(trans["continuous"], dtype=np.float32),
        "move_ids": np.array([ms[0][0] for ms in move_slots], dtype=np.int32),
        "move_banks": np.array([ms[1] for ms in move_slots], dtype=np.int32),
        "move_cont": np.array([ms[2] for ms in move_slots], dtype=np.float32),
        "switch_ids": np.array([ss[0][0] for ss in switch_slots], dtype=np.int32),
        "switch_cont": np.array([ss[1] for ss in switch_slots], dtype=np.float32),
        "legal": feat["legal_mask"].astype(np.float32),
    }


# =============================
# Validation
# =============================

def validate_episode(records: List[dict]) -> Tuple[bool, str]:
    """Validate a completed episode. Returns (is_valid, reason)."""
    if not records:
        return False, "empty"
    if not records[-1].get("done", False):
        return False, "no terminal"
    if records[-1].get("result", -1.0) < 0:
        return False, "no result"

    for i, rec in enumerate(records):
        # Check action
        if not (0 <= rec["action"] <= 8):
            return False, f"bad action {rec['action']} at t={i}"
        # Check legal mask
        if rec["legal"].sum() <= 0:
            return False, f"no legal actions at t={i}"
        # Check dims
        if rec["our_poke_cont"].shape != (6, POKEMON_CONT_DIM):
            return False, f"bad our_poke_cont shape {rec['our_poke_cont'].shape} at t={i}"
        if rec["field_cont"].shape[0] != FIELD_CONT_DIM:
            return False, f"bad field_cont dim {rec['field_cont'].shape} at t={i}"
        if rec["trans_cont"].shape[0] != TRANSITION_CONT_DIM:
            return False, f"bad trans_cont dim {rec['trans_cont'].shape} at t={i}"
        # Check NaN/inf
        for key in ["our_poke_cont", "opp_poke_cont", "field_cont", "trans_cont", "move_cont"]:
            arr = rec[key]
            if np.isnan(arr).any() or np.isinf(arr).any():
                return False, f"NaN/inf in {key} at t={i}"

    return True, "ok"


# =============================
# Memmap writer
# =============================

def _trim_files_to_n_rows(out_dir: Path, N: int, dims: Dict[str, int]) -> None:
    """Truncate raw-memmap .npy files to exactly N rows.

    These files were created by np.memmap(mode='w+') with no numpy header, so they
    are flat row-major float32/int32 binaries. Truncating to N * per_row_bytes
    yields a valid memmap of shape (N, ...). Skips files that already match.

    Does NOT touch episode_index.npy (written by np.save, has a header) or
    metadata.json.
    """
    pcd = dims["poke_cont_dim"]; fcd = dims["field_cont_dim"]
    tcd = dims["trans_cont_dim"]; mcd = dims["move_cont_dim"]
    scd = dims["switch_cont_dim"]
    i4 = np.dtype(np.int32).itemsize
    f4 = np.dtype(np.float32).itemsize

    # (filename, bytes-per-row) for every raw-memmap file written by MemmapV8Writer.
    # Keep in sync with __init__ shapes.
    targets = [
        ("our_pokemon_ids.npy",   6 * 7 * i4),
        ("our_pokemon_banks.npy", 6 * 10 * i4),
        ("our_pokemon_cont.npy",  6 * pcd * f4),
        ("our_pokemon_mcont.npy", 6 * 4 * 23 * f4),
        ("opp_pokemon_ids.npy",   6 * 7 * i4),
        ("opp_pokemon_banks.npy", 6 * 10 * i4),
        ("opp_pokemon_cont.npy",  6 * pcd * f4),
        ("opp_pokemon_mcont.npy", 6 * 4 * 23 * f4),
        ("field_banks.npy", 4 * i4),
        ("field_cont.npy",  fcd * f4),
        ("trans_ids.npy",   2 * i4),
        ("trans_cont.npy",  tcd * f4),
        ("move_ids.npy",    4 * i4),
        ("move_banks.npy",  4 * 4 * i4),
        ("move_cont.npy",   4 * mcd * f4),
        ("switch_ids.npy",  5 * i4),
        ("switch_cont.npy", 5 * scd * f4),
        ("legal.npy",       9 * f4),
        ("action.npy",      i4),
        ("result.npy",      f4),
        ("turn.npy",        i4),
    ]
    reclaimed = 0
    for name, row_bytes in targets:
        path = out_dir / name
        if not path.exists():
            continue
        cur = path.stat().st_size
        want = N * row_bytes
        if cur == want:
            continue
        if cur < want:
            # Unexpected — file is smaller than N rows. Skip to avoid corruption.
            print(f"  [WARN] {name}: size {cur} < expected {want}, skipping")
            continue
        os.truncate(str(path), want)
        reclaimed += cur - want
    if reclaimed:
        print(f"  Trimmed memmap files: reclaimed {reclaimed/1e9:.2f} GB")


class MemmapV8Writer:
    """Writes v8 records directly to pre-allocated memmap arrays."""

    def __init__(self, out_dir: str, max_rows: int,
                 poke_cont_dim: int, field_cont_dim: int,
                 trans_cont_dim: int, move_cont_dim: int, switch_cont_dim: int):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_rows = max_rows
        self.cursor = 0
        self.episodes = []  # list of (episode_id, start, length)

        def _mm(name, shape, dtype):
            return np.memmap(str(self.out_dir / name), dtype=dtype, mode="w+", shape=shape)

        N = max_rows
        self.our_poke_ids = _mm("our_pokemon_ids.npy", (N, 6, 7), np.int32)
        self.our_poke_banks = _mm("our_pokemon_banks.npy", (N, 6, 10), np.int32)
        self.our_poke_cont = _mm("our_pokemon_cont.npy", (N, 6, poke_cont_dim), np.float32)
        self.our_poke_mcont = _mm("our_pokemon_mcont.npy", (N, 6, 4, 23), np.float32)
        self.opp_poke_ids = _mm("opp_pokemon_ids.npy", (N, 6, 7), np.int32)
        self.opp_poke_banks = _mm("opp_pokemon_banks.npy", (N, 6, 10), np.int32)
        self.opp_poke_cont = _mm("opp_pokemon_cont.npy", (N, 6, poke_cont_dim), np.float32)
        self.opp_poke_mcont = _mm("opp_pokemon_mcont.npy", (N, 6, 4, 23), np.float32)
        self.field_banks = _mm("field_banks.npy", (N, 4), np.int32)
        self.field_cont = _mm("field_cont.npy", (N, field_cont_dim), np.float32)
        self.trans_ids = _mm("trans_ids.npy", (N, 2), np.int32)
        self.trans_cont = _mm("trans_cont.npy", (N, trans_cont_dim), np.float32)
        self.move_ids = _mm("move_ids.npy", (N, 4), np.int32)
        self.move_banks = _mm("move_banks.npy", (N, 4, 4), np.int32)
        self.move_cont = _mm("move_cont.npy", (N, 4, move_cont_dim), np.float32)
        self.switch_ids = _mm("switch_ids.npy", (N, 5), np.int32)
        self.switch_cont = _mm("switch_cont.npy", (N, 5, switch_cont_dim), np.float32)
        self.legal = _mm("legal.npy", (N, 9), np.float32)
        self.action = _mm("action.npy", (N,), np.int32)
        self.result = _mm("result.npy", (N,), np.float32)
        self.result[:] = -1.0
        self.turn = _mm("turn.npy", (N,), np.int32)

        self._dims = {
            "poke_cont_dim": poke_cont_dim, "field_cont_dim": field_cont_dim,
            "trans_cont_dim": trans_cont_dim, "move_cont_dim": move_cont_dim,
            "switch_cont_dim": switch_cont_dim,
        }

    def write_episode(self, records: List[dict]) -> bool:
        """Write a validated episode to memmap. Returns True if written."""
        n = len(records)
        if self.cursor + n > self.max_rows:
            return False

        ep_id = records[0]["episode_id"]
        start = self.cursor

        for rec in records:
            i = self.cursor
            self.our_poke_ids[i] = rec["our_poke_ids"]
            self.our_poke_banks[i] = rec["our_poke_banks"]
            self.our_poke_cont[i] = rec["our_poke_cont"]
            self.our_poke_mcont[i] = rec["our_poke_mcont"]
            self.opp_poke_ids[i] = rec["opp_poke_ids"]
            self.opp_poke_banks[i] = rec["opp_poke_banks"]
            self.opp_poke_cont[i] = rec["opp_poke_cont"]
            self.opp_poke_mcont[i] = rec["opp_poke_mcont"]
            self.field_banks[i] = rec["field_banks"]
            self.field_cont[i] = rec["field_cont"]
            self.trans_ids[i] = rec["trans_ids"]
            self.trans_cont[i] = rec["trans_cont"]
            self.move_ids[i] = rec["move_ids"]
            self.move_banks[i] = rec["move_banks"]
            self.move_cont[i] = rec["move_cont"]
            self.switch_ids[i] = rec["switch_ids"]
            self.switch_cont[i] = rec["switch_cont"]
            self.legal[i] = rec["legal"]
            self.action[i] = rec["action"]
            self.turn[i] = rec["t"]
            r = rec.get("result", -1.0)
            if r >= 0:
                self.result[i] = r
            self.cursor += 1

        # Propagate result to all rows in episode
        result_val = records[-1].get("result", -1.0)
        if result_val >= 0:
            self.result[start:start + n] = result_val

        self.episodes.append((ep_id, start, n))
        return True

    def finalize(self):
        """Trim memmaps to actual size, write episode_index and metadata."""
        N = self.cursor
        E = len(self.episodes)

        # Build episode index
        ep_index = np.zeros((E, 3), dtype=np.int64)
        for i, (eid, start, length) in enumerate(self.episodes):
            h = hashlib.sha1(eid.encode()).digest()[:8]
            ep_hash = int.from_bytes(h, "big") & ((1 << 63) - 1)
            ep_index[i] = [start, length, ep_hash]
        np.save(str(self.out_dir / "episode_index.npy"), ep_index)

        # Write metadata
        meta = {
            "version": "v8",
            "num_records": N,
            "num_episodes": E,
            **self._dims,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(str(self.out_dir / "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Flush all memmaps, then release handles so we can truncate.
        mm_attrs = [a for a in dir(self) if isinstance(getattr(self, a), np.memmap)]
        for attr in mm_attrs:
            getattr(self, attr).flush()
        for attr in mm_attrs:
            mm = getattr(self, attr)
            if hasattr(mm, "_mmap") and mm._mmap is not None:
                mm._mmap.close()
            setattr(self, attr, None)
        import gc
        gc.collect()

        # Physically trim each memmap file to N rows (reclaims unused preallocation).
        if N < self.max_rows:
            _trim_files_to_n_rows(self.out_dir, N, self._dims)

        # Report size
        total_bytes = sum(
            os.path.getsize(str(self.out_dir / fn))
            for fn in os.listdir(str(self.out_dir))
            if fn.endswith(".npy")
        )
        print(f"\nMemmap finalized: {N:,} rows, {E:,} episodes, {total_bytes/1e9:.2f} GB")
        print(f"  Max allocated: {self.max_rows:,} rows")
        print(f"  Utilization: {N/self.max_rows*100:.1f}%")


# =============================
# Main
# =============================

def main():
    p = argparse.ArgumentParser(description="Stream human replays directly to v8 memmap")
    p.add_argument("--format", default="gen9ou")
    p.add_argument("--min-rating", type=int, default=1500)
    p.add_argument("--max-replays", type=int, default=80000)
    p.add_argument("--max-rows", type=int, default=5_000_000,
                   help="Pre-allocated memmap size (rows). Excess is unused.")
    p.add_argument("--out-dir", default="data/datasets/human_v8_memmap")
    p.add_argument("--log-both", action="store_true", default=True)
    p.add_argument("--no-log-both", dest="log_both", action="store_false")
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print(f"Streaming replays: format={args.format}, min_rating={args.min_rating}, "
          f"max_replays={args.max_replays}")
    print(f"Output: {args.out_dir}")
    print(f"Pre-allocated: {args.max_rows:,} rows ({args.max_rows * 22 / 1e6:.0f} MB estimate)")
    print()

    # Initialize writer
    writer = MemmapV8Writer(
        args.out_dir, args.max_rows,
        poke_cont_dim=POKEMON_CONT_DIM,
        field_cont_dim=FIELD_CONT_DIM,
        trans_cont_dim=TRANSITION_CONT_DIM,
        move_cont_dim=MOVE_SLOT_CONT_DIM,
        switch_cont_dim=SWITCH_SLOT_CONT_DIM,
    )

    # Stream dataset
    print("Loading HuggingFace dataset (streaming)...")
    ds = load_dataset("jakegrigsby/metamon-raw-replays", split="train", streaming=True)

    n_processed = 0
    n_skipped_fmt = 0
    n_skipped_rat = 0
    n_skipped_parse = 0
    n_skipped_valid = 0
    n_records = 0
    n_episodes = 0
    t_start = time.time()

    fmt_clean = re.sub(r"[^a-z0-9]", "", args.format.lower())

    for row in ds:
        if n_processed >= args.max_replays:
            break

        # Filter format
        format_id = row.get("formatid") or row.get("format") or ""
        fid_clean = re.sub(r"[^a-z0-9]", "", str(format_id).lower())
        if fmt_clean not in fid_clean:
            n_skipped_fmt += 1
            continue

        # Filter rating
        rating_val = row.get("rating")
        try:
            rating_int = int(rating_val) if rating_val is not None else 0
        except (ValueError, TypeError):
            rating_int = 0
        if rating_int < args.min_rating:
            n_skipped_rat += 1
            continue

        # Parse
        log_text = row.get("log", "")
        replay_id = str(row.get("id", f"replay-{n_processed}"))
        if not log_text or len(log_text) < 50:
            n_skipped_parse += 1
            continue

        try:
            lines = _split_log_lines(log_text)
            players = _extract_players(lines)
            winner_name = _extract_winner(lines)
            tie = _is_tie(lines)
            moves_map = _prescan_moves(lines)
            turn_bounds = _find_turn_boundaries(lines)
            gen = _parse_gen_from_format(args.format)

            if not turn_bounds or len(players) < 2:
                n_skipped_parse += 1
                continue

            winner_role = None
            if winner_name:
                for role, name in players.items():
                    if name == winner_name:
                        winner_role = role
                        break

            perspectives = list(players.keys())
            if not args.log_both:
                perspectives = perspectives[:1]

            for persp in perspectives:
                records = _parse_perspective_v8(
                    replay_id, lines, persp, players, winner_role, tie,
                    args.format, rating_int, moves_map, turn_bounds, gen,
                )

                if not records:
                    continue

                # Validate
                valid, reason = validate_episode(records)
                if not valid:
                    n_skipped_valid += 1
                    continue

                # Write to memmap
                if not writer.write_episode(records):
                    print(f"\n[WARN] Memmap full at {writer.cursor:,} rows. Stopping.")
                    n_processed = args.max_replays  # break outer loop
                    break

                n_records += len(records)
                n_episodes += 1

        except Exception:
            n_skipped_parse += 1
            if n_skipped_parse <= 5:
                traceback.print_exc()
            continue

        n_processed += 1

        if n_processed % 500 == 0:
            elapsed = time.time() - t_start
            rate = n_processed / elapsed if elapsed > 0 else 0
            mem_gb = writer.cursor * 22 / 1e9
            print(f"  [{n_processed:,}/{args.max_replays:,}] "
                  f"episodes={n_episodes:,} records={n_records:,} "
                  f"memmap={mem_gb:.1f}GB "
                  f"skip(fmt={n_skipped_fmt:,} rat={n_skipped_rat:,} "
                  f"err={n_skipped_parse:,} val={n_skipped_valid:,}) "
                  f"({rate:.1f} rep/s)", flush=True)

    # Finalize
    writer.finalize()
    elapsed = time.time() - t_start

    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Replays processed: {n_processed:,}")
    print(f"  Episodes written:  {n_episodes:,}")
    print(f"  Records written:   {n_records:,}")
    print(f"  Skipped (format):  {n_skipped_fmt:,}")
    print(f"  Skipped (rating):  {n_skipped_rat:,}")
    print(f"  Skipped (parse):   {n_skipped_parse:,}")
    print(f"  Skipped (validate): {n_skipped_valid:,}")


if __name__ == "__main__":
    main()
