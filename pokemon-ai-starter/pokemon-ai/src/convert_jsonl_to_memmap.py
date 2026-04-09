# convert_jsonl_to_memmap_v8.py
# Converts v8 JSONL (structured per-entity features) to memmap format.
#
# Two-pass streaming:
#   Pass 1: Scan — count episodes, rows, detect dims, validate
#   Pass 2: Write — pre-allocate memmaps, write data, propagate results
#
# Output files (all in --out-dir):
#   our_pokemon_ids.npy      (N, 6, 7) int32
#   our_pokemon_banks.npy    (N, 6, 10) int32
#   our_pokemon_cont.npy     (N, 6, POKE_CONT_DIM) float32
#   our_pokemon_mcont.npy    (N, 6, 4, 23) float32  -- compact move encoding per mon
#   opp_pokemon_ids.npy      (N, 6, 7) int32
#   opp_pokemon_banks.npy    (N, 6, 10) int32
#   opp_pokemon_cont.npy     (N, 6, POKE_CONT_DIM) float32
#   opp_pokemon_mcont.npy    (N, 6, 4, 23) float32
#   field_banks.npy          (N, 4) int32
#   field_cont.npy           (N, FIELD_CONT_DIM) float32
#   trans_ids.npy            (N, 2) int32
#   trans_cont.npy           (N, TRANS_CONT_DIM) float32
#   move_ids.npy             (N, 4) int32
#   move_banks.npy           (N, 4, 4) int32      -- bp, acc, pp, prio
#   move_cont.npy            (N, 4, MOVE_CONT_DIM) float32
#   switch_ids.npy           (N, 5) int32
#   switch_cont.npy          (N, 5, SWITCH_CONT_DIM) float32
#   legal.npy                (N, 9) float32
#   action.npy               (N,) int32
#   result.npy               (N,) float32
#   turn.npy                 (N,) int32
#   episode_index.npy        (E, 3) int64  -- [start, length, hash]
#   metadata.json

from __future__ import annotations
import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from features import DIMS


def episode_hash(eid: str) -> int:
    h = hashlib.sha1(eid.encode()).digest()[:8]
    return int.from_bytes(h, "big") & ((1 << 63) - 1)


def get_episode_id(row: dict, fallback_ctr: int) -> str:
    for k in ("episode_id", "battle_tag", "battleId", "battle_id"):
        if k in row and row[k]:
            return str(row[k])
    return f"_standalone_{fallback_ctr}"


def validate_v8_row(row: dict) -> bool:
    """Check that a row has the required v8 fields."""
    if not row.get("v8", False):
        return False
    required = ["our_poke_ids", "our_poke_cont", "opp_poke_ids", "opp_poke_cont",
                 "field_cont", "trans_cont", "legal", "action"]
    for k in required:
        if k not in row:
            return False
    if not isinstance(row["action"], int) or not (0 <= row["action"] <= 8):
        return False
    if len(row.get("legal", [])) != 9:
        return False
    return True


def main():
    p = argparse.ArgumentParser(description="Convert v8 JSONL to structured memmap")
    p.add_argument("--data", default="data/datasets/obs_v8/*.jsonl",
                   help="Glob pattern for input JSONL files")
    p.add_argument("--out-dir", default="data/datasets/memmap_v8",
                   help="Output directory for memmap files")
    args = p.parse_args()

    files = sorted(glob.glob(args.data))
    if not files:
        print(f"No files found matching {args.data}")
        sys.exit(1)
    print(f"Found {len(files)} JSONL files")

    # =====================
    # Pass 1: Scan
    # =====================
    print("\n=== Pass 1: Scanning ===")
    t0 = time.time()

    episode_rows: Dict[str, int] = {}
    episode_complete: Dict[str, bool] = {}
    episode_order: List[str] = []
    total_rows = 0
    skipped = 0
    fallback_ctr = 0

    # Detect dims from first valid row
    poke_cont_dim = 0
    field_cont_dim = 0
    trans_cont_dim = 0
    move_cont_dim = 0
    switch_cont_dim = 0

    for fi, fpath in enumerate(files):
        with open(fpath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                if not validate_v8_row(row):
                    skipped += 1
                    continue

                eid = get_episode_id(row, fallback_ctr)
                if eid.startswith("_standalone_"):
                    fallback_ctr += 1

                if eid not in episode_rows:
                    episode_rows[eid] = 0
                    episode_complete[eid] = False
                    episode_order.append(eid)

                episode_rows[eid] += 1
                total_rows += 1

                if row.get("done", False):
                    episode_complete[eid] = True

                # Detect dims from first row
                if poke_cont_dim == 0:
                    poke_cont_dim = len(row["our_poke_cont"][0])
                    field_cont_dim = len(row["field_cont"])
                    trans_cont_dim = len(row["trans_cont"])
                    from features import MOVE_SLOT_CONT_DIM, SWITCH_SLOT_CONT_DIM
                    move_cont_dim = len(row["move_cont"][0]) if "move_cont" in row else MOVE_SLOT_CONT_DIM
                    switch_cont_dim = len(row["switch_cont"][0]) if "switch_cont" in row else SWITCH_SLOT_CONT_DIM

        if (fi + 1) % 10 == 0 or fi == len(files) - 1:
            print(f"  Scanned {fi+1}/{len(files)} files, {total_rows:,} rows, "
                  f"{len(episode_rows):,} episodes, {skipped:,} skipped")

    # Filter to complete episodes
    complete_eids = [eid for eid in episode_order if episode_complete.get(eid, False)]
    incomplete = len(episode_order) - len(complete_eids)
    N = sum(episode_rows[eid] for eid in complete_eids)
    E = len(complete_eids)

    print(f"\nPass 1 done in {time.time()-t0:.1f}s")
    print(f"  Total rows: {total_rows:,}, Complete episodes: {E:,}, "
          f"Incomplete dropped: {incomplete:,}, Skipped rows: {skipped:,}")
    print(f"  Final: {N:,} rows from {E:,} episodes")
    print(f"  Dims: poke_cont={poke_cont_dim}, field_cont={field_cont_dim}, "
          f"trans_cont={trans_cont_dim}, move_cont={move_cont_dim}, switch_cont={switch_cont_dim}")

    if N == 0:
        print("No complete episodes found. Exiting.")
        sys.exit(1)

    # =====================
    # Pass 2: Allocate and write
    # =====================
    print("\n=== Pass 2: Writing memmaps ===")
    t1 = time.time()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute episode start offsets
    episode_start: Dict[str, int] = {}
    complete_set = set(complete_eids)
    offset = 0
    for eid in complete_eids:
        episode_start[eid] = offset
        offset += episode_rows[eid]

    # Allocate memmaps
    def make_mm(name, shape, dtype):
        path = str(out_dir / name)
        mm = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
        return mm

    our_poke_ids_mm = make_mm("our_pokemon_ids.npy", (N, 6, 7), np.int32)
    our_poke_banks_mm = make_mm("our_pokemon_banks.npy", (N, 6, 10), np.int32)
    our_poke_cont_mm = make_mm("our_pokemon_cont.npy", (N, 6, poke_cont_dim), np.float32)
    our_poke_mcont_mm = make_mm("our_pokemon_mcont.npy", (N, 6, 4, 23), np.float32)

    opp_poke_ids_mm = make_mm("opp_pokemon_ids.npy", (N, 6, 7), np.int32)
    opp_poke_banks_mm = make_mm("opp_pokemon_banks.npy", (N, 6, 10), np.int32)
    opp_poke_cont_mm = make_mm("opp_pokemon_cont.npy", (N, 6, poke_cont_dim), np.float32)
    opp_poke_mcont_mm = make_mm("opp_pokemon_mcont.npy", (N, 6, 4, 23), np.float32)

    field_banks_mm = make_mm("field_banks.npy", (N, 4), np.int32)
    field_cont_mm = make_mm("field_cont.npy", (N, field_cont_dim), np.float32)

    trans_ids_mm = make_mm("trans_ids.npy", (N, 2), np.int32)
    trans_cont_mm = make_mm("trans_cont.npy", (N, trans_cont_dim), np.float32)

    move_ids_mm = make_mm("move_ids.npy", (N, 4), np.int32)
    move_banks_mm = make_mm("move_banks.npy", (N, 4, 4), np.int32)
    move_cont_mm = make_mm("move_cont.npy", (N, 4, move_cont_dim), np.float32)

    switch_ids_mm = make_mm("switch_ids.npy", (N, 5), np.int32)
    switch_cont_mm = make_mm("switch_cont.npy", (N, 5, switch_cont_dim), np.float32)

    legal_mm = make_mm("legal.npy", (N, 9), np.float32)
    action_mm = make_mm("action.npy", (N,), np.int32)
    result_mm = make_mm("result.npy", (N,), np.float32)
    result_mm[:] = -1.0
    turn_mm = make_mm("turn.npy", (N,), np.int32)

    # Episode buffers (buffer per episode, flush when complete)
    ep_buf: Dict[str, List] = {}
    ep_write_offset: Dict[str, int] = {}  # next write position within episode
    written_rows = 0
    fallback_ctr2 = 0

    def flush_ep(eid, rows):
        nonlocal written_rows
        rows.sort(key=lambda r: r.get("t", 0))
        start = episode_start[eid]
        for i, row in enumerate(rows):
            idx = start + i
            # Our pokemon
            for j in range(6):
                our_poke_ids_mm[idx, j, :len(row["our_poke_ids"][j])] = row["our_poke_ids"][j]
                our_poke_banks_mm[idx, j, :len(row["our_poke_banks"][j])] = row["our_poke_banks"][j]
                our_poke_cont_mm[idx, j, :len(row["our_poke_cont"][j])] = row["our_poke_cont"][j][:poke_cont_dim]
                mcont = row.get("our_poke_mcont", [[[]]*4]*6)
                for k in range(4):
                    if j < len(mcont) and k < len(mcont[j]) and len(mcont[j][k]) >= 23:
                        our_poke_mcont_mm[idx, j, k, :23] = mcont[j][k][:23]
            # Opp pokemon
            for j in range(6):
                opp_poke_ids_mm[idx, j, :len(row["opp_poke_ids"][j])] = row["opp_poke_ids"][j]
                opp_poke_banks_mm[idx, j, :len(row["opp_poke_banks"][j])] = row["opp_poke_banks"][j]
                opp_poke_cont_mm[idx, j, :len(row["opp_poke_cont"][j])] = row["opp_poke_cont"][j][:poke_cont_dim]
                mcont = row.get("opp_poke_mcont", [[[]]*4]*6)
                for k in range(4):
                    if j < len(mcont) and k < len(mcont[j]) and len(mcont[j][k]) >= 23:
                        opp_poke_mcont_mm[idx, j, k, :23] = mcont[j][k][:23]
            # Field
            fb = row.get("field_banks", [0, 0, 0, 0])
            field_banks_mm[idx, :len(fb)] = fb[:4]
            fc = row.get("field_cont", [])
            field_cont_mm[idx, :min(len(fc), field_cont_dim)] = fc[:field_cont_dim]
            # Transition
            ti = row.get("trans_ids", [0, 0])
            trans_ids_mm[idx, :len(ti)] = ti[:2]
            tc = row.get("trans_cont", [])
            trans_cont_mm[idx, :min(len(tc), trans_cont_dim)] = tc[:trans_cont_dim]
            # Active moves
            mi = row.get("move_ids", [0, 0, 0, 0])
            move_ids_mm[idx, :min(len(mi), 4)] = mi[:4]
            mb = row.get("move_banks", [[0]*4]*4)
            for j in range(4):
                if j < len(mb):
                    move_banks_mm[idx, j, :min(len(mb[j]), 4)] = mb[j][:4]
            mc = row.get("move_cont", [])
            for j in range(4):
                if j < len(mc):
                    move_cont_mm[idx, j, :min(len(mc[j]), move_cont_dim)] = mc[j][:move_cont_dim]
            # Switches
            si = row.get("switch_ids", [0]*5)
            switch_ids_mm[idx, :min(len(si), 5)] = si[:5]
            sc = row.get("switch_cont", [])
            for j in range(5):
                if j < len(sc):
                    switch_cont_mm[idx, j, :min(len(sc[j]), switch_cont_dim)] = sc[j][:switch_cont_dim]
            # Legal / action / turn
            legal_mm[idx] = row["legal"][:9]
            action_mm[idx] = int(row["action"])
            turn_mm[idx] = int(row.get("t", 0))
            # Result
            if row.get("done", False) and row.get("result") is not None:
                result_mm[idx] = float(row["result"])
            written_rows += 1

    # Stream through files
    for fi, fpath in enumerate(files):
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not validate_v8_row(row):
                    continue
                eid = get_episode_id(row, fallback_ctr2)
                if eid.startswith("_standalone_"):
                    fallback_ctr2 += 1
                if eid not in complete_set:
                    continue

                if eid not in ep_buf:
                    ep_buf[eid] = []
                ep_buf[eid].append(row)

                # Flush when episode has all expected rows
                if len(ep_buf[eid]) >= episode_rows[eid]:
                    flush_ep(eid, ep_buf[eid])
                    del ep_buf[eid]

        if (fi + 1) % 10 == 0 or fi == len(files) - 1:
            print(f"  Wrote {fi+1}/{len(files)} files, {written_rows:,}/{N:,} rows")

    # Flush any remaining buffered episodes
    for eid, rows in ep_buf.items():
        if eid in complete_set:
            flush_ep(eid, rows)
    ep_buf.clear()

    # Propagate results within episodes
    for eid in complete_eids:
        start = episode_start[eid]
        length = episode_rows[eid]
        sl = result_mm[start:start+length]
        # Find last known result
        known = np.where(sl >= 0)[0]
        if len(known) > 0:
            val = float(sl[known[-1]])
            sl[:] = val

    # Build episode index
    ep_index = np.zeros((E, 3), dtype=np.int64)
    for i, eid in enumerate(complete_eids):
        ep_index[i] = [episode_start[eid], episode_rows[eid], episode_hash(eid)]
    np.save(str(out_dir / "episode_index.npy"), ep_index)

    # Flush all memmaps
    for mm in [our_poke_ids_mm, our_poke_banks_mm, our_poke_cont_mm, our_poke_mcont_mm,
               opp_poke_ids_mm, opp_poke_banks_mm, opp_poke_cont_mm, opp_poke_mcont_mm,
               field_banks_mm, field_cont_mm, trans_ids_mm, trans_cont_mm,
               move_ids_mm, move_banks_mm, move_cont_mm,
               switch_ids_mm, switch_cont_mm,
               legal_mm, action_mm, result_mm, turn_mm]:
        mm.flush()
        del mm

    # Write metadata
    meta = {
        "version": "v8",
        "num_records": int(N),
        "num_episodes": int(E),
        "poke_cont_dim": poke_cont_dim,
        "field_cont_dim": field_cont_dim,
        "trans_cont_dim": trans_cont_dim,
        "move_cont_dim": move_cont_dim,
        "switch_cont_dim": switch_cont_dim,
        "source_pattern": args.data,
        "source_files": len(files),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(str(out_dir / "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t1
    size_gb = sum(os.path.getsize(str(out_dir / fn))
                  for fn in os.listdir(str(out_dir))
                  if fn.endswith(".npy")) / 1e9
    print(f"\nPass 2 done in {elapsed:.1f}s")
    print(f"  Written {N:,} rows, {E:,} episodes")
    print(f"  Total memmap size: {size_gb:.2f} GB")
    print(f"  Output: {out_dir}")
    print(f"  Metadata: {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
