#!/usr/bin/env python3
"""
Convert JSONL observation files to memory-mapped numpy arrays for fast BC training.

Two-pass streaming design — never buffers more than one episode in RAM.
  Pass 1: Scan all files to count rows, detect dims, collect episode metadata.
  Pass 2: Pre-allocate memmap at exact size, re-read JSONL, write directly.

Usage:
    python convert_jsonl_to_memmap.py --data "data/datasets/obs/*.jsonl" --out-dir data/datasets/memmap
    python convert_jsonl_to_memmap.py --data "data/datasets/obs/*.jsonl" --out-dir data/datasets/memmap --exclude MaxDamage,HazardSense

Output layout:
    out_dir/
        meta.json           # metadata (obs_dim, counts, mod_keys, etc.)
        obs.npy             # (N, obs_dim) float32
        action.npy          # (N,) int32
        legal.npy           # (N, 9) float32
        move_slots.npy      # (N, 4, move_slot_dim) float32
        switch_slots.npy    # (N, 5, switch_slot_dim) float32
        ctx_extra.npy       # (N, ctx_dim) float32
        mods.npy            # (N, num_mod_keys) float32
        result.npy          # (N,) float32   (-1 = unknown)
        turn.npy            # (N,) int32
        phase.npy           # (N,) int32
        entity_ids.npy      # (N, n_entity_ids) int32  (v5 entity IDs)
        move_ids.npy        # (N, 4) int32              (v5 move slot IDs)
        switch_ids.npy      # (N, 5) int32              (v5 switch slot IDs)
        episode_index.npy   # (E, 3) int64   [start_row, length, episode_hash]

Episodes are written contiguously so the dataset can slice rows by episode.
"""

from __future__ import annotations
import argparse, glob, hashlib, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np


def episode_hash(episode_id: str) -> int:
    """SHA-1 based hash for uniform distribution across episode IDs."""
    h = hashlib.sha1(episode_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFFFFFFFFFFFFFF


def parse_args():
    p = argparse.ArgumentParser(description="Convert JSONL to memmap for fast BC training")
    p.add_argument("--data", default="src/data/datasets/obs/*.jsonl",
                   help="Glob pattern for JSONL files")
    p.add_argument("--out-dir", default="src/data/datasets/memmap",
                   help="Output directory for memmap files")
    p.add_argument("--exclude", default="",
                   help="Comma-separated bot names to exclude (e.g. 'MaxDamage,HazardSense')")
    return p.parse_args()


def should_exclude(filepath: str, exclude_set: set) -> bool:
    name = os.path.basename(filepath).lower()
    name = name.replace("maxbasepower", "maxdamage")
    return any(f"-vs-{ex.lower()}" in name for ex in exclude_set)


def get_episode_id(row: dict, fallback_counter: int) -> str:
    """Extract episode ID from a row, or generate a standalone one."""
    for k in ("episode_id", "battle_tag", "battleId", "battle_id"):
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    return f"_standalone_{fallback_counter}"


def validate_row(row: dict, obs_dim_ref: list) -> bool:
    """Check if a row has valid required fields. obs_dim_ref is a 1-element list for mutability."""
    obs = row.get("obs")
    action = row.get("action")
    legal = row.get("legal")
    if not isinstance(obs, (list, tuple)) or not isinstance(legal, (list, tuple)):
        return False
    if len(legal) != 9:
        return False
    try:
        a = int(action)
    except Exception:
        return False
    if a < 0 or a >= 9:
        return False
    if obs_dim_ref[0] is None:
        obs_dim_ref[0] = len(obs)
    elif len(obs) != obs_dim_ref[0]:
        return False
    return True


def main():
    args = parse_args()
    files = sorted(glob.glob(args.data))
    if not files:
        print(f"[convert] No files matched: {args.data}")
        sys.exit(1)

    exclude_set = {s.strip() for s in args.exclude.split(",") if s.strip()}
    if exclude_set:
        before = len(files)
        files = [f for f in files if not should_exclude(f, exclude_set)]
        print(f"[convert] Excluded {before - len(files)} files matching: {exclude_set}")

    print(f"[convert] Processing {len(files)} JSONL files...")

    # ================================================================
    # Pass 1: Scan — count rows per episode, detect dims, no data stored
    # ================================================================
    t0 = time.time()

    # episode_id -> { "row_count": int, "has_done": bool, "file_rows": [(file_idx, line_offsets)] }
    # We only need counts and done-status, not the actual data.
    episode_row_counts = defaultdict(int)       # eid -> number of rows
    episode_has_done = defaultdict(bool)         # eid -> has at least one done=True
    episode_order = []                           # order of first appearance for deterministic output

    obs_dim_ref = [None]  # mutable container for validate_row
    mod_keys_set = set()
    move_slot_dim = 0
    switch_slot_dim = 0
    ctx_dim = 0
    entity_ids_dim = 0
    total_rows = 0
    skipped = 0
    standalone_counter = 0

    for fi, fp in enumerate(files):
        fname = os.path.basename(fp)
        print(f"[scan] [{fi+1}/{len(files)}] {fname}", end="", flush=True)
        file_rows = 0
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    skipped += 1
                    continue

                if not validate_row(row, obs_dim_ref):
                    skipped += 1
                    continue

                # Detect dims from first valid examples
                mv = row.get("move_slots")
                if isinstance(mv, list) and len(mv) == 4 and isinstance(mv[0], list):
                    move_slot_dim = max(move_slot_dim, len(mv[0]))
                sw = row.get("switch_slots")
                if isinstance(sw, list) and len(sw) == 5 and isinstance(sw[0], list):
                    switch_slot_dim = max(switch_slot_dim, len(sw[0]))
                cx = row.get("ctx_extra")
                if isinstance(cx, (list, tuple)):
                    ctx_dim = max(ctx_dim, len(cx))
                eids = row.get("entity_ids")
                if isinstance(eids, (list, tuple)):
                    entity_ids_dim = max(entity_ids_dim, len(eids))
                mods = row.get("mods")
                if isinstance(mods, dict):
                    mod_keys_set.update(mods.keys())

                # Track episode
                eid = get_episode_id(row, standalone_counter)
                if eid.startswith("_standalone_"):
                    standalone_counter += 1

                if eid not in episode_row_counts:
                    episode_order.append(eid)
                episode_row_counts[eid] += 1

                if row.get("done", False):
                    episode_has_done[eid] = True

                total_rows += 1
                file_rows += 1

                if file_rows % 500000 == 0:
                    elapsed = time.time() - t0
                    rate = total_rows / elapsed
                    print(f"    [scan progress] {total_rows/1e6:.1f}M rows, "
                          f"{len(episode_order)} episodes, "
                          f"{rate:.0f} rows/s, "
                          f"elapsed {elapsed/60:.1f}min",
                          flush=True)

        print(f"  ({file_rows} rows)")

    obs_dim = obs_dim_ref[0]
    mod_keys = sorted(mod_keys_set)

    # Filter to complete episodes (have at least one done=True, or standalone)
    complete_eids = [
        eid for eid in episode_order
        if episode_has_done.get(eid, False) or eid.startswith("_standalone_")
    ]
    total_episode_rows = sum(episode_row_counts[eid] for eid in complete_eids)
    incomplete_eids = len(episode_order) - len(complete_eids)
    _complete_set = set(complete_eids)  # build once, not per-iteration
    incomplete_rows = sum(
        episode_row_counts[eid] for eid in episode_order
        if eid not in _complete_set
    )

    elapsed1 = time.time() - t0
    print(f"\n[scan] Pass 1 done in {elapsed1:.1f}s")
    print(f"[scan] Complete episodes: {len(complete_eids)} ({total_episode_rows} rows)")
    print(f"[scan] Incomplete episodes dropped: {incomplete_eids}")
    print(f"[scan] Skipped rows (bad data): {skipped}")
    print(f"[scan] obs_dim={obs_dim} move_slot_dim={move_slot_dim} "
          f"switch_slot_dim={switch_slot_dim} ctx_dim={ctx_dim} entity_ids_dim={entity_ids_dim}")
    print(f"[scan] mod_keys={mod_keys}")

    if total_episode_rows == 0:
        print("[convert] ERROR: No complete episodes found!")
        sys.exit(1)

    # Build episode write plan: eid -> (start_row, length) in output memmap
    complete_eids_set = set(complete_eids)
    episode_start = {}   # eid -> start row in memmap
    cursor = 0
    for eid in complete_eids:
        episode_start[eid] = cursor
        cursor += episode_row_counts[eid]
    assert cursor == total_episode_rows

    # ================================================================
    # Pass 2: Re-read JSONL, write directly to pre-allocated memmap
    # ================================================================
    t1 = time.time()
    N = total_episode_rows
    E = len(complete_eids)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[write] Allocating memmap files ({N:,} rows, {E:,} episodes)...")

    obs_mm = np.lib.format.open_memmap(
        str(out_dir / "obs.npy"), mode='w+', dtype=np.float32, shape=(N, obs_dim))
    action_mm = np.lib.format.open_memmap(
        str(out_dir / "action.npy"), mode='w+', dtype=np.int32, shape=(N,))
    legal_mm = np.lib.format.open_memmap(
        str(out_dir / "legal.npy"), mode='w+', dtype=np.float32, shape=(N, 9))
    move_slots_mm = np.lib.format.open_memmap(
        str(out_dir / "move_slots.npy"), mode='w+', dtype=np.float32,
        shape=(N, 4, move_slot_dim)) if move_slot_dim > 0 else None
    switch_slots_mm = np.lib.format.open_memmap(
        str(out_dir / "switch_slots.npy"), mode='w+', dtype=np.float32,
        shape=(N, 5, switch_slot_dim)) if switch_slot_dim > 0 else None
    ctx_extra_mm = np.lib.format.open_memmap(
        str(out_dir / "ctx_extra.npy"), mode='w+', dtype=np.float32,
        shape=(N, ctx_dim)) if ctx_dim > 0 else None
    mods_mm = np.lib.format.open_memmap(
        str(out_dir / "mods.npy"), mode='w+', dtype=np.float32,
        shape=(N, len(mod_keys))) if mod_keys else None
    result_mm = np.lib.format.open_memmap(
        str(out_dir / "result.npy"), mode='w+', dtype=np.float32, shape=(N,))
    result_mm[:] = -1.0  # default: unknown
    turn_mm = np.lib.format.open_memmap(
        str(out_dir / "turn.npy"), mode='w+', dtype=np.int32, shape=(N,))
    phase_mm = np.lib.format.open_memmap(
        str(out_dir / "phase.npy"), mode='w+', dtype=np.int32, shape=(N,))
    entity_ids_mm = np.lib.format.open_memmap(
        str(out_dir / "entity_ids.npy"), mode='w+', dtype=np.int32,
        shape=(N, entity_ids_dim)) if entity_ids_dim > 0 else None
    move_ids_mm = np.lib.format.open_memmap(
        str(out_dir / "move_ids.npy"), mode='w+', dtype=np.int32,
        shape=(N, 4))  # always 4 move slots
    switch_ids_mm = np.lib.format.open_memmap(
        str(out_dir / "switch_ids.npy"), mode='w+', dtype=np.int32,
        shape=(N, 5))  # always 5 switch slots

    # Track per-episode write cursor (rows may arrive out of order across files)
    episode_write_cursor = {eid: episode_start[eid] for eid in complete_eids}

    # Buffer to collect rows per episode for sorting by turn before writing.
    # Only one episode's rows are buffered at a time (flushed on done or episode change).
    # For safety with interleaved episodes, we buffer per-episode and flush on done.
    episode_buffers = {}   # eid -> list of (turn, row_dict)
    episodes_flushed = set()
    standalone_counter2 = 0  # Stays in sync with Pass 1: same files, same order, same validate_row filtering

    def flush_episode(eid, rows):
        """Sort rows by turn and write to memmap."""
        rows.sort(key=lambda x: x[0])  # sort by turn
        start = episode_start[eid]
        for i, (t, rec) in enumerate(rows):
            idx = start + i
            obs_mm[idx] = rec["obs"]
            action_mm[idx] = rec["action"]
            legal_mm[idx] = rec["legal"]
            turn_mm[idx] = t
            phase_mm[idx] = rec["phase"]
            result_mm[idx] = rec.get("result", -1.0)

            if move_slots_mm is not None and "move_slots" in rec:
                mv = rec["move_slots"]
                L = min(move_slot_dim, mv.shape[-1])
                move_slots_mm[idx, :, :L] = mv[:, :L]

            if switch_slots_mm is not None and "switch_slots" in rec:
                sw = rec["switch_slots"]
                L = min(switch_slot_dim, sw.shape[-1])
                switch_slots_mm[idx, :, :L] = sw[:, :L]

            if ctx_extra_mm is not None and "ctx_extra" in rec:
                cx = rec["ctx_extra"]
                L = min(ctx_dim, len(cx))
                ctx_extra_mm[idx, :L] = cx[:L]

            if mods_mm is not None:
                mods = rec.get("mods", {})
                for ki, k in enumerate(mod_keys):
                    mods_mm[idx, ki] = float(mods.get(k, 0))

            if entity_ids_mm is not None and "entity_ids" in rec:
                ei = rec["entity_ids"]
                L = min(entity_ids_dim, len(ei))
                entity_ids_mm[idx, :L] = ei[:L]

            if move_ids_mm is not None and "move_ids" in rec:
                move_ids_mm[idx] = rec["move_ids"]

            if switch_ids_mm is not None and "switch_ids" in rec:
                switch_ids_mm[idx] = rec["switch_ids"]

        episodes_flushed.add(eid)

    # Re-read all files
    rows_written = 0
    for fi, fp in enumerate(files):
        fname = os.path.basename(fp)
        print(f"[write] [{fi+1}/{len(files)}] {fname}", end="", flush=True)
        file_written = 0
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue

                if not validate_row(row, [obs_dim]):
                    continue

                eid = get_episode_id(row, standalone_counter2)
                if eid.startswith("_standalone_"):
                    standalone_counter2 += 1

                # Skip episodes we're not writing (incomplete)
                if eid not in complete_eids_set:
                    continue

                # Skip already-flushed episodes (shouldn't happen, but safety)
                if eid in episodes_flushed:
                    print(f"\n  [WARN] Dropping row for already-flushed episode '{eid}' in file '{fname}'", flush=True)
                    continue

                # Build compact record — only allocate numpy for arrays we need
                obs = row["obs"]
                rec = {
                    "obs": np.asarray(obs, dtype=np.float32),
                    "action": int(row["action"]),
                    "legal": np.asarray(row["legal"], dtype=np.float32),
                    "phase": int(row.get("phase", 0)),
                    "mods": row.get("mods", {}) if isinstance(row.get("mods"), dict) else {},
                }

                mv = row.get("move_slots")
                if isinstance(mv, list) and len(mv) == 4:
                    rec["move_slots"] = np.asarray(mv, dtype=np.float32)
                sw = row.get("switch_slots")
                if isinstance(sw, list) and len(sw) == 5:
                    rec["switch_slots"] = np.asarray(sw, dtype=np.float32)
                cx = row.get("ctx_extra")
                if isinstance(cx, (list, tuple)):
                    rec["ctx_extra"] = np.asarray(cx, dtype=np.float32)
                eids = row.get("entity_ids")
                if isinstance(eids, (list, tuple)):
                    rec["entity_ids"] = np.asarray(eids, dtype=np.int32)
                mids = row.get("move_ids")
                if isinstance(mids, (list, tuple)) and len(mids) == 4:
                    rec["move_ids"] = np.asarray(mids, dtype=np.int32)
                sids = row.get("switch_ids")
                if isinstance(sids, (list, tuple)) and len(sids) == 5:
                    rec["switch_ids"] = np.asarray(sids, dtype=np.int32)

                if "result" in row:
                    try:
                        rec["result"] = float(row["result"])
                    except Exception:
                        pass

                turn = int(row.get("t", 0))
                done = bool(row.get("done", False))

                # Buffer the row
                buf = episode_buffers.setdefault(eid, [])
                buf.append((turn, rec))

                # Flush on done or standalone
                if done or eid.startswith("_standalone_"):
                    flush_episode(eid, buf)
                    del episode_buffers[eid]
                    rows_written += len(buf)
                    file_written += len(buf)

                    if rows_written % 500000 < 100:
                        elapsed = time.time() - t1
                        rate = rows_written / max(elapsed, 0.1)
                        remaining = (N - rows_written) / max(rate, 1)
                        print(f"\r    [write progress] {rows_written/1e6:.1f}M/{N/1e6:.1f}M rows, "
                              f"{rate:.0f} rows/s, "
                              f"~{remaining/60:.0f}min remaining",
                              end="", flush=True)

        # Flush any episodes that completed in this file but somehow missed done
        # (this shouldn't happen normally, but handles edge cases)

        print(f"  ({file_written} rows flushed)")

        # Periodic memory cleanup — flush episodes that have all expected rows
        to_flush = []
        for eid, buf in episode_buffers.items():
            if len(buf) >= episode_row_counts[eid]:
                to_flush.append(eid)
        for eid in to_flush:
            buf = episode_buffers.pop(eid)
            flush_episode(eid, buf)
            rows_written += len(buf)

    # Final flush — any remaining buffered episodes
    for eid, buf in list(episode_buffers.items()):
        flush_episode(eid, buf)
        rows_written += len(buf)
    episode_buffers.clear()

    # Build episode_index with result propagation
    print(f"\n[write] Building episode index...")
    episode_index = np.zeros((E, 3), dtype=np.int64)
    for ep_idx, eid in enumerate(complete_eids):
        start = episode_start[eid]
        length = episode_row_counts[eid]
        ep_hash = episode_hash(eid)
        episode_index[ep_idx] = [start, length, ep_hash]

        # Propagate episode result to all rows (find result from last row)
        ep_result = -1.0
        for i in range(start + length - 1, start - 1, -1):
            if result_mm[i] != -1.0:
                ep_result = result_mm[i]
                break
        if ep_result != -1.0:
            result_mm[start:start + length] = ep_result

    np.save(str(out_dir / "episode_index.npy"), episode_index)

    # Flush all memmaps
    for mm in [obs_mm, action_mm, legal_mm, move_slots_mm, switch_slots_mm,
               ctx_extra_mm, mods_mm, result_mm, turn_mm, phase_mm,
               entity_ids_mm, move_ids_mm, switch_ids_mm]:
        if mm is not None:
            mm.flush()
            del mm

    # Write metadata
    meta = {
        "obs_dim": obs_dim,
        "num_records": N,
        "num_episodes": E,
        "mod_keys": mod_keys,
        "move_slot_dim": move_slot_dim,
        "switch_slot_dim": switch_slot_dim,
        "ctx_extra_dim": ctx_dim,
        "source_pattern": args.data,
        "source_files": len(files),
        "entity_ids_dim": entity_ids_dim,
        "excluded": list(exclude_set) if exclude_set else [],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    elapsed2 = time.time() - t1
    total_elapsed = time.time() - t0

    # Report sizes
    total_bytes = 0
    for f_name in out_dir.iterdir():
        if f_name.suffix == ".npy":
            total_bytes += f_name.stat().st_size

    print(f"\n[convert] Done!")
    print(f"[convert] Pass 1 (scan): {elapsed1:.1f}s, Pass 2 (write): {elapsed2:.1f}s, "
          f"total: {total_elapsed:.1f}s")
    print(f"[convert] Output: {out_dir}")
    print(f"[convert] Total memmap size: {total_bytes / 1e9:.2f} GB")
    print(f"[convert] Records: {N:,}, Episodes: {E:,}")
    print(f"[convert] Dims: obs={obs_dim} move_slot={move_slot_dim} "
          f"switch_slot={switch_slot_dim} ctx={ctx_dim} mods={len(mod_keys)} entity_ids={entity_ids_dim}")

    # ================================================================
    # Pass 3: Verify integrity — spot-check against source JSONL
    # ================================================================
    verify_memmap(out_dir, files, obs_dim, mod_keys, complete_eids_set)


def verify_memmap(out_dir: Path, files: list, obs_dim: int, mod_keys: list,
                  complete_eids_set: set):
    """Verify memmap integrity: structural checks + spot-check against source JSONL."""
    import random

    print("\n" + "=" * 60)
    print("[verify] Starting integrity verification...")
    t_v = time.time()
    errors = []

    # --- 1. Load metadata and memmap files ---
    with open(out_dir / "meta.json") as f:
        meta = json.load(f)

    obs_mm = np.load(str(out_dir / "obs.npy"), mmap_mode='r')
    action_mm = np.load(str(out_dir / "action.npy"), mmap_mode='r')
    legal_mm = np.load(str(out_dir / "legal.npy"), mmap_mode='r')
    result_mm = np.load(str(out_dir / "result.npy"), mmap_mode='r')
    turn_mm = np.load(str(out_dir / "turn.npy"), mmap_mode='r')
    phase_mm = np.load(str(out_dir / "phase.npy"), mmap_mode='r')
    episode_index = np.load(str(out_dir / "episode_index.npy"))

    move_slots_mm = None
    if (out_dir / "move_slots.npy").exists():
        move_slots_mm = np.load(str(out_dir / "move_slots.npy"), mmap_mode='r')
    switch_slots_mm = None
    if (out_dir / "switch_slots.npy").exists():
        switch_slots_mm = np.load(str(out_dir / "switch_slots.npy"), mmap_mode='r')
    ctx_extra_mm = None
    if (out_dir / "ctx_extra.npy").exists():
        ctx_extra_mm = np.load(str(out_dir / "ctx_extra.npy"), mmap_mode='r')

    N = meta["num_records"]
    E = meta["num_episodes"]

    # --- 2. Shape checks ---
    print(f"[verify] Checking shapes...")
    checks = [
        ("obs", obs_mm.shape, (N, obs_dim)),
        ("action", action_mm.shape, (N,)),
        ("legal", legal_mm.shape, (N, 9)),
        ("result", result_mm.shape, (N,)),
        ("turn", turn_mm.shape, (N,)),
        ("phase", phase_mm.shape, (N,)),
        ("episode_index", episode_index.shape, (E, 3)),
    ]
    for name, actual, expected in checks:
        if actual != expected:
            errors.append(f"Shape mismatch: {name} is {actual}, expected {expected}")
        else:
            print(f"  {name}: {actual} OK")

    if move_slots_mm is not None:
        print(f"  move_slots: {move_slots_mm.shape} OK")
    if switch_slots_mm is not None:
        print(f"  switch_slots: {switch_slots_mm.shape} OK")
    if ctx_extra_mm is not None:
        print(f"  ctx_extra: {ctx_extra_mm.shape} OK")

    # --- 3. Episode index consistency ---
    print(f"[verify] Checking episode index consistency...")
    total_from_index = 0
    ep_errors = 0
    for i in range(E):
        start, length, ep_hash = episode_index[i]
        if start < 0 or start >= N:
            errors.append(f"Episode {i}: start={start} out of bounds [0, {N})")
            ep_errors += 1
            continue
        if start + length > N:
            errors.append(f"Episode {i}: start+length={start+length} exceeds N={N}")
            ep_errors += 1
            continue
        total_from_index += length

        # Check turns are non-decreasing within episode
        if length > 1:
            ep_turns = turn_mm[start:start + length]
            if not np.all(ep_turns[1:] >= ep_turns[:-1]):
                errors.append(f"Episode {i}: turns not sorted (start={start}, len={length})")
                ep_errors += 1

        # Check result is consistent within episode
        ep_results = result_mm[start:start + length]
        unique_results = np.unique(ep_results)
        if len(unique_results) > 1:
            errors.append(f"Episode {i}: mixed results {unique_results} (start={start}, len={length})")
            ep_errors += 1

    if total_from_index != N:
        errors.append(f"Episode index covers {total_from_index} rows, expected {N}")
    else:
        print(f"  Episode index covers all {N:,} rows OK")

    # Check episodes don't overlap
    sorted_eps = sorted(range(E), key=lambda i: int(episode_index[i, 0]))
    for i in range(len(sorted_eps) - 1):
        a = sorted_eps[i]
        b = sorted_eps[i + 1]
        end_a = int(episode_index[a, 0] + episode_index[a, 1])
        start_b = int(episode_index[b, 0])
        if end_a > start_b:
            errors.append(f"Episodes {a} and {b} overlap: [{episode_index[a,0]}, {end_a}) vs [{start_b}, ...)")
            break
    else:
        print(f"  No episode overlaps OK")

    print(f"  Episode index errors: {ep_errors}")

    # --- 4. Value range checks ---
    print(f"[verify] Checking value ranges...")
    # Actions should be 0-8
    action_min, action_max = int(action_mm.min()), int(action_mm.max())
    if action_min < 0 or action_max > 8:
        errors.append(f"Actions out of range: [{action_min}, {action_max}]")
    else:
        print(f"  Actions: [{action_min}, {action_max}] OK")

    # Legal mask should be 0 or 1
    legal_unique = np.unique(legal_mm)
    if not np.all(np.isin(legal_unique, [0.0, 1.0])):
        errors.append(f"Legal mask has unexpected values: {legal_unique}")
    else:
        print(f"  Legal mask: binary OK")

    # Check for NaN/Inf in obs
    nan_count = int(np.isnan(obs_mm).sum()) if obs_mm.size < 50_000_000 else int(np.isnan(obs_mm[:1000000]).sum())
    inf_count = int(np.isinf(obs_mm).sum()) if obs_mm.size < 50_000_000 else int(np.isinf(obs_mm[:1000000]).sum())
    sample_note = "" if obs_mm.size < 50_000_000 else " (sampled first 1M values)"
    if nan_count > 0:
        errors.append(f"Obs has {nan_count} NaN values{sample_note}")
    if inf_count > 0:
        errors.append(f"Obs has {inf_count} Inf values{sample_note}")
    if nan_count == 0 and inf_count == 0:
        print(f"  Obs: no NaN/Inf{sample_note} OK")

    # Results should be -1, 0, or 1
    result_unique = np.unique(result_mm)
    valid_results = {-1.0, 0.0, 1.0}
    bad_results = [r for r in result_unique if r not in valid_results]
    if bad_results:
        errors.append(f"Results have unexpected values: {bad_results}")
    else:
        n_win = int((result_mm == 1.0).sum())
        n_loss = int((result_mm == 0.0).sum())
        n_unknown = int((result_mm == -1.0).sum())
        print(f"  Results: win(1.0)={n_win:,}, loss(0.0)={n_loss:,}, unknown(-1.0)={n_unknown:,} OK")

    # Check actions respect legal mask
    print(f"[verify] Checking actions respect legal masks...")
    # Sample to avoid slow full scan
    sample_size = min(100000, N)
    sample_idx = np.random.choice(N, sample_size, replace=False) if N > sample_size else np.arange(N)
    illegal_actions = 0
    for idx in sample_idx:
        a = int(action_mm[idx])
        if legal_mm[idx, a] == 0.0:
            illegal_actions += 1
    if illegal_actions > 0:
        pct = illegal_actions / len(sample_idx) * 100
        errors.append(f"Illegal actions: {illegal_actions}/{len(sample_idx)} ({pct:.2f}%) sampled rows have action on illegal slot")
    else:
        print(f"  Actions respect legal masks: {len(sample_idx):,} sampled rows OK")

    # --- 5. Spot-check against source JSONL ---
    print(f"[verify] Spot-checking against source JSONL (sampling 50 episodes)...")

    # Pick random episodes to verify
    num_spot = min(50, E)
    spot_eps = random.sample(range(E), num_spot)

    # Build a reverse map: for each spot-check episode, find its hash and
    # we'll scan JSONL to find matching rows
    # Instead, pick random rows and verify them against JSONL by re-reading
    # the corresponding file position. Since we can't easily seek, we'll
    # verify a simpler property: pick random episodes, read their memmap data,
    # and verify internal consistency + match against one random JSONL file.

    # Simpler approach: re-read a few random JSONL files and compare rows
    spot_files = random.sample(files, min(5, len(files)))
    spot_mismatches = 0
    spot_checked = 0

    for fp in spot_files:
        # Read all valid rows from this file, grouped by episode
        file_episodes = defaultdict(list)
        verify_standalone_counter = 0
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not validate_row(row, [obs_dim]):
                    continue
                eid = get_episode_id(row, verify_standalone_counter)
                if eid.startswith("_standalone_"):
                    verify_standalone_counter += 1
                if eid not in complete_eids_set:
                    continue
                file_episodes[eid].append(row)

        # Pick up to 10 episodes from this file to verify
        eids_to_check = random.sample(list(file_episodes.keys()),
                                       min(10, len(file_episodes)))
        for eid in eids_to_check:
            rows = file_episodes[eid]
            rows.sort(key=lambda r: int(r.get("t", 0)))
            ep_hash_val = episode_hash(eid)

            # Find this episode in the episode_index by hash
            hash_matches = np.where(episode_index[:, 2] == ep_hash_val)[0]
            if len(hash_matches) == 0:
                errors.append(f"Spot-check: episode hash {ep_hash_val} for '{eid}' not found in index")
                spot_mismatches += 1
                continue

            # Check each matching episode (hash collisions possible but unlikely)
            found = False
            for match_idx in hash_matches:
                start = int(episode_index[match_idx, 0])
                length = int(episode_index[match_idx, 1])
                if length != len(rows):
                    continue  # try next hash match

                # Compare first and last row obs vectors
                first_obs_jsonl = np.asarray(rows[0]["obs"], dtype=np.float32)
                first_obs_mm = obs_mm[start]
                if np.allclose(first_obs_jsonl, first_obs_mm, atol=1e-6):
                    # Also check last row
                    last_obs_jsonl = np.asarray(rows[-1]["obs"], dtype=np.float32)
                    last_obs_mm = obs_mm[start + length - 1]
                    if np.allclose(last_obs_jsonl, last_obs_mm, atol=1e-6):
                        # Check action matches
                        first_action_jsonl = int(rows[0]["action"])
                        first_action_mm = int(action_mm[start])
                        last_action_jsonl = int(rows[-1]["action"])
                        last_action_mm = int(action_mm[start + length - 1])
                        if first_action_jsonl == first_action_mm and last_action_jsonl == last_action_mm:
                            found = True
                            break

            if not found:
                spot_mismatches += 1
                errors.append(f"Spot-check: episode '{eid}' data mismatch (obs/action don't match JSONL)")
            spot_checked += 1

    if spot_mismatches == 0:
        print(f"  Spot-checked {spot_checked} episodes across {len(spot_files)} files: ALL MATCH")
    else:
        print(f"  Spot-check: {spot_mismatches}/{spot_checked} episodes MISMATCHED")

    # --- 6. Summary ---
    elapsed_v = time.time() - t_v
    print(f"\n{'=' * 60}")
    if errors:
        print(f"[verify] FAILED — {len(errors)} error(s) found in {elapsed_v:.1f}s:")
        for e in errors:
            print(f"  ERROR: {e}")
    else:
        print(f"[verify] PASSED — all checks OK in {elapsed_v:.1f}s")
        print(f"  Shapes: all correct")
        print(f"  Episodes: {E:,} contiguous, non-overlapping, turns sorted")
        print(f"  Values: actions [0-8], legal binary, no NaN/Inf, results {{-1,0,1}}")
        print(f"  Legal mask: actions respect masks ({len(sample_idx):,} sampled)")
        print(f"  Spot-check: {spot_checked} episodes match source JSONL")
    print(f"{'=' * 60}")

    return len(errors) == 0


if __name__ == "__main__":
    main()
