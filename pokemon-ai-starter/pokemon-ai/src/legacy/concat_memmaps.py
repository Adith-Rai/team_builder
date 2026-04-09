#!/usr/bin/env python3
"""Concatenate two memmap datasets into a combined one.

Usage:
    python concat_memmaps.py --dirs dir1 dir2 [dir3 ...] --out combined_dir

Each input dir must have the standard memmap files (obs.npy, action.npy, etc.)
plus meta.json. Output will be a new memmap dir with all records concatenated
and episode_index recomputed with correct offsets.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ARRAY_FILES = [
    "obs", "action", "legal", "result", "turn", "phase",
    "ctx_extra", "mods", "entity_ids", "move_ids", "switch_ids",
    "move_slots", "switch_slots",
]


def load_meta(d: str) -> dict:
    with open(os.path.join(d, "meta.json")) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(description="Concatenate memmap datasets")
    p.add_argument("--dirs", nargs="+", required=True, help="Input memmap directories")
    p.add_argument("--out", required=True, help="Output directory")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load all metas and validate compatibility
    metas = []
    for d in args.dirs:
        if not os.path.isfile(os.path.join(d, "meta.json")):
            print(f"ERROR: {d}/meta.json not found")
            sys.exit(1)
        metas.append(load_meta(d))

    obs_dim = metas[0]["obs_dim"]
    for i, m in enumerate(metas):
        if m["obs_dim"] != obs_dim:
            print(f"ERROR: obs_dim mismatch: dir0={obs_dim}, dir{i}={m['obs_dim']}")
            sys.exit(1)

    total_records = sum(m["num_records"] for m in metas)
    total_episodes = sum(m["num_episodes"] for m in metas)
    print(f"Concatenating {len(args.dirs)} dirs: {total_records:,} records, {total_episodes:,} episodes")

    # Concatenate each array file
    t0 = time.time()
    for name in ARRAY_FILES:
        arrays = []
        for d in args.dirs:
            path = os.path.join(d, f"{name}.npy")
            if os.path.isfile(path):
                arr = np.load(path, mmap_mode="r")
                arrays.append(arr)
            else:
                print(f"  WARN: {path} not found, skipping")
                break
        else:
            combined = np.concatenate(arrays, axis=0)
            out_path = os.path.join(args.out, f"{name}.npy")
            np.save(out_path, combined)
            print(f"  {name}: {combined.shape} -> {out_path} ({combined.nbytes / 1e9:.1f} GB)")
            del combined
        del arrays

    # Rebuild episode_index with corrected offsets
    print("  Rebuilding episode_index...")
    offset = 0
    all_episodes = []
    for d in args.dirs:
        ep = np.load(os.path.join(d, "episode_index.npy"), mmap_mode="r")
        for row in ep:
            start, length = int(row[0]), int(row[1])
            extra = int(row[2]) if row.shape[-1] > 2 else 0
            all_episodes.append([start + offset, length, extra])
        offset += load_meta(d)["num_records"]

    ep_combined = np.array(all_episodes, dtype=np.int64)
    np.save(os.path.join(args.out, "episode_index.npy"), ep_combined)
    print(f"  episode_index: {ep_combined.shape}")

    # Write combined meta
    combined_meta = dict(metas[0])  # Copy first meta as template
    combined_meta["num_records"] = total_records
    combined_meta["num_episodes"] = total_episodes
    combined_meta["source_dirs"] = args.dirs
    combined_meta["created"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(combined_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s: {total_records:,} records, {total_episodes:,} episodes -> {args.out}")


if __name__ == "__main__":
    main()
