#!/usr/bin/env python
# subset_memmap.py — Extract the first N episodes from a v8 memmap dir.
#
# Use case (S68): Cloudflare R2 has the full 104 GB human_v8_100k memmap
# that BC v10 was trained on. Prod container disk only has ~46 GB free.
# We don't need the full 100k episodes for AWR — sampling 16 episodes/iter
# × 150 iters = 2400 samples needs at most ~3k distinct episodes for
# good variety (even with replacement).
#
# This script:
#   1. Reads source memmap_dir's metadata.json + episode_index.npy
#   2. Slices first N episodes (= sum of their record lengths)
#   3. For each per-record .npy file, copies the first N_records rows
#      to a new .npy in the output dir
#   4. Writes a new metadata.json + new episode_index.npy reflecting
#      only the kept episodes
#
# Memory profile: each .npy is sliced one-at-a-time via np.memmap; max
# RAM spike = largest sliced array's bytes. For 5k episodes of
# our_pokemon_cont (full = 34 GB / 100k eps), subset is ~1.7 GB.
# Fine on prod (232 GB cgroup).
#
# Usage:
#   python subset_memmap.py --src /dev/shm/human_v8_full \
#       --dst /workspace/.../human_v8_5k --n-episodes 5000

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


# Layout schema mirrors dataset.py:84-103 (_open_memmaps).
# (filename_stem, per_record_shape_after_N_dim, dtype)
# N_dim is the leading record dimension; we slice along axis 0.
_FILE_SCHEMA = [
    ("our_pokemon_ids",    (6, 7),         np.int32),
    ("our_pokemon_banks",  (6, 10),        np.int32),
    # 'D' = pokemon_cont_dim from metadata
    ("our_pokemon_cont",   ("6", "PCD"),   np.float32),
    ("our_pokemon_mcont",  (6, 4, 23),     np.float32),
    ("opp_pokemon_ids",    (6, 7),         np.int32),
    ("opp_pokemon_banks",  (6, 10),        np.int32),
    ("opp_pokemon_cont",   ("6", "PCD"),   np.float32),
    ("opp_pokemon_mcont",  (6, 4, 23),     np.float32),
    ("field_banks",        (4,),           np.int32),
    ("field_cont",         ("FCD",),       np.float32),
    ("trans_ids",          (2,),           np.int32),
    ("trans_cont",         ("TCD",),       np.float32),
    ("move_ids",           (4,),           np.int32),
    ("move_banks",         (4, 4),         np.int32),
    ("move_cont",          ("4", "MCD"),   np.float32),
    ("switch_ids",         (5,),           np.int32),
    ("switch_cont",        ("5", "SCD"),   np.float32),
    ("legal",              (9,),           np.float32),
    ("action",             (),             np.int32),
    ("result",             (),             np.float32),
    ("turn",               (),             np.int32),
]


def _resolve_shape(tmpl, meta):
    """Replace string placeholders in shape tuple with metadata dims."""
    sub = {
        "PCD": meta["poke_cont_dim"],
        "FCD": meta["field_cont_dim"],
        "TCD": meta["trans_cont_dim"],
        "MCD": meta["move_cont_dim"],
        "SCD": meta["switch_cont_dim"],
    }
    return tuple(int(sub.get(d, d)) for d in tmpl)


def main():
    p = argparse.ArgumentParser(description="Subset a v8 memmap dir to first N episodes")
    p.add_argument("--src", required=True, help="source memmap dir (has metadata.json + .npy)")
    p.add_argument("--dst", required=True, help="output memmap dir (created)")
    p.add_argument("--n-episodes", type=int, required=True, help="how many episodes to keep")
    p.add_argument("--dry-run", action="store_true",
                   help="print plan + sizes without writing output")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    # 1. Load source metadata + episode_index.
    meta_path = src / "metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"missing metadata.json at {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"src metadata: version={meta.get('version')}, "
          f"num_episodes={meta['num_episodes']:,}, "
          f"num_records={meta['num_records']:,}")
    if meta.get("version") != "v8":
        raise SystemExit(f"expected v8, got {meta.get('version')}")

    ep_index_path = src / "episode_index.npy"
    ep_index = np.load(str(ep_index_path))
    print(f"episode_index: shape={ep_index.shape}, dtype={ep_index.dtype}")
    if ep_index.shape[0] != meta["num_episodes"]:
        raise SystemExit(
            f"episode_index rows ({ep_index.shape[0]}) != metadata num_episodes "
            f"({meta['num_episodes']})"
        )

    N_ep_target = args.n_episodes
    if N_ep_target >= meta["num_episodes"]:
        raise SystemExit(
            f"--n-episodes {N_ep_target} >= source num_episodes "
            f"{meta['num_episodes']}; nothing to do"
        )

    # 2. Compute how many records to keep.
    # episode_index format from dataset.py: [start, length, hash] per row.
    # We keep first N_ep_target episodes whose record range is [0, last_record_end).
    keep_starts  = ep_index[:N_ep_target, 0].astype(np.int64)
    keep_lengths = ep_index[:N_ep_target, 1].astype(np.int64)
    last_end = int(keep_starts[-1] + keep_lengths[-1])
    n_records_keep = last_end
    print(f"keeping first {N_ep_target:,} episodes -> {n_records_keep:,} records "
          f"({100 * n_records_keep / meta['num_records']:.1f}% of source)")

    # Sanity: assert episodes are contiguous in record space (start[i] == end[i-1]).
    expected_starts = np.zeros(N_ep_target, dtype=np.int64)
    expected_starts[1:] = np.cumsum(keep_lengths[:-1])
    if not np.array_equal(keep_starts, expected_starts):
        # Non-contiguous (gaps between episodes) is theoretically possible if
        # the memmap was built from multiple sources. Fall back to dropping
        # non-contiguous tail — keep up to first gap.
        diffs = keep_starts - expected_starts
        first_bad = int(np.argmax(diffs != 0))
        print(f"WARNING: episodes are NOT contiguous starting at episode "
              f"{first_bad}. Truncating to {first_bad} contiguous episodes.")
        N_ep_target = first_bad
        keep_starts = keep_starts[:N_ep_target]
        keep_lengths = keep_lengths[:N_ep_target]
        last_end = int(keep_starts[-1] + keep_lengths[-1])
        n_records_keep = last_end

    # 3. Plan + size accounting.
    total_src_bytes = 0
    total_dst_bytes = 0
    plan = []
    for name, shape_tmpl, dtype in _FILE_SCHEMA:
        rec_shape = _resolve_shape(shape_tmpl, meta) if shape_tmpl else ()
        full_shape = (meta["num_records"],) + rec_shape
        sub_shape  = (n_records_keep,)   + rec_shape
        bytes_per_rec = int(np.dtype(dtype).itemsize * np.prod(rec_shape)
                            if rec_shape else np.dtype(dtype).itemsize)
        src_bytes = bytes_per_rec * meta["num_records"]
        dst_bytes = bytes_per_rec * n_records_keep
        total_src_bytes += src_bytes
        total_dst_bytes += dst_bytes
        plan.append((name, full_shape, sub_shape, dtype, src_bytes, dst_bytes))

    print()
    print(f"{'file':<24} {'src_size':>12} {'dst_size':>12}")
    for name, _, _, _, src_b, dst_b in plan:
        print(f"  {name:<22} {src_b/1e9:>9.2f} GB {dst_b/1e9:>9.2f} GB")
    print(f"  {'TOTAL':<22} {total_src_bytes/1e9:>9.2f} GB "
          f"{total_dst_bytes/1e9:>9.2f} GB")
    print()

    if args.dry_run:
        print("--dry-run set; exiting without writing")
        return

    # 4. Create dst dir + copy each file's first n_records_keep rows.
    dst.mkdir(parents=True, exist_ok=True)

    for name, full_shape, sub_shape, dtype, src_b, dst_b in plan:
        src_path = src / f"{name}.npy"
        dst_path = dst / f"{name}.npy"
        if not src_path.exists():
            raise SystemExit(f"missing source file: {src_path}")

        t0 = time.time()
        # CRITICAL: R2 source .npy files are RAW BYTES (no .npy header) —
        # MemmapDataset uses np.memmap(...) which assumes the same.
        # If we used np.save() here, headers would be added and MemmapDataset
        # would read header bytes as data -> garbage IDs -> CUDA OOB asserts.
        # Confirmed empirically during S68 Phase 2D.
        src_arr = np.memmap(str(src_path), dtype=dtype, mode="r",
                            shape=full_shape)
        sub_view = src_arr[:n_records_keep]
        sub_copy = np.ascontiguousarray(sub_view)
        sub_copy.tofile(str(dst_path))  # raw bytes, no .npy header
        # Release source memmap immediately to avoid holding it across files
        del src_arr, sub_view, sub_copy
        dt = time.time() - t0
        actual_size = os.path.getsize(dst_path)
        rate_mb = (actual_size / 1e6) / max(dt, 1e-3)
        print(f"  wrote {name}.npy ({actual_size/1e9:.2f} GB) in {dt:.1f}s "
              f"({rate_mb:.0f} MB/s)")

    # 5. New episode_index.npy — keep just the first N_ep_target rows.
    # ALSO raw bytes (no .npy header) to match R2 format + MemmapDataset.
    new_ep_index = np.ascontiguousarray(ep_index[:N_ep_target])
    new_ep_index.tofile(str(dst / "episode_index.npy"))
    print(f"  wrote episode_index.npy ({new_ep_index.shape})")

    # 6. New metadata.json (preserve dims, update counts).
    new_meta = dict(meta)
    new_meta["num_records"] = n_records_keep
    new_meta["num_episodes"] = N_ep_target
    new_meta["subset_of"] = str(src)
    new_meta["subset_created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(dst / "metadata.json", "w") as f:
        json.dump(new_meta, f, indent=2)
    print(f"  wrote metadata.json (num_episodes={N_ep_target:,}, "
          f"num_records={n_records_keep:,})")

    print()
    print(f"DONE. Subset at {dst} ({total_dst_bytes/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
