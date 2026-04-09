#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behavior Cloning trainer (production-ready).

Two dataset modes:
  1) stream (row-wise): memory-light, shuffles individual rows
  2) stream + --seq-mode: groups by episode_id, yields full sequences [B,T,*]

Original features preserved:
- Episode-aware StreamingEpisodeDataset
- Row-wise StreamingJSONLDataset with shuffle-reservoir
- Sequence collate with padding and legal-mask-aware CE
- Modifier BCE (gated on “is a MOVE”)
- TensorBoard logging, progress scanning, best checkpoint by val_loss

Added/fixed features (this drop-in):
- FIX: ctx_extra now flows end-to-end (extract_sample preserves it; collates handle sizing + dropout)
- FIX: episode terminal result is propagated so value head can learn/eval in seq mode
- AMP mixed precision (--amp, default: ON) with GradScaler
- Determinism controls (--seed, --deterministic)
- DataLoader throughput: persistent_workers, prefetch_factor, pin-memory
- Cosine LR schedule with warmup (--sched cosine, --warmup-steps)
- Train/val split modes (default: chronological here, can use hash_episode)
- OOD holdouts by opponent style (default: MaxDamage & HazardSense excluded from train/val)
- CSV run log next to checkpoints (metrics.csv)
- Ensure policy_cfg carries action_dim in checkpoints
- Configurable AdamW weight decay via --weight-decay (default 0.01)
- Exponential Moving Average of weights:
    * --ema <decay in (0,1)>; 0 disables EMA
    * --use-ema-for-eval (default: True) temporarily swaps EMA weights for validation
    * EMA state saved in checkpoints; loaded on --resume/--init-from when enabled
- Label smoothing for legal actions via --label-smoothing (default 0.0)
- EMA warmup via --ema-warmup-steps (don’t update EMA for first N optimizer steps)
- Dual validation metrics: raw vs EMA in the same epoch
- Top-K symlinks (best by val_loss) + latest symlink via --topk (default 3)
- Hard-turn mining hooks (--hardmine-pct, --hardmine-weight) – light-touch, off by default.
"""

from __future__ import annotations
from datetime import datetime
import argparse, glob, io, json, math, os, random, sys, time, csv, zlib, heapq
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Callable
from contextlib import nullcontext

import numpy as np
_arraylike = (list, tuple, np.ndarray)  # for isinstance checks in collate (supports cached numpy)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
torch.set_float32_matmul_precision("high")

# --- Your policy heads (kept) ---
from policy_heads import BattlePolicy, PolicyConfig, ModifierSpec
from features import step_type_from_pos, step_type_from_abs_t

# ============================================================
#                       UTILITIES
# Global knob for collates (set in main from args)
_GLOBAL_STEP_TYPE_BINS = 3
# ============================================================

def coerce_int(x) -> Optional[int]:
    try: return int(x)
    except Exception:
        try: return int(float(x))
        except Exception: return None

def iter_jsonl(files: List[str], report_every: int = 5000, strict: bool = False) -> Iterator[Dict[str, Any]]:
    count = 0
    for fp in files:
        print(f"[BC][scan] starting: {os.path.basename(fp)}", flush=True)
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try: row = json.loads(line)
                except Exception:
                    if strict: raise
                    continue
                yield row
                count += 1
                if report_every and (count % report_every == 0):
                    print(f"[BC][scan] ... {count} rows", flush=True)
    print(f"[BC][scan] done. total rows seen: {count}", flush=True)

def extract_sample(row: Dict[str, Any], obs_dim_override: int = 0) -> Optional[Dict[str, Any]]:
    """
    Normalize a training sample from a raw JSONL row.
    Required keys: 'obs' (list/array of floats), 'action' (int 0..8), 'legal' (len=9)
    Optional: 'mods' (dict of modifier_name -> 0/1), 'ctx_extra' (list/array)
    Returns None if row is unusable.
    """
    try:
        obs = row["obs"]
        action = int(row["action"])
        legal = row["legal"]
    except Exception:
        return None
    if not isinstance(obs, (list, tuple)) or not isinstance(legal, (list, tuple)):
        return None
    if len(legal) != 9:
        return None
    if action < 0 or action >= 9:
        return None
    if obs_dim_override and len(obs) != obs_dim_override:
        return None

    mods = row.get("mods", {}) if isinstance(row.get("mods"), dict) else {}
    out = {"obs": obs, "action": action, "legal": legal, "mods": mods}

    # carry ctx_extra through if present so collate/seq can use it.
    cx = row.get("ctx_extra", None)
    if isinstance(cx, (list, tuple)):
        out["ctx_extra"] = list(cx)

    # carry move/switch slot tensors through if present
    mv = row.get("move_slots", None)
    sw = row.get("switch_slots", None)
    if isinstance(mv, (list, tuple)) and len(mv) == 4 and isinstance(mv[0], (list, tuple)):
        out["move_slots"] = mv
    if isinstance(sw, (list, tuple)) and len(sw) == 5 and isinstance(sw[0], (list, tuple)):
        out["switch_slots"] = sw

    # v5 entity IDs (integer IDs for embeddings)
    eids = row.get("entity_ids", None)
    if isinstance(eids, (list, tuple)):
        out["entity_ids"] = eids
    mids = row.get("move_ids", None)
    if isinstance(mids, (list, tuple)) and len(mids) == 4:
        out["move_ids"] = mids
    sids = row.get("switch_ids", None)
    if isinstance(sids, (list, tuple)) and len(sids) == 5:
        out["switch_ids"] = sids

    # carry terminal result (if per-row logged; seq collate will broadcast)
    if "result" in row:
        try:
            out["result"] = float(row["result"])
        except Exception:
            pass

    return out

def detect_modifier_keys(files: List[str], limit_rows: int = 5000) -> List[str]:
    seen = set(); scanned = 0
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if scanned >= limit_rows: break
                scanned += 1
                try: row = json.loads(line)
                except Exception: continue
                mods = row.get("mods")
                if isinstance(mods, dict):
                    for k, v in mods.items():
                        if isinstance(k, str) and (isinstance(v, (int, float, bool))):
                            seen.add(k)
        if scanned >= limit_rows: break
    return sorted(seen)
    
def _infer_slot_and_ctx_dims_from_loader(train_loader) -> tuple[int, int, int]:
    """
    Returns (move_slot_dim, switch_slot_dim, ctx_dim) observed in the first non-empty batch.
    Falls back to 0 if tensors are missing.
    """
    mv_dim = sw_dim = cx_dim = 0
    for batch in train_loader:
        # move / switch slots
        if "move_slots" in batch and hasattr(batch["move_slots"], "shape"):
            mv_dim = int(batch["move_slots"].shape[-1])
        if "switch_slots" in batch and hasattr(batch["switch_slots"], "shape"):
            sw_dim = int(batch["switch_slots"].shape[-1])
        # ctx_extra (row mode: [B,D], seq: [B,T,D])
        if "ctx_extra" in batch and hasattr(batch["ctx_extra"], "shape"):
            cx_shape = batch["ctx_extra"].shape
            cx_dim = int(cx_shape[-1]) if len(cx_shape) >= 2 else 0
        break  # only need the first batch
    return mv_dim, sw_dim, cx_dim

def row_episode_id(row: Dict[str, Any]) -> Optional[str]:
    for k in ("episode_id", "battle_tag", "battleId", "battle_id"):
        v = row.get(k, None)
        if isinstance(v, str) and v: return v
    return None

def row_step_index(row: Dict[str, Any]) -> Optional[int]:
    for k in ("t", "turn", "step"):
        v = row.get(k, None)
        if v is None: continue
        out = coerce_int(v)
        if out is not None: return out
    return None

def row_done(row: Dict[str, Any]) -> bool:
    v = row.get("done", None)
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(int(v))
    if isinstance(v, str): return v.strip().lower() in ("1","true","yes","y")
    return ("result" in row) or ("winner" in row)

def _hash_to_unit(s: str) -> float:
    return (zlib.adler32(s.encode("utf-8")) & 0xFFFFFFFF) / 0xFFFFFFFF

def is_heldout_shard(path: str, holdouts: set) -> bool:
    name = os.path.basename(path).lower()
    name = name.replace("maxbasepower", "maxdamage")
    return any(f"-vs-{h.lower()}" in name for h in holdouts)

def split_files(files: List[str], split_mode: str, val_ratio: float) -> Tuple[List[str], List[str]]:
    """
    Returns (files_train, files_val) with guarantees:
      - If len(files) == 0: returns ([], []).
      - If len(files) == 1: returns (files, files) so the dataset can do episode-level split.
      - If len(files) >= 2: validation is non-empty (at least 1 file), and train is non-empty.
    Behavior by mode:
      - chronological: sort by mtime; last N go to val (N = max(1, round(len*val_ratio))).
      - hash_file:     hash(basename) into [0,1); <val_ratio -> val. If empty, force 1 val.
      - hash_episode:  returns (files, files); dataset does per-episode split.
    """
    import hashlib
    from pathlib import Path

    if not files:
        return [], []

    # De-dup and stable sort by path for reproducibility before mode-specific logic
    files = sorted({str(Path(f)) for f in files})

    if split_mode == "hash_episode":
        # Dataset will split episodes by episode_id hash; give it all files for both folds.
        return files, files

    if len(files) == 1:
        # Single-file case: let the dataset do episode-level split
        return files, files

    if split_mode == "chronological":
        # Sort by mtime ascending; take the newest N as validation
        stat_list = []
        for f in files:
            try:
                mtime = Path(f).stat().st_mtime
            except Exception:
                mtime = 0.0
            stat_list.append((mtime, f))
        stat_list.sort(key=lambda x: (x[0], x[1]))  # (mtime, path) asc
        sorted_files = [f for _, f in stat_list]

        val_count = max(1, int(round(len(sorted_files) * float(val_ratio))))
        val_files = sorted_files[-val_count:]
        train_files = sorted_files[:-val_count]

        # Ensure both sides non-empty
        if not train_files:
            train_files = sorted_files[:-1]
            val_files = sorted_files[-1:]
        return train_files, val_files

    if split_mode == "hash_file":
        # Deterministic bucket per file name
        def hunit(p: str) -> float:
            h = hashlib.sha1(Path(p).name.encode("utf-8")).digest()
            v = int.from_bytes(h[:8], "big") / 2**64
            return v

        val_mask = [hunit(f) < float(val_ratio) for f in files]
        val_files = [f for f, m in zip(files, val_mask) if m]
        train_files = [f for f, m in zip(files, val_mask) if not m]

        # Guarantee at least one val and one train (when possible)
        if not val_files:
            val_files = [files[-1]]
            train_files = [f for f in files if f not in val_files]
        if not train_files:
            train_files = [files[0]]
            val_files = [f for f in files if f not in train_files]

        return train_files, val_files

    # Default: treat as chronological
    return split_files(files, "chronological", val_ratio)

# ============================================================
#                      DATASETS
# ============================================================

class StreamingJSONLDataset(IterableDataset):
    def __init__(self, files: List[str], shuffle_buffer: int, report_every: int,
                 max_rows: int = 0, strict_json: bool = False,
                 split_mode: str = "hash_episode", val_ratio: float = 0.1, want_val: bool = False):
        super().__init__()
        self.files = files
        self.shuffle_buffer = shuffle_buffer
        self.report_every = report_every
        self.max_rows = max_rows
        self.strict_json = strict_json
        self._epoch = 0
        self.obs_dim = 0
        self.split_mode = split_mode
        self.val_ratio = float(val_ratio)
        self.want_val = bool(want_val)
        self._file_fold = {f: (_hash_to_unit(os.path.basename(f)) < self.val_ratio) for f in self.files}

    def set_epoch(self, epoch: int): self._epoch = epoch

    def _infer_obs_dim_once(self):
        if self.obs_dim: return
        w = get_worker_info()
        files = self.files[w.id::w.num_workers] if (w and w.num_workers > 1) else self.files
        for row in iter_jsonl(files, report_every=self.report_every, strict=self.strict_json):
            s = extract_sample(row, obs_dim_override=0)
            if s is None: continue
            self.obs_dim = len(s["obs"])
            print(f"[BC][stream] inferred obs_dim={self.obs_dim}", flush=True)
            break
        if self.obs_dim == 0:
            raise RuntimeError("[BC][stream][fatal] could not infer obs_dim from any file")

    def _episode_in_split(self, episode_id: Optional[str], file_path: str, want_val: bool) -> bool:
        if self.split_mode == "chronological": return True
        if self.split_mode == "hash_file":
            is_val = self._file_fold.get(file_path, False)
            return is_val if want_val else (not is_val)
        if not episode_id:
            is_val = self._file_fold.get(file_path, False)
            return is_val if want_val else (not is_val)
        u = _hash_to_unit(episode_id)
        is_val = (u < self.val_ratio)
        return is_val if want_val else (not is_val)

    def __iter__(self):
        self._infer_obs_dim_once()
        w = get_worker_info()
        files = self.files[w.id::w.num_workers] if (w and w.num_workers > 1) else self.files
        if not files: return

        rng = random.Random(1337 + (self._epoch * 997))
        buf: List[Dict[str, Any]] = []
        seen = 0
        yielded = 0

        for fp in files:
            for row in iter_jsonl([fp], report_every=self.report_every, strict=self.strict_json):
                if self.max_rows and seen >= self.max_rows: break
                seen += 1
                if not self._episode_in_split(row_episode_id(row), fp, self.want_val): continue
                s = extract_sample(row, obs_dim_override=self.obs_dim)
                if s is None: continue
                buf.append(s)
                if len(buf) >= self.shuffle_buffer:
                    idx = rng.randrange(len(buf))
                    yield buf.pop(idx)
                    yielded += 1

        while buf:
            idx = rng.randrange(len(buf))
            yield buf.pop(idx)
            yielded += 1

        if yielded == 0:
            raise RuntimeError("[BC][stream][fatal] No usable samples yielded. "
                               "Check your --data glob or JSONL content.")

class StreamingEpisodeDataset(IterableDataset):
    def __init__(self, files: List[str], report_every: int,
                 max_rows: int = 0, strict_json: bool = False,
                 max_live_episodes: int = 1024,
                 split_mode: str = "hash_episode", val_ratio: float = 0.1, want_val: bool = False,
                 cache_in_ram: bool = False):
        super().__init__()
        self.files = files
        self.report_every = report_every
        self.max_rows = max_rows
        self.strict_json = strict_json
        self.max_live_episodes = max_live_episodes
        self._epoch = 0
        self.obs_dim = 0
        self.split_mode = split_mode
        self.val_ratio = float(val_ratio)
        self.want_val = bool(want_val)
        self._file_fold = {f: (_hash_to_unit(os.path.basename(f)) < self.val_ratio) for f in self.files}
        self.cache_in_ram = cache_in_ram
        self._episode_cache: Optional[List[Dict[str, Any]]] = None

    def set_epoch(self, epoch: int): self._epoch = epoch

    def _infer_obs_dim_once(self):
        if self.obs_dim: return
        w = get_worker_info()
        files = self.files[w.id::w.num_workers] if (w and w.num_workers > 1) else self.files
        for row in iter_jsonl(files, report_every=self.report_every, strict=self.strict_json):
            s = extract_sample(row, obs_dim_override=0)
            if s is None: continue
            self.obs_dim = len(s["obs"])
            print(f"[BC][stream] inferred obs_dim={self.obs_dim}", flush=True)
            break
        if self.obs_dim == 0:
            raise RuntimeError("[BC][stream][fatal] could not infer obs_dim from any file")

    def _episode_in_split(self, episode_id: Optional[str], file_path: str, want_val: bool) -> bool:
        if self.split_mode == "chronological": return True
        if self.split_mode == "hash_file":
            is_val = self._file_fold.get(file_path, False)
            return is_val if want_val else (not is_val)
        if not episode_id:
            is_val = self._file_fold.get(file_path, False)
            return is_val if want_val else (not is_val)
        u = _hash_to_unit(episode_id)
        is_val = (u < self.val_ratio)
        return is_val if want_val else (not is_val)

    def _iter_from_disk(self):
        """Read episodes from disk (original streaming path)."""
        self._infer_obs_dim_once()
        w = get_worker_info()
        files = self.files[w.id::w.num_workers] if (w and w.num_workers > 1) else self.files
        if not files: return

        episodes: Dict[str, List[Dict[str, Any]]] = {}
        seen_rows = 0
        yielded_eps = 0

        def emit_episode(eid: str):
            nonlocal yielded_eps
            seq = episodes.pop(eid, [])
            if not seq: return
            if all(("t" in r) for r in seq):
                seq.sort(key=lambda r: r["t"])
            yielded_eps += 1
            yield {"episode": seq}

        for fp in files:
            for row in iter_jsonl([fp], report_every=self.report_every, strict=self.strict_json):
                if self.max_rows and seen_rows >= self.max_rows: break
                seen_rows += 1

                eid = row_episode_id(row)
                if not self._episode_in_split(eid, fp, want_val=self.want_val): continue

                s = extract_sample(row, obs_dim_override=self.obs_dim)
                if s is None: continue

                # Preserve per-row result if present, so collate_seq can broadcast it.
                if "result" in row and ("result" not in s):
                    try:
                        s["result"] = float(row["result"])
                    except Exception:
                        pass

                t = row_step_index(row)
                done = row_done(row)

                if eid is None:
                    s["t"] = 0; s["done"] = True
                    yield {"episode": [s]}
                    continue

                if t is not None: s["t"] = int(t)
                s["done"] = bool(done)

                bucket = episodes.setdefault(eid, [])
                bucket.append(s)

                if s["done"]:
                    for out in emit_episode(eid):
                        yield out

                if len(episodes) > self.max_live_episodes:
                    oldest = next(iter(episodes.keys()))
                    print(f"[BC][stream][warn] dropping incomplete episode buffer: {oldest}", flush=True)
                    episodes.pop(oldest, None)

        leftovers = []
        for eid, seq in list(episodes.items()):
            if any(x.get("done", False) for x in seq):
                leftovers.append(eid)
        for eid in leftovers:
            for out in emit_episode(eid):
                yield out

        if yielded_eps == 0 and files:
            raise RuntimeError("[BC][stream][fatal] No complete episodes yielded. "
                               "If your logs lack episode_id/turn/done, consider fixing observer.py "
                               "or temporarily use the non-episode StreamingJSONLDataset.")

    @staticmethod
    def _compact_sample(s: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Python float lists to numpy float32 arrays for ~7x memory reduction."""
        out = dict(s)
        for key in ("obs", "legal"):
            if key in out and isinstance(out[key], list):
                out[key] = np.asarray(out[key], dtype=np.float32)
        if "ctx_extra" in out and isinstance(out["ctx_extra"], list):
            out["ctx_extra"] = np.asarray(out["ctx_extra"], dtype=np.float32)
        if "move_slots" in out and isinstance(out["move_slots"], list):
            out["move_slots"] = np.asarray(out["move_slots"], dtype=np.float32)
        if "switch_slots" in out and isinstance(out["switch_slots"], list):
            out["switch_slots"] = np.asarray(out["switch_slots"], dtype=np.float32)
        for id_key in ("entity_ids", "move_ids", "switch_ids"):
            if id_key in out and isinstance(out[id_key], list):
                out[id_key] = np.asarray(out[id_key], dtype=np.int32)
        return out

    def __iter__(self):
        if self._episode_cache is not None:
            # Serve from RAM cache with epoch-seeded shuffle
            indices = list(range(len(self._episode_cache)))
            rng = random.Random(self._epoch + 42)
            rng.shuffle(indices)
            for i in indices:
                yield self._episode_cache[i]
            return

        # First pass: stream from disk, optionally building cache
        building_cache = self.cache_in_ram and (self._episode_cache is None)
        cache_buf: List[Dict[str, Any]] = []

        for ep in self._iter_from_disk():
            if building_cache:
                compact_ep = {"episode": [self._compact_sample(s) for s in ep["episode"]]}
                cache_buf.append(compact_ep)
            yield ep

        if building_cache and cache_buf:
            self._episode_cache = cache_buf
            n_rows = sum(len(e["episode"]) for e in cache_buf)
            # Estimate actual memory: sum array nbytes
            mem_bytes = 0
            for e in cache_buf:
                for s in e["episode"]:
                    for v in s.values():
                        if isinstance(v, np.ndarray):
                            mem_bytes += v.nbytes
            mb = mem_bytes / 1e6
            print(f"[BC][cache] cached {len(cache_buf)} episodes ({n_rows} rows, ~{mb:.0f} MB numpy) in RAM", flush=True)

# ============================================================
#                  MEMMAP EPISODE DATASET
# ============================================================

class MemmapEpisodeDataset(Dataset):
    """
    Map-style dataset that reads pre-converted memmap .npy files.
    Each __getitem__ returns {"episode": [list of sample dicts]} matching collate_seq.
    Memmaps are opened lazily to avoid pickle issues with DataLoader workers on Windows.
    """

    def __init__(self, memmap_dir: str, split: str = "train", val_ratio: float = 0.1,
                 ood_holdout_hashes: Optional[set] = None):
        super().__init__()
        self.memmap_dir = Path(memmap_dir)
        meta_path = self.memmap_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"[MemmapDataset] meta.json not found in {memmap_dir}")
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.obs_dim = self.meta["obs_dim"]
        self.mod_keys = self.meta.get("mod_keys", [])
        self.move_slot_dim = self.meta.get("move_slot_dim", 0)
        self.switch_slot_dim = self.meta.get("switch_slot_dim", 0)
        self.ctx_extra_dim = self.meta.get("ctx_extra_dim", 0)
        self.entity_ids_dim = self.meta.get("entity_ids_dim", 0)

        # Load episode index (small array, OK to keep in RAM)
        full_index = np.load(str(self.memmap_dir / "episode_index.npy"))  # (E, 3) int64
        # Split by episode hash — uses SHA-1 based hash stored as uint64 in column 2
        hashes = full_index[:, 2].astype(np.float64) / float(2**63 - 1)
        if split == "val":
            mask = hashes < val_ratio
        else:
            mask = hashes >= val_ratio
        self.ep_indices = np.where(mask)[0]
        self.episode_index = full_index
        self._memmaps = None  # lazy open — DO NOT open here (pickle/spawn issues)

    def _open_memmaps(self):
        """Lazily open memmap files. Called on first __getitem__ in each worker."""
        if self._memmaps is not None:
            return
        d = self.memmap_dir
        mm = {}
        mm["obs"] = np.load(str(d / "obs.npy"), mmap_mode='r')
        mm["action"] = np.load(str(d / "action.npy"), mmap_mode='r')
        mm["legal"] = np.load(str(d / "legal.npy"), mmap_mode='r')
        mm["result"] = np.load(str(d / "result.npy"), mmap_mode='r')
        mm["turn"] = np.load(str(d / "turn.npy"), mmap_mode='r')
        mm["phase"] = np.load(str(d / "phase.npy"), mmap_mode='r')
        if (d / "move_slots.npy").exists():
            mm["move_slots"] = np.load(str(d / "move_slots.npy"), mmap_mode='r')
        if (d / "switch_slots.npy").exists():
            mm["switch_slots"] = np.load(str(d / "switch_slots.npy"), mmap_mode='r')
        if (d / "ctx_extra.npy").exists():
            mm["ctx_extra"] = np.load(str(d / "ctx_extra.npy"), mmap_mode='r')
        if (d / "mods.npy").exists():
            mm["mods"] = np.load(str(d / "mods.npy"), mmap_mode='r')
        if (d / "entity_ids.npy").exists():
            mm["entity_ids"] = np.load(str(d / "entity_ids.npy"), mmap_mode='r')
        if (d / "move_ids.npy").exists():
            mm["move_ids"] = np.load(str(d / "move_ids.npy"), mmap_mode='r')
        if (d / "switch_ids.npy").exists():
            mm["switch_ids"] = np.load(str(d / "switch_ids.npy"), mmap_mode='r')
        self._memmaps = mm

    def __len__(self):
        return len(self.ep_indices)

    def __getitem__(self, idx):
        self._open_memmaps()
        real_idx = self.ep_indices[idx]
        start, length, _ep_hash = self.episode_index[real_idx]
        start, length = int(start), int(length)
        end = start + length
        mm = self._memmaps

        samples = []
        for row_idx in range(start, end):
            s = {
                "obs": mm["obs"][row_idx].copy(),
                "action": int(mm["action"][row_idx]),
                "legal": mm["legal"][row_idx].copy(),
                "t": int(mm["turn"][row_idx]),
            }

            # Result
            r = float(mm["result"][row_idx])
            if r >= 0.0:
                s["result"] = r

            # Mods
            if "mods" in mm and self.mod_keys:
                mods_row = mm["mods"][row_idx]
                s["mods"] = {k: float(mods_row[i]) for i, k in enumerate(self.mod_keys)}
            else:
                s["mods"] = {}

            # Move/switch slots
            if "move_slots" in mm:
                s["move_slots"] = mm["move_slots"][row_idx].copy()
            if "switch_slots" in mm:
                s["switch_slots"] = mm["switch_slots"][row_idx].copy()

            # Ctx extra
            if "ctx_extra" in mm:
                s["ctx_extra"] = mm["ctx_extra"][row_idx].copy()

            # v5 entity IDs
            if "entity_ids" in mm:
                s["entity_ids"] = mm["entity_ids"][row_idx].copy()
            if "move_ids" in mm:
                s["move_ids"] = mm["move_ids"][row_idx].copy()
            if "switch_ids" in mm:
                s["switch_ids"] = mm["switch_ids"][row_idx].copy()

            # Done flag (last row in episode)
            s["done"] = (row_idx == end - 1)

            samples.append(s)

        return {"episode": samples}

    def set_epoch(self, epoch: int):
        """No-op for API compatibility — DataLoader handles shuffling for map-style datasets."""
        pass


# ============================================================
#                      COLLATE FNS
# ============================================================

def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Row-wise collate (no sequence dimension)."""
    obs = torch.tensor([b["obs"] for b in batch], dtype=torch.float32)
    act = torch.tensor([b["action"] for b in batch], dtype=torch.long)
    legal = torch.tensor([b["legal"] for b in batch], dtype=torch.float32)

    mod_keys = sorted({k for b in batch for k in (b.get("mods", {}) or {}).keys()})
    if mod_keys:
        mods = torch.tensor([[float((b.get("mods", {}) or {}).get(k, 0)) for k in mod_keys]
                             for b in batch], dtype=torch.float32)
    else:
        mods = None

    # Hard guards: 9-way mask and at least one legal action per row
    assert legal.shape[1] == 9, f"legal mask must be 9-way, got {legal.shape[1]}"
    if torch.any(legal.sum(dim=1) <= 0):
        bad = (legal.sum(dim=1) <= 0).nonzero(as_tuple=False).view(-1).tolist()
        raise RuntimeError(f"[BC][fatal] Found {len(bad)} rows with no legal actions in minibatch")

    out = {"obs": obs, "action": act, "legal": legal, "mod_keys": mod_keys}
    if mods is not None:
        out["mods"] = mods

    # Step-type: absolute bucketing (match inference)
    ts = []
    for b in batch:
        t = b.get("t", None)
        if t is None:
            ts.append(0)
        else:
            ts.append(step_type_from_abs_t(int(t),
                                           bins=max(1, int(_GLOBAL_STEP_TYPE_BINS)),
                                           cap=50))
    out["step_type"] = torch.tensor(ts, dtype=torch.long)
    
    # discover dims
    mv_dim = 0
    sw_dim = 0
    for b in batch:
        mv = b.get("move_slots"); sw = b.get("switch_slots")
        if isinstance(mv, _arraylike) and len(mv) == 4 and isinstance(mv[0], _arraylike):
            mv_dim = max(mv_dim, len(mv[0]))
        if isinstance(sw, _arraylike) and len(sw) == 5 and isinstance(sw[0], _arraylike):
            sw_dim = max(sw_dim, len(sw[0]))
    if mv_dim > 0:
        ms = np.zeros((len(batch), 4, mv_dim), dtype=np.float32)
        for i,b in enumerate(batch):
            mv = b.get("move_slots")
            if isinstance(mv, _arraylike):
                v = np.asarray(mv, dtype=np.float32)
                L = min(mv_dim, v.shape[-1]); ms[i, :, :L] = v[:, :L]
        out["move_slots"] = torch.from_numpy(ms)
    if sw_dim > 0:
        ss = np.zeros((len(batch), 5, sw_dim), dtype=np.float32)
        for i,b in enumerate(batch):
            sw = b.get("switch_slots")
            if isinstance(sw, _arraylike):
                v = np.asarray(sw, dtype=np.float32)
                L = min(sw_dim, v.shape[-1]); ss[i, :, :L] = v[:, :L]
        out["switch_slots"] = torch.from_numpy(ss)

    # ctx_extra: auto-detect dimension from batch data
    ctx_dim = 0
    for b in batch:
        v = b.get("ctx_extra")
        if isinstance(v, _arraylike) and len(v) > 0:
            ctx_dim = max(ctx_dim, len(v))
    if ctx_dim > 0:
        cx = np.zeros((len(batch), ctx_dim), dtype=np.float32)
        for i, b in enumerate(batch):
            if "ctx_extra" in b and isinstance(b["ctx_extra"], _arraylike):
                v = np.asarray(b["ctx_extra"], dtype=np.float32)
                L = min(ctx_dim, len(v))
                cx[i, :L] = v[:L]
        out["ctx_extra"] = torch.from_numpy(cx)

    # v5 entity IDs
    eid_dim = 0
    for b in batch:
        v = b.get("entity_ids")
        if isinstance(v, _arraylike) and len(v) > 0:
            eid_dim = max(eid_dim, len(v))
    if eid_dim > 0:
        eids = np.zeros((len(batch), eid_dim), dtype=np.int32)
        for i, b in enumerate(batch):
            if "entity_ids" in b and isinstance(b["entity_ids"], _arraylike):
                v = np.asarray(b["entity_ids"], dtype=np.int32)
                L = min(eid_dim, len(v))
                eids[i, :L] = v[:L]
        out["entity_ids"] = torch.from_numpy(eids).long()

    # move_ids / switch_ids
    has_mids = any(isinstance(b.get("move_ids"), _arraylike) for b in batch)
    has_sids = any(isinstance(b.get("switch_ids"), _arraylike) for b in batch)
    if has_mids:
        mids = np.zeros((len(batch), 4), dtype=np.int32)
        for i, b in enumerate(batch):
            if isinstance(b.get("move_ids"), _arraylike):
                mids[i] = np.asarray(b["move_ids"], dtype=np.int32)[:4]
        out["move_ids"] = torch.from_numpy(mids).long()
    if has_sids:
        sids = np.zeros((len(batch), 5), dtype=np.int32)
        for i, b in enumerate(batch):
            if isinstance(b.get("switch_ids"), _arraylike):
                sids[i] = np.asarray(b["switch_ids"], dtype=np.int32)[:5]
        out["switch_ids"] = torch.from_numpy(sids).long()

    return out

def collate_seq(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Sequence collate: pads to max T; produces masks and per-step step_type bins."""
    episodes = [b["episode"] for b in batch]
    B = len(episodes)
    T = max(len(ep) for ep in episodes)
    F = len(episodes[0][0]["obs"])
    A = 9

    obs   = np.zeros((B, T, F), dtype=np.float32)
    act   = np.full((B, T), fill_value=-1, dtype=np.int64)
    legal = np.zeros((B, T, A), dtype=np.float32)
    mask  = np.zeros((B, T), dtype=np.float32)
    step_type = np.zeros((B, T), dtype=np.int64)
    result_all = np.full((B, T), fill_value=-1.0, dtype=np.float32)
    
    # ---- Discover slot dims & allocate tensors ----
    move_slot_dim = 0
    switch_slot_dim = 0
    for ep in episodes:
        for s in ep:
            mv = s.get("move_slots")
            sw = s.get("switch_slots")
            if isinstance(mv, _arraylike) and len(mv) == 4 and isinstance(mv[0], _arraylike):
                move_slot_dim = max(move_slot_dim, len(mv[0]))
            if isinstance(sw, _arraylike) and len(sw) == 5 and isinstance(sw[0], _arraylike):
                switch_slot_dim = max(switch_slot_dim, len(sw[0]))
    move_slots = None
    switch_slots = None
    if move_slot_dim > 0:
        move_slots = np.zeros((B, T, 4, move_slot_dim), dtype=np.float32)
    if switch_slot_dim > 0:
        switch_slots = np.zeros((B, T, 5, switch_slot_dim), dtype=np.float32)
    
    # Collect modifier keys
    mod_keys = set()
    for ep in episodes:
        for s in ep:
            md = s.get("mods")
            if isinstance(md, dict):
                mod_keys |= set(md.keys())
    mod_keys = sorted(list(mod_keys))
    mods_arr = None
    if mod_keys:
        mods_arr = np.zeros((B, T, len(mod_keys)), dtype=np.float32)

    # Decide ctx_dim ONCE (prefer logged > args > 0)
    ctx_dim = 0
    for ep in episodes:
        for s in ep:
            v = s.get("ctx_extra", None)
            if isinstance(v, _arraylike) and len(v) > 0:
                ctx_dim = max(ctx_dim, len(v))
    # ctx_dim is auto-detected from data above; no global args fallback needed

    ctx_extra = None
    if ctx_dim > 0:
        ctx_extra = np.zeros((B, T, ctx_dim), dtype=np.float32)

    # v5 entity IDs
    eid_dim = 0
    for ep in episodes:
        for s in ep:
            v = s.get("entity_ids")
            if isinstance(v, _arraylike) and len(v) > 0:
                eid_dim = max(eid_dim, len(v))
    entity_ids_arr = None
    if eid_dim > 0:
        entity_ids_arr = np.zeros((B, T, eid_dim), dtype=np.int32)

    has_mids = any(isinstance(s.get("move_ids"), _arraylike) for ep in episodes for s in ep)
    has_sids = any(isinstance(s.get("switch_ids"), _arraylike) for ep in episodes for s in ep)
    move_ids_arr = np.zeros((B, T, 4), dtype=np.int32) if has_mids else None
    switch_ids_arr = np.zeros((B, T, 5), dtype=np.int32) if has_sids else None

    # ---- Fill tensors ----
    seq_lens = np.zeros((B,), dtype=np.int64)
    for i, ep in enumerate(episodes):
        Tlen = len(ep)
        seq_lens[i] = Tlen

        # Terminal result (if any) copied across the episode
        ep_result = None
        for s in reversed(ep):
            if "result" in s:
                try:
                    ep_result = float(s["result"])
                except Exception:
                    ep_result = None
                break

        for t, s in enumerate(ep):
            obs[i, t]   = np.asarray(s["obs"], dtype=np.float32)
            act[i, t]   = int(s["action"])
            legal[i, t] = np.asarray(s["legal"], dtype=np.float32)
            mask[i, t]  = 1.0

            # Step-type: absolute turn bucket (matches inference in bc_policy_player)
            abs_turn = int(s.get("turn", t))
            step_type[i, t] = step_type_from_abs_t(
                abs_turn, bins=max(1, int(_GLOBAL_STEP_TYPE_BINS)), cap=50
            )
            
            # per-slot tensors
            if move_slots is not None and isinstance(s.get("move_slots"), _arraylike):
                v = np.asarray(s["move_slots"], dtype=np.float32)  # [4, M]
                L = min(move_slot_dim, v.shape[-1])
                move_slots[i, t, :, :L] = v[:, :L]
            if switch_slots is not None and isinstance(s.get("switch_slots"), _arraylike):
                v = np.asarray(s["switch_slots"], dtype=np.float32)  # [5, S]
                L = min(switch_slot_dim, v.shape[-1])
                switch_slots[i, t, :, :L] = v[:, :L]

            # ctx per-step (only if we decided to include it)
            if ctx_extra is not None:
                if isinstance(s.get("ctx_extra"), _arraylike):
                    v = np.asarray(s["ctx_extra"], dtype=np.float32)
                else:
                    v = np.zeros((ctx_dim,), dtype=np.float32)
                L = min(ctx_dim, len(v))
                ctx_extra[i, t, :L] = v[:L]

            # mods
            if mods_arr is not None:
                md = (s.get("mods", {}) or {})
                for k_idx, k in enumerate(mod_keys):
                    mods_arr[i, t, k_idx] = float(md.get(k, 0))

            # v5 entity IDs
            if entity_ids_arr is not None and isinstance(s.get("entity_ids"), _arraylike):
                v = np.asarray(s["entity_ids"], dtype=np.int32)
                L = min(eid_dim, len(v))
                entity_ids_arr[i, t, :L] = v[:L]
            if move_ids_arr is not None and isinstance(s.get("move_ids"), _arraylike):
                v = np.asarray(s["move_ids"], dtype=np.int32)
                move_ids_arr[i, t, :min(4, len(v))] = v[:4]
            if switch_ids_arr is not None and isinstance(s.get("switch_ids"), _arraylike):
                v = np.asarray(s["switch_ids"], dtype=np.int32)
                switch_ids_arr[i, t, :min(5, len(v))] = v[:5]

            # episode-level result
            if ep_result is not None:
                result_all[i, t] = ep_result

    # Hard guards: 9-way mask and at least one legal action at valid positions
    if legal.shape[-1] != 9:
        raise RuntimeError(f"sequence legal mask must be 9-way, got {legal.shape[-1]}")
    if np.any((legal.sum(axis=-1) <= 0) & (mask > 0)):
        raise RuntimeError("[BC][fatal] Found time-steps with no legal actions (where mask=1) in sequence batch")

    out: Dict[str, Any] = {
        "obs": torch.from_numpy(obs),
        "action": torch.from_numpy(act),
        "legal": torch.from_numpy(legal),
        "mask": torch.from_numpy(mask),
        "seq_lens": torch.from_numpy(seq_lens),
        "mod_keys": mod_keys,
        "step_type": torch.from_numpy(step_type),
        "result": torch.from_numpy(result_all),
    }
    if move_slots is not None:
        out["move_slots"] = torch.from_numpy(move_slots)
    if switch_slots is not None:
        out["switch_slots"] = torch.from_numpy(switch_slots)
    if ctx_extra is not None:
        out["ctx_extra"] = torch.from_numpy(ctx_extra)
    if mods_arr is not None:
        out["mods"] = torch.from_numpy(mods_arr)
    if entity_ids_arr is not None:
        out["entity_ids"] = torch.from_numpy(entity_ids_arr).long()
    if move_ids_arr is not None:
        out["move_ids"] = torch.from_numpy(move_ids_arr).long()
    if switch_ids_arr is not None:
        out["switch_ids"] = torch.from_numpy(switch_ids_arr).long()
    return out

# ============================================================
#                       LOSSES
# ============================================================

def _big_neg_like(x: torch.Tensor) -> torch.Tensor:
    # Large negative that plays nicely with AMP float16 (max ~65504)
    return torch.full_like(x, -6e4 if x.dtype == torch.float16 else -1e9)

def masked_policy_ce(
    logits: torch.Tensor,
    actions: torch.Tensor,
    legal_mask: torch.Tensor,
    inputs_are_log_probs: bool = False,
    valid_mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0
) -> Tuple[torch.Tensor, float]:
    """
    Cross-entropy over *legal* actions, optionally with legal-aware label smoothing.
    Works for batched sequences too.
    """
    # Sanity: 9-way action space everywhere
    assert logits.shape[-1] == 9 and legal_mask.shape[-1] == 9, \
        f"expected 9-way actions, got logits={logits.shape} legal={legal_mask.shape}"
    if logits.dim() == 3:
        B, T, A = logits.shape
        masked_logits = torch.where(legal_mask > 0, logits, _big_neg_like(logits))
        logp_all = masked_logits if inputs_are_log_probs else torch.log_softmax(masked_logits, dim=-1)

        flat_actions = actions.reshape(B*T)
        flat_logp = logp_all.reshape(B*T, A)
        flat_legal = legal_mask.reshape(B*T, A)
        vm = (flat_actions >= 0)
        if valid_mask is not None:
            vm = vm & (valid_mask.reshape(B*T) > 0)
        if vm.sum() == 0:
            return torch.tensor(0.0, device=logits.device), 0.0

        flat_actions = flat_actions[vm]
        flat_logp = flat_logp[vm]
        flat_legal = flat_legal[vm]

        if label_smoothing > 0.0:
            with torch.no_grad():
                legal_counts = flat_legal.sum(dim=-1, keepdim=True).clamp_min(1.0)
                smooth = (label_smoothing / legal_counts)  # [N, 1]
                target = smooth * flat_legal               # [N, A]
                target.scatter_(-1, flat_actions.view(-1,1),
                                (1.0 - label_smoothing) + smooth)  # smooth is broadcast
            loss = -(target * flat_logp).sum(dim=-1).mean()
        else:
            nll = nn.NLLLoss(reduction="mean")
            loss = nll(flat_logp, flat_actions)

        with torch.no_grad():
            pred = torch.argmax(masked_logits.reshape(B*T, A)[vm], dim=-1)
            acc = (pred == flat_actions).float().mean().item()
        return loss, acc
    else:
        masked_logits = torch.where(legal_mask > 0, logits, _big_neg_like(logits))
        logp = masked_logits if inputs_are_log_probs else torch.log_softmax(masked_logits, dim=-1)
        if label_smoothing > 0.0:
            A = logits.shape[-1]
            with torch.no_grad():
                legal_counts = legal_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                smooth = (label_smoothing / legal_counts)
                target = smooth * legal_mask
                target.scatter_(-1, actions.view(-1,1),
                                (1.0 - label_smoothing) + smooth)
            loss = -(target * logp).sum(dim=-1).mean()
        else:
            loss = nn.NLLLoss(reduction="mean")(logp, actions)
        with torch.no_grad():
            pred = torch.argmax(masked_logits, dim=-1)
            acc = (pred == actions).float().mean().item()
        return loss, acc

def modifier_bce(
    mod_logits_cat: Optional[torch.Tensor],  # [B,K] or [B,T,K]
    mods_cat: Optional[torch.Tensor],        # [B,K] or [B,T,K]
    actions: torch.Tensor,                   # [B] or [B,T]
    valid_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, float]:
    """Binary cross-entropy for modifier heads (only when choosing a MOVE: action<4)."""
    if (mod_logits_cat is None) or (mods_cat is None):
        return torch.tensor(0.0, device=actions.device), 0.0

    if mod_logits_cat.dim() == 3:
        B, T, K = mod_logits_cat.shape
        is_move = (actions < 4)
        if valid_mask is not None: is_move = is_move & (valid_mask > 0)
        valid = is_move.unsqueeze(-1).expand_as(mod_logits_cat)  # [B,T,K]
        if not valid.any():
            return torch.tensor(0.0, device=actions.device), 0.0
        loss = F.binary_cross_entropy_with_logits(
            mod_logits_cat[valid], mods_cat[valid], reduction='sum'
        ) / valid.float().sum()
        with torch.no_grad():
            probs = torch.sigmoid(mod_logits_cat)
            pred = (probs > 0.5).float()
            correct = ((pred == (mods_cat > 0.5).float()) & valid).sum().item()
            acc = correct / float(valid.sum().item())
        return loss, acc
    else:
        is_move = (actions < 4)
        valid = is_move.unsqueeze(-1).expand_as(mod_logits_cat)  # [B,K]
        if not valid.any():
            return torch.tensor(0.0, device=actions.device), 0.0
        loss = F.binary_cross_entropy_with_logits(
            mod_logits_cat[valid], mods_cat[valid], reduction='sum'
        ) / valid.float().sum()
        with torch.no_grad():
            probs = torch.sigmoid(mod_logits_cat)
            pred = (probs > 0.5).float()
            correct = ((pred == (mods_cat > 0.5).float()) & valid).sum().item()
            acc = correct / float(valid.sum().item())
        return loss, acc

# ============================================================
#                TRAIN / EVAL + MODEL FACTORY
# ============================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    tb: Optional[SummaryWriter],
    global_step: int,
    args,
    scaler: Optional[torch.cuda.amp.GradScaler],
    sched_step: Optional[Callable[[int], None]],
    max_batches: Optional[int] = None,
    opt_steps: int = 0,
    ema_update_fn: Optional[Callable[[int], None]] = None,  # now passes global opt_steps
    action_class_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, float, float, int, int]:
    model.train()
    total_loss = 0.0; total_acc = 0.0; total_macc = 0.0; count = 0

    use_amp = bool(args.amp) and str(device).startswith("cuda")
    autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    accum = max(1, int(getattr(args, "grad_accum", 1)))
    opt.zero_grad(set_to_none=True)

    opt_steps_local = 0

    for i, batch in enumerate(train_loader):
        if max_batches is not None and i >= max_batches: break

        obs = batch["obs"].to(device)
        act = batch["action"].to(device)
        legal = batch["legal"].to(device)
        batch_move_slots = batch.get("move_slots")
        batch_move_slots = batch_move_slots.to(device) if batch_move_slots is not None else None
        batch_switch_slots = batch.get("switch_slots")
        batch_switch_slots = batch_switch_slots.to(device) if batch_switch_slots is not None else None
        valid_mask = batch.get("mask", None)
        if valid_mask is not None: valid_mask = valid_mask.to(device)

        with autocast_ctx():
            st = batch.get("step_type", None)
            if st is not None: st = st.to(device)

            # Prepare optional ctx_extra (pad/truncate to cfg size; apply dropout if requested)
            cx = None
            ctx_dim = int(getattr(args, "ctx_extra_dim", 0))
            if ctx_dim > 0:
                if "ctx_extra" in batch:
                    cx = batch["ctx_extra"].to(device)
                    if cx.dim() == 3:
                        B,T,D = cx.shape
                        if D < ctx_dim:
                            pad = torch.zeros(B, T, ctx_dim-D, device=device, dtype=cx.dtype)
                            cx = torch.cat([cx, pad], dim=-1)
                        elif D > ctx_dim:
                            cx = cx[..., :ctx_dim]
                    else:
                        B,D = cx.shape
                        if D < ctx_dim:
                            pad = torch.zeros(B, ctx_dim-D, device=device, dtype=cx.dtype)
                            cx = torch.cat([cx, pad], dim=-1)
                        elif D > ctx_dim:
                            cx = cx[:, :ctx_dim]

                    drop_p = float(getattr(args, "ctx_dropout", 0.0) or 0.0)
                    if drop_p > 0.0:
                        if cx.dim() == 3:
                            m = (torch.rand(cx.shape[0], cx.shape[1], 1, device=device) > drop_p).float()
                            cx = cx * m
                        else:
                            m = (torch.rand(cx.shape[0], 1, device=device) > drop_p).float()
                            cx = cx * m

            # v5 entity IDs
            b_entity_ids = batch.get("entity_ids")
            if b_entity_ids is not None: b_entity_ids = b_entity_ids.to(device)
            b_move_ids = batch.get("move_ids")
            if b_move_ids is not None: b_move_ids = b_move_ids.to(device)
            b_switch_ids = batch.get("switch_ids")
            if b_switch_ids is not None: b_switch_ids = b_switch_ids.to(device)
            b_seq_lens = batch.get("seq_lens")
            if b_seq_lens is not None: b_seq_lens = b_seq_lens.to(device)

            out = model(obs, action_mask=legal, step_type=st, ctx_extra=cx,
                        move_slots=batch_move_slots, switch_slots=batch_switch_slots,
                        entity_ids=b_entity_ids, move_ids=b_move_ids, switch_ids=b_switch_ids,
                        seq_lens=b_seq_lens)

            logits = out["action_logits"]
            pol_loss, pol_acc = masked_policy_ce(
                logits, act, legal,
                valid_mask=valid_mask,
                label_smoothing=float(getattr(args, "label_smoothing", 0.0))
            )

            # Advantage weighting: recompute policy loss with per-step weights
            adv_w = float(getattr(args, "advantage_weight", 0.0))
            use_acw = action_class_weights is not None
            if (adv_w > 0.0 and "result" in batch) or use_acw:
                w_win = float(getattr(args, "w_win", 2.0))
                w_loss = float(getattr(args, "w_loss", 0.5))
                if adv_w > 0.0 and "result" in batch and act.dim() >= 2:
                    result_t = batch["result"].to(device)
                    # result: 1.0=win, 0.0=loss, -1.0=unknown
                    # Build per-sample weights: win->w_win, loss->w_loss, unknown->1.0
                    weights = torch.ones_like(result_t)
                    weights = torch.where(result_t == 1.0, torch.full_like(weights, w_win), weights)
                    weights = torch.where(result_t == 0.0, torch.full_like(weights, w_loss), weights)
                elif adv_w > 0.0 and "result" in batch and act.dim() == 1:
                    result_t = batch["result"].to(device)
                    weights = torch.ones_like(result_t)
                    weights = torch.where(result_t == 1.0, torch.full_like(weights, w_win), weights)
                    weights = torch.where(result_t == 0.0, torch.full_like(weights, w_loss), weights)
                elif act.dim() >= 2:
                    B, T = act.shape
                    weights = torch.ones(B, T, device=device)
                else:
                    weights = torch.ones_like(act, dtype=torch.float32)

                if act.dim() >= 2:
                    # Mask out padded positions
                    if valid_mask is not None:
                        vm_f = valid_mask.to(device)
                        weights = weights * vm_f
                    # Recompute weighted CE manually (with label smoothing support)
                    masked_logits = torch.where(legal > 0, logits, _big_neg_like(logits))
                    logp_all = torch.log_softmax(masked_logits, dim=-1)
                    B, T, A = logp_all.shape
                    flat_act = act.reshape(B*T)
                    flat_logp = logp_all.reshape(B*T, A)
                    flat_legal = legal.reshape(B*T, A)
                    flat_w = weights.reshape(B*T)
                    vm = (flat_act >= 0)
                    if valid_mask is not None:
                        vm = vm & (vm_f.reshape(B*T) > 0)
                    if vm.sum() > 0:
                        ls = float(getattr(args, "label_smoothing", 0.0))
                        if ls > 0.0:
                            # Label-smoothed cross-entropy over legal actions
                            with torch.no_grad():
                                legal_counts = flat_legal[vm].sum(dim=-1, keepdim=True).clamp_min(1.0)
                                smooth = ls / legal_counts  # [N, 1]
                                target = smooth * flat_legal[vm]  # [N, A]
                                target.scatter_(-1, flat_act[vm].view(-1, 1),
                                                (1.0 - ls) + smooth)
                            ce = -(target * flat_logp[vm]).sum(dim=-1)  # [N]
                        else:
                            ce = -flat_logp[vm].gather(1, flat_act[vm].unsqueeze(1)).squeeze(1)
                        w_valid = flat_w[vm]
                        # Apply action class weights: multiply per-step weight by action's class weight
                        if use_acw:
                            acw = action_class_weights[flat_act[vm]]  # [N]
                            w_valid = w_valid * acw
                        pol_loss = (ce * w_valid).sum() / w_valid.sum().clamp_min(1e-6)
                else:
                    # Row mode (1D): act is [B], logits is [B,A], legal is [B,A]
                    masked_logits = torch.where(legal > 0, logits, _big_neg_like(logits))
                    logp_all = torch.log_softmax(masked_logits, dim=-1)  # [B, A]
                    vm = (act >= 0)
                    if vm.sum() > 0:
                        ls = float(getattr(args, "label_smoothing", 0.0))
                        if ls > 0.0:
                            with torch.no_grad():
                                legal_counts = legal[vm].sum(dim=-1, keepdim=True).clamp_min(1.0)
                                smooth = ls / legal_counts
                                target = smooth * legal[vm]
                                target.scatter_(-1, act[vm].view(-1, 1),
                                                (1.0 - ls) + smooth)
                            ce = -(target * logp_all[vm]).sum(dim=-1)
                        else:
                            ce = -logp_all[vm].gather(1, act[vm].unsqueeze(1)).squeeze(1)
                        w_valid = weights[vm]
                        if use_acw:
                            acw = action_class_weights[act[vm]]
                            w_valid = w_valid * acw
                        pol_loss = (ce * w_valid).sum() / w_valid.sum().clamp_min(1e-6)

            mod_loss, mod_acc = torch.tensor(0.0, device=device), 0.0
            if "mods" in batch and len(batch["mod_keys"]) > 0 and isinstance(out.get("mod_logits", None), dict):
                keys = batch["mod_keys"]
                avail = [out["mod_logits"][k] for k in keys if k in out["mod_logits"]]
                if len(avail) > 0:
                    mod_logits_cat = torch.cat(avail, dim=-1)
                    mods_cat = batch["mods"].to(device)
                    mod_loss, mod_acc = modifier_bce(mod_logits_cat, mods_cat, act, valid_mask=valid_mask)

            val_w = float(getattr(args, "value_loss_weight", 0.0))
            v_loss = torch.tensor(0.0, device=device)
            if val_w > 0.0 and "result" in batch and out.get("value", None) is not None:
                v_pred = out["value"]                      # [B] or [B,T,1]
                y = batch["result"].to(device)             # [B,T] with -1 as "unknown"
                if v_pred.dim() == 3: v_pred = v_pred.squeeze(-1)  # [B,T]
                valid = (y >= 0.0)
                if valid.any():
                    if bool(getattr(args, "value_loss_bce", True)):
                        v_loss = nn.BCEWithLogitsLoss(reduction="mean")(v_pred[valid], y[valid])
                    else:
                        v_loss = nn.MSELoss(reduction="mean")(torch.sigmoid(v_pred[valid]), y[valid])
            v_loss = val_w * v_loss

            loss = (pol_loss + mod_loss + v_loss) / accum

        if use_amp: scaler.scale(loss).backward()
        else: loss.backward()

        if (i + 1) % accum == 0:
            if use_amp: scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            if use_amp: scaler.step(opt); scaler.update()
            else: opt.step()
            opt.zero_grad(set_to_none=True)

            opt_steps_local += 1
            if sched_step is not None:
                sched_step(opt_steps + opt_steps_local)

            if ema_update_fn is not None:
                ema_update_fn(opt_steps + opt_steps_local)

        total_loss += float((pol_loss + mod_loss).detach().item())
        total_acc  += float(pol_acc)
        total_macc += float(mod_acc)
        count += 1

        if tb and (i % 50 == 0):
            tb.add_scalar("train/loss", float((pol_loss + mod_loss).detach().item()), global_step)
            tb.add_scalar("train/policy_acc_legal_top1", float(pol_acc), global_step)
            tb.add_scalar("train/mod_acc", float(mod_acc), global_step)
            tb.add_scalar("train/value_loss", float(v_loss.detach().item()), global_step)

        global_step += 1

    remainder = count % accum
    if remainder != 0:
        if use_amp: scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        if use_amp: scaler.step(opt); scaler.update()
        else: opt.step()
        opt.zero_grad(set_to_none=True)
        opt_steps_local += 1
        if sched_step is not None:
            sched_step(opt_steps + opt_steps_local)
        if ema_update_fn is not None:
            ema_update_fn(opt_steps + opt_steps_local)

    if count == 0:
        raise RuntimeError("[BC][fatal] Training loader produced no batches. "
                           "If using --seq-mode, verify your logs contain episode_id/t/done.")

    return (total_loss / count, total_acc / count, total_macc / count, global_step, opt_steps + opt_steps_local)

@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    tb: Optional[SummaryWriter],
    global_step: int,
    args
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0; total_acc = 0.0; total_macc = 0.0; count = 0
    vloss_sum, vloss_n = 0.0, 0
    use_amp = bool(args.amp) and str(device).startswith("cuda")
    autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    for batch in val_loader:
        obs = batch["obs"].to(device)
        act = batch["action"].to(device)
        legal = batch["legal"].to(device)
        batch_move_slots = batch.get("move_slots")
        batch_move_slots = batch_move_slots.to(device) if batch_move_slots is not None else None
        batch_switch_slots = batch.get("switch_slots")
        batch_switch_slots = batch_switch_slots.to(device) if batch_switch_slots is not None else None
        valid_mask = batch.get("mask", None)
        if valid_mask is not None: valid_mask = valid_mask.to(device)

        with autocast_ctx():
            st = batch.get("step_type", None)
            if st is not None: st = st.to(device)

            cx = None
            ctx_dim = int(getattr(args, "ctx_extra_dim", 0))
            if ctx_dim > 0:
                if "ctx_extra" in batch:
                    cx = batch["ctx_extra"].to(device)
                    if cx.dim() == 3:
                        B,T,D = cx.shape
                        if D < ctx_dim:
                            pad = torch.zeros(B, T, ctx_dim-D, device=device, dtype=cx.dtype)
                            cx = torch.cat([cx, pad], dim=-1)
                        elif D > ctx_dim:
                            cx = cx[..., :ctx_dim]
                    else:
                        B,D = cx.shape
                        if D < ctx_dim:
                            pad = torch.zeros(B, ctx_dim-D, device=device, dtype=cx.dtype)
                            cx = torch.cat([cx, pad], dim=-1)
                        elif D > ctx_dim:
                            cx = cx[:, :ctx_dim]

            # v5 entity IDs
            b_entity_ids = batch.get("entity_ids")
            if b_entity_ids is not None: b_entity_ids = b_entity_ids.to(device)
            b_move_ids = batch.get("move_ids")
            if b_move_ids is not None: b_move_ids = b_move_ids.to(device)
            b_switch_ids = batch.get("switch_ids")
            if b_switch_ids is not None: b_switch_ids = b_switch_ids.to(device)
            b_seq_lens = batch.get("seq_lens")
            if b_seq_lens is not None: b_seq_lens = b_seq_lens.to(device)

            out = model(obs, action_mask=legal, step_type=st, ctx_extra=cx,
                        move_slots=batch_move_slots, switch_slots=batch_switch_slots,
                        entity_ids=b_entity_ids, move_ids=b_move_ids, switch_ids=b_switch_ids,
                        seq_lens=b_seq_lens)

            logits = out["action_logits"]
            pol_loss, pol_acc = masked_policy_ce(
                logits, act, legal,
                valid_mask=valid_mask,
                label_smoothing=float(getattr(args, "label_smoothing", 0.0))
            )

            mod_loss, mod_acc = torch.tensor(0.0, device=device), 0.0
            if "mods" in batch and len(batch["mod_keys"]) > 0 and isinstance(out.get("mod_logits", None), dict):
                keys = batch["mod_keys"]
                avail = [out["mod_logits"][k] for k in keys if k in out["mod_logits"]]
                if len(avail) > 0:
                    mod_logits_cat = torch.cat(avail, dim=-1)
                    mods_cat = batch["mods"].to(device)
                    mod_loss, mod_acc = modifier_bce(mod_logits_cat, mods_cat, act, valid_mask=valid_mask)

            v_loss = torch.tensor(0.0, device=device)
            val_w = float(getattr(args, "value_loss_weight", 0.0))
            if val_w > 0.0 and "result" in batch and out.get("value", None) is not None:
                v_pred = out["value"]
                if v_pred.dim() == 3: v_pred = v_pred.squeeze(-1)
                y = batch["result"].to(device)
                valid = (y >= 0.0)
                if valid.any():
                    if bool(getattr(args, "value_loss_bce", True)):
                        v_loss = nn.BCEWithLogitsLoss(reduction="mean")(v_pred[valid], y[valid])
                    else:
                        v_loss = nn.MSELoss(reduction="mean")(torch.sigmoid(v_pred[valid]), y[valid])
                v_loss = val_w * v_loss
                vloss_sum += float(v_loss.detach().item()); vloss_n += 1

            loss = pol_loss + mod_loss + v_loss

        total_loss += float(loss.detach().item())
        total_acc  += float(pol_acc)
        total_macc += float(mod_acc)
        count += 1

    if count == 0:
        print("[BC][warn] Validation loader produced no batches.")
        return 0.0, 0.0, 0.0

    val_loss = total_loss / count
    val_acc  = total_acc / count
    val_macc = total_macc / count
    if tb:
        tb.add_scalar("val/loss", val_loss, global_step)
        tb.add_scalar("val/policy_acc_legal_top1", val_acc, global_step)
        tb.add_scalar("val/mod_acc", val_macc, global_step)
        if vloss_n > 0:
            tb.add_scalar("val/value_loss", vloss_sum / vloss_n, global_step)
    
    return val_loss, val_acc, val_macc

def build_model(obs_dim: int, args, move_slot_dim: int = 0, switch_slot_dim: int = 0,
                ctx_dim_override: int = 0, entity_ids_dim: int = 0) -> Tuple[nn.Module, Dict[str, Any]]:
    train_files = getattr(args, "_train_files", None)
    memmap_mod_keys = getattr(args, "_memmap_mod_keys", None)
    mod_keys: List[str] = []
    if memmap_mod_keys is not None:
        # Memmap path: use mod_keys from metadata
        mod_keys = memmap_mod_keys
    elif getattr(args, "mods", "auto") == "auto":
        scan_list = train_files if train_files else glob.glob(args.data)
        try: mod_keys = detect_modifier_keys(scan_list)
        except Exception: mod_keys = []
    elif args.mods.strip():
        mod_keys = [k.strip() for k in args.mods.split(",") if k.strip()]
    mod_specs = [ModifierSpec(name=k) for k in mod_keys]

    # v5 embedding config
    n_entity_ids = int(getattr(args, "n_entity_ids", entity_ids_dim))
    embed_dim = int(getattr(args, "embed_dim", 32))

    # Resolve sequence core: --use-lstm overrides --use-transformer
    _use_lstm = bool(getattr(args, "use_lstm", False))
    _use_transformer = bool(getattr(args, "use_transformer", True)) and not _use_lstm

    cfg = PolicyConfig(
        obs_dim=obs_dim,
        action_dim=9,
        use_lstm=_use_lstm,
        use_transformer=_use_transformer,
        lstm_hidden=int(getattr(args, "lstm_hidden", 256)),
        mlp_hidden=int(getattr(args, "mlp_hidden", 256)),
        lstm_layers=int(getattr(args, "lstm_layers", 1)),
        mlp_layers=int(getattr(args, "mlp_layers", 2)),
        modifiers=mod_specs or None,
        # new cfg knobs (default off)
        hierarchical=bool(getattr(args, "hierarchical", False)),
        step_type_bins=int(getattr(args, "step_type_bins", 0)),
        ctx_extra_dim=int(ctx_dim_override if ctx_dim_override > 0 else getattr(args, "ctx_extra_dim", 0)),
        move_slot_dim=int(move_slot_dim),
        switch_slot_dim=int(switch_slot_dim),
        # v5 embeddings
        n_entity_ids=n_entity_ids,
        embed_dim=embed_dim,
        # Transformer config
        n_transformer_layers=int(getattr(args, "n_transformer_layers", 6)),
        n_heads=int(getattr(args, "n_heads", 4)),
        transformer_dropout=float(getattr(args, "transformer_dropout", 0.1)),
        context_length=int(getattr(args, "context_length", 128)),
    )
    model = BattlePolicy(cfg)
    return model, {"mod_keys": mod_keys, "policy_cfg": asdict(cfg)}

# ============================================================
#                 DATA BUILDERS (train/val)
# ============================================================

def build_datasets(files: List[str], args) -> Tuple:
    # ---- MEMMAP path ----
    if getattr(args, "data_format", "jsonl") == "memmap":
        memmap_dir = getattr(args, "memmap_dir", "data/datasets/memmap")
        print(f"[BC][memmap] Loading from {memmap_dir}", flush=True)
        train_ds = MemmapEpisodeDataset(memmap_dir, split="train", val_ratio=args.val_ratio)
        val_ds = MemmapEpisodeDataset(memmap_dir, split="val", val_ratio=args.val_ratio)
        obs_dim = train_ds.obs_dim
        print(f"[BC][memmap] train={len(train_ds)} episodes, val={len(val_ds)} episodes, "
              f"obs_dim={obs_dim}", flush=True)
        return train_ds, val_ds, obs_dim

    # ---- JSONL path (original) ----
    holdouts = {h.strip() for h in args.ood_holdout.split(",") if h.strip()}
    filtered = [f for f in files if not is_heldout_shard(f, holdouts)]
    ood = [f for f in files if f not in filtered]
    if ood:
        print(f"[split] OOD holdout files: {len(ood)} ({args.ood_holdout})", flush=True)

    files_train, files_val = split_files(filtered, args.split_mode, args.val_ratio)

    def _count_complete(fs):
        ds = StreamingEpisodeDataset(fs, strict_json=args.strict_json, report_every=max(10000, args.scan_log_every))
        c = 0
        for _ in ds: c += 1
        return c

    cache_in_ram = bool(getattr(args, "cache_in_ram", False))

    print(f"[BC][split] files: train={len(files_train)} val={len(files_val)}")
    if cache_in_ram:
        print("[BC][split] skipping episode count scans (--cache-in-ram; counts available after epoch 1)")
        train_eps = val_eps = -1  # unknown until first epoch
    else:
        train_eps = _count_complete(files_train); val_eps = _count_complete(files_val)
        print(f"[BC][split] complete episodes: train={train_eps} val={val_eps}")

    if val_eps == 0:
        print("[BC][split][warn] validation has 0 complete episodes with current flags "
              f"(split_mode={args.split_mode}, val_ratio={args.val_ratio}, ood_holdout='{args.ood_holdout}'). "
              "Consider --split-mode chronological or a larger --val-ratio.")

    if args.dataset_mode != "stream":
        raise ValueError("This drop-in supports dataset_mode=stream only (row or episode).")

    if getattr(args, "seq_mode", False):
        train_ds = StreamingEpisodeDataset(
            files_train,
            report_every=args.scan_log_every,
            max_rows=args.max_rows,
            strict_json=args.strict_json,
            split_mode=args.split_mode,
            val_ratio=args.val_ratio,
            want_val=False,
            cache_in_ram=cache_in_ram,
        )
        val_ds = StreamingEpisodeDataset(
            files_val,
            report_every=args.scan_log_every,
            max_rows=min(args.max_rows, 20000) if args.max_rows else 20000,
            strict_json=args.strict_json,
            split_mode=args.split_mode,
            val_ratio=args.val_ratio,
            want_val=True,
            cache_in_ram=cache_in_ram,
        )
        mode_str = "stream_seq"
    else:
        train_ds = StreamingJSONLDataset(
            files_train,
            shuffle_buffer=args.shuffle_buffer,
            report_every=args.scan_log_every,
            max_rows=args.max_rows,
            strict_json=args.strict_json,
            split_mode=args.split_mode,
            val_ratio=args.val_ratio,
            want_val=False,
        )
        val_ds = StreamingJSONLDataset(
            files_val,
            shuffle_buffer=max(1024, args.shuffle_buffer // 4),
            report_every=args.scan_log_every,
            max_rows=min(args.max_rows, 20000) if args.max_rows else 20000,
            strict_json=args.strict_json,
            split_mode=args.split_mode,
            val_ratio=args.val_ratio,
            want_val=True,
        )
        mode_str = "stream"

    train_ds._infer_obs_dim_once()
    obs_dim = train_ds.obs_dim
    print(f"[BC] dataset({mode_str}): train_files={len(files_train)} val_files={len(files_val)} (ood_holdout={len(ood)})", flush=True)
    return train_ds, val_ds, obs_dim

def build_dataloaders(train_ds, val_ds, args):
    is_memmap = isinstance(train_ds, MemmapEpisodeDataset)
    stream_mode = (args.dataset_mode == "stream")
    seq_mode = bool(getattr(args, "seq_mode", False))

    # Memmap always uses collate_seq (episodes)
    if is_memmap:
        coll = collate_seq
    else:
        coll = collate_seq if (stream_mode and seq_mode) else collate

    dl_common = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.workers > 0),
    )
    if args.workers > 0:
        dl_common["prefetch_factor"] = args.prefetch_factor

    # Map-style datasets (memmap) use DataLoader's built-in shuffling
    if is_memmap:
        train_shuffle = True
    else:
        train_shuffle = not seq_mode  # IterableDataset handles its own shuffle

    train_loader = DataLoader(
        train_ds, shuffle=train_shuffle, drop_last=True, collate_fn=coll, **dl_common
    )
    val_loader = DataLoader(
        val_ds, shuffle=False, drop_last=False, collate_fn=coll, **dl_common
    )
    return train_loader, val_loader

# ============================================================
#                    SCHEDULER (with warmup)
# ============================================================

def build_scheduler(optimizer, args, steps_per_epoch: int):
    if args.sched == "none":
        return None, (lambda step: None)
    if args.sched == "cosine":
        total_steps = max(1, args.epochs * max(1, steps_per_epoch))
        warmup = getattr(args, "warmup_steps", 0)
        cosine_steps = max(1, total_steps - warmup)
        sched = CosineAnnealingLR(optimizer, T_max=cosine_steps)
        def step_fn(global_step):
            if global_step < warmup:
                for pg in optimizer.param_groups:
                    base = pg.get("initial_lr", pg["lr"])
                    pg["lr"] = base * (float(global_step + 1) / float(max(1, warmup)))
            else:
                sched.step()
        return sched, step_fn
    if args.sched == "step":
        sched = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        return sched, (lambda _gs: sched.step())
    return None, (lambda step: None)

# ============================================================
#                        EMA Helper
# ============================================================

class EMAHelper:
    """
    CPU-resident EMA of model weights.
    Keeps state on CPU to save VRAM. Call update() after each optimizer step.
    """
    def __init__(self, decay: float, warmup_steps: int = 0):
        self.decay = float(decay)
        self.state: Optional[Dict[str, torch.Tensor]] = None
        self.warmup_steps = int(max(0, warmup_steps))

    def enabled(self) -> bool:
        return self.decay > 0.0

    def init_from(self, model: nn.Module):
        if not self.enabled() or self.state is not None: return
        self.state = {k: v.detach().clone().to("cpu") for k, v in model.state_dict().items()}
        print(f"[EMA] initialized (decay={self.decay}) with {len(self.state)} tensors.", flush=True)

    def update(self, model: nn.Module, global_opt_steps: int):
        if not self.enabled() or self.state is None: return
        if global_opt_steps < self.warmup_steps:  # warmup gate
            return
        with torch.no_grad():
            cur = model.state_dict()
            d = self.decay
            for k, v in cur.items():
                if k not in self.state:
                    self.state[k] = v.detach().clone().to("cpu")
                else:
                    self.state[k].mul_(d).add_(v.detach().to("cpu"), alpha=(1.0 - d))

    def apply_to(self, model: nn.Module) -> Optional[Dict[str, torch.Tensor]]:
        if not (self.enabled() and self.state): return None
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        missing, unexpected = model.load_state_dict(self.state, strict=False)
        if missing or unexpected:
            print(f"[EMA][warn] apply_to: missing={missing} unexpected={unexpected}", flush=True)
        return backup

    @staticmethod
    def restore(model: nn.Module, backup: Optional[Dict[str, torch.Tensor]]):
        if backup is not None:
            model.load_state_dict(backup, strict=False)

    def state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        return self.state

    def load_state_dict(self, state: Optional[Dict[str, torch.Tensor]]):
        self.state = {k: v.to("cpu") for k, v in (state or {}).items()} if state is not None else None
        if self.state is not None:
            print("[EMA] loaded state.", flush=True)

# ============================================================
#                         MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/datasets/obs/*.jsonl")

    # training
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=0,
                   help="Early stopping: halt after N epochs with no val_loss improvement (0=disabled)")
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--run-name", default="bc_lstm_seq")

    # dataset mode
    p.add_argument("--dataset-mode", default="stream", choices=["stream"])
    p.add_argument("--seq-mode", action="store_true", default=True)

    # streaming & IO
    p.add_argument("--shuffle-buffer", type=int, default=50000)
    p.add_argument("--scan-log-every", type=int, default=5000)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--mods", default="auto",
                   help="Comma list (e.g., 'tera'), 'auto' to detect, or '' to disable.")
    p.add_argument("--strict-json", action="store_true", default=False)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prefetch-factor", type=int, default=2)

    # model config — sequence core: --use-transformer (default) or --use-lstm
    p.add_argument("--use-transformer", action=argparse.BooleanOptionalAction, default=True,
                   help="Use causal transformer core (default). --no-use-transformer to disable.")
    p.add_argument("--use-lstm", action="store_true", default=False,
                   help="Use LSTM core instead of transformer (legacy).")
    p.add_argument("--n-transformer-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--transformer-dropout", type=float, default=0.1)
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--lstm-hidden", type=int, default=256)
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--lstm-layers", type=int, default=1)
    p.add_argument("--mlp-layers", type=int, default=2)
    p.add_argument("--hierarchical", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable hierarchical head (MvS → which) while keeping 9 logits outwardly.")
    p.add_argument("--step-type-bins", type=int, default=3,
                   help=">0 to enable step-type embeddings (e.g., 3 for early/mid/late).")
    p.add_argument("--ctx-extra-dim", type=int, default=41,
                   help="Reserve a small context vector concatenated before heads (0=off).")
    p.add_argument("--ctx-dropout", type=float, default=0.0,
                   help="Probability to zero ctx features during training (0.0 = off).")

    # stability/perf
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--deterministic", action="store_true", default=False)
    p.add_argument("--grad-accum", type=int, default=1)

    # scheduler
    p.add_argument("--sched", choices=["none","cosine","step"], default="cosine")
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--step-size", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--resume", default="")
    p.add_argument("--init-from", default="")
    p.add_argument("--auto-calibrate", action=argparse.BooleanOptionalAction, default=False)

    # split & OOD
    p.add_argument("--split-mode", choices=["hash_episode","chronological","hash_file"], default="chronological")
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--ood-holdout", default="MaxDamage,HazardSense")

    # logs
    p.add_argument("--log-csv", action="store_true", default=True)
    p.add_argument("--epoch-metrics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--print-epoch-steps", action=argparse.BooleanOptionalAction, default=True)

    # optimizer & EMA
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--ema", type=float, default=0.0, help="e.g., 0.999; 0 disables")
    p.add_argument("--ema-warmup-steps", type=int, default=0,
                   help="Do not update EMA for the first N optimizer steps.")
    p.add_argument("--use-ema-for-eval", action=argparse.BooleanOptionalAction, default=True)

    # label smoothing / value head
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="Legal-aware smoothing, e.g., 0.05")
    p.add_argument("--value-loss-weight", type=float, default=0.0,
               help=">0 to train value head against terminal result (per-step broadcast).")
    p.add_argument("--value-loss-bce", action=argparse.BooleanOptionalAction, default=True,
                   help="If true, BCEWithLogits vs {0,1} result; else MSE to float.")

    # hard-turn mining (hooks only)
    p.add_argument("--hardmine-pct", type=float, default=0.0,
                   help="0..1 fraction of recent hard turns to resurface next epoch (0=off).")
    p.add_argument("--hardmine-weight", type=float, default=2.0,
                   help="Relative frequency to sample hard turns when resurfaced.")

    # checkpoint mgmt
    p.add_argument("--topk", type=int, default=3,
                   help="Maintain best-K symlinks by val_loss (keep all epoch_* files regardless).")
    p.add_argument("--cache-in-ram", action="store_true", default=False,
                   help="Cache all episodes in RAM after first epoch scan (faster epochs, higher memory).")

    # memmap support
    p.add_argument("--data-format", choices=["jsonl", "memmap"], default="jsonl",
                   help="Data format: 'jsonl' (stream from text) or 'memmap' (pre-converted binary).")
    p.add_argument("--memmap-dir", default="src/data/datasets/memmap",
                   help="Directory containing memmap files (used when --data-format memmap).")

    # v5 embeddings
    p.add_argument("--n-entity-ids", type=int, default=0,
                   help="Number of entity ID slots (0=auto from data, 82=v5).")
    p.add_argument("--embed-dim", type=int, default=32,
                   help="Embedding dimension for entity IDs.")

    # advantage weighting
    p.add_argument("--advantage-weight", type=float, default=0.0,
                   help=">0 to enable advantage-weighted BC (scales policy loss by game result).")
    p.add_argument("--w-win", type=float, default=2.0,
                   help="Weight multiplier for winning episodes (used with --advantage-weight).")
    p.add_argument("--w-loss", type=float, default=0.5,
                   help="Weight multiplier for losing episodes (used with --advantage-weight).")

    # action class weighting
    p.add_argument("--action-class-weight", action="store_true", default=False,
                   help="Apply inverse-frequency weighting per action class to counter action imbalance.")

    return p.parse_args()

def _update_latest_and_topk(run_dir: Path, epoch_ckpt: Path, val_loss: float, topk: int):
    # latest symlink
    latest = run_dir / "latest.pt"
    try:
        if latest.exists() or latest.is_symlink(): latest.unlink()
        latest.symlink_to(epoch_ckpt.name)
    except Exception:
        pass

    # track topk in a small json index
    idx_path = run_dir / "best_index.json"
    data = {"items": []}
    if idx_path.exists():
        try:
            data = json.loads(idx_path.read_text())
        except Exception:
            data = {"items": []}
    items = data.get("items", [])
    items.append({"epoch": int(epoch_ckpt.stem.split("_")[-1]), "path": epoch_ckpt.name, "val_loss": float(val_loss)})
    items = sorted(items, key=lambda x: x["val_loss"])[:max(1, int(topk))]
    data["items"] = items
    idx_path.write_text(json.dumps(data, indent=2))

    # refresh symlinks best_1.pt, best_2.pt, ...
    for i, it in enumerate(items, 1):
        link = run_dir / f"best_{i}.pt"
        try:
            if link.exists() or link.is_symlink(): link.unlink()
            link.symlink_to(it["path"])
        except Exception:
            pass

def main():
    args = parse_args()
    global _GLOBAL_STEP_TYPE_BINS
    _GLOBAL_STEP_TYPE_BINS = int(args.step_type_bins) if int(getattr(args, 'step_type_bins', 0)) > 0 else 3
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    random.seed(args.seed); os.environ["PYTHONHASHSEED"] = str(args.seed)
    try:
        import numpy as _np; _np.random.seed(args.seed)
    except Exception: pass
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True

    is_memmap = (getattr(args, "data_format", "jsonl") == "memmap")

    if is_memmap:
        files = []  # not used for memmap path
        train_ds, val_ds, obs_dim = build_datasets(files, args)
        args._train_files = None
        # Infer slot dims and mod_keys from memmap metadata
        mv_dim = train_ds.move_slot_dim
        sw_dim = train_ds.switch_slot_dim
        eid_dim = train_ds.entity_ids_dim
        args._memmap_mod_keys = train_ds.mod_keys
        # Auto-set n_entity_ids from data if not explicitly provided
        if int(getattr(args, "n_entity_ids", 0)) == 0 and eid_dim > 0:
            args.n_entity_ids = eid_dim
        print(f"[BC][memmap] dims: move_slot={mv_dim} switch_slot={sw_dim} "
              f"entity_ids={eid_dim} mod_keys={train_ds.mod_keys}", flush=True)
    else:
        files = sorted(glob.glob(args.data))
        if not files: raise SystemExit(f"[BC][fatal] No files matched: {args.data}")
        print("[BC] Found %d file(s)/dir(s) for --data:" % len(files))
        for f in files: print("      -", f)
        train_ds, val_ds, obs_dim = build_datasets(files, args)
        args._train_files = getattr(train_ds, "files", None)
        # Infer slot dims from first row of data
        mv_dim = sw_dim = 0
        eid_dim = 0
        for row in iter_jsonl(files[:1], report_every=0):
            s = extract_sample(row)
            if s is None: continue
            if "move_slots" in s and s["move_slots"] is not None:
                mv_dim = len(s["move_slots"][0]) if len(s["move_slots"]) > 0 else 0
            if "switch_slots" in s and s["switch_slots"] is not None:
                sw_dim = len(s["switch_slots"][0]) if len(s["switch_slots"]) > 0 else 0
            if "entity_ids" in s and s["entity_ids"] is not None:
                eid_dim = len(s["entity_ids"])
            break
        if int(getattr(args, "n_entity_ids", 0)) == 0 and eid_dim > 0:
            args.n_entity_ids = eid_dim
        print(f"[BC] inferred dims from data: move_slot={mv_dim} switch_slot={sw_dim} entity_ids={eid_dim}", flush=True)

    train_loader, val_loader = build_dataloaders(train_ds, val_ds, args)

    # Compute action class weights (inverse-frequency) if requested
    action_class_weights = None
    if getattr(args, "action_class_weight", False):
        print("[BC] Computing action class weights from training data...", flush=True)
        if is_memmap:
            action_arr = np.load(str(Path(args.memmap_dir) / "action.npy"), mmap_mode='r')
            action_counts = np.bincount(action_arr[action_arr >= 0].astype(np.int64), minlength=9)
        else:
            action_counts = np.zeros(9, dtype=np.int64)
            for batch in train_loader:
                acts = batch.get("action", batch.get("actions"))
                if acts is not None:
                    flat = acts.numpy().flatten()
                    flat = flat[flat >= 0]
                    action_counts += np.bincount(flat.astype(np.int64), minlength=9)
                if action_counts.sum() > 500000:
                    break
        # Inverse frequency, normalized so mean weight = 1.0
        freq = action_counts / action_counts.sum().clip(1)
        inv_freq = 1.0 / freq.clip(1e-6)
        inv_freq = inv_freq / inv_freq.mean()  # normalize to mean=1
        action_class_weights = torch.tensor(inv_freq, dtype=torch.float32, device=device)
        print(f"[BC] Action counts: {action_counts.tolist()}", flush=True)
        print(f"[BC] Action class weights: {[f'{w:.3f}' for w in inv_freq]}", flush=True)

    model, meta = build_model(obs_dim, args, move_slot_dim=mv_dim, switch_slot_dim=sw_dim,
                              entity_ids_dim=eid_dim); model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for pg in opt.param_groups: pg.setdefault("initial_lr", pg["lr"])

    steps_per_epoch = args.steps_per_epoch
    try: steps_per_epoch = max(1, len(train_loader))
    except Exception: pass
    sched, sched_step = build_scheduler(opt, args, steps_per_epoch)

    scaler = torch.cuda.amp.GradScaler(enabled=(bool(args.amp) and str(device).startswith("cuda")))

    ema = EMAHelper(args.ema, warmup_steps=int(getattr(args, "ema_warmup_steps", 0)))
    if ema.enabled(): ema.init_from(model)

    start_epoch = 0; opt_steps = 0; global_step = 0
    if args.resume:
        print(f"[BC][resume] loading full state from: {args.resume}", flush=True)
        ckpt = torch.load(args.resume, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            print(f"[BC][resume][warn] state_dict mismatches: missing={missing} unexpected={unexpected}", flush=True)
        try: opt.load_state_dict(ckpt["opt"])
        except Exception as e: print(f"[BC][resume][warn] optimizer state not loaded: {e}", flush=True)
        if scaler is not None and ckpt.get("amp_scaler") is not None:
            try: scaler.load_state_dict(ckpt["amp_scaler"])
            except Exception as e: print(f"[BC][resume][warn] amp scaler not loaded: {e}", flush=True)
        if sched is not None and ckpt.get("sched") is not None:
            try: sched.load_state_dict(ckpt["sched"])
            except Exception as e: print(f"[BC][resume][warn] scheduler state not loaded: {e}", flush=True)
        if ema.enabled() and ("ema" in ckpt) and (ckpt["ema"] is not None):
            ema.load_state_dict(ckpt["ema"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        opt_steps   = int(ckpt.get("opt_steps", max(0, global_step // max(1, args.grad_accum))))
        print(f"[BC][resume] start_epoch={start_epoch} global_step={global_step} opt_steps={opt_steps}", flush=True)
    elif args.init_from:
        print(f"[BC][init] loading WEIGHTS ONLY from: {args.init_from}", flush=True)
        ckpt = torch.load(args.init_from, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            print(f"[BC][init][warn] state_dict mismatches: missing={missing} unexpected={unexpected}", flush=True)
        if ema.enabled() and ("ema" in ckpt) and (ckpt["ema"] is not None):
            ema.load_state_dict(ckpt["ema"])

    def _timestamp(): return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = Path("data/models/bc"); base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = (base_dir / args.run_name) if args.run_name else (base_dir / _timestamp())
    if run_dir.exists(): run_dir = base_dir / f"{args.run_name}-{_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tb = SummaryWriter(log_dir=str(Path("data/logs/tb/bc") / run_dir.name))
    csv_path = run_dir / "metrics.csv"
    if args.log_csv and not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["epoch","train_loss","val_loss_raw","val_loss_ema","lr","step"])

    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"created": _timestamp(), "resolved_run_dir": str(run_dir), "args": vars(args)}, f, indent=2)

    best_val = float("inf"); did_auto_calibrate = False
    epochs_no_improve = 0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Clear CUDA cache between epochs to reduce fragmentation
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if hasattr(train_ds, "set_epoch"): train_ds.set_epoch(epoch)
        if hasattr(val_ds, "set_epoch"): val_ds.set_epoch(epoch)

        opt_steps_at_epoch_start = opt_steps

        train_loss, train_pacc, train_macc, global_step, opt_steps = train_one_epoch(
            model, train_loader, opt, device, tb, global_step, args, scaler, sched_step,
            opt_steps=opt_steps,
            ema_update_fn=(lambda gos: ema.update(model, gos)) if ema.enabled() else None,
            action_class_weights=action_class_weights,
        )

        # Dual eval: raw AND EMA (if enabled)
        if ema.enabled():
            # raw
            val_loss_raw, val_pacc_raw, val_macc_raw = eval_one_epoch(model, val_loader, device, tb, global_step, args)
            # ema
            _bak = ema.apply_to(model)
            val_loss_ema, val_pacc_ema, val_macc_ema = eval_one_epoch(model, val_loader, device, tb, global_step, args)
            EMAHelper.restore(model, _bak)
            val_loss = val_loss_ema if args.use_ema_for_eval else val_loss_raw
            val_pacc = val_pacc_ema if args.use_ema_for_eval else val_pacc_raw
            val_macc = val_macc_ema if args.use_ema_for_eval else val_macc_raw
        else:
            val_loss_raw, val_pacc_raw, val_macc_raw = eval_one_epoch(model, val_loader, device, tb, global_step, args)
            val_loss_ema = val_pacc_ema = val_macc_ema = float("nan")
            val_loss, val_pacc, val_macc = val_loss_raw, val_pacc_raw, val_macc_raw

        cur_lr = opt.param_groups[0]["lr"]
        if tb:
            tb.add_scalar("train/lr", float(cur_lr), global_step)
            if ema.enabled(): tb.add_scalar("ema/decay", float(ema.decay), global_step)

        if tb and args.epoch_metrics:
            tb.add_scalars("epoch/loss", {"train": float(train_loss), "val_raw": float(val_loss_raw), "val_ema": float(val_loss_ema)}, epoch)
            tb.add_scalars("epoch/pacc", {"train": float(train_pacc), "val": float(val_pacc)}, epoch)
            tb.add_scalar("epoch/lr", float(cur_lr), epoch)
            tb.add_scalars("epoch/pacc_pct", {"train": float(train_pacc * 100.0), "val": float(val_pacc * 100.0)}, epoch)

        if args.print_epoch_steps:
            opt_steps_this_epoch = opt_steps - opt_steps_at_epoch_start
            print(f"[BC][epoch {epoch:03d}] optimizer steps this epoch = {opt_steps_this_epoch}  (running total = {opt_steps})", flush=True)

        warmup_done = (opt_steps >= args.warmup_steps)
        if args.auto_calibrate and args.sched == "cosine" and warmup_done and not did_auto_calibrate:
            measured_steps_per_epoch = max(1, opt_steps - opt_steps_at_epoch_start)
            remaining_epochs = max(0, args.epochs - epoch)
            remaining_steps  = remaining_epochs * measured_steps_per_epoch
            if remaining_steps > 0:
                current_lr = opt.param_groups[0]["lr"]
                sched = CosineAnnealingLR(opt, T_max=remaining_steps)
                for pg in opt.param_groups: pg["initial_lr"] = current_lr
                sched.base_lrs = [current_lr for _ in sched.base_lrs]
                def sched_step(new_opt_step: int): sched.step()
                did_auto_calibrate = True
                print(f"[BC][sched] Auto-calibrated cosine: steps/epoch~{measured_steps_per_epoch}, remaining_steps={remaining_steps}", flush=True)
            else:
                print("[BC][sched] Auto-calibration skipped (no remaining steps).", flush=True)
        elif args.auto_calibrate and not warmup_done and epoch == 1:
            print("[BC][sched] Warmup not finished; deferring auto-calibration until warmup completes.", flush=True)

        print(f"[BC][epoch {epoch}/{args.epochs}] "
              f"train_loss={train_loss:.4f} val_loss(raw)={val_loss_raw:.4f} val_loss(ema)={val_loss_ema:.4f} "
              f"train_pacc={train_pacc:.3f} val_pacc={val_pacc:.3f} train_macc={train_macc:.3f} val_macc={val_macc:.3f}", flush=True)

        # Save checkpoint
        policy_cfg = dict(meta["policy_cfg"]); policy_cfg.setdefault("action_dim", 9)
        ckpt = {
            "model": model.state_dict(),
            "obs_dim": obs_dim,
            "policy_cfg": policy_cfg,
            "args": vars(args),
            "epoch": epoch,
            "global_step": global_step,
            "opt_steps": opt_steps,
            "opt": opt.state_dict(),
            "amp_scaler": (scaler.state_dict() if scaler is not None else None),
            "sched": (sched.state_dict() if sched is not None else None),
            "ema": (ema.state_dict() if ema.enabled() else None),
        }
        ckpt_path = run_dir / f"epoch_{epoch:03d}.pt"
        torch.save(ckpt, ckpt_path)
        print(f"[BC] saved checkpoint: {ckpt_path}", flush=True)
        _update_latest_and_topk(run_dir, ckpt_path, val_loss, topk=int(getattr(args, "topk", 3)))

        if args.log_csv:
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([epoch, f"{train_loss:.6f}", f"{val_loss_raw:.6f}", f"{val_loss_ema:.6f}", f"{cur_lr:.8f}", global_step])

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(ckpt, run_dir / "best.pt")
        else:
            epochs_no_improve += 1

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"[BC] early stopping: no val_loss improvement for {args.patience} epochs", flush=True)
            break

    print("[BC] training complete.", flush=True)

if __name__ == "__main__":
    main()
