# dataset.py
# Memmap dataset and collation for PokeTransformer.
#
# MemmapDataset: reads structured memmap files, returns episodes
# collate_seq: pads and batches episodes into the dict PokeTransformer.forward() expects
#
# Usage in train_bc.py:
#   from dataset import MemmapDataset, collate_seq
#   dataset = MemmapDataset(memmap_dir, split="train")
#   loader = DataLoader(dataset, collate_fn=collate_seq, ...)

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapDataset(Dataset):
    """Map-style dataset for v8 structured memmap files.

    Each __getitem__ returns {"episode": [list of sample dicts]}.
    Memmaps are opened lazily to avoid pickle issues with DataLoader workers.
    """

    def __init__(self, memmap_dir: str, split: str = "train", val_ratio: float = 0.1):
        super().__init__()
        self.memmap_dir = Path(memmap_dir)
        meta_path = self.memmap_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {memmap_dir}")
        with open(meta_path) as f:
            self.meta = json.load(f)
        assert self.meta.get("version") == "v8", f"Expected v8 memmap, got {self.meta.get('version')}"

        # Load episode index
        full_index = np.load(str(self.memmap_dir / "episode_index.npy"))
        hashes = full_index[:, 2].astype(np.float64) / float(2**63 - 1)
        if split == "val":
            mask = hashes < val_ratio
        else:
            mask = hashes >= val_ratio
        self.ep_indices = np.where(mask)[0]
        self.episode_index = full_index
        self._memmaps = None

    def _open_memmaps(self):
        if self._memmaps is not None:
            return
        d = self.memmap_dir
        N = self.meta["num_records"]
        pcd = self.meta["poke_cont_dim"]
        fcd = self.meta["field_cont_dim"]
        tcd = self.meta["trans_cont_dim"]
        mcd = self.meta["move_cont_dim"]
        scd = self.meta["switch_cont_dim"]

        # Track dimension gaps for zero-padding in collate
        from features import POKEMON_CONT_DIM, FIELD_CONT_DIM, TRANSITION_CONT_DIM, MOVE_SLOT_CONT_DIM, SWITCH_SLOT_CONT_DIM
        self._pad_move = max(0, MOVE_SLOT_CONT_DIM - mcd)
        self._pad_switch = max(0, SWITCH_SLOT_CONT_DIM - scd)
        if self._pad_move > 0 or self._pad_switch > 0:
            import logging, sys
            msg = (
                f"Memmap dimension mismatch (will zero-pad): "
                f"move_cont {mcd}->{MOVE_SLOT_CONT_DIM} (+{self._pad_move}), "
                f"switch_cont {scd}->{SWITCH_SLOT_CONT_DIM} (+{self._pad_switch}). "
                f"Consider regenerating memmaps with current features.py."
            )
            logging.getLogger("pokemon_ai").warning(msg)
            # Also print so it's visible when no logging handler is configured
            # (train_bc.py / train_rl.py don't set one up by default).
            print(f"  [WARN] {msg}", file=sys.stderr, flush=True)
        assert pcd == POKEMON_CONT_DIM, f"pokemon_cont_dim mismatch: memmap has {pcd}, features.py expects {POKEMON_CONT_DIM}"
        assert fcd == FIELD_CONT_DIM, f"field_cont_dim mismatch: memmap has {fcd}, features.py expects {FIELD_CONT_DIM}"
        assert tcd == TRANSITION_CONT_DIM, f"trans_cont_dim mismatch: memmap has {tcd}, features.py expects {TRANSITION_CONT_DIM}"

        def _mm(name, shape, dtype):
            return np.memmap(str(d / name), dtype=dtype, mode="r", shape=shape)

        mm = {}
        for side in ["our", "opp"]:
            mm[f"{side}_pokemon_ids"] = _mm(f"{side}_pokemon_ids.npy", (N, 6, 7), np.int32)
            mm[f"{side}_pokemon_banks"] = _mm(f"{side}_pokemon_banks.npy", (N, 6, 10), np.int32)
            mm[f"{side}_pokemon_cont"] = _mm(f"{side}_pokemon_cont.npy", (N, 6, pcd), np.float32)
            mm[f"{side}_pokemon_mcont"] = _mm(f"{side}_pokemon_mcont.npy", (N, 6, 4, 23), np.float32)
        mm["field_banks"] = _mm("field_banks.npy", (N, 4), np.int32)
        mm["field_cont"] = _mm("field_cont.npy", (N, fcd), np.float32)
        mm["trans_ids"] = _mm("trans_ids.npy", (N, 2), np.int32)
        mm["trans_cont"] = _mm("trans_cont.npy", (N, tcd), np.float32)
        mm["move_ids"] = _mm("move_ids.npy", (N, 4), np.int32)
        mm["move_banks"] = _mm("move_banks.npy", (N, 4, 4), np.int32)
        mm["move_cont"] = _mm("move_cont.npy", (N, 4, mcd), np.float32)
        mm["switch_ids"] = _mm("switch_ids.npy", (N, 5), np.int32)
        mm["switch_cont"] = _mm("switch_cont.npy", (N, 5, scd), np.float32)
        mm["legal"] = _mm("legal.npy", (N, 9), np.float32)
        mm["action"] = _mm("action.npy", (N,), np.int32)
        mm["result"] = _mm("result.npy", (N,), np.float32)
        mm["turn"] = _mm("turn.npy", (N,), np.int32)
        self._memmaps = mm

    @staticmethod
    def _pad_array(arr: np.ndarray, pad: int) -> np.ndarray:
        """Zero-pad the last dimension of an array if pad > 0."""
        if pad <= 0:
            return arr
        pad_shape = list(arr.shape)
        pad_shape[-1] = pad
        return np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=-1)

    def __len__(self):
        return len(self.ep_indices)

    def __getitem__(self, idx):
        self._open_memmaps()
        real_idx = self.ep_indices[idx]
        start, length, _ = self.episode_index[real_idx]
        start, length = int(start), int(length)
        end = start + length
        mm = self._memmaps

        samples = []
        for i in range(start, end):
            # No .copy() needed — memmap views are read-only and collate copies
            # into pre-allocated tensors. .astype() in collate creates new arrays
            # for int fields. _pad_array creates new arrays when padding is needed.
            s = {
                # Pokemon (both sides)
                "our_pokemon_ids": mm["our_pokemon_ids"][i],       # (6, 7) int32
                "our_pokemon_banks": mm["our_pokemon_banks"][i],   # (6, 10) int32
                "our_pokemon_cont": mm["our_pokemon_cont"][i],     # (6, D) float32
                "our_pokemon_mcont": mm["our_pokemon_mcont"][i],   # (6, 4, 23) float32
                "opp_pokemon_ids": mm["opp_pokemon_ids"][i],
                "opp_pokemon_banks": mm["opp_pokemon_banks"][i],
                "opp_pokemon_cont": mm["opp_pokemon_cont"][i],
                "opp_pokemon_mcont": mm["opp_pokemon_mcont"][i],
                # Field
                "field_banks": mm["field_banks"][i],               # (4,) int32
                "field_cont": mm["field_cont"][i],                 # (D,) float32
                # Transition
                "trans_ids": mm["trans_ids"][i],                    # (2,) int32
                "trans_cont": mm["trans_cont"][i],                 # (D,) float32
                # Active moves
                "move_ids": mm["move_ids"][i],                     # (4,) int32
                "move_banks": mm["move_banks"][i],                 # (4, 4) int32
                "move_cont": self._pad_array(mm["move_cont"][i], self._pad_move),  # (4, mcd) float32
                # Switches
                "switch_ids": mm["switch_ids"][i],                 # (5,) int32
                "switch_cont": self._pad_array(mm["switch_cont"][i], self._pad_switch),  # (5, scd) float32
                # Action/legal
                "legal": mm["legal"][i],                           # (9,) float32
                "action": int(mm["action"][i]),
                "t": int(mm["turn"][i]),
                "done": (i == end - 1),
            }
            r = float(mm["result"][i])
            if r >= 0:
                s["result"] = r
            samples.append(s)

        return {"episode": samples}


def collate_seq(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate v8 episodes into padded batch tensors for PokeTransformer.

    Input: list of {"episode": [list of sample dicts]} from MemmapDataset
    Output: dict matching PokeTransformer.forward() batch argument
    """
    episodes = [item["episode"] for item in batch]
    B = len(episodes)
    T = max(len(ep) for ep in episodes)

    # Get shapes from first sample
    s0 = episodes[0][0]
    poke_cont_dim = s0["our_pokemon_cont"].shape[1]
    field_cont_dim = s0["field_cont"].shape[0]
    trans_cont_dim = s0["trans_cont"].shape[0]
    move_cont_dim = s0["move_cont"].shape[1]
    switch_cont_dim = s0["switch_cont"].shape[1]

    # Pre-allocate tensors
    our_poke_ids = torch.zeros(B, T, 6, 7, dtype=torch.long)
    our_poke_banks = torch.zeros(B, T, 6, 10, dtype=torch.long)
    our_poke_cont = torch.zeros(B, T, 6, poke_cont_dim, dtype=torch.float32)
    our_poke_mcont = torch.zeros(B, T, 6, 4, 23, dtype=torch.float32)

    opp_poke_ids = torch.zeros(B, T, 6, 7, dtype=torch.long)
    opp_poke_banks = torch.zeros(B, T, 6, 10, dtype=torch.long)
    opp_poke_cont = torch.zeros(B, T, 6, poke_cont_dim, dtype=torch.float32)
    opp_poke_mcont = torch.zeros(B, T, 6, 4, 23, dtype=torch.float32)

    field_banks_t = torch.zeros(B, T, 4, dtype=torch.long)
    field_cont_t = torch.zeros(B, T, field_cont_dim, dtype=torch.float32)

    trans_ids_t = torch.zeros(B, T, 2, dtype=torch.long)
    trans_cont_t = torch.zeros(B, T, trans_cont_dim, dtype=torch.float32)

    active_move_ids = torch.zeros(B, T, 4, dtype=torch.long)
    active_move_banks = torch.zeros(B, T, 4, 4, dtype=torch.long)
    active_move_cont = torch.zeros(B, T, 4, move_cont_dim, dtype=torch.float32)

    switch_ids_t = torch.zeros(B, T, 5, dtype=torch.long)
    switch_cont_t = torch.zeros(B, T, 5, switch_cont_dim, dtype=torch.float32)

    legal_t = torch.zeros(B, T, 9, dtype=torch.float32)
    action_t = torch.full((B, T), -1, dtype=torch.long)
    result_t = torch.full((B, T), -1.0, dtype=torch.float32)
    mask_t = torch.zeros(B, T, dtype=torch.float32)
    seq_lens = torch.zeros(B, dtype=torch.long)

    for b, ep in enumerate(episodes):
        L = len(ep)
        seq_lens[b] = L
        mask_t[b, :L] = 1.0

        # Stack all turns for this episode at once (numpy), then convert to tensor once
        our_poke_ids[b, :L] = torch.from_numpy(np.stack([s["our_pokemon_ids"] for s in ep]).astype(np.int64))
        our_poke_banks[b, :L] = torch.from_numpy(np.stack([s["our_pokemon_banks"] for s in ep]).astype(np.int64))
        our_poke_cont[b, :L] = torch.from_numpy(np.stack([s["our_pokemon_cont"] for s in ep]))
        our_poke_mcont[b, :L] = torch.from_numpy(np.stack([s["our_pokemon_mcont"] for s in ep]))

        opp_poke_ids[b, :L] = torch.from_numpy(np.stack([s["opp_pokemon_ids"] for s in ep]).astype(np.int64))
        opp_poke_banks[b, :L] = torch.from_numpy(np.stack([s["opp_pokemon_banks"] for s in ep]).astype(np.int64))
        opp_poke_cont[b, :L] = torch.from_numpy(np.stack([s["opp_pokemon_cont"] for s in ep]))
        opp_poke_mcont[b, :L] = torch.from_numpy(np.stack([s["opp_pokemon_mcont"] for s in ep]))

        field_banks_t[b, :L] = torch.from_numpy(np.stack([s["field_banks"] for s in ep]).astype(np.int64))
        field_cont_t[b, :L] = torch.from_numpy(np.stack([s["field_cont"] for s in ep]))

        trans_ids_t[b, :L] = torch.from_numpy(np.stack([s["trans_ids"] for s in ep]).astype(np.int64))
        trans_cont_t[b, :L] = torch.from_numpy(np.stack([s["trans_cont"] for s in ep]))

        active_move_ids[b, :L] = torch.from_numpy(np.stack([s["move_ids"] for s in ep]).astype(np.int64))
        active_move_banks[b, :L] = torch.from_numpy(np.stack([s["move_banks"] for s in ep]).astype(np.int64))
        active_move_cont[b, :L] = torch.from_numpy(np.stack([s["move_cont"] for s in ep]))

        switch_ids_t[b, :L] = torch.from_numpy(np.stack([s["switch_ids"] for s in ep]).astype(np.int64))
        switch_cont_t[b, :L] = torch.from_numpy(np.stack([s["switch_cont"] for s in ep]))

        legal_t[b, :L] = torch.from_numpy(np.stack([s["legal"] for s in ep]))
        actions_np = np.array([s["action"] for s in ep], dtype=np.int64)
        action_t[b, :L] = torch.from_numpy(actions_np)
        results_np = np.array([s.get("result", -1.0) for s in ep], dtype=np.float32)
        result_t[b, :L] = torch.from_numpy(results_np)

    # The PokeTransformer processes turns sequentially, building up history.
    # For BC training, we process each turn with its accumulated history.
    # The collate returns the full sequence; the training loop handles temporal buffering.

    return {
        # Pokemon features — indexed as [B, T, 6, ...]
        "our_pokemon_ids": our_poke_ids,
        "our_pokemon_banks": our_poke_banks,
        "our_pokemon_cont": our_poke_cont,
        "our_pokemon_move_ids": our_poke_ids[:, :, :, 3:7],  # extract move IDs from pokemon_ids
        "our_pokemon_move_cont": our_poke_mcont,
        "opp_pokemon_ids": opp_poke_ids,
        "opp_pokemon_banks": opp_poke_banks,
        "opp_pokemon_cont": opp_poke_cont,
        "opp_pokemon_move_ids": opp_poke_ids[:, :, :, 3:7],
        "opp_pokemon_move_cont": opp_poke_mcont,
        # Field — need to unpack into dict for model
        "field_banks_raw": field_banks_t,  # [B, T, 4] — training loop unpacks to dict
        "field_cont_raw": field_cont_t,    # [B, T, field_cont_dim]
        # Transition
        "trans_ids_raw": trans_ids_t,      # [B, T, 2]
        "trans_cont_raw": trans_cont_t,    # [B, T, trans_cont_dim]
        # Active moves
        "active_move_ids_raw": active_move_ids,    # [B, T, 4]
        "active_move_banks_raw": active_move_banks, # [B, T, 4, 4]
        "active_move_cont_raw": active_move_cont,  # [B, T, 4, move_cont_dim]
        # Switches
        "switch_ids_raw": switch_ids_t,     # [B, T, 5]
        "switch_cont_raw": switch_cont_t,   # [B, T, 5, switch_cont_dim]
        # Action / mask
        "legal_mask_raw": legal_t,          # [B, T, 9]
        "action": action_t,                 # [B, T]
        "result": result_t,                 # [B, T]
        "mask": mask_t,                     # [B, T]
        "seq_lens": seq_lens,               # [B]
    }


def unpack_turn_batch(collated: dict, t: int, device: torch.device) -> dict:
    """Extract a single turn's features from the collated batch and format
    them as the dict that PokeTransformer.forward() expects.

    collated: output of collate_seq (all [B, T, ...])
    t: turn index (0-based)
    device: torch device

    Returns: dict ready for model.forward(batch=...)
    """
    B = collated["our_pokemon_ids"].shape[0]

    # Pokemon IDs: [B, 6, 3] (species, item, ability only — model indexes by position)
    our_poke_ids = collated["our_pokemon_ids"][:, t, :, :3].to(device)  # [B, 6, 3]
    opp_poke_ids = collated["opp_pokemon_ids"][:, t, :, :3].to(device)

    # Pokemon banks: [B, 6, 10]
    our_poke_banks = collated["our_pokemon_banks"][:, t].to(device)
    opp_poke_banks = collated["opp_pokemon_banks"][:, t].to(device)

    # Pokemon continuous: [B, 6, D]
    our_poke_cont = collated["our_pokemon_cont"][:, t].to(device)
    opp_poke_cont = collated["opp_pokemon_cont"][:, t].to(device)

    # Pokemon move IDs: [B, 6, 4]
    our_poke_move_ids = collated["our_pokemon_move_ids"][:, t].to(device)
    opp_poke_move_ids = collated["opp_pokemon_move_ids"][:, t].to(device)

    # Pokemon move cont: [B, 6, 4, 23]
    our_poke_mcont = collated["our_pokemon_move_cont"][:, t].to(device)
    opp_poke_mcont = collated["opp_pokemon_move_cont"][:, t].to(device)

    # Field banks → dict
    fb = collated["field_banks_raw"][:, t].to(device)  # [B, 4]
    field_banks = {
        "turn": fb[:, 0], "weather_dur": fb[:, 1],
        "terrain_dur": fb[:, 2], "tr_dur": fb[:, 3],
    }
    field_cont = collated["field_cont_raw"][:, t].to(device)

    # Transition ids → dict
    ti = collated["trans_ids_raw"][:, t].to(device)  # [B, 2]
    transition_ids = {"our_action": ti[:, 0], "opp_action": ti[:, 1]}
    transition_cont = collated["trans_cont_raw"][:, t].to(device)

    # Active moves
    active_move_ids = collated["active_move_ids_raw"][:, t].to(device)  # [B, 4]
    amb = collated["active_move_banks_raw"][:, t].to(device)  # [B, 4, 4]
    active_move_banks = {
        "bp": amb[:, :, 0], "acc": amb[:, :, 1],
        "pp": amb[:, :, 2], "prio": amb[:, :, 3],
    }
    active_move_cont = collated["active_move_cont_raw"][:, t].to(device)

    # Switches
    switch_ids = collated["switch_ids_raw"][:, t].to(device)
    switch_cont = collated["switch_cont_raw"][:, t].to(device)

    # Legal mask
    legal_mask = collated["legal_mask_raw"][:, t].to(device)

    return {
        "our_pokemon_ids": our_poke_ids,
        "our_pokemon_banks": our_poke_banks,
        "our_pokemon_cont": our_poke_cont,
        "our_pokemon_move_ids": our_poke_move_ids,
        "our_pokemon_move_cont": our_poke_mcont,
        "opp_pokemon_ids": opp_poke_ids,
        "opp_pokemon_banks": opp_poke_banks,
        "opp_pokemon_cont": opp_poke_cont,
        "opp_pokemon_move_ids": opp_poke_move_ids,
        "opp_pokemon_move_cont": opp_poke_mcont,
        "field_banks": field_banks,
        "field_cont": field_cont,
        "transition_ids": transition_ids,
        "transition_cont": transition_cont,
        "active_move_ids": active_move_ids,
        "active_move_banks": active_move_banks,
        "active_move_cont": active_move_cont,
        "switch_ids": switch_ids,
        "switch_cont": switch_cont,
        "legal_mask": legal_mask,
    }
