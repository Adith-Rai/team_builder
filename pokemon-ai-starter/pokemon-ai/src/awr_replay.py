#!/usr/bin/env python
# awr_replay.py — Replay buffer for AWR replay rehearsal during PPO.
#
# S68 Task #125 Phase 2B. See:
#   - docs/AWR_REPLAY_REHEARSAL_DESIGN.md           (full design)
#   - docs/REPLAY_REHEARSAL_AWR_VS_OFFPOLICY_PPO.md (AWR vs Off-Policy PPO)
#   - memory/project_plateau_hypothesis_and_experiments.md (why)
#
# Architecture:
#   - Reuses MemmapDataset (the SAME data loader BC v10 was trained on).
#   - Returns BC-format collated batches (output of dataset.collate_seq),
#     consumed by model.forward_sequence — NOT forward_ppo_sequence.
#     forward_sequence is the BC training path; it outputs the same
#     (B, T, n_actions) action_logits + (B, T, v_bins) v_logits that
#     AWR needs, in the format the memmap already produces. Skipping
#     the BC→PPO feat reshape avoids ~hundreds of lines of plumbing.
#   - Replay batches sampled per PPO iter; "batch_size" = number of
#     EPISODES (atomic units in the memmap). 16 episodes × avg 30 turns
#     ≈ 480 transitions — same ballpark as a PPO minibatch.
#
# What's NOT here (deferred to Phase 2C):
#   - The AWR loss term itself (lives in ppo.py near _ppo_loss_*_internal).
#   - Integration into ppo_update_batched / train_rl.py training loop.

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from dataset import MemmapDataset, collate_seq


_LOG = logging.getLogger(__name__)


class AWRReplayBuffer:
    """Replay buffer for AWR rehearsal.

    Wraps MemmapDataset. Sampling returns BC-format collated batches
    plus per-episode terminal_result for advantage computation.

    Args:
        memmap_path: path to a v8 memmap dir (must contain metadata.json
            + .npy files). Typically the SAME dir BC v10 was trained on
            (data/datasets/human_v8_memmap on the cloud pod). The
            MemmapDataset assertions enforce strict-equality on
            POKEMON_CONT/FIELD_CONT/TRANSITION_CONT dims; will fail loudly
            if features.py has drifted incompatibly. Move/switch cont
            dims auto zero-pad (see dataset.py:62-78).
        min_rating: Elo filter threshold (default 1500, matches BC v10
            training filter). NOTE v1 limitation: the current
            episode_index.npy is [start, length, hash] only — no rating
            column. If filtering can't be applied, a WARN is logged and
            ALL episodes are sampled. The human_v8_memmap was already
            pre-filtered to 1500+ during BC v10 data prep, so this is OK
            in practice for that specific dataset.
        val_ratio: hold-out fraction for MemmapDataset's train/val split
            (default 0.05). AWR samples from the 'train' split.
        rng_seed: optional RNG seed for reproducible sampling.

    Smoke test:
        python -m awr_replay <memmap_path> [--batch-size 16]
    """

    def __init__(
        self,
        memmap_path: str,
        min_rating: int = 1500,
        val_ratio: float = 0.05,
        rng_seed: Optional[int] = None,
    ):
        self.memmap_path = Path(memmap_path)
        self.min_rating = min_rating
        self.dataset = MemmapDataset(
            str(memmap_path), split="train", val_ratio=val_ratio
        )
        self.n_episodes = len(self.dataset)

        # Rating filter: only applies if episode_index has a rating column.
        # Current v8 layout is [start, length, hash]; flag this limitation
        # loudly so users know what's actually being sampled.
        ep_index_cols = self.dataset.episode_index.shape[1]
        self._rating_col_available = ep_index_cols >= 4
        if not self._rating_col_available:
            msg = (
                f"AWRReplayBuffer: episode_index has {ep_index_cols} cols "
                f"(expected >=4 for rating filter). Rating filter "
                f"--awr-min-rating={min_rating} will be IGNORED — sampling "
                f"from ALL {self.n_episodes} train-split episodes in "
                f"'{memmap_path}'. If this memmap was pre-filtered to "
                f"{min_rating}+ at creation time (true for human_v8_memmap), "
                f"this is fine in practice. Otherwise rebuild the memmap "
                f"with rating metadata."
            )
            _LOG.warning(msg)
            print(f"  [WARN] {msg}", file=sys.stderr, flush=True)
        else:
            # Future path: filter ep_indices by rating column. Not exercised
            # yet because v8 has no rating col; once a v9 memmap with rating
            # ships, drop this branch in and validate.
            ratings = self.dataset.episode_index[:, 3]
            keep = ratings >= min_rating
            keep_train = keep[self.dataset.ep_indices]
            self.dataset.ep_indices = self.dataset.ep_indices[keep_train]
            self.n_episodes = len(self.dataset.ep_indices)
            print(
                f"  [AWR] rating filter {min_rating}+ kept "
                f"{self.n_episodes} train-split episodes",
                flush=True,
            )

        if self.n_episodes == 0:
            raise ValueError(
                f"AWRReplayBuffer: no episodes after filtering "
                f"(rating>={min_rating}, train split). Check memmap_path "
                f"and rating threshold."
            )

        self._rng = np.random.default_rng(rng_seed)

        print(
            f"  [AWR] buffer ready: {self.n_episodes:,} episodes from "
            f"'{memmap_path}' (rating filter "
            f"{'ACTIVE' if self._rating_col_available else 'INACTIVE — see WARN'})",
            flush=True,
        )

    def __len__(self) -> int:
        return self.n_episodes

    def sample(self, batch_episodes: int, device: torch.device) -> dict:
        """Sample batch_episodes random episodes from the buffer.

        Returns:
            dict with the same keys as dataset.collate_seq output, PLUS:
              terminal_result: (B,) float32 — per-episode outcome
                  (+1 win, -1 loss, 0 tie). Sourced from the
                  per-turn 'result' field (constant within an episode
                  by construction in the memmap).
            All tensors moved to `device`.

        Behavior:
          - Sampling is WITHOUT REPLACEMENT within a single call
            (np.random.choice replace=False) but WITH REPLACEMENT across
            calls — different PPO iters re-roll independently. For 16
            episodes from 160K+ available, collision probability is
            negligible.
          - Episodes are NOT shuffled within themselves (sequence order
            preserved — required for temporal context correctness).
          - Episodes longer than 200 turns are NOT truncated here
            (collate_seq pads to the longest in the batch). The model's
            temporal context (cfg.temporal_context = 200) handles
            longer sequences internally via tail truncation.
        """
        if batch_episodes > self.n_episodes:
            raise ValueError(
                f"AWRReplayBuffer.sample: requested {batch_episodes} but "
                f"buffer has only {self.n_episodes} episodes"
            )

        idxs = self._rng.choice(
            self.n_episodes, size=batch_episodes, replace=False
        )
        items = [self.dataset[int(i)] for i in idxs]

        collated = collate_seq(items)

        # Per-episode terminal result, RESCALED to PPO's reward convention.
        # The memmap stores result as {0.0, 1.0} (loss, win) — see
        # train_bc.py + dataset.py:159-161 (only sets result when r >= 0;
        # confirmed via .npy distribution: 184k 0.0, 176k 1.0, no other
        # values).
        # PPO's terminal reward is {-1, +1} (rewards.py:140-141
        # terminal_sparse) and the value head is calibrated to [v_min,
        # v_max] = [-1.6, 1.6] (model_transformer.py:355-356) on PPO-scale
        # rewards. For AWR's advantage A = R - V_theta(s) to land in a
        # signed range consistent with what the value head was trained on,
        # we rescale: R = 2 * memmap_result - 1 → {-1, +1}. Without this,
        # losses would have R=0 vs V_theta∈[-1.6, 1.6] → A always > -1.6,
        # so the binary filter (1[A>0]) would weight loss-state actions
        # nearly as often as win-state actions, destroying the AWR signal.
        result_t = collated["result"]  # (B, T) in {0.0, 1.0, -1.0=unset}
        seq_lens = collated["seq_lens"]  # (B,)
        B = result_t.shape[0]
        terminal_result = torch.zeros(B, dtype=torch.float32)
        for b in range(B):
            T_b = int(seq_lens[b].item())
            if T_b == 0:
                terminal_result[b] = 0.0
                continue
            r_raw = float(result_t[b, T_b - 1].item())
            if r_raw < 0:
                # No terminal info recorded — episode is unusable for AWR.
                # Mark with NaN; downstream loss should mask these out.
                terminal_result[b] = float('nan')
            else:
                terminal_result[b] = 2.0 * r_raw - 1.0  # {0,1} → {-1,+1}
        collated["terminal_result"] = terminal_result

        # Move everything to device.
        out = {}
        for k, v in collated.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device, non_blocking=True)
            else:
                out[k] = v
        return out


# =============================
# Standalone smoke test
# =============================
#
# Usage:
#   python -m awr_replay data/datasets/memmap_v8 --batch-size 4
#
# Verifies:
#   1. Buffer loads without crashing
#   2. Sampling returns expected keys + shapes
#   3. terminal_result has expected value distribution
#   4. No NaN/inf in feature tensors

def _smoke_main():
    import argparse

    p = argparse.ArgumentParser(description="AWRReplayBuffer smoke test")
    p.add_argument("memmap_path", help="path to v8 memmap dir")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--min-rating", type=int, default=1500)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-samples", type=int, default=3,
                   help="how many sample() calls to run")
    p.add_argument("--ckpt", default=None,
                   help="optional checkpoint path to verify "
                        "model.forward_sequence accepts the batch format")
    args = p.parse_args()

    device = torch.device(args.device)

    print(f"=== AWRReplayBuffer smoke test ===")
    print(f"  memmap_path: {args.memmap_path}")
    print(f"  batch_episodes: {args.batch_size}")
    print(f"  min_rating: {args.min_rating}")
    print(f"  device: {device}")
    print()

    buf = AWRReplayBuffer(
        memmap_path=args.memmap_path,
        min_rating=args.min_rating,
        rng_seed=42,
    )
    print(f"  len(buffer): {len(buf):,}")
    print()

    expected_keys = {
        "our_pokemon_ids", "our_pokemon_banks", "our_pokemon_cont",
        "our_pokemon_move_ids", "our_pokemon_move_cont",
        "opp_pokemon_ids", "opp_pokemon_banks", "opp_pokemon_cont",
        "opp_pokemon_move_ids", "opp_pokemon_move_cont",
        "field_banks_raw", "field_cont_raw",
        "trans_ids_raw", "trans_cont_raw",
        "active_move_ids_raw", "active_move_banks_raw", "active_move_cont_raw",
        "switch_ids_raw", "switch_cont_raw",
        "legal_mask_raw", "action", "result", "mask", "seq_lens",
        "terminal_result",
    }

    for i in range(args.n_samples):
        print(f"--- sample #{i+1} ---")
        batch = buf.sample(args.batch_size, device)
        actual_keys = set(batch.keys())
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        if missing:
            print(f"  ERROR: missing keys: {sorted(missing)}")
        if extra:
            print(f"  INFO:  extra keys: {sorted(extra)}")
        if not missing and not extra:
            print(f"  keys: OK ({len(actual_keys)} keys)")

        B, T = batch["mask"].shape
        n_valid = int(batch["mask"].sum().item())
        print(f"  shape: B={B}, T_max={T}, n_valid_transitions={n_valid}")
        print(f"  seq_lens: {batch['seq_lens'].tolist()}")

        # Check terminal_result distribution (PPO-scale, {-1, +1, nan}).
        tr = batch["terminal_result"]
        n_nan = int(torch.isnan(tr).sum().item())
        finite = tr[~torch.isnan(tr)]
        n_wins = int((finite > 0.5).sum().item())
        n_losses = int((finite < -0.5).sum().item())
        n_other = int(finite.numel()) - n_wins - n_losses
        print(f"  terminal_result (PPO-scale {-1, +1}): "
              f"wins={n_wins}, losses={n_losses}, "
              f"other={n_other}, no-terminal-info(nan)={n_nan}")
        if not torch.isnan(tr).all():
            print(f"  terminal_result values: {tr.tolist()}")

        # Check no NaN/inf in continuous features
        any_nan = False
        for k in ["our_pokemon_cont", "opp_pokemon_cont", "field_cont_raw",
                  "trans_cont_raw", "active_move_cont_raw", "switch_cont_raw"]:
            t = batch[k]
            if torch.isnan(t).any() or torch.isinf(t).any():
                print(f"  ERROR: NaN/inf in {k}")
                any_nan = True
        if not any_nan:
            print(f"  feature sanity: OK (no NaN/inf in continuous tensors)")

        # Sanity check: at every valid position, action should be in [0, 8]
        # (9 actions: 4 moves + 5 switches). action_t is -1 at padding.
        valid_actions = batch["action"][batch["mask"].bool()]
        if valid_actions.numel() > 0:
            a_min = int(valid_actions.min().item())
            a_max = int(valid_actions.max().item())
            print(f"  action range at valid positions: [{a_min}, {a_max}] "
                  f"(expected [0, 8])")
            if a_min < 0 or a_max > 8:
                print(f"  WARNING: action out of expected range")

        print()

    # =========================================================
    # Round-trip check: does this batch actually flow through
    # model.forward_sequence? If yes, AWR loss can attach here.
    # =========================================================
    if args.ckpt:
        print("--- model.forward_sequence round-trip ---")
        from train_rl import load_checkpoint
        model, cfg, _ = load_checkpoint(args.ckpt, device)
        model.eval()
        with torch.no_grad():
            batch = buf.sample(args.batch_size, device)
            out = model.forward_sequence(batch, device)
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    has_nan = bool(torch.isnan(v).any())
                    print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}, "
                          f"nan={has_nan}")
            # Specifically check action_logits + v_logits (what AWR consumes)
            assert "action_logits" in out, "AWR needs action_logits"
            assert "v_logits" in out, "AWR needs v_logits"
            B_in = batch["our_pokemon_ids"].shape[0]
            T_in = batch["our_pokemon_ids"].shape[1]
            assert out["action_logits"].shape[:2] == (B_in, T_in), \
                f"action_logits shape mismatch: got {out['action_logits'].shape[:2]}, expected ({B_in}, {T_in})"
            assert out["v_logits"].shape[:2] == (B_in, T_in), \
                f"v_logits shape mismatch: got {out['v_logits'].shape[:2]}, expected ({B_in}, {T_in})"
            print(f"  OK: model.forward_sequence accepts batch + outputs "
                  f"(B={B_in}, T={T_in}) action_logits + v_logits")
        print()

    print("=== smoke test DONE ===")


if __name__ == "__main__":
    _smoke_main()
