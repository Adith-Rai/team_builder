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
import torch.nn.functional as F

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
# BC-format -> PPO-format adapter (Phase 2C v2, Task #125)
# =============================
#
# AWRReplayBuffer outputs BC-format batches (dataset.collate_seq), but
# AWR's forward goes through forward_ppo_sequence (the same path Tier 3
# PPO uses) for consistency. forward_ppo_sequence expects PPO-format
# (ppo.collate_episodes output) which differs structurally:
#   - field_banks_raw (B,L,4)       -> field_banks dict of 4 (B,L) tensors
#   - trans_ids_raw (B,L,2)         -> transition_ids dict of 2 (B,L) tensors
#   - active_move_banks_raw (B,L,4,4) -> active_move_banks dict of 4 (B,L,4)
#   - *_raw keys                     -> non-raw keys
#   - flat top level                 -> {"feat_batches": ..., "pad_mask": ...}
# This adapter does that conversion in O(1) tensor views (no copies).

def bc_to_ppo_format(bc_collated: dict) -> dict:
    """Convert AWRReplayBuffer.sample() output to forward_ppo_sequence input.

    No copies (tensor views + slices). The terminal_result field is NOT
    forwarded to PPO format — AWR keeps it at top level for advantage
    computation outside the model.
    """
    B, L = bc_collated["our_pokemon_ids"].shape[:2]
    device = bc_collated["our_pokemon_ids"].device
    feat_batches = {
        "our_pokemon_ids":      bc_collated["our_pokemon_ids"],
        "our_pokemon_banks":    bc_collated["our_pokemon_banks"],
        "our_pokemon_cont":     bc_collated["our_pokemon_cont"],
        "our_pokemon_move_ids": bc_collated["our_pokemon_move_ids"],
        "our_pokemon_move_cont": bc_collated["our_pokemon_move_cont"],
        "opp_pokemon_ids":      bc_collated["opp_pokemon_ids"],
        "opp_pokemon_banks":    bc_collated["opp_pokemon_banks"],
        "opp_pokemon_cont":     bc_collated["opp_pokemon_cont"],
        "opp_pokemon_move_ids": bc_collated["opp_pokemon_move_ids"],
        "opp_pokemon_move_cont": bc_collated["opp_pokemon_move_cont"],
        "field_cont":           bc_collated["field_cont_raw"],
        "transition_cont":      bc_collated["trans_cont_raw"],
        "active_move_ids":      bc_collated["active_move_ids_raw"],
        "active_move_cont":     bc_collated["active_move_cont_raw"],
        "switch_ids":           bc_collated["switch_ids_raw"],
        "switch_cont":          bc_collated["switch_cont_raw"],
        "legal_mask":           bc_collated["legal_mask_raw"],
    }
    fb = bc_collated["field_banks_raw"]
    feat_batches["field_banks"] = {
        "turn":        fb[:, :, 0],
        "weather_dur": fb[:, :, 1],
        "terrain_dur": fb[:, :, 2],
        "tr_dur":      fb[:, :, 3],
    }
    ti = bc_collated["trans_ids_raw"]
    feat_batches["transition_ids"] = {
        "our_action": ti[:, :, 0],
        "opp_action": ti[:, :, 1],
    }
    amb = bc_collated["active_move_banks_raw"]
    feat_batches["active_move_banks"] = {
        "bp":   amb[:, :, :, 0],
        "acc":  amb[:, :, :, 1],
        "pp":   amb[:, :, :, 2],
        "prio": amb[:, :, :, 3],
    }
    feat_batches["gen_id"] = torch.full(
        (B, L), 9, dtype=torch.long, device=device
    )

    mask = bc_collated["mask"]
    pad_mask = (mask > 0.5)

    return {
        "feat_batches": feat_batches,
        "pad_mask": pad_mask,
        "seq_lens": bc_collated["seq_lens"],
        "B": B,
        "L_max": L,
    }


# =============================
# AWR loss + optimizer step (Phase 2C, Task #125)
# =============================
#
# compute_awr_loss is pure math (forward → loss + diagnostics dict).
# awr_step wraps it with optimizer.zero_grad + backward + grad_clip + step.
# Called from train_rl.py AFTER ppo_update_batched returns, when
# --awr-replay-memmap is set AND not in warmup. Single optimizer step
# per PPO iter (one forward, one backward, one step) for AWR's part.
#
# Why detach V_θ for advantage (default in v1):
#   AWR with terminal reward can pull V_θ toward "win probability" while
#   PPO with shaped reward pulls V_θ toward "discounted shaped return."
#   With terminal-only PPO + binary AWR (the planned setup), the two
#   reward distributions match → detach matters less. But we keep
#   detach=True in v1 for surface area reduction: AWR updates POLICY
#   only, value head is trained exclusively by PPO. Disable via
#   `detach_value=False` once we want to ablate.

def compute_awr_loss(
    model,
    replay_batch: dict,
    device: torch.device,
    awr_binary: bool = True,
    awr_beta: float = 1.0,
    awr_clip_high: float = 20.0,
    detach_value: bool = True,
) -> dict:
    """Compute AWR loss for a replay batch (pure math, no optimizer).

    Loss formula:
        L = - E_(s,a)~replay [ w(s,a) * log pi_theta(a | s) ]

        w(s,a) = 1[A(s,a) > 0]                       if awr_binary
               = clip(exp(A(s,a)/beta), max=clip_high) otherwise

        A(s,a) = R - V_theta(s)

    Where:
        R = terminal_result of the episode (broadcast to all states).
            In {-1, +1, NaN}. NaN means no terminal info — those
            positions are masked out.
        V_theta(s) = scalar value head output (detached by default).

    Aggregation: masked-mean over (B, T) positions where both the
    pad mask AND the terminal-result-validity mask are True. Same
    aggregation style as the BC anchor KL term.

    Args:
        model: TransformerBattlePolicy (will run forward_sequence on it)
        replay_batch: output of AWRReplayBuffer.sample() — BC-format
            with terminal_result (B,) tensor in {-1, +1, NaN}
        device: torch device
        awr_binary: if True (default), use 1[A>0] weight. Mirrors
            metamon binary_rl.gin (SyntheticRLV2 / Minikazam paradigm).
        awr_beta: temperature for exp(A/beta). Only used if not binary.
        awr_clip_high: max weight to prevent extreme single-sample
            domination. Only matters for the exp variant (binary
            weights are {0, 1}).
        detach_value: if True (default), V_theta is detached for the
            advantage. AWR only updates policy. See docstring rationale.
            NOTE: for the BINARY variant, value head gradient is blocked
            REGARDLESS of detach_value, because `(A > 0)` is a
            non-differentiable comparison. detach_value only matters
            for the exp(A/beta) variant. Verified end-to-end with all
            4 (binary x detach) combos at Phase 2C.

    Returns:
        dict (all scalar tensors except 'loss' which has grad):
            loss:               scalar AWR loss (with grad)
            advantage_mean:     mean A over valid positions
            advantage_pos_frac: fraction of valid positions with A > 0
            weight_max:         max weight applied
            weight_mean:        mean weight over valid positions
            n_valid:            number of (B, T) positions with valid R + non-pad
            n_total_valid_pad:  number of non-pad positions (for ratio)
    """
    # Use forward_ppo_sequence — the SAME forward path Tier 3 PPO uses
    # for updates. Critical: forward_sequence (BC path) and
    # forward_ppo_sequence (PPO/Tier 3 path) DIVERGE by ~1.9e-3 in
    # action_logits and ~3.6e-3 in v_logits at valid positions, verified
    # via test_forward_paths_equivalence.py. Using forward_sequence here
    # would pull the policy toward an inconsistent distribution vs PPO's
    # updates -> active pollution.
    # Adapter bc_to_ppo_format converts dataset.collate_seq output to
    # the dict format forward_ppo_sequence expects.
    ppo_batch = bc_to_ppo_format(replay_batch)
    out = model.forward_ppo_sequence(ppo_batch, device)
    action_logits = out["action_logits"]  # (B, L_max, n_actions)
    value = out["value"]                  # (B, L_max) scalar V_theta

    if detach_value:
        value = value.detach()

    mask = replay_batch["mask"]                       # (B, L_max) {0, 1}
    actions = replay_batch["action"]                  # (B, L_max) long, -1 at pad
    terminal_result = replay_batch["terminal_result"] # (B,) in {-1, +1, NaN}

    B, T = mask.shape

    # Broadcast terminal_result to per-state R.
    R = terminal_result.unsqueeze(1).expand(B, T)  # (B, T)

    # Valid mask: non-pad AND R is finite (NaN means no terminal info,
    # those episodes can't contribute to AWR).
    valid_R = ~torch.isnan(R)
    valid_mask = (mask > 0.5) & valid_R
    valid_mask_f = valid_mask.float()
    n_valid = valid_mask_f.sum().clamp(min=1.0)

    # Replace NaN in R with 0 to avoid NaN propagation; masking handles them.
    R_clean = torch.nan_to_num(R, nan=0.0)

    # Advantage. Float for numerical safety (value head outputs may be
    # in autocast dtype like bf16).
    advantage = R_clean - value.float()  # (B, T)

    # Weight per position.
    if awr_binary:
        weight = (advantage > 0).float()
    else:
        weight = torch.exp(advantage / awr_beta).clamp(max=awr_clip_high)

    # Log probability of the chosen action. Pad-positions have action=-1;
    # clamp to a valid index (we mask the loss anyway, so the actual
    # value doesn't propagate).
    actions_clean = actions.clamp(min=0)
    log_probs = F.log_softmax(action_logits.float(), dim=-1)  # (B, T, n_actions)
    chosen_log_probs = log_probs.gather(
        -1, actions_clean.unsqueeze(-1)
    ).squeeze(-1)  # (B, T)

    # Per-position AWR loss, masked-mean over valid positions.
    awr_loss_per_pos = -weight * chosen_log_probs
    awr_loss = (awr_loss_per_pos * valid_mask_f).sum() / n_valid

    # Diagnostics.
    weighted_mask = weight * valid_mask_f
    pos_advantage_frac = ((advantage > 0).float() * valid_mask_f).sum() / n_valid
    weight_max_val = (weight * valid_mask_f).max()
    weight_mean_val = weighted_mask.sum() / n_valid
    advantage_mean = (advantage * valid_mask_f).sum() / n_valid

    return {
        "loss": awr_loss,
        "advantage_mean": advantage_mean,
        "advantage_pos_frac": pos_advantage_frac,
        "weight_max": weight_max_val,
        "weight_mean": weight_mean_val,
        "n_valid": n_valid,
        "n_total_valid_pad": (mask > 0.5).float().sum(),
    }


def awr_step(
    model,
    optimizer,
    replay_buffer: "AWRReplayBuffer",
    device: torch.device,
    awr_batch_episodes: int,
    awr_mix_weight: float,
    awr_binary: bool = True,
    awr_beta: float = 1.0,
    awr_clip_high: float = 20.0,
    max_grad_norm: float = 0.5,
    detach_value: bool = True,
) -> dict:
    """Single AWR optimizer step. Called after ppo_update_batched.

    Pattern mirrors a PPO step: zero_grad → forward → loss × mix_weight
    → backward → clip → step. One step per PPO iter.

    The AWR forward shares the optimizer with PPO (same AdamW). PPO's
    accumulated state at zero_grad ensures the AWR backward starts
    clean; PPO's optimizer.step at the end of its own loop already
    flushed PPO gradients.

    Args:
        model: TransformerBattlePolicy. Will be set to model.train().
        optimizer: the same AdamW used by PPO.
        replay_buffer: AWRReplayBuffer (from awr_replay module).
        device: torch device.
        awr_batch_episodes: number of episodes to sample per AWR step.
        awr_mix_weight: scalar multiplier on AWR loss before backward.
            Per design memo Q2 calibration table, picked from 5-iter
            smoke data (Phase 2D).
        awr_binary / awr_beta / awr_clip_high: see compute_awr_loss.
        max_grad_norm: passed to clip_grad_norm_.
        detach_value: see compute_awr_loss.

    Returns:
        dict of Python floats (ready for TensorBoard logging):
            awr_loss              raw AWR loss value
            awr_loss_scaled       AWR loss × mix_weight (what backward sees)
            awr_advantage_mean    mean advantage over valid positions
            awr_advantage_pos_frac  frac of valid positions with A > 0
            awr_weight_max        max applied weight
            awr_weight_mean       mean applied weight
            awr_n_valid           positions used (post-NaN-mask)
            awr_n_total_valid     positions before NaN-mask
            awr_grad_norm         gradient norm post-AWR-backward (pre-clip)
            awr_mix_weight        the mix weight used (for logging)
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    batch = replay_buffer.sample(awr_batch_episodes, device)

    # bf16 autocast iff enabled (matches PPO update path; see ppo.py).
    from precision_config import get_amp_dtype, autocast_ctx
    import contextlib
    _amp_ctx = (autocast_ctx() if get_amp_dtype() is torch.bfloat16
                else contextlib.nullcontext())

    with _amp_ctx:
        awr_out = compute_awr_loss(
            model, batch, device,
            awr_binary=awr_binary,
            awr_beta=awr_beta,
            awr_clip_high=awr_clip_high,
            detach_value=detach_value,
        )
        loss_scaled = awr_mix_weight * awr_out["loss"]

    loss_scaled.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm
    )
    optimizer.step()

    return {
        "awr_loss": float(awr_out["loss"].item()),
        "awr_loss_scaled": float(loss_scaled.item()),
        "awr_advantage_mean": float(awr_out["advantage_mean"].item()),
        "awr_advantage_pos_frac": float(awr_out["advantage_pos_frac"].item()),
        "awr_weight_max": float(awr_out["weight_max"].item()),
        "awr_weight_mean": float(awr_out["weight_mean"].item()),
        "awr_n_valid": float(awr_out["n_valid"].item()),
        "awr_n_total_valid": float(awr_out["n_total_valid_pad"].item()),
        "awr_grad_norm": float(grad_norm.item()),
        "awr_mix_weight": float(awr_mix_weight),
    }


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
