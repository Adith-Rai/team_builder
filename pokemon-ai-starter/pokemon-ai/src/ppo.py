#!/usr/bin/env python
# ppo.py — PPO utilities: Trajectory, GAE, PPO update, checkpoint I/O.
#
# Used by train_rl.py and the RL collection modules.
# All training-loop-specific code lives in train_rl.py.
# All collection code lives in rl_collection.py / rl_pipeline.py.
# All player classes live in rl_player.py / battle_agent.py.
#
# Key functions:
#   Trajectory — per-episode storage for PPO (CPU batch dicts)
#   compute_gae — Generalized Advantage Estimation
#   build_ppo_episodes — convert trajectories to PPO episodes with global advantage norm
#   ppo_update — PPO update with distributional value loss + KL early stopping
#   load_checkpoint / save_checkpoint — checkpoint I/O with dim expansion support
#   _cancel_listener — cleanup for poke-env PSClient websocket

from __future__ import annotations

import contextlib
import gc
import os
import random
import traceback
from typing import Dict, List, Optional

# Used in ppo_update to conditionally apply torch.no_grad() during warmup
# (when only value_head trains, skipping autograd tape on frozen backbone).
_nullcontext = contextlib.nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arch_compat import (
    call_action_encoder,
    call_policy_logits,
    call_value_logits,
    get_v_support,
)
from model import PokeTransformer, PokeTransformerConfig
from precision_config import autocast_ctx, get_amp_dtype


# =============================
# Trajectory
# =============================

class Trajectory:
    """Per-episode storage for PPO. Stores CPU batch dicts to save GPU memory."""
    __slots__ = (
        "feat_batches", "actions", "log_probs", "values",
        "rewards", "dones", "action_masks",
    )

    def __init__(self):
        for slot in self.__slots__:
            setattr(self, slot, [])

    def __len__(self):
        return len(self.actions)


# =============================
# GAE + Episode Building
# =============================

def compute_gae(rewards, values, dones, gamma=0.9999, lam=0.8):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_val = 0.0 if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def build_ppo_episodes(trajectories: List[Trajectory],
                       gamma: float = 0.9999, lam: float = 0.8) -> List[dict]:
    episodes = []
    all_advs = []
    for traj in trajectories:
        if len(traj) == 0:
            continue
        adv, ret = compute_gae(traj.rewards, traj.values, traj.dones, gamma, lam)
        all_advs.append(adv)
        episodes.append({
            "feat_batches": traj.feat_batches,
            "actions": traj.actions,
            "old_logp": traj.log_probs,
            "advantages": adv,
            "returns": ret,
            "action_masks": [m.tolist() for m in traj.action_masks],
        })
    # Global advantage normalization
    if all_advs:
        all_flat = np.concatenate(all_advs)
        mean, std = all_flat.mean(), all_flat.std()
        if std > 1e-8:
            for ep in episodes:
                ep["advantages"] = ((ep["advantages"] - mean) / std).tolist()
        else:
            for ep in episodes:
                ep["advantages"] = ep["advantages"].tolist()
    return episodes


# =============================
# Tier 3 (Phase 4.7+, S55): Episode collation for sequence-batched PPO
# =============================
#
# C1 (this function): collate B episodes into padded (B, L_max, *) tensors
# with pad_mask. Foundational change for Tier 3 — does NOT alter the
# existing per-episode `ppo_update()` path. C2/C3/C4 will wire collated
# data into the new sequence-batched forward + masked loss.
#
# Design constraints (per docs/PHASE1_V3_OBSERVATIONS.md + boot doc):
#   - Reference shape from Metamon's metamon_to_amago.py:
#     (B episodes, L_max turns, ...) + pad_mask
#   - PPO is on-policy → no off-policy reweighting needed (vs Metamon V-trace)
#   - Causal masking lives in C2 (temporal attention); collate is just shape
#   - Per-episode storage (Trajectory, build_ppo_episodes) unchanged

def collate_episodes(episodes, L_max=None, device=None) -> dict:
    """Collate B episode dicts into padded (B, L_max, *) tensors + pad_mask.

    Args:
      episodes: list of episode dicts as returned by `build_ppo_episodes`.
        Each must contain: feat_batches (list of T per-turn feat dicts),
        actions/old_logp/advantages/returns (length-T sequences),
        action_masks (list of T arrays/tensors of shape (A,)).
      L_max: optional max sequence length. Defaults to longest episode in
        the bundle. Episodes longer than L_max are right-truncated.
      device: optional torch.device — if given, output tensors are moved
        there. If None, output stays on CPU (move at use site).

    Returns:
      dict with the following keys:
        feat_batches: dict — each leaf is (B, L_max, ...) padded tensor.
          Nested dicts are recursed; non-tensor leaves are stacked as-is.
        actions:      (B, L_max) long tensor — padded with 0
        old_logp:     (B, L_max) float tensor — padded with 0.0
        advantages:   (B, L_max) float tensor — padded with 0.0
        returns:      (B, L_max) float tensor — padded with 0.0
        action_masks: (B, L_max, A) float tensor — padded with 0.0
        pad_mask:     (B, L_max) bool tensor — True at valid positions,
                      False at padding. Multiply loss by pad_mask to zero
                      gradient at padding positions.
        seq_lens:     (B,) long tensor — actual T per episode (post-truncation)
        B:            int — batch size (number of episodes)
        L_max:        int — padded sequence length

    Memory notes:
      - At production scale (B=48 episodes, L_max=200 turns, ~few hundred
        feature dims): collated tensors are ~100-500 MB on GPU. Manageable.
      - For very large bundles, caller can pass L_max < max(seq_lens) to
        cap memory at the cost of right-truncating long episodes (rare).

    Acceptance gate (C1 unit test): reduce-sum equivalence on valid positions
      sum(collated[k] * pad_mask) == sum_per_episode(k) for all k.
    """
    import torch as _t

    if not episodes:
        raise ValueError("collate_episodes: empty episode list")

    # 1. Determine L_max + seq_lens
    seq_lens_list = [len(ep["actions"]) for ep in episodes]
    if L_max is None:
        L_max = max(seq_lens_list)
    # Right-truncate any episode longer than L_max (defensive)
    seq_lens_list = [min(s, L_max) for s in seq_lens_list]
    B = len(episodes)

    seq_lens = _t.tensor(seq_lens_list, dtype=_t.long)
    # pad_mask[b, t] = True iff t < seq_lens[b]
    arange_L = _t.arange(L_max).unsqueeze(0)            # (1, L_max)
    pad_mask = arange_L < seq_lens.unsqueeze(1)          # (B, L_max) bool

    # 2. Pad scalar-per-turn fields (actions, old_logp, advantages, returns)
    #    to (B, L_max). Each ep[k] is a length-T list/array.
    def _pad_1d(ep_list, T_actual, dtype, fill=0.0):
        """Convert a length-T list/array to a length-L_max tensor padded with `fill`."""
        x = _t.as_tensor(list(ep_list)[:T_actual], dtype=dtype)
        if T_actual < L_max:
            pad = _t.full((L_max - T_actual,), fill, dtype=dtype)
            x = _t.cat([x, pad], dim=0)
        return x

    actions = _t.stack([_pad_1d(ep["actions"], s, _t.long, fill=0)
                         for ep, s in zip(episodes, seq_lens_list)], dim=0)
    old_logp = _t.stack([_pad_1d(ep["old_logp"], s, _t.float32, fill=0.0)
                          for ep, s in zip(episodes, seq_lens_list)], dim=0)
    advantages = _t.stack([_pad_1d(ep["advantages"], s, _t.float32, fill=0.0)
                            for ep, s in zip(episodes, seq_lens_list)], dim=0)
    returns = _t.stack([_pad_1d(ep["returns"], s, _t.float32, fill=0.0)
                         for ep, s in zip(episodes, seq_lens_list)], dim=0)

    # 3. action_masks: list of T per-turn (A,) arrays/tensors → (B, L_max, A)
    #    Determine A from first non-empty episode.
    A = None
    for ep in episodes:
        if ep["action_masks"]:
            first_m = ep["action_masks"][0]
            A = (first_m.shape[0] if hasattr(first_m, "shape")
                 else len(first_m))
            break
    if A is None:
        raise ValueError("collate_episodes: no action_masks found")

    def _pad_2d(am_list, T_actual, A):
        """Stack T (A,) masks into (T, A), pad to (L_max, A) with zeros."""
        if T_actual == 0:
            stacked = _t.zeros(0, A, dtype=_t.float32)
        else:
            stacked = _t.stack([_t.as_tensor(m, dtype=_t.float32)
                                for m in am_list[:T_actual]], dim=0)
        if T_actual < L_max:
            pad = _t.zeros(L_max - T_actual, A, dtype=_t.float32)
            stacked = _t.cat([stacked, pad], dim=0)
        return stacked

    action_masks = _t.stack([_pad_2d(ep["action_masks"], s, A)
                              for ep, s in zip(episodes, seq_lens_list)], dim=0)

    # 4. feat_batches: per-episode list of T per-turn dicts. Each per-turn
    #    dict has tensor leaves of shape (1, ...) and possibly nested dict
    #    leaves of the same shape. Need to:
    #      (a) per-episode: stack T turn dicts → leaves (T, ...)
    #      (b) per-episode: pad to (L_max, ...) with zeros
    #      (c) across episodes: stack to (B, L_max, ...)
    #    Recurse for nested dicts. Non-tensor leaves return None (caller
    #    must handle; current production has no non-tensor leaves).
    def _stack_pad_one_episode(turn_dicts, T_actual):
        """Return a dict with leaves stacked + padded to (L_max, ...).
        Recurses for nested dicts."""
        if T_actual == 0:
            # Empty episode: synthesize zero-shaped leaves matching schema
            # of OTHER episodes. Caller should not normally pass empty episodes
            # (build_ppo_episodes filters them), but be defensive.
            raise ValueError("collate_episodes: T_actual==0 episode "
                             "(should be filtered upstream)")

        sample = turn_dicts[0]
        out = {}
        for k, v in sample.items():
            if isinstance(v, _t.Tensor):
                # Stack along dim 0 (each leaf is (1, ...) per turn) — squeeze
                # the leading-1 if present (use cat which preserves shape).
                # Cat gives (T, ...) — same as existing _stack_field at L168.
                stacked = _t.cat([turn_dicts[t][k]
                                  for t in range(T_actual)], dim=0)
                # Pad to (L_max, ...) with zeros
                if T_actual < L_max:
                    pad_shape = (L_max - T_actual,) + tuple(stacked.shape[1:])
                    pad = _t.zeros(pad_shape, dtype=stacked.dtype,
                                    device=stacked.device)
                    stacked = _t.cat([stacked, pad], dim=0)
                out[k] = stacked
            elif isinstance(v, dict):
                # Recurse: build a nested dict where each key's value is the
                # padded tensor across turns of the OUTER episode.
                inner_out = {}
                for inner_k, inner_v in v.items():
                    if isinstance(inner_v, _t.Tensor):
                        inner_stacked = _t.cat(
                            [turn_dicts[t][k][inner_k]
                             for t in range(T_actual)], dim=0)
                        if T_actual < L_max:
                            pad_shape = ((L_max - T_actual,)
                                         + tuple(inner_stacked.shape[1:]))
                            pad = _t.zeros(pad_shape, dtype=inner_stacked.dtype,
                                            device=inner_stacked.device)
                            inner_stacked = _t.cat([inner_stacked, pad], dim=0)
                        inner_out[inner_k] = inner_stacked
                    else:
                        # Non-tensor nested leaf: not supported in current
                        # production schema. Skip rather than crash.
                        pass
                out[k] = inner_out
            # else: non-tensor non-dict leaf — skip (not used in production)
        return out

    # Per-episode collated dicts, then stack across batch dim
    per_episode_collated = [
        _stack_pad_one_episode(ep["feat_batches"], s)
        for ep, s in zip(episodes, seq_lens_list)
    ]

    def _stack_batch_dim(per_ep_list):
        """Given list of per-episode dicts where each leaf is (L_max, ...),
        stack across batch to give dicts where each leaf is (B, L_max, ...).
        Recurses for nested dicts."""
        sample = per_ep_list[0]
        out = {}
        for k, v in sample.items():
            if isinstance(v, _t.Tensor):
                out[k] = _t.stack([d[k] for d in per_ep_list], dim=0)
            elif isinstance(v, dict):
                inner_out = {}
                for inner_k in v:
                    inner_out[inner_k] = _t.stack(
                        [d[k][inner_k] for d in per_ep_list], dim=0)
                out[k] = inner_out
        return out

    feat_batches = _stack_batch_dim(per_episode_collated)

    # 5. Optional device move
    if device is not None:
        def _to_device(d):
            r = {}
            for k, v in d.items():
                if isinstance(v, _t.Tensor):
                    r[k] = v.to(device, non_blocking=True)
                elif isinstance(v, dict):
                    r[k] = _to_device(v)
                else:
                    r[k] = v
            return r
        feat_batches = _to_device(feat_batches)
        actions = actions.to(device, non_blocking=True)
        old_logp = old_logp.to(device, non_blocking=True)
        advantages = advantages.to(device, non_blocking=True)
        returns = returns.to(device, non_blocking=True)
        action_masks = action_masks.to(device, non_blocking=True)
        pad_mask = pad_mask.to(device, non_blocking=True)
        seq_lens = seq_lens.to(device, non_blocking=True)

    return {
        "feat_batches": feat_batches,
        "actions": actions,
        "old_logp": old_logp,
        "advantages": advantages,
        "returns": returns,
        "action_masks": action_masks,
        "pad_mask": pad_mask,
        "seq_lens": seq_lens,
        "B": B,
        "L_max": L_max,
    }


# =============================
# Tier 3 C3 (Phase 4.7+, S55): masked PPO loss for sequence-batched update
# =============================
#
# C3: compute PPO loss components over a (B, L_max) batched forward output
# with pad_mask weighting. Replaces the per-episode loss math in
# `ppo_update`'s inner loop. Used by C4's switch-to-batched-update.
#
# Aggregation choice — IMPORTANT: per-transition mean (sum / pad_mask.sum())
# rather than per-episode mean (mean of per-episode means). This is the
# Metamon / standard-PPO aggregation and is what produces the 4-10× speedup
# in C4 (one optimizer step per WHOLE batch rather than one per episode).
#
# Equivalence with current per-episode loss path is EXACT only when B=1
# (single episode); for B>1 with variable T_i, batched per-transition mean
# weights longer episodes more (each transition equal), while current
# per-episode-mean weights each episode equally. The C3 unit test verifies
# B=1 equivalence; the multi-episode case is a deliberate semantic shift
# that ships in C4 + validated end-to-end in C6 vs Phase 1 v3 baseline.

def _ppo_loss_batched_internal(collated: dict, forward_out: dict, model, cfg,
                               ent_coef, vf_coef, clip_eps,
                               normalize_advantages: bool = False) -> dict:
    """Tensor-only output of the masked PPO loss. Tier 3 C5 split: this is
    the compile-friendly core (no .item() calls, no Python control flow).
    Both the eager wrapper `ppo_loss_batched` and the compiled train_step
    (via `make_compiled_train_step`) consume this directly.

    `ent_coef`, `vf_coef`, `clip_eps` accept either Python floats (eager
    path) or 0-dim tensors (compiled path — passing as tensors avoids
    recompile when adaptive-entropy moves ent_coef per iter).

    Returns dict with the same keys as `ppo_loss_batched` but `approx_kl`,
    `ratio_clip_frac`, `n_valid` are TENSORS (not Python scalars). Callers
    needing scalars must call `.item()` themselves.
    """
    pad_mask = collated["pad_mask"]
    actions = collated["actions"]
    old_logp = collated["old_logp"]
    advantages = collated["advantages"]
    returns = collated["returns"]

    logits_all = forward_out["action_logits"]
    vlogits_all = forward_out["v_logits"]

    device = logits_all.device
    pad_mask_f = pad_mask.to(device).float()
    n_valid = pad_mask_f.sum().clamp(min=1.0)
    actions = actions.to(device)
    old_logp = old_logp.to(device).float()
    advantages = advantages.to(device).float()
    returns = returns.to(device).float()

    if normalize_advantages:
        adv_mean = (advantages * pad_mask_f).sum() / n_valid
        adv_var = ((advantages - adv_mean).pow(2) * pad_mask_f).sum() / n_valid
        adv_std = adv_var.clamp(min=1e-8).sqrt()
        advantages = (advantages - adv_mean) / adv_std
        advantages = advantages * pad_mask_f

    lp = F.log_softmax(logits_all.float(), dim=-1)
    new_logp = lp.gather(2, actions.unsqueeze(-1)).squeeze(-1)
    ratio = torch.exp(new_logp - old_logp)

    with torch.no_grad():
        clipped_per_pos = ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float()
        ratio_clip_frac = (clipped_per_pos * pad_mask_f).sum() / n_valid

    s1 = ratio * advantages
    s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    pi_loss_per_pos = -torch.min(s1, s2)
    pi_loss = (pi_loss_per_pos * pad_mask_f).sum() / n_valid

    probs = F.softmax(logits_all.float(), dim=-1)
    entropy_per_pos = -(probs * lp).sum(-1)
    entropy = (entropy_per_pos * pad_mask_f).sum() / n_valid

    ret_c = returns.clamp(cfg.v_min, cfg.v_max)
    B_dim, L_max_dim = ret_c.shape
    vtgt_flat = model.twohot_target(ret_c.reshape(-1))
    vtgt = vtgt_flat.reshape(B_dim, L_max_dim, -1).float()
    v_loss_per_pos = -(vtgt * F.log_softmax(vlogits_all.float(), dim=-1)).sum(-1)
    v_loss = (v_loss_per_pos * pad_mask_f).sum() / n_valid

    with torch.no_grad():
        kl_per_pos = (old_logp - new_logp)
        approx_kl = (kl_per_pos * pad_mask_f).sum() / n_valid

    total_loss = pi_loss - ent_coef * entropy + vf_coef * v_loss

    return {
        "total_loss":      total_loss,
        "pi_loss":         pi_loss,
        "entropy":         entropy,
        "v_loss":          v_loss,
        "approx_kl":       approx_kl,        # TENSOR scalar
        "ratio_clip_frac": ratio_clip_frac,  # TENSOR scalar
        "n_valid":         n_valid,          # TENSOR scalar
    }


def ppo_loss_batched(collated: dict, forward_out: dict, model, cfg,
                     ent_coef: float = 0.02, vf_coef: float = 0.5,
                     clip_eps: float = 0.2,
                     normalize_advantages: bool = False) -> dict:
    """Compute PPO loss components over a sequence-batched forward output
    with pad_mask weighting.

    Args:
      collated: output of `ppo.collate_episodes()`. Reads:
        actions       (B, L_max) long
        old_logp      (B, L_max) float
        advantages    (B, L_max) float
        returns       (B, L_max) float
        pad_mask      (B, L_max) bool — True at valid positions
      forward_out: output of `model.forward_ppo_sequence()`. Reads:
        action_logits (B, L_max, n_actions) — -100.0 at padding
        v_logits      (B, L_max, v_bins)    — 0.0 at padding
        (value not used; we recompute from v_logits via twohot_target inverse
        only if needed for diagnostics — not used in loss)
      model: TransformerBattlePolicy — needed for `model.twohot_target()`
        (distributional value targets)
      cfg: model config — reads cfg.v_min, cfg.v_max for return clamping
      ent_coef, vf_coef, clip_eps: PPO hyperparameters
      normalize_advantages: if True, advantages are normalized in-place over
        valid positions (zero-mean, unit-std). If False, assume caller
        already normalized in build_ppo_episodes (current production path).

    Returns:
      dict with:
        total_loss:      scalar tensor — pi - ent_coef*ent + vf_coef*v
        pi_loss:         scalar tensor — policy clip loss
        entropy:         scalar tensor — mean entropy across valid positions
        v_loss:          scalar tensor — distributional value CE
        approx_kl:       scalar (Python float) — old_logp vs new_logp diff
        ratio_clip_frac: scalar (Python float) — fraction of ratios outside [1-clip, 1+clip]
        n_valid:         int — number of valid (b, t) positions in batch

    Aggregation: ALL losses use per-transition mean over valid positions:
      loss = (per_pos_loss * pad_mask).sum() / pad_mask.sum().clamp(min=1)

    Implementation note (Tier 3 C5): thin wrapper over
    `_ppo_loss_batched_internal` (the tensor-only compile-friendly core).
    Existing callers preserve their scalar-returning contract; the
    compiled train_step path uses the internal function directly.
    """
    out = _ppo_loss_batched_internal(
        collated, forward_out, model, cfg,
        ent_coef=ent_coef, vf_coef=vf_coef, clip_eps=clip_eps,
        normalize_advantages=normalize_advantages,
    )
    # Match the prior contract: approx_kl, ratio_clip_frac, n_valid as scalars.
    out["approx_kl"]       = out["approx_kl"].item()
    out["ratio_clip_frac"] = out["ratio_clip_frac"].item()
    out["n_valid"]         = int(out["n_valid"].item())
    return out


# =============================
# PPO Update (batched spatial, sequential temporal)
# =============================

def ppo_update(model: PokeTransformer, optimizer, episodes: List[dict],
                  device: torch.device, cfg: PokeTransformerConfig,
                  epochs: int = 3, clip_eps: float = 0.2, ent_coef: float = 0.02,
                  vf_coef: float = 0.5, max_grad_norm: float = 0.5,
                  target_kl: float = 0.02, grad_accum: int = 1,
                  in_warmup: bool = False,
                  ) -> dict:
    """PPO update with distributional value loss + KL early stopping (ps-ppo style).
    No KL penalty term — instead stops PPO epochs early if policy changes too much.
    grad_accum: accumulate gradients over N episodes before each optimizer step."""
    model.train()
    stats = {"pi": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0,
             # Diagnostic counters — catch silent policy/value pathologies early
             # (Exp 4-style collapse: value drifts while policy keeps training).
             "ratio_clip_frac": 0.0,   # fraction of ratios outside [1-clip, 1+clip]
             "value_mean": 0.0,        # mean predicted value (drift indicator)
             "return_mean": 0.0,       # mean return target (compare to value_mean)
             "adv_abs_mean": 0.0,      # advantage magnitude (normalization sanity)
             }
    n = 0
    n_failed = 0  # episodes that raised an exception (CUDA error, OOM, etc.)
    n_skipped_kl = 0       # episodes gated by per-episode KL check (silent otherwise)
    n_skipped_nan = 0      # episodes skipped due to NaN in advantages/returns/forward
    kl_early_stopped = False

    for ppo_ep in range(epochs):
        if kl_early_stopped:
            break
        random.shuffle(episodes)
        epoch_kl_sum = 0.0
        epoch_kl_count = 0
        accum_count = 0
        for ep in episodes:
            T = len(ep["actions"])
            if T == 0:
                continue

            try:
                actions = torch.tensor(ep["actions"], dtype=torch.long, device=device)
                old_logp = torch.tensor(ep["old_logp"], dtype=torch.float32, device=device)
                advantages = torch.tensor(ep["advantages"], dtype=torch.float32, device=device)
                returns = torch.tensor(ep["returns"], dtype=torch.float32, device=device)
                masks = torch.tensor(ep["action_masks"], dtype=torch.float32, device=device)

                # Defensive: NaN in advantages/returns indicates upstream GAE bug
                # or corrupt trajectory. Skip loudly rather than silently propagate
                # NaN into the optimizer.
                if (torch.isnan(advantages).any() or torch.isinf(advantages).any()
                        or torch.isnan(returns).any() or torch.isinf(returns).any()):
                    print(f"  [WARN] NaN/Inf in advantages/returns (T={T}), skipping episode", flush=True)
                    n_skipped_nan += 1
                    continue

                # --- Batch all T turns' spatial processing at once ---
                def _stack_field(key):
                    vals = [ep["feat_batches"][t][key] for t in range(T)]
                    if isinstance(vals[0], torch.Tensor):
                        return torch.cat(vals, dim=0).to(device)
                    elif isinstance(vals[0], dict):
                        return {k: torch.cat([v[k] for v in vals], dim=0).to(device)
                                for k in vals[0]}
                    return vals

                mega = {k: _stack_field(k) for k in ep["feat_batches"][0].keys()}
                # Warmup optimization: skip autograd through frozen backbone +
                # policy. Backward stops at value_head's input (the only thing
                # being trained). Saves ~50% on update wall-time during warmup.
                _backbone_ctx = torch.no_grad() if in_warmup else _nullcontext()
                # Autocast on update path: bf16 only. fp16 backward without a
                # GradScaler underflows on small gradients (we don't use a scaler
                # by precision_config design); bf16 has fp32 dynamic range so
                # backward is stable without scaling. fp32 path stays unchanged.
                _update_amp_ctx = (autocast_ctx()
                                   if get_amp_dtype() is torch.bfloat16
                                   else _nullcontext())
                with _update_amp_ctx:
                    with _backbone_ctx:
                        spatial_out, all_summaries = model.forward_spatial(mega)
                        action_ctx = call_action_encoder(model, mega, spatial_out)
                    legal_all = mega["legal_mask"]

                    # --- Sequential temporal + heads ---
                    all_logits = []
                    all_vlogits = []
                    # Summary buffer dim = resolved d_temporal (falls back to d_model for legacy configs)
                    _d_sum = cfg.d_temporal if cfg.d_temporal is not None else cfg.d_model
                    summary_buf = torch.zeros(1, 0, _d_sum, device=device)

                    for t in range(T):
                        s = all_summaries[t:t+1].unsqueeze(0)
                        summary_buf = torch.cat([summary_buf, s], dim=1)
                        if summary_buf.shape[1] > 200:
                            summary_buf = summary_buf[:, -200:]

                        # Backbone + policy path: in warmup, no_grad to skip autograd
                        # tape (policy is frozen; logits used only for KL/entropy
                        # stats, not for gradient).
                        with _backbone_ctx:
                            temporal_ctx = model.temporal(summary_buf)
                            actor_out = spatial_out[t, 0, :]
                            critic_out = spatial_out[t, 1, :]
                            act_ctx = action_ctx[t]

                            at = torch.cat([actor_out, temporal_ctx.squeeze(0)], dim=-1)
                            at_exp = at.unsqueeze(0).expand(9, -1)
                            pi_input = torch.cat([at_exp, act_ctx], dim=-1)
                            logits = call_policy_logits(model, pi_input)
                            logits = logits.masked_fill(legal_all[t] < 0.5, -100.0)
                        all_logits.append(logits)

                        # Value path: gradient flows through value_head only (its
                        # input critic_out + temporal_ctx are detached if in warmup).
                        vi = torch.cat([critic_out, temporal_ctx.squeeze(0)], dim=-1)
                        vl = call_value_logits(model, vi.unsqueeze(0)).squeeze(0)
                        all_vlogits.append(vl)

                    logits_seq = torch.stack(all_logits)
                    vlogits_seq = torch.stack(all_vlogits)

                    # NaN check
                    if logits_seq.isnan().any() or vlogits_seq.isnan().any():
                        print(f"  [WARN] NaN in forward, skip (T={T})", flush=True)
                        n_skipped_nan += 1
                        continue

                    # Policy loss
                    lp = F.log_softmax(logits_seq, dim=-1)
                    new_logp = lp.gather(1, actions.unsqueeze(1)).squeeze(1)
                    ratio = torch.exp(new_logp - old_logp)
                    # Track ratio-clip fraction — if consistently high, policy is drifting
                    # too fast per update (Exp 4-style instability signal).
                    with torch.no_grad():
                        clipped_frac = ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float().mean().item()
                    s1 = ratio * advantages
                    s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
                    pi_loss = -torch.min(s1, s2).mean()

                    # Entropy
                    probs = F.softmax(logits_seq, dim=-1)
                    entropy = -(probs * lp).sum(-1).mean()

                    # Value loss (distributional two-hot CE with per-step clamping)
                    ret_c = returns.clamp(cfg.v_min, cfg.v_max)
                    vtgt = model.twohot_target(ret_c)
                    v_loss_per_step = -(vtgt * F.log_softmax(vlogits_seq, dim=-1)).sum(-1)
                    v_loss = v_loss_per_step.mean()

                    # Approximate KL — check BEFORE applying gradient (ps-ppo style)
                    with torch.no_grad():
                        approx_kl = (old_logp - lp.gather(1, actions.unsqueeze(1)).squeeze(1)).mean().item()

                    # Per-episode KL gate: skip this episode entirely if policy diverged too much
                    if abs(approx_kl) > target_kl * 5:
                        n_skipped_kl += 1
                        continue  # no backward, no step — episode discarded

                    # Loss: no KL penalty term — early stopping replaces it
                    loss = (pi_loss - ent_coef * entropy + vf_coef * v_loss) / grad_accum

                if loss.isnan() or loss.isinf():
                    print(f"  [WARN] NaN/inf loss (pi={pi_loss.item():.4f} v={v_loss.item():.4f}), T={T}, aborting PPO update", flush=True)
                    optimizer.zero_grad()
                    kl_early_stopped = True
                    break

                # Gradient accumulation: accumulate N episodes before stepping
                if accum_count == 0:
                    optimizer.zero_grad()
                loss.backward()
                accum_count += 1

                if accum_count >= grad_accum:
                    if max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    accum_count = 0

                stats["pi"] += pi_loss.item()
                stats["v"] += v_loss.item()
                stats["ent"] += entropy.item()
                stats["kl"] += abs(approx_kl)
                stats["ratio_clip_frac"] += clipped_frac
                # Drift diagnostics: value_mean vs return_mean — if they diverge,
                # critic is learning the wrong scale (Exp 4 post-mortem symptom).
                with torch.no_grad():
                    v_probs_step = F.softmax(vlogits_seq, dim=-1)
                    v_pred_step = (v_probs_step * get_v_support(model)).sum(-1)
                    stats["value_mean"] += v_pred_step.mean().item()
                    stats["return_mean"] += returns.mean().item()
                    stats["adv_abs_mean"] += advantages.abs().mean().item()
                epoch_kl_sum += abs(approx_kl)
                epoch_kl_count += 1
                n += 1

            except Exception as e:
                print(f"  [ERROR] PPO episode failed (T={T}): {e}", flush=True)
                traceback.print_exc()
                n_failed += 1
                # Reset gradient state to prevent stale/partial gradients
                optimizer.zero_grad()
                accum_count = 0
                continue

            # Free GPU memory periodically
            if n % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # Flush remaining accumulated gradients
        if accum_count > 0:
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            accum_count = 0

        # KL early stopping: if avg KL this epoch exceeds threshold, stop
        if epoch_kl_count > 0:
            avg_epoch_kl = epoch_kl_sum / epoch_kl_count
            if avg_epoch_kl > target_kl * 1.5:
                print(f"    KL early stop: epoch {ppo_ep}, avg_kl={avg_epoch_kl:.4f} > {target_kl*1.5:.4f}", flush=True)
                kl_early_stopped = True

    for k in stats:
        stats[k] /= max(1, n)
    # Bookkeeping (added after the divide so they're not normalized)
    stats["n_succeeded"] = n
    stats["n_failed"] = n_failed
    stats["n_skipped_kl"] = n_skipped_kl
    stats["n_skipped_nan"] = n_skipped_nan
    # Surface silent discards if substantial — these episodes produce no gradient
    # but still consume a training slot. Useful for diagnosing low-effective-batch regimes.
    total_eps = n + n_failed + n_skipped_kl + n_skipped_nan
    if total_eps > 0 and (n_skipped_kl + n_skipped_nan) >= max(3, total_eps // 10):
        print(f"  [NOTICE] PPO discarded {n_skipped_kl} KL + {n_skipped_nan} NaN "
              f"episodes out of {total_eps} ({100*(n_skipped_kl+n_skipped_nan)/total_eps:.1f}%)",
              flush=True)
    # Surface value/return drift — if |value_mean - return_mean| grows, critic is wrong.
    vm, rm = stats["value_mean"], stats["return_mean"]
    if abs(vm - rm) > 0.3:
        print(f"  [NOTICE] Value drift: value_mean={vm:.3f} vs return_mean={rm:.3f} "
              f"(gap={abs(vm-rm):.3f}). Critic may be miscalibrated.", flush=True)
    return stats


# =============================
# Tier 3 C5 (Phase 4.7+, S56): single-graph compiled train_step
# =============================

def make_compiled_train_step(model, optimizer, cfg, vf_coef: float = 0.5,
                              max_grad_norm: float = 0.5,
                              normalize_advantages: bool = False):
    """Build the Tier 3 C5 compiled train_step.

    Returns an eager-wrapper callable that orchestrates:
      1. eager:    optimizer.zero_grad()           — clears .grad
      2. COMPILED: forward + loss + masked-backward — single fused graph via
                   torch.compile + aot_autograd; fuses forward AND backward
                   kernels across the boundary (the big speedup)
      3. eager:    clip_grad_norm_ + optimizer.step — torch 2.2.x dynamo
                   can't trace `model.parameters()` iterator through the
                   `isinstance` check inside clip_grad (verified failure
                   on dev pod); fused AdamW step is already kernel-fused
                   internally so eager is fine. Cost: ~1-2% of step time.

    The user-facing constraint ("full train_step in one graph") is satisfied
    in spirit by including backward in the compiled region — that's where
    the aot_autograd fusion lives. The optimizer.step boundary split is
    forced by torch 2.2.x dynamo limitations (would need torch 2.4+ to
    move further), NOT by conservative engineering.

    Per S55 wrap design intent ("surgical compile region around forward+
    loss; control flow stays uncompiled"), the control flow IS surgical:
    safety gates (NaN check, KL gate) are applied via in-graph TENSOR
    MASKS, not Python branches. When a gate trips, the masked loss is 0
    → backward computes zero gradients → eager optimizer.step is a no-op
    in expectation (AdamW momentum decays slightly toward zero, harmless
    for skipped steps). The decision to skip is made entirely from a
    tensor compare with no host sync, so dynamo doesn't graph-break.

    Compile mode: "default" + dynamic=True. Matches the S51 per-submodule
    pattern (avoids cudagraph aliasing pitfalls + recompile churn on
    variable L_max / B). The S51 per-submodule wrappers (tokenizer, spatial,
    temporal, action_head, value_head) are transparently traced through by
    dynamo when compiling the outer fwd_bwd — net result is one fused
    graph spanning forward + loss + backward.

    Args fixed at compile time (closure-captured constants):
        model, optimizer, cfg, vf_coef, max_grad_norm, normalize_advantages

    Args passed at call time as 0-dim tensors (avoid recompile when values
    change; ent_coef in particular moves per-iter under adaptive-entropy):
        ent_coef_t, clip_eps_t, target_kl_t

    Returns: eager callable
        (collated, ent_coef_t, clip_eps_t, target_kl_t) -> dict
    All return values are TENSORS — caller (ppo_update_batched compiled
    path) calls .item() outside for stats + KL early-stop check.

    Returned dict keys:
        total_loss, pi_loss, entropy, v_loss, approx_kl, ratio_clip_frac,
        value_mean, return_mean, adv_abs_mean,
        step_mask  — 1.0 if backward had nonzero grad, 0.0 if skipped
        nan_safe   — 1.0 if forward+loss finite, 0.0 if NaN/inf
        kl_safe    — 1.0 if |approx_kl| <= target_kl × 5, 0.0 if gate fired
    """
    def fwd_bwd(collated, ent_coef_t, clip_eps_t, target_kl_t):
        """Compiled inner: forward + loss + masked-backward. Compile boundary."""
        device = collated["actions"].device

        # Forward (Tier 3 sequence-batched) + loss (tensor-only)
        forward_out = model.forward_ppo_sequence(collated, device)
        loss_dict = _ppo_loss_batched_internal(
            collated, forward_out, model, cfg,
            ent_coef=ent_coef_t, vf_coef=vf_coef, clip_eps=clip_eps_t,
            normalize_advantages=normalize_advantages,
        )

        total_loss = loss_dict["total_loss"]
        approx_kl_t = loss_dict["approx_kl"]

        # Safety masks — in-graph tensor compares (no host sync, no graph break)
        nan_safe = torch.isfinite(total_loss).float()
        kl_safe = (approx_kl_t.abs() <= target_kl_t * 5.0).float()
        step_mask = nan_safe * kl_safe
        # nan_to_num: even if step_mask is 0, NaN × 0 = NaN. Replace with 0
        # so backward gives clean zero gradients on the skip path.
        loss_safe = torch.nan_to_num(total_loss * step_mask,
                                      nan=0.0, posinf=0.0, neginf=0.0)
        # Backward INSIDE compile so aot_autograd captures fwd+bwd graphs together
        loss_safe.backward()

        # Drift diagnostics — in graph, eager .item() at use site
        with torch.no_grad():
            pad_mask_f = collated["pad_mask"].to(device).float()
            n_valid_t = pad_mask_f.sum().clamp(min=1.0)
            returns_t = collated["returns"].to(device).float()
            advantages_t = collated["advantages"].to(device).float()
            v_probs = F.softmax(forward_out["v_logits"].float(), dim=-1)
            v_pred = (v_probs * get_v_support(model)).sum(-1)
            value_mean_t = (v_pred * pad_mask_f).sum() / n_valid_t
            return_mean_t = (returns_t * pad_mask_f).sum() / n_valid_t
            adv_abs_mean_t = (advantages_t.abs() * pad_mask_f).sum() / n_valid_t

        return {
            "total_loss":      total_loss,
            "pi_loss":         loss_dict["pi_loss"],
            "entropy":         loss_dict["entropy"],
            "v_loss":          loss_dict["v_loss"],
            "approx_kl":       approx_kl_t,
            "ratio_clip_frac": loss_dict["ratio_clip_frac"],
            "value_mean":      value_mean_t,
            "return_mean":     return_mean_t,
            "adv_abs_mean":    adv_abs_mean_t,
            "step_mask":       step_mask,
            "nan_safe":        nan_safe,
            "kl_safe":         kl_safe,
        }

    # dynamic=True so variable B/L_max don't trigger per-shape recompile.
    # mode="default" matches S51 production choice (avoids cudagraph aliasing).
    compiled_fwd_bwd = torch.compile(fwd_bwd, mode="default", dynamic=True)

    def train_step(collated, ent_coef_t, clip_eps_t, target_kl_t):
        """Eager wrapper: zero_grad → COMPILED fwd+loss+bwd → (gated) clip+step.

        The optimizer.step is ELIDED when step_mask is 0 (safety gate fired)
        to match the eager `ppo_update_batched` semantic — `continue`-equivalent.
        Without this gate, AdamW's weight_decay (default 0.01) would cause
        small parameter drift even on zero gradients, since the AdamW update
        is `param ← param - lr * (m_hat / (sqrt(v_hat) + eps) + wd * param)`.
        Costs one .item() host sync per step (~10 μs, negligible).
        """
        optimizer.zero_grad()
        out = compiled_fwd_bwd(collated, ent_coef_t, clip_eps_t, target_kl_t)
        if out["step_mask"].item() > 0.5:
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        return out

    return train_step


# =============================
# Tier 3 C4 (Phase 4.7+, S55): sequence-batched PPO update
# =============================

def ppo_update_batched(model, optimizer, episodes, device, cfg,
                       epochs: int = 3, clip_eps: float = 0.2,
                       ent_coef: float = 0.02, vf_coef: float = 0.5,
                       max_grad_norm: float = 0.5, target_kl: float = 0.02,
                       L_max: Optional[int] = None,
                       normalize_advantages: bool = False,
                       in_warmup: bool = False,
                       compiled_step=None) -> dict:
    """Sequence-batched PPO update — Tier 3's payoff. Composes C1/C2/C3:
        collate_episodes (C1) → forward_ppo_sequence (C2) → ppo_loss_batched (C3)
    → backward → optimizer.step().

    Replaces the per-episode loop in `ppo_update` with a SINGLE
    forward+loss+backward+step per epoch over the WHOLE batch. Where the
    4-10× update-phase speedup lives (per-iter optimizer.step calls drop
    from B*epochs to epochs).

    Drop-in replacement for `ppo_update`: same arguments, same returned
    stats dict shape, same KL early-stop semantics. Caller (train_rl.py)
    selects this path via a flag (added in C5 wiring).

    Aggregation choice (intentional, see C3 docstring): per-transition
    mean over valid positions. Differs from current per-episode mean for
    multi-episode batches with variable T. Larger effective batch per
    gradient step → enables higher lr safely (re-ablate after Tier 3).

    NOT supported in v1 (will add in subsequent commit if needed):
      - in_warmup=True (per-step value-only training): currently raises
        NotImplementedError. Warmup is 5 iters; production launches with
        --warmup-iters 5 or 10 and then proceeds normally. For Tier 3
        warmup support, callers can use the existing per-episode
        ppo_update for warmup iters, then switch to ppo_update_batched.
      - Minibatching within an epoch: currently 1 batch per epoch (one
        gradient step per epoch). For very large batches (>2000
        transitions) consider splitting into 2-4 minibatches per epoch.
        At our production scale (B≈48, L_avg≈30 → ~1500 transitions),
        single batch is fine.

    Args:
      model, optimizer, episodes, device, cfg: same as ppo_update
      epochs, clip_eps, ent_coef, vf_coef, max_grad_norm, target_kl: same
      L_max: optional cap on episode length passed to collate_episodes
      normalize_advantages: passed through to ppo_loss_batched
      in_warmup: rejected with NotImplementedError in v1 (see above)
      compiled_step: optional callable from `make_compiled_train_step`. When
        provided, dispatches to the C5 single-graph compiled train_step
        (forward+loss+backward+clip+optimizer.step in one fused graph).
        Safety gates (NaN, KL) are tensor-mask based inside the graph;
        n_skipped_nan / n_skipped_kl counters classified eager-side from
        returned mask tensors. Caller is responsible for verifying compile-
        time invariants match (vf_coef, max_grad_norm, normalize_advantages,
        cfg) — these are closure-captured constants in the compiled graph.

    Returns:
      stats dict with same keys as ppo_update:
        pi, v, ent, kl, ratio_clip_frac, value_mean, return_mean, adv_abs_mean
        n_succeeded, n_failed, n_skipped_kl, n_skipped_nan
      Stats are normalized over the number of EPOCHS that ran (not
      episodes), since batched path runs 1 step per epoch.
    """
    if in_warmup:
        raise NotImplementedError(
            "ppo_update_batched does not support in_warmup=True yet. "
            "Use the per-episode ppo_update for warmup iters, then switch "
            "to ppo_update_batched for the main training loop."
        )
    if not episodes:
        # No episodes — nothing to do. Return zero-stats.
        return {"pi": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0,
                "ratio_clip_frac": 0.0, "value_mean": 0.0,
                "return_mean": 0.0, "adv_abs_mean": 0.0,
                "n_succeeded": 0, "n_failed": 0,
                "n_skipped_kl": 0, "n_skipped_nan": 0}

    model.train()
    stats = {"pi": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0,
             "ratio_clip_frac": 0.0,
             "value_mean": 0.0, "return_mean": 0.0, "adv_abs_mean": 0.0}
    n = 0                # number of epochs that ran without skip
    n_failed = 0         # epochs that raised an exception
    n_skipped_kl = 0     # epochs gated by per-batch KL check (target_kl × 5)
    n_skipped_nan = 0    # epochs skipped due to NaN/inf in loss
    kl_early_stopped = False

    # bf16 autocast on update (same gating as ppo_update line 226-228):
    # autocast_ctx() only when amp_dtype is bf16; fp16 backward without
    # GradScaler underflows. fp32 path stays unchanged.
    _update_amp_ctx = (autocast_ctx()
                       if get_amp_dtype() is torch.bfloat16
                       else _nullcontext())

    for ppo_ep in range(epochs):
        if kl_early_stopped:
            break

        # Shuffle episode order each epoch — same intent as ppo_update's
        # random.shuffle(episodes), now applied before collation.
        random.shuffle(episodes)

        if compiled_step is not None:
            # ---- Tier 3 C5 compiled path: single-graph train_step ----
            try:
                collated = collate_episodes(episodes, L_max=L_max, device=device)
                # 0-dim tensors — value-change does NOT trigger recompile.
                # Adaptive entropy moves ent_coef per iter; passing as tensor
                # keeps the compile cache hot.
                ent_coef_t = torch.tensor(ent_coef, device=device, dtype=torch.float32)
                clip_eps_t = torch.tensor(clip_eps, device=device, dtype=torch.float32)
                target_kl_t = torch.tensor(target_kl, device=device, dtype=torch.float32)

                with _update_amp_ctx:
                    out = compiled_step(collated, ent_coef_t, clip_eps_t, target_kl_t)

                # All values are tensors. .item() AFTER the compiled call
                # — the host sync here is OUTSIDE the graph, no graph break.
                nan_safe_v = out["nan_safe"].item()
                kl_safe_v = out["kl_safe"].item()
                approx_kl_v = out["approx_kl"].item()

                # Classify the skip reason from the in-graph masks
                if nan_safe_v < 0.5:
                    print(f"  [WARN] NaN/inf in compiled train_step "
                          f"(epoch {ppo_ep}), step skipped via mask",
                          flush=True)
                    n_skipped_nan += 1
                    continue
                if kl_safe_v < 0.5:
                    n_skipped_kl += 1
                    continue

                # Stats (one optimizer.step actually landed)
                stats["pi"] += out["pi_loss"].item()
                stats["v"] += out["v_loss"].item()
                stats["ent"] += out["entropy"].item()
                stats["kl"] += abs(approx_kl_v)
                stats["ratio_clip_frac"] += out["ratio_clip_frac"].item()
                stats["value_mean"] += out["value_mean"].item()
                stats["return_mean"] += out["return_mean"].item()
                stats["adv_abs_mean"] += out["adv_abs_mean"].item()
                n += 1

                # KL early-stop check (same threshold + semantics as eager path)
                if abs(approx_kl_v) > target_kl * 1.5:
                    print(f"    KL early stop (batched-compiled): epoch "
                          f"{ppo_ep}, kl={abs(approx_kl_v):.4f} > "
                          f"{target_kl*1.5:.4f}", flush=True)
                    kl_early_stopped = True

            except Exception as e:
                print(f"  [ERROR] Compiled batched PPO epoch {ppo_ep} "
                      f"failed: {e}", flush=True)
                traceback.print_exc()
                n_failed += 1
                optimizer.zero_grad()
                continue

            del collated, out
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue  # next ppo_ep

        # ---- Eager path (compiled_step is None) ----
        try:
            # Collate B episodes → (B, L_max, *) padded batch on device
            collated = collate_episodes(episodes, L_max=L_max, device=device)

            with _update_amp_ctx:
                # Forward: collated → (B, L_max, n_actions) logits etc.
                forward_out = model.forward_ppo_sequence(collated, device)

                # NaN/inf check on forward
                if (forward_out["action_logits"].isnan().any()
                        or forward_out["v_logits"].isnan().any()):
                    print(f"  [WARN] NaN in batched forward, skip epoch "
                          f"{ppo_ep}", flush=True)
                    n_skipped_nan += 1
                    continue

                # Loss: ppo_loss_batched (C3) — pi + entropy + value + kl
                loss_dict = ppo_loss_batched(
                    collated, forward_out, model, cfg,
                    ent_coef=ent_coef, vf_coef=vf_coef,
                    clip_eps=clip_eps,
                    normalize_advantages=normalize_advantages,
                )
                total_loss = loss_dict["total_loss"]
                approx_kl = loss_dict["approx_kl"]

            # NaN/inf check on loss
            if total_loss.isnan() or total_loss.isinf():
                print(f"  [WARN] NaN/inf loss (pi={loss_dict['pi_loss'].item():.4f} "
                      f"v={loss_dict['v_loss'].item():.4f}), aborting batched "
                      f"PPO update", flush=True)
                optimizer.zero_grad()
                kl_early_stopped = True
                break

            # Per-batch KL gate: skip this epoch's update if policy diverged
            # too much. Coarser than the per-episode gate in ppo_update
            # (which skips just outlier episodes), but the effect is
            # similar at the optimizer-step level (no harmful update lands).
            if abs(approx_kl) > target_kl * 5:
                n_skipped_kl += 1
                # Don't break — we might recover next epoch on shuffled batch
                continue

            # Backward + clip + step
            optimizer.zero_grad()
            total_loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Accumulate stats (per-epoch since batched path is 1 step / epoch)
            stats["pi"] += loss_dict["pi_loss"].item()
            stats["v"] += loss_dict["v_loss"].item()
            stats["ent"] += loss_dict["entropy"].item()
            stats["kl"] += abs(approx_kl)
            stats["ratio_clip_frac"] += loss_dict["ratio_clip_frac"]

            # Drift diagnostics: value vs return (same as ppo_update lines
            # 532-539). Computed over valid positions of the collated batch.
            with torch.no_grad():
                pad_mask_f = collated["pad_mask"].to(device).float()
                n_valid = pad_mask_f.sum().clamp(min=1.0)
                returns_t = collated["returns"].to(device).float()
                advantages_t = collated["advantages"].to(device).float()
                v_probs = F.softmax(forward_out["v_logits"].float(), dim=-1)
                v_pred = (v_probs * get_v_support(model)).sum(-1)  # (B, L_max)
                stats["value_mean"]  += ((v_pred * pad_mask_f).sum() / n_valid).item()
                stats["return_mean"] += ((returns_t * pad_mask_f).sum() / n_valid).item()
                stats["adv_abs_mean"]+= ((advantages_t.abs() * pad_mask_f).sum() / n_valid).item()

            n += 1

            # KL early-stop: avg-KL > target_kl × 1.5 → break out of epoch loop
            # (same threshold + intent as ppo_update lines 700-706, computed
            # over per-batch KL since each epoch IS one batch update).
            if abs(approx_kl) > target_kl * 1.5:
                print(f"    KL early stop (batched): epoch {ppo_ep}, "
                      f"kl={abs(approx_kl):.4f} > {target_kl*1.5:.4f}",
                      flush=True)
                kl_early_stopped = True

        except Exception as e:
            print(f"  [ERROR] Batched PPO epoch {ppo_ep} failed: {e}", flush=True)
            traceback.print_exc()
            n_failed += 1
            optimizer.zero_grad()
            continue

        # Memory cleanup between epochs (collated batch can be 100-500 MB)
        del collated, forward_out, loss_dict, total_loss
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Normalize stats by number of completed epochs
    for k in stats:
        stats[k] /= max(1, n)

    stats["n_succeeded"] = n
    stats["n_failed"] = n_failed
    stats["n_skipped_kl"] = n_skipped_kl
    stats["n_skipped_nan"] = n_skipped_nan

    # Surface silent discards (same heuristic as ppo_update line 717-721)
    total_epochs = n + n_failed + n_skipped_kl + n_skipped_nan
    if total_epochs > 0 and (n_skipped_kl + n_skipped_nan) >= max(2, total_epochs // 3):
        print(f"  [NOTICE] PPO-batched discarded {n_skipped_kl} KL + "
              f"{n_skipped_nan} NaN epochs out of {total_epochs} "
              f"({100*(n_skipped_kl+n_skipped_nan)/total_epochs:.1f}%)",
              flush=True)
    # Surface value/return drift (same heuristic as ppo_update line 723-726)
    vm, rm = stats["value_mean"], stats["return_mean"]
    if abs(vm - rm) > 0.3:
        print(f"  [NOTICE] Value drift (batched): value_mean={vm:.3f} vs "
              f"return_mean={rm:.3f} (gap={abs(vm-rm):.3f}). Critic may be "
              f"miscalibrated.", flush=True)
    return stats


# =============================
# Utility
# =============================

def _cancel_listener(player):
    """Cancel PSClient websocket listener to prevent zombie coroutines."""
    try:
        ps = getattr(player, "ps_client", None) or getattr(player, "_ps_client", None)
        if ps and hasattr(ps, "_listening_coroutine"):
            ps._listening_coroutine.cancel()
    except Exception:
        pass


# =============================
# Checkpoint I/O
# =============================

def _infer_arch_from_state_dict(state: dict) -> str:
    """Pick 'transformer' vs 'mlp' from state-dict key prefixes. Used when a
    legacy checkpoint (no `arch` field) is loaded — defaults to 'mlp'."""
    # Both arches use spatial.*/temporal.* prefixes (legacy PokeTransformer
    # also instantiates SpatialTransformer/TemporalTransformer as `self.spatial`
    # and `self.temporal` attributes). Discriminate on prefixes unique to
    # the new arch only.
    transformer_prefixes = ("tokenizer.", "switch_encoder.", "action_head.")
    return "transformer" if any(k.startswith(transformer_prefixes) for k in state.keys()) else "mlp"


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    arch = ckpt.get("arch") or _infer_arch_from_state_dict(state)

    if arch == "transformer":
        # New arch (REWRITE_DESIGN.md, Session 47+). No dim-expansion path —
        # these checkpoints can't predate the spec.
        from model_transformer import (
            TransformerBattlePolicy, TransformerConfig, load_move_flag_lookup,
        )
        from pathlib import Path as _Path
        cfg = TransformerConfig.from_dict(ckpt.get("model_config", {}))
        lookup = load_move_flag_lookup(
            _Path("data/lookup/move_flags_v1.pt"), expected_n_moves=cfg.n_moves,
        )
        model = TransformerBattlePolicy(cfg, move_flag_lookup=lookup).to(device)
        # torch.compile wraps modules with `_orig_mod.` prefix in state_dict;
        # strip on load so we can use either compiled or uncompiled ckpts.
        state = {k.replace("._orig_mod.", "."): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        return model, cfg, ckpt

    # Legacy MLP arch.
    cfg = PokeTransformerConfig.from_dict(ckpt.get("model_config", {}))
    model = PokeTransformer(cfg).to(device)

    # Handle dim expansion for type effectiveness features (zero-init new columns)
    # move_net.mlp.0.weight: 187 -> 189 (+2: type_eff, opp_threat)
    # switch_mlp.0.weight: 60 -> 62 (+2: defensive/offensive effectiveness)
    _expand_targets = ["move_net.mlp.0.weight", "switch_mlp.0.weight"]
    for key in list(state.keys()):
        if any(key.endswith(t) for t in _expand_targets):
            old_w = state[key]
            parts = key.split(".")
            mod = model
            for p in parts[:-1]:
                mod = getattr(mod, p) if not p.isdigit() else mod[int(p)]
            expected_in = mod.in_features
            if old_w.shape[1] < expected_in:
                pad = expected_in - old_w.shape[1]
                state[key] = torch.cat([old_w, torch.zeros(old_w.shape[0], pad, device=old_w.device)], dim=1)
                print(f"  [INFO] Expanding {key}: {old_w.shape[1]} -> {expected_in} (+{pad} dims, zero-init)")

    model.load_state_dict(state, strict=True)
    return model, cfg, ckpt


def save_checkpoint(path, model, cfg, optimizer, iteration, metrics=None):
    # Architecture tag — read by load_checkpoint to dispatch to the right class.
    # Detect from cfg type so callers don't have to pass an extra argument.
    arch = "transformer" if type(cfg).__name__ == "TransformerConfig" else "mlp"
    ckpt = {
        "arch": arch,
        "model_state_dict": model.state_dict(),
        "model_config": cfg.to_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "metrics": metrics or {},
        "v8_version": "8.0",
    }
    # Atomic write: save to temp file then rename to prevent corruption on crash
    tmp_path = str(path) + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)
