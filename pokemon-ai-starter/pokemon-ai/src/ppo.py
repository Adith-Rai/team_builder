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

import gc
import os
import random
import traceback
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import PokeTransformer, PokeTransformerConfig


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
# PPO Update (batched spatial, sequential temporal)
# =============================

def ppo_update(model: PokeTransformer, optimizer, episodes: List[dict],
                  device: torch.device, cfg: PokeTransformerConfig,
                  epochs: int = 3, clip_eps: float = 0.2, ent_coef: float = 0.02,
                  vf_coef: float = 0.5, max_grad_norm: float = 0.5,
                  target_kl: float = 0.02, grad_accum: int = 1,
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
                spatial_out, all_summaries = model.forward_spatial(mega)
                action_ctx = model.action_encoder(
                    mega["active_move_ids"], mega["active_move_banks"],
                    mega["active_move_cont"], mega["switch_ids"], mega["switch_cont"],
                )
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

                    temporal_ctx = model.temporal(summary_buf)
                    actor_out = spatial_out[t, 0, :]
                    critic_out = spatial_out[t, 1, :]
                    act_ctx = action_ctx[t]

                    at = torch.cat([actor_out, temporal_ctx.squeeze(0)], dim=-1)
                    at_exp = at.unsqueeze(0).expand(9, -1)
                    pi_input = torch.cat([at_exp, act_ctx], dim=-1)
                    logits = model.policy_head(pi_input).squeeze(-1)
                    logits = logits.masked_fill(legal_all[t] < 0.5, -100.0)
                    all_logits.append(logits)

                    vi = torch.cat([critic_out, temporal_ctx.squeeze(0)], dim=-1)
                    vl = model.value_head(vi.unsqueeze(0)).squeeze(0)
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
                    v_pred_step = (v_probs_step * model.v_support).sum(-1)
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

def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = PokeTransformerConfig.from_dict(ckpt.get("model_config", {}))
    model = PokeTransformer(cfg).to(device)

    # Handle dim expansion for type effectiveness features (zero-init new columns)
    # move_net.mlp.0.weight: 187 -> 189 (+2: type_eff, opp_threat)
    # switch_mlp.0.weight: 60 -> 62 (+2: defensive/offensive effectiveness)
    state = ckpt["model_state_dict"]
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
    ckpt = {
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
