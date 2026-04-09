#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO reinforcement learning trainer for Pokemon battles.

Usage:
    SHOWDOWN_HOST=127.0.0.1 python rl_train.py --init-from checkpoints/bc_best.pt --device cuda

Collects trajectories via poke-env self-play, then updates the policy with PPO.
Supports warm-starting from a BC checkpoint.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from poke_env import AccountConfiguration
from poke_env.concurrency import POKE_LOOP
from poke_env.player import Player
from poke_env.player.baselines import SimpleHeuristicsPlayer, MaxBasePowerPlayer
from poke_env.ps_client.server_configuration import ServerConfiguration

from features import (
    featurize, action_mask, make_obs_mask_and_slots,
    step_type_from_abs_t, derive_ctx_extra_live,
    encode_opp_last_ctx, _opp_last_action_from_logs,
    encode_move_and_switch_slots,
)
from policy_heads import BattlePolicy, PolicyConfig, ModifierSpec, ppo_losses
from rewards import RewardShaper, terminal_sparse
from teams_ou import random_teambuilder, random_pool_teambuilder
from policy_rulebots import (
    GreedySEPlayer, HazardSensePlayer,
    SwitchAwareEscapePlayer, SetupThenSweepPlayer,
)
from policy_smartbots import SmartDamagePlayer, TacticalPlayer, StrategicPlayer
from bc_policy_player import BCPolicyPlayer


# =========================================================================
# Server config
# =========================================================================
_SD_HOST = os.environ.get("SHOWDOWN_HOST", "127.0.0.1")
_SD_PORT = os.environ.get("SHOWDOWN_PORT", "8000")


def make_server(port=None):
    host = _SD_HOST
    p = port or _SD_PORT
    return ServerConfiguration(
        f"ws://{host}:{p}/showdown/websocket",
        f"http://{host}:{p}/action.php?"
    )


def make_server_pool(ports_csv: str):
    """Create a list of ServerConfiguration objects from comma-separated ports.

    E.g. "8000,8001,8002" -> [ServerConfiguration(...:8000), ...]
    Returns at least one server (default port) if ports_csv is empty.
    """
    if not ports_csv or ports_csv.strip() == "":
        return [make_server()]
    ports = [p.strip() for p in ports_csv.split(",") if p.strip()]
    return [make_server(port=p) for p in ports]


# =========================================================================
# Trajectory storage
# =========================================================================
class Trajectory:
    """Stores one episode's transitions."""
    __slots__ = (
        "obs", "actions", "action_masks", "log_probs", "values",
        "rewards", "dones", "ctx_extras", "move_slots", "switch_slots",
        "step_types", "entity_ids", "move_ids", "switch_ids",
    )

    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.action_masks: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.ctx_extras: List[Optional[np.ndarray]] = []
        self.move_slots: List[np.ndarray] = []
        self.switch_slots: List[np.ndarray] = []
        self.step_types: List[int] = []
        self.entity_ids: List[np.ndarray] = []
        self.move_ids: List[np.ndarray] = []
        self.switch_ids: List[np.ndarray] = []

    def __len__(self):
        return len(self.obs)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else 0.0
        next_done = dones[t]
        delta = rewards[t] + gamma * next_val * (1 - next_done) - values[t]
        last_gae = delta + gamma * lam * (1 - next_done) * last_gae
        advantages[t] = last_gae
    returns = advantages + np.array(values[:T], dtype=np.float32)
    return advantages, returns


# =========================================================================
# RL Player — collects trajectories during battles
# =========================================================================
class RLPlayer(Player):
    """
    poke-env Player that uses a BattlePolicy for decisions and records
    trajectories for PPO training.
    """

    def __init__(
        self,
        model: BattlePolicy,
        device: torch.device,
        reward_shaper: RewardShaper,
        ctx_extra_dim: int = 0,
        step_type_bins: int = 0,
        temperature: float = 1.0,
        **player_kwargs,
    ):
        player_kwargs.pop("replay_folder", None)
        super().__init__(**player_kwargs)
        self.model = model
        self.device = device
        self.reward_shaper = reward_shaper
        self.ctx_extra_dim = ctx_extra_dim
        self.step_type_bins = step_type_bins
        self.temperature = temperature

        # Per-battle trajectory storage
        self._trajectories: Dict[str, Trajectory] = {}
        self._shapers: Dict[str, RewardShaper] = {}
        self._hidden: Dict[str, Optional[Tuple]] = {}
        self._history: Dict[str, list] = {}  # transformer: per-battle input history
        self._is_transformer = getattr(self.model, 'transformer_core', None) is not None
        self.completed_trajectories: List[Trajectory] = []

    def _get_btag(self, battle) -> str:
        return getattr(battle, "battle_tag", None) or f"battle-{id(battle)}"

    def _ensure_trajectory(self, battle):
        btag = self._get_btag(battle)
        if btag not in self._trajectories:
            self._trajectories[btag] = Trajectory()
            self._shapers[btag] = RewardShaper(
                ko_coef=self.reward_shaper.ko_coef,
                hp_coef=self.reward_shaper.hp_coef,
                terminal_coef=self.reward_shaper.terminal_coef,
                clip_abs=self.reward_shaper.clip_abs,
            )
            self._hidden[btag] = None
        return self._trajectories[btag], self._shapers[btag]

    def choose_move(self, battle):
        traj, shaper = self._ensure_trajectory(battle)
        btag = self._get_btag(battle)

        # Compute shaping reward for previous step
        shape_r = shaper.step(battle)
        if len(traj.rewards) > 0:
            traj.rewards[-1] += shape_r

        # Featurize
        obs_vec, legal_mask, ctx, mv_slots, sw_slots, entity_ids, move_ids, switch_ids = make_obs_mask_and_slots(battle)

        obs_np = np.array(obs_vec, dtype=np.float32)
        mask_np = np.array(legal_mask, dtype=np.float32)
        mv_np = np.array(mv_slots, dtype=np.float32)
        sw_np = np.array(sw_slots, dtype=np.float32)

        # Guard: replace NaN in features with 0 (can occur on edge-case battle states)
        np.nan_to_num(obs_np, copy=False)
        np.nan_to_num(mv_np, copy=False)
        np.nan_to_num(sw_np, copy=False)

        # Guard: if no actions are legal, allow all (fallback to random)
        if mask_np.sum() == 0:
            mask_np[:] = 1.0

        eid_np = np.array(entity_ids, dtype=np.int64)
        mid_np = np.array(move_ids, dtype=np.int64)
        sid_np = np.array(switch_ids, dtype=np.int64)

        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)
        mv_t = torch.from_numpy(mv_np).unsqueeze(0).to(self.device)
        sw_t = torch.from_numpy(sw_np).unsqueeze(0).to(self.device)
        eid_t = torch.from_numpy(eid_np).unsqueeze(0).to(self.device)
        mid_t = torch.from_numpy(mid_np).unsqueeze(0).to(self.device)
        sid_t = torch.from_numpy(sid_np).unsqueeze(0).to(self.device)

        # Context
        ctx_t = None
        if self.ctx_extra_dim > 0:
            if ctx is not None:
                ctx_np = np.array(ctx, dtype=np.float32)
            else:
                ctx_np = np.zeros(self.ctx_extra_dim, dtype=np.float32)
            np.nan_to_num(ctx_np, copy=False)
            ctx_t = torch.from_numpy(ctx_np).unsqueeze(0).to(self.device)

        step_t = None
        if self.step_type_bins > 0:
            t_abs = max(0, int(getattr(battle, "turn", 1)) - 1)
            bin_idx = step_type_from_abs_t(t_abs, bins=self.step_type_bins, cap=50)
            step_t = torch.tensor([bin_idx], dtype=torch.long, device=self.device)

        # Forward pass
        with torch.no_grad():
            if self._is_transformer:
                # Transformer: accumulate history on CPU, pass full sequence
                hist = self._history.get(btag, [])
                hist.append({
                    "obs": obs_t.cpu(), "mask": mask_t.cpu(),
                    "step_type": step_t.cpu() if step_t is not None else None,
                    "ctx_extra": ctx_t.cpu() if ctx_t is not None else None,
                    "entity_ids": eid_t.cpu() if eid_t is not None else None,
                    "move_ids": mid_t.cpu() if mid_t is not None else None,
                    "switch_ids": sid_t.cpu() if sid_t is not None else None,
                    "move_slots": mv_t.cpu(), "switch_slots": sw_t.cpu(),
                })
                ctx_len = getattr(self.model.cfg, "context_length", 128)
                if len(hist) > ctx_len:
                    hist = hist[-ctx_len:]
                self._history[btag] = hist

                T = len(hist)
                dev = self.device

                def _cat(key):
                    vals = [h[key] for h in hist]
                    if vals[0] is None:
                        return None
                    return torch.cat(vals, dim=0).unsqueeze(0).to(dev)

                obs_seq = _cat("obs")
                mask_seq = _cat("mask")
                st_seq = _cat("step_type")
                ctx_seq = _cat("ctx_extra")
                eid_seq = _cat("entity_ids")
                mid_seq = _cat("move_ids")
                sid_seq = _cat("switch_ids")
                ms_seq = _cat("move_slots")
                ss_seq = _cat("switch_slots")
                sl = torch.tensor([T], device=dev)

                out = self.model(
                    obs_seq, action_mask=mask_seq,
                    step_type=st_seq, ctx_extra=ctx_seq,
                    move_slots=ms_seq, switch_slots=ss_seq,
                    entity_ids=eid_seq, move_ids=mid_seq, switch_ids=sid_seq,
                    seq_lens=sl,
                )
                logits = out["action_logits"][:, -1, :]  # last timestep [1, 9]
                value = out["value"][:, -1, :].squeeze(-1).item()
            else:
                # LSTM: single-step with hidden state
                h0 = self._hidden.get(btag, None)
                out = self.model(
                    obs_t, action_mask=mask_t,
                    step_type=step_t, ctx_extra=ctx_t,
                    move_slots=mv_t, switch_slots=sw_t,
                    entity_ids=eid_t, move_ids=mid_t, switch_ids=sid_t,
                    h0=h0,
                )
                self._hidden[btag] = out.get("h_n", None)
                logits = out["action_logits"]  # [1, 9]
                value = out["value"].squeeze(-1).item()  # scalar

            # Guard: if logits or value contain NaN (edge-case battle states or corrupted weights)
            if torch.isnan(logits).any():
                if not hasattr(self, '_nan_count'):
                    self._nan_count = 0
                self._nan_count += 1
                if self._nan_count <= 3:
                    print(f"  [WARN] NaN logits at turn {getattr(battle, 'turn', '?')}", flush=True)
                elif self._nan_count == 4:
                    print(f"  [WARN] Suppressing further NaN warnings (count={self._nan_count})", flush=True)
                logits = torch.where(mask_t > 0, torch.zeros_like(logits),
                                     torch.full_like(logits, -1e9))
            if not np.isfinite(value):
                value = 0.0

            # Temperature-scaled sampling
            scaled_logits = logits / max(self.temperature, 0.01)
            probs = F.softmax(scaled_logits, dim=-1)  # [1, 9]
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()  # [1]
            action_idx = action.item()

            # Store UNSCALED log_prob for PPO importance ratio consistency.
            # Temperature only affects which action is sampled, not the stored
            # log_prob. During PPO update, model re-forward produces unscaled
            # logits, so old_logp must also be unscaled to avoid biased ratios.
            unscaled_dist = torch.distributions.Categorical(logits=logits.squeeze(0))
            log_prob = unscaled_dist.log_prob(action.squeeze(0)).item()

        # Store transition
        traj.obs.append(obs_np)
        traj.actions.append(action_idx)
        traj.action_masks.append(mask_np)
        traj.log_probs.append(log_prob)
        traj.values.append(value)
        traj.rewards.append(0.0)  # will be filled in next step / end
        traj.dones.append(False)
        if ctx is not None:
            traj.ctx_extras.append(np.array(ctx, dtype=np.float32))
        else:
            traj.ctx_extras.append(None)
        traj.move_slots.append(mv_np)
        traj.switch_slots.append(sw_np)
        traj.entity_ids.append(eid_np)
        traj.move_ids.append(mid_np)
        traj.switch_ids.append(sid_np)
        # Store step_type bin index for PPO re-forward
        if self.step_type_bins > 0:
            t_abs = max(0, int(getattr(battle, "turn", 1)) - 1)
            traj.step_types.append(step_type_from_abs_t(t_abs, bins=self.step_type_bins, cap=50))
        else:
            traj.step_types.append(0)

        # Map action to battle order
        avail_moves = list(battle.available_moves or [])
        avail_switches = list(battle.available_switches or [])
        num_moves = min(4, len(avail_moves))

        if action_idx < num_moves:
            return self.create_order(avail_moves[action_idx])
        sidx = action_idx - 4
        if 0 <= sidx < len(avail_switches):
            return self.create_order(avail_switches[sidx])
        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        """Called by poke-env when a battle ends."""
        super()._battle_finished_callback(battle)
        btag = self._get_btag(battle)

        if btag in self._trajectories:
            traj = self._trajectories[btag]
            shaper = self._shapers[btag]

            # Final dense reward for the last turn (shaper.step not called at episode end)
            final_shape = shaper.step(battle)
            if len(traj.rewards) > 0:
                traj.rewards[-1] += final_shape

            # Terminal reward
            won = battle.won
            if won is not None:
                terminal_r = terminal_sparse(won)
            else:
                terminal_r = 0.0

            # Reset shaper state
            shaper.end_episode()

            if len(traj.rewards) > 0:
                traj.rewards[-1] += terminal_r
                traj.dones[-1] = True

            if len(traj) > 0:
                self.completed_trajectories.append(traj)

            # Cleanup
            del self._trajectories[btag]
            del self._shapers[btag]
            if btag in self._hidden:
                del self._hidden[btag]
            self._history.pop(btag, None)

    def reset_trajectories(self):
        """Clear completed trajectories for next collection round."""
        self.completed_trajectories = []
        self._trajectories.clear()
        self._shapers.clear()
        self._hidden.clear()
        self._history.clear()


# =========================================================================
# Opponent pool + curriculum tiers (v3)
# =========================================================================
OPPONENT_BOTS = {
    "SimpleHeuristics": SimpleHeuristicsPlayer,
    "SmartDamage": SmartDamagePlayer,
    "Tactical": TacticalPlayer,
    "Strategic": StrategicPlayer,
    "MaxBasePower": MaxBasePowerPlayer,
    "GreedySE": GreedySEPlayer,
    "HazardSense": HazardSensePlayer,
    "SwitchAwareEscape": SwitchAwareEscapePlayer,
    "SetupThenSweep": SetupThenSweepPlayer,
}

# v2 flat weights (used when curriculum is off) — uniform across all 9 bots
OPPONENT_WEIGHTS_FLAT = {
    "SimpleHeuristics": 1.0,
    "SmartDamage": 1.0,
    "Tactical": 1.0,
    "Strategic": 1.0,
    "MaxBasePower": 1.0,
    "GreedySE": 1.0,
    "HazardSense": 1.0,
    "SwitchAwareEscape": 1.0,
    "SetupThenSweep": 1.0,
}

# v4 self-play weights: self-play dominant, smart bots moderate, easy bots low for diversity.
# Rationale: self-play provides adaptive opponents that scale with skill.
# Smart bots test specific skills (type calc, hazards, setup).
# Easy bots at low weight prevent forgetting basic play + add strategy diversity.
# BCPolicy anchor prevents catastrophic forgetting (KL penalty also helps).
OPPONENT_WEIGHTS_SELFPLAY = {
    "SimpleHeuristics": 1.0,
    "SmartDamage": 1.0,
    "Tactical": 1.0,
    "Strategic": 1.0,
    "MaxBasePower": 0.3,
    "GreedySE": 0.3,
    "HazardSense": 0.3,
    "SwitchAwareEscape": 0.3,
    "SetupThenSweep": 0.3,
    # BCPolicy and SelfPlay added dynamically in collect_trajectories
}

# Curriculum tiers (v3): unlock harder bots as the model improves
# Promotion requires minimum win rate vs EVERY bot in the tier (not average).
# This prevents easy bots from inflating the score and causing premature promotion.
CURRICULUM_TIERS = [
    {
        "name": "Tier 1 (basics)",
        "bots": {"MaxBasePower": 3.0, "GreedySE": 2.0, "SetupThenSweep": 2.0},
        "promote_min_wr": 0.40,  # need 40%+ vs EACH bot to advance
    },
    {
        "name": "Tier 2 (intermediate)",
        "bots": {"MaxBasePower": 1.0, "GreedySE": 1.0, "SetupThenSweep": 1.0,
                 "SimpleHeuristics": 3.0},
        "promote_min_wr": 0.40,  # need 40%+ vs EACH bot
    },
    {
        "name": "Tier 3 (advanced)",
        "bots": {"MaxBasePower": 1.0, "SimpleHeuristics": 2.0,
                 "SmartDamage": 2.0, "Tactical": 3.0, "Strategic": 3.0,
                 "HazardSense": 1.0, "SwitchAwareEscape": 1.0},
        "promote_min_wr": 0.40,  # need 40%+ vs EACH bot
    },
    {
        "name": "Tier 4 (full)",
        "bots": {"SimpleHeuristics": 2.0, "SmartDamage": 2.0,
                 "Tactical": 3.0, "Strategic": 3.0,
                 "MaxBasePower": 1.0, "GreedySE": 1.0,
                 "HazardSense": 1.0, "SwitchAwareEscape": 1.0,
                 "SetupThenSweep": 1.0},
        "promote_min_wr": 1.0,  # never promote beyond this
    },
]


def get_curriculum_weights(
    tier_idx: int,
    eval_results: Optional[Dict[str, float]] = None,
    adaptive: bool = True,
    min_weight: float = 0.5,
) -> Dict[str, float]:
    """Return opponent weights for the given curriculum tier.

    If adaptive=True and eval_results are provided, dynamically adjusts weights:
    bots below the promotion threshold get higher weight (proportional to gap),
    bots above get reduced weight (floored at min_weight).
    This focuses training on the actual bottleneck opponents.
    """
    tier = CURRICULUM_TIERS[min(tier_idx, len(CURRICULUM_TIERS) - 1)]
    base_weights = dict(tier["bots"])

    if not adaptive or eval_results is None:
        return base_weights

    threshold = tier["promote_min_wr"]
    adjusted = {}
    for bot, base_w in base_weights.items():
        wr = eval_results.get(bot, -1)
        if wr < 0:
            # No eval data for this bot — keep base weight
            adjusted[bot] = base_w
        elif wr >= threshold:
            # Already mastered — reduce weight but keep minimum for diversity
            adjusted[bot] = max(min_weight, base_w * 0.5)
        else:
            # Below threshold — increase weight proportional to gap
            # e.g., threshold=0.50, wr=0.10 → scale = 0.50/0.10 = 5.0
            # e.g., threshold=0.50, wr=0.40 → scale = 0.50/0.40 = 1.25
            scale = threshold / max(wr, 0.05)
            adjusted[bot] = base_w * min(scale, 3.0)  # cap at 3x base

    return adjusted


def check_curriculum_promotion(
    tier_idx: int, eval_results: Dict[str, float]
) -> Tuple[int, bool, str]:
    """Check if we should promote to the next tier.

    Requires minimum win rate vs EVERY bot in the current tier (not average).
    Returns (new_tier, promoted, reason_string).
    """
    tier = CURRICULUM_TIERS[min(tier_idx, len(CURRICULUM_TIERS) - 1)]
    if tier_idx >= len(CURRICULUM_TIERS) - 1:
        return tier_idx, False, ""

    threshold = tier["promote_min_wr"]
    tier_bots = list(tier["bots"].keys())
    # Only check bots that are in the eval results (some may not be in eval_bots)
    bot_wrs = {}
    for b in tier_bots:
        wr = eval_results.get(b, -1)
        if wr >= 0:
            bot_wrs[b] = wr

    if not bot_wrs:
        return tier_idx, False, "no eval data"

    # Check if ALL tier bots meet the minimum
    failing = {b: wr for b, wr in bot_wrs.items() if wr < threshold}
    if not failing:
        min_bot = min(bot_wrs, key=bot_wrs.get)
        reason = (f"all bots >= {threshold:.0%}, "
                  f"min={min_bot}@{bot_wrs[min_bot]:.0%}")
        return tier_idx + 1, True, reason
    else:
        fail_str = ", ".join(f"{b}={wr:.0%}" for b, wr in failing.items())
        return tier_idx, False, f"below {threshold:.0%}: {fail_str}"



# =========================================================================
# KL divergence against BC reference (prevents catastrophic forgetting)
# =========================================================================
def compute_kl_from_ref(model_logits: torch.Tensor, ref_logits: torch.Tensor,
                         action_mask: torch.Tensor) -> torch.Tensor:
    """KL(ref || model) — penalizes model for diverging from reference.

    Uses masked log-softmax so illegal actions don't affect the KL.
    """
    # Mask illegal actions with large negative (NOT -inf, which causes 0*-inf=NaN)
    illegal = (action_mask <= 0)
    # Use -6e4 instead of -1e9: safe for AMP float16 (max ~65504)
    ml = model_logits.masked_fill(illegal, -6e4)
    rl = ref_logits.masked_fill(illegal, -6e4)

    model_logp = F.log_softmax(ml, dim=-1)
    ref_logp = F.log_softmax(rl, dim=-1)
    ref_p = ref_logp.exp()

    # KL(ref || model) = sum ref_p * (ref_logp - model_logp)
    kl_terms = ref_p * (ref_logp - model_logp)
    # Safety: clamp any residual numerical noise
    kl_terms = kl_terms.clamp(min=0.0)
    return kl_terms.sum(dim=-1).mean()


# =========================================================================
# PPO batch builder — preserves episode structure for LSTM
# =========================================================================
def build_ppo_episodes(
    trajectories: List[Trajectory],
    gamma: float = 0.99,
    lam: float = 0.95,
    ctx_extra_dim: int = 0,
):
    """Convert completed trajectories into per-episode tensors (LSTM-friendly).

    Returns a list of episode dicts, each containing tensors for one episode.
    Advantages are normalized globally across all episodes.
    """
    episodes = []
    all_advantages_flat = []

    for traj in trajectories:
        if len(traj) == 0:
            continue
        # Clean NaN/inf in rewards and values before GAE
        clean_rewards = [r if np.isfinite(r) else 0.0 for r in traj.rewards]
        clean_values = [v if np.isfinite(v) else 0.0 for v in traj.values]
        advantages, returns = compute_gae(
            clean_rewards, clean_values, traj.dones, gamma, lam
        )
        all_advantages_flat.append(advantages)

        # Build context
        ctx_list = []
        for c in traj.ctx_extras:
            if c is not None:
                ctx_list.append(c)
            elif ctx_extra_dim > 0:
                ctx_list.append(np.zeros(ctx_extra_dim, dtype=np.float32))

        ep = {
            "obs": torch.from_numpy(np.stack(traj.obs)),
            "actions": torch.tensor(traj.actions, dtype=torch.long),
            "action_masks": torch.from_numpy(np.stack(traj.action_masks)),
            "old_logp": torch.tensor(traj.log_probs, dtype=torch.float32),
            "advantages": torch.from_numpy(advantages),
            "returns": torch.from_numpy(returns),
            "move_slots": torch.from_numpy(np.stack(traj.move_slots)),
            "switch_slots": torch.from_numpy(np.stack(traj.switch_slots)),
            "step_types": torch.tensor(traj.step_types, dtype=torch.long),
            "entity_ids": torch.from_numpy(np.stack(traj.entity_ids)),
            "move_ids": torch.from_numpy(np.stack(traj.move_ids)),
            "switch_ids": torch.from_numpy(np.stack(traj.switch_ids)),
        }
        if ctx_list:
            ep["ctx_extra"] = torch.from_numpy(np.stack(ctx_list))
        episodes.append(ep)

    if not episodes:
        return None

    # Global advantage normalization
    all_adv = np.concatenate(all_advantages_flat)
    adv_mean, adv_std = all_adv.mean(), all_adv.std() + 1e-8
    for ep in episodes:
        ep["advantages"] = (ep["advantages"] - adv_mean) / adv_std

    return episodes


# =========================================================================
# PPO update step — processes episodes sequentially for correct LSTM states
# =========================================================================
def ppo_update(
    model: BattlePolicy,
    optimizer: torch.optim.Optimizer,
    episodes: list,
    device: torch.device,
    epochs: int = 4,
    clip_eps: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    use_amp: bool = False,
    ref_model: Optional[BattlePolicy] = None,
    kl_coef: float = 0.0,
):
    """Run PPO updates processing whole episodes to preserve LSTM hidden states.

    Instead of shuffling individual steps (which destroys LSTM context),
    we shuffle the episode ORDER and process each episode sequentially,
    threading the LSTM hidden state through the episode.
    """
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_accum = defaultdict(float)
    n_updates = 0

    for epoch in range(epochs):
        # Shuffle episode order, not individual steps
        ep_order = np.random.permutation(len(episodes))

        for ep_idx in ep_order:
            ep = episodes[ep_idx]
            T = ep["obs"].shape[0]
            if T == 0:
                continue

            # Process the whole episode as a single sequence [1, T, ...]
            ep_obs = ep["obs"].unsqueeze(0).to(device)        # [1, T, F]
            ep_actions = ep["actions"].to(device)              # [T]
            ep_masks = ep["action_masks"].unsqueeze(0).to(device)  # [1, T, 9]
            ep_old_logp = ep["old_logp"].to(device)            # [T]
            ep_adv = ep["advantages"].to(device)               # [T]
            ep_ret = ep["returns"].to(device)                  # [T]
            ep_mv = ep["move_slots"].unsqueeze(0).to(device)   # [1, T, 4, M]
            ep_sw = ep["switch_slots"].unsqueeze(0).to(device) # [1, T, 5, S]
            ep_eid = ep["entity_ids"].unsqueeze(0).to(device)  # [1, T, N]
            ep_mid = ep["move_ids"].unsqueeze(0).to(device)    # [1, T, 4]
            ep_sid = ep["switch_ids"].unsqueeze(0).to(device)  # [1, T, 5]

            ep_ctx = None
            if "ctx_extra" in ep:
                ep_ctx = ep["ctx_extra"].unsqueeze(0).to(device)

            ep_step = None
            if "step_types" in ep:
                ep_step = ep["step_types"].unsqueeze(0).to(device)

            # seq_lens for transformer padding mask (single episode, no padding)
            ep_seq_lens = torch.tensor([T], device=device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # Forward pass over full episode — LSTM threads hidden state,
                # transformer uses causal attention over full sequence
                out = model(
                    ep_obs,
                    action_mask=ep_masks,
                    step_type=ep_step,
                    ctx_extra=ep_ctx,
                    move_slots=ep_mv,
                    switch_slots=ep_sw,
                    entity_ids=ep_eid,
                    move_ids=ep_mid,
                    switch_ids=ep_sid,
                    seq_lens=ep_seq_lens,
                    h0=None,  # Fresh hidden for episode start (ignored by transformer)
                )

                # Reshape outputs from [1, T, ...] to [T, ...]
                logits_seq = out["action_logits"].squeeze(0)  # [T, 9]
                value_seq = out["value"].squeeze(0)           # [T, 1]

                # Replace NaN in logits/values to prevent gradient corruption
                if torch.isnan(logits_seq).any():
                    logits_seq = torch.nan_to_num(logits_seq, nan=0.0)
                if torch.isnan(value_seq).any():
                    value_seq = torch.nan_to_num(value_seq, nan=0.0)

                # Build per-step output dict for ppo_losses
                step_out = {
                    "action_logits": logits_seq,
                    "value": value_seq,
                    "mod_logits": {k: v.squeeze(0) for k, v in out["mod_logits"].items()},
                }

                total_loss, comps = ppo_losses(
                    out=step_out,
                    actions=ep_actions,
                    old_logp=ep_old_logp,
                    advantages=ep_adv,
                    returns=ep_ret,
                    action_mask=ep_masks.squeeze(0),
                    mod_labels={},
                    mod_masks={},
                    clip_eps=clip_eps,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                )

                # KL penalty against BC reference model
                kl_loss = torch.tensor(0.0, device=device)
                if ref_model is not None and kl_coef > 0:
                    with torch.no_grad():
                        ref_out = ref_model(
                            ep_obs,
                            action_mask=ep_masks,
                            step_type=ep_step,
                            ctx_extra=ep_ctx,
                            move_slots=ep_mv,
                            switch_slots=ep_sw,
                            entity_ids=ep_eid,
                            move_ids=ep_mid,
                            switch_ids=ep_sid,
                            seq_lens=ep_seq_lens,
                        )
                    ref_logits = ref_out["action_logits"].squeeze(0)
                    kl_loss = kl_coef * compute_kl_from_ref(
                        logits_seq, ref_logits, ep_masks.squeeze(0)
                    )
                    total_loss = total_loss + kl_loss

            # Guard: skip this episode if loss is NaN (prevents weight corruption)
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)

            # Check for NaN gradients before clipping/stepping
            has_nan_grad = any(
                p.grad is not None and torch.isnan(p.grad).any().item()
                for p in model.parameters()
            )
            if has_nan_grad:
                print(f"  [WARN] NaN gradients detected, skipping this step", flush=True)
                optimizer.zero_grad()  # Clear the bad gradients
                continue

            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            # Check for NaN in gradients/weights after update
            has_nan_weights = any(
                torch.isnan(p).any().item() for p in model.parameters() if p is not None
            )
            if has_nan_weights:
                print(f"  [FATAL] NaN in model weights after PPO update (epoch {epoch}, ep {ep_idx})!", flush=True)
                # Try to identify which parameter has NaN
                for name, p in model.named_parameters():
                    if torch.isnan(p).any():
                        print(f"    NaN param: {name} shape={p.shape}", flush=True)
                break

            for k, v in comps.items():
                loss_accum[k] += v.item() if torch.is_tensor(v) else float(v)
            loss_accum["kl"] += kl_loss.item() if torch.is_tensor(kl_loss) else float(kl_loss)
            n_updates += 1
        else:
            continue  # inner loop didn't break
        break  # inner loop broke (NaN weights), stop outer loop too

    return {k: v / max(1, n_updates) for k, v in loss_accum.items()}


# =========================================================================
# Evaluation
# =========================================================================
_eval_round = 0
_pid_tag = os.getpid() % 10000  # Per-process tag to avoid name collisions


async def evaluate_vs_bot(
    model: BattlePolicy,
    device: torch.device,
    bot_cls,
    server,
    n_battles: int = 20,
    battle_format: str = "gen9ou",
    ctx_extra_dim: int = 0,
    step_type_bins: int = 0,
    max_concurrent: int = 5,
    timeout_per_battle: float = 45.0,
    use_direct: bool = False,
):
    """Evaluate current policy against a bot. Returns win rate.

    Uses per-battle team randomization to eliminate team-matchup variance.
    Uses unique player names (with PID) to avoid stale connection conflicts.
    Includes timeout to prevent hanging on Showdown websocket issues.
    """
    global _eval_round
    _eval_round += 1
    eid = _eval_round

    # Common kwargs; when --direct, skip websocket listener
    extra_kwargs = {}
    if use_direct:
        extra_kwargs["start_listening"] = False
    else:
        extra_kwargs["server_configuration"] = server

    eval_player = RLPlayer(
        model=model,
        device=device,
        reward_shaper=RewardShaper(),
        ctx_extra_dim=ctx_extra_dim,
        step_type_bins=step_type_bins,
        temperature=0.01,  # near-greedy
        battle_format=battle_format,
        team=random_pool_teambuilder(),
        max_concurrent_battles=max_concurrent,
        account_configuration=AccountConfiguration(f"Ev{_pid_tag}r{eid}", None),
        **extra_kwargs,
    )
    opponent = bot_cls(
        battle_format=battle_format,
        team=random_pool_teambuilder(),
        max_concurrent_battles=max_concurrent,
        account_configuration=AccountConfiguration(f"EO{_pid_tag}r{eid}", None),
        **extra_kwargs,
    )

    # Patch players for direct transport if --direct
    if use_direct:
        from direct_player import patch_to_direct, direct_battle_against
        patch_to_direct(eval_player)
        patch_to_direct(opponent)

    timeout = timeout_per_battle * n_battles
    try:
        if use_direct:
            fut = asyncio.run_coroutine_threadsafe(
                direct_battle_against(eval_player, opponent, n_battles=n_battles),
                POKE_LOOP,
            )
            fut.result(timeout=min(timeout, 300))
        else:
            await asyncio.wait_for(
                eval_player.battle_against(opponent, n_battles=n_battles),
                timeout=min(timeout, 300),  # Cap at 5 min per eval
            )
    except (asyncio.TimeoutError, TimeoutError):
        print(f"  Eval vs {bot_cls.__name__} timed out after {min(timeout, 300):.0f}s")
    wins = eval_player.n_won_battles
    total = eval_player.n_won_battles + eval_player.n_lost_battles + eval_player.n_tied_battles
    wr = wins / max(1, total)

    # Cleanup eval players to free websocket buffers and battle state
    for p in (eval_player, opponent):
        try:
            ps = getattr(p, "ps_client", None) or getattr(p, "_ps_client", None)
            if ps and hasattr(ps, "_listening_coroutine"):
                ps._listening_coroutine.cancel()
        except Exception:
            pass
    del eval_player, opponent

    return wr


# =========================================================================
# Checkpoint helpers
# =========================================================================
def load_bc_checkpoint(path: str, device: torch.device):
    """Load a BC checkpoint and return (model, cfg, metadata)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model", {})
    pcfg_dict = ckpt.get("policy_cfg", None)

    if isinstance(pcfg_dict, dict):
        cfg = PolicyConfig(**pcfg_dict)
    else:
        # Infer from state dict
        obs_dim = ckpt.get("obs_dim", 988)
        use_lstm = any("lstm" in k for k in sd.keys())
        cfg = PolicyConfig(
            obs_dim=obs_dim,
            action_dim=9,
            use_lstm=use_lstm,
            lstm_hidden=256,
            mlp_hidden=256,
        )

    model = BattlePolicy(cfg).to(device)

    # Prune state dict to matching shapes
    msd = model.state_dict()
    pruned = {k: v for k, v in sd.items()
              if k in msd and v.shape == msd[k].shape}
    model.load_state_dict(pruned, strict=False)
    print(f"Loaded BC checkpoint: {len(pruned)}/{len(msd)} params matched")
    return model, cfg


def save_rl_checkpoint(model, optimizer, cfg, step, metrics, path,
                       lr_scheduler=None):
    """Save RL training checkpoint."""
    data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "policy_cfg": {
            "obs_dim": cfg.obs_dim,
            "action_dim": cfg.action_dim,
            "use_lstm": cfg.use_lstm,
            "use_transformer": cfg.use_transformer,
            "lstm_hidden": cfg.lstm_hidden,
            "mlp_hidden": cfg.mlp_hidden,
            "mlp_layers": cfg.mlp_layers,
            "lstm_layers": cfg.lstm_layers,
            "hierarchical": cfg.hierarchical,
            "step_type_bins": cfg.step_type_bins,
            "ctx_extra_dim": cfg.ctx_extra_dim,
            "ctx_proj_dim": cfg.ctx_proj_dim,
            "move_slot_dim": cfg.move_slot_dim,
            "switch_slot_dim": cfg.switch_slot_dim,
            "slot_hidden": cfg.slot_hidden,
            "n_entity_ids": cfg.n_entity_ids,
            "embed_dim": cfg.embed_dim,
            "n_species": cfg.n_species,
            "n_moves": cfg.n_moves,
            "n_items": cfg.n_items,
            "n_abilities": cfg.n_abilities,
            "n_transformer_layers": cfg.n_transformer_layers,
            "n_heads": cfg.n_heads,
            "transformer_dropout": cfg.transformer_dropout,
            "context_length": cfg.context_length,
            "modifiers": [{"name": m.name} if hasattr(m, 'name') else m for m in (cfg.modifiers or [])],
        },
        "obs_dim": cfg.obs_dim,
        "step": step,
        "metrics": metrics,
    }
    if lr_scheduler is not None:
        data["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(data, path)


# =========================================================================
# Main training loop
# =========================================================================
_collect_round = 0


async def collect_trajectories(
    model: BattlePolicy,
    device: torch.device,
    server,
    n_games: int = 100,
    battle_format: str = "gen9ou",
    ctx_extra_dim: int = 0,
    step_type_bins: int = 0,
    temperature: float = 1.0,
    bc_checkpoint: str = None,
    max_concurrent: int = 5,
    opponent_weights: Optional[Dict[str, float]] = None,
    self_play_checkpoint: Optional[str] = None,
    self_play_weight: float = 2.0,
    self_play_pool: Optional[List[str]] = None,
    bc_opponent_weight: float = 0.5,
    opponent_device: str = "cpu",
    reward_shaper: Optional[RewardShaper] = None,
    server_pool: Optional[list] = None,
    use_direct: bool = False,
):
    """Play n_games against random opponents, return trajectories.

    Batches games by opponent type for efficiency (fewer connections).
    Uses concurrent battles to speed up collection.
    Uses unique player names per iteration to avoid stale connection conflicts.

    v3: supports curriculum weights, self-play opponent, configurable reward shaper.
    v4: server_pool for multi-server parallelism (round-robin batches across servers).
    v5: self_play_pool — list of checkpoint paths for fictitious play (uniform over pool).
    """
    global _collect_round
    _collect_round += 1
    rid = _collect_round

    if reward_shaper is None:
        reward_shaper = RewardShaper()

    # Server pool: round-robin batches across available Showdown instances
    servers = server_pool if server_pool and len(server_pool) > 0 else [server]

    # Build opponent pool with weights (v3/v5: curriculum-aware + self-play pool)
    if opponent_weights is None:
        opponent_weights = OPPONENT_WEIGHTS_FLAT
    names = list(opponent_weights.keys())
    weights = [opponent_weights[n] for n in names]
    if bc_checkpoint is not None and bc_opponent_weight > 0:
        names.append("BCPolicy")
        weights.append(bc_opponent_weight)

    # v5: Self-play pool — distribute self_play_weight across all pool members uniformly.
    # Each pool member is a past snapshot or hall-of-fame checkpoint.
    # Falls back to single snapshot for backward compat.
    _sp_pool = self_play_pool or ([self_play_checkpoint] if self_play_checkpoint else [])
    _sp_pool = [p for p in _sp_pool if p is not None]
    if _sp_pool:
        per_sp_weight = self_play_weight / len(_sp_pool)
        for si, sp_path in enumerate(_sp_pool):
            names.append(f"SelfPlay_{si}")
            weights.append(per_sp_weight)
    opp_counts = defaultdict(int)
    opp_names = []
    for _ in range(n_games):
        chosen = random.choices(names, weights=weights, k=1)[0]
        opp_names.append(chosen)
        opp_counts[chosen] += 1

    # Play batches per opponent type.
    # Create a FRESH player per batch to avoid stale challenge cascade:
    # if one batch times out, the player's pending websocket challenge blocks
    # all subsequent battle_against() calls. Fresh player = clean connection.
    all_trajectories = []
    total_wins, total_losses, total_ties = 0, 0, 0
    batch_idx = 0
    for opp_name, count in opp_counts.items():
        batch_idx += 1
        # Round-robin server assignment across batches
        srv = servers[(batch_idx - 1) % len(servers)]

        # Common kwargs; when --direct, skip websocket listener
        extra_kwargs = {}
        if use_direct:
            extra_kwargs["start_listening"] = False
        else:
            extra_kwargs["server_configuration"] = srv

        player = RLPlayer(
            model=model,
            device=device,
            reward_shaper=reward_shaper,
            ctx_extra_dim=ctx_extra_dim,
            step_type_bins=step_type_bins,
            temperature=temperature,
            battle_format=battle_format,
            team=random_pool_teambuilder(),
            max_concurrent_battles=max_concurrent,
            account_configuration=AccountConfiguration(
                f"RL{_pid_tag}r{rid}b{batch_idx}", None),
            **extra_kwargs,
        )
        opp_acct = AccountConfiguration(f"Op{_pid_tag}r{rid}b{batch_idx}", None)
        if opp_name == "BCPolicy" or opp_name.startswith("SelfPlay"):
            if opp_name == "BCPolicy":
                ckpt = bc_checkpoint
            elif opp_name.startswith("SelfPlay_") and _sp_pool:
                sp_idx = int(opp_name.split("_")[1])
                ckpt = _sp_pool[sp_idx % len(_sp_pool)]
            else:
                ckpt = self_play_checkpoint
            opponent = BCPolicyPlayer(
                checkpoint_path=ckpt,
                device=opponent_device,
                battle_format=battle_format,
                team=random_pool_teambuilder(),
                max_concurrent_battles=max_concurrent,
                account_configuration=opp_acct,
                **extra_kwargs,
            )
        else:
            cls = OPPONENT_BOTS[opp_name]
            opponent = cls(
                battle_format=battle_format,
                team=random_pool_teambuilder(),
                max_concurrent_battles=max_concurrent,
                account_configuration=opp_acct,
                **extra_kwargs,
            )

        # Patch players for direct transport if --direct
        if use_direct:
            from direct_player import patch_to_direct, direct_battle_against
            patch_to_direct(player)
            patch_to_direct(opponent)

        try:
            if use_direct:
                # direct_battle_against must run on POKE_LOOP
                fut = asyncio.run_coroutine_threadsafe(
                    direct_battle_against(player, opponent, n_battles=count),
                    POKE_LOOP,
                )
                fut.result(timeout=min(45 * count, 300))
            else:
                await asyncio.wait_for(
                    player.battle_against(opponent, n_battles=count),
                    timeout=min(45 * count, 300),  # Cap at 5 min per batch
                )
        except (asyncio.TimeoutError, TimeoutError):
            print(f"  Batch vs {opp_name} ({count} games) timed out")

        total_wins += player.n_won_battles
        total_losses += player.n_lost_battles
        total_ties += player.n_tied_battles
        all_trajectories.extend(player.completed_trajectories)

        # Free battle objects + trajectory buffers to prevent memory leak.
        # Without cleanup: 100 games/iter × 200 iters = 20K Battle objects (~1GB)
        # + trajectory numpy arrays (~2GB).
        player.reset_battles()
        opponent.reset_battles()
        player.completed_trajectories.clear()
        # Cancel PSClient.listen coroutines to prevent zombie listeners on POKE_LOOP
        for p in (player, opponent):
            try:
                ps = getattr(p, "ps_client", None) or getattr(p, "_ps_client", None)
                if ps and hasattr(ps, "_listening_coroutine"):
                    ps._listening_coroutine.cancel()
            except Exception:
                pass
        del player, opponent

    total_steps = sum(len(t) for t in all_trajectories)

    return all_trajectories, total_wins, total_losses, total_ties, total_steps, opp_names


def main():
    p = argparse.ArgumentParser(description="PPO RL Training for Pokemon AI")
    p.add_argument("--init-from", type=str, default=None, help="BC checkpoint to warm-start from")
    p.add_argument("--resume", type=str, default=None, help="RL checkpoint to resume from")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--format", default="gen9ou")

    # PPO hyperparameters
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--kl-coef", type=float, default=0.01,
                   help="KL penalty coefficient against BC reference (0=off)")

    # Collection
    p.add_argument("--games-per-iter", type=int, default=100, help="Games to play per iteration")
    p.add_argument("--n-iters", type=int, default=200, help="Total training iterations")
    p.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent battles")
    p.add_argument("--servers", type=str, default="",
                   help="Comma-separated Showdown ports for multi-server parallelism "
                        "(e.g. '8000,8001,8002'). Empty = single default server.")
    p.add_argument("--temperature", type=float, default=1.0, help="Action sampling temperature")
    p.add_argument("--temp-decay", type=float, default=0.995, help="Temperature decay per iter")
    p.add_argument("--temp-min", type=float, default=0.3, help="Min temperature")

    # LR schedule
    p.add_argument("--lr-schedule", default="cosine", choices=["none", "cosine"],
                   help="Learning rate schedule")

    # Evaluation
    p.add_argument("--eval-interval", type=int, default=10, help="Evaluate every N iters")
    p.add_argument("--eval-games", type=int, default=100, help="Games per eval opponent")

    # Checkpointing
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--out-dir", default="checkpoints/rl")

    # AMP
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--no-amp", dest="amp", action="store_false")

    # === v3 feature flags ===
    # Curriculum learning
    p.add_argument("--curriculum", action="store_true", default=False,
                   help="Enable curriculum learning (start with easy bots, unlock harder ones)")
    p.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    p.add_argument("--curriculum-tier", type=int, default=0,
                   help="Starting curriculum tier (0=easiest)")
    p.add_argument("--adaptive-weights", action="store_true", default=False,
                   help="Dynamically focus training on struggling opponents")
    p.add_argument("--no-adaptive-weights", dest="adaptive_weights",
                   action="store_false")
    p.add_argument("--promote-temp-bump", type=float, default=0.15,
                   help="Temperature increase on tier promotion (0=no bump)")
    p.add_argument("--promote-temp-cap", type=float, default=0.8,
                   help="Max temperature after promotion bump")
    p.add_argument("--promote-lr-restart", action="store_true", default=True,
                   help="Warm-restart LR cosine schedule on tier promotion")
    p.add_argument("--no-promote-lr-restart", dest="promote_lr_restart",
                   action="store_false")

    # Dense reward shaping (unified with IQL — see rewards.py)
    p.add_argument("--dense-rewards", action="store_true", default=True,
                   help="Enable dense reward shaping (KO delta + HP delta)")
    p.add_argument("--no-dense-rewards", dest="dense_rewards", action="store_false")
    p.add_argument("--ko-coef", type=float, default=0.05,
                   help="Reward per net KO delta (matched to IQL)")
    p.add_argument("--hp-coef", type=float, default=0.02,
                   help="Reward per unit HP advantage delta (matched to IQL)")
    p.add_argument("--reward-clip", type=float, default=2.0,
                   help="Clamp per-step shaped reward")
    # Legacy aliases (backward compat with old configs/scripts)
    p.add_argument("--ko-bonus", type=float, default=None,
                   help="DEPRECATED: use --ko-coef instead")
    p.add_argument("--faint-penalty", type=float, default=None,
                   help="DEPRECATED: use --ko-coef instead")

    # Self-play
    p.add_argument("--self-play", action="store_true", default=True,
                   help="Include self-play opponent (snapshot of current model)")
    p.add_argument("--no-self-play", dest="self_play", action="store_false")
    p.add_argument("--self-play-interval", type=int, default=10,
                   help="Update self-play snapshot every N iters")
    p.add_argument("--self-play-weight", type=float, default=4.0,
                   help="Sampling weight for self-play opponent (total across pool)")
    p.add_argument("--bc-opponent-weight", type=float, default=0.5,
                   help="Sampling weight for BC reference as opponent (0=disable)")
    p.add_argument("--snapshot-pool-size", type=int, default=5,
                   help="Number of recent self-play snapshots to keep in opponent pool")
    p.add_argument("--snapshot-hall-of-fame", type=int, default=3,
                   help="Number of best-performing snapshots to keep permanently")
    p.add_argument("--selfplay-weights", action="store_true", default=True,
                   help="Use self-play-focused opponent weights (OPPONENT_WEIGHTS_SELFPLAY)")
    p.add_argument("--no-selfplay-weights", dest="selfplay_weights", action="store_false",
                   help="Use flat uniform weights for all bots")
    p.add_argument("--opponent-device", type=str, default="auto",
                   help="Device for opponent model inference (auto/cpu/cuda). Auto uses same as --device.")
    p.add_argument("--history-dirs", type=str, default="",
                   help="Comma-separated list of additional run dirs for historical sampling (one-time seeding)")
    p.add_argument("--direct", action="store_true",
                   help="Use direct BattleStream transport (no websockets/Docker)")

    args = p.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    # Resolve opponent device
    if args.opponent_device == "auto":
        args.opponent_device = str(device)
    print(f"Device: {device} (opponent: {args.opponent_device})")

    # Load model
    if args.resume:
        print(f"Resuming from RL checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        pcfg = ckpt.get("policy_cfg", {})
        cfg = PolicyConfig(**pcfg)
        model = BattlePolicy(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        start_iter = ckpt.get("step", 0)
    elif args.init_from:
        print(f"Warm-starting from BC checkpoint: {args.init_from}")
        model, cfg = load_bc_checkpoint(args.init_from, device)
        start_iter = 0
    else:
        print("Training from scratch (no checkpoint)")
        cfg = PolicyConfig(obs_dim=1480, action_dim=9, use_lstm=True,
                           lstm_hidden=256, mlp_hidden=256)
        model = BattlePolicy(cfg).to(device)
        start_iter = 0

    # Frozen BC reference model for KL penalty (prevents catastrophic forgetting)
    ref_model = None
    bc_ckpt_path = args.init_from  # Keep path for BC opponent
    if args.resume and not args.init_from:
        print("[WARN] Resuming without --init-from! BC opponent and KL penalty will be DISABLED. "
              "Add --init-from <BC_checkpoint> for proper self-play training.", flush=True)
    if args.init_from and args.kl_coef > 0:
        print(f"Creating frozen BC reference for KL penalty (coef={args.kl_coef})")
        ref_model, _ = load_bc_checkpoint(args.init_from, device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.resume:
        opt_sd = ckpt.get("optimizer")
        if opt_sd:
            optimizer.load_state_dict(opt_sd)

    # LR schedule
    total_iters = start_iter + args.n_iters  # total iters including prior training
    lr_scheduler = None
    if args.lr_schedule == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_iters, eta_min=args.lr * 0.05
        )
        # Restore scheduler state from checkpoint, or fast-forward if not saved
        if args.resume and start_iter > 0:
            sched_sd = ckpt.get("lr_scheduler")
            if sched_sd is not None:
                lr_scheduler.load_state_dict(sched_sd)
                # Sync optimizer LR with scheduler state (load_state_dict
                # restores scheduler internals but not the optimizer's LR)
                last_lrs = lr_scheduler.get_last_lr()
                for pg, lr_val in zip(optimizer.param_groups, last_lrs):
                    pg["lr"] = lr_val
                print(f"LR scheduler restored from checkpoint "
                      f"(lr={optimizer.param_groups[0]['lr']:.2e})")
            else:
                for _ in range(start_iter):
                    lr_scheduler.step()
                print(f"LR scheduler fast-forwarded to step {start_iter} "
                      f"(lr={optimizer.param_groups[0]['lr']:.2e})")

    # Output
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"ppo_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir / "tb"))

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    server = make_server()
    server_pool = make_server_pool(args.servers) if args.servers else [server]
    if len(server_pool) > 1:
        print(f"Multi-server: {len(server_pool)} Showdown instances "
              f"(ports: {args.servers})")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Restore temperature to where it should be after start_iter decays
    temperature = max(args.temp_min,
                      args.temperature * (args.temp_decay ** start_iter))
    if args.resume and start_iter > 0:
        print(f"Temperature restored to {temperature:.4f} (after {start_iter} decays)")

    # Restore best eval win rate from checkpoint metrics
    best_eval_wr = 0.0
    if args.resume and start_iter > 0:
        prev_metrics = ckpt.get("metrics", {})
        best_eval_wr = prev_metrics.get("best_eval_wr", 0.0)
        if best_eval_wr > 0:
            print(f"Best eval win rate restored: {best_eval_wr:.1%}")

    # === v3: Curriculum setup ===
    curriculum_tier = args.curriculum_tier
    if args.resume and start_iter > 0:
        prev_tier = ckpt.get("metrics", {}).get("curriculum_tier", 0)
        curriculum_tier = prev_tier
    if args.curriculum:
        tier_info = CURRICULUM_TIERS[min(curriculum_tier, len(CURRICULUM_TIERS) - 1)]
        print(f"Curriculum: {tier_info['name']} (tier {curriculum_tier})")
    else:
        print("Curriculum: OFF (flat opponent weights)")

    # === Dense reward setup (unified with IQL) ===
    # Handle legacy --ko-bonus/--faint-penalty args
    ko_coef = args.ko_coef
    hp_coef = args.hp_coef
    if args.ko_bonus is not None:
        print(f"[WARN] --ko-bonus is deprecated, mapping to --ko-coef={args.ko_bonus}")
        ko_coef = args.ko_bonus
    if not args.dense_rewards:
        ko_coef = 0.0
        hp_coef = 0.0
    reward_shaper_template = RewardShaper(
        ko_coef=ko_coef,
        hp_coef=hp_coef,
        clip_abs=args.reward_clip,
    )
    if args.dense_rewards:
        print(f"Dense rewards: ON (ko_coef={ko_coef}, hp_coef={hp_coef}, "
              f"clip={args.reward_clip})")
    else:
        print("Dense rewards: OFF (terminal reward only)")

    # === v6: Self-play setup with snapshot pool + hall of fame + lineage tracking ===
    self_play_ckpt_path = None
    snapshot_pool: List[str] = []       # Recent snapshots (FIFO, max 2)
    hall_of_fame: List[Tuple[float, str]] = []  # (eval_wr, path) — best-ever snapshots
    history_dirs: List[str] = [str(run_dir)]   # All run dirs in this training lineage
    if args.self_play:
        # Add resume checkpoint's parent dir to lineage
        if args.resume:
            resume_parent = str(Path(args.resume).parent)
            if resume_parent not in history_dirs:
                history_dirs.append(resume_parent)

        # Add CLI-specified history dirs (one-time seeding for first resume)
        if args.history_dirs:
            for d in args.history_dirs.split(","):
                d = d.strip()
                if d and d not in history_dirs and Path(d).exists():
                    history_dirs.append(d)

        # Save initial self-play snapshot
        sp_path = run_dir / "snapshot_init.pt"
        save_rl_checkpoint(model, optimizer, cfg, start_iter,
                          {"self_play_init": True}, sp_path)
        self_play_ckpt_path = str(sp_path)
        snapshot_pool.append(str(sp_path))

        # Restore pool + hall of fame from checkpoint if resuming
        if args.resume:
            ckpt_metrics = ckpt.get("metrics", {})
            saved_pool = ckpt_metrics.get("snapshot_pool", [])
            saved_hof = ckpt_metrics.get("hall_of_fame", [])
            # Only restore paths that still exist on disk
            for p in saved_pool:
                if p not in snapshot_pool and Path(p).exists():
                    snapshot_pool.append(p)
            for wr, p in saved_hof:
                if Path(p).exists():
                    hall_of_fame.append((wr, p))

            # Fallback: if no HoF in checkpoint, scan for hof_*.pt files
            # in the resume checkpoint's directory AND out-dir for orphaned HoF files
            if not hall_of_fame:
                import glob as _glob
                resume_dir = Path(args.resume).parent
                scan_dirs = {str(resume_dir)}
                if args.out_dir:
                    for d in Path(args.out_dir).iterdir() if Path(args.out_dir).exists() else []:
                        if d.is_dir():
                            scan_dirs.add(str(d))
                for scan_dir in scan_dirs:
                    for hof_file in sorted(_glob.glob(str(Path(scan_dir) / "hof_*.pt"))):
                        # Parse WR from filename: hof_iterXXXX_wrY.YYY.pt
                        try:
                            wr_str = Path(hof_file).stem.split("_wr")[1]
                            wr = float(wr_str)
                            hall_of_fame.append((wr, hof_file))
                        except (IndexError, ValueError):
                            continue
                if hall_of_fame:
                    # Keep only top N
                    hall_of_fame.sort(key=lambda x: x[0], reverse=True)
                    hall_of_fame = hall_of_fame[:args.snapshot_hall_of_fame]
                    hof_wrs = [f"{wr:.1%}" for wr, _ in hall_of_fame]
                    print(f"  Discovered {len(hall_of_fame)} hall-of-fame from disk: {', '.join(hof_wrs)}")

            # Restore lineage dirs
            saved_dirs = ckpt_metrics.get("history_dirs", [])
            for d in saved_dirs:
                if d not in history_dirs and Path(d).exists():
                    history_dirs.append(d)

            if saved_pool or hall_of_fame:
                print(f"  Restored pool: {len(snapshot_pool)} snapshots + "
                      f"{len(hall_of_fame)} hall-of-fame, "
                      f"lineage: {len(history_dirs)} dirs")

        print(f"Self-play: ON (weight={args.self_play_weight}, "
              f"update every {args.self_play_interval} iters, "
              f"pool_size={args.snapshot_pool_size}, "
              f"hall_of_fame={args.snapshot_hall_of_fame})")
    else:
        print("Self-play: OFF")

    # Track latest eval results for adaptive weighting
    latest_eval_results: Optional[Dict[str, float]] = None

    # Use self-play focused weights if enabled
    if args.selfplay_weights and args.self_play and not args.curriculum:
        print("Opponent weights: SELFPLAY (smart bots 1.0, easy bots 0.3, "
              f"self-play {args.self_play_weight}, BC 2.0)")
    elif not args.curriculum:
        print("Opponent weights: FLAT (all bots equal)")

    print(f"\nStarting PPO training: {args.n_iters} iterations, "
          f"{args.games_per_iter} games/iter")
    print(f"Model: obs_dim={cfg.obs_dim}, transformer={cfg.use_transformer}, "
          f"hidden={cfg.mlp_hidden}")
    print(f"Output: {run_dir}\n")

    historical_picks = []  # initialized empty, populated on first snapshot interval

    for it in range(start_iter, start_iter + args.n_iters):
        t0 = time.time()

        # === v5: Get opponent weights ===
        if args.curriculum:
            opp_weights = get_curriculum_weights(
                curriculum_tier,
                eval_results=latest_eval_results,
                adaptive=args.adaptive_weights,
            )
            if latest_eval_results is not None and args.adaptive_weights:
                w_str = ", ".join(f"{k}={v:.2f}" for k, v in opp_weights.items())
                print(f"  Adaptive weights: {w_str}")
        elif args.selfplay_weights and args.self_play:
            opp_weights = OPPONENT_WEIGHTS_SELFPLAY
        else:
            opp_weights = OPPONENT_WEIGHTS_FLAT

        # === Collect trajectories ===
        model.eval()
        # Build full self-play pool: 2 recent + 3 random historical + 3 hall of fame
        full_sp_pool = list(snapshot_pool) + historical_picks + [p for _, p in hall_of_fame]
        # Deduplicate while preserving order, skip missing files
        seen = set()
        deduped_pool = []
        for p in full_sp_pool:
            if p not in seen and Path(p).exists():
                seen.add(p)
                deduped_pool.append(p)

        trajectories, wins, losses, ties, total_steps, opp_names = loop.run_until_complete(
            collect_trajectories(
                model, device, server,
                n_games=args.games_per_iter,
                battle_format=args.format,
                ctx_extra_dim=int(cfg.ctx_extra_dim or 0),
                step_type_bins=int(cfg.step_type_bins or 0),
                temperature=temperature,
                bc_checkpoint=bc_ckpt_path,
                max_concurrent=args.max_concurrent,
                opponent_weights=opp_weights,
                self_play_checkpoint=self_play_ckpt_path,
                self_play_weight=args.self_play_weight if args.self_play else 0.0,
                self_play_pool=deduped_pool if deduped_pool else None,
                bc_opponent_weight=args.bc_opponent_weight,
                opponent_device=args.opponent_device,
                reward_shaper=reward_shaper_template,
                server_pool=server_pool,
                use_direct=args.direct,
            )
        )

        total_games = wins + losses + ties
        win_rate = wins / max(1, total_games)
        collect_time = time.time() - t0

        tier_str = f" T{curriculum_tier}" if args.curriculum else ""
        # Opponent distribution summary
        from collections import Counter
        _opp_dist = Counter(opp_names)
        _opp_str = " ".join(f"{k.replace('SelfPlay_','SP')}={v}" for k, v in sorted(_opp_dist.items()))
        print(f"Iter {it+1}/{start_iter + args.n_iters}: "
              f"W/L/T={wins}/{losses}/{ties} ({win_rate:.1%}), "
              f"{total_steps} steps, {len(trajectories)} episodes, "
              f"temp={temperature:.3f}{tier_str}, collect={collect_time:.1f}s "
              f"[{_opp_str}]", end="")

        # === Build PPO episodes (preserves LSTM context) ===
        episodes = build_ppo_episodes(
            trajectories,
            gamma=args.gamma,
            lam=args.lam,
            ctx_extra_dim=int(cfg.ctx_extra_dim or 0),
        )

        # Free trajectory data now that it's been converted to tensors
        del trajectories

        if episodes is None:
            print(" — no data, skipping update")
            continue

        # === PPO update (episode-sequential for LSTM) ===
        model.train()
        t1 = time.time()
        loss_info = ppo_update(
            model, optimizer, episodes, device,
            epochs=args.ppo_epochs,
            clip_eps=args.clip_eps,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            use_amp=args.amp and device.type == "cuda",
            ref_model=ref_model,
            kl_coef=args.kl_coef,
        )

        # Free GPU tensors from PPO episodes + force garbage collection
        del episodes
        if device.type == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        update_time = time.time() - t1
        kl_str = f" kl={loss_info.get('kl',0):.4f}" if args.kl_coef > 0 else ""
        print(f", update={update_time:.1f}s, "
              f"pi={loss_info.get('pi',0):.4f} v={loss_info.get('v',0):.4f} "
              f"ent={loss_info.get('ent',0):.4f}{kl_str}")

        # === Logging ===
        writer.add_scalar("train/win_rate", win_rate, it)
        writer.add_scalar("train/total_steps", total_steps, it)
        writer.add_scalar("train/temperature", temperature, it)
        if args.curriculum:
            writer.add_scalar("train/curriculum_tier", curriculum_tier, it)
        for k, v in loss_info.items():
            writer.add_scalar(f"loss/{k}", v, it)

        # === Decay temperature ===
        temperature = max(args.temp_min, temperature * args.temp_decay)

        # === LR schedule step ===
        if lr_scheduler is not None:
            lr_scheduler.step()
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], it)

        # === v6: Update self-play snapshot + uniform historical sampling ===
        if args.self_play and (it + 1) % args.self_play_interval == 0:
            sp_path = run_dir / f"snapshot_iter{it+1:04d}.pt"
            save_rl_checkpoint(model, optimizer, cfg, it,
                              {"self_play_update": True}, sp_path)
            self_play_ckpt_path = str(sp_path)

            # FIFO: keep only 2 most recent snapshots, delete older
            snapshot_pool.append(str(sp_path))
            while len(snapshot_pool) > 2:
                old = snapshot_pool.pop(0)
                hof_paths = {p for _, p in hall_of_fame}
                if old not in hof_paths and Path(old).exists():
                    Path(old).unlink()

            # Scan known lineage dirs for iter_*.pt + snapshot_*.pt checkpoints
            import glob as _glob
            all_history = set()
            for d in history_dirs:
                if Path(d).exists():
                    for f in _glob.glob(str(Path(d) / "iter_*.pt")):
                        all_history.add(f)
                    for f in _glob.glob(str(Path(d) / "snapshot_*.pt")):
                        all_history.add(f)
            # Exclude files already in pool or HoF
            pool_set = set(snapshot_pool) | {p for _, p in hall_of_fame}
            all_history = [p for p in all_history if Path(p).exists() and p not in pool_set]

            # Sample 3 random from history (uniform)
            n_hist = min(3, len(all_history))
            historical_picks = random.sample(all_history, n_hist) if all_history else []

            pool_size = len(snapshot_pool) + n_hist + len(hall_of_fame)
            hist_names = [Path(p).stem for p in historical_picks]
            print(f"  Self-play: snapshot saved, pool={len(snapshot_pool)} recent + "
                  f"{n_hist} historical [{', '.join(hist_names)}] + "
                  f"{len(hall_of_fame)} hall-of-fame = {pool_size} total")

        # === Evaluation ===
        if (it + 1) % args.eval_interval == 0:
            model.eval()
            # Eval against all registered bots
            eval_bots = dict(OPPONENT_BOTS)
            eval_results = {}
            for ei, (bot_name, bot_cls) in enumerate(eval_bots.items()):
                eval_srv = server_pool[ei % len(server_pool)]
                try:
                    wr = loop.run_until_complete(
                        evaluate_vs_bot(
                            model, device, bot_cls, eval_srv,
                            n_battles=args.eval_games,
                            battle_format=args.format,
                            ctx_extra_dim=int(cfg.ctx_extra_dim or 0),
                            step_type_bins=int(cfg.step_type_bins or 0),
                            max_concurrent=args.max_concurrent,
                            use_direct=args.direct,
                        )
                    )
                    eval_results[bot_name] = wr
                    writer.add_scalar(f"eval/{bot_name}", wr, it)
                except Exception as e:
                    print(f"  Eval vs {bot_name} failed: {e}")
                    eval_results[bot_name] = -1

            # Weighted avg: smart bots 1.0, easy bots 0.3 (matches opponent weights)
            _SMART_BOTS = {"SimpleHeuristics", "SmartDamage", "Tactical", "Strategic"}
            _eval_w_sum, _eval_v_sum = 0.0, 0.0
            for k, v in eval_results.items():
                if v < 0:
                    continue
                w = 1.0 if k in _SMART_BOTS else 0.3
                _eval_w_sum += w
                _eval_v_sum += w * v
            avg_wr = _eval_v_sum / max(_eval_w_sum, 1e-6)
            smart_wr = np.mean([v for k, v in eval_results.items()
                                if k in _SMART_BOTS and v >= 0]) if any(
                                    k in _SMART_BOTS for k in eval_results) else 0.0
            writer.add_scalar("eval/avg_win_rate", avg_wr, it)
            writer.add_scalar("eval/smart_avg_wr", smart_wr, it)
            print(f"  EVAL: " + ", ".join(f"{k}={v:.1%}" for k, v in eval_results.items())
                  + f"  smart_avg={smart_wr:.1%} weighted_avg={avg_wr:.1%}")

            # Store for adaptive weighting next iteration
            latest_eval_results = eval_results

            # Free VRAM after eval round
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if avg_wr > best_eval_wr:
                best_eval_wr = avg_wr
                save_rl_checkpoint(
                    model, optimizer, cfg, it,
                    {"best_eval_wr": best_eval_wr, "curriculum_tier": curriculum_tier,
                     **eval_results},
                    run_dir / "best.pt",
                    lr_scheduler=lr_scheduler,
                )
                print(f"  New best model saved (avg wr={best_eval_wr:.1%})")

            # === Hall of fame: keep top N performers across all evals ===
            if args.self_play:
                hall_of_fame.sort(key=lambda x: x[0])
                worst_hof_wr = hall_of_fame[0][0] if hall_of_fame else -1.0
                should_add = (len(hall_of_fame) < args.snapshot_hall_of_fame
                              or avg_wr > worst_hof_wr)
                if should_add:
                    hof_path = run_dir / f"hof_iter{it+1:04d}_wr{avg_wr:.3f}.pt"
                    save_rl_checkpoint(model, optimizer, cfg, it,
                                      {"hall_of_fame": True, "eval_wr": avg_wr},
                                      hof_path)
                    hall_of_fame.append((avg_wr, str(hof_path)))
                    # Evict weakest if over capacity
                    if len(hall_of_fame) > args.snapshot_hall_of_fame:
                        hall_of_fame.sort(key=lambda x: x[0])
                        removed_wr, removed_path = hall_of_fame.pop(0)
                        if removed_path not in snapshot_pool and Path(removed_path).exists():
                            Path(removed_path).unlink()
                    hof_wrs = [f"{wr:.1%}" for wr, _ in hall_of_fame]
                    print(f"  Hall of fame updated: {len(hall_of_fame)} entries ({', '.join(hof_wrs)})")

            # === v3: Curriculum promotion check ===
            if args.curriculum:
                new_tier, promoted, reason = check_curriculum_promotion(
                    curriculum_tier, eval_results)
                if promoted:
                    curriculum_tier = new_tier
                    new_info = CURRICULUM_TIERS[curriculum_tier]
                    print(f"  CURRICULUM: Promoted to {new_info['name']} "
                          f"(tier {curriculum_tier}, {reason})")

                    # Bump temperature for exploration against new opponents
                    if args.promote_temp_bump > 0:
                        old_temp = temperature
                        temperature = min(temperature + args.promote_temp_bump,
                                         args.promote_temp_cap)
                        print(f"  CURRICULUM: Temperature bumped "
                              f"{old_temp:.3f} -> {temperature:.3f}")

                    # Warm-restart LR schedule for fresh learning capacity
                    if args.promote_lr_restart and args.lr_schedule == "cosine":
                        remaining_iters = (start_iter + args.n_iters) - it
                        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, T_max=max(remaining_iters, 10),
                            eta_min=args.lr * 0.05
                        )
                        print(f"  CURRICULUM: LR warm-restarted "
                              f"(T_max={remaining_iters}, "
                              f"lr={optimizer.param_groups[0]['lr']:.2e})")
                else:
                    print(f"  CURRICULUM: Staying at tier {curriculum_tier} "
                          f"({reason})")

        # === Save checkpoint ===
        if (it + 1) % args.save_interval == 0:
            save_rl_checkpoint(
                model, optimizer, cfg, it,
                {"win_rate": win_rate, "best_eval_wr": best_eval_wr,
                 "temperature": temperature, "curriculum_tier": curriculum_tier,
                 "snapshot_pool": snapshot_pool,
                 "hall_of_fame": hall_of_fame,
                 "history_dirs": history_dirs,
                 **loss_info},
                run_dir / f"iter_{it+1:04d}.pt",
                lr_scheduler=lr_scheduler,
            )

    # Final save
    save_rl_checkpoint(
        model, optimizer, cfg, start_iter + args.n_iters,
        {"final": True, "best_eval_wr": best_eval_wr,
         "temperature": temperature, "curriculum_tier": curriculum_tier},
        run_dir / "final.pt",
        lr_scheduler=lr_scheduler,
    )
    print(f"\nTraining complete. Best eval win rate: {best_eval_wr:.1%}")
    print(f"Checkpoints saved to: {run_dir}")
    writer.close()


if __name__ == "__main__":
    main()
    if "--direct" in sys.argv:
        import os
        os._exit(0)
