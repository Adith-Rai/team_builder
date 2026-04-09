#!/usr/bin/env python
# ppo.py — PPO utilities (trajectory, GAE, PPO update, checkpoint I/O)
#
# Core PPO functions used by train_rl.py:
#   - features.make_features() for structured entity features
#   - PokeTransformer with temporal history buffer
#   - Distributional value head (51-bin two-hot cross-entropy)
#   - All v7 self-play infrastructure: snapshot pool, HoF, historical sampling
#   - Memory management: reset_battles, PSClient cancel, gc.collect, empty_cache
#   - Speed optimization: batch spatial during PPO update where possible
#
# Hyperparameters (V8_PLAN.md + ps-ppo + Metamon research):
#   gamma=0.9999, lambda=0.8, entropy=0.02, lr=1e-4, kl=0.05
#   Distributional value: 51 bins, [-1.6, 1.6], two-hot CE loss
#   Reward: terminal ±1.0, ko_coef=0.05, hp_coef=0.02
#
# Usage:
#   python -u rl_train_v8.py \
#     --init-from data/models/bc/v8_bc_human_metamon/best.pt \
#     --device cuda --servers 9000 \
#     --games-per-iter 100 --max-concurrent 10 \
#     --self-play --n-iters 500

from __future__ import annotations
import argparse
import asyncio
import gc
import glob
import json
import math
import os
import random
import signal
import sys
import time
import traceback
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from poke_env.player import Player
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.player.baselines import RandomPlayer as PokeRandomPlayer

from features import make_features
from model import PokeTransformer, PokeTransformerConfig, add_model_args, config_from_args
from battle_agent import BattleAgent
from rewards import RewardShaper
from teams_ou import random_teambuilder, random_pool_teambuilder
from policy_rulebots import (
    GreedySEPlayer, HazardSensePlayer, SwitchAwareEscapePlayer, SetupThenSweepPlayer,
)
from policy_smartbots import SmartDamagePlayer, TacticalPlayer, StrategicPlayer


# =============================
# Constants
# =============================

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

OPPONENT_WEIGHTS_SELFPLAY = {
    "SimpleHeuristics": 1.0, "SmartDamage": 1.0,
    "Tactical": 1.0, "Strategic": 1.0,
    "MaxBasePower": 0.3, "GreedySE": 0.3,
    "HazardSense": 0.3, "SwitchAwareEscape": 0.3, "SetupThenSweep": 0.3,
}

SMART_BOTS = {"SimpleHeuristics", "SmartDamage", "Tactical", "Strategic"}

_pid_tag = os.getpid() % 10000
_collect_round = 0


# =============================
# V8 Trajectory
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
# V8 RL Player
# =============================

class V8RLPlayer(Player):
    """PPO player using PokeTransformer v8."""

    def __init__(self, model: PokeTransformer, device: torch.device,
                 reward_shaper_cfg: Optional[dict] = None,
                 temperature: float = 1.0, fp16: bool = False,
                 turn_cap: int = 300, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.device = device
        self._rs_cfg = reward_shaper_cfg or {"ko_coef": 0.05, "hp_coef": 0.02, "clip_abs": 2.0}
        self.temperature = temperature
        self.fp16 = fp16 and device.type == "cuda"
        self.turn_cap = turn_cap
        self._history: Dict[str, torch.Tensor] = {}
        self._trajectories: Dict[str, Trajectory] = {}
        self._reward_shapers: Dict[str, RewardShaper] = {}  # per-battle!
        self.completed_trajectories: List[Trajectory] = []

    def _get_shaper(self, btag: str) -> RewardShaper:
        if btag not in self._reward_shapers:
            self._reward_shapers[btag] = RewardShaper(**self._rs_cfg)
        return self._reward_shapers[btag]

    def _get_traj(self, btag: str) -> Trajectory:
        if btag not in self._trajectories:
            self._trajectories[btag] = Trajectory()
        return self._trajectories[btag]

    def _build_turn_batch(self, feat: dict) -> dict:
        """Convert features_v8 output to model batch dict on self.device.
        Optimized: build all numpy arrays first, then transfer to GPU in bulk."""
        dev = self.device

        def _pi(p):
            i = p["ids"]
            return [i["species"], i["item"], i["ability"]]
        def _pb(p):
            b = p["banks"]
            return [b["hp_pct"], b["level"], b["weight"], b["height"],
                    b["stat_hp"], b["stat_atk"], b["stat_def"],
                    b["stat_spa"], b["stat_spd"], b["stat_spe"]]
        def _pmi(p):
            i = p["ids"]
            return [i["move0"], i["move1"], i["move2"], i["move3"]]
        def _pmc(p):
            from features import extract_move_cont
            return extract_move_cont(p["continuous"])

        our, opp = feat["our_pokemon"], feat["opp_pokemon"]

        # Build all arrays on CPU as numpy, then one bulk transfer
        int_arrays = {
            "our_pokemon_ids": np.array([[_pi(p) for p in our]], dtype=np.int64),
            "our_pokemon_banks": np.array([[_pb(p) for p in our]], dtype=np.int64),
            "our_pokemon_move_ids": np.array([[_pmi(p) for p in our]], dtype=np.int64),
            "opp_pokemon_ids": np.array([[_pi(p) for p in opp]], dtype=np.int64),
            "opp_pokemon_banks": np.array([[_pb(p) for p in opp]], dtype=np.int64),
            "opp_pokemon_move_ids": np.array([[_pmi(p) for p in opp]], dtype=np.int64),
        }
        float_arrays = {
            "our_pokemon_cont": np.array([[p["continuous"] for p in our]], dtype=np.float32),
            "our_pokemon_move_cont": np.array([[_pmc(p) for p in our]], dtype=np.float32),
            "opp_pokemon_cont": np.array([[p["continuous"] for p in opp]], dtype=np.float32),
            "opp_pokemon_move_cont": np.array([[_pmc(p) for p in opp]], dtype=np.float32),
            "field_cont": np.array([feat["field"]["continuous"]], dtype=np.float32),
            "transition_cont": np.array([feat["transition"]["continuous"]], dtype=np.float32),
            "legal_mask": feat["legal_mask"].reshape(1, 9).astype(np.float32),
        }

        # Active moves
        mids, mbp, mac, mpp, mpr, mco = [], [], [], [], [], []
        for m in feat["active_moves"]:
            if m is None:
                mids.append(0); mbp.append(0); mac.append(0); mpp.append(0); mpr.append(6)
                from features import MOVE_SLOT_CONT_DIM
                mco.append([0.0]*MOVE_SLOT_CONT_DIM)
            else:
                mids.append(m["move_id"]); mbp.append(m["bp_int"]); mac.append(m["acc_int"])
                mpp.append(m["pp_int"]); mpr.append(m["priority_int"]); mco.append(m["continuous"])
        int_arrays["active_move_ids"] = np.array([mids], dtype=np.int64)
        float_arrays["active_move_cont"] = np.array([mco], dtype=np.float32)

        # Switches
        sids, sco = [], []
        for s in feat["switch_slots"]:
            if s is None:
                from features import SWITCH_SLOT_CONT_DIM
                sids.append(0); sco.append([0.0]*SWITCH_SLOT_CONT_DIM)
            else:
                sids.append(s["species_id"]); sco.append(s["continuous"])
        int_arrays["switch_ids"] = np.array([sids], dtype=np.int64)
        float_arrays["switch_cont"] = np.array([sco], dtype=np.float32)

        # Bulk transfer: numpy -> pinned CPU tensor -> GPU (faster than individual torch.tensor(device=cuda))
        batch = {}
        for k, arr in int_arrays.items():
            batch[k] = torch.from_numpy(arr).to(dev, non_blocking=True)
        for k, arr in float_arrays.items():
            batch[k] = torch.from_numpy(arr).to(dev, non_blocking=True)

        # Small dicts (4 scalars each) — overhead is minimal
        fb = feat["field"]["banks"]
        batch["field_banks"] = {k: torch.tensor([fb[k]], dtype=torch.long, device=dev) for k in fb}
        ti = feat["transition"]["ids"]
        batch["transition_ids"] = {k: torch.tensor([ti[k]], dtype=torch.long, device=dev) for k in ti}
        batch["active_move_banks"] = {
            "bp": torch.tensor([mbp], dtype=torch.long, device=dev),
            "acc": torch.tensor([mac], dtype=torch.long, device=dev),
            "pp": torch.tensor([mpp], dtype=torch.long, device=dev),
            "prio": torch.tensor([mpr], dtype=torch.long, device=dev),
        }
        return batch

    def _to_cpu(self, batch: dict) -> dict:
        """Move batch dict to CPU for trajectory storage."""
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.cpu()
            elif isinstance(v, dict):
                out[k] = {kk: vv.cpu() if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
            else:
                out[k] = v
        return out

    def choose_move(self, battle):
        btag = battle.battle_tag
        traj = self._get_traj(btag)
        shaper = self._get_shaper(btag)

        # Turn cap: forfeit to prevent OOM on extremely long battles
        if len(traj) >= self.turn_cap:
            print(f"  [TURN CAP] {btag} hit {self.turn_cap} turns, forfeiting", flush=True)
            try:
                self.forfeit_battle(battle)
            except Exception:
                pass
            return self.choose_random_move(battle)

        # Dense reward for previous step
        if len(traj.rewards) > 0:
            traj.rewards[-1] += shaper.step(battle)

        feat = make_features(battle)
        batch = self._build_turn_batch(feat)
        history = self._history.get(btag)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.fp16):
            out = self.model(batch, history=history)

        # Update temporal history (always float32 for stable accumulation)
        summary = out["summary"].float().unsqueeze(1)
        if history is None:
            self._history[btag] = summary
        else:
            self._history[btag] = torch.cat([history, summary], dim=1)
            if self._history[btag].shape[1] > 200:
                self._history[btag] = self._history[btag][:, -200:]

        # Sample action with temperature
        logits = out["action_logits"][0]
        if self.temperature != 1.0:
            scaled_logits = logits / self.temperature
        else:
            scaled_logits = logits
        probs = F.softmax(scaled_logits, dim=-1)
        action_idx = torch.multinomial(probs, 1).item()

        # Store UNSCALED log_prob (critical for PPO importance ratio)
        log_prob = F.log_softmax(logits, dim=-1)[action_idx].item()
        value = out["value"][0].item()

        # Store in trajectory (CPU)
        traj.feat_batches.append(self._to_cpu(batch))
        traj.actions.append(action_idx)
        traj.log_probs.append(log_prob)
        traj.values.append(value)
        traj.rewards.append(0.0)
        traj.dones.append(False)
        traj.action_masks.append(feat["legal_mask"].copy())

        return self._action_to_order(battle, action_idx)

    def _action_to_order(self, battle, idx):
        if idx < 4:
            moves = list(battle.available_moves or [])
            if idx < len(moves):
                return self.create_order(moves[idx])
        else:
            sw = list(battle.available_switches or [])
            si = idx - 4
            if si < len(sw):
                return self.create_order(sw[si])
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])
        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        btag = battle.battle_tag
        traj = self._trajectories.get(btag)
        if traj and len(traj) > 0:
            shaper = self._get_shaper(btag)
            traj.rewards[-1] += shaper.step(battle)
            if battle.won:
                traj.rewards[-1] += 1.0
            elif battle.lost:
                traj.rewards[-1] -= 1.0
            traj.dones[-1] = True
            self.completed_trajectories.append(traj)

        self._trajectories.pop(btag, None)
        self._history.pop(btag, None)
        self._reward_shapers.pop(btag, None)  # clean up per-battle shaper
        super()._battle_finished_callback(battle)


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
    stats = {"pi": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0}
    n = 0
    n_failed = 0  # episodes that raised an exception (CUDA error, OOM, etc.)
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
                summary_buf = torch.zeros(1, 0, cfg.d_model, device=device)

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
                    continue

                # Policy loss
                lp = F.log_softmax(logits_seq, dim=-1)
                new_logp = lp.gather(1, actions.unsqueeze(1)).squeeze(1)
                ratio = torch.exp(new_logp - old_logp)
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
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                accum_count += 1

                if accum_count >= grad_accum:
                    optimizer.step()
                    accum_count = 0

                stats["pi"] += pi_loss.item()
                stats["v"] += v_loss.item()
                stats["ent"] += entropy.item()
                stats["kl"] += abs(approx_kl)
                epoch_kl_sum += abs(approx_kl)
                epoch_kl_count += 1
                n += 1


            except Exception as e:
                print(f"  [ERROR] PPO episode failed (T={T}): {e}", flush=True)
                n_failed += 1
                continue

            # Free GPU memory periodically
            if n % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # Flush remaining accumulated gradients
        if accum_count > 0:
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
    return stats


# =============================
# Battle Collection
# =============================

def _cancel_listener(player):
    """Cancel PSClient websocket listener to prevent zombie coroutines."""
    try:
        ps = getattr(player, "ps_client", None) or getattr(player, "_ps_client", None)
        if ps and hasattr(ps, "_listening_coroutine"):
            ps._listening_coroutine.cancel()
    except Exception:
        pass


async def collect_trajectories(
    model: PokeTransformer, device: torch.device, server,
    n_games: int = 100, battle_format: str = "gen9ou",
    temperature: float = 1.0, bc_checkpoint: str = None,
    max_concurrent: int = 10, opponent_weights: Dict[str, float] = None,
    self_play_pool: List[str] = None, self_play_weight: float = 4.0,
    bc_opponent_weight: float = 0.5, opponent_device: str = "cpu",
    reward_shaper: Optional[RewardShaper] = None,
    server_pool: Optional[list] = None,
    fp16: bool = False,
):
    """Collect trajectories via battles. Returns (trajectories, wins, losses, ties, steps)."""
    global _collect_round
    _collect_round += 1
    rid = _collect_round

    if reward_shaper is None:
        reward_shaper = RewardShaper()

    servers = server_pool if server_pool else [server]
    if opponent_weights is None:
        opponent_weights = OPPONENT_WEIGHTS_SELFPLAY

    # Build opponent pool
    names = list(opponent_weights.keys())
    weights = [opponent_weights[n] for n in names]
    if bc_checkpoint and bc_opponent_weight > 0:
        names.append("BCPolicy")
        weights.append(bc_opponent_weight)

    sp_pool = [p for p in (self_play_pool or []) if p and Path(p).exists()]
    if sp_pool:
        per_w = self_play_weight / len(sp_pool)
        for i, _ in enumerate(sp_pool):
            names.append(f"SelfPlay_{i}")
            weights.append(per_w)

    # Sample opponents
    opp_counts = Counter()
    for _ in range(n_games):
        opp_counts[random.choices(names, weights=weights, k=1)[0]] += 1

    all_trajs = []
    wins, losses, ties = 0, 0, 0
    per_opp_wr = {}  # {opp_name: (wins, total_games)}
    batch_idx = 0

    for opp_name, count in opp_counts.items():
        batch_idx += 1
        srv = servers[(batch_idx - 1) % len(servers)]

        rs_cfg = {"ko_coef": reward_shaper.ko_coef, "hp_coef": reward_shaper.hp_coef,
                  "clip_abs": reward_shaper.clip_abs}
        player = V8RLPlayer(
            model=model, device=device,
            reward_shaper_cfg=rs_cfg,
            temperature=temperature, fp16=fp16,
            battle_format=battle_format,
            team=random_pool_teambuilder(),
            max_concurrent_battles=max_concurrent,
            account_configuration=AccountConfiguration(f"RL{_pid_tag}r{rid}b{batch_idx}", None),
            server_configuration=srv,
        )

        opp_acct = AccountConfiguration(f"Op{_pid_tag}r{rid}b{batch_idx}", None)
        if opp_name == "BCPolicy" or opp_name.startswith("SelfPlay"):
            ckpt = bc_checkpoint if opp_name == "BCPolicy" else (
                sp_pool[int(opp_name.split("_")[1]) % len(sp_pool)] if sp_pool else bc_checkpoint)
            opponent = BattleAgent(
                checkpoint_path=ckpt, device=opponent_device, fp16=fp16,
                battle_format=battle_format, team=random_pool_teambuilder(),
                max_concurrent_battles=max_concurrent,
                account_configuration=opp_acct, server_configuration=srv,
            )
        else:
            cls = OPPONENT_BOTS[opp_name]
            opponent = cls(
                battle_format=battle_format, team=random_pool_teambuilder(),
                max_concurrent_battles=max_concurrent,
                account_configuration=opp_acct, server_configuration=srv,
            )

        try:
            await asyncio.wait_for(
                player.battle_against(opponent, n_battles=count),
                timeout=max(120, 30 * count),  # no cap — 200 games needs ~6000s
            )
        except (asyncio.TimeoutError, TimeoutError):
            print(f"  Batch vs {opp_name} ({count}) timed out")

        opp_w = player.n_won_battles
        opp_l = player.n_lost_battles
        wins += opp_w
        losses += opp_l
        ties += player.n_tied_battles
        all_trajs.extend(player.completed_trajectories)
        per_opp_wr[opp_name] = (opp_w, count)  # track per-opponent wins

        # Memory cleanup (critical for long runs)
        try:
            player.reset_battles()
        except EnvironmentError:
            pass
        try:
            opponent.reset_battles()
        except EnvironmentError:
            pass
        player.completed_trajectories.clear()
        _cancel_listener(player)
        _cancel_listener(opponent)
        del player, opponent

    total_steps = sum(len(t) for t in all_trajs)
    return all_trajs, wins, losses, ties, total_steps, dict(opp_counts), per_opp_wr


# =============================
# Checkpoint I/O
# =============================

def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = PokeTransformerConfig.from_dict(ckpt.get("model_config", {}))
    model = PokeTransformer(cfg).to(device)

    # Handle dim expansion for type effectiveness features (zero-init new columns)
    # move_net.mlp.0.weight: 187 -> 189 (+2: type_eff, opp_threat)
    # switch_mlp.0.weight: 60 -> 61 (+1: defensive_effectiveness)
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
    torch.save(ckpt, path)


# =============================
# CLI
# =============================

def parse_args():
    p = argparse.ArgumentParser(description="PPO self-play training for PokeTransformer v8")
    p.add_argument("--init-from", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--opponent-device", default="cuda",
                   help="Device for opponent models. 'cuda' is faster with batched entity encoding")
    p.add_argument("--servers", default="9000")
    p.add_argument("--format", default="gen9ou")
    p.add_argument("--games-per-iter", type=int, default=100)
    p.add_argument("--max-concurrent", type=int, default=10)
    p.add_argument("--n-iters", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.9999)
    p.add_argument("--lam", type=float, default=0.75)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=5)
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--vf-coef", type=float, default=1.0)
    p.add_argument("--target-kl", type=float, default=0.03,
                   help="KL early stopping threshold (ps-ppo style, stops PPO epochs at 1.5x this)")
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--grad-accum", type=int, default=10,
                   help="Accumulate gradients over N episodes before each optimizer step")
    p.add_argument("--warmup-iters", type=int, default=5,
                   help="Iterations to train only value head before full PPO (ps-ppo style)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temp-decay", type=float, default=0.999)
    p.add_argument("--temp-min", type=float, default=0.5)
    p.add_argument("--ko-coef", type=float, default=0.05)
    p.add_argument("--hp-coef", type=float, default=0.02)
    p.add_argument("--reward-clip", type=float, default=2.0)
    p.add_argument("--fp16", action="store_true",
                   help="Use FP16 inference during collection (faster, GPU only)")
    p.add_argument("--self-play", action="store_true")
    p.add_argument("--self-play-interval", type=int, default=10)
    p.add_argument("--self-play-weight", type=float, default=4.0)
    p.add_argument("--snapshot-pool-size", type=int, default=5)
    p.add_argument("--snapshot-hall-of-fame", type=int, default=3)
    p.add_argument("--bc-opponent-weight", type=float, default=0.5)
    p.add_argument("--eval-interval", type=int, default=20)
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--save-interval", type=int, default=20)
    p.add_argument("--out-dir", default="data/models/rl_v8")
    add_model_args(p)
    return p.parse_args()


# =============================
# Main Training Loop
# =============================

def main():
    args = parse_args()
    device = torch.device(args.device)
    opp_device = args.opponent_device  # default 'cpu' for parallel inference

    # Server pool
    ports = [int(p.strip()) for p in args.servers.split(",")]
    server_pool = [ServerConfiguration(f"ws://127.0.0.1:{p}/showdown/websocket", None) for p in ports]
    server = server_pool[0]

    # Load model
    print(f"Loading BC checkpoint: {args.init_from}")
    model, cfg, _ = load_checkpoint(args.init_from, device)
    print(f"Model: {model.count_parameters():,} params, d_model={cfg.d_model}")

    # No ref_model needed — KL early stopping replaces KL penalty (ps-ppo style)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Output
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / f"ppo_v8_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir / "tb"))

    with open(str(run_dir / "config.json"), "w") as f:
        json.dump({"model_config": cfg.to_dict(), "args": vars(args)}, f, indent=2)

    # Reward shaper
    reward_shaper = RewardShaper(ko_coef=args.ko_coef, hp_coef=args.hp_coef, clip_abs=args.reward_clip)

    # Self-play state
    snapshot_pool: List[str] = []
    hall_of_fame: List[Tuple[float, str]] = []
    historical_picks: List[str] = []
    bc_ckpt = args.init_from

    if args.self_play:
        sp0 = str(run_dir / "snapshot_init.pt")
        save_checkpoint(sp0, model, cfg, optimizer, 0)
        snapshot_pool.append(sp0)

    # Resume
    start_iter = 0
    best_eval_wr = 0.0
    temperature = args.temperature
    if args.resume:
        rckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(rckpt["model_state_dict"])
        optimizer.load_state_dict(rckpt["optimizer_state_dict"])
        start_iter = rckpt.get("iteration", 0) + 1
        m = rckpt.get("metrics", {})
        best_eval_wr = m.get("best_eval_wr", 0.0)
        temperature = m.get("temperature", args.temperature)
        snapshot_pool = m.get("snapshot_pool", snapshot_pool)
        hall_of_fame = m.get("hall_of_fame", hall_of_fame)
        print(f"Resumed from {args.resume}, iter {start_iter}, best_wr={best_eval_wr:.1f}%")

    print(f"\nPPO Training: {args.n_iters} iters, {args.games_per_iter} games/iter")
    print(f"gamma={args.gamma}, lam={args.lam}, ent={args.ent_coef}, target_kl={args.target_kl}")
    print(f"Self-play: {'ON' if args.self_play else 'OFF'}")
    print(f"Value warmup: {args.warmup_iters} iters (freeze backbone+policy, train value only)")
    print(f"Opponent device: {opp_device}")
    print(f"FP16 inference: {'ON' if args.fp16 else 'OFF'}")
    print(f"Output: {run_dir}\n")

    # Async event loop
    loop = asyncio.new_event_loop()

    for it in range(start_iter, start_iter + args.n_iters):
        # Value warmup phase (ps-ppo style): freeze backbone + policy, train only value head
        in_warmup = (it - start_iter) < args.warmup_iters
        if in_warmup:
            # Freeze everything except value_head
            for name, param in model.named_parameters():
                param.requires_grad = "value_head" in name
        elif (it - start_iter) == args.warmup_iters:
            # Unfreeze all at end of warmup
            for param in model.parameters():
                param.requires_grad = True
            print(f"  Value warmup complete, unfreezing all parameters", flush=True)
        t0 = time.time()

        # Build self-play pool
        full_pool = list(set(snapshot_pool + historical_picks + [p for _, p in hall_of_fame]))
        full_pool = [p for p in full_pool if Path(p).exists()]

        # Collect trajectories
        model.eval()
        trajs, wins, losses, ties, steps, opp_dist, per_opp_wr = loop.run_until_complete(
            collect_trajectories(
                model, device, server, n_games=args.games_per_iter,
                battle_format=args.format, temperature=temperature,
                bc_checkpoint=bc_ckpt, max_concurrent=args.max_concurrent,
                opponent_weights=OPPONENT_WEIGHTS_SELFPLAY,
                self_play_pool=full_pool if args.self_play else None,
                self_play_weight=args.self_play_weight,
                bc_opponent_weight=args.bc_opponent_weight,
                opponent_device=opp_device, reward_shaper=reward_shaper,
                server_pool=server_pool, fp16=args.fp16,
            )
        )

        total_games = wins + losses + ties
        wr = wins / max(1, total_games)
        ct = time.time() - t0

        # Build PPO episodes
        episodes = build_ppo_episodes(trajs, gamma=args.gamma, lam=args.lam)
        del trajs
        gc.collect()

        if not episodes:
            print(f"Iter {it+1}: no episodes, skipping")
            continue

        # PPO update
        model.train()
        t1 = time.time()
        loss_info = ppo_update(
            model, optimizer, episodes, device, cfg,
            epochs=args.ppo_epochs, clip_eps=args.clip_eps,
            ent_coef=args.ent_coef, vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            grad_accum=args.grad_accum,
        )

        del episodes
        gc.collect()
        torch.cuda.empty_cache()
        ut = time.time() - t1

        kl_str = f" kl={loss_info['kl']:.4f}"
        warmup_str = " [WARMUP]" if in_warmup else ""
        opp_str = " ".join(f"{k.replace('SelfPlay_','SP')}={v}" for k, v in sorted(opp_dist.items()))
        print(f"Iter {it+1}: W/L/T={wins}/{losses}/{ties} ({wr:.1%}), "
              f"{steps} steps, collect={ct:.0f}s, update={ut:.0f}s, "
              f"pi={loss_info['pi']:.4f} v={loss_info['v']:.4f} "
              f"ent={loss_info['ent']:.4f}{kl_str}{warmup_str}")
        # Per-opponent win rates
        wr_parts = []
        for oname in sorted(per_opp_wr.keys()):
            ow, ot = per_opp_wr[oname]
            owr = ow / max(1, ot) * 100
            short = oname.replace("SelfPlay_", "SP").replace("SimpleHeuristics", "SH")
            short = short.replace("SmartDamage", "SmD").replace("Strategic", "Str")
            short = short.replace("Tactical", "Tac").replace("MaxBasePower", "MBP")
            short = short.replace("SwitchAwareEscape", "SwE").replace("SetupThenSweep", "StS")
            short = short.replace("HazardSense", "HzS").replace("GreedySE", "GSE")
            short = short.replace("BCPolicy", "BC")
            wr_parts.append(f"{short}={owr:.0f}%")
        print(f"  WR: [{' '.join(wr_parts)}] pool={len(snapshot_pool)} hof={len(hall_of_fame)}")

        # Logging
        writer.add_scalar("train/win_rate", wr, it)
        writer.add_scalar("train/steps", steps, it)
        writer.add_scalar("train/temperature", temperature, it)
        for k, v in loss_info.items():
            writer.add_scalar(f"loss/{k}", v, it)

        # Temperature decay
        temperature = max(args.temp_min, temperature * args.temp_decay)

        # Self-play snapshot update
        if args.self_play and (it + 1) % args.self_play_interval == 0:
            sp_path = str(run_dir / f"snapshot_{it+1:04d}.pt")
            save_checkpoint(sp_path, model, cfg, optimizer, it)
            snapshot_pool.append(sp_path)
            while len(snapshot_pool) > args.snapshot_pool_size:
                old = snapshot_pool.pop(0)
                hof_set = {p for _, p in hall_of_fame}
                if old not in hof_set and Path(old).exists():
                    Path(old).unlink()
            # Historical sampling
            _g = glob
            all_hist = set()
            for f in _g.glob(str(run_dir / "snapshot_*.pt")) + _g.glob(str(run_dir / "iter_*.pt")):
                all_hist.add(f)
            pool_set = set(snapshot_pool) | {p for _, p in hall_of_fame}
            all_hist = [p for p in all_hist if p not in pool_set and Path(p).exists()]
            historical_picks = random.sample(all_hist, min(3, len(all_hist))) if all_hist else []
            print(f"  Self-play: pool={len(snapshot_pool)} + {len(historical_picks)} hist + {len(hall_of_fame)} HoF")

        # Eval
        if (it + 1) % args.eval_interval == 0:
            model.eval()
            # Save temp checkpoint for eval
            tmp = str(run_dir / "_eval_temp.pt")
            save_checkpoint(tmp, model, cfg, optimizer, it)
            try:
                from train_bc import eval_vs_bots
                eval_replay_dir = str(run_dir / f"replays_iter{it+1:04d}")
                results = eval_vs_bots(tmp, device=str(device), n_battles=args.eval_games,
                                       server_url=f"ws://127.0.0.1:{ports[0]}/showdown/websocket",
                                       replay_dir=eval_replay_dir)
                smart_avg = results.get("smart_avg", 0.0)
                # eval_vs_bots returns abbreviated keys: SH, SmartDmg, Tactical, Strategic
                _key_map = {"SimpleHeuristics": "SH", "SmartDamage": "SmartDmg",
                            "Tactical": "Tactical", "Strategic": "Strategic"}
                for k in SMART_BOTS:
                    rk = _key_map.get(k, k)
                    val = results.get(rk, results.get(k, None))
                    if val is not None:
                        writer.add_scalar(f"eval/{k}", val, it)
                writer.add_scalar("eval/smart_avg", smart_avg, it)
                print(f"  EVAL: " + ", ".join(f"{k}={v:.0f}%" for k, v in results.items()))

                if smart_avg > best_eval_wr:
                    best_eval_wr = smart_avg
                    save_checkpoint(str(run_dir / "best.pt"), model, cfg, optimizer, it,
                                       {"best_eval_wr": best_eval_wr, **results})
                    print(f"  New best: smart_avg={best_eval_wr:.1f}%")

                # Hall of fame
                if args.self_play:
                    hof_worst = hall_of_fame[0][0] if hall_of_fame else -1
                    if len(hall_of_fame) < args.snapshot_hall_of_fame or smart_avg > hof_worst:
                        hp = str(run_dir / f"hof_{it+1:04d}_wr{smart_avg:.1f}.pt")
                        save_checkpoint(hp, model, cfg, optimizer, it)
                        hall_of_fame.append((smart_avg, hp))
                        if len(hall_of_fame) > args.snapshot_hall_of_fame:
                            hall_of_fame.sort(key=lambda x: x[0])
                            _, rp = hall_of_fame.pop(0)
                            if rp not in snapshot_pool and Path(rp).exists():
                                Path(rp).unlink()
            except Exception as e:
                print(f"  Eval failed: {e}")
                traceback.print_exc()
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            gc.collect()
            torch.cuda.empty_cache()

        # Save
        if (it + 1) % args.save_interval == 0:
            save_checkpoint(
                str(run_dir / f"iter_{it+1:04d}.pt"), model, cfg, optimizer, it,
                {"win_rate": wr, "best_eval_wr": best_eval_wr,
                 "temperature": temperature,
                 "snapshot_pool": snapshot_pool,
                 "hall_of_fame": hall_of_fame},
            )

    # Final save
    save_checkpoint(str(run_dir / "final.pt"), model, cfg, optimizer,
                       start_iter + args.n_iters, {"final": True, "best_eval_wr": best_eval_wr})
    print(f"\nDone. Best eval: {best_eval_wr:.1f}%. Output: {run_dir}")
    writer.close()
    loop.close()


if __name__ == "__main__":
    main()
