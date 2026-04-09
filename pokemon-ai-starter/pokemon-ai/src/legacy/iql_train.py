#!/usr/bin/env python3
"""
IQL (Implicit Q-Learning) trainer for Pokemon AI.

Three networks trained jointly:
  - Q(s,a): predicts expected return for (state, action) pairs
  - V(s):   predicts state value via expectile regression on Q
  - pi(a|s): policy extracted via advantage-weighted behavioral cloning

Data: same memmap episode pipeline as bc_train.py.
Initialize policy from BC checkpoint for warm start.

v2 fixes (2026-03-17):
  - Dense reward shaping from obs features (KO bonuses + HP advantage deltas)
  - LR scheduler steps per batch (was per epoch)
  - Higher default beta (10.0) and tau (0.9) for stronger advantage amplification
  - --resume support to continue training from checkpoint
  - Increased default patience (25)

v4 fixes (2026-03-17):
  - Removed checkpoint pruning — all epoch checkpoints kept
  - --lr-restart flag for warm restart (reset schedulers, keep weights)
  - --eval-every N for periodic eval-based checkpointing during training
  - --eval-bots / --eval-games to configure eval checkpointing
  - Saves eval_best_policy.pt based on actual smart bot win rate
  - Saves policy-only checkpoint every epoch for easy eval
"""

from __future__ import annotations
import argparse, asyncio, csv, json, os, random, shutil, sys, time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from policy_heads import BattlePolicy, PolicyConfig, ModifierSpec
from bc_train import MemmapEpisodeDataset, collate_seq

torch.set_float32_matmul_precision("high")


# ── helpers ──────────────────────────────────────────────────────────

def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def make_model(cfg_dict: dict) -> BattlePolicy:
    """Build a BattlePolicy from a config dict (as stored in checkpoints)."""
    mods = cfg_dict.get("modifiers", None)
    if mods:
        mods = [ModifierSpec(**m) if isinstance(m, dict) else m for m in mods]
    # Backward compat: old checkpoints have use_lstm=True but no use_transformer key.
    # New checkpoints have both. When use_transformer is absent, infer from use_lstm.
    _use_lstm = cfg_dict.get("use_lstm", False)
    _use_transformer = cfg_dict.get("use_transformer", not _use_lstm)

    cfg = PolicyConfig(
        obs_dim=cfg_dict["obs_dim"],
        action_dim=cfg_dict.get("action_dim", 9),
        use_lstm=_use_lstm,
        use_transformer=_use_transformer,
        lstm_hidden=cfg_dict.get("lstm_hidden", 256),
        mlp_hidden=cfg_dict.get("mlp_hidden", 256),
        lstm_layers=cfg_dict.get("lstm_layers", 1),
        mlp_layers=cfg_dict.get("mlp_layers", 2),
        modifiers=mods,
        hierarchical=cfg_dict.get("hierarchical", False),
        step_type_bins=cfg_dict.get("step_type_bins", 3),
        ctx_extra_dim=cfg_dict.get("ctx_extra_dim", 0),
        ctx_proj_dim=cfg_dict.get("ctx_proj_dim", 32),
        move_slot_dim=cfg_dict.get("move_slot_dim", 0),
        switch_slot_dim=cfg_dict.get("switch_slot_dim", 0),
        slot_hidden=cfg_dict.get("slot_hidden", 32),
        n_entity_ids=cfg_dict.get("n_entity_ids", 0),
        embed_dim=cfg_dict.get("embed_dim", 32),
        n_species=cfg_dict.get("n_species", 1548),
        n_moves=cfg_dict.get("n_moves", 953),
        n_items=cfg_dict.get("n_items", 2340),
        n_abilities=cfg_dict.get("n_abilities", 314),
        n_transformer_layers=cfg_dict.get("n_transformer_layers", 6),
        n_heads=cfg_dict.get("n_heads", 4),
        transformer_dropout=cfg_dict.get("transformer_dropout", 0.1),
        context_length=cfg_dict.get("context_length", 128),
    )
    return BattlePolicy(cfg)


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Polyak averaging: target <- tau*source + (1-tau)*target."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def compute_done_mask(mask: torch.Tensor) -> torch.Tensor:
    """Given valid mask [B,T], return done mask [B,T] where 1 = last valid step."""
    done = torch.zeros_like(mask)
    # Interior: valid now but invalid next step
    done[:, :-1] = mask[:, :-1] * (1.0 - mask[:, 1:])
    # Last position: terminal if valid
    done[:, -1] = mask[:, -1]
    return done


# ── Observation feature indices for reward shaping ────────────────
# From features.py v5 layout:
#   obs[1430] = our_alive / 6.0
#   obs[1431] = opp_alive / 6.0
#   obs[1473] = hp_advantage (our_hp% - opp_hp%, clipped to [-1, 1])
OBS_ALIVE_OUR = 1430
OBS_ALIVE_OPP = 1431
OBS_HP_ADV = 1473


def compute_shaped_reward(
    obs: torch.Tensor,
    done: torch.Tensor,
    result: torch.Tensor,
    mask: torch.Tensor,
    ko_coef: float = 0.05,
    hp_coef: float = 0.02,
    terminal_coef: float = 1.0,
) -> torch.Tensor:
    """
    Compute dense per-step reward from observation features + terminal result.
    Unified with rewards.py RewardShaper (same coefficients, same formula).

    Intermediate reward (non-terminal steps):
      - KO delta: +ko_coef per net KO (opp lost - our lost)
      - HP delta: +hp_coef * (change in HP advantage)

    Terminal reward:
      - terminal_coef * result (0 or 1)

    Returns reward tensor [B, T].
    """
    alive_our = obs[:, :, OBS_ALIVE_OUR] * 6.0  # [B, T] actual count
    alive_opp = obs[:, :, OBS_ALIVE_OPP] * 6.0
    hp_adv = obs[:, :, OBS_HP_ADV]

    # Temporal differences: delta from step t to step t+1
    # reward[t] reflects what happened AFTER taking action at step t
    # = f(obs[t+1]) - f(obs[t])
    d_ko = torch.zeros_like(mask)   # [B, T]
    d_hp = torch.zeros_like(mask)

    # For steps 0..T-2, compute delta to next step
    # KO delta: (opp lost mons) - (our lost mons) = positive when we KO, negative when we faint
    d_ko[:, :-1] = (alive_opp[:, :-1] - alive_opp[:, 1:]) - (alive_our[:, :-1] - alive_our[:, 1:])
    # HP advantage delta: positive = we gained HP advantage
    d_hp[:, :-1] = hp_adv[:, 1:] - hp_adv[:, :-1]

    # Zero out deltas at episode boundaries (where done=1, the "next step" is padding)
    d_ko[:, :-1] = d_ko[:, :-1] * mask[:, 1:]
    d_hp[:, :-1] = d_hp[:, :-1] * mask[:, 1:]

    # Terminal step KO bonus: at done=1, compare alive counts at step t vs step t-1
    # This captures KOs that happen on the final turn (otherwise lost)
    # For t >= 1, d_ko[t] = delta from t-1 to t (same formula, reversed direction)
    terminal_mask = done * mask  # [B, T] — 1 only at last valid step
    if terminal_mask.any():
        # KO delta for terminal step: compare current step vs previous step
        d_ko_term = torch.zeros_like(mask)
        d_ko_term[:, 1:] = (alive_opp[:, :-1] - alive_opp[:, 1:]) - (alive_our[:, :-1] - alive_our[:, 1:])
        d_hp_term = torch.zeros_like(mask)
        d_hp_term[:, 1:] = hp_adv[:, 1:] - hp_adv[:, :-1]
        # Only apply at terminal positions (overwrite the zero from the forward-delta)
        d_ko = d_ko + d_ko_term * terminal_mask
        d_hp = d_hp + d_hp_term * terminal_mask

    shaped = ko_coef * d_ko + hp_coef * d_hp
    # result is 1.0=win, 0.0=loss, 0.5=tie, -1.0=unknown fill value
    # Transform to +1/-1/0 to match PPO reward scale, masking out -1 fill values
    result_valid = torch.where(result >= 0, result, torch.zeros_like(result))
    result_transformed = 2.0 * result_valid - 1.0
    terminal = done * result_transformed * terminal_coef

    return shaped + terminal


# ── forward helper ───────────────────────────────────────────────────

def forward_net(net: BattlePolicy, batch: dict, device: torch.device,
                ctx_dim: int = 0) -> dict:
    """Run a BattlePolicy on a collated batch, returning the output dict."""
    obs = batch["obs"].to(device)
    legal = batch["legal"].to(device)
    st = batch.get("step_type")
    if st is not None:
        st = st.to(device)

    # Context extra
    cx = None
    if ctx_dim > 0 and "ctx_extra" in batch:
        cx = batch["ctx_extra"].to(device)
        if cx.shape[-1] < ctx_dim:
            pad = torch.zeros(*cx.shape[:-1], ctx_dim - cx.shape[-1],
                              device=device, dtype=cx.dtype)
            cx = torch.cat([cx, pad], dim=-1)
        elif cx.shape[-1] > ctx_dim:
            cx = cx[..., :ctx_dim]

    # Optional slot / entity tensors
    kw = {}
    for key in ("move_slots", "switch_slots", "entity_ids", "move_ids", "switch_ids"):
        if key in batch and batch[key] is not None:
            kw[key] = batch[key].to(device)

    return net(obs, action_mask=legal, step_type=st, ctx_extra=cx, **kw)


# ── IQL losses ───────────────────────────────────────────────────────

def iql_losses(
    q_net: BattlePolicy,
    v_net: BattlePolicy,
    v_target: BattlePolicy,
    policy: BattlePolicy,
    batch: dict,
    device: torch.device,
    gamma: float = 0.99,
    tau_expectile: float = 0.9,
    beta: float = 10.0,
    ctx_dim: int = 0,
    ko_coef: float = 0.05,
    hp_coef: float = 0.02,
    terminal_coef: float = 1.0,
    binary_advantage: bool = False,
) -> dict:
    """
    Compute IQL losses for one batch of episodes.

    Returns dict with keys: q_loss, v_loss, pi_loss, and diagnostics.
    """
    mask = batch["mask"].to(device)           # [B, T]
    action = batch["action"].to(device)       # [B, T]
    result = batch["result"].to(device)       # [B, T] (broadcast episode outcome)
    legal = batch["legal"].to(device)         # [B, T, 9]
    obs_raw = batch["obs"].to(device)         # [B, T, F]

    # Valid mask: mask AND valid action
    valid = (mask > 0) & (action >= 0)

    done = compute_done_mask(mask)  # [B, T]

    # ── Dense reward shaping from obs features + terminal result ──
    reward = compute_shaped_reward(
        obs_raw, done, result, mask,
        ko_coef=ko_coef, hp_coef=hp_coef, terminal_coef=terminal_coef,
    )

    # ── forward passes ──
    q_out = forward_net(q_net, batch, device, ctx_dim)
    v_out = forward_net(v_net, batch, device, ctx_dim)
    pi_out = forward_net(policy, batch, device, ctx_dim)

    with torch.no_grad():
        vt_out = forward_net(v_target, batch, device, ctx_dim)

    q_all = q_out["action_logits"]    # [B, T, 9] -- repurposed as Q-values
    v = v_out["value"].squeeze(-1)    # [B, T]
    v_targ = vt_out["value"].squeeze(-1)  # [B, T]
    pi_logits = pi_out["action_logits"]   # [B, T, 9]

    # ── gather Q at taken action ──
    # Clamp action to valid range for gather (padded steps have -1)
    act_clamped = action.clamp(min=0)
    q_a = q_all.gather(-1, act_clamped.unsqueeze(-1)).squeeze(-1)  # [B, T]

    # ── Q-loss: Bellman backup using target V ──
    # V(s') = V_target at next timestep; 0 at terminal
    v_next = torch.zeros_like(v_targ)
    v_next[:, :-1] = v_targ[:, 1:]
    # Zero out next-value at terminal and at padding-adjacent steps
    v_next = v_next * (1.0 - done)

    q_target = reward + gamma * v_next  # [B, T]
    q_loss = ((q_a - q_target.detach()) ** 2 * valid.float()).sum() / valid.float().sum().clamp(min=1)

    # ── V-loss: expectile regression toward Q(s,a) ──
    diff = q_a.detach() - v  # [B, T]
    weight = torch.where(diff > 0, tau_expectile, 1.0 - tau_expectile)
    v_loss = (weight * diff ** 2 * valid.float()).sum() / valid.float().sum().clamp(min=1)

    # ── Policy loss: advantage-weighted BC ──
    advantage = (q_a - v).detach()  # [B, T]

    if binary_advantage:
        # Binary filtering (Metamon-style): only train on above-median advantage actions.
        valid_adv = advantage[valid]
        if valid_adv.numel() > 0:
            adv_median = valid_adv.median()
        else:
            adv_median = torch.tensor(0.0, device=device)
        adv_weights = (advantage >= adv_median).float()  # 1.0 above median, 0.0 below
    else:
        # Exponential weighting (original IQL)
        adv_weights = torch.exp(beta * advantage).clamp(max=100.0)  # [B, T]

    # Masked log-softmax over legal actions
    MASK_VAL = -1e9
    masked_logits = torch.where(legal > 0, pi_logits, torch.full_like(pi_logits, MASK_VAL))
    log_pi = torch.log_softmax(masked_logits, dim=-1)  # [B, T, 9]
    nll = -log_pi.gather(-1, act_clamped.unsqueeze(-1)).squeeze(-1)  # [B, T]

    # Normalize by sum of weights (not just valid count) so binary filtering loss scale is correct
    pi_loss = (adv_weights * nll * valid.float()).sum() / (adv_weights * valid.float()).sum().clamp(min=1)

    # ── diagnostics ──
    with torch.no_grad():
        q_mean = (q_a * valid.float()).sum() / valid.float().sum().clamp(min=1)
        v_mean = (v * valid.float()).sum() / valid.float().sum().clamp(min=1)
        adv_mean = (advantage * valid.float()).sum() / valid.float().sum().clamp(min=1)
        adv_std = ((advantage - adv_mean) ** 2 * valid.float()).sum() / valid.float().sum().clamp(min=1)
        adv_std = adv_std.sqrt()
        reward_mean = (reward * valid.float()).sum() / valid.float().sum().clamp(min=1)

        # Policy accuracy (top-1 among legal)
        pred = masked_logits.argmax(dim=-1)  # [B, T]
        flat_valid = valid.reshape(-1)
        if flat_valid.sum() > 0:
            acc = (pred.reshape(-1)[flat_valid] == action.reshape(-1)[flat_valid]).float().mean().item()
        else:
            acc = 0.0

    return {
        "q_loss": q_loss,
        "v_loss": v_loss,
        "pi_loss": pi_loss,
        "q_mean": q_mean.item(),
        "v_mean": v_mean.item(),
        "adv_mean": adv_mean.item(),
        "adv_std": adv_std.item(),
        "reward_mean": reward_mean.item(),
        "pi_acc": acc,
    }


# ── training loop ────────────────────────────────────────────────────

def train_one_epoch(
    q_net, v_net, v_target, policy,
    q_opt, v_opt, pi_opt,
    schedulers,
    train_loader, device, args,
    global_step: int,
    tb: Optional[SummaryWriter] = None,
) -> Tuple[dict, int]:
    """Train for one epoch. Returns (avg_metrics_dict, updated_global_step)."""
    q_net.train(); v_net.train(); policy.train()
    q_sched, v_sched, pi_sched = schedulers

    sums = {"q_loss": 0, "v_loss": 0, "pi_loss": 0,
            "q_mean": 0, "v_mean": 0, "adv_mean": 0, "adv_std": 0,
            "reward_mean": 0, "pi_acc": 0}
    count = 0

    for i, batch in enumerate(train_loader):
        losses = iql_losses(
            q_net, v_net, v_target, policy, batch, device,
            gamma=args.gamma,
            tau_expectile=args.tau,
            beta=args.beta,
            ctx_dim=args.ctx_extra_dim,
            ko_coef=args.reward_ko,
            hp_coef=args.reward_hp,
            terminal_coef=args.reward_terminal,
            binary_advantage=getattr(args, "binary_advantage", False),
        )

        # Update Q
        q_opt.zero_grad(set_to_none=True)
        losses["q_loss"].backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), args.grad_clip)
        q_opt.step()

        # Update V
        v_opt.zero_grad(set_to_none=True)
        losses["v_loss"].backward()
        torch.nn.utils.clip_grad_norm_(v_net.parameters(), args.grad_clip)
        v_opt.step()

        # Update policy
        pi_opt.zero_grad(set_to_none=True)
        losses["pi_loss"].backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        pi_opt.step()

        # Step LR schedulers per batch (not per epoch)
        q_sched.step()
        v_sched.step()
        pi_sched.step()

        # Soft-update V target
        soft_update(v_target, v_net, args.target_tau)

        # Accumulate metrics
        for k in sums:
            sums[k] += losses[k] if isinstance(losses[k], float) else losses[k].item()
        count += 1

        # Logging
        if tb and (i % 50 == 0):
            tb.add_scalar("train/q_loss", losses["q_loss"].item(), global_step)
            tb.add_scalar("train/v_loss", losses["v_loss"].item(), global_step)
            tb.add_scalar("train/pi_loss", losses["pi_loss"].item(), global_step)
            tb.add_scalar("train/q_mean", losses["q_mean"], global_step)
            tb.add_scalar("train/v_mean", losses["v_mean"], global_step)
            tb.add_scalar("train/adv_mean", losses["adv_mean"], global_step)
            tb.add_scalar("train/adv_std", losses["adv_std"], global_step)
            tb.add_scalar("train/reward_mean", losses["reward_mean"], global_step)
            tb.add_scalar("train/pi_acc", losses["pi_acc"], global_step)

        if i % 100 == 0:
            print(f"  [step {i}] q={losses['q_loss'].item():.4f} "
                  f"v={losses['v_loss'].item():.4f} pi={losses['pi_loss'].item():.4f} "
                  f"Qm={losses['q_mean']:.3f} Vm={losses['v_mean']:.3f} "
                  f"Am={losses['adv_mean']:.3f}+/-{losses['adv_std']:.3f} "
                  f"Rm={losses['reward_mean']:.4f} acc={losses['pi_acc']:.3f}",
                  flush=True)

        global_step += 1

        # Free VRAM
        del losses
        if device.type == "cuda" and i % 200 == 0:
            torch.cuda.empty_cache()

    avgs = {k: v / max(count, 1) for k, v in sums.items()}
    return avgs, global_step


@torch.no_grad()
def eval_epoch(
    q_net, v_net, v_target, policy,
    val_loader, device, args,
) -> dict:
    """Evaluate on validation set."""
    q_net.eval(); v_net.eval(); policy.eval()

    sums = {"q_loss": 0, "v_loss": 0, "pi_loss": 0,
            "q_mean": 0, "v_mean": 0, "adv_mean": 0, "adv_std": 0,
            "reward_mean": 0, "pi_acc": 0}
    count = 0

    for batch in val_loader:
        losses = iql_losses(
            q_net, v_net, v_target, policy, batch, device,
            gamma=args.gamma,
            tau_expectile=args.tau,
            beta=args.beta,
            ctx_dim=args.ctx_extra_dim,
            ko_coef=args.reward_ko,
            hp_coef=args.reward_hp,
            terminal_coef=args.reward_terminal,
            binary_advantage=getattr(args, "binary_advantage", False),
        )
        for k in sums:
            sums[k] += losses[k] if isinstance(losses[k], float) else losses[k].item()
        count += 1

    return {k: v / max(count, 1) for k, v in sums.items()}


# ── live eval during training ────────────────────────────────────────

def run_eval_checkpoint(policy, policy_cfg, device, epoch, run_dir, args,
                        best_eval_wr: float) -> float:
    """
    Run a quick eval of the policy against smart bots.
    Saves eval_best_policy.pt if win rate improves.
    Returns the new best_eval_wr.
    """
    from poke_env.ps_client.server_configuration import ServerConfiguration
    from bc_policy_player import BCPolicyPlayer
    from poke_env.player.baselines import SimpleHeuristicsPlayer
    from policy_smartbots import SmartDamagePlayer, TacticalPlayer, StrategicPlayer
    from teams_ou import random_pool_teambuilder

    # Save a temp policy checkpoint for eval
    tmp_ckpt = run_dir / "_eval_tmp_policy.pt"
    torch.save({
        "model": policy.state_dict(),
        "policy_cfg": policy_cfg,
        "obs_dim": policy_cfg["obs_dim"],
        "epoch": epoch,
    }, tmp_ckpt)

    bots = {
        "SimpleHeuristics": SimpleHeuristicsPlayer,
        "SmartDamage": SmartDamagePlayer,
        "Tactical": TacticalPlayer,
        "Strategic": StrategicPlayer,
    }

    server_url = getattr(args, 'eval_server', 'ws://127.0.0.1:8000/showdown/websocket')
    ws = server_url.rstrip("/")
    http = ws.replace("wss://", "https://").replace("ws://", "http://")
    if ws.endswith("/showdown/websocket"):
        http = http[: http.rfind("/showdown/websocket")] + "/action.php?"
    server = ServerConfiguration(ws, http)

    n_games = getattr(args, 'eval_games', 20)
    total_wins = 0
    total_games = 0
    results = {}

    for bot_name, BotClass in bots.items():
        try:
            our = BCPolicyPlayer(
                checkpoint_path=str(tmp_ckpt),
                device="cpu",  # eval on CPU to not compete with GPU training
                battle_format="gen9ou",
                server_configuration=server,
                team=random_pool_teambuilder(),
                save_replays=False,
                max_concurrent_battles=2,
            )
            opp = BotClass(
                battle_format="gen9ou",
                server_configuration=server,
                team=random_pool_teambuilder(),
                save_replays=False,
                max_concurrent_battles=2,
            )

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(our.battle_against(opp, n_battles=n_games))
            finally:
                loop.close()

            wins = our.n_won_battles
            total = our.n_won_battles + our.n_lost_battles + our.n_tied_battles
            wr = wins / max(1, total)
            results[bot_name] = wr
            total_wins += wins
            total_games += total
        except Exception as e:
            print(f"  [eval] WARNING: {bot_name} failed: {e}", flush=True)
            results[bot_name] = -1

    # Clean up temp file
    tmp_ckpt.unlink(missing_ok=True)

    avg_wr = total_wins / max(1, total_games)
    results_str = " ".join(f"{k[:6]}={v:.0%}" for k, v in results.items() if v >= 0)
    print(f"  [eval] epoch {epoch}: avg_smart_wr={avg_wr:.1%} ({results_str})", flush=True)

    # Save eval best
    if avg_wr > best_eval_wr:
        best_eval_wr = avg_wr
        torch.save({
            "model": policy.state_dict(),
            "policy_cfg": policy_cfg,
            "obs_dim": policy_cfg["obs_dim"],
            "epoch": epoch,
            "eval_wr": avg_wr,
            "eval_results": results,
        }, run_dir / "eval_best_policy.pt")
        print(f"  [eval] * New eval best: {avg_wr:.1%} (saved eval_best_policy.pt)", flush=True)

    # Log to CSV
    eval_csv = run_dir / "eval_metrics.csv"
    write_header = not eval_csv.exists()
    with open(eval_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["epoch", "avg_smart_wr"] + list(bots.keys()))
        w.writerow([epoch, f"{avg_wr:.4f}"] + [f"{results.get(k, -1):.4f}" for k in bots.keys()])

    return best_eval_wr


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IQL training for Pokemon AI")

    # Data
    parser.add_argument("--memmap-dir", default="data/datasets/memmap")
    parser.add_argument("--val-ratio", type=float, default=0.1)

    # Model init
    parser.add_argument("--init-from", type=str, default="data/models/bc/v5_vanilla_bc/best.pt",
                        help="BC checkpoint to initialize all networks from")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume training from a full IQL checkpoint (.pt)")

    # IQL hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.9,
                        help="Expectile for V-loss (0.5=mean, higher=more optimistic)")
    parser.add_argument("--beta", type=float, default=10.0,
                        help="Inverse temperature for advantage weighting (higher=greedier)")
    parser.add_argument("--binary-advantage", action="store_true", default=False,
                        help="Use binary advantage filtering: train only on above-median advantage actions (Metamon-style). Ignores --beta.")
    parser.add_argument("--target-tau", type=float, default=0.005,
                        help="Polyak averaging rate for V-target network")

    # Reward shaping
    parser.add_argument("--reward-ko", type=float, default=0.05,
                        help="Reward per net KO delta (unified with PPO rewards.py)")
    parser.add_argument("--reward-hp", type=float, default=0.02,
                        help="Reward per unit HP advantage delta (unified with PPO rewards.py)")
    parser.add_argument("--reward-terminal", type=float, default=1.0,
                        help="Terminal reward multiplier (win=+coef, loss=0)")

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-policy", type=float, default=0.0,
                        help="Separate LR for policy (0=use --lr)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience (0=off)")

    # LR restart (warm restart from checkpoint with fresh LR schedule)
    parser.add_argument("--lr-restart", action="store_true",
                        help="Reset LR schedulers when resuming (warm restart)")

    # Eval-based checkpointing
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Run eval vs smart bots every N epochs (0=off)")
    parser.add_argument("--eval-games", type=int, default=20,
                        help="Games per bot during eval checkpoint")
    parser.add_argument("--eval-server", type=str,
                        default="ws://127.0.0.1:8000/showdown/websocket",
                        help="Showdown server for eval")

    # Infrastructure
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--log-every", type=int, default=100)

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"[IQL] device={device}", flush=True)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Resume or fresh init ──
    start_epoch = 1
    global_step = 0
    best_val_pi = float("inf")
    best_eval_wr = 0.0
    epochs_no_improve = 0
    resume_ckpt = None

    if args.resume:
        print(f"[IQL] Resuming from checkpoint: {args.resume}", flush=True)
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        policy_cfg = resume_ckpt["policy_cfg"]
        start_epoch = resume_ckpt["epoch"] + 1
        global_step = resume_ckpt["global_step"]
        args.ctx_extra_dim = policy_cfg.get("ctx_extra_dim", 0)
        print(f"[IQL] Resuming from epoch {resume_ckpt['epoch']}, "
              f"global_step={global_step}", flush=True)
    else:
        print(f"[IQL] Loading BC checkpoint: {args.init_from}", flush=True)
        bc_ckpt = torch.load(args.init_from, map_location="cpu")
        policy_cfg = bc_ckpt["policy_cfg"]
        args.ctx_extra_dim = policy_cfg.get("ctx_extra_dim", 0)

    print(f"[IQL] Model config: hidden={policy_cfg['mlp_hidden']}, "
          f"lstm={policy_cfg['lstm_hidden']}, mlp_layers={policy_cfg['mlp_layers']}, "
          f"obs_dim={policy_cfg['obs_dim']}, entity_ids={policy_cfg.get('n_entity_ids', 0)}",
          flush=True)

    # ── Build 4 networks ──
    q_net = make_model(policy_cfg).to(device)
    v_net = make_model(policy_cfg).to(device)
    v_target = make_model(policy_cfg).to(device)
    policy = make_model(policy_cfg).to(device)

    if resume_ckpt:
        # Restore network weights from checkpoint
        q_net.load_state_dict(resume_ckpt["q_net"])
        v_net.load_state_dict(resume_ckpt["v_net"])
        v_target.load_state_dict(resume_ckpt["v_target"])
        policy.load_state_dict(resume_ckpt["policy"])
        print(f"[IQL] Restored all 4 networks from checkpoint", flush=True)
    else:
        # Initialize all from BC weights (backbone features are useful)
        bc_state = bc_ckpt["model"]
        for net, name in [(q_net, "Q"), (v_net, "V"), (v_target, "V_target"), (policy, "Policy")]:
            missing, unexpected = net.load_state_dict(bc_state, strict=False)
            n_loaded = len(net.state_dict()) - len(missing)
            print(f"[IQL] {name}: loaded {n_loaded} params, {len(missing)} missing, "
                  f"{len(unexpected)} unexpected", flush=True)

        # V_target starts as copy of V (no grad)
        v_target.load_state_dict(v_net.state_dict())
        del bc_ckpt, bc_state  # free ~150MB

    for p in v_target.parameters():
        p.requires_grad_(False)

    # Count parameters
    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    total_params = count_params(q_net) + count_params(v_net) + count_params(policy)
    print(f"[IQL] Trainable params: Q={count_params(q_net)/1e6:.1f}M, "
          f"V={count_params(v_net)/1e6:.1f}M, Pi={count_params(policy)/1e6:.1f}M, "
          f"total={total_params/1e6:.1f}M", flush=True)

    # ── Dataset ──
    print(f"[IQL] Loading memmap data from {args.memmap_dir}", flush=True)
    train_ds = MemmapEpisodeDataset(args.memmap_dir, split="train", val_ratio=args.val_ratio)
    val_ds = MemmapEpisodeDataset(args.memmap_dir, split="val", val_ratio=args.val_ratio)
    print(f"[IQL] Train episodes: {len(train_ds)}, Val episodes: {len(val_ds)}", flush=True)

    dl_common = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
        collate_fn=collate_seq,
    )
    if args.workers > 0:
        dl_common["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **dl_common)

    steps_per_epoch = max(1, len(train_loader))
    print(f"[IQL] Steps/epoch: ~{steps_per_epoch}", flush=True)

    # ── Optimizers ──
    lr_pi = args.lr_policy if args.lr_policy > 0 else args.lr
    q_opt = torch.optim.AdamW(q_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    v_opt = torch.optim.AdamW(v_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pi_opt = torch.optim.AdamW(policy.parameters(), lr=lr_pi, weight_decay=args.weight_decay)

    # Cosine schedule — T_max = total batch steps across all epochs
    total_steps = steps_per_epoch * args.epochs
    q_sched = CosineAnnealingLR(q_opt, T_max=total_steps, eta_min=1e-6)
    v_sched = CosineAnnealingLR(v_opt, T_max=total_steps, eta_min=1e-6)
    pi_sched = CosineAnnealingLR(pi_opt, T_max=total_steps, eta_min=1e-6)

    if resume_ckpt:
        if args.lr_restart:
            # Warm restart: keep network weights, fresh optimizers + schedulers
            # Recompute total_steps for the NEW epochs only
            remaining_epochs = args.epochs - start_epoch + 1
            total_steps = steps_per_epoch * remaining_epochs
            q_sched = CosineAnnealingLR(q_opt, T_max=total_steps, eta_min=1e-6)
            v_sched = CosineAnnealingLR(v_opt, T_max=total_steps, eta_min=1e-6)
            pi_sched = CosineAnnealingLR(pi_opt, T_max=total_steps, eta_min=1e-6)
            print(f"[IQL] LR restart: fresh schedulers for {remaining_epochs} epochs "
                  f"({total_steps} steps), lr={args.lr}", flush=True)
        else:
            # Full restore: optimizer + scheduler states
            q_opt.load_state_dict(resume_ckpt["q_opt"])
            v_opt.load_state_dict(resume_ckpt["v_opt"])
            pi_opt.load_state_dict(resume_ckpt["pi_opt"])
            if "q_sched" in resume_ckpt:
                q_sched.load_state_dict(resume_ckpt["q_sched"])
                v_sched.load_state_dict(resume_ckpt["v_sched"])
                pi_sched.load_state_dict(resume_ckpt["pi_sched"])
                # Warn if T_max changed (e.g., --epochs differs from original run)
                saved_T_max = resume_ckpt["q_sched"].get("T_max", total_steps)
                if saved_T_max != total_steps:
                    print(f"[IQL] WARNING: CosineAnnealingLR T_max mismatch — "
                          f"saved={saved_T_max}, current={total_steps} "
                          f"(steps_per_epoch={steps_per_epoch} * epochs={args.epochs}). "
                          f"The restored scheduler state uses the saved T_max. "
                          f"Use --lr-restart to reset schedulers with the new T_max.",
                          flush=True)
            else:
                # No saved scheduler state — fast-forward to match global_step.
                # NOTE: T_max = steps_per_epoch * args.epochs which may differ from
                # the original training run if --epochs changed. Use --lr-restart
                # for a clean LR schedule when extending training.
                print(f"[IQL] WARNING: No saved scheduler state. Fast-forwarding "
                      f"{global_step} steps with T_max={total_steps}. If --epochs "
                      f"differs from original run, LR schedule may be incorrect. "
                      f"Consider --lr-restart.", flush=True)
                for _ in range(global_step):
                    q_sched.step(); v_sched.step(); pi_sched.step()
        # Restore best val pi-loss and patience counter
        best_val_pi = resume_ckpt.get("best_val_pi", float("inf"))
        best_eval_wr = resume_ckpt.get("best_eval_wr", 0.0)
        epochs_no_improve = resume_ckpt.get("epochs_no_improve", 0)
        print(f"[IQL] Restored (best_val_pi={best_val_pi:.4f}, "
              f"best_eval_wr={best_eval_wr:.1%}, "
              f"epochs_no_improve={epochs_no_improve})", flush=True)
        del resume_ckpt  # free memory

    schedulers = (q_sched, v_sched, pi_sched)

    # ── Run directory ──
    if args.resume:
        # Continue in the same run directory as the checkpoint
        run_dir = Path(args.resume).parent
        run_name = run_dir.name
    else:
        run_name = args.run_name or f"v5_iql-{_timestamp()}"
        run_dir = Path("data/models/iql") / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

    # Save run config (append resume info if resuming)
    meta_path = run_dir / "run_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    meta.update({
        "last_updated": _timestamp(),
        "args": vars(args),
        "policy_cfg": policy_cfg,
    })
    if not args.resume:
        meta["created"] = _timestamp()
        meta["init_from"] = args.init_from
    else:
        meta["resumed_from"] = args.resume
        meta["resumed_at_epoch"] = start_epoch
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # CSV log — always write fresh header (resume starts a new metrics file
    # since reward shaping changes the metrics being tracked)
    csv_path = run_dir / "metrics.csv"
    csv_header = ["epoch", "q_loss", "v_loss", "pi_loss",
                  "val_q_loss", "val_v_loss", "val_pi_loss",
                  "q_mean", "v_mean", "adv_mean", "adv_std",
                  "reward_mean", "pi_acc", "lr"]
    if args.resume and csv_path.exists():
        # Backup old CSV before overwriting
        backup_csv = csv_path.with_suffix(f".pre_resume_ep{start_epoch}.csv")
        if not backup_csv.exists():
            import shutil
            shutil.copy2(str(csv_path), str(backup_csv))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_header)

    # TensorBoard
    tb = SummaryWriter(log_dir=str(Path("data/logs/tb/iql") / run_name))

    # ── Training loop ──
    print(f"\n[IQL] Starting training: epochs {start_epoch}-{args.epochs}, "
          f"gamma={args.gamma}, tau={args.tau}, beta={args.beta}, "
          f"target_tau={args.target_tau}, lr={args.lr}", flush=True)
    print(f"[IQL] Reward shaping: ko={args.reward_ko}, hp={args.reward_hp}, "
          f"terminal={args.reward_terminal}", flush=True)
    print(f"[IQL] Run dir: {run_dir}\n", flush=True)

    epoch = start_epoch - 1  # default in case loop doesn't run
    if start_epoch > args.epochs:
        print(f"[IQL] start_epoch ({start_epoch}) > epochs ({args.epochs}), nothing to train. "
              f"Increase --epochs to continue.", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Train
        train_metrics, global_step = train_one_epoch(
            q_net, v_net, v_target, policy,
            q_opt, v_opt, pi_opt,
            schedulers,
            train_loader, device, args,
            global_step, tb,
        )

        # Validate
        val_metrics = eval_epoch(q_net, v_net, v_target, policy, val_loader, device, args)

        elapsed = time.time() - t0
        lr_now = q_opt.param_groups[0]["lr"]

        # Log
        print(f"[IQL][epoch {epoch}/{args.epochs}] "
              f"q={train_metrics['q_loss']:.4f}/{val_metrics['q_loss']:.4f} "
              f"v={train_metrics['v_loss']:.4f}/{val_metrics['v_loss']:.4f} "
              f"pi={train_metrics['pi_loss']:.4f}/{val_metrics['pi_loss']:.4f} "
              f"Qm={val_metrics['q_mean']:.3f} Vm={val_metrics['v_mean']:.3f} "
              f"Am={val_metrics['adv_mean']:.3f}+/-{val_metrics['adv_std']:.3f} "
              f"Rm={val_metrics['reward_mean']:.4f} "
              f"acc={val_metrics['pi_acc']:.3f} lr={lr_now:.6f} ({elapsed:.1f}s)",
              flush=True)

        # CSV
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch,
                f"{train_metrics['q_loss']:.6f}", f"{train_metrics['v_loss']:.6f}",
                f"{train_metrics['pi_loss']:.6f}",
                f"{val_metrics['q_loss']:.6f}", f"{val_metrics['v_loss']:.6f}",
                f"{val_metrics['pi_loss']:.6f}",
                f"{val_metrics['q_mean']:.4f}", f"{val_metrics['v_mean']:.4f}",
                f"{val_metrics['adv_mean']:.4f}", f"{val_metrics['adv_std']:.4f}",
                f"{val_metrics['reward_mean']:.4f}", f"{val_metrics['pi_acc']:.4f}",
                f"{lr_now:.8f}",
            ])

        # TensorBoard epoch-level
        for prefix, m in [("train", train_metrics), ("val", val_metrics)]:
            for k, v in m.items():
                tb.add_scalar(f"{prefix}_epoch/{k}", v, epoch)

        # Checkpoint — includes scheduler state for correct resume
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "policy_cfg": policy_cfg,
            "q_net": q_net.state_dict(),
            "v_net": v_net.state_dict(),
            "v_target": v_target.state_dict(),
            "policy": policy.state_dict(),
            "q_opt": q_opt.state_dict(),
            "v_opt": v_opt.state_dict(),
            "pi_opt": pi_opt.state_dict(),
            "q_sched": q_sched.state_dict(),
            "v_sched": v_sched.state_dict(),
            "pi_sched": pi_sched.state_dict(),
            "best_val_pi": best_val_pi,
            "best_eval_wr": best_eval_wr,
            "epochs_no_improve": epochs_no_improve,
            "args": vars(args),
        }
        torch.save(ckpt, run_dir / f"epoch_{epoch:03d}.pt")

        # Save best by val policy loss (primary metric for IQL)
        val_pi = val_metrics["pi_loss"]
        if val_pi < best_val_pi:
            best_val_pi = val_pi
            epochs_no_improve = 0
            torch.save(ckpt, run_dir / "best.pt")
            # Also save policy-only checkpoint for easy eval
            torch.save({
                "model": policy.state_dict(),
                "policy_cfg": policy_cfg,
                "obs_dim": policy_cfg["obs_dim"],
                "epoch": epoch,
                "iql_metrics": val_metrics,
            }, run_dir / "best_policy.pt")
            print(f"  * New best val pi-loss: {val_pi:.4f} (saved best.pt + best_policy.pt)",
                  flush=True)
        else:
            epochs_no_improve += 1

        # Save policy-only checkpoint every epoch for easy eval
        torch.save({
            "model": policy.state_dict(),
            "policy_cfg": policy_cfg,
            "obs_dim": policy_cfg["obs_dim"],
            "epoch": epoch,
        }, run_dir / f"epoch_{epoch:03d}_policy.pt")

        # Eval-based checkpointing
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            best_eval_wr = run_eval_checkpoint(
                policy, policy_cfg, device, epoch, run_dir, args, best_eval_wr)

        # Early stopping
        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"[IQL] Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)", flush=True)
            break

    # Final save
    torch.save({
        "model": policy.state_dict(),
        "policy_cfg": policy_cfg,
        "obs_dim": policy_cfg["obs_dim"],
        "epoch": epoch,
    }, run_dir / "final_policy.pt")

    # Run final eval if eval checkpointing is enabled
    if args.eval_every > 0:
        best_eval_wr = run_eval_checkpoint(
            policy, policy_cfg, device, epoch, run_dir, args, best_eval_wr)

    print(f"\n[IQL] Training complete. Best val pi-loss: {best_val_pi:.4f}")
    if best_eval_wr > 0:
        print(f"[IQL] Best eval win rate: {best_eval_wr:.1%}")
    print(f"[IQL] Policy checkpoints in {run_dir}/")
    print(f"[IQL]   best_policy.pt - best by val pi-loss")
    if args.eval_every > 0:
        print(f"[IQL]   eval_best_policy.pt - best by smart bot win rate")
    print(f"[IQL]   final_policy.pt - last epoch")
    print(f"[IQL]   epoch_XXX_policy.pt - every epoch (for comparison)")
    tb.close()


if __name__ == "__main__":
    main()
