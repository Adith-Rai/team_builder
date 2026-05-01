#!/usr/bin/env python
# train_bc.py — Behavioral Cloning training for PokeTransformer (v8)
#
# Usage:
#   python -u bc_train_v8.py --memmap-dir data/datasets/memmap_v8 \
#     --device cuda --epochs 10 --batch-size 16 --lr 1e-4 --workers 0

from __future__ import annotations
import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import asyncio

from dataset import MemmapDataset, collate_seq, unpack_turn_batch
from model import PokeTransformer, PokeTransformerConfig, add_model_args, config_from_args


def masked_policy_ce(logits: torch.Tensor, actions: torch.Tensor,
                     legal: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """Cross-entropy loss over legal actions.

    When label_smoothing > 0, smooths ONLY over legal actions (not all action
    classes). Using F.cross_entropy's built-in label_smoothing with masked
    logits would distribute probability mass uniformly to every class
    including illegal ones, giving illegal actions non-zero target weight.
    """
    masked_logits = logits.masked_fill(legal < 0.5, -100.0)
    if label_smoothing <= 0.0:
        return F.cross_entropy(masked_logits, actions)

    # Defensive: zero-legal-actions should never happen (every state has at least
    # one legal action in poke-env). If it does, fall back to uniform.
    n_legal = legal.sum(dim=-1).clamp(min=1.0)  # (N,)
    log_probs = F.log_softmax(masked_logits, dim=-1)  # (N, A)
    # Uniform distribution over legal actions
    smooth_target = legal.float() / n_legal.unsqueeze(-1)  # (N, A)
    # One-hot target for the demonstrator's action
    hot_target = F.one_hot(actions, num_classes=logits.shape[-1]).float()  # (N, A)
    target = (1.0 - label_smoothing) * hot_target + label_smoothing * smooth_target
    return -(target * log_probs).sum(-1).mean()


def compute_accuracy(logits: torch.Tensor, actions: torch.Tensor,
                     legal: torch.Tensor) -> float:
    """Top-1 accuracy over legal actions."""
    masked_logits = logits.masked_fill(legal < 0.5, -100.0)
    preds = masked_logits.argmax(dim=-1)
    return (preds == actions).float().mean().item()


def train_one_epoch(model: PokeTransformer, loader: DataLoader,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, tb: Optional[SummaryWriter], global_step: int,
                    grad_clip: float = 1.0, label_smoothing: float = 0.0,
                    scheduler=None, save_fn=None, save_every: int = 1000,
                    scaler: Optional[torch.amp.GradScaler] = None,
                    ) -> tuple:
    """Train one epoch using batched spatial + sequential temporal.
    Returns (avg_loss, avg_acc, avg_v_loss, global_step)."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_vloss = 0.0
    n_batches = 0
    n_turns = 0
    n_skipped_empty = 0   # batches dropped because no valid turns
    n_skipped_nan = 0     # batches dropped due to NaN loss
    t0 = time.time()
    use_amp = scaler is not None

    for batch_idx, collated in enumerate(loader):
        B = collated["seq_lens"].shape[0]
        T = collated["mask"].shape[1]
        mask = collated["mask"].to(device)  # (B, T)
        actions = collated["action"].to(device)  # (B, T)
        results = collated["result"].to(device)  # (B, T)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            # Batched forward: spatial all-at-once, temporal per-turn
            out = model.forward_sequence(collated, device)
            logits = out["action_logits"]  # (B, T, 9)
            v_logits_all = out["v_logits"]  # (B, T, 51)

            # Flatten valid turns for loss computation
            valid = (mask > 0.5) & (actions >= 0)  # (B, T)
            if not valid.any():
                # Silent skips hide systematic data issues. Track and surface.
                n_skipped_empty += 1
                continue

            flat_logits = logits[valid].float()  # (N_valid, 9) — float32 for loss
            flat_actions = actions[valid]  # (N_valid,)
            flat_legal = collated["legal_mask_raw"].to(device)[valid]  # (N_valid, 9)

            pi_loss = masked_policy_ce(flat_logits, flat_actions, flat_legal,
                                       label_smoothing=label_smoothing)
            acc = compute_accuracy(flat_logits, flat_actions, flat_legal)

            # Value loss on turns with known results
            flat_results = results[valid]  # (N_valid,)
            v_valid = flat_results >= 0
            v_loss = torch.tensor(0.0, device=device)
            if v_valid.any():
                vl = v_logits_all[valid][v_valid].float()  # float32 for CE loss
                vt = flat_results[v_valid]
                twohot = model.twohot_target(vt)
                v_loss = F.cross_entropy(vl, twohot) * 0.5

            total_batch_loss = pi_loss + v_loss

        # Guard against NaN/Inf loss before stepping optimizer — a silent NaN
        # step propagates NaN to all parameters permanently.
        if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
            print(f"  [WARN] NaN/Inf loss at batch {batch_idx} (pi={pi_loss.item()}, v={v_loss.item()}), skipping", flush=True)
            n_skipped_nan += 1
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(total_batch_loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_batch_loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        n_valid = int(valid.sum().item())
        total_loss += pi_loss.item()
        total_vloss += v_loss.item()
        total_acc += acc * n_valid
        n_turns += n_valid
        n_batches += 1

        if scheduler is not None:
            scheduler.step()  # per-batch LR stepping

        if tb and n_batches % 10 == 0:
            tb.add_scalar("train/pi_loss", pi_loss.item(), global_step)
            tb.add_scalar("train/v_loss", v_loss.item(), global_step)
            tb.add_scalar("train/acc", acc, global_step)
            tb.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
        global_step += 1

        # Free batch tensors aggressively to prevent RAM + VRAM leak
        del out, logits, v_logits_all, mask, actions, results, collated
        try:
            del flat_logits, flat_actions, flat_legal, flat_results, total_batch_loss
        except NameError:
            pass
        if n_batches % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        # Mid-epoch checkpoint (prevents losing hours of training to crashes)
        if save_fn and save_every > 0 and n_batches % save_every == 0:
            save_fn(n_batches, global_step)
            print(f"  [checkpoint saved at batch {n_batches}, step {global_step}]", flush=True)

        if n_batches % 20 == 0:
            elapsed = time.time() - t0
            avg_loss = total_loss / n_batches
            avg_acc = total_acc / max(1, n_turns)
            print(f"  [{n_batches}] loss={avg_loss:.4f} acc={avg_acc:.3f} "
                  f"turns={n_turns} elapsed={elapsed:.0f}s", flush=True)

    avg_loss = total_loss / max(1, n_batches)
    avg_acc = total_acc / max(1, n_turns)
    avg_vloss = total_vloss / max(1, n_batches)
    # Surface silent batch skips if substantial — helps diagnose data/loss issues
    # that would otherwise just look like slow training.
    total_batches_seen = n_batches + n_skipped_empty + n_skipped_nan
    if total_batches_seen > 0:
        skip_frac = (n_skipped_empty + n_skipped_nan) / total_batches_seen
        if skip_frac >= 0.01 or n_skipped_nan > 0:
            print(f"  [epoch {epoch} skip summary] empty={n_skipped_empty} nan={n_skipped_nan} "
                  f"total_seen={total_batches_seen} ({100*skip_frac:.1f}% skipped)", flush=True)
    return avg_loss, avg_acc, avg_vloss, global_step


@torch.no_grad()
def validate(model: PokeTransformer, loader: DataLoader, device: torch.device) -> tuple:
    """Validate using batched forward. Returns (avg_loss, avg_acc, avg_vloss)."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_vloss = 0.0
    n_batches = 0
    n_turns = 0

    for collated in loader:
        mask = collated["mask"].to(device)
        actions = collated["action"].to(device)
        results = collated["result"].to(device)

        out = model.forward_sequence(collated, device)
        logits = out["action_logits"]
        v_logits_all = out["v_logits"]

        valid = (mask > 0.5) & (actions >= 0)
        if not valid.any():
            continue

        flat_logits = logits[valid]
        flat_actions = actions[valid]
        flat_legal = collated["legal_mask_raw"].to(device)[valid]

        pi_loss = masked_policy_ce(flat_logits, flat_actions, flat_legal)
        acc = compute_accuracy(flat_logits, flat_actions, flat_legal)

        flat_results = results[valid]
        v_valid = flat_results >= 0
        v_loss_val = 0.0
        if v_valid.any():
            vl = v_logits_all[valid][v_valid]
            vt = flat_results[v_valid]
            twohot = model.twohot_target(vt)
            v_loss_val = F.cross_entropy(vl, twohot).item() * 0.5

        n_valid = int(valid.sum().item())
        total_loss += pi_loss.item()
        total_vloss += v_loss_val
        total_acc += acc * n_valid
        n_turns += n_valid
        n_batches += 1

    return (total_loss / max(1, n_batches),
            total_acc / max(1, n_turns),
            total_vloss / max(1, n_batches))


def eval_vs_bots(checkpoint_path: str, device: str = "cuda", n_battles: int = 200,
                  server_url: str = "ws://127.0.0.1:9000/showdown/websocket",
                  replay_dir: str = None, battle_format: str = "gen9ou",
                  team_set: str = "pool") -> dict:
    """Run bot eval on a checkpoint. Returns dict of win rates + smart_avg.
    Must be called when no other async loop is running.

    Args:
        team_set: "pool" (default — 70-team teams_ou pool, RandomPoolTeambuilder)
                  or "metamon-competitive" (16 curated Smogon teams from
                  metamon_cache/teams/competitive/gen9ou; lower team-quality
                  variance → cleaner skill measurement, ladder-validated).
                  S44 finding: same-policy variance is ~3.6pt smart_avg /
                  ~12pt single-bot at 200×4 games on either pool, but the
                  metamon-competitive pool has tighter team-quality spread
                  so absolute scores are more reproducible across evals.
    """
    from poke_env.ps_client.account_configuration import AccountConfiguration
    from poke_env.ps_client.server_configuration import ServerConfiguration
    from poke_env.player.baselines import SimpleHeuristicsPlayer
    from policy_smartbots import SmartDamagePlayer, TacticalPlayer, StrategicPlayer
    from battle_agent import BattleAgent

    if team_set == "metamon-competitive":
        from eval_metamon_competitive import MetamonCompetitiveTeambuilder
        # ONE shared teambuilder instance — both sides sample from same
        # 16-team pool. Loaded once, used for all matchups in this eval.
        _shared_tb = MetamonCompetitiveTeambuilder()
        def _make_tb():
            return _shared_tb
    elif team_set == "pool":
        from teams_ou import random_pool_teambuilder
        def _make_tb():
            return random_pool_teambuilder()
    else:
        raise ValueError(
            f"Unknown team_set={team_set!r}. Must be 'pool' or 'metamon-competitive'."
        )

    SERVER = ServerConfiguration(server_url, None)
    opponents = [
        (SimpleHeuristicsPlayer, "SH"),
        (SmartDamagePlayer, "SmartDmg"),
        (TacticalPlayer, "Tactical"),
        (StrategicPlayer, "Strategic"),
    ]

    async def _run():
        results = {}
        for opp_cls, opp_name in opponents:
            save_dir = None
            if replay_dir:
                save_dir = os.path.join(replay_dir, opp_name)
                os.makedirs(save_dir, exist_ok=True)
            p1 = BattleAgent(
                checkpoint_path, device=device,
                account_configuration=AccountConfiguration.generate("Eval", rand=True),
                battle_format=battle_format, max_concurrent_battles=5,
                server_configuration=SERVER, team=_make_tb(),
                save_replays=save_dir if save_dir else False,
            )
            p2 = opp_cls(
                account_configuration=AccountConfiguration.generate(opp_name, rand=True),
                battle_format=battle_format, max_concurrent_battles=5,
                server_configuration=SERVER, team=_make_tb(),
            )
            await p1.battle_against(p2, n_battles=n_battles)
            wr = p1.n_won_battles / n_battles * 100
            results[opp_name] = wr
            print(f"    vs {opp_name:12s}: {p1.n_won_battles}/{n_battles} = {wr:.0f}%", flush=True)
            try:
                p1.reset_battles()
            except EnvironmentError:
                pass
            try:
                p2.reset_battles()
            except EnvironmentError:
                pass
            del p1, p2
        gc.collect(); torch.cuda.empty_cache()
        results["smart_avg"] = sum(results[k] for k in ["SH", "SmartDmg", "Tactical", "Strategic"]) / 4
        return results

    return asyncio.run(_run())


def main():
    parser = argparse.ArgumentParser(description="BC training for PokeTransformer v8")
    parser.add_argument("--memmap-dir", required=True, help="Path to memmap directory")
    parser.add_argument("--format", default="gen9ou", help="Battle format (gen9ou, gen8ou, etc.)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    # Defaults match Metamon's IL recipe (il/train.py): WD 1e-4, grad clip 2.0.
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--sched", choices=["cosine", "constant", "psppo"], default="cosine",
                        help="LR schedule: cosine decay, constant, or psppo (warmup->hold->power-law)")
    parser.add_argument("--hold-steps", type=int, default=20000,
                        help="Steps to hold at full LR before decay (psppo schedule only)")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--lr-restart", action="store_true",
                        help="Reset optimizer + scheduler on resume (keep weights only). "
                             "Use when changing LR/weight_decay from a previous run.")
    parser.add_argument("--eval-games", type=int, default=200,
                        help="Games per bot for epoch-end eval (0 to disable)")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="Save mid-epoch checkpoint every N batches (increase for 30M+ scaling)")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable mixed precision training (AMP) for ~2x speedup on CUDA")
    parser.add_argument("--server", type=str, default="ws://127.0.0.1:9000/showdown/websocket",
                        help="Battle server URL for bot eval")
    add_model_args(parser)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Datasets
    print(f"Loading dataset from {args.memmap_dir}")
    train_ds = MemmapDataset(args.memmap_dir, split="train", val_ratio=args.val_ratio)
    val_ds = MemmapDataset(args.memmap_dir, split="val", val_ratio=args.val_ratio)
    print(f"Train: {len(train_ds)} episodes, Val: {len(val_ds)} episodes")

    use_cuda = device.type == "cuda"
    nw = args.workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_seq, num_workers=nw,
                              pin_memory=use_cuda, drop_last=True,
                              persistent_workers=(nw > 0),
                              prefetch_factor=(2 if nw > 0 else None))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_seq, num_workers=nw,
                            pin_memory=use_cuda,
                            persistent_workers=(nw > 0),
                            prefetch_factor=(2 if nw > 0 else None))

    # Model
    cfg = config_from_args(args)
    model = PokeTransformer(cfg).to(device)
    print(f"Model: {model.count_parameters():,} params")
    # Print effective (resolved) dims so reshape runs are obvious in logs.
    print(f"Config: d_spatial={model.d_spatial}, d_temporal={model.d_temporal}, "
          f"spatial={cfg.n_spatial_layers}L, temporal={cfg.n_temporal_layers}L, "
          f"heads={cfg.n_heads}, n_summary_tokens={cfg.n_summary_tokens}, "
          f"dropout={cfg.dropout}")
    # Echo training hyperparams to log — prevents silent-config debugging pain.
    print(f"Train: lr={args.lr}, wd={args.weight_decay}, grad_clip={args.grad_clip}, "
          f"label_smoothing={args.label_smoothing}, batch_size={args.batch_size}, "
          f"epochs={args.epochs}, fp16={args.fp16}, sched={args.sched}, "
          f"warmup={args.warmup_steps}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

    # LR scheduler: warmup then constant or cosine decay
    total_batches = len(train_loader) * args.epochs
    sched_type = getattr(args, "sched", "cosine")
    hold_steps = getattr(args, "hold_steps", 20000)  # for psppo schedule
    def lr_lambda(step):
        # Warmup phase
        if step < args.warmup_steps:
            return max(0.01, float(step) / max(1, args.warmup_steps))
        if sched_type == "constant":
            return 1.0
        elif sched_type == "psppo":
            # ps-ppo schedule: warmup -> hold flat -> power-law decay 1/(8p+1)^1.5
            post_warmup = step - args.warmup_steps
            if post_warmup < hold_steps:
                return 1.0  # hold at full LR
            decay_progress = (post_warmup - hold_steps) / max(1, total_batches - args.warmup_steps - hold_steps)
            return max(0.05, 1.0 / (8.0 * decay_progress + 1.0) ** 1.5)
        else:  # cosine
            progress = (step - args.warmup_steps) / max(1, total_batches - args.warmup_steps)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    best_smart_avg = 0.0

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if args.lr_restart:
            print(f"LR restart: loaded weights only, fresh optimizer + scheduler")
        else:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = max(0, ckpt.get("epoch", -1))
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_smart_avg = ckpt.get("best_smart_avg", 0.0)
        print(f"Resumed from {args.resume}, epoch {start_epoch}, step {global_step}, "
              f"best_smart_avg={best_smart_avg:.1f}%")

    # Output
    run_name = args.run_name or f"v8_bc_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(f"data/models/bc/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(str(out_dir / "tb"))

    # Save config
    with open(str(out_dir / "config.json"), "w") as f:
        json.dump({"model_config": cfg.to_dict(), "args": vars(args)}, f, indent=2)

    # Mixed precision (AMP)
    scaler = torch.cuda.amp.GradScaler() if args.fp16 and device.type == "cuda" else None
    if scaler:
        print(f"Mixed precision: ENABLED (fp16)")

    print(f"\nTraining {args.epochs} epochs, output: {out_dir}\n")

    # Compute global epoch offset so filenames don't collide on resume
    batches_per_epoch = len(train_loader)
    epoch_offset = global_step // max(1, batches_per_epoch)  # how many epochs already done

    _current_actual_epoch = [0]  # mutable container so closure can read it

    def _save_checkpoint(n_batches: int, current_step: int):
        ae = _current_actual_epoch[0]
        ckpt = {
            "epoch": -1,  # mid-epoch
            "global_step": current_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": cfg.to_dict(),
            "best_val_loss": best_val_loss,
        }
        torch.save(ckpt, str(out_dir / f"mid_step{current_step}_epoch{ae}_batch{n_batches}.pt"))

    for epoch in range(start_epoch, args.epochs):
        actual_epoch = epoch + epoch_offset
        _current_actual_epoch[0] = actual_epoch
        t0 = time.time()
        train_loss, train_acc, train_vloss, global_step = train_one_epoch(
            model, train_loader, optimizer, device, epoch, tb, global_step,
            grad_clip=args.grad_clip, label_smoothing=args.label_smoothing,
            scheduler=scheduler, save_fn=_save_checkpoint, save_every=args.save_every,
            scaler=scaler,
        )
        elapsed = time.time() - t0

        # Validate
        val_loss, val_acc, val_vloss = validate(model, val_loader, device)

        print(f"Epoch {actual_epoch} (local {epoch}): train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
              f"v_loss={val_vloss:.4f} time={elapsed:.0f}s")

        tb.add_scalar("epoch/train_loss", train_loss, actual_epoch)
        tb.add_scalar("epoch/train_acc", train_acc, actual_epoch)
        tb.add_scalar("epoch/val_loss", val_loss, actual_epoch)
        tb.add_scalar("epoch/val_acc", val_acc, actual_epoch)
        tb.add_scalar("epoch/val_vloss", val_vloss, actual_epoch)
        tb.add_scalar("epoch/lr", optimizer.param_groups[0]["lr"], actual_epoch)

        # Bot eval (the real metric)
        smart_avg = 0.0
        bot_results = {}
        if args.eval_games > 0:
            print(f"  Running bot eval ({args.eval_games} games per bot)...", flush=True)
            # Save temp checkpoint for eval
            temp_ckpt_path = str(out_dir / "_eval_temp.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": cfg.to_dict(),
            }, temp_ckpt_path)
            try:
                eval_replay_dir = str(out_dir / f"replays_epoch{actual_epoch:03d}")
                bot_results = eval_vs_bots(
                    temp_ckpt_path, device=str(device),
                    n_battles=args.eval_games, server_url=args.server,
                    replay_dir=eval_replay_dir,
                )
                smart_avg = bot_results.get("smart_avg", 0.0)
                print(f"  Bot eval: smart_avg={smart_avg:.1f}%", flush=True)
                for k in ["SH", "SmartDmg", "Tactical", "Strategic"]:
                    if k in bot_results:
                        tb.add_scalar(f"eval/{k}", bot_results[k], actual_epoch)
                tb.add_scalar("eval/smart_avg", smart_avg, actual_epoch)
            except Exception as e:
                print(f"  [WARN] Bot eval failed: {e}", flush=True)
            finally:
                try:
                    os.remove(temp_ckpt_path)
                except OSError:
                    pass

        # Save checkpoint
        ckpt = {
            "epoch": actual_epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": cfg.to_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "smart_avg": smart_avg,
            "bot_results": bot_results,
            "best_val_loss": best_val_loss,
            "best_smart_avg": best_smart_avg,
        }

        # Save best by bot win rate (the metric that matters)
        if smart_avg > best_smart_avg:
            best_smart_avg = smart_avg
            ckpt["best_smart_avg"] = best_smart_avg
            torch.save(ckpt, str(out_dir / "best.pt"))
            print(f"  -> New best smart_avg={smart_avg:.1f}%, saved best.pt")
        elif args.eval_games == 0 and val_loss < best_val_loss:
            # Fallback to val_loss if bot eval is disabled
            best_val_loss = val_loss
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, str(out_dir / "best.pt"))
            print(f"  -> New best val_loss={val_loss:.4f}, saved best.pt (no bot eval)")

        torch.save(ckpt, str(out_dir / f"epoch_{actual_epoch:03d}.pt"))
        torch.save(ckpt, str(out_dir / f"step_{global_step}.pt"))

    print(f"\nTraining complete. Best smart_avg={best_smart_avg:.1f}%, best val_loss={best_val_loss:.4f}")
    print(f"Output: {out_dir}")
    tb.close()


if __name__ == "__main__":
    main()
