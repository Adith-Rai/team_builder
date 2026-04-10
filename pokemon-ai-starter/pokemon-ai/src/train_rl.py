#!/usr/bin/env python
# train_rl.py — Pure self-play PPO training with batched GPU inference.
#
# Main training loop. All infrastructure lives in separate modules:
#   inference_batcher.py — async batched GPU forward
#   rl_player.py — V9RLPlayer, SelfPlayOpponent
#   rl_collection.py — collect_v9, BackgroundCollector
#   rl_pipeline.py — multiprocess collection (InferenceServer, MPRLPlayer)
#   ppo.py — Trajectory, GAE, PPO update, checkpoint I/O
#
# Usage:
#   python -u train_rl.py \
#     --init-from data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt \
#     --device cuda --servers 9000,9001 --fp16 \
#     --games-per-iter 200 --max-concurrent 20 --n-iters 500

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

from model import PokeTransformer, PokeTransformerConfig, add_model_args
from ppo import (
    Trajectory, compute_gae, build_ppo_episodes, ppo_update,
    load_checkpoint, save_checkpoint,
)
from rewards import RewardShaper
from teams_ou import random_pool_teambuilder
from team_generator import ProceduralTeambuilder, procedural_teambuilder
from rl_collection import _make_server, collect_v9, BackgroundCollector


# =============================
# Argument parsing
# =============================

def parse_args():
    p = argparse.ArgumentParser(description="Self-Play PPO with Batched Inference")
    p.add_argument("--init-from", required=True, help="Init checkpoint (e.g. iter80)")
    p.add_argument("--resume", default=None, help="Resume from checkpoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--opponent-device", default="cuda")
    p.add_argument("--servers", default="9000", help="Comma-separated ports")
    p.add_argument("--format", default="gen9ou", help="Battle format (gen9ou, gen8ou, etc.)")
    p.add_argument("--games-per-iter", type=int, default=200)
    p.add_argument("--max-concurrent", type=int, default=20)
    p.add_argument("--n-iters", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.9999)
    p.add_argument("--lam", type=float, default=0.75)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=5)
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--adaptive-entropy", action="store_true",
                   help="Auto-adjust ent_coef to keep entropy in [0.55, 0.80] range")
    p.add_argument("--vf-coef", type=float, default=1.0)
    p.add_argument("--target-kl", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--grad-accum", type=int, default=10,
                   help="Accumulate gradients over N episodes before each optimizer step")
    p.add_argument("--warmup-iters", type=int, default=5)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--ko-coef", type=float, default=0.05)
    p.add_argument("--hp-coef", type=float, default=0.02)
    p.add_argument("--reward-clip", type=float, default=2.0)
    p.add_argument("--temp-min", type=float, default=1.0, help="Opponent temp range min")
    p.add_argument("--temp-max", type=float, default=2.25, help="Opponent temp range max")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile spatial encoder (Linux only)")
    p.add_argument("--pipeline", action="store_true",
                   help="Pipeline collection and PPO update (overlap on GPU)")
    p.add_argument("--snapshot-interval", type=int, default=5, help="Save snapshot every N iters")
    p.add_argument("--eval-interval", type=int, default=20)
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--out-dir", default="data/models/rl_v9")
    p.add_argument("--immune-penalty", type=float, default=0.0,
                   help="Per-step penalty when our move hits immunity")
    p.add_argument("--procedural-teams", default=None,
                   help="Path to Smogon usage stats dir for procedural team generation")
    p.add_argument("--random-team-pct", type=float, default=0.05,
                   help="Fraction of procedural teams with uniform weights")
    p.add_argument("--lr-restart", action="store_true",
                   help="Reset optimizer on resume (use when dims/hyperparams changed)")
    p.add_argument("--mp", action="store_true",
                   help="Use multiprocess collection (workers on CPU, GPU inference centralized)")
    p.add_argument("--batch-timeout-ms", type=float, default=15,
                   help="InferenceBatcher batch timeout in ms")
    p.add_argument("--reward-style", choices=["dense", "sparse", "terminal"], default="dense",
                   help="Reward shaping style: dense (KO+HP+terminal), sparse (terminal+immune), terminal (win/loss only)")
    add_model_args(p)
    return p.parse_args()


# =============================
# Setup helpers
# =============================

def _build_reward_config(args) -> dict:
    """Build reward shaper config dict from args."""
    style = getattr(args, 'reward_style', 'dense')
    if style == 'dense':
        cfg = {"ko_coef": args.ko_coef, "hp_coef": args.hp_coef,
               "clip_abs": args.reward_clip, "immune_penalty": args.immune_penalty}
    elif style == 'sparse':
        cfg = {"ko_coef": 0.0, "hp_coef": 0.0,
               "clip_abs": args.reward_clip, "immune_penalty": args.immune_penalty}
    elif style == 'terminal':
        cfg = {"ko_coef": 0.0, "hp_coef": 0.0,
               "clip_abs": args.reward_clip, "immune_penalty": 0.0}
    else:
        raise ValueError(f"Unknown reward_style: {style}")
    print(f"Reward style: {style} ({cfg})", flush=True)
    return cfg


def _resume_from_checkpoint(args, model, optimizer, snapshot_pool, device):
    """Load model/optimizer state from resume checkpoint. Returns start_iter."""
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    resume_state = ckpt["model_state_dict"]

    # Handle dim expansion for checkpoints from before type_eff features
    _expand_targets = ["move_net.mlp.0.weight", "switch_mlp.0.weight"]
    for key in list(resume_state.keys()):
        if any(key.endswith(t) for t in _expand_targets):
            old_w = resume_state[key]
            parts = key.split(".")
            mod = model
            for p in parts[:-1]:
                mod = getattr(mod, p) if not p.isdigit() else mod[int(p)]
            expected_in = mod.in_features
            if old_w.shape[1] < expected_in:
                pad = expected_in - old_w.shape[1]
                resume_state[key] = torch.cat([old_w, torch.zeros(old_w.shape[0], pad, device=old_w.device)], dim=1)
                print(f"  [INFO] Expanding {key}: {old_w.shape[1]} -> {expected_in} (+{pad} dims, zero-init)")

    model.load_state_dict(resume_state)
    if args.lr_restart:
        print("  [INFO] --lr-restart: optimizer reset (fresh Adam state)")
    else:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    start_iter = ckpt.get("iteration", 0) + 1
    pool = ckpt.get("metrics", {}).get("snapshot_pool", snapshot_pool)

    # Scan disk for ALL existing snapshots and add to pool
    import glob as _glob, re as _re
    # Snapshots before iter 260 are from the pre-type-effectiveness era
    # (eval 25-44%, suboptimal play). Including them corrupts the value function.
    MIN_SNAPSHOT_ITER = 260
    all_disk = sorted(set(_glob.glob("data/models/rl_v9/selfplay_v9_*/snapshot_*.pt")))
    def _snap_iter(path):
        m = _re.search(r'snapshot_(\d+)\.pt$', path)
        return int(m.group(1)) if m else 0
    all_disk = [s for s in all_disk if _snap_iter(s) >= MIN_SNAPSHOT_ITER]
    existing = set(pool)
    new_snaps = [s for s in all_disk if s not in existing]
    if new_snaps:
        pool = new_snaps + pool
    print(f"Resumed from {args.resume}, starting at iter {start_iter}, "
          f"pool: {len(pool)} checkpoints (+{len(new_snaps)} from disk scan, "
          f"filtered sp<{MIN_SNAPSHOT_ITER})", flush=True)

    return start_iter, pool


# =============================
# Per-iter step helpers
# =============================

def _collect_data(args, model, device, server_pool, snapshot_pool,
                  rs_cfg, train_teambuilder, battle_format,
                  loop, pending_collection, _flow):
    """Run one collection step. Returns (trajs, wins, losses, ties, steps, opp_name, collect_time)."""
    if pending_collection is not None:
        _flow("using pre-collected data from background")
        result = pending_collection
        _flow(f"unpacked pre-collected: {len(result[0])} trajs, {result[4]} steps")
        return result

    if getattr(args, 'mp', False):
        from mp_collect_v2 import mp_collect_v2
        model.eval()
        latest_sp = snapshot_pool[-1] if len(snapshot_pool) > 1 else None
        return mp_collect_v2(
            model, device, server_pool,
            n_games=args.games_per_iter,
            max_concurrent=args.max_concurrent,
            snapshot_pool=snapshot_pool,
            fp16=args.fp16,
            reward_shaper_cfg=rs_cfg,
            temp_range=(args.temp_min, args.temp_max),
            latest_snapshot=latest_sp,
            teambuilder_path=getattr(args, 'procedural_teams', None),
            opponent_device=args.opponent_device,
            batch_timeout_ms=args.batch_timeout_ms,
        )

    _flow("starting SYNC collection")
    model.eval()
    latest_sp = snapshot_pool[-1] if len(snapshot_pool) > 1 else None
    result = loop.run_until_complete(
        collect_v9(
            model, device, server_pool,
            n_games=args.games_per_iter,
            max_concurrent=args.max_concurrent,
            snapshot_pool=snapshot_pool,
            fp16=args.fp16,
            reward_shaper_cfg=rs_cfg,
            temp_range=(args.temp_min, args.temp_max),
            opponent_device=args.opponent_device,
            latest_snapshot=latest_sp,
            teambuilder=train_teambuilder,
            battle_format=battle_format,
        )
    )
    _flow(f"sync collection done: {result[6]:.0f}s, {len(result[0])} trajs")
    return result


def _start_background_collection(args, model, device, server_pool, snapshot_pool,
                                  collect_args, bg_collector, mp_bg_collector,
                                  in_warmup, _flow):
    """Kick off background collection for the NEXT iteration (pipeline mode)."""
    if args.mp and args.pipeline and not in_warmup:
        from mp_collect_v2 import MPPipelineCollector
        if mp_bg_collector is None:
            mp_bg_collector = MPPipelineCollector()
        mp_collect_args = {
            "games_per_iter": args.games_per_iter,
            "max_concurrent": args.max_concurrent,
            "fp16": args.fp16,
            "rs_cfg": collect_args["rs_cfg"],
            "temp_range": collect_args["temp_range"],
            "teambuilder_path": getattr(args, 'procedural_teams', None),
            "opponent_device": collect_args["opponent_device"],
            "batch_timeout_ms": args.batch_timeout_ms,
        }
        mp_bg_collector.start(model, device, server_pool, snapshot_pool, mp_collect_args)
    elif bg_collector and not in_warmup and not args.mp:
        _flow("starting BACKGROUND collection for next iter")
        bg_collector.start(model, device, server_pool, snapshot_pool, collect_args)
    return mp_bg_collector


def _join_background(bg_collector, mp_bg_collector, _flow):
    """Wait for background collection to finish. Returns pending_collection or None."""
    if mp_bg_collector is not None and getattr(mp_bg_collector, 'running', False):
        _flow("waiting for MP background collection")
        result = mp_bg_collector.join()
        _flow(f"MP background done, result={'OK' if result else 'NONE'}")
        return result
    if bg_collector and bg_collector.running:
        _flow("waiting for background collection")
        result = bg_collector.join()
        _flow(f"background done, result={'OK' if result else 'NONE'}")
        return result
    if bg_collector and not bg_collector.running and bg_collector._result is not None:
        _flow("background ALREADY DONE (good overlap!)")
        return bg_collector.join()
    return None


def _log_iter(writer, it, wins, losses, ties, steps, collect_time, update_time,
              loss_info, opp_name, snapshot_pool, in_warmup):
    """Print iter summary and write TensorBoard scalars."""
    total_games = wins + losses + ties
    wr = wins / max(1, total_games)
    kl_str = f" kl={loss_info['kl']:.4f}" if 'kl' in loss_info else ""
    warmup_str = " [WARMUP]" if in_warmup else ""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] Iter {it}: W/L/T={wins}/{losses}/{ties} ({wr:.1%}), {steps} steps, "
          f"collect={collect_time:.0f}s, update={update_time:.0f}s, "
          f"pi={loss_info['pi']:.4f} v={loss_info['v']:.4f} "
          f"ent={loss_info['ent']:.4f}{kl_str}{warmup_str} "
          f"vs={opp_name} pool={len(snapshot_pool)}",
          flush=True)

    writer.add_scalar("train/win_rate", wr, it)
    writer.add_scalar("train/pi_loss", loss_info["pi"], it)
    writer.add_scalar("train/v_loss", loss_info["v"], it)
    writer.add_scalar("train/entropy", loss_info["ent"], it)
    if "kl" in loss_info:
        writer.add_scalar("train/kl", loss_info["kl"], it)
    writer.add_scalar("train/collect_time", collect_time, it)
    writer.add_scalar("train/update_time", update_time, it)
    writer.add_scalar("train/steps", steps, it)
    writer.add_scalar("train/pool_size", len(snapshot_pool), it)
    return wr


def _maybe_save_snapshot(it, args, model, cfg, optimizer, steps, loss_info,
                         wr, best_eval_wr, snapshot_pool, run_dir):
    """Save snapshot if interval reached and iter is clean."""
    if (it + 1) % args.snapshot_interval != 0:
        return
    if steps < 100:
        print(f"  Snapshot SKIPPED: only {steps} steps (min 100 required)", flush=True)
    elif loss_info.get("n_succeeded", 1) == 0:
        print(f"  Snapshot SKIPPED: 0 PPO episodes succeeded (tainted iter)", flush=True)
    else:
        sp_path = str(run_dir / f"snapshot_{it:04d}.pt")
        save_checkpoint(sp_path, model, cfg, optimizer, it, metrics={
            "win_rate": wr, "best_eval_wr": best_eval_wr,
            "snapshot_pool": snapshot_pool[-500:],
        })
        snapshot_pool.append(sp_path)
        print(f"  Snapshot saved: {sp_path} (pool={len(snapshot_pool)})", flush=True)


def _maybe_eval(it, args, model, cfg, optimizer, device, writer, run_dir,
                best_eval_wr, battle_format):
    """Run bot evaluation if interval reached. Returns updated best_eval_wr."""
    if (it + 1) % args.eval_interval != 0:
        return best_eval_wr
    try:
        tmp = str(run_dir / f"iter_{it:04d}.pt")
        save_checkpoint(tmp, model, cfg, optimizer, it)

        from train_bc import eval_vs_bots
        srv_url = f"ws://127.0.0.1:{args.servers.split(',')[0].strip()}/showdown/websocket"
        replay_path = str(run_dir / f"replays_iter{it:04d}")
        results = eval_vs_bots(tmp, device=str(device), n_battles=args.eval_games,
                               server_url=srv_url, replay_dir=replay_path,
                               battle_format=battle_format)
        sh = results.get("SH", 0)
        smd = results.get("SmartDmg", results.get("SmD", 0))
        tac = results.get("Tactical", results.get("Tac", 0))
        stra = results.get("Strategic", results.get("Str", 0))
        smart_avg = (sh + smd + tac + stra) / 4

        print(f"  EVAL: SH={sh:.0f}%, SmartDmg={smd:.0f}%, Tactical={tac:.0f}%, "
              f"Strategic={stra:.0f}%, smart_avg={smart_avg:.0f}%", flush=True)

        writer.add_scalar("eval/smart_avg", smart_avg, it)
        writer.add_scalar("eval/SH", sh, it)
        writer.add_scalar("eval/SmartDmg", smd, it)
        writer.add_scalar("eval/Tactical", tac, it)
        writer.add_scalar("eval/Strategic", stra, it)

        # Persist to registry (fire-and-forget)
        from registry import log_eval
        log_eval(it, str(run_dir), sh, smd, tac, stra, smart_avg)

        if smart_avg > best_eval_wr:
            best_eval_wr = smart_avg
    except Exception as e:
        print(f"  [ERROR] Eval failed: {e}", flush=True)
    return best_eval_wr


# =============================
# Main training loop
# =============================

def main():
    args = parse_args()
    device = torch.device(args.device)
    battle_format = args.format

    # Load model
    model, cfg, _ = load_checkpoint(args.init_from, device)
    model.to(device)

    # torch.compile (Linux/cloud only)
    compiled = False
    if args.compile:
        try:
            model.forward_spatial = torch.compile(model.forward_spatial, mode="reduce-overhead")
            compiled = True
            print("torch.compile: spatial encoder compiled successfully", flush=True)
        except Exception as e:
            print(f"torch.compile: SKIPPED ({e})", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Run directory + TensorBoard
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / f"selfplay_v9_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    # Infrastructure
    server_pool = [_make_server(s.strip()) for s in args.servers.split(",")]
    snapshot_pool = [args.init_from]
    rs_cfg = _build_reward_config(args)

    # Team builder (procedural for training, handcrafted for eval)
    train_teambuilder = (procedural_teambuilder(args.procedural_teams, random_pct=args.random_team_pct)
                         if args.procedural_teams else None)

    # Save config
    config = vars(args)
    config["run_dir"] = str(run_dir)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Resume
    start_iter = 0
    if args.resume:
        start_iter, snapshot_pool = _resume_from_checkpoint(
            args, model, optimizer, snapshot_pool, device)

    loop = asyncio.new_event_loop()

    # Print banner
    print(f"\n=== Self-Play PPO Training ===")
    print(f"Init: {args.init_from} | Format: {battle_format} | Run: {run_dir}")
    print(f"Iters: {args.n_iters}, Games/iter: {args.games_per_iter}, Concurrent: {args.max_concurrent}")
    print(f"gamma={args.gamma}, lam={args.lam}, ent={args.ent_coef}, target_kl={args.target_kl}, grad_accum={args.grad_accum}")
    print(f"FP16: {'ON' if args.fp16 else 'OFF'}, Compile: {'ON' if compiled else 'OFF'}, "
          f"Pipeline: {'ON' if args.pipeline else 'OFF'}, Device: {device}")
    print(f"Snapshot pool: {len(snapshot_pool)} checkpoints\n", flush=True)

    # Register this run (fire-and-forget)
    from registry import log_run
    log_run(str(run_dir), config, start_iter, start_iter + args.n_iters - 1)

    # Training state
    best_eval_wr = 0.0
    ent_coef = args.ent_coef
    bg_collector = BackgroundCollector(cpu_inference=False) if args.pipeline else None
    collect_args = {
        "games_per_iter": args.games_per_iter,
        "max_concurrent": args.max_concurrent,
        "fp16": args.fp16,
        "rs_cfg": rs_cfg,
        "temp_range": (args.temp_min, args.temp_max),
        "opponent_device": args.opponent_device,
        "teambuilder": train_teambuilder,
    }
    pending_collection = None
    mp_bg_collector = None

    # ---- Training loop ----
    for it in range(start_iter, start_iter + args.n_iters):
        _flow_t0 = time.time()
        def _flow(msg):
            elapsed = time.time() - _flow_t0
            print(f"  [FLOW {datetime.now().strftime('%H:%M:%S')} +{elapsed:6.1f}s] {msg}", flush=True)
        _flow("iter start")

        # Value warmup (freeze backbone+policy, train only value head)
        in_warmup = (it - start_iter) < args.warmup_iters
        if in_warmup:
            for name, param in model.named_parameters():
                param.requires_grad = "value_head" in name
        elif (it - start_iter) == args.warmup_iters:
            for param in model.parameters():
                param.requires_grad = True
            print(f"  Value warmup complete, unfreezing all parameters", flush=True)

        # ---- Collect ----
        trajs, wins, losses, ties, steps, opp_name, collect_time = _collect_data(
            args, model, device, server_pool, snapshot_pool,
            rs_cfg, train_teambuilder, battle_format,
            loop, pending_collection, _flow)
        pending_collection = None
        wr = wins / max(1, wins + losses + ties)

        # ---- Start background collection for next iter ----
        mp_bg_collector = _start_background_collection(
            args, model, device, server_pool, snapshot_pool,
            collect_args, bg_collector, mp_bg_collector, in_warmup, _flow)

        # ---- PPO Update ----
        _flow("building PPO episodes")
        episodes = build_ppo_episodes(trajs, gamma=args.gamma, lam=args.lam)
        _flow(f"PPO episodes built: {len(episodes)} episodes")

        model.train()
        if in_warmup:
            for name, param in model.named_parameters():
                param.requires_grad = "value_head" in name

        _flow("starting PPO update")
        t_update = time.time()
        loss_info = ppo_update(
            model, optimizer, episodes, device, cfg,
            epochs=args.ppo_epochs, clip_eps=args.clip_eps,
            ent_coef=ent_coef, vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm, target_kl=args.target_kl,
            grad_accum=args.grad_accum,
        )
        update_time = time.time() - t_update
        _flow(f"PPO update DONE: {update_time:.0f}s")

        # ---- Catastrophic-failure guard (Session 33) ----
        if loss_info.get("n_succeeded", 1) == 0:
            print(f"  [FATAL] PPO update: 0 succeeded ({loss_info.get('n_failed', '?')} failed, "
                  f"{len(episodes)} episodes). Saving emergency checkpoint.", flush=True)
            try:
                emerg = str(run_dir / f"emergency_iter_{it:04d}.pt")
                save_checkpoint(emerg, model, cfg, optimizer, it, metrics={
                    "win_rate": wr, "snapshot_pool": snapshot_pool[-500:]})
                print(f"  [FATAL] Saved: {emerg}", flush=True)
            except Exception as e:
                print(f"  [FATAL] Save failed: {e}", flush=True)
            writer.close()
            sys.exit(2)

        # ---- Wait for background collection ----
        pending_collection = _join_background(bg_collector, mp_bg_collector, _flow)

        # ---- Log + TensorBoard ----
        wr = _log_iter(writer, it, wins, losses, ties, steps, collect_time, update_time,
                       loss_info, opp_name, snapshot_pool, in_warmup)

        # ---- Adaptive entropy ----
        if args.adaptive_entropy and loss_info["ent"] > 0.01:
            if loss_info["ent"] < 0.55:
                ent_coef = min(ent_coef * 1.05, 0.06)
                print(f"  [ENT] Low ({loss_info['ent']:.3f}), ent_coef → {ent_coef:.4f}")
            elif loss_info["ent"] > 0.80:
                ent_coef = max(ent_coef * 0.95, 0.01)
                print(f"  [ENT] High ({loss_info['ent']:.3f}), ent_coef → {ent_coef:.4f}")

        # ---- Snapshot + Eval ----
        _maybe_save_snapshot(it, args, model, cfg, optimizer, steps, loss_info,
                             wr, best_eval_wr, snapshot_pool, run_dir)
        best_eval_wr = _maybe_eval(it, args, model, cfg, optimizer, device, writer,
                                    run_dir, best_eval_wr, battle_format)

        # Memory cleanup
        del trajs, episodes
        gc.collect()
        torch.cuda.empty_cache()

    # Final save
    final_path = str(run_dir / "final.pt")
    save_checkpoint(final_path, model, cfg, optimizer, start_iter + args.n_iters - 1,
                    metrics={"best_eval_wr": best_eval_wr, "snapshot_pool": snapshot_pool[-500:]})
    print(f"\nTraining complete. Final checkpoint: {final_path}", flush=True)
    writer.close()
    loop.close()


if __name__ == "__main__":
    main()
