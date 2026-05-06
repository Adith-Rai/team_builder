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

# Set sharing strategy BEFORE any other imports that touch torch.multiprocessing.
# file_system uses ref-counted /tmp files instead of POSIX shm_open per tensor;
# avoids vm.max_map_count exhaustion on linux containers (default cap 65530)
# under high-volume tensor IPC. RunPod containers don't allow sysctl bumps.
import torch.multiprocessing as _mp_train
try:
    _mp_train.set_sharing_strategy('file_system')
except Exception:
    pass

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
    p.add_argument("--init-from", default=None,
                   help="Init checkpoint (e.g. iter80). Optional when --resume is provided; "
                        "the resume checkpoint is used as the init source in that case.")
    p.add_argument("--resume", default=None, help="Resume from checkpoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--opponent-device", default="cuda")
    p.add_argument("--servers", default="9000", help="Comma-separated ports")
    p.add_argument("--format", default="gen9ou", help="Battle format (gen9ou, gen8ou, etc.)")
    p.add_argument("--games-per-iter", type=int, default=200)
    p.add_argument("--max-concurrent", type=int, default=20)
    p.add_argument("--n-iters", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-5,
                   help="Adam learning rate. Default 3e-5 — the value S39 used "
                        "to set the smart_avg-64% record (sp_0229). Default was "
                        "1e-4 historically, but that consistently caused KL "
                        "early-stop on every iter, ~10%% per-episode KL discards, "
                        "and policy drift from sharp PPO checkpoints (observed in "
                        "S39 from a 45%% BC base, and again in S43's first attempt "
                        "from sp_0229). 3e-5 is safe for both BC->PPO transitions "
                        "and PPO->PPO continuation. Override at your own risk.")
    p.add_argument("--gamma", type=float, default=0.9999)
    p.add_argument("--lam", type=float, default=0.75)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=5)
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--adaptive-entropy", action="store_true",
                   help="Auto-adjust ent_coef to keep entropy in [low, high] range")
    p.add_argument("--adaptive-entropy-low", type=float, default=0.65,
                   help="Raise ent_coef when entropy falls below this (default: 0.65, was 0.55)")
    p.add_argument("--adaptive-entropy-high", type=float, default=0.95,
                   help="Lower ent_coef when entropy exceeds this (default: 0.95, was 0.80)")
    p.add_argument("--adaptive-entropy-max", type=float, default=0.08,
                   help="Cap for ent_coef under adaptive adjustment (default: 0.08)")
    p.add_argument("--adaptive-entropy-min", type=float, default=0.01,
                   help="Floor for ent_coef under adaptive adjustment (default: 0.01)")
    p.add_argument("--adaptive-entropy-step", type=float, default=0.1,
                   help="Per-iter multiplicative change to ent_coef (default: 0.1 = ±10%)")
    # Early stopping (composite: savg + per-bot consensus)
    p.add_argument("--early-stop", action="store_true",
                   help="Enable composite early stopping based on eval regression")
    p.add_argument("--early-stop-patience", type=int, default=3,
                   help="Consecutive regressing evals required to stop (default: 3)")
    p.add_argument("--early-stop-savg-threshold", type=float, default=2.0,
                   help="Minimum savg regression (percent) from best rm3 to count (default: 2.0)")
    p.add_argument("--early-stop-bot-threshold", type=float, default=3.0,
                   help="Minimum per-bot regression (percent) from best rm3 to count (default: 3.0)")
    p.add_argument("--early-stop-bot-count", type=int, default=3,
                   help="How many of 4 bots must regress (default: 3)")
    p.add_argument("--early-stop-min-evals", type=int, default=5,
                   help="Minimum eval points before checking stop condition (default: 5)")
    # PFSP win-rate tracking mode
    p.add_argument("--win-rate-mode", choices=["cumulative", "ema"], default="cumulative",
                   help="How PFSP tracks opponent win rates. cumulative=all history (default), "
                        "ema=exponential moving average (forgets old data, fixes staleness)")
    p.add_argument("--win-rate-ema-alpha", type=float, default=0.3,
                   help="EMA blend weight for new encounters (default: 0.3). Only used with --win-rate-mode=ema")
    p.add_argument("--win-rate-ema-window", type=int, default=50,
                   help="Cap on effective_games in EMA mode (default: 50). "
                        "Prevents unbounded growth and ensures old data fades.")
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
    p.add_argument("--eval-team-set", choices=["pool", "metamon-competitive"], default="pool",
                   help="Team source for in-training bot evals. 'pool' = 70-team "
                        "teams_ou pool (legacy default; ~30pt strength spread → "
                        "noisy smart_avg). 'metamon-competitive' = 16 curated "
                        "Smogon teams from metamon_cache (lower team-quality "
                        "variance, ladder-validated, ~3.6pt same-policy noise "
                        "floor at 200×4 games).")
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
    p.add_argument("--external-adapters", default=None,
                   help="Path to external_adapters.yaml — adds in-process opponent "
                        "adapters (e.g. PokeEnginePlayer) to the PFSP pool")
    # Pool curation (Session 44 — anti-dilution). Both default to old behavior.
    p.add_argument("--pool-anchors", default="",
                   help="Comma-separated paths to fixed anchor checkpoints kept in the "
                        "PFSP pool throughout training (e.g. peak-era references). "
                        "Always present; never pruned. Default empty = old behavior.")
    p.add_argument("--pool-max-current-run", type=int, default=-1,
                   help="Cap on number of self-play snapshots from the CURRENT run "
                        "kept in the pool. When N>=0 and the current run has produced "
                        "more than N snapshots, the oldest ones are dropped from the "
                        "pool (still saved on disk). Anchor checkpoints and the init "
                        "checkpoint are not affected. Default -1 = unbounded (old "
                        "behavior — caused S43/S44 dilution).")
    # Memory: per-battle turn cap. New arch's per-attribute tokenization makes
    # PPO's per-episode forward over T turns scale ~quadratically (T=45 ≈ 1.7 GB,
    # T=200 ≈ 8 GB on a 6 GB GPU). Lowering turn_cap on local; cloud can keep 300.
    p.add_argument("--turn-cap", type=int, default=300,
                   help="Per-battle turn cap before forfeit. Local 6 GB GPU: 200. "
                        "Cloud 80 GB: 300 (default).")
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

    # Normalize all pool paths to forward slashes (fixes Windows \/  duplicates)
    pool = [p.replace("\\", "/") for p in pool]

    # Scan disk for ALL existing snapshots and add to pool
    import glob as _glob, re as _re
    # Snapshots before iter 260 are from the pre-type-effectiveness era
    # (eval 25-44%, suboptimal play). Including them corrupts the value function.
    MIN_SNAPSHOT_ITER = 260
    all_disk = sorted(set(_glob.glob("data/models/rl_v9/selfplay_v9_*/snapshot_*.pt")))
    all_disk = [p.replace("\\", "/") for p in all_disk]
    def _snap_iter(path):
        m = _re.search(r'snapshot_(\d+)\.pt$', path)
        return int(m.group(1)) if m else 0
    all_disk = [s for s in all_disk if _snap_iter(s) >= MIN_SNAPSHOT_ITER]
    existing = set(pool)
    new_snaps = [s for s in all_disk if s not in existing]
    if new_snaps:
        pool = new_snaps + pool

    # Deduplicate (same file, different path variants)
    seen = set()
    deduped = []
    for p in pool:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    n_dupes = len(pool) - len(deduped)
    pool = deduped

    print(f"Resumed from {args.resume}, starting at iter {start_iter}, "
          f"pool: {len(pool)} checkpoints (+{len(new_snaps)} from disk scan, "
          f"filtered sp<{MIN_SNAPSHOT_ITER})"
          f"{f', removed {n_dupes} path duplicates' if n_dupes else ''}", flush=True)

    return start_iter, pool


# =============================
# Per-iter step helpers
# =============================

def _collect_data(args, model, device, server_pool, snapshot_pool,
                  rs_cfg, train_teambuilder, battle_format,
                  loop, pending_collection, _flow, win_rates=None,
                  external_manager=None):
    """Run one collection step. Returns (trajs, wins, losses, ties, steps, opp_name, collect_time, opp_records)."""
    if pending_collection is not None:
        _flow("using pre-collected data from background")
        result = pending_collection
        _flow(f"unpacked pre-collected: {len(result[0])} trajs, {result[4]} steps")
        return result

    if getattr(args, 'mp', False):
        from mp_collect_v2 import mp_collect_v2
        model.eval()
        latest_sp = snapshot_pool[-1] if len(snapshot_pool) > 1 else None
        mp_result = mp_collect_v2(
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
        # mp_collect_v2 returns 7-tuple; add empty opp_records for compatibility
        return mp_result + ({},)

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
            win_rates=win_rates,
            external_manager=external_manager,
            turn_cap=args.turn_cap,
        )
    )
    _flow(f"sync collection done: {result[6]:.0f}s, {len(result[0])} trajs")
    return result


def _start_background_collection(args, model, device, server_pool, snapshot_pool,
                                  collect_args, bg_collector, mp_bg_collector,
                                  in_warmup, _flow, external_manager=None):
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
        bg_collector.start(model, device, server_pool, snapshot_pool, collect_args,
                           win_rates=collect_args.get("win_rates"),
                           external_manager=external_manager)
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
                         wr, best_eval_wr, snapshot_pool, run_dir,
                         protected_paths=None):
    """Save snapshot if interval reached and iter is clean.

    `protected_paths` (set of str) is the set of pool entries that must NEVER
    be pruned: the init checkpoint and any --pool-anchors. When
    --pool-max-current-run >= 0, the function caps the number of *unprotected*
    self-play snapshots from this run; the oldest current-run snapshots beyond
    the cap are dropped from the pool (still saved to disk).
    """
    if (it + 1) % args.snapshot_interval != 0:
        return
    if steps < 100:
        print(f"  Snapshot SKIPPED: only {steps} steps (min 100 required)", flush=True)
    elif loss_info.get("n_succeeded", 1) == 0:
        print(f"  Snapshot SKIPPED: 0 PPO episodes succeeded (tainted iter)", flush=True)
    else:
        sp_path = str(run_dir / f"snapshot_{it:04d}.pt").replace("\\", "/")
        save_checkpoint(sp_path, model, cfg, optimizer, it, metrics={
            "win_rate": wr, "best_eval_wr": best_eval_wr,
            "snapshot_pool": [s for s in snapshot_pool if isinstance(s, str)],
        })
        snapshot_pool.append(sp_path)

        # Layer-5 anti-dilution prune (Session 44). When --pool-max-current-run
        # is set, drop oldest current-run snapshots beyond the cap. Anchors and
        # init are protected.
        n_pruned = 0
        if args.pool_max_current_run >= 0 and protected_paths is not None:
            run_dir_prefix = str(run_dir).replace("\\", "/")
            current_run_idx = [
                i for i, s in enumerate(snapshot_pool)
                if isinstance(s, str)
                and s.replace("\\", "/").startswith(run_dir_prefix)
                and s.replace("\\", "/") not in protected_paths
            ]
            excess = len(current_run_idx) - args.pool_max_current_run
            if excess > 0:
                # current_run_idx is in pool order, so oldest first → drop those
                drop_indices = set(current_run_idx[:excess])
                snapshot_pool[:] = [s for i, s in enumerate(snapshot_pool)
                                    if i not in drop_indices]
                n_pruned = excess

        prune_str = f", pruned={n_pruned}" if n_pruned else ""
        print(f"  Snapshot saved: {sp_path} (pool={len(snapshot_pool)}{prune_str})",
              flush=True)


def _maybe_eval(it, args, model, cfg, optimizer, device, writer, run_dir,
                best_eval_wr, battle_format, eval_history=None):
    """Run bot evaluation if interval reached.

    Returns (updated_best_eval_wr, eval_dict or None, should_stop bool).
    should_stop is True when early stopping condition triggers.
    """
    if (it + 1) % args.eval_interval != 0:
        return best_eval_wr, None, False

    eval_dict = None
    should_stop = False
    try:
        tmp = str(run_dir / f"iter_{it:04d}.pt")
        save_checkpoint(tmp, model, cfg, optimizer, it)

        from train_bc import eval_vs_bots
        srv_url = f"ws://127.0.0.1:{args.servers.split(',')[0].strip()}/showdown/websocket"
        replay_path = str(run_dir / f"replays_iter{it:04d}")
        results = eval_vs_bots(tmp, device=str(device), n_battles=args.eval_games,
                               server_url=srv_url, replay_dir=replay_path,
                               battle_format=battle_format,
                               team_set=args.eval_team_set)
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

        eval_dict = {"iter": it, "savg": smart_avg, "SH": sh, "SmartDmg": smd,
                     "Tactical": tac, "Strategic": stra}

        # ---- Composite early stopping check ----
        if args.early_stop and eval_history is not None:
            eval_history.append(eval_dict)
            should_stop = _check_early_stop(eval_history, args)
    except Exception as e:
        print(f"  [ERROR] Eval failed: {e}", flush=True)
    return best_eval_wr, eval_dict, should_stop


def _check_early_stop(eval_history, args):
    """Composite early stopping: requires BOTH savg AND multi-bot regression.

    Best = max rolling-3 mean from history (smoothed baseline, noise-resistant).
    Stop if the LAST `patience` RAW evals are ALL below best by threshold,
    AND at least `bot_count` of 4 bots are regressing on each of those evals.

    A single bad eval followed by recovery won't trigger (raw check resets).
    Sustained degradation across multiple evals and multiple bots triggers.
    """
    if len(eval_history) < args.early_stop_min_evals:
        return False  # not enough data

    bots = ["SH", "SmartDmg", "Tactical", "Strategic"]

    def rm3(history, key, i):
        start = max(0, i - 2)
        window = history[start:i + 1]
        return sum(e[key] for e in window) / len(window)

    n = len(eval_history)
    rm3_savg = [rm3(eval_history, "savg", i) for i in range(n)]
    rm3_bots = {b: [rm3(eval_history, b, i) for i in range(n)] for b in bots}

    best_savg = max(rm3_savg)
    best_bots = {b: max(rm3_bots[b]) for b in bots}

    patience = args.early_stop_patience
    if n < patience:
        return False

    savg_th = args.early_stop_savg_threshold
    bot_th = args.early_stop_bot_threshold
    bot_cnt = args.early_stop_bot_count

    # Use RAW recent evals for stop trigger (rolling baseline for best).
    # Trigger if EITHER:
    #   (a) savg regressed by threshold AND `bot_cnt` of 4 bots regressing, OR
    #   (b) savg regressed severely (>2x threshold) — handles specialization cases
    #       where 2 bots tank while 2 improve (net bad but doesn't meet bot consensus).
    for i in range(n - patience, n):
        raw = eval_history[i]
        raw_savg_bad = raw["savg"] < best_savg - savg_th
        savg_very_bad = raw["savg"] < best_savg - (2 * savg_th)
        bot_bad_count = sum(1 for b in bots if raw[b] < best_bots[b] - bot_th)
        consensus_bad = raw_savg_bad and (bot_bad_count >= bot_cnt)
        combined_bad = consensus_bad or savg_very_bad
        if not combined_bad:
            return False

    # All `patience` recent raw evals show degradation on both signals
    last_savg = eval_history[-1]["savg"]
    print(f"  [EARLY STOP] {patience} consecutive raw evals show savg regression >{savg_th:.1f}% "
          f"AND >={bot_cnt} bots regressing >{bot_th:.1f}%. "
          f"Best rm3_savg={best_savg:.1f}, last raw savg={last_savg:.1f}", flush=True)
    return True


# =============================
# Main training loop
# =============================

def main():
    args = parse_args()
    device = torch.device(args.device)
    battle_format = args.format

    # Resolve initial checkpoint source. Require at least one of --init-from / --resume
    # (previously --init-from was always required; making it fallback-friendly so you
    # can resume sp2979-style runs without passing a BC checkpoint path).
    init_path = args.init_from or args.resume
    if init_path is None:
        raise SystemExit("ERROR: must provide --init-from or --resume")
    if args.init_from is None:
        print(f"[init] --init-from not given; using --resume path ({args.resume}) as init source", flush=True)
    # Track the effective init path for downstream code (snapshot pool, logs, etc.)
    args.init_from = init_path

    # Load model
    model, cfg, _ = load_checkpoint(init_path, device)
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

    # Anchors — fixed checkpoints kept in the pool throughout training.
    # Used to prevent self-play drift / cycling by giving PFSP stable
    # reference points (e.g. peak-era sp_2979). Anchors are NEVER pruned by
    # --pool-max-current-run.
    anchor_set = set()
    if args.pool_anchors:
        for raw in args.pool_anchors.split(","):
            p = raw.strip().replace("\\", "/")
            if not p:
                continue
            if not Path(p).exists():
                print(f"  [WARN] --pool-anchors path does not exist: {p} (skipping)",
                      flush=True)
                continue
            if p == args.init_from.replace("\\", "/"):
                continue  # init is already in the pool
            if p in anchor_set:
                continue
            anchor_set.add(p)
            snapshot_pool.append(p)
            print(f"  [pool] anchor added: {p}", flush=True)
    rs_cfg = _build_reward_config(args)

    # Team builder. Training MUST use procedural Smogon-usage teams to avoid
    # overtraining on the 70 hand-curated teams in teams_ou.py (those are eval-only).
    # If --procedural-teams isn't passed, try the canonical project path; otherwise
    # raise loudly. The previous silent fallback to random_pool_teambuilder() (= the
    # 70 static teams) caused thousands of iters of training on the same teams.
    _CANON_PROC_PATH = Path(__file__).resolve().parents[3] / "raw_data" / "pokemon_usage" / "2024-04"
    proc_path = args.procedural_teams or (str(_CANON_PROC_PATH) if _CANON_PROC_PATH.exists() else None)
    if not proc_path:
        raise SystemExit(
            "ERROR: training requires --procedural-teams <stats_dir>. "
            f"Canonical path is {_CANON_PROC_PATH} (not found). "
            "The 70 static teams in teams_ou.py are eval-only — "
            "do not use them for training."
        )
    train_teambuilder = procedural_teambuilder(proc_path, random_pct=args.random_team_pct)
    print(f"  Train teambuilder: ProceduralTeambuilder({proc_path}, random_pct={args.random_team_pct})", flush=True)

    # Save config
    config = vars(args)
    config["run_dir"] = str(run_dir)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # PFSP win rate tracking: {checkpoint_path: [wins, games]}
    # Load from previous run if resuming, otherwise start fresh (all default 0.5)
    win_rates_path = run_dir / "win_rates.json"
    win_rates = {}
    if args.resume:
        # Try loading from the PREVIOUS run's directory
        prev_run_dir = Path(args.resume).parent
        prev_wr = prev_run_dir / "win_rates.json"
        if prev_wr.exists():
            try:
                with open(prev_wr) as f:
                    win_rates = json.load(f)
                print(f"  [PFSP] Loaded {len(win_rates)} win rates from {prev_wr}")
            except Exception as e:
                print(f"  [PFSP] Failed to load win_rates: {e}, starting fresh")
    if win_rates_path.exists():
        try:
            with open(win_rates_path) as f:
                win_rates = json.load(f)
            print(f"  [PFSP] Loaded {len(win_rates)} win rates from {win_rates_path}")
        except Exception:
            pass
    # Normalize win_rates keys and merge duplicates from path separator issues
    if win_rates:
        normalized = {}
        for k, v in win_rates.items():
            nk = k.replace("\\", "/")
            if nk in normalized:
                normalized[nk][0] += v[0]
                normalized[nk][1] += v[1]
            else:
                normalized[nk] = list(v)
        if len(normalized) < len(win_rates):
            print(f"  [PFSP] Merged {len(win_rates) - len(normalized)} duplicate path entries")
        win_rates = normalized

    # Resume
    start_iter = 0
    if args.resume:
        start_iter, snapshot_pool = _resume_from_checkpoint(
            args, model, optimizer, snapshot_pool, device)

    # External opponents — appended AFTER resume so resumed pool state stays clean
    # (resume loads only local snapshot paths; externals are re-instantiated each
    # run from the YAML). Subprocess adapters (metamon) get spawned + supervised
    # by an ExternalOpponentManager which we keep alive for the rest of training.
    external_manager = None
    if getattr(args, "external_adapters", None):
        from external_adapters import load_pool_entries
        default_port = int(args.servers.split(",")[0].strip())
        ext_entries, external_manager = load_pool_entries(
            args.external_adapters, default_server_port=default_port
        )
        if ext_entries:
            snapshot_pool.extend(ext_entries)
            ext_keys = ", ".join(e.key for e in ext_entries)
            print(f"  [PFSP] +{len(ext_entries)} external adapters: {ext_keys}", flush=True)
        if external_manager is not None:
            print(f"  [PFSP] starting {len(external_manager.opponents)} subprocess adapter(s)",
                  flush=True)
            external_manager.start_all()
            # Block until every subprocess has logged into Showdown and entered
            # its accept loop. Metamon's model-load takes ~30s, Foul Play ~10s.
            # Without this gate, V9RLPlayer's challenges hit not-yet-ready
            # subprocesses → wait_for timeout per opponent → wave-time blow up.
            print(f"  [PFSP] waiting up to 180s for subprocess adapter(s) to be ready...",
                  flush=True)
            ready = external_manager.wait_until_ready(per_opp_timeout_s=180.0)
            if ready:
                print(f"  [PFSP] all subprocess adapter(s) ready", flush=True)
            else:
                print(f"  [PFSP] WARN — one or more subprocess adapter(s) not ready; "
                      f"proceeding anyway, expect timeouts", flush=True)
            # NOTE: the 30s GUARD sleep we tried in attempt 6 was REPLACED by
            # the dispatch watchdog in rl_collection.py `_play_one_opponent`.
            # Watchdog catches the same login-race symptoms AND any other
            # subprocess flakiness (post-login crashes, _challenge_queue
            # binding race on subsequent waves not just iter 0, etc.) without
            # imposing a fixed startup cost on healthy runs.

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
        "win_rates": win_rates,
        "turn_cap": args.turn_cap,
    }
    pending_collection = None
    mp_bg_collector = None
    eval_history = []  # list of eval dicts for early-stopping check

    # Pool entries that prune-on-save must NEVER drop: init + anchors. (Layer 5)
    protected_paths = {args.init_from.replace("\\", "/")} | anchor_set

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
        collect_result = _collect_data(
            args, model, device, server_pool, snapshot_pool,
            rs_cfg, train_teambuilder, battle_format,
            loop, pending_collection, _flow, win_rates=win_rates,
            external_manager=external_manager)
        trajs, wins, losses, ties, steps, opp_name, collect_time = collect_result[:7]
        opp_records = collect_result[7] if len(collect_result) > 7 else {}
        pending_collection = None
        wr = wins / max(1, wins + losses + ties)

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
                    "win_rate": wr, "snapshot_pool": [s for s in snapshot_pool if isinstance(s, str)]})
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

        # ---- PFSP win rate update ----
        if opp_records:
            for ckpt, (w, g) in opp_records.items():
                nk = ckpt.replace("\\", "/")
                rec = win_rates.get(nk, [0, 0])
                if args.win_rate_mode == "ema":
                    # EMA mode: blend old rate with new batch rate.
                    # Old rec is stored as [eff_wins, eff_games] representing
                    # the smoothed rate. effective_games is capped to prevent
                    # unbounded growth and ensure old data is forgotten.
                    alpha = args.win_rate_ema_alpha
                    old_rate = (rec[0] / rec[1]) if rec[1] > 0 else 0.5
                    batch_rate = (w / g) if g > 0 else 0.5
                    new_rate = (1.0 - alpha) * old_rate + alpha * batch_rate
                    # Cap effective games at ema_window (default 50) so old data fades.
                    eff_games = min(rec[1] + g, args.win_rate_ema_window)
                    win_rates[nk] = [new_rate * eff_games, eff_games]
                else:
                    # Cumulative (default): just add wins and games
                    rec[0] += w
                    rec[1] += g
                    win_rates[nk] = rec
            # Save periodically (every 5 iters to avoid IO bottleneck)
            if (it + 1) % 5 == 0:
                try:
                    with open(win_rates_path, "w") as f:
                        json.dump(win_rates, f)
                except Exception:
                    pass

        # ---- Adaptive entropy ----
        # Raises ent_coef when entropy drops (prevents collapse).
        # Lowers ent_coef when entropy is too exploratory.
        if args.adaptive_entropy and loss_info["ent"] > 0.01:
            low = args.adaptive_entropy_low
            high = args.adaptive_entropy_high
            max_coef = args.adaptive_entropy_max
            min_coef = args.adaptive_entropy_min
            step = args.adaptive_entropy_step
            if loss_info["ent"] < low:
                ent_coef = min(ent_coef * (1.0 + step), max_coef)
                print(f"  [ENT] Low ({loss_info['ent']:.3f} < {low:.2f}), ent_coef -> {ent_coef:.4f}",
                      flush=True)
            elif loss_info["ent"] > high:
                ent_coef = max(ent_coef * (1.0 - step), min_coef)
                print(f"  [ENT] High ({loss_info['ent']:.3f} > {high:.2f}), ent_coef -> {ent_coef:.4f}",
                      flush=True)

        # ---- Snapshot (before background collection so new snapshot is in pool) ----
        _maybe_save_snapshot(it, args, model, cfg, optimizer, steps, loss_info,
                             wr, best_eval_wr, snapshot_pool, run_dir,
                             protected_paths=protected_paths)

        # ---- Start background collection for next iter ----
        # Moved here from before PPO update so that the latest snapshot is in the
        # pool when background collection begins. Previously, background collection
        # started before snapshot save, so the model never fought its most recent self.
        mp_bg_collector = _start_background_collection(
            args, model, device, server_pool, snapshot_pool,
            collect_args, bg_collector, mp_bg_collector, in_warmup, _flow,
            external_manager=external_manager)

        # ---- Eval (runs while background collection is in progress) ----
        best_eval_wr, _, should_stop = _maybe_eval(
            it, args, model, cfg, optimizer, device, writer,
            run_dir, best_eval_wr, battle_format,
            eval_history=eval_history if args.early_stop else None)
        if should_stop:
            print(f"\n[EARLY STOP] Terminating at iter {it}. Best snapshots saved; "
                  f"check evals registry and snapshot_*.pt files in {run_dir}.", flush=True)
            break

        # Memory cleanup
        del trajs, episodes
        gc.collect()
        torch.cuda.empty_cache()

    # Final save
    final_path = str(run_dir / "final.pt")
    save_checkpoint(final_path, model, cfg, optimizer, start_iter + args.n_iters - 1,
                    metrics={"best_eval_wr": best_eval_wr, "snapshot_pool": [s for s in snapshot_pool if isinstance(s, str)]})
    print(f"\nTraining complete. Final checkpoint: {final_path}", flush=True)
    writer.close()
    loop.close()


if __name__ == "__main__":
    main()
