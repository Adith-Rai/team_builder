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
import torch._dynamo  # imported at module top so any later reference inside main() doesn't shadow `torch` as a local

# Linux/Ampere optimizations (parity with train_bc.py — Session 50 audit found
# these were missing from train_rl.py despite being free wins):
# - TF32 matmul: 5-15% speedup on Ampere (A100) for fp32 matmuls
# - cuDNN benchmark: 5-10% speedup by autotuning kernels for stable shapes
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

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
    ppo_update_batched, make_compiled_train_step,
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
                   help="Per-iter multiplicative change to ent_coef (default: 0.1 = +/-10 percent)")
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
    # --fp16 / --bf16 are mutually exclusive autocast precision flags.
    # PREFER --bf16 for new runs (Ampere+ default): same Tensor Core throughput as fp16,
    # no GradScaler needed (fp32 dynamic range), avoids the -1e9 mask overflow trap.
    # bf16 also enables autocast on the PPO update path (ppo.py); fp16 update stays
    # fp32 because fp16 backward without a GradScaler underflows on small gradients.
    # --fp16 kept for backward compat with in-flight runs (Phase 1 v3).
    _amp_group = p.add_mutually_exclusive_group()
    _amp_group.add_argument("--fp16", action="store_true",
                            help="Mixed-precision autocast in fp16 (legacy; prefer --bf16)")
    _amp_group.add_argument("--bf16", action="store_true",
                            help="Mixed-precision autocast in bf16 (Ampere+; recommended default)")
    p.add_argument("--ko-coef", type=float, default=0.05)
    p.add_argument("--hp-coef", type=float, default=0.02)
    p.add_argument("--reward-clip", type=float, default=2.0)
    p.add_argument("--temp-min", type=float, default=1.0, help="Opponent temp range min")
    p.add_argument("--temp-max", type=float, default=2.25, help="Opponent temp range max")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile spatial encoder (Linux only)")
    p.add_argument("--tier3", action="store_true",
                   help="Tier 3 sequence-batched PPO update (S55+). Composes "
                        "collate_episodes + forward_ppo_sequence + ppo_loss_batched "
                        "into ONE forward+backward+step per epoch instead of one "
                        "per episode (4-10× update phase speedup). When combined "
                        "with --compile, additionally enables C5 single-graph "
                        "torch.compile of forward+loss+backward+clip+optimizer.step. "
                        "NOT supported in warmup iters (auto-falls-back to per-"
                        "episode ppo_update during warmup). Linux + transformer "
                        "arch only.")
    p.add_argument("--bc-anchor-ckpt", default=None,
                   help="S57 BC anchor: path to a frozen BC reference checkpoint. "
                        "When set, an auxiliary KL(BC || model) loss term anchors "
                        "the model to BC's policy distribution during PPO, "
                        "preventing the type-knowledge erosion observed in pure "
                        "self-play on the new transformer arch (Phase 1 v3 SE "
                        "rate fell 44%→31% over 60 iters). Requires --tier3 (eager "
                        "batched path); NOT supported with --compile in v1.")
    p.add_argument("--tier3-minibatch-size", type=int, default=None,
                   help="S57 task #10: when --tier3 is set, splits each "
                        "epoch's episodes into minibatches of N for "
                        "memory-bounded forward+backward (gradients "
                        "accumulate, ONE optimizer.step per epoch). "
                        "Required for production scale (1600+ games) on "
                        "A100 80GB — mega-batching all episodes blows "
                        "past activation memory. Typical: 16-32 for "
                        "production scale, 8 for tighter memory. None "
                        "= one chunk = old mega-batch behavior (smoke "
                        "scale only).")
    p.add_argument("--packed", action="store_true",
                   help="S64: route the Tier 3 eager update through the "
                        "packed-sequence path (collate_episodes_packed + "
                        "forward_ppo_sequence_packed + _ppo_loss_packed_internal). "
                        "Eliminates pad_mask waste in the collate+temporal+loss "
                        "stack (~38%% of update CPU at prod scale per S64 step-back "
                        "profile). Requires --tier3. NOT supported with --compile "
                        "(eager-only in v1). Bit-equivalent to legacy at fp32 eval "
                        "and within bf16 noise at training (B.2-B.5 gates).")
    p.add_argument("--no-per-chunk-gc", action="store_true",
                   help="S64 2b experiment: disable per-chunk gc.collect() + "
                        "torch.cuda.empty_cache() in the Tier 3 eager update path. "
                        "Default is per-chunk gc ON (matches legacy). At mb=64+ "
                        "with packed memory savings, per-chunk gc may be redundant "
                        "(activation memory is well-bounded by Python ref counting "
                        "alone). If this flag is set: faster wall but risk OOM if "
                        "memory accumulates. Test at prod before shipping.")
    p.add_argument("--diag-grad-norms", action="store_true",
                   help="S67-EXT observability: per-epoch grad-norm decomposition "
                        "(PPO part vs BC anchor part) + cosine similarity. Cost: "
                        "1 extra backward per epoch (~3-5% update wall). Use to "
                        "diagnose whether BC anchor is mechanically dominating "
                        "updates vs PPO objective. See project_bc_anchor_dominance_diagnosis.md.")
    p.add_argument("--bc-anchor-coef", type=float, default=0.1,
                   help="Coefficient for the BC anchor KL term. Typical 0.05-0.2. "
                        "0.0 disables anchor even if --bc-anchor-ckpt given. "
                        "Default 0.1 — caps drift without capping PPO improvement "
                        "direction (S57 isolation experiment starting point).")
    p.add_argument("--pipeline", action="store_true",
                   help="Pipeline collection and PPO update (overlap on GPU)")
    p.add_argument("--profile-iters", type=str, default="",
                   help="S64 forensics: comma-separated list of iter indices "
                        "to capture with torch.profiler around ppo_update_batched. "
                        "Trace saved as {out_dir}/profile_iter{N}.json. Adds "
                        "~20-30%% wall overhead during profiled iters. Use sparingly. "
                        "Example: --profile-iters 0,2")
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
                   help="Use disk-backed multiprocess collection (mp_disk_collect.py). "
                        "N forkserver workers, each with own GPU model copy + own "
                        "InferenceBatcher; trajectories written to /tmp at iter end. "
                        "Cloud-only, transformer arch only. See docs/MP_DISK_REDESIGN.md.")
    p.add_argument("--cis", action="store_true",
                   help="Use centralized inference server (mp_centralized_collect.py). "
                        "Single CIS subprocess holds the GPU model; N workers pipe "
                        "obs to CIS via numpy IPC, no per-worker model copy. Unlocks "
                        "real --pipeline overlap via CUDA stream priority (main HIGH, "
                        "CIS LOW). Mutually exclusive with --mp. Cloud-only, "
                        "transformer arch only. See docs/CENTRALIZED_INFERENCE_DESIGN.md.")
    p.add_argument("--mp-workers", type=int, default=8,
                   help="Number of mp-disk OR cis workers when --mp/--cis is set "
                        "(default 8 for RunPod A100 80GB; tune down for smaller VRAM).")
    p.add_argument("--mp-cache-size", type=int, default=3,
                   help="Per-worker LRU cache size for opponent ckpts (default 3).")
    p.add_argument("--batch-timeout-ms", type=float, default=15,
                   help="InferenceBatcher batch timeout in ms")
    # CIS batch-formation tuning (S62 Option B). Defaults match the pre-S62
    # production behavior. Bumping these widens CIS's per-slot batch window
    # → potentially fewer-larger fires, higher GPU util on the inference path.
    # See memory/project_s61_fix1_design.md for design + A/B gate spec.
    p.add_argument("--cis-min-batch", type=int, default=8,
                   help="CIS per-slot batch-fire threshold. Fires when the slot "
                        "accumulates this many requests OR --cis-timeout-ms "
                        "elapses since last fire. Only effective when --cis is "
                        "set. Default 8 = pre-S62 production.")
    p.add_argument("--cis-timeout-ms", type=int, default=15,
                   help="CIS per-slot batch-fire timeout in ms. See "
                        "--cis-min-batch. Default 15 = pre-S62 production.")
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
    p.add_argument("--max-opponents-per-iter", type=int, default=10,
                   help="Max active opponents per iter. When pool > N, the "
                        "select_opponents_phase2_stage1 composition function selects "
                        "N from pool (2 forced anchors + 2 random + N-4 PFSP). "
                        "Default 10 matches Phase 2 Stage 1 design. "
                        "Set -1 to disable (use full pool — wall time grows with pool). "
                        "S67 fix: previously only applied to legacy paths; now CIS too.")
    p.add_argument("--force-anchors", default="",
                   help="Comma-separated paths to opponents ALWAYS included in active "
                        "set every iter (in addition to 2 self-forced anchors). "
                        "S67 NEW. Use case: terminal-self regression check during "
                        "Phase 2 Stage 2 external curriculum (force Stage 1 final "
                        "checkpoint as anchor to detect regression below pure-self-play). "
                        "Each path MUST also be in --pool-anchors (or already in pool). "
                        "Limit: K force-anchors + 2 self-forced ≤ max_opponents_per_iter.")
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

    # Strip torch.compile's `_orig_mod.` prefix if present. Snapshots saved
    # from a compile-wrapped model carry this prefix (each wrapped submodule
    # has its parameters under self._orig_mod.*). Resume always targets a
    # fresh, un-wrapped model (compile is applied AFTER this function returns,
    # see main()), so keys must match the un-prefixed form. Mirrors the same
    # strip in load_checkpoint at ppo.py:397.
    resume_state = {k.replace("._orig_mod.", "."): v for k, v in resume_state.items()}

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

    # Scan disk for snapshots saved by THIS run that aren't yet in pool.
    # (S58 fix) Previously hardcoded to data/models/rl_v9/ + MIN_SNAPSHOT_ITER=260
    # — that was V9-era cleanup logic (pre-type-effectiveness snapshots had
    # eval 25-44%, polluted value function). It's stale for V10+ runs because
    # (a) the glob misses rl_v10/... paths entirely, and (b) V10 snapshot iter
    # numbers are tiny (10s-100s, all < 260). Now derives the scan dir from
    # the resume ckpt's parent (= the actual run dir).
    from pathlib import Path as _Path
    run_dir = _Path(args.resume).parent
    all_disk = sorted(run_dir.glob("snapshot_*.pt"))
    all_disk = [str(p).replace("\\", "/") for p in all_disk]
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
          f"pool: {len(pool)} checkpoints (+{len(new_snaps)} from disk scan "
          f"of {run_dir})"
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
        # Pipeline-timing trap (S58): if bg-collect produced 0 trajectories,
        # accepting it leads to FATAL on the PPO update (n_succeeded=0). Common
        # cause: stall-mode policy → games taking >> bg-collect window because
        # KL early-stop made the prior update phase too short to drain games.
        # Fall back to synchronous collect (full time budget) instead of dying.
        # If even sync collect produces 0 trajs, the FATAL safety net still
        # fires downstream (genuine failure: CIS dead, workers wedged, etc.).
        if len(pending_collection[0]) == 0:
            _flow("pre-collected was EMPTY — falling back to sync collect "
                  "(likely stall-mode + short update window race)")
            # Fall through to sync collect path below — do NOT return.
        else:
            _flow("using pre-collected data from background")
            result = pending_collection
            _flow(f"unpacked pre-collected: {len(result[0])} trajs, {result[4]} steps")
            return result

    if getattr(args, 'cis', False):
        # --cis routes to centralized inference server (mp_centralized_collect.py).
        # Workers don't own a model; CIS is the single GPU model holder.
        # See docs/CENTRALIZED_INFERENCE_DESIGN.md.
        from mp_centralized_collect import mp_centralized_collect_sync

        # S67 FIX: apply composition function for CIS (previously only applied
        # to legacy paths). Without this, CIS plays ALL pool members per iter
        # → wall time + GPU slot count grow unboundedly with pool size.
        # Composition rule (2 forced + 2 random + N-4 PFSP) preserved via flag.
        # S67-EXT: optional --force-anchors list always included (Phase 2 Stage 2
        # use case: force terminal-self as regression check vs external curriculum).
        max_opps = getattr(args, 'max_opponents_per_iter', 10)
        force_anchors_str = getattr(args, 'force_anchors', '') or ''
        force_anchors_list = [p.strip().replace("\\", "/") for p in force_anchors_str.split(",") if p.strip()] or None
        effective_pool = snapshot_pool
        if max_opps > 0 and len(snapshot_pool) > max_opps:
            from rl_collection import select_opponents_phase2_stage1
            effective_pool = select_opponents_phase2_stage1(
                snapshot_pool, win_rates, max_n=max_opps,
                force_anchors=force_anchors_list,
            )
            n_force = len(force_anchors_list) if force_anchors_list else 0
            n_pfsp = max_opps - 2 - 2 - n_force  # 2 self-forced + 2 random
            force_desc = f"{n_force} force-ext + 2 self-forced + 2 random + {n_pfsp} PFSP"
            print(f"  [composition] pool={len(snapshot_pool)} active={len(effective_pool)} "
                  f"({force_desc} per Phase 2 Stage 1 design)",
                  flush=True)

        # S67-EXT CIS+EXTERNAL Tier 1: convert PoolEntry objects (external
        # opps from --external-adapters) to dict format for CIS pipeline.
        # Local entries (strings) pass through unchanged. External subprocess
        # entries become {"kind": "external_subprocess", "key": str, "username": str}.
        # In-process external (factory-based MCTS, Tier 2) is REJECTED here —
        # not yet supported in CIS mode (deferred per
        # project_cis_external_integration_design).
        cis_pool = []
        for item in effective_pool:
            if isinstance(item, str):
                cis_pool.append(item)  # local checkpoint path — unchanged
            else:
                # PoolEntry-like object
                kind = getattr(item, "kind", "local")
                if kind == "local":
                    cis_pool.append(item.path)
                elif getattr(item, "showdown_username", None):
                    cis_pool.append({
                        "kind": "external_subprocess",
                        "key": item.key,
                        "username": item.showdown_username,
                    })
                elif getattr(item, "factory", None):
                    raise NotImplementedError(
                        f"In-process external opp '{item.key}' (factory-based, "
                        f"e.g. MCTS) is not supported in --cis mode (Tier 2 "
                        f"deferred per project_cis_external_integration_design). "
                        f"Use legacy non-CIS path or drop this entry."
                    )
                else:
                    raise ValueError(
                        f"Unknown PoolEntry: kind={kind} key={item.key}"
                    )

        model.eval()
        _flow(f"starting CIS collection (n_workers={args.mp_workers})")
        cis_result = mp_centralized_collect_sync(
            model, device, server_pool,
            n_games=args.games_per_iter,
            max_concurrent=args.max_concurrent,
            snapshot_pool=cis_pool,
            fp16=args.fp16,
            reward_shaper_cfg=rs_cfg,
            temp_range=(args.temp_min, args.temp_max),
            opponent_device=args.opponent_device,
            win_rates=win_rates,
            turn_cap=args.turn_cap,
            battle_format=battle_format,
            procedural_teams_path=getattr(args, 'procedural_teams', None),
            iter_n=getattr(args, '_current_iter', 0),
            n_workers=args.mp_workers,
            amp_dtype=getattr(args, 'amp_dtype_name', None),
            cis_min_batch=args.cis_min_batch,
            cis_timeout_ms=args.cis_timeout_ms,
        )
        _flow(f"cis collect done: {cis_result[6]:.0f}s, "
              f"{len(cis_result[0])} trajs")
        return cis_result

    if getattr(args, 'mp', False):
        # --mp now routes to disk-backed implementation (mp_disk_collect.py).
        # Old mp_collect_v2.py kept in repo for reference; not called.
        # Transformer arch only; CPU not supported. See docs/MP_DISK_REDESIGN.md.
        from mp_disk_collect import mp_disk_collect_sync
        model.eval()
        _flow(f"starting MP-DISK collection (n_workers={args.mp_workers})")
        mp_result = mp_disk_collect_sync(
            model, device, server_pool,
            n_games=args.games_per_iter,
            max_concurrent=args.max_concurrent,
            snapshot_pool=snapshot_pool,
            fp16=args.fp16,
            reward_shaper_cfg=rs_cfg,
            temp_range=(args.temp_min, args.temp_max),
            opponent_device=args.opponent_device,
            win_rates=win_rates,
            turn_cap=args.turn_cap,
            battle_format=battle_format,
            procedural_teams_path=getattr(args, 'procedural_teams', None),
            iter_n=getattr(args, '_current_iter', 0),
            n_workers=args.mp_workers,
            amp_dtype=getattr(args, 'amp_dtype_name', None),
        )
        _flow(f"mp-disk collect done: {mp_result[6]:.0f}s, "
              f"{len(mp_result[0])} trajs")
        return mp_result

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
                                  in_warmup, _flow, external_manager=None,
                                  iter_n: int = 0):
    """Kick off background collection for the NEXT iteration (pipeline mode)."""
    if args.mp and args.pipeline and not in_warmup:
        # KNOWN LIMITATION (Session 50): mp+pipeline overlap causes worker
        # GPU contention deadlock when bg cmd processing runs in parallel
        # with main's PPO update. Workers' inference forwards stall when
        # main is doing optimizer.step()-heavy update; deadlock doesn't
        # recover after main finishes. See docs/MP_DISK_REDESIGN.md.
        # The CIS path (--cis --pipeline) was built to fix this via
        # centralized inference + low-priority CUDA stream arbitration
        # (Phase 4.2/4.3). Use --cis instead of --mp when pipeline overlap
        # is wanted. mp-only is fully validated; --mp --pipeline still
        # silently behaves as --mp only.
        pass  # no-op; mp_bg_collector stays None
    elif args.cis and args.pipeline and not in_warmup:
        # Phase 4.3b: CIS bg overlap re-enabled. CISBgCollector mirrors
        # MPDiskBgCollector's start/join interface; CIS forwards run on a
        # low-priority CUDA stream behind main's optimizer.step on the
        # high-priority default stream. Workers progress during update.
        if mp_bg_collector is not None and not mp_bg_collector.running:
            _flow("starting CIS BACKGROUND collection for next iter")
            mp_bg_collector.start(
                model, device, server_pool, snapshot_pool, collect_args,
                win_rates=collect_args.get("win_rates"),
                iter_n=iter_n,
            )
    elif bg_collector and not in_warmup and not args.mp and not args.cis:
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
    bc_kl_str = (f" bc_kl={loss_info['bc_kl']:.4f}"
                  if loss_info.get('bc_kl', 0.0) > 0 else "")
    warmup_str = " [WARMUP]" if in_warmup else ""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] Iter {it}: W/L/T={wins}/{losses}/{ties} ({wr:.1%}), {steps} steps, "
          f"collect={collect_time:.0f}s, update={update_time:.0f}s, "
          f"pi={loss_info['pi']:.4f} v={loss_info['v']:.4f} "
          f"ent={loss_info['ent']:.4f}{kl_str}{bc_kl_str}{warmup_str} "
          f"vs={opp_name} pool={len(snapshot_pool)}",
          flush=True)

    writer.add_scalar("train/win_rate", wr, it)
    writer.add_scalar("train/pi_loss", loss_info["pi"], it)
    writer.add_scalar("train/v_loss", loss_info["v"], it)
    writer.add_scalar("train/entropy", loss_info["ent"], it)
    if "kl" in loss_info:
        writer.add_scalar("train/kl", loss_info["kl"], it)
    if loss_info.get("bc_kl", 0.0) > 0:
        writer.add_scalar("train/bc_kl", loss_info["bc_kl"], it)
    writer.add_scalar("train/collect_time", collect_time, it)
    writer.add_scalar("train/update_time", update_time, it)
    writer.add_scalar("train/steps", steps, it)
    writer.add_scalar("train/pool_size", len(snapshot_pool), it)

    # S67-EXT observability: per-epoch trajectories + Tier 2 diagnostics +
    # (flag-gated) grad-norm decomposition. See project_bc_anchor_dominance_diagnosis.md
    # for what each metric tells us and how to interpret results.
    def _fmt_list(xs, fmt="{:.4f}"):
        return "[" + ", ".join(fmt.format(x) for x in xs) + "]"

    kl_traj   = loss_info.get("epoch_kl_traj", [])
    pi_traj   = loss_info.get("epoch_pi_traj", [])
    bc_traj   = loss_info.get("epoch_bc_kl_traj", [])
    ent_traj  = loss_info.get("epoch_ent_traj", [])
    if kl_traj:
        print(f"  [TRAJ] kl/ep={_fmt_list(kl_traj)}  "
              f"pi/ep={_fmt_list(pi_traj)}  "
              f"bc_kl/ep={_fmt_list(bc_traj)}  "
              f"ent/ep={_fmt_list(ent_traj, '{:.3f}')}", flush=True)
        # TensorBoard: log first + last epoch + delta for trend monitoring
        writer.add_scalar("train/kl_epoch_0", kl_traj[0], it)
        writer.add_scalar("train/kl_epoch_last", kl_traj[-1], it)
        writer.add_scalar("train/kl_epoch_delta", kl_traj[-1] - kl_traj[0], it)
        writer.add_scalar("train/pi_epoch_0", pi_traj[0], it)
        writer.add_scalar("train/pi_epoch_last", pi_traj[-1], it)
        if bc_traj and bc_traj[0] > 0:
            writer.add_scalar("train/bc_kl_epoch_0", bc_traj[0], it)
            writer.add_scalar("train/bc_kl_epoch_last", bc_traj[-1], it)

    adv_pos = loss_info.get("adv_pos_frac", None)
    n_valid_ep = loss_info.get("n_valid_per_epoch", None)
    rcf       = loss_info.get("ratio_clip_frac", None)
    if adv_pos is not None and rcf is not None:
        print(f"  [DIAG] adv_pos_frac={adv_pos:.3f}  "
              f"n_valid_per_epoch={int(n_valid_ep) if n_valid_ep else '-'}  "
              f"ratio_clip_frac={rcf:.4f}", flush=True)
        writer.add_scalar("train/adv_pos_frac", adv_pos, it)
        writer.add_scalar("train/n_valid_per_epoch", n_valid_ep or 0, it)
        writer.add_scalar("train/ratio_clip_frac", rcf, it)

    ppo_norms = loss_info.get("epoch_grad_ppo_norm_traj", [])
    bc_norms  = loss_info.get("epoch_grad_bc_norm_traj", [])
    cos_sims  = loss_info.get("epoch_grad_cos_traj", [])
    if ppo_norms and bc_norms:
        # Mean across epochs (first chunk sample per epoch)
        mean_ppo = sum(ppo_norms) / len(ppo_norms)
        mean_bc  = sum(bc_norms)  / len(bc_norms)
        mean_cos = sum(cos_sims)  / len(cos_sims) if cos_sims else 0.0
        ratio = mean_bc / max(mean_ppo, 1e-12)
        dom_str = ("BC-DOMINATED" if ratio > 1.5 else
                   "PPO-dominated" if ratio < 0.67 else
                   "balanced")
        print(f"  [GRAD] ppo_norm/ep={_fmt_list(ppo_norms, '{:.4f}')}  "
              f"bc_norm/ep={_fmt_list(bc_norms, '{:.4f}')}  "
              f"cos/ep={_fmt_list(cos_sims, '{:+.3f}')}", flush=True)
        print(f"  [GRAD] mean: ppo={mean_ppo:.4f}  bc={mean_bc:.4f}  "
              f"bc/ppo={ratio:.2f}× ({dom_str})  cos={mean_cos:+.3f}",
              flush=True)
        writer.add_scalar("train/grad_ppo_norm", mean_ppo, it)
        writer.add_scalar("train/grad_bc_norm", mean_bc, it)
        writer.add_scalar("train/grad_bc_ppo_ratio", ratio, it)
        writer.add_scalar("train/grad_cosine", mean_cos, it)
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
                best_eval_wr, battle_format, eval_history=None,
                snapshot_pool=None):
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
        # S67-EXT bugfix: include snapshot_pool in metrics so resume from
        # iter_XXXX.pt doesn't lose the pool. Previously this save site
        # passed no metrics, so any resume from iter_XXXX.pt got an empty
        # pool — relying on disk scan of the run dir alone (which misses
        # snaps in other selfplay subdirs and prior-run inherited snaps).
        # Mirrors snapshot_XXXX.pt save at line 704.
        _iter_metrics = {}
        if snapshot_pool is not None:
            _iter_metrics["snapshot_pool"] = [s for s in snapshot_pool if isinstance(s, str)]
        save_checkpoint(tmp, model, cfg, optimizer, it,
                        metrics=_iter_metrics if _iter_metrics else None)

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

    # --mp and --cis are mutually exclusive collect strategies.
    if getattr(args, "mp", False) and getattr(args, "cis", False):
        raise SystemExit("ERROR: --mp and --cis are mutually exclusive. "
                         "--mp = per-worker GPU model (mp_disk_collect.py); "
                         "--cis = centralized inference server "
                         "(mp_centralized_collect.py). Pick one.")

    # Set the global amp dtype ONCE here so every autocast site (in this
    # process AND in mp workers via cmd dict propagation) picks it up.
    # See precision_config.py docstring for rationale.
    from precision_config import set_amp_dtype, parse_amp_dtype, amp_dtype_name
    if args.bf16:
        args.amp_dtype_name = "bf16"
    elif args.fp16:
        args.amp_dtype_name = "fp16"
    else:
        args.amp_dtype_name = "fp32"
    set_amp_dtype(parse_amp_dtype(args.amp_dtype_name))
    # Keep args.fp16 True if either flag implies amp on, so legacy callers
    # that read fp16 bool still get autocast enabled (with the right dtype).
    args.fp16 = args.fp16 or args.bf16
    print(f"AMP dtype: {args.amp_dtype_name}", flush=True)

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

    # NOTE: torch.compile() is applied AFTER `_resume_from_checkpoint` (see
    # below near line ~820). Wrapping submodules with torch.compile rewrites
    # their state_dict keys to include `_orig_mod.` - if compile happens
    # BEFORE resume, the resume's `load_state_dict(strict=True)` fails on
    # snapshots saved without that prefix (the typical case for production
    # snapshots). Caught Session 51 cont. on the iter-14 cutover relaunch.
    compiled = False

    # fused=True uses CUDA-native fused AdamW kernel on supported devices
    # (Ampere+ A100 supports it). 3-7% saving on optimizer.step. Falls back
    # to non-fused on CPU/older GPUs automatically.
    _fused_supported = device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4,
        fused=_fused_supported,
    )
    if _fused_supported:
        print("optimizer: AdamW fused kernel enabled", flush=True)

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
        # S67 FIX: insert anchors at position 1 (right after init_from), NOT append.
        # Previously appended at end → anchors took the snapshot_pool[-2] / [-3] slots
        # that select_opponents_phase2_stage1 uses for "prev" and "prev-of-prev" forced
        # self-play snapshots. This made external anchors forced every iter, kicking
        # out the recent self-play snapshots intended for regression-detection.
        # Inserting at position 1 keeps init_from at [0], anchors at [1..N], and
        # self-play snapshots accumulate chronologically at the END of pool — so
        # pool[-1] / [-2] / [-3] are correctly the recent self snaps as designed.
        insert_pos = 1  # after init_from
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
            snapshot_pool.insert(insert_pos, p)
            insert_pos += 1  # next anchor goes after this one
            print(f"  [pool] anchor added (at pos {insert_pos-1}): {p}", flush=True)
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

        # S67-EXT-FIX2: re-apply --pool-anchors after resume. Previously the ckpt
        # pool overwrote whatever was set up at lines 859-892, so NEW anchors added
        # via --pool-anchors on a resume launch were silently dropped (only anchors
        # already in ckpt's saved pool persisted). Now we re-insert them at pos 1
        # after the init checkpoint, deduping against the ckpt-loaded pool.
        if args.pool_anchors:
            insert_pos = 1  # after init_from
            anchors_added = 0
            existing_keys = set(snapshot_pool)
            for raw in args.pool_anchors.split(","):
                p = raw.strip().replace("\\", "/")
                if not p or p in existing_keys:
                    continue
                if not Path(p).exists():
                    print(f"  [WARN] --pool-anchors (post-resume) path not found: {p} "
                          f"(skipping)", flush=True)
                    continue
                snapshot_pool.insert(insert_pos, p)
                existing_keys.add(p)
                insert_pos += 1
                anchors_added += 1
                print(f"  [pool] anchor re-applied post-resume (at pos {insert_pos-1}): {p}",
                      flush=True)
            if anchors_added:
                print(f"  [pool] post-resume: added {anchors_added} new anchor(s), "
                      f"pool now {len(snapshot_pool)} entries", flush=True)

    # torch.compile (Linux/cloud only - Windows local has no compile support;
    # the try/except below degrades gracefully there). Applied AFTER resume
    # because torch.compile wraps modules with `_orig_mod.` prefix in their
    # state_dict; loading a non-prefixed checkpoint into a wrapped model
    # fails with strict=True. See note near `compiled = False` above.
    #
    # Path 2 per-submodule compile (Session 51 finding, 2026-05-07): the prior
    # single-method `torch.compile(model.forward_spatial)` call only covered
    # tokenizer + spatial + summary projection (~40-60% of the inference path).
    # Compiling each nn.Module submodule separately covers ~90+%: temporal,
    # action_head, value_head also get optimized.
    #
    # mode="default" + dynamic=True chosen over mode="reduce-overhead" for
    # production robustness:
    # - reduce-overhead uses CUDA Graph replay (faster steady state, ~1.45x
    #   on full forward) but requires `torch.compiler.cudagraph_mark_step_begin()`
    #   between invocations to avoid tensor-aliasing crashes, and recompiles
    #   per shape (high cost under InferenceBatcher's variable B + PPO update's
    #   variable T).
    # - default uses Inductor codegen without cudagraphs. ~1.25x on full
    #   forward, identical performance across all shapes once compiled, no
    #   aliasing constraints. Predictable over multi-gen 5-7 week runs.
    #
    # `dynamic=True` hints Dynamo to mark batch dim dynamic from the start,
    # avoiding the second recompile pass when a new B arrives.
    #
    # `_dynamo.config.suppress_errors=True` is the safety net for the known
    # B=1 dynamic-shape edge in the tokenizer's `_encode_pokemon_block` (concat
    # of symint-derived tensors fails on torch 2.2.x). Falls back to eager for
    # that single call, no crash.
    #
    # Env requirement: torch + triton version match (torch 2.2.x needs triton
    # 2.2.x; torch 2.4.x needs triton 3.0.x). Mismatch triggers `ImportError:
    # cannot import name 'get_cuda_stream' from 'triton.runtime.jit'` at
    # compile-decoration time. Pod fix: `pip install triton==2.2.0` for
    # torch 2.2.1. See `docs/PPO_CLOUD_COOKBOOK.md` §3i for full history.
    if args.compile:
        try:
            torch._dynamo.config.suppress_errors = True
            submodules_to_compile = [
                "tokenizer", "spatial", "temporal", "action_head", "value_head",
            ]
            n_ok = 0
            for name in submodules_to_compile:
                if not hasattr(model, name):
                    print(f"torch.compile: model has no '{name}' submodule, skipping",
                          flush=True)
                    continue
                try:
                    sub = getattr(model, name)
                    setattr(model, name,
                            torch.compile(sub, mode="default", dynamic=True))
                    n_ok += 1
                except Exception as sub_e:
                    print(f"torch.compile: '{name}' SKIPPED ({sub_e})", flush=True)
            compiled = n_ok > 0
            if compiled:
                print(f"torch.compile: {n_ok}/{len(submodules_to_compile)} submodules "
                      f"compiled (mode=default, dynamic=True)", flush=True)
            else:
                print("torch.compile: no submodules compiled successfully", flush=True)
        except Exception as e:
            # Outermost guard: if torch._dynamo import or anything else fails,
            # don't crash the run - just fall back to eager.
            print(f"torch.compile: SKIPPED ({e})", flush=True)

    # BC anchor (S57): load frozen reference model into a SEPARATE local
    # variable (NOT an attribute of `model`). Storing on the model would
    # make PyTorch include the BC ref's parameters in `model.state_dict()`,
    # breaking snapshot save/load + mp_disk worker respawn (verified on
    # dev pod S57). Pass `bc_ref` directly to ppo_update_batched.
    #
    # S60 Fix #2: BC anchor loading was moved BEFORE make_compiled_train_step
    # so we can pass `bc_anchor_enabled` as a compile-time flag. The previous
    # FATAL mutex with --compile + --tier3 is REMOVED — BC anchor is now
    # plumbed through the compile boundary (bc_logits passed at call time
    # via eager BC ref forward per chunk; bc_anchor_coef as a 0-dim tensor
    # so per-iter coef changes don't trigger recompile). See `make_compiled_train_step`
    # docstring + `project_bc_anchor_design.md`.
    bc_ref = None
    if args.bc_anchor_ckpt:
        if args.bc_anchor_coef == 0.0:
            print("[WARN] --bc-anchor-coef is 0.0; BC anchor will be a no-op. "
                  "Set --bc-anchor-coef 0.1 (or similar) to enable.", flush=True)
        bc_path = args.bc_anchor_ckpt
        print(f"BC anchor: loading reference from {bc_path}", flush=True)
        bc_ref, _bc_cfg, _ = load_checkpoint(bc_path, device)
        bc_ref.eval()
        for p in bc_ref.parameters():
            p.requires_grad_(False)
        n_bc_params = sum(p.numel() for p in bc_ref.parameters())
        print(f"BC anchor: ref loaded ({n_bc_params:,} params, frozen, "
              f"coef={args.bc_anchor_coef})", flush=True)

    # Tier 3 C5 (S55+): single-graph compiled train_step for the batched PPO
    # update. Built AFTER per-submodule compile so the per-submodule compile
    # wrappers are transparently traced through by the outer graph (net effect
    # is one fused forward+loss+backward+clip+optimizer.step graph). Stashed
    # on the model object so it survives across iters and is reused (compile
    # cache stays warm). Only built when both --tier3 + --compile are set
    # AND the per-submodule compile succeeded; otherwise the eager batched
    # path runs (still 4-10× faster than per-episode ppo_update).
    #
    # S60 Fix #2: `bc_anchor_enabled` closure-bound at compile time. When
    # True, the compiled fwd_bwd accepts bc_logits + bc_anchor_coef_t as
    # call-time tensor inputs. When False, no BC code is in the compiled
    # region (zero overhead on non-BC training runs).
    #
    # Per S56 design (no shortcuts, ship optimal once): the full train_step
    # in one graph maximizes kernel fusion across forward/backward/step
    # boundaries. Safety gates (NaN, KL > target × 5) are in-graph tensor
    # masks (no host sync, no graph break). See `make_compiled_train_step`
    # docstring for details.
    model._tier3_step = None
    if args.tier3 and args.compile and compiled:
        _bc_anchor_compile_flag = (bc_ref is not None and args.bc_anchor_coef != 0.0)
        try:
            model._tier3_step = make_compiled_train_step(
                model, optimizer, cfg,
                vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm,
                normalize_advantages=False,  # build_ppo_episodes already normalizes
                bc_anchor_enabled=_bc_anchor_compile_flag,
            )
            _bc_suffix = (" + BC anchor" if _bc_anchor_compile_flag else "")
            print(f"torch.compile: Tier 3 C5 train_step compiled "
                  f"(fwd+loss+backward+clip+step in one graph{_bc_suffix})",
                  flush=True)
        except Exception as e:
            print(f"torch.compile: Tier 3 C5 train_step SKIPPED ({e}); "
                  f"--tier3 will use eager batched path", flush=True)
            model._tier3_step = None

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
    _collect_mode = "CIS" if args.cis else ("MP" if args.mp else "SYNC")
    _tier3_status = ("OFF" if not args.tier3
                     else ("ON+compile" if model._tier3_step is not None
                           else "ON (eager)"))
    _bc_anchor_status = (f"ON (coef={args.bc_anchor_coef})"
                         if bc_ref is not None else "OFF")
    print(f"FP16: {'ON' if args.fp16 else 'OFF'}, Compile: {'ON' if compiled else 'OFF'}, "
          f"Pipeline: {'ON' if args.pipeline else 'OFF'}, Collect: {_collect_mode}, "
          f"Tier3: {_tier3_status}, BCAnchor: {_bc_anchor_status}, Device: {device}")
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
        # S62: CIS batching window (Option B). Read by CISBgCollector.start
        # at mp_centralized_collect.py:2792-2793. Defaults preserved on the
        # CISBgCollector side (8/15) if absent, but we always pass to keep
        # sync + bg paths consistent.
        "cis_min_batch": args.cis_min_batch,
        "cis_timeout_ms": args.cis_timeout_ms,
    }
    pending_collection = None
    # Background collector for --mp/--cis + --pipeline. Mirrors the
    # BackgroundCollector interface (start, join, running). Phase 4.3b
    # added the CIS variant; the --mp variant is still no-op'd.
    mp_bg_collector = None
    if args.cis and args.pipeline:
        from mp_centralized_collect import CISBgCollector
        mp_bg_collector = CISBgCollector()
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
        # mp-disk path needs current iter index for traj/weights filenames
        args._current_iter = it
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
        # Tier 3 (--tier3) routes to ppo_update_batched which is a drop-in
        # replacement: same args, same returned stats. Warmup iters fall
        # back to per-episode ppo_update because ppo_update_batched does NOT
        # support the value-head-only frozen-backbone warmup pattern (raises
        # NotImplementedError). Warmup is 5 iters / 200 — trivial.
        # When --tier3 + --compile + per-submodule compile succeeded,
        # `model._tier3_step` holds the C5 single-graph compiled train_step;
        # ppo_update_batched dispatches to it via compiled_step kwarg. Else
        # ppo_update_batched runs its eager path (still 4-10× faster than
        # ppo_update via single-step-per-epoch batching).
        _profile_this_iter = (
            args.profile_iters
            and it in {int(x) for x in args.profile_iters.split(",") if x.strip()}
        )
        if _profile_this_iter:
            import torch.profiler as _tp
            _prof_ctx = _tp.profile(
                activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA],
                record_shapes=False,
                with_stack=False,
            )
            _prof_ctx.__enter__()
            _flow(f"profiler ENABLED for iter {it}")
        else:
            _prof_ctx = None

        # S67 (2026-05-21): Tier 3 batched path now supports warmup iters.
        # Previously dispatched to legacy ppo_update during warmup because
        # ppo_update_batched raised NotImplementedError; that limitation is
        # lifted (requires_grad set by lines 1175-1176 controls grad flow).
        # Warmup iters here are ~3-5× faster than legacy path. Legacy still
        # available via direct call if memory-constrained env needs the
        # no_grad backbone optimization.
        if args.tier3:
            loss_info = ppo_update_batched(
                model, optimizer, episodes, device, cfg,
                epochs=args.ppo_epochs, clip_eps=args.clip_eps,
                ent_coef=ent_coef, vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm, target_kl=args.target_kl,
                normalize_advantages=False,
                compiled_step=getattr(model, "_tier3_step", None),
                bc_ref=bc_ref,
                bc_anchor_coef=args.bc_anchor_coef,
                minibatch_size=args.tier3_minibatch_size,
                packed=args.packed,
                per_chunk_gc=not args.no_per_chunk_gc,
                in_warmup=in_warmup,  # noop in batched path; caller controls requires_grad
                diag_grad_norms=getattr(args, 'diag_grad_norms', False),
            )
        else:
            loss_info = ppo_update(
                model, optimizer, episodes, device, cfg,
                epochs=args.ppo_epochs, clip_eps=args.clip_eps,
                ent_coef=ent_coef, vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm, target_kl=args.target_kl,
                grad_accum=args.grad_accum,
                in_warmup=in_warmup,  # skips autograd through frozen backbone
                bc_ref=bc_ref,
                bc_anchor_coef=args.bc_anchor_coef,
            )

        if _prof_ctx is not None:
            _prof_ctx.__exit__(None, None, None)
            _prof_dir = Path(args.out_dir)
            _prof_dir.mkdir(parents=True, exist_ok=True)
            _prof_path = str(_prof_dir / f"profile_iter{it}.json")
            _prof_ctx.export_chrome_trace(_prof_path)
            _flow(f"profiler trace saved: {_prof_path}")

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
            external_manager=external_manager, iter_n=it + 1)

        # ---- Eval (runs while background collection is in progress) ----
        best_eval_wr, _, should_stop = _maybe_eval(
            it, args, model, cfg, optimizer, device, writer,
            run_dir, best_eval_wr, battle_format,
            eval_history=eval_history if args.early_stop else None,
            snapshot_pool=snapshot_pool)
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

    # Clean shutdown of mp-disk workers if --mp was used.
    if getattr(args, 'mp', False):
        try:
            from mp_disk_collect import shutdown_workers
            shutdown_workers()
            print("[mp-disk] workers shut down cleanly.", flush=True)
        except Exception as e:
            print(f"[mp-disk] shutdown error (non-fatal): {e}", flush=True)

    writer.close()
    loop.close()


if __name__ == "__main__":
    main()
