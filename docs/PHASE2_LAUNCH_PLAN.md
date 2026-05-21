# Phase 2 Launch Plan — S67 Final (2026-05-21)

**Status**: Design FINALIZED. Pending validation: 30w/5-iter memory stability test (in flight at handover).

**Authoritative companion**: `memory/project_phase2_launch_plan.md` (same content, auto-loaded).

---

## §-1. CURRENT STATE (as of 2026-05-21 11:35 UTC)

**Phase 2 Stage 1 IS RUNNING.** Do not relaunch.

- Started: 11:32:16 UTC, run name `phase2_stage1_v1`
- Prod log: `/tmp/phase2_stage1_v1.log` on prod (port 47913 / 195.26.233.30)
- Run dir: `data/models/rl_v10/phase2_stage1_v1/selfplay_v9_20260521_113214/`
- Local Elo poller running (background task at session wrap; if machine restarted, relaunch via §0 step 5)
- HEAD on prod: `7509467` (includes the ppo_update_batched warmup-mode support — see §X.5 below)

**Check status before any action**:
```bash
ssh -i ~/.ssh/id_ed25519 -p 47913 -o StrictHostKeyChecking=no root@195.26.233.30 \
  'grep -E "^\[..:..:..\] Iter [0-9]+:" /tmp/phase2_stage1_v1.log | tail -5'
```

If no iter lines yet at expected time, see §9 for troubleshooting.

---

## §X.5 Last-minute warmup fix (commit 7509467, 2026-05-21 evening)

**Without this fix**: `--warmup-iters 10` makes Stage 1 unusable. Warmup iter updates take 15-25 min each on legacy `ppo_update` path (because `ppo_update_batched` raised NotImplementedError for `in_warmup=True`). 10 warmup iters = 4-6 hours of wasted compute.

**The fix**: remove the `NotImplementedError` block in `ppo_update_batched`, flip `train_rl.py` dispatch from `if args.tier3 and not in_warmup:` to `if args.tier3:`. Functionally correct because `train_rl.py` already sets `param.requires_grad = "value_head" in name` before the call; PyTorch autograd respects this — backward computes value_head gradients only, optimizer skips frozen params.

**Smoke result** (3-iter test before relaunch):
- Iter 0 [WARMUP]: update **87s** (vs 21+ min on legacy → **~14× speedup**)
- Iter 1 [WARMUP]: update 81s (consistent)
- Iter 2 (full): update 163s (matches 30w validation 158s)
- `Value warmup complete, unfreezing all parameters` transition clean

**One minor oddity observed** (NOT a functional issue): `kl` was 0.01-0.012 during warmup when theoretically should be 0 (frozen policy → new=old). Caused by bf16 non-determinism in GPU forward (~1e-3 per-logit divergence aggregates to ~0.01 approx KL). Well below `target_kl 0.03`, same magnitude as iter 2 post-warmup. Don't fix; don't worry.

**Why this matters for Stage 2 too**: every tier-addition restart in Stage 2 uses `--warmup-iters 10` to re-stabilize value head. Without the fix, each restart cost ~4-6 hours. With the fix, ~105 min.

---

## §0. QUICK START — launch Stage 1 in 4 commands

**Prerequisite**: 30w validation completed cleanly (iter 4 lands without OOM in
`/tmp/exp30w_validate.log`). If not yet validated, see §5 first.

**1. SSH to prod + sync to latest master:**
```bash
ssh -i ~/.ssh/id_ed25519 -p 47913 -o StrictHostKeyChecking=no root@195.26.233.30
cd /workspace/team_builder
# Verify no zombie training procs from any previous run:
pgrep -c python  # expect 0 (orphan multiprocessing wrappers OK)
nvidia-smi --query-gpu=memory.used --format=csv,noheader  # expect ≤500 MiB
# Sync master:
git checkout master && git pull origin master
git log -1 --oneline  # confirm cb7ea7ca or newer
```

**2. Make wrapper + poller executable** (one-time after first pull):
```bash
chmod +x pokemon-ai-starter/pokemon-ai/src/launch_phase2_with_oom_fallback.sh
```

**3. Launch Stage 1** (the actual one-line training launch):
```bash
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
STAGE=stage1 nohup bash launch_phase2_with_oom_fallback.sh phase2_stage1_v1 \
  > /tmp/phase2_stage1_v1_wrapper.log 2>&1 &
```

**4. Verify training started cleanly** (~30s after launch):
```bash
sleep 30
tail -20 /tmp/phase2_stage1_v1.log
# Expect: "BC anchor: loading reference..." + CIS startup banner + iter 0 collect lines
pgrep -c python  # expect ~33 (30 workers + main + CIS + orchestrator)
nvidia-smi --query-gpu=memory.used --format=csv,noheader  # expect ~15-20 GB
```

**First iter line lands** at ~25 min after launch. Subsequent iters every ~24 min.

### Concurrently on local (Elo poller — starts gathering data as snapshots land):

**5. Start the Elo poller on local** (needs 3 BS running locally on 9000/9001/9002):
```bash
cd C:/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src
python elo_poller.py --run-name phase2_stage1_v1
```

The poller idles 5 min between checks when caught up. First snapshot lands on prod
at iter 10 (snapshot-interval=10), so first Elo eval starts ~250 min into Stage 1.

### Monitoring after launch
```bash
# Iter completion (one line per iter):
ssh ... 'grep -E "^\[..:..:..\] Iter [0-9]+:" /tmp/phase2_stage1_v1.log'

# OOM safety check:
ssh ... 'grep -iE "out of memory|FATAL" /tmp/phase2_stage1_v1.log'

# Smart-bot eval trajectory (auto-logged every 10 iters):
ssh ... 'tail -20 data/eval/registry/evals.jsonl'
```

If you see anything weird: see §9 for deeper operational details, troubleshooting,
and Stage 2 launch procedure.

---

## TL;DR

Two-stage Phase 2:
- **Stage 1**: Pure self-play warmup (~50-100 iters). Init from BC v10. Builds initial snapshot pool + value-head stability before externals.
- **Stage 2**: External opps added (6 day-1 + manual curriculum additions). Init from best Elo snapshot from Stage 1. ~150 iters.

Canonical infrastructure (locked):
- `--mp-workers 30` (architectural ceiling at mb=64; see §3.1)
- 10 active opps per iter
- `--snapshot-interval 10`, `--warmup-iters 10`
- `--bc-anchor-coef 0.1`
- Wrapper: `launch_phase2_with_oom_fallback.sh` (detect+notify only, no auto-fallback)
- Eval: existing `eval_elo_ladder.py --add-to` with curated 15-anchor subset (drops 3 abandoned), 500 g/opp, local 3-server sharded

---

## §1. Architectural ceiling — WHY 30 workers

The 79.14 GB GPU on the A100 80GB is partitioned as:

| Component | Memory | Notes |
|---|---|---|
| Per-worker CUDA primary context | **~490 MB / worker** | Fixed at process start; PyTorch + cuBLAS + cuDNN |
| CIS slots (max_pool_size + 1) | **~1.4 GB total** | Default 17 slots × ~80 MB each, fixed at launch |
| Main process — KL-early-stopped update (iter 0) | **~54 GB** | Lower iter-0 footprint |
| Main process — **steady-state full update (iter 1+)** | **~57 GB** | **The killer** |

The architectural ceiling table:

| Workers | Worker GPU | Total peak | Margin (79.14 - total) | Verdict |
|---|---|---|---|---|
| **30 (canonical)** | **14.7 GB** | **~73 GB** | **~6 GB** ✓ | **SHIP** |
| 40 (proven OOM 2026-05-21) | 19.6 GB | ~78.4 GB | 0.4 GB | OOM iter 1 |
| 48 (proven OOM via Exp 1m) | 23.5 GB | ~82 GB | overflows | OOM iter 3 |

**Crucial finding**: Exp 1j (48w/pool=8) was "smooth" for 3 iters because iter 0 update was KL-early-stopped at lower memory (~47s elapsed). The full steady-state ~57 GB peak only appears at iter 1+. Earlier sessions undercounted iters and concluded 48w worked. **30w is the real ceiling at mb=64.**

---

## §2. Two-stage Phase 2 plan

### §2.1 Stage 1 — pure self-play warmup

**Purpose**: Build initial snapshot pool (for Stage 2 PFSP), let value head stabilize against self, accumulate Elo trajectory data, choose Stage 2 init checkpoint.

**Init**: `data/models/bc/v10_padded_for_cis_dev.pt` (BC v10)

**Composition (per iter, 10 active opps)**:

| Role | Count | Selection |
|---|---|---|
| Forced anchor: prev snapshot (10 iters back) | 1 | Deterministic — regression detection |
| Forced anchor: prev-of-prev snapshot (20 iters back) | 1 | Deterministic — medium-term regression |
| Random self-play from pool | 2 | Random — anti-PFSP-collapse, anti-staleness |
| PFSP self-play | 6 | PFSP-weighted, adaptive to current model strength |
| **Total** | **10** | |

Early-Stage notes:
- Iter 0: pool = 1 (just BC init). Active = 1 (mirror self-play).
- Iter 10+: snapshots accumulate, anchors become meaningful (prev = iter 0 snapshot at iter 10).
- Iter 30+: full 10-opp composition functional (2 anchors + 2 random + 6 PFSP, all distinct).

**Duration**: 50-100 iters. Trigger Stage 2 transition when:
- Smart-bot eval (`eval --interval 10`) shows clear Elo trajectory peak or plateau
- Best Elo iter has been identified (typically 30-80 range; varies)

**Output**: best Elo snapshot from Stage 1 → init for Stage 2

### §2.2 Stage 2 — with externals

**Purpose**: Train against published bots + diverse external opponents → elite gen 9 OU.

**Init**: best Stage 1 Elo snapshot (or terminal Stage 1 iter if within ~15 Elo of best)

**Day-1 external pool** (`external_adapters_phase2_day1.yaml`):
- mcts-fast (in-process, search 80ms)
- mcts-medium (in-process, search 200ms)
- mm-minikazam (Metamon Minikazam, 4.7M)
- mm-smallil (Metamon SmallIL, 13.7M)
- mm-smallrl (Metamon SmallRL, 13.9M)
- mm-mediumil (Metamon MediumIL, 50M) — **validate in smoke** (toId concerns from curated.yaml)

= **6 externals day 1**. Self-play snapshot pool starts at ~20+ snapshots from Stage 1 (KEPT, not reset).

**Composition (per iter, 10 active opps)**:

| Role | Count | Selection |
|---|---|---|
| PFSP externals | 4 | PFSP-weighted from 6-ext pool (so 4 of 6 active per iter) |
| Random external | 1 | Random from 6-ext pool |
| Forced anchor: prev snapshot | 1 | Deterministic — regression detection |
| Forced anchor: prev-of-prev snapshot | 1 | Deterministic — medium-term regression |
| Random self-play | 1 | Random from grown sp pool |
| PFSP self-play | 2 | PFSP-weighted, adaptive |
| **Total** | **10** | |

Per-iter games: 1600 / 10 = **160 games/opp** — comfortable PFSP signal for both external + self-play.

**Manual curriculum additions** (NOT auto-gated; user adds manually based on Elo poller data):

| Trigger | Action |
|---|---|
| WR vs day-1 pool ≥ 55% | Add MediumRL + MediumRL_Aug (strong tier) |
| WR vs tier-1+2 ≥ 55% | Add LargeRL |
| Stable on tier-1+2 + on cloud | Add Kakuna + Superkazam (Phase 3 tier, 140M+ params, NOT feasible on local 3060) |
| WR vs FP ≥ 30% in side eval | Add 1 FoulPlay entry at weight 0.2 |

On "manual add new tier" event:
- **Restart from best Elo snapshot of current sub-phase** (NOT hot-add). Cleaner: value head re-warms cleanly against new opp distribution. Loses ~10 iters to re-warmup, gains stability.

**Duration**: ~150 iters across Stage 2 (with potential restarts for tier additions).

---

## §3. Infrastructure

### §3.1 Canonical launch flags

```bash
./launch_rl.sh \
  --init-from <ckpt> \                            # BC v10 (Stage 1) or best Stage 1 snap (Stage 2)
  --bc-anchor-ckpt data/models/bc/v10_padded_for_cis_dev.pt \
  --bc-anchor-coef 0.1 \
  --cis --bf16 --tier3 --tier3-minibatch-size 64 \
  --packed --no-per-chunk-gc \
  --cis-min-batch 32 --cis-timeout-ms 50 \
  --mp-workers 30 \                               # LOCKED — architectural ceiling
  --games-per-iter 1600 --max-concurrent 200 \
  --n-iters 200 --warmup-iters 10 \               # 10 warmup iters for value head
  --lr 1e-5 --lam 0.95 --ent-coef 0.02 --target-kl 0.03 \
  --grad-accum 1 --turn-cap 300 \
  --reward-style terminal \
  --procedural-teams /workspace/raw_data/pokemon_usage/2024-04 \
  --adaptive-entropy --adaptive-entropy-low 0.65 --adaptive-entropy-high 0.95 \
  --eval-interval 10 --eval-games 200 --eval-team-set metamon-competitive \
  --snapshot-interval 10 \
  --early-stop --early-stop-patience 3 \
  --win-rate-mode ema --win-rate-ema-alpha 0.3 --win-rate-ema-window 50 \
  --out-dir data/models/rl_v10/<run_name> \
  [--external-adapters external_adapters_phase2_day1.yaml]  # Stage 2 only
```

**Cost projection (final)**: 23.7 min/iter × 200 iters ÷ 60 × $1.50/hr = **~$118/run** (single run). 200 iters with tier-add restarts may run ~$150-180 total across Stage 1 + Stage 2 + restarts.

### §3.2 OOM monitor wrapper

**File**: `pokemon-ai-starter/pokemon-ai/src/launch_phase2_with_oom_fallback.sh`

**Behavior** (option A, S67 decision):
- Launches training with 30w (locked)
- Monitors log every 60s for OOM / FATAL / "Training complete" / process death
- **NO auto-fallback** on OOM — exits with error code, preserves checkpoints + battle servers for manual investigation
- Rationale: 30w has 6 GB margin (vs 0.4 GB at 40w, 0.2 GB at 48w). If 30w OOMs, that signals an architectural change (model size grew, sequence packing changed, etc) — we want the signal, NOT a silent hedge to 24w that masks the real cause.

**Usage**:
```bash
# Stage 1:
STAGE=stage1 bash launch_phase2_with_oom_fallback.sh stage1_run1

# Stage 2:
STAGE=stage2 INIT_CKPT=data/models/rl_v10/stage1_run1/selfplay_v9_.../iter_0079.pt \
  bash launch_phase2_with_oom_fallback.sh stage2_run1
```

### §3.3 Pre-launch checklist

Before every launch:
1. Verify launch_rl.sh bakes in `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `LD_LIBRARY_PATH` for cuDNN 9 (already done; committed `c03af174`).
2. **Battle server cleanup** (mandatory after any prior python kill):
   ```bash
   pkill -9 -f battle_server.js
   sleep 5
   for p in 9000 9001 9002 9003 9004 9005 9006 9007; do
     nohup node battle_server.js --port $p > /tmp/bs_$p.log 2>&1 &
   done
   sleep 10
   ```
   (The wrapper handles this automatically.)
3. Verify 8 battle servers running on ports 9000-9007.
4. For Stage 2: confirm `external_adapters_phase2_day1.yaml` exists in src/.
5. For Stage 2: confirm INIT_CKPT path exists and is the best Stage 1 Elo snapshot.

---

## §4. Elo evaluation poller

### §4.1 Architecture (user S67 design)

Lightweight local process polls prod pod for new snapshots, pulls them down, runs Elo eval, adds to ladder, sleeps 5 min between polls when idle.

**Flow**:
1. Every 5 min when idle: ssh prod, list snapshots in run dir, diff against locally-evaluated set
2. New snapshot found → scp pull → rename to `<run-name>_iter<N>.pt` locally → add to local registry
3. Run `eval_elo_ladder.py --add-to <ladder.json> --snapshots <new_snap.pt> --n-games 500 --concurrency 50 --shard 0/3 ...` (3-server sharded for ~3× throughput)
4. After eval completes, append to ladder JSON
5. If 2 consecutive Elo dips of ≥25 Elo → pull intermediate snapshots between last-good and dipped, eval them in order to map the dip trajectory
6. Otherwise → poll for next new snapshot

### §4.2 Eval lineup (curated 15 anchors)

Use existing `data/eval/registry/elo_v10_500g_focused_plus_iter49.json` as base ladder via `--add-to`. **Drop the 3 abandoned snapshots** before continuing:
- `phase1v3_abandoned_iter19`
- `phase1v3_abandoned_iter39`
- `phase1v3_abandoned_iter59`

Remaining **15 anchor opps** (4 bots + 11 snapshots):

**Bots** (heuristic baselines):
- SH (anchor @ 1000 Elo)
- SmartDmg
- Tactical
- Strategic

**Snapshots** (strong reference points for Bradley-Terry calibration):
- bc_v10_cloud_e1 / e2 / e3 (BC reference)
- ppo_s39_iter229 (strong sp reference, Session 39)
- ppo_curated_iter119 (strong sp reference, curated pool)
- rl_v10_200g_iter9 (dev-prod H2H reference)
- rl_v10_1600g_iter9 / iter19 / iter29 / iter39 / iter49 (Session 65-era prod reference)

500 g/opp × 15 opps = **7500 games per new snapshot**.

### §4.3 Performance estimate (local RTX 3060)

| Setup | Throughput | Per snapshot | 20 snapshots over Phase 2 |
|---|---|---|---|
| Single server | ~30-50 g/min | 2.5-4 hours | 50-80 hours |
| **3-server sharded** | **~100-150 g/min** | **~1-1.5 hours** | **~20-30 hours** |

Local can comfortably handle this in parallel to training (poll + eval in background while prod trains).

### §4.4 Dip detection + backtrack

**Trigger**: -25 Elo over 2 consecutive evals (about 1 standard error at 500g/opp).

**Backtrack action**: pull all snapshots between last-good iter and dipped iter (e.g., if iter 50 was good and iter 70 dipped, pull iter 60 and any other intermediate not yet evaluated). Eval them in order to map the dip trajectory and identify exact iter where regression started.

**Don't auto-decide on smart-bot eval** (`eval --interval 10 --eval-team-set metamon-competitive`). It gives signal but is too noisy for auto-early-stop. Reserve for manual review.

---

## §5. Phase 2 prep TODO before launch

Status of pre-launch requirements:

| Item | Status | Notes |
|---|---|---|
| 30w validation test (5 iters) | **IN FLIGHT** | Watcher `b1qdcs6e2` waiting |
| Source 100-150 real elite gen-9-OU teams | **PENDING** | NEVER mix with 16 mm-competitive eval set (data leakage) |
| `external_adapters_phase2_day1.yaml` | **DONE** | Created 2026-05-21 |
| OOM monitor wrapper | **DONE** | `launch_phase2_with_oom_fallback.sh` updated 2026-05-21 |
| `MediumIL` smoke validation | **PENDING** | Validate Metamon MediumIL works without toId() issue before Phase 2 |
| Drop 3 abandoned snapshots from ladder | **PENDING** | Manual step before first Elo poller run |
| Elo poller script | **NOT WRITTEN** | Design spec'd in §4 above; implement as separate task |
| lr re-ablation post-S67 | **PENDING** | Confirm `--lr 1e-5` still optimal after S64 sequence packing era |
| Stage 1 launch | **READY** (pending validation + team prep) | Use wrapper with STAGE=stage1 |

---

## §6. Anti-patterns explicitly avoided (lessons captured)

1. **Iter-count fluke**: 3-iter tests are NOT enough to validate memory stability. Exp 1j (48w/pool=8 "smooth" for 3 iters) led us astray; iter 1+ steady-state full update is ~3 GB heavier than iter 0 KL-early-stopped. **Run 5+ iters for memory validation.**

2. **40w gambling on 0.4 GB margin**: tempting at $90/run vs 30w's $118, but proved structurally non-viable. Don't re-attempt 40w+ without an architectural change (CPU-only-workers refactor would free ~15 GB).

3. **Soft fallback as architectural mask**: auto-fallback from 30w → 24w on OOM would HIDE the signal that something architectural moved. Explicit failure is better than silent hedge.

4. **Pool capping as wall optimization**: REFUTED in S67 (Exp 5a). Pool size doesn't affect wall; active opps per iter does. Pool can grow unbounded.

5. **PFSP-only on critical opps**: PFSP weights by current WR, missing regression. Forced inclusion of prev + prev-of-prev anchors is cheap insurance against the Phase 1 v3 type-knowledge-erosion failure mode.

6. **Hot-add new opp tier mid-run**: cleaner to restart from best Elo snapshot (re-warms value head). The ~10 iter cost is worth the stability gain.

---

## §7. Cross-references

**Memory (auto-loaded)**:
- [project_s67_final_handover.md](project_s67_final_handover.md) — S67 worker scaling + 30w ceiling
- [project_s67_worker_scaling_findings.md](project_s67_worker_scaling_findings.md) — full experiment data
- [feedback_battle_server_restart_after_kill.md](feedback_battle_server_restart_after_kill.md) — BS cleanup recipe
- [feedback_dont_propose_principle_violations.md](feedback_dont_propose_principle_violations.md) — Track B rejection lesson
- [project_phase1_v3_diagnosis.md](project_phase1_v3_diagnosis.md) — regression-detection rationale

**Docs in git**:
- [docs/S67_WORKER_SCALING_RESULTS.md](S67_WORKER_SCALING_RESULTS.md) — worker ratio + memory data
- [docs/PHASE2_LAUNCH_PLAN.md](PHASE2_LAUNCH_PLAN.md) — THIS FILE

**Scripts**:
- `pokemon-ai-starter/pokemon-ai/src/launch_phase2_with_oom_fallback.sh` — OOM monitor wrapper
- `pokemon-ai-starter/pokemon-ai/src/external_adapters_phase2_day1.yaml` — day-1 external pool
- `pokemon-ai-starter/pokemon-ai/src/eval_elo_ladder.py` — Elo eval (existing, mature)
- `pokemon-ai-starter/pokemon-ai/src/run_elo_shards.sh` — 3-server parallel eval launcher

---

## §8. Open design questions (none blocking launch)

1. **Elo poller implementation**: spec'd in §4 but not yet written. Can be a separate session task.
2. **Stage 1 → Stage 2 transition timing**: planned ~50-100 iters but exact cutoff depends on Stage 1 trajectory data. Decide when Stage 1 plateau is observed.
3. **lr re-ablation**: should be done in a quick smoke (~$10) post-S67 to confirm `--lr 1e-5` still optimal with packed/sequence + 30w.
4. **MediumIL smoke**: $1-2 smoke to confirm the day-1 YAML doesn't have toId() issues.

---

## §9. Operational HOW-TO (next-session ready-to-run)

This section is for any session (fresh Claude or human) to execute Phase 2 without
needing to re-derive operational details. Verified working 2026-05-21.

### §9.1 Pod connection

```bash
# Prod pod (current as of 2026-05-21):
ssh -i ~/.ssh/id_ed25519 -p 47913 -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@195.26.233.30
# Code root:    /workspace/team_builder/
# Source dir:   /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src/
# torch:        2.5.1+cu121, triton 3.1.0 (verified 2026-05-21)
# GPU:          A100 80GB SXM, 79.14 GB usable, $1.50/hr
# BS ports:     9000-9007 (8 battle servers)
```

If SSH details change in future, verify via RunPod dashboard — IP/port can drift if pod restarts.

### §9.2 Pre-Phase-2 prod setup (ONE TIME before first Phase 2 launch)

```bash
# 1. SSH to prod
ssh -i ~/.ssh/id_ed25519 -p 47913 -o StrictHostKeyChecking=no root@195.26.233.30

# 2. Verify clean state (no zombie training procs)
pgrep -c python  # expect 0 (or only orphan multiprocessing wrappers)
nvidia-smi --query-gpu=memory.used --format=csv,noheader  # expect ≤ 500 MiB

# 3. Checkout master + pull latest (prod might be on a feature branch)
cd /workspace/team_builder
git status  # check current branch
git checkout master
git pull origin master
git log -1 --oneline  # confirm has the S67 REV 2 commit 92751bc or newer

# 4. Verify the wrapper + YAML are present (committed in 92751bc)
ls -la pokemon-ai-starter/pokemon-ai/src/launch_phase2_with_oom_fallback.sh
ls -la pokemon-ai-starter/pokemon-ai/src/external_adapters_phase2_day1.yaml

# 5. Verify torch + triton (S51 lesson — version drift is a real failure mode)
python -c "import torch, triton; print(torch.__version__, triton.__version__)"
# Expected: 2.5.1+cu121 3.1.0

# 6. Verify battle servers running
ss -ltn | grep -cE ':900[0-7]'  # expect 8
```

If any of #2-#6 fails, fix before proceeding. Do NOT launch over a dirty state.

### §9.3 Launch Stage 1 (pure self-play warmup)

```bash
# On prod, in src/ directory:
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
STAGE=stage1 nohup bash launch_phase2_with_oom_fallback.sh phase2_stage1_v1 \
  > /tmp/phase2_stage1_v1_wrapper.log 2>&1 &

# Wrapper writes training log to: /tmp/phase2_stage1_v1.log
# Wrapper handles BS restart automatically.
```

**Expected behavior**:
- BS restart in first ~10s
- Training process starts (banner "BC anchor: loading reference..." appears)
- Iter 0 collection begins ~30s after launch
- First "Iter 0:" line lands in ~25 minutes (collect ~21 min + update ~2.5 min)
- Subsequent iters land every ~24 minutes

**Live monitoring** (from local or another SSH session):
```bash
# Iter completion lines (one per iter):
ssh ... 'grep -E "^\[..:..:..\] Iter [0-9]+:" /tmp/phase2_stage1_v1.log'

# Last 30 lines (to see in-progress collect):
ssh ... 'tail -30 /tmp/phase2_stage1_v1.log'

# OOM check:
ssh ... 'grep -iE "out of memory|FATAL" /tmp/phase2_stage1_v1.log'

# Process + GPU health:
ssh ... 'pgrep -c python; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader'
```

### §9.4 Launch Stage 2 (with externals — after Stage 1 best Elo identified)

```bash
# Identify best Stage 1 snapshot (run AFTER Stage 1 + Elo eval complete):
# From local Elo ladder JSON, find snapshot with highest Elo from Stage 1 run.
# Then pull that snapshot's path on prod:
ssh ... 'ls -la /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src/data/models/rl_v10/phase2_stage1_v1/selfplay_v9_*/iter_*.pt | sort -k9 -V'

# Pick the best-Elo iter checkpoint, then launch Stage 2:
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
STAGE=stage2 \
  INIT_CKPT=data/models/rl_v10/phase2_stage1_v1/selfplay_v9_XXXX/iter_NNNN.pt \
  nohup bash launch_phase2_with_oom_fallback.sh phase2_stage2_v1 \
  > /tmp/phase2_stage2_v1_wrapper.log 2>&1 &
```

### §9.5 Snapshot pull to local (for Elo eval poller)

```bash
# Pull a single snapshot from prod to local
LOCAL_DIR=C:/Users/raiad/OneDrive/Desktop/team_builder/data/cloud_runs/phase2_stage1_v1
mkdir -p "$LOCAL_DIR"

scp -i ~/.ssh/id_ed25519 -P 47913 \
  root@195.26.233.30:/workspace/team_builder/pokemon-ai-starter/pokemon-ai/src/data/models/rl_v10/phase2_stage1_v1/selfplay_v9_*/iter_NNNN.pt \
  "$LOCAL_DIR/phase2_stage1_v1_iter_NNNN.pt"

# Pull all snapshots not yet locally (Elo poller pattern):
ssh ... 'ls /workspace/team_builder/.../selfplay_v9_*/iter_*.pt' \
  > /tmp/prod_snapshots.txt
# diff against local, scp the new ones with name rewrite to phase2_stage1_v1_iter_NNNN.pt
```

### §9.6 Run an Elo eval against the curated ladder (3-server sharded)

```bash
cd C:/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src

# Make sure 3 battle servers are running locally (ports 9000/9001/9002).
# Then run 3 parallel shards:
NEW_CKPT=../data/cloud_runs/phase2_stage1_v1/phase2_stage1_v1_iter_0049.pt
NEW_NAME=phase2_stage1_v1_iter49
LADDER=data/eval/registry/elo_v10_500g_focused_plus_iter49.json
OUT_BASE=data/eval/registry/elo_phase2_stage1_v1_iter49

python -u eval_elo_ladder.py --add-to "$LADDER" \
  --snapshots "$NEW_CKPT" --names "$NEW_NAME" \
  --n-games 500 --concurrency 50 --device cuda \
  --shard 0/3 --server ws://127.0.0.1:9000/showdown/websocket \
  --out-json "${OUT_BASE}_shard0.json" &

python -u eval_elo_ladder.py --add-to "$LADDER" \
  --snapshots "$NEW_CKPT" --names "$NEW_NAME" \
  --n-games 500 --concurrency 50 --device cuda \
  --shard 1/3 --server ws://127.0.0.1:9001/showdown/websocket \
  --out-json "${OUT_BASE}_shard1.json" &

python -u eval_elo_ladder.py --add-to "$LADDER" \
  --snapshots "$NEW_CKPT" --names "$NEW_NAME" \
  --n-games 500 --concurrency 50 --device cuda \
  --shard 2/3 --server ws://127.0.0.1:9002/showdown/websocket \
  --out-json "${OUT_BASE}_shard2.json" &
wait

# Combine shards into final result:
python -u eval_elo_ladder.py --combine "${OUT_BASE}_shard0.json" \
  "${OUT_BASE}_shard1.json" "${OUT_BASE}_shard2.json" \
  --out-json "${OUT_BASE}_FINAL.json"
```

**Before first use**: manually drop the 3 abandoned snapshots from the base ladder
JSON (`phase1v3_abandoned_iter19/39/59`) so they don't pollute future BT fits.

### §9.7 MediumIL smoke (before Stage 2 launch)

The day-1 YAML includes `mm-mediumil`. Curated.yaml flagged it as "suspicious" for
toId() / showdown_username issues. Validate before committing to Phase 2:

```bash
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
# Quick smoke: 50 games against ONLY mediumil
cat > /tmp/mediumil_smoke.yaml <<EOF
opponents:
  - name: mm-mediumil
    type: metamon
    model: MediumIL
    temperature: 1.0
    server_port: 9000
    weight: 1.0
EOF

nohup ./launch_rl.sh \
  --init-from data/models/bc/v10_padded_for_cis_dev.pt \
  --bc-anchor-ckpt data/models/bc/v10_padded_for_cis_dev.pt \
  --cis --bf16 --tier3 --tier3-minibatch-size 64 --packed --no-per-chunk-gc \
  --cis-min-batch 32 --cis-timeout-ms 50 --mp-workers 8 \
  --games-per-iter 50 --max-concurrent 50 \
  --n-iters 1 --warmup-iters 0 --lr 1e-5 \
  --reward-style terminal --procedural-teams /workspace/raw_data/pokemon_usage/2024-04 \
  --external-adapters /tmp/mediumil_smoke.yaml \
  --out-dir /tmp/mediumil_smoke_run \
  > /tmp/mediumil_smoke.log 2>&1 &

# Watch for either:
# - "Iter 0:" line with responded=N matching (success)
# - "hang at iter 0 — 0 of 8 expected MM challenges sent" pattern (failure → toId issue)
```

If smoke hangs at iter 0 with MM never challenging: open issue, drop MediumIL from
day-1 YAML, replace with another easy/medium metamon (e.g., add a 2nd Minikazam or
double SmallRL/SmallIL weight).

### §9.8 Stage 1 → Stage 2 transition criteria

**When is Stage 1 done?** Look at smart-bot eval trajectory (auto-logged at
`data/eval/registry/evals.jsonl` every 10 iters via `--eval-interval 10`) AND
Elo poller results:

- **Eval trajectory plateau**: smart_avg WR stops improving for 20+ iters
- **Elo trajectory peak**: best Elo iter has been ≥10 iters ago and not surpassed
- **Operational floor**: at minimum, 50 iters of Stage 1 before considering done

**Pick Stage 2 init**:
- If terminal iter Elo is within ~15 points of best-Elo iter → use terminal (simpler)
- Otherwise → use the best-Elo snapshot explicitly

**Don't auto-decide**. This is a manual review point with the user. Surface the trajectory data, recommend, and confirm before proceeding.

### §9.9 Python shutdown hang workaround (S64-era issue)

After "Training complete" prints, the main python proc often hangs in cleanup
because CIS workers don't exit gracefully. Kill explicitly:

```bash
ssh ... 'pgrep -af "python.*train_rl" | head -1'  # find PID
ssh ... 'kill -9 <PID>; sleep 3; pkill -9 -f spawn_main; sleep 5; pgrep -c python'
```

If pgrep shows lingering procs, find spawn-main parent + kill it:
```bash
ssh ... 'ps auxf | grep -i "spawn_main" | grep -v grep | head -5'
ssh ... 'kill -9 <PARENT_PID>; sleep 5; pgrep -c python'
```

### §9.10 SSH disconnect during operations

`pkill -9 python` on prod can drop your SSH connection (exit code 255 because the
SSH-child process gets killed too). This is normal. Reconnect and verify state
with the §9.2 checks. The kill landed; the SSH just got caught in the blast radius.

### §9.11 Cost monitoring during long runs

Pod cost = $1.50/hr × hours running. To check pod uptime / billing:
- RunPod dashboard (browser, user account)
- `uptime` on the pod gives system uptime (not billing exactly, but proxies)

For Phase 2:
- Stage 1 (50-100 iters × 24 min) ≈ $30-60 in pod time
- Stage 2 (150 iters × 24 min) ≈ $90
- Total Phase 2 first run ≈ **$120-150**; restarts for tier additions could add $30-60

Watch for OOM mid-run (wrapper exits with code 1 + preserved checkpoints) — that's
when human review is required.

