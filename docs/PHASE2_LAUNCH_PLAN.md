# Phase 2 Launch Plan — S67 Final (2026-05-21)

**Status**: Design FINALIZED. Pending validation: 30w/5-iter memory stability test (in flight at handover).

**Authoritative companion**: `memory/project_phase2_launch_plan.md` (same content, auto-loaded).

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
