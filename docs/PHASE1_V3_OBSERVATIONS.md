# Phase 1 v3 Observations — what S50 fixed, what remains, Phase 2 design implications

**Run:** `data/models/rl_v10/ppo_phase1_v3_compiled/selfplay_v9_20260507_235306/`
**Status:** in flight, iter 41+ at time of writing (2026-05-09)
**Purpose:** measure pure self-play baseline with S50 postmortem fixes applied; characterize what those fixes did and didn't address; inform Phase 2 design.

This doc is a **follow-up** to `PHASE1_POSTMORTEM.md`, not a replacement. The S50 postmortem identified three compounding causes; only one was applied to Phase 1 v3. This doc documents which.

---

## TL;DR

Phase 1 v3 is **tracking the S50 trajectory at iter 39+49 despite the lr=1e-5 fix being applied** — 53% / 49% smart_avg vs S50's 54% / ~51% (interp) at the same iters. Different per-bot signature (we're stronger vs SH, weaker vs Tactical). The lr fix removed cause #1 from S50; causes #2 (team distribution) and #3 (perm/canonical) are still present, accounting for the remaining drop.

**Update (iter 49 EVAL — 2026-05-09)**: smart_avg dropped from iter 39's 53% to **49%**. Per-bot SH=59 SmartDmg=44 Tactical=42 Strategic=52. Below the 50% concerning-band threshold. Still within expected baseline range (per §7 below); hard abort triggers (smart_avg <30%, W/L <35% sustained, KL >0.06, v_loss >5.0) all NOT firing. Continue. Iter 59 next eval data point.

**Phase 1 v3 will likely land at smart_avg ~35-50% by iter 200**, similar to S50. That's the **expected baseline measurement** for "pure self-play with lr fix only." Not elite. The point of the run is to confirm this ceiling so Phase 2 can target the remaining gaps.

**Phase 2 launches with the S50 postmortem fixes that were never applied** + the multi-gen-AGNOSTIC infra (Phase 4.6 already shipped, Tier 3 + train_step compile pending, bf16 update enable, H2H gauntlet wiring, Network Volume).

**Strict sequencing for Phase 2 launch:**
1. Phase 1 v3 must complete cleanly (current run, ~6 days remaining)
2. Infra must be ready: Tier 3 + train_step compile shipped + validated; bf16 update flag enabled; H2H gauntlet wired; Network Volume verified
3. THEN Phase 2 launches with all training-distribution + lr-scheduling changes

The training-distribution + lr changes documented below are **NOT mid-run interventions**. They go into Phase 2 launch, after the infra optimizations are in place.

---

## 1. What S50 identified, what Phase 1 v3 applied, what remains

| S50 cause | Estimated impact | Applied to Phase 1 v3? | Status for Phase 2 |
|---|---|---|---|
| **lr=3e-5 too high** | -20-25pp smart_avg | ✅ `--lr 1e-5` locked | Re-ablate after Tier 3 lands; consider scheduling |
| **Procedural train, mm-competitive eval distribution shift** | -15-20pp | ❌ Still 100% procedural in training | **Fix in Phase 2** (see §2) |
| **Perm features at training, canonical at eval** | -7pp | ❌ Mismatch still present | **Fix in Phase 2** (see §3) |
| **Self-play attractor (no external opps)** | (also contributed) | ❌ Pure self-play | **Fix in Phase 2** (see §4) |
| **Tighter early-stop** | (catches drift sooner) | ⚠️ Partially (`--early-stop --early-stop-patience 3`) | Add absolute threshold |

Net expected from causes #2 + #3 still present: **~22-27pp combined drop potential**. We've seen 16pp at iter 39, tracking the math.

---

## 2. Team distribution mix — the big lever

**Goal**: bridge the train/eval distribution gap without leaking the eval set into training.

**Critical methodological constraint**: the 16 metamon-competitive teams used for eval **MUST NOT be used during training**. Mixing them in would be data leakage — the model would memorize those specific compositions, eval scores would inflate artificially, and we'd lose the eval signal.

**Correct three-pool design for Phase 2 PPO self-play:**

| Pool | Source | Purpose | Used in |
|---|---|---|---|
| **Procedural** | `raw_data/pokemon_usage/2024-04` (current) | Pokemon-fundamentals learning, type-matchup breadth, slot-permutation robustness | Training only |
| **Real elite teams (NEW)** | 100-150 teams **copied verbatim from real competitive sources** — Smogon team archive, Showdown ladder replays, tournament team posts, /r/stunfisk OU sample teams, top-ladder usage stats. **Not generated, not estimated, not synthesized — actual teams used by real strong players.** | Coordinated-strategy learning, eval-distribution proxy | Training only |
| **mm-competitive (16)** | Existing `metamon_cache/teams/competitive/gen9ou/` | Eval baseline | **Eval only — never in training** |

**Mix during training**: ~50-70% procedural / ~30-50% real elite teams, randomly sampled per battle, both sides independently. Procedural keeps Pokemon-fundamentals breadth (the user's load-bearing intent: "teach Pokemon, not specific teams"). Real elite teams add coordinated-team distribution coverage.

**Why both sides should sample independently**: avoids the "both sides have random uncoordinated teams" degenerate-equilibrium problem the S50 postmortem identified, while preserving slot-permutation invariance.

**Sourcing the 100-150 real elite teams** (one-time effort before Phase 2):
- Smogon team archive (https://www.smogon.com/teams/) — gen 9 OU teams from RMTs and tournament reports
- Showdown ladder replay scraping — top-100 ladder players, gen 9 OU, last 30 days
- /r/stunfisk OU sample teams threads
- Recent OST / OLT tournament team posts
- VR (Viability Rankings) sample teams from Smogon

Quality bar: must be a team a top-100 OU ladder player ACTUALLY USED, not a hypothetical "good team". 100-150 teams is roughly tournament-level diversity coverage.

---

## 3. Perm/canonical feature mismatch

**S50 postmortem (§7.3)**: V9RLPlayer trains with permuted features (training=True, slot order randomized per turn), eval uses canonical (training=False, fixed slot order). The model becomes asymmetric — it learns position-dependent representations that don't transfer to canonical eval.

**Why perm at training is correct (load-bearing design intent)**: prevents the model from learning "slot 0 is always X" patterns. Forces position-invariant representations. This is the right training design for general Pokemon AI.

**The fix is NOT to remove perm.** Two options:
- **(a) Enable perm at eval**: match training distribution. Should give ~+7pp smart_avg per S50 measurement.
- **(b) Train longer to enforce invariance**: the transformer arch SHOULD be permutation-invariant by attention construction. The fact that perm/canonical differs (-7pp) means residual position info is leaking. More training with perm augmentation might converge to true invariance.

Option (a) is cheap (one-line config change), option (b) is principled but uncertain. **Recommendation for Phase 2: ship (a), measure perm-eval delta. If still significant after Phase 2 trains with perm + team-mix, investigate (b).**

---

## 4. External opponents — confirmed Phase 2

Per `project_strategic_frame.md` and S50 postmortem: pure self-play creates an attractor where strategies that "don't lose to yourself" win. External opponents break this attractor — model has to develop strategies that beat opponents IT DIDN'T TRAIN AGAINST.

**Phase 2 PFSP pool composition** (post-Phase 1 v3):
- Phase 1 v3 final.pt (current run end-state) as main player init
- Phase 1 v3 snapshots (existing ~40 snapshots, more by iter 200)
- `mcts-fast` (in-process MCTS — no extra subprocess infra needed)
- Kakuna / SmallRL / Minikazam (Metamon published bots — needs H2H gauntlet wiring done in STEP 6)
- BC v10 e3 (the project peak Elo baseline)

PFSP weighting (existing logic): `(1 - wr)^2` — opponents we lose to get sampled more. External opps will dominate sampling early, naturally pulling the model toward strategies that beat strong play.

**The first 200-iter Phase 1 v3 is intentionally pure self-play** — the baseline measurement of where self-play alone caps out. Phase 2 measures the lift from external opps.

---

## 5. lr — third-tier optimization, not the bottleneck (but needs revisit)

The S50 4-point lr ablation validated lr=1e-5 as **stable** over 5 iters, not as **optimal across the full training curve**. Phase 1 v3's tracking of S50's iter 39 trajectory despite lr=1e-5 confirms: **lr is not the current bottleneck — team distribution + perm mismatch are.**

But lr does deserve revisit for Phase 2, in three specific ways:

### 5.1 lr scheduling (constant lr is probably wrong for full curve)

S50 ablation only ran 5 iters per point. Constant lr=1e-5 across all 200 iters is convenient but probably suboptimal:
- **Early iters (0-15)**: gradients are accurate (large 1600-game batch), policy is far from optimum → larger steps tolerable
- **Late iters (50+)**: policy is more refined, large steps overshoot → smaller lr safer

**Phase 2 candidate**: cosine decay from `5e-6` to `5e-7` over 200 iters, OR step decay (1e-5 for iter 0-50, 5e-6 for iter 50-150, 1e-6 for iter 150+). Need ablation.

### 5.2 Re-ablation after Tier 3 lands

Tier 3 transition-level minibatching changes effective batch size per gradient step from ~48 episodes to 256-2048 transitions. Larger accurate batches → can justify higher lr per the linear lr scaling rule (broadly applies to PPO too).

**Mandatory after Tier 3**: 4-point lr ablation again. Probably 2e-5 to 5e-5 becomes optimal once batch shape changes.

### 5.3 KL early-stop currently masks the lr question

Phase 1 v3 KL fires at epoch 2-3 of 5 every iter — meaning `lr × n_epochs` is already at the trust-region wall. Lowering lr just delays hitting the wall (slower learning, same wall). Raising lr pushes past the wall faster (bad). Until we change either Tier 3 batching OR the KL target, lr changes alone won't move the needle.

**Practical Phase 2 plan for lr**:
- Initial: keep `--lr 1e-5` (validated baseline)
- Add `--lr-schedule cosine --lr-min 1e-6` if scheduling is implemented (TODO)
- After Tier 3 ships: full 4-point ablation, 5 iters each: 5e-6, 1e-5, 2e-5, 5e-5
- Update cookbook + lock new winner

---

## 6. Sequencing for Phase 2 launch — strict order

**Phase 2 cannot launch until ALL of the following are in place** (locked by user constraint):

1. **Phase 1 v3 completes cleanly** (currently running, ~6 days remaining)
2. **Infra optimizations all shipped + validated:**
   - ✅ CIS Phase 4.3 + 4.4 + 4.5 (S53)
   - ✅ CIS Phase 4.6 unified recovery (S54, just shipped)
   - ⏳ Tier 3 transition-level minibatching (~1.5-2 wks dev + 20-iter A/B)
   - ⏳ `torch.compile(train_step)` (bundled with Tier 3)
   - ⏳ bf16 update autocast (code shipped S52, needs `--bf16` flag at relaunch)
3. **Eval infra ready:**
   - ⏳ H2H gauntlet wired (Kakuna/SmallRL/Minikazam adapters, STEP 6, ~3-4 hrs)
   - ⏳ Network Volume verified on Phase 2 pod (STEP 7, ~10 min)
4. **Training-distribution changes prepared:**
   - ⏳ 100-150 real elite teams sourced + packaged as a separate teambuilder pool
   - ⏳ Mix logic in `team_generator.py` or self-play config (50-70% procedural / 30-50% elite mix per battle, both sides independent)
   - ⏳ Eval set isolation verified (16 mm-competitive teams NEVER in training pool)
5. **Perm/canonical eval consistency**:
   - ⏳ Enable perm at eval (one-line config) OR document why we keep canonical-at-eval
6. **lr scheduling** (optional but recommended):
   - ⏳ Implement `--lr-schedule cosine` in train_rl.py
   - ⏳ Pre-Phase-2 5-iter validation

The training-distribution changes (§2-4) are **multi-iter aggregate effects**, NOT mid-Phase-1-v3 interventions. Don't apply mid-run. Apply at Phase 2 launch.

---

## 7. What Phase 1 v3's expected outcome means

Phase 1 v3 will measure: "with all S50 fixes that don't require changing training distribution applied (lr, lam, ent_coef, warmup, adaptive entropy), where does pure self-play cap?"

Best estimate based on current trajectory (53% at iter 39, 49% at iter 49, S50 trajectory):
- iter 59: ~46-50%
- iter 79: ~35-45%
- iter 200: ~30-40%

This is **NOT elite**. It's a baseline. Phase 2 will measure how much the team-mix + perm-fix + external-opps + tier-3 stack add on top.

**Don't abort Phase 1 v3.** The smart_avg=53% at iter 39 IS the expected outcome of "lr fix only, no team mix, no external opps." Aborting would lose the baseline measurement.

**Watch for emergency rollback only if:**
- smart_avg drops below 30% (catastrophic, worse than S50)
- W/L vs PFSP pool drops below 35% sustained (model can't even beat itself)
- KL trajectory diverges (going up sustainably above 0.06)
- v_loss explodes (above 5.0)

None of these are happening as of iter 41. Continue to iter 200.

---

## 8. Memory + cross-references

- **Postmortem (root)**: `docs/PHASE1_POSTMORTEM.md` — S50 catastrophic regression, lr=3e-5 root cause
- **Diagnosis report**: `docs/PHASE1_DIAGNOSIS_REPORT.md` — lr ablation matrix
- **Investigation plan**: `docs/PHASE1_INVESTIGATION_PLAN.md` — 7 Tier-1/2 tests for the regression
- **Strategic frame**: `memory/project_strategic_frame.md` — sequencing rules, multi-gen scope
- **Phase 4.6 design**: `memory/project_cis_4_6_design.md` — failure recovery (just shipped)
- **Engineering standard**: `memory/feedback_engineering_standard.md` — read FIRST every session
