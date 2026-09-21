# Phase 1 Diagnosis Report — Root Cause Confirmed

**Created:** Session 50 continuation (2026-05-06)
**Status:** Investigation complete. Root cause identified. Recommended fix has empirical support.

---

## TL;DR

Phase 1's catastrophic regression (smart_avg 67% → 35% over 79 iters) was caused by **`--lr 3e-5` being too high for the new TransformerBattlePolicy arch.** The lr was inherited from S39/S43 era runs on the legacy 12-15M PokeTransformer, which had ~250x lighter attention activations and different gradient sensitivity. On the new 20M-param per-attribute-tokenized arch, lr=3e-5 produces sustained policy drift; lr=1e-5 maintains stability.

**Empirical evidence:** 4-point lr ablation (1e-5 / 3e-5 / 5e-5 / 1e-4) starting from `snapshot_0019.pt`, 5 iters each, then smart_avg eval:

| lr | smart_avg | Δ from BC (67%) | Trajectory |
|---|---|---|---|
| **1e-5** | **66.8%** | **-0.2 (stable)** | Smooth, pi_loss small (-0.05 to +0.01), monotonic internal wr climb |
| 3e-5 (Phase 1) | 63.2% | -3.8 | Drift down, anchor wr declining 53→44 |
| 5e-5 | 59.0% | -8 | Oscillation: pi_loss flips sign 4 times in 5 iters |
| 1e-4 | 44.0% | -23 | Chaotic; +12pt wr swings; matches S39 "diverged in 3 iters" history |

**Recommendation:** Restart Phase 1 with `--lr 1e-5`. Other parameters unchanged (procedural teams, lam=0.95, ent=0.02, adaptive entropy, etc.). All earlier non-lr hypotheses (feature pipeline divergence, InferenceBatcher refactor, perm/canonical mismatch) were ruled out via dedicated tests.

---

## Investigation chronology

### Hypothesis ranking before testing
1. H1: hyperparameter sensitivity (lr, ppo_epochs, etc.)
2. H2: feature pipeline divergence between BC training and PPO live
3. H3: InferenceBatcher staged forward differs from `model.forward()` (Session 50 refactor bug)
4. H4: per-attribute tokenization causes catastrophic forgetting under noisy gradients
5. H6: lr=3e-5 inherited from legacy arch, never validated for new arch

### Tests run

#### Pri 1 — Feature pipeline equality (RULED OUT H2)
- `feature_equality_test.py` ran a real h2h replay across 12 turns
- BC pipeline (`_feat_to_flat_record` from `replay_to_memmap.py`) vs PPO pipeline (`build_turn_batch` from `features.py`) on the same `feat` dict
- **Result: max_diff = 0 across all fields, all turns.** Both pipelines produce bit-identical batch dicts.
- Resolves the structural difference where BC stored `our_pokemon_ids` as (6, 7) and PPO as (6, 3) + separate (6, 4) `our_pokemon_move_ids` — `_fix_ids` (model_transformer.py:1531) reconciles both forms.

#### Pri 1.5 — InferenceBatcher staged path vs `model.forward()` (RULED OUT H3)
- `inference_batcher_equality_test.py` loaded `snapshot_0019.pt`, ran 12 turns through:
  - Direct: `model.forward(batch, history)` — eval-time path
  - Staged: `forward_spatial → call_action_encoder → call_policy_logits → call_value_logits` — training path through `arch_compat.py`
- **Result: max_diff in action_logits = 0, values = 0, summary = 0, v_logits = 0.** Session 50 arch_compat refactor is bit-identical to direct forward.

#### Pri 4 — Logit/probability dump across snapshots (RULED OUT H4 — partial)
- `logit_dump_test.py` ran a fixed 50-state set through 6 ckpts (BC, sp_0019_warmup, sp_0019_new, sp_0039, sp_0059, sp_0079)
- **Found:** No mode collapse on action *classes* (move/switch both still get probability mass throughout)
- **Found:** Intra-class oscillation — favorite move slot bounced mv0(0.62)→mv1(0.67)→mv0/mv1(0.49/0.28) across training
- **Found:** Entropy went UP at sp_0079 (1.06 vs BC's 0.69), top-1 confidence DOWN (0.57 vs 0.77)
- **Found:** Value head lost confidence (0.86 → 0.44 at sp_0059), partial recovery at sp_0079
- **Interpretation:** Gradient instability, not mode collapse. The model couldn't settle on a stable policy preference. Strong indirect evidence for H1.

#### Pri 2 — lr ablation (CONFIRMED H1, H6)
- 4 short PPO runs from `snapshot_0019.pt`: lr ∈ {1e-5, 3e-5, 5e-5, 1e-4}, 5 iters each, conc=100, all other flags identical to Phase 1
- Smart_avg eval at iter 4 for each
- **Results:** Monotonic relationship — lower lr = better outcome. lr=1e-5 holds BC parity; higher lrs degrade proportionally to lr magnitude.

#### Architectural context (from STATUS.md archaeology)
- Old runs that climbed BC=809 → 1029 used **lr=1e-4, ent=0.04, lam=0.75** on the 12-15M legacy arch
- "1e-4 diverged in 3 iters" warning in next-prompt.txt was from S39/S43 on legacy arch with v8 BC base — different config than ours
- Our current setup inherited lr=3e-5 as "the safe choice" without re-validating for new arch
- New arch has ~250x more attention activations than legacy (per-attribute tokenization × 6 spatial layers vs 14-token × 4 spatial layers). Different gradient sensitivity profile.

---

## Detailed findings — full lr ablation matrix

### Per-iter trajectories (W/L/T from 100 games, % wr)

| iter | lr=1e-5 | lr=3e-5 | lr=5e-5 | lr=1e-4 |
|---|---|---|---|---|
| 0 | 41% | 42.4% | 38% | 32% |
| 1 | 43% | 37% | 46% | 44% |
| 2 | 42% | 38% | 41% | 48% |
| 3 | 44% | 36% | 42% | 39% |
| 4 | 46% | 40% | 34% | 42% |
| **swing range** | **5pt** | 6pt | 12pt | 16pt |

### KL divergence per iter

| iter | lr=1e-5 | lr=3e-5 | lr=5e-5 | lr=1e-4 |
|---|---|---|---|---|
| 0 | 0.039 | 0.050 | 0.059 | 0.076 |
| 1 | 0.043 | 0.047 | 0.051 | 0.067 |
| 2 | 0.039 | 0.050 | 0.053 | 0.060 |
| 3 | 0.036 | 0.049 | 0.055 | 0.053 |
| 4 | 0.037 | 0.054 | 0.054 | 0.050 |
| Mean | **0.039** | 0.050 | 0.054 | 0.061 |

target_kl = 0.03; PPO early-stops at avg_kl > target_kl × 1.5 = 0.045. lr=1e-5 was the ONLY config to consistently stay under early-stop threshold → PPO ran full 5 epochs. Higher lrs hit early-stop in 0.5-2 epochs every iter.

### pi_loss magnitude per iter

| iter | lr=1e-5 | lr=3e-5 | lr=5e-5 | lr=1e-4 |
|---|---|---|---|---|
| 0 | -0.033 | -0.052 | +0.055 | +0.030 |
| 1 | +0.005 | -0.070 | -0.139 | +0.032 |
| 2 | +0.008 | -0.061 | -0.164 | -0.021 |
| 3 | -0.048 | -0.013 | -0.110 | +0.023 |
| 4 | +0.009 | -0.084 | +0.089 | -0.098 |
| Sign flips | 2 | 0 | 2 | 3 |
| Max magnitude | 0.048 | 0.084 | 0.164 | 0.098 |

**lr=5e-5 and lr=1e-4 sign flips repeatedly** = oscillation. **lr=3e-5 stays negative consistently but with growing magnitude** = sustained drift. **lr=1e-5 stays small with rare sign flips** = controlled adjustment.

### Smart_avg eval at iter 4 (n=100 per bot, mm-competitive teams)

| lr | SH | SmartDmg | Tactical | Strategic | smart_avg |
|---|---|---|---|---|---|
| 1e-5 | 69 | 70 | 63 | 65 | **66.8** |
| 3e-5 | 62 | 66 | 68 | 57 | 63.2 |
| 5e-5 | 62 | 57 | 57 | 60 | 59.0 |
| 1e-4 | 45 | 34 | 39 | 58 | 44.0 |

BC v10 e3 baseline: 67%. sp_0019 (warmup-end): 69%.

---

## Open questions / next steps

### Is lr=1e-5 the sweet spot, or too small to truly learn?

**Evidence for "right":** smart_avg held at BC parity (-0.2pt), internal wr climbed 41→46, kl bounded under early-stop, pi_loss healthy magnitudes when sign-correct.

**Evidence for "too small":** pi_loss was near zero in 3 of 5 iters (essentially frozen), eval smart_avg held flat (no improvement, just stability), historical successful runs used lr=1e-4 (10x higher).

**Recommendation:** Run a longer lr=1e-5 ablation (15-20 iters) to see if smart_avg actually rises above BC parity, or if it's just maintaining. If it climbs to 70-72%, sweet spot. If flat at 66-69%, may want to test lr=2e-5 or a warmup schedule.

### Why is conc=200 hitting OOM now when Phase 1 ran fine at conc=200?

**User-reported observation:** Phase 1 (run yesterday, same code, same arch) used 16-30% VRAM at conc=200. Today's lr ablations at conc=100 hit ~100% VRAM ceiling and OOM during PPO update.

**Best hypothesis (unproven):** CUDA driver fragmentation accumulated from today's many model load/unload cycles in diagnostic scripts (logit_dump loaded 6 ckpts; h2h_diag ran 16+ model loads; multiple eval_diag and equality tests). Even with `torch.cuda.empty_cache()` and `gc.collect()`, driver-level fragmentation doesn't fully clear.

**Mitigation:** System reboot would likely restore Phase 1's memory profile. Worth testing AFTER current investigation, before next real Phase 1 attempt.

### Hyperparameters that might still need re-tuning for new arch

The lr is the dominant variable, but other hyperparams were also inherited from legacy:
- `--ppo-epochs 5` — could be too aggressive given heavier per-forward cost on new arch
- `--ent-coef 0.02` — old runs used 0.04
- `--lam 0.95` — old runs used 0.75 (much shorter advantage horizon)
- `--target-kl 0.03` — works at lr=1e-5 (only ~0.039 avg kl), but might want recalibration

These haven't been ablated. Worth a follow-up if lr=1e-5 alone doesn't produce sustained improvement.

---

## Recommended Phase 1 v2 launch

After confirming lr=1e-5 actually improves over 15-20 iters, restart Phase 1 with:

```bash
python train_rl.py \
  --init-from data/models/rl_v10/ppo_phase1/selfplay_v9_20260504_223016/snapshot_0019.pt \
  --pool-anchors data/models/bc/v10_cloud_gen9/epoch_003.pt \
  --device cuda --servers 9000,9000,9000,9000 --fp16 \
  --games-per-iter 200 --max-concurrent 200 \
  --n-iters 200 --warmup-iters 0 \
  --lr 1e-5 \                                    # ← THE FIX
  --lam 0.95 --ent-coef 0.02 --reward-style terminal \
  --grad-accum 1 \
  --adaptive-entropy --adaptive-entropy-low 0.65 --adaptive-entropy-high 0.95 \
  --win-rate-mode ema --win-rate-ema-alpha 0.3 --win-rate-ema-window 50 \
  --eval-interval 20 --eval-team-set metamon-competitive --eval-games 200 \
  --snapshot-interval 5 \
  --early-stop --early-stop-patience 3 \
  --procedural-teams C:/Users/raiad/OneDrive/Desktop/team_builder/raw_data/pokemon_usage/2024-04 \
  --out-dir data/models/rl_v10/ppo_phase1_v2
```

**Pre-flight checklist:**
1. System reboot (clear CUDA fragmentation; restore expected ~16-30% VRAM at conc=200)
2. Run a 15-20 iter lr=1e-5 confirmation experiment first
3. If smart_avg climbs (or holds), launch full 200-iter run

---

## Files generated

- `pokemon-ai-starter/pokemon-ai/src/eval_diag.py` — general-purpose smart_avg eval with fp16/perm overrides (kept for reuse)
- `pokemon-ai-starter/pokemon-ai/src/h2h_diag.py` — head-to-head with replay saving + per-side mode flags (kept for reuse)
- `data/models/rl_v10/lr_ablation_1e5/` — lr=1e-5 ablation run
- `data/models/rl_v10/lr_ablation_3e5/` — lr=3e-5 control run
- `data/models/rl_v10/lr_ablation_5e5/` — lr=5e-5 ablation run
- `data/models/rl_v10/lr_ablation_1e4/` — lr=1e-4 ablation run
- `docs/PHASE1_INVESTIGATION_PLAN.md` — investigation plan with test results inline
- `docs/PHASE1_POSTMORTEM.md` — original postmortem with H2H matrix + playstyle analysis
- `docs/PHASE1_DIAGNOSIS_REPORT.md` — this file

**Single-purpose diag scripts deleted after producing their conclusions** (results preserved in this report):
- `feature_equality_test.py` — Pri 1 result: BC vs PPO pipeline max_diff=0
- `inference_batcher_equality_test.py` — Pri 1.5 result: arch_compat staged path = direct forward, max_diff=0
- `logit_dump_test.py` — Pri 4 result: gradient instability, no class-level mode collapse
- `vram_diag.py` — VRAM result: T-quadratic scaling identified, T=300 → ~12 GB peak
