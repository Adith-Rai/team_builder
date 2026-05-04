# PPO_PHASED_TRAINING.md — Curriculum plan for v10 PPO

**Status as of Session 48 (2026-05-03)**: BC training complete on cloud
(`bc_v10_cloud_e1` already +17.7pt smart_avg over legacy v8 BC, beats every
legacy PPO peak in head-to-head). PPO is the next deliverable. This doc is
the agreed-upon training plan and the success/failure thresholds at each
phase.

> **A note for future sessions reading this fresh:** This plan is a
> well-considered consensus from Session 48 — but it's not sacred. If
> you're approaching this from a more distant vantage point and you see a
> better idea, or evidence that one of the assumptions below is wrong, the
> user is open to discussion. Surface your reasoning explicitly (don't just
> deviate quietly), point at the assumption you're challenging, and propose
> the alternative concretely. The phased structure here is a tool for
> extracting clean signal at each stage; if a different structure extracts
> cleaner signal, swap it in.

---

## The motivating problem

Our previous PPO runs (S39 `selfplay_v9_20260425_062416`, S43-44 curated
pool `selfplay_v9_20260501_011537`) used **mixed-pool training from the
start**: self-play + mcts + foul_play subprocesses + metamon subprocesses,
all in the pool simultaneously, weighted by PFSP. The runs converged to
~64-66% smart_avg (sp_0119, sp_0229), which we initially read as "the
architectural ceiling." We had no way to attribute the plateau to:

- Self-play itself being exhausted at this architecture
- External opponents being too strong / dragging convergence
- Pool composition / PFSP weighting being wrong
- Architecture capacity bottleneck
- LR/HP tuning being suboptimal

**Curriculum / phased training fixes this attribution problem.** Each
phase narrows the opponent distribution. Saturation at any phase is a
specific signal pointing at a specific intervention.

---

## The four phases

Each phase trains until the **`--early-stop` patience** trigger fires
(currently `--early-stop-patience 3` in our infra → 3 consecutive eval
intervals without smart_avg improvement). Each phase ends with a "best
checkpoint" that seeds the next phase.

### Phase 1 — Self-play + PFSP only (no external opponents)

**Pool composition:**
```yaml
opponents:
  - {name: sp, adapter: self_play, weight: 1.0}
```

**Init from**: `bc_v10_cloud_e1` (or whatever the BC convergence
checkpoint ends up as).

**Why this phase first:**
- BC has already taught the model strong fundamentals. Self-play refines
  through PFSP weighting on past versions.
- No subprocess bottleneck — `--max-concurrent 100-200` works freely.
- Local-friendly: doesn't need cloud's CPU advantage. Saves $20-40.
- Cleanest architectural signal: if the model has more to learn from
  refining its own play, this phase reveals it.

**Compute estimate**: 100-200 iters at ~3-5 min/iter local
(RTX 3060 6 GB, B=4-8 batches per PPO update with no external
subprocess overhead) → **5-15 hr local, free**.

**Success criterion**: smart_avg ≥3pt above BC baseline (so v10_cloud_e1's
~63.8% would mean Phase 1 hits ~67% sustained). If the H2H gauntlet vs
phase-1-best shows it's stronger than the BC, that's the same signal.

**Failure modes / triggers for architecture intervention:**

| Symptom | Likely cause | Intervention |
|---------|--------------|--------------|
| smart_avg never > BC by 1pt | Model has converged and self-play can't find more | **Don't go to Phase 2 yet — need a bigger model OR more BC data first.** Consider d_model 256→384, or pivot to multi-gen BC. |
| H2H WR vs `bc_v10_cloud_e1` plateaus < 53% | Same — self-play exhausted | Same as above |
| Loss/entropy/value diverging in <20 iters | LR or hyperparam issue | `--lr-restart` with `--lr 1e-5` or smaller. Check entropy floor. |
| Big gap between train smart_avg and held-out (Metamon competitive) eval | Overfitting to pool | Add `--pool-anchors bc_v10_cloud_e1`, push pool diversity sooner — go to Phase 2 |

### Phase 2 — Add original-tier external opponents

**Pool composition** (matches the curated_pool runs we already validated):
```yaml
opponents:
  - {name: sp, adapter: self_play, weight: 0.50}
  - {name: mcts-fast, adapter: mcts, weight: 0.20, params: {time_ms: 100}}
  - {name: foulplay-100ms, adapter: foulplay, weight: 0.15, params: {time_ms: 100}}
  - {name: metamon-mini, adapter: metamon, weight: 0.15, params: {model: Minikazam}}
```

**Init from**: Phase-1-best.

**Why this set first**: these are the opponents we already validated
infra-side (Sessions 42-44 protocol fixes, 5 defense layers). Same as
the recent curated runs that hit 66.15% smart_avg. Now we're starting
from a stronger BC + a phase-1-strengthened model, so the ceiling
should be higher.

**Compute estimate**: ~22 min/iter at C=6 (current production runbook),
~5-8 min/iter on cloud with Phase 1 of CLOUD_RUNBOOK §11 (more subprocess
instances). 100-200 iters → **8-30 hr** depending on platform.

**Success criterion**: smart_avg ≥3pt above Phase-1-best AND H2H
WR vs Phase-1-best ≥53%.

**Failure modes**:

| Symptom | Cause | Intervention |
|---------|-------|--------------|
| Mixed-pool wins regress vs Phase-1 baseline | Pool diversity overwhelming model | Drop external weights (sp 0.7, mcts 0.15, others 0.075 each) |
| Specific bot dominating (e.g., FP 60%) | One opponent style is unfamiliar | Boost that opponent's weight temporarily; over-train to that style |
| Forfeit/crash/stall warnings spike | Layer 1-5 defenses (S43-44) catching infra issues | Same as production: investigate per `EXTERNAL_OPPONENTS_PHASE2.md` |

### Phase 3 — Stronger Metamon variants

**Pool composition** adds the harder Metamon models:
```yaml
opponents:
  - {name: sp, adapter: self_play, weight: 0.30}
  - {name: mcts-fast, adapter: mcts, weight: 0.10, params: {time_ms: 100}}
  - {name: mcts-slow, adapter: mcts, weight: 0.05, params: {time_ms: 300}}
  - {name: foulplay-100ms, adapter: foulplay, weight: 0.10, params: {time_ms: 100}}
  - {name: foulplay-300ms, adapter: foulplay, weight: 0.05, params: {time_ms: 300}}
  - {name: metamon-mini, adapter: metamon, weight: 0.10, params: {model: Minikazam}}
  - {name: metamon-small, adapter: metamon, weight: 0.15, params: {model: SmallRL}}
  - {name: metamon-medium, adapter: metamon, weight: 0.15, params: {model: MediumRL}}
```

(In Pokemon's "Abra → Kadabra → Alakazam" naming convention, this is the
Kadabra tier — strong but not the bosses yet.)

**Init from**: Phase-2-best.

**Why this set**: SmallRL/MediumRL are 15M / 50M Metamon-trained models,
genuinely stronger than Minikazam (4.7M). Tests whether our model can
generalize against opponents trained on a different (multi-gen,
larger) data distribution.

**Compute estimate**: 100-200 iters cloud, **15-40 hr, $25-60**.
Strongly cloud-only — multiple Metamon model variants need significant GPU
+ host RAM for inference.

**Success criterion**: smart_avg ≥2pt above Phase-2-best AND positive H2H
record vs Metamon-SmallRL specifically.

**Failure modes**:

| Symptom | Cause | Intervention |
|---------|-------|--------------|
| Lose to Metamon-SmallRL despite winning vs SH/Tactical | Capacity bottleneck — our model can't out-think a 15M opponent at this width | **First architecture intervention trigger.** Bump d_model 256→384 (~+7M params), retrain BC + Phase 1 + Phase 2 + Phase 3 from scratch. |
| Win vs Mini, draw vs SmallRL, lose vs Medium | Cleanly graded — model has more to learn here, not a ceiling | Just train longer (more iters, different LR schedule) |
| Catastrophic forgetting (lose to Phase-2-best opponents) | Pool drift | Stronger `--pool-anchors` (include phase-1-best + phase-2-best + bc_v10_cloud_e1) |

### Phase 4 — Final boss tier (Kakuna et al.)

**Pool composition** — the "league final":
```yaml
opponents:
  - {name: sp, adapter: self_play, weight: 0.25}
  - {name: mcts-300ms, adapter: mcts, weight: 0.10, params: {time_ms: 300}}
  - {name: foulplay-300ms, adapter: foulplay, weight: 0.10, params: {time_ms: 300}}
  - {name: foulplay-1s, adapter: foulplay, weight: 0.05, params: {time_ms: 1000}}
  - {name: metamon-small, adapter: metamon, weight: 0.10, params: {model: SmallRL}}
  - {name: metamon-medium, adapter: metamon, weight: 0.10, params: {model: MediumRL}}
  - {name: metamon-kakuna, adapter: metamon, weight: 0.20, params: {model: Kakuna}}
  - {name: legacy-sp_0229, adapter: legacy_self_play, weight: 0.10, params: {checkpoint: data/models/rl_v9/_init_sp_0229/snapshot_0229.pt}}
```

**Init from**: Phase-3-best.

**Why include Kakuna here**: 142M params, the strongest published Metamon
model. PokeAgent ladder shows Kakuna at ~409 SR points above us
(MEMORY.md Session 35 audit). This phase tests our model against the
hardest reasonable opponent we can run.

**Compute estimate**: 200-500 iters cloud, **24-72 hr, $40-110**.
Kakuna inference alone needs serious GPU memory (10+ GB) — definitely cloud.

**Success criterion (V1)**: smart_avg ≥75% sustained over 3+ evals AND
H2H record vs Metamon-Kakuna ≥45% (we don't have to beat Kakuna; closing
the gap is the win).

**Failure modes**:

| Symptom | Cause | Intervention (V2 territory) |
|---------|-------|------------------------------|
| Plateau < 70% smart_avg after 200 iters | Capacity / data ceiling | (V2) Multi-gen BC scaling, bigger model |
| Kakuna H2H < 30% | Architecture insufficient for Kakuna's strategy depth | (V2) test-time adaptation OR move to bigger model + more BC |

---

## Phase boundary protocol

At the end of each phase:

1. **Run eval**: `eval_metamon_competitive.py` 500 games × 4 bots on the phase-best ckpt.
2. **H2H gauntlet vs prior phases**: run `eval_h2h_gauntlet.py` with the new phase-best as champion vs all prior phase-bests + key legacy peaks (sp_0229, iter_0119, sp_2979) + bc_v10_cloud_e1.
3. **Append result to model registry** (`docs/MODEL_REGISTRY.md`).
4. **Update Elo ladder** via `eval_elo_ladder.py --add-to <existing-ladder>.json --snapshots <phase-best.pt> --names ppo_v10_phase<N>_best`.
5. **Decide**: success criterion met → next phase. Failure → trigger one of the architectural interventions above before retrying.

---

## What this plan deliberately doesn't do

- **No "all opponents from start."** Sacrifices speed-to-final-ckpt for clean attribution.
- **No async actor-learner** (AMAGO-style). That's V2 architecture work; defer.
- **No `inference_batcher.py` rewrite.** Phase 1 happens entirely on local where current pipeline works fine.
- **No multi-gen during PPO.** Phase 1-4 are gen-9-only; multi-gen is post-V1.

---

## When the cloud is needed

| Phase | Cloud needed | Why |
|-------|--------------|-----|
| 1 | No | self-play only, no subprocess bottleneck, runs on local 6 GB GPU |
| 2 | Optional (cheaper if cloud) | local C=6 cap is fine; cloud gives ~3× faster iters |
| 3 | Yes | many concurrent FP/MM subprocesses needs cloud cores |
| 4 | Definitely | Kakuna inference + many subprocesses doesn't fit on 6 GB GPU |

Per-phase cost ceiling (worst case): Phase 1 ~$0, Phase 2 ~$25, Phase 3 ~$60, Phase 4 ~$110. Total V1 PPO budget: **~$200**, fits the $100-150 stated budget at the lower end of phase estimates.

---

## What this is NOT a substitute for

- **The success criteria thresholds are educated guesses, not laws.** If smart_avg trajectory looks healthy but the gain at a phase boundary is +2pt instead of +3pt, use judgment — don't blindly trigger an architecture intervention.
- **The pool weights in each phase are starting points.** Real production runs will adjust based on per-opponent W/L drift (`analyze_eval.py --by-opponent`).
- **Each phase's "best ckpt" should be defined by the same metric.** Suggest: Metamon competitive smart_avg sustained over 3 consecutive eval intervals. Don't switch metrics mid-run.

---

## Triggers that SHOULD make a future session pause and discuss

If you're a future session executing this plan and you see:

- **Phase 1 saturation at <BC baseline + 1pt** → STOP. The model can't even refine its own play meaningfully. Go talk to the user about model capacity / data scaling before any PPO work.
- **Catastrophic forgetting between phases** (Phase-2-best loses H2H to Phase-1-best) → STOP. Pool anchoring is broken.
- **Compute budget exceeding $200 across all phases** → STOP. Discuss whether to abandon V1 and pivot to V2 architecture or multi-gen.
- **Any phase taking >2× the time estimate** → STOP. Something is wrong; debug the infra before burning more compute.

The "open to better ideas" note at the top applies most strongly to these stop-and-discuss triggers. If you see one and you have a non-obvious diagnostic insight, surface it before the user has to ask.

---

## Lineage of references this plan builds on

- **AlphaStar (DeepMind)**: league-style training with explicit opponent pool tiers
- **OpenAI Five**: pure self-play to expert level (we use it for Phase 1, mix for Phase 2+)
- **AlphaGo**: BC anchor + self-play → human-level play (mirrors our v10 pipeline)
- **Metamon (AMAGO)**: actor-learner asynchrony enables their scale (V2 destination, not V1)
- **VGC-Bench BCFP**: BC + uniform-fictitious-play pool. Validates the BC-anchor approach (we extend it with curriculum).
- **REWRITE_DESIGN.md §6.3, §7 Week 4**: high-level PPO continuation plan we're filling in here.
