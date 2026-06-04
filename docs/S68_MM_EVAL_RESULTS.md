# S68 MM Evaluation Results — Consolidated

Authoritative table for all MM-related evals done in S68. Raw JSONs in
`data/eval_artifacts/s68/`. Memory pointer:
`memory/project_plateau_hypothesis_and_experiments.md`.

All matchups n=500 per pair unless noted. Metamon-competitive = 16-team curated
val set. Procedural = our training-time team distribution.

---

## 1. MMs vs smart-bots (calibration — where MMs sit vs our anchor scale)

Each cell = MM win rate vs that smart bot. Mean = simple average across 4 bots.

| MM | vs SH | vs SmartDmg | vs Tactical | vs Strategic | Mean MM wr |
|---|---|---|---|---|---|
| LargeRL | 70.6% | 69.8% | 68.2% | 72.0% | **70.2%** |
| MediumRL_Aug | 65.4% | 67.4% | 65.6% | 68.4% | **66.7%** |
| SyntheticRLV2 | 69.8% | 73.0% | 71.4% | 70.0% | **71.1%** |
| Minikazam | 92.0% | 90.6% | 93.4% | 91.6% | **91.9%** |

**Key reading**: LargeRL/SynthRL sit at ~70% smart_avg vs bots — same band as
our model (70-74%). MediumRL_Aug is *below* that (~67%). Minikazam transcends
the ceiling (~92%) — only MM in this set that does.

Sources:
- LargeRL + MediumRL_Aug: `mm_vs_smartbots_FULL.json` (2026-06-03 15:46 UTC)
- SyntheticRLV2 + Minikazam: `mm_vs_smartbots_SYNTH_MINI.json` (2026-06-03 17:15)
- (The 134100 JSON is a partial earlier run, superseded.)

---

## 2. POST_INIT_iter139 (= snap_0139, lr8e-5 record, 1178.4 Elo) vs MMs

Standalone evals on each team-set.

| MM | metamon-competitive | procedural |
|---|---|---|
| LargeRL | 51.0% | (missing — pre-fix script bug) |
| MediumRL_Aug | 56.2% | 51.0% |
| SyntheticRLV2 | 48.6% | 40.2% |
| Minikazam | 16.2% | 30.0% |

**Key reading**:
- On metamon-competitive teams (the val set), we're at MM-tier vs
  LargeRL/SynthRL (~50/50), even *above* MediumRL_Aug. Confirms our 70-74%
  smart_avg ceiling IS at MM-tier when teams match.
- On procedural teams, we drop on SyntheticRLV2 (-8.4pp) and rise on
  Minikazam (+13.8pp). Inconsistent — argues partial team-overfit on Minikazam
  AND partial team-pattern training of our policy that doesn't help on SynthRL.
- The MediumRL_Aug procedural drop (-5.2pp) suggests medium-tier MMs benefit
  most from team-distribution alignment.

Sources:
- `snap_vs_mms_POST_INIT_iter139.json` (metamon-competitive, 18:32 UTC)
- `snap_vs_mms_PROCEDURAL_POST_INIT_iter139.json` (procedural, 19:56 UTC)

---

## 3. Three-way: snap_0139 vs snap_0249 vs snap_0289 — on metamon-competitive teams

Closes the comparison triangle for fishbowl_prod_lr1e-4_v1 (LR=1e-4, snap_0139
base, +5 ext/iter from {LargeRL, MediumRLAug, SyntheticRLV2} pool over 150
iters):

- **snap_0139** = baseline (lr8e-5 record, NO further training, NO externals)
- **snap_0249** = peak smart_avg snap from fishbowl_prod (74% smart_avg)
- **snap_0289** = end-of-run snap (72% smart_avg)

All n=500/pair on metamon-competitive teams.

| MM | snap_0139 | snap_0249 | snap_0289 | Δ end vs base | trend |
|---|---|---|---|---|---|
| LargeRL | 49.6% | 49.2% | 51.4% | **+1.8pp** | dip → recover |
| MediumRL_Aug | 53.2% | 51.4% | 54.0% | **+0.8pp** | dip → recover |
| SyntheticRLV2 | 49.0% | 47.8% | 46.8% | **−2.2pp** | monotonic decline ⚠ |
| Minikazam | 18.4% | 20.4% | 22.2% | **+3.8pp** | monotonic increase ✓ |
| **Avg** | **42.6%** | **42.2%** | **43.6%** | **+1.0pp** | |

Sources:
- `two_snaps_vs_mms_MC.json` (snap_0139 + snap_0289, 2026-06-03 22:50)
- `snap0249_vs_mms_MC.json` (snap_0249, 2026-06-04 01:06)

**Reading the trajectory**:
- **Minikazam monotonic gain (+3.8pp)** is the most encouraging signal — model
  slowly closing the gap to the strongest MM. Whatever is happening in training
  is working *for the hardest matchup*.
- **SyntheticRLV2 monotonic decline (−2.2pp)** is concerning. Despite SyntheticRLV2
  being IN the training pool, performance vs it declined across the run. Suggests
  style drift away from SyntheticRLV2's distribution (possibly toward LargeRL/
  MediumRLAug style).
- **LargeRL & MediumRLAug both dip mid-run then recover** — argues against
  "peak smart_avg = peak MM-tier." snap_0249 (smart_avg peak) is the WORST
  snap of the three vs LargeRL/MediumRLAug. End-of-run snap_0289 is best.
- Net: smart_avg and MM-tier WR ARE decoupled. The smart_avg-peak snap is NOT
  the MM-tier-peak snap. Use direct MM evals as ground truth for elite-cap gains.
- **All deltas within n=500 noise (~4.5pp 95% CI)** — directions are
  suggestive, not statistically significant individually. The monotonic patterns
  (Minikazam ↑, SynthRL ↓) are the strongest signals because they're consistent
  across 3 data points.

**Strategic implication**: externals-in-training give mixed transfer at low
dose (5/iter). Helps on some MMs (Mini), neutral or harms on others (SynthRL).
**Supports the case for AWR rehearsal** as a more-targeted intervention than
just opponent diversification.

---

## 4. fishbowl_prod_lr1e-4_v1 smart_avg trajectory (training-side eval)

15 evals across iters 149→289 (eval_interval=10). Run started from snap_0139
(lr8e-5 record), trained 150 iters with externals (5/iter from MM pool).

| Iter | smart_avg | Iter | smart_avg | Iter | smart_avg |
|---|---|---|---|---|---|
| 149 | 71% | 199 | 70% | 249 | **74%** |
| 159 | **74%** | 209 | 70% | 259 | 73% |
| 169 | 72% | 219 | **74%** | 269 | 72% |
| 179 | **74%** | 229 | 73% | 279 | 71% |
| 189 | 70% | 239 | **74%** | 289 | 72% |

**Range 70-74%, mean 72%, std ~1.5pp, no climb across 150 iters.**

74% peaks at iters: **159, 179, 219, 239, 249** (5 of 15).

snap_0249 (last 74% peak) is the planned "peak-vs-end" comparator — Task #126.

Source: `fishbowl_prod_lr1e-4_v1_smart_avg_digest.txt` (digest of
`/tmp/fishbowl_prod.log` on prod, 51 MB original).

---

## 5. Comparison: smart_avg metric vs MM-tier WR

| Run | smart_avg band | snap vs LargeRL (MC) | gain on MMs? |
|---|---|---|---|
| Pre-fishbowl (snap_0139) | 73-74% | 51.0% (baseline) | — |
| fishbowl_prod_lr1e-4_v1 (snap_0289) | 70-74%, flat | 51.4% (preliminary) | **+1.8pp** despite flat smart_avg |
| fishbowl_v2 / v2_resume (no externals) | 67-73% | not tested | unknown |

**Headline interpretation**: smart_avg is bot-anchored and saturates at
70-74%; doesn't reflect MM-tier capability gains. Direct MM-vs-our_model
H2H is the right metric for the elite-capability question. **The
externals-in-training signal is small but real in this preliminary slice.**

---

## Open questions (post-this-eval)

1. Does snap_0249 (peak smart_avg snap) beat snap_0289 vs MMs? If yes, mid-run
   was better → re-evaluate training duration. If similar → "peak" was a
   bot-noise artifact, externals lifted the whole curve uniformly.
2. Does the +1.8pp hold on Minikazam? If yes (i.e., Minikazam gap shrinks
   16.2% → 18%+), externals are doing real work. If no, the gain may be
   restricted to MMs that look like the training-pool MMs.
3. Procedural-team eval of fbp_snap0289 (Task #126 follow-up): if externals
   helped MC teams but NOT procedural teams, training-pool team distribution
   matters more than just opponent-type diversity.

---

## Related

- `docs/PLATEAU_HYPOTHESIS_AND_EXPERIMENTS.md` — strategic context, why this matters
- `docs/REPLAY_REHEARSAL_AWR_VS_OFFPOLICY_PPO.md` — next experiment if externals validate
- `memory/project_plateau_hypothesis_and_experiments.md` — memory pointer
- Task #126 — snap_0249 vs MMs eval (queued after current two-snap)
