# Phase 1 Postmortem — PPO from BC v10 e3, Session 50

**Run:** `data/models/rl_v10/ppo_phase1/selfplay_v9_20260505_060410/`
**Status:** Stopped at iter 79/200 (manually killed due to confirmed regression)
**Outcome:** Negative result — model regressed -36pt smart_avg, -25pt H2H vs BC
**Best checkpoint:** `snapshot_0019.pt` (BC + value-head warmup, no policy training)

---

## TL;DR

Pure self-play PPO from BC v10 e3 on the new transformer arch + procedural Smogon teams **monotonically degraded the model** across 79 iters. Internal self-play wr climbed (35% → 50%) while every external metric collapsed. The degradation has three compounding causes, ranked by contribution:

1. **Real skill loss (~20-25pt of the smart_avg drop):** specific behaviors got worse — model attacks into resistances 50% more, switches voluntarily less than half as often, KO ratio inverts from 1.65 to 0.60.
2. **Team distribution shift (~15-20pt):** PPO trained on procedural Smogon-stat teams; eval is on metamon-competitive curated teams. On procedural, sp79 only loses 60/40; on competitive it loses 80/20.
3. **Perm/canonical feature mismatch (~7pt):** training=True at training, training=False at eval. Real but smaller than initially measured because BC's perm performance was already poor (sp19: 69% canon vs 30% perm).

---

## 1. Eval trajectory

Smart_avg over the run, 4 eval points × 800 games each:

| iter | SH | SmartDmg | Tactical | Strategic | smart_avg | Δ vs BC |
|---|---|---|---|---|---|---|
| 0 (=BC) | 66 | 70 | 66 | 66 | **67** | — |
| 19 | 65 | 67 | 56 | 69 | 64 | -3 (noise) |
| 39 | 51 | 59 | 52 | 55 | 54 | -13 |
| 59 | 50 | 46 | 47 | 48 | 48 | -19 |
| 79 | 31 | 32 | 36 | 40 | **35** | **-32** |

Slope was accelerating: -10pt → -6pt → -13pt per 20-iter window. Early-stop wouldn't have fired until iter 99 (requires 5 evals minimum).

---

## 2. Internal self-play vs external eval — the divergence

Concrete contradiction the user flagged:
- At iter 79, **internal self-play wr** vs sp_0019 ≈ ~50% (fair fight in self-play).
- At iter 79, **head-to-head vs sp_0019** at canonical/competitive eval ≈ 25% (sp19 wins 75/25).

Same models, same matchup, different conclusion. Resolution: self-play used **procedural teams + sp79 in perm features** (its training distribution); eval used **competitive teams + canonical features** (very different).

---

## 3. Head-to-head matrix (200 games each)

| # | p1 (mode) | p2 (mode) | Teams | p1 wr | sp79 wr |
|---|---|---|---|---|---|
| Original 1 | epoch3 (canon) | sp79 (canon) | mm-competitive | **78.5%** | 21.5% |
| Original 2 | sp19 (canon) | sp79 (canon) | mm-competitive | **75.5%** | 24.5% |
| A | epoch3 (canon) | sp79 (**perm**) | mm-competitive | **79.5%** | 20.5% |
| B | sp19 (canon) | sp79 (**perm**) | mm-competitive | **75.0%** | 25.0% |
| C | epoch3 (canon) | sp79 (canon) | **procedural** | **58.5%** | 41.5% |
| D | sp19 (canon) | sp79 (canon) | **procedural** | **59.5%** | 40.5% |
| E | epoch3 (canon) | sp79 (**perm**) | **procedural** | **61.5%** | 38.5% |
| F (regression knee) | sp_0049 (canon) | sp79 (canon) | mm-competitive | **65.5%** | 34.5% |

### Observations
- **Perm mode for sp79 doesn't help in H2H** (Original vs A: 78.5 → 79.5; Original 2 vs B: 75.5 → 75.0). The 7pt smart_avg recovery in eval-mode-perm doesn't translate to direct play.
- **Procedural teams shift everything ~17-20pt in sp79's favor** (78.5 → 58.5 for epoch3; 75.5 → 59.5 for sp19). Team distribution is the biggest single axis.
- **Even peak-vs-peak (Test E)** — sp79 in perm + procedural — STILL loses to BC by 23pt. sp79 has **no regime where it matches BC**.
- **Regression was monotonic** (Test F): sp_0049 beats sp_0079 65.5%. No "best policy at iter X" intermediate peak to rescue.

---

## 4. Diagnostic eval matrix (sp_0079, single-side smart_avg)

100 games × 4 bots, on metamon-competitive teams:

| Eval mode | smart_avg | Δ vs control |
|---|---|---|
| Control (fp32, canonical) | **31.0%** | — |
| training=True (perm features) | 38.0% | +7pt |
| fp16 inference | 36.2% | +5pt |
| Combined fp16 + perm | 36.8% | +6pt (NOT additive) |

For comparison, **sp_0019 (BC + warmup)**:
- Canonical: **69.2%**
- Perm: **29.8%** (-39pt!)

**Key finding from the BC perm baseline:** the model was ALWAYS asymmetric between perm and canonical. BC's "good" eval was riding on canonical-specific representations that augmentation didn't fully erase.

---

## 5. Playstyle analysis — concrete behavioral changes (sp79 vs sp19)

From `analyze_eval.py` on H2H replays (200 battles each, n_moves ~6800):

### Decision quality (the smoking gun)

| Metric | sp19 | sp79 | Δ |
|---|---|---|---|
| Super-effective hits (of flagged) | **54%** | **31%** | **-23pt** |
| Resisted hits | 30% | **51%** | +21pt |
| Immune hits | 14% | 17% | +4pt |

**sp79 attacks into resistance/immunity 68.5% of the time vs sp19's 44%.** The model literally picks worse moves — type-effectiveness understanding has been damaged.

### Combat outcomes

| Metric | sp19 | sp79 |
|---|---|---|
| KO ratio | **1.65** | **0.60** |
| Our faints/game | 3.3 | 5.5 |
| Opp faints/game | 5.5 | 3.3 |

sp79 trades unfavorably almost 2:1.

### Move usage shifts

| Move | sp19 % | sp79 % | Note |
|---|---|---|---|
| Stealth Rock | 4.6% | **12.1%** | spammed at 2.5x rate |
| Protect | 2.5% | **6.3%** | passive stalling |
| Earthquake | 5.7% | 3.4% | high-quality move dropped |
| Recovery (any) | 9.2% | 6.0% | less self-preservation |
| Status moves | 3.0% | 2.3% | slightly less utility |

**sp79 leans heavily on Stealth Rock and Protect — passive moves that don't deal damage.** It has converged to a "set up hazards and stall" template that might be okay vs procedural-team novices but fails against competitive teams that have hazard removal.

### Switching behavior

| Metric | sp19 | sp79 |
|---|---|---|
| Voluntary switch rate | **13.0%** | **5.4%** |
| Forced (post-faint) | 7.1% | 11.4% |
| Total switch rate | 20.1% | 16.8% |

**sp79 voluntarily switches less than HALF as often as sp19.** It stays in to attack (badly) where sp19 would pivot. The forced-switch rate going UP is the consequence: getting KO'd → forced switch.

### HP management

| Metric | sp19 | sp79 |
|---|---|---|
| Setup at <25% HP | 14% | **28%** |
| Recovery at >90% HP | 9.6% | 5.3% |

sp79 sets up while almost dead twice as often (reckless), but recovers wastefully less often.

---

## 6. L2 weight delta per submodule (iter 79 vs warmup baseline sp_0019)

| Submodule | params (% of total) | L2 delta % |
|---|---|---|
| tokenizer | 5.0% | 0.10% (effectively frozen — embeddings preserved) |
| spatial | 23.7% | 10.0% |
| summary_to_temporal | 2.6% | 16.5% |
| temporal | **63.6%** | **14.4%** |
| switch_encoder | 0.3% | 7.0% |
| action_head | 2.6% | **20.5%** |
| value_head | 2.1% | 20.4% |

**Verdict:** all submodules training (no gradient routing bug). Temporal stack got rewritten 14% — that's the multi-turn reasoning being substantially modified. Action head moved most in % terms.

---

## 7. Root-cause synthesis

The full explanation requires THREE compounding effects (in order of contribution):

### 7.1 Real skill loss in damage-trade fundamentals (~20-25pt smart_avg)

The most striking finding. PPO trained the model to:
- Select moves with much lower super-effective rate (54% → 31%)
- Switch voluntarily much less (13% → 5%)
- Spam Stealth Rock and Protect at high rates (~3x and ~2.5x)
- Set up at low HP twice as often

These are **objectively worse decisions** regardless of distribution. The model is making textbook-bad Pokemon play more often.

**Why?** Self-play with procedural-Smogon-stat teams creates a degenerate training signal:
- Both sides have random, uncoordinated teams
- Type effectiveness matters less when both sides have a random mix
- "Always-okay" defaults like Stealth Rock and Protect become safer than committed type-aware attacks
- Voluntary switches don't help vs random opponents that aren't punishing matchup
- KL early-stop (kl=0.045+ every iter for 79 iters) compounded into a drift toward this safe-defaults policy

The model collapsed to a **"play it safe, set hazards, hope" policy** that passes the procedural self-play training distribution but fails competitive eval.

### 7.2 Team distribution shift (~15-20pt smart_avg)

Quantified directly by Test C/D: epoch3 vs sp79 wr drops from 78.5% (mm-competitive) to 58.5% (procedural). That's a 20pt swing entirely from team distribution.

Procedural teams = random Smogon usage stats samples. No coherent cores, no archetypes, no "stall vs offense" style matchups. Just random combos. sp79 is OK against this.

Metamon-competitive teams = 16 hand-curated competitive teams with structured game plans. The "spam SR + Protect" policy fails because competitive teams have Defog/Rapid Spin and aren't punished by passive play.

### 7.3 Perm/canonical feature mismatch (~7pt smart_avg, ~0pt H2H)

The smallest of the three. V9RLPlayer trains with permuted features; eval uses canonical. PPO trained the model toward permutation-mode operation, which doesn't translate cleanly to canonical eval.

But the H2H matrix (Tests A, B, E) shows that **forcing perm at eval barely changes head-to-head outcomes** (78.5 → 79.5). Within noise. So this is a real but minor effect.

---

## 8. Why early signals didn't catch this

Internal training looked healthy throughout:
- pi_loss: trending negative (good)
- v_loss: dropping 2.88 → 2.46 (good)
- entropy: 1.20 → 0.83 (sharpening, expected)
- kl: bounded 0.045-0.053 (PPO correctly limiting)
- Self-play wr: 35% → 50% (climbing within distribution)

**All the green signals were measured on the training distribution (procedural + perm + self-play). The training distribution diverged from the eval distribution, so progress on training didn't transfer.**

`smart_avg` eval at iter 19 was -3pt (looked like noise). Iter 39 was -13pt — should have been a stop signal. Iter 59 was -19pt — definitively a stop signal. The early-stop mechanism's `--early-stop-min-evals 5` requirement meant it wouldn't fire until iter 99.

**Lesson:** lower the min-evals threshold, OR add an *absolute* threshold (e.g., "if smart_avg drops 10pt below init, stop immediately") in addition to the relative-to-best logic.

---

## 9. Recommendations for Phase 2

### Must-fix before any retry
1. **Procedural teams in training is the wrong baseline.** Either:
   - Mix in metamon-competitive teams (~30-50%) during training, OR
   - Switch to a curated team set entirely, OR
   - Add a "team augmentation" step that perturbs competitive teams instead of generating random ones
2. **Make V9RLPlayer use `training=False`** — match eval distribution. Removes the perm/canonical asymmetry. ~7pt of recoverable headroom.

### Strongly recommended
3. **Add at least one external opponent** to break the self-play attractor. `mcts-fast` is in-process, no extra subprocess infrastructure needed. PFSP will sample it when we lose, naturally pulling us toward strategies that beat it (which won't be "spam SR + Protect").
4. **Tighten early-stop:** add absolute threshold (smart_avg < init_baseline - 10pt → stop), reduce `--early-stop-min-evals` from 5 to 3.

### Nice-to-have
5. **Anchor weight floor for BC** — raise BC anchor PFSP weight to a minimum 15-20% so it doesn't decay to <1% and stop providing canonical-aware play in the pool.
6. **Improve replay save** — `poke-env` save_replays cuts off before the `|win|` frame. Causes `analyze_eval` to report 0W/200L. Worth fixing for future analysis.

---

## 10. Salvage: best Phase 1 deliverable

**`snapshot_0019.pt` is THE Phase 1 result.** It's at smart_avg ≈ 67% (BC parity). The 19 warmup iters trained only the value head — it's effectively BC v10 e3 with a calibrated value estimate. Path: `data/models/rl_v10/ppo_phase1/selfplay_v9_20260504_223016/snapshot_0019.pt`.

This is the checkpoint to **init from for Phase 2**.

---

## 11. Checkpoints for the project registry

| label | path | smart_avg | notes |
|---|---|---|---|
| BC v10 e3 (epoch_003) | `data/models/bc/v10_cloud_gen9/epoch_003.pt` | 67% | original BC, untouched by PPO |
| sp_0019 (warmup-end) | `data/models/rl_v10/ppo_phase1/selfplay_v9_20260504_223016/snapshot_0019.pt` | 69% | best Phase 1 result, value-only training |
| sp_0049 (mid-PPO) | `data/models/rl_v10/ppo_phase1/selfplay_v9_20260505_060410/snapshot_0049.pt` | ~52% | mid-regression, kept for trajectory analysis |
| sp_0079 (failed end) | `data/models/rl_v10/ppo_phase1/selfplay_v9_20260505_060410/snapshot_0079.pt` | 35% | degraded model — useful for diagnosis only |

---

## 12. Files generated by this postmortem

- `pokemon-ai-starter/pokemon-ai/src/eval_diag.py` — diagnostic eval with fp16/perm overrides
- `pokemon-ai-starter/pokemon-ai/src/h2h_diag.py` — h2h with per-side fp16/perm + team-set
- `data/replays/h2h_diag/` — ~3200 saved replays from 8 H2H runs across the matrix
- `docs/PHASE1_POSTMORTEM.md` — this file
