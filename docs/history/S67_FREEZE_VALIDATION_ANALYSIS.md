# S67 Track B Quality Validation — A/B Analysis

**Status**: ARCHIVED — Track B (freeze-spatial PPO) was REJECTED on principle
before the A/B experiment's quality data could meaningfully change the
decision. See `feedback_dont_propose_principle_violations.md` (auto-loaded
memory) for the rejection rationale.

**Outcome**: Both A (full-finetune) and B (frozen-spatial) completed 30
iters. Smart_avg scores were within ±5pt noise at all 3 eval points (iter
10/20/30) — methodology insufficient to detect the model-ceiling concerns
the user raised (smart_avg vs Elo, early-PPO vs late-PPO regime, fixed-bot
eval vs out-of-distribution opponents). Quality decision is principle-based,
not data-based.

**Wall-time observation** (the only useful data from variant B): freeze
saved ~3-5% per iter at active=8 (small). Confirmed that freeze is NOT
useful as a single-run optimization. The freeze pattern is preserved as
a TEMPORARY SCAFFOLD lever for specific use cases (Phase 2 early-PPO
anti-erosion, multi-gen transfer, etc.) per `feedback_freeze_spatial_use_cases.md`.

**For the actual Phase 2 wall reduction**: see
`docs/S67_WORKER_SCALING_RESULTS.md`. The ship-ready answer is
`--mp-workers 30` (-34% wall at pool=15, no quality compromise).

---

**Last updated**: 2026-05-19, variant A iter ~25/30 in flight.

---

## §1. Experiment design

Validation gate for the Track B shared-backbone CIS architecture
(`docs/SHARED_BACKBONE_INVESTIGATION.md`). Tests whether freezing the
backbone (tokenizer + spatial + summary_to_temporal, 6.27M params, 31.4%
of model) during PPO hurts model quality vs current full-finetune.

| Setting | Both variants |
|---|---|
| Init | `data/models/bc/v10_padded_for_cis_dev.pt` (BC v10) |
| BC anchor | Same ckpt, coef=0.1 |
| Iters | 30 |
| Games/iter | 1600 |
| Concurrent | 200 |
| Workers | 8 |
| Pool ramp | snapshot-interval=10 → pool 1 (iters 0-9), 2 (10-19), 3 (20-29) |
| LR | 1e-5 |
| KL target | 0.03 |
| Eval | Every 10 iters, 200 games, metamon-competitive teams |
| Tier 3 | mb=64, packed, no-per-chunk-gc |
| Compile | OFF |

| Variant | Difference |
|---|---|
| **A** | Current canonical Phase 2 stack. All params trainable. |
| **B** | A + `--freeze-spatial`. Backbone frozen, only temporal + heads + switch_encoder trainable. |

Code: `perf/freeze-spatial` branch at `671e19e4`.

---

## §2. Pass/fail gate

**Primary metric**: `smart_avg` at iter 10, 20, 30 (3 eval points each).

**Noise floor**: metamon-competitive eval = ~3.6pt same-policy noise at
200×4 games (per `--eval-team-set metamon-competitive` help text).
Conservative gate uses **±5pt** to absorb noise + per-iter PPO variance.

| Outcome | Verdict | Next move |
|---|---|---|
| B ≥ A − 5pt across all 3 eval points | **PASS** | Commit to Track B impl (CIS shared-backbone refactor). Project ~3-4 sessions. |
| B < A − 5pt at any eval point | **FAIL** | Step back. Three pivot options: LoRA-PFSP, cold-freeze-then-unfreeze, accept moderate pool ceiling. |
| B > A + 5pt | **WIN** | Strong commit to Track B. Possibly also investigate whether Phase 2 itself should use freeze (S57 erosion mitigation). |
| Inconclusive (all within ±5pt but no clear trend) | **SOFT PASS** | Proceed with Track B, flag uncertainty. |

Sanity checks (don't change verdict; flag if anomalous):
- kl trajectory (both should be ≤ target 0.03)
- bc_kl (B's should be slightly LOWER than A's, since spatial is bit-identical to BC v10)
- ent (stable in adaptive band 0.65-0.95)
- pi loss / v loss (both stable, no spikes)
- W/L self-play (around 50%, can't really diverge)

---

## §3. Variant A data (baseline)

**Run dir**: `/tmp/variant_A_baseline/selfplay_v9_20260519_000624/`
**Log**: `/tmp/variant_A.log`
**Launched**: 00:06:25 UTC

### §3.1 Iter trajectory (filled live)

| Iter | Pool | collect | update | W/L | kl | bc_kl | ent |
|---|---|---|---|---|---|---|---|
| 0  | 1 | 565s | 154s | 48.0% | 0.0118 | 0.0119 | 1.178 |
| 1  | 1 | 601s | 163s | 50.6% | 0.0124 | 0.0154 | 1.183 |
| 2  | 1 | 590s | 155s | 52.5% | 0.0128 | 0.0231 | 1.174 |
| 3  | 1 | 595s | 161s | 52.8% | 0.0128 | 0.0328 | 1.174 |
| 4  | 1 | 574s | 150s | 55.2% | 0.0134 | 0.0459 | 1.179 |
| 5  | 1 | 582s | 156s | 52.0% | 0.0139 | 0.0596 | 1.174 |
| 6  | 1 | 577s | 155s | 53.7% | 0.0138 | 0.0739 | 1.176 |
| 7  | 1 | 582s | 155s | 52.7% | 0.0139 | 0.0829 | 1.175 |
| 8  | 1 | 573s | 156s | 52.4% | 0.0137 | 0.0839 | 1.169 |
| 9  | 1 | 588s | 158s | 54.1% | 0.0124 | 0.0825 | 1.170 |
| 10 | 2 | 782s | 170s | 50.1% | 0.0119 | 0.0858 | 1.175 |
| 11 | 2 | 786s | 169s | 51.6% | 0.0119 | 0.0900 | 1.181 |
| 12 | 2 | 777s | 165s | 52.6% | 0.0112 | 0.0961 | 1.182 |
| 13 | 2 | 770s | 162s | 52.3% | 0.0114 | 0.1018 | 1.180 |
| 14 | 2 | 765s | 159s | 49.9% | 0.0108 | 0.1063 | 1.176 |
| 15 | 2 | 766s | 167s | 51.1% | 0.0111 | 0.1105 | 1.173 |
| 16 | 2 | 777s | 161s | 52.8% | 0.0104 | 0.1149 | 1.165 |
| 17 | 2 | 779s | 163s | 50.3% | 0.0108 | 0.1196 | 1.163 |
| 18 | 2 | 784s | 165s | 51.2% | 0.0111 | 0.1215 | 1.159 |
| 19 | 2 | 781s | 166s | 51.0% | 0.0117 | 0.1259 | 1.161 |
| 20 | 3 | 842s | 157s | 50.8% | 0.0114 | 0.1223 | 1.157 |
| 21 | 3 | 944s | 154s | 50.4% | 0.0112 | 0.1242 | 1.142 |
| 22 | 3 | 912s | 158s | 50.7% | 0.0110 | 0.1239 | 1.145 |
| 23 | 3 | 918s | 156s | 52.4% | 0.0114 | 0.1258 | 1.142 |
| 24 | 3 | 945s | 157s | 52.7% | 0.0111 | 0.1257 | 1.138 |
| 25 | 3 | 925s | 157s | 51.3% | 0.0119 | 0.1287 | 1.124 |
| 26 | 3 | ... | ... | ... | ... | ... | ... |
| 27 | 3 | ... | ... | ... | ... | ... | ... |
| 28 | 3 | ... | ... | ... | ... | ... | ... |
| 29 | 3 | ... | ... | ... | ... | ... | ... |

### §3.2 Eval points (filled live)

| Eval after iter | SH | SmartDmg | Tactical | Strategic | **smart_avg** |
|---|---|---|---|---|---|
| 10 | 66% | 66% | 70% | 72% | **69%** |
| 20 | 74% | 68% | 66% | 72% | **70%** |
| 30 | ... | ... | ... | ... | **...** |

---

## §4. Variant B data (frozen-spatial)

**Run dir**: `/tmp/variant_B_freeze/selfplay_v9_<timestamp>/`
**Log**: `/tmp/variant_B.log`
**Launched**: TBD (after A completes)

### §4.1 Iter trajectory (to fill)

| Iter | Pool | collect | update | W/L | kl | bc_kl | ent |
|---|---|---|---|---|---|---|---|
| 0  | 1 | ... | ... | ... | ... | ... | ... |
| ... | | | | | | | |
| 29 | 3 | ... | ... | ... | ... | ... | ... |

### §4.2 Eval points (to fill)

| Eval after iter | SH | SmartDmg | Tactical | Strategic | **smart_avg** |
|---|---|---|---|---|---|
| 10 | ... | ... | ... | ... | **...** |
| 20 | ... | ... | ... | ... | **...** |
| 30 | ... | ... | ... | ... | **...** |

### §4.3 Freeze verification (sanity check)

Verified at smoke (S67, 2026-05-18):
- `[freeze-spatial] frozen params: 6,274,040 (31.4%)`
- `[freeze-spatial] trainable params: 13,723,956 (68.6%)`
- Inline backbone-vs-init compare: 191/191 backbone params bit-equal, 61/62 specialized changed.

Will re-verify at variant B iter 1 snapshot.

---

## §5. Side-by-side comparison (to fill)

### §5.1 Eval at matched iters (THE PRIMARY GATE)

| After iter | Variant A smart_avg | Variant B smart_avg | Δ (B−A) | Within ±5pt? |
|---|---|---|---|---|
| 10 | 69% | ... | ... pp | ... |
| 20 | 70% | ... | ... pp | ... |
| 30 | ... | ... | ... pp | ... |

### §5.2 Trajectory comparison

KL trajectories (sample iter 5, 15, 25):
- A iter 5  kl=0.0139, iter 15 kl=0.0111, iter 25 kl=0.0119
- B iter 5  kl=..., iter 15 kl=..., iter 25 kl=...

BC KL trajectories:
- A iter 5  bc_kl=0.0596, iter 15 bc_kl=0.1105, iter 25 bc_kl=0.1287
- B iter 5  bc_kl=..., iter 15 bc_kl=..., iter 25 bc_kl=...

**Expected**: B's bc_kl should be SLIGHTLY LOWER (spatial bit-identical to BC v10 → bc_kl reflects only temporal+heads drift, not spatial drift). If B's bc_kl is dramatically lower, it confirms freeze is "working" as intended (less drift). If higher, something's off.

Wall time per iter:
- A average: ... s/iter (pool=1: ~585s, pool=2: ~775s, pool=3: ~928s)
- B average: ... s/iter
- Difference: ... (not the point of this experiment, but logged)

---

## §6. Verdict (TO FILL)

**Gate result**: ___ (PASS / FAIL / WIN / SOFT PASS)

**Reasoning**: ...

**Recommendation**: ...

---

## §7. Next-move tree (filled based on §6)

### If PASS or WIN
1. Commit to Track B impl. Create `perf/cis-shared-backbone` branch off master.
2. Design memo update: `docs/SHARED_BACKBONE_INVESTIGATION.md` gets a §8 IMPLEMENTATION section.
3. Refactor `mp_centralized_collect.py`:
   - Backbone forward runs ONCE per fire on combined-batch from all slots
   - Specialized routes per-snapshot
   - Snapshot file format changes: specialized-only state_dict
   - Per-slot reload becomes "load specialized only"
4. Quality gate at every milestone (smoke + prod).
5. Expected wall savings at pool=4-8: **-25 to -35%** (per S67 measured data, NOT the original §1 SHARED_BACKBONE memo's optimistic -54%).
6. Estimated cost: ~$30-50 + 3-4 sessions.

### If FAIL
1. **Don't immediately abandon Track B.** Three pivots to consider:
   a. **LoRA-PFSP**: Add small per-snapshot LoRA delta to the frozen backbone. Each snapshot still has unique behavior; LoRA delta is the per-snapshot specialized weight. Less invasive than full freeze, smaller quality risk.
   b. **Cold-freeze-then-unfreeze**: Phase 1 with freeze (establish stable backbone), then unfreeze for Phase 2 fine-tune. Phase 1 from `--init-from BC v10` is fast; Phase 2 starts from Phase 1 final.
   c. **Accept pool ceiling**: Phase 2 runs at pool=2-4 with current canonical stack. No Track B. Move on to Phase 2 itself, defer pool experimentation.
2. Update `docs/REFUTED_LOG.md` with "freeze-spatial full PPO refuted" entry + evidence.
3. Surface options to user with cost/risk for each.

### If SOFT PASS (inconclusive but no harm)
1. Proceed with Track B impl. Flag uncertainty.
2. After Track B integration, plan a longer (60-100 iter) head-to-head A/B at production scale to confirm the gate at higher iter count.

---

## §8. Notes / observations during the run

### A side
- Iter 0-9 (pool=1) clean. Average ~585s. KL hovering 0.012-0.014, well under target.
- Iter 10 transition to pool=2 added ~30% to collect (782s vs 585s) — matches S65/S66 documented pool slowdown.
- Iter 20 transition to pool=3 added another step — collect=842s, growing to 925-945s by iter 25. Within S65 expectations.
- Eval iter 10: smart_avg=69%, iter 20: smart_avg=70%. Essentially FLAT — model not gaining vs the fixed eval set in first 20 iters of PPO. Expected behavior with BC anchor (kl/bc_kl growing slowly, model conservatively).
- W/L hovering 50-55% in self-play (normal noise).
- No NaN, no FATAL, no anomalies.

### B side
(to fill)

---

## §9. References

- `memory/project_s66_collect_arch_findings.md` — S66 investigation that motivated Track B
- `memory/project_phase1_v3_diagnosis.md` — S57 type-knowledge erosion (load-bearing for freeze quality hypothesis)
- `memory/project_bc_anchor_design.md` — BC anchor mechanism (relevant since both variants use it)
- `docs/SHARED_BACKBONE_INVESTIGATION.md` — Track B architecture target
- `docs/WAVE_BASED_CIS_INVESTIGATION.md` — Track A (alternative, deprioritized)
- `docs/REFUTED_LOG.md` — will be updated regardless of outcome
- `memory/feedback_battle_server_restart_after_kill.md` — S67 ops lesson learned during this experiment

---

End of template. Will be finalized when both variants complete.
