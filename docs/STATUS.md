# Project Status

**Last Updated:** 2026-05-14 (S64 Phase A wrap)

---

## CURRENT STATE — S64 Phase A (sequence packing arc)

**Where we are**: Phase 1 v3 DONE (diagnosed as failing via type-knowledge erosion, S57). Phase 2 DEFERRED until update-optimization sequence completes. Currently in multi-session optimization arc; S64 Phase A SHIPPED on `perf/seq-packing` branch.

**Branch state**:
- `master` at S64 Phase A wrap commit (post `1ebf45a4`, plus this STATUS refresh)
- `perf/seq-packing` at `70fd33df` — **S64 Phase A SHIPPED**, origin pushed

**Active work**: S64 Phase B NEXT (awaiting fresh session + user authorization). Sub-phases B.1-B.7 spec'd in `memory/project_s64_phase_a_results.md` §4. Estimated $3-5 pod, 1-2 sessions. Bit-equivalence gate at fp32 + bf16.

**Canonical Phase 2 launch stack** (post-S62/S63 refutations):
```
--cis --pipeline --bf16 --tier3 --tier3-minibatch-size 16 \
--bc-anchor-ckpt v10 --bc-anchor-coef 0.1 \
--cis-min-batch 32 --cis-timeout-ms 50
```
NO `--compile` (REFUTED at prod S62 per REFUTED_LOG.md #1). NO perm-at-eval (REFUTED S60 per REFUTED_LOG.md #17).

**Cumulative optimization arc cost**: ~$20.50 (S62 $15 + S63 $2 + S64 step-back ~$3 + S64 Phase 1 isolation $0.50 + S64 Phase A $0).

**Read for full context**:
- `next-prompt.txt` at project root — session-specific deep state (S64 Phase A wrap at top)
- `docs/PROFILE_BOTTLENECKS_REPORT.md` — bottleneck data + optimization arc state
- `docs/REFUTED_LOG.md` — techniques tried and refuted (don't retry these)
- `docs/SESSION_BOOT_PROTOCOL.md` — standing orders
- `memory/project_optimization_tracker.md` — optimization arc roadmap + session protocols
- `memory/project_s64_phase_a_results.md` — Phase A results + §4 detailed Phase B plan

---

## ERA INDEX — S30-S64 high-level arc

For session-by-session detail, see the historical record below this section (preserved verbatim from S33 era). For S35+ detail not in this file, see the corresponding memory files in `memory/project_session*` or session wraps in `next-prompt.txt`.

| Era | Sessions | Theme | Outcome |
|---|---|---|---|
| BC + LSTM IQL | S20-S30 | First architecture iterations; 3.85M LSTM hits 25-30% plateau vs smart bots | Plateau identified; transformer needed |
| Transformer + BC v6/v7 | S31-S35 | New 20M arch (CausalTransformerCore); BC v6 SmallRL-level | Established transformer baseline |
| RL v9 self-play | S35-S43 | PPO self-play on transformer; multiple runs to 60% smart_avg ceiling | Hit perm/canonical eval distribution mismatch |
| mp_disk + POKE_LOOP infrastructure | S43-S50 | Multi-process collection architecture; per-worker POKE_LOOP | Production-scale collection enabled |
| S50 lr regression + Phase 1 v1/v2 | S50 | `--lr 3e-5` caused catastrophic regression on transformer arch; 4-point ablation locked `--lr 1e-5` | See `docs/PHASE1_DIAGNOSIS_REPORT.md` |
| CIS (Centralized Inference Service) | S52-S54 | Pool-mirror multi-slot, async-dispatch, Phase 4.6 Option B full-reset | `memory/project_cis_4_6_design.md` |
| Tier 3 sequence-batched PPO + compile | S55-S57 | C1-C5 + train_rl wiring + BC anchor + `--tier3-minibatch-size N` | All SHIPPED; required for prod scale |
| Phase 1 v3 + collapse diagnosis | S57 | Production run to ~iter 90; diagnosed 3-stage collapse (exploration → type-knowledge erosion → strategic collapse) | BC anchor designed to prevent in Phase 2. `memory/project_phase1_v3_diagnosis.md` |
| S58 ghost-tie fixes | S58 | `asyncio.gather` of N heterogeneous opp coros was starving small batches → wait_for cancels → `battle.won=None`; replaced with one-opp-per-worker | `memory/project_s58_session_narrative.md` |
| S59 profile-driven Track B | S59 | Worker viztracer at prod: 82% poll-wait on CIS; bottleneck is Python orchestration, NOT sim/WS | See `docs/PROFILE_BOTTLENECKS_REPORT.md` |
| S60-S61 Fix #1/#2/#3 design + smoke | S60-S61 | Fix #2 (--compile) shipped capability; Fix #3 (vectorize collate) REFUTED via microbench; Fix #1 design memo | `memory/project_s61_fix1_design.md` |
| S62 prod validations + update profile | S62 | Fix #1 Option B SHIPPED at prod (-27% collect); Fix #2 REFUTED at prod (8% slower); torch.profiler reveals update is ORCHESTRATION-bound (CUDA = 8% of update wall) | `memory/project_s62_fix*.md`, `memory/project_s62_update_profile_findings.md` |
| S63 free wins | S63 | `optimizer.zero_grad(set_to_none=True)` + `.item()` audit defer SHIPPED → -4.2% update wall. Below 8-15% projection but ceiling for the free-wins category. | `memory/project_s63_free_wins_results.md` |
| S64 step-back + Phase 1 + Phase A (CURRENT) | S64 | Step-back: sequence packing prioritized over ARCH (drop temporal stack); torch 2.5.1 venv-isolation PASSED; `collate_episodes_packed` SHIPPED on `perf/seq-packing` at `70fd33df` with 11/11 equivalence tests | `memory/project_s64_*.md` (4 memos) |
| S64 Phase B NEXT | (next session) | Refactor `forward_ppo_sequence` to consume packed via `flex_attention` + per-episode causal `BlockMask` | Awaiting authorization |

---

## Historical record (preserved from S33 era, 2026-04-08)

The content below is the doc as it existed at S33 wrap. Preserved verbatim because it captures the per-session detail of the era it covers. For S35+ content, see memory files and session wraps in `next-prompt.txt`.

---

## Current Phase: Phase E — TRAINING at confirmed plateau, decision point: CLOUD. (S33 era — preserved)

### Session 33: Terminal-reward experiment ran 10 evals. Plateau confirmed. Cloud is next.

**TRAINING RUNNING** with `--pipeline --reward-style terminal` from snapshot_1519.
Currently at iter ~1720+. Pool=570. Wall ~270s/iter, healthy (kl~0.04, ent~0.89).

**Hypothesis tested in Session 33:** Dense reward (per-step KO/HP shaping) was causing
reward hacking — model optimized short-term damage at expense of type accuracy and
strategic play. Switched to terminal-only reward (ko_coef=0, hp_coef=0, terminal=1.0)
with 20-iter value-warmup (freeze policy/backbone, train value head only) starting at
iter 1520.

**Result: 10 evals, no breakout. Plateau is confirmed.**

| Iter | SH | SD | T | S | avg |
|---|---|---|---|---|---|
| 1539 | 52 | 50 | 48 | 46 | 49 |
| 1559 | 52 | 55 | 47 | 54 | 52 |
| 1579 | 62 | 50 | 46 | 60 | 55 |
| 1599 | 48 | 50 | 52 | 56 | 52 |
| 1619 | 59 | 56 | 49 | 56 | 55 |
| 1639 | 54 | 50 | 52 | 49 | 51 |
| 1659 | 62 | 52 | 52 | 50 | 54 |
| 1679 | 54 | 60 | 50 | 49 | 54 |
| 1699 | 56 | 54 | 52 | 56 | 54 |
| 1719 | 50 | 54 | 50 | 48 | 50 |

- **Mean 52.6, stdev ~2.1, range 49-55. No drift between halves.**
- All-time peak still snapshot_0699 at 57%. Plateau ceiling unchanged.
- Floor "lift" was reading a pattern into 9 samples; the 10th broke it.

**What terminal-reward DID accomplish (real but small):**
- Removed reward hacking failure mode. Behavioral metrics genuinely better.
- Recovery use 2-3x'd across all bots (was ~2%, now 4-7%).
- Voluntary switching up everywhere (10-13% → 13-17%).
- SE accuracy +5-8 vs Tactical and Strategic.
- Immune % improved on 3 of 4 bots.
- New behaviors appeared: Tactical-side experimented with 7.6% recovery.

**The smoking gun: behavioral/win-rate decoupling.**
At iter 1719, model was making *more* correct decisions than at 1699 by every observable
metric (SE up, immune down, switching up, recovery up) — and won FEWER games (54→50).
The model has learned WHAT to do but not WHEN. That's a credit assignment / sample
volume problem, not a reward shaping problem. Local optimization cannot fix this.

**Decision: cloud burst is the elimination test.**
- If cloud → 60%+ in 2 days: architecture is fine, was data-bound. Continue scaling.
- If cloud → still 50-55%: architecture has a real ceiling. Need bigger model.
- Either outcome is a definitive answer. Cost: $50-100. Cheaper than rebuilding model first.

### Session 33 POST-SCRIPT — CUDA context loss + resilience patches

**Incident (2026-04-08 ~03:28):** Training run died at iter 1728 with a cascade
of `CUDA error: unknown error` (one per PPO episode), then a final raise from
`torch.cuda.empty_cache()` at end of iter. Likely cause: Windows TDR / driver-level
context loss after ~16h of continuous pipeline + fp16 + concurrent batched inference.
Machine subsequently rebooted with a "Windows ran into an issue" screen.

**Last clean checkpoint:** `selfplay_v9_20260407_124041/snapshot_1724.pt` (Apr 8 03:17).
Pool ~571. Resume from this is safe — iter 1728 had `pi=0.0000 v=0.0000 ent=0.0000`
in the log (every PPO episode caught a CUDA exception, so backward never ran and
weights are unchanged from snapshot_1724). Snapshot 1729 was never written (1729 % 5 != 0).

**Failure-mode gap found:** `ppo_update_v8` catches per-episode exceptions
(intentional since the Session-29 1017-turn OOM disaster — DO NOT REMOVE). But when
*all* episodes raise (CUDA context death), it returned a clean dict of zeros and the
outer loop happily logged the iter and proceeded toward (a) writing a tainted snapshot
into the pool, (b) running an eval on a wedged GPU. The empty_cache() at end-of-iter
was the only thing that finally surfaced the error.

**Resilience patches applied (this post-script):**
1. `ppo_update_v8` now returns `n_succeeded` and `n_failed` in its stats dict
   (additive, doesn't change any existing behavior).
2. `rl_train_v9.py` main loop checks `n_succeeded == 0` immediately after the
   PPO update; on hit, saves `emergency_iter_NNNN.pt` and `sys.exit(2)`.
3. Snapshot save block also gates on `n_succeeded > 0` (belt-and-suspenders).

These are additive — Session-29-era per-episode resilience is preserved exactly.
Cloud relevance: same gap exists for `--mp` failures (worker death, IPC timeouts,
NCCL hangs). Patches apply equally there. They are now a prerequisite for trusting
a 2-day spot run's eval signal.

**Optional Windows-only mitigation (not auto-applied):** raise TDR delay via
`HKLM\System\CurrentControlSet\Control\GraphicsDrivers\TdrDelay` (DWORD = 60 seconds),
then reboot. Standard ML-on-Windows workaround. May or may not have been the actual
trigger — can't be proven without a reproducer — but it's free insurance.

### Session 33 ERA TRAJECTORY ANALYSIS — what the Elo curve actually shows

After the extended ladder, we mapped each measured snapshot to its training era from the
canonical era breakdown (Era 1-6 below + S32/S33). The result is a **per-era Elo trajectory**
that shows the plateau is NOT flat — it has real structure that maps directly to known
session work.

**The trajectory plot (sorted by iter, ASCII):**

```
Iter   Elo    Era                          Trajectory (Elo 700 .................... 1050)
----   ----   --------------------------   ------------------------------------------------
   0    806   E0  BC base                  ##################
  14    795   E1  pre-fix collapse         ################
 194    727   E1  pre-fix collapse         ####                       ← deepest collapse
 284    945   E2-3 type+switch eff         ##########################################
 324    955   E2-3 type+switch eff         ###########################################
 409    982   E4  stability patches        ################################################
 589   1015   E4  stability patches        #####################################################
 699    998   E5  KL gate / old "peak"     ###################################################
 724    989   E6  S31 root-cause fixes     #################################################
 824   1018   E6  S31 root-cause fixes     ######################################################
 824   1013   E6  S31 root-cause fixes     #####################################################
 879    992   E6  S31 root-cause fixes     ##################################################
 949   1000   E7  S32 speed work           ###################################################
1019   1000   E7  S32 speed work           ###################################################
1059    973   E7  S32 speed work           ##############################################
1089    998   E7  S32 speed work           ###################################################
1149    988   E7  S32 speed work           #################################################
1239    970   E7  S32 speed work           ##############################################
1289    980   E7  S32 speed work           ################################################
1349   1018   E7  S32 speed work           ######################################################
1419    990   E7  S32 speed work           #################################################
1499    984   E7  S32 speed work           ################################################
1599   1014   E8  S33 terminal reward      #####################################################
1724   1027   E8  S33 terminal reward      #######################################################
1739   1021   E8  S33 terminal reward      #######################################################
1759   1004   E8  S33 terminal reward      ####################################################
1779   1013   E8  S33 terminal reward      #####################################################
1784   1032   E8  S33 terminal reward      ########################################################
```

**Era aggregates (mean, range, delta vs previous era):**

```
  Era    Iters       n   Mean    Range          vs prev   Description
  -----  ----------  --  ------  -------------  --------  -------------------------
  E0     0-0          1   806    806             —        BC base (init)
  E1     1-279        2   761    727-795        -45       Pre-fix early PPO collapse
  E2-3   280-339      2   950    945-955       +189       Type+Switch eff breakthrough
  E4     340-699      3   998    982-1015       +48       Stability patches era
  E5     700-723      1   998    998             0        KL gate / old smart_avg peak
  E6     724-939      4  1003    989-1018        +5       S31 root-cause fixes
  E7     940-1499    10   990    970-1018       -13       S32 speed work + many tweaks
  E8     1500-1784    6  1018   1004-1032       +28       S33 terminal reward (stable)
```

**The key finding from the era analysis** (validated user intuition):

The "plateau" interpretation that came out of the first Elo ladder run is INCOMPLETE. The
trajectory has real structure:

1. **Genuine breakthrough phase (E1 → E4):** PPO climbed from Elo 761 (collapse low) to Elo
   998 (E4 mean). This is where type effectiveness + switch effectiveness features unlocked
   ~240 Elo of real improvement. Big architectural lever.

2. **Natural steady state (E4 → E6):** Mean climbed from 998 to 1003. **+5 Elo over ~290
   iters.** This is what the actual "training-time-only" improvement rate looks like once the
   architecture is at its capacity — extremely slow but slightly positive.

3. **S32 disruption regression (E6 → E7):** Mean DROPPED from 1003 to 990. **-13 Elo.** This
   is the user's "post-1000 had many tweaks where we had to keep resetting starts" intuition
   confirmed empirically. Session 32 included: snapshot pool overhauls, batched temporal
   optimization (with bugs along the way), `--lr-restart` events that destroyed optimizer
   momentum, adaptive entropy experiments, reward shaping changes. **Each disruption cost Elo.**

4. **S33 recovery (E7 → E8):** Mean recovered from 990 to 1018. **+28 Elo.** Session 33's
   stability (terminal reward locked in, no more reward experiments, deeper pool used
   consistently, no `--lr-restart` events) let the model claw back what S32 lost AND slightly
   exceed the previous high (snapshot_1784 at 1032 is now the best ever measured).

**What this means for the "training has plateaued" interpretation:**

- **NOT a flat plateau.** It's a real ceiling around Elo 1018-1032 (E6 best + E8 best) with
  a real ~13 Elo dip in S32 caused by disruptive tweaks.
- **The model's "useful improvement rate" from E4 onward is ~0.018 Elo/iter** — 36x slower
  than the early phase but not zero. The signal is REAL, just glacial.
- **The S32 regression was avoidable.** If we hadn't been doing disruptive experiments in S32,
  the trajectory might have been a cleaner climb from 1003 (E6 mean) to ~1020-1030 (E8 best)
  without the dip in between.
- **The architectural ceiling is ~Elo 1018-1032, not ~1029.** The first ladder run gave a
  point estimate of "snapshot_1784 at 1029" which I called the ceiling. The actual ceiling is
  better described as a band of 1015-1032 across multiple stable-era snapshots.

**Lessons for the next phase (BC scaling, multi-gen):**

1. **Don't tweak training mid-run.** Each `--lr-restart`, optimizer hyperparam change, or
   pool composition change has a measurable cost. Plan experiments as separate clean runs.
2. **The improvement rate metric matters more than absolute Elo.** If BC scaling moves us
   from 0.018 Elo/iter back to 0.1+ Elo/iter early in a new training run, that's a strong
   positive signal even before the absolute Elo lands.
3. **The architecture genuinely has more headroom than the smart_avg plateau suggested**, but
   only marginally — maybe ~30 more Elo if we trained cleanly without disruptions for another
   1000 iters. Not a transformative gain. The big lever is still bigger BC base / architectural
   change, not training-time scaling.

**Does the model have room to grow with more iters? (Session 33 final analysis)**

Short answer: **probably yes, but bounded and uneconomic.**

The stable-era rates (E4->E6: 0.015 Elo/iter, E6->E8: 0.019 Elo/iter) are remarkably similar
at ~0.017 Elo/iter. The rate is NOT visibly decelerating between these two measurement points.
This is **weak evidence for roughly linear improvement within this regime** (i.e., the asymptote
isn't hit yet). Extrapolating at 0.018 Elo/iter:

  +50 Elo (clearly above top bots): ~2,800 iters = ~9 days @ 270s/iter
  +100 Elo (Elo ~1130): ~5,600 iters = ~17 days
  +200 Elo (Elo ~1230): ~11,100 iters = ~35 days
  +700 Elo (VGC-Bench territory): ~39,000 iters = ~4 months continuous

BUT: (1) most ML training curves are logarithmic/saturating, not linear, so the rate will
almost certainly decay further at some point — we just can't tell when from 2 data points;
(2) even the optimistic linear case gives uneconomic returns compared to architectural changes
(BC scaling is expected to give +50-200 Elo from a single experiment, not months of grinding);
(3) to experimentally determine where the rate starts bending, we'd need a focused 1000-2000
iter clean continuation experiment with Elo measurements every ~200 iters (~3.5 days compute).

Decision: **don't run this experiment standalone.** The BC scaling experiment (which takes
similar compute) is more likely to move things by an order of magnitude. If BC scaling fails,
then revisiting "just train more from snapshot_1784" becomes a meaningful backup plan.

**Visualization tool:** `analyze_elo_trajectory.py` + `data/eval/eras.json` produce the full
trajectory plot (scatter + era bands + bot anchors + per-era means) as PNG or interactive
matplotlib. CSV export is also available for further analysis.

### Session 33 ELO LADDER EXTENDED — added mid-era snapshots, killed the dip hypothesis

After the first ladder run, added 7 mid-era snapshots (iter 949-1499) to verify whether
the "iter 880-1500 dip" hypothesis from the first run was real. **It wasn't.** The mid-region
has both highs (dip_1349 at Elo 1018, #4 overall) and lows (dip_1289 at 980), oscillating
around the same noise band as everything else.

**Extended ladder: 38 players, 703 matches, anchored SH=1000.** Canonical result file:
`pokemon-ai-starter/pokemon-ai/src/data/eval/elo_session33_EXTENDED_FINAL.json`.

**Top 12:**
```
 #1  snapshot_1784      Elo 1032  [1009-1055]   ← latest, still top
 #2  pre_crash_1724     Elo 1027  [1005-1049]
 #3  iter1739_eval      Elo 1021  [ 998-1044]
 #4  dip_1349           Elo 1018  [ 997-1041]   ← MID-ERA, top tier!
 #5  snapshot_0824      Elo 1018  [ 994-1040]
 #6  snapshot_0589      Elo 1015  [ 990-1037]
 #7  snapshot_1599      Elo 1014  [ 991-1036]
 #8  snapshot_0824_1    Elo 1013  [ 992-1037]
 #9  iter1779_eval      Elo 1013  [ 991-1035]
#10  iter1759_eval      Elo 1004  [ 982-1025]
#11  Tactical           Elo 1000  [top bot]
#12  SH                 Elo 1000  [anchor]
```

**Key updates from the first run:**
1. **dip_1349 (iter 1349) is at Elo 1018, #4 overall** — statistically tied with snapshot_0589
   and snapshot_0824. The "mid-era dip" interpretation is dead. The mid-region is noisy
   like everything else.
2. **snapshot_0589 dropped Elo 1019 → 1015** with more data. Regression to the mean as
   predicted. NOT a freak outlier — just one of ~12 plateau-band snapshots.
3. **All shifts are 3-5 Elo, well within bootstrap noise.** Both ladders are consistent.
   The first run's headlines stand.

**Sharper interpretation enabled by the extended data:**

The plateau era (iter 0589 onward) has:
- Min Elo: 970 (snapshot_1239)
- Max Elo: 1032 (snapshot_1784)
- Range: 62 Elo total
- Most snapshots within 980-1020 (40 Elo wide)

**1200 iters of training produced 62 Elo of total range, mostly noise.** No clear trajectory,
no improvement curve, just oscillation in a fixed noise band. The model reached the
architectural ceiling by iter ~590 (right after the type effectiveness breakthrough) and has
stayed there. The latest snapshot is the top of the noise band (~14 Elo above #4 dip_1349,
within CI overlap), but NOT meaningfully better than snapshots from 1200 iters earlier.

**The case for "architectural change > more training" is stronger:**
- Extended data shows training has produced essentially zero improvement since iter ~590
- More compute won't break this — we have 1200 iters of empirical proof
- The lever must be pre-PPO (bigger BC) or architectural (capacity, head count, etc.)

**Methodology note (resume + merge):**
First attempt at extending the ladder hit a partition mismatch — adding 7 new players
shifted matchup indices, so each shard's per-shard JSONL only matched ~8 of its old matches
to the new partition. Fixed by merging all 4 old JSONLs into one master and copying to each
shard's slot, so every shard's resume sees ALL 469 previously-completed matches and skips
them. After fix: each shard ran ~58 new matchups (vs the broken ~168), saving ~90 minutes.
Total wall for the extension: ~57 minutes (4 shards in parallel) for the 234 new matches.

### Session 33 ELO LADDER RESULT — first real Elo measurement (2026-04-08 ~15:02)

**TL;DR: snapshot_0699 was a smart_avg illusion. The latest snapshot is the strongest. The plateau
is real and we're at "barely above bot tier" in absolute Elo terms.** Full results in
`pokemon-ai-starter/pokemon-ai/src/data/eval/elo_session33_FINAL.json`.

Tournament: 31 players (3 current-run eval-point snapshots + BC_base + peak_0699 + pre_crash_1724
+ 15 sampled historical + 10 bot anchors), 50 games per matchup, 465 total matchups, 4 parallel
shards with the new permanent-fix script (PlayerPool + checkpoint cache + JSONL incremental save).
**Wall time: ~93 minutes.** SH anchored at Elo 1000.

**Top 12:**
```
 #1  snapshot_1784      Elo 1029  [1003-1053]   ← latest (the strongest checkpoint we have)
 #2  pre_crash_1724     Elo 1024  [ 998-1049]
 #3  iter1739_eval      Elo 1020  [ 994-1044]
 #4  snapshot_0589      Elo 1019  [ 996-1042]
 #5  snapshot_0824_1    Elo 1017  [ 993-1041]
 #6  snapshot_0824      Elo 1013  [ 988-1039]
 #7  snapshot_1599      Elo 1007  [ 981-1034]
 #8  iter1759_eval      Elo 1007  [ 982-1032]
 #9  iter1779_eval      Elo 1007  [ 980-1030]
#10  Tactical           Elo 1004  [ 976-1030]   ← top bot
#11  peak_0699          Elo 1000  [ 973-1026]   ← old "all-time peak"
#12  SH                 Elo 1000  [anchor]
```

**Bottom (for reference):**
```
#23  BC_base            Elo  809  ← PPO starting point
#24  snapshot_0014      Elo  795
#25  SetupThenSweep     Elo  785
...
#31  Random             Elo  433
```

**THE BIG FINDINGS:**

1. **snapshot_0699 (the long-claimed 57% smart_avg "all-time peak") is at Elo 1000** — tied with SH
   anchor and BELOW most other snapshots. We've been chasing a smart_avg variance spike for
   1000+ iters thinking it was a real strength peak. It wasn't. The "we never beat 0699" panic
   was looking at the wrong metric. **Smart_avg has been actively misleading us about which
   snapshot is best.**
2. **Latest snapshot (snapshot_1784) is the strongest at Elo 1029** — by a hair, but consistently
   above most predecessors and clearly above peak_0699. Training has been monotonically improving
   slowly all along; we just couldn't see it in smart_avg's noise.
3. **PPO actually worked: BC_base 809 → snapshot_1784 1029 = +220 Elo.** That's substantial
   (~78% expected win rate of latest vs BC). The Phase D/E pipeline is not broken.
4. **The plateau is real at the Elo level**: top 9 snapshots within 22 Elo (1007-1029). All within
   bootstrap CI overlap. This is the actual ceiling of the current architecture+training regime
   over 1500+ iters of training.
5. **We're at "tournament bot tier"** — latest beats top bot (Tactical) by 25 Elo = ~54% expected
   win rate. Marginal edge over the heuristics, not dominant.
6. **Cross-reference to VGC-Bench BCFP (1768 ELO at our compute scale)**: we're at 1029 (different
   anchors, different format — not directly comparable, but the magnitude of the gap is striking).
   Their architectural choices produce ~700 more Elo of headroom that we lack.
7. **Two real breakthroughs in the trajectory**:
   - snapshot_0194 (729) → snapshot_0284 (951): **+222 Elo from type effectiveness features**
     (Session 30's big win, now confirmed in absolute Elo terms)
   - snapshot_0284 → snapshot_0589: +68 Elo (steady improvement post-features)
   - Everything after snapshot_0589: ±30 Elo oscillation around ~1015 (the plateau)

**Implications for cloud burst decision:**
- Cloud burst on current architecture probably gives +50-100 Elo over weeks. Useful but small.
- The bigger lever is upstream: **Metamon's paper says model size matters for BC > RL**. Our
  BC_base at Elo 809 is the foundation everything is built on. A bigger BC trained on the full
  389K replay corpus might land at 1100-1200, and PPO from there could reach 1300-1400.
- **Do not interpret the plateau as "architecture has zero headroom"** — it's "this specific
  PPO setup with this specific BC base has zero remaining headroom." The BC base IS the
  architectural lever to test next.

**Methodology validation (the script that produced this):**
The `eval_elo_ladder.py` script went through several debugging iterations during this session:
- v1 had a `n_battles` positional vs keyword bug, unicode arrow crash, and most critically,
  a "fresh player per matchup" pattern that caused CUDA allocator fragmentation and a death
  spiral (per-matchup time degraded from 80s to 595s over 2 hours, ETA jumped to 14h).
- v2 (the permanent fix, applied this session) added: checkpoint state_dict caching (CPU),
  PlayerPool with LRU eviction (5 snapshots resident, 10 bots persistent), incremental JSONL
  save with auto-resume on restart. **Wall time dropped from projected 14h to actual 93 min.**
  Also added `recover_elo_from_log.py` for crash recovery from the log files of the v1 attempt.

### Session 33 RESEARCH ROUND — Architecture deep dive (post-crash)

After resilience patches were in and training was resumed cleanly, did a fresh round
of source-code research on Metamon, VGC-Bench, and ps-ppo to verify or disprove
several plateau hypotheses. **See `docs/RESEARCH.md` section 0 for the canonical
comparison table.** This is the summary of what changed.

**Concrete configs found** (verified from actual repo source, not paper claims):
- **VGC-Bench BCFP** (1768 Elo @ 5M states, our compute scale): d_model=256, 3 layers,
  4 heads, ff_dim=256 (NO expansion), 13 tokens (1 CLS + 12 mons), **stateless core**,
  optional frame stack. Likely <10M params.
- **Metamon Small** (15M, gins/small_agent.gin): per-step encoder d=100/3L/5H,
  trajectory transformer d=512/3L/8H/ff=2048, 200-turn context, 4 critics with popart.
- **Metamon Medium** (50M): per-step d=100/3L/5H, trajectory d=768/6L/8H/ff=3072.
- **Metamon Large** (200M): per-step d=160/5L/8H, trajectory d=1280/9L/20H/ff=5120.
- **ps-ppo**: 4-layer transformer, 8 heads, 15 tokens, **stateless**, **NO opponent pool**
  (continuously synced live policy via Ray, no checkpoint history at all).
- **Ours**: per-step d=384/4L/4H/ff=768, temporal d=384/2L/4H, 16 tokens, single
  distributional critic.

**Five things research VINDICATED about our setup:**
1. Temporal model is justified for OU. Metamon (closest format match) uses heavy temporal.
2. Uniform-over-history pool is justified. VGC-Bench's BCFP **wins their pool A/B**
   over both Nash-weighted (BCDO) and latest-only (BCSP). Our filtered uniform is BCFP.
3. Distributional value head is justified. ps-ppo and Metamon both use it.
4. Entity tokenization is justified. All references use it.
5. BC pretrain → PPO fine-tune is justified. All references use it.

**Five things research raised QUESTIONS about:**
1. **Capacity allocation is inverted vs Metamon.** We have heavy per-step (384d) and
   light temporal (384d/2L). Metamon Small has light per-step (100d) and heavy
   temporal (512d/3L/8H). Even Metamon's smallest has more temporal capacity than us.
   We may be putting capacity in the wrong place for OU's strategic depth.
2. **Per-step ff_dim doubled** (768=2×d_model) but VGC-Bench uses ff_dim=d_model.
   May be wasted parameters per layer.
3. **Single critic** vs Metamon's 4-critic ensemble with popart. Variance reduction
   benefit unmeasured.
4. **4 attention heads** vs ps-ppo's 8. Free A/B test (param-neutral if d_head halves).
5. **Pool filter `sp≥260`** vs BCFP's no-filter. May discard useful signal — important
   caveat: pool was capped at 100 or smaller for most of training history; deep pool
   only used in recent ~500 iters. The plateau predates the deep pool.

**The biggest single new insight:**
We're at almost exactly VGC-Bench's compute scale (5M states). VGC-Bench BCFP achieved
**1768 Elo** at this compute, on a HARDER format (doubles' combinatorial action space).
**We don't know our Elo.** Our smart_avg of 52% is on a metric Session 29 already
flagged as a poor predictor of H2H strength. We could be at 1700 (architecture fine,
scale to break through) or at 1500 (real local gap to close before scaling).
Measuring actual Elo is now the gating prerequisite for the cloud burst decision.

**Retractions from earlier Session 33 analysis:**
- "Stateless variant" suggestion: was too one-sided. Two of three references are
  stateless or have stateless cores, but the only one closest to our format (Metamon)
  is heavily temporal. For OU, temporal is justified. Don't strip it. (Possibly do
  redistribute capacity into it, though.)
- "Shrink the pool" suggestion: wrong. BCFP wins the pool A/B in the closest published
  comparison. Question the filter, not the depth.
- "Architecture has hit ceiling" claim: cannot be made until Elo is measured. The
  metric we used to call the plateau is the same metric we know is unreliable.

**New canonical order of operations** (from Session 33 research synthesis):
- **Step (a):** Build `eval_elo_ladder.py` — round-robin tournament across N snapshots
  + 4 smart bots as anchors, BayesElo computation, uses handcrafted 70 OU teams
  (lower variance than procedural for measurement clarity). This produces our actual
  Elo. **GATING for everything else.**
- **Step (b):** Code refactor — decompose `rl_train_v9.py` into focused modules,
  prune dead v7 files, remove obsolete backup dirs (keep `v9_pre_cloud/` as fallback),
  add 60-second integration smoke test, switch ad-hoc prints to `logging` module.
  Can run in parallel to (a).
- **Step (c):** After (a) gives Elo:
  - If Elo > 1700: architecture is fine, run cloud burst as scaling test
  - If Elo < 1600: local fixes first
    - Test pool filter: sp≥0 vs sp≥260 (1-day)
    - Test 4 heads → 8 heads (1-day, param-neutral)
    - Consider capacity redistribution (lighter spatial, heavier temporal)
    - Re-measure Elo, decide on cloud
- **Step (d):** Cloud burst, with revised success criterion:
  delta-vs-baseline AND states-consumed, NOT absolute smart_avg %.

### Session 32 — what's in the codebase now

**Speed optimizations (all verified):**
1. Batched temporal — one GPU call for all N histories. ~5% faster collection.
2. Yield-before-fire — `await asyncio.sleep(0)` before batch fire. Batch sizes max 26.
3. Pipeline (`--pipeline`) — overlaps collection with PPO update. **~270s/iter vs 340s baseline.**
4. FLOW instrumentation — every iter prints exact timing of each phase.
5. Profiling built into InferenceBatcher — batch size/GPU time stats per wave.

**Multiprocess infrastructure (cloud-ready, SLOWER locally):**
- `--mp` flag — workers in separate processes (mp_collect_v2.py)
- `mp_collect_v3.py` — PSPPO style with InferenceServer as separate PROCESS (correct, not thread)
- Verified correct: trajectories pass integrity tests, no battle mangling
- Locally: 415s/iter (mp+pipeline thread had bugs, mp alone too slow due to CPU opponents)
- On cloud: should be the fastest option with GPU opponents + 8 workers

**Snapshot pool changes:**
- Disk scan on resume rebuilds full pool (was sliding window of 100 only)
- Pool now ~430 snapshots from disk (was always 100-200 max)
- **Filter: sp >= 200** — excludes pre-type-effectiveness era (random play, useless signal)
- Saves last 500 in checkpoint (was 100)

**Things that did NOT help locally (DON'T retry):**
- Multiple servers without `--mp`: single event loop can't use them.
- `run_in_executor` for GPU forward: GIL + CUDA contention = 7x SLOWER.
- Multiprocess locally: IPC queue overhead + CPU opponents.
- CPU pipeline collection: MKL saturates cores, starves GPU PPO.
- PSPPO separate process: dual CUDA contexts thrash on 3060 (60% slower).
- Direct player (`--direct`): capped at 1.5 g/s.
- High concurrency (c=30): event loop can't process 30 battles fast enough.
- mp+pipeline (thread version): GPU contention + race conditions = all-zero PPO.

**Profiling (collection bottleneck is event loop, not GPU):**
- GPU inference: ~18s total per iter (12% of collection time)
- Websocket I/O + event loop: ~130s (88% of collection time)
- GPU forward blocks event loop for ~14ms per batch
- Multiprocess separates these → cloud win, local loss

**Key invariants from Session 31 still in effect:**
- FP32 summary_attn prevents NaN
- grad_accum=1 is safe
- ent_coef=0.04 fixed (no adaptive entropy)
- "Listen interrupted" = normal cleanup, not crashes
- --lr-restart destroys optimizer momentum, avoid

**IMPORTANT - process hygiene:**
After mp/spawn experiments, kill ALL python processes (taskkill //F //IM python.exe).
Zombie processes from spawn'd workers can persist and contend for GPU compute, causing
unexplained slowdowns (we saw 415s/iter when zombies were holding resources, vs 270s clean).

### Session 33 additions to codebase

**`--reward-style {dense,sparse,terminal}` flag in rl_train_v9.py:**
- `dense` (default): ko_coef=0.05, hp_coef=0.02, terminal=1.0, immune_penalty=0.01
- `sparse`: ko_coef=0, hp_coef=0, terminal=1.0, immune_penalty kept
- `terminal`: ko_coef=0, hp_coef=0, terminal=1.0, immune_penalty=0 (pure outcome)

**Value-warmup behavior:**
- `--warmup-iters N` freezes backbone+policy, trains only value head for N iters
- Used when switching reward shape so value estimates re-anchor before policy updates
- Resume + warmup is supported (no --lr-restart needed; preserves optimizer momentum)

### Resume command (current run, terminal reward):
```
python -u rl_train_v9.py --init-from data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt \
  --resume LATEST_SNAPSHOT.pt --device cuda --servers 9000,9001,9002 --fp16 --pipeline \
  --games-per-iter 200 --max-concurrent 10 --n-iters 500 \
  --warmup-iters 0 --reward-style terminal --ent-coef 0.04 --grad-accum 1 \
  --procedural-teams C:/Users/raiad/OneDrive/Desktop/team_builder/raw_data/pokemon_usage/2024-04
```
Find latest: `ls -t data/models/rl_v9/selfplay_v9_2026040*/snapshot_*.pt | head -1`

**IMPORTANT:** `--procedural-teams` needs ABSOLUTE path. Relative path from src/ loads 0 Pokemon.

### Battle servers (start 3):
```
tools/node-v20.18.1-win-x64/node.exe battle_server.js --port 9000
tools/node-v20.18.1-win-x64/node.exe battle_server.js --port 9001
tools/node-v20.18.1-win-x64/node.exe battle_server.js --port 9002
```

### Trail logs (PowerShell):
```powershell
Get-Content "C:\Users\raiad\OneDrive\Desktop\team_builder\pokemon-ai-starter\pokemon-ai\src\training_pipeline_run.log" -Wait | Select-String "^Iter|EVAL|Snapshot|TAINTED|NaN|ERROR"
```

### Key checkpoints:
- `selfplay_v9_20260401_141524/snapshot_0699.pt` — 57% eval peak (original optimizer)
- `selfplay_v9_20260405_164115/snapshot_1279.pt` — current resume point (53%)
- `BEST_PPO_iter80_h2h_52.8pct.pt` — init-from (BC base, required for dim expansion)

### Hyperparameters (current):
- ent_coef=0.04, grad_accum=1, target_kl=0.03, clip_eps=0.2, ppo_epochs=5
- lr=1e-4, gamma=0.9999, lam=0.75, max_grad_norm=0.5, vf_coef=1.0
- immune_penalty=0.01, ko_coef=0.05, hp_coef=0.02
- Pipeline: ON (--pipeline), Adaptive entropy: OFF

### Analysis tools:
- `python analyze_eval.py --replay-dir DIR --deep` — playstyle, switch quality, momentum
- `python analyze_eval.py --replay-dir DIR/SH DIR/Strategic --labels SH Strat` — per-bot
- `python analyze_status_targeting.py REPLAY_DIR` — burn/para/toxic targeting intelligence
- `python analyze_sp_wr.py` — self-play WR by opponent era (edit log paths inside)

### Cloud deployment: see docs/CLOUD_DEPLOY.md
Ready: `--mp` flag, GPU opponent support, torch.compile, multiprocess workers.
Upgraded version in `mp_collect_v2.py` (configurable timeout, GPU opponents, pipeline combo).

### Backups:
- `backups/v9_session32_final/` — after all Session 32 optimizations
- `backups/v9_pre_batch_temporal/` — before Session 32 changes
- `backups/v8_pre_switch_offensive/` — before type effectiveness features

### Files (Session 32 additions):
- `mp_collect_v2.py` — multiprocess collection v2 (queue-based, reuses MPRLPlayer)
- `mp_collect_v3.py` — PSPPO style: InferenceServer as separate process. Tested correct.
- `test_batched_temporal.py`, `test_batched_temporal_v2.py` — temporal batching correctness
- `test_mp_collection.py` — mp correctness verification
- `analyze_status_targeting.py` — burn/para/toxic targeting intelligence

## Next Steps (Session 33 final — post-Elo measurement)

**Canonical source: `docs/NEXT_SESSION.md`**. The plan below is a summary that may go stale;
treat NEXT_SESSION.md as truth.

### Status: Training STOPPED, Elo measured, plan revised

- Latest snapshot: `selfplay_v9_20260408_042048/snapshot_1784.pt` at Elo 1032 (top of ladder)
- Elo measurement complete: extended ladder, 38 players, 703 matches, anchored SH=1000
- Result: **the architecture is at its ceiling.** 1200+ iters of training between snapshot_0589
  (Elo 1015) and snapshot_1784 (Elo 1032) produced ~17 Elo of net change, all within bootstrap
  CI overlap. More training compute is empirically NOT the lever.
- **VGC-Bench BCFP at our compute scale: Elo 1768.** ~700 Elo gap. Architecture-level intervention
  needed.

### NEXT: Code refactor (Task 8 — pending)

Decompose `rl_train_v9.py` (1900+ lines) into focused modules. Add `test_smoke_train.py` for
crash safety. Move v7 legacy files to `legacy/`. ~1-2 days. Unblocks clean A/B testing for
the BC scaling experiments that follow.

See `docs/NEXT_SESSION.md` step (b) for the detailed sub-steps and file decomposition.

### THEN: Multi-gen prep BEFORE BC scaling (per user direction)

The BC scaling experiment (Metamon's "size matters for BC > RL" thesis) is the highest-leverage
architectural lever. But per Session 33 user direction, the multi-gen prep happens FIRST so the
scaled BC base is multi-gen-capable from day one (avoiding a redo when multi-gen lands later).

Sequence:
1. Multi-gen vocab + feature prep (1-2 weeks)
2. Multi-gen replay scrape — gen 6/7/8 OU (1-2 weeks, mostly automated background)
3. **30M BC scaling test** on multi-gen data (3-7 days local, 6GB GPU max)
   - Decision threshold: 30M BC base reaches Elo 900+ (vs current BC_base 806)
4. PPO from new BC base + Elo measurement
   - Decision threshold: new PPO ceiling beats snapshot_1784 (Elo 1032) by ≥+50 Elo
5. (only if 3+4 succeed) cloud BC at 50M+ params
6. Cloud burst — see `docs/CLOUD_DEPLOY.md` for revised checklist + decision criteria

If the 30M BC scaling test FAILS to move things, the lever is elsewhere — pivot to capacity
reallocation (`memory/feedback_capacity_allocation.md`) or other architectural changes.

### What NOT to do (Session 33 retractions)

- Don't run cloud burst on the current architecture (1200 iters of plateau proves it won't work)
- Don't grind smart_avg as the primary metric (snapshot_0699 was a smart_avg lie at Elo 998)
- Don't try to "reproduce snapshot_0699" — latest snapshot (1784) is genuinely stronger by ~34 Elo
- Don't strip the temporal module (Metamon, our closest format match, uses heavy temporal)
- Don't use the original c1/c2 plan (head count A/B + filter loosening) as standalone fixes —
  they each give ~30-80 Elo, but the gap is ~700 Elo. Smaller experiments are not the lever.

---

## Full Feature Timeline & Impact History

### Era 1: Baseline (iters 140-252) — 24-26% eval
**Features:** Basic move flags (BP, accuracy, category, STAB). No type effectiveness.
**Runs:** phase_e, phase_e_clean, phase_e_ent04. ent=0.02→0.04.
**Evals:** 6 evals, all 24-26%. Dead flat. Model couldn't distinguish SE from neutral moves.
**CONCRETE:** The 22-26% ceiling was real — measured across 100+ iters, multiple restarts.

### Era 2: Type Effectiveness (iters 160-340) — 26→47%
**Features added:** `type_effectiveness` + `opp_threat` per move. MOVE_SLOT 107→109 dims. lr-restart.
**Run:** type_eff (160-340), 181 iters, 9 evals, zero spikes.
**Evals:** 30→37→42→44→44→47→47→47→45%. The breakthrough.
**CONCRETE:** +21 points. Weight analysis: type_eff was #1 signal (35.1), 5x > base power.
**CONJECTURE (high):** Model learned "pick SE move" but not yet "switch to resist attacks."

### Era 3: Switch Effectiveness + Floor Fix (iters 280-337) — 47→52%
**Features added:** `defensive_eff` + `offensive_eff` per switch. SWITCH_SLOT 28→30 dims.
Floor fix: 0.25→0.0 (immune matchups were invisible). lr-restart.
**Runs:** switch_off (280-306), switch_off_v2 (285-337). First spikes at iter 306.
**Evals:** 51→50→52%.
**CONCRETE:** +5 points. Defensive_eff = #1 switch signal (23.8). Model learned to switch into resistances.
Collapsed at 333 — per-episode stepping amplified bad gradients.

### Era 4: Stability Patches (iters 320-634) — oscillating 49-55%
**Code fixes (no new features):** NaN abort, -100.0 masking, 0.01 floor, ent_coef=0.03.
**Runs:** vclip, ent03, ent03_safenan (327 iters, longest run). Many restarts from iter 319 checkpoint.
**Evals:** Oscillated 49-55%. Peaked 55% at iter 339. Self-recovered from collapses.
**CONCRETE:** No eval improvement over era 3 peak. 40 collapse-era snapshots entered pool.
**CONJECTURE (high):** Long ent03_safenan run built optimizer momentum encoding strategic timing.

### Era 4.5: Feature Encoding Refactor (deployed during era 4 via cached imports)
**Code changes (same dims):** Enum refactor (all poke-env enums), secondary stat boosts,
confusion chance fix, status_to fix, Chilly Reception detection.
**CONCRETE:** No dim changes. Deployed mid-run. Hard to isolate impact.
**CONJECTURE (medium):** Minimal eval impact. status_to fix most significant (burn/toxic were invisible).

### Era 5: KL Gate + Peak (iters 635-699) — 52→57%
**Fixes:** Per-episode KL gate, adaptive entropy, opponent temperature fix (pool>15 = full strength).
**Run:** klgate (635-719), 85 iters, 4 evals.
**Evals:** 52→55→55→57% (all-time peak at iter 699).
**CONCRETE:** +5 points. KL gate prevented worst gradient updates.
**CONJECTURE (high):** 57% peak was partly from 700+ iters of accumulated optimizer momentum
encoding strategic timing (when to hazard, pivot, recover). This was lost on lr-restart.

### Era 6: Session 31 — Root Cause Fixes (iters 700-939+) — oscillating 48-56%
**Three root causes found:**
1. FP16 NaN: summary_attn overflow → FP32 cast fix (CONCRETE)
2. Per-episode stepping + NaN → grad_accum CLI flag (CONCRETE)
3. Snapshot pool: 40% collapse-era → pruned to 60 healthy (CONCRETE)

**Additional:** lr-restart destroys optimizer momentum (CONCRETE: pi=+0.23 vs -0.02).
Adaptive entropy unnecessary (CONCRETE: research doesn't use it). Tainted discard feedback
loop when NaN existed (CONCRETE: eval dropped as tainted rate climbed).

**Runs:** Multiple restarts from 699. Tested grad_accum=10, 1. Tested lr-restart vs preserved.
**Evals:** 50→53→52→51→52→48→56→54→48→50→52→54%
**Playstyle evolution (CONCRETE):**
- Pivoting: 5.3%(699) → 3.4%(low) → 5.4%(recovering)
- Recovery: 4.7%(699) → 2.3%(low) → 3.7%(recovering)
- Hazards: 2.0%(699) → 0.3%(low) → 0.8%(slowly recovering)
- Attack%: 86%(699) → 92%(peak aggression) → 89%(diversifying)

**CONJECTURE (high):** lr-restart caused strategic regression. Fresh optimizer overreacted on
first updates, partially overwriting learned strategic patterns. Not "forgetting" — the weights
still encoded the knowledge, but uncalibrated optimizer pushed them in wrong directions.
Model is rebuilding: switch quality and pivoting back to 699 levels, hazards slowly returning.

**CONJECTURE (medium):** May surpass 57% once hazards/recovery fully recover. Underlying play
quality (momentum, coverage prediction, waste rate) approaching/exceeding 699 levels.

**CONJECTURE (low):** May be near architecture ceiling ~55-57%. Cloud scaling might be needed.

---

### Session 30 details (for reference)

Session 30: Type effectiveness broke 26%→47%. Floor fix + offensive_eff broke 47%→52%.
Collapsed at iter 333 from value instability. Resume from BEST_switch_off_iter319_52pct.pt.
Need --lr-restart (status_to fix + refactor changed feature values).

**Full eval history:**

| Eval | SH | SmDmg | Tact | Strat | smart_avg | SE% | Immune% | KO avg |
|------|-----|-------|------|-------|-----------|-----|---------|--------|
| iter 159 (base) | 25% | 30% | 24% | 24% | 26% | 12% | 8.8% | 0.76 |
| iter 179 (+20) | 28% | 32% | 32% | 26% | 30% | 15% | 10% | 0.81 |
| iter 199 (+40) | 43% | 40% | 33% | 32% | 37% | 19% | 6% | 0.89 |
| iter 219 (+60) | 40% | 46% | 43% | 41% | 42% | 21% | 5% | 0.94 |
| iter 239 (+80) | 45% | 38% | 41% | 50% | 44% | 24% | 5.5% | 0.97 |
| iter 259 (+100) | 50% | 42% | 42% | 40% | 44% | 23% | 5.5% | 0.95 |
| iter 279 (+120) | 52% | 45% | 44% | 46% | **47%** | 23% | 4.3% | 0.98 |
| iter 299 (+140) | 49% | 46% | 51% | 42% | **47%** | 25% | 4.8% | 1.00 |
| iter 319 (+160) | 54% | 47% | 42% | 44% | **47%** | 25% | 5.0% | 0.99 |
| iter 339 (+180) | 46% | 46% | 43% | 46% | 45% | 25% | 5.3% | 0.98 |

Plateaued at 47% for 3 evals (279-319), then dipped to 45%.
Resumed from iter 279 with offensive_eff + floor fix + --lr-restart.

**Post-fix results (iter 280+ with offensive_eff + floor fix, collapsed at 333):**

| Eval | SH | SmDmg | Tact | Strat | smart_avg | SE% | Immune% | KO avg |
|------|-----|-------|------|-------|-----------|-----|---------|--------|
| iter 299 (off_eff +20) | 58% | 52% | 46% | 44% | **50%** | 28% | 5.8% | 1.04 |
| iter 319 (off_eff +40) | 52% | 57% | 52% | 44% | **52%** | 28% | 5.8% | 1.07 |
| iter 333+ | COLLAPSED — value instability, 10% WR, 0 steps | | | | | |

Peak: 52% at iter 319. KO ratio 1.07. Offensive_eff weight grew 5.7→8.6 (+50% in 20 iters).
Collapsed at iter 333: v-spikes at 289,296,306,322,325,330,332 → fatal at 333.

**Bugs found and fixed during this run:**
- `reset_battles` crash in rl_train_v9.py collection path (same bug as eval, different code path)
- NaN in FP16 inference (cascading from reset_battles crash). Added NaN guard in choose_move.
- Deleted run dir while training was running (test cleanup accident). Fixed by relaunching.

**Model weight analysis (iter 299 snapshot — what the model actually learned):**

Move decisions (action_encoder.move_net weights by importance):
```
Type_Effectiveness  ██████████████████████████████████  34.2  #1 — THE dominant signal
Opp_Threat          ████████████████                    16.6  #2 — should I stay or switch?
STAB                ████████████                        13.0  #3 — same-type bonus
Powder              ████████████                        12.8  #4 — Grass immune to powder
Heal                ████████████                        12.7  #5 — Recover/Roost value
Pivot               █████████                            9.5  #6 — U-turn/Volt Switch value
Base Power          ██████                               6.4  #10 — raw damage << type matchup
```

Switch decisions (action_encoder.switch_mlp weights by importance):
```
Defensive_Eff       ████████████████████████            23.7  #1 — "will I survive the switch-in?"
HP%                 ██████████████                      14.1  #2 — "is this mon healthy enough?"
Status:Paralysis    █████████████                       13.1  #3 — speed crippled = bad switch
Status:Sleep        ████████                             8.9  #4 — asleep = useless
Status:Toxic        ████████                             8.7  #5 — ticking damage = dying
Offensive_Eff       █████                                5.7  #6 — "can it fight back?" (still learning)
```

Key insight: Type_Effectiveness (34.2) is 5x more important than Base Power (6.4) — the model
learned "a weak SE move beats a strong neutral move." Defensive_Eff (23.7) dominates switch
decisions — the floor fix (0.25→0.0) making immune visible was the main driver of 47%→50%.
Paralysis ranked higher than other statuses for switching — correct (speed matters on switch-in).

**New features deployed (iter 280+):**
- `offensive_effectiveness` per switch slot (SWITCH_SLOT 29->30): "how effective are my STAB types vs opponent?"
- `max_eff` floor fix: 0.25->0.0 in _opp_type_threat and _switch_offensive/defensive_effectiveness.
  Bug caused immune matchups to show as neutral. Active during ALL prior training (159-339).
- Both E2E tested: 972+ type matchups verified, 40 live battle switch targets, full pipeline.

**Current training:** From `BEST_type_eff_iter279_47pct.pt`, --lr-restart, ent_coef=0.04.
Run dir: check latest selfplay_v9_2026033* in training_switch_off.log.

**Checkpoints saved:**
- `BEST_phase_e_iter159_base.pt` — base before type_eff (26%, pool=29)
- `BEST_type_eff_iter279_47pct.pt` — peak type_eff (47%, pool=51)
- `backups/v8_pre_switch_offensive/` — all source files before switch offensive_eff addition

### Session 30 — Procedural Teams + Immune Penalty + Type Effectiveness

**Type Effectiveness Features (the big one):**
- MOVE_SLOT_CONT_DIM: 107 -> 109 (+2 dims)
  - `type_effectiveness`: our move vs opp active, via poke-env `damage_multiplier()`, scaled mult/4.0
  - `opp_threat`: max of opp's known moves (or STAB types) vs our active
- SWITCH_SLOT_CONT_DIM: 28 -> 29 (+1 dim)
  - `defensive_effectiveness`: max opp threat vs switch target (lower = safer to switch into)
- All computed via poke-env `Pokemon.damage_multiplier()` — no hardcoded type chart
- Scale: 0.0=immune, 0.0625=4x resist, 0.125=NVE, 0.25=neutral, 0.5=SE, 1.0=4x SE
- Old checkpoints auto-expanded with zero-init on load (3 MoveNets + 1 SwitchMLP)
- Minor fixes: recoil keeps abs() (poke-env stores positive), PP normalized /64 not /40
- `_safe_getattr` added for moves with incomplete poke-env data (e.g. "recharge" pseudo-move)
- `--lr-restart` flag added to rl_train_v9.py (required when dims change)

**E2E tested across FULL pipeline:**
- Observer: 130 rows, correct dims, 70% non-neutral type_eff values
- JSONL->memmap: shapes (N,4,109) and (N,5,29), values preserved
- BC training: 1 epoch trains + saves, 13,382,444 params
- Bot eval: old checkpoint loads with expansion, 4 bots complete
- H2H eval: 2 old checkpoints both expanded, 5 battles complete
- Replay->memmap: 5 HuggingFace replays, 243 rows, correct dims
- RL training: 1 iter PPO with --lr-restart, non-zero stats confirmed
- Correctness: 8 turns hand-verified (Ice vs Dragon/Ground=1.0, Ground vs Grass=0.125, etc.)

**Procedural Team Generator (team_generator.py):**
- 545 Pokemon from gen9 OU/UU/RU/NU/PU/ZU, weighted by Smogon usage
- 99.9% Showdown-valid (3996/4000), 0.3% iter overhead
- Bans: Gouging Fire, Volcarona (post-2024-04 OU bans)

**Immune Penalty:** --immune-penalty 0.01 in RewardShaper

**Phase E training results (iters 140-252, pre-type-eff):**

| Eval | SH | SmDmg | Tact | Strat | smart_avg | Entropy |
|------|-----|-------|------|-------|-----------|---------|
| 159 (ent=0.02) | 25% | 30% | 24% | 24% | 26% | 0.61 |
| 179 (ent=0.02) | 29% | 27% | 24% | 26% | 27% | 0.56 |
| 199 (ent=0.02) | 24% | 28% | 24% | 22% | 24% | 0.54 |
| 219 (ent=0.04) | 28% | 22% | 28% | 22% | 25% | 0.81 |
| 239 (ent=0.04) | 26% | 22% | 22% | 24% | 24% | 0.86 |

Entropy collapsed to 0.51 under ent=0.02, recovered to 0.80+ with ent=0.04.
Win rates stayed at 24-27% plateau — type effectiveness features needed to break through.
Immune rate slowly declined: 12.8% avg -> 10.8% avg across evals.

**Base checkpoint:** `BEST_phase_e_iter159_base.pt` (iter 159, pool=29, best immune 8.8%, most balanced playstyle)
Chosen over iter 249 because: pre-entropy-collapse, best type accuracy, earned strategic diversity.

**Command to start type-effectiveness training:**
```
python -u rl_train_v9.py \
  --init-from data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt \
  --resume data/models/rl_v9/BEST_phase_e_iter159_base.pt \
  --device cuda --servers 9000 --fp16 \
  --games-per-iter 200 --max-concurrent 10 --n-iters 500 \
  --warmup-iters 0 --immune-penalty 0.01 --ent-coef 0.04 --lr-restart \
  --procedural-teams raw_data/pokemon_usage/2024-04
```

Session 29: Hybrid PPO ran 240 iterations, confirmed 22-26% plateau. H2H tournament identified iter80 as best
init (52.8% H2H). Built v9: batched async inference + pure self-play. 36% faster per step vs v8 (49 vs 36 steps/s).
Research finding: ps-ppo is STATELESS (no temporal), making batching trivially fast. torch.compile is a potential win.

### Session 29 — PPO Hybrid Run + Analysis + Self-Play Preparation

**PPO Hybrid Run (240 iterations, iter 41-240 in ppo_v8_20260327_053625):**
Resumed from iter 40 with: target_kl=0.03, ppo_epochs=5, lambda=0.75, warmup_iters=5,
200 games/iter, opponent on GPU, 4 battle servers, snapshot pool=5, HoF=3.
Opponent mix: ~40% self-play snapshots, ~40% smart bots, ~15% easy bots, ~5% BC anchor.

**Bot Eval Results (10 evals, 200 games × 4 bots, random teams per battle):**
| Iter | SH | SmDmg | Tact | Strat | Smart Avg | Notes |
|------|-----|-------|------|-------|-----------|-------|
| 60 | 26% | 25% | 23% | 20% | 24% | |
| 80 | 30% | 26% | 22% | 22% | 25% | SH peak |
| 100 | 24% | 26% | 19% | 22% | 23% | |
| **120** | 26% | 24% | **28%** | 24% | **26%** | Best bot avg, most balanced (4pp spread) |
| 140 | 22% | 24% | 20% | 25% | 23% | |
| 160 | 25% | 20% | 22% | 20% | 22% | |
| 180 | 23% | 25% | 22% | 25% | 24% | |
| 200 | 24% | 24% | 16% | 26% | 22% | OOM aftershock |
| 220 | 28% | 28% | 21% | 18% | 24% | |
| 240 | 22% | 32% | 18% | 19% | 23% | SmDmg peak, most specialized |

**Plateau confirmed at 22-26% smart_avg.** 200 iterations, no breakthrough. Bot mix creates conflicting
optimization signals — model oscillates between strategies that favor different bots.

**Type Accuracy (immune % of all moves, avg across 4 bots):**
| Iter | SE% | Neutral+other% | Resisted% | Immune% |
|------|-----|----------------|-----------|---------|
| 60 | 24% | 51% | 15% | 10% |
| 80 | 26% | 54% | 13% | 8% |
| 100 | 24% | 52% | 15% | 10% |
| 120 | 24% | 49% | 15% | 12% |
| 140 | 24% | 54% | 13% | 10% |
| 160 | 26% | 52% | 14% | 9% |
| 180 | 24% | 53% | 13% | 10% |
| 200 | 22% | 55% | 12% | 11% |
| 220 | — | — | — | — |
| 240 | 25% | 52% | 13% | 11% |

Type accuracy flat throughout. SE ~24%, immune ~10%. Not improving. Metric corrected this session:
old metric excluded neutral hits (misleading ratios). New metric: SE/resisted/immune as % of ALL moves.

**KO Ratios (avg across 4 bots):** Slowly improving: 1.30 → 1.32 → 1.36. Better tactical execution
even though win rates plateaued. Model trading more efficiently but not converting to wins.

**Playstyle Evolution:** Model learned genuine per-opponent adaptation:
- vs SH: Setup sweeping (Gholdengo NP), minimal switching
- vs SmDmg: Heavy pivoting (8-13%), VoltSwitch/U-turn momentum
- vs Tactical: Toxapex utility (Haze, Toxic Spikes), status moves
- vs Strategic: Hazard stacking (Spikes+SR), Heatran Taunt (anti-stall)
But adaptation is strategic only — type targeting remained flat.

**OOM Bug Found & Fixed:**
- iter 163 PPO update hit CUDA OOM on a 1017-turn battle (no turn cap existed)
- 1,903 OOM errors in that single update, pi_loss spiked to +0.19 (corrupted update)
- Caused entropy spike (0.70→1.08) lasting ~30 iterations (iters 170-190)
- **Fix: Added 300-turn cap** to V8RLPlayer.choose_move() — forfeits if trajectory exceeds 300 turns.

**FP16 Inference Added:**
- `--fp16` flag for rl_train_v8.py and bc_policy_player_v8.py
- `torch.amp.autocast("cuda")` wraps forward pass during collection only
- Summary cast to float32 before temporal history storage (prevents dtype mismatch)
- PPO update stays full FP32 (gradient precision). Eval stays FP32 (consistency).
- Expected ~1.5x collection speedup.

**H2H Tournament (11 checkpoints, 200 games per matchup, random teams per battle):**
| Rank | Model | Avg WR | W/G |
|------|-------|--------|-----|
| #1 | iter100 | 52.8% | 1057/2000 |
| #2 | iter80 | 52.8% | 1056/2000 |
| #3 | iter120 | 51.5% | 1031/2000 |
| #4 | iter240 | 51.2% | 1025/2000 |
| #5 | iter180 | 51.1% | 1022/2000 |
| #6 | iter160 | 50.9% | 1019/2000 |
| #7 | iter140 | 50.0% | 999/2000 |
| #8 | iter60 | 49.2% | 985/2000 |
| #9 | iter220 | 49.2% | 985/2000 |
| #10 | iter200 | 46.8% | 935/2000 |
| #11 | BC | 43.5% | 870/2000 |

Key findings: (1) iter100 and iter80 tied at #1, NOT iter30 as bot eval suggested.
(2) BC is clearly weakest — every PPO checkpoint beats it. PPO improved the model.
(3) Spread is compressed (43.5-52.8%) — models play very similarly.
(4) iter200 notably weak (46.8%) — OOM aftershock.
(5) Bot eval (smart_avg) is a poor predictor of H2H strength.

**Playstyle Comparison of Top Models (vs BC, 200 games):**
| Metric | iter80 (#2) | iter100 (#1) | iter120 (#3) | iter180 (#5) | iter240 (#4) |
|--------|------------|-------------|-------------|-------------|-------------|
| Attack% | 76.6% | 82.7% | 87.1% | 88.4% | 88.4% |
| Hazards% | 7.0% | 3.4% | 2.3% | 1.7% | 2.1% |
| Recovery% | 7.1% | 4.8% | 3.9% | 4.6% | 5.1% |
| Status% | 1.7% | 0.6% | 0% | 0% | 0% |
| Setup% | 2.8% | 1.9% | 0% | 1.7% | 0.5% |
| Immune% | 9% | 15% | 14% | 10% | 12% |
| KO ratio | 1.17 | 1.11 | 1.09 | 1.16 | 1.09 |

**iter80 chosen as self-play init** because:
- Most strategically diverse (hazards 7%, recovery 7%, status, setup — all present)
- Best type accuracy (9% immune — lowest of all checkpoints)
- Best KO ratio (1.17)
- Most diverse movesets (mons use 3-4 moves, not just spam one)
- Self-play rewards generalist play — iter80's balanced style is the best foundation
- iter100 tied in H2H WR but one-dimensional (Clefable=100% Moonblast, Garchomp=87% EQ)

**V8_PPO_TODO.md updated:** Items 2,3,5,6,7 marked DONE. Batched inference deferred to v9.
Priority order updated: let hybrid run finish → pure self-play → FP16 → future items.

**eval_h2h_v8.py created:** V8 H2H tournament script with random teams per battle, replay saving,
round-robin win matrix + rankings. Uses BCPolicyPlayerV8.

**analyze_eval.py improved:**
- Added `--player-prefix` arg (supports `p1`/`p2` for H2H analysis)
- Added "vs all moves" metric: SE/resisted/immune as % of total moves (includes neutral+other)
- Old metric (% of flagged hits only) was misleading — excluded neutral hits from denominator

**V9 Built: rl_train_v9.py (batched async inference + pure self-play)**
- InferenceBatcher: collects pending requests from concurrent battles, ONE batched forward pass
- V9RLPlayer: async choose_move (poke-env natively supports Awaitable returns)
- SelfPlayOpponent: BCPolicyPlayerV8 wrapper with temp randomization [1.0, 2.25]
- Architecture: batched spatial (the expensive part) + per-item temporal (cheap, ~0.1ms)
  Temporal can't be batched due to positional embedding mismatch with padded histories.
- Unit tested: batched vs sequential outputs match within 1e-5. ALL TESTS PASSED.
- FP16: RL player uses autocast, opponent stays FP32 (-1e9 masking overflows float16)

**V9 Benchmark (50 games, neural opponent, 2 servers):**
| Config | Steps/sec | Games/sec | Notes |
|--------|-----------|-----------|-------|
| V9 conc=10, fp16, batched | **49** | 1.16 | Batched spatial, per-item temporal |
| V9 conc=20, fp16, batched | **50** | 1.14 | Higher conc doesn't help (GPU-bound) |
| V8 conc=10, sync, neural opp | **36** | 1.05 | Sequential GPU inference |
| V8 conc=10, sync, SH bot | 61 | 1.91 | No GPU cost for opponent |

**V9 is 36% faster per step** vs V8 with neural opponents (49 vs 36 steps/s).
Higher concurrency (20 vs 10) doesn't help because both models serialize on one 6GB GPU.
The speedup comes from batching the RL player's spatial encoding across concurrent battles.

**Research Speed Findings (ps-ppo source code analysis):**
- ps-ppo's model is **STATELESS** — no temporal/recurrent processing. Temporal info baked into
  observation features (transition token with recent move history + effectiveness flags).
  This makes batching trivially fast — stack observations, one forward, done.
- Our temporal transformer (2L/4H/200-turn) prevents full batching — each battle needs its own
  sequential temporal pass. This is why our batching gives 36% not 300%.
- ps-ppo batches 256-2048 observations per GPU call (3ms wait timeout, greedy drain).
- Metamon uses `torch.compile` for model optimization (potential win for us).
- Neither uses ONNX, TorchScript, or quantization. Speed comes from massive batching.
- **Future optimization**: torch.compile, or a "fast policy" mode that skips temporal during
  collection (use spatial-only for action selection, full model during PPO update).

**Bugs Found & Fixed:**
- `_make_server()` double-appended `/showdown/websocket` for port-only args
- FP16 overflow: `-1e9` masking overflows float16. Fix: `.float()` before masking in batcher,
  `fp16=False` for opponent (SelfPlayOpponent).
- `asyncio.to_thread` overhead: ~5-10ms scheduling cost negated batching gains. Fix: run
  `_gpu_forward` directly on event loop (fast enough at ~10ms, blocking is tolerable).
- History padding bug: padded histories give wrong positional embeddings in temporal transformer.
  Fix: batch spatial only, run temporal per-item (cheap, correct).

**Speed optimizations investigated:**
1. torch.compile: NOT SUPPORTED on Windows. Would work on Linux/cloud.
2. KV-cache for temporal: INVESTIGATED but NOT WORTH IT. Profiling revealed temporal
   is only 14% of forward pass (0.95ms). Spatial (54%, 3.7ms) and action_encoder (24%, 1.7ms)
   are the real bottlenecks — both already batched in v9.

**Forward pass breakdown (B=1, 15-turn history):**
| Component | Time | % | Batched in v9? |
|-----------|------|---|----------------|
| forward_spatial | 3.70ms | 54% | YES (main win) |
| action_encoder | 1.67ms | 24% | YES |
| temporal | 0.95ms | 14% | Per-item (cheap enough) |
| heads | 0.54ms | 8% | Per-item |
| TOTAL | 6.86ms | 100% | |

**Conclusion:** v9's 36% speedup from spatial batching is near the practical limit for
this architecture on this hardware. Further gains require: Linux (torch.compile),
bigger GPU (more parallelism), or architectural changes (stateless model like ps-ppo).

**V9 Comprehensive Test Results:**
| Test | Result |
|------|--------|
| A/B vs V8 (neural opp) | V9 13% faster (48 vs 43 stp/s). Modest but real. |
| Opponent GPU vs CPU | GPU 46 stp/s, CPU 35 stp/s. **Keep opponent on GPU.** |
| PPO correctness | **PASS.** V9 trajectories → valid PPO update. |
| 1 vs 2 servers | No benefit (0.99x). GPU is bottleneck, not servers. |
| Concurrency sweep | conc=10-15 optimal. Higher doesn't help. |
| Bot eval compat | Fixed (server_url param). Works. |

**Optimal config:** `--max-concurrent 10 --servers 9000 --fp16 --opponent-device cuda`
Only 1 server needed. Concurrency 10. FP16 for RL player, FP32 for opponent.

**V9 Self-Play RUNNING (iter 136+, run dir: selfplay_v9_20260328_104814/):**
Note: first run (selfplay_v9_20260328_063629/) discarded — all opponents had random temp,
inflating WR and degrading training. Restarted from warmup checkpoint with fixed temp logic.

**Bot eval trend (6 evals so far):**
| Iter | SH | SmDmg | Tact | Strat | Smart Avg |
|------|-----|-------|------|-------|-----------|
| 19 | 26% | 26% | 20% | 22% | 23% |
| 39 | 20% | 25% | 27% | 19% | 23% |
| 59 | 28% | 28% | 21% | 28% | **26%** |
| 79 | 18% | 26% | 23% | 20% | 21% |
| 99 | 19% | 28% | 24% | 18% | 22% |
| 119 | 22% | 24% | 21% | 20% | 22% |

Same 21-26% range as v8 hybrid. No breakthrough.

**H2H vs v8 champions (200 games each, iter99):**
- vs v8 iter80 (init): 100-100 (50-50)
- vs v8 iter100: 99-99 (50-50)
- vs BC base: 101-99 (50-50)
Self-play hasn't improved absolute H2H strength.

**Playstyle evolution (real, steady):**
- Attack: 87% → 82% (less blind aggression)
- Hazards: 3.7% → 6.3% (learned hazard setting)
- Status: 0.1% → 1.6% (using WoW, TWave)
- KO ratio: 1.25 → 1.31 (better trading)

**Type accuracy:** Flat at 10-12% immune vs smart bots. Tested vs dumb bots (Random, MaxBP):
immune 6% vs Random (good — opponent doesn't switch into immunities), but 22% vs MaxBP
(neither side type-aware). Model learned WHEN to switch, not WHICH move to pick.

**Training metrics:** WR ~49-53% vs pool (healthy). Entropy 0.59-0.63 (stable, low). Pool=26.

**Pipeline benchmark result: SLOWER on single GPU (0.80x).** GPU contention between inference
and gradient computation. PPO update took 44% longer. Don't use `--pipeline` locally.
Would help with 2+ GPUs (cloud only).

**Resume if crashed:**
Find latest: `find data/models/rl_v9/selfplay_v9_20260328_104814 -name "snapshot_*.pt" | sort | tail -1`
```
python -u rl_train_v9.py --init-from data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt \
  --resume LATEST_SNAPSHOT.pt \
  --device cuda --servers 9000 --fp16 --games-per-iter 200 --max-concurrent 10 \
  --n-iters 500 --warmup-iters 0 \
  --out-dir data/models/rl_v9/selfplay_v9_20260328_104814
```
Trail: `Get-Content OUTPUT_FILE -Wait -Tail 20`

**Next phase (build while training runs):**
1. **Procedural team generator** from Smogon usage stats (gen9 OU/UU/RU/NU/PU/ZU).
   Weighted sampling of mons/moves/abilities/items/spreads. Rule enforcement (species/item clause,
   ban list). 200-500 generated teams for training diversity. Eval stays on 70 handcrafted teams.
   Data at: `datasets/raw/pokemon_usage/2024-04/gen9{ou,uu,ru,nu,pu,zu}-0.txt`
2. **Immune penalty** in reward shaping. `--immune-penalty 0.01` flag. No NVE penalty
   (legitimate uses: priority finishing, scouting, PP saving).
3. Start new run with generated teams + immune penalty from best v9 checkpoint.
See `memory/project_next_phase.md` for full plan.

**Self-play plan:**
- Init from iter80 (52.8% H2H, most diverse playstyle)
- Pure self-play: opponent from uniform snapshot pool (starts with iter80)
- Temperature randomization [1.0, 2.25] per game for diversity
- Snapshot every 5 iters (pool grows over time)
- Bot eval every 20 iters (generalization tracking)
- Same hyperparams as v8: gamma=0.9999, lam=0.75, target_kl=0.03, ppo_epochs=5
- Same reward: ko_coef=0.05, hp_coef=0.02, terminal ±1.0
- Value warmup: 5 iters
- Watch for: entropy collapse (<0.5 = bad), H2H vs iter80 improvement

**v8 source backed up to `backups/v8_source_backup/`**

### Session 28 — v8 Architecture Planning & Implementation Start

**Architecture Finalized:**
- PokeTransformer: 17 entity tokens (actor, critic, field, transition, 6 our + 6 opp pokemon)
- Spatial transformer: d_model=384, 4 layers, 4 heads with Poke-Mask attention
- Temporal transformer: d_model=384, 2 layers, 4 heads over 200-turn summaries (attention-pooled)
- Sub-networks: PokemonNet (384d), MoveNet (128d), FieldNet (384d), TransitionNet (384d)
- NumericalBanks for all continuous values (HP, stats, BP, accuracy, PP, turn, level, weight, height)
- Distributional value head: 51-bin categorical with two-hot encoding
- CLI flags for all architecture params: `--d-model`, `--temporal-mode summary|frames`, etc.
- Estimated: 12-15M params, 2.5-3.5 GB VRAM (local). Scalable to 120M+ (cloud).

**Feature Audit (91 dims removed, 57+45 added):**
- REMOVED: v4_computed_features (48d), matchup_scalars (3d), move_type_histograms (38d), ctx_extra (41d as concept). All pre-computed damage/matchup estimates removed — entity attention learns these.
- REMOVED: hand-rolled type chart (_type_effectiveness) — poke-env Pokemon.damage_multiplier() used instead.
- ADDED: Transition token (57 features): effectiveness flags (immune/barely/NVE/neutral/SE/ultra for both sides), crit, flinch, status applied, stat changes, KO events, entry hazard damage, field changes, who moved first, our/opp last action IDs.
- ADDED: Expanded volatiles (17→38): MAGNET_RISE, FLASH_FIRE, SMACK_DOWN, HEAL_BLOCK, DESTINY_BOND, IMPRISON, GLAIVE_RUSH, TAR_SHOT, GASTRO_ACID, NO_RETREAT, INGRAIN, SALT_CURE, ENDURE, LOCKED_MOVE, AQUA_RING, SYRUP_BOMB, THROAT_CHOP, STOCKPILE1-3, LASER_FOCUS.
- ADDED: 7-dim Paradox boost encoding (Protosynthesis/Quark Drive + which stat).
- ADDED: Level, ability_revealed, item_revealed per Pokemon token.
- CHANGED: Boosts from normalized float to one-hot(13) per stat (ps-ppo finding).
- KEPT: All 107-dim move slot features, 28-dim switch slots, board state, all per-Pokemon raw state.

**Key Design Decisions:**
- Temporal via summaries (not frame stacking): 200-turn context on 6GB via attention-pooled turn summaries. Frame stacking documented as cloud-GPU ideal with CLI flag.
- No episode length clamp: battles run to completion. Temporal window is sliding 200-turn.
- Clean break from v7: no backward compat with v7 checkpoints.
- All poke-env native API: no hand-rolled type charts or damage calculations.
- New structured memmap format: separate .npy arrays per entity type.

**Files created:** features_v8.py, policy_heads_v8.py, dataset_v8.py, bc_train_v8.py, bc_policy_player_v8.py, convert_jsonl_to_memmap_v8.py, replay_to_memmap_v8.py. Modified: observer.py (--v8 flag), replay_parser.py (v8 import).

**Bot Data Generated:**
- 115 JSONL files, 360,881 rows, 12,104 episodes, 0 errors, 7.86 GB memmap
- All 10 bot pairings, 50 games each, both perspectives

**BC Training Results (5 epochs, bot data, bs=8, lr=1e-4):**
- Epoch 0: val_acc=26.0% → Epoch 4: val_acc=66.6% (loss 1.693 → 0.884)
- Distributional value head converged (v_loss 1.99 → 0.40)
- Total time: 2.6 hours (31 min/epoch). Optimized to ~18 min/epoch with batched spatial.

**Eval vs Bots (50 games each, epoch 4 checkpoint):**
| Opponent | V8 BC | V7 BC | V7 PPO |
|----------|-------|-------|--------|
| Random | 96% | 94% | — |
| MaxBasePower | 62% | 60% | — |
| SimpleHeuristics | **64%** | 20% | 23% |
| SmartDamage | 24% | 26% | 27% |
| Tactical | 24% | 18% | 22% |
| Strategic | 0% | 12% | 20% |
| **Smart avg** | **28%** | **19%** | **23%** |

- v8 BC on bot data alone beats v7 BC (19%) and v7 self-play PPO (23%) at smart avg
- 64% vs SH proves entity attention learns type relationships
- 0% vs Strategic expected: bot data lacks strategic play, model needs human data + more epochs
- Architecture validated. Training speed optimized (16x inference, 1.7x training)

**Human Data Pipeline (in progress):**
- replay_to_memmap_v8.py: streams HuggingFace replays direct to memmap (no JSONL)
- 80K replays at 1500+, ~4M records, ~84 GB memmap
- Speed: 14 replays/sec, ~1.6 hours total
- Zero validation failures in test run

**Training Speed Optimization:**
- forward_sequence() batches all turns' spatial transformer in one GPU call
- Then temporal runs per-turn batched across all batch items
- Result: 1.7s/batch → 0.1s inference (16x), training ~1.7x faster overall

**Human Data Generated (direct-to-memmap, no JSONL):**
- 80K replays at 1500+ from HuggingFace, streamed directly to memmap
- 4,066,905 rows, 159,934 episodes, 108.84 GB memmap
- Zero parse errors, zero validation failures
- Pipeline: replay_to_memmap_v8.py, 95 min at 14.3 replays/sec

**Human BC Training (3 epochs, lr=3e-4, bs=8):**
- Epoch 0: val_acc=43.5% (v7 BC peaked at 41.3% — already surpassed)
- Epoch 1: val_acc=44.1%
- Epoch 2 (partial): accuracy 43.1% (training continuing)
- LR bug found and fixed: warmup lambda was per-epoch not per-batch, giving near-zero LR in epoch 0
- RAM leak mitigated: gc.collect + empty_cache every 50 batches, mid-epoch checkpoints every 1000 batches
- Checkpoint: `data/models/bc/v8_bc_human_v3/mid_epoch2_batch1000.pt` (best eval performance)

**Human BC Eval Results (100 games each, epoch 2 batch 1000 checkpoint):**
| Opponent | v8 human BC | v8 bot BC | v7 BC | v7 PPO |
|----------|-------------|-----------|-------|--------|
| Random | 98% | 98% | 94% | -- |
| MaxBasePower | 44% | 46% | 60% | -- |
| SimpleHeuristics | **89%** | 12% | 20% | 23% |
| SmartDamage | **27%** | 0% | 26% | 27% |
| Tactical | 5% | 15% | 18% | 22% |
| Strategic | **84%** | 17% | 12% | 20% |
| **Smart avg** | **51.2%** | 11% | 19% | 23% |

- **51.2% smart avg** — more than double v7's best ever (23% PPO self-play)
- **89% vs SimpleHeuristics** — entity attention mastered type relationships
- **84% vs Strategic** — human data taught long-term planning (hazards, momentum, safe switches)
- **5% vs Tactical** — hard-coded damage calc still dominates; needs self-play to overcome
- **27% vs SmartDamage** — matches v7 PPO peak; entity attention helps but not enough for raw damage optimization
- Clear progression: bot_bc(11%) -> human_ep0(16%) -> human_ep1(22%) -> human_ep2(51%)

**Bugs Found & Fixed:**
- LR scheduler bug: warmup lambda stepped per-epoch not per-batch, giving ~0 LR during epoch 0
- RAM leak: collated batch tensors + memmap page cache not freed between batches. Fixed with explicit del + gc.collect + torch.cuda.empty_cache every 50 batches
- VRAM creep: PyTorch CUDA allocator fragmentation from variable-size batches. Fixed with torch.cuda.empty_cache every 50 batches. Peak VRAM stable at ~5960/6144 MiB.
- Mid-epoch checkpointing added (every 1000 batches) to prevent losing hours of training to crashes

**Playstyle Analysis (v8 BC progression, vs SH):**
| Metric | ep0 (0% WR) | ep1 (17%) | ep2b1k (97%) |
|--------|-------------|-----------|-------------|
| Attack % | 36% | 43% | **60%** |
| Switch % | 22% | 24% | **13%** |
| Recovery % | 16% | 15% | **8%** |
| Hazards % | 8% | 4% | **8%** |
| KO ratio | 0.39 | 0.64 | **1.48** |
| Top move | Seismic Toss | Roost | **Hurricane** |
| Style | Passive stall | Timid offense | **Aggressive STAB + hazards** |

- ep0: Imitates defensive human play. Seismic Toss/Soft-Boiled/Protect. Loses 2.5 mons per kill.
- ep1: Shifts to offense (Dark Pulse, Discharge) but still too much switching (24%). KO ratio improving.
- ep2b1k: **Competent aggressive player.** Hurricane/Surf/Earthquake (strong STAB coverage), Stealth Rock for chip, low switching. Kills 1.48 mons per loss. Plays like a strong human vs heuristic bots.

**Playstyle vs Strategic (ep2b1k, 10% WR):** Tries Nasty Plot setup sweeping, but Strategic has phazing/Haze to counter. Model needs to learn opponent-specific adaptation — will come from self-play PPO.

**Key Finding — Accuracy Is Not Win Rate:**
Val accuracy plateaued at 43-44% across epochs, but win rate jumped massively (ep1 22.5% → ep2b1k 51.2%). The model learned WHICH decisions to get right (attacking with coverage moves, committing to plays) even though overall prediction accuracy barely changed. BC val accuracy is a poor proxy for battle strength.

**Key Finding — No LR Spike At Epoch Boundaries:**
The apparent accuracy jump at epoch starts is a REPORTING ARTIFACT. The training loop reports a running average from batch 1. End-of-epoch average is dragged down by early batches (when model was weaker). Start-of-next-epoch resets the running average, showing the model's current ability.
Verified: LR is smooth cosine decay, no resets. LR at ep2 start = 0.000075 (25% of peak), no spike.

**H2H Tournament (100 games per matchup):**
| v8_human_best vs: | Result |
|-------------------|--------|
| v8_bot_bc | 62-38 (62%) |
| v8_human_ep0 | 87-13 (87%) |
| v8_human_ep1 | 62-38 (62%) |
| v7_lstm_bc | 56-44 (56%) |
| v7_trans_bc | 98-2 (98%) |
| v7_ppo_best | 56-44 (56%) |
| v7_iql_ep10 | 30-70 (30%) |

Round-robin (key models, avg win %): v7_iql_ep10 (65%) > v7_lstm_bc (49%) > v7_ppo (48%) > v8_human (38%).
v8 dominates bots (51.2% smart avg) but loses H2H to v7 RL models that learned opponent exploitation through self-play. BC teaches general strategy; RL teaches adversarial adaptation. v8's much stronger BC base (51% vs 19%) should produce much stronger self-play PPO.

**Eval Variance Discovery:**
50-game evals with random teams are unreliable (same checkpoint: 51% in one eval, 20% in another).
Root cause: random_teambuilder() picks from 70 teams — some hard-counter specific bots.
Fix: bc_train_v8.py now runs 200 games × 4 bots automatically after each epoch. best.pt saved by smart_avg.

**BC Hyperparameter Tuning (Metamon research):**
Switched from 3e-4 cosine to Metamon-validated config: lr=1e-4 constant, weight_decay=1e-4, grad_clip=2.0.
Both Metamon (human data, top 10%) and ps-ppo (bot data, >1900 Elo) use lr=1e-4.
Constant LR is better for open-ended training where epoch count is unknown.
Active training: `data/models/bc/v8_bc_human_metamon/` with auto bot eval.

**Phase D Ready — rl_train_v8.py (574 lines):**
Complete PPO self-play training loop for v8 PokeTransformer. Includes:
- V8RLPlayer with make_v8_features(), temporal history, CPU trajectory storage
- ppo_update_v8() with distributional value loss (two-hot CE), KL penalty
- Full self-play infrastructure: snapshot pool, HoF, historical sampling
- Memory management: reset_battles, PSClient cancel, gc.collect, empty_cache
- Bot eval via bc_train_v8.eval_vs_bots() with replays
- Hyperparams: gamma=0.9999, lam=0.8, ent=0.02, lr=1e-4, kl=0.05 (research-validated)
- Command: `python -u rl_train_v8.py --init-from best.pt --device cuda --servers 9000 --self-play`

**Disk Cleanup:** Deleted 35 old mid-epoch checkpoints (5.3 GB freed). 213 GB free.

**Optimizer Mismatch Discovery (Metamon-config run):**
Resumed from SAVE_ep2b1k_51pct.pt (trained at 3e-4 cosine, weight_decay=1e-2) with new config
(1e-4 constant, weight_decay=1e-4) but loaded the OLD optimizer state. Adam momentum terms
from 3e-4 training fought the 1e-4 config — accuracy DECLINED from 42% to 39.9% over epoch 2.
Bot eval: 18.6% smart avg (down from 51.2%). Epoch 3 partially recovered to 28.5%.
Playstyle analysis: model degenerated into U-turn spam (26-30% pivot rate) in epoch 2,
then recovered to balanced attacker in epoch 3 (Magma Storm + Taunt vs Strategic = clever).

**Lesson: ALWAYS use --lr-restart when changing LR or weight_decay on resume.**
Fresh optimizer resets Adam momentum. Weights stay. Added --lr-restart flag to bc_train_v8.py.

**Fresh Optimizer Run (current, v8_bc_human_fresh_opt):**
Resumed from SAVE_ep2b1k_51pct.pt with --lr-restart (fresh Adam, lr=1e-4 constant, wd=1e-4).
Accuracy: started at 44.0% (correct — matches checkpoint), stable at 43.9% through 8K batches.
Loss: 1.387 → 1.357 (slowly decreasing, lower than original run at same point).
No decline — fresh optimizer holds the good minimum. Epoch completion + 800-game bot eval pending.

**Metamon-config Run Playstyle Analysis (auto bot eval, 200 games each):**
| | Epoch 2 (18.6%) | Epoch 3 (28.5%) |
|---|---|---|
| vs SH | 1% — U-turn spam, KO 0.51 | 24% — Salt Cure, KO 0.75 |
| vs SmartDmg | 1% — U-turn spam, KO 0.49 | 36% — Fire Blast/Tbolt/EQ coverage |
| vs Tactical | 18% — Shadow Ball + U-turn | 5% — Phantom Force (poor choice) |
| vs Strategic | 54% — Calm Mind + Surf setup | 50% — Magma Storm + Taunt (clever trap) |

Epoch 2 was U-turn degeneration from stale optimizer momentum. Epoch 3 showed recovery with
new strategies (trapping + anti-stall) but lost the original SH-crushing STAB aggression.

**PPO Update Optimization (rl_train_v8.py):**
Applied same batched-spatial optimization from bc_train_v8.py to ppo_update_v8():
- Batch all T turns' spatial processing in one GPU call (the expensive part)
- Sequential temporal only over lightweight 384-dim summaries (cheap)
- KL reference model also uses batched spatial
- Estimated ~30x speedup for PPO update step
- Battle collection unchanged (bottleneck is Showdown sim, not model)
- For faster collection: use --servers 9000,9001 (2 battle servers = ~5.4 g/s vs 4.7 g/s)

**Direct Mode (--direct) Not Recommended for PPO:**
Single battle_worker.js subprocess serializes I/O — 1.5 g/s regardless of concurrency.
battle_server.js with 2 instances = 5.4 g/s. Not worth fixing direct mode when server is better.

**Checkpoint Naming Fix:**
Mid-epoch checkpoints now include correct global step and epoch:
`mid_step37988_epoch2_batch1000.pt` instead of wrong `mid_step35988_epoch0_batch1000.pt`.
Global step comes from inside train_one_epoch, epoch uses epoch_offset from resume.

**Fresh Optimizer Run Results (v8_bc_human_fresh_opt):**
| Epoch | Train Loss | Val Loss | Val Acc | SH | SmDmg | Tact | Strat | Smart Avg |
|-------|-----------|---------|---------|-----|-------|------|-------|-----------|
| 2 (first) | 1.3552 | 1.3501 | 44.2% | 39% | 13% | **58%** | 24% | **33.6%** |
| 3 (second) | 1.3523 | 1.3469 | 44.5% | 30% | 8% | 12% | 10% | **15.0%** |

Critical finding: Val loss IMPROVED (1.3501->1.3469) while win rate COLLAPSED (33.6%->15.0%).
Model became more accurate at predicting human actions (conservative play) while losing battle
effectiveness. Recovery moves exploded (11%->20% vs SH). KO ratios dropped across all matchups.

This is "BC overfitting to the mean" — constant LR keeps pushing toward average human behavior
(defensive, recovery-heavy) rather than winning behavior (aggressive, decisive). Extended BC
training actively HURTS battle performance after the sweet spot.

**BC Sweet Spot Discovery (in progress):**
Running 200-game eval sweep across 7 mid-epoch checkpoints (ep2 batch 5K/10K/15K/end + ep3
batch 5K/10K/end) to find exactly where battle performance peaks. Using 2 battle servers at
conc=15 for faster eval. Results will determine the best BC checkpoint for PPO init.

**PPO Training (Session 28 continued):**
- First run (iter 1-60): KL drifted to 0.33, bot eval dropped 28%→7%. Fixed with:
  - KL early stopping (ps-ppo style, target_kl=0.02) replaces KL penalty
  - Value warmup phase (5 iters freeze backbone+policy, train value head only)
  - Lambda 0.75 (from ps-ppo, was 0.8)
  - Opponent on CPU (parallel inference)
- Second run with fixes: KL stable at 0.017 during warmup, 0.030 post-warmup. Value head
  calibrated (16→2.1 in 2 iters). Win rate 20-27% and stable.
- Entity encoding BATCHED: 12 pokemon processed in ONE PokemonNet call (was 12 separate).
  Forward pass 46ms→9.2ms (5x speedup). Collection 480s→133s (3.6x faster, verified).
- **CRITICAL BUG FOUND:** eval_vs_bots used ConstantTeambuilder (one fixed team per 200-game eval).
  ALL previous eval results were team-luck artifacts, not model quality measurements.
  The "51% smart avg" BC, "43% PPO iter 20", playstyle evolution — all unreliable.
  FIXED: changed to random_pool_teambuilder (per-battle random teams).
- **TRUE results (first reliable eval ever, 200 games with random teams per battle):**
  BC init: 20.5% | PPO iter 20: 24.6% | PPO iter 40: 25.1% smart avg.
  PPO is improving steadily (+5% over BC in 40 iters). No wild specialization.
- **Pipeline audit:** observer.py also had ConstantTeambuilder — fixed to random_pool.
  Training (rl_train_v8.py) was correct (random_pool). Pruned: batched_player_v8.py, backup.
- **TRUE BC re-eval (all checkpoints, 200 games with random teams):**
  orig_ep0=20.0% | orig_ep1=22.1% | **orig_ep2b1k=24.6% (REAL BEST)** | fresh_ep2=21.5% | BC_BEST(ep3_b5k)=19.0%
  BC was steadily improving (20→22→24.6%), NOT oscillating. Our ep3_b5k selection was wrong.
  orig_ep2b1k should have been the PPO init checkpoint. (Though later confirmed statistically identical to BC_BEST.)

**PPO Evolution (TRUE eval, 200 games × 4 bots, random teams per battle):**
| Checkpoint | SH | SmDmg | Tac | Strat | Smart Avg | 95% CI |
|-----------|-----|-------|-----|-------|-----------|--------|
| BC init | 18% | 23% | 25% | 26% | 22.8% | 20.0-25.8% |
| PPO iter10 | 27% | 25% | 25% | 26% | 25.6% | 22.7-28.8% |
| PPO iter20 | 30% | 22% | 25% | 24% | 25.0% | 22.1-28.1% |
| **PPO iter30** | 24% | 27% | **28%** | **27%** | **26.0%** | **23.1-29.1%** |
| PPO iter40 | 24% | 23% | 24% | 20% | 22.8% | 20.0-25.8% |

Peak at iter 30 (26.0%). Most balanced — all bots 24-28%. After iter 30, model starts declining
(potential oscillation, as seen in v7). iter 30 saved as `BEST_PPO_iter30_26pct.pt`.
PPO playstyle: 60-66% attack (vs BC's 42-44%), shorter games (22-25 vs 32 turns), better KO
ratios (0.70-0.77 vs 0.61-0.68). Learned aggression + recovery timing.
Type awareness: SE improved 6%→9%, but immune also rose 6%→10% and resisted 14%→19%.

**eval_report_v8.py built:** comprehensive eval with CI, playstyle, per-mon, team performance.
**If oscillation continues:** research shows pure self-play (no bots) works — ps-ppo >1900 Elo
with just model-vs-self. See docs/V8_PPO_TODO.md item #10.

**PPO iter 60 eval: smart_avg=24%** (SH=26%, SmD=25%, Tac=23%, Str=20%).
Partial recovery from iter 40 dip (22.8%). Not back to iter 30 peak (26.0%).
Training oscillating in 22-26% range — possible plateau.

**PPO iter 60 playstyle (TRUE eval, 200 games, random teams):**
- Attack moderated: 61-66% (was 72% at iter 40). Healthier aggression.
- Immune rate IMPROVED: 7-8% (was 9-11%). Genuine type learning over time.
- Hazards returned: 3-5% with Spikes as top move. Rediscovered hazard value.
- Switching increased: 23-24% (was 19-21%). Learning when brute force won't work.
- Consistent across all bots: same strategy vs everyone. No per-opponent adaptation.
- Top moves: Earthquake, Shadow Ball, Moonblast, Hurricane, Hydro Pump (5-type coverage).
- New mon: Ting-Lu appearing (Ground/Dark bulky tank).
- KO ratio: 0.71-0.77 (same as iter 30 peak).

**Full PPO evolution (all reliable, TRUE evals):**
| Iter | Smart Avg | Atk% | Imm% | KO | Recovery | Style |
|------|-----------|------|------|-----|----------|-------|
| BC | 22.8% | 43% | 6-8% | 0.64 | 13% | Balanced/defensive |
| 10 | 25.6% | 60% | 8-11% | 0.70 | 4% | Aggressive shift |
| 20 | 25.0% | 65% | 9-11% | 0.74 | 9% | Peak aggression + recovery |
| **30** | **26.0%** | **64%** | **10%** | **0.72** | **10%** | **Best balanced** |
| 40 | 22.8% | 72% | 9% | 0.76 | 3% | Over-aggressive dip |
| 60 | 24.0% | 63% | **7%** | 0.74 | 3% | Recovering, better types |

Key trend: immune rate improving (8%→10%→7%), attack moderating (72%→63%),
hazards returning. Model maturing toward balanced play.

**Potential plateau at 22-26%.** If persists, consider:
1. Pure self-play (drop bots) — ps-ppo reached >1900 this way
2. More games per iter (currently 200, ps-ppo uses ~800 equivalent)
3. Temperature randomization (Metamon uses [1.0, 2.25] range)
- Resumed from iter 20 with: target_kl=0.03, ppo_epochs=5, opponent on GPU, batched entities.
  Total iter time: ~5.5 min (was 10 min). 200-game eval with replays every 20 iters.
- Memmaps compressed: human_v8 108GB→~2-3GB, bot_v8 7.8GB→75MB.
- Research findings: ps-ppo uses 3-phase (BC→value warmup→PPO) with optimizer reset at each.
  No KL penalty — only KL early stopping. Metamon uses FBC filter + continuous offline data.
  See docs/V8_PPO_TODO.md for full research + implementation priorities.
- Best BC checkpoint: `data/models/bc/BEST_v8_bc_step58982_aggressive_priority.pt`
- Backup: `rl_train_v8_backup_pre_batch.py`

**Key Architectural Findings:**
- Entity tokenization works: immune rate dropped from v7's 29-40% to 0-9%
- NumericalBanks: all continuous values (HP, stats, BP, accuracy, PP, priority, turn, weight,
  height, durations) embedded through learned lookups — 13 banks total
- Temporal context: 200-turn summaries via learned attention pooling (model decides what to
  compress per turn — emergent, not hand-designed)
- Transition token: 57 features including 6-way effectiveness (immune/barely/NVE/neutral/SE/ultra)
  gives direct per-turn feedback for type learning
- Move features: explicit type one-hot (19d) + base power bank + category + STAB flag
- Pokemon features: explicit type multi-hot (19d) + base stat banks (6x256 bins)
- The model has ALL ingredients for type/damage reasoning; architecture enables entity-level
  attention between move types and pokemon types. Learning is confirmed but incomplete —
  avoids immune moves (good) but doesn't fully exploit SE targeting yet (needs PPO).

**BC Checkpoint Sweep (7 checkpoints, 100 games/bot, fresh-opt run):**
| Checkpoint | Step | SH | SmDmg | Tact | Strat | Avg | Style |
|-----------|------|-----|-------|------|-------|-----|-------|
| ep2_b5k | 40988 | 0% | 8% | 5% | 33% | 11.5% | Setup/stall |
| ep2_b10k | 45988 | 0% | 17% | 6% | 17% | 10.0% | Aggressive but type-blind (15% imm) |
| ep2_b15k | 50988 | 14% | 27% | 2% | 30% | 18.2% | Hazard setter |
| ep2_end | 53982 | 0% | 7% | 52% | 15% | 18.5% | Wish/Protect staller (81 turns vs Tact) |
| **ep3_b5k** | **58982** | **28%** | **56%** | **28%** | 2% | **28.5%** | **Aggressive + priority moves** |
| ep3_b10k | 63982 | 8% | 52% | 26% | 36% | 22.0% | Balanced/pivoting |
| ep3_end | 71976 | 22% | 52% | 12% | 2% | 16.2% | Regressed defensive |

Best: **ep3_b5k (step 58982)** — discovered priority moves (Aqua Jet, Bullet Punch, Mach Punch),
71% attack rate vs SH, 0% immune vs SmDmg/Strategic, 56% vs SmDmg. Closest to competitive play.
Saved as: `data/models/bc/BEST_v8_bc_step58982_aggressive_priority.pt`

Key finding: BC playstyle OSCILLATES across training — setup→attack→stall→aggressive→balanced→passive.
No monotonic improvement. The sweet spot is wherever the model happens to be in the aggressive phase.
Extended training always regresses toward passive/recovery play (average human behavior).

**PPO Plan (Phase D):**
- Init from: BEST_v8_bc_step58982_aggressive_priority.pt
- Command: `python -u rl_train_v8.py --init-from data/models/bc/BEST_v8_bc_step58982_aggressive_priority.pt --device cuda --servers 9000,9001 --self-play --games-per-iter 100 --max-concurrent 10 --n-iters 500`
- Opponent distribution: ~24% self-play pool, ~24% smart bots, ~9% easy bots, ~3% BC anchor
- Hyperparams: gamma=0.9999, lam=0.8, ent=0.02, lr=1e-4, kl=0.05 (research-validated)
- From v7 lessons: NO curriculum, NO adaptive weights, YES uniform historical sampling, YES KL penalty
- Target: 40%+ smart avg (v7 PPO peaked at 23% from a 19% BC base; v8 starts from 28%)

Full spec: docs/V8_PLAN.md. Commands: docs/V8_COMMANDS.md.

### Session 27 — Full Summary

**BC Transformer (DONE)**
- 10 epochs on human_v3_memmap (10.1M records, 397K episodes). lr=1e-4 + warmup. Val acc 41.3%.
- Checkpoint: `bc/v7_bc_transformer_lr1e4/best.pt`. Smart avg: ~20% (same ceiling as LSTM BC).
**IQL (ABANDONED)**
- 3 runs (beta=10 exponential, beta=5 exponential, binary filtering) all failed.
- Q-network learns state values but not action-level values from offline data. Advantage signal (±0.10) too weak.
- Binary advantage filtering implemented (--binary-advantage flag) but did not help.

**Self-Play PPO (DONE — plateaued at 20-23%)**
- Population-based self-play with snapshot pool + hall of fame + historical sampling + bots + BC.
- Evolved through multiple iterations of infrastructure improvements:
  - v1 (iter 1-100): 50 games/iter, 5 recent snapshots. Smart avg 10%→26%. Broke fixed-bot ceiling.
  - v2 (iter 100-260): 100 games/iter, conc=10, opponent on GPU. Discovered strategy oscillation.
  - v3 (iter 260-320): Uniform historical sampling. Dampened oscillation but ceiling held.
  - v4 (iter 320-380+): 200 games/iter, conc=15, BC weight 0.5. Stable at 20-23%.
- **Strategy oscillation**: ~100 iter cycle: aggressive → stalling (Roost spam) → pivoting (U-turn spam) → aggressive. Root cause: recent snapshots converge to same style. Partially fixed by uniform historical sampling.
- **H2H**: PPO best (iter 80) = 57.2% overall, 2nd strongest model ever behind LSTM BC (58.8%). But later iters degraded — iter 380 only 45.7%.
- **Playstyle evolution**: BC→PPO learned hazards (Clodsire SR), pivoting, recovery, setup. More strategic than BC but couldn't sustain it — cycled between styles.
- **Type blindness**: 29-40% immune hits throughout. Self-play can't teach type chart because neither side punishes it.

**Key Infrastructure Built**
- Snapshot pool (FIFO 2 recent) + Hall of Fame (top 3 by weighted eval) + uniform historical sampling
- HoF selection by weighted avg (smart bots 1.0, easy bots 0.3)
- Lineage tracking across run directories (history_dirs in checkpoint)
- Configurable BC weight, opponent device, concurrency
- Pool/HoF/lineage persistence in checkpoints for clean resume
- Resume safety: warning if --init-from missing, dead path filtering
- Opponent distribution logging per iter

**Critical Findings**
- KL penalty + BC opponent are CRITICAL — model collapsed in <10 iters without them
- No curriculum/adaptive weights — causes oscillation (beat A → upweight B → forget A)
- IQL is dead end for our setup — skip to self-play PPO from BC
- Self-play plateau at 20-23% is ARCHITECTURAL, not training-related
- v7 source code backed up to `backups/v7_source_backup/`

**Architecture Gaps Identified (via comparison with ps-ppo >1900 Elo, Metamon top 10%)**
1. **Entity tokenization**: Our flat vector → MLP → transformer crushes entity relationships. ps-ppo processes each Pokemon as a separate token. This is the #1 gap.
2. **Previous action + outcome**: Both Metamon and ps-ppo include what happened last turn (move used, SE/immune/crit flags). We don't — model can't learn cause-effect.
3. **Distributional value head**: Both successful approaches use classification bins. We use scalar regression — weaker gradient signal.
4. **Structured attention (Poke-Mask)**: ps-ppo separates state from policy/value attention. We don't.
5. **None of the successful approaches inject type effectiveness** — they learn it from entity embeddings + entity-level attention. The architecture enables this, not the features.

**Data**: human_v3_memmap compressed to 2.2 GB tar.gz. combined_v6_memmap.tar.gz (1.6 GB). 322 GB free.
**v7 backup**: `backups/v7_source_backup/` — all source (26 files), docs (6), logs (16), model checkpoints (BC best + PPO best + PPO latest). 716 MB.

### Next: v8 Architecture Overhaul
Full plan in `docs/V8_PLAN.md`. Four phases:
1. **Transition features** — previous action + SE/immune/crit outcome flags (~15 new features)
2. **Entity tokenization** — PokeTransformer with per-Pokemon tokens, sub-networks, Poke-Mask (major refactor)
3. **Distributional value head** — 51-bin categorical with two-hot encoding
4. **Temporal attention** — frame stacking for cross-turn strategy (OU-specific, beyond ps-ppo)

Key insight: none of the successful approaches inject type effectiveness — they learn it from entity-level attention. The architecture enables learning, not the features.

Session 26 (continued): LSTM IQL training completed through epoch 21 + comprehensive eval + transformer architecture implemented.
- **LSTM IQL results (v6_iql_combined_bs32)**: Trained 21 epochs on combined data. Bot WR flat at 23-28% smart avg. But h2h shows clear improvement: ep5 53.2% → ep10 57.2% → ep16 58.0% → ep20 57.3%. **EP10 is best h2h model (58.3%).**
- **100-game bot evals**: EP5 28.3%, EP10 24.5%, EP16 27.0%, EP20 23.8% smart avg. Noisy but centered ~25%.
- **7-way H2H tournament** (100 games per matchup): ep10 58.3% > ep20 57.3% > ep16 55.5% > BC 51.8% > ep05 50.7% > PPO 48.2% > iql_human 28.2%.
- **Playstyle evolution**: EP5→EP20 learns recovery (5.5%), pivoting (3.5%), status (WoW 13%), hazards (SR 8%). Later epochs drop status/recovery, become more positional (hazards). Corviknight Brave Bird spam persists across all (86-97%).
- **Plateau analysis**: All approaches (BC, IQL, PPO) converge to 25-30% vs smart bots regardless of training method. Root cause: model capacity (3.85M LSTM) can't learn Pokemon mechanics (type charts, damage calc, Choice-lock tracking, burn effects). Smart bots hard-code this knowledge.
- **Transformer implemented** (v7): CausalTransformerCore added to policy_heads.py. 6 layers, 4 heads, 512 dim = 20.78M params. Fits in 0.92 GB VRAM for IQL (3 models + backprop). New default. LSTM still available via --use-lstm flag. All files backward compatible with old checkpoints.
- **Files changed**: policy_heads.py (CausalTransformerCore + use_transformer config), bc_train.py (--use-transformer flag), iql_train.py (make_model reads transformer config), bc_policy_player.py (transformer inference with history buffer).
- **Memory leak fix**: PSClient._listening_coroutine.cancel() added to observer.py, rl_train.py. Prevents zombie websocket listeners accumulating over thousands of games.
- **Self-play support**: observer.py now accepts --model name:path to register model checkpoints as bots for self-play data generation.
- **Data**: combined_v6_memmap (7.6M records, 277K episodes, 63 GB). Source JSONLs deleted. 252 GB free.
- **Research survey**: docs/RESEARCH.md — Metamon (transformers + self-play = top 10%), MIT thesis (PPO+MCTS), PokeChamp (LLM minimax). Self-play is biggest gap vs SOTA.
- **Next**: Train IQL with transformer on combined_v6_memmap, then self-play PPO.
- **Workers**: --workers 2 is OK if bloatware killed (Adobe, Overwolf, Norton UI). --workers 0 is safe fallback (~2.5x slower). --workers 1 is middle ground.
- **Node 20 path**: `tools/node-v20.18.1-win-x64/node.exe` (project root, NOT src/tools/)

Session 25 (v6): Deep codebase audit across all source files → 40 bugs fixed (11 HIGH, 18 MEDIUM, 11 LOW). Human replay data acquired: gen9ou_rating1500.jsonl (30 GB, 2.54M records, 99,955 episodes) → converted to memmap at data/datasets/human_memmap/. Key HIGH fixes: (1) IQL terminal reward 0 for losses not -1 — Q-function couldn't learn to avoid losing, (2) PPO missing final dense reward — last turn KO/HP delta lost, (3) features.py recoil always 0.0 — model blind to Flare Blitz/Brave Bird, (4) bc_train.py modifier_bce leak from masked positions (log(2) per slot), (5) battle_server.js finished battles never cleaned up (memory leak), (6) battle_worker.js no chunk coalescing — |request| and |switch| sent separately breaking poke-env, (7) direct_player.py maybe_restart kills worker mid-battle. Infrastructure: memory leak fixes (bounded _processed_wins, DirectClient.unregister_battle, battle timeouts), rewards.py hp_frac now assumes unseen mons at 100%, bot fixes (setup at +6, accuracy, SR ignores Boots, opp remaining count). All 30 fixes verified with import checks + logic tests + model forward pass.

Session 24: Deep codebase audit (27+ bugs fixed) + reward unification + battle_server.js + PPO to iter 409 + comprehensive eval. Infrastructure: battle_server.js (minimal Showdown, 5.4 g/s, 31 MB, replaces Docker) + direct_player.py (--direct flag, no server). Bugfixes: PPO temperature, rewards double-counting, features.py×5, bc_train.py×3, memory leak (reset_battles+gc.collect). PPO eval (6,000 battles): **iter 350 best** (27% smart avg, SH 30%, Strategic 30%). IQL wins h2h 58% vs RL (reads opponents better). RL learns bot-specific attack strategies, IQL has broader adaptation. Memory leak: PSClient.listen tasks accumulate on POKE_LOOP — needs cancel on cleanup.

Session 19: v5 data generation complete (3.9M records, 143K episodes, 34.7 GB memmap). A-W-BC training started.

Session 20: A-W-BC training converged (epoch ~16). Found and fixed critical LSTM concurrency bug in bc_policy_player.py — single shared hidden state across concurrent battles effectively disabled LSTM at eval. Fix: per-battle hidden state dict. +8% win rate. Action class weighting experiment failed (made things worse). BC ceiling reached at ~20-26% vs smart bots. Ready for IQL.

Session 21: Found and fixed observer.py batch counter bug — `_wins_before` not reset between games, corrupting 38% of episodes (losses mislabeled as wins). Fixed memmap result labels post-hoc using alive_counts + hp_adv from obs features (88% wins -> 50/50 split). Deleted 48GB JSONL source files (memmap is sufficient). IQL v2 trained with fixed data but showed no improvement over BC — advantage weights all ~1.0 due to sparse terminal-only reward. Full audit identified 3 bugs: (1) LR scheduler stepped per-epoch not per-step, (2) sparse reward gave zero intermediate signal, (3) V-network value_head started random. IQL v3 implements dense reward shaping from obs features (KO bonuses + HP deltas), higher beta/tau, fixed scheduler, --resume support.

Session 23: Comprehensive 13-model evaluation (BC + 12 IQL checkpoints, 4,680 h2h games). Confirmed offline RL ceiling — all models Elo 1471-1473 vs bots, h2h compressed (46.9%-54.2%). Bot round-robin established hierarchy: SmartDamage (79%) > Tactical (78%) > Strategic (73%) > SH (70%) >> mid-tier (~39%) >> Random (4%). Full audit of rl_train.py found and fixed 9 bugs total: (1) make_obs_mask_and_slots unpacking only 5 of 8 return values, (2) undefined rl_player reference, (3) entity_ids/move_ids/switch_ids not captured or passed to model, (4) save_rl_checkpoint missing v5 config fields, (5) torch.load missing weights_only=False, (6) default obs_dim=2442 (v4) instead of 1480 (v5), (7) ref_model KL forward missing entity_ids, (8) LR scheduler state not saved/restored on resume, (9) optimizer LR not synced from scheduler on resume. Also: added HazardSense+SwitchAwareEscape to OPPONENT_BOTS, multi-server support (--servers flag), bumped defaults to max_concurrent=15 and games_per_iter=50. All verified with 51-point smoke test. Online RL (PPO) with IQL ep22 init is next.

Session 22: IQL v3 completed 30 epochs. Comprehensive evaluation: 1,620 bot games + 450 head-to-head games across 6 checkpoints. Key findings: (1) Val pi-loss is **anti-correlated** with actual play strength — ep7 (best val loss) was worst in h2h (45.3%), ep30 (worst val loss trend) was best (61.3%). (2) IQL ep30 surpasses BC in h2h (61.3% vs 48.7%) but bot win rates are similar — suggesting IQL learns better adaptation/reads rather than raw move quality. (3) Playstyle evolves: BC→ep7 (more pivoting) → ep18 (Dragon Dance degeneration) → ep22-30 (balanced, highest switch rates). (4) Decision Transformers rejected — can only stitch training data, not discover new strategies. IQL v4 implemented: LR warm restart, eval-based checkpointing (4 smart bots × 20 games), keep all checkpoints, policy-only saves every epoch.

---

## What Exists

### Code (pokemon-ai-starter/pokemon-ai/src/)
| File | Status | Notes |
|------|--------|-------|
| observer.py | v5 FIXED | Emits entity_ids (84), move_ids (4), switch_ids (5); ties=0.5; **batch counter bug fixed** (Session 21) |
| iql_train.py | v4 UPDATED | IQL trainer: dense reward shaping, per-batch LR schedule, --resume, --lr-restart, eval-based checkpointing, keep all checkpoints |
| eval_head_to_head.py | NEW | Round-robin head-to-head tournament between model checkpoints |
| analyze_eval.py | REWRITTEN | Playstyle analysis from replays: move categories, switch rates, KO ratios, comparison tables |
| fix_memmap_results.py | DONE | Post-hoc fix for corrupted result labels using obs alive_counts + hp_adv |
| features.py | v5 REWRITTEN | 1480-dim continuous obs + 84 entity IDs + 107-dim move slots + 28-dim switch slots |
| policy_heads.py | v5 UPDATED | nn.Embedding layers (species/move/item/ability), 19 sum-pooled groups, ~3.85M params |
| bc_train.py | v5 UPDATED | Entity IDs flow end-to-end, advantage-weighted BC, auto-detects v5 dims |
| convert_jsonl_to_memmap.py | v5 UPDATED | entity_ids.npy, move_ids.npy, switch_ids.npy arrays added |
| bc_policy_player.py | FIXED | Fixed ctx_extra_dim/step_type_bins override bug |
| eval_bc_vs_bots.py | FIXED | Fixed --use-lstm default (was False, now None → respects checkpoint) |
| eval_bots_roundrobin.py | DONE | Bot-vs-bot round-robin evaluation |
| policy_rulebots.py | DONE | 4 basic bots: GreedySE, HazardSense, SwitchAwareEscape, SetupThenSweep |
| policy_smartbots.py | DONE | 3 advanced bots extending SimpleHeuristics |
| policy_random.py | DONE | Returns BattleOrder directly |
| rl_train.py | v5 READY | PPO trainer: 9 bugs fixed, multi-server support, LR scheduler persistence, all 9 bots registered |
| rewards.py | OK | Clean reward shaping implementation (for RL phase) |
| env_wrapper.py | DONE | Legacy env wrapper, not used by rl_train.py |
| teams_ou.py | DONE | 30 teams, all Gen 9 OU legal, diverse archetypes |

### Data
| Data | Size | Status | Notes |
|------|------|--------|-------|
| Human replay JSONL (gen9ou 1500+) | 2.54M records, 30 GB | SOURCE | src/data/datasets/human_replays/gen9ou_rating1500.jsonl |
| Human replay memmap (v6) | ~20 GB, 99,955 episodes | CURRENT | src/data/datasets/human_memmap/, v5-compatible dims (1480+84) |
| JSONL observations (v5, 1480+84) | 3.94M records, 108 files | DELETED | Was 48GB, deleted after memmap conversion confirmed |
| Memmap dataset (v5, bot data) | 34.7 GB, 143,212 episodes | CURRENT | Memory-mapped numpy, src/data/datasets/memmap/ |
| JSONL observations (2442-dim, v4) | 3.72M records, 198 files | SUPERSEDED | 10 bots, 2442-dim obs |
| Memmap dataset (2442-dim, v4) | 52.61 GB, 132,900 episodes | SUPERSEDED | Memory-mapped numpy |
| JSONL observations (2394-dim) | 540k records, 130 files | SUPERSEDED | backed up to obs_v3_backup/ |
| Memmap dataset (2394-dim) | 7.53 GB, 18682 episodes | SUPERSEDED | backed up to memmap_v3_backup/ |
| JSONL observations (988-dim) | 473k records, 217 files | SUPERSEDED | 10 bots, 988-dim, concatenated as obs_all_988dim.jsonl |
| v5 A-W-BC checkpoint | best.pt (epoch ~16) | CURRENT | v5_awbc-2026-03-15_18-38-23, 512 hidden, LSTM, entity embeddings |
| BC flat checkpoint (v3) | epoch_041 best.pt | SUPERSEDED | 988-dim, 85.9% val accuracy |
| BC hierarchical checkpoint | epoch_060 best.pt | SUPERSEDED | Hierarchical head causes move-bias |
| Bot round-robin | results.csv | CURRENT | 10 bots x 50 battles x 45 matchups |

### Infrastructure
| Component | Status | Notes |
|-----------|--------|-------|
| Docker Showdown | WORKING | Port 8000, use 127.0.0.1 (not localhost) on Windows |
| GPU Training | WORKING | RTX 3060 Laptop GPU (6GB VRAM) with CUDA 12.1 |
| TensorBoard | CONFIGURED | Port 6006 (may need `pip install --upgrade tensorboard` for duplicate plugin fix) |

---

## BC Training Results (988-dim, Flat Model)

**Run:** `bc_flat_988dim_full` (41 epochs, converged)

| Metric | Epoch 1 | Epoch 41 (best) |
|--------|---------|-----------------|
| Train loss | 16.41 | 12.76 |
| Val loss (raw) | 12.61 | 8.81 |
| Train accuracy | 28.7% | 71.7% |
| Val accuracy | 57.6% | 85.9% |

## BC Evaluation (988-dim, Flat, 50 battles each)

### v3 (slot encoders active, Session 10)
| Opponent | v3 Win Rate | v1 (old) | Change |
|----------|-------------|----------|--------|
| Random | **96%** | 96% | = |
| MaxBasePower | **48%** | 28% | +20% |
| SimpleHeuristics | 6% | 8% | -2% |
| SmartDamage | 12% | 2% | +10% |
| Tactical | 10% | 6% | +4% |
| Strategic | 14% | 0% | +14% |
| **Elo** | **1464** | — | — |

### v5 A-W-BC (1480-dim + embeddings, h512, LSTM fixed, Session 20)
| Opponent | v5 Win Rate | v4 (old, LSTM broken) | Change |
|----------|-------------|----------------------|--------|
| Random | **94%** | 98% | -4% |
| MaxBasePower | **60%** | 54% | +6% |
| SimpleHeuristics | **20%** | 14% | +6% |
| SmartDamage | **26%** | 12% | +14% |
| Tactical | **18%** | 16% | +2% |
| Strategic | **12%** | 14% | -2% |
| **Elo** | **~1477** | ~1470 | +7 |

Note: v4 and earlier evals had LSTM disabled at inference (bug). v5 is the FIRST clean eval. Real improvement is likely larger than the numbers suggest since v4 numbers were artificially close due to both models being crippled.

**Behavioral analysis (Session 20):** One-move spam (Garchomp EQ 89%, Corviknight Brave Bird 100%), low voluntary switch rate (6.4%), mediocre type awareness (SE ratio 31-57%). BC imitates average bot behavior, can't reason about when to switch. Ceiling reached — moving to offline RL.

## IQL v3 Evaluation (Session 22)

### Bot Win Rates (90 games each, all models)
| Model | Random | MaxBP | SH | SmartDmg | Tactical | Strategic | Avg Smart |
|-------|--------|-------|----|----------|----------|-----------|-----------|
| BC | 94% | 60% | 20% | 26% | 18% | 12% | 19.0% |
| IQL ep7 | 96% | 54% | 20% | 16% | 16% | 12% | 16.0% |
| IQL ep16 | 96% | 52% | 18% | 22% | 16% | 18% | 18.5% |
| IQL ep18 | 90% | 46% | 16% | 14% | 18% | 14% | 15.5% |
| IQL ep22 | 90% | 52% | 14% | 20% | 16% | 16% | 16.5% |
| IQL ep30 | 92% | 52% | 16% | 22% | 16% | 16% | 17.5% |

### Head-to-Head Round-Robin (30 games per matchup)
| Model | W | L | Games | WR |
|-------|---|---|-------|----|
| **ep30** | 92 | 58 | 150 | **61.3%** |
| ep22 | 78 | 72 | 150 | 52.0% |
| BC | 73 | 77 | 150 | 48.7% |
| ep16 | 71 | 79 | 150 | 47.3% |
| ep7 | 68 | 82 | 150 | 45.3% |
| ep18 | 68 | 82 | 150 | 45.3% |

**Key insight:** Val pi-loss is anti-correlated with h2h strength. ep7 (best val loss) performs worst; ep30 (latest) performs best. IQL improves at reading opponents, not raw move accuracy — explaining why bot win rates stay flat while h2h strength increases.

### v4 (2442-dim, h512, Session 17) — LSTM BROKEN AT EVAL
| Opponent | v4 Win Rate | v3 (old) | Change |
|----------|-------------|----------|--------|
| Random | **98%** | 96% | +2% |
| MaxBasePower | **54%** | 48% | +6% |
| SimpleHeuristics | **14%** | 6% | +8% |
| SmartDamage | **12%** | 12% | = |
| Tactical | **16%** | 10% | +6% |
| Strategic | **14%** | 14% | = |
| **Elo** | **~1470** | 1464 | +6 |

WARNING: All v4 and earlier evals ran without LSTM (bug in eval script). These numbers reflect a crippled model.

### v1 (old baseline, no slot encoders)
| Opponent | Win Rate | vs Old 978-dim |
|----------|----------|----------------|
| Random | **96%** | 82% |
| MaxBasePower | **28%** | 24% |
| SimpleHeuristics | **8%** | 3% |
| SmartDamage | 2% | - |
| Tactical | 6% | - |
| Strategic | 0% | - |

---

## Critical Bugs Found & Fixed (Sessions 6-7)

### Bug 1: bc_policy_player.py ctx_extra_dim override
- `ctx_extra_dim` and `step_type_bins` defaulted to 0, overriding checkpoint config
- Model ran with wrong input dimensions, obs_encoder weights got pruned
- Fix: Changed defaults to None, only override when explicitly passed

### Bug 2: eval_bc_vs_bots.py --use-lstm default
- `--use-lstm` was `store_true` with default=False, always dropping LSTM
- Model rebuilt WITHOUT LSTM → random MLP instead of trained LSTM
- Fix: Changed default to None, letting checkpoint config take precedence

### Bug 3: Hierarchical action head move bias
- The mvs_head (move-vs-switch) creates a structural bias toward moves
- Move logits always +3-4 higher than switch logits regardless of battle state
- Model never voluntarily switches, leading to 0% vs any smart bot
- Fix: Use flat (non-hierarchical) action head

---

## Bot Rankings (Full Round-Robin: 50 battles x 45 matchups)

| Rank | Bot | Win Rate | Type |
|------|-----|----------|------|
| 1 | **Tactical** | **81.8%** | NEW (extends SimpleHeuristics) |
| 2 | **Strategic** | **81.3%** | NEW (extends SimpleHeuristics) |
| 3 | **SmartDamage** | **78.0%** | NEW (extends SimpleHeuristics) |
| 4 | SimpleHeuristics | 67.8% | poke-env baseline |
| 5 | SetupThenSweep | 44.7% | basic heuristic |
| 6 | GreedySE | 38.4% | basic heuristic |
| 7 | HazardSense | 36.7% | basic heuristic |
| 8 | MaxBasePower | 33.3% | poke-env baseline |
| 9 | SwitchAwareEscape | 32.7% | basic heuristic |
| 10 | Random | 5.3% | uniform random |

---

## Completed TODO

- [x] Fix observer, features, all policy files, env_wrapper
- [x] Add volatile statuses (15) + tera type (20) to features (908 -> 978)
- [x] Expand team pool to 30 teams, all Gen 9 OU legal
- [x] Generate 195k records with 978-dim features + 30 teams
- [x] Train BC + LSTM (50 epochs, CUDA) — 978-dim
- [x] Fix bc_policy_player.py, eval_bc_vs_bots.py bugs
- [x] Evaluate against all 7 bots (100 battles each)
- [x] Audit obs fields — found 27 missing poke-env fields
- [x] Add combat state to obs (978 -> 988 dims)
- [x] Expand move slot vector (34 -> 226 dims with 38 new fields)
- [x] Build 3 smart bots that beat SimpleHeuristics
- [x] Full round-robin evaluation (10 bots, 2250 battles)
- [x] Regenerate data with 988-dim obs + 10 bots (473k records)
- [x] Fix bc_policy_player.py ctx_extra_dim/step_type_bins bug
- [x] Fix eval_bc_vs_bots.py --use-lstm default bug
- [x] Retrain BC flat model (41 epochs, 85.9% val accuracy)
- [x] Evaluate flat vs hierarchical (flat wins)

## Current TODO

### Immediate
- [x] RL training (PPO) using BC flat checkpoint as warm start
- [x] RL reward shaping (rewards.py integrated)
- [x] KL divergence penalty against BC reference (prevents catastrophic forgetting)
- [x] BC model as RL training opponent
- [x] Retrain BC with slot encoders active (bc_flat_988dim_v3_slotenc, 60 epochs, best val_loss=7.57 @ epoch 32, ~89.4% val acc)
- [x] Retrain RL with all Session 9 fixes (RL v1: best 18.3% at iter 100)
- [x] Evaluate RL v1 best checkpoint with larger eval (50+ games per bot)
- [x] Expand obs vector: 988 → 2394 dims (stats, revealed moves, bench items/abilities, alive counts)
- [x] Regenerate training data with 2394-dim obs (540k records, 10 bots, 130 files)
- [x] Convert JSONL → memmap for fast BC training (convert_jsonl_to_memmap.py)
- [x] Add MemmapEpisodeDataset to bc_train.py (lazy memmap open, episode-contiguous, SHA-1 split)
- [x] BC v2 training with 2394-dim memmap data (bc_v2_2394dim_bs16, completed)
- [x] RL v2 training with BC v2 warm start (199 iters, best 28% avg at iter 130)
- [x] Diagnose RL plateau — implemented v3 features (curriculum, dense rewards, self-play)
- [x] RL v3 training attempted 3 times — all failed (oscillates at Tier 0/1, 20-36% avg)
- [x] Implemented v4 computed features (48 dims, obs 2394 → 2442)
- [x] v4 data generation COMPLETE (198 JSONL files, ~5M records)
- [x] Fixed convert_jsonl_to_memmap.py OOM crash (two-pass streaming rewrite + verification)
- [x] Fixed OOM risks: rl_train.py trajectory/episode cleanup, bc_train.py VRAM defrag, bc_policy_player.py -inf masking
- [x] Fixed boost formula bug in features.py (_stat_estimation used > 1 instead of > 0)
- [x] Fixed RewardShaper clone missing ko_bonus/faint_penalty/dense_events in rl_train.py
- [x] Convert v4 JSONL → memmap (DONE: 3,721,587 rows, 132,900 episodes, 52.61 GB)
- [x] BC v4 training (h512, 25 epochs, best val_loss=5.52 @ epoch 13)
- [x] Evaluate BC v4 vs all bots (50 games each): Random 98%, MaxBP 54%, SH 14%, SmartDmg 12%, Tactical 16%, Strategic 14%, Elo ~1470
- [x] Deleted old training data (JSONL 67GB + memmap 49GB) — v5 requires data regen anyway
- [x] **v5 Phase 1**: Built vocabulary files (species 1548, moves 953, items 2340, abilities 314) — `src/vocab.py`
- [x] **v5 Phase 2**: Feature pipeline rewrite — hash buckets → integer IDs, obs 2442→1454 continuous + 82 entity IDs, move_slots 226→98, switch_slots 24→27. Fixed: Perish encoding (4 separate slots), opponent trapped (checks volatiles+abilities), normalization (STAT 255, BP 250, PP 40)
- [x] **v5 Phase 3**: Model architecture — added nn.Embedding layers (species/move/item/ability) to policy_heads.py, 17 sum-pooled groups → 544 dims. Total ~3.85M params (+165K embeddings)
- [x] **v5 Phase 4**: Training pipeline — updated bc_train.py (entity_ids/move_ids/switch_ids flow end-to-end), observer.py (emits v5 IDs), convert_jsonl_to_memmap.py (entity_ids/move_ids/switch_ids arrays). Added advantage-weighted BC (--advantage-weight, --w-win, --w-loss). Auto-detects n_entity_ids from data.
- [x] **v5 Phase 5**: Data pipeline — observer.py and convert_jsonl_to_memmap.py updated for v5 (entity_ids, move_ids, switch_ids). Ready to generate v5 data.
- [x] **v5 Phase 5b (Session 18)**: Comprehensive audit — 23+ fixes across all files. Feature dims: obs 1454→1480 (+12: toxic fraction, future sight, tailwind/screen turn counters), move_slots 98→107, switch_slots 27→28, entity_ids 82→84 (preparing moves), embed groups 17→19. Bot logic fixes, observer fixes, dead code removal.
- [ ] **v5 Phase 6**: Generate v5 data, train BC v5 baseline (no weighting), then advantage-weighted BC, evaluate both
- [ ] Decision Transformer prototype (v5 Step 3, after advantage-weighted BC proves signal)

### Known Bugs (to fix before next data regeneration)
- [x] **policy_rulebots.py: broken STAB calculation** — `move.stab` doesn't exist in poke-env; now checks `move.type in active_pokemon.types`. FIXED.
- [x] **policy_rulebots.py: broken switch logic** — `last_used_move` doesn't exist; now uses opponent's STAB types as proxy. FIXED.
- [x] **policy_rulebots.py: SetupThenSweep inverted** — `best_eff <= 1.0` meant "setup when walled" — fixed to `>= 1.0`. FIXED (Session 18).
- [x] **policy_smartbots.py: SmartDamage pivot filter** — `m.base_power > 0` excluded Parting Shot. FIXED (Session 18).
- [x] **policy_smartbots.py: n_opp_remaining overcounted** — `6 - fainted` counts unrevealed as alive. FIXED to `len(not fainted)` (Session 18).
- [x] **policy_smartbots.py: SR damage uncapped** — could exceed 50% for 4x weakness. Added `min(0.5, ...)` (Session 18).
- [x] **observer.py: ties = losses** — `result=0` for ties indistinguishable from losses. FIXED to `result=0.5` (Session 18).
- [x] **observer.py: opponent data loss** — Opponent player never flushed with `--log-both`. Added explicit flush (Session 18).
- [x] **convert_jsonl_to_memmap.py: standalone row miscount** — Counter used `1` instead of `len(buf)`. FIXED (Session 18).
- [x] **convert_jsonl_to_memmap.py: incomplete_rows always 0** — Double-subtraction cancelled out. FIXED (Session 18).
- [x] **features.py: dead code** — Removed hashlib, `_hash_to_index`, `_hashed_one_hot`, `_hashed_k_hot`, `_encode_bench_items_abilities` (Session 18).
- [x] **features.py: action_mask fallback** — checked `len(moves_meta)` (always 4) instead of `len(moves)`. FIXED.
- [x] **features.py: _count_revealed_se_in_party** — Returned raw count 0-2 instead of fraction 0-1. FIXED (Session 18).
- [x] **features.py: Normal type missing from NVE** — Normal→Rock/Steel was 1.0 instead of 0.5. FIXED (Session 24).
- [x] **features.py: BP normalization inconsistency** — `_encode_move_compact` used cap=200 vs slot encoder cap=250. Unified to 250. FIXED (Session 24).
- [x] **features.py: Fixed damage detection** — `damage="level"` (Seismic Toss) returned 0. FIXED (Session 24).
- [x] **features.py: trap feature never fired** — `getattr(m, "trap")` doesn't exist. Now checks volatile_status. FIXED (Session 24).
- [x] **features.py: stat normalization unbounded** — actual stats >255 produced values >1.0. Added min(val, 1.5) clamp. FIXED (Session 24).
- [x] **bc_train.py: value loss masking bug** — BCEWithLogitsLoss(0,0)=log(2) at masked positions. FIXED (Session 24).
- [x] **bc_train.py: label smoothing bypassed** — advantage weighting path skipped label smoothing. FIXED (Session 24).
- [x] **rl_train.py: PPO temperature mismatch** — old_logp from scaled distribution, new_logp unscaled. FIXED (Session 24).
- [x] **rewards.py: KO double-counting** — phi included fainted counts AND event rewards added ko_bonus. FIXED (Session 24).
- [x] **policy_rulebots.py: switch candidate type math** — nested loop multiplied across both dimensions. FIXED (Session 24).
- [x] **teams_ou.py: illegal/nonfunctional sets** — Pincurchin Recover, Hawlucha Electric Seed, Politoed Helping Hand, Indeedee-F Heal Pulse. FIXED (Session 24).
- [ ] **features.py: ctx_keys order alignment** — `extract_env_presence` ctx_keys list order could potentially misalign. Low risk edge case.
- [ ] **Data regeneration needed** — features.py and teams_ou.py changes (Session 24) require data regen for full effect.

### Deferred Features (not worth adding now)
These were identified in the Session 18 audit but deferred as low-value relative to implementation cost:
- **Opponent PP tracking**: poke-env doesn't expose opponent PP; would need manual tracking from battle log. Low signal — PP stalling is rare at current bot skill level.
- **Choice lock detection**: Inferring if opponent is Choice-locked requires tracking move history + item identification. Complex to implement correctly, marginal benefit for BC.
- **Substitute HP%**: poke-env tracks Substitute presence (already in volatile bits) but not remaining HP. Would need manual tracking. Edge case.
- **Ability trigger history**: Tracking which abilities have activated (e.g., Intimidate on switch-in) is observable from logs but complex to encode. Low priority until the model is strong enough that this matters.

### Observation Improvements v2 — COMPLETE
**988 → 2394 dims (+1406 new) — IMPLEMENTED, VERIFIED, DATA REGENERATED**

All v2 observation improvements have been implemented, verified, and training data regenerated:
- [x] Pokemon Stats (+72 dims): our active/bench actual stats + opponent toggleable base stats
- [x] Revealed Moves (+1012 dims): opponent active/bench + our bench move encodings (23-dim per slot)
- [x] Bench Items & Abilities (+320 dims): 16-dim hashes for items and abilities
- [x] Alive Counts (+2 dims): remaining Pokemon per side
- [x] Tera used flags: already existed in mechanic_flags
- [x] Hazard layers: already encoded as fractional values

### Cross-Battle Memory (Phase 4+)
- [ ] **Opponent modeling**: Track observed sets across battles (moves, items, abilities seen on each species). Build internal usage-stats memory bank. Useful for ladder play where you face the same opponent multiple times.
- [ ] **Metagame context vector**: Feed format-level usage stats as input (e.g., "70% of Garchomp in OU run Choice Scarf"). Provides priors even on turn 1.
- [ ] **Memory bank architecture**: Separate memory module from within-battle LSTM. Current LSTM resets per battle (correct for independent battles). Cross-battle memory would be a retrieval-augmented module that queries a database of past observations — architecturally distinct from the within-battle sequential memory.
- [ ] **Decision**: Cross-battle memory is a long-term goal. Current priority is getting strong single-battle play first. The within-battle LSTM already handles sequential reasoning within a game. Cross-battle features would layer on top, not replace it.

### Future
- [ ] Team expansion (more variety, use scraped data)
- [ ] Live ladder play
- [ ] Team building module
- [ ] Multi-format: doubles (VGC, Doubles OU), triples
- [ ] Multi-gen: Gen 4-8 singles, doubles, triples

---

## RL Training Results (PPO, KL-penalized)

### RL v2 (Session 14 — 2394-dim obs, resumed from iter 120)
**Run:** `ppo_20260311_204716` (iters 1-120) + `ppo_20260312_004922` (iters 120-199)
**Config:** lr=3e-4, clip=0.2, ent=0.01, kl_coef=0.1, temp=1.0→0.37, no-amp, 100 games/iter
**Init:** BC v2 checkpoint (bc_v2_2394dim_bs16/best.pt, 2394-dim obs)

| Iter | MaxBP | SH | SmartDmg | Tactical | Strategic | **Avg** |
|------|-------|----|----------|----------|-----------|---------|
| 120 | 40% | 5% | 15% | 15% | 5% | 16.0% |
| **130** | **55%** | **30%** | **15%** | **20%** | **20%** | **28.0%** (best) |
| 140 | 50% | 10% | 30% | 30% | 10% | 26.0% |
| 150 | 60% | 15% | 15% | 15% | 10% | 23.0% |
| 160 | 45% | 5% | 15% | 5% | 10% | 16.0% |
| 170 | 50% | 20% | 5% | 15% | 5% | 19.0% |
| 180 | 45% | 15% | 5% | 15% | 10% | 18.0% |
| 190 | 60% | 10% | 25% | 20% | 10% | 25.0% |

**Per-bot trend analysis:**
- **MaxBasePower (40-60%):** Strongest and most consistent matchup. Ranges 40-60% — model handles pure-damage bots well. No clear trend (flat).
- **SimpleHeuristics (5-30%):** Most volatile. Spiked to 30% at iter 130 but dropped to 5% at iters 120/160. No sustained improvement — model hasn't learned to counter SH's switching/status logic.
- **SmartDamage (5-30%):** Also volatile. Two good evals (30% at 140, 25% at 190) but also 5% at 170/180. No trend direction.
- **Tactical (5-30%):** Peaked at 30% (iter 140) but dropped to 5% at iter 160. No clear improvement over time.
- **Strategic (5-20%):** Consistently the hardest opponent. Never above 20%. The model cannot learn to counter Strategic's long-term play (setup moves, recovery, hazards).

**Overall assessment:**
- Best avg 28% (iter 130) vs RL v1 best 18.3% (iter 100) — the 2394-dim obs helped
- But performance is **flat with high variance**, not improving over iterations 120-199
- Training win rate stable 17-37% per iter (against weighted opponent pool including BC)
- KL well-controlled (0.013-0.018), no NaN issues, no connection hangs (timeout fix worked)
- The model is **plateaued** — PPO is not finding useful gradients to improve beyond BC+noise level

**Likely bottlenecks (why RL isn't learning more):**
1. **Reward sparsity**: Win/loss is +1/-1 terminal, shaping is weak. Most episodes are losses (75%), giving mostly negative signal. The model can't extract fine-grained strategy from this.
2. **Opponent too strong**: Training against 80%+ win rate bots means the RL agent almost always loses. Hard to learn from a 25% win rate — the signal-to-noise ratio is poor.
3. **KL anchor too tight**: kl_coef=0.1 keeps the model close to BC, which limits how far RL can deviate. But loosening it risks catastrophic forgetting.
4. **20-game eval variance**: With only 20 games per bot, the eval numbers have ±15-20% noise. Need 50+ game evals for reliable signal.
5. **No curriculum**: Model faces the same difficulty throughout. A curriculum (start with weaker bots, gradually add stronger ones) could help build foundational skills first.

### RL v1 (Session 11 — all bug fixes applied)
**Run:** `ppo_20260310_215746` (150/200 iters completed, hung on websocket timeout)
**Config:** lr=3e-5, clip=0.1, ent=0.005, kl_coef=0.1, temp=0.8→0.3, no-amp
**Init:** BC v3 checkpoint (bc_flat_988dim_v3_slotenc/best.pt)

| Iter | SimpleHeuristics | SmartDamage | Tactical | Avg |
|------|-----------------|-------------|----------|-----|
| BC baseline | 14% | 18% | 16% | 16% |
| 10 | 10% | 15% | 15% | 13.3% |
| 20 | 20% | 10% | 15% | 15.0% |
| 30 | 10% | 20% | 20% | 16.7% |
| 40 | 20% | 25% | 5% | 16.7% |
| 50 | 20% | 15% | 15% | 16.7% |
| 60 | 15% | 15% | 15% | 15.0% |
| 70 | 15% | 15% | 10% | 13.3% |
| 80 | 10% | 15% | 5% | 10.0% |
| 90 | 5% | 10% | 15% | 10.0% |
| **100** | **20%** | **20%** | **15%** | **18.3%** (best) |
| 110 | 5% | 5% | 10% | 6.7% |

**50-game eval (best checkpoint, iter 100):**
| Opponent | Win Rate | W/L |
|----------|----------|-----|
| Random | 90% | 45/5 |
| MaxBasePower | 54% | 27/23 |
| SimpleHeuristics | 12% | 6/44 |
| SmartDamage | 14% | 7/43 |
| Tactical | 20% | 10/40 |
| Strategic | 14% | 7/43 |
| **Overall** | **32%** | **102/198** |

**Key findings (v1):**
- NaN gradient bug fixed (`-inf` → `-1e9` masking) — no NaN issues throughout training
- KL divergence well-controlled (0.0006 → ~0.01, never exceeded 0.011)
- Best at iter 100 (18.3%), marginal improvement over BC baseline (16%)
- RL didn't significantly improve over BC — likely needs richer observations (pokemon stats) and more diverse training signal
- Training win rate fluctuated 20-35% per iter (against mixed opponent pool)
- Process hung at iter 150+ due to Showdown websocket timeout — best model already saved

### RL v0 (Session 8 — before bug fixes, unreliable)
**Run:** `ppo_20260310_053326` (200 iters, 50 games/iter, CUDA)

| Iter | SimpleHeuristics | SmartDamage | Tactical | Avg |
|------|-----------------|-------------|----------|-----|
| **80** | **40%** | **65%** | **20%** | **41.7%** (best) |

Note: v0 results are unreliable — slot encoders were ignored, LSTM states shuffled, reward double-counted, entropy gradients blocked. The high numbers may reflect exploiting broken evaluation rather than genuine improvement.

---

## Session Log

### Session 1 (2026-03-07)
- Explored project, fixed observer.py, features.py (908-dim)

### Session 2 (2026-03-07)
- Fixed all policy files, removed _OrderStr wrappers

### Session 3 (2026-03-07)
- Fixed env_wrapper, generated first dataset (85k, 908-dim)
- Added volatile statuses + tera type (908 -> 978 dims)
- Expanded teams from 10 -> 30

### Session 4 (2026-03-07)
- Fixed 30 teams for Gen 9 OU legality (6 rounds of fixes)
- Regenerated data: 64 files, 195k records, 978-dim
- Installed PyTorch CUDA, trained BC LSTM (50 epochs)
- Fixed bc_train.py bugs (label smoothing, AMP overflow)
- Fixed bc_policy_player.py (action_mask returns dicts, not objects)
- Evaluated: 82% vs Random, 3-34% vs heuristic bots, Elo 1462

### Session 5 (2026-03-07)
- Audited all obs fields, found 27 missing poke-env properties
- Added combat state (5 dims per active: first_turn, must_recharge, preparing, protect_counter, status_counter)
- Expanded move slot vector (34 -> 226 dims) with category, boosts, screens, volatile, secondary effects
- Built 3 smart bots (SmartDamage, Tactical, Strategic) extending SimpleHeuristicsPlayer
- Initial bots failed (0% vs SimpleHeuristics) — opponent stats are all None, must use base_stats
- Tuned utility gating (only use status/recovery when best attack < 80 score)
- Full round-robin: Tactical 81.8%, Strategic 81.3%, SmartDamage 78.0% > SimpleHeuristics 67.8%

### Session 6 (2026-03-08/09)
- Generated 473k records with 988-dim obs + 10 bots (concatenated to obs_all_988dim.jsonl)
- Trained hierarchical BC model (60 epochs, 89.4% val accuracy)
- Found 0% live win rate — debugged extensively
- Fixed bc_policy_player.py: ctx_extra_dim/step_type_bins override bug
- Fixed bc_train.py: --amp flag now uses BooleanOptionalAction
- Discovered hierarchical action head causes permanent move bias (never switches)

### Session 7 (2026-03-09/10)
- Identified --use-lstm bug in eval_bc_vs_bots.py (was always False, dropping LSTM)
- Confirmed flat model works: 96% vs Random, 28% vs MaxBasePower, 8% vs SimpleHeuristics
- Confirmed hierarchical model still worse even with LSTM fix (move bias persists)
- Trained flat model to convergence (41 epochs, 85.9% val accuracy)
- Updated docs, ready for RL phase

### Session 8 (2026-03-10)
- Added KL divergence penalty to rl_train.py (frozen BC reference model, configurable kl_coef)
- Added BC flat model as RL training opponent (BCPolicyPlayer in opponent pool, weight=2.0)
- Added compute_kl_from_ref() function for masked KL divergence computation
- Ran PPO training on CUDA: 200 iters, 50 games/iter, kl_coef=0.1
- Best RL checkpoint at iter 80: 41.7% avg eval (40% SimpleHeuristics, 65% SmartDamage, 20% Tactical)
- KL penalty successfully prevents catastrophic forgetting — model stays functional throughout training
- Updated STATUS.md with RL results table

### Session 9 (2026-03-10)
- Full codebase audit — found and fixed 6 critical bugs:
  1. **policy_heads.py**: Flat action head ignored slot encoders (`action_head(core)` → per-slot scoring with move_enc/switch_enc)
  2. **rl_train.py**: PPO shuffled mini-batches destroyed LSTM hidden states → episode-sequential processing
  3. **rl_train.py**: Reward shaping double-counted (terminal_r included accumulated shaping)
  4. **policy_heads.py**: Entropy gradient blocked by `torch.no_grad()` wrapper
  5. **features.py**: Type effectiveness missing immunities (Fighting→Ghost, Psychic→Dark) + broken dual-type calc
  6. **bc_train.py**: step_type mismatch (training used relative position, inference used absolute turn)
- Fixed eval harness: per-battle team randomization (RandomPoolTeambuilder) in eval_bc_vs_bots.py, eval_bots_roundrobin.py
- Reliable BC baseline (50 games, per-battle randomization): Random 94%, MaxBP 56%, SH 14%, SmartDmg 18%, Tactical 16%, Strategic 18%
- Kicked off BC retrain with slot encoders active (bc_flat_988dim_v2_slotenc)

### Session 10 (2026-03-10)
- Fixed policy_rulebots.py: STAB calculation (move.stab doesn't exist → check move.type in active.types)
- Fixed policy_rulebots.py: switch logic (last_used_move doesn't exist → use opponent STAB types as proxy)
- Fixed features.py: action_mask fallback checked len(moves_meta) (always 4) instead of len(moves)
- Fixed eval_bc_vs_bots.py: IndexError crash when all_meta_rows is empty (added guard)
- Confirmed bc_train.py RAM cache is correct (no disk re-reading bug; ~10 min/epoch is GPU-bound)
- Confirmed --cache-in-ram risky on 16GB system (cache ~8-12GB → swap thrashing)
- BC retrain complete: bc_flat_988dim_v3_slotenc (60 epochs, 22 min total, 22 sec/epoch with numpy cache)
- Best checkpoint: epoch 32 (val_loss=7.57, val_acc~89.4%) — better than old run (8.81, 85.9%)
- Numpy cache optimization: converted Python float lists (28 bytes/float) to numpy arrays (4 bytes/float), 35GB→4GB cache, 27x speedup

### Session 11 (2026-03-10)
- **Critical fix: NaN gradient bug in PPO training**
  - Root cause: `float("-inf")` masking in policy_heads.py causes `0 * -inf = NaN` in entropy backprop
  - `-inf` masked logits → softmax gives exact 0 probs → `probs * log_softmax = 0 * -inf = NaN` → NaN gradients corrupt all weights
  - Fix: Replace `float("-inf")` with `-1e9` in action_mask and mod_legal masking (policy_heads.py)
  - Also added NaN guards in rl_train.py: obs/slots/ctx nan_to_num, value NaN guard, gradient NaN skip, loss NaN skip
- RL v1 training completed: 150/200 iters (hung at iter 150+ on websocket timeout)
  - Best model saved at iter 100: SH=20%, SmartDmg=20%, Tactical=15%, avg=18.3%
  - Marginal improvement over BC baseline (~16%) — model needs richer observations
  - KL divergence well-controlled throughout (0.0006 → 0.01)
  - No NaN issues — `-1e9` masking fix worked perfectly
- Batched game collection by opponent type for efficiency
- Updated observation improvement TODOs with decisions:
  - Our Pokemon: actual stats (pokemon.stats), no separate speed calc needed
  - Opponent Pokemon: toggleable base stats (always reserve dims, fill or zero via flag)
  - Reserve 72 new dims total (6 active + 30 bench ours + 36 opp toggleable)
- RL v1 50-game eval completed:
  - vs Random 90%, MaxBP 54%, SH 12%, SmartDmg 14%, Tactical 20%, Strategic 14%
  - Overall 32% (102/198), Elo ~1469-1501

### Session 12 (2026-03-11)
- Expanded v2 observation plan from 72 → 1408 new dims (988 → 2396 total):
  - Pokemon stats: 72 dims (our actual stats + opponent toggleable base stats)
  - Revealed moves: 1012 dims (opponent active 92 + opponent bench 460 + our bench 460)
  - Bench items & abilities: 320 dims (our 160 + opponent revealed 160, 16-dim hashes)
  - Battle state: 4 dims (tera used flags + alive counts)
  - Hazard layers already encoded (spikes_layers/tspikes_layers existed)
  - Tera used flags already existed in mechanic_flags (no new dims needed)
  - Alive counts: 2 dims
- Implemented in features.py: new helpers (_get_sorted_bench, _encode_stats, _encode_move_compact, _encode_pokemon_moves_compact, _encode_bench_stats, _encode_bench_moves, _encode_bench_items_abilities, _alive_counts)
- Live-tested: 2394 dims confirmed, no NaN/Inf, all blocks populated correctly

### Session 13 (2026-03-11)
- Validated generated 2394-dim training data: 540k records, 130 files, 0 skipped, all fields correct
- Data richness confirmed: moves/mons revealed correctly over turns, move slots zero only on forced switches
- Identified BC training bottleneck: 10GB JSONL re-parsed every epoch (~2.5 min/epoch just I/O)
- **Implemented memmap data pipeline (Option 2)**:
  - New file: `convert_jsonl_to_memmap.py` — streams JSONL → buffers complete episodes → writes contiguous memmap
  - Output: separate .npy files per field (obs, action, legal, move_slots, switch_slots, ctx_extra, mods, result, turn, phase, episode_index) + meta.json
  - SHA-1 based episode hashing for deterministic train/val split (10.2% val, uniform distribution)
  - Fixed int64 overflow: mask hash to `& 0x7FFFFFFFFFFFFFFF` to fit signed int64
  - Full conversion: 540,059 rows, 18,682 episodes, 7.53 GB, completed in ~690s
- **Added MemmapEpisodeDataset to bc_train.py**:
  - Map-style Dataset with lazy memmap opening (Windows `spawn` compatible)
  - `.copy()` on memmap slices to avoid page pinning
  - Episode-contiguous layout for efficient LSTM sequential reads
  - New CLI args: `--data-format memmap`, `--memmap-dir`
- Kicked off BC v2 training: `bc_v2_2394dim_memmap` (60 epochs, 2394-dim obs, LSTM, no-AMP)
  - Early results: epoch 5 val_acc=60.4% (up from 42.5% at epoch 1)

### Session 14 (2026-03-12)
- Killed 4 stray RL processes (7.5 GB RAM hung on websocket timeout)
- **Fixed rl_train.py connection issues:**
  - Added `asyncio.wait_for()` timeout to `evaluate_vs_bot()` (was missing — caused infinite hangs)
  - Added PID-based player names (`_pid_tag = os.getpid() % 10000`) to prevent name collisions between concurrent processes
- **Fixed rl_train.py resume bugs:**
  - Temperature reset to 1.0 on resume → now computes `temp * decay^start_iter`
  - LR scheduler reset from scratch → now fast-forwards to `start_iter`
  - `best_eval_wr` reset to 0.0 → now restored from checkpoint metrics
  - Periodic checkpoints now save `best_eval_wr` and `temperature` for proper resume
  - Cosine LR `T_max` now uses `start_iter + n_iters` (total), not just `n_iters`
- RL v2 training completed: 199 iters total (120 from prior run + 80 resumed)
  - Best: iter 130, avg 28% (MaxBP 55%, SH 30%, SmartDmg 15%, Tactical 20%, Strategic 20%)
  - Performance flat with high variance across iters 120-199 — RL has plateaued
  - No connection issues — timeout fix worked, zero hangs across 80 iters
- **Implemented RL v3 features** (all flag-controlled, default ON):
  - **(A) Curriculum learning** (`--curriculum`): 4 tiers of opponent difficulty
    - Tier 1: MaxBP, GreedySE, SetupThenSweep (promote at 55% avg)
    - Tier 2: + SimpleHeuristics (promote at 45% avg)
    - Tier 3: + SmartDamage, Tactical, Strategic (promote at 35% avg)
    - Tier 4: full pool (no promotion)
    - Promotion requires minimum win rate vs EVERY bot in tier (not average) — prevents easy bots inflating scores
    - Thresholds: Tier 1→2: 55% each, Tier 2→3: 50% each, Tier 3→4: 40% each
    - On promotion: temperature bumped +0.15 (explore new opponents), LR cosine warm-restarted (fresh learning capacity), optimizer state preserved (stability)
    - Configurable: `--promote-temp-bump`, `--promote-temp-cap`, `--promote-lr-restart`/`--no-promote-lr-restart`
  - **(B) Dense event rewards** (`--dense-rewards`): per-KO bonuses on top of existing potential shaping
    - +0.15 per opponent KO, -0.15 per own faint (configurable via `--ko-bonus`, `--faint-penalty`)
    - Raised clip_abs 0.5 → 2.0 (`--reward-clip`) so shaping signal isn't squashed
  - **(C) Self-play** (`--self-play`): periodic model snapshot as opponent
    - Saves snapshot every N iters (`--self-play-interval`, default 10)
    - Loaded via BCPolicyPlayer on CPU (no VRAM competition)
    - Sampling weight configurable (`--self-play-weight`, default 2.0)
  - **(D) Adaptive opponent weighting** (`--adaptive-weights`): dynamically adjusts training opponent distribution
    - After each eval, bots below promotion threshold get upweighted (scale = threshold/wr, capped at 5x)
    - Bots above threshold get downweighted (0.5x base, floored at `min_weight=0.5`)
    - Focuses training on hardest matchups (e.g., SimpleHeuristics) instead of wasting games on mastered bots
    - Uses latest eval results; first iteration before any eval uses uniform tier weights
  - Backed up v2 code: `rl_train_v2.py`, `rewards_v2.py`
  - **Architecture lesson**: Curriculum promotion should use min-per-bot threshold, not average — weak bots inflate averages and cause premature promotion

### Session 15 (2026-03-12)
- RL v3 runs (3 attempts) all stuck at Tier 0/1 — model oscillates 20-36% avg, can't simultaneously beat all tier bots
- **Root cause**: model can't learn basic Pokemon math (type charts, damage formulas, speed comparisons) from raw stats via RL alone — needs pre-computed features
- **Implemented v4 computed features** (48 new dims, obs 2394 → 2442):
  - Group A (16 dims): active vs active — matchup score, type advantages, speed tier (with TR/Tailwind/PAR), phys/spec ratios (both directions), best move damage score, KO checks, priority signals
  - Group B (10 dims): our bench vs opp active — per-slot matchup scores + defensive resist scores
  - Group C (5 dims): opp bench vs our active — threat scores for revealed bench mons
  - Group D (5 dims): aggregate bench signals — best matchup, best resist, positive matchup count, hazard-aware switch score, SR entry cost
  - Group E (12 dims): game context — endgame flag, boost totals (both sides), remaining counts, HP advantage, status flags, weather benefit, STAB signal, bench threat signal
  - All features use revealed info only (no omniscience), base_stats for opponents
  - Formulas match SimpleHeuristics/SmartBots exactly (_estimate_matchup, _stat_estimation, _move_damage_score)
- **Tuned adaptive weights**: cap reduced 5x → 3x (prevents overwhelming model with hard opponents)
- **Eval improvements**: now evals all bots in current curriculum tier (was hardcoded 5), 50 games/eval (was 20)
- **Lowered promotion thresholds**: Tier 0: 55% → 40%, Tier 1: 50% → 40%
- **Observer optimized**: default concurrency 4 → 8, parallel pairings mode (3 concurrent pairings by default)
- Backed up v3 code: `features_v3.py`, `observer_v3.py`
- **Observer further optimized**: `play_batch()` reuses player ws connections across multiple games (was 1 game per player), `--batch-per-worker` CLI flag (default 5)
- **v4 data generation COMPLETE** (~3 hours, finished 16:27):
  - 198 JSONL files, ~5M records at 2442 dims
  - Weighted toward smart bots: Smart×Smart 16 pairs × 2500 games (~52%), Smart×Medium 32 × 500 (~21%), Medium×Medium 16 × 400 (~8%), Smart×Weak 16 × 300 (~6%), Medium×Weak 16 × 200 (~4%), Weak×Weak 4 × 100 (~0.5%), plus existing 444K base data
  - Smart bots = SimpleHeuristics, SmartDamage, Tactical, Strategic
  - Medium bots = GreedySE, HazardSense, SwitchAwareEscape, SetupThenSweep
  - Weak bots = MaxBasePower, Random
- **Model upgrade planned**: 256 → 512 hidden (3.6M params, ~1.4x data/param ratio with 5M records)
- Old data backed up: `obs_v3_backup/`, `memmap_v3_backup/`
- **NEXT steps (in order)**:
  1. Convert JSONL → memmap (conversion rewritten — two-pass streaming, won't OOM)
  2. BC v4 training with 512 hidden, 2442-dim obs, memmap data, --no-amp
  3. Evaluate BC v4 — if substantially better than BC v2, proceed
  4. Decide: RL v4 (PPO, needs 500+ games/iter) vs offline RL (v5 approach)

### Session 16 (2026-03-12)
- **Codebase audit & OOM fixes:**
  - `convert_jsonl_to_memmap.py`: Complete rewrite — two-pass streaming + integrity verification. Old version buffered all episodes in RAM (70GB+ at 5M records → guaranteed OOM). New version: Pass 1 scans (no data stored, <100MB RAM), Pass 2 writes directly to pre-allocated memmap, Pass 3 verifies integrity (shapes, episode consistency, NaN/Inf, legal masks, spot-check vs JSONL)
  - `rl_train.py`: `del trajectories` + `reset_trajectories()` after build_ppo_episodes, `del episodes` + `torch.cuda.empty_cache()` after PPO update, `del eval_player/opponent` after eval, cache clear after eval loop
  - `bc_train.py`: `torch.cuda.empty_cache()` between epochs (prevents VRAM fragmentation over 60 epochs)
  - `bc_policy_player.py`: `float("-inf")` → `-1e9` masking (consistency with session 11 fix)
- **Bug fixes found by codebase audit:**
  - `features.py:1008,1017`: Boost formula used `> 1` instead of `> 0`. A +1 stat boost gave 2x multiplier instead of correct 1.5x. Affected ALL v4 computed features (matchup scores, damage estimates, KO checks). Data generated with wrong formula but impact is narrow (+1 boost only, not ±2-6). Pragmatic decision: train on current data, RL will correct.
  - `rl_train.py:151-157`: RewardShaper clone in `_ensure_trajectory()` didn't copy `ko_bonus`, `faint_penalty`, `dense_events` from template. `--no-dense-rewards` flag had zero effect — per-battle shapers always used constructor defaults (dense_events=True).
  - `bc_policy_player.py:315`: Dead code after `torch.no_grad()` block re-extracts logits (already extracted on line 307). Fragile but not currently broken.
  - `bc_train.py:1106`: Logged training loss excludes v_loss component (only pol_loss + mod_loss). Training is correct, just logging underreports.

### Session 17 (2026-03-15)
- BC v4 already trained (25 epochs, best val_loss=5.52 @ epoch 13) and evaluated:
  - Random 98%, MaxBP 54%, SH 14%, SmartDmg 12%, Tactical 16%, Strategic 14%, Elo ~1470
  - Essentially same as prior versions — confirms BC ceiling
- **Disk cleanup**: deleted JSONL (67 GB) + memmap (49 GB) training data — v5 requires data regen anyway
  - Also cleaned conda envs (~11 GB) and caches (~10 GB)
  - Disk: 800 GB used → 666 GB used (266 GB free, 72%)
- **v5 plan finalized**: feature pipeline rewrite first (learned embeddings, bug fixes), then generate data once
  - Split obs into continuous (~1450 dims) + integer IDs (~90) for nn.Embedding
  - Advantage-weighted BC: weight loss by game result (winners 2-3x, losers 0.3-0.5x)
  - 6 phases: vocab → features → model → training → data → train+eval
- **v5 Phases 1-5 completed**: vocab, features, model architecture, training pipeline, data pipeline all done
  - observer.py emits entity_ids (82), move_ids (4), switch_ids (5) in JSONL rows
  - convert_jsonl_to_memmap.py writes entity_ids.npy, move_ids.npy, switch_ids.npy
  - bc_train.py: auto-detects entity IDs from data, passes to model, advantage-weighted BC via --advantage-weight
  - End-to-end live test passed (3.85M params, 165K embedding params)
- **Next**: Generate v5 training data (~3 hours), train BC v5 baseline, then advantage-weighted BC

---

## v5 Strategic Analysis — Architecture & Training Paradigm

### Why RL (PPO) Has Plateaued

After 4 iterations of online RL (v1→v2→v3→v4), each adding more features and tricks, the pattern is clear: small improvement, then plateau. The bottleneck is the **training paradigm**, not features or hyperparameters.

**Fundamental issues with PPO for Pokemon:**

1. **Collection speed**: Each game requires a full Showdown websocket round-trip per turn. 100 games/iter ≈ 30s collection. For stable PPO gradients with Pokemon's variance (team matchups, crits, misses), you need 500-1000 games/iter — 5-10x slower. By the time you collect enough data, the on-policy assumption is strained.

2. **Reward sparsity**: Win/loss at turn 30-80 with γ=0.99 means reward is attenuated by 0.99^40 ≈ 0.67 by mid-game. Dense KO rewards (+0.15) are faint signal relative to policy entropy. The model can't extract fine-grained strategy from this.

3. **Sample inefficiency**: PPO is on-policy — each batch of trajectories is used for ~4 epochs then discarded. With 100 games/iter × 50 turns = 5000 steps, the policy update uses 5000 steps then throws them away. We have 5M expert demonstrations sitting on disk, unused during RL.

4. **Variance**: 20-game evals have ±15-20% noise. Even 50-game evals have ±10%. The "best 28%" at iter 130 could be noise — performance at iter 140 was 26%, iter 160 was 16%. We can't distinguish signal from noise at this scale.

5. **Opponent too strong**: Training against 80%+ win rate bots means the RL agent mostly loses. Curriculum helps sequence difficulty but doesn't solve the fundamental signal-to-noise ratio of learning from a 25% win rate.

### Model Architecture Concerns

- **Model too small**: 3.6M params for 2442-dim input. The 512-dim LSTM bottleneck must compress entire battle history (reveals, boosts, hazards, team info) into a single vector. Simple Atari agents use 3-5M params on 84x84 pixel inputs with simpler decision spaces.
- **Hash-bucketed features are inefficient**: 128-dim k-hot species hashing means most dims are zero. A learned embedding (species → 32-dim via lookup table) would be far more parameter-efficient.
- **No attention mechanism**: The LSTM must "remember" all opponent reveals sequentially. A simple attention layer over the last N turns would let the model "look back" at specific events rather than hoping LSTM state encodes them.
- **Hardware constraint**: RTX 3060 Laptop 6GB VRAM limits model size. 512 hidden is near the sweet spot for this GPU. Bigger models → cloud training.

### Websocket / Infrastructure Bottleneck

- poke-env 0.10.0 wasn't designed for high-throughput automated play
- Every battle = websocket connection, every move = round-trip message
- PID-based player names are a workaround for Showdown rejecting duplicates
- `docker restart showdown` to clear stale connections is manual intervention
- **Scaling makes this worse**: 500+ games/iter will hit Showdown connection limits
- Options: multiple Showdown instances (round-robin), poke-env local sim mode, or direct engine interface (eliminates websocket entirely)

### Data Quality Ceiling

- 5M records is excellent volume but **expert quality is capped** — best bot (Tactical) wins 81.8% in round-robin
- BC ceiling is bounded by expert quality: perfect imitation of Tactical = ~82% vs bot pool
- Bots don't predict opponent switches, don't plan multi-turn sequences, don't bait
- The model is learning to imitate heuristic players, not to play optimally
- 2442-dim obs is rich but sparse (hash buckets mostly zero). Data/param ratio (5M/3.6M ≈ 1.4) is low — model could be larger or data more efficiently encoded

### v5 Recommended Approach: Offline RL

**Instead of online PPO, use offline RL on the existing 5M demonstrations.**

**Why offline RL fits this project:**
- Uses the 5M demonstrations directly as training data (no websocket collection needed)
- No Showdown bottleneck during training — pure GPU workload
- Can leverage cloud GPUs for larger models without needing Docker/websocket infra
- Well-suited to "learn from demonstrations then improve" setting
- No stale policy problem, no collection variance, no connection hangs

**Candidate algorithms (in order of recommendation):**

1. **Decision Transformer (DT)**: Frames RL as sequence modeling. Input: (return-to-go, state, action) tuples. Model: Transformer that predicts actions conditioned on desired return. At inference, condition on high return = play to win. Natural fit for Pokemon (sequential decisions, variable-length episodes). Can use larger models than LSTM. Drawback: needs return conditioning tuning.

2. **Implicit Q-Learning (IQL)**: Learns a Q-function from offline data without querying out-of-distribution actions. Conservative but stable. Can extract policy better than BC by learning which expert actions were "good" vs "lucky". Works well with moderate data/model sizes.

3. **Conservative Q-Learning (CQL)**: Adds a conservative penalty to prevent Q-value overestimation on unseen actions. More complex than IQL but stronger theoretical guarantees. Needs careful hyperparameter tuning.

4. **Filtered BC / Advantage-Weighted BC**: Simplest approach — weight BC loss by estimated advantage. Actions from games the expert won get higher weight. Actions from losses get downweighted. Nearly free to implement on top of existing bc_train.py.

**v5 implementation plan (phased, prove-then-scale):**

**Step 1 — BC v4 baseline (local, free)**
- Train BC v4 on 5M memmap data (512 hidden, 2442-dim, --no-amp)
- Evaluate vs all bots (50+ games each). This is the baseline to beat.

**Step 2 — Advantage-Weighted BC (local, free, ~20 lines of code)**
- Modify bc_train.py: weight loss by game result. Winners get 2-3x weight, losers 0.3-0.5x.
- Hypothesis: model learns more from winning expert actions than losing ones.
- Success criteria: >40% avg vs strong bots (up from BC's ~28%).
- If this works, it proves offline RL signal exists in the data.

**Step 3 — Decision Transformer prototype (local, free)**
- Replace LSTM with small Transformer (3.6M params, fits in 6GB VRAM).
- Add return conditioning: input (return-to-go, state, action) tuples.
- At inference, condition on return=1.0 (win) to play aggressively.
- Train locally, evaluate locally. Prove the architecture before spending money.

**Step 4 — Iterate locally, scale model (still local, free)**
- Scale DT/LSTM to 10-20M params (fits in 6GB VRAM with batch size tuning).
- Learned embeddings instead of hash buckets (requires feature pipeline rewrite + data regen).
- 200+ game evals for statistical significance.
- Cloud only needed later for Phase 4+ (live ladder, cross-battle memory, multi-gen, 50M+ models).

**Key principle: all v5 work is local and free. Cloud is a Phase 4-5 concern.**

### Cloud Training Strategy

**v5 does NOT require cloud.** RTX 3060 Laptop (6GB VRAM) has sufficient headroom:
- Current model uses ~1 GB of 6 GB available
- A 10-20M param model with learned embeddings fits locally with batch size tuning
- The biggest v5 wins are algorithmic (advantage weighting, learned embeddings, fixed type chart), not scale
- Decision Transformer prototype (5-20M params) fits locally

**When cloud becomes necessary (Phase 4-5):**
- **Live ladder play** (Phase 4): continuous learning loop, potentially 24/7 training + inference
- **Cross-battle memory** (Phase 4+): retrieval-augmented module over a database of past observations, significantly larger model footprint
- **Multi-gen / multi-format training** (Phase 5): training across Gen 4-9, all formats — either multiple specialized models or one large multi-task model
- **Hyperparameter sweeps**: running 10+ experiments in parallel to find optimal configs
- **50M+ param models** with long sequences that exceed 6GB activation memory

**Cloud options (for when needed):**
- **Spot/preemptible GPU instances**: 70-90% discount vs on-demand. Providers: Vast.ai, RunPod, Lambda Labs, AWS Spot, GCP Preemptible.
- **Estimated costs**: A100 40GB spot ≈ $0.80-1.50/hr. Full training run ≈ $2-30 depending on model size and duration.
- **Workflow**: Upload memmap data to cloud storage, spin up spot instance, train, download checkpoint, tear down. Evaluate locally (evals need Showdown, not GPU).

**Current plan: prove everything locally (free) through v5 Steps 1-3. Cloud only if hitting a clear hardware wall.**

### Data Generation vs Training Split

A key v5 insight: **separate data generation (needs Showdown) from training (needs GPU).**
- Data gen: run locally on CPU (Showdown + poke-env, no GPU needed). Can run 24/7 on local machine.
- Training: run on cloud GPU. Upload memmap, train, download checkpoint.
- Evaluation: run locally (Showdown + checkpoint inference, minimal GPU).
- This eliminates the need for Showdown on cloud, keeping cloud costs pure GPU time.

---

## Known Issues for v5 — Feature Pipeline & Training Quality

### CRITICAL: Hash Bucketing Destroys Information (features.py)

The entire feature pipeline encodes categorical identities (species, moves, items, abilities) via **hash bucketing** — run the name through blake2b, set k=2 bits in a fixed-size vector. This causes massive collisions where completely different entities produce identical feature patterns, making them indistinguishable to the model.

**Collision analysis:**

| Feature | Unique Values | Bucket Size | k | Saturation | Impact |
|---------|--------------|-------------|---|------------|--------|
| Species (active) | ~1025 | H_SPECIES=128 | 2 | 16x oversaturated | Model can't distinguish many Pokemon |
| Moves (slot encoder) | ~900 | H_MOVE=128 | 2 | 14x oversaturated | Earthquake and Thunderbolt may hash the same |
| Items (active) | ~800 | H_ITEM=64 | 2 | 25x oversaturated | Choice Scarf = Leftovers to the model |
| Abilities (active) | ~300 | H_ABILITY=64 | 2 | 9x oversaturated | Intimidate = Levitate to the model |
| Items (bench) | ~800 | H_ITEM_BENCH=16 | 2 | 100x oversaturated | Bench items are noise |
| Abilities (bench) | ~300 | H_ABILITY_BENCH=16 | 2 | 37x oversaturated | Bench abilities are noise |
| Move ID (mini payload) | ~900 | 4 bits | 2 | 450x oversaturated | **Functionally useless** |
| Species (mini payload) | ~1025 | 8 bits | 2 | 256x oversaturated | **Functionally useless** |

**Why this matters:** The model literally cannot tell apart Pokemon, moves, items, or abilities that hash to the same buckets. A correct response to Earthquake (ground, physical, 100 BP) vs Thunderbolt (electric, special, 90 BP) requires knowing which move it is. With 900 moves in 128 buckets, ~7 moves share each bucket on average.

**Solution: Learned embeddings (v5 Step 3+)**
Replace hash bucketing with `nn.Embedding` lookup tables. Each species/move/item/ability gets a unique, trainable vector:
```python
# Instead of: _hashed_k_hot("Garchomp", 128, k=2)  → 128-dim, collides
# Use:        species_embed = nn.Embedding(1200, 64)  → 64-dim, unique, learned
```
Requires: vocabulary mapping (name → integer ID), minor architecture change. Adds negligible parameters (~200K total for all embeddings). Should be combined with data regeneration.

### CRITICAL: Conflicting Labels from Mixed-Quality Data

Training data comes from bots of vastly different skill levels (Random 5.3% win rate to Tactical 81.8%). Given the **exact same game state**, different bots choose different actions. The model sees contradictory "correct" answers and learns to average across incompatible strategies.

Data composition (v4, ~3.7M records):
- ~52% Smart×Smart (good quality, consistent strategies)
- ~21% Smart×Medium (mixed — medium bot actions are suboptimal)
- ~8% Medium×Medium (mediocre quality)
- ~6% Smart×Weak (mixed — weak bot actions are bad)
- ~4% Medium×Weak, ~0.5% Weak×Weak (poor quality)
- **~30-40% of data contains actions from bots with <50% win rate**

**Solution: Advantage-Weighted BC (v5 Step 2)**
Weight BC loss by game outcome. Actions from winning games get amplified (2-3x), losing games get suppressed (0.3-0.5x). ~20 lines change to bc_train.py. The `result` field (1.0=win, 0.0=loss) is already in the memmap data.

### HIGH: Type Effectiveness Chart Missing NORMAL Immunity (features.py)

The type effectiveness dictionaries (lines 880-917) are **missing NORMAL → GHOST immunity**. Normal-type moves are always calculated as neutral (1.0x) against Ghost-type Pokemon, when they should be 0.0x (immune). This affects:
- All matchup score calculations (v4 computed features)
- All damage estimation features
- The model's ability to learn that Normal moves don't hit Ghosts

**Fix:** Add `"normal": ["ghost"]` to the immunity dict. Must regenerate data after fixing.

### HIGH: Trick Room Ignored in Speed Heuristic (features.py)

The `moved_first_heuristic_bits()` fallback (line 838-850) and `_eff_speed_with_conditions()` do NOT check for Trick Room. Under Trick Room, the slower Pokemon moves first — the speed comparison should be reversed. The main featurize path (lines 863-866) does check Trick Room, but the heuristic fallback doesn't.

**Fix:** Check `battle.fields` for Trick Room in the speed heuristic. Must regenerate data after fixing.

### MEDIUM: Boost Formula Bug in v4 Computed Features (features.py:1008,1017)

Fixed in code (Session 16) but v4 training data was generated with the bug. `boost_val > 1` should be `> 0`. A +1 stat boost gave 2x multiplier instead of correct 1.5x. Affects all v4 computed features: matchup scores, damage estimates, KO checks, speed tiers. Impact is narrow (+1 boost only, not ±2-6).

### MEDIUM: Perish Song Urgency Lost (features.py:166-170)

PERISH0 (faint next turn) and PERISH3 (faint in 3 turns) are mapped to PERISH1/PERISH2 slots with value 2.0. This loses the critical distinction between "you faint NOW" and "you have 3 turns to switch." Switching on PERISH0 is meaningless (too late) but switching on PERISH3 is optimal play.

### MEDIUM: Opponent Trapped Flag Always Zero (features.py:1528)

`opp_trapped` is hardcoded to 0.0. The model never knows if it has the opponent trapped (via Arena Trap, Shadow Tag, Magnet Pull, etc.), which is critical for deciding whether to use setup moves or go for the KO.

### LOW: Stat Normalization Inconsistencies (features.py)

- Base power capped at 200, but some moves exceed this (Explosion 250)
- PP normalized by /24.0, but PP-Up maxes reach 32-40
- Stat normalization uses /200.0, but Pokemon with 150+ base stats can produce values >1.0
- These create minor information loss at the extremes

### LOW: Mini Payloads Use Useless Hash Sizes (features.py:242-270)

The `_mini_move_payload_from_meta()` function encodes the opponent's last action into 22 dims. The move ID is hashed into **4 bits** (for ~900 unique moves) — 100% collision, no information content. The species hash (8 bits for ~1025 species) is similarly useless. **The information itself is critical** (knowing which move the opponent just used matters for set deduction, choice-lock detection, etc.) — the problem is the encoding is too small to carry it. Fix: replace with integer IDs for learned embeddings, same as the main hash bucket fix.

### NOTE: Modifier Features (Z-move, Dynamax, Mega) — Keep for Multi-Gen

Z-move modifier is always 0 in Gen 9 OU (Z-moves don't exist in Gen 9), and Dynamax is banned in Gen 9 OU. However, **these must be kept** for multi-gen support: Gen 7 uses Z-moves, Gen 8 uses Dynamax, Gen 6 uses Mega Evolution. The project plan targets Gen 4 through latest gen. These features are dead weight in Gen 9 OU training but will be essential when expanding to other generations. No action needed — leave as-is.

---

## v5 Feature Pipeline Rewrite Checklist

When regenerating data for v5, fix ALL of the above in a single pass:

- [x] Replace hash bucketing with integer IDs (prepare for learned embeddings) — DONE (Session 17-18)
- [x] Build vocabulary files: species→ID, move→ID, item→ID, ability→ID — DONE (Session 17)
- [x] Fix Normal→Ghost immunity in type effectiveness chart — DONE (was already in imm dict)
- [x] Fix Normal→Rock/Steel NVE in type effectiveness chart — DONE (Session 24)
- [ ] Fix Trick Room in speed heuristic
- [ ] Fix Perish song urgency encoding
- [x] Encode opponent trapped status from battle state — DONE via trap feature fix (Session 24)
- [x] Fix stat/BP/PP normalization ranges — stat clamp + BP cap unified (Session 24)
- [x] Expand mini payloads to integer IDs (move ID and species currently too small to carry info) — DONE (Session 17-18)
- [x] Keep modifier features (z-move, dynamax, mega) — needed for multi-gen support (Gen 6-8) — N/A (kept as-is)
- [ ] Add observation dimension validation in observer.py before recording
- [ ] Regenerate full dataset with fixed pipeline (features.py + teams_ou.py changes from Session 24)

---

## How Each Stage Was Run (Commands Reference)

All commands run from `pokemon-ai-starter/pokemon-ai/src/` unless noted. Docker Showdown must be running.

### Infrastructure

Three options for running the Showdown battle server, from fastest to most compatible:

**Option A: battle_server.js (RECOMMENDED — fastest, lightest)**
Minimal websocket Showdown server. No Docker needed. ~31 MB per instance.
5.4 g/s with 2 servers × conc=5. Works with all existing Python code via `--server` flag.

```bash
# Requires Node 20+. We have a portable install:
NODE20="tools/node-v20.18.1-win-x64/node.exe"

# Start one server (from src/ directory):
"$NODE20" battle_server.js --port 9000

# Start multiple for parallel training (each ~31 MB):
"$NODE20" battle_server.js --port 9000 &
"$NODE20" battle_server.js --port 9001 &
"$NODE20" battle_server.js --port 9002 &

# Use in Python scripts (zero code changes — just --server flag):
python eval_bc_vs_bots.py --server ws://127.0.0.1:9000/showdown/websocket ...
python observer.py --server ws://127.0.0.1:9000/showdown/websocket ...
python rl_train.py --servers 9000,9001,9002 ...

# Why battle_server.js over Docker:
#   - 2x faster (5.4 g/s vs 2.5 g/s with Docker)
#   - 1/16th RAM (31 MB vs 500 MB per container + Docker Desktop overhead)
#   - No Docker Desktop needed (saves ~1 GB RAM)
#   - Same poke-env code, same features, same concurrency
#   - Supports all formats (gen9ou, gen9randombattle, gen4ou, etc.)
```

**Option B: --direct flag (no server at all)**
Uses battle_worker.js subprocess with stdin/stdout. No server process needed.
1.5 g/s sequential. Good for lightweight testing.

```bash
# No server to start — just add --direct flag:
python eval_bc_vs_bots.py --direct --max-concurrent 1 ...
python observer.py --direct ...
python rl_train.py --direct ...

# Why --direct:
#   - Zero dependencies (no Docker, no server process)
#   - Works with Node 14+ (system node is fine)
#   - Slower than server mode (1.5 g/s vs 5.4 g/s) — no websocket concurrency
```

**Option C: Docker (legacy, most compatible)**
Full Showdown server in Docker container. Battle-tested, most robust.

```bash
# Start Showdown (run once, persists). Image name: pokemon-ai-showdown
docker start showdown
# Or create: docker run -d --name showdown -p 8000:8000 pokemon-ai-showdown

# For multi-server RL, start extra instances on different ports:
docker run -d --name showdown2 -p 8001:8000 pokemon-ai-showdown
docker run -d --name showdown3 -p 8002:8000 pokemon-ai-showdown

# Restart if stale connections (|nametaken| errors):
docker restart showdown

# Why Docker:
#   - Most robust (full Showdown with all features)
#   - Slowest (2.5 g/s) and heaviest (~500 MB + Docker Desktop)
#   - Required for live ladder play (needs full server features)
```

**Benchmark results (Session 24):**

| Setup | Eval speed | Observer speed | RAM per instance |
|-------|-----------|---------------|-----------------|
| battle_server.js (1 srv, conc=10) | 4.7 g/s | 2.8 g/s | 31 MB |
| battle_server.js (2 srv, conc=5) | **5.4 g/s** | — | 31 MB each |
| Docker (1 container, conc=2) | 2.5 g/s | 1.6 g/s | ~500 MB |
| --direct (conc=1) | 1.4 g/s | 2.0 g/s | 60 MB |

```bash
# Check GPU
nvidia-smi
```

### Stage 1: Data Generation (observer.py)

Plays bots against each other, records observations as JSONL files. CPU-only, runs locally.
10 bots (4 basic + 3 smart + MaxBP + SH + Random), all permutations, 200 games per matchup.

```bash
# Generate v5 observations — all bot permutations
# observer.py auto-runs all bot×bot matchups; each batch = 200 games, 10 concurrent
python observer.py \
    --n-games 200 \
    --max-concurrent 10 \
    --format gen9ou \
    --output-dir data/datasets/obs

# Convert JSONL to memmap (faster training I/O)
python convert_jsonl_to_memmap.py \
    --data "data/datasets/obs/*.jsonl" \
    --out-dir data/datasets/memmap
```

**Output**: `data/datasets/memmap/` — obs.npy (1480-dim), entity_ids.npy (84), move_ids.npy (4), switch_ids.npy (5), actions.npy, results.npy, episode_starts.npy
**Result**: 3,944,845 records, 143,212 episodes, 34.7 GB memmap (50.4% wins / 49.6% losses after fix)
**Key config**: v5 obs = 1480 continuous + 84 entity IDs. 30 teams in teams_ou.py.

### Stage 2: Behavioral Cloning (bc_train.py)

Supervised learning — imitate the training data's action distribution. GPU training.

```bash
# Actual command used (Session 19-20):
python bc_train.py \
    --data-format memmap \
    --memmap-dir data/datasets/memmap \
    --device cuda \
    --epochs 60 \
    --batch-size 256 \
    --lr 3e-4 \
    --sched cosine \
    --use-lstm \
    --lstm-hidden 512 \
    --mlp-hidden 512 \
    --mlp-layers 3 \
    --n-entity-ids 84 \
    --embed-dim 32 \
    --advantage-weight 1.0 \
    --w-win 2.0 --w-loss 0.5 \
    --ctx-extra-dim 41 \
    --step-type-bins 3 \
    --val-ratio 0.1 \
    --run-name v5_awbc
# Converged at epoch ~16. Best model saved via EMA.
```

**Current best**: `data/models/bc/v5_awbc-2026-03-15_18-38-23/best.pt` (~3.85M params, LSTM+embeddings)
**Performance**: Random 94%, MaxBP 60%, SH 20%, SmartDmg 26%, Tactical 18%, Strategic 12%. Elo ~1477.

### Stage 3: Offline RL — IQL (iql_train.py)

Implicit Q-Learning on the same memmap data. Learns Q/V networks, extracts improved policy. GPU training.

```bash
# IQL v3 — actual command (Session 21-22):
python iql_train.py \
    --memmap-dir data/datasets/memmap \
    --init-from data/models/bc/v5_vanilla_bc/best.pt \
    --device cuda \
    --epochs 30 \
    --batch-size 64 \
    --lr 3e-4 \
    --beta 10 \
    --tau 0.9 \
    --gamma 0.99 \
    --reward-ko 0.1 --reward-hp 0.05 --reward-terminal 1.0 \
    --patience 25 \
    --val-ratio 0.1 \
    --ctx-extra-dim 41
# Run: v5_iql-2026-03-17_05-36-21/

# IQL v4 — resume from epoch 30, fresh LR, run to 60 (Session 22):
python iql_train.py \
    --memmap-dir data/datasets/memmap \
    --resume data/models/iql/v5_iql-2026-03-17_05-36-21/epoch_030_policy.pt \
    --device cuda \
    --epochs 60 \
    --lr 1e-4 \
    --lr-restart \
    --beta 15 \
    --eval-every 5 \
    --patience 999
```

**Key insight**: Val pi-loss is ANTI-CORRELATED with play strength. Always use eval-based checkpointing.
**Result**: 60 epochs total. Offline RL ceiling — all models Elo 1471-1473 vs bots, h2h compressed 46.9%-54.2%.
**Best init for online RL**: `epoch_022_policy.pt` (best KO ratio 0.96, best bot Elo 1482)

### Stage 4: Online RL — PPO (rl_train.py)

Live reinforcement learning — plays actual games and learns from outcomes. Uses Showdown server.

```bash
# Actual command running (Session 23) — multi-server, curriculum tier 3:
python -u -X utf8 rl_train.py \
    --init-from data/models/iql/v5_iql-2026-03-17_05-36-21/epoch_022_policy.pt \
    --device cuda \
    --servers 8000,8001,8002 \
    --games-per-iter 50 \
    --max-concurrent 10 \
    --n-iters 200 \
    --lr 1e-4 \
    --lr-schedule cosine \
    --ppo-epochs 4 \
    --ent-coef 0.01 \
    --kl-coef 0.1 \
    --dense-rewards \
    --curriculum \
    --curriculum-tier 3 \
    --amp \
    --temperature 1.0 \
    --temp-decay 0.998 \
    --eval-interval 10 \
    --eval-games 50 \
    --save-interval 10 \
    --no-self-play \
    --out-dir data/models/rl
# Run: data/models/rl/ppo_20260318_032552/
# Use -u for unbuffered output. -X utf8 for Windows Unicode.
# --no-self-play: disabled to reduce complexity on first run.
# --curriculum-tier 3: start with all bots (skip easy tiers since IQL init already strong).

# Resume from RL checkpoint:
python -u -X utf8 rl_train.py \
    --resume data/models/rl/ppo_20260318_032552/iter_0100.pt \
    --init-from data/models/iql/v5_iql-2026-03-17_05-36-21/epoch_022_policy.pt \
    --device cuda \
    --servers 8000,8001,8002 \
    --n-iters 100
# Note: --init-from still needed for BC reference model (KL penalty)
```

**Key flags**:
- `--servers 8000,8001,8002`: Round-robins opponent batches across Showdown instances
- `--max-concurrent 15`: Concurrent battles per batch (was 3, now 15)
- `--games-per-iter 50`: Games per iteration (was 100, now 50 for faster feedback)
- `--curriculum`: Start with weak bots, promote to harder ones as model improves
- `--kl-coef 0.1`: Prevents catastrophic forgetting from BC/IQL knowledge
- `--dense-rewards`: KO bonuses + HP potential shaping (critical — sparse doesn't work)
- `--amp`: Mixed precision on GPU (~2x faster PPO updates)

### Evaluation

```bash
# Eval model vs specific bots (e.g. 100 games each)
python eval_bc_vs_bots.py \
    --checkpoint data/models/iql/v5_iql-2026-03-17_05-36-21/epoch_022_policy.pt \
    --n-battles 100 \
    --device cpu \
    --bots "Random,MaxBasePower,SimpleHeuristics,SmartDamage,Tactical,Strategic" \
    --save-replays \
    --replays-root data/replays/eval_ep22

# Head-to-head tournament between multiple checkpoints
python eval_head_to_head.py \
    --checkpoints ckpt1.pt ckpt2.pt ckpt3.pt \
    --names BC ep22 ep30 \
    --n-battles 30 \
    --device cpu \
    --save-replays \
    --replay-root data/replays/h2h

# Bot round-robin (all bots vs each other, no model)
python eval_bots_roundrobin.py --n-battles 30

# Playstyle analysis from replays
python analyze_eval.py --replay-dir data/replays/eval_ep22

# Full evaluation pipeline (bot evals + h2h + playstyle for multiple models)
bash run_full_eval.sh
```

### Monitoring

```bash
# TensorBoard (RL training)
tensorboard --logdir checkpoints/rl/<run>/tb

# Tail RL training logs (PowerShell)
Get-Content -Path training.log -Wait -Tail 20
```

---

## Session Log (continued)

### Session 24 (2026-03-18)
- **Deep codebase audit**: Read and analyzed all 20+ source files, found 27+ bugs across 13 files
- **All files backed up** to `src/_backups_session24/` before any changes
- **HIGH severity fixes (3)**:
  1. **rl_train.py: PPO temperature mismatch** — `old_logp` stored from temperature-scaled distribution but PPO re-forward used unscaled logits. Fixed: store unscaled log_prob during collection (temperature only affects sampling, not stored log_prob). This likely contributed to PPO plateaus.
  2. **rewards.py: KO double-counting** — phi included `alpha*(opp_fainted - our_fainted)` AND event rewards added ko_bonus separately. Removed fainted counts from phi (now HP-only). Also fixed: first-step accumulator exclusion, per-step clipping inconsistency, misleading "potential-based" docstring.
  3. **eval_bc_vs_bots.py: Default host `localhost`** → `127.0.0.1` (Docker IPv6 issue on Windows).
- **MEDIUM severity fixes (14)**:
  4. **features.py: Normal type missing from NVE** — added `"NORMAL": {"ROCK","STEEL"}` to type effectiveness table.
  5. **features.py: BP normalization inconsistency** — unified `_encode_move_compact` from cap=200 to cap=250 (matches slot encoder).
  6. **features.py: Fixed damage detection** — `damage="level"` (Seismic Toss/Night Shade) now detected.
  7. **features.py: Trap feature** — replaced nonexistent `getattr(m, "trap")` with `volatile_status` check for `partiallytrapped`/`no_retreat`/`octolock` + move ID checks.
  8. **features.py: Stat normalization** — added `min(val/255, 1.5)` clamp for actual stats exceeding 255.
  9. **bc_train.py: Value loss masking** — `BCEWithLogitsLoss(0,0)=log(2)` at masked positions. Fixed with boolean indexing `[valid]`. Applied in both training and eval.
  10. **bc_train.py: Label smoothing + advantage weighting** — label smoothing was silently bypassed when advantage weighting was active. Fixed: smoothed targets computed within the weighted path.
  11. **bc_train.py: ctx-extra-dim default** — changed from 51 to 41 (v5 format).
  12. **iql_train.py: Terminal step dense reward** — last valid step got no KO/HP bonus. Fixed: backward-delta at terminal positions captures final-turn KO/HP changes.
  13. **iql_train.py: CosineAnnealingLR T_max** — added warning when T_max mismatches on resume.
  14. **policy_heads.py: Packed sequence support** — LSTM now uses `pack_padded_sequence`/`pad_packed_sequence` when `seq_lens` provided. No-op when seq_lens is None (backward compatible).
  15. **policy_heads.py: Partial slot encoding** — flat head now works with only move_slots OR only switch_slots (previously required both or fell back to linear head).
  16. **policy_rulebots.py: `_best_switch_candidate` type math** — fixed nested loop to compute per-STAB-type effectiveness product, then MAX across opponent types.
  17. **eval_head_to_head.py: Battle timeout** — added `asyncio.wait_for(timeout=300)`.
- **LOW severity fixes (10+)**:
  18. **rl_train.py: Removed unused `mini_batch_size` param** from ppo_update.
  19. **rl_train.py: Deprecated `torch.cuda.amp`** → `torch.amp` API.
  20. **policy_heads.py: Stale comment** — "17 groups" → "19 groups" (two locations).
  21. **policy_rulebots.py: Priority bonus** — changed from `+5.0` (negligible) to `*1.2` (multiplicative).
  22. **eval_bc_vs_bots.py + eval_head_to_head.py + iql_train.py: Event loop leaks** — added `finally: loop.close()`.
  23. **analyze_eval.py: SE/resisted/immune attribution** — track `last_move_player` to attribute effectiveness events to correct player.
  24. **analyze_eval.py: `|drag|` events** — excluded from player switch counts (forced switches, not decisions).
  25. **teams_ou.py: 4 illegal/nonfunctional sets** — Hawlucha Electric Seed→Focus Sash, Pincurchin Recover→Toxic Spikes, Politoed Helping Hand→Icy Wind, Indeedee-F Heal Pulse→Shadow Ball.
  26. **teams_ou.py: Stale comment** — "30 teams" → "70 teams".
  27. **convert_jsonl_to_memmap.py: Multi-file episode warning** — silently dropped rows now logged with episode ID.
  28. **convert_jsonl_to_memmap.py: Standalone verification counter** — fixed from hardcoded -1 to proper incrementing counter.
- **Reward system unification**:
  - **rewards.py complete rewrite** — unified with IQL's formula: `ko_coef * KO_delta + hp_coef * HP_delta + terminal`. Removed: tempo_tax, gamma-discounted potential phi, separate ko_bonus/faint_penalty. Legacy constructor params accepted but ignored.
  - **New defaults**: ko_coef=0.05, hp_coef=0.02, terminal_coef=1.0. Terminal dominates (~75% of total signal vs old ~20%).
  - **rl_train.py**: `--ko-coef`/`--hp-coef` replace `--ko-bonus`/`--faint-penalty`. Legacy `--ko-bonus` still accepted with deprecation warning.
  - **iql_train.py**: Function signature and argparse defaults aligned to 0.05/0.02/1.0.
  - **Rationale**: Old PPO rewards (ko_bonus=0.15 + phi double-count = 0.65 per KO) caused attack spam. RL iter69 degenerated to 94% attack rate, 0% status/recovery. New coefficients: 6 KOs = 0.30 shaping vs 1.0 terminal.
- **RL iter69 evaluation**: Model degraded from IQL ep22 baseline. 12% avg vs smart bots (IQL was ~17-19%). 94.4% attack rate, 0% status, 1.3% recovery. One-move spam (Ting-Lu 99% EQ, Corviknight 100% Brave Bird). Caused by: temperature mismatch + KO double-counting + kl_coef=0.01 (too loose).
- **Bugs introduced and fixed during session**:
  - features.py: `mid` variable used before defined in trap fix (reordered)
  - teams_ou.py: Politoed Toxic illegal in Gen 9 (→ Icy Wind)
  - teams_ou.py: Indeedee-F Mystical Fire illegal in Gen 9 (→ Shadow Ball)
- **NOT changed (would break existing models/data)**:
  - Duplicate `preparing` bit (removing it changes obs dim from 1480 to 1479)
  - Duplicate `alive_counts` in obs and v4 computed features
- **Verification**: All 12 modified files pass comprehensive verification suite (syntax, import, runtime logic, forward pass, mock battles, coefficient alignment, no old params in active code paths).
- **RISK NOTE**: features.py and teams_ou.py changes affect data generation — existing memmap data was generated with old type tables and old team sets. A full data regeneration is needed for these fixes to take full effect in training. However, existing trained models will continue to work since obs_dim=1480 is unchanged.
- **battle_server.js** — Minimal websocket Showdown server (~320 lines):
  - Uses BattleStream + ws package, no auth/chat/sqlite/user management
  - Handles: `/trn` login, `/utm` team set, `/challenge`/`/accept`, `/choose`, `/team` (team preview)
  - Supports all formats (gen9ou, gen9randombattle, etc.) and multiple concurrent battles
  - Requires Node 20+ (portable install at `tools/node-v20.18.1-win-x64/`)
  - **Benchmarks**: 5.4 g/s (2 servers × conc=5), 31 MB/instance, zero Python code changes
  - Bugs fixed during development: missing `/team` handler (team preview hang), observer.py double URL path in `resolve_server`
- **direct_player.py** — poke-env Player subclass with subprocess transport:
  - `DirectClient` replaces `PSClient`, `DirectPlayer` replaces `Player`
  - `patch_to_direct(player)` converts any existing Player instance
  - `direct_battle_against(p1, p2, n_battles)` orchestrator
  - `_safe_handle_battle_message` wrapper prevents double |win| deadlock
  - `--direct` flag added to: observer.py, eval_bc_vs_bots.py, eval_head_to_head.py, rl_train.py
  - 1.5 g/s sequential, no server needed
- **battle_worker.js** — Node subprocess for `--direct` mode, manages BattleStream via JSON stdin/stdout
- **PPO experiments** (Session 24):
  - Run 1 (kl=0.05, 100 games/iter, 200 iters): iter 150 best — discovered Volt Switch (7.5%) + Spikes (4.6%), 23% smart bot WR, 80.5% attack rate (most balanced playstyle). Later degenerated into Roost-stalling (19.9% recovery, 60.5% attack).
  - Key finding: PPO CAN learn beyond IQL but oscillates — needs more games or stopping at the right time
  - Run 2 (resumed from 150, 300 games/iter): killed at iter 170, 20-24% WR, stable
- **New files created**: battle_server.js, battle_worker.js, direct_player.py, direct_env.py, benchmark_direct.py, package.json
- **Node 20 portable**: tools/node-v20.18.1-win-x64/ (downloaded for battle_server.js)

### Session 25 (2026-03-19/20)
- **Deep codebase audit v6**: 40 bugs fixed (11 HIGH, 18 MEDIUM, 11 LOW). See Session 26 header for details.
- **Human replay scraping v1**: gen9ou 1500+ Elo, 99,955 episodes → human_memmap. IQL trained 30 epochs → 15% smart avg (worse than BC). Human strategies don't transfer to our teams.
- **Scraping v2**: gen9ou 1700+ Elo, 200K replays, --log-both → 4.9M records (62 GB JSONL)
- **v6 bot data gen**: 10 strong bot matchups → 2.7M records
- **Combined memmap**: 7.6M records, 277K episodes, 63 GB. Completed successfully before system crash.
- **System crash**: Screen went black (taskbar visible) during/after memmap conversion. Cause: OneDrive syncing 63 GB + heavy I/O + 16 GB RAM exhaustion. Power button shutdown required.

### Session 26 (2026-03-20/21)
- **Recovery**: Verified combined_v6_memmap intact. Deleted source JSONLs (148 GB freed).
- **LSTM IQL on combined data** (v6_iql_combined_bs32): 21 epochs trained, batch_size=32.
  - Terminal reward fix confirmed working (adv_std 0.17-0.20 vs old 0.06)
  - Bot WR flat at 23-28% smart avg across all epochs
  - H2H improves: ep5 53.2% → ep10 58.3% (best) → ep16 55.5% → ep20 57.3%
  - 100-game evals: EP5 28.3%, EP10 24.5%, EP16 27.0%, EP20 23.8%
  - Playstyle: learned recovery (5.5%), pivoting (3.5%), WoW (13%), Stealth Rock (8%) from human data. Later epochs become more positional, drop status.
- **7-way H2H tournament** (100 games/matchup, 2100 games total):
  - ep10 58.3% > ep20 57.3% > ep16 55.5% > BC 51.8% > ep05 50.7% > PPO 48.2% > iql_human 28.2%
- **Research survey** (docs/RESEARCH.md): Metamon (transformers + 18M self-play = top 10%), MIT thesis (PPO+MCTS = rank 8), PokeChamp (LLM minimax = 1300-1500 Elo), VGC-Bench (BC+self-play).
- **v6 plateau analysis**: All approaches converge to 25-30% vs smart bots. Root cause identified:
  1. **Model capacity**: 3.85M LSTM can't learn Pokemon mechanics (type charts, damage calc, Choice-lock, burn halving attack). Smart bots hard-code this. The 512-dim hidden state compresses too much.
  2. **No long-range memory**: LSTM can't recall "opponent used Swords Dance on turn 3" from turn 40. Can't track Choice-lock (same move twice = locked). Can't connect burn to reduced damage over many turns.
  3. **No status reward signal**: Status moves (WoW, Thunder Wave) give 0 immediate reward. Model learns them from human imitation but IQL advantage weighting suppresses them vs direct attacks.
  4. **Self-play data missing**: Bot data provides noisy signal (random wins ≠ good play). Human data provides quality but mismatched teams. Self-play provides adaptive diversity.
- **v7 decision: Transformer architecture**
  - CausalTransformerCore: 6 layers, 4 heads, 512 dim = 20.78M params (5.3x LSTM)
  - Fits 0.92 GB for IQL (3 models + backprop on 6 GB GPU)
  - Causal attention: each turn attends to all previous turns (no compression)
  - Learned positional embeddings, 128-turn context length
  - New default in policy_heads.py. --use-lstm for backward compat.
  - bc_train.py: --use-transformer (default), --n-transformer-layers, --n-heads, --context-length
  - iql_train.py: make_model() reads transformer config from checkpoint, backward compat with old LSTM checkpoints
  - bc_policy_player.py: transformer inference via per-battle history buffer (accumulates past observations, passes full sequence each turn)
- **Memory leak fix**: PSClient._listening_coroutine.cancel() in observer.py + rl_train.py. Each Player creation spawned an uncancelled listen() coroutine on POKE_LOOP, leaking across thousands of batches.
- **Self-play observer support**: observer.py --model name:path registers model checkpoints as bots. Enables self-play data generation with diverse agent populations.
- **Workers finding**: --workers 2 uses ~9.4 GB RAM (each subprocess caches memmap). Kill Adobe/Overwolf/Norton UI first. --workers 0 is safe but 2.5x slower (GPU waits for CPU data loading).
