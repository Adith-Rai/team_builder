# NEXT_SESSION.md — Concrete TODO Order

**Last updated: 2026-04-15 (Session 36 — Exp 2+3+4 complete, safeguards added, ready for next phase)**

This file is the canonical "if you're starting a new session, do these things in this order"
reference. **It is intentionally self-contained** — a future session reading only this file
plus `docs/RESEARCH.md` §0 should have full context to execute every pending task without
asking for re-explanation.

If you read nothing else, **read this top-to-bottom, then `docs/RESEARCH.md` §0**.

---

## Where things stand right now (Session 36 end — 2026-04-15)

- **Current all-time best checkpoint:** `selfplay_v9_20260413_061236/snapshot_2979.pt` at
  **Elo 1058** (confirmed by elo_session36_FINAL.json). +30 Elo over Session 35 top (sp1984=1028).
  **Maybe higher:** sp2999 measurement in progress (~90 min) — had savg=61% peak, likely new best.
- **Exp 4 (LR refinement)** revealed: lower LR can hit better peaks (sp2999 savg=61% all-time high)
  but entropy collapse risk is real. Run crashed at iter 3055 via FATAL zero-PPO guard.
  Entropy collapsed 0.84→0.58 over 70 iters despite PFSP + permutation. Safety mechanisms were OFF.
- **Safeguards added (Session 36):**
  - **Adaptive entropy**: raised default thresholds (low=0.65 was 0.55, step=10% was 5%, max=0.08 was 0.06).
    All configurable via `--adaptive-entropy-{low,high,step,max,min}`. Would have caught Exp 4 collapse.
  - **Composite early stopping** (`--early-stop`): tracks rolling-3 savg + per-bot. Stops when last
    `patience` RAW evals show BOTH savg regression AND 3+ bots regressing, OR savg drops >2x threshold
    (handles specialization). 9 unit tests pass including Exp 4 replay.
- **Root cause of drift:** PFSP cumulative WR is stale — 12% of training compute was on opponents
  the model had mastered (recent WR >10% higher than historical cum). Combined with weak ent_coef,
  this caused specialization → entropy collapse.
- **All experiments summary:**
  - Exp 1 (lam 0.75→0.95): +25 Elo
  - Exp 1b/1c (ent=0.03→0.02): +7 Elo, entropy stable
  - Exp 2+3 (permutation + PFSP): peak sp2979 at 1058 (+30 over sp1984)
  - Exp 4 (LR=3e-5, games=400): sp2999 possibly new peak, then collapsed
- **Elo canonical:** `data/eval/elo_session36_FINAL.json` (52 players, 1326 matches).
- **Machine stability:** BSODs pattern ended after March, no issues this session.
- **Stale memmaps**: move_cont_dim=107, switch_cont_dim=28 (expects 109/30). Auto-padded.

### Run history (Sessions 35-36)

| Exp | lam | ent | LR | Iters | Notes |
|-----|-----|-----|------|-------|-------|
| Exp 1 | 0.95 | 0.02 | 1e-4 | 1789-2019 | +25 Elo, peak savg 60% at 2019, entropy decline after |
| Exp 1b | 0.95 | 0.03 | 1e-4 | 2020-2230 | Entropy overshot to 1.06, SP win% declining |
| Exp 1c | 0.95 | 0.02 | 1e-4 | 2120-2544 | Stable entropy, savg ~54.6. BSOD at 2544 |
| Exp 2+3 | 0.95 | 0.02 | 1e-4 | 2540-3199 | Slot permutation + PFSP. Peak sp2979=1058 Elo. Drifted to 1046 by end |
| **Exp 4** | 0.95 | 0.02 | **3e-5** | 2980-3054 | LR refinement + 400 games. sp2999 savg=61% (all-time high). **Entropy collapse → FATAL at 3055** |

### Key checkpoints (with measured Elo)

- BC_base: `data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt` (Elo 817)
- Exp1 best: `selfplay_v9_20260409_080620/snapshot_1984.pt` (Elo 1031)
- Exp 1c final: `selfplay_v9_20260411_115905/snapshot_2539.pt` (Elo 1030)
- **Exp 2+3 peak (current best): `selfplay_v9_20260413_061236/snapshot_2979.pt` (Elo 1058)**
- Exp 4 candidate peak: `selfplay_v9_20260415_083340/snapshot_2999.pt` (savg=61%, Elo measuring now)
- Exp 4 pre-crash: `selfplay_v9_20260415_083340/snapshot_3054.pt` (ent=0.66, collapsing)
- **Exp1c latest (resume from here):** `selfplay_v9_20260411_115905/snapshot_2539.pt`

## The Session 33 Elo result — read this carefully

The extended ladder (38 players, 703 matches, anchored SH=1000):

```
 #1  snapshot_1784      Elo 1032  [1009-1055]   ← latest, top of plateau
 #2  pre_crash_1724     Elo 1027  [1005-1049]
 #3  iter1739_eval      Elo 1021  [ 998-1044]
 #4  dip_1349           Elo 1018  [ 997-1041]   ← MID-ERA tied with the early "good" snapshots
 #5  snapshot_0824      Elo 1018  [ 994-1040]
 #6  snapshot_0589      Elo 1015  [ 990-1037]
 #7  snapshot_1599      Elo 1014  [ 991-1036]
 ...
#11  Tactical           Elo 1000  [top bot]
#12  SH                 Elo 1000  [anchor]
 ...
#16  peak_0699          Elo  998  [ 976-1020]   ← old "all-time peak" — middle of pack
 ...
#30  BC_base            Elo  806  [ 784- 830]   ← starting point
 ...
#38  Random             Elo  444  [ 403- 480]   ← floor anchor
```

**Key findings (these drive every decision below):**

1. **PPO improved over BC by +226 Elo.** BC_base 806 → snapshot_1784 1032. ~78% expected
   win rate. PPO is doing real work. The pipeline is not broken.

2. **The "all-time peak" snapshot_0699 was a smart_avg lie.** Elo 998, middle of pack, BELOW
   most v9 PPO snapshots. The "57% smart_avg peak" we chased for 1000+ iters was variance,
   not strength. **Smart_avg is dead as a primary metric.**

3. **The "plateau" has structure, not zero trend** (per the era trajectory analysis in
   STATUS.md "Session 33 ERA TRAJECTORY ANALYSIS"). The actual trajectory:
   - **E4 (iters 340-699): mean Elo 998** — natural steady-state after the type-eff breakthrough
   - **E6 (iters 724-939): mean 1003** — +5 Elo from S31 stability fixes
   - **E7 (iters 940-1499): mean 990** — **-13 Elo regression caused by S32's many disruptive tweaks**
     (snapshot pool overhauls, lr-restarts, batched temporal experiments, adaptive entropy, reward shaping)
   - **E8 (iters 1500-1784): mean 1018** — +28 Elo recovery from S32 disruption + slow new high
   The improvement rate from E4 onward is ~0.018 Elo/iter — glacial but **not zero**. Latest
   snapshot (snapshot_1784, Elo 1032) is the genuine new high, but most of the apparent gain
   from snapshot_0589 (Elo 1015) is "recovery from S32 disruption + slow trickle." The
   architecture has an asymptotic ceiling around Elo 1018-1032 (multiple stable-era snapshots
   land in this band) — better described as a **noisy band** than a hard wall.

4. **There is no "mid-era dip."** First ladder suggested an iter 880-1500 dip. Extended
   ladder added 7 mid-era snapshots and found `dip_1349` at Elo 1018 (#4 overall, tied with
   the early "good" snapshots). The mid-region oscillates like everywhere else. The plateau
   is uniform noise, not "two peaks separated by a dip."

5. **We're at "barely above bot tier."** Latest snapshot (Elo 1032) is 32 Elo above Tactical
   (top bot, Elo 1000). ~55% expected win rate vs the top bot. Marginal edge, not dominant.

6. **⚠ Cross-reference CORRECTED (Session 35):** VGC-Bench BCFP is +147 above SH on their
   scale; we're +32 above SH on ours. **The real gap is ~115 Elo, NOT 700.** The raw numbers
   (1768 vs 1032) are on incompatible scales (VGC-Bench: Random=1127, SH=1621; ours:
   Random=444, SH=1000). Additionally, OU singles is harder than VGC doubles (longer games,
   more hidden info). The gap is closable with targeted fixes.

**Implication (revised Session 35):** more training time at the current rate (~0.018 Elo/iter)
is still uneconomic. But the lever is **hyperparameter fixes first** (lambda=0.95 is the
#1 anomaly — see RESEARCH.md §0.7), then augmentation, THEN architectural changes or BC
scaling. The original "700 Elo gap → need fundamental overhaul" framing was wrong.

**Lesson from the S32 regression:** plan experiments as separate clean runs. Disruptive
mid-run tweaks (lr-restart, optimizer hyperparam changes, pool composition shifts) cost
measurable Elo. Session 32's many changes cost ~13 Elo on average vs E6 (a "no major changes"
era). For the BC scaling experiments in step (c), use clean fresh runs from the new BC base,
not mid-run hot-swaps.

**Does the model have room to grow with more iters?** Probably yes — the stable-era rate
(~0.018 Elo/iter) is roughly constant between E4->E6 and E6->E8, suggesting the asymptote
isn't fully hit yet. But the rate is 28x slower than the early breakthrough phase, making it
uneconomic compared to architectural changes. Extrapolation: +100 Elo = ~17 days training,
+700 Elo (VGC-Bench territory) = ~4 months continuous. Verdict: **bounded headroom exists,
not worth pursuing when BC scaling is expected to give more improvement per compute unit.**
Full analysis in STATUS.md "Does the model have room to grow" section.

**Visualization tool:** `analyze_elo_trajectory.py` generates the full trajectory plot (PNG +
interactive) from any Elo ladder JSON + era config. Eras are editable at `data/eval/eras.json`.
Run `python analyze_elo_trajectory.py --show` for interactive, or the default saves to
`data/eval/elo_trajectory.png`.

Full research and comparison: `docs/RESEARCH.md` §0.

---

## Order of operations

### Step (a) — Elo ladder ✅ **DONE (Session 33)**

`eval_elo_ladder.py` was built, gauntlet of bugs fixed, permanent fix applied (PlayerPool +
checkpoint cache + JSONL save/resume), and run to completion **twice**: first with 31 players
(465 matches, ~93 min), then extended with 7 mid-era snapshots (38 players, 703 matches, ~57
additional min via JSONL resume).

**Result file:** `pokemon-ai-starter/pokemon-ai/src/data/eval/elo_session33_EXTENDED_FINAL.json`
(canonical). First-run kept at `elo_session33_FINAL.json` for history.

**Headlines:** see "The Session 33 Elo result" section above. The TL;DR is:
- Latest snapshot (1784) is top at Elo 1032
- snapshot_0699 was a smart_avg lie (Elo 998, middle of pack)
- 1200+ iters of training produced ~17 Elo of net change (within noise)
- Architecture ceiling was reached at iter ~590
- We're 32 Elo above the top bot (marginal edge)

**No need to re-run the Elo ladder until after a major experiment** (refactor, multi-gen,
bigger BC, etc.). The current measurement is the canonical baseline that future experiments
will be compared against.

**Skip to step (b) below** unless you want to first review the historical script journey
(it's documented in STATUS.md "Session 33 ELO LADDER" sections).

**[Original step (a) notes preserved for reference, now superseded:]**

**Why we did this:** Session 33 deep research found we're at almost exactly VGC-Bench
BCFP's compute scale (5M states), and they hit 1768 Elo on a HARDER format. We didn't know
our actual Elo. The `smart_avg` metric was already documented (Session 29) as a poor predictor
of H2H strength. The cloud-burst decision is gated on this number — see `docs/CLOUD_DEPLOY.md`.

**Exact command to run** (kill training first; both want the GPU):
```bash
# 1. Stop training (Ctrl-C the running python process or kill via taskkill)
# 2. Make sure 1 battle server is running on port 9000:
#    tools/node-v20.18.1-win-x64/node.exe battle_server.js --port 9000
# 3. Run the ladder:
cd pokemon-ai-starter/pokemon-ai/src
python -u eval_elo_ladder.py \
  --snapshot-glob "data/models/rl_v9/selfplay_v9_*/snapshot_*.pt" \
  --sample-n 20 --bots all --n-games 50 \
  --device cuda --server ws://127.0.0.1:9000/showdown/websocket \
  --out-json data/eval/elo_ladder_session33.json \
  2>&1 | tee elo_ladder_session33.log
```

**What the script does** (full details inside the file's docstring):
- Round-robin tournament: every player plays every other N games. Default N=50.
- Players: 20 evenly-spaced snapshots sampled from the entire `selfplay_v9_*` glob
  (covers all historical runs) + all 10 heuristic bots as anchors.
- All 10 bots: Random, MaxBP, GreedySE, HazardSense, SwitchAwareEscape, SetupThenSweep,
  SH, SmartDmg, Tactical, Strategic. Per Session 23 round-robin, these span ~75% win rate
  spread (Random ~5% to Tactical ~82%) → ~400 Elo of dynamic range.
- **Teams: handcrafted 70 OU via `random_pool_teambuilder()`**, NOT procedural. Lower
  variance per game = tighter signal with fewer games. We TRAIN on procedural for
  generalization but EVAL on handcrafted for measurement clarity.
- Fits Bradley-Terry MLE via Hunter (2004) MM algorithm to compute self-consistent
  player strengths (see "How Elo is computed" section below).
- Anchors SH at Elo 1000 by default so the scale is absolutely interpretable.
- Bootstrap 95% CIs from binomial resampling per matchup (B=200).
- Total players: 20 snapshots + 10 bots = 30. Matchups: C(30,2) = 435.
- Expected runtime: ~2-3 hours at 50 games/matchup on one server with concurrency 10.
  Bump to 30 games/match for ~70 min if you want a quick first pass.
- Output: text Elo table + JSON dump with full results.

**Parallelization (built in via `--shard i/N`):** the script supports auto-partitioning
across N shards. Run 3 instances simultaneously on the 3 battle servers for ~3× speedup:

```bash
# Terminal 1
python -u eval_elo_ladder.py --snapshot-glob "data/models/rl_v9/selfplay_v9_*/snapshot_*.pt" \
  --sample-n 20 --bots all --n-games 50 \
  --server ws://127.0.0.1:9000/showdown/websocket \
  --shard 0/3 --out-json data/eval/elo_shard0.json

# Terminal 2 (port 9001, --shard 1/3, --out-json elo_shard1.json)
# Terminal 3 (port 9002, --shard 2/3, --out-json elo_shard2.json)

# When all 3 shards finish, combine into the final ladder:
python -u eval_elo_ladder.py --combine \
  data/eval/elo_shard0.json data/eval/elo_shard1.json data/eval/elo_shard2.json \
  --out-json data/eval/elo_ladder_session33.json
```

Shard partition is deterministic and balanced — each pair `k` goes to shard `k % N`, so
matchup load is evenly split AND each shard sees a representative slice of player types
(rather than e.g. shard 0 only doing snapshot-vs-snapshot).

Verified: 30 players × 435 matchups → 145/145/145 split across 3 shards, no overlap, no
missing pairs. Combine path verified end-to-end on synthetic data (Session 33).

**Time table for parallelized run** (30 players, 50 games/match, 3 servers):

| `--n-games` | Single server | 3 shards (parallel) |
|---|---|---|
| 30 | ~75 min | ~25 min |
| 50 | ~2.5 hr | **~50 min** ← recommended first run |
| 100 | ~5 hr | ~100 min |
| 200 | ~10 hr | ~3.3 hr |

**Recommendation:** start with `--n-games 50 --shard i/3` (3 parallel terminals, ~50 min total).
Each player participates in ~29 matchups, so even at 50 games/match each player has ~1450
games of evidence behind their Elo. If CIs come back wider than ±50, bump to 100. 200 is
overkill — reserve for the official cloud-burst baseline.

#### Snapshot selection cadence — when to run, which snapshots to include

The Elo ladder is a measurement tool, not a continuous metric. Run it when you need a number;
don't try to track it on every iter (variance per re-run + cost). Three modes for different
purposes:

| Use case | Mode | Selection | Cost |
|---|---|---|---|
| **First measurement** (Session 33 step a) | History sweep | `--snapshot-glob "selfplay_v9_*/snapshot_*.pt" --sample-n 20` | ~50 min parallel |
| **After step (c) experiment** | Milestone | `--snapshots <baseline.pt> <experiment.pt>` + bots | ~15 min parallel |
| **Cloud burst tracking** | Recent-focus | `--snapshots <last 10 snapshots from cloud>` + bots | ~25 min parallel |
| **Ongoing during long runs** | Periodic | History sweep every ~500 iters | as above |

**Don't:** run the Elo ladder every 20 iters (the current bot-eval cadence). It's expensive,
adds variance per re-run, and the signal you'd track is similar to the trajectory you can see
in retrospect from a single sweep.

**Do:** run it (a) once now to establish baseline, (b) after each step (c) experiment to
measure delta, (c) periodically during cloud burst (every 6-12 hours), (d) once at the end
of cloud burst as the official result.

**Snapshot save cadence in training is unchanged:** snapshots still save every 5 iters
(`--snapshot-interval 5`), bot eval runs every 20 iters (`--eval-interval 20`). Those are
the existing knobs, the Elo ladder is a separate manual process layered on top.

**Success criterion for this step:**
- Latest snapshot has a measured Elo with bootstrap CI ±50
- All 10 bot anchors have measured Elos (absolute reference)
- snapshot_0699 (the unbeaten 57% smart_avg peak) has a measured Elo for direct comparison
- The Elo trajectory across 20 sampled snapshots tells you whether training is monotonically
  improving (which `smart_avg` cannot reveal due to the documented variance issues)

### How Elo is computed (the algorithm details)

The script implements the standard Bradley-Terry → Elo pipeline. Here's exactly what happens:

**Step 1 — Build win matrix.** After running all matchups, we have for each pair (i, j):
- `wins[i][j]` = number of times i beat j
- `games[i][j]` = total games between i and j (= games[j][i])

**Step 2 — Fit Bradley-Terry model via Hunter (2004) MM algorithm.**

The Bradley-Terry model assumes each player has a "strength" parameter `π_i > 0`, and:
```
P(i beats j) = π_i / (π_i + π_j)
```

The MLE for π given win counts has no closed form, but Hunter's MM (Minorization-Maximization)
algorithm gives a simple iterative update that always converges:
```
π_i_new = W_i / Σ_j (N[i][j] / (π_i_old + π_j_old))
```
where `W_i` is the total wins by player i across all opponents, and `N[i][j]` is the symmetric
game count. After each iteration, normalize so Σπ = n (BT is identifiable only up to scale).
Converges in ~50-200 iterations for our scale; we cap at 1000 with a 1e-7 tolerance.

**Step 3 — Convert BT strengths to Elo.**
```
elo_i = 400 * log10(π_i) + offset
```
The 400/log10 constant is the standard Elo definition. A 400-point Elo advantage means 10×
the BT strength, which corresponds to an expected win rate of `1/(1+10^-1) = 90.9%`. A
200-point advantage = √10× strength = ~76% expected win rate. A 100-point advantage = 64%.

**Step 4 — Anchor scale.** Pick a reference player (default SH = 1000 Elo). Compute
`offset = 1000 - elo[SH]`, add to all players. This pins the scale to a known baseline so
the resulting Elo numbers have absolute meaning. Without an anchor, BT only gives RELATIVE
strengths between players, which is uninformative if you only have your own snapshots.

**Step 5 — Bootstrap 95% CIs.** For B=200 iterations:
- Resample each matchup's outcomes binomially: `new_p1_wins ~ Binomial(N, observed_p_win)`
- Refit BT on the resampled win matrix
- Recompute Elos with the same anchor
- Take the 2.5/97.5 percentiles of each player's Elo distribution as the 95% CI

This captures the sampling noise from N games per matchup. With N=50 games, expect ±30-60
Elo CI on competitive players (those near the median strength of the field). With N=100,
expect ±20-40. Bot anchors (which play many games via their many matchups) get tighter CIs.

**Why Bradley-Terry over per-pair win rates:**
Pairwise win rates ignore opponent strength. Beating a strong player 60% is much more
impressive than beating a weak one 60%. BT solves the entire league simultaneously and
produces a self-consistent strength estimate where each player's number reflects who they
beat AND who those opponents also beat. This is the same approach used by Elo, Glicko,
BayesElo, and TrueSkill — BT is the underlying math that all of these are variants of.

**Why we need bot anchors (and many of them):**
A pure snapshot-vs-snapshot Elo ladder gives ONLY relative strengths. The output number
could be anywhere — Elo 1000 in the snapshot ladder means nothing absolute. Bots fix this:
- SH at 1000 (anchor) gives absolute scale
- The other 9 bots span ~75% win-rate spread (Random ~5% vs Tactical ~82%), which translates
  to ~400 Elo of dynamic range. This wide range lets the snapshots interpolate accurately.
- If we used only SH as the anchor, snapshots much stronger or weaker than SH would have
  imprecise relative position. Having Tactical at ~1200 (top) and Random at ~600 (floor)
  gives the BT solver real reference points across the full strength spectrum.

**Cross-format comparability caveat:**
Even with bot anchors, our resulting Elo is NOT directly comparable to VGC-Bench's published
1768 Elo. They use different anchors (different bots, different format). The two scales
differ. What IS meaningful: the SHAPE of the curve (is our latest dramatically better than
our earliest? are we stronger than the smart bots? by how much?) and the rough order of
magnitude relative to the anchor spread.

### Step (b) — Code refactor + smoke test ✅ **DONE (Session 34)**

Completed in Session 34. See "Session 34 refactor summary" section above for full details.
All files renamed, monolith decomposed, code deduplicated, dead code removed, multi-gen
plumbing added, FormatConfig created, --format flags wired, git initialized. Smoke tested.

**Goal:** Make the codebase A/B-test-friendly and crash-safe so steps (c1)–(c5) can be
implemented as small clean diffs rather than 200-line edits to a 1900-line file.

**Why this matters before step (c):** the planned experiments (head count A/B, capacity
redistribution, ff_dim audit) all need to pass new flags through ~6 classes in 2 files.
With the current monolith, each experiment becomes a multi-hundred-line edit that's hard
to review and easy to break. With the refactor, each becomes a small change to one focused
module.

#### b1 — Decompose `rl_train_v9.py` (1900 lines → ~5 files)

**Current state** (verified Session 33 via `grep -n "^class\|^def main"`):
- `rl_train_v9.py` lines 66-272: `InferenceBatcher` class
- lines 280-525: `V9RLPlayer` class
- lines 530-590: `SelfPlayOpponent` class
- lines 595-720: `collect_v9` async function + helpers
- lines 730-815: `BackgroundCollector` class
- lines 820-900+: `InferenceServer` class (multiprocess)
- lines 1000+: `MPRLPlayer` class (multiprocess worker)
- lines 1250+: `MPPipelineCollector` class
- lines 1500-1899: `main()` — argparse + main training loop
- Imports `ppo_update_v8`, `V8Trajectory`, `compute_gae`, `build_ppo_episodes`,
  `save_v8_checkpoint` from `rl_train_v8.py` (lines 50-52)

**Proposed decomposition** (target file sizes in parens):
- `pokemon-ai-starter/pokemon-ai/src/inference_batcher.py` (~250 lines):
  Move `InferenceBatcher` class (rl_train_v9.py:66-272). It's already self-contained
  — only depends on `PokeTransformer` and torch. Trivial to extract.
- `pokemon-ai-starter/pokemon-ai/src/rl_player.py` (~350 lines):
  Move `V9RLPlayer` (rl_train_v9.py:280-525) and `SelfPlayOpponent` (530-590).
  Imports `InferenceBatcher`, `RewardShaper`, `V8Trajectory`, `BCPolicyPlayerV8`.
- `pokemon-ai-starter/pokemon-ai/src/rl_collection.py` (~200 lines):
  Move `collect_v9` async function and helpers. Imports `V9RLPlayer`, `SelfPlayOpponent`.
- `pokemon-ai-starter/pokemon-ai/src/rl_pipeline.py` (~600 lines):
  Move `BackgroundCollector`, `InferenceServer`, `MPRLPlayer`, `MPPipelineCollector`.
  All multiprocess + threaded collection infrastructure in one place.
- `pokemon-ai-starter/pokemon-ai/src/rl_train_v9.py` (REDUCED to ~400 lines):
  Keep only `main()` + argparse + the training loop body. Imports from all modules above.
  All existing CLI flags and behavior preserved exactly.

**Key invariant during refactor:** the existing resume command from `MEMORY.md` MUST work
unchanged with no flag or behavior changes. Verify by:
1. Running training for 1 iter with old `rl_train_v9.py` (current monolith), save log
2. Running training for 1 iter with new module layout, save log
3. Diff the FLOW lines and loss values — should match within float precision

**Caveat:** the multiprocess code in `mp_collect_v2.py` and `mp_collect_v3.py` ALSO has
classes that may overlap with the rl_pipeline.py extraction. Don't merge those — leave
mp_collect_v2.py / mp_collect_v3.py as-is. They're standalone-tested and not in the main
training path. The refactor is ONLY about rl_train_v9.py's monolith.

#### b2 — Move v7 legacy files to `legacy/`

**Files to move** (verified v7 path, not imported by current v8/v9 training):
```bash
cd pokemon-ai-starter/pokemon-ai/src
mkdir -p legacy
git mv features.py policy_heads.py bc_train.py bc_policy_player.py legacy/  # v7 versions
# observer.py is v7+v8 (--v8 flag) — KEEP at top level, do NOT move
# rl_train.py is v5/v6 PPO — move to legacy
git mv rl_train.py iql_train.py legacy/  # if they still exist
```

Verify nothing in the current training path imports them:
```bash
grep -rE "from features import|from policy_heads import|from bc_train import" \
  *.py | grep -v legacy
# should be empty
```

#### b3 — Prune obsolete backups

Backups are tiny (~400KB each, source-only) so this is symbolic. **Already documented in
`backups/README.md` (created Session 33).** The README marks `v9_pre_cloud/` as the canonical
fallback. No actual deletion needed unless you want to. If you do delete:
```bash
cd pokemon-ai-starter/pokemon-ai/src/backups
# Keep: v9_pre_cloud (current fallback), v8_source_backup (oldest reference)
# Delete (symbolic, ~1.4MB total):
rm -rf v9_session32_final v9_pre_batch_temporal v8_pre_switch_offensive
```

#### b4 — Add `test_smoke_train.py`

**Goal:** 60-second integration test that catches the class of bugs that Session 33's
zero-PPO incident would have caught at commit time instead of 16 hours into a run.

**Sketch** (`pokemon-ai-starter/pokemon-ai/src/test_smoke_train.py`):
```python
"""60-second smoke test for the v9 training loop. Run before any commit that touches
rl_train_v9.py, rl_train_v8.py, ppo_update_v8, or features_v8.py.

Usage:
  python test_smoke_train.py            # runs 1 iter, asserts everything works
  pytest -k smoke                        # if pytest is set up
"""
import subprocess, sys, json, os
from pathlib import Path

def test_one_iter():
    # Assumes battle server already running on port 9000
    # Run 1 iter from a known-good checkpoint with minimal games
    cmd = [
        "python", "-u", "rl_train_v9.py",
        "--init-from", "data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt",
        "--device", "cuda", "--servers", "9000", "--fp16",
        "--games-per-iter", "20",  # minimum viable
        "--max-concurrent", "5",
        "--n-iters", "1",
        "--warmup-iters", "0",
        "--reward-style", "terminal",
        "--ent-coef", "0.04",
        "--out-dir", "data/models/rl_v9/smoke_test",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, f"smoke train failed:\n{out.stderr[-2000:]}"

    # Assert non-zero loss in the iter line
    iter_lines = [l for l in out.stdout.splitlines() if "] Iter " in l]
    assert iter_lines, f"no Iter line in output:\n{out.stdout[-1000:]}"
    assert "pi=0.0000 v=0.0000 ent=0.0000" not in iter_lines[-1], \
        f"zero PPO update — the bug Session 33 fixed:\n{iter_lines[-1]}"

    # Assert snapshot saved
    snap = list(Path("data/models/rl_v9/smoke_test").glob("snapshot_*.pt"))
    assert snap, "no snapshot saved after 1 iter"

    print("SMOKE TEST PASSED")

if __name__ == "__main__":
    test_one_iter()
```

**Why this exact set of asserts:** they cover the three failure classes that have eaten
the most debugging time historically:
1. `pi=0.0000 v=0.0000 ent=0.0000` — the Session 33 zero-PPO failure
2. No snapshot saved — the "steps < 100" gate or "n_succeeded == 0" gate firing
3. Non-zero exit code — any uncaught exception

#### b5 — Switch ad-hoc print prefixes to `logging` module

**Current state** (`grep -n "^\s*print(f\"  \[" rl_train_v9.py | wc -l` ≈ 50+):
- `[FLOW HH:MM:SS +N.Ns]` — pipeline timing
- `[PROF wave N]` — batch profiling
- `[INFO]` — checkpoint expansion, init messages
- `[ERROR]` — caught exceptions
- `[NaN-DIAG]` — NaN diagnostic from inference batcher
- `[FATAL]` — Session 33 zero-PPO guard
- `[TAINTED]` — NaN trajectory discard
- `[ENT]` — adaptive entropy adjustments

**Replace with** `logging.getLogger("pokemon_ai")` calls at appropriate levels:
- `INFO`: FLOW, INFO, snapshot saved, eval results
- `WARNING`: NaN-DIAG, TAINTED (recoverable issues), entropy adjustments
- `ERROR`: caught exceptions
- `CRITICAL`: FATAL guard

Keep human-readable format via a custom Formatter that preserves the bracket prefixes.
Optional: also write a parallel JSONL structured log so future sessions can `jq` historical
runs.

**Success criterion (whole step b):** old training command still produces equivalent output
(same FLOW timing, same iter results) and the smoke test passes.

### Step (b) completion notes

**Completed Session 34.** The modular file structure is now in place. Step (c) experiments
can be implemented as small clean diffs to focused modules instead of 200-line edits to
a 1900-line monolith. `--format` flags are wired, FormatConfig is the single source of
truth for magic numbers, and build_turn_batch/action_to_order are shared (not duplicated).

### ⚠ Step (c-pre) — Session 35 Hyperparameter Experiments (NEW, BEFORE multi-gen)

**Status: NOT STARTED.** Added by Session 35 deep audit. The corrected Elo gap (~115, not
700 — see "Session 35 Elo scale correction" above) means cheap hyperparameter fixes should
be tested BEFORE expensive multi-gen/BC-scaling work. If these close most of the gap,
the multi-gen plan proceeds on a stronger foundation.

**PPO hyperparameter comparison (Session 35 audit finding):**

| Parameter | **Ours** | **VGC-Bench** | **ps-ppo** | **OpenAI Five** |
|-----------|----------|---------------|------------|-----------------|
| gamma | 0.9999 | 1.0 | 0.99 | 0.9998 |
| **lambda** | **0.75** | **0.95** | **0.95** | **0.95** |
| clip | 0.2 | 0.2 | 0.2 | 0.2 |
| **ent_coef** | **0.04** | **0.001** | **0.01** | **0.01** |
| PPO epochs | 5 | 10 | 3-4 | 4 |
| lr | 1e-4 | 1e-5 | 2.5e-4 | ~1e-4 |

**Lambda = 0.75 is the single most anomalous parameter.** Every published system uses 0.95.
At 0.75, GAE advantage estimates are heavily myopic in a 30-60 turn game with terminal reward.

**Entropy history caveat:** at ent=0.02 during early training (iters 159-199), entropy
collapsed to 0.51. Win rates didn't change (plateau was architectural, not entropy). But
reduce cautiously: try 0.02 first, not 0.01.

#### Pre-flight verification checklist (Session 35 code changes)

Session 35 made infrastructure fixes that need quick verification before Exp 1.
Run these ONCE before starting any training. Should take ~5 minutes total.

```bash
cd pokemon-ai-starter/pokemon-ai/src

# 1. Verify all files import cleanly (no broken deps)
python -c "from ppo import compute_gae, save_checkpoint; print('ppo OK')"
python -c "from dataset import MemmapDataset, collate_seq; print('dataset OK')"
python -c "import train_bc; print('train_bc OK')"
python -c "import train_rl; print('train_rl OK')"

# 2. Start 1 battle server
tools/../node.exe battle_server.js --port 9000 &

# 3. Run 2 iters of training with NEW hyperparams — verify output matches expected format
python -u train_rl.py --init-from data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt \
  --resume data/models/rl_v9/selfplay_v9_20260408_042048/snapshot_1784.pt \
  --device cuda --servers 9000 --fp16 \
  --games-per-iter 20 --max-concurrent 5 --n-iters 2 --warmup-iters 0 \
  --reward-style terminal --lam 0.95 --ent-coef 0.02 --grad-accum 1 \
  --out-dir data/models/rl_v9/verify_s35 \
  --procedural-teams C:/Users/raiad/OneDrive/Desktop/team_builder/raw_data/pokemon_usage/2024-04

# VERIFY in output:
# - [FLOW] lines appear (collection + PPO update working)
# - Iter line shows non-zero pi/v/ent values (NOT "pi=0.0000 v=0.0000 ent=0.0000")
# - "lam=0.95" appears in config printout (not 0.75)
# - "ent=0.02" appears in config printout (not 0.04)
# - Snapshot saved without errors
# - No NaN warnings, no traceback output
# - Entropy value reported (should be 0.5-1.0 range, watch for collapse toward 0)

# 4. Verify checkpoint round-trip (atomic write)
python -c "
import torch, os
p = 'data/models/rl_v9/verify_s35'
snaps = [f for f in os.listdir(p) if f.startswith('snapshot_')]
if snaps:
    ckpt = torch.load(os.path.join(p, snaps[0]), map_location='cpu', weights_only=False)
    print(f'Checkpoint OK: iter={ckpt[\"iteration\"]}, keys={list(ckpt.keys())}')
    assert not os.path.exists(os.path.join(p, snaps[0] + '.tmp')), 'Temp file leaked!'
    print('Atomic write OK')
else:
    print('WARNING: no snapshot saved — check n-iters and output')
"

# 5. Clean up verification run
# rm -rf data/models/rl_v9/verify_s35  # optional, small
```

**If anything fails:** do NOT proceed to Exp 1. Debug first. The most likely failure is
a battle server not running (ECONNREFUSED) or a stale checkpoint path.

**If all passes:** proceed to Exp 1 below.

#### Exp 1 — Lambda + entropy fix (CLI flags only)

**Status: RUNNING (Session 35, started 2026-04-09 ~08:10)**
Log file: `pokemon-ai-starter/pokemon-ai/src/exp1_lambda095.log`
Run dir: `data/models/rl_v9/selfplay_v9_20260409_075821/`

**Starting battle servers** (must be in separate persistent terminals or detached):
```bash
# Option A: Detached processes (from any terminal, survives terminal close)
cd C:\Users\raiad\OneDrive\Desktop\team_builder\pokemon-ai-starter\pokemon-ai\src
cmd.exe /c "start /B C:\Users\raiad\OneDrive\Desktop\team_builder\tools\node-v20.18.1-win-x64\node.exe battle_server.js --port 9000"
cmd.exe /c "start /B C:\Users\raiad\OneDrive\Desktop\team_builder\tools\node-v20.18.1-win-x64\node.exe battle_server.js --port 9001"
cmd.exe /c "start /B C:\Users\raiad\OneDrive\Desktop\team_builder\tools\node-v20.18.1-win-x64\node.exe battle_server.js --port 9002"

# Verify:
curl -s -o /dev/null -w "9000:%{http_code} " http://127.0.0.1:9000 && \
curl -s -o /dev/null -w "9001:%{http_code} " http://127.0.0.1:9001 && \
curl -s -o /dev/null -w "9002:%{http_code}\n" http://127.0.0.1:9002
# Should show: 9000:200 9001:200 9002:200
```

**Training command:**
```bash
python -u train_rl.py --init-from data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt \
  --resume data/models/rl_v9/selfplay_v9_20260408_042048/snapshot_1784.pt \
  --device cuda --servers 9000,9001,9002 --fp16 --pipeline \
  --games-per-iter 200 --max-concurrent 50 --n-iters 200 --warmup-iters 0 \
  --reward-style terminal --lam 0.95 --ent-coef 0.02 --grad-accum 1 \
  --procedural-teams C:/Users/raiad/OneDrive/Desktop/team_builder/raw_data/pokemon_usage/2024-04 \
  2>&1 | tee exp1_lambda095.log
```

**Session 35 findings on concurrency and speed:**
- `--max-concurrent 50` is ~3-4x faster than the old default of 10. With 50 concurrent
  battles, inference batcher processes 5-7 requests per GPU forward call (vs 1 at conc=10).
  GPU time goes from 11ms to 20ms per batch but does 5-7x more work per call.
- 3 servers + 50 concurrent: ~4 min/iter. At 200 iters ≈ **~13 hours total.**
- Previous training used `--max-concurrent 10` → ~4.5 min/iter with 3 servers.
- **Use `--max-concurrent 50` going forward** unless OOM occurs.

**FP16/FP32 note:** Spatial encoder runs in FP16 (via autocast). Temporal transformer runs
in **FP32** explicitly (`all_summaries.float()` in inference_batcher.py:169). This is
intentional — FP16 accumulation errors compound across 200 turns of causal attention.
Summaries stored in history are always FP32.

**Exp 1 results (215 iters: 1785-1999, measured by Elo ladder):**

Elo result file: `data/eval/elo_session35_exp1.json` (33 players, 528 matchups, 100 games each)

| Era | Mean Elo | Snapshots | vs Previous |
|-----|----------|-----------|-------------|
| E4 Stability | 1015 | 2 | — |
| E7 S32 disruption | 1001 | 5 | -14 |
| E8 S33 stable | 1003 | 2 | +2 |
| **E9 Exp1 lam=0.95** | **1019** | **8** | **+16** |

Key: **sp1784 (baseline) = Elo 1003, sp1984 (Exp1 best) = Elo 1028, delta = +25 Elo.**
Below +50 threshold but **E9 is the highest era mean ever** (1019). 5 of top 8 players
are Exp 1 snapshots. Improvement rate ~0.12 Elo/iter (8x faster than old 0.014 Elo/iter).

**The trajectory was still climbing when we stopped.** First-half Exp1 mean 1015 → second-half
1023 = +8 within the run. Self-play win rate still >53% (model still improving vs pool).
**Training resumed for 500 more iters (2000-2499) to see if trajectory continues.**

**Smart_avg was flat at ~52% across E4-E9** while Elo showed real structure. Smart_avg
is indicative (trends per-bot are real signal) but saturates — can't resolve 990 vs 1030.
Elo is the primary metric. Smart_avg is a secondary check.

**Exp 1 (ent=0.02) result:** Peak at iter 2019 (savg=60%, all-time high). Then entropy
drifted from 0.84 to 0.52 and evals declined: 60→57→56→53→58→55→53→52. Entropy=0.02
was too aggressive — productive sharpening initially, then over-sharpening.

**Exp 1b (ent=0.03) result:** Branched from iter 2019 peak. Ran 210 iters (2020-2230).
Entropy recovered and stabilized (0.77→1.06) but OVERSHOT — became too exploratory.
SP win% was actually declining (-3.6/100it). Mean savg=54.6 (good) but std=1.4.
Decision: ent=0.03 too high, switch back to ent=0.02 from a point with healthy entropy.

**Exp 1c (ent=0.02, from iter 2119) — BEST RUN. Completed 425 iters (2120-2544).**
Resumed from Exp 1b's snapshot_2119 (where entropy was healthy at ~0.93) with ent=0.02.
Log: `exp1c_ent02_from2119.log`. Crashed at iter 2544 (Windows BSOD, not training bug).

**Exp 1c analytics (from `analyze_experiments.py`):**

Cross-experiment comparison (smart_avg):
```
                           Experiment    N   Mean   Std    Min    Max   <52%   >=56%
  Pre-Exp (lam=0.75, ent=0.04)          9   52.9   1.6   50.2   55.0   33%     0%
  Exp 1 (lam=0.95, ent=0.02)           11   52.4   1.9   48.8   56.4   45%     9%
  Exp 1b (lam=0.95, ent=0.03)          10   54.6   1.4   52.5   57.2    0%    20%
  Exp 1c (lam=0.95, ent=0.02)          21   54.6   2.0   50.8   58.4   10%    29%
```

Per-bot means across experiments:
```
       Bot       Pre-Exp    Exp 1     Exp 1b    Exp 1c
       SH        55.8±3.0   52.9±4.0  56.8±3.4  56.1±3.3
       SmartDmg  53.5±3.1   52.9±2.9  54.6±3.7  55.3±3.3   ← most improved
       Tactical  50.5±2.0   52.0±3.9  53.6±3.0  54.1±3.9   ← was weakest, now solid
       Strategic 52.0±3.8   51.6±4.1  53.4±3.3  53.0±3.6   ← stubborn, still weakest
```

Key findings:
- **+2.2 savg** over Pre-Exp/Exp 1 — real, sustained across 21 eval points
- **0% evals below 50%** (vs 9% Exp 1), only 10% below 52% (vs 45% Exp 1)
- **SH: 13 consecutive evals at 55+** (iters 2199-2439) — longest hot streak ever
- **Tactical: transformed** from 50.5 mean/44% dips to 54.1 mean/5% dips
- **Strategic: still weakest** (24% below 50, 4 consecutive dips at 2159-2219), but
  improved in second half (mean 51.5→54.3)
- **Entropy stable at ~0.85** (no drift, no collapse). Solved the Exp 1 entropy problem.
- **Flat trend** (-0.23/100it). Plateau reached — not declining, not climbing.
- **Bot correlations (Exp 1c):** SH-SmartDmg mildly correlated (r=0.35), Tactical
  negatively correlated with SH (r=-0.19), Strategic uncorrelated with everything.
- **Last 50-100 iters were the strongest self-play window:** SP win% 55.0-55.4,
  52-54% of iters at >=55%. Eval dips at 2519/2539 were noise, not decline.

**CONCLUSION: Hyperparameter experiments (Exp 1/1b/1c) DONE.** Lambda fix from 0.75→0.95
was the main lever (+25 Elo confirmed). Entropy tuning (0.02→0.03→0.02) found the sweet
spot: start with healthy entropy from Exp 1b, then maintain with ent=0.02. No further
gains available from ent tuning. **Proceed to Exp 2 (slot permutation) or Exp 3
(recency-weighted pool).**

Resume command (Exp 2+3: slot permutation + PFSP active):
```bash
# snapshot_2544.pt.tmp already deleted (Session 36)
python -u train_rl.py --init-from data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt \
  --resume data/models/rl_v9/selfplay_v9_20260411_115905/snapshot_2539.pt \
  --device cuda --servers 9000,9001,9002 --fp16 --pipeline \
  --games-per-iter 200 --max-concurrent 75 --n-iters 500 --warmup-iters 0 \
  --reward-style terminal --lam 0.95 --ent-coef 0.02 --grad-accum 1 \
  --procedural-teams C:/Users/raiad/OneDrive/Desktop/team_builder/raw_data/pokemon_usage/2024-04 \
  2>&1 | tee exp2_exp3_pfsp.log
```
Slot permutation + PFSP are active automatically (built into features.py and
rl_collection.py). PFSP warms up over ~50 iters. conc=75 (up from 50).

**Incremental Elo measurement (--add-to mode, ~30 min per snapshot):**
```bash
python eval_elo_ladder.py \
  --add-to data/eval/elo_session35_exp1.json \
  --snapshots data/models/rl_v9/.../snapshot_XXXX.pt \
  --names spXXXX \
  --n-games 100 --concurrency 100 --device cuda \
  --server ws://127.0.0.1:9000/showdown/websocket \
  --out-json data/eval/elo_session35_updated.json
```

**Decision: proceed to Exp 2 + Exp 3 (REVISED).** Hyperparameter experiments raised
the plateau but didn't break through. Session 36 research (web + literature review)
found that (a) our Exp 3 plan of 70/30 recency weighting is suboptimal — PFSP is
strictly better and proven in AlphaStar, and (b) slot permutation is strongly supported
by equivariance theory (E2GN2, NeurIPS 2024: 10x sample efficiency from equivariance).

**Starting checkpoint: `snapshot_2539.pt`** (latest good). Rationale:
- Last 50 iters had strongest SP win% (55.4, 54% of iters at >=55%, only 8% below 50%)
- Entropy healthy at 0.86 — no need to branch from an earlier "healthier" point
- Eval dips at 2519/2539 were noise (training signal was strong underneath)
- Largest pool (847 checkpoints) — more opponents for PFSP to weight
- snapshot_2379 (peak rm3=56.8) is 160 iters behind with a smaller pool — not worth reverting
- We're changing the training dynamics (PFSP + augmentation) so eval at the old regime
  is less relevant than having the strongest underlying policy + biggest pool

#### Exp 2 — Slot permutation augmentation

**What it is:** During training only, randomly shuffle the order of bench Pokemon (slots
1-5) and moves within each Pokemon (4 moves per mon). Active mon stays at slot 0.
This is data augmentation — the model sees more orderings of the same game state.

**Why it helps (concrete):** Our bench Pokemon are sorted by species name in
`features.py:_get_sorted_bench()`. This means Garchomp is ALWAYS at a specific position
relative to other team members. The model can memorize "the Pokemon at position 3 is
usually Garchomp, so attend to position 3 for ground coverage" instead of learning the
general rule "attend to whichever entity provides ground coverage." Shuffling forces
content-based attention instead of position-based memorization.

Self-attention in our spatial transformer IS mathematically permutation-equivariant over
input tokens — but the fixed input ordering breaks this in practice because the model
can learn position-dependent biases through training. Augmentation restores the symmetry.

**Evidence:**
- E2GN2 (NeurIPS 2024): 10x sample efficiency from equivariance in multi-agent settings.
  Architectural equivariance > augmentation, but augmentation is cheap and our entity
  tokenization already handles most of the symmetry.
- No Pokemon-specific paper tests slot permutation. This is our own experiment.
- CONJECTURE: Expected +5-15 Elo. Could be more if position memorization is a bigger
  bottleneck than we think. Could be near-zero if entity tokenization already forces
  content-based attention. We'll know in 200 iters.

**Downsides:**
- Slight training noise increase (same state seen in different orderings = noisier gradients)
- If the model has learned useful positional heuristics (e.g., "slot 1 is the fastest
  bench mon" via species sort), shuffling destroys those. But these are spurious
  correlations — they don't generalize to new teams. Net positive.
- Zero compute overhead (permutation is O(6) per sample).

**Implementation** (in `features.py:build_turn_batch()`):
```python
# After building the 6-pokemon feature arrays, before packing into tensors:
import random
if training:  # only during training, not inference
    # Shuffle ally bench order (slots 1-5, keep slot 0 = active)
    ally_perm = [0] + random.sample(range(1, 6), 5)
    our_pokemon_ids = our_pokemon_ids[ally_perm]
    our_pokemon_cont = our_pokemon_cont[ally_perm]
    # ... same for banks, mcont

    # Shuffle opponent bench order (same pattern)
    opp_perm = [0] + random.sample(range(1, 6), 5)
    # ... same for opp arrays

    # Shuffle move order within each pokemon (4 moves)
    for i in range(6):
        move_perm = random.sample(range(4), 4)
        move_ids[i] = move_ids[i][move_perm]
        move_cont[i] = move_cont[i][move_perm]
```

**Key constraints:**
- Active Pokemon MUST stay at index 0 (model uses position 0 for active mon)
- Bench Pokemon (indices 1-5) can be freely permuted
- Move order within each Pokemon can be freely permuted
- Switch action targets (switch_slots) must be permuted with the SAME permutation as
  bench Pokemon, so "switch to slot 2" still maps to the correct Pokemon after shuffling

**Where to add the `training` flag:** `build_turn_batch()` is called from both RL
(rl_player.py — training) and eval (battle_agent.py — inference). Add a `training=False`
parameter, pass `training=True` from rl_player.py only.

**Also apply in BC training:** the collate path in dataset.py should also shuffle. Add
permutation in `collate_seq()` or in `MemmapDataset.__getitem__()` with a `training` flag.

Implementation: ~2 hours. Run jointly with Exp 3.

#### Exp 3 (REVISED) — PFSP opponent sampling (replaces recency-weighted pool)

**What it is:** Replace uniform random opponent sampling with Prioritized Fictitious
Self-Play (PFSP). Instead of `random.sample(pool, 15)`, weight each pool checkpoint by
how hard it is for the current model to beat, then sample proportionally.

**Why it helps (concrete):** Our pool has 847 checkpoints. With uniform sampling, 15
opponents per iter means each iter samples ~1.8% of the pool. Most of those 847 are
from early training (iters 0-1000) when the model was weak. The current model beats
those easily — training against them produces near-zero learning signal (the model
already knows how to beat them). PFSP focuses training compute on the opponents that
are actually challenging.

Current sampling code (`rl_collection.py:76`):
```python
selected = random.sample(snapshot_pool, max_opponents)  # pure uniform
```

This is the single most wasteful thing in our training pipeline. Every serious self-play
system since AlphaStar (2019) uses prioritized sampling.

**Evidence:**
- AlphaStar (Nature 2019): PFSP is the core of their league training. Priority function
  `f(x) = (1-x)^p` weights harder opponents higher. Used in all three agent types
  (main, main exploiter, league exploiter).
- VGC-Bench (AAMAS 2026): Uses Double Oracle (a PSRO variant with Nash-weighted
  opponent selection). Outperforms uniform fictitious play.
- Self-play survey (Aug 2024): Explicitly identifies opponent sampling as a key lever.
  PFSP strictly dominates both uniform and recency-weighted approaches.
- The original Exp 3 plan (70% recent, 30% old) was a crude approximation of what PFSP
  does properly. Recency is a weak proxy for difficulty — some old checkpoints may
  still be hard (different style), some recent ones may be easy (similar style).

**Downsides:**
- Need to track win rates vs pool checkpoints. This adds bookkeeping but the data is
  already partially available (each iter logs per-opponent results like `sp1784=7/14`).
- Cold start: new checkpoints have no win-rate estimate. Solution: initialize at 50%
  (assume even match) and update after first encounter.
- Staleness: win rates shift as the model improves. Solution: use exponential moving
  average or only count recent encounters (last 5-10 meetings).
- Risk of exploitation narrowing: if the model only trains vs hard opponents, it might
  forget how to beat easy ones. Mitigation: keep a small fraction (10-20%) of random
  uniform samples alongside PFSP-weighted samples.

**Implementation** (in `rl_collection.py`):
```python
# Replace uniform sampling with PFSP
# win_rates: dict mapping checkpoint_path -> EMA win rate (maintained across iters)
# Default 0.5 for unseen checkpoints

def pfsp_sample(snapshot_pool, win_rates, n_opponents, uniform_frac=0.15):
    \"\"\"Sample opponents using PFSP weighting + uniform safety margin.\"\"\"
    n_uniform = max(1, int(n_opponents * uniform_frac))
    n_pfsp = n_opponents - n_uniform

    # PFSP weights: harder opponents get higher weight
    # f(x) = (1 - win_rate)^2  (AlphaStar uses p=2 for main agent)
    weights = []
    for sp in snapshot_pool:
        wr = win_rates.get(sp, 0.5)
        weights.append((1.0 - wr) ** 2)
    total = sum(weights)
    probs = [w / total for w in weights]

    # Weighted sample without replacement
    pfsp_selected = np.random.choice(
        len(snapshot_pool), size=min(n_pfsp, len(snapshot_pool)),
        replace=False, p=probs
    )
    selected = [snapshot_pool[i] for i in pfsp_selected]

    # Add uniform random for diversity / anti-forgetting
    remaining = [sp for sp in snapshot_pool if sp not in selected]
    if remaining and n_uniform > 0:
        selected += random.sample(remaining, min(n_uniform, len(remaining)))

    return selected

# After each iter, update win_rates from per-opponent results:
# ema_rate = 0.3 (weight recent encounters more)
# win_rates[opp] = (1 - ema_rate) * win_rates[opp] + ema_rate * (wins / games)
```

**Always include latest snapshot** in the sample (existing behavior, keep it).

Implementation: ~3 hours (sampling + win-rate tracking + EMA updates). Run jointly
with Exp 2 for 200 iters, then measure.

#### Why implement Exp 2 + Exp 3 together

These changes are orthogonal:
- Exp 2 changes HOW the model sees each game state (augmented input)
- Exp 3 changes WHICH opponents it trains against (harder ones)

Running them together is fine because:
1. Neither requires the other as a baseline
2. Both are cheap to implement (<1 day total)
3. Combined measurement (200 iters + Elo) tells us if cheap fixes work
4. If combined gain is large, we can ablate later to understand which contributed
5. If combined gain is small, we know to move to Exp 5 (capacity reallocation)

The 70/30 recency approach from the old Exp 3 is superseded. Do NOT implement it.

#### Exp 4 — Elo measurement after Exp 2+3

Run incremental Elo via `--add-to` for snapshot_2539 (pre-experiment baseline) and
2-3 post-experiment snapshots. ~30 min each via `eval_elo_ladder.py --add-to`.

**Decision criteria:**
- If total gain >=+30 Elo → cheap fixes working, run 200 more iters
- If total gain >=+50 Elo → gap substantially closed, proceed to multi-gen (Step c)
- If total gain <+30 Elo → move to Exp 5 (capacity reallocation)

#### Exp 5 — Capacity reallocation (if Exp 2+3 insufficient)

Redistribute model capacity: spatial 256d/3L → temporal 512d/3L. Requires BC retrain.

**Why this is the right architectural change (concrete):**
- Metamon allocates 5-8x more capacity to temporal than spatial across ALL model sizes
  (15M: 100d→512d, 50M: 100d→768d, 200M: 160d→1280d). This isn't arbitrary.
- OU singles games are 30-60 turns with heavy hidden information. Long-range temporal
  dependencies (what happened 20 turns ago) dominate per-turn spatial pattern matching.
- Our 1:1 ratio (384d spatial / 384d temporal) means the temporal model is underweight.
- Entity tokenization (the breakthrough) lives in the spatial encoder. Shrinking spatial
  from 384d→256d preserves entity attention — verified in code, the spatial transformer
  processes entity tokens the same regardless of d_model.
- Net param count stays ~13M (spatial shrinks, temporal grows).

**Downsides:** Requires BC retrain from scratch (~days), then PPO from new BC base.

### Session 36 research findings (for future reference)

**New landscape information (April 2026 web research):**
1. **PokeAgent Challenge (NeurIPS 2025)** is now a live benchmark at pokeagentchallenge.com.
   100+ teams, 650+ community. We could submit for ground-truth Elo against Metamon, Foul
   Play, and others. Not in previous docs.
2. **Metamon Kakuna** (57M, Gen9OU) exists but has no published public Elo. Internal only.
3. **Foul Play** (pure MCTS, no neural net) co-won PokeAgent Track 1 alongside PA-Agent (RL).
   Validates MIT thesis: "RL alone plateaus but RL+search breaks through."
4. **No pure RL agent has demonstrated >1800 Gen9 OU on public ladder.** PokeChamp (LLM
   minimax) gets 1300-1500 but times out 1/3 of games. ps-ppo >1900 is Random Battles only.
5. **Top human Gen9 OU: 2030-2115 Elo** (Showdown Glicko-1 + GXE).
6. **EPO (2025)** confirms constant entropy > entropy decay in multi-turn RL. Validates our
   finding that stable ent=0.02 works.
7. **"Self-Play Only Evolves When..." (March 2026)** argues sustained improvement requires
   new learnable signal. PFSP provides this by always presenting challenging opponents.

**Longer-term options (not immediate priority):**
- **Search (MCTS) at inference time:** Highest-ceiling option per MIT thesis + Foul Play
  results. Would require opponent modeling + forward sim via battle server. ~1-2 weeks impl.
  CONJECTURE: could be +200-500 Elo based on MIT result (Rank 8, 1693 Elo with PPO+MCTS
  in Gen4). Risk: inference latency, imperfect info handling.
- **PokeAgent Challenge submission:** Ground-truth benchmarking. Low cost (API wrapper).
- **Scale to 50M+ params:** Metamon uses 57M for Gen9OU. Our 13.38M may be undersized.
  But requires more BC data + cloud compute. This is the Step c/d path.

**What the docs previously got wrong:**
- Exp 3 (70/30 recency) was a weak proxy for PFSP. Literature is clear: win-rate
  weighting > recency weighting. Superseded.
- The "115 Elo gap" framing (vs VGC-Bench) is an estimate across incompatible scales
  and formats. The real benchmark should be public ladder or PokeAgent, not cross-scale
  inference. Treat 115 as directional, not precise.
- Search was mentioned (MIT thesis) but not prioritized. With Foul Play (pure MCTS)
  winning PokeAgent, search deserves higher priority than previously documented.

### Old Exp 3 plan (SUPERSEDED — do NOT implement)

~~Change opponent sampling from uniform to: 70% from last 200 checkpoints, 30% from older.~~

```python
# OLD PLAN — SUPERSEDED BY PFSP (see revised Exp 3 above)
# Do NOT implement this. PFSP is strictly better.
recent_cutoff = max(0, len(snapshot_pool) - 200)
recent = snapshot_pool[recent_cutoff:]
older = snapshot_pool[:recent_cutoff]
n_recent = int(max_opponents * 0.7)
n_older = max_opponents - n_recent
selected = random.sample(recent, min(n_recent, len(recent)))
if older and n_older > 0:
    selected += random.sample(older, min(n_older, len(older)))
```

**Rationale:** OpenAI Five uses 80/20 recent/older split. VGC-Bench uses pure uniform
(which works at their scale). Our pool is ~610 deep — pure uniform wastes ~70% of training
signal on weak old policies. Recency weighting focuses learning on current-strength opponents.

Implementation: ~30 min. Run 200 iters on top of Exp 1+2, measure.
*(This plan was replaced by PFSP — see revised Exp 3 above.)*

---

### Step (c) — Multi-gen vocab prep, BEFORE BC scaling

**Status: NOT STARTED.** ~~This step replaces the original "head-count A/B + filter loosening"
plan.~~ **Session 35 update:** This step is now AFTER the hyperparameter experiments above.
The original "700 Elo gap" justification for prioritizing BC scaling was based on incompatible
Elo scales. The real gap (~115 Elo) may be partially closed by Exp 1-3. Multi-gen remains
the long-term goal regardless.

**User direction (confirmed Session 33):** do multi-gen prep BEFORE the BC scaling test,
not after. Reasoning: multi-gen is the long-term project goal regardless. Doing it before
BC scaling means the scaled BC base is multi-gen-capable from day one and doesn't need to
be redone when we add gens 6/7/8 later. The cost of multi-gen prep is roughly the same
whether done before or after BC training, so do it before.

**Sub-steps in order:**

#### c1 — Multi-gen vocab + feature prep (1-2 weeks)

**What:** expand species/move/ability/item embedding tables to cover gens 6-9, expand
`features_v8._VOLATILE_EFFECTS` for gen-specific volatiles, add per-gen team generators,
gen-aware feature handling shells. Per `memory/project_multigen_plan.md`: "Architecture is
gen-agnostic. Multi-gen is mostly data pipeline work after gen9ou ladder Elo lands." That
data-pipeline work is now this step.

**Where to change:**
- `vocab.py` (or wherever the embedding tables live) — bump table sizes
- `features_v8.py` — add gen 6/7/8 volatiles, abilities, items not present in gen 9
- `team_generator.py` — per-gen support, currently gen9-only
- `policy_heads_v8.py` — verify embedding sizes flow through correctly (architecture itself
  doesn't change)

**Validation:** train a tiny (1 iter) BC sanity check with the expanded vocab on the existing
gen9 data. Confirm the model still trains without errors and the embedding tables aren't
exploding memory.

**See also:** `memory/project_multigen_plan.md` for the full multi-gen TODO breakdown.

#### c2 — Multi-gen replay scrape (1-2 weeks, mostly automated)

**What:** scrape human replays for gens 6/7/8 OU at sufficient ratings (1500+). Add to the
existing `human_v3_memmap` (currently gen9ou 1500+, 200K replays, 10.1M records). Target
sizes:
- gen 6/7/8 OU 1500+: ~100K replays per gen (HuggingFace `jakegrigsby/metamon-raw-replays`)
- combined memmap: ~500K replays, ~25M records, ~50 GB
- This pipeline already exists for gen9, just needs gen extension

**Where to change:**
- `replay_to_memmap_v8.py` — gen filter parameter
- Possibly `replay_parser.py` — gen-specific parsing rules if needed

**Background work, not blocking.** Can run while you do c1.

#### c3 — 30M BC scaling test on multi-gen data (3-7 days)

**What:** train a 30M-param BC model (scaled up from current 13.4M) on the union of
`human_v3_memmap` + new multi-gen replays. **30M is the realistic local ceiling on the 6GB
GPU** — see VRAM math in MEMORY.md or session notes. 50M would need cloud.

**Scaling approach:** spatial encoder dim 384 → 512 OR temporal dim 384 → 512. Adding layers
is more expensive in activations than widening, so prefer widening if VRAM allows.

**Training config:** FP16 mixed precision, batch size 4-8 with gradient accumulation, otherwise
standard BC settings. Run 3-5 epochs.

**Validation:** measure the new BC's Elo via `eval_elo_ladder.py` (with `BC_base` as one of
the players). Target: Elo 900+ (vs current BC_base 806). If the 30M BC base hits Elo 900+,
that's confirmation the scaling lever exists. If it stays at ~810, the lever doesn't exist
locally and we re-evaluate (cloud BC, architecture pivot, etc.).

**Decision:**
- If 30M BC ≥ Elo 900 → continue to c4 (PPO from new BC)
- If 30M BC ≈ Elo 810 → BC scaling isn't the lever. Pivot to architectural changes (head
  count, capacity reallocation, ensemble critic) or cloud BC at 50M+ params.

#### c4 — PPO from the new BC base, plus Elo measurement

**What:** identical to the previous PPO setup but starting from the 30M BC base instead of
the 13.4M one. ~200-500 iters of self-play. Run the Elo ladder against the existing baseline
checkpoints to measure the delta.

**Decision threshold:** does the new PPO base beat snapshot_1784 (Elo 1032) by >50 Elo? If
yes, BC scaling worked and we have a new ceiling to optimize toward. If no, the bottleneck
is upstream of BC base size (architecture, training method, etc.).

#### c5 — (after c4 succeeds) cloud-scale BC at 50M+ params

**Only if c3/c4 confirm scaling helps locally.** Spend $20-50 of A100 time training a 50M+
parameter BC base on the multi-gen data. Same recipe as c3 but with the bigger model size
that doesn't fit on the 3060.

#### c6 — Multi-gen training data extension (the long game)

After 30M+ BC + PPO is validated on gen9 data, gradually add gen 6/7/8 training data and
re-train. Eventually have a single model trained on all 4 gens. This is the actual "all
formats, all gens" goal from the original project plan.

### Step (d) — Cloud burst, AFTER (a) and (c)

See `docs/CLOUD_DEPLOY.md` for the revised plan + decision tree. **Success criterion is now
`+50 Elo delta AND ≥20M states`, not absolute smart_avg %.**

---

## Open questions for next session

These are the questions still genuinely open after Session 33's Elo measurement.

1. **Does BC scaling actually transfer to PPO ceiling?** Metamon's paper says "size matters
   for BC > RL." Their BC scaling was clear but their RL variants showed diminishing returns.
   **The whole c3-c4 sequence is built on the bet that scaling our BC base will lift the PPO
   ceiling.** It's plausible but not proven for our format. The 30M experiment in step (c3)
   tests this directly. **If it fails, we need a different theory.** Have a backup architectural
   hypothesis ready (probably capacity reallocation per `feedback_capacity_allocation.md`).

2. **What's the right BC scaling target on 6GB?** Current: 13.4M, fits at bs=8. Estimated
   max local: 30M with FP16 + bs=4 + grad accumulation. **Untested.** First task in c3 should
   be a VRAM probe — train 1 epoch of 30M BC and see what peak VRAM looks like. If OOM, drop
   to 25M or aggressively reduce batch size. If comfortable headroom, push to 35-40M.

3. **Multi-gen data overlap with gen9 — do gens compete or complement?** Open empirical
   question. If gen6/7/8 data adds noise that hurts gen9 performance, the union dataset
   isn't a clean win. Mitigation: train on union but evaluate on gen9 separately to track
   per-gen impact. If gen9 Elo drops vs gen9-only baseline, we have a curriculum problem
   (need either per-gen heads, or gen-conditioning, or sequential gen training).

4. **Per-opponent signal question (from Session 33 user discussion):** Currently we get ~14
   games per opponent per iter (200 games / ~14 unique opponents from pool). High diversity,
   low per-opponent precision. ps-ppo gets 200 games per opponent (since they only have 1
   opponent — themselves). Would FEWER opponents per iter with MORE games each give better
   per-opponent learning? No published comparison exists. **Test as a side experiment if BC
   scaling is somehow blocked.**

5. **Eval team distribution question (from Session 33 user discussion):** the Elo ladder uses
   handcrafted 70 OU teams for measurement clarity. Procedural would test the training
   distribution more directly. Future improvement: add `--teambuilder procedural:PATH` flag
   to `eval_elo_ladder.py` and run BOTH; if rankings differ meaningfully, investigate.

6. **N=50 vs N=100 games per matchup:** N=50 gives ±25 Elo CIs on top players, which is fine
   for headlines (BC vs PPO, plateau width) but borderline for distinguishing experiment
   outcomes. **For measuring c3-c4 deltas, bump to N=100** so we can detect 30+ Elo improvements
   with confidence. Cost: ~2x wall time per ladder run (~3 hours instead of ~1.5).

---

## What to absolutely NOT do (Session 33 retractions + Elo-result findings)

- **Don't use smart_avg as a primary metric anymore.** Session 33 Elo measurement proved it's
  actively misleading. snapshot_0699 had the highest smart_avg ever (57%) but is at Elo 998
  (middle of pack, BELOW most v9 PPO snapshots). Smart_avg is a side check at best, not a
  decision driver. Use the Elo ladder for any ranking question.
- **Don't try to "reproduce snapshot_0699."** We've been trying to recreate the 57% smart_avg
  peak for 1000+ iters thinking it was real strength. It wasn't. The latest snapshot
  (snapshot_1784, Elo 1032) is genuinely stronger than snapshot_0699 by ~34 Elo.
- **Don't grind training time to break the plateau.** Session 33 Elo has 1200 iters of empirical
  proof: training from snapshot_0589 (Elo 1015) to snapshot_1784 (Elo 1032) = +17 Elo, all
  within bootstrap CI overlap. More iters at this scale produce noise, not improvement.
- **Don't strip the temporal module.** Two of three published references are stateless or have
  stateless cores, but the only one closest to our format (Metamon) is heavily temporal. For
  OU, temporal is justified. Keep it. Possibly redistribute capacity into it (later), but don't
  remove it.
- **Don't shrink the pool to "recent only".** VGC-Bench BCFP wins their pool A/B over both
  Nash-weighted and latest-only. Uniform-over-history is the published winner.
- **Don't run the cloud burst with the Session 32 success criterion** (60% smart_avg). It's
  the wrong metric. Use the Session 33 criterion (+50 Elo delta AND ≥20M states).
- **Don't grind on reward shaping.** Sessions 31-33 tried dense, sparse, terminal, immune
  penalty, KO bonus, HP delta. None broke the plateau. Reward isn't the lever.
- **Don't try the "head count A/B" or "filter loosening" experiments as standalone fixes.**
  They were in the original step (c) plan but remain low-priority.
- **⚠ Don't assume the "700 Elo gap" is real (Session 35 correction).** The VGC-Bench and our
  Elo scales are incompatible. Apples-to-apples gap is ~115 Elo (BCFP +147 above SH vs our
  +32 above SH). Plans premised on "700 Elo gap" were overreacting. See RESEARCH.md §0.5.
- **Don't skip the lambda fix.** GAE lambda=0.75 was uniquely anomalous — every published
  system uses 0.95. **DONE (Session 35 Exp 1): +25 Elo so far, still climbing.**
- **Don't compare raw Elo numbers across different ladder runs.** Pool composition changes
  shift absolute ratings. sp1784 was Elo 1032 in Session 33 ladder (38 players) but 1003 in
  Session 35 ladder (33 players, different composition). **Relative ordering within a ladder
  is what matters.** Use `--add-to` for consistent incremental measurement.
- **Smart_avg is indicative but not primary.** Per-bot trends ARE real signal (if Tactical
  goes from 48% to 57%, that's real improvement). But smart_avg saturates — it was flat at
  ~52% across E4-E9 while Elo showed E4(1015)→E9(1019) improvement. Use Elo for decisions.

## Reference files

- `docs/RESEARCH.md` §0 — canonical architecture comparison (updated Session 35)
- `docs/STATUS.md` — full session history
- `docs/CLOUD_DEPLOY.md` — revised cloud plan with decision tree gated on Elo
- `memory/MEMORY.md` — quick-reference summary, loaded each session
- `memory/feedback_session35_audit.md` — S35 audit: lambda, entropy, Elo scale, ff_dim
- `memory/project_infra_audit.md` — S35 code audit: deferred fixes, cloud readiness
- `memory/feedback_capacity_allocation.md` — capacity inversion vs Metamon
- `memory/project_multigen_plan.md` — multi-gen pipeline TODO
- `data/eval/elo_session35_exp1.json` — **canonical Elo (Session 35, 33 players, 528 matchups)**
- `data/eval/elo_session33_EXTENDED_FINAL.json` — Session 33 Elo (historical reference)
- `data/eval/eval_history.csv` — all bot eval results from TensorBoard
- `data/eval/eras.json` — era definitions (E0-E9) for trajectory analysis
- `extract_eval_csv.py` — extract eval data from TB to CSV + plots
- `plot_session35.py` — 3-panel Elo + eval trajectory plot

---

## Session 34 refactor summary

**All file renames (old → new):**
- `features_v8.py` → `features.py` | `policy_heads_v8.py` → `model.py`
- `bc_policy_player_v8.py` → `battle_agent.py` | `rl_train_v8.py` → `ppo.py`
- `bc_train_v8.py` → `train_bc.py` | `dataset_v8.py` → `dataset.py`
- `replay_to_memmap_v8.py` → `replay_to_memmap.py` | `convert_jsonl_to_memmap_v8.py` → `convert_jsonl_to_memmap.py`

**Internal symbol renames:** make_features (was make_v8_features), BattleAgent (was BCPolicyPlayerV8),
Trajectory (was V8Trajectory), ppo_update (was ppo_update_v8), load/save_checkpoint (was *_v8_*)

**Monolith decomposition (rl_train_v9.py → 5 files):**
- `train_rl.py` (~540 lines) — main loop with 8 extracted helpers
- `inference_batcher.py` (~200 lines) — async batched GPU inference
- `rl_player.py` (~180 lines) — V9RLPlayer + SelfPlayOpponent
- `rl_collection.py` (~250 lines) — collect_v9 + BackgroundCollector
- `rl_pipeline.py` (~350 lines) — multiprocess infrastructure

**New shared code:** `format_config.py` (FormatConfig dataclass), `features.build_turn_batch()`,
`features.action_to_order()` — eliminated 300+ lines of duplication across 3 files.

**Multi-gen plumbing:** `--format` flag on train_rl.py, train_bc.py, eval_elo_ladder.py.
team_generator.py has per-gen ban lists (gens 6-9) and gen parameter. dataset.py validates
dimensions and zero-pads old memmaps. vocab.py already covers gens 1-9.

**Dead code removed:** V8RLPlayer + old collect + old main from ppo.py (718 lines), old
rl_train_v9.py monolith (1931 lines), v7 files moved to legacy/.

## Session 35 deep methodology audit (2026-04-09)

**What happened:** Full audit of ML methodology, PPO hyperparameters, architecture, self-play,
reward design, and data efficiency — cross-referenced against published systems (Metamon,
VGC-Bench, ps-ppo, OpenAI Five, AlphaStar) and our own training history.

**Major findings (ranked by impact):**

1. **GAE lambda=0.75 is a major outlier.** Every published system uses 0.95. Most anomalous
   hyperparameter. Causes myopic credit assignment in 30-60 turn games. #1 priority fix.
2. **The "700 Elo gap" vs VGC-Bench is an artifact of incompatible Elo scales.** Real gap:
   ~115 Elo (BCFP +147 above SH, we +32 above SH). Changes optimal experiment order.
3. **Entropy=0.04 is 4x standard** (0.01 typical). But our history shows collapse at 0.02
   during early training. Reduce cautiously to 0.02 first.
4. **ff_dim was documented as 2x but is actually 4x** (standard). Doc error, not code error. Fixed.
5. **Slot permutation augmentation** is near-free regularization we're not using.
6. **Capacity allocation (spatial-heavy, temporal-light) is inverted vs Metamon**, but the
   breakthrough was entity tokenization (preserved at any spatial dim), not spatial size.
   Reallocation is justified but expensive (requires retrain). Do after cheap fixes.
7. **Offline RL was tried (IQL, 3 runs) and failed**, but with scalar value head (now fixed
   to distributional). Metamon's Binary+MaxQ was NOT fully implemented. Worth revisiting
   but lower priority than hyperparameter fixes.
8. **Recency-weighted pool** (70/30) may outperform pure uniform per OpenAI Five's pattern.

**What was validated (don't change):**
BC→PPO pipeline, terminal-only reward, entity tokenization, distributional value head,
temporal model for OU, uniform-ish pool strategy, gamma=0.9999, clip=0.2.

**Revised experiment order:** See Step (c-pre) above. Hyperparameter fixes → augmentation →
capacity reallocation → multi-gen/BC scaling. The original "30M BC scaling first" plan is
deprioritized since the gap it was designed to close is much smaller than believed.

**Infrastructure fixes (also Session 35):**
- Atomic checkpoint writes (ppo.py) — prevents corruption on crash
- Traceback in PPO exception handler — full stack trace for debugging
- Grad accumulation bug fixed — accum_count reset on exception, grad clip at step not backward
- Temporal history pre-slice — prevents OOM on 300+ turn battles
- BC training AMP support — `--fp16` flag for ~2x speedup on CUDA
- Vectorized collate function — numpy stack per-episode instead of per-turn Python loop
- DataLoader optimization — persistent_workers, prefetch_factor, drop_last
- Removed 18 unnecessary .copy() per sample in memmap reads

**Deferred infrastructure items (do when relevant):**

| Item | When to do | Effort |
|------|-----------|--------|
| DDP/FSDP multi-GPU wrapper | Before cloud training | 1-2 days |
| Hydra/OmegaConf config management | Before running many experiments | Half day |
| Structured logging (Python logging module) | Before long runs | Half day |
| Test suite (pytest, smoke tests) | Ongoing, before major changes | Ongoing |
| Remote checkpoint storage (S3/GCS) | Before cloud deployment | Half day |
| Health monitoring / heartbeat | Before cloud deployment | 2 hours |
| Signal handler for graceful Ctrl+C shutdown | Before next long run | 1 hour |

---

## Current state snapshot (as of Session 35, 2026-04-10)

### Training state
- **TRAINING RUNNING.** Exp 1 continuation, iters 2000-2499. `--lam 0.95 --ent-coef 0.02`.
- Log: `exp1_lambda095.log`. Run dir: `selfplay_v9_20260410_*/`
- **Exp 1 result so far:** +25 Elo (sp1784=1003 → sp1984=1028). Still climbing. 500 more iters.
- Resume command in MEMORY.md if interrupted.

### Key checkpoints
- `data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt` — **BC base, Elo 818.**
- `selfplay_v9_20260408_042048/snapshot_1784.pt` — **pre-Exp1 baseline, Elo 1003.**
- `selfplay_v9_20260409_080620/snapshot_1984.pt` — **Exp1 best so far, Elo 1028.**
- `selfplay_v9_20260410_001804/snapshot_1999.pt` — **Exp1 last measured, Elo 1021.**

### Key data
- `data/eval/elo_session35_exp1.json` — **canonical Elo (33 players, 528 matchups, 100 games)**
- `data/eval/eval_history.csv` — bot eval history from all TensorBoard logs
- `data/eval/eras.json` — era definitions E0-E9
- Existing memmaps (human_v8, memmap_v8) are stale (move=107, switch=28). Regenerate before BC scaling.

### Git
`git log --oneline` for history. All Session 35 changes tracked.

### Tools added this session
- `extract_eval_csv.py` — TensorBoard eval → CSV + trend plots
- `plot_session35.py` — 3-panel combined Elo + eval trajectory
- `run_elo_shards.sh` — parallel Elo shard launcher
- `eval_elo_ladder.py --add-to` — incremental Elo measurement (~30 min per new snapshot)
- `registry.py` — fire-and-forget append helpers for persistent data registry
- `build_registry.py` — rebuild registry from scratch (safety net)

### Persistent data registry

All training runs, bot evals, and Elo measurements are **automatically** saved to
`data/eval/registry/` in JSONL format. No manual action needed during training.

| File | Updated by | Contents |
|------|-----------|----------|
| `runs.jsonl` | train_rl.py (auto at start) | Run configs: lam, ent, lr, iter range, etc. |
| `evals.jsonl` | train_rl.py (auto every 20 iters) | SH%, SD%, Tac%, Str%, smart_avg% per eval |
| `elos.jsonl` | eval_elo_ladder.py (auto after BT fit) | Elo, CI, ladder source per snapshot |
| `eras.json` | Manual edit | Era definitions (id, iter range, color, description) |

If the registry gets out of sync: `python build_registry.py` rebuilds from scratch.
Scans all run dirs, TensorBoard logs, and Elo JSONs — nothing lost if source data exists.

### Session handover checklist (for future sessions)

1. Read `docs/NEXT_SESSION.md` top-to-bottom (this file)
2. Check training status: `grep "Iter " exp1_lambda095.log | tail -5`
3. Check registry: `python build_registry.py --print-summary`
4. Check `data/eval/eras.json` — add new era if training regime changed
5. Check `memory/MEMORY.md` for current state summary
