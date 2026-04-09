# NEXT_SESSION.md — Concrete TODO Order

**Last updated: 2026-04-09 (Session 33 end — Elo ladder + extension complete)**

This file is the canonical "if you're starting a new session, do these things in this order"
reference. **It is intentionally self-contained** — a future session reading only this file
plus `docs/RESEARCH.md` §0 should have full context to execute every pending task without
asking for re-explanation.

If you read nothing else, **read this top-to-bottom, then `docs/RESEARCH.md` §0**.

---

## Where things stand right now (Session 33 end)

- **Training is STOPPED** at `selfplay_v9_20260408_042048/snapshot_1784.pt` (the latest checkpoint).
  Was killed cleanly to free the GPU for the Elo ladder run. Resume command in `MEMORY.md` if
  you want to continue training, but **don't, until step (b)/(c) is done** — Session 33 proved
  more iters at this scale produce essentially zero Elo improvement.
- **First-ever Elo measurement is DONE.** Two ladder runs:
  - **Initial** (31 players, 465 matches): `data/eval/elo_session33_FINAL.json`
  - **Extended** (38 players, 703 matches, the canonical one): `data/eval/elo_session33_EXTENDED_FINAL.json`
- **Resilience patches are in place** in `rl_train_v8.py` (`n_succeeded`/`n_failed`) and
  `rl_train_v9.py` (FATAL guard at zero-PPO, snapshot save gate). Verified working.
- **`eval_elo_ladder.py` v2** has the permanent fix: PlayerPool with LRU cache, checkpoint
  state_dict caching (CPU), incremental JSONL save, resume support. Wall time for the 38-player
  ladder is ~93 min on 4 shards. ~10x faster than v1's death-spiral pattern.
- **Pre-cloud backup** at `pokemon-ai-starter/pokemon-ai/src/backups/v9_pre_cloud/` — patched
  and working source for fallback if anything breaks.

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

6. **Cross-reference: VGC-Bench BCFP claimed 1768 Elo at our compute scale.** Different
   anchors, can't directly compare, but the magnitude (~700 Elo gap) suggests substantial
   architectural headroom we lack. The gap is too large to close with more training time.

**Implication: more training time at this rate is uneconomic.** At ~0.018 Elo/iter, gaining
+50 Elo costs ~2800 iters (~9 days at 270s/iter). Gaining +700 Elo to close the gap to
VGC-Bench's reference is fantasy at this rate (~50 weeks). The lever must be **pre-PPO**
(bigger BC base — Metamon's "size matters for BC > RL" thesis) or **architectural** (capacity
reallocation, head count, ensemble critic). Cloud burst on the current architecture might give
+50-150 Elo over a few days; it will not close the 700 Elo gap.

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

### Step (b) — Code refactor + smoke test

**Status: NOT STARTED.** Multi-day work. Can run in parallel to step (a) or after.
The patches Session 33 added (n_succeeded detection, FATAL guard) work fine in the
current monolithic file — refactor is for maintainability, not correctness.

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

### Step (b) DEFERRED status

**Has not been started as of Session 33.** Estimated 1-2 days of focused work. Should be
done before step (c) experiments start, because step (c) needs the modular file structure
to add experiment-specific flags cleanly. If you're impatient and want to run step (c)
first, you can — but expect ugly diffs to `rl_train_v9.py`.

### Step (c) — Multi-gen vocab prep, BEFORE BC scaling

**Status: NOT STARTED.** This step replaces the original "head-count A/B + filter loosening"
plan. The Session 33 Elo result killed those experiments as standalone fixes — they would
each be expected to give ~30-80 Elo improvement, but the gap to architectural reference
points is ~700 Elo. Smaller experiments are not the lever. The lever is **a stronger
foundation (bigger BC base)** combined with **multi-gen capability** so we don't have to
retrain BC for every gen.

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
  They were in the original step (c) plan but the Session 33 Elo result killed them as
  meaningful levers — they each give ~30-80 Elo improvement at best, but the gap to
  reference points is ~700 Elo. Smaller experiments are not the lever. Bigger BC base is.
- **Don't run another Elo ladder until you have something to measure.** N=50 ladder takes
  ~93 min. Don't burn that to "see if anything changed" — only run it after a meaningful
  experiment (refactor doesn't count, BC scaling does, PPO from new BC does).

## Reference files

- `docs/RESEARCH.md` §0 — canonical architecture comparison (Session 33 research findings)
- `docs/STATUS.md` Session 33 sections (POST-SCRIPT, RESEARCH ROUND, ELO LADDER RESULT, ELO
  LADDER EXTENDED) — full session narrative top-to-bottom
- `docs/CLOUD_DEPLOY.md` — revised cloud plan with decision tree gated on Elo
- `MEMORY.md` — quick-reference summary, exactly 200 lines, fully loaded each session
- `memory/feedback_capacity_allocation.md` — Session 33 finding on capacity inversion vs Metamon
- `memory/project_multigen_plan.md` — multi-gen pipeline TODO (referenced by step c1/c2)
- `memory/project_not_needed.md` — features the model learns; don't hand-engineer
- `pokemon-ai-starter/pokemon-ai/src/backups/v9_pre_cloud/` — Session 33 patched source fallback
- `pokemon-ai-starter/pokemon-ai/src/backups/README.md` — which backup is which
- `pokemon-ai-starter/pokemon-ai/src/data/eval/elo_session33_EXTENDED_FINAL.json` — canonical
  Session 33 Elo result (the baseline future experiments are compared against)

---

## Current state snapshot (as of Session 33 end, 2026-04-09)

For a future session that needs to start cold without hunting:

### Training state
- **Training is STOPPED.** Was killed cleanly Session 33 to free the GPU for the Elo ladder.
- **Latest snapshot:** `pokemon-ai-starter/pokemon-ai/src/data/models/rl_v9/selfplay_v9_20260408_042048/snapshot_1784.pt`
- **Reward style at stop:** terminal (`--reward-style terminal`)
- **Pool size at stop:** 617 (filtered sp ≥ 260)
- **Why we stopped:** Session 33 Elo measurement showed the architecture is at its ceiling.
  Additional training iters would just produce more plateau-band noise. Resume only if you
  have a specific reason (e.g., to validate something during the refactor work).

### To resume training (only if needed)
```bash
cd pokemon-ai-starter/pokemon-ai/src
# Find latest snapshot:
ls -t data/models/rl_v9/selfplay_v9_2026040*/snapshot_*.pt | head -1
# Then use the resume command in MEMORY.md, pointing --resume at that snapshot.
```

### Battle servers
4 servers were left running on ports 9000/9001/9002/9003 at end of Session 33 (port 9003
added for parallelization of the Elo ladder). Restart with the commands in MEMORY.md if
they've stopped.

### Key checkpoints (canonical references — Elo numbers are from extended ladder)
- `data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt` — **BC base**, init-from for all v9 runs.
  **Elo: 806.** The starting point everything is built on.
- `selfplay_v9_20260401_141524/snapshot_0699.pt` — old "all-time peak" by smart_avg (57%).
  **Actual Elo: 998.** Middle of the pack. Smart_avg lied. **NOT** the target to reproduce.
- `selfplay_v9_20260407_124041/snapshot_1724.pt` — pre-CUDA-crash clean state. **Elo: 1027**.
- `selfplay_v9_20260408_042048/snapshot_1784.pt` — **latest, current top of ladder. Elo: 1032.**
  The strongest checkpoint we have.

### Key code files (and their state at Session 33 end)
- `eval_elo_ladder.py` — **Session 33 build, with permanent fix.** PlayerPool + checkpoint
  cache + JSONL save/resume. ~93 min for full 38-player ladder on 4 shards. Verified working.
- `recover_elo_from_log.py` — log-parsing recovery tool, used during the v1 death-spiral.
  Still useful as a recovery tool if the main script's incremental save somehow fails.
- `bc_policy_player_v8.py` — modified Session 33 to accept `_cached_ckpt` parameter (bypasses
  disk read). Backwards-compatible — existing callers don't change.
- `rl_train_v9.py` — **1900+ lines, monolith.** Has the Session 33 resilience patches but
  is the refactor target for step (b).
- `rl_train_v8.py` — has `ppo_update_v8` with `n_succeeded`/`n_failed` return additions.
- `policy_heads_v8.py` — 13.38M params, target for c3 (BC scaling — needs to grow to ~30M).
- `features_v8.py` — 16 entity tokens, type effectiveness features. Will need vocab expansion
  for c1 (multi-gen prep).
- `vocab.py` — embedding tables for species/move/ability/item. Will need expansion for multi-gen.
- `team_generator.py` — currently gen9-only. Will need per-gen support.
- `mp_collect_v2.py`, `mp_collect_v3.py` — multiprocess collection, cloud-ready, NOT in main path.
- `backups/v9_pre_cloud/` — Session 33 patched source fallback.
- `backups/README.md` — which backup is which.

### Key data files
- `pokemon-ai-starter/pokemon-ai/src/data/eval/elo_session33_EXTENDED_FINAL.json` — **canonical
  Session 33 Elo result.** 38 players, 703 matches, BayesElo + bootstrap CIs. The baseline
  every future experiment will be compared against.
- `pokemon-ai-starter/pokemon-ai/src/data/eval/elo_session33_FINAL.json` — first ladder run
  (31 players, 465 matches). Kept for history; superseded by EXTENDED.
- `pokemon-ai-starter/pokemon-ai/src/data/eval/elo_session33_shard{0,1,2,3}.{json,jsonl}` —
  per-shard incremental saves. The JSONLs are resume sources for any future ladder extension.
- `data/datasets/human_v3_memmap/` — 200K gen9ou 1500+ replays, 10.1M records. Source for
  the existing 13.4M BC. Needs gen6/7/8 extension for c2.

### If you need to resume right now (5-line cheat sheet)
```bash
# 1. Resume training from latest snapshot (only if you have a reason):
cd pokemon-ai-starter/pokemon-ai/src
ls -t data/models/rl_v9/selfplay_v9_2026040*/snapshot_*.pt | head -1
# Use that path with the resume command in MEMORY.md.

# 2. Re-run Elo ladder against the canonical baseline (kill training first, both want GPU):
python -u eval_elo_ladder.py --combine \
  data/eval/elo_session33_shard0.json \
  data/eval/elo_session33_shard1.json \
  data/eval/elo_session33_shard2.json \
  data/eval/elo_session33_shard3.json \
  --out-json /tmp/elo_recheck.json
# (Pure combine — no new matches needed unless you have new snapshots to add.)

# 3. To add NEW snapshots to the existing ladder (e.g., post-experiment):
# - Merge all 4 existing shard JSONLs into a single master (see Session 33 STATUS notes)
# - Copy master to each shard's JSONL slot
# - Re-launch with --snapshots that include the new ones (resume will skip old pairs)
```
