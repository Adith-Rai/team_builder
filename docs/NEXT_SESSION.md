# NEXT_SESSION.md — Project Handover

**Last updated: 2026-04-22 (Session 37 — Metamon study + multi-gen prep + capacity reshape done)**

This is the canonical reference for resuming work on this project. It's self-contained —
read this top-to-bottom and you should have full context to execute every pending task.

Supporting documents:
- `docs/METAMON_LEARNINGS.md` — Session 37 Metamon architecture study + recommendations
- `docs/RESEARCH.md` — architecture research, published system comparisons, experiment order
- `docs/STATUS.md` — full historical narrative if deep context needed (long, usually skippable)
- `docs/CLOUD_DEPLOY.md` — cloud migration plan

---

## Current position

**Best checkpoint: `selfplay_v9_20260413_061236/snapshot_2979.pt` at Elo 1058**
(confirmed by `data/eval/elo_exp5_FINAL.json`, 62 players, 1891 matches).

- +38 Elo above strongest heuristic bot (Tactical=1019)
- +30 Elo above previous session-top (sp1984=1028)
- +241 Elo above BC base (817)

Near-tied: `snapshot_2999.pt` (Elo 1055, CI overlaps). Top 7 snapshots are all 1048-1058.

**The architectural ceiling is CONFIRMED at ~1058 Elo.** Exp 5 (200 iters with all
safeguards: adaptive entropy + early stop + EMA PFSP + 400 games/iter) produced no
checkpoint that exceeded sp2979. The plateau is real, measured across 62 players and
1891 matchups. Further gains require architectural changes.

**Team selection test in progress:** testing sp2979 with each of 70 handcrafted OU teams
individually against bots (14,000 games). Early results show savg 51-77% depending
on team — massive variance. Fixed-team play is MUCH stronger than random-from-pool.
Results will be in `team_selection_results.json` when complete.

**Session 37 status — TWO major prep milestones done:**

1. **Metamon study complete** (`docs/METAMON_LEARNINGS.md`). Confirmed from direct config
   read that every Metamon size variant uses 5-8× temporal:spatial d_model — strongest
   evidence yet that Option A (capacity reallocation) is the right next architectural lever.

2. **Multi-gen prep DONE** (commit `18e965c`). Previous sessions had already implemented
   90%: vocab loops gens 1-9, features has Mega/Z-move/Dynamax/Tera flags, team_generator
   has per-gen ban lists, format_from_str parses gen. Session 37 fixed two real bugs —
   `ProceduralTeambuilder` wasn't passing `gen` to `load_pokemon_pool` (always loaded
   gen9 tiers), and OU rating threshold was hardcoded `1695` (gen9 only; gen6-8 use `1760`
   in 2024-04 data). Verified gen6/7/8/9 teams now generate era-appropriate mons
   (gen6 HP Ice Thundurus, gen7 Z-crystal Dragonite, gen8 Double Iron Bash Melmetal).

3. **Capacity reshape implemented** (commit `76acca8`). Added `d_spatial`, `d_temporal`,
   `n_summary_tokens` to PokeTransformerConfig. Legacy defaults (None/0) preserve exact
   sp2979 param count (13,382,572, strict=True load verified). New reshape config
   (256/512/3L/3L/K=4/dropout=0.05) instantiates at 14.28M, forward + multi-turn works
   cleanly, FP16 smoke-tested on CUDA. See `docs/METAMON_LEARNINGS.md` §5 for design rationale.

**Next concrete step:** Launch BC retrain with reshape config. Command below.

---

## Quick-reference commands

### BC retrain with capacity reshape (Session 37 — the NEXT step)
```bash
cd pokemon-ai-starter/pokemon-ai/src
python -u train_bc.py \
  --memmap-dir data/datasets/memmap_v8 \
  --device cuda --fp16 \
  --d-spatial 256 --d-temporal 512 \
  --n-spatial-layers 3 --n-temporal-layers 3 \
  --n-summary-tokens 4 \
  --dropout 0.05 \
  --lr 1e-4 --weight-decay 1e-4 --grad-clip 2.0 \
  --batch-size 16 --epochs 10 --sched cosine --warmup-steps 200 \
  --eval-games 200 \
  2>&1 | tee bc_reshape.log
```

Expected: ~14.28M params. The reshape tests whether Metamon-style temporal:spatial
(2:1 d_model, equal layer count, K=4 summary scratch tokens) closes the gap to
MM-SmallRLG9 on the PokeAgent ladder. Runs entirely on existing memmap (stale 107/28
dims, auto-padded — the 2 missing type_eff dims are not the bottleneck per METAMON_LEARNINGS §5).

### Resume training from current best (with safeguards ON)
BC_base (`data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt`) was removed during the
Session 37 cleanup. `train_rl.py` still requires `--init-from` (argparse-enforced),
but when `--resume` is also provided the init weights are overwritten. Pass the same
sp2979 path for both — it's a safe workaround until `--init-from` is made optional.
```bash
cd pokemon-ai-starter/pokemon-ai/src
# Start battle servers first (see below)
python -u train_rl.py \
  --init-from data/models/rl_v9/selfplay_v9_20260413_061236/snapshot_2979.pt \
  --resume data/models/rl_v9/selfplay_v9_20260413_061236/snapshot_2979.pt \
  --device cuda --servers 9000,9001,9002 --fp16 --pipeline \
  --games-per-iter 200 --max-concurrent 200 --n-iters 500 --warmup-iters 0 \
  --reward-style terminal --lam 0.95 --ent-coef 0.02 --grad-accum 1 \
  --adaptive-entropy --early-stop \
  --procedural-teams C:/Users/raiad/OneDrive/Desktop/team_builder/raw_data/pokemon_usage/2024-04 \
  2>&1 | tee new_run.log
```

Add `--win-rate-mode ema` to enable EMA PFSP (fixes stale-rating issue). Default is cumulative.

### Start 3 battle servers (Windows)
```bash
NODE=/c/Users/raiad/OneDrive/Desktop/team_builder/tools/node-v20.18.1-win-x64/node.exe
BS=/c/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src/battle_server.js
$NODE "$BS" --port 9000 &
$NODE "$BS" --port 9001 &
$NODE "$BS" --port 9002 &
# Verify:
curl -s -o /dev/null -w "9000:%{http_code} " http://127.0.0.1:9000/action.php?
curl -s -o /dev/null -w "9001:%{http_code} " http://127.0.0.1:9001/action.php?
curl -s -o /dev/null -w "9002:%{http_code}\n" http://127.0.0.1:9002/action.php?
```

### Incremental Elo measurement
```bash
# 3 shards on 3 servers, parallel (~5 hours for 19 snapshots)
cd pokemon-ai-starter/pokemon-ai/src
SNAPS="path/to/snap1.pt path/to/snap2.pt ..."
NAMES="sp1000 sp1010 ..."

# Shard 0 (on server 9000):
python -u eval_elo_ladder.py --add-to data/eval/elo_exp5_FINAL.json \
  --snapshots $SNAPS --names $NAMES \
  --n-games 100 --concurrency 70 --device cuda \
  --server ws://127.0.0.1:9000/showdown/websocket \
  --shard 0/3 --out-json data/eval/elo_sessionXX_shard0.json &

# Shards 1 and 2 identical but with --server 9001/9002 and --shard 1/3, 2/3
# Then combine:
python eval_elo_ladder.py --combine \
  data/eval/elo_sessionXX_shard0.json \
  data/eval/elo_sessionXX_shard1.json \
  data/eval/elo_sessionXX_shard2.json \
  --out-json data/eval/elo_sessionXX_FINAL.json
```

### Monitor training (PowerShell)
```powershell
Get-Content "path\to\log.log" -Wait | Select-String "\] Iter|EVAL:|Snapshot|KL early|FATAL|ERROR|PFSP|EARLY STOP|ENT"
```

---

## What's been tried and what works (consolidated)

### What's validated — don't change these
- **BC → PPO pipeline** (standard, works)
- **Terminal-only reward** (tried shaping, always hurt)
- **Entity tokenization** (the architectural breakthrough — preserved at any spatial dim)
- **Distributional value head** (51-bin categorical, not scalar)
- **Temporal model for OU** (needed for 30-60 turn games)
- **gamma=0.9999, clip=0.2, lam=0.95** (PPO standards, confirmed)
- **lam=0.95** (not 0.75 — that was our outlier, fixed in Exp 1 for +25 Elo)

### What's been tested and learned

| Experiment | Change | Result | Conclusion |
|-----------|--------|--------|------------|
| Lambda fix | 0.75 → 0.95 | +25 Elo | **Keep.** |
| Entropy tuning | ent 0.04 → 0.02 → 0.03 → 0.02 | 0.02 from healthy base is sweet spot | **Keep 0.02.** |
| Slot permutation | bench/move shuffle per turn | Small impact, but helped stability | **Keep.** |
| PFSP (vs uniform) | (1-wr)² weighted | +11 Elo, but 12% stale-rating waste | **Keep. Consider EMA to fix staleness.** |
| LR refinement | 1e-4 → 3e-5 | Hit peak sp2999 (Elo 1055), then collapsed | **Lower LR is risky without safeguards.** |
| Games/iter 400 | paired with low LR | More data per update | Secondary — paired with LR choice. |
| **Exp 5 safeguards** | adaptive ent + early stop + EMA | 200 iters, savg=55.3% mean, no collapse | **Ceiling confirmed at ~1058. Safeguards work.** |

### What causes collapse (learned the hard way)

Exp 4 collapsed from Elo 1055 (sp2999) to 785 (sp3054) in 55 iters. Root causes:

1. **Weak ent_coef** (0.02) + consistent "improvement" gradients → slow policy sharpening
2. **Stale PFSP cumulative ratings** → 12% of training on opponents we'd already mastered →
   reinforces narrow strategies → over-specialization
3. **Entropy dropped below 0.65 without triggering safeguards** (old adaptive threshold was 0.55)

This is now fixed — see safeguards below.

### Safeguards available (Session 36 additions)

All configurable, defaults preserve old behavior. Enable explicitly for safety.

**Adaptive entropy (`--adaptive-entropy`):**
Auto-raises ent_coef when entropy drops, auto-lowers when too exploratory. Defaults:
- `--adaptive-entropy-low 0.65` (raise ent_coef below this; was 0.55)
- `--adaptive-entropy-high 0.95` (lower ent_coef above this)
- `--adaptive-entropy-step 0.1` (±10% per iter; was ±5%)
- `--adaptive-entropy-max 0.08` (cap)
- `--adaptive-entropy-min 0.01` (floor)

Would have caught Exp 4 collapse at iter 3020 with ent_coef rising to ~0.05.

**Composite early stopping (`--early-stop`):**
Stops training if `patience` (default 3) consecutive RAW evals show BOTH savg regression
AND 3+ of 4 bots regressing, OR savg drops >2× threshold (handles specialization).
- Based on rolling-3 "best" baseline, raw-eval comparison for current
- Would have stopped Exp 4 at iter 3055 with only ~3 Elo loss from peak

Parameters: `--early-stop-patience 3`, `--early-stop-savg-threshold 2.0`,
`--early-stop-bot-threshold 3.0`, `--early-stop-bot-count 3`, `--early-stop-min-evals 5`.

9 unit tests in `test_early_stop.py` validate all trigger conditions including Exp 4 replay.

**EMA PFSP (`--win-rate-mode ema`):**
Replaces cumulative win-rate tracking with exponential moving average. Fixes the 12%
stale-oversampling problem. Old cumulative data naturally fades; current policy strength
reflected in rating.
- `--win-rate-ema-alpha 0.3` (blend weight for new batches)
- `--win-rate-ema-window 50` (cap on effective_games to prevent unbounded growth)

Default `cumulative` preserves exact old behavior. 8 unit tests in `test_ema_pfsp.py`.

---

## Code architecture — key files

### Training pipeline
```
train_rl.py           Main training loop + argparse. Calls collect → PPO → eval → save
 ├─ rl_collection.py    collect_v9 async collection + BackgroundCollector (pipeline)
 │   └─ pfsp_sample()   Prioritized opponent sampling — cumulative or EMA
 ├─ rl_player.py        V9RLPlayer + SelfPlayOpponent (shared build_turn_batch)
 ├─ rl_pipeline.py      Multiprocess variants (InferenceServer, MPRLPlayer)
 ├─ inference_batcher.py Async batched GPU forward — key for concurrency speedup
 ├─ ppo.py              Trajectory, GAE, ppo_update, save_checkpoint (atomic writes)
 ├─ model.py            PokeTransformer 13.38M: spatial 384d/4L + temporal 384d/2L
 └─ features.py         Entity tokenization, build_turn_batch, slot permutation
```

### Evaluation / analysis
```
eval_elo_ladder.py     Full Bradley-Terry ladder + --add-to incremental + --shard parallel
battle_agent.py        Inference player (eval, no training augmentation)
analyze_experiments.py Cross-experiment comparison (per-bot stats, correlations)
analyze_pfsp.py        PFSP opponent tracking analysis (staleness, concentration)
analyze_pfsp_timing.py Opponent rotation / encounter timing analysis
analyze_elo_trajectory.py  Elo plot + era aggregates from ladder JSON + eras.json
build_registry.py      Rebuild persistent registry from scratch
```

### Data / registry
```
data/eval/registry/
├── runs.jsonl         Training run configs (auto-logged by train_rl.py)
├── evals.jsonl        Per-iter bot eval results (auto-logged every 20 iters)
└── elos.jsonl         Per-snapshot Elo measurements (auto-logged by eval_elo_ladder.py)

data/eval/
├── elo_exp5_FINAL.json    Canonical Elo ladder (55 players)
├── eras.json                     Training era definitions for trajectory plots
└── trajectory_session36.png      Latest trajectory plot
```

### Tests
```
test_exp2_exp3.py      Slot permutation + PFSP correctness (14 tests)
test_early_stop.py     Composite early stopping trigger logic (9 tests)
test_ema_pfsp.py       EMA win-rate tracking (8 tests)
```

---

## What to do next — MULTI-GEN IS THE PRIORITY

The hyperparameter refinement lever is exhausted (confirmed by Exp 5). Multi-gen
prep comes FIRST before any model size changes, so all future work builds on a
multi-gen-ready foundation.

### IMMEDIATE NEXT: Study Metamon architecture + apply learnings

**PokeAgent ladder results (Session 36):**
Our model (13.38M) submitted to the PokeAgent Challenge live ladder.
- **Skill Rating 1444±30 with TEAM_T** — rank #12, ABOVE MM-Minikazam (4.7M, 1429)
- **Skill Rating 1376±24 with TEAM_AU** — rank #15
- Team choice moved us 68 points (more than all hyperparameter experiments combined)
- Gap to MM-SmallRLG9 (15M, same size as us): 73 points — they use params more efficiently
- Gap to MM-Kakuna (142M, best): 409 points
- We beat ALL heuristic baselines (BH tier: 1185-1240) by 200+ points

**Key insight: Metamon's 4.7M model nearly matches our 13.38M.** The gap is architecture
and methodology, not model size. Metamon is open source — study their design, apply to ours.

**Metamon reference repo cloned:** `metamon_ref/` (shallow clone, read-only reference).
Study these files (DO NOT modify our code to copy theirs — learn principles, apply to our design):
- Architecture: temporal:spatial ratio, entity token handling, layer configs
- BC data pipeline: preprocessing, quality filters, data scale
- Offline RL recipe: Binary+MaxQ methodology (different from our PPO)
- Minikazam (4.7M) vs Kakuna (142M): what scales and what doesn't

**After understanding Metamon's design, THEN do:**

### Multi-gen + gen-agnostic prep

**Why first:** User direction (Session 33, reconfirmed Session 36): multi-gen before
BC scaling or capacity reallocation. Reasoning: if we change model size or retrain BC
BEFORE multi-gen, we'd redo all that work when we add multi-gen later. Do it once.

**What needs to change (the architecture is ALREADY gen-agnostic):**
- `vocab.py`: verify/expand species/move/ability/item tables for gens 6-9
- `features.py`: add gen-specific volatile effects (Mega Evolution for gen6, Z-moves for gen7)
- `team_generator.py`: per-gen team generation + ban lists (currently gen9 only)
- `format_config.py`: add gen6/7/8 FormatConfig entries (n_actions, team_size, etc.)
- `replay_to_memmap_v8.py`: gen filter for scraping gen6/7/8 OU replays
- `battle_server.js`: verify it handles gen6/7/8 formats

**What does NOT change:** model.py, ppo.py, train_rl.py, rl_collection.py, rl_player.py
(all already parameterized via FormatConfig + --format flag).

**Gen scope:** Start with gens 6-9 (fully compatible features). Gens 4-5 need minor
adjustments. Skip gens 1-3 for now (missing abilities/items, different mechanics).

**Estimated effort:** 1-2 sessions for vocab + features + team gen. Multi-gen replay
scraping can run in background.

**Validation:** Train a 1-iter BC sanity check with expanded vocab on existing gen9
data → confirm no breakage. Then scrape gen6-8 replays and train multi-gen BC.

### AFTER multi-gen: architectural experiments

### Option A — Capacity reallocation (Exp 5 from old plan)

**What:** shrink spatial encoder, grow temporal encoder. Current ratio 1:1 (384d/4L spatial,
384d/2L temporal). Metamon uses 5-8:1 temporal:spatial across all model sizes. Proposed:
spatial 256d/3L, temporal 512d/3L (same param count, redistributed).

**Why:** Our temporal model is undersized for OU's 30-60 turn games with heavy hidden info.
The inverted ratio is our biggest architectural anomaly vs published systems.

**Cost:** Requires BC retrain from scratch (~1-2 days compute) + PPO from new BC (~1 week).

**Decision criterion:** If new model hits Elo 1100+ on existing ladder, ratio matters.
If flat at ~1058, capacity allocation isn't the bottleneck.

**Implementation:** change config values in `model.py` (d_model and n_spatial/temporal_layers).
The architecture supports any combination. Main work is BC retrain, not code.

### Option B — BC scaling (multi-gen path)

**What:** scale BC base from 13.4M to 30M+ params. Train on multi-gen data (gens 6-9 OU, not
just gen9). Per `memory/project_multigen_plan.md` this is mostly data-pipeline work since
architecture is already gen-agnostic.

**Why:** Our BC_base is Elo 817. Metamon's base is 1500+. PPO can only climb from where BC
starts — if BC ceiling is low, PPO ceiling is bounded. Bigger BC + more diverse data
= higher starting floor.

**Cost:** Several weeks. Multi-gen vocab prep (1-2 weeks), multi-gen replay scrape (1-2
weeks, mostly automated), 30M BC training (3-7 days), PPO from new base (1-2 weeks).

**Local limit:** 30M is realistic max on 6GB VRAM. For 50M+, need cloud.

**Decision criterion:** 30M BC at Elo 900+ → scaling helps → continue. 30M BC at ~810 → pivot.

**Steps in detail:**
1. Multi-gen vocab expansion (`vocab.py`, `features.py`)
2. Multi-gen replay memmap (`replay_to_memmap_v8.py` + gen filter)
3. 30M BC training (existing `train_bc.py` with larger config)
4. PPO from new base (same recipe, different checkpoint)

### Option C — Search at inference (MCTS)

**What:** add MCTS on top of learned policy/value. Use NN as prior/evaluator,
search for better moves. MIT thesis finding: "RL alone plateaus but RL+search breaks
through." Foul Play (PokeAgent Challenge co-winner) uses pure MCTS with no NN.

**Why:** Our policy has hit its representational ceiling. Search can compensate by
computing forward through difficult positions instead of relying on amortized NN output.

**Cost:** ~1-2 weeks implementation. Needs opponent modeling (opponent team is partially
hidden) + forward simulation via battle_server.

**Risk:** Inference latency. PokeChamp's LLM search times out 1/3 of games. Our search
would be faster (NN forward vs LLM call) but still adds per-move overhead.

**Expected gain:** MIT thesis achieved 1693 Elo with PPO+MCTS. Gen9 OU is harder than
Gen4 Random, but +200-400 Elo over the current policy is plausible.

### Option D — PokeAgent Challenge submission (READY TO GO)

**What:** Submit our model to the PokeAgent Challenge live benchmark. Script written
(`pokeagent_submit.py`), top teams identified (TEAM_AU at 78.5% savg).

**Why:** Ground-truth Elo against Metamon, Foul Play, and other agents. Our internal
"1058" is bot-anchored; external Elo could be very different.

**How to submit (exact steps):**

1. Go to https://battling.pokeagentchallenge.com
2. Click Login -> Create Team (creates an organization account)
3. Open "My Team" -> create a named AI agent (gets username + password credentials)
4. Run:
```bash
cd pokemon-ai-starter/pokemon-ai/src
python pokeagent_submit.py \
    --checkpoint data/models/rl_v9/selfplay_v9_20260413_061236/snapshot_2979.pt \
    --username <agent_username_from_step_3> \
    --password <agent_password_from_step_3> \
    --team TEAM_AU \
    --n-games 50 \
    --device cuda
```
5. Check rating at https://battling.pokeagentchallenge.com/ladder

Server: `wss://battling.pokeagentchallenge.com/showdown/websocket`
Discord for support: https://discord.gg/E2DuX5FWF7
Formats supported: Gen9 OU (our format), Gen1-4 OU, Gen9 VGC

**Cost:** ~1-2 hours for 50 games. No code changes needed. Uses existing BattleAgent.
Top team (TEAM_AU) scores 78.5% vs smart bots locally.

**Team selection results** in `team_selection_results.json`. Top 5: TEAM_AU (78.5%),
TEAM_T (77.5%), TEAM_G (77.0%), TEAM_B (75.0%), TEAM_C (73.5%). Use `--team random`
to rotate among top 10.

### Decision recommendation

**If you want decisive answers fast:** do D (PokeAgent submission) first — gives us an
external benchmark in days, informs A/B/C choice.

**If you want to push forward:** do A (capacity reallocation) before B (BC scaling).
A is cheaper (no new data) and tests the architectural hypothesis directly.

**If you want the highest ceiling:** do C (search). Highest-variance outcome but best
potential gain.

**For cloud deployment:** B + C + D path together. Scale up BC, add search, submit to
benchmark. Requires significant compute budget.

---

## Research context

See `docs/RESEARCH.md` §0 for full comparison tables. Key landscape facts (as of Apr 2026):

| System | Format | Elo | Model | Notes |
|--------|--------|-----|-------|-------|
| **Ours (sp2979)** | Gen9 OU | 1058 internal (bot-anchored) | 13.38M | Current peak |
| ps-ppo | Gen9 Random | 1900+ public ladder | ~15M | Random is easier than OU |
| VGC-Bench BCFP | VGC Doubles | 1768 internal | ~10M | Different format/scale |
| Metamon (Gen1OU) | Gen1 OU | 1500+ public | 15-200M | Published best for public ladder |
| Metamon Kakuna | Gen9OU | Unknown | 57M | Post-paper, no public Elo |
| PokeChamp | Gen9 OU | 1300-1500 | GPT-4 (no training) | LLM minimax search |
| PokeAgent PA-Agent | Gen9 OU | Unknown | RL-based | Co-won PokeAgent Track 1 |
| Foul Play | Multi-gen OU | Unknown | No NN (pure MCTS) | Co-won PokeAgent Track 1 |

**No pure RL agent has demonstrated >1800 Gen9 OU on public ladder** as of Apr 2026.
Top human Gen9 OU peaks: 2030-2115 Elo.

**Validated techniques from literature:**
- Entity tokenization (ps-ppo, VGC-Bench) — we have this
- Distributional value head (ps-ppo, Metamon via two-hot) — we have this
- Transformer with long context (Metamon, VGC-Bench) — we have this
- BC → PPO pipeline (all systems) — we have this
- PFSP opponent sampling (AlphaStar, VGC-Bench Double Oracle) — we have this
- Adaptive/constant entropy (EPO 2025) — we have this
- Slot permutation augmentation (no Pokemon paper; E2GN2 theory) — we have this
- Search + RL (MIT thesis, Foul Play, PokeChamp) — **we don't have this**

---

## Reference data

### Confirmed Elo (from `elo_exp5_FINAL.json`)

Top 10 snapshots:
```
 #1  sp2979  1058  [1044, 1070]   ← current all-time best
 #2  sp2999  1055  [1040, 1068]   ← tied (CI overlap)
 #3  sp3179  1053  [1039, 1065]
 #4  sp2299  1049  [1036, 1060]
 #5  sp2039  1047  [1034, 1058]
 #6  sp3199  1045  [1031, 1058]
 #7  sp2799  1042  [1027, 1056]
 #8  sp2359  1041  [1029, 1055]
 #9  sp2139  1040  [1026, 1051]
 #10 sp3019  1039  [1024, 1049]
```

Bot anchors:
```
  Tactical        1019  (top bot)
  Strategic       1018
  SmartDmg/SH     1000  (SH is anchor)
  Random          462   (floor)
```

Collapse reference:
```
  sp3054          785   ← Exp 4 post-collapse (below BC_base!)
  BC_base         817
```

### Critical checkpoints

- `data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt` — BC base (Elo 817), init-from for all PPO
- `selfplay_v9_20260413_061236/snapshot_2979.pt` — current all-time best (Elo 1058)
- `selfplay_v9_20260415_083340/emergency_iter_3055.pt` — Exp 4 collapse (DO NOT USE)

### Data files / registry
- `data/eval/elo_exp5_FINAL.json` — canonical Elo ladder (55 players, ~1400 matches)
- `data/eval/eras.json` — training era definitions (E0-E12)
- `data/eval/registry/runs.jsonl` — all training run configs
- `data/eval/registry/evals.jsonl` — per-iter bot evals (~200+ entries)
- `data/eval/registry/elos.jsonl` — all Elo measurements

### Infrastructure
- **Device:** RTX 3060 Laptop 6GB VRAM, 16GB RAM, Win10, Python 3.11, CUDA 12.1
- **Battle servers:** portable Node 20 at `tools/node-v20.18.1-win-x64/node.exe`
- **Use `127.0.0.1` not `localhost`** (Windows DNS quirk)

### Stale memmaps warning
`human_v8/memmap_v8` have `move_cont_dim=107, switch_cont_dim=28`. Current code expects
109/30. `dataset.py` auto-pads, but **regenerate memmaps before any BC training** to
get full type_eff features. Regeneration via `replay_to_memmap_v8.py`.

---

## Session handover checklist

For starting a new session:

1. **Read this file top-to-bottom.** It's self-contained.
2. **Check git status** (`git log --oneline -10`) for any un-handed-over work.
3. **Check for running processes** (`nvidia-smi | grep python`) — anything still training?
4. **Check latest Elo:** `data/eval/elo_exp5_FINAL.json` → confirm sp2979=1058 is still the best.
5. **If training didn't finish cleanly, check the emergency checkpoint** in the run dir.
6. **Pick a next step** from "What to do next" above based on time budget.
7. **If running any PPO training, enable safeguards**: `--adaptive-entropy --early-stop`.
8. **For cloud runs especially**: also `--win-rate-mode ema` (fixes staleness, saves compute).

### Known machine-level issues
- BSODs observed Mar 14-Apr 12 (multiple). Pattern stopped after Apr 12. If they resume,
  update NVIDIA drivers (nvlddmkm errors in Event Log) and uninstall MSI bloatware
  (Dragon Center/NBFoundation — crashed on reboot previously, implicated in WMI bugchecks).

---

## Memory file index

User-level memories in `memory/`:
- `MEMORY.md` — compact project summary loaded every session
- `feedback_*.md` — decisions/lessons that should persist
- `project_*.md` — project state notes
- `reference_*.md` — device/landscape references
