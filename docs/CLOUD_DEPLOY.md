# Cloud Deployment Guide (Session 33 final — post-Elo measurement)

## STATUS: Cloud is GATED on a successful BC scaling experiment first. Don't burn cloud money on the current architecture.

Session 33 measured our actual Elo via `eval_elo_ladder.py`. **Latest snapshot is at Elo 1032**
(extended ladder, 38 players, 703 matches, anchored to SH=1000). VGC-Bench BCFP claimed
**1768 Elo at our exact compute scale (5M states)** — a ~700 Elo gap (with cross-format anchor
caveats, but the magnitude is striking).

**Critical finding:** between snapshot_0589 (Elo 1015) and snapshot_1784 (Elo 1032), we have
**1200+ iters of training producing ~17 Elo of net change**, all within bootstrap CI overlap.
**More compute on the current architecture is empirically NOT the lever.** The plateau is real,
the architectural ceiling was reached at iter ~590 (right after type effectiveness features),
and 1500+ iters of additional training have produced statistical noise.

**Therefore: do NOT run the cloud burst on the current architecture.** The expected outcome
is +50-100 Elo over 2 days, which won't close the 700 Elo gap. The lever must be either
(a) stronger BC base — Metamon's "size matters for BC > RL" thesis — or (b) architectural
changes (capacity reallocation, ensemble critic, etc.). Both should be tested locally first.

## Revised plan: BC scaling first, cloud burst LATER

The new sequence (canonical: `docs/NEXT_SESSION.md` step c1-c5):

```
1. Code refactor (Task 8 — pending) — 1-2 days
2. Multi-gen vocab + feature prep — 1-2 weeks
3. Multi-gen replay scrape (gens 6/7/8 OU) — 1-2 weeks (mostly automated background)
4. 30M BC scaling test on multi-gen data — 3-7 days local (FP16, bs=2-4, 6GB ceiling)
   - Target: BC base reaches Elo 900+ (vs current BC_base 806)
5. PPO from new BC base + Elo measurement — 3-5 days
   - Target: PPO ceiling beats snapshot_1784 (Elo 1032) by >50 Elo
6. ONLY IF (4) and (5) succeed → cloud BC at 50M+ params, then cloud PPO
   - Cloud burst becomes valuable here because we're scaling a validated architecture
7. After cloud succeeds → multi-gen training data extension, eventually multi-gen ladder
```

## Decision criterion for cloud burst (revised, applies to step 6 above)

Two conditions, both required:
1. **Local BC scaling test (step 4) shows ≥ +90 Elo improvement** in the BC base
   (806 → 900+). This proves the scaling lever exists for our format at all.
2. **Local PPO from scaled BC (step 5) shows ≥ +50 Elo improvement** vs current PPO ceiling
   (1032 → 1080+). This proves the PPO ceiling actually moves with a stronger BC base.

If both are met → cloud burst is justified to scale the model further (50M+ params, more
training states). Success criterion for the cloud burst itself:
- ≥ +50 Elo delta vs the local PPO baseline (validated post-experiment)
- ≥ 20M states consumed during the cloud window (otherwise we haven't tested scaling)

If either local test fails → DO NOT run cloud burst. The architecture has issues that more
compute won't fix. Time for architectural pivot (capacity reallocation, ensemble critic,
different attention scheme).

## Why Cloud (eventually)

Compute reality:
- ps-ppo (1900 Elo, Random Battles): **250M states**, 2 days, RTX 3090
- VGC-Bench BCFP (1768 Elo, VGC Doubles): **5M states**
- Metamon (top 10%, OU): hundreds of millions+
- **Ours**: ~5-6M states total, Elo 1032

We're at VGC-Bench's compute scale but their setup was very different. The honest read:
**compute alone is not the missing piece** (we have empirical proof from the plateau). But
once we've validated that BC scaling helps locally, cloud lets us push BC scaling beyond
the 6GB GPU ceiling (50M+ instead of 30M) and gives us much faster training iterations.

## Decision Criterion (LEGACY — kept for context, superseded above)

~~**Success:** smart_avg breaks 60% within 2 days~~ (Session 32 framing — wrong metric)
~~**Success:** Elo > 1700 + cloud burst as scaling test~~ (Session 33 mid-research framing —
   superseded by the actual measurement showing Elo 1032)

These have been superseded. The current criterion is "BC scaling test passes locally, then
cloud burst scales the validated approach."

## Hardware Requirements
- **GPU**: A100 80GB (preferred) or 4090 24GB
- **CPU**: 16+ cores (1 per worker + main + servers)
- **RAM**: 32GB+
- **Disk**: 50GB for code + checkpoints + datasets

Recommended providers: Lambda, Vast.ai, RunPod (spot instances $1-2/hr A100)

## Files to Upload (~3GB total)

```
pokemon-ai-starter/pokemon-ai/src/
  rl_train_v9.py             # Main training script (with --mp, --pipeline, --batch-timeout-ms)
  rl_train_v8.py             # PPO update, trajectories, checkpoints
  mp_collect_v2.py           # Multiprocess collection v2 (queue-based)
  mp_collect_v3.py           # PSPPO style (separate InferenceServer process)
  policy_heads_v8.py         # Model architecture (FP32 summary_attn fix)
  features_v8.py             # Feature extraction (pure CPU function)
  bc_policy_player_v8.py     # Inference wrapper
  rewards.py                 # Reward shaping
  teams_ou.py                # Handcrafted teams (eval only)
  team_generator.py          # Procedural teams from Smogon usage
  battle_server.js           # Showdown server
  node_modules/              # Showdown npm package (or pip install pokemon-showdown)

  # Resume checkpoints
  data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt   # init-from (BC base)
  data/models/rl_v9/selfplay_v9_*/snapshot_LATEST.pt  # Latest local snapshot

  # Procedural teams (Smogon usage stats)
  raw_data/pokemon_usage/2024-04/

# Linux node binary (download separately)
tools/node-v20.18.1-linux-x64/
```

Don't upload backup folders, test scripts, or old logs — those are local-only.

## Setup Sequence

### 1. Provision instance
```bash
# Lambda Labs example
# Choose: 1x A100 80GB, Ubuntu 22.04, ~$1.10/hr spot
ssh ubuntu@<instance-ip>
```

### 2. Install dependencies
```bash
# Python 3.11
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip
python3.11 -m venv pokemon_env
source pokemon_env/bin/activate

# PyTorch with CUDA 12.1
pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Other deps
pip install poke-env==0.10.0 numpy pandas tensorboard tqdm

# Node.js (or use system node)
wget https://nodejs.org/dist/v20.18.1/node-v20.18.1-linux-x64.tar.xz
tar xf node-v20.18.1-linux-x64.tar.xz
export PATH=$PWD/node-v20.18.1-linux-x64/bin:$PATH

# Pokemon Showdown
npm install pokemon-showdown
```

### 3. Upload code
```bash
# From local machine
rsync -avz --exclude='backups/' --exclude='*.log' --exclude='__pycache__/' \
  pokemon-ai-starter/pokemon-ai/src/ ubuntu@<ip>:~/pokemon-ai/src/
rsync -avz raw_data/pokemon_usage/2024-04/ ubuntu@<ip>:~/pokemon-ai/raw_data/pokemon_usage/2024-04/
rsync -avz data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt ubuntu@<ip>:~/pokemon-ai/data/models/rl_v8/
rsync -avz data/models/rl_v9/selfplay_v9_<latest>/snapshot_<N>.pt ubuntu@<ip>:~/pokemon-ai/data/models/rl_v9/selfplay_v9_<latest>/
```

### 4. Start Showdown servers (8 instances)
```bash
cd ~/pokemon-ai/src
for port in 9000 9001 9002 9003 9004 9005 9006 9007; do
  node battle_server.js --port $port &
done
sleep 5  # let servers start
```

### 5. Launch training (Option A: Single-process + pipeline + compile)
```bash
# Simplest, ~5-8x faster than local due to A100 + torch.compile
python -u rl_train_v9.py \
  --init-from data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt \
  --resume data/models/rl_v9/selfplay_v9_<latest>/snapshot_<N>.pt \
  --device cuda --servers 9000 --fp16 --compile --pipeline \
  --games-per-iter 400 --max-concurrent 30 --n-iters 1000 \
  --warmup-iters 0 --immune-penalty 0.01 --ent-coef 0.04 --grad-accum 1 \
  --procedural-teams raw_data/pokemon_usage/2024-04 \
  2>&1 | tee training_cloud.log
```

### 5b. Launch training (Option B: Multiprocess + GPU opponents — MAX throughput)
```bash
# Need to set --opponent-device cuda for mp workers (A100 has 80GB, fits multiple CUDA contexts)
# IMPORTANT: this is what enables --mp to actually be faster than single-process
python -u rl_train_v9.py \
  --init-from data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt \
  --resume data/models/rl_v9/selfplay_v9_<latest>/snapshot_<N>.pt \
  --device cuda --servers 9000,9001,9002,9003,9004,9005,9006,9007 \
  --fp16 --compile --mp --opponent-device cuda \
  --games-per-iter 400 --max-concurrent 30 --n-iters 1000 \
  --warmup-iters 0 --immune-penalty 0.01 --ent-coef 0.04 --grad-accum 1 \
  --procedural-teams raw_data/pokemon_usage/2024-04 \
  --batch-timeout-ms 10 \
  2>&1 | tee training_cloud_mp.log
```

### 5c. Launch training (Option C: PSPPO style — InferenceServer separate process)
NOT YET WIRED INTO TRAINING LOOP. Standalone in `mp_collect_v3.py`.
On A100 this would beat both A and B but requires integration work first.

## Monitoring

### Trail logs
```bash
tail -f training_cloud.log | grep -E "FLOW|Iter|EVAL|Snapshot|ERROR|NaN"
```

### GPU
```bash
watch -n 5 nvidia-smi
```

### Track eval progression
```bash
grep "EVAL" training_cloud.log
```

## Expected Throughput

| Config | Expected iter time | States/day | Notes |
|--------|-------------------|------------|-------|
| Local (current) | 270s | ~2.1M | 3060, pipeline |
| Cloud A: single+pipeline+compile | ~50s | ~12M | A100 baseline |
| Cloud B: mp+compile (8 workers) | ~25-30s | ~22M | requires GPU opponents |
| Cloud C: PSPPO style (future) | ~20s | ~30M | separate process |

**At Cloud B rates: ~44M states in 2 days. ~6x ps-ppo's daily rate.**

## Cost Estimate

- A100 80GB spot: $1.00-2.00/hr
- 2-day burst: $48-96
- Includes setup time + buffer for restarts

## Fallback Plan

If --mp causes issues on cloud:
1. Switch to Option A (single-process + pipeline + compile)
2. Still gets ~12M states/day, ~24M in 2 days
3. Enough to test the compute hypothesis

If torch.compile causes issues (occasional incompatibilities):
1. Drop --compile, use --pipeline alone
2. Maybe 3-5x local speedup instead of 5-8x
3. Still meaningful, ~10M states/day

## Pre-Cloud Checklist (Session 33 final — post-Elo measurement)

**Gate 1 — Elo baseline established:** ✅ DONE Session 33
- [x] `eval_elo_ladder.py` built with permanent fix (PlayerPool + checkpoint cache + JSONL)
- [x] Latest snapshot Elo measured: **1032** [1009-1055] (extended ladder, 38 players, 703 matches)
- [x] BC_base Elo measured: **806** (the foundation)
- [x] Baseline JSON locked in: `data/eval/elo_session33_EXTENDED_FINAL.json`

**Gate 2 — code stability:**
- [x] Resilience patches in place (n_succeeded detection, FATAL guard) — done Session 33
- [x] Pre-cloud backup at `backups/v9_pre_cloud/` — done Session 33
- [ ] Code refactor (Task 8) — PENDING. rl_train_v9.py 1900 lines is portable as-is, but the
      refactor is the prerequisite for clean A/B experiments in step (c). Do this before
      the BC scaling test.
- [ ] Smoke test (`test_smoke_train.py`) — PENDING, part of refactor task

**Gate 3 — multi-gen prep done (per Session 33 user direction, do BEFORE BC scaling):**
- [ ] Vocab tables expanded for gens 6/7/8 (`vocab.py`, `features_v8.py` embedding sizes)
- [ ] Per-gen volatile lists added to `features_v8.py`
- [ ] Per-gen team generation in `team_generator.py`
- [ ] Multi-gen replay scrape complete (gen 6/7/8 OU added to `human_v3_memmap`)
- [ ] Sanity-check 1-epoch BC trains cleanly on the union dataset

**Gate 4 — local BC scaling test passes:**
- [ ] 30M BC trained on union dataset, fits on 6GB GPU (FP16, bs=2-4 + grad accum)
- [ ] **30M BC base Elo ≥ 900** (vs current BC_base 806). This is the gating threshold.
      If 30M BC stays at ~810, BC scaling isn't the lever — STOP, don't do cloud.
- [ ] PPO from new BC base, ~200-500 iters
- [ ] **PPO ceiling beats snapshot_1784 (Elo 1032) by ≥ +50 Elo.** This validates that
      BC scaling actually transfers to PPO ceiling. If not, STOP.

**Gate 5 — operational:**
- [ ] Cloud account, SSH key, budget alert set ($50-100 cap)
- [ ] `eval_elo_ladder.py` + the validated 30M architecture uploaded
- [ ] Cloud-side success criterion locked in: **≥ +50 Elo delta vs validated local baseline AND ≥20M states**

**Gate 6 — pull trigger:** all gates above checked. Cloud burst now scales a VALIDATED
architecture, not a hypothesis.

## Post-Cloud Analysis

After the burst, evaluate:
1. **Eval trajectory:** did it improve linearly, plateau, or oscillate?
2. **Final Elo vs validated local baseline:** Elo 1080+ (local target) → ?Elo (cloud)
3. **State count**: actual vs expected
4. **Cost**: actual spend
5. **Decision**: continue cloud (if working), or pivot architecture (if not)

## After Success (60%+)

Path to top 10%:
1. Continue cloud training to 65-70% (more days, ~$200-500)
2. Multi-gen pipeline (mostly data work)
3. Eventually live ladder play

## After Failure (still 50-55%)

Architecture changes to consider:
- Bigger model (50M+ params, like Metamon Kakuna)
- Stateless architecture (drop temporal, encode in features like ps-ppo)
- Different exploration (curiosity bonus, parameter noise)
- Offline RL on collected data (Metamon-style)
