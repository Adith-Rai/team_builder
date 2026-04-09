# V8 Commands — PokeTransformer Training Pipeline

## Prerequisites

```bash
# Battle server (from project root)
NODE20="C:/Users/raiad/OneDrive/Desktop/team_builder/tools/node-v20.18.1-win-x64/node.exe"
cd pokemon-ai-starter/pokemon-ai/src
"$NODE20" battle_server.js --port 9000 &
```

## Architecture Summary

PokeTransformer: 16 entity tokens, spatial (384d/4L/4H) + temporal (384d/2L/4H/200-turn).
13.38M params. 0.35 GB peak VRAM (PPO scenario). Full spec: docs/V8_PLAN.md.

## New Files (v8)

| File | Purpose |
|------|---------|
| `features_v8.py` | Structured per-entity feature extraction (replaces flat 1480-dim vector) |
| `policy_heads_v8.py` | PokeTransformer model (spatial + temporal + distributional value) |
| `dataset_v8.py` | MemmapV8Dataset + collate_seq_v8 + unpack_turn_batch |
| `bc_train_v8.py` | BC training loop with batched spatial + sequential temporal |
| `bc_policy_player_v8.py` | Inference wrapper for live battles (per-battle temporal history) |
| `convert_jsonl_to_memmap_v8.py` | JSONL to structured memmap converter |
| `replay_to_memmap_v8.py` | HuggingFace replays direct to memmap (no JSONL intermediate) |

---

## Step 1: Generate Bot Data (proof of concept)

Bot-vs-bot data using all 10 bots. Fast to generate, validates the pipeline.
observer.py with --v8 flag calls features_v8.make_v8_features() instead of flat vector.
Output JSONL has "v8": true marker and structured per-entity fields.

```bash
# Generate v8 bot data (~360K records, ~17 min)
# All 10 bot pairings, 50 games each, both perspectives
python -u -X utf8 observer.py --v8 \
  --bots all --all-mode all_pairings \
  --games 50 --max-concurrent 8 --log-both \
  --format gen9ou \
  --server ws://127.0.0.1:9000/showdown/websocket \
  --batch-per-worker 5 --parallel-pairings 3

# Convert JSONL to structured memmap
python -u convert_jsonl_to_memmap_v8.py \
  --data "data/datasets/obs_v8/*.jsonl" \
  --out-dir data/datasets/memmap_v8

# Delete JSONL after conversion (memmap is authoritative)
rm -rf data/datasets/obs_v8/
```

### Bot data results (Session 28)
- 115 JSONL files, 360,881 rows, 12,104 episodes, 0 skipped
- 7.86 GB memmap, all dimensions validated
- Conversion: 2-pass streaming, 249s scan + 424s write

---

## Step 2: Generate Human Data (1500+ rating)

Streams replays from HuggingFace directly to memmap — no JSONL intermediate.
Saves ~150+ GB of disk vs the old JSONL pipeline.
Each episode is validated before writing (dims, NaN, completeness, legal mask, result).

```bash
# Stream 80K human replays at 1500+ directly to v8 memmap
# Both perspectives (--log-both default), ~4M records, ~84 GB memmap
# Pre-allocates 5M rows, trims unused space via episode_index
python -u replay_to_memmap_v8.py \
  --min-rating 1500 \
  --max-replays 80000 \
  --max-rows 5000000 \
  --out-dir data/datasets/human_v8_memmap

# Why 1500+: Balances diversity (wider skill range, more strategies) with quality.
# ps-ppo used only bot demos for BC. Metamon used all available replays.
# 1500+ players know type matchups, hazards, switching — exactly what we need.
#
# Why 80K replays: ~4M records at ~22 KB/row = ~84 GB memmap.
# Fits within 100 GB budget. More than Metamon's 950K IL sequences.
#
# Why direct-to-memmap: 80K replays at v8 JSONL size (~60 KB/row) would be
# ~240 GB of intermediate JSONL. Direct-to-memmap skips this entirely.
#
# Estimated time: ~1.6 hours at 14 replays/sec.
# RAM usage: ~9 GB (poke-env battle objects + numpy buffers). 7-8 GB free.
# Disk: pre-allocates 105 GB, actual usage depends on episode count.
```

---

## Step 3: BC Training

Batched spatial processing (all turns at once) + sequential temporal (per-turn).
16x faster inference, 1.7x faster training vs naive turn-by-turn approach.

```bash
# BC on bot data (proof of concept, ~2.5 hours for 5 epochs)
python -u -X utf8 bc_train_v8.py \
  --memmap-dir data/datasets/memmap_v8 \
  --device cuda --epochs 5 --batch-size 8 --lr 1e-4 \
  --warmup-steps 100 --workers 0 --run-name v8_bc_bot

# BC on human data — Metamon-style config (RECOMMENDED)
# Metamon (UT-Austin, human replays, top 10%): lr=1e-4, constant schedule,
# weight_decay=1e-4, grad_clip=2.0, 500 epochs with early stopping patience=2.
# ps-ppo (bot data only): lr=1e-4, warmup->hold->power-law, weight_decay=1e-2.
# Both use lr=1e-4. We follow Metamon since we train on human data like they did.
# Keep dropout=0.1 (higher than Metamon's 0.05) because we have 6x less data.
python -u -X utf8 bc_train_v8.py \
  --memmap-dir data/datasets/human_v8_memmap \
  --device cuda --epochs 10 --batch-size 8 --lr 1e-4 \
  --sched constant --warmup-steps 1000 \
  --weight-decay 1e-4 --grad-clip 2.0 \
  --workers 0 --run-name v8_bc_human_metamon \
  --resume data/models/bc/v8_bc_human_v3/SAVE_ep2b1k_51pct.pt

# Alternative: ps-ppo schedule (warmup -> hold 20K steps -> power-law decay)
python -u -X utf8 bc_train_v8.py \
  --memmap-dir data/datasets/human_v8_memmap \
  --device cuda --epochs 10 --batch-size 8 --lr 1e-4 \
  --sched psppo --warmup-steps 1000 --hold-steps 20000 \
  --workers 0 --run-name v8_bc_human_psppo

# Resume from checkpoint (same hyperparams — keeps optimizer state)
python -u -X utf8 bc_train_v8.py \
  --memmap-dir data/datasets/human_v8_memmap \
  --device cuda --epochs 20 --batch-size 8 --lr 1e-4 \
  --sched constant --warmup-steps 1000 --weight-decay 1e-4 --grad-clip 2.0 \
  --eval-games 200 --server ws://127.0.0.1:9000/showdown/websocket \
  --workers 0 --run-name v8_bc_human_fresh_opt \
  --resume data/models/bc/v8_bc_human_v3/SAVE_ep2b1k_51pct.pt

# Resume with DIFFERENT hyperparams — MUST use --lr-restart to reset optimizer!
# Without --lr-restart, Adam momentum from old config fights new config.
# Proven: old optimizer caused accuracy to DECLINE from 42% to 39.9%.
python -u -X utf8 bc_train_v8.py \
  --memmap-dir data/datasets/human_v8_memmap \
  --device cuda --epochs 10 --batch-size 8 --lr 1e-4 \
  --sched constant --warmup-steps 1000 --weight-decay 1e-4 --grad-clip 2.0 \
  --eval-games 200 --server ws://127.0.0.1:9000/showdown/websocket \
  --workers 0 --run-name v8_bc_human_fresh_opt \
  --resume data/models/bc/v8_bc_human_v3/SAVE_ep2b1k_51pct.pt \
  --lr-restart

# Architecture scaling (all are CLI flags with defaults for RTX 3060):
#   --d-model 384           hidden dim (default, fits 6GB)
#   --n-spatial-layers 4    spatial transformer depth
#   --n-temporal-layers 2   temporal transformer depth
#   --n-heads 4             attention heads
#   --temporal-context 200  turns of history
#   --temporal-mode summary summary (local) or frames (cloud)
#   --v-bins 51             distributional value bins
#   --gradient-checkpoint   save VRAM (~30% slower)
```

### Training speed benchmarks (RTX 3060 6GB, bs=8)

| Version | Time/batch | Time/epoch (10.8K ep) | Bottleneck |
|---------|------------|----------------------|------------|
| v1 (naive turn-by-turn) | 6.7 s | 2.5 hours | Sequential GPU calls |
| v2 (batched spatial) | 1.7 s | 38 min | Temporal loop + backward |
| v3 (batched temporal too) | 0.1 s (inference) | ~18 min est. | Backward pass |

Key optimization: forward_sequence() batches ALL turns' spatial processing in one
GPU call (the expensive part), then runs temporal + heads per-turn (cheap: just
vector operations on 384-dim summaries).

### Bot BC results (Session 28, 5 epochs, 360K rows)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 0 | 1.699 | 26.1% | 1.693 | 26.0% |
| 1 | 1.380 | 54.6% | 1.129 | 58.5% |
| 2 | 1.085 | 59.7% | 1.009 | 61.1% |
| 3 | 0.990 | 63.6% | 0.923 | 65.6% |
| 4 | 0.927 | 65.8% | 0.884 | 66.6% |

Loss still decreasing at epoch 4. Val > train (no overfitting). 66.6% accuracy on bot data
(bots are predictable, but validates the architecture learns entity relationships).

---

## Step 4: Eval vs Bots

```bash
# Quick inline eval (50 games per opponent)
python -u -X utf8 -c "
import asyncio
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer, RandomPlayer
from policy_smartbots import SmartDamagePlayer, TacticalPlayer, StrategicPlayer
from bc_policy_player_v8 import BCPolicyPlayerV8
from teams_ou import random_teambuilder

SERVER = ServerConfiguration('ws://127.0.0.1:9000/showdown/websocket', None)

async def eval_vs(opp_cls, opp_name, n=50):
    p1 = BCPolicyPlayerV8(
        'data/models/bc/v8_bc_bot_opt/best.pt', device='cuda',
        account_configuration=AccountConfiguration.generate('V8Bot', rand=True),
        battle_format='gen9ou', max_concurrent_battles=5,
        server_configuration=SERVER, team=random_teambuilder(),
    )
    p2 = opp_cls(
        account_configuration=AccountConfiguration.generate(opp_name, rand=True),
        battle_format='gen9ou', max_concurrent_battles=5,
        server_configuration=SERVER, team=random_teambuilder(),
    )
    await p1.battle_against(p2, n_battles=n)
    wr = p1.n_won_battles / n * 100
    print(f'  vs {opp_name:20s}: {p1.n_won_battles}/{n} = {wr:.0f}%')
    return wr

async def main():
    for cls, name in [
        (RandomPlayer, 'Random'), (MaxBasePowerPlayer, 'MaxBasePower'),
        (SimpleHeuristicsPlayer, 'SimpleHeuristics'),
        (SmartDamagePlayer, 'SmartDamage'), (TacticalPlayer, 'Tactical'),
        (StrategicPlayer, 'Strategic'),
    ]:
        await eval_vs(cls, name)

asyncio.run(main())
"
```

### Bot eval results (Session 28, v8 BC on bot data, epoch 4)

| Opponent | V8 BC | V7 BC | V7 PPO (self-play) | Change vs V7 BC |
|----------|-------|-------|-------------------|-----------------|
| Random | 96% | 94% | — | +2% |
| MaxBasePower | 62% | 60% | — | +2% |
| SimpleHeuristics | **64%** | 20% | 23% | **+44%** |
| SmartDamage | 24% | 26% | 27% | -2% |
| Tactical | 24% | 18% | 22% | +6% |
| Strategic | 0% | 12% | 20% | -12% |
| **Smart avg** | **28.0%** | **19.0%** | **23.0%** | **+9.0%** |

Key findings:
- **28% smart avg beats both v7 BC (19%) and v7 self-play PPO ceiling (23%)** with
  just 5 epochs of BC on bot data. No RL, no human data yet.
- **64% vs SimpleHeuristics** (was 20%) — entity attention is learning type relationships.
  SH relies on type-based heuristics; v8 model learned to exploit/counter them.
- **0% vs Strategic** — Strategic does long-term planning (hazard stacking, PP stalling,
  safe switches). Model needs more training + human data to learn these patterns.
  Also only 50 games (high variance).
- **SmartDamage/Tactical at 24%** — these bots hard-code damage calculations.
  Model is still learning damage estimation from entity embeddings (5 epochs is early).

Why the skew: The bot training data is dominated by SimpleHeuristics patterns
(most common bot across pairings). The model learned SH-like play well → beats SH.
Strategic-like play is rare in bot data → model can't handle it yet.
Human data will fix this — human players use diverse strategies including Strategic-like play.

### Human BC results (Session 28, 3 epochs, 4M rows, 1500+ rating)

Training: lr=3e-4, bs=8, cosine decay over 54K batches, warmup 200 steps.
~4 hrs/epoch. RAM leak mitigated with gc.collect + empty_cache every 50 batches.
Mid-epoch checkpoints every 1000 batches.

| Epoch | Val Acc | Smart Avg | SH WR | Strategic WR | Notes |
|-------|---------|-----------|-------|-------------|-------|
| 0 | 43.5% | 16.5% | 13% | 18% | Learning basic patterns |
| 1 | 44.1% | 22.5% | 43% | 41% | Offense emerging |
| 2 (b1k) | ~43.8% | **51.2%** | **89%** | **84%** | Aggressive STAB + hazards |

KEY INSIGHT: Val accuracy is a POOR proxy for battle strength. Accuracy plateaued at
43-44% but win rate jumped from 22% to 51%. The model learned WHICH decisions to get
right (attack with coverage, commit to plays) even though overall prediction barely changed.

The "accuracy jump" at epoch boundaries is a REPORTING ARTIFACT: the training loop
reports a running average from batch 1, which is dragged down by early batches. At
epoch start, the counter resets, showing the model's current (higher) ability.
LR is smooth cosine decay (no reset): 0.000300 -> 0.000226 -> 0.000075 -> 0.000030.

### Playstyle evolution (human BC vs SimpleHeuristics)

| Metric | ep0 (0%) | ep1 (17%) | ep2b1k (97%) |
|--------|----------|-----------|-------------|
| Attack % | 36% | 43% | 60% |
| Switch % | 22% | 24% | 13% |
| Recovery % | 16% | 15% | 8% |
| Hazards % | 8% | 4% | 8% |
| KO ratio | 0.39 | 0.64 | 1.48 |
| Top move | Seismic Toss | Roost | Hurricane |

ep0: Passive stall (imitating defensive human play without understanding)
ep1: Timid offense (diverse moves but too much switching)
ep2b1k: Aggressive STAB player (Hurricane/Surf/EQ + SR, commits to attacks, KO ratio 1.48)

### H2H vs v7 models (100 games each)

v8_human_best beats v7_lstm_bc (56%) and v7_ppo_best (56%) but loses to v7_iql_ep10 (30%).
BC teaches general strategy; RL teaches adversarial adaptation. v8's stronger base (51% smart avg)
should produce much stronger self-play PPO than v7's 19% base did.

Best checkpoint: data/models/bc/v8_bc_human_v3/mid_epoch2_batch1000.pt

### Research-validated BC hyperparameters

| Setting | Metamon (human data) | ps-ppo (bot data) | Ours (current) | Ours (next) |
|---------|---------------------|-------------------|----------------|-------------|
| LR | 1e-4 | 1e-4 | 3e-4 | **1e-4** |
| Schedule | Constant (cosine eta_min=lr) | Warmup->hold->power-law | Cosine | **Constant** |
| Weight decay | 1e-4 | 1e-2 | 1e-2 | **1e-4** |
| Grad clip | 2.0 | 0.5 | 1.0 | **2.0** |
| Dropout | 0.05 | 0.0 | 0.1 | **0.1** (more data=lower, less data=higher) |
| Warmup | None | 1000 steps | 200 steps | **1000 steps** |
| Epochs | 500 (early stop pat=2) | Step-based 500K | 3-5 | **10+** |
| Data | 475K replays (human) | Bot demos | 80K replays (human) | Same |

Key decisions:
- Follow Metamon over ps-ppo for BC config since we train on human data like they did
- Keep dropout at 0.1 (higher than Metamon's 0.05) because we have ~6x less data
- lr=1e-4 is standard — both papers agree. Our 3e-4 was too aggressive for human patterns
- Constant LR lets the model keep learning at full strength across all epochs
- Light weight decay (1e-4) preserves rare strategic patterns in human data
- Eval each epoch's win rate (not accuracy) to decide when to stop — val accuracy is a
  poor proxy for battle strength (plateaued at 43% while win rate jumped 22%->51%)

LR schedule options (--sched flag):
- constant: flat after warmup (Metamon-style, RECOMMENDED for human data BC)
- psppo: warmup -> hold N steps -> power-law decay (ps-ppo style)
- cosine: single cosine decay (default, decays too fast for open-ended training)

---

## Step 5: Self-Play PPO

`rl_train_v8.py` — complete PPO self-play training loop for v8 PokeTransformer.

All v7 self-play infrastructure ported: snapshot pool (FIFO 2), hall of fame (top 3),
uniform historical sampling, BC anchor + KL penalty, dense reward shaping.

Memory management from v7: reset_battles(), PSClient listener cancel, gc.collect +
empty_cache after every batch/update/eval.

```bash
# Start battle servers (2 instances recommended for PPO — 5.4 g/s vs 4.7 g/s single)
"$NODE20" battle_server.js --port 9000 &
"$NODE20" battle_server.js --port 9001 &

# Self-play PPO from best BC checkpoint
# Best BC: ep3_b5k (step 58982) — aggressive/priority style, 56% SmDmg, 28% SH/Tactical
# Chose this over balanced ep3_b10k because aggressive style gives PPO better foundation.
# Passive models learn to stall in self-play; aggressive models learn to adapt.
python -u rl_train_v8.py \
  --init-from data/models/bc/BEST_v8_bc_step58982_aggressive_priority.pt \
  --device cuda --opponent-device auto \
  --servers 9000,9001 \
  --games-per-iter 100 --max-concurrent 10 \
  --n-iters 500 --lr 1e-4 \
  --gamma 0.9999 --lam 0.8 \
  --ppo-epochs 5 --ent-coef 0.02 --vf-coef 0.5 \
  --target-kl 0.03 --max-grad-norm 0.5 \
  --warmup-iters 5 \
  --self-play --self-play-interval 10 --self-play-weight 4.0 \
  --snapshot-pool-size 5 --snapshot-hall-of-fame 3 \
  --bc-opponent-weight 0.5 \
  --eval-interval 20 --eval-games 200 \
  --save-interval 20 \
  --temperature 1.0 --temp-decay 0.999 \
  --out-dir data/models/rl_v8

# Resume from RL checkpoint (no warmup since value head already calibrated)
python -u rl_train_v8.py \
  --init-from data/models/bc/BEST_v8_bc_step58982_aggressive_priority.pt \
  --resume data/models/rl_v8/ppo_v8_XXXXXXXX_XXXXXX/iter_XXXX.pt \
  --device cuda --servers 9000 \
  --games-per-iter 100 --max-concurrent 10 \
  --self-play --n-iters 500
```

### PPO Hyperparameters (research-validated)

| Param | Value | Source | Reason |
|-------|-------|--------|--------|
| gamma | 0.9999 | ps-ppo | Value winning regardless of game length |
| lambda | 0.8 | V8_PLAN | Less variance with distributional value head |
| lr | 1e-4 | Metamon + ps-ppo | Both papers agree |
| entropy | 0.02 | V8_PLAN | More exploration (ps-ppo used 0.001 but had 250M states) |
| kl_coef | 0.05 | v7 finding | Higher than v7's 0.01 — v8 BC is stronger, worth preserving |
| vf_coef | 1.0 | new | Distributional CE loss is already well-scaled |
| clip_eps | 0.2 | standard | PPO default |
| ppo_epochs | 3 | ps-ppo | Per update |
| grad_norm | 0.5 | ps-ppo | Tighter than v7's 1.0 |
| temperature | 1.0→0.5 | v7 | Decay 0.999/iter for exploration→exploitation |

### Reward Shaping (unchanged from v7)

terminal ±1.0 + ko_coef=0.05 per KO + hp_coef=0.02 per HP delta.
Returns fit within distributional support [-1.6, 1.6].
No tempo_tax (penalizes strategic play). No hazard/status bonuses (let model learn from outcomes).

### Self-Play Opponent Distribution (~100 games/iter)

- ~24% self-play pool (2 recent + 3 historical + 3 HoF)
- ~5% BC reference (anchor against forgetting)
- ~40% smart bots (SH, SmartDmg, Tactical, Strategic at 1.0 each)
- ~12% easy bots (MaxBP, GreedySE, etc. at 0.3 each)

### Memory Management (from v7 lessons)

- `reset_battles()` + `completed_trajectories.clear()` after each opponent batch
- `_cancel_listener()` kills PSClient zombie websocket coroutines
- Trajectory data stored on CPU (not GPU) to prevent VRAM accumulation
- `gc.collect()` + `torch.cuda.empty_cache()` after PPO update and eval
- Each opponent batch uses fresh Player objects (prevents stale connection cascade)

### Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `rl_train_v8.py` | 574 | Full PPO self-play loop |
| `bc_policy_player_v8.py` | 200 | Inference player (opponents + eval) |
| `rewards.py` | — | Dense reward shaping (unchanged from v7) |
| `bc_train_v8.py` | — | `eval_vs_bots()` reused for PPO eval |

---

## Resource Usage

### Training (BC, bs=8, RTX 3060 6GB)
- GPU: 27% util, 5.0/6.1 GB VRAM, 60-65C
- CPU: 25% (temporal loop is single-threaded Python)
- RAM: ~9.5 GB used (memmap + tensors + model)
- Bottleneck: temporal per-turn loop (CPU-bound)

### Data generation (observer.py --v8, bot data)
- GPU: 0% (CPU only)
- CPU: ~40% (battle sim + feature extraction)
- RAM: ~3 GB
- Speed: 34 g/s (bot-vs-bot), ~17 min for all pairings

### Human data (replay_to_memmap_v8.py)
- GPU: 0% (CPU only)
- CPU: ~25% (single-threaded streaming)
- RAM: ~9 GB (poke-env battle objects)
- Speed: 14 replays/sec, ~1.6 hours for 80K replays
- Disk: pre-allocates memmap at max_rows, actual usage depends on episode count

### Inference (bc_policy_player_v8.py)
- GPU: ~15% per concurrent battle
- Latency: ~6ms per turn (spatial + temporal forward)
- Memory: per-battle history buffer (384 dims × turns, negligible)

---

## Device Config

- GPU: RTX 3060 Laptop, 6 GB VRAM, CUDA 12.1
- CPU: i7-11375H, 4 cores
- RAM: 16.9 GB (need ~8 GB free for training)
- Disk: 931 GB total, target 200+ GB free
- Node 20: `tools/node-v20.18.1-win-x64/node.exe`
- poke-env 0.10.0, Python 3.11, Windows 10

---

## Lessons Learned (Session 28)

### Optimizer state matters on resume
When changing LR or weight_decay, ALWAYS use --lr-restart. Adam's momentum terms (m, v)
are calibrated for the old config. Loading stale momentum with new hyperparams causes the
model to DECLINE. Proven: accuracy dropped 42% → 39.9% and smart_avg dropped 51% → 18.6%.
Fresh optimizer from same weights stays stable at 43.9%.

### Val accuracy is NOT win rate
Accuracy plateaued at 43-44% across epochs, but bot eval smart_avg varied 18-51%.
The model learns WHICH decisions matter, not overall prediction accuracy.
The "jump" at epoch boundaries is a reporting artifact (running average resets).
Always use bot eval (--eval-games 200) as the real metric. best.pt saved by smart_avg.

### Eval needs 200+ games
50-game evals with random teams are unreliable. Same checkpoint: 51% in one eval, 20%
in another. Root cause: random_teambuilder() picks from 70 teams, some hard-counter specific
bots. 200 games per bot gives statistically meaningful results.

### RAM leaks on Windows with large memmaps
108 GB memmap + PyTorch DataLoader on 16 GB RAM causes gradual RAM exhaustion.
Mitigated with gc.collect() + torch.cuda.empty_cache() every 50 batches.
Mid-epoch checkpoints (every 1000 batches) prevent losing progress to OOM crashes.

### VRAM fluctuation is normal
GPU memory varies 1100-5960 MiB between batches because episode lengths vary.
Short episodes (10 turns) = small spatial mega-batch. Long episodes (50 turns) = large.
torch.cuda.empty_cache() every 50 batches prevents fragmentation buildup.

### Constant LR > cosine for open-ended training
Cosine decay assumes a fixed compute budget. We keep extending training and changing
epoch counts. Constant LR (after warmup) keeps full learning power throughout.
Both Metamon and ps-ppo effectively used constant LR for their IL/BC phases.

### Playstyle analysis reveals what accuracy hides
Two models at 43% accuracy can play completely differently:
- ep2b1k: 60% attacking, 13% switching, Hurricane/Surf/EQ, KO ratio 1.48 (dominant)
- Metamon-run ep2: 37% attacking, 27% switching, U-turn spam, KO ratio 0.51 (degenerate)
Always analyze replays alongside win rates.

### Direct mode (--direct) is slower than battle_server.js
Direct mode: 1.5 g/s (single subprocess serializes I/O regardless of concurrency).
Server mode: 5.4 g/s with 2 instances. Use --servers 9000,9001 for PPO, not --direct.

### PPO update optimization
Same batched-spatial trick from BC applies to PPO: batch all T turns' spatial processing,
then sequential temporal only. ~30x speedup for PPO update step (360ms → 12ms per episode).
Battle collection is still the bottleneck (Showdown sim speed, not model inference).
