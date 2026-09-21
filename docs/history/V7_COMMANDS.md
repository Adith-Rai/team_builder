# V7 Training Commands — Transformer Architecture

## Prerequisites
```bash
# Start battle server (from project root)
NODE20="C:/Users/raiad/OneDrive/Desktop/team_builder/tools/node-v20.18.1-win-x64/node.exe"
cd pokemon-ai-starter/pokemon-ai/src
"$NODE20" battle_server.js --port 9000 &
```

## Step 1: Prepare Data — DONE

Using `human_v3_memmap` directly (10.1M records, 397K episodes, 83 GB).
Skipping concat with combined_v6 — the human data in v6 (1700+ scrape) overlaps
with v3 (1500+ superset), and bot data may hurt transformer learning.
Source JSONL (119 GB) deleted. combined_v6_memmap.tar.gz kept as backup.

## Step 2: BC with Transformer — DONE

Best: epoch 8 (val_loss=1.3281, val_acc=41.3%).
Checkpoint: `data/models/bc/v7_bc_transformer_lr1e4/best.pt`

```bash
# lr=1e-4 (not 3e-4 — diverged at bs=32). 500-step warmup.
python -u -X utf8 bc_train.py \
  --data-format memmap \
  --memmap-dir data/datasets/human_v3_memmap \
  --device cuda --epochs 10 --batch-size 32 --lr 1e-4 \
  --sched cosine --warmup-steps 500 --use-transformer \
  --n-transformer-layers 6 --n-heads 4 --transformer-dropout 0.1 \
  --context-length 128 --mlp-hidden 512 --mlp-layers 3 \
  --n-entity-ids 84 --embed-dim 32 --ctx-extra-dim 41 --step-type-bins 3 \
  --val-ratio 0.1 --run-name v7_bc_transformer_lr1e4 \
  --workers 2 --no-amp --seed 42
```

### BC Eval Results
- vs Random: 94% | vs MaxBP: 50% | vs SH: 23% | vs SmartD: 20% | vs Tactical: 22% | vs Strategic: 14%
- Smart avg: ~20% (same ceiling as LSTM BC)
- H2H vs LSTM models: 43.7% (loses to all — expected, BC only, no RL yet)
- Playstyle: heavy pivoting (17.5%), Spikes spam, learned competitive strategies from human data

## Step 3: IQL with Transformer — ABANDONED (offline RL dead end)

Three runs attempted, all failed to improve over BC:

| Run | Config | Eval WR | Result |
|-----|--------|---------|--------|
| 1 | beta=10, lr=1e-4 (ep1-8) | 12.5% (ep5) | Q val plateaued at 0.040. Acc stuck 40.6% |
| 2 | beta=5, lr=3e-4, lr-restart (ep9-11) | 14.0% (ep10) | LR destroyed value networks. Adv std: 0.11→0.02 |
| 3 | binary filtering, lr=1e-4, lr-restart (ep9-10) | 12.0% (ep10) | Healthy adv std (±0.11) but Q can't rank actions |

**Root cause:** IQL's Q-network learns state values but not action-specific values from offline data.
The advantage signal (±0.10) is too weak and noisy to meaningfully reshape the policy.
All three weighting schemes (exp beta=10, exp beta=5, binary median) produce the same result.

**Conclusion:** Offline RL cannot improve over BC with our setup. The Q-network needs
online interaction (self-play) to learn precise action values, not static replay data.

## Step 4: Self-Play PPO from BC — IN PROGRESS

Init from BC transformer checkpoint. Population-based self-play with:
- 2 most recent snapshots + 3 uniform random historical checkpoints + 3 hall of fame (top performers)
- KL penalty against BC reference (prevent forgetting)
- 9 rule-based bots in the opponent mix (smart=1.0, easy=0.3 weight)
- BC as permanent opponent (weight 2.0)
- Dense reward shaping (ko + hp + terminal)
- Pool/HoF/lineage state persisted in checkpoints for clean resume

### Self-play opponent pool (v6 — uniform historical sampling)
Pool rebuilt every 10 iters:
- **2 most recent** snapshots (FIFO, older deleted)
- **3 random historical** checkpoints (uniform sample from ALL checkpoints across training lineage)
- **3 hall of fame** (top 3 weighted-avg performers, auto-evicted when beaten)
- **BC** reference (permanent, weight 2.0)
- Total: 8 self-play + BC + 9 bots = 18 opponent sources

**Why uniform historical:** With only recent snapshots, the model oscillated between
degenerate strategies every ~40-60 iters (aggressive → stalling → pivoting → loop).
Random historical picks force the model to beat ALL phases of its own training, not
just the current one. Lineage tracking (`history_dirs`) ensures only same-architecture
checkpoints are sampled.

### Results
| Eval (iter) | Smart Avg | Weighted Avg | Notes |
|-------------|-----------|-------------|-------|
| 20 | 10.0% | 18.4% | Starting point |
| 40 | 21.5% | 30.5% | Strong climb |
| 80 | **26.0%** | **32.8%** | **Peak (50 games/iter era)** |
| 100 | 16.8% | 25.8% | Post-resume dip |
| 140 | 23.8% | 30.9% | Recovery |
| 180 | 16.8% | 24.9% | Stalling collapse |
| 240 | **26.2%** | **31.3%** | Bounce back |
| 260 | 19.0% | 26.4% | Oscillation continues |

**Oscillation pattern identified at iter 180:** Model cycles aggressive → stalling → pivoting.
Root cause: 5 recent snapshots all converge to same style, drowning out HoF diversity.
Fix: uniform historical sampling (v6) + 200 games/iter for more signal per opponent.

### H2H Tournament (100 games per matchup, 6 models)
| Model | Overall WR |
|-------|-----------|
| LSTM BC | 58.8% |
| **PPO SP Best (iter 80)** | **57.2%** |
| LSTM IQL ep10 | 53.0% |
| PPO SP iter 140 | 51.5% |
| PPO SP Latest (iter 100) | 47.0% |
| Transformer BC | 39.0% |

### Playstyle Evolution (BC → PPO iter 80 → PPO iter 140)
- Attack %: 69% → 87% → 62-73% (aggressive peak then balanced)
- Pivot %: 17.5% → 3% → 11.5% (recovered pivoting)
- Hazards %: 3.2% → 1% → 8.5% (learned Stealth Rock/Spikes)
- Recovery %: 4.9% → 2% → 7.4-8.9% (learned Roost/Recover)
- KO ratio: 0.86 → 1.14 → 0.89-1.04 (context-dependent)
- Immune hits: 32% → 29-40% (still type-blind)
- Clodsire emerged as key support mon (SR + EQ + Recover + Toxic)
- Corviknight shifted from Brave Bird spam to U-turn pivoting

### Current run (v6 — from iter 320)
- **200 games/iter**, max-concurrent 15
- Uniform historical sampling — 3 random picks from full lineage every 10 iters
- **BC weight reduced** to 0.5 (was 2.0) — ~5% of games. KL penalty is the real anchor.
- Self-play now ~51% of games (was ~35%), bots ~44%
- Lineage tracking saves history_dirs in checkpoint (no `--history-dirs` needed on future resumes)
- Eval trend (v6 era): smart avg stabilized at 20-23%, oscillation dampened

```bash
# Start battle server first (see Prerequisites)
# KEY DIFFERENCES from previous PPO (Session 24, plateaued at 27%):
#   1. Self-play with snapshot pool — adaptive opponents, not fixed bots
#   2. Hall of fame — best checkpoints permanently in opponent pool (VGC-Bench fictitious play)
#   3. Smart bot weights reduced — self-play is primary signal, bots provide diversity
#   4. All bots included at low weight — diverse strategies prevent overfitting
#   5. KL penalty anchors to BC — prevents catastrophic forgetting
#   6. Transformer (20.78M) not LSTM (3.85M)
#
# Previous PPO plateau cause: fixed bots → model learns bot-specific exploits,
#   adaptive weights cause oscillation (beats A → upweight B → learns B → forgets A → loop)
#
# Opponent distribution per iter (~200 games):
#   ~24% self-play pool (4.0 weight across 8 entries: 2 recent + 3 historical + 3 HoF)
#   ~24% BC reference (2.0 weight, prevents forgetting)
#   ~40% smart bots (SH, SmartDamage, Tactical, Strategic at 1.0 each)
#   ~12% easy bots (MaxBP, GreedySE, HazardSense, SwitchAware, Setup at 0.3 each)
#
# v6 uniform historical: every 10 iters, 3 random checkpoints sampled from full
#   training lineage (all run dirs). Prevents strategy oscillation.
#   Lineage tracked in checkpoints — no manual dir management after first seed.
# BC weight 0.5 (not 2.0): KL penalty is the real anchor, BC opponent is just diversity.

python -u -X utf8 rl_train.py \
  --init-from data/models/bc/v7_bc_transformer_lr1e4/best.pt \
  --device cuda \
  --opponent-device auto \
  --servers 9000 \
  --games-per-iter 200 \
  --max-concurrent 15 \
  --n-iters 1000 \
  --lr 1e-4 \
  --lr-schedule cosine \
  --ppo-epochs 4 \
  --ent-coef 0.01 \
  --kl-coef 0.1 \
  --dense-rewards \
  --self-play \
  --self-play-interval 10 \
  --self-play-weight 4.0 \
  --snapshot-pool-size 5 \
  --snapshot-hall-of-fame 3 \
  --bc-opponent-weight 0.5 \
  --selfplay-weights \
  --no-curriculum --no-adaptive-weights \
  --temperature 1.0 \
  --temp-decay 0.999 \
  --eval-interval 20 \
  --eval-games 100 \
  --save-interval 20 \
  --no-amp \
  --out-dir data/models/rl
```

## Step 5: Eval

```bash
# 100-game eval vs bots
python -u eval_bc_vs_bots.py \
  --checkpoint data/models/rl/BEST_CHECKPOINT.pt \
  --bots "Random,MaxBasePower,SimpleHeuristics,SmartDamage,Tactical,Strategic" \
  --n-battles 100 --device cuda --max-concurrent 5 \
  --server ws://127.0.0.1:9000/showdown/websocket

# H2H vs BC + LSTM models
python -u eval_head_to_head.py \
  --checkpoints \
    data/models/rl/BEST_CHECKPOINT.pt \
    data/models/bc/v7_bc_transformer_lr1e4/best.pt \
    data/models/iql/v6_iql_combined_bs32/epoch_010_policy.pt \
  --names ppo_selfplay transformer_bc lstm_iql_ep10 \
  --n-battles 100 --device cuda \
  --server ws://127.0.0.1:9000/showdown/websocket
```

## Resume Commands

```bash
# Resume PPO (MUST include --init-from for BC opponent + KL penalty)
# --history-dirs only needed on FIRST resume if checkpoint lacks lineage data.
# After that, lineage is saved in checkpoints automatically.
python -u -X utf8 rl_train.py \
  --resume data/models/rl/RUNDIR/iter_XXXX.pt \
  --init-from data/models/bc/v7_bc_transformer_lr1e4/best.pt \
  --device cuda --opponent-device auto --servers 9000 \
  --games-per-iter 200 --max-concurrent 15 --n-iters 1000 \
  --lr 1e-4 --lr-schedule cosine --ppo-epochs 4 \
  --ent-coef 0.01 --kl-coef 0.1 --dense-rewards \
  --self-play --self-play-interval 10 --self-play-weight 4.0 \
  --bc-opponent-weight 0.5 \
  --snapshot-pool-size 5 --snapshot-hall-of-fame 3 \
  --selfplay-weights --no-curriculum --no-adaptive-weights \
  --eval-interval 20 --eval-games 100 --save-interval 20 \
  --no-amp --out-dir data/models/rl
```

## Notes
- Transformer is the new default (20.78M params vs 3.85M LSTM)
- Old LSTM checkpoints still work via --use-lstm flag
- **LR**: 1e-4 for transformers. 3e-4 diverges at bs=32.
- **Self-play**: population-based (snapshots + hall of fame + BC + bots), NOT pure self-play (causes strategy collapse)
- **KL penalty**: anchors policy to BC baseline, prevents catastrophic forgetting
- **Opponent device**: `--opponent-device auto` uses same as training (cuda). Speeds up self-play inference.
- **Concurrency**: `--max-concurrent 10` safe with current resources (GPU 11%, CPU 54%, 8GB RAM free)
- **No curriculum/adaptive weights**: causes oscillation (beat A → upweight B → forget A → loop)
- **Hall of fame selection**: weighted avg (smart bots 1.0, easy bots 0.3) — prevents gaming easy bots
- **Pool/HoF/lineage persistence**: snapshot pool, hall of fame, and history_dirs all saved in checkpoints
- **Uniform historical sampling**: every 10 iters, 3 random checkpoints from full training lineage. Prevents strategy oscillation.
- **Lineage tracking**: `history_dirs` persisted in checkpoint. `--history-dirs` CLI flag for one-time seeding only. Not needed on subsequent resumes.
- **Resume MUST include --init-from**: without it, BC opponent and KL penalty are missing (bc_ckpt_path = None). Warning printed if missing.
- Workers=2 for BC, workers=1 for IQL/PPO (RAM constraint)
- Workers=0 is safe fallback (2.5x slower)

## Device Resources (during PPO self-play training)
- **GPU**: 39% util, 1.5 GB / 6 GB VRAM, 53°C — with opponent on GPU + HoF loaded
- **CPU**: 54% — battle sim + opponent inference
- **RAM**: 7.5 GB free / 16 GB — training uses 2.2 GB, rest is Windows/system
- **Scaling tested**: max-concurrent 10, opponent on GPU, 200 games/iter — all stable
- **Timing (200 games/iter)**: ~141s collect + ~45s update = ~190s/iter. Eval ~6 min every 20 iters.

## Lessons Learned (v7)

### BC
- Transformer BC matches LSTM BC ceiling (~20% vs smart bots) — architecture doesn't matter for imitation
- lr=1e-4 + 500 warmup for small-batch transformers. lr=3e-4 diverges.
- Val accuracy plateaus at ~41% on diverse 1500+ human data (many equally valid actions per turn)

### IQL (offline RL) — dead end
- Q-network learns state values but not action-specific values from offline data
- Advantage signal too weak (±0.10 std) and noisy to reshape policy meaningfully
- Exponential weighting (any beta), binary filtering — all produce same result
- **Offline RL can only reshuffle existing data.** It cannot discover new strategies.

### Self-Play PPO — working (with oscillation challenge)
- **Self-play breaks the fixed-bot ceiling.** Previous PPO on fixed bots plateaued at 27%. Self-play reached 26% smart avg.
- **BC anchor is CRITICAL.** KL penalty + BC opponent prevent catastrophic forgetting. Model collapsed in <10 iters without them.
- **Strategy oscillation identified.** Model cycles through degenerate strategies every ~40-60 iters: aggressive → stalling (Roost spam) → over-pivoting (U-turn spam) → aggressive. Root cause: 5 recent snapshots all converge to same style, drowning out HoF diversity.
- **Fix: uniform historical sampling (v6).** Sample 3 random checkpoints from full training lineage instead of only keeping recent snapshots. Forces model to beat ALL phases of its own history.
- **200 games/iter better than 100.** More signal per opponent = stabler gradients. Each opponent gets ~10-15 games instead of 4-5.
- **Type blindness persists.** 29-40% immune hits. Self-play can't teach the type chart because neither side punishes type-blind play.
- **Playstyle evolves toward strategy.** Later iters develop hazards (Clodsire SR), pivoting (Corviknight U-turn), recovery, status — more like competitive play than pure aggression.

### What to try in v8 (if self-play PPO plateaus)
- **Per-move type effectiveness features** — inject damage_multiplier per move slot. Model doesn't need to learn the type chart, just when to exploit it. Easy features.py change.
- **Metamon-style actor-critic**: two-hot value classification + Q-maximization in policy loss (major refactor)
- **Search at inference**: use trained NN as value function for 1-2 step lookahead (MIT thesis approach)
- **Include our previous action in observations** — Metamon does this, we don't. Helps cause-effect learning.
