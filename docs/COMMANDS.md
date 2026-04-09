# Commands & Workflows

## Prerequisites

```bash
# Install PyTorch with CUDA (for GPU training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install poke-env
pip install poke-env==0.10.0

# Ensure numpy < 2 (PyTorch compatibility)
pip install "numpy<2"
```

## Docker: Showdown Server

The Showdown server runs battles. Must be running for data generation and evaluation.

```bash
# Start Showdown (port 8000)
docker start showdown
# Or if container doesn't exist:
cd pokemon-ai-starter && docker compose up -d showdown

# Verify it's running (expect HTTP 404 — that's normal)
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/action.php

# Restart (clears stale player connections)
docker restart showdown

# IMPORTANT: Use 127.0.0.1 not localhost (IPv6 issue on Windows Docker Desktop)
```

## Data Generation (Observer)

Records bot-vs-bot battles as JSONL observation files.

```bash
cd pokemon-ai-starter/pokemon-ai/src

# Generate all bot pairings (11 bots × 11 = 121 pairings, 50 games each)
# Produces ~600k+ decision records with 2442-dim observations (v4)
PYTHONUNBUFFERED=1 python observer.py \
  --bots all \
  --all-mode all_pairings \
  --games 50 \
  --max-concurrent 8 \
  --batch-per-worker 5 \
  --parallel-pairings 3 \
  --format gen9ou \
  --log-both

# Generate extra data for strong bots (100 games each, 4×4=16 pairings)
for a in SimpleHeuristics SmartDamage Tactical Strategic; do
  for b in SimpleHeuristics SmartDamage Tactical Strategic; do
    PYTHONUNBUFFERED=1 python observer.py \
      --bots "$a,$b" --games 100 --max-concurrent 8 \
      --batch-per-worker 10 --format gen9ou --log-both
  done
done

# Generate specific bot pairing
python observer.py \
  --bots GreedySE,Random \
  --games 100 \
  --max-concurrent 8

# Available bots: Random, MaxDamage/MaxBasePower, SimpleHeuristics, GreedySE,
#   HazardSense, SwitchAwareEscape, SetupThenSweep, SmartDamage, Tactical, Strategic
# Output: src/data/datasets/obs/*.jsonl
```

### Observer CLI Options

| Arg | Purpose | Default |
|-----|---------|---------|
| `--max-concurrent` | Concurrent battles per pairing | 8 |
| `--batch-per-worker` | Games per player object (reuses ws connection) | 5 |
| `--parallel-pairings` | Pairings to run concurrently (for --bots all) | 3 |
| `--log-both` | Record both perspectives | off |
| `--battle-timeout` | Timeout per battle (seconds) | 600 |
| `--turn-cap` | Max turns before forfeit | 300 |

### Convert JSONL to Memmap

```bash
python convert_jsonl_to_memmap.py \
  --input-dir data/datasets/obs \
  --output-dir data/datasets/memmap
```

### Validate Generated Data

```bash
# Quick validation
python -c "
import json, glob
files = sorted(glob.glob('data/datasets/obs/*.jsonl'))
total = sum(1 for f in files for _ in open(f))
print(f'{len(files)} files, {total} records')
d = json.loads(open(files[0]).readline())
print(f'obs dim: {len(d[\"obs\"])}, ctx_extra: {len(d.get(\"ctx_extra\",[]))}, legal: {len(d[\"legal\"])}')
"
```

## BC Training (Behavioral Cloning)

Trains a policy network to imitate bot decisions.

### LSTM + MLP (recommended — captures turn-to-turn memory)

```bash
cd pokemon-ai-starter/pokemon-ai/src

# v4: 2442-dim observations with computed features
PYTHONIOENCODING=utf-8 python bc_train.py \
  --data "data/datasets/memmap" \
  --epochs 60 \
  --batch-size 16 \
  --lr 3e-4 \
  --device cuda \
  --run-name bc_v4_2442dim \
  --use-lstm \
  --lstm-hidden 256 \
  --lstm-layers 2 \
  --mlp-hidden 256 \
  --mlp-layers 2 \
  --seq-mode \
  --ctx-extra-dim 51 \
  --ctx-dropout 0.1 \
  --mods auto \
  --sched cosine \
  --warmup-steps 500 \
  --weight-decay 0.01 \
  --ema 0.999 \
  --label-smoothing 0.05 \
  --seed 42 \
  --log-csv \
  --workers 0 \
  --topk 3 \
  --no-amp

# Checkpoints saved to: data/models/bc/<run-name>/
```

## RL Training (PPO v4)

```bash
cd pokemon-ai-starter/pokemon-ai/src

# v4: from BC checkpoint, all features enabled
PYTHONUNBUFFERED=1 python rl_train.py \
  --init-from data/models/bc/bc_v4_2442dim/best.pt \
  --device cuda \
  --no-amp \
  --n-iters 200 \
  --games-per-iter 100 \
  --curriculum \
  --dense-rewards \
  --self-play \
  --adaptive-weights

# Resume from checkpoint
PYTHONUNBUFFERED=1 python rl_train.py \
  --resume checkpoints/rl/<run>/iter_XXXX.pt \
  --device cuda \
  --no-amp \
  --n-iters 200
```

### RL CLI Options (v4)

| Arg | Purpose | Default |
|-----|---------|---------|
| `--curriculum` / `--no-curriculum` | Tiered opponent difficulty | ON |
| `--dense-rewards` / `--no-dense-rewards` | Per-KO reward bonuses | ON |
| `--self-play` / `--no-self-play` | Periodic model snapshots as opponents | ON |
| `--adaptive-weights` / `--no-adaptive-weights` | Dynamic opponent weighting | ON |
| `--eval-games` | Games per eval opponent | 50 |
| `--promote-temp-bump` | Temperature bump on tier promotion | 0.15 |
| `--promote-lr-restart` | Cosine LR warm restart on promotion | ON |

## Evaluation

Test a trained model against bot opponents.

```bash
cd pokemon-ai-starter/pokemon-ai/src

PYTHONIOENCODING=utf-8 python eval_bc_vs_bots.py \
  --checkpoint data/models/bc/<run>/best.pt \
  --bots Random,MaxDamage,SimpleHeuristics,GreedySE,HazardSense,SwitchAwareEscape,SetupThenSweep,SmartDamage,Tactical,Strategic \
  --n-battles 50 \
  --format gen9ou \
  --device cuda \
  --max-concurrent 4
```

## Observation Vector (2442 dims — v4)

```
v1 (988): turn(1) + our_active(330) + opp_active(330) + move_type_hists(38)
  + our_bench(120) + opp_bench(120) + board_bits(43) + matchup(3) + moved_first(3)

v2 (+1406): active_stats(12) + bench_stats(60) + opp_active_moves(92)
  + opp_bench_moves(460) + our_bench_moves(460) + bench_items_abilities(320)
  + alive_counts(2)

v4 (+48): computed battle features
  Group A (16): matchup_score, type_advantages, speed_tier, stat_ratios,
    damage_scores, KO_checks, priority_signals
  Group B (10): bench_vs_opp matchups + defensive resist scores
  Group C (5): opp_bench threats vs our active
  Group D (5): aggregate bench signals (best switch, hazard cost)
  Group E (12): game context (endgame, boosts, HP_adv, status, weather, STAB)

ctx_extra (51): 26 board presence + 25 opponent last-action
```

## Key File Locations

| What | Path |
|------|------|
| Source code | `pokemon-ai-starter/pokemon-ai/src/` |
| JSONL data | `pokemon-ai-starter/pokemon-ai/src/data/datasets/obs/` |
| Memmap data | `pokemon-ai-starter/pokemon-ai/src/data/datasets/memmap/` |
| BC checkpoints | `pokemon-ai-starter/pokemon-ai/src/data/models/bc/` |
| RL checkpoints | `pokemon-ai-starter/pokemon-ai/src/checkpoints/rl/` |
| Training logs | `pokemon-ai-starter/pokemon-ai/src/data/logs/` |
| Teams | `pokemon-ai-starter/pokemon-ai/src/teams_ou.py` (30 teams) |
| Features | `pokemon-ai-starter/pokemon-ai/src/features.py` (2442-dim v4 encoder) |
| Project docs | `docs/` |
| Docker config | `pokemon-ai-starter/docker-compose.yml` |
