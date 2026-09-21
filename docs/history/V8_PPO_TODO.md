# V8/V9 PPO — TODO List (prioritized)

## Status: Session 29 — v9 built, self-play ready to launch
## Hybrid PPO ran 240 iters, confirmed 22-26% plateau. v9 built with batched inference + pure self-play.
## Init from iter80 (52.8% H2H). 36% faster per step than v8.

---

## DONE (already implemented in rl_train_v8.py)

### ~~2. KL EARLY STOPPING~~ [DONE]
Implemented `--target-kl` with break at 1.5x. KL stable at ~0.030. Replaced KL penalty.

### ~~3. VALUE WARMUP PHASE~~ [DONE]
Implemented `--warmup-iters 5`. Freezes backbone + policy, trains only value head.
Value head calibrated (16 → 2.1 in 2 iters).

### ~~5. OPPONENT ON CPU~~ [DONE]
`--opponent-device` flag implemented. Currently running with `cuda` (GPU has headroom).

### ~~6. LAMBDA 0.75~~ [DONE]
Default changed to 0.75 in argparse.

### ~~7. OPTIMIZER RESET~~ [DONE]
Fresh optimizer created at PPO init in main(). No stale Adam momentum.

---

## REMAINING

## ~~1. BATCHED INFERENCE~~ [DONE — rl_train_v9.py]
Implemented via async choose_move + InferenceBatcher. poke-env natively supports Awaitable returns.
Batches spatial encoding across concurrent battles. Temporal runs per-item (can't batch due to
positional embedding mismatch with padded histories). 36% speedup (49 vs 36 steps/s with neural opp).

## ~~4. FP16 INFERENCE~~ [DONE — rl_train_v8.py + v9]
RL player uses torch.amp.autocast. Opponent stays FP32 (-1e9 masking overflows float16).
Summary cast to float32 before history storage.

## NEW: torch.compile [SPEED — FUTURE]
**What:** `torch.compile(model)` or `torch.compile(model.forward_spatial)` for optimized kernels.
**Why:** Metamon uses torch.compile for their GRU model. Could reduce per-forward latency.
**Risk:** May not work with our dynamic shapes (variable history length). Test carefully.
**Impact:** Potentially 1.5-2x speedup on spatial encoder.

## NEW: STATELESS MODEL (ps-ppo approach) [ARCHITECTURE — FUTURE/v10]
**What:** Remove temporal transformer entirely. Bake temporal info into observation features
(expand transition token with recent N turns of move history, effectiveness, KO events).
**Why:** ps-ppo is stateless and reached >1900 Elo. Makes batching trivially fast (no per-battle
state to manage). Would enable 256-2048 batch sizes instead of per-item temporal.
**Impact:** Massive speedup + simpler architecture. But loses 200-turn attention memory.
**When:** v10 consideration if temporal isn't providing enough value.

## 8. LARGER ROLLOUT BUFFER [TRAINING — LOW PRIORITY]
**What:** Accumulate multiple iterations of data before PPO update (ps-ppo uses 32768 steps).
**Why:** More stable gradient estimates per update.
**Current:** 200 games × ~33 turns = ~6500 steps/update. Smaller than ps-ppo's 32768 but
with KL early stopping already preventing bad updates, not clearly needed.
**Note:** Same total steps reached either way — this only affects per-update stability.

## 9. CONTINUOUS OFFLINE DATA (Metamon approach) [TRAINING — FUTURE]
**What:** Feed human replay data alongside self-play during PPO updates.
Metamon uses 90% offline (human) + 10% online (self-play) per batch.
**Why:** Continuous BC anchor — model can't forget human strategies.
**Impact:** Strongest possible BC anchoring. Requires infrastructure changes.

## ~~10. PURE SELF-PLAY~~ [DONE — rl_train_v9.py]
Implemented in v9. No bots in training. Uniform snapshot pool sampling.

## ~~11. TEMPERATURE RANDOMIZATION~~ [DONE — rl_train_v9.py]
SelfPlayOpponent randomizes temp from [1.0, 2.25] per game.

## 12. GAMMA=1.0 + TERMINAL-ONLY REWARD [TRAINING — FUTURE]
**What:** VGC-Bench uses gamma=1.0 (no discounting) + terminal-only reward (±1 win/loss).
**Why:** Simpler reward signal, no dense shaping noise. Model focuses purely on winning.
**When:** If self-play with current reward (ko_coef+hp_coef) plateaus.

## 13. CONTINUOUS OFFLINE DATA (Metamon approach) [TRAINING — FUTURE]
**What:** Feed human replay data alongside self-play during PPO updates.
Metamon uses 90% offline (human) + 10% online (self-play) per batch.
**Why:** Continuous BC anchor — model can't forget human strategies.
**When:** If self-play shows catastrophic forgetting of human strategies.

---

## Commands

### Local (Windows, RTX 3060 Laptop)
```
python -u rl_train_v9.py \
  --init-from data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt \
  --device cuda --servers 9000 --fp16 \
  --games-per-iter 200 --max-concurrent 10 --n-iters 500
```

### Cloud (Linux, A100/4090, 16+ cores)
```
python -u rl_train_v9.py \
  --init-from data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt \
  --device cuda --servers 9000,9001,9002,9003 --fp16 --compile \
  --games-per-iter 400 --max-concurrent 50 --n-iters 1000 \
  --snapshot-interval 5 --eval-interval 20
```

Cloud flags:
- `--compile`: torch.compile spatial encoder (~1.5-2x spatial, Linux only, safe no-op on Windows)
- `--max-concurrent 50`: more concurrent battles (bigger GPU headroom)
- `--games-per-iter 400`: more data per update (more cores = faster collection)
- More `--servers`: start 4+ battle_server.js instances on different ports
- `--opponent-device cpu`: viable on cloud with many cores (frees GPU for RL player)

Cloud setup:
1. Install Python 3.11+, PyTorch with CUDA, poke-env 0.10.0, Node 20+
2. Copy src/ directory + checkpoint file to cloud machine
3. Install npm packages: `cd src && npm install pokemon-showdown`
4. Start N Showdown servers: `for port in 9000 9001 9002 9003; do node battle_server.js --port $port & done`
5. Run with cloud flags above
6. Results compatible with local eval (same checkpoint format, same poke-env version)

Cloud status:
- All code is cloud-ready. No Windows-specific dependencies.
- `--compile` tested locally (safe no-op on Windows, works on Linux)
- `--pipeline` tested locally (slower on 1 GPU, should help with 2+ GPUs on cloud)
- Checkpoint format identical between local and cloud runs
- Expected cloud speedup: ~3-5x collection (torch.compile + higher concurrency + more CPU cores)
- No Ray dependency needed (our InferenceBatcher works with standard asyncio)

## Key Checkpoints
- Self-play init: `rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt` (52.8% H2H, best playstyle)
- BC safe copy: `bc/SAFE_v8_bc_human_bot_mix.pt`
- v8 backup: `backups/v8_source_backup/`
