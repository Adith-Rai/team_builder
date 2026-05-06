# PPO Cloud Cookbook — current state

**Authoritative reference for running PPO training on cloud (RunPod A100 80GB).**
Last validated Session 50 (2026-05-06). Phase 1 v3 production launched 21:46 UTC.

---

## TL;DR — canonical command

```bash
python train_rl.py \
  --init-from data/models/bc/v10_cloud_gen9/epoch_003.pt \
  --pool-anchors data/models/bc/v10_cloud_gen9/epoch_003.pt \
  --device cuda \
  --servers 9000,9001,9002,9003,9004,9005,9006,9007 \
  --fp16 --mp --mp-workers 8 \
  --games-per-iter 1600 --max-concurrent 200 \
  --n-iters 200 --warmup-iters 20 \
  --lr 1e-5 --lam 0.95 --ent-coef 0.02 --reward-style terminal \
  --grad-accum 1 \
  --adaptive-entropy --adaptive-entropy-low 0.65 --adaptive-entropy-high 0.95 \
  --win-rate-mode ema --win-rate-ema-alpha 0.3 --win-rate-ema-window 50 \
  --eval-interval 20 --eval-team-set metamon-competitive --eval-games 200 \
  --snapshot-interval 5 --early-stop --early-stop-patience 3 \
  --turn-cap 300 \
  --procedural-teams /workspace/raw_data/pokemon_usage/2024-04 \
  --out-dir data/models/rl_v10/<run_name>
```

Expected: ~12-15 min/iter steady, **~$60-70 for 200 iters on A100 SXM 80GB**.

---

## 1. Pre-flight checklist

Run BEFORE every cloud launch to avoid surprises.

### 1a. Data files on pod

| Path | Source | Purpose | Size |
|---|---|---|---|
| `data/models/bc/v10_cloud_gen9/epoch_003.pt` | R2: `models/bc/v10_cloud_gen9/` | BC base ckpt for `--init-from` | 240 MB |
| `data/vocab/*.json` (5 files) | scp from local | Species/items/abilities/moves vocab | ~120 KB |
| `data/lookup/move_flags_v1.pt` | scp from local | Move flags lookup for transformer | 558 KB |
| `/workspace/raw_data/pokemon_usage/2024-04/` (256 files) | R2: `raw_data/pokemon_usage/2024-04/` | Procedural team generation (training) | ~5 MB |
| `/workspace/metamon_cache/teams/competitive/gen9ou/` (16 files) | scp from local | Eval team set (`--eval-team-set metamon-competitive`) | 75 KB |

### 1b. System setup (Linux/RunPod container)

```bash
# CRITICAL — required for --mp to work
ulimit -n 65536              # default 1024 fails on N>=4 mp workers (FD exhaustion)

# REQUIRED for torch CUDA on RunPod base images
apt-get install -y libcudnn8  # without this: torch import fails with libcudnn.so.8 missing

# Recommended (we use these in launch script):
export OMP_NUM_THREADS=4      # if seeing CPU oversubscription
```

**`vm.max_map_count`**: read-only in RunPod containers. We work around this with Pipe-based IPC (no SemLock) — see §3 below.

### 1c. Pod cleanup before launch

```bash
# Kill any stale processes
pkill -9 python 2>/dev/null
pkill -9 -f train_rl 2>/dev/null
pkill -9 -f forkserver 2>/dev/null
sleep 2

# Wipe stale shared memory + tmp files
rm -f /dev/shm/sem.mp-* /dev/shm/torch_*
rm -f /tmp/weights_iter*.pt /tmp/traj_w*_iter*.pkl.gz
rm -rf /workspace/sweep_runs   # leftover test runs

# Wipe any zombie screens (keep battle_servers!)
screen -wipe
```

### 1d. Verify battle_servers running

```bash
# Should see 8 ports listening (9000-9007)
ss -ltn | grep -E ':900[0-7]'

# If not, start them via:
for p in 9000 9001 9002 9003 9004 9005 9006 9007; do
  screen -dmS bs_$p bash -c "node battle_server.js --port $p 2>&1 | tee /tmp/battle_server_$p.log"
done
```

### 1e. R2 sync loop (recovery insurance)

Run this in a separate screen alongside training:

```bash
# /tmp/r2_sync_loop.sh
source /workspace/team_builder/pokemon-ai-starter/pokemon-ai/scripts/r2_env.local.sh
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
while true; do
  aws s3 sync data/models/rl_v10/<run_name>/ \
    s3://team-builder-data/models/rl_v10/<run_name>/ \
    --endpoint-url $S3_ENDPOINT_URL \
    --exclude "*" \
    --include "snapshot_*.pt" --include "*.json" --include "config.json" \
    --include "win_rates.json" --include "evals.json" --include "final.pt" \
    --quiet 2>&1
  sleep 300
done
```

Worst-case loss bound on pod death: 5 min of progress.

---

## 2. Architecture: which flag does what

| Flag combo | Implementation | Status | Use case |
|---|---|---|---|
| (none) | `collect_v9` sync, single python process | ✅ works | Local dev, smoke tests |
| `--pipeline` only | `BackgroundCollector` (rl_pipeline.py) — main process bg thread + deepcopy model | ✅ works | Pipeline-only baseline |
| `--mp` only | `mp_disk_collect.py` — N forkserver workers, per-worker GPU model copy + own InferenceBatcher, traj→disk at iter end | ✅ **production** | Cloud throughput, transformer arch only |
| `--mp --pipeline` | Falls through to `--mp` only (no-op for bg overlap — see §6 known limitations) | ⚠️ silent no-op | Treats as `--mp` only |

**Critical**: `--mp` is **transformer-only**. Legacy `BattleAgent` ckpts cannot be `--init-from` for `--mp`. Use them only as PFSP opponents (factory dispatches via `is_transformer_checkpoint`).

### Recommended config matrix

| Run type | Flags | Why |
|---|---|---|
| Phase 1 production | `--mp --mp-workers 8` | 4× faster collect than pipeline-only at production scale |
| Local 6GB GPU smoke | `--pipeline` (no `--mp`) | mp not supported on CPU; pipeline gives modest speedup |
| Numerical baseline | (no flags) | Slowest but simplest reference |

---

## 3. Cloud quirks (Session 50 hard-won lessons)

### 3a. SemLock race at N>=4 mp workers

**Symptom**: `FileNotFoundError: [Errno 2]` in child during spawn. Reliably fires at N≥4 spawn workers, near-100% at N=8.
**Root cause**: CPython 3.11 `multiprocessing.resource_tracker` unlinks SemLock files in `/dev/shm/sem.mp-*` before spawn children open them.
**Why our containers**: RunPod containers have `vm.max_map_count` capped + can't bump (sysctl read-only). Plus shared `/dev/shm` is contention-prone.

**Fix (already in `mp_disk_collect.py`)**: replace ALL `mp.Queue` with `mp.Pipe`. Pipes are FD-only (no SemLock). Per-worker `ctrl_pipe` + `result_pipe`, multiplexed via `multiprocessing.connection.wait`. **0 SemLocks per spawn.**

```python
# Why we use spawn context (not forkserver):
# - forkserver had same SemLock race
# - spawn fully isolates child Python state, sidesteps shared resource_tracker
# - spawn is slower per-startup but workers persist across iters (one-time cost)
```

### 3b. Heartbeat starvation during model load

**Symptom**: workers spawn fine, then declared `stale_heartbeat` ~60s later. Watchdog respawns. Cascade.
**Root cause**: 8 workers loading 240MB ckpt simultaneously from disk takes 30-60s under contention. Worker can't send heartbeat until load done. Default 60s timeout fires.

**Fix (in `mp_disk_collect.py`)**:
- `HEARTBEAT_TIMEOUT_S = 300.0` (was 60s)
- Workers send ack-heartbeat IMMEDIATELY on cmd receipt (before slow model load)
- Liveness probe in separate thread — distinguishes "asyncio dead" from "process dead"

### 3c. mp+pipeline overlap GPU contention

**Symptom**: `--mp --pipeline` works for iter 0, hangs at iter 1. Workers stall, never recover.
**Root cause**: when `mp_bg_collector.start()` runs at end of iter K, workers begin processing iter K+1 cmd in PARALLEL with main's PPO update (heavy `optimizer.step()`). GPU contention causes worker CUDA forwards to stall. Stalled forwards don't recover even after main's update finishes.

**Current workaround**: `--mp --pipeline` silently treats as `--mp` only (no-op for bg overlap). See `train_rl.py:_start_background_collection`.

**Real fix (deferred to multi-gen prep)**: redesign as centralized inference server (workers send obs to single GPU process, which queues forwards on appropriate CUDA streams). ~2-3 day project. Saves $200-300 over multi-gen run vs $10-15 on Phase 1.

### 3d. SelfPlayOpponent factory dispatch

**Symptom**: with `--mp` and transformer init, workers crash with `Missing key(s) in state_dict: tokenizer.actor_token, ...`.
**Root cause**: `mp_collect_v2.py:402`, `mp_collect_v3.py:312`, `rl_pipeline.py:429` originally used raw `SelfPlayOpponent(...)` (BattleAgent class). When loading transformer ckpt, key shape mismatch.

**Fix (already in repo)**: replaced with `make_self_play_opponent(...)` factory in all 3 files. Factory dispatches on `is_transformer_checkpoint(_cached_ckpt)`. Legacy ckpts still work as PFSP opps.

### 3e. argparse `%` escape bug

**Symptom**: `--help` raises `ValueError: unsupported format character ')' (0x29) at index 63`.
**Root cause**: pre-existing bug at `train_rl.py:96` — help text contains `%)` which argparse tries to format.
**Fix**: replace `±10%)` with `+/-10 percent)` in the help string. Already patched.

### 3f. Forkserver cmd queue draining (legacy `mp_collect_v2`)

**Note**: legacy `mp_collect_v2.py` is no longer reachable from `--mp` flag (replaced by `mp_disk_collect.py`). Kept in repo for reference. Don't use.

---

## 4. Hyperparameters (validated for transformer arch + lr=1e-5)

| Flag | Value | Why this value |
|---|---|---|
| `--lr` | **`1e-5`** | Transformer arch (20M params, 220 tokens) is sensitive. 3e-5 (legacy validated) caused regression at the new scale. Confirmed by 4-point lr ablation Session 50. |
| `--lam` | `0.95` | GAE lambda. Session 39 validated. |
| `--ent-coef` | `0.02` | Session 39 validated. With adaptive-entropy active, this is just the starting point. |
| `--target-kl` | `0.03` (default) | KL early stop threshold. Validated. |
| `--grad-accum` | **`1`** | **Mandatory.** Values >1 caused stability issues historically (per docs). |
| `--reward-style` | `terminal` | Session 43+ validated. Was `dense` earlier. |
| `--adaptive-entropy-low/high` | `0.65 / 0.95` | Session 43 safeguards entropy collapse. |
| `--win-rate-mode` | `ema` | Forgets old data in PFSP weighting; prevents stuck weights when policy beats old snapshot. |
| `--win-rate-ema-alpha` | `0.3` | Smoothing constant. |
| `--win-rate-ema-window` | `50` | Effective games cap; bounds influence of single batch. |
| `--turn-cap` | `300` | Forfeit turn budget. T-quadratic memory means going higher (e.g., 1000) costs ~10× more VRAM per battle. |
| `--snapshot-interval` | `5` | Save every 5 iters. With 200 iters → ~40 snapshots → pool curated to 15 via `--pool-max-current-run`. |
| `--eval-interval` | `20` | Smart-bot eval every 20 iters. With 200 iters → 10 eval points. |
| `--eval-games` | `200` | 200 × 4 bots = 800 games per eval. SE ≈ ±3.5%. |
| `--eval-team-set` | `metamon-competitive` | Fixed 16-team set. Consistent benchmark. |
| `--early-stop-patience` | `3` | Stops if 3 consecutive evals regress past noise threshold. |
| `--mp-workers` | `8` | Matches 8 battle_servers (one server per worker). VRAM: ~17GB on 80GB A100. |
| `--max-concurrent` | `200` | 8 × 200 = 1600 simul battles. Battle_server capacity validated. |
| `--games-per-iter` | `1600` | Matches `--mp-workers 8 × --max-concurrent 200`. 100 games per opp at pool=15. |
| `--warmup-iters` | `20` | Value head re-equilibration at lr=1e-5. Eval at iter 19 (= end of warmup, first real signal). |

### Init from BC vs PPO snapshot

For Phase 1 (BC→PPO), use `--init-from data/models/bc/v10_cloud_gen9/epoch_003.pt`. **NOT** legacy snapshot files (sp_NNNN). Legacy ckpts can be PFSP opponents but not init.

`--pool-anchors` should be the same BC ckpt — pins it in pool forever (never pruned), so weight only decays via PFSP `(1-wr)²`.

---

## 5. Validation pattern (small-scale → production)

Before launching production (~$60-70 commit), validate at small scale (~$1):

| Test | Scale | Validates |
|---|---|---|
| **A** | 5-iter `--mp --mp-workers N` at games=200, conc=20 | Sustained correctness, no NaN/drift, weight reload across iters |
| **B-mp** | 1-iter `--mp` at games=200, conc=200 | mp metrics match pipeline-only baseline |
| **B-pipe** | 1-iter `--pipeline` at games=200, conc=200 | Reference baseline metrics |
| **C** | 3-iter `--mp --pipeline` at games=200 | Currently no-op for bg overlap; will revisit |
| **D** | 5-iter `--mp` + manual `kill -9 worker` mid-iter | Watchdog respawn, slice drop, run continues |

**Pre-launch acceptance**:
- Iter line metrics within noise: |wr_diff| < 5%, |pi_loss_diff| < 0.05, |v_loss_diff| < 0.3, |kl_diff| < 0.01
- No NaN in any metric
- Workers shut down cleanly at end
- smart_avg ≥ BC baseline (67% for v10 e3) on iter 4 of sustained test

---

## 6. Wall time + cost (empirical, Session 50)

### Per-iter timing (RunPod A100 80GB)

| Config | games | iter 0 | iter 1+ steady |
|---|---|---|---|
| `--pipeline` only (Test A) | 1500, conc=500 | 30 min | 30 min |
| `--mp` only (Test A) | 200, conc=20 | ~6 min | ~8-12 min (with pool growth) |
| `--mp` only (extrapolated production) | 1600, conc=200, N=8 | ~13-14 min | ~12-13 min |
| `--mp --pipeline` (broken) | — | iter 0 OK, iter 1 hang | — |

### 200-iter cost projection

| Config | Wall time | Cost ($1.50/hr) |
|---|---|---|
| `--mp` only @ games=1600 | 40-45 hr | **$60-70** |
| `--pipeline` only | ~100 hr | $150 |
| `--mp --pipeline` (if fixed) | ~33 hr | $50 |

---

## 7. Common errors + immediate fixes

| Error | Cause | Fix |
|---|---|---|
| `OSError [Errno 24] Too many open files` | ulimit -n default 1024 | `ulimit -n 65536` before launch |
| `libcudnn.so.8: cannot open shared object` | Missing libcudnn8 on container | `apt install -y libcudnn8` |
| `FileNotFoundError: SemLock._rebuild` | mp.Queue race at N>=4 spawn | Already patched (Pipe-only IPC). If reproducing in new code, use `mp.Pipe` not `mp.Queue` |
| `Missing key(s) in state_dict: tokenizer.*` | SelfPlayOpponent loaded transformer ckpt with legacy class | Use `make_self_play_opponent()` factory (already patched) |
| Workers `stale_heartbeat` immediately | Heartbeat timeout too short for model load contention | Already 300s tolerance. If hitting, reduce N or increase tolerance |
| iter 1 hangs with `--mp --pipeline` | mp+pipeline overlap = GPU contention | Already silently downgrades to `--mp` only |
| `[FATAL] PPO update: 0 succeeded (0 failed, 0 episodes)` | Workers never returned trajs | Check log for traceback above this line; usually upstream worker crash |
| argparse `--help` crashes | Pre-existing `%` in help text | Already patched |

---

## 8. Active TODOs (post-Phase 1 v3)

### Multi-gen prep (next major work session)

1. **mp+pipeline redesign** (~2-3 day engineering project): centralized inference server. Saves $200-300 over multi-gen run.
2. **Multi-gen support** (per `MULTIGEN_FEASIBILITY.md`):
   - Gen-id token in TransformerBattlePolicy (~1 week)
   - Per-gen procedural teambuilder (gen-aware filtering)
   - Multi-gen replay assembly (HuggingFace)
   - BC v11 multi-gen retrain (~1-2 weeks compute)
3. **Per-gen eval bots** if multi-gen run goes through

### Smaller cleanups

- Migrate old runs from `data/models/rl_v9/*` to R2-only (free local disk)
- Document smart-bot eval baseline ranges per gen for benchmarking

---

## 9. Pod bootstrap (fresh A100 SXM 80GB)

### One-time setup commands

```bash
# 1. Repo + deps
cd /workspace
git clone https://github.com/Adith-Rai/team_builder.git
cd team_builder/pokemon-ai-starter/pokemon-ai
pip install --no-deps -r requirements.txt
apt install -y libcudnn8  # required for torch import

# 2. Verify GPU + cuda
python -c "import torch; assert torch.cuda.is_available(); print('cuda ok:', torch.cuda.get_device_name(0))"

# 3. Node 20 + npm deps for battle_server
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs
cd pokemon-ai-starter/pokemon-ai/src && npm install

# 4. R2 credentials (recreate from local r2_env.local.sh, gitignored)
cat > pokemon-ai-starter/pokemon-ai/scripts/r2_env.local.sh <<EOF
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="auto"
export S3_ENDPOINT_URL="https://....r2.cloudflarestorage.com"
export S3_BUCKET="team-builder-data"
EOF
chmod 600 pokemon-ai-starter/pokemon-ai/scripts/r2_env.local.sh
pip install awscli
```

### Sync data from R2 + scp from local

```bash
source pokemon-ai-starter/pokemon-ai/scripts/r2_env.local.sh

# BC ckpt (240 MB, ~2 min)
mkdir -p pokemon-ai-starter/pokemon-ai/src/data/models/bc/v10_cloud_gen9
aws s3 cp s3://team-builder-data/models/bc/v10_cloud_gen9/epoch_003.pt \
  pokemon-ai-starter/pokemon-ai/src/data/models/bc/v10_cloud_gen9/epoch_003.pt \
  --endpoint-url $S3_ENDPOINT_URL --quiet

# Procedural teams (256 files, ~10 sec)
aws s3 sync s3://team-builder-data/raw_data/pokemon_usage/2024-04 \
  /workspace/raw_data/pokemon_usage/2024-04 \
  --endpoint-url $S3_ENDPOINT_URL --quiet
```

### scp from local (vocab, lookup, eval teams — gitignored)

```bash
# From your laptop, after pod is up:
scp -i ~/.ssh/id_ed25519 -P <pod_port> \
  pokemon-ai-starter/pokemon-ai/src/data/vocab/*.json \
  root@<pod_ip>:/workspace/team_builder/pokemon-ai-starter/pokemon-ai/src/data/vocab/

scp -i ~/.ssh/id_ed25519 -P <pod_port> \
  pokemon-ai-starter/pokemon-ai/src/data/lookup/move_flags_v1.pt \
  root@<pod_ip>:/workspace/team_builder/pokemon-ai-starter/pokemon-ai/src/data/lookup/

scp -i ~/.ssh/id_ed25519 -P <pod_port> -r \
  metamon_cache/teams/competitive/gen9ou/*.gen9ou_team \
  root@<pod_ip>:/workspace/metamon_cache/teams/competitive/gen9ou/
```

### Start battle_servers

```bash
cd pokemon-ai-starter/pokemon-ai/src
for p in 9000 9001 9002 9003 9004 9005 9006 9007; do
  screen -dmS bs_$p bash -c "node battle_server.js --port $p 2>&1 | tee /tmp/battle_server_$p.log"
done
ss -ltn | grep -E ':900[0-7]'  # verify all 8 listening
```

### Then proceed with §1 pre-flight + canonical launch

---

## 10. Architecture summary (mp-disk design)

For deeper architectural reference see `docs/MP_DISK_REDESIGN.md`. Quick summary:

```
┌─ Main process ─────────────────────────────┐
│  - Owns model + optimizer + scheduler      │
│  - PPO update loop, eval, snapshots        │
│  - Per-worker ctrl_pipes (parent→worker)   │
│  - Per-worker result_pipes (worker→parent) │
│  - Multiplexes via mp.connection.wait      │
│  - Saves weights atomically + signals      │
│    workers via small ctrl msg (filename    │
│    only, NEVER torch.tensor)               │
└─────────────┬──────────────────────────────┘
              │ Pipes only (no SemLock)
              ▼
┌─ Worker × N=8 ──────────────────────────────┐
│  Spawn-context Python process. Each:        │
│  - Loads main model from /tmp/weights_iter*│
│  - Maintains LRU cache of opp ckpts (=3)   │
│  - Owns InferenceBatcher (private)         │
│  - Owns asyncio loop, conc=200 battles     │
│  - Plays games_per_iter/N games            │
│  - Liveness probe thread (heartbeats main) │
│  - At iter end: writes traj_w<id>_iter<N>  │
│    .pkl.gz to /tmp, posts done             │
│  - Listens on ctrl_pipe for next cmd       │
└─────────────────────────────────────────────┘
```

**Why this design**:
- Tensor IPC is the diagnosed root cause of mmap explosion in v2 mp design
- Disk I/O at iter boundaries is bounded (~50 MB/iter)
- Workers stay alive across iters; spawn cost paid once per run (~25-40s)
- Backward compat preserved: `make_self_play_opponent` factory dispatches on arch

**Worker manager / health**:
- Heartbeat protocol (workers send every 15-30s via async + immediate ack on cmd receipt)
- Watchdog in main: `is_alive()` + `last_heartbeat < 300s`
- Respawn on death/hang. Cap 3 respawns/5 iters → mark dead.
- Liveness probe thread (stdout-only, separate from asyncio) — distinguishes "asyncio stuck" from "process dead"

---

## 11. References

- **Architectural design**: `docs/MP_DISK_REDESIGN.md`
- **Multi-gen scope**: `docs/MULTIGEN_FEASIBILITY.md`
- **Phase 1 history (postmortems)**: `docs/PHASE1_POSTMORTEM.md`, `docs/PHASE1_DIAGNOSIS_REPORT.md`, `docs/PHASE1_INVESTIGATION_PLAN.md`
- **Cloud BC training**: `docs/CLOUD_RUNBOOK.md` §3-5 (BC-specific)
- **Phased curriculum**: `docs/PPO_PHASED_TRAINING.md`
- **Earlier PPO scaling notes (Session 35 — superseded)**: `docs/CLOUD_RUNBOOK.md` §11 — outdated; this cookbook is current

---

## Quick reference — commands

```bash
# Tail run progress
tail -F /workspace/logs/ppo_phase1_v3.log | grep -E "Iter [0-9]+: W/L|EVAL|Snapshot|FATAL"

# Check GPU + iter elapsed
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
ps -o etime= -p $(pgrep -f 'python train_rl' | head -1)

# Per-port battle activity
for p in 9000 9001 9002 9003 9004 9005 9006 9007; do
  log=/tmp/battle_server_$p.log; [ "$p" = "9000" ] && log=/tmp/battle_server.log
  echo "port $p: $(tail -1 $log 2>/dev/null | grep -oE '[0-9]+ active') active"
done

# Sync to R2 manually
source pokemon-ai-starter/pokemon-ai/scripts/r2_env.local.sh
aws s3 sync data/models/rl_v10/<run_name>/ \
  s3://team-builder-data/models/rl_v10/<run_name>/ \
  --endpoint-url $S3_ENDPOINT_URL --include "snapshot_*.pt" --include "*.json"

# Pull results to local at end of run
aws s3 sync s3://team-builder-data/models/rl_v10/<run_name>/ \
  data/models/rl_v10/<run_name>/ --endpoint-url $S3_ENDPOINT_URL
```
