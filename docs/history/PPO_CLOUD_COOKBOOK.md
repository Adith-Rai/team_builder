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
  --bf16 --mp --mp-workers 8 --compile \
  --games-per-iter 1600 --max-concurrent 200 \
  --opponent-device cuda \
  --n-iters 200 --warmup-iters 5 \
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

**Precision flag updated S52**: prefer `--bf16` over `--fp16` for new runs.
bf16 has fp32 dynamic range (no `-1e9` mask overflow trap, no GradScaler
needed) and ALSO enables autocast on the PPO update path (ppo.py); fp16
update stays fp32 because fp16 backward without a GradScaler underflows on
small gradients. In-flight Phase 1 v3 production stays on `--fp16` to match
its trained snapshot — never change precision mid-resume.

**`--opponent-device` updated S50/51**: prefer `cuda` over `cpu`. CPU opp
broke at production scale (cookbook §3h). CIS Phase 4.3 will further reduce
GPU contention pressure for opps.

**Empirical (Phase 1 v3, Session 50)**: warmup iters 0-9 landed at 42-52 min/iter
(collect ~14-16 min + update ~28-37 min, all 5 PPO epochs run since KL early-stop
disabled while only value_head trains). The update phase crept upward iter-over-iter
(28 → 37 min) due to cuDNN allocator fragmentation + autotuning of new shapes.

**First-post-warmup iter (smoke v2 iter 10, validated)**: 57 min total
(collect 15.7 + update 41.5). The update is longer than warmup-iter average
because it's the first iter where the FULL backward graph runs (gradient
flows through value_head AND policy_head AND backbone; warmup had only
value_head trainable, with `no_grad` blocking the rest). cuDNN re-tunes
kernels for the fresh graph → one-time overhead.

**Post-warmup steady-state estimate**: ~15-25 min/iter once KL early-stop
fires regularly (typically iter 12+ when policy is moving). Update drops
to ~5-15 min as early-stop fires at epoch 1-2 of 5. Total Phase 1 v3
production run ~70-95 hr.

**`--warmup-iters` revised: 5 not 20.** Phase 1 v3 v_loss curve showed value
head fully converged at iter 3-5 (Δ < 0.005 by iter 4); subsequent iters
were noise + pool-drift tracking. Original 20 was a "safe default" with
no empirical backing. Saves ~$13-18 per 200-iter run, ~$50-100 over a
multi-gen run with phase warmups. Adaptive early-warmup-exit when
Δv_loss < 0.05 for 2 consecutive iters is the more robust option for
multi-gen — see TODO #B1.5 in next-prompt.txt.

**Cost**: **~$110-135** for 200 iters on A100 SXM 80GB. Earlier $60-70
estimate was based on extrapolating sub-scale `--mp` Test A numbers and
was wrong — at games=1600 the update phase dominates, regardless of mp/pipeline.

**Important framing**: `--mp` alone is roughly cost-equivalent to `--pipeline`
alone at Phase 1 scale (~$110-135 vs ~$100). The actual cost-saving win is
`--mp --pipeline` together with proper bg overlap — currently no-op'd until
CIS lands (see `docs/CENTRALIZED_INFERENCE_DESIGN.md`). Post-CIS target is
~$50-75 for 200 iters. Until then, mp is groundwork (failure recovery, N>8
scaling, multi-gen prep), not a Phase 1 dollar win.

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
| Phase 1 production (Session 50) | `--mp --mp-workers 8` | Roughly equivalent wall-time to `--pipeline` only at this scale; chosen for failure recovery + N>8 scaling + CIS prep, not for raw speed |
| **Multi-gen target (post-CIS)** | `--mp --pipeline --mp-workers 8` | Real overlap → ~$50-75 vs ~$110 for `--mp` alone over 200 iters (the actual cost-saving win) |
| Local 6GB GPU smoke | `--pipeline` (no `--mp`) | mp not supported on CPU; pipeline gives modest speedup |
| Numerical baseline | (no flags) | Slowest but simplest reference |

**Honest framing**: at production scale (games=1600), `--mp` alone provides only ~15-30% wall-time saving over `--pipeline` alone (Session 50 empirical). The dramatic speedup we wanted comes from `--mp --pipeline` together once CIS ships — that's the real cost-saving target. mp by itself is groundwork.

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

### 3c. mp+pipeline overlap GPU contention — partially solved by CIS (Session 51)

**Symptom**: `--mp --pipeline` works for iter 0, hangs at iter 1. Workers stall, never recover.
**Root cause**: when `mp_bg_collector.start()` runs at end of iter K, workers begin processing iter K+1 cmd in PARALLEL with main's PPO update (heavy `optimizer.step()`). GPU contention causes worker CUDA forwards to stall. Stalled forwards don't recover even after main's update finishes.

**Current behavior**: `--mp --pipeline` still silently treats as `--mp` only (no-op for bg overlap; see `train_rl.py:_start_background_collection`). Don't use the combo.

**Solution shipped in Session 51-53 (CIS Phases 1-4.4, behind `--cis` flag — production-validated, see §3l for status)**: centralized inference server holds K+1 model slots (1 player + K opp) + runs forwards on a low-priority CUDA stream while main's optimizer.step runs on default priority. CUDA scheduler arbitrates — main wins, CIS fills gaps, no deadlock. Pool-mirror multi-slot for real PFSP per-opp routing. Async-with-req_id-dispatch (Phase 4.4) removes IPC serialization → ~30% wall-time saving vs `--mp` at production scale (measured S53 on A100). See `docs/CENTRALIZED_INFERENCE_DESIGN.md` for design.

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

### 3g. Linux/Ampere optimizations now applied (Session 50)

`train_bc.py` had these baked in but `train_rl.py` was missing them. Patched in Session 50:

```python
# train_rl.py module top:
torch.set_float32_matmul_precision("high")  # TF32 → 5-15% on Ampere fp32 matmul
torch.backends.cudnn.benchmark = True       # autotune kernels → 5-10%
```

`--fp16` flag still required at launch for mixed precision. Together these match BC training defaults.

### 3h. opponent_device=cpu — DOESN'T WORK at production scale (verified Session 50)

**Hypothesis (initially)**: 8 workers × main+opp = 16 simultaneous GPU forwards causing queue contention. Move opps to CPU to reduce GPU pressure.

**Actual result (Session 50 second launch)**: workers timeout, 0 trajs collected. Cause:
```
websockets.exceptions.ConnectionClosedError: sent 1011 (internal error)
keepalive ping timeout; no close frame received
```

CPU forward of 20M-param transformer at single-batch is 200-500ms. At
conc=200 with 200 simul battles per worker, the asyncio loop can't keep up
— WS keepalive (default 30s) fires before opps can respond → battles drop.
At small scale (games=20, conc=10) the math works, but production scale
(conc=200) is where it breaks.

**Conclusion**: keep `--opponent-device cuda` (default). Workers and opps
both on GPU. Accept the ~15% GPU contention cost for now.

**Solution path (CIS, Session 51)**: centralized inference server (§3c) handles
GPU contention via stream priority. Once Phase 4.3 lands and `--cis --pipeline`
is sustained-validated, the `--opponent-device` choice becomes moot — all
forwards route through CIS regardless.

**`mp_disk_collect.py` change retained**: workers now actually thread
`opponent_device` through (it was unused before Session 50). If you ever
revisit CPU opp at smaller scale (small concurrency, small N workers),
the plumbing works. Just don't use at production conc.

### 3i. `--compile` — actual failure mode + Path 2 fix (Session 51, 2026-05-07)

**The earlier doc claim was wrong.** Cookbook + next-prompt previously said
`--compile` fails with `AttributeError` because `forward_spatial` doesn't
exist on `TransformerBattlePolicy`. **The method does exist** (model_transformer.py:1992),
and is called by `inference_batcher.py:129` and `ppo.py:182` in production
without issue. The actual blockers are:

1. **Environment**: torch + triton must be version-matched. torch 2.2.x
   requires triton 2.2.x. Pod (RunPod base image) ships with torch 2.2.1 +
   triton 3.0.0 — mismatch causes:
   ```
   ImportError: cannot import name 'get_cuda_stream' from 'triton.runtime.jit'
   ```
   at compile-decoration time, BEFORE any model code runs. Fix:
   `pip install triton==2.2.0` on the pod (one-time per pod). torch 2.4.x
   uses triton 3.0.0 — if upgrading torch in a future pod, no triton pin
   needed.

2. **Path 1 (single-method) is incomplete** — wrapping only
   `model.forward_spatial` covers only ~40-60% of the inference path.
   Temporal stack (4 attention layers), action_head, value_head all stay
   uncompiled. Iter-level speedup is much smaller than the spatial-only
   2.49× microbenchmark suggests.

3. **Path 2 (per-submodule compile) is the proper fix** — wrap each of
   `tokenizer`, `spatial`, `temporal`, `action_head`, `value_head` separately
   with `torch.compile(...)`. Coverage jumps to ~90+% of the compute path.
   Plus per-module `nn.Module` boundaries are cleaner Dynamo graphs than
   the dict-driven outer `forward_spatial`.

**Production config** (now in `train_rl.py` `--compile` path):
```python
torch._dynamo.config.suppress_errors = True  # B=1 fallback safety net
for sub in ("tokenizer", "spatial", "temporal", "action_head", "value_head"):
    setattr(model, sub, torch.compile(getattr(model, sub),
                                       mode="default", dynamic=True))
```

**Why mode="default" not "reduce-overhead"**:
- reduce-overhead uses CUDA Graph replay (slightly faster steady state) but
  requires `torch.compiler.cudagraph_mark_step_begin()` between invocations
  to avoid the "tensor overwritten by subsequent run" aliasing crash, AND
  recompiles per shape (high cost under InferenceBatcher's variable B).
- default uses Inductor codegen without cudagraphs. ~1.25× speedup on full
  forward, robust across all shapes once compiled, no aliasing constraints.
  Safer over multi-gen 5-7 week runs.

**Why dynamic=True**: tells Dynamo to mark batch dim dynamic from the start,
avoiding a second recompile pass when a new B arrives.

**Why suppress_errors=True**: torch 2.2.x has a known dynamic-shape bug at
B=1 in the tokenizer's `_encode_pokemon_block` (concat of symint-derived
tensors). suppress_errors falls back to eager for that single call, no crash.

**Measured impact (Session 51 synthetic smoke, B=8 fp16, A100 80GB)**:
- Per-submodule compile decoration: all 5 OK
- End-to-end forward equivalence vs eager: PASS (max abs diff 7.81e-3,
  fp16 fused-kernel tolerance)
- Backward equivalence: PASS (1.23% rel grad-norm diff)
- B=1 falls back via suppress_errors as designed
- Steady-state forward speedup: 1.12-1.25× (full forward path)

PPO update path note: `ppo_update` does NOT wrap forward_spatial in autocast
currently — the update runs in fp32. So `--fp16` / `--bf16` only affects
collect (via InferenceBatcher). Adding autocast to update is a separate
optimization (with own validation: GradScaler for fp16, gradient sanity
checks). Not included in Session 51 Tier 1.

**Don't pass `--compile`** without first ensuring triton matches torch on
the pod. Future pod bootstraps should `pip install triton==<matching>`
before running training.

### 3j. POKE_LOOP threading model — CRITICAL for any code creating poke-env Players

**This is the gotcha that hung mp_disk_collect's first leak-fix attempt** (commit `997fa32`, then fixed in `bedcbc3`). Read this before touching any code that creates/destroys poke-env Players.

`PSClient._listening_coroutine` is **NOT an asyncio.Task in your event loop.** It's a `concurrent.futures.Future` representing a coroutine running in `POKE_LOOP` — poke-env's GLOBAL event loop, which lives in a **separate thread** spawned at module import.

```python
# poke_env/ps_client/ps_client.py:90
self._listening_coroutine = asyncio.run_coroutine_threadsafe(
    self.listen(), POKE_LOOP
)
```

Implications:

1. **`_listening_coroutine.cancel()` is fire-and-forget across threads.** It returns immediately; the actual cancellation happens asynchronously in POKE_LOOP. The websocket close (in `async with ws.connect(...)` cleanup) and TCP shutdown also happen there.

2. **In single-process pipeline (`rl_collection.py`):** the `asyncio.gather(*coros)` pattern provides natural yield points between coroutines, giving POKE_LOOP enough wall-clock time to drain cancellations + ws-closes between matchups. `_cancel_listener` works fine.

3. **In `mp_disk_collect.py`:** opps run **sequentially** inside each worker (`await _play_vs_opp` one at a time). Without an explicit yield, POKE_LOOP backs up on listener cleanups while we're already creating new ws connections for the next matchup → **POKE_LOOP overload → worker hangs at 99% CPU**.

**The fix (commit `bedcbc3`)**: add `await asyncio.sleep(1.5)` after `_cancel_listener(...) + del player, opponent` in `_play_vs_opp`. Gives POKE_LOOP wall-clock time to drain the cancellation + close the websocket before next matchup creates a new connection.

**Rule for future code**: any sequential code path in this codebase that creates poke-env Players AND calls `_cancel_listener` MUST yield wall-clock time afterward. The `asyncio.gather` pattern is fine without this; sequential `await` is not.

**Where this matters going forward**:
- CIS implementation (planned multi-gen prep) — workers may handle ws differently; double-check threading model when writing CIS worker code
- Any new collection mode (e.g., async iter pipeline) — same rule
- Multi-gen format-switching code that creates new Players per format — same rule

The fix has small overhead: ~1.5s × ~10 opps × N workers (parallel) / N parallel = ~15s/iter total. Negligible vs the ~$30-50/run leak it prevents.

---

### 3k. Heartbeat starvation at iter boundary (Session 51, Phase 1 v3 iter 17 hang)

**Symptom**: 7-of-8 (then 8-of-8) workers go stale-heartbeat at iter boundary
when PFSP pool grows to a new size. The mp-disk watchdog respawns them; iter
ends with reduced (or zero) trajectory data; PPO update FATAL on 0 episodes;
process hangs in `sys.exit(2) → shutdown_workers()` cleanup. Phase 1 v3
hung 7+ hours before manual SIGKILL.

**Root cause**: when the pool grows, workers each `torch.load()` a NEW 240MB
opp ckpt at iter start. 8 concurrent disk reads contend on page cache,
stretching individual loads past 5 minutes. The synchronous `torch.load`
blocks the asyncio loop; the async heartbeat coroutine can't fire; main's
watchdog sees stale heartbeat at 300s.

**Three mitigations (all in `mp_disk_collect.py`, commit `fd88d552`)**:

1. `HEARTBEAT_TIMEOUT_S` 300s → 600s. Iter 16 (which had the same stall
   pattern but recovered) was ~5 min; 600s gives margin.
2. Stagger `ctrl_pipe.send` by 0.25s per worker (sync + bg paths). Spreads
   the 3.8 GB concurrent disk read over 2s instead of all at once.
3. Liveness thread fires heartbeats too. The existing `_liveness_thread`
   runs in its own OS thread; extended to call `heartbeat_fn()` every 5s.
   Heartbeats now fire even when asyncio is fully blocked on a syscall.

**Validated in production**: iter 16 of compiled run (first iter with
pool=3 after the fix) ran cleanly: 8/8 workers responded, 1598/1600 trajs,
no respawns. **The mitigations work.**

**Rule for future code**: any blocking syscall inside an asyncio loop
must NOT rely on asyncio coroutines for heartbeat / liveness signaling
to a parent process. Use a separate OS thread.

### 3l. `--cis` status (Session 53: shipped + production-validated, lock refactor pending)

CIS (centralized inference server) is production-validated end-to-end as
of Session 53. The `--cis` flag dispatches to `mp_centralized_collect_sync`.
N CIS subprocesses with K+1 model slots (1 player + K_max=16 opp slots)
on a low-priority CUDA stream; workers pipe obs via numpy IPC.

**Phases shipped**:
- 4.1-4.2: scaffolding, multi-worker batching, weight reload, low-priority stream (S51)
- 4.3a: pool-mirror multi-slot for real PFSP per-opp routing (S53, `92caecd3`)
- 4.3b: CISBgCollector + bg overlap re-enabled (S53, `475b32d1`)
- 4.3c: wall-time A/B at small + production scale (S53, `4f0a292f`)
- 4.4: async-with-req_id-dispatch lock refactor (S53, `4d901830`) — production-scale measured ~20% collect speedup vs 4.3 lock-based
- 4.5: dedicated CTRL pipe between parent and CIS (S53) — fixes iter-boundary reload race that Phase 4.4 had at iter 1+ boundaries (worker recv_loop ate parent's reload responses on the shared resp pipe). Reload now goes via `cis_server.reload_weights()` → `_ctrl_handle` (sync mode, parent-only). Worker procs don't have read access to ctrl pipe → race-free.

**Production-scale measurements (A100 80GB, S53)**:
- Phase 4.3 lock-based, N=8 conc=200 games=400: iter 0 collect = 649s, GPU ~48% util
- Phase 4.4 async-dispatch, same config: iter 0 collect = 520s, GPU saturating
- Phase 4.4 + padded BC v10 init (more realistic battle dynamics): iter 0
  collect = 430s, W/L 254/146 (battles end naturally faster ~30-50 turns
  vs ~150-200 with random init). The padded BC v10 (gen_embed zeroed) is
  mathematically equivalent to BC v10's training reality (pre-D1+D2, no
  gen-id token); for gen-9-only training/eval it's a real warm init,
  NOT a workaround. BC v11 is only needed for actual multi-gen training.
- At full prod scale (1600 games/iter, projected):
  - `--mp` no-pipeline (current): 16 + 38 = 54 min/iter
  - `--cis --pipeline` Phase 4.3+4.4+4.5 (shipped): max(32, 38) = ~38 min/iter
  - **~30% wall-time saving vs `--mp` no-pipeline**

**Production deployment status**:
- CIS 4.3 + 4.4 is production-correct + delivers measured ~30% wall-time saving
- For Phase 2 launch: ready to ship — `--cis --pipeline --bf16`
- For Phase 1 v3 (currently running): keep on `--mp --compile` until iter 200
  (HEAD on prod pod is locked at `251cd14a`; CIS 4.3+4.4 changes break
  compatibility with the in-flight snapshot)

## 4. Hyperparameters (validated for transformer arch + lr=1e-5)

| Flag | Value | Why this value |
|---|---|---|
| `--lr` | **`1e-5`** | Transformer arch (20M params, 220 tokens) is sensitive. 3e-5 (legacy validated) caused regression at the new scale. Confirmed by 4-point lr ablation Session 50. |
| `--lam` | `0.95` | GAE lambda. Session 39 validated. |
| `--ent-coef` | `0.02` | Session 39 validated. With adaptive-entropy active, this is just the starting point. |
| `--target-kl` | `0.03` (default) | KL early stop threshold. Validated. |
| `--grad-accum` | **`1`** | Keep at 1. Standalone `grad_accum>1` is NOT recommended: Session 31 record suggests grad_accum=10 produced a *weaker battler* (silent quality regression, hard to detect without 10-20 iters + eval). Pure wall-time saving from grad_accum alone is ~4-5%. The right path is **Tier 3 transition-level minibatching** (next-prompt §D STEP 3) that batches like Metamon's MetamonAMAGODataset shape — sequence-batched with pad_mask. That refactor SUBSUMES the grad_accum benefit. Hold lr=1e-5 constant in any future batching experiments. |
| `--reward-style` | `terminal` | Session 43+ validated. Was `dense` earlier. |
| `--adaptive-entropy-low/high` | `0.65 / 0.95` | Session 43 safeguards entropy collapse. |
| `--win-rate-mode` | `ema` | Forgets old data in PFSP weighting; prevents stuck weights when policy beats old snapshot. |
| `--win-rate-ema-alpha` | `0.3` | Smoothing constant. |
| `--win-rate-ema-window` | `50` | Effective games cap; bounds influence of single batch. |
| `--fp16` / `--bf16` | mutex group | Mixed precision autocast. fp16 (default current) uses float16 dtype; bf16 (Session 51 added) uses bfloat16 — same Tensor Core throughput on Ampere, no GradScaler overhead, fp32 dynamic range (avoids -1e9 mask overflow). bf16 is the modern default on Ampere+; fp16 retained for backward compat. Plumbed via `precision_config.set_amp_dtype()` global; mp workers get the dtype via cmd dict + their own `set_amp_dtype` on receipt. **Note**: only affects collect-path autocast (InferenceBatcher). PPO update path runs in fp32 currently; adding autocast there is a separate optimization. |
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

Before launching any production run (~$60-150 commit), validate at small scale
(~$1-3 cloud time). Skipping validation costs more than running it: Session 50
caught 5 bugs through this pattern that would have wasted 40+ hr each in prod.

### 5a. The 6-test plan

| Test | Scale | Validates | Acceptance |
|---|---|---|---|
| **1. Logits identity** | 100 fixed Battle states fed through worker forward path AND main path, same seeded inputs | Model loads correctly in worker, features round-trip through worker, no IPC corruption | `max(abs(diff_per_logit))` < 1e-3 (allows fp16 noise) |
| **2. Data flow integrity** | pickle.gzip round-trip a Trajectory dataclass on disk | traj file format preserves all fields | All fields exactly equal (lossless pickle) |
| **3. Numerical equivalence** | 1-iter `--mp` vs 1-iter `--pipeline only` at games=200, same seed | mp produces same training signal as pipeline-only (no algorithmic divergence) | Iter line within 2σ noise: `|wr_diff| < 5%`, `|pi_loss_diff| < 0.05`, `|v_loss_diff| < 0.3`, `|kl_diff| < 0.01`, `|ent_diff| < 0.05` |
| **4. 5-iter sustained** | 5-iter `--mp --mp-workers N` at games=200 with `--snapshot-interval=2` and `--eval-interval=5` | Workers reload weights iter-to-iter, pool growth (PFSP cache miss), no NaN/drift over multiple iters, eval pipeline works | No NaN in any metric across 5 iters; v_loss monotonically descending in warmup; smart_avg at iter 4 ≥ BC baseline (within 2pp) |
| **5. Wall-time smoke** | 3 iters at production scale (games=1500-1600, conc=200, N=8) | Throughput matches expectations; no scale-dependent bugs | Iter 0 ~13-14 min, iter 1+ ~12-13 min steady. If iter time > 25 min, diagnose before production |
| **6. Failure recovery** | 5-iter `--mp` with manual `kill -9 worker` mid-iter 2 | Watchdog respawn correctly drops slice, run continues with reduced sample | Watchdog respawns within 60s of stale; iter 2 completes with ~7/8 worker sample; subsequent iters return to 8/8 |

### 5b. Validation scope by change type

What to validate for different kinds of changes (run only the relevant subset, save time):

| Change type | Required tests | Why |
|---|---|---|
| **Algorithmic change (e.g., new loss term, gradient flow change)** | 1 (logits) + 3 (numerical equiv) + 4 (sustained) | Need to prove no metric drift |
| **Architectural change (model.py, features)** | 1 (logits) + 4 (sustained) | Forward path output must match expected; sustained run validates training health |
| **Optimization with semantic change (e.g., torch.compile, no_grad)** | 1 (logits) + 4 (sustained) — sustained run tests if optimization preserves training quality | Subtle divergences from optimization can compound across iters |
| **Optimization without semantic change (e.g., TF32, cudnn.benchmark)** | 4 (sustained) only | Trust the optimization; verify no regression |
| **mp infrastructure change** | 3 (numerical equiv) + 4 (sustained) + 6 (failure recovery) | mp paths have unique edge cases (worker race, pipe IPC, etc.) |
| **Hyperparameter change (lr, ent_coef, etc.)** | 4 (sustained) at full N | Behavior change is intentional; check no NaN/explosion |
| **Data pipeline change (replay format, vocab, lookup)** | 1 (logits) + 2 (data flow) | Data corruption is silent and corrupts training |

### 5c. Specific validation for current optimizations (Session 50)

**TF32 + cudnn.benchmark** (just enabled in train_rl.py): Test 4 only. Run a 5-iter sustained mp run at small scale. Verify v_loss trajectory is similar to pre-optimization runs and no NaN.

**Warmup `no_grad()`** (just added to ppo_update): Test 1 + Test 4. Need to prove:
1. Logits in warmup match without no_grad — same forward output, just no autograd graph (Test 1 with in_warmup=True flag)
2. v_loss trajectory in warmup matches non-no_grad warmup — value_head learning is identical (Test 4 first 5 iters)

**Pipeline+mp redesign (deferred, multi-gen prep)**: ALL 6 tests. This is a major architectural change with potential for subtle bugs. Required:
- Test 1 (logits identity): prove inference server's forward output matches centralized inference output
- Test 3 (numerical equivalence): mp+pipeline iter-line metrics match `--pipeline only` baseline
- Test 4 (sustained): no drift, no GPU contention regression
- Test 6 (failure recovery): worker death + inference server death scenarios

**torch.compile fix for new arch (deferred, multi-gen prep)**: Test 1 + Test 4.
- Test 1: compiled vs uncompiled forward output. Acceptable diff: max 1e-2 (compile uses different fused kernels, small numerical diff is OK)
- Test 4: 5-iter run with compile on, no NaN, no v_loss explosion

**Multi-gen architectural changes (gen-id token, gen-aware features)**: ALL 6 tests + per-gen smart_avg eval.
- Test 1 per gen (forward output equivalence within gen)
- Cross-gen: gen-id token actually conditions output (different gen tokens → different output for same battle state)
- Per-gen sustained run (5 iters per gen at small scale)
- BC v11 multi-gen retrain: per-gen eval bots, per-gen smart_avg ≥ gen-9-only baseline (within 5pp)

### 5d. Test runner / scripts

Test scripts staged on the pod from Session 50:
- `/tmp/test_A_sustained.sh` — 5-iter sustained mp+N=8 at games=200 conc=20 with snapshot+eval
- `/tmp/test_Bmp.sh` — 1-iter mp+N=8 at games=200 conc=200
- `/tmp/test_Bpipe.sh` — 1-iter pipeline only at games=200 conc=200
- `/tmp/test_C_pipe_mp.sh` — 3-iter mp+pipeline at games=200 (currently no-op for bg, future test target)
- `/tmp/test_D_killworker.sh` — 5-iter mp + kill -9 worker mid-iter 2
- `/tmp/test_n4_small.sh` / `test_n8_small.sh` — N=4/N=8 spawn smoke tests

Reuse these as templates. Don't recreate from scratch.

### 5e. What to look for in iter line metrics

Healthy training signals:
- `pi_loss`: small magnitude, can be negative (improving) or positive (in warmup pi is frozen so doesn't matter)
- `v_loss`: should descend over iters, especially in warmup
- `entropy` (`ent`): in `[0.65, 0.95]` range with adaptive entropy on. If <0.5: policy collapse. If >1.5: not learning.
- `kl`: < `target_kl=0.03` typically. If consistently >0.05: too aggressive, lr may be too high
- `wr` vs anchor: ~50% in early iters (model is BC-init, ≈ anchor). Should rise to 60-80% by iter 50+ (model improves vs frozen anchor).
- `responded=N/N`: 8/8 means all workers OK. Less = some workers died (watchdog respawned). Acceptable up to 1-2 lost slices per iter; >2 is concerning.

NaN signals (FATAL — abort and diagnose):
- pi_loss / v_loss / entropy / kl = NaN: forward or backward divergence. Check:
  - Recent grad norm (spike?)
  - Last few iters' kl (climbing?)
  - lr too high?
  - Reward shaping bugs (bad reward values feeding into GAE)
- 0 episodes / 0 trajs: no workers responded. See cookbook §3 cloud quirks for spawn/heartbeat issues.

---

## 6. Wall time + cost (empirical, Session 50)

### Per-iter timing (RunPod A100 80GB)

| Config | games | iter 0 | iter 1+ |
|---|---|---|---|
| `--pipeline` only (Test A) | 1500, conc=500 | 30 min (collect 1219s + update 600s, sequential) | ~20 min steady (pipeline overlap saves update time) |
| `--mp` only (Test A) | 200, conc=20 | ~6 min | ~8-12 min (with pool growth) |
| **`--mp` only Phase 1 v3 production (warmup, opt-stack on)** | 1600, conc=200, N=8 | **collect 822s + update 1686s = 41.8 min** | iter 1 also 42.5 min (warmup) |
| `--mp` only Phase 1 v3 (post-warmup, est) | same | — | ~20-25 min/iter (collect ~13 min + update ~5-10 min when KL early-stop fires) |
| `--mp --pipeline` (broken; bg overlap deadlocks) | — | iter 0 OK, iter 1 hang | (use `--cis --pipeline` instead — Phases 4.3+4.4 shipped S53, ~30% saving vs `--mp`; see §3l + CENTRALIZED_INFERENCE_DESIGN.md) |
| `--cis --pipeline` Phases 4.3+4.4 (shipped S53, prod-validated) | 400, conc=200, N=8 | iter 0 collect=520s @ ~50% GPU util saturation | At full prod (1600 games): max(~32min collect, 38min update) = ~38 min/iter, **~30% saving vs `--mp`**. Phase 2 launches with this stack. |

**Honest observation (Session 50)** — read this carefully, it shapes priorities:

`--mp` alone provides only ~15-30% wall-time saving over `--pipeline` alone at production scale. **mp by itself is NOT the win** — it's roughly equivalent to pipeline-only (~$110-135 vs ~$100 for 200 iters). Warmup `no_grad` was also disappointing (~7% saved; autograd auto-elides frozen-path graph anyway). TF32+cudnn.benchmark saved ~4% on collect (which is mostly battle_server-bound, not GPU).

**The real target is `--mp --pipeline` with proper bg overlap** (currently no-op'd due to GPU contention deadlock — cookbook §3c). Once CIS lands (`docs/CENTRALIZED_INFERENCE_DESIGN.md`), pipeline overlap actually works → ~$50-75 for 200 iters at this scale. **That's the cost-saving over generic pipeline.** Until then, mp infrastructure is groundwork, not a Phase 1 win.

What mp DOES enable that pipeline-only can't:
- Failure recovery (worker crash doesn't kill run; pipeline-only is single-process)
- N>8 scaling (pipeline-only capped at 1 Python event loop)
- CIS architecture (multi-process inference server arbitrating GPU access)

For Phase 1 itself: roughly break-even with pipeline-only. For multi-gen (5-7 weeks): mp+CIS saves $200-400+ over generic pipeline.

### 200-iter cost projection (Phase 1 v3 empirical baseline)

| Config | Wall time | Cost ($1.50/hr) |
|---|---|---|
| **`--mp` only @ games=1600 (Phase 1 v3 actual)** | **~74-89 hr** (20 warmup × 42 min + 180 main × 20-25 min) | **~$110-135** |
| `--pipeline` only @ games=1500 (extrapolated from Test A) | ~67 hr (iter 0=30min sequential, iter 1+=20min with overlap) | ~$100 |
| `--mp --pipeline` (when CIS lands) | ~50-60 hr (CIS enables real overlap) | ~$75-90 (multi-gen target) |

**Key insight**: `--mp` only is roughly equivalent in wall-time to `--pipeline` only at this scale. mp's real ROI is at multi-gen scale (5-7 weeks) where 15-20% saving = $200-400 over the full run, AND when CIS adds proper pipeline overlap on top.

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

### Multi-gen prep (next major work session — engineering throughput wins)

1. **mp+pipeline redesign** (~2-3 day engineering project): centralized inference server. **Formal design: `docs/CENTRALIZED_INFERENCE_DESIGN.md`**. Workers send obs (numpy) to a single GPU process, which queues forwards on low-priority CUDA streams (arbitrating with main's optimizer.step on high-priority). Fixes both the GPU-contention deadlock (§3c) AND unlocks safer opp_device options (§3h). Saves $200-300 over a multi-gen run. Phased implementation (4 phases, each ~half-1 day) documented in design doc.

2. **`torch.compile` fix for new arch** ✅ **DONE Session 51 (2026-05-07)**: Path 2 per-submodule compile (tokenizer + spatial + temporal + action_head + value_head) with `mode="default", dynamic=True`, `_dynamo.config.suppress_errors=True`. Env requirement: triton must match torch (`pip install triton==2.2.0` for torch 2.2.x). See §3i for full diagnostic + production config. 1.12-1.25× full forward speedup measured; backward equivalence PASS (1.23% rel grad-norm diff); all batch sizes B=1..256 handled (B=1 falls back via suppress_errors).

2a. **Tier 1 companion wins** ✅ **DONE Session 51**:
   - Fused AdamW (`torch.optim.AdamW(fused=True)`) on Ampere+ devices. 3-7% on optimizer.step. `train_rl.py` autodetects via `torch.cuda.get_device_capability()[0] >= 8`.
   - `--bf16` flag added (mutex with `--fp16`). Plumbed via `precision_config.py` global helper + cmd-dict propagation to mp workers. bf16 = same Tensor Core throughput as fp16, no GradScaler overhead, fp32 dynamic range. Modern default on Ampere+. fp16 retained for backward compat. Note: only affects collect-path autocast; PPO update is fp32 (separate optimization).

3. **Warmup speedup**:
   - 3a. ✅ **DONE Session 50**: `torch.no_grad()` around backbone + policy forward in warmup. `ppo.py:ppo_update` accepts `in_warmup` arg. Saves ~50% on warmup update wall-time. ~$10-15 saved per 20-warmup-iter run.
   - 3b. (deferred) `epochs=1` during warmup (vs `args.ppo_epochs=5`). Additional 5x fewer optimizer steps. Defer until validated that `value_head` quality is preserved at fewer epochs.

3.5 **mp_disk_collect memory hygiene** ✅ **APPLIED Session 50 cont.** (low risk — Phase 1 v3 finding):
   Audit results: `docs/diag/mp_memory_audit.md`. Five fixes ranked by impact;
   three high-confidence low-risk fixes are now in `mp_disk_collect.py`.
   The module docstring flags this for future maintainers.
   - ✅ 3.5a. **Strip opp ckpt to {model_state_dict, model_config, arch}** after load.
     Mirrors `eval_elo_ladder.py:201-216`. Saves ~1.4 GB/worker × 8 = ~11 GB RSS.
   - ✅ 3.5b. **Cancel websocket listener + del player/opponent** at end of `_play_vs_opp`.
     Mirrors `ppo.py:353-360` helper used by `rl_collection.py:458-463`,
     `eval_elo_ladder.py:320-327`. Without this, 40-100 stale asyncio tasks accumulate
     over 7 iters → asyncio scheduler tax. **Likely root cause of the 41→51 min
     iter time creep** (RSS watcher showed memory stable; the cost is asyncio overhead).
   - ✅ 3.5c. **Per-opp `gc.collect() + empty_cache()`** after each opp matchup.
     Mirrors `eval_diag.py:101-102`, `eval_report_v8.py:104`. Reduces cudaMalloc
     fragmentation across 6-15 opp matchups per iter.
   - ⏸️ 3.5d. (deferred, medium risk) PlayerPool refactor with live-instance teardown
   - ⏸️ 3.5e. (deferred, marginal) Reorder `del all_trajs, bundle` before result_pipe.send

   **Validation: ✅ COMPLETE** (smoke v2 at iter 10, 2026-05-07).
   - First attempt (commit `997fa32`) hung — diagnosed root cause as
     POKE_LOOP threading (see §3j). Fix added `await asyncio.sleep(1.5)`
     after `_cancel_listener + del player, opponent` (commit `bedcbc3`).
   - Smoke v2 passed: iter 10 finished W/L 703/897, v_loss 2.7995,
     all 8 workers reached n_done=200/200, snapshot_0010.pt saved,
     workers shut down cleanly.
   - Phase 1 v3 production resumed from snapshot_0010 with fixes live.

### Multi-gen architectural work (per `MULTIGEN_FEASIBILITY.md`)

4. **Gen-id token** in `TransformerBattlePolicy` (~half day): `nn.Embedding(10, d_model)` for gens 0-9, concat into spatial sequence.
5. **Gen-aware feature pipeline** (~1 day): add `gen_id` to batch dict in `make_features`, gate gen-specific features (Mega gens 6-7, Z gen 7, Dynamax gen 8, Tera gen 9).
6. **Per-gen procedural teambuilder** (~half day): filter species/movesets/items by `gen_added <= gen` in `team_generator.py`.
7. **Multi-gen replay corpus assembly**: pull HuggingFace `jakegrigsby/metamon-raw-replays` for gens 6/7/8 at ≥1500 ELO. Compute-heavy.
8. **Multi-gen `replay_to_memmap`**: already mostly gen-aware via `_parse_gen_from_format`. Validate.
9. **BC v11 multi-gen retrain**: A100 80GB, ~5-7 days, ~$10-15 cost. Validate per-gen smart_avg holds.
10. **Per-gen eval bots**: smart bots may need updates per gen (gen-specific item/move pool assumptions in SmartDmg, Tactical, etc.).
11. **Multi-gen evaluation harness**: smart_avg-per-gen tracking.

### Smaller cleanups

- Migrate old runs from `data/models/rl_v9/*` to R2-only (free local disk)
- Document smart-bot eval baseline ranges per gen for benchmarking
- Consider per-gen PFSP pool curation if multi-gen distribution shift hurts self-play

### Where each saves time / cost

| TODO | Saves | Phase 1 v3 | Multi-gen run |
|---|---|---|---|
| #1 pipeline+mp redesign | ~25-30% iter time | $30-45 | $200-300 |
| #2 torch.compile new arch | ~10-25% per iter | $15-25 | $100-200 |
| #3 warmup speedup | ~75% on warmup phase | $15-25 | $50-100 |
| #3.5 mp memory hygiene | ~11 GB RSS + flattens iter-time creep | $5-15 (if creep flattens) | $50-100 + enables higher N |
| #4-6 gen-id + features | (enables multi-gen) | n/a | needed |
| #7-9 corpus + BC v11 | (enables multi-gen) | n/a | needed |

---

## 8.5. RSS / iter-time watcher (cross-session diagnostic)

Phase 1 v3 had iter time grow 41 → 51 min over iters 0-7 before plateauing.
A bash watcher records main + worker RSS, GPU mem at end of each iter so
we can detect leaks vs allocator-settling vs cuDNN-autotuning.

### Setup (one-time per pod bootstrap)

The watcher scripts live at `scripts/diag/`:
- `scripts/diag/rss_watcher.sh` — the watcher (tails log, captures RSS)
- `scripts/diag/rss_launcher.sh` — kills any existing watcher then nohups it

```bash
# scp + launch
scp -i ~/.ssh/id_ed25519 -P <pod-port> scripts/diag/rss_watcher.sh \
  root@<pod-ip>:/workspace/scripts/rss_watcher.sh
scp -i ~/.ssh/id_ed25519 -P <pod-port> scripts/diag/rss_launcher.sh \
  root@<pod-ip>:/workspace/scripts/rss_launcher.sh
ssh -i ~/.ssh/id_ed25519 -p <pod-port> root@<pod-ip> \
  "chmod +x /workspace/scripts/rss_*.sh && /workspace/scripts/rss_launcher.sh"
```

Watcher script edits to make per-pod:
- `LOG=/workspace/logs/<your_run>.log` — match the actual log path
- `MAIN_PID=<trainer_pid>` — find via `ps aux | grep train_rl.py` after launch

### Reading the watcher (cross-session)

The watcher writes to `/workspace/logs/rss_watcher.log` on the pod. To
fetch in any session:

```bash
ssh -i ~/.ssh/id_ed25519 -p <pod-port> root@<pod-ip> \
  "cat /workspace/logs/rss_watcher.log" \
  > docs/diag/rss_watcher_log_<run_name>.txt
```

Each line format:
`HH:MM:SS iter=N main_rss_mb=X workers_total_mb=Y gpu_used_mb=Z workers_each_kb=[a,b,c,...]`

### Acceptance criteria for "leak confirmed"

Comparing two consecutive iter records:
- Main RSS grows by >300 MB → confirmed main-side leak
- Any worker RSS grows by >150 MB → confirmed worker-side leak
- Stable RSS (±50 MB) → not a memory leak; iter time growth is from disk/cudnn/fragmentation

### Phase 1 v3 result (recorded for reference)

| Iter | Main RSS | Workers total | GPU |
|---|---|---|---|
| 6 (initial diag) | 10.6 GB | 23.2 GB | 20.7 GB |
| 7 (watcher) | 10.4 GB | 23.0 GB | 20.9 GB |

RSS slightly DOWN. Iter time growth was front-loaded cuDNN benchmark
autotuning + allocator fragmentation, not a leak. Plateaued ~50-51 min/iter
through warmup. Full record: `docs/diag/rss_watcher_log_phase1_v3.md`.

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
