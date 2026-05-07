# Centralized Inference Server — design (Session 50 → multi-gen prep)

**Status**: design, not yet implemented. Drafted Session 50 end. Blocks
multi-gen efficient run; targets ~$200-300 savings over a 5-7 week multi-gen
training run + unlocks safer `--opponent-device` options.

**Supersedes**: `docs/MULTIPROCESS_COLLECTION.md` (Session 32 — same architecture
concept but predates the SemLock race fix, spawn-vs-forkserver decision, and
the failed `--opp-cpu` experiment from Session 50).

---

## Problem statement

Two parallel issues we've hit:

### Issue 1: mp+pipeline GPU contention deadlock

`--mp --pipeline` was meant to overlap collect (workers) with update (main).
**Workers' inference forwards stall when main is doing `optimizer.step()`** on
the same GPU — CUDA scheduler queueing causes worker forwards to never complete.
Currently silently no-op'd (cookbook §3c).

### Issue 2: opp on GPU contention pressure

8 workers × (own model + opp model) = 16 simul GPU forwards per turn at
production scale. Each forward queues behind others in CUDA scheduler, raising
effective per-forward latency 2-3x. We measured ~14% of collect time being GPU
forward wait (rest is battle simulation + asyncio loop), but at higher
concurrency this fraction grows.

We tried `--opponent-device cpu` — broke at production scale because CPU forward
of 20M-param transformer at conc=200 exceeds WS keepalive timeout (~30s default;
CPU forward is 200-500ms per call × asyncio scheduling).

### Common root cause

**Main process and workers compete for unmanaged GPU access.** No arbitration,
no stream priorities, no batching across workers. CUDA scheduler resolves each
kernel launch independently → no global view of priorities.

---

## Solution: centralized inference server (CIS)

A dedicated GPU process arbitrates all inference. Workers send observations
via small numpy/scalar IPC; CIS does GPU forward; CIS returns logits. Main
process trains on its own CUDA stream with priority over CIS.

### Architecture

```
┌─ Main process ──────────────────────────────────┐
│  - Owns model + optimizer + scheduler           │
│  - Single source of truth for model weights     │
│  - PPO update on HIGH-priority CUDA stream      │
│  - Eval, snapshot, PFSP wr update               │
│  - Sends weight-updated signal to CIS           │
└──────────────────┬──────────────────────────────┘
                   │ shared memory (single GPU)
                   │ separate CUDA streams
                   ▼
┌─ Centralized Inference Server (CIS) ─────────────┐
│  - Owns model COPY (sync'd from main on update)  │
│  - LOW-priority CUDA stream for forwards         │
│  - Receives obs from workers via mp.Pipe        │
│    (numpy arrays, no torch tensor IPC)           │
│  - Batches forwards across worker requests       │
│    (same InferenceBatcher pattern as today)      │
│  - Returns logits to workers via per-worker Pipe │
└──────────────────┬───────────────────────────────┘
                   │ Pipes (no SemLock)
                   ▼
┌─ Worker × N=8 ──────────────────────────────────┐
│  - NO GPU model on worker (different from       │
│    current mp-disk where each worker has own)   │
│  - Just CPU asyncio loop running poke-env       │
│    battles                                       │
│  - For each turn: extract obs (numpy)            │
│    → send to CIS via Pipe                        │
│    → await result (logits, numpy)                │
│    → pick action → send to battle_server         │
│  - Trajectories collected in worker memory       │
│  - At iter end: write traj to disk (same as     │
│    mp-disk current)                              │
└──────────────────────────────────────────────────┘
```

### Why this fixes both issues

**mp+pipeline GPU contention**: Main runs optimizer.step() on high-priority
stream. CIS runs forwards on low-priority stream. CUDA scheduler honors
priority — main gets GPU first, CIS fills gaps. Workers' inference pipeline
never stalls completely; it just slows down during update windows. **No
deadlock** because forwards always make progress.

**Opp on GPU**: opp inference goes through CIS too (same path as main player).
CIS has visibility across all forwards, batches them efficiently. No more
16 independent forward streams competing — single batched stream.

---

## Implementation plan

### Files

**New**: `pokemon-ai-starter/pokemon-ai/src/mp_centralized_collect.py`
- Replaces `mp_disk_collect.py` for `--mp`-style runs (or new flag, see §design decisions)
- Implements: CIS class, worker entry point, weight sync protocol

**Modified**:
- `train_rl.py`: dispatch new path; pause CIS during update windows
- `inference_batcher.py`: extract a CIS-friendly batching primitive (or keep current InferenceBatcher inside CIS)
- `rl_player.py`: V9RLPlayer's `_build_turn_batch` may need to produce numpy not torch (since IPC is numpy)
- `ppo.py`: probably unchanged (workers read weights via shm, main's update path is same)

### Phases (incremental implementation)

**Phase 1: skeleton (~1 day)**
- CIS process that owns model, listens on a single shared request_queue (numpy arrays via mp.Queue with `set_sharing_strategy('file_system')` — but small numpy arrays should be safe)
  - Actually use mp.Pipe per worker since we proved Pipes work without SemLock issues
- One worker → CIS → one worker test, no batching, sync inference
- Validate logits identity (Test 1 from cookbook §5)

**Phase 2: batching + N workers (~half day)**
- CIS runs InferenceBatcher internally, accumulates requests until min_batch
  or timeout
- N=4 workers, validate Test 4 (5-iter sustained, no NaN)
- Compare wall time vs current mp-disk

**Phase 3: weight sync + low-priority stream (~half day)**
- Main writes weights atomically (already implemented for mp-disk)
- CIS reloads on signal
- CIS forward on `torch.cuda.Stream(priority=-1)` (low priority)
- Main's optimizer on default priority stream
- Validate: pipeline overlap actually overlaps now (collect during update)

**Phase 4: pipeline overlap + production validation (~half day)**
- Re-enable `--mp --pipeline` bg overlap (the current no-op workaround in
  `train_rl.py:_start_background_collection` becomes a real bg call)
- Validate Test 6 (failure recovery), Test 5 (production wall time at games=1500)
- Production smoke at small games (games=100, 3 iters)

### Design decisions to make in implementation

**A. Flag naming**: replace `--mp` (which currently routes to mp_disk_collect.py)?
- Pro: single flag, simpler
- Con: silent semantic change for users who knew --mp's old behavior
- **Recommendation**: keep both. New flag `--mp-cis` or `--cis`, old `--mp` keeps
  routing to mp_disk_collect.py. After CIS validates, deprecate `--mp` (still
  reachable via that path). Eventually remove.

**B. Weight sync mechanism**: same atomic file write + signal as mp-disk?
- Pro: proven, works
- Con: CIS reloading 240MB ckpt every iter is overhead
- Alternative: shared memory (mp.shared_memory.SharedMemory) for weights
- **Recommendation**: start with file write (proven). Optimize to shared memory
  in phase 3 if profiling shows reload is hot.

**C. IPC type — Pipe (per-worker) vs single Queue (shared)**:
- Pipes: no SemLock, FD-based. Each worker has own ctrl + result pipes.
- Queue: SemLock-based. Failed at N≥4 in Session 50.
- **Recommendation**: Pipes (mp.Pipe(duplex=False) per worker). Same as
  current mp-disk. Numpy arrays serialize via pickle through Pipes — small
  arrays (a few KB per obs) are fine.

**D. Numpy vs torch tensor IPC**:
- Numpy: pickle through Pipe, no torch shm tracking → no mmap explosion at scale
- Torch tensor: triggers torch.multiprocessing.set_sharing_strategy issues
- **Recommendation**: numpy only across IPC. Workers do `feat_dict_torch →
  numpy_serializable_dict → Pipe.send`. CIS does `numpy_dict → torch.tensor on
  GPU → forward → logits.cpu().numpy() → Pipe.send`. This trades a small
  serialize/deserialize cost (~few ms per request) for IPC safety at scale.

**E. CUDA stream priority**:
- Main: default priority (priority=0, high)
- CIS: low priority (`torch.cuda.Stream(priority=-1)` or `priority=-2`)
- This is the KEY mechanism for arbitration. Without it, CIS and main race
  the same scheduler.

**F. Pipeline overlap re-enable**:
- After CIS validated, the existing `_start_background_collection` logic in
  `train_rl.py` flips back from no-op to actually starting bg collect.
- The bg collect uses CIS for inference (which uses low-priority stream)
- Main's optimizer.step() runs on high-priority stream
- Both overlap on GPU, properly arbitrated
- This is the actual mp+pipeline win (~25-30% wall time saving)

---

## Validation plan (per cookbook §5)

| # | Test | Acceptance |
|---|---|---|
| 1 | **Logits identity**: 100 fixed Battle states fed through CIS forward path AND uncentralized (mp-disk) forward path, same seed | `max(abs(diff_per_logit))` < 1e-3 (allow fp16 noise) |
| 2 | **Numpy round-trip**: serialize a feat_dict via numpy, deserialize, recompute on GPU. Compare to direct GPU compute | Logits within 1e-6 (numpy roundtrip is lossless) |
| 3 | **Numerical equivalence**: 1-iter `--mp-cis` vs 1-iter `--pipeline only`, same seed, games=200 | Iter line within 2σ noise |
| 4 | **Sustained 5-iter `--mp-cis`** at games=200, conc=20 | No NaN, v_loss descending, smart_avg ≥ BC baseline |
| 5 | **Pipeline+CIS overlap (Test 5 of cookbook §5b)**: 3-iter `--mp-cis --pipeline`, games=200 | ALL workers complete iter 1+ collect; no stall (the original mp+pipeline failure mode is gone) |
| 6 | **Failure recovery**: kill -9 CIS process mid-iter | Main detects, respawns CIS, run continues. Tests robustness. |
| 7 | **Wall-time benchmark**: 3-iter at production scale, games=1600, conc=200, N=8 | Steady-state iter time ≤10 min (vs current ~12-15 min mp-only). Pipeline overlap actually saves time. |

---

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| CUDA stream priority not honored on RunPod containers | Test phase 3 with explicit `torch.cuda.synchronize()` between main step and CIS forward to verify ordering. If priorities are ignored, use sync barriers as fallback (slightly slower than true pipelining). |
| CIS becomes a SPOF (single CIS process, if it dies, run dies) | Test 6 (failure recovery) validates respawn. Watchdog monitors CIS heartbeat similar to mp-disk worker watchdog. Worst case: short stall while CIS respawns + reloads model. |
| Numpy serialization overhead | Profile during phase 1. If hot, switch to mp.shared_memory.SharedMemory for the obs buffer (pre-allocated, ring-buffer style) — known pattern. |
| Pipeline benefit < expected (similar to no_grad warmup surprise) | Phase 4 wall-time test answers this empirically. If <10% saving, abandon pipeline overlap; just use CIS for the contention-arbitration win on `--mp` path alone. |
| Multi-gen complications: gen-id token affects CIS forward | CIS just sees feat_dict; gen_id is part of the dict. Same forward path as main. No extra complication. |

---

## Effort estimate

- Phase 1 (skeleton): ~1 day (8 hours)
- Phase 2 (batching + N): ~half day (4 hours)
- Phase 3 (weight sync + streams): ~half day (4 hours)
- Phase 4 (pipeline overlap + prod): ~half day (4 hours)
- Validation runs (small + production smoke): ~6 hours of cloud time, ~$10
- **Total**: ~2-3 days dedicated dev work + ~$10 cloud compute

Recommended for a focused session, not interleaved with other work. The
debugging cycles (especially CUDA stream priority semantics) need contiguous
attention.

---

## When to do this

**Before multi-gen training launches**, NOT before Phase 1 v3 finishes.

Reasoning:
1. Phase 1 v3 (currently running) is the V1 baseline and must complete
   without disruption.
2. Multi-gen run is 5-7 weeks of cloud time. CIS savings ($200-300) compound
   over that scale.
3. Per-Phase-1 savings ($30-45) doesn't justify rushing into a new
   architecture before the V1 baseline lands.

Estimated calendar slot: 2-3 day session 1-2 weeks after Phase 1 v3 wraps.
Validate, merge to master, then BC v11 multi-gen retrain + multi-gen PPO run
both use CIS path.

---

## What this enables that we can't do today

1. **mp+pipeline true overlap** (collect during update) — actual ~25-30%
   wall time saving on long runs (multi-gen scale)
2. **Higher worker count** without GPU contention pressure (currently N=8 is
   sweet spot; with CIS could go N=16+ on large pods)
3. **Mixed device opps**: opps on different priority stream than main, which
   was the original intent of `--opp-cpu` (which broke for unrelated reasons —
   WS timeout). With CIS, this is moot — all forwards go through CIS.
4. **Multi-GPU scaling** future option: CIS lives on its own GPU, main trains
   on another. Enables 2-A100 setups for very long runs.

---

## References

- `docs/MP_DISK_REDESIGN.md` — current mp-disk architecture (this design supersedes it for `--mp` path long-term)
- `docs/MULTIPROCESS_COLLECTION.md` — Session 32 original design (this was the right concept; Session 50 lessons make it implementable now)
- `docs/PPO_CLOUD_COOKBOOK.md` §3c, §3h, §8 — current state notes pointing to this redesign as the deferred fix
- ps-ppo (Nebraskinator GitHub) — Ray-based centralized inference reference. Different IPC layer (Ray vs stdlib mp) but same architectural pattern.
- Metamon's AMAGO trainer (`metamon_ref/metamon/rl/`) — uses `gym.vector.AsyncVectorEnv` with forkserver, disk-backed trajectories. Different design (no centralized inference), but informs failure modes.
