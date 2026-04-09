# Multiprocess Collection Design (Session 32)

## Problem
Single-process asyncio event loop is the bottleneck. GPU sits idle 88% of collection time
(18s GPU vs 148s total). The event loop blocks for ~14ms during each GPU forward pass,
stalling all websocket I/O for 30+ concurrent battles. Adding more servers doesn't help
because the event loop can only process one message at a time.

## Solution
Separate GPU inference (main process) from battle I/O (worker processes).
Workers' event loops are free to process websockets at full speed.

## Architecture
```
Main Process (GPU):
  ├── InferenceServer: drain request queue → batch forward → return results
  ├── History store: {(worker_id, battle_tag): tensor} — managed centrally
  ├── PPO update: same ppo_update_v8, pauses inference during update
  └── Trajectory accumulator: collects finished episodes from all workers

Worker Process × N (CPU only, NO GPU):
  ├── asyncio event loop (never blocked by GPU)
  ├── MPRLPlayer: extract features → send obs to main → await result → play
  ├── Opponent: loads snapshot, runs OWN inference on CPU (independent)
  ├── Trajectory recording: (obs, action, logp, value, reward) per turn
  └── Sends completed trajectories to main when episode ends
```

## Data Flow Per Turn
```
Worker                          Main Process
  │                                │
  │ 1. poke-env delivers battle    │
  │ 2. extract features (CPU)      │
  │ 3. put (worker_id, btag,       │
  │    obs_dict) → request_queue   │
  │                                │ 4. drain queue (greedy + timeout)
  │                                │ 5. look up histories
  │                                │ 6. batched GPU forward
  │                                │    (spatial+temporal+heads)
  │                                │ 7. update histories with summaries
  │ 8. get (logits, value)         │ 8. put results → worker's result_queue
  │    ← from result_queue         │
  │ 9. sample action, record traj  │
  │ 10. return action to poke-env  │
  │                                │
  │ [on battle end]                │
  │ 11. send trajectory →          │
  │     trajectory_queue           │ 12. accumulate for PPO
  │ 13. send CLEAR signal →        │
  │     request_queue              │ 14. delete history entry
```

## What crosses process boundary

| Direction | Data | Size/turn | Serialization |
|-----------|------|-----------|---------------|
| Worker → Main | obs_dict (feature tensors) | ~20KB | torch.mp shared memory |
| Worker → Main | (worker_id, battle_tag, msg_type) | ~100 bytes | pickle |
| Main → Worker | (logits: 9 floats, value: 1 float) | 40 bytes | torch.mp shared memory |
| Worker → Main (end) | trajectory list | ~500KB/episode | once per episode |

**History tensors NEVER cross the boundary.** Created and managed in main process only.

## Critical Correctness Invariants

### 1. Request-Response Mapping
- Each worker has its OWN result_queue — responses CANNOT cross workers
- Each request carries (worker_id, request_id) 
- request_id is monotonically increasing per worker
- VERIFY: assert request_id matches in every response

### 2. Temporal History
- Key: (worker_id, battle_tag) — composite key is unique across all workers
- Created on first obs for a battle, updated each turn, cleared on END signal
- VERIFY: numerical test — same obs+history → same output in single vs multi process
- VERIFY: no stale histories after collection (all cleared)

### 3. Trajectory Integrity  
- Each worker's MPRLPlayer manages its own trajectories (same code as V9RLPlayer)
- Trajectory sent to main ONLY when episode is complete (done=True)
- VERIFY: every trajectory has sequential turns, one done=True, consistent episode_id

### 4. No Battle Jumbling
- Each worker connects to one Showdown server with unique account names
- Battles are isolated per-worker (different server connections)
- poke-env manages battle state per-player-instance — no shared state
- VERIFY: battle count matches expected (n_games total)

## Verification Tests (must ALL pass before use)

### Test 1: Numerical Equivalence
Run same model through single-process InferenceBatcher and multiprocess InferenceServer.
Same observations, same histories → outputs must match within 1e-4.

### Test 2: Single Worker Smoke Test
1 worker, 1 server, 20 games. Verify:
- All 20 games complete
- All trajectories have valid structure
- No NaN in any output
- History store is empty after collection (all cleared)
- collect time is reasonable

### Test 3: Multi-Worker Correctness
3 workers, 3 servers, 60 games. Verify:
- All 60 games complete
- Trajectories from all 3 workers present
- No duplicate battle_tags across workers
- Batch sizes in profiling show > 10 average (proving cross-worker batching works)
- No NaN, no errors

### Test 4: Full Scale Test
3 workers, 3 servers, 200 games. Compare to single-process baseline:
- collect time must be faster
- trajectory count and total steps similar
- PPO update runs normally on collected data
- No NaN, tainted, or errors

### Test 5: Training Continuity
Resume from snapshot_1164, run 5 iters multiprocess. Verify:
- Eval doesn't collapse (similar WR to single-process)
- Entropy, KL, pi in normal ranges
- Snapshots save correctly

## Opponent Handling
- Opponent (SelfPlayOpponent) runs in worker process on CPU
- Loads snapshot checkpoint, runs BCPolicyPlayerV8 inference on CPU
- CPU inference ~15-20ms vs 7ms GPU — slower per battle but event loop isn't blocked
- With N workers, net throughput is higher despite slower per-battle speed
- On cloud: opponent can also use centralized inference (future optimization)

## Files Modified
- rl_train_v9.py — new InferenceServer class, MPRLPlayer class, mp_collect_v9()
- NO changes to: policy_heads_v8.py, features_v8.py, rl_train_v8.py, rewards.py

## Risks
- torch.multiprocessing + poke-env websockets: need to verify child process forking works
- Opponent on CPU: slower per-battle, verify net throughput is positive
- Queue serialization overhead: verify with profiling
- Process crash: main detects via timeout, continues with remaining workers

## Fallback
Backup at backups/v9_pre_batch_temporal/. Single-process code fully functional.
Can also use --no-mp flag (TBD) to fall back to single-process collection.
