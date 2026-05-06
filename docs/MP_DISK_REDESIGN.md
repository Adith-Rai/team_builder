# --mp redesign — disk-backed worker collection

**Created:** Session 50 cont. (2026-05-06)
**Branch:** `mp-redesign`
**Status:** Design — not yet implemented. Awaiting signoff before code.

---

## Goal

Replace the current `--mp` (`mp_collect_v2.py`) implementation. Current path crashes on cloud due to torch tensor IPC saturating the kernel `vm.max_map_count` cap (~64K, RunPod containers don't allow `sysctl -w`).

New design eliminates cross-process tensor IPC entirely. Each worker owns its model copy, drives its own InferenceBatcher, writes trajectories to disk at iter boundaries. Main reads disk, runs PPO update, signals workers to reload.

**Primary success metric:** wall-clock per iter on RunPod A100 80GB at games_per_iter=1500.
- Today (`--pipeline` only): ~20 min/iter (steady state, with overlap)
- Target with `--mp` (8 workers): ~10 min/iter
- Target with `--mp --pipeline` together: ~10 min/iter (collect ≤ update; update is the floor)

**Secondary metrics:** sustained-run stability across 200 iters, no NaN, no policy drift vs `--pipeline` baseline beyond noise. Cost target: ~$50 for a 200-iter Phase 1 v3 run on RunPod A100 ($1.50/hr × ~33 hr).

---

## Constraints carried from project goal

- **Cloud-only.** Local 6 GB GPU keeps using `--pipeline` or sync paths. `--mp` errors out fast on `device == "cpu"`.
- **Transformer architecture only.** Legacy `BattleAgent` ckpts (sp_NNNN, etc.) raise `ValueError("--mp requires transformer arch")`. Backward compat is a non-goal here; legacy ckpts only matter as PFSP opponents (loaded by workers via `make_self_play_opponent` factory which already arch-dispatches), and as Elo-eval opponents (separate code path, unaffected).
- **All training guardrails preserved** (see §6).
- **No new heavy dependencies.** Stdlib `multiprocessing` + `forkserver` context, `pickle.gzip` for traj files. No Ray, no shared memory libs.
- **Format-agnostic.** Workers receive `format_config` from main; nothing hardcodes gen9ou.

---

## Architecture

```
┌─ Main ─────────────────────────────────────────────────────┐
│  - Owns model + optimizer + scheduler                      │
│  - PPO update loop, eval, snapshots, PFSP wr update        │
│  - Communicates with workers via two stdlib queues:        │
│      ctrl_queue:    main → workers (iter cmd, reload)      │
│      result_queue:  workers → main (done, error)           │
│  - Writes weights_iter{N}.pt to /tmp atomically            │
│  - Reads traj_w{id}_iter{N}.pkl.gz from /tmp at iter end   │
└──────────────────┬─────────────────────────────────────────┘
                   │ ctrl/result queues — STRINGS + DICTS only
                   │ (NEVER torch.tensor; eliminates mmap explosion)
                   ▼
┌─ Worker × N=8 ─────────────────────────────────────────────┐
│  Forkserver-spawned. Each worker:                          │
│  - Loads main model from weights_iter{N}.pt (own GPU copy) │
│  - Maintains LRU cache of opponent ckpts (cap=3)           │
│  - Owns InferenceBatcher (private to worker)               │
│  - Owns asyncio loop; runs games_per_iter/N games          │
│  - Reward shaping in V9RLPlayer (unchanged)                │
│  - Trajectories collected in memory                        │
│  - At iter end: writes traj_w{id}_iter{N}.pkl.gz to /tmp,  │
│    posts {worker_id, iter_n, traj_path, n_games_done,      │
│            wr_per_opp, n_forfeit_wins, n_forfeit_losses}   │
│    to result_queue                                         │
│  - Listens on ctrl_queue for next iter cmd or shutdown     │
└────────────────────────────────────────────────────────────┘
```

### Why per-worker GPU model (and not central inference)?

GPU contention managed by CUDA scheduler — multi-process forwards on the same A100 interleave on its 108 SMs efficiently. We measured 7% GPU util in single-process mode; per-worker GPU forwards should bring this up to 30-50% (rough estimate; exact number doesn't matter as long as wall-clock drops).

Central inference (option A in design discussion) requires sending obs to a central process. Even with numpy-IPC bypassing the mmap issue, pickle CPU overhead on 75K req/iter would eat into the speedup. Per-worker keeps inference inside a single process, no IPC needed during collect.

VRAM budget on A100 80GB:
- Main: model + optimizer + scheduler ≈ 320 MB
- 8 workers × main model ≈ 640 MB
- 8 workers × LRU 3 opp ckpts × 80 MB ≈ 1.9 GB
- 8 workers × inference activations (T=300, B=200, fp16) ≈ 16 GB
- **Total: ~19 GB on 80 GB. Sustainable.**

---

## Message formats (queues)

All values JSON-serializable (dicts, lists, str, int, float). **No torch.tensor or np.ndarray.**

### `ctrl_queue` (main → workers)

Each worker has its own ctrl_queue (one queue per worker, not shared).

```python
{"cmd": "collect_iter", "iter_n": int, "weights_path": str,
 "n_games": int, "max_concurrent": int,
 "opp_pool": [{"path": str, "wr": float, "weight": float}, ...],
 "opp_temp_range": [float, float], "fp16": bool,
 "rs_cfg": dict, "format_config": dict, "turn_cap": int,
 "battle_format": str, "procedural_teams_path": str | None,
 "server_url": str,
 "rng_seed": int}

{"cmd": "shutdown"}
```

### `result_queue` (workers → main)

Single shared queue, all workers post to it.

```python
# Success
{"status": "done", "worker_id": int, "iter_n": int,
 "traj_path": str,             # /tmp/traj_w{id}_iter{N}.pkl.gz
 "n_games_played": int,
 "wins": int, "losses": int, "ties": int,
 "n_forfeit_wins": int, "n_forfeit_losses": int,
 "wr_per_opp": {opp_path: {"w": int, "g": int}, ...},
 "elapsed_s": float}

# Error
{"status": "error", "worker_id": int, "iter_n": int,
 "exc_type": str, "exc_msg": str, "traceback": str}
```

---

## Trajectory file format

`/tmp/traj_w{id}_iter{N}.pkl.gz`

```python
{
  "trajectories": [Trajectory, ...],   # existing Trajectory dataclass
  "iter_n": int,
  "worker_id": int,
  "n_games": int,
  "wr_per_opp": dict,
  "elapsed_s": float,
}
```

Pickled with protocol 4, gzip-compressed (level 1; fast). Sizes: ~5-30 MB per worker per iter. 8 workers × 30 MB = 240 MB max per iter (~50 MB typical).

---

## Weight sync protocol

1. Main saves snapshot at iter end via existing `save_checkpoint(model, opt, sched, ...)`.
2. Main writes lightweight worker-only weights:
   ```python
   tmp = f"/tmp/weights_iter{N}.pt.tmp"
   torch.save({"model_state_dict": model.state_dict(), "model_config": cfg.to_dict()}, tmp)
   os.replace(tmp, f"/tmp/weights_iter{N}.pt")  # atomic
   ```
   (No optimizer/scheduler — workers don't train.)
3. Main signals workers via `ctrl_queue.put({"cmd": "collect_iter", "weights_path": "/tmp/weights_iter{N}.pt", ...})`.
4. Worker receives cmd → `model.load_state_dict(torch.load(weights_path)["model_state_dict"])` → starts collect.
5. After iter K, main can `os.remove(f"/tmp/weights_iter{K-2}.pt")` to bound disk usage. (Keep last-2 in case a slow worker is still using K-1.)

Old traj files cleaned the same way: `os.remove(f"/tmp/traj_*_iter{K-1}.pkl.gz")` after main's PPO update completes.

---

## Pipeline overlap (with `--mp --pipeline`)

Sequence per iter:

```
Main timeline:        [iter N update][iter N eval (every 20)][iter N+1 wait→read trajs][iter N+1 update]…
Worker timeline:    [iter N collect (~5min)]    [iter N+1 collect using iter N's weights]    [reload to iter N+1]    …
```

After main writes weights_iter{N}.pt + posts ctrl msg, workers receive next-iter cmd. Workers **do not block** on main's PPO update finishing — they keep collecting while main updates.

Steady-state iter wall time: `max(worker_collect, main_update + main_eval_if_any)`.

---

## Worker lifecycle & health management

Long-run resilience is a hard requirement (200+ iters, multi-gen runs in Phase 3 will be longer). A `WorkerManager` class in main supervises the pool:

```python
class WorkerManager:
    def __init__(self, n_workers, ctrl_factory, result_queue):
        self.workers: Dict[int, mp.Process] = {}
        self.last_heartbeat: Dict[int, float] = {}
        self.ctrl_queues: Dict[int, mp.Queue] = {}  # one per worker
        ...

    def spawn(self, worker_id):
        """Spawn a fresh worker via forkserver context."""

    def respawn(self, worker_id):
        """Kill (if alive) + spawn fresh. Used when worker dies or hangs."""

    def health_check(self, timeout_s=60):
        """Returns list of worker_ids that have missed heartbeat or have
        is_alive()=False. Called from main loop between iters AND every
        ~30s during long collect waits via a watchdog thread."""

    def kill_all(self):
        """SIGTERM with 5s grace, then SIGKILL."""
```

### Heartbeat protocol

Workers send a lightweight heartbeat to `result_queue` every 30s during collect:
```python
{"status": "heartbeat", "worker_id": int, "iter_n": int,
 "n_games_done": int, "n_games_total": int, "ts": float}
```

Main's watchdog thread reads heartbeats, updates `last_heartbeat[worker_id]`. If a worker is stale (>60s since last heartbeat) AND `is_alive()` is False → respawn. If stale AND `is_alive()` is True → it's stuck; SIGKILL + respawn.

### Failure modes

| Mode | Detection | Response |
|---|---|---|
| Worker crash mid-iter | `is_alive()` False + missing heartbeat | Watchdog respawns worker. Current iter continues with reduced sample (slice dropped); next iter the respawned worker rejoins. **No iter abort.** |
| Worker hangs (live process, no progress) | `is_alive()` True but heartbeat stale > 60s | SIGKILL + respawn (same response). |
| Worker error during collect | Worker posts `{"status": "error"}` to result_queue | Log full traceback, drop slice, **respawn worker for next iter** (the worker exits cleanly after posting error). |
| Disk full (/tmp) | `OSError` writing traj | Worker posts error; main aborts iter, surfaces error to user (this is unrecoverable without ops). |
| Weights file missing/corrupt | Worker's `torch.load` raises | Worker posts error; main re-attempts atomic write, re-signals. If second attempt fails, abort iter. |
| All N workers fail in same iter | `result_queue` has 0 successful msgs after timeout (4× expected collect time) | Abort run, log all errors, exit gracefully (preserve last good snapshot). |
| Persistent failure of same worker (3 respawns in 5 iters) | Counter in WorkerManager | Run continues with N-1 workers; logged warning. Avoids infinite respawn loops on toxic worker state. |

The watchdog thread runs at 10s polling cadence; cheap.

---

## Guardrails preservation

| Guard | Where today | Where in mp-disk |
|---|---|---|
| `_finish_looks_real` forfeit filter | `rl_player.py` `V9RLPlayer` | Worker (V9RLPlayer used unchanged) |
| Turn cap | `V9RLPlayer.__init__(turn_cap=...)` | Passed in ctrl msg |
| `is_transformer_checkpoint` arch dispatch | `make_self_play_opponent` factory | Worker calls factory (factory unchanged) |
| KL early stop | `ppo.py:ppo_update` | Main (unchanged) |
| Adaptive entropy | `train_rl.py` main loop | Main (unchanged) |
| EMA win-rate tracker | `train_rl.py` main loop | Main reads `wr_per_opp` from result msgs, applies EMA |
| Perm augmentation (`training=True` in collect) | `V9RLPlayer.choose_move` / `_build_turn_batch` | Worker (V9RLPlayer unchanged) |
| Snapshot pool & `--pool-anchors` | Main loop | Main (unchanged) |
| `cfg.format_config` (no gen9 hardcoding) | Throughout | Passed in ctrl msg, workers respect |
| PFSP `(1-wr)²` weighting | Main loop | Main computes; workers receive `weight` per opp in ctrl msg |
| `--snapshot-interval`, `--eval-interval`, `--early-stop` | Main loop | Main (unchanged) |
| Reward shaping (`rs_cfg`) | Worker's V9RLPlayer | Worker (passed in ctrl msg) |

---

## Validation plan (must pass before launching Phase 1 v3)

| # | Test | Acceptance |
|---|---|---|
| 1 | **Logits identity test**: small-scale, identical seed, 100 fixed Battle states fed through `--mp` worker forward path AND `--pipeline` main path. Compare action logits + value logits tensor-equal at fp16 precision. | Max abs diff < 1e-3 per logit (allows fp16 noise). Validates: model loads correctly in worker, features round-trip through worker, no IPC corruption. |
| 2 | **Data flow integrity test**: pickle.gzip round-trip a Trajectory dataclass on disk. Compare every field (rewards, action_masks, advantages, returns, action_logp, etc.) post-deserialize. | All fields exactly equal (lossless pickle). Validates traj file format. |
| 3 | **End-to-end numerical equivalence**: 1-iter `--mp` vs 1-iter `--pipeline` on `epoch_003.pt`, identical seed | Iter-line metrics within 2σ noise: `\|wr_diff\| < 5%`, `\|pi_loss_diff\| < 0.05`, `\|v_loss_diff\| < 0.3`, `\|kl_diff\| < 0.01` |
| 4 | **5-iter sustained run**: `--mp --pipeline` continuous | No NaN in pi/v_loss/kl/ent. wr trajectory monotonic-ish (no >10% drop iter-to-iter at lr=1e-5). |
| 5 | **Wall-time smoke (3 iters)**: at games=1500, conc=200, N=8 workers on RunPod A100 80GB | Iter 0 sequential ~12 min, iter 1+ pipelined ~10 min. If iter time > 15 min, throughput goal failed → diagnose before production launch. |
| 6 | **Failure recovery**: Kill -9 a random worker at iter 2 mid-collect. | Watchdog detects within 60s, respawns worker, current iter completes with reduced sample, run continues to iter 5 cleanly. No NaN. |

Tests 1-4 + 6 done locally on 6GB GPU at small scale (games=20, conc=20, N=2 workers) for fast iteration. Test 5 done on cloud A100. Production launch (20 warmup + 200 main iters) only after all 6 pass.

---

## Implementation file plan

**New file:** `pokemon-ai-starter/pokemon-ai/src/mp_disk_collect.py` (~250 LOC)

Public interface:
```python
def mp_disk_collect(
    model: PokeTransformer,
    optimizer,                      # only for main; ignored by workers
    snapshot_pool: List[str],
    n_games: int,
    n_workers: int = 8,
    max_concurrent_per_worker: int = 200,
    fp16: bool = True,
    rs_cfg: dict,
    temp_range: Tuple[float, float],
    teambuilder_path: Optional[str],
    server_pool: List[ServerConfiguration],
    win_rates: Optional[dict],      # PFSP wr state from main
    iter_n: int,
    rng_seed: int,
    fp_workers_alive: List[mp.Process],   # state passed across iters
) -> CollectResult:
    """Disk-backed mp collection for cloud transformer-only PPO."""
```

Module-level:
- `_worker_main(worker_id, ctrl_queue, result_queue)` — entrypoint for forkserver children
- `_collect_one_iter_in_worker(worker_id, cmd_msg)` — runs one iter's collect using V9RLPlayer
- `_make_worker_args(...)` — builds ctrl msg dict
- Worker lifecycle helpers: spawn, signal, monitor, shutdown

**Modified file:** `pokemon-ai-starter/pokemon-ai/src/train_rl.py` (~30 LOC delta)

```python
# In _collect_data():
if args.mp and not args.device.startswith("cpu"):
    from mp_disk_collect import mp_disk_collect
    return mp_disk_collect(...)
elif args.mp and args.device.startswith("cpu"):
    raise ValueError("--mp not supported on CPU; use --pipeline or sync")
```

Plus arg additions: `--mp-workers` (default 8), `--mp-cache-size` (LRU cap, default 3).

**Untouched:** `mp_collect_v2.py`, `mp_collect_v3.py` (kept in repo for reference; no longer reachable from `--mp` flag). `rl_pipeline.py` (still used by `--pipeline` only).

---

## Open issues / non-goals

- **Multi-node cloud**: out of scope. Single A100 80GB target.
- **Bigger inference batch via cross-worker batching**: not pursued; per-worker independent batching is simpler and benchmarks suggest sufficient.
- **Trajectory format optimization**: pickle.gzip is fine for now; if disk becomes a bottleneck (it won't at our scale) we can switch to numpy memmap or arrow.

---

## Decision log

- **Per-worker GPU model copy** chosen over central inference because pickle CPU overhead on 75K reqs/iter would eat the speedup; mmap risk eliminated by construction.
- **Disk traj at iter boundary** chosen over inter-process queues because (a) tensor IPC is the diagnosed root cause and (b) iter boundary is rare (every ~5 min), so disk I/O cost is negligible.
- **Forkserver context** chosen over spawn because forkserver is faster (workers stay alive across iters) and avoids the spawn-time CUDA re-init overhead. Workers reload weights via filesystem rather than restart.
- **Stdlib multiprocessing** chosen over Ray because single-node A100 doesn't need Ray's strengths (multi-node, plasma store) and Ray adds ~200 MB install + runtime overhead.
- **No backward compat with legacy `BattleAgent` arch** per user direction (Session 50 cont.). Legacy ckpts will only ever be Elo-eval opponents going forward; that path is unrelated to `--mp`.
