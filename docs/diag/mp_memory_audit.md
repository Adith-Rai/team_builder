# mp_disk_collect.py — memory hygiene audit (Session 50 cont.)

Comparison of memory-cleanup patterns in local single-process code paths
vs `mp_disk_collect.py` worker path. Audit triggered by Phase 1 v3 iter
time growing 41 → 51 min over iters 0-7 of warmup.

**Total achievable savings**: ~1.7-2.0 GB RSS per worker (~13-16 GB across
8 workers) AND flattens the iter-time creep that watcher data couldn't
explain via RSS alone (the creep is asyncio listener accumulation, not
memory pressure).

**Five fixes, sorted by impact:**

---

## 1. Strip opp ckpt to {model_state_dict, model_config, arch} after load

**Local pattern**: `eval_elo_ladder.py:201-216` (`load_ckpt_cached`):
```python
full = torch.load(path, weights_only=True, map_location="cpu")
ckpt = {
    "model_state_dict": full["model_state_dict"],
    "model_config": full["model_config"],
    "arch": full.get("arch", "legacy"),
}
del full
gc.collect()
return ckpt
```

**mp_disk_collect bug** (`mp_disk_collect.py:535-536`):
```python
cached_ckpt = torch.load(opp_path, map_location=opponent_device, weights_only=False)
opp_cache[opp_path] = {"ckpt": cached_ckpt, "last_used": time.time()}
```

**Two issues**:
1. `weights_only=False` loads the full ckpt incl. AdamW optimizer state (~480 MB per ckpt)
2. Even with `weights_only=True`, the snapshot pool list, scheduler state, metadata are still attached. The strip pattern explicitly drops them.

**Saving**: ~1.4 GB RSS per worker (3 cached × ~480 MB). ~11 GB total across 8 workers.

**Risk**: low. `make_self_play_opponent` only reads `model_state_dict` + `model_config` + `arch`.

---

## 2. Cancel websocket listener task + explicit `del player; del opponent` per matchup

**Local pattern**: `ppo.py:353-360` (`_cancel_listener` helper) called at:
- `rl_collection.py:458-463`
- `eval_elo_ladder.py:320-327`
- `rl_pipeline.py:462-463`

**mp_disk_collect bug** (`mp_disk_collect.py:587-591`): only calls `reset_battles()`. The poke-env websocket listener coroutine (`ps_client._listening_coroutine`) is NOT cancelled. The asyncio loop holds a strong ref via the running task → `gc.collect()` cannot reclaim the player.

Each iter creates ~6-15 player+opponent pairs (one per opp-pool entry). Over 7 iters that's 40-100 lingering listeners + their poke-env battle history dicts, each pinning ~5-10 MB.

**Saving**: ~200-500 MB RSS per worker after 7 iters. **More importantly**: the asyncio scheduler now has to walk 100+ stale tasks per turn → measurable iter-time tax. This likely explains the 41→51 min creep that wasn't visible in RSS (objects already counted in 2.9 GB baseline; the slowdown is asyncio overhead).

**Risk**: low. Pattern is identical to four other call sites.

---

## 3. Live-instance opp model pool with teardown on LRU eviction

**Local pattern**: `eval_elo_ladder.py:280-313` (`PlayerPool`) keeps live `BattleAgent` instances, not just ckpt dicts. On LRU eviction, calls `_teardown_player` (`eval_elo_ladder.py:315-327`) which: cancels listener, releases GPU model params, `gc.collect() + empty_cache()`.

**mp_disk_collect bug** (`mp_disk_collect.py:529-536`): `opp_cache` only stores ckpt dicts. The actual `SelfPlayOpponentTransformer` instance (built fresh from cached ckpt at line 538) goes out of scope only at end of `_play_vs_opp`. With `opponent_device="cuda"` this fragments cudaMalloc as each game-batch creates+destroys the instance.

**Saving**: ~200-400 MB VRAM per worker (reduced fragmentation), latent flatten of iter-time creep on cuda opp paths.

**Risk**: medium. Replicating PlayerPool inside the worker is a non-trivial refactor. Defer until #1+2 confirmed insufficient.

---

## 4. `gc.collect() + torch.cuda.empty_cache()` per opp matchup (not just per iter)

**Local pattern**: per-opp cleanup at:
- `eval_diag.py:101-102`
- `eval_report_v8.py:104`
- `eval_elo_ladder.py:488-490`
- `train_rl.py:949-950` (per iter, in main)

**mp_disk_collect bug** (`mp_disk_collect.py:585-591`): cleanup only at end of `_play_vs_opp` indirectly (no explicit cleanup; relies on Python GC). The single `gc.collect() + empty_cache()` at iter end (`mp_disk_collect.py:646-648`) is too coarse — fragmentation accumulates through 6-15 opp matchups.

**Saving**: ~100-300 MB VRAM per worker steady-state. Addresses iter-time creep more than RSS.

**Risk**: low.

---

## 5. Reorder cleanup before result_pipe.send()

**Issue** (`mp_disk_collect.py:425-428` + 646-648): `gc.collect()` at line 647 fires AFTER pickle+disk-write of `all_trajs`. Trajectories sit in RSS during the heavy serialization step + result-pipe send.

**Fix**: `del all_trajs, bundle; gc.collect()` between trajectory dump (line 425) and result_pipe.send (line 428).

**Saving**: ~200-500 MB RSS at iter boundary per worker (depends on n_games × avg trajectory length).

**Risk**: low. Just a reorder.

---

## Combined effect

If we apply #1 + #2 + #4 (low-risk, ~1 hour total work):
- Worker RSS: 2.9 GB → ~1.0-1.3 GB
- Workers total: 23 GB → ~10 GB
- Iter time creep: should flatten (no asyncio listener accumulation)

Adding #5: peak trim at iter boundary, marginal but free.

Adding #3: requires PlayerPool refactor. Defer unless #1+2+4 leave residual issues.

---

## Validation plan

1. Apply #1 first (5 min change). Smoke 1-iter mp run. Confirm worker RSS drops to ~1.5 GB.
2. Apply #2 + #4. Smoke 5-iter mp run. Confirm iter time stays flat (not 41 → 51 → 60 → ...).
3. Apply #5. Smoke 1-iter mp run. Confirm peak RSS drop at iter boundary.
4. Production validation: run a Phase-1-v3-style 5-iter mp burst. Compare iter time series vs current Phase 1 v3 watcher data.

---

## Why we missed this in original mp_disk_collect design

Session 50 mp-redesign focused on:
- SemLock race fix (mp.Pipe instead of mp.Queue)
- Heartbeat protocol with respawn
- Per-worker GPU model + LRU opp_cache

The cleanup patterns that local code had developed over 30+ sessions
(rl_collection, eval_elo_ladder, eval_diag) didn't get propagated.
mp_disk_collect was written as a "fresh slate" that imported the
collection logic but reimplemented its own cleanup — and missed several
patterns that local had learned the hard way.

**Lesson for CIS implementation**: when implementing centralized
inference, audit local patterns the SAME way before committing the
worker side. Specifically check: ckpt strip, listener cancel, per-opp
empty_cache, peak-trim ordering.

---

## Source: audit agent run

Triggered Session 50 cont. `2026-05-07 ~05:50 UTC`. Agent searched 6
local files + mp_disk_collect.py for memory-hygiene patterns. Full
output preserved in conversation transcript.
