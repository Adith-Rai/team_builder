---
name: s68-cleanup-design
description: Design memo for collect-side state cleanup to enable 60w + mb=128 sustainably. Targets ~7 min iter wall by making iter N look like iter 0. Concrete component-by-component analysis + implementation options + validation plan.
metadata: 
  node_type: memory
  type: project
  originSessionId: 38b24000-12d2-44d7-8996-2ca8fd44815c
---

# S68 cleanup design memo

**Goal**: enable sustainable 60w + mb=128 (~7 min/iter target) by making iter N look like iter 0.

**Root cause** (S67 found): state grows iter 0 → iter 1 from ~16 GB to ~76 GB on a 79 GB GPU. mb=128 update needs ~30 GB → OOMs at iter 1+.

**This memo is a DESIGN, not a decision.** Implementation needs code verification (open questions section).

## 🔴 IMPORTANT: S68 Phase A FINDINGS (2026-05-30 evening) — read before implementing

**The 47 GB main proc hypothesis was WRONG.** Phase A instrumentation (commit `aaef191` on `instrumentation/s68-phase-a-memory-diag` branch) measured main proc at all 7 boundaries with `--mem-diag` flag. Concrete data from iter 140 at 60w + mb=128:

| Boundary | alloc | reserved |
|---|---|---|
| iter_start | 0.37 GB | 0.61 GB |
| after_collect_with_trajs | 0.37 GB | 0.61 GB |
| after_build_episodes (trajs+episodes both alive) | 0.37 GB | 0.61 GB |
| after_trajs_None_gc_only | 0.37 GB | 0.61 GB |
| after_empty_cache | 0.37 GB | **0.40 GB** |
| before_ppo_update | 0.37 GB | 0.40 GB |
| after_ppo_update | 0.38 GB | 0.53 GB |

**Main proc steady-state = 0.37 GB.** Trajs+episodes combined < 0.5 GB. The "15 GB trajs" estimate in this memo was wrong by 30×.

**The 47 GB OOM is the PPO update TRANSIENT PEAK at mb=128** — gradients + activations during forward/backward at packed sequence size 1600 trajs × mb=128. After update completes (or fails on OOM), memory drops back to 0.38 GB.

**Revised cleanup picture**:
- Total OOM math: ~47 GB (update transient peak) + ~29 GB (60 worker residual) = 76 GB / 79 GB cap
- Margin: ~3 GB before fragmentation pushes over

**Implications for this memo**:
- ❌ **Target #1 (drop trajs) — SKIP**. trajs are <0.5 GB, releasing buys nothing.
- ✅ **Target #2 (worker kill+respawn) — STILL HIGH LEVERAGE**. Frees 29 GB → leaves 50 GB for update → mb=128 fits cleanly (peak 47 + cushion 3 = 50).
- ⚖️ **NEW: Target #1' — Reduce PPO update peak** via grad checkpointing OR smaller mb (mb=96 likely fits without worker cleanup). Big engineering for the former, easy config for the latter.

**Recommended next step**: skip Target #1, go directly to Target #2 (worker cleanup). If implementation cost is high, do the cheaper alternative first: test mb=96 + worker stays at 60 = should fit with current code (no cleanup needed). If mb=96 update is fast enough (~15-20s vs mb=64's 117s), that's a no-code-change win.

---

---

## Memory accounting — where each GB lives

Based on S67 OOM error data + code inspection (mp_centralized_collect.py, train_rl.py):

### Main proc (train_rl.py — the python -u train_rl process)
At iter 1 OOM: **47.59 GB allocated** by PyTorch.

| Component | Estimated size | Held where | Lifetime |
|---|---|---|---|
| Player model (transformer, ~20M params) | ~80 MB at bf16 | model on cuda | full run |
| Player optimizer state (Adam, 2× params) | ~160 MB at fp32 | optimizer.state | full run |
| Player model grads | ~80 MB | model.grad | created during update |
| BC anchor model (frozen) | ~80 MB | bc_anchor.state | full run |
| Collected trajectories `trajs` | **~10-15 GB** | trajs list of dicts | collect end → update start |
| PPO `episodes` built from trajs | **~10-15 GB** | episodes list | build_ppo_episodes → end of update |
| PPO update activations (mb=128) | **~10-15 GB** | per-minibatch fwd + backward | active during epoch |
| Cached allocator (fragmentation residue) | <1 GB | PyTorch caching | between allocs |

**Likely 47 GB ≈ trajs (15) + episodes (15) + activations (15)**.

**Critical insight**: `trajs` and `episodes` may COEXIST in memory if `build_ppo_episodes` doesn't move-construct (i.e., trajs references aren't released). They overlap during the build step, then trajs should be droppable.

### CIS subprocess (mp.Process spawned by CISServer)
Holds K+1 model slots:
- Default `_CIS_NUM_SLOTS=16` opp slots + 1 player = 17 slots
- Each slot: ~80 MB at bf16
- Total slot memory: **~1.36 GB**
- Plus per-request KV caches during inference (transient, freed after fire)
- Plus low-priority CUDA stream metadata

**CIS subprocess GPU footprint: ~2 GB** (slots + transients). This is SEPARATE from main proc accounting.

### Worker subprocesses (60 mp.Process spawned by `_start_background_collection`)
Each worker:
- Holds its own CUDA runtime context: **~490 MB per worker** (confirmed in OOM listings)
- 60 workers × 490 MB = **29.4 GB**
- This is the bulk of "non-PyTorch memory" in the main proc OOM accounting

**Worker GPU residual is the single biggest blocker for scaling beyond 30w.**

---

## Cleanup targets — ranked by impact

### Target #1: Drop `trajs` after `build_ppo_episodes` returns
**Estimated savings**: ~10-15 GB main proc

**Code location**: train_rl.py line 1440-1448 (after `build_ppo_episodes`).

**Current state** (`experimental/empty-cache-before-update` branch did this):
```python
episodes = build_ppo_episodes(trajs, ...)
trajs = None  # already added in S67 patch
gc.collect()
torch.cuda.empty_cache()
```

**Open question**: does `build_ppo_episodes` deep-copy tensors or share refs?
- If shares refs → `trajs = None` doesn't free GPU memory (episodes still hold same tensors)
- If deep-copies → `trajs = None` frees, but build time peaks at trajs+episodes both alive

**Action**: read `build_ppo_episodes` in ppo.py. Determine ownership model. If shared-ref, restructure to release trajs piecewise OR allow build to take ownership.

### Target #2: Release worker CUDA contexts before update
**Estimated savings**: ~29 GB across 60 worker processes

**Concept**: Workers don't need GPU during update phase. They're idle.

**Options ranked by risk**:

**Option A — Process kill + respawn each iter** (simplest, highest correctness risk-free)
- After `collect_done`, kill all worker processes
- Their CUDA contexts die with them
- Respawn fresh workers at next iter start
- Cost: ~5-10s startup overhead per iter (model state already on disk, just process creation + CUDA init)
- Risk: if worker has in-flight work it loses it. Need clean barrier.
- Net: +5-10s collect-start cost, but +29 GB free during update → mb=128 fits

**Option B — Workers explicitly release CUDA via IPC**
- Send "release CUDA" message to each worker
- Worker calls `torch.cuda.synchronize() + del all gpu tensors + torch.cuda.empty_cache()`
- Worker resumes when CIS sends "warmup" command before next iter
- Cost: ~1-2s per worker, parallel = <2s total
- Risk: torch CUDA context might not actually release without process death
- Verification needed: does Worker's `torch.cuda.empty_cache()` actually return memory to driver?
- Net: cleaner if it works, but unproven release behavior

**Option C — Workers detach CUDA, reattach next iter** (most complex)
- Requires unloading torch from each worker, releasing CUDA driver context, reloading
- High complexity, unclear benefit over Option A

**Recommendation: Option A.** Process kill is well-understood, no ambiguity about whether memory is freed. The 5-10s startup cost is amortized over a ~7 min iter (<2% wall).

### Target #3: Drop CIS opp slots that aren't needed for next iter
**Estimated savings**: ~640 MB (8 unused slots × 80 MB)

**Current**: CIS holds all 17 slots loaded permanently.
**Idea**: at iter end, send "unload slot N" command for slots not picked by next iter's composition.
**Issue**: at next iter start, need to reload — disk read ~3-5s per slot.
**Net**: marginal benefit (640 MB) vs added complexity + reload time. **DEFER** unless main proc cleanup isn't enough.

### Target #4: Release PPO optimizer momentum buffers between epochs
**Estimated savings**: tiny (optimizer is ~160 MB), but might help fragmentation.
**DEFER**: marginal, not worth coupling complexity.

---

## Implementation plan (ordered)

### Phase A — instrument first
Before any cleanup code, add memory diagnostics:
1. Print `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` at:
   - Start of collect
   - End of collect, before build_ppo_episodes
   - After build_ppo_episodes, before trajs cleanup
   - After trajs = None + gc.collect + empty_cache
   - Before PPO update epoch 0
   - After PPO update epoch N
2. Print per-worker `torch.cuda.memory_allocated()` via IPC (one round trip per iter)
3. **Goal**: verify the 47 GB accounting from S67. Identify which component is the biggest.

**This is the "no shortcuts, no assumptions" gate.** Don't implement cleanup until we know exactly what's holding the 47 GB.

### Phase B — main proc cleanup
1. Implement `trajs` release (verify `build_ppo_episodes` ownership)
2. Add explicit `episodes = None` after PPO update done
3. Add `torch.cuda.empty_cache()` at iter end (not just before update)
4. Test at 60w + mb=128 + n_iters=3: does iter 1 OOM disappear?

### Phase C — worker cleanup
1. Implement worker kill+respawn between iters (Option A)
2. Hook into `mp_bg_collector` start/stop cycle
3. Test at 60w + mb=128 + n_iters=5: sustained iter 1-4 all succeed?

### Phase D — validation
1. Run 10-iter test at 60w + mb=128 with full cleanup
2. Validate: zero OOMs, iter wall ~7 min sustained, training metrics equivalent to baseline
3. Numerical equivalence check: PPO update produces same parameter deltas as baseline (within tolerance)

### Phase E — production rollout
1. Merge cleanup behind a flag: `--worker-cleanup-mode {none, kill-respawn}` defaulting to `none`
2. Test in actual long Phase 2 run (50+ iters)
3. If clean, flip default to `kill-respawn`

---

## Risks + fallback

### Risk 1: `build_ppo_episodes` shares refs → trajs release doesn't free memory
**Mitigation**: Phase A instrumentation surfaces this. If true, refactor build_ppo_episodes to take ownership (clear `trajs` list as it consumes).

### Risk 2: Worker kill+respawn breaks state
**Mitigation**: workers don't hold per-iter state across iters (each iter starts fresh from CIS server's slot pool). Killing should be safe.

### Risk 3: Worker respawn slow (model load per worker)
**Mitigation**: workers don't load the model — they connect to CIS server which holds it. Worker just establishes CUDA context + connects pipes. Should be <5s.

### Risk 4: Test methodology — iter 1 OOM was iter-1-specific, not deterministic
**Mitigation**: validate with n_iters=5+, not n_iters=2. Need to see iters 1, 2, 3 all succeed.

### Risk 5: empty_cache showed it could be wrong direction
**Mitigation**: Phase A measurements eliminate guessing. Don't ship Phase B/C until A confirms hypothesis.

### Fallback at every phase
- All changes flag-gated. Default = `none` (current behavior).
- `experimental/` branch isolation until validated for 20+ iters.
- Reproducible v2_dense production config is the safe state to revert to.

---

## Open questions (must answer in implementation)

1. **Does `build_ppo_episodes` (ppo.py) share trajectory tensor refs or deep-copy?** Determines if `trajs = None` actually frees GPU memory.
2. **What's the actual per-component breakdown of main proc 47 GB?** Phase A diagnostics will answer.
3. **Can workers be killed without disrupting in-flight CIS requests?** Need to verify CIS server tolerates worker disconnects gracefully.
4. **Worker respawn cost — actual seconds?** Need empirical measurement, not estimate.
5. **Does `torch.cuda.empty_cache()` in a worker process release memory to the driver (visible from main proc) or just to PyTorch cache?** Critical for Option B vs Option A decision.

---

## Success criteria

Cleanup is "done" when:
- **Zero OOMs** across 10+ consecutive iters at 60w + mb=128 + SP+MCTS
- **Iter 1+ wall ≈ iter 0 wall** (~7 min ± 1 min)
- **Numerically equivalent** updates to baseline (no training-quality regression)
- **Worker respawn overhead** < 30s/iter (i.e., < 7% wall)

If success: merge `--worker-cleanup-mode kill-respawn` as default → production v3 config at ~7 min/iter, ~2.7× faster than current v2_dense.

---

## Out of scope (separate investigations)

- Dropping mcts-hard from training (training-strategy question, not cleanup)
- torch.compile (separate experiment, mile 4)
- Async update overlap (architectural, mile 4)
- CIS opp slot eviction policy (deferred unless Phase B/C insufficient)
- Allocator fragmentation (empty_cache showed it's <1% of the problem)

---

## Effort estimate

- Phase A (instrumentation): **1-2 hours** (add prints, run test, analyze)
- Phase B (main proc cleanup): **1-2 hours** (code change + smoke)
- Phase C (worker kill+respawn): **3-5 hours** (IPC + lifecycle changes + smoke)
- Phase D (validation): **2-3 hours** (10-iter run, analysis, numerical check)
- Total: **~1-1.5 session** of focused engineering work

Risk that requires extra effort: if `build_ppo_episodes` needs refactoring (Risk 1), add 2-4 hours.
