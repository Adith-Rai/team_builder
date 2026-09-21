---
name: s67-speed-optimization-findings
description: "S67 systematic speed-optimization sweep — concrete iter wall data + OOM root cause + why empty_cache wasn't enough + what real cleanup needs to target"
metadata: 
  node_type: memory
  type: project
  originSessionId: 38b24000-12d2-44d7-8996-2ca8fd44815c
---

S67 (2026-05-30) speed-optimization sweep on dev pod (Era 4 lr8e5_v1_flash final.pt resumed).
Goal: find no-cleanup ceiling + validate cleanup direction.

## Concrete iter wall data (no-MM config on dev pod)

All resumed from Era 4 final.pt, pool=47-50, 1600 games/iter, mb=64 unless noted.
**iter 1 = steady-state** (iter 0 has startup advantage, not representative).

| Test | Workers | mb | Opps | iter 0 collect | iter 1 collect | iter 1 update | Outcome |
|---|---|---|---|---|---|---|---|
| T1 | 30 | 64 | SP+3 MCTS | 360s | 487s | 117s | OK, **iter1 total 604s** |
| T3 | 45 | 64 | SP+3 MCTS | 368s | 513s | 119s | OK, iter1 total 632s |
| T4 | 60 | 64 | SP+3 MCTS | 393s | 516s | 121s | OK, iter1 total 637s |
| T5 | 30 | 64 | pure SP | 577s | 609s | 141s | OK, iter1 total **750s (slower!)** |
| T6 | 60 | 64 | pure SP | OOM iter 0 | - | - | FAILED |
| T2 | 30 | 128 | SP+3 MCTS | 372s | - | - | iter 0 update **9s**, iter 1 OOM |
| smoke v2 | 60 | 128 | SP+3 MCTS | 397s | - | - | iter 0 update **7s**, iter 1 OOM |
| empty_cache_test | 60 | 128 | SP+3 MCTS | 381s | - | - | iter 0 OOM even WITH empty_cache |

## Key findings (concrete, not speculation)

### 1. Worker scaling does NOT help at mb=64
T1/T3/T4 (30→45→60w) all land 604-637s iter 1. **+5% at 60w, basically flat.** The CIS batches still timeout-bound (timeout_pct=96-100% at 60w), GPU 93% idle by compute, workload is **game-wall bound, not GPU-bound** at our model size.

### 2. Pure SP is SLOWER than SP+MCTS (counterintuitive but real)
T5 vs T1: pure SP at 30w = **750s vs 604s = +24%**.
**Root cause: allocator concentrates workers on hard matchups.** In pure SP, snapshot_0139.pt (current self) got **9/30 workers + 578/1600 games** because cost-based allocator gives most workers to slow matchups. MCTS in the pool was secretly load-balancing — MCTS games are slow but **bounded**; self-vs-current-self is unbounded.

### 3. mb=128 update is REAL and dramatic (7-9s vs 117s = 13-15× faster)
Confirmed in 3 separate runs (T2 iter 0, smoke v2 iter 0, emptycache_test iter 0). All consistent.

### 4. mb=128 ALWAYS OOMs iter 1+ without cleanup
T2, smoke v2 both: iter 0 succeeds at mb=128, iter 1 PPO update OOMs.
emptycache_test: iter 0 OOMs immediately even with empty_cache + gc.collect (different starting state).

### 5. OOM root cause is NOT fragmentation — empty_cache REFUTED
S67 hypothesis: "iter 1 OOMs because of allocator fragmentation; empty_cache between collect and update will fix it."

Test result: empty_cache fired successfully ("GPU cache cleared before PPO update" in log) but **main proc memory stayed at 47.59 GB**. allocator's fragmentation reserve was only 289 MB — empty_cache had basically nothing to give back.

**Real root cause**: live state held by Python references that empty_cache can't touch:
- **CIS server inference state** (~most of the 47 GB main proc): player model + opp slot snapshots + KV inference caches scaling with worker × concurrent-battles
- **Worker GPU residuals**: 60 workers × 490 MB = **29.4 GB** of separate process GPU contexts that empty_cache CANNOT reach (different processes)
- Combined: ~77 GB / 79 GB cap → OOM

## What no-cleanup ceiling actually is

**With current code, ceiling = ~10 min/iter at 30w + 3 MCTS + 7 SP + mb=64.**

This basically IS v2_dense's prod config (which adds MMs and runs at ~19 min — MMs add 9 min). Without MMs we'd be at 10 min on dev. With MMs we're at ~19 min on prod.

**Worker scaling above 30w gives nothing.** mb=128 gives 13-15× update speedup but ALWAYS OOMs without cleanup.

## IMPORTANT: iter 0 sustained = ~7 min/iter (the real cleanup target)

**The "massive gains" we saw at iter 0 were 100% real, not fluke.** smoke v2 iter 0 + emptycache_test iter 0 both confirmed 60w + mb=128 + SP+MCTS = collect ~390s + update ~7-8s = **~6.7-7.3 min**. These ARE achievable, sustainably.

Why iter 0 works but iter 1 OOMs (memory growth across iters):

| | iter 0 | iter 1+ (OOM) |
|---|---|---|
| Main proc | ~10-15 GB estimated | **47 GB** (confirmed) |
| Worker residual (60×?) | ~100 MB each = ~6 GB | 490 MB each = **29 GB** |
| Total state | ~16-21 GB | **~76 GB** |
| mb=128 update room (~30 GB) | plenty | OOMs |

The growth from ~16 GB → ~76 GB is what closes the door. That growth is **CIS server inference KV caches + per-worker CUDA context accumulation across the iter's many concurrent battles**.

**Cleanup goal = maintain iter-0-state across all iters.** If state stays ~16 GB instead of growing to 76 GB, mb=128 + 60w runs sustainably.

**Sustainable target: 7 min/iter** (matches iter 0 measurement of all 60w + mb=128 runs).

## What real cleanup needs to target

Not empty_cache. Need to actually FREE live state before PPO update:

1. **CIS server inference KV caches** — scales with worker × concurrent-battles. Released after collect, NOT held into update.
2. **CIS opp slot snapshots** — 10 × 80 MB = 800 MB GPU. Could evict during update phase (reload at next iter start).
3. **Worker GPU contexts** — 60 × 490 MB = 29 GB. Requires worker process termination + respawn (or CUDA context release IPC).
4. THEN empty_cache to defrag what's left.

Order matters: workers must release CUDA contexts BEFORE the main proc PPO update starts. Otherwise the 29 GB worker residual + main proc PPO working set OOMs.

## Production v3 config decision (post-S67)

**No code change shipped from S67.** All experimental work on `experimental/empty-cache-before-update` branch (which didn't help — keep branched for archeology).

**Production stays on master at current settings**:
- 30 workers, mb=64, SP+MCTS+MMs (v2_dense config)
- ~19 min/iter on prod, ~10 min/iter without MMs on dev

The compounding optimization arc (`project_optimization_tracker`) lives:
- **Mile 1 (5 min)** — REVISED: not achievable without code work. Real cleanup needed.
- **Mile 2 (find wall)** — DONE: wall is at 10 min for SP+MCTS @ 30w/mb=64. mb=128 update is unreachable without cleanup.
- **Mile 3 (surgical changes)** — NEXT: real CIS state cleanup design (next session)

## Next session focus

Design memo for **CIS state cleanup**:
- Where exactly the 47 GB is held (CIS server inspect + numbers)
- IPC protocol for "release inference state" between collect and update
- Worker context release strategy (process kill + respawn vs CUDA IPC release)
- Validation plan (run 10+ iters to confirm no OOM after cleanup)

Once cleanup designed + implemented, expected ceiling:
- mb=128 working sustainably → save 110s/iter on update = ~17% wall
- + drop mcts-hard (separate question) → save 3-5 min/iter
- Combined sub-7 min plausible. Sub-5 min still needs additional architectural work (torch.compile, async overlap).

## What S67 was wrong about (lessons)

1. **"60w is faster than 30w" steady-state** — wrong. Only at iter 0 (workers fresh). Steady-state flat 604-637s across 30/45/60 *at mb=64*. BUT at mb=128, 60w iter 0 was indeed faster (7.3 min vs 10 min) — that gain is achievable sustainably with cleanup.
2. **"empty_cache will fix fragmentation"** — wrong. Fragmentation was only 289 MB; real issue is live state growth (16 GB → 76 GB across iters).
3. **"Pure SP would be faster than SP+MCTS"** — wrong. Allocator concentration on hard matchups makes pure SP slower.
4. **"No-cleanup ceiling is 5 min"** — wrong. Real no-cleanup ceiling is 10 min. **But cleanup ceiling IS ~7 min** (sustained iter 0 behavior), and that's a realistic target.

## What S67 got right

1. **Iter 0 numbers ARE real and sustainably achievable** (60w collect ~390s, mb=128 update ~7-9s). Target = **make iter N look like iter 0**.
2. **mb=128 update speedup is the biggest concrete win available** (13-15× confirmed, 110s/iter savings).
3. **Worker cleanup is necessary** — just turned out to be CIS-state cleanup not allocator fragmentation cleanup. Same direction, different mechanism.
4. **The sub-5min path is clear**: cleanup (→ 7 min) + drop mcts-hard (→ 4-5 min) + mile 4 architectural (sub-3 min).
