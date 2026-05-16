# Profile-Driven Bottleneck Report — Session 59, 2026-05-13

**This document is AUTHORITATIVE for "where the time goes" findings.**
Supersedes prior framings in `next-prompt.txt` (`§task #22 simulator rewrite`,
`18× speed gap to ps-ppo`) which were CONJECTURE without empirical backing.
Where this doc disagrees with prior docs, trust this one — it's measurement-based.

**Last updated**: S64 Phase B wrap, 2026-05-15 (sequence packing SHIPPED + merged to master).

**See also**: `docs/REFUTED_LOG.md` — companion doc consolidating all techniques tried + refuted across sessions, with evidence + rationale + revisit-conditions.

---

## S64 Phase B update (SHIPPED): sequence packing measured at prod, merged to master

**S64 Phase B status**: full pipeline SHIPPED. Measured at prod (1600g/200conc, BC v10 init, fresh battle servers, 1-iter A/B):

- **Update wall: 1865s → 1729s = -7.3%**
- **Overall wall: 2451s → 2320s = -5.3%** (collect ~unchanged, packed only affects update)
- 200-iter Phase 2 saving: ~7.3 hr / ~$11
- Numerical drift: small (kl/bc_kl ~-0.003), stable across smoke/prod scales, **favorable direction** for bc_kl (closer to BC anchor)

5 bit-equiv gates passed (B.2 temporal, B.3+B.4 forward fp32+bf16, B.5 loss 12-combo with non-zero BC anchor, B.6 e2e smoke). Code is pure additive — legacy `forward_ppo_sequence` / `collate_episodes` / `_ppo_loss_batched_internal` untouched; `--packed` flag opts in. Compiled-path branch raises NotImplementedError if `--packed + --compile` (compile REFUTED at prod anyway).

Master at **`ba2ced64`** (merge commit `cf0963fc` + launch_rl.sh exec-mode fix). Canonical Phase 2 launch updated to include `--packed` + `./launch_rl.sh` wrapper. See `memory/project_s64_phase_b_results.md` for full details.

### S64 Phase B finding — update phase is ~4× super-linear in B

Surfaced at Phase B wrap (user question). Update wall scaling 100g smoke (30s) → 1600g prod (1865s) is **62× for 16× games**. Collect scales correctly (15.4×). Decomposition under S62 profile data:

- GPU compute: scales ~linearly (~25× for 16× games)
- **Python/CPU orchestration: scales 71×** — this is the super-linearity

Mechanism: Tier 3 minibatch=16 → ceil(B/16) chunks. Smoke 100g = 21 chunks, prod 1600g = 300 chunks (14× more). Per-chunk Python overhead is ~constant per chunk (`gc.collect`, `torch.cuda.empty_cache`, dict construction, loop iteration, tensor accumulation). Each per-chunk invocation has a large constant overhead, so 14× more invocations → 14× more overhead pile-up.

**Implication for technique projections — CORRECTED**: my initial revision to 1.5-3× for CUDA Graphs was OVER-projection. CUDA Graphs eliminates per-launch/per-alloc CUDA overhead (5-15% honest range). It does NOT eliminate the Python control flow (gc.collect, empty_cache, dict construction, loop iteration). The per-chunk Python loop is attacked by separate techniques in the tracker: **2a `--tier3-minibatch-size 32`** (halves chunk count if memory fits, est. 1.2-1.4×), **2b gc.collect/empty_cache audit** (est. 1.05-1.20×), **2c BC anchor caching** (est. 1.05-1.10×). They stack additively. Combined optimistic ceiling 2a×2b×2c×#2 ≈ 1.5-2.5× update wall. **ALSO**: the "92% Python orchestration" decomposition is from S62 prod profile only — at smoke scale the split may differ. Step A of post-wrap investigation profiles smoke + compares.

---

## S64 Phase A historical (precursor to Phase B): collate_episodes_packed shipped to branch

**S64 status pre-Phase-B**: sequence packing was the right priority. Phase A shipped; Phase B was next.

**S64 step-back** (full-confirmed at prod): Option A (seq-packing on existing arch) wins over Option B (drop temporal stack). Step-back was prompted because seq-packing is itself an arch-touching commitment; the question was whether we'd want to drop the temporal stack anyway (making seq-packing partially sunk cost). Profile data answered: NO — the cat hot-spot lives in `collate_episodes` data prep, not in the temporal stack.

**Prod profile (CONCRETE, with_stack=True + record_shapes=True at 1600g/8w/conc=200, 1 iter)**:
- Total `aten::cat` self CPU = 93.89s = **23.6% of update CPU** (398s) — matches S62 24% headline
- 541,593 cat events parsed and shape-bucketed:
  - Big batch-stack (n≥8 same shape): 60.7% (57.0s) — `_stack_batch_dim` (ppo.py:301)
  - Padding pair (two tensors summing to 200): 27.4% (25.7s) — `_stack_pad_one_episode` (ppo.py:267)
  - Attn-internal (empty Input Dims): 11.2% (10.5s) — qkv reshapes inside FA
  - Other misc: 0.7%
- **88.1% of cat self CPU lives in collate-driven patterns at prod scale.** Reachable by sequence packing.
- Top single signature: `cat([(200, 6, 285)] × 16)` at 990 events × 36.66ms avg = **38.6% of all cat time = ~9% of update CPU from one callsite** (`_stack_batch_dim` cross-episode stack of spatial-feature tensors).

**Bonus prod finding**: `aten::fill_` count at prod = 160,486 (vs S62 160,501) DESPITE S63's `set_to_none=True`. Means most fill_ calls are NOT grad zeroing — they're padding zero allocations from `collate_episodes` (`torch.zeros(pad_shape, ...)` at ppo.py:202/235/265/279). Seq-packing eliminates these too. Adds 62s CPU + 4.2s GPU to savings projection.

**Total reachable by seq-packing**: ~150s CPU + ~5s GPU = **~38% of update CPU**.

**Phase 1 (torch 2.5.1 upgrade isolation) PASSED**: 8/8 compat tests on prod via isolated venv. Three operational findings baked in: LD_LIBRARY_PATH wrapper for cuDNN 9, varlen API = flex_attention not dedicated module, triton 3.1.0 (re-test C5 compile path at Phase B).

**Phase A SHIPPED** on branch `perf/seq-packing` at `70fd33df`: `collate_episodes_packed(episodes, max_seqlen, device, tail)` in `ppo.py:347-557` alongside legacy `collate_episodes` (untouched). 11/11 equivalence unit tests pass first run (`test_collate_packed.py`). Pure additive change — function is DEAD CODE by design until Phase B wires it. Zero pod time.

**Phase B NEXT** (awaiting fresh session + user authorization): refactor `forward_ppo_sequence` to consume packed via `flex_attention` with per-episode causal `BlockMask`. Estimated 1-2 sessions, $3-5 pod, bit-equivalence gate at fp32 + bf16.

**ARCH revisit REFUTED AS PRIORITY**: dropping the temporal stack addresses ~0 of dominant waste (temporal stack consumes <8% of update wall, bounded by total CUDA kernel time = 8%). May revisit post-Phase-2 as quality exercise; NOT as part of optimization arc. See `REFUTED_LOG.md` entry #33.

**Memory sources**:
- `memory/project_s64_arch_revisit_profile_findings.md` — full step-back reasoning + prod data
- `memory/project_s64_phase1_torch_upgrade_results.md` — torch 2.5.1 venv isolation
- `memory/project_s64_phase_a_results.md` — Phase A shipped + §4 detailed Phase B plan

---

## S63 update (post-free-wins): -4.2% update wall SHIPPED

**S63 outcome**: free wins SHIPPED at -4.2% update wall (below the 8-15% a-priori projection from S62 profile but accepted because numerically equivalent + best-practice + audit showed remaining `.item()` calls are NaN gates that can't be safely removed).

**Two changes shipped to master** (`463827b4`):
1. `optimizer.zero_grad(set_to_none=True)` at 12 callsites in `ppo.py` — eliminates allocate-then-fill pattern for grad zeroing.
2. Defer chunk-stats `.item()` to epoch boundary in eager Tier3 hot path — `_ppo_loss_batched_internal` called directly, tensor accumulators, one `.item()` per epoch instead of ~3600 per epoch (chunks × stats keys).

**Measurement** (1600g/8w/conc=200, 1 iter, canonical Phase 2 stack):
- collect: 588s (unchanged from S62 — same Option B)
- update: 1850s vs 1931s baseline = **-81s = -4.2%**
- Health stats clean (kl=0.0155, bc_kl=0.0156)

**Why projection was off**: profile showed 9000 `.item()` calls/iter at avg 4.3ms = 38s sync. My changes eliminated ~3600 (the obvious chunk-loop ones). Remaining ~5400 are NaN gates (forward NaN check ppo.py:1435, loss NaN check ppo.py:1455) — NOT removable without correctness risk. The 81s wall reduction exceeded the direct sync-elimination prediction (~50s), suggesting set_to_none also enabled better GPU pipelining.

**Memory source**: `memory/project_s63_free_wins_results.md`

---

## S62 update (post-prod-validation + update profile): orchestration-bound, optimization arc begins

**S62 was decisive**: Fix #1 Option B SHIPPED (-27% collect at prod), Fix #2 (--compile) REFUTED at prod (-8% i.e. SLOWER). Update profile with torch.profiler reveals the update phase is ORCHESTRATION-bound, not GPU-compute-bound. Pivots us from one-fix-at-a-time to a multi-session optimization sequence.

**Fix #1 Option B SHIPPED**: `--cis-min-batch 32 --cis-timeout-ms 50` at prod scale (1600g/8w/conc=200) → collect 809s → 592s = **-27%**. H3 (slot starvation under one-opp-per-worker + tight 15ms timeout) confirmed dominant via per-slot CIS-STATS: meanq 3.9 → 15, maxq 6 → 32-35. Update flags into canonical Phase 2 launch stack. See `memory/project_s62_fix1_b_results.md`.

**Fix #2 (--compile whole-fn) REFUTED at prod**: 4 prod-scale data points at 1600g/8w showed --compile is **8% SLOWER per step** than eager (27μs vs 25μs). S60 9× speedup was a smoke-only (64g/2w) anomaly. Recompile loop hypothesis refuted via TORCH_LOGS=recompiles (only 2 recompile events in 34-min update). The S60 capability work (BC anchor + Tier3 + compile composition) stands as engineering; we just don't pull the --compile trigger. Canonical Phase 2 stack has --compile REMOVED. See `memory/project_s62_fix2_prod_validation.md`.

**Update profile findings (torch.profiler kernel breakdown at prod, 38% overhead)**:
- **CUDA kernels = only 8% of update wall** (147s of 1931s). Update is orchestration-bound, NOT GPU-compute-bound.
- `aten::copy_`: 28% of GPU time, 583k calls/iter — most are padding copies inside `collate_episodes`
- `aten::cat`: 24% of CPU time, 541k calls/iter — also collate-driven (confirmed S64 step-back)
- `aten::fill_`: 13.3% CPU, 160k calls/iter — grad zeroing AND padding zero allocations
- `_local_scalar_dense` (`.item()` syncs): 9% CPU, 9000 calls × avg 4.3ms = 38s sync wait
- `_efficient_attention_forward`: 9.5% — some is padding compute waste

**Two free wins identified** (8-15% wall projection, one-line changes):
- `optimizer.zero_grad(set_to_none=True)` (eliminates fill_ for grad zeroing) → SHIPPED S63
- `.item()` audit + batch (eliminates obvious chunk-loop syncs) → SHIPPED S63

**Big architectural opportunity**: sequence packing + varlen attention → S64.

**Demoted via profile**:
- Liger-Kernel — we use GELU + LayerNorm, NOT SwiGLU + RMSNorm (does not apply). See REFUTED_LOG.md entry #3.
- Regional torch.compile as primary win — not compute-bound on individual kernels; CUDA Graphs covers same ground better. See REFUTED_LOG.md DEMOTED #1.

**Memory source**: `memory/project_s62_update_profile_findings.md`

**Stacked optimistic ceiling across the optimization arc**: 3-5× wall reduction. 32 min/iter → 6-10 min/iter at prod scale (CONJECTURE, depends on Phase B/seq-packing landing + CUDA Graphs).

---

## S60 update (post-recon): Fix #2 SHIPPED, Fix #3 REFUTED

**Fix #2 (BC anchor + Tier3 + compile composition) — SHIPPED in S60**
on branch `perf/compile-bc-anchor-composition` (commits `cbc9ec01` +
`a6743f0d`). Prod-pod smoke confirmed: compile cache hit gives ~9×
speedup on update phase iter-2 (190s → 21s with BC anchor + minibatch=16
at 64 games / 2 workers / --cis). See `memory/project_s60_fix2_design.md`
for design notes. The mutex that previously forced "BC anchor OR
compile, not both" is gone.

**Fix #3 (vectorize collate_episodes) — REFUTED in S60** by microbench.
Two variants tested: advanced-index scatter (V3) was 0.68-0.81× of V1
(strictly SLOWER); pre-alloc + slice-assign (V2) was 0.98-1.10× of V1
(within noise). The 55s/iter `collate_episodes` figure in §4.1 was
viztracer-INCLUSIVE time — most of it is bandwidth-bound child C ops
(`torch.cat`, `torch.as_tensor`) that Python-level vectorization can't
speed up. See `memory/project_s60_fix3_refuted.md` for the data + variants
attempted. Diagnostic scripts kept on master at `scripts/diag/bench_collate_*.py`
+ `scripts/diag/test_collate_episodes_vec_bitexact.py`.

**Updated §5 action plan (priority for S61+)**:

| # | Fix | Status | Notes |
|---|---|---|---|
| 1 | Worker↔CIS dispatch redesign | **NEXT — design session first** | 30-60% collect savings, HIGH risk, 1-2wk impl |
| 2 | BC anchor + Tier3 + compile composition | **SHIPPED (S60)** | 9× update speedup measured; ready for Phase 2 |
| 3 | Vectorize `collate_episodes` | **REFUTED (S60)** | Python-level vectorization doesn't pay off |

For the S61 Fix #1 design session, see the §5.1 sequencing notes below
(the original sequencing assumed Fix #3 first as a warm-up; that's been
removed since Fix #3 doesn't pay off).

---

---

## TL;DR

**Current state (S64 Phase B + 2a BOTH SHIPPED at mb=64)**: collect-side fix SHIPPED (-27% via Fix #1 Option B); update-side free wins SHIPPED (-4.2% via S63 set_to_none + .item() defer); **sequence packing SHIPPED** at prod (-7.3% update via S64 Phase A+B `--packed` flag, merged to master at `ba2ced64`); **2a (minibatch=64 at bf16) SHIPPED** at prod after bf16 sweep (mb=16→551s, **-68.1% / 3.14× incremental**; corrects initial mb=32 ship from fp32-smoke defect). **Cumulative -70.5% / 3.38× update wall at packed mb=64 vs legacy mb=16.** 200-iter Phase 2: ~$200/5.7d → ~$90/2.5d. NEXT: 2b (gc/empty_cache audit), 2c (BC anchor caching), #2 CUDA Graphs. --compile DROPPED from canonical Phase 2 stack (S62 prod-refuted).

| Claim | Status | Evidence |
|---|---|---|
| WS+Node layer is the bottleneck | **FALSE** | WS round-trip = 0.165ms localhost, 6000 req/sec |
| Showdown sim is the bottleneck | **FALSE** | Direct BattleStream ceiling = 529-623 turns/sec/Node, multi-Node ~5000 |
| Our throughput is 83 states/sec vs ps-ppo 1500 (18× gap) | **MISLEADING** | Our actual = 22-41 turns/sec; gap framing conflated infra + architectural confounds. Real infra-only-closable gap is smaller and entirely in **Python orchestration**, not sim/WS. |
| **Workers blocked 82% of wall time on `mp.connection.poll()` waiting for CIS** (S59) | **TRUE then; FIXED S62** | Worker_0 viztracer, prod-scale (800g/8w) — Fix #1 Option B `--cis-min-batch 32 --cis-timeout-ms 50` recovers -27% at prod. See §3 + S62 update at top. |
| GPU is bottleneck during collect | **FALSE** | 57% GPU util during collect — CIS not saturated |
| Update is ~70% GPU-bound (S59 inference) | **REFUTED at prod scale, S62 profile** | torch.profiler shows CUDA kernels = only 8% of update wall (147s of 1931s). Update is ORCHESTRATION-bound. See S62 update at top. |
| Update has 11% in `collate_episodes` Python loop (S59) | **SUPERSEDED — actual prod is 24% CPU cat, 88% collate-driven** | S64 prod with_stack profile: 541k cat events, 23.6% of update CPU, 88.1% in `_stack_batch_dim` + `_stack_pad_one_episode`. Plus 160k `aten::fill_` calls (most are padding zeros from collate, not grad zeroing). Sequence packing addresses both. |
| --compile delivers 15-25% update savings | **REFUTED at prod, S62** | 4 prod data points at 1600g/8w show --compile is 8% SLOWER per step (27μs vs 25μs). S60 9× was smoke-only anomaly. DROPPED from Phase 2 stack. |
| Liger-Kernel drop-in saves update time | **REFUTED (arch mismatch), S62** | We use GELU + LayerNorm, not SwiGLU + RMSNorm. Drop-in doesn't apply. |
| Dropping temporal stack saves update time | **REFUTED AS PRIORITY, S64 step-back** | Temporal stack <8% of update wall. Dominant waste is in data prep (collate_episodes), not in temporal stack. May revisit post-Phase-2. |

---

## §1. How we measured (methodology)

Three independent measurement passes on RunPod A100 80GB (port 47913):

### Pass 1 — Bench scripts (no instrumented training)
- `scripts/bench/bench_ws_echo.{js,py}` — WS layer floor
- `scripts/bench/bench_direct.js` — direct `BattleStream` (no WS) sim ceiling
- `scripts/bench/bench_parallel.js` — N concurrent battles in one Node process

### Pass 2 — viztracer profile training run, 200g/4w/--pipeline (round 2)
- Branch `diag/profile-bottlenecks-s59`, commit `ea3ff92f`
- env-var-gated viztracer hooks at 3 process entry points
- Main process JSON captured (379MB); worker JSONs failed (atexit didn't fire
  because workers blocked on `recv()` after main exited)

### Pass 3 — viztracer with SIGUSR1 save handler, 800g/8w/--pipeline (round 3)
- Branch same, commit `7e3423b4`
- Added SIGUSR1 handler to `profile_hook.py` for synchronous save
- All 10 process JSONs captured (~1.8GB total)
- Real prod-scale collect (958s) and update (506s)

**Caveat: viztracer adds ~20-30% overhead.** Absolute numbers in profiles are
slightly inflated; relative breakdowns are reliable.

**Caveat: CIS process trace buffer filled after 10s** (1M entries × ~96K
events/sec from torch op tracing). CIS data captures startup behavior only —
sufficient for confirming per-batch cost (22ms, matches independent CIS-STATS)
but not steady-state distribution.

---

## §2. Bench results — sim & WS are NOT the bottleneck

### §2.1 WS round-trip floor (`bench_ws_echo`)
```
WS echo round-trip (10000 msgs):
  mean   = 0.1648ms
  median = 0.1516ms
  p95    = 0.2263ms
  p99    = 0.3145ms
  total  = 1652.6ms => 6051 req/sec
```
**Reading**: localhost WS overhead is essentially free (~0.165ms/round-trip).
Each turn has ~2 WS round-trips → ~0.33ms of WS time per turn. **Bypassing WS
captures <1% of iter time.**

### §2.2 Direct BattleStream sim ceiling (`bench_direct.js`)
```
Direct BattleStream, 50 battles of gen9randombattle:
  total turns:   2018
  total wall:    3814ms
  turns/sec:     529.1
  battles/sec:   13.11
  avg turns/btl: 40.4
```
**Reading**: pure sim (no WS, no model, default move every turn) at **529 turns/sec on single Node thread**.

### §2.3 Parallel BattleStream (`bench_parallel.js`)
```
conc=8:    484 turns/sec
conc=32:   527 turns/sec
conc=100:  538 turns/sec
conc=200:  623 turns/sec (best)
```
**Reading**: single Node process maxes ~620 turns/sec. Multi-Node (matching our 8 battle_servers) → projected ceiling ~4000-5000 turns/sec.

### §2.4 Production reality (from prod iter lines)
- v4_freshbs iter 50: 72233 steps / 1759s collect = **41 turns/sec**
- v4_freshbs iter 51: 56455 steps / 2601s collect = **22 turns/sec**

**Gap analysis**:
- Sim ceiling: ~5000 turns/sec (8 Nodes parallel)
- Prod reality: 22-41 turns/sec
- **We extract ~0.5-1% of available sim throughput** → bottleneck is NOT sim

---

## §3. Worker collect bottleneck — IPC waiting (the headline)

### §3.1 Worker_0 profile (round 3, 800g/8w)

Worker_0 had 200 games (PFSP-weighted, 2× the median worker). Most representative.

**Trace span: 1518s** (iter 50 collect+update window + spawn).

**Top 10 by inclusive total time:**

| Function | Total | %wall | Calls | Avg |
|---|---|---|---|---|
| `CISInferenceBatcher.submit.<locals>.<lambda>` | 1672.30s | 110%* | 585 | 2.86s |
| `CISClientHandle.infer` | 1672.29s | 110%* | 585 | 2.86s |
| `CISClientHandle._send_with_future` | 1554.89s | 102%* | 588 | 2.64s |
| **`_ConnectionBase.poll` (the headline)** | **1244.35s** | **81.95%** | **6770** | **184ms** |
| `Connection._poll` | 1244.32s | 81.94% | 6770 | 184ms |
| `wait` (mp.connection) | 1244.28s | 81.94% | 6770 | 184ms |
| `_PollLikeSelector.select` | 1243.88s | 81.92% | 6770 | 184ms |
| `_do_collect_iter_cis` | 869.91s | 57.29% | 1 | 869.91s |
| `CISClientHandle._await_future` | 125.84s | 8.29% | 585 | 215ms |
| `Future.result` | 125.84s | 8.29% | 603 | 209ms |

*\>100% = concurrent across asyncio coros / OS threads*

### §3.2 Reading

**The dominant cost is `_ConnectionBase.poll` — 82% of wall time on workers, blocked on the mp.Pipe response from CIS.** This is partly the recv_loop OS thread sitting on poll(), partly asyncio coros awaiting future results.

Cross-reference with prior CIS-STATS data (NOT from this profile, from prod iter lines):
- CIS forward latency = 16ms
- CIS fire_latency = 20ms (so 4ms IPC overhead per request)
- GPU util during collect = 57%

**Synthesis**: CIS GPU is NOT saturated (57%). CIS per-batch is fast (22ms). But workers spend 82% of wall blocked on IPC. The bottleneck is **round-trip latency × number of round-trips**, not throughput.

### §3.3 Why round-trips dominate (hypothesis, CONJECTURE)

- Workers run 25 concurrent battles each via asyncio
- Each battle's turn requires an inference (or 2 — player + opp)
- asyncio interleaves submissions; CIS sees them arrive over time
- CIS's batch-formation window (`timeout_ms=15`) doesn't aggregate enough → small batches (CIS-STATS showed `maxq` 2-8 typical, not 25)
- Many small batches → many round-trips → high cumulative IPC wait time per worker

### §3.4 What this means for fixes

**Fix the IPC pattern, not the IPC speed.** Replacing `mp.Pipe` with shared memory might shave 0.5-2ms per round-trip — incremental. **Reducing the number of round-trips by aggregating submissions** (or by changing the asyncio pattern so CIS sees a full batch's worth of requests at once) could save 30-60% of collect time.

---

## §4. Update phase bottleneck — GPU-bound + collate Python overhead

### §4.1 Main process profile (round 3, prod-scale)

**Trace span: 1507s** (covers iter 50 collect+update).

**Top during update phase (ppo_update_batched = 506s):**

| Cost center | Time | % of update |
|---|---|---|
| GPU-bound CUDA kernel time (async, not in Python timing) | **~350s** | **~70%** |
| `Module._call_impl` (all torch fwd calls, Python overhead) | 68s | 13% |
| **`collate_episodes` (Python data prep loop)** | **55s** | **11%** |
| `Tensor.backward` (Python-side wrapper, not GPU) | 27s | 5% |
| `TransformerEncoder.forward` (cumulative) | 20s | 4% |
| Misc (BC anchor fwd, optimizer.step, KL check) | ~6s | 1% |

### §4.2 Reading

**~70% of update is GPU compute** (CUDA kernels executing async; viztracer can't see inside them). That's a fundamental cost of the model + batch size — not Python-overhead-fixable without compile or arch changes.

**The 11% in `collate_episodes` IS fixable.** It's a per-episode Python loop with `.stack()` / `.pad()` that could be vectorized. 55s/iter saved on update.

**CONJECTURE on the 70% GPU-bound share**: torch.compile typically saves 20-40% on transformer forward kernels. Applied to update's GPU share: 70-140s saved per iter. **NOT YET MEASURED on our specific arch + Tier3 + BC anchor composition** — task #12 must land to enable.

### §4.3 CIS process — Tokenizer is surprisingly heavy

Only 10s of CIS captured (buffer overrun), but informative:

| Cost center in CIS forward | % of forward (22ms) |
|---|---|
| TransformerEncoder | 58% (~13ms) |
| **Tokenizer** | **38% (~8.5ms)** |
| `_encode_pokemon_block` (within Tokenizer) | 25% (~5.5ms) |
| numpy_dict_to_torch | 17% (~3.7ms) |

Tokenizer at 38% of forward is more than expected. Possibly optimizable.

---

## §5. Action plan — current state of all techniques (post-S64 Phase A)

### §5.0 Historical fixes (S60-S62)

| # | Fix | Targets | Expected savings | Effort | Risk | Status |
|---|---|---|---|---|---|---|
| **1** | **Worker↔CIS dispatch redesign** — Option B: bump CIS batch window (`min_batch=8→32, timeout_ms=15→50`) to fix slot-starvation (H3) | small CIS fires (maxq=2-8 → 32-35) | **-40% collect at Phase C scale; -27% at prod 1600g/8w** | 1 session (~$4) | low | **SHIPPED S62, in canonical Phase 2 stack** |
| **2** | **BC anchor + Tier3 + compile composition** | ~70% GPU-bound update share | smoke 9× (64g/2w); prod -8% i.e. SLOWER | 1-2 days | medium | **CAPABILITY shipped S60, --compile DROPPED at prod S62. See REFUTED_LOG.md #1** |
| **3** | **Vectorize `collate_episodes` (Python-side)** | 11% of update Python loop (S59 framing — superseded) | ~10% of update (~50s) | 2-3 days | low | **REFUTED S60. See REFUTED_LOG.md #2. Superseded by S64 sequence packing.** |

### §5.1 Optimization arc (S63+, multi-session, profile-validated priorities)

Source: `memory/project_optimization_tracker.md`. Current state below.

| # | Technique | Status | Measured / projected | Effort | Risk | Branch | Memo |
|---|---|---|---|---|---|---|---|
| **0a** | `optimizer.zero_grad(set_to_none=True)` (12 callsites in `ppo.py`) | **SHIPPED S63** | combined w/ 0b: **-4.2% wall** | 0.25 session | ZERO | merged to master | `project_s63_free_wins_results.md` |
| **0b** | `.item()` audit + batch (defer chunk-stats to epoch boundary in eager Tier3) | **SHIPPED S63 w/ 0a** | combined: -4.2% wall | 0.5 session | ZERO | same branch as 0a | same memo |
| **1** | **Sequence packing + varlen attention** — eliminates 24% CPU `aten::cat`, 28% GPU `aten::copy_`, 16% CPU `aten::fill_` (padding zeros), most FA padding waste. Requires torch upgrade 2.2.1 → 2.5+. | **PHASE A SHIPPED S64; PHASE B NEXT** | **1.5-3× wall (prod-data-supported, ~38% update CPU reachable)** | 2-4 sessions total | MEDIUM-HIGH (cascades through collate + forward_ppo_sequence + loss) | `perf/seq-packing` at `70fd33df` | `project_s64_phase_a_results.md` (Phase B plan in §4) |
| 2 | CUDA Graphs over train_step — eliminates 5.9% direct launch overhead (1.8M launches) + most of 1.4M `aten::empty` allocations | PENDING (after #1) | 1.3-2× | 1-2 sessions | MEDIUM | `perf/cuda-graphs` | tbd |
| 3 | Investigate `cudaMemcpyAsync` 62k @ 990μs root cause | PENDING (opportunistic) | UNKNOWN | 0.5-1 session | LOW | `perf/memcpy-investigation` | tbd |
| ❌ | ~~Liger-Kernel~~ | **REFUTED — arch mismatch (GELU + LayerNorm, not SwiGLU + RMSNorm)** | — | — | — | — |
| ❌ | ~~Regional `torch.compile` as primary win~~ | **DEMOTED — CUDA Graphs covers same ground better** | — | — | — | — |
| ❌ | ~~Token-bucket dynamic batching as separate technique~~ | **DEMOTED — overlaps with #1** | — | — | — | — |
| ARCH | ~~Drop temporal stack~~ | **REFUTED AS PRIORITY (S64 step-back) — addresses ~0 of dominant waste; may revisit post-Phase-2** | — | — | — | — |

**Stacked optimistic ceiling (profile-validated)**: ~3-5× wall reduction. 32 min/iter → 6-10 min/iter at prod scale.

**Sequence packing reachable wall** (CONCRETE from S64 prod profile, with caveat band):
- Padding cats (`_stack_pad_one_episode`): 27.4% of cat time = ~6.5% of update CPU — ELIMINATED
- Padding fill_ (`torch.zeros(pad_shape, ...)`): ~16% of update CPU — ELIMINATED
- Big batch-stack cat: ~14% of update CPU — reduced (still cats once per feature key, less dispatch overhead)
- Realistic floor: ~25% update wall reduction. Realistic ceiling: ~40%. Phase E gate at 30% achievable but not comfortable.

**S62 Option B outcome**: tested `--cis-min-batch 32 --cis-timeout-ms 50` at 100g/2w smoke (-46% collect) + 3-iter A/B at 200g/4w (-40%/-41%/-40% per iter, mean -40%). Confirmed mechanism via per-slot CIS-STATS instrumentation (S62 A3): meanq 3.9 → 15, maxq 6 → 32-35, timeout_pct 100% → 84-94%, fire rate halved. H3 (slot starvation under one-opp-per-worker dispatch + tight 15ms timeout) was the dominant bottleneck at our scales — matches §3.4's prediction. See `memory/project_s62_fix1_b_results.md` for full data + mechanism analysis. Production launch should pass these flags.

**Options A (worker-side aggregation) + C (shared-mem ring buffer)** designed in `project_s61_fix1_design.md` but skipped per S62 result — Option B captured ≥25% of collect (the Phase D threshold), so the more expensive options weren't needed. Could be revisited if production-scale (8w/conc=200) shows much smaller gain than Phase C (would imply H1/H2 contributing more than at our test scale).

**Fix #1 prod-scale validation (S62)**: 1600g/8w/conc=200, 1 iter A/B. Baseline collect=809s, variant collect=592s (-27%). Compression vs Phase C (-40%) was as predicted: at prod scale slot 0 baseline meanq=7.7 (vs Phase C 3.9) was already approaching min_batch=8, so 71% of baseline fires were minbatch-driven (vs Phase C's 100% timeout). Variant gain shrinks accordingly. Still above 25% Phase D threshold. See `memory/project_s62_fix1_b_results.md`.

**Fix #2 prod-scale validation (S62)**: 4 data points at 1600g/8w/conc=200 confirm `--compile` is **8% SLOWER per step** than eager at prod scale. S60 "9× speedup" was a smoke-only (64g/2w) anomaly that does NOT generalize. Recompile loop hypothesis REFUTED (only 2 recompiles in 34-min update via TORCH_LOGS=recompiles). The `forward_ppo_sequence` graph break + per-call compile dispatch overhead exceeds whatever kernel-fusion savings exist at prod chunk size. **Decision: drop `--compile` from canonical Phase 2 launch stack.** The S60 capability work (BC anchor + Tier3 + compile composition + bc_anchor_enabled closure flag) is still useful as engineering — we just don't pull the `--compile` trigger in production. See `memory/project_s62_fix2_prod_validation.md`.

**Fix #2 actual measurement (S60 prod-pod smoke)**: compile cache hit
gave **9× speedup on update phase iter-2** (190s → 21s with BC anchor +
minibatch=16). Beats the 15-25% projection by a wide margin — most of
the iter-0 overhead was compile time, not steady-state compute.

**Fix #3 actual measurement (S60 microbench)**: V3 (advanced-index
scatter) was 0.68-0.81× of V1 — strictly SLOWER. V2 (simpler pre-alloc
+ slice-assign) was within noise (0.98-1.10×). The projected 10% savings
were not achievable via Python-level vectorization (the 55s was
viztracer-INCLUSIVE time including child C ops that are bandwidth-bound).

**Secondary** (lower priority):
- Compile CIS inference model — small collect gain (1-3% — workers are IPC-bound, not CIS-compute-bound)
- Tokenizer `_encode_pokemon_block` optimization — small CIS forward gain
- CUDA graphs for full update step — possible 10-20% but out of scope unless others tap out

### §5.2 Historical sequencing (post-S60 era — superseded by §5.1 above)

The below sequencing reflects S60-era thinking, before S62 update profile revealed update is orchestration-bound (not GPU-bound). Preserved as historical record; current authoritative sequencing is §5.1.

1. **S61: Fix #1 design session** — no code; compare IPC redesign options
   (worker-side submission aggregation, CIS `timeout_ms` window tuning,
   shared-memory ring buffer, Unix socket binary protocol, hybrid).
   Pick lowest-risk path that captures most of the 30-60% gain. Design
   correctness A/B gate at the same time.
2. **S62+: Fix #1 implementation** across multiple sessions with strict
   correctness gate (A/B against current master at same init/seed; smart_avg
   + eval signal must match within noise or strictly improve).
3. **After Fix #1 lands**: Phase 2 prep (task #14 Metamon scaling, source
   100-150 elite teams, etc.) on the now-fast/cheap infra.

### §5.3 Combined effect (S60-era projection — partially superseded)

S60-era projection assumed Fix #2 9× speedup would hold at prod (refuted S62) and that Fix #1 captured 30-60% of collect (Option B captured -27%, in range). Current authoritative projections in §5.1 above.

If Fix #1 hits its estimate (Fix #2 already shipped, Fix #3 dropped):
- collect: 958s → ~400-600s (30-60% reduction) — pending Fix #1
- update: 506s → much lower (Fix #2's 9× cache hit means update is no
  longer the bottleneck; warm-cache update at 64g/2w was 21s, scaling
  linearly to 800g/8w ~ 26s/iter)
- **iter total at 800g**: was 1464s → projected ~430-630s = **~2.3-3.4×
  speedup** if Fix #1 lands. Most of the remaining cost is collect.

---

## §6. Things this profile DID NOT capture

### §6.1 CIS steady-state behavior
Viztracer buffer overran after 10s of CIS activity. We have CIS's startup pattern, not the steady-state distribution. For more CIS detail, either:
- Reduce viztracer's tracer_entries to capture longer (paradoxically — limits per-event detail but extends time window)
- Use torch.profiler with scheduled capture window
- Use sampling profiler

### §6.2 GPU kernel-level timing
The ~70% "GPU-bound" inference about update is from the gap between Python time and wall time, not direct GPU profiling. To confirm + see per-kernel breakdown, use **torch.profiler with CUDA activity tracking** around `ppo_update_batched`. Would tell us *which* kernels dominate.

### §6.3 Network Volume / disk I/O
Not measured. Likely small but unconfirmed.

### §6.4 What workers OTHER than worker_0 see
Only worker_0 analyzed in depth. Other 7 workers have similar JSONs (184MB each) but not yet analyzed. Worth a cross-worker comparison if Fix #1 design needs more detail.

---

## §7. Artifacts (all under `data/profiling/round3/`, gitignored)

| File | Size | What |
|---|---|---|
| `profile_main_3293518.json` | 199MB | Main process viztracer (full iter 50 + start of 51) |
| `profile_cis_3295243.json` | 193MB | CIS subprocess viztracer (first 10s only — buffer overrun) |
| `profile_worker_0_3308356.json` | 184MB | Worker 0 (200 games — heaviest worker) |
| `profile_worker_1...7_*.json` | 184MB each | Workers 1-7 (~85 games each) |
| `gpu_util.csv` | 46KB | nvidia-smi 1Hz utilization+memory time-series |
| `profile_run_diag.log` | 122KB | Training log with iter lines + PROFILE messages |

To re-analyze:
```bash
python scripts/bench/analyze_viztracer.py <profile_*.json>
```

For interactive flame-graph viewing:
```bash
vizviewer data/profiling/round3/profile_worker_0_*.json
# Opens browser; navigate timeline, click functions for details
```

---

## §8. Concrete vs Conjecture labels (audit trail)

| Statement | Status | Why |
|---|---|---|
| WS round-trip = 0.165ms | **CONCRETE** | Bench, 10000 samples |
| Direct BattleStream = 529 turns/sec single-thread | **CONCRETE** | Bench, 50 battles |
| Workers 82% poll-wait (S59) | **CONCRETE then; REMEDIATED S62 via Fix #1 Option B (-27% collect at prod)** | viztracer worker_0; Option B batch-window tuning resolved most |
| CIS GPU util 57% | **CONCRETE** | nvidia-smi during prod |
| CIS forward 16ms | **CONCRETE** | CIS-STATS instrumentation in prod |
| Update is ~70% GPU-bound (S59 inference) | **REFUTED at prod scale, S62 torch.profiler** | CUDA kernels = only 8% of update wall (147s of 1931s). Update is ORCHESTRATION-bound. |
| `aten::cat` is 24% of update CPU, 541k calls/iter | **CONCRETE, S62 prod torch.profiler** | 1600g/8w/conc=200, no stacks, single iter |
| 88.1% of `aten::cat` self CPU is in collate-driven patterns | **CONCRETE, S64 prod with_stack=True profile** | shape decomposition of 541,593 events + python_function frame stack reconstruction |
| `aten::fill_` 160k calls at prod are mostly padding zeros (not grad zeroing) | **STRONG INFERENCE, S64** | Count unchanged 160,501 → 160,486 between S62 baseline and S63 post-set_to_none — implies fill_ isn't grad zeroing. Shape signatures match `torch.zeros(pad_shape, ...)` calls in collate_episodes. |
| compile would save 20-40% of GPU update share (S59 conjecture) | **REFUTED at prod scale, S62** | 4 prod data points showed --compile is 8% SLOWER per step. Not compute-bound on individual kernels at prod scale. |
| Liger-Kernel saves update time (S62 research candidate) | **REFUTED (arch mismatch), S62** | We use GELU + LayerNorm, not SwiGLU + RMSNorm |
| Fix #1 saves 30-60% of collect (S59 upper bound) | **CONCRETE at prod: -27%**, S62 | Option B tuning delivered as Phase D threshold met (-27% ≥ 25% gate) |
| Sequence packing saves 1.5-3× update wall | **CONJECTURE, prod-data-supported** | ~38% of update CPU reachable per S64 prod profile; varlen attention floor unmeasured |
| S63 free wins save 8-15% update wall (S62 projection) | **PARTIALLY REFUTED, SHIPPED at -4.2%, S63** | Obvious .item() calls captured; remaining are NaN gates that aren't safely removable. Ceiling for the free-wins category. |
| Drop temporal stack saves update time | **REFUTED AS PRIORITY, S64 step-back** | Temporal stack <8% of update wall; dominant waste is in data prep |
| ps-ppo at 1500 turns/sec on RTX 3090 | **CONCRETE FROM EXTERNAL DOC** | `docs/CLOUD_DEPLOY.md:65` — but irrelevant since architecturally different |
| "18× speed gap to ps-ppo" original framing | **REFUTED, S59** | Gap is mostly architectural (stateless, Random Battles, no temporal); infra-only closable gap is much smaller and entirely Python orchestration |
