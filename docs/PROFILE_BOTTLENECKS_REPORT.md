# Profile-Driven Bottleneck Report — Session 59, 2026-05-13

**This document is AUTHORITATIVE for "where the time goes" findings.**
Supersedes prior framings in `next-prompt.txt` (`§task #22 simulator rewrite`,
`18× speed gap to ps-ppo`) which were CONJECTURE without empirical backing.
Where this doc disagrees with prior docs, trust this one — it's measurement-based.

---

## TL;DR

| Claim | Status | Evidence |
|---|---|---|
| WS+Node layer is the bottleneck | **FALSE** | WS round-trip = 0.165ms localhost, 6000 req/sec |
| Showdown sim is the bottleneck | **FALSE** | Direct BattleStream ceiling = 529-623 turns/sec/Node, multi-Node ~5000 |
| Our throughput is 83 states/sec vs ps-ppo 1500 (18× gap) | **MISLEADING** | Our actual = 22-41 turns/sec; gap framing conflated infra + architectural confounds. Real infra-only-closable gap is smaller and entirely in **Python orchestration**, not sim/WS. |
| **Workers blocked 82% of wall time on `mp.connection.poll()` waiting for CIS** | **TRUE** | Worker_0 viztracer, prod-scale (800g/8w) — see §1 |
| GPU is bottleneck during collect | **FALSE** | 57% GPU util during collect — CIS not saturated |
| Update is fast in prod | **FALSE** | 506s update at 800g, ~70% GPU-bound, ~11% in `collate_episodes` Python loop |

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

## §5. Action plan — top 3 fixes ranked by ROI

| # | Fix | Targets | Expected savings | Effort | Risk |
|---|---|---|---|---|---|
| **1** | **Worker↔CIS dispatch redesign** — aggregate submissions, reduce round-trips per turn; possibly faster IPC | 82% poll-wait on workers | **30-60% of collect** (~300-600s per iter at 800g) | **1-2 wk** (design + impl + validation) | **HIGH** — load-bearing prod path |
| **2** | **Task #12: BC anchor + Tier3 + compile composition** | ~70% GPU-bound update share | **15-25% of update** (~70-140s) | 1-2 days | medium (torch 2.2.x dynamo limits) |
| **3** | **Vectorize `collate_episodes`** | 11% of update (Python loop) | ~10% of update (~50s) | 2-3 days | low (clean unit-testable optimization) |

**Secondary** (lower priority):
- Compile CIS inference model — small collect gain (1-3% — workers are IPC-bound, not CIS-compute-bound)
- Tokenizer `_encode_pokemon_block` optimization — small CIS forward gain
- CUDA graphs for full update step — possible 10-20% but out of scope unless others tap out

### §5.1 Recommended sequencing

1. **First session after this one**: Fix #3 (collate_episodes vectorize). Low-risk independent win. Proves the profile-driven methodology delivers.
2. **In parallel or next**: Fix #2 (compile composition). 1-2 days focused. Biggest update-phase win.
3. **After 1+2 land**: **Design session** for Fix #1 (no code yet). We compare IPC redesign options (shared memory ring buffer vs explicit submit-aggregation vs Unix socket vs hybrid) with the user before any code.
4. **Implementation of Fix #1**: 1-2 weeks across multiple sessions.

### §5.2 Combined effect (CONJECTURE — must validate per-fix)

If all three fixes hit their estimates:
- collect: 958s → ~400-600s (30-60% reduction)
- update: 506s → ~300-370s (~25-30% reduction)
- **iter total**: 1464s → ~700-970s = **~1.5-2× speedup**

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
| Workers 82% poll-wait | **CONCRETE** | viztracer worker_0 |
| CIS GPU util 57% | **CONCRETE** | nvidia-smi during prod |
| CIS forward 16ms | **CONCRETE** | CIS-STATS instrumentation in prod |
| Update is ~70% GPU-bound | **STRONG INFERENCE** | gap between Python time + wall time; expected for transformer training |
| compile would save 20-40% of GPU update share | **CONJECTURE** | based on typical transformer compile gains, NOT measured on our specific composition |
| Fix #1 saves 30-60% of collect | **CONJECTURE/UPPER BOUND** | based on "82% IPC wait" with most being avoidable through aggregation |
| Fix #2 saves 70-140s of update | **CONJECTURE** | derived from typical compile gains; verify per-fix |
| Fix #3 saves ~50s of update | **CONJECTURE** | vectorization typically captures 80-95% of Python loop time |
| ps-ppo at 1500 turns/sec on RTX 3090 | **CONCRETE FROM EXTERNAL DOC** | `docs/CLOUD_DEPLOY.md:65` — but irrelevant to our infra since architecturally different |
| "18× speed gap to ps-ppo" original framing | **REFUTED** | the gap is mostly architectural (stateless, Random Battles, no temporal); infra-only closable gap is much smaller and entirely Python orchestration |
