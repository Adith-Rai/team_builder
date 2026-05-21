# S67 Worker Scaling — Final Results

**Date**: 2026-05-19 → 2026-05-21
**Status**: COMPLETE. Canonical Phase 2 stack updated.
**Author session**: S67
**Companion memo**: `memory/project_s67_final_handover.md` (auto-loaded — read first)

---

## TL;DR

**SHIP: `--mp-workers 48 --pool-anchors 11` (12 active opps per iter, mb=64).**
**Wall: 18.0 min/iter iter 0 (Exp 1m), steady-state ~18.5-19 min/iter projected.**
**Phase 2 cost: ~$155 (8w) → ~$93 (48w/pool=12). Save $62/run.**
**Fallback if 1m iter 1 OOMs: `--mp-workers 30` at pool=15, 23.7 min/iter → $115/run.**
**User-decided: 10-12 active opps per iter, snapshot-interval 5-10, 10-iter warmup.**
**Zero quality compromise. Zero code change required (just launch flags).**

**Architectural ceiling confirmed**: per-worker CUDA context = ~490 MB GPU. 48 workers
fit only if active opps capped to 12 (pool=15 OOMs on update at 48 workers).

---

## §1. Experiment design

All experiments shared the same canonical Phase 2 stack except for the variable being tested:

```
--init-from BC v10 --bc-anchor-ckpt BC v10 --bc-anchor-coef 0.1
--cis --bf16 --tier3 --tier3-minibatch-size 64 --packed --no-per-chunk-gc
--cis-min-batch 32 --cis-timeout-ms 50
--games-per-iter 1600 --max-concurrent 200
--n-iters 3 --warmup-iters 0
[hyperparams unchanged]
--snapshot-interval 100 (no growth during 3-iter test)
```

Pool setup via `--pool-anchors`: 7 anchors for pool=8, 14 anchors for pool=15.
All anchors duplicated from variant A/B snapshots (weight identity irrelevant
for wall measurement; what matters is slot count + worker allocation).

---

## §2. Full results table

| Exp | Workers | Pool | Active opps | Ratio | BS | Wall avg | Δ from 8w pool=15 |
|---|---|---|---|---|---|---|---|
| (variant A) | 8 | 1 | 1 | 8:1 | 8 | 9.4 min | — (different scale) |
| (variant A iter 10) | 8 | 2 | 2 | 4:1 | 8 | 13 min | — |
| (variant A iter 20) | 8 | 3 | 3 | 2.67:1 | 8 | 15 min | — |
| (S65 runA) | 8 | 5 | 5 | 1.6:1 | 8 | 17 min | — |
| **5a** | 8 | 8 | 8 | 1:1 | 8 | **31 min** | -3% |
| (pool=15 baseline) | 8 | 15 | 8 (PFSP-selected) | 1:1 | 8 | **32 min** | baseline |
| **1 redo** | 15 | 15 | 15 | 1:1 | 8 | **29 min** | -9% |
| **1c** | 24 | 15 | 15 | 1.6:1 mixed | 8 | **25.5 min** | -20% |
| **1d** | 30 | 15 | 15 | **2:1 clean** | 8 | **21.2 min** ✓ | **-30%** (fallback ship) |
| **1f** | 30 | 15 | 15 | 2:1 | **30** | 21.2 min | -30% (BS noop) |
| **1g** | 8 | 15 | 8 | 1:1 | 8 | 31.8 min (conc=53 forced) | -1% (conc noop) |
| **1e** | 24 | 8 | 8 | **3:1 clean** | 8 | **17.6 min** | (diagnostic only) |
| **1h** | 32 | 8 | 8 | **4:1 clean** | 8 | **15.9 min** | (diagnostic only) |
| **1j** | 48 | 8 | 8 | **6:1 clean** | 8 | collect 13.6 / update 1.8* / **15.4 min** | (diagnostic only) |
| **1k** | 48 | 15 | 15 | 3.2:1 mixed | 8 | collect 18.1 only — **OOM at mb=64 update** | — |
| **1m iter 0** | **48** | **12** | **12** | **4:1 clean** | 8 | **collect 16.1 / update 1.9* / 18.0 min** ✓ | **-47% — NEW SHIP** |

Notes:
- Exp 1e/1h/1j operate at active=8 (below user's diversity floor 10-12). Diagnostic
  only — characterize the ratio mechanism, not shippable.
- *Update wall in iter 0 is typically KL-early-stopped low. Exp 1j iter 0 update = 47s
  vs iter 1-2 average = 140s. Steady-state Exp 1m update expected 130-150s → total
  ~18.5-19 min/iter.
- Exp 1k confirmed the architectural ceiling: 48 workers × ~490 MB CUDA context per
  worker = 23.5 GB GPU eaten by workers, leaving ~53 GB for the main process; mb=64
  update needs ~62 GB → OOM. Exp 1m sidesteps this by capping active opps to 12 (one
  fewer CIS slot, slightly cheaper inference, plus less pressure during update).
- Exp UP-mb 80 was attempted at 30 workers and OOM'd immediately — confirms mb=64
  is at the practical ceiling for the worker counts we're using.

---

## §3. Findings

### §3.1 SHIPPED (NEW): 48 workers + pool=12 (Exp 1m)

48 workers (`--mp-workers 48`) + pool=12 (`--pool-anchors 11` + init) gives a
clean **4:1 worker:opp ratio** and **fits at mb=64** (the prior conjectured
ceiling was mb=64 + 30 workers).

| Step | Wall reduction (vs 8w/32 min baseline) |
|---|---|
| 8w → 15w (1:1 → 1:1) | -9% |
| 15w → 24w (1.6:1 mixed) | -20% |
| 24w → 30w (2:1 clean) | -30% |
| 30w pool=15 → 48w pool=12 (4:1 clean) | -47% (iter 0; ~-42% steady projected) |

### §3.1b FALLBACK SHIP: 30 workers at pool=15 (Exp 1d)

If Exp 1m iter 1 OOMs or steady-state wall > 20 min (validate before launch):
fall back to 30 workers at pool=15. -30% vs 8w baseline. 23.7 min/iter, $115/run.
Safer ceiling, no memory risk. Use as fallback only.

### §3.2 REFUTED: per-worker concurrency confounder (Exp 1g)

**Hypothesis**: 30w might be winning not from ratio but from reduced per-worker
concurrency (200 games/worker at 8w → 53 games/worker at 30w → less asyncio
scheduling overhead per worker).

**Test (Exp 1g)**: 8 workers with `--max-concurrent 53` forces per-worker
concurrency to match 30w's effective value.

**Result**: 31.8 min/iter — identical to 8w baseline (32 min). Concurrency
reduction alone gives zero improvement. The 30w win is genuinely from the
ratio mechanism, NOT from concurrency artifact.

### §3.3 REFUTED: battle server contention as bottleneck at 30w (Exp 1f)

**Hypothesis**: 30 workers / 8 battle servers = 3.75 workers per BS WebSocket
event loop might saturate.

**Test (Exp 1f)**: 30 workers + 30 battle servers on ports 9000-9029, 1:1
worker-to-BS mapping.

**Result**: 21.2 min/iter — identical to 30w/8BS. Battle server event loops
are NOT the bottleneck at 30 workers. **Don't ship 30 BS** — wasted memory
+ process overhead.

### §3.4 REFUTED: pool capping as wall optimization (Exp 5a)

**Hypothesis**: Pool=15 wall comes from rotation overhead + loaded slot count.
Capping pool at 8 would eliminate both.

**Test (Exp 5a)**: pool=8 (7 anchors + init), no rotation between iters.

**Result**: 31 min/iter — identical to pool=15 8w (32 min). **Active opps per
iter** is what matters, not total pool size. Pool capping reduces PFSP
diversity for zero wall benefit. Rejected as path.

### §3.5 REFUTED (by user, on principle): active-opps-per-iter cap

**Proposal**: Cap active opps per iter at 8 from a pool of N, via PFSP
sub-sampling. Would let pool grow arbitrarily while keeping active=8 wall
(31 min).

**User rejection**: Reduces per-iter PFSP diversity. Self-play + external
opps needs 8+ active per iter. Doesn't actually solve the high-wall problem
— just hides it. Same handicap framing as Track B rejection.

### §3.6 REFUTED: S66's mp.send mechanism (S67 stage profile)

S66 attributed the per-fire overhead to the N×mp.send dispatch loop at
`mp_centralized_collect.py:770-784`.

S67 stage profile patches (on `diag/cis-stage-profile` branch) decomposed
the per-fire wall directly. **Dispatch is 0.1-3% of per-fire wall, not the
bottleneck.** The actual dominant components:
- `setup` (forward_spatial + action_encoder, GPU compute): 12ms, batch-invariant
- `output` (temporal + heads + GPU sync): 4-5ms
- `to_torch` (numpy → torch H2D): 5-22ms, scales with batch
- `dispatch` (N×mp.send): 0.1-0.8ms — NOT the bottleneck

The 12ms forward_spatial floor is what the rejected Track B (shared backbone)
would have amortized.

### §3.7 Ratio scaling continues past 4:1 (PRIOR CLAIM CORRECTED)

Earlier S67 framing: "diminishing returns past 2:1-3:1". **REFUTED** by Exp 1j:

At pool=8 (diagnostic):
- 1:1 → 3:1: -43%
- 3:1 → 4:1: -10%
- **4:1 → 6:1 (Exp 1j): -14% additional** (15.9 → 13.6 min collect)

At pool=12 (operating point, Exp 1m iter 0):
- 4:1 clean = 16.1 min collect / 18.0 min total — **the cleanest data point at user's target diversity**

The mechanism keeps paying off; the real ceiling is **GPU memory** (per-worker
CUDA context eating budget) — not the ratio mechanism plateauing.

### §3.8 Architectural ceiling at mb=64 (per-worker CUDA context)

**Each Python worker process consumes ~490 MB GPU just for CUDA context** (PyTorch
primary context + lazy-loaded cuBLAS + cuDNN libraries). Loaded the moment a worker
touches CUDA. Cannot be released without exiting the process.

| Workers | Worker GPU | Update GPU budget | mb=64 fit? |
|---|---|---|---|
| 8 | ~4 GB | ~73 GB | ✓ comfortable |
| 24 | ~12 GB | ~65 GB | ✓ comfortable |
| **30 (fallback ship)** | **~15 GB** | **~62 GB** | **✓ tight (~400 MB margin)** |
| 48 + pool=15 (Exp 1k) | ~24 GB | ~53 GB | ❌ OOM (update needs ~62 GB) |
| **48 + pool=12 (Exp 1m, NEW SHIP)** | **~24 GB + smaller CIS** | **~54 GB** | **✓ fits (Exp 1m iter 0 confirmed)** |
| 60 | ~30 GB | ~47 GB | ❌ (extrapolated) |

To push past 48 workers without OOM would require a **CPU-only-workers refactor**
(workers build CPU tensors only, never touch CUDA — frees ~24 GB GPU). ~50-150 LOC
in V9RLPlayer + build_turn_batch + CIS client. **NOT urgent** — Exp 1m result is
already strong; refactor only if pushing past current limits becomes a priority.

---

## §4. Mechanism (why ratio matters)

CIS slot fires based on `min_batch=32` reached OR `timeout_ms=50` elapsed.
At pool=15 with 8 workers (1:1 ratio), each opp slot receives requests from
exactly 1 worker:
- Per-slot arrival rate ≈ 1 worker × ~50 reqs/sec = 50/sec
- min_batch=32 → fills in 640ms (way past 50ms timeout)
- Slot fires on timeout every 50ms with batch ≈ 2-3 requests
- Lots of small fires, each paying ~12ms architectural floor

With 30 workers (2:1 ratio), each opp slot receives requests from 2 workers:
- Per-slot arrival rate ≈ 2 × 50 = 100/sec
- Still doesn't hit min_batch (would need 320ms)
- Still fires on timeout, but batch ≈ 4-6 requests
- Same per-fire architectural cost amortized over more requests
- Better effective throughput per fire

The mechanism is **per-slot batch fan-in**, not asyncio scheduling or other
worker-side effects. Confirmed by Exp 1g (concurrency control).

---

## §5. Implications for Phase 2

### Updated cost projection (post-Exp 1m)

| Stack | Per-iter wall | 200-iter cost @ $1.50/hr | Savings vs status quo |
|---|---|---|---|
| 8w status quo (pre-S67 canonical) | 32 min | $155 | — |
| 30w (S67 fallback ship) | 21.2 min | $115 | -$40 |
| **48w + pool=12 (S67 NEW SHIP — Exp 1m)** | **~18.5 min** | **~$93** | **-$62** |
| Multi-GPU (not pursued) | <15 min projected | hardware cost trades back | — |

### Canonical launch (UPDATED post-Exp 1m)

See `memory/project_s67_final_handover.md` §2.

**User decisions (2026-05-21) integrated**:
- `--mp-workers 48 --pool-anchors 11` → 12 active opps per iter (user target 10-12)
- `--warmup-iters 10` → stabilize value head over first 10 iters before main training
- `--snapshot-interval 10` → user accepts 5-10 range; keep default 10

**One key flag change vs S64 canonical**: `--mp-workers 48`. Plus `--pool-anchors 11`
sized to 12 active opps + `--warmup-iters 10`.

### Pre-launch checklist (UPDATED per S67)

1. **Verify Exp 1m iter 1 steady-state wall** (`/tmp/exp1m_48w_pool12.log` on prod).
   If update wall > 200s or OOM, fall back to `--mp-workers 30` at pool=15.
2. Restart 8 battle_server.js processes (state degrades after kills, per
   `feedback_battle_server_restart_after_kill`)
3. Verify torch 2.5.1+cu121 + LD_LIBRARY_PATH on prod
4. Verify all 8 battle servers listening on 9000-9007
5. Either `--init-from BC v10` (fresh) or `--resume <SNAPSHOT>` (continuation)
6. Use the updated canonical stack with `--mp-workers 48 --pool-anchors 11 --warmup-iters 10`

---

## §6. Hardware analysis (corrected post-Exp 1k/1m)

Pod: RunPod A100 80GB community, AMD EPYC 7763 64-core (256 logical), **27.2 CPU cores
quota** (cgroup-enforced via `/sys/fs/cgroup/cpu.max` — NOT 256, NOT 16). Steal time = 0.

At 48 workers (Exp 1m):
- CPU load avg: 6-13 of 27.2 quota (22-48% used during steady-state)
- GPU memory: 27 GB used / 79.14 GB total (worker contexts + CIS + main process)
- GPU compute (during collect): 60-90% util
- Steal time: 0 (no neighbor contention)

**Actual ceiling is GPU memory, NOT CPU**. The earlier "16-core" or "256-core" framings
were both wrong:
- 16 was an underestimate (we have 27.2 quota)
- 256 is the physical count; cgroup quota limits us to 27.2
- CPU is not saturated even at 48 workers — load avg 6-13 of 27.2

**Real ceiling**: 48 workers + pool=12 + mb=64 = ~78 GB / 79.14 GB available.
Pushing past 48 workers requires CPU-only-workers refactor (frees ~15-24 GB).

---

## §7. What's left

### Optional further optimizations (if more wall reduction needed)

- **CPU-only-workers refactor** — workers build CPU tensors only, never touch CUDA.
  Frees ~24 GB GPU at 48 workers, unlocks mb=80+ or pool=15 at 48 workers (Exp 1k
  configuration). Real ~50-150 LOC change to V9RLPlayer + build_turn_batch + CIS
  client. Projected 10-15% additional wall reduction OR enables 60-80 workers. Real
  engineering effort. NOT urgent given Exp 1m result.

- **Exp 2: numpy_dict_to_torch H2D optimization** — pinned-memory buffer pool +
  non_blocking H2D. Projected 5-7% additive. Real code change. Quality-preserving.

- **Multi-GPU evaluation** — preserved as option. Would distribute slot fires across
  GPUs, removing serial-stream bottleneck. Real cost change (2-4× hourly hardware).
  NOT urgent given 48w/pool=12 result.

### Phase 2 prep (separate sessions, no architectural changes left)

- **Validate Exp 1m iter 1** steady-state wall before launching production Phase 2.
- Source 100-150 real elite gen-9-OU teams (NEVER mix with 16 mm-competitive eval set).
- lr re-ablation post-arc to confirm `--lr 1e-5` still optimal.
- External opps in PFSP pool (mcts-fast + Kakuna + SmallRL + Minikazam adapters).
- H2H gauntlet wiring (post-Phase-2 eval against published bots).
- Phase 2 launch script with the validated stack.

---

## §8. Cross-references

**Memory (auto-loaded)**:
- `project_s67_worker_scaling_findings.md` — primary memo
- `feedback_dont_propose_principle_violations.md` — Track B rejection lesson
- `feedback_battle_server_restart_after_kill.md` — BS cleanup recipe
- `feedback_freeze_spatial_use_cases.md` — when freeze IS legitimate
- `project_s66_collect_arch_findings.md` — superseded by this memo

**Branches**:
- `diag/cis-stage-profile` @ 668df0f8 — diagnostic patches (CIS_STAGE_PROFILE)
- `perf/freeze-spatial` @ 671e19e4 — preserved but NEVER merged
- master — receives this docs + next-prompt.txt update

**Docs**:
- `docs/S67_WORKER_SCALING_RESULTS.md` — THIS FILE
- `docs/S67_FREEZE_VALIDATION_ANALYSIS.md` — Track B A/B (rejected on principle pre-completion)
- `docs/SHARED_BACKBONE_INVESTIGATION.md` — Track B design (rejected)
- `docs/WAVE_BASED_CIS_INVESTIGATION.md` — Track A design (refuted)
- `docs/REFUTED_LOG.md` — should be updated with S67 refutations (TODO)

---

## §9. Session cost tally

| Item | Cost |
|---|---|
| Diagnostic patches (S67 stage profile + smokes) | ~$5 |
| Track B A/B (variant A + variant B, rejected post-principle) | ~$20 |
| Pool=15 wall measurement (original + redo) | ~$3 |
| Exp 5a (pool=8 baseline) | ~$3 |
| Exp 1 redo (15w pool=15) | ~$5 |
| Exp 1c (24w pool=15) | ~$5 |
| Exp 1d (30w pool=15) | ~$5 |
| Exp 1e (24w pool=8, 3:1 diagnostic) | ~$5 |
| Exp 1f (30w + 30 BS) | ~$5 |
| Exp 1g (8w + conc=53 control) | ~$5 |
| Exp 1h (32w pool=8, 4:1 diagnostic) | ~$3 |
| Exp 1j (48w pool=8, 6:1 diagnostic) | ~$3 |
| Exp 1k (48w pool=15 — discovered OOM ceiling) | ~$2 |
| Exp UP-mb (mb=80 sweep, OOM'd immediately) | ~$1 |
| Exp 1m (48w pool=12 — THE NEW SHIP) | ~$3 (iter 0 done, iter 1 in flight) |
| Failed retries / battle server lesson | ~$3 |
| **Total** | **~$76** |

Pays back in ~1.2 Phase 2 runs (savings of $62/run vs 8w baseline).

---

End of S67 worker scaling results memo.
