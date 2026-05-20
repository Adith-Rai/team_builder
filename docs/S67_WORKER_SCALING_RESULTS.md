# S67 Worker Scaling — Final Results

**Date**: 2026-05-20
**Status**: COMPLETE. Canonical Phase 2 update ready to ship.
**Author session**: S67
**Companion memo**: `memory/project_s67_worker_scaling_findings.md` (auto-loaded)

---

## TL;DR

**Ship one flag change: `--mp-workers 30` (vs default 8).**
**Wall reduction: -34% at pool=15 (32 → 21.2 min/iter).**
**Phase 2 cost: ~$155 → ~$105 per 200-iter run.**
**Zero quality compromise. Zero code change.**

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
| **1d** | 30 | 15 | 15 | **2:1 clean** | 8 | **21.2 min** ✓ | **-34%** |
| **1f** | 30 | 15 | 15 | 2:1 | **30** | 21.2 min | -34% (BS noop) |
| **1g** | 8 | 15 | 8 | 1:1 | 8 | 31.8 min (conc=53 forced) | -1% (conc noop) |
| **1e** | 24 | 8 | 8 | **3:1 clean** | 8 | **17.6 min** | (diagnostic only) |
| **1h** | 32 | 8 | 8 | **4:1 clean** | 8 | **15.9 min** | (diagnostic only) |

Note: Exp 1e and 1h operated at active=8 which is below the user's stated
diversity floor (self-play + external opps need 8+ active per iter). Their
data is diagnostic — characterizes the ratio mechanism — not shippable
configurations.

---

## §3. Findings

### §3.1 SHIPPED: 30 workers at pool=15

The combination of more workers AND clean 2:1 worker:opp ratio gives -34%
vs the 8w status quo. Each ratio step contributes meaningfully:

| Step | Wall reduction |
|---|---|
| 8w → 15w (1:1 → 1:1, just more workers) | -9% |
| 15w → 24w (1:1 → 1.6:1 mixed ratio) | -11% |
| 24w → 30w (1.6:1 → 2:1 clean) | -17% |
| **Cumulative 8w → 30w** | **-34%** |

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

### §3.7 Ratio scaling diminishes past 2:1-3:1

At pool=8 (diagnostic only):
- 1:1 → 3:1: -43%
- 3:1 → 4:1: only -10%

Returns diminish. Practical sweet spot is 2:1-3:1 depending on operating point.
At pool=15 (user's operating point), 2:1 = 30 workers = ship-ready.

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

### Updated cost projection

| Stack | Per-iter wall | 200-iter cost @ $1.50/hr | Savings vs status quo |
|---|---|---|---|
| 8w status quo (pre-S67 canonical) | 32 min | $155 | — |
| **30w (S67 SHIP)** | **21.2 min** | **$105** | **-$50** |
| 30w + Exp 2 (H2D opt, projected) | ~20 min | ~$95-100 | -$55-60 |
| Multi-GPU (not needed) | <15 min projected | depends on hardware cost | — |

### Canonical launch (UPDATED)

See `memory/project_s67_worker_scaling_findings.md` §8.

Only **one flag change** from the prior S64-era canonical stack:
`--mp-workers 30` (was default 8). Everything else unchanged.

### Pre-launch checklist (UPDATED per S67)

1. Restart 8 battle_server.js processes (state degrades, per `feedback_battle_server_restart_after_kill`)
2. Verify torch 2.5.1+cu121 + LD_LIBRARY_PATH on prod
3. Verify all 8 battle servers listening on 9000-9007
4. Either `--init-from BC v10` (fresh) or `--resume <SNAPSHOT>` (continuation)
5. Use the updated canonical stack with `--mp-workers 30`

---

## §6. Hardware analysis (max workers limit)

Pod: RunPod A100 80GB community, AMD EPYC 7763, **27.2 CPU cores quota** (cgroup-enforced).

At 30 workers:
- CPU: ~50% of 27.2 quota used
- GPU memory: ~18 GB of 80 GB
- GPU compute (during collect): 60-90% util
- Steal time: 0 (no neighbor contention)

**Theoretical max workers on current pod: ~50-60** before CPU quota saturation.

**Ratio ceiling at pool=15 without architecture change**:
- 30 workers → 2:1 ratio (SHIPPED)
- 45 workers → 3:1 ratio (untested, projected ~10% further gain to ~19 min/iter)
- 60 workers → 4:1 ratio (probably hits CPU/orch limits)

**Diminishing returns** observed past 2:1-3:1 (only -10% from 3:1 to 4:1 at pool=8).
**30 workers is the practical sweet spot** balancing gain vs resource pressure.

---

## §7. What's left

### Optional optimizations (if more wall reduction needed)

- **Exp 2: numpy_dict_to_torch H2D optimization** — pinned-memory buffer
  pool + non_blocking H2D. Projected 5-7% additive at active=15. Real code
  change. Quality-preserving. 1 session.

- **Exp 1i: 45 workers at pool=15** — untested. Projected ~10% further gain
  (~19 min/iter, ~$95 Phase 2). Worth testing only if Exp 2 isn't enough
  and user wants below $100/run.

- **Multi-GPU evaluation** — preserved as option. Would distribute slot fires
  across GPUs, removing serial-stream bottleneck. Real cost change (2-4×
  hourly hardware). NOT urgent given 30w/2:1 result.

### Phase 2 prep (separate sessions, no architectural changes left)

- Source 100-150 real elite gen-9-OU teams (NEVER mix with 16 mm-competitive eval set)
- lr re-ablation post-arc to confirm `--lr 1e-5` still optimal
- External opps in PFSP pool (mcts-fast + Kakuna + SmallRL + Minikazam)
- H2H gauntlet wiring
- Phase 2 launch script with the validated stack

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
| Exp 1d (30w pool=15) — THE SHIP-READY ANSWER | ~$5 |
| Exp 1e (24w pool=8, 3:1 diagnostic) | ~$5 |
| Exp 1f (30w + 30 BS) | ~$5 |
| Exp 1g (8w + conc=53 control) | ~$5 |
| Exp 1h (32w pool=8, 4:1 diagnostic) | ~$3 |
| **Total** | **~$64** |

Pays back in ~1.3 Phase 2 runs (savings of $50/run).

---

End of S67 worker scaling results memo.
