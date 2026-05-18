# Multi-Process CIS + CUDA MPS — Design Memo

**Branch**: `perf/multi-process-cis-mps`
**Authored**: S66 (2026-05-18), from master `fe9df8b9`
**Goal**: solve the pool-growth collect-time slowdown that left Phase 2 not launch-ready at S65 wrap.
**Status**: ⚠️ **SUPERSEDED — see STATUS UPDATE below before reading the original design.**

---

## ⚠️ STATUS UPDATE (S66 wrap, 2026-05-18) — READ FIRST

**This memo's proposed architecture (1 player + N opp processes via CUDA MPS) is REFUTED at our pool scale.**

Phase A bench measured:
- N=2: 1.93× throughput @ batch=4, 1.75× @ batch=16 (matches Databricks H100 sweet spot ✓)
- N=6: **0.90× @ batch=4, 0.74× @ batch=16** (WORSE than single process)

MPS scheduler thrashes when 6+ clients each want 100% of SMs. Combined with the user's clarification that pool can reach 17 currently and possibly 33+ later, the multi-process-per-slot pattern is not viable for our pool target.

**Pivoted direction**: shared backbone with frozen spatial during PPO, per-snapshot temporal+heads trainable. See `docs/SHARED_BACKBONE_INVESTIGATION.md` (next session continues there) and `memory/project_s66_collect_arch_findings.md` for full context.

**What still applies from this memo**:
- §1 raw-data findings (per-fire overhead 21-37ms, mechanism: N×mp.send) — VERIFIED CONCRETE, source-of-truth checked
- §2.1-§2.3 options analysis — STILL valid as decision log
- §3 (CUDA MPS technical context) — partially valid; sweet spot is N=2 not N=6 for us
- §5 operational concerns — apply to ANY multi-process approach; relevant for future hardware change

**What's REFUTED from this memo**:
- §2.4 Option D (multi-process per slot for our pool scale)
- §4 staged plan (Phase A.3 gate failed, B/C/D obsolete as written)
- §0 "this is the correct architecture" claim

Investigation continues. The original design below is preserved for the decision-log value (why we explored this, what we measured, what we learned).

---

## §0. TL;DR

S65 declared "all cheap tuning levers exhausted, pool-growth slowdown unsolved." S66 re-investigation against raw logs found the S65 framing was right in direction but wrong in mechanism:

- S65 memo claimed per-fire fixed overhead was 3-5ms. **Raw CIS-STATS data shows it's 21-37ms at prod scale** (4-7× higher).
- The dominant component is **`worker_resp_writers[widx].send(msg)` called in a tight loop** (`mp_centralized_collect.py:770-784`), one mp.send per request per fire. At batch=24 across 15 workers, that's 24 separate IPC sends per fire.
- This means the centralized-in-one-process CIS pattern fights the architectural grain. Industry standard for multi-agent RL inference is **process-per-task** (Metamon's `subprocess.Popen` per agent, OpenAI Five's distributed Forward Pass GPUs, AlphaStar's TPU-per-agent).

**Proposal**: refactor CIS into **multi-process per slot** (1 process for player, 1 per opp snapshot), all sharing the single A100 via **CUDA MPS daemon**. Each process implements widx-batched send-back. Workers route inference requests via a `pool_slot_map` to the correct process.

**Why this is "correct, not convenient"**: eliminates GIL serialization of slot fires, eliminates N×mp.send-per-fire overhead, eliminates per-slot starvation (each opp slot has its own process running concurrent on GPU via MPS). Future-proofs for external opp adapters, multi-gen, doubles/VGC.

**Load-bearing unknown**: MPS scaling efficiency at our specific workload (small transformer forwards, ~10 fires/sec/process, N=6+ processes). Documented up to N=2 (Databricks H100), N=16-32 (GROMACS compute). N=6 on A100 with our workload is not directly measured. Phase A's three sub-experiments derisk this.

---

## §1. The problem — what the raw data says

### §1.1 Pool growth slowdown is real and architectural

From S65 raw `stage_a_test.log` at prod scale (1600g/15w/conc=200):

| Iter | Pool | Collect (s) | Δ from pool=1 |
|---|---|---|---|
| 0 | 1 | 555 (Run A baseline 32/50ms) | — |
| 2 | 2 | 646 | +16% |
| 4 | 3 | 776 | +40% |
| 6 | 4 | 978 | +76% |
| 8 | 5 | 1026 | +85% |
| 9 | 5 | 1263 (Stage A) | +127% |

User's stop criterion: collect ≤600s (10 min/iter) at steady state. We meet this at pool=1, miss decisively at pool≥3. For a 200-iter Phase 2 with snapshot-interval=10, most iters are pool≥5 → projected wall ~$100+/run with current architecture, plus inability to scale to the multi-gen final form.

### §1.2 The mechanism — per-fire overhead at prod scale

`mean_fire` in CIS-STATS measures wall time from `_t_fire_start` (line 659) to `_t_fire_end` (line 801). `fwd` measures GPU forward time only. The difference is per-fire dispatch overhead.

**Direct A/B at pool=1 from runA_15w_retest.log vs stage_a_test.log**:

| Config | mean_fire | fwd | **Overhead** | Batch (meanq) | timeout_pct |
|---|---|---|---|---|---|
| Run A (32, 50ms) | 57.6ms | 21.1ms | **36.5ms** | 24 | 60% |
| Stage A (10, 25ms) | 39.9ms | 18.8ms | **21.1ms** | 14 | 8% |

**Per-fire overhead is 21-37ms at prod scale.** The S65 memo's "3-5ms fixed cost" claim was inferred, not measured, and is wrong by 4-7×.

**Mechanism (from code reading at mp_centralized_collect.py:770-784)**:

```python
# Dispatch per-request slices
cursor = 0
for entry in pending:
    widx = entry[0]
    req_id = entry[1]
    bsize = entry[2]
    end = cursor + bsize
    slice_out = _slice_numpy_output(mega_np, cursor, end)
    cursor = end
    msg = {"status": "ok", "out": slice_out}
    if req_id is not None:
        msg["req_id"] = req_id
    try:
        worker_resp_writers[widx].send(msg)   # ONE mp.send per request
    except Exception:
        closed_workers.add(widx)
```

At batch=24, this loop does 24 separate mp.Pipe sends. Plus `_stack_numpy_batches` (line 667) and `numpy_dict_to_torch` (line 668) before forward. These together account for the measured 21-37ms.

**Per-fire overhead SCALES with batch size**, not constant. Bigger batches → more sends → more overhead. This shape was missed in the S65 framing.

### §1.3 Pool growth compounds the problem

At pool=N, opp slot arrival rate drops to ~480/N inf/sec total (PFSP allocator may bias). With static (32, 50ms), batches fire on timeout with mean batch ~480×0.05/N = 24/N. At pool=5, batch ≈ 5.

Smaller batches → less GPU amortization (per-inference GPU time grows). PLUS the same per-fire dispatch overhead per fire. Combined effect: per-fire wall stays roughly constant but FIRE COUNT grows (more slots × similar fire rate).

CIS dispatch loop is **single-threaded** (one Python interpreter, single CUDA stream shared across slots — line 535-544 + 678-680). At high pool, the loop saturates: 6 slots × 12 fires/sec/slot × 30ms wall ≈ 2.16 sec/sec → impossible → fires queue → collect wall balloons.

This is the architectural wall S65 hit. Tuning (min_batch, timeout) inside this single-process pattern can shift where the saturation point is but cannot remove the saturation.

### §1.4 What S65's tuning experiments actually proved

S65 refuted Stage A's specific formula. The CIS-STATS data confirms:
- Stage A's (10, 25ms) at pool=1: 652s collect (vs Run A's 555s = +17% worse)
- Stage A at pool=5: 1231s collect (Run A pool=5 ~1026s)

But Stage A confounded two variables (min_batch AND timeout). A clean experiment varying one at a time hasn't been done. **However**, the architectural saturation analysis above predicts no (min_batch, timeout) pair in this single-process pattern closes the pool=5 gap. So the clean re-test, while methodologically valid, has a low ceiling. Skipped.

CPU opp inference refutation HOLDS (verified against raw `bench_cpu.log`): 344 inf/s aggregate vs 2300 GPU baseline. Benchmark methodology was actually optimistic for CPU (tested `forward_spatial` only, prod runs full forward path).

Battle-server degradation refutation HOLDS (verified against runA/B/C logs): Run A iter 0 = 555s, Run B mean = 567s, Run C mean = 567s. Within noise. Effect magnitude excludes BS state as cause.

---

## §2. Architecture options considered

### §2.1 Option A — Status quo + widx-batched mp.send (Stage 1 alone)

**Idea**: in the existing `_dispatch_slot`, group `pending` by widx and send one msg-dict per worker instead of one per request. Worker recv-loop demuxes.

**Impact**: bounded. At batch=24 with 15 workers: 24 sends → 15 sends = -38% of mp.send time = saves ~12-15ms per fire = -10-15% collect at pool=1. **Doesn't address the architectural saturation at high pool** because the central problem is the dispatch loop being serialized in one process, not the per-fire cost.

**Verdict**: REJECTED as standalone. Folded into Option D as a piece of per-process dispatch.

### §2.2 Option B — Cross-slot batching via shared backbone

**Idea**: train all snapshots with a shared spatial encoder, only the temporal stack + heads specialize per snapshot. Then at inference, batch the spatial forward across all opp slots in one call.

**Pros**: addresses the root issue (different models can't batch). Single-process simplicity. No MPS daemon.

**Cons**:
- Training-side surgery. BC v10 incompatible — would need new BC training (Phase 1 v4).
- Locks ALL future snapshots into shared-backbone constraint.
- May reduce PFSP diversity (snapshots become more similar at the encoder).
- **Doesn't work for external opp adapters** (Metamon, foul_play models have their own architectures we don't control). We'd need a hybrid anyway → architectural complexity comparable to Option D.
- Quality risk on a path the long-term goal depends on.

**Verdict**: REJECTED. Training-side risk + external-adapter incompatibility > infra cleanliness gain.

### §2.3 Option C — NVIDIA Triton Inference Server

**Idea**: replace CIS with Triton. Triton handles multi-model concurrent execution + dynamic batching natively.

**Pros**: production-grade, NVIDIA-supported, already installed on prod pod.

**Cons**:
- **IPC latency regression risk**: mp.Pipe ~100µs vs gRPC ~1-5ms (even local). At 22 fires/sec × 6 processes, the gRPC delta compounds. Triton's shared-memory feature avoids gRPC but requires meaningful integration work.
- **Model export pain**: our `forward_ppo_sequence` has Python control flow (history conditional, padded temporal, BC anchor logic). TorchScript export fragile. ONNX export typically can't handle dynamic shapes/control flow.
- **Triton Python backend** avoids export but then we're wrapping our existing code in Triton's API — most of Triton's value disappears.
- **Batching mismatch**: Triton's dynamic batching is across-request-to-same-model. We have across-slot-different-models. We'd configure separate model instances per opp → not really using Triton strategically, just its server scaffolding.
- **Continuation vs rewrite**: we have CIS code we understand. Triton requires learning a new system.

**Verdict**: REJECTED. The "already installed" point removes the dependency argument but doesn't remove the integration, export, and IPC-latency arguments.

### §2.4 Option D — Multi-process per slot + CUDA MPS (RECOMMENDED)

**Idea**:
- 1 player-slot process (high arrival rate, big batches, current pattern works well here)
- N opp-slot processes (1 per pool snapshot, lower arrival per slot)
- All share single A100 via CUDA MPS daemon (concurrent kernel execution, eliminates context-switch overhead)
- Each process implements widx-batched send-back internally
- `train_rl.py` orchestrator routes worker inference requests to the right process based on `pool_slot_map` (which is already maintained)

**Why this is "correct"**:
- **Industry standard pattern** (Metamon process-per-agent; OpenAI Five separate Forward Pass GPUs; AlphaStar TPU-per-agent). Every successful multi-agent RL system uses some variant; we're the outlier with centralized-in-one-process.
- **Eliminates ALL the documented failure modes**: GIL serialization (each process has its own thread), N×mp.send-per-fire (each process serves its own workers, fewer sends per fire), per-slot starvation (each opp slot's process runs independently on GPU).
- **Future-proofs**: external opp adapters drop in naturally (each adapter = one process with its own arch). Multi-gen drops in naturally. Doubles/VGC drops in naturally. **The pattern doesn't need to be touched again.**
- **Doesn't touch the model architecture** — no BC retraining, no shared-backbone constraint forever.

**Tradeoffs**:
- More processes = more orchestration complexity
- MPS daemon as operational dependency
- Per-process CUDA context memory overhead (mitigated by MPS; budget at N=17 max is ~2-3GB extra, fine on 80GB A100)
- MPS isolation loss: one process's OOM/segfault can kill the daemon → need a process supervisor
- N>2 MPS at our specific workload not directly documented (Databricks tested N=2, GROMACS at higher N is compute-bound — neither maps 1:1 to our small transformer + sparse arrival regime)

---

## §3. Why CUDA MPS is the right enabling tech

MPS allows multiple CUDA processes to share a single GPU through one CUDA context, multiplexing their kernels on the hardware scheduler. Without MPS, multi-process on one GPU = serialized via time-slicing + each process has its own CUDA context (~1-2GB overhead each). With MPS = concurrent kernels possible when GPU has spare SMs, single shared context.

**Hardware limit**: 60 client CUDA contexts per device on Ampere/Volta+ (CUDA 13.1+). We need 6-17. Comfortably within.

**Sweet spot for MPS**:
- Small models (≤3B params) — we're 20M, 150× smaller than 3B sweet-spot upper bound
- Short context — battle turns, not LLM long-context
- GPU underutilization — our CIS-STATS data shows ~37% GPU util at pool=2 (fwd 17-21ms × ~22 fires/sec)
- Concurrent processes with independent kernel streams — our exact pattern

**Documented evidence**:
- Databricks (small LLM, N=2, H100): >50% throughput uplift for small models in sweet spot
- Pebble case study: similar findings, gains for sub-3B models
- GROMACS at N=16-32 (compute-bound): up to 3.5× throughput improvement
- NVIDIA Triton uses similar concurrent-execution pattern via CUDA streams within one process; reports 80-90% invocations/min increase

**Not documented at our specific N + workload**. That's what Phase A measures.

**Risks specific to MPS**:
1. **Isolation loss**: one client process OOM/segfault → MPS daemon may go down → all clients die. Mitigation: process supervisor + per-process memory monitoring + fast-restart on death.
2. **Daemon as failure dependency**: extra process to start/monitor/restart. Mitigation: bake into launch script with health check.
3. **N=6+ scaling not measured for our workload**: the load-bearing unknown. Phase A.3 measures it directly.

---

## §4. Staged plan with risk gates

### §4.1 Phase A — Derisk MPS at our N (single session, ~$5-15 pod)

**A.1 — MPS daemon smoke** (15 min, ~$0.50)
- Start `nvidia-cuda-mps-control` on prod
- Verify environment vars (`CUDA_MPS_PIPE_DIRECTORY`, `CUDA_MPS_LOG_DIRECTORY`) set correctly
- Single-process PyTorch test in MPS client mode → confirms hookup works

**A.2 — N=2 throughput bench** (30-60 min, ~$1-2)
- Bench `forward_spatial` (and full forward) at batch=4, 16 with N=2 concurrent processes
- Measure aggregate inf/s + per-process latency
- Compare to N=1 baseline
- **Gate**: if N=2 aggregate ≥ 1.5× N=1 throughput at batch=4, MPS works as Databricks suggests. If not, MPS gains don't materialize at our model size → diagnose before A.3.

**A.3 — N=6 scaling bench** (1-2 hours, ~$2-5)
- Bench at N=6 (1 player + 5 opp target)
- Measure aggregate inf/s, per-process latency, total GPU memory
- Vary batch sizes (1, 4, 16) to characterize where overlap helps most
- **Gate**: if N=6 aggregate ≥ 2× N=1 throughput at our typical batch sizes (3-24), MPS scaling is viable for Phase B. If gains diminish past N=2, our workload doesn't get MPS's full benefit at N=6 → re-evaluate.

**Fallback if A.3 fails**: Stage 1 alone (widx-batched send on current single-process CIS). Buys ~10-15% at pool=1. Phase 2 launches accepting pool>1 cost. Multi-gen wait for hardware change or revisit shared-backbone.

### §4.2 Phase B — Implement hybrid multi-process CIS (~2-3 weeks)

GATED by Phase A.3 passing.

**B.1**: Process supervisor scaffolding — spawn/monitor/restart N CIS processes from train_rl.py orchestrator. Each process is essentially the current `_cis_main` adapted to serve ONE slot instead of N.

**B.2**: Worker routing layer — `pool_slot_map` is already maintained by orchestrator. Workers route inference requests via the right process's pipe based on which opp they're currently playing.

**B.3**: Per-process widx-batched send-back (Option A folded in) — group `pending` by widx, one mp.send per worker per fire.

**B.4**: MPS launch wrapper in `launch_rl.sh` — start MPS daemon before train_rl.py, verify daemon health, set CUDA env vars.

**B.5**: Bit-equivalence gate — running multi-process CIS at pool=1 should produce identical trajectories to single-process CIS at pool=1. Gates correctness of the IPC + routing refactor.

**B.6**: Smoke validation — 100g/4w/conc=25 single-iter at pool=1. Confirms end-to-end works.

### §4.3 Phase C — Prod-scale validation (~$20-30 pod)

10-iter prod run at 15w/1600g/conc=200, snapshot-interval=2 to walk pool 1→5.

**Gate**: collect ≤600s at every pool size (1-5).
- If met → Phase 2 launch-ready. Merge to master.
- If not at one pool only → diagnose that specific regime.
- If not at multiple → Phase B implementation has a bug or MPS overlap is worse than Phase A predicted. Diagnose.

### §4.4 Phase D — Operational hardening (~1 session)

- Process supervisor for OOM isolation (kill + restart dead processes, alert if daemon dies)
- Launch/shutdown runbook for MPS
- Per-process memory + throughput monitoring
- Memory + tracker + boot-protocol updates
- Handover doc for next-session-me

---

## §5. Operational concerns

### §5.1 MPS daemon failure modes

**One process OOM kills daemon**: documented risk. Mitigation: per-process memory budgeting (hard cap via `torch.cuda.set_per_process_memory_fraction`), monitor RSS, restart daemon + all clients on detection.

**Daemon down on launch**: pre-flight check in `launch_rl.sh` — verify `nvidia-cuda-mps-control` running before spawning train_rl.py. If not, start daemon, wait for ready signal, then proceed.

**Daemon down mid-run**: detect via per-process health pings. On detection: stop training cleanly, alert, restart full stack.

### §5.2 GPU memory budgeting at N=17 (max_pool_size + 1)

Per process:
- CUDA context (with MPS): ~100-200MB
- Model weights bf16 (20M params): ~40MB
- Activation buffers during forward (batch=24): ~100-200MB
- PyTorch caching allocator overhead: ~200-500MB

Per-process total: ~500MB-1GB. At N=17: ~8-17GB. Fine on 80GB A100 even with player slot training using full update memory.

### §5.3 Worker→process routing

`pool_slot_map` is already maintained in `_cis_main_multi` (line 1804 + 1942 + 2450 reads). Currently it maps `opp_path → slot_idx`. We extend this to map `opp_path → process_pid` (or pipe handle). Worker code already knows which opp it's playing (from `pool_slot_map` at iter start); it routes to the right pipe.

Player slot has a fixed pipe. Opp slots have one pipe per opp process.

### §5.4 Process lifecycle

Spawn at iter start (for new opps), reuse across iters (for unchanged opps), terminate at shutdown. Each opp process loads its checkpoint via the existing `reload` cmd path (line 892-957). Player process loaded once at startup, reloaded at warmup-end if needed.

### §5.5 Bit-equivalence gate at pool=1

At pool=1, multi-process CIS and single-process CIS should produce identical trajectories. The only difference is dispatch routing (which process handles the request). Forward computation is identical, model weights identical, batching is identical (single opp slot, single player slot — no overlap).

Test: run the same iter under both configs, compare action distributions per turn. Numerically equivalent → IPC refactor is correct.

---

## §6. What this proposal explicitly does NOT do

- **Cross-slot batching via shared backbone** — REJECTED §2.2. Training-side risk, external-adapter incompatibility.
- **Triton replacement** — REJECTED §2.3. IPC regression risk, export pain, no strategic benefit.
- **Standalone Stage 1 fix** — REJECTED §2.1. Bounded gain, doesn't address architectural saturation. Folded into Phase B.
- **Pool size cap** — user explicitly rejected.
- **Multi-GPU** — budget rejected.
- **Distillation of opp models** — training-side change, out of Phase 2 prep scope.
- **Pure multi-process for ALL slots including player** — breaks player-slot amortization that works well today. Hybrid is more conservative.

---

## §7. Cross-references

**Raw data sources** (verified S66):
- `pokemon-ai-starter/pokemon-ai/src/data/s65_artifacts/stage_a_test.log` — Stage A 10-iter prod run, CIS-STATS per-slot timing
- `pokemon-ai-starter/pokemon-ai/src/data/s65_artifacts/battle_server_diag/runA_15w_retest.log` — baseline 15w retest CIS-STATS
- `pokemon-ai-starter/pokemon-ai/src/data/s65_artifacts/battle_server_diag/runB_warm_bs.log`, `runC_cold_bs.log` — battle server degradation A/B/C
- `pokemon-ai-starter/pokemon-ai/src/data/s65_artifacts/bench_cpu.log` — CPU inference benchmark
- `pokemon-ai-starter/pokemon-ai/src/data/s65_artifacts/cis_adaptive_smoke.log` — adaptive batching smoke

**Code sources cited**:
- `pokemon-ai-starter/pokemon-ai/src/mp_centralized_collect.py` — CIS implementation
  - Lines 659, 801: `_t_fire_start`, `_t_fire_end` timing (mean_fire scope)
  - Lines 667-668: stack_numpy_batches + numpy_dict_to_torch
  - Lines 770-784: per-request mp.send loop (PRIMARY OVERHEAD)
  - Lines 681-685, 690-744: forward path (forward_spatial + temporal + heads for history-bearing requests)
  - Lines 535-544, 678-680: shared low-pri CUDA stream
  - Lines 892-957: slot reload (snapshot loading per slot)
  - Lines 1804, 1942, 2450: pool_slot_map maintenance

**Memory files**:
- `memory/project_s65_arch_impasse.md` — S65 wrap, original problem statement
- `memory/project_optimization_tracker.md` — overall optimization arc
- `memory/feedback_asyncio_gather_at_scale.md` — prior learning on process-per-task pattern
- `memory/reference_cloud_pods_usage.md` — pod connection details for Phase A

**External references** (Phase 4):
- [NVIDIA MPS docs](https://docs.nvidia.com/deploy/mps/latest/index.html) — 60-client limit, sweet-spot characterization
- [Databricks: Scaling Small LLMs with NVIDIA MPS](https://www.databricks.com/blog/scaling-small-llms-nvidia-mps) — N=2 H100 benchmark
- [NVIDIA Triton concurrent execution](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_execution.html) — alternative pattern, rejected §2.3
- [OpenAI Five paper](https://arxiv.org/abs/1912.06680) — multi-agent inference at scale (multi-GPU)
- [AlphaStar DeepMind blog](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/) — TPU-per-agent league
- Metamon self_play (in-repo at `metamon_ref/metamon/rl/self_play/`) — subprocess-per-agent

---

## §8. Open questions for next session

1. **MPS overlap efficiency at N=6 with our specific workload** — Phase A measures.
2. **Process supervisor implementation** — what's the failure-detection / restart strategy? Pending Phase B.1 design.
3. **Phase A budget approval** — estimated $5-15 pod time for full Phase A. Confirm with user before pod time burns.
4. **External adapter integration** — does the multi-process pattern need any special handling for Metamon / foul_play adapters? Defer until Phase B.2 routing design.

---

## §9. Decision log (for next-session-me)

| Decision | Why | Could revisit if |
|---|---|---|
| Multi-process per slot, not centralized | Industry standard, eliminates GIL+mp.send overhead | MPS unavailable on future hardware |
| CUDA MPS, not raw multi-process | Eliminates context-switch overhead, allows concurrent kernels | MPS daemon proves operationally fragile |
| Custom CIS refactor, not Triton | gRPC latency, export pain, strategic value low | We need multi-model versioning / persistence |
| Hybrid (single player process + multi opp processes) | Player slot's centralized pattern works well at high arrival | Player slot needs scaling too |
| Phase A first (derisk MPS) | The load-bearing unknown is overlap efficiency | A.3 fails → fallback to widx-batched single-process |
| Don't propose Stage 1 standalone | Bounded gain, doesn't solve pool growth | Phase A.3 fails AND shared-backbone also rejected |
| Don't propose shared backbone | Training-side risk, external-adapter incompat | MPS proves unworkable AND we're willing to retrain BC |

End of design memo.
