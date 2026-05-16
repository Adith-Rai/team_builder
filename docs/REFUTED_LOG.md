# REFUTED LOG — Don't retry, or approach differently

**Purpose**: durable record of what has been explored and **does NOT work** at our scale/arch, with evidence + rationale + circumstances-that-would-warrant-revisit. Read this before proposing or re-exploring an "optimization" or "fix" — many things on this list looked promising on paper or worked at smoke scale, then failed at prod.

**How to use this doc**:
- Browsing for an old idea? Ctrl-F by technique name first.
- Adding a new entry? Cite the session and the evidence. Don't just say "didn't work."
- A REFUTED item is rebuttable only with NEW evidence (different scale, different arch, different hardware). "We should try again because it's been a while" is not new evidence.

**Last updated**: S64 post-Phase-B `--pipeline` A/B, 2026-05-16.

**Authoritative status (where these decisions live)**:
- `memory/project_optimization_tracker.md` §2 — optimization-arc refutations (with current session links)
- `memory/project_session_boot_protocol.md` §5 — long-term "Don't reopen" list
- This doc — durable consolidation of both, with evidence

---

## Categories

- **REFUTED** — Data shows it doesn't work at our scale/arch. Don't retry without new evidence.
- **DEMOTED** — Not refuted, but lower priority than alternatives. Could revisit if alternative paths are exhausted.
- **DEFERRED** — Not refuted, but blocked on circumstance change (e.g., hardware upgrade, scope expansion).

---

## REFUTED

### Performance / training-loop optimizations

#### 0. `--pipeline` at current scale (single A100 + CIS + conc=200 + 8 battle servers + packed mb=64)
- **Refuted**: S64 (2026-05-16), 3-iter prod A/B at `data/s64_artifacts/2b_validation/`
- **Evidence**: Same config (packed mb=64, BC v10 init, 1600g/200conc, fresh battle servers, 3 iters), only difference is `--pipeline` on vs off.
  - **Total run wall: 2880s with --pipeline → 2191s without = -24% savings.**
  - Iter 1 collect: 674s on → 559s off (-17% battle-server contention removed)
  - Iter 1 update: **254s on → 156s off (-39% GPU contention removed)**
  - Plus drain-pending-BG wait: +288s at iter 1 end, +577s at iter 2 end
- **Three contention vectors**:
  1. **GPU contention (CONCRETE)**: BG CIS inference and update share the GPU. CIS Phase 4.3b's low-priority CUDA stream design was supposed to prevent this — measurement shows the streams don't fully isolate at our scale (root cause: memory bandwidth / scheduler / allocator not separately profiled).
  2. **Battle-server contention (STRONG INFERENCE)**: BG workers and foreground collect workers share 8 battle servers.
  3. **Drain wait (CONCRETE from FLOW timestamps)**: must drain current BG before launching next BG; drain runs after update, blocking iter exit.
- **Historical context**: pipeline was a 21% win at Session 32 (conc=10, ~14M model, RTX 3060). Session 50 found it broken at conc=200 + 20M transformer. CIS Phase 4.3b (S54) re-enabled it on the CONJECTURE that low-priority CUDA streams would fix the contention; **end-to-end validation deferred at S54 until now**.
- **Could revisit if**: multi-GPU configuration available (BG inference on GPU2, training on GPU1) — would eliminate vector #1, potentially restoring the 10-22% win. At single-A100 + current concurrency/server count, don't reopen.
- **Memory source**: `project_optimization_tracker.md` §1.2, `project_s64_phase_b_results.md` §3.7

#### 1. `--compile` (whole-function `torch.compile` on `forward_ppo_sequence`) at prod scale
- **Refuted**: S62 (`869ef285`)
- **Evidence**: 4 prod-scale data points (1600g/8w/conc=200, --tier3-minibatch-size 16). Per-step update cost: **27 μs with --compile, 25 μs without** = **8% SLOWER** with compile. Recompile loop hypothesis tested with `TORCH_LOGS=recompiles` — only 2 recompile events total over 34-min update, both early. NOT a recompile churn issue; compile dispatch overhead exceeds kernel-fusion savings at prod scale.
- **The S60 9× speedup was a smoke-only (64g/2w) anomaly**: at smoke, GPU work per chunk is small relative to eager Python overhead, so compile fusion wins big. At prod, GPU work scales but Python dispatch stays constant — eager already amortizes well.
- **Why not retry**: Forward graph break at `model_transformer.py:2350,2378` means only the resume portion gets compiled. Eager forward+backward is fast enough at prod scale.
- **Could revisit if**: minibatch size drops far below 16, OR `forward_ppo_sequence` is rewritten without graph breaks, OR hardware change makes per-call overhead matter more (e.g., much smaller GPU).
- **Memory source**: `project_s62_fix2_prod_validation.md`

#### 2. Vectorize `collate_episodes` in Python (Fix #3)
- **Refuted**: S60 (microbench on `perf/vectorize-collate-episodes` branch, NOT merged)
- **Evidence**: Two variants implemented + bit-equivalence tested. V2 (pre-alloc + slice-assign) ran within noise (0.98-1.10× of baseline V1). V3 (advanced-index scatter) was **strictly SLOWER** (0.68-0.81×). Diagnostic scripts kept: `scripts/diag/test_collate_episodes_vec_bitexact.py`, `scripts/diag/bench_collate_variants.py`.
- **Why not retry**: The S59 "55s/iter in collate_episodes" was viztracer-INCLUSIVE time; most of it lives in C ops (`torch.cat`, `torch.as_tensor`) that Python-level vectorization cannot speed up. Pure Python overhead is a small fraction of the 55s; V1 is already near-optimal for that fraction.
- **Could revisit if**: A future profile shows time in a SPECIFIC C op (e.g., `torch.as_tensor` on list-of-lists for action_masks) that's reachable by surgical fix. Don't redesign the whole function.
- **Superseded by**: S64 sequence-packing approach (`collate_episodes_packed`) — eliminates the padding C ops entirely instead of trying to vectorize around them.
- **Memory source**: `project_s60_fix3_refuted.md`

#### 3. Liger-Kernel drop-in (RMSNorm / SwiGLU / RoPE fused kernels)
- **Refuted**: S62 (profile-driven, no impl attempted)
- **Evidence**: S62 update profile inspection — our model uses `nn.GELU` + `nn.LayerNorm`, NOT SwiGLU + RMSNorm. Liger-Kernel's drop-in replacement targets the latter; does not apply to our arch.
- **Why not retry**: Architectural mismatch. Would require changing model to SwiGLU/RMSNorm first, which is a separate (non-trivial) decision not motivated by data.
- **Could revisit if**: We change to SwiGLU/RMSNorm for other reasons (e.g., post-multi-gen arch refresh).
- **Memory source**: `project_s62_update_profile_findings.md` §4.4

#### 4. `--tier3-minibatch-size 16 → 48`
- **Refuted**: user's prior test (pre-S62)
- **Evidence**: Wall-time was NULL — bigger minibatch alone doesn't help when GPU is saturated.
- **Why not retry**: At prod scale we're orchestration-bound, not minibatch-size-bound.
- **Could revisit if**: A future optimization (e.g., CUDA Graphs or sequence packing) makes the GPU non-saturated again.

#### 5. Multi-GPU training for the 20M-param model
- **Refuted**: S62 research (no impl)
- **Evidence**: NCCL allreduce ~5-10ms per step vs per-step compute ~1.5ms → communication would dominate. Single A100 80GB is the correct hardware sizing for a 20M model.
- **Why not retry**: Communication-bound for parameter counts << 1B. Multi-GPU is for parameter scale, not data scale.
- **Could revisit if**: Model parameter count grows >1B (not on roadmap).

#### 6. FlashAttention-3 / FP8 / MXFP4
- **Refuted**: hardware constraint (Ampere A100, not Hopper)
- **Evidence**: FA3, FP8, and MXFP4 all require Hopper-class GPUs. We are on Ampere A100 SXM 80GB.
- **Could revisit if**: Hardware upgrade to H100/H200.

#### 7. Standalone `grad_accum > 1`
- **Refuted**: Session 31
- **Evidence**: Silent quality regression observed (specifics in S31 logs).
- **Why not retry**: Tier 3 transition-level minibatching (`--tier3-minibatch-size N`) provides the memory-bound benefit safely. Standalone grad_accum is the wrong abstraction for our setup.

### CIS / collection infrastructure

#### 8. `--opponent-device cpu` (`--opp-cpu`)
- **Refuted**: S50
- **Evidence**: Broke at production scale. CPU 20M-param forward × conc=200 exceeded WS keepalive limits.
- **Why not retry**: We need GPU forward at scale; CPU can't keep up.
- **Source**: `docs/PPO_CLOUD_COOKBOOK.md §3h`

#### 9. REBIND_WORKER ctrl protocol for CIS worker respawn
- **Refuted**: Phase 4.6, S54
- **Evidence**: Connection-passing-via-`mp.Pipe.send` hangs in RunPod environment. Replaced with Option B full-reset.
- **Why not retry**: RunPod-specific networking quirk that doesn't go away. Option B (full reset) is simpler and correct.
- **Memory source**: `project_cis_4_6_design.md`

#### 10. Dual-slot CIS (slot 0 = player, slot 1 = active opponent)
- **Refuted**: S53
- **Evidence**: Doesn't compose with `asyncio.gather` opponent concurrency. Replaced with pool-mirror multi-slot.
- **Why not retry**: Architectural mismatch with how workers handle concurrent opponents.
- **Memory source**: `project_cis_4_3_design.md`

#### 11. `forkserver` mp context (instead of `spawn`)
- **Refuted**: S50
- **Evidence**: SemLock race at N≥4 workers (CPython 3.11 bug).
- **Why not retry**: Upstream Python bug; not within our control to fix.
- **Workaround**: Use `spawn` (standard, slightly slower process start but no SemLock issue).

#### 12. Torch tensor IPC across `mp.Pipe`
- **Refuted**: S50
- **Evidence**: mmap explosion at production scale.
- **Why not retry**: Torch tensor pickling for IPC creates per-tensor shared-memory mappings that don't get released; at scale this exhausts mmap.
- **Workaround**: Numpy-only IPC across `mp.Pipe`.

#### 13. `asyncio.gather` of N heterogeneous coroutines in one event loop (e.g., 5 opp coros per worker)
- **Refuted**: S58
- **Evidence**: Small batches starve under shared async scheduling, hit per-coro `wait_for` timeouts → `battle.won = None` ghost ties. Iter 40-49 of Phase 1 v3 saw this in prod.
- **Why not retry**: Industry standard (Metamon, OpenAI Five, AlphaStar, ps-ppo) is process-per-task for a reason — shared-loop scheduling doesn't fairly serve heterogeneous concurrent work.
- **Fix shipped**: one-opp-per-worker (commit `548033e8`) + defensive `cmds_sent` filter (`641de8b8`) + deadline 8.0× (`d092c4c5`).
- **Applies to**: ANY future "fan out parallel work in one process" instinct.
- **Memory source**: `feedback_asyncio_gather_at_scale.md`, `project_s58_session_narrative.md`

#### 14. Per-worker trajectory checkpointing on CIS failure
- **Refuted**: Q3 cost analysis (S54)
- **Evidence**: <$3/run savings vs half-day dev time. Empirical CIS failure rate <5/run.
- **Why not retry**: Cost-benefit doesn't pencil out unless failure rate exceeds 5/run.
- **Could revisit if**: Empirical failure rate climbs above the threshold.
- **Memory source**: `feedback_phase46_traj_checkpoint_rejected.md`

### Model / training-correctness

#### 15. `--lr 3e-5` on transformer architecture
- **Refuted**: S50, 4-point ablation
- **Evidence**: Caused regression on TransformerBattlePolicy. New arch has 3.3× lower gradient sensitivity vs the legacy PokeTransformer; needs lower lr.
- **Why not retry**: Hard-won settled decision. `--lr 1e-5` is locked.
- **Could revisit if**: Tier 3's effective batch-size change requires re-ablation. Do it as part of C6 A/B, not as a standalone change.
- **Source**: `docs/PHASE1_DIAGNOSIS_REPORT.md`

#### 16. Mixing the 16 metamon-competitive eval teams into training
- **Refuted**: project design intent (S54 reinforced)
- **Evidence**: Data leakage — model memorizes specific compositions, eval scores inflate artificially, eval signal is lost.
- **Why not retry**: Eval set integrity is non-negotiable.
- **Fix forward**: Phase 2 sources 100-150 SEPARATE real elite teams (Smogon archive, Showdown ladder replays, tournament posts). Copy verbatim, never generate or estimate.

#### 17. Perm-at-eval (adding slot permutation at evaluation time)
- **Refuted**: S60 (user correction)
- **Evidence**: Perm exists for TRUE Pokemon learning — invariance to slot order. Adding it at eval would MASK invariance bugs by making eval easier, not improve the model. The "+7pp smart_avg" framing in next-prompt §S58 is measurement-fudging.
- **Why not retry**: Other Pokemon PPO codebases (Metamon, ps-ppo) keep eval canonical because that's the deployment distribution. The fix for "model isn't fully invariant" is in TRAINING (more iters, better perm coverage), not in eval.
- **Memory source**: `feedback_perm_at_eval_wrong.md`

#### 18. Slot permutation REMOVAL from training (i.e., training on canonical slot order)
- **Refuted**: project design intent (S50 postmortem reinforced)
- **Evidence**: Procedural teams + slot perm are LOAD-BEARING for general Pokemon learning vs gen-9-OU memorization.
- **Why not retry**: The S50 critique was about train/eval DISTRIBUTION mismatch (fix = mix real elite teams in training), not about perm-or-procedural-being-wrong.

### Torch / compile internals (torch 2.2.x specific)

#### 19. `F.one_hot` inside `torch.compile` dynamic-shape region
- **Refuted**: S56 (`project_tier3_c5_design.md` Limit 1)
- **Evidence**: Fails on torch 2.2.x with `Cannot call numel() on tensor with symbolic sizes/strides`.
- **Why not retry**: torch 2.2.x dynamo limitation.
- **Workaround**: Broadcast-equality `(arange == idx.unsqueeze(-1)).to(dtype)` — bit-equivalent and dynamic-safe.
- **Could revisit if**: Torch upgrade to 2.4+ (where this is fixed). S64 Phase 1 verified torch 2.5.1 venv-isolation; bears re-testing at Phase B.

#### 20. `nn.utils.clip_grad_norm_(model.parameters(), ...)` inside `torch.compile`
- **Refuted**: S56 (Limit 2)
- **Evidence**: Graph-breaks on torch 2.2.x with `ListIteratorVariable() has no type` at the internal isinstance check.
- **Workaround**: Move clip_grad_norm to eager wrapper outside the compile region.

#### 21. Single-graph optimizer.step on safety-mask-zeroed loss
- **Refuted**: S56 (Limit 3)
- **Evidence**: AdamW's `weight_decay` drifts params even with zero gradients (`param ← param - lr·wd·param`).
- **Workaround**: Gate `optimizer.step()` on `step_mask.item() > 0.5` in eager wrapper.

#### 22. `mode="reduce-overhead"` for full forward compile
- **Refuted**: S51
- **Evidence**: cudagraph aliasing breaks on multi-module patterns + recompile churn.
- **Workaround**: `mode="default"` + `dynamic=True`.

#### 23. Path 1 single-method compile (`forward_spatial` only)
- **Refuted**: S51
- **Evidence**: Covers only ~40% of forward.
- **Workaround**: Path 2 (per-submodule, all 5 modules).

#### 24. `--compile` on legacy `PokeTransformer` (not `TransformerBattlePolicy`)
- **Refuted**: S51
- **Evidence**: Different forward shape; not supported.
- **Why not retry**: Architectural — legacy arch is on deprecation path anyway.

### Tier 3 / forward_ppo_sequence (S57 era)

#### 25. Tier 3 `ppo_update_batched` mega-batching all episodes into one forward
- **Refuted**: S57 OOM at production scale
- **Evidence**: OOMs at 200+ games on A100 80GB (activation memory ~36 GB per FF layer × 6).
- **Workaround**: `--tier3-minibatch-size N` (chunks of N episodes, accumulate grads, ONE optimizer.step per epoch). Shipped S57 (`6821a907`).
- **Required at production scale**: Phase 2 launch with --tier3 REQUIRES `--tier3-minibatch-size` set. Suggested: 16 for memory headroom.

#### 26. `forward_ppo_sequence` with `L_max > cfg.temporal_context`
- **Refuted**: S57 bug fix (`d1b101bb`)
- **Evidence**: `TemporalTransformer.forward` truncates summaries internally to `temporal_context` (default 200), but outer indexing `temporal_ctx_grid[b_idx, t_idx]` uses `t_idx` values up to `L_max-1` → CUDA "index out of bounds" assert.
- **Workaround**: Pre-truncate via `collate_episodes(L_max=cfg.temporal_context, tail=True)`. All five production callsites use this.

#### 27. `torch.inference_mode()` for BC reference forward
- **Refuted**: S57
- **Evidence**: Interacts badly with autocast bf16 + eager `forward_ppo_sequence`. Causes CUDA "index out of bounds" + `CUBLAS_STATUS_EXECUTION_FAILED` on downstream GEMM.
- **Workaround**: `torch.no_grad()` (standard frozen-teacher pattern, ~5 sec total perf cost over 7-hour run).
- **Memory source**: `project_bc_anchor_design.md`

#### 28. Storing BC reference as `model._bc_ref` attribute
- **Refuted**: S57
- **Evidence**: PyTorch picks up the BC ref's parameters in `model.state_dict()` → mp_disk worker respawn fails with `Unexpected key(s) in state_dict: "_bc_ref.tokenizer..."`.
- **Workaround**: Store as a LOCAL variable in `main()`, pass directly to `ppo_update` / `ppo_update_batched`.

### Diagnostic / instrumentation

#### 29. `CUDA_LAUNCH_BLOCKING=1` for diagnosing PPO update crashes
- **Refuted**: S57
- **Evidence**: Makes ALL CUDA calls synchronous, INCLUDING battle-server inference during collect. Slowed collect to a crawl — timed out at 20 min still in iter 0 collect.
- **Why not retry**: Too broad-scope for crash diagnosis.
- **Workaround**: Targeted instrumentation in the suspect code path. Or set the env var only for a single-iter `--n-iters 1` run.

#### 30. Removing `CISClientHandle` thread-safety (the per-handle Lock + async-dispatch)
- **Refuted**: design
- **Evidence**: Both async-dispatch + sync lock paths are load-bearing. `mp.Pipe.send` is not byte-atomic; removing the safety causes intermittent corruption.
- **Memory source**: `feedback_cis_thread_safety.md`

### Architecture / scope

#### 31. VGC-Bench as Tier 3 reference (their batching approach)
- **Refuted**: S37 study
- **Evidence**: They are stateless per-turn (no temporal stack); their batching freedom doesn't transfer.
- **Why not retry**: Architectural mismatch.
- **Workaround**: Use Metamon's `metamon_to_amago.py` as the Tier 3 reference instead.

#### 32. Explicit per-gen feature gating in code (e.g., "if gen ≤ 4, skip Mega Evolution")
- **Refuted**: S51 D2
- **Evidence**: poke-env API already returns False/0 for unavailable mechanics per gen.
- **Why not retry**: Belt-and-suspenders gating is dead weight.

#### 33. Drop temporal stack as PRIORITY in the optimization arc
- **Refuted**: S64 step-back (PRIORITY-refuted, not technique-refuted)
- **Evidence**: S62 + S64 prod profiles show temporal stack consumes <8% of update wall (bounded by total CUDA kernel time = 8% of update). Dominant 24% CPU cat waste is in DATA PREP (`collate_episodes`), not in the temporal stack. Top single cat signature (38.6% of cat time) lives at `_stack_batch_dim` (ppo.py:301), inside `collate_episodes`.
- **Why not retry as priority**: Cost (4-8 sessions, BC v10 compat risk, Phase 2 timeline risk) >> reachable wall.
- **Could revisit if**: After Phase 2 ships, as a quality-not-speed exercise. NOT as part of this optimization arc.
- **Memory source**: `project_s64_arch_revisit_profile_findings.md` §5.2

#### 34. Network Volume migration before multi-gen launch
- **Refuted**: user decision
- **Evidence**: Container Disk OK for Phase 1/2 since user monitors and won't stop pod without informing.
- **Why not retry**: Risk accepted.
- **Could revisit if**: Approaching multi-gen launch (5-7 wk / $400-600 run) where container disk loss would be much more expensive.

#### 35. Gigantamax / Doubles / Triples in V1
- **Refuted**: scope decision
- **Why not retry**: Out of scope for V1 (gen 9 OU singles). Post-multi-gen format work.

---

## DEMOTED — Not refuted, lower priority than alternatives

### 1. Regional `torch.compile` as primary win
- **Status**: DEMOTED via S62 profile
- **Why demoted**: We are not compute-bound on individual kernels (FA engaged via SDPA ✓, bf16 fused GEMM engaged ✓). CUDA Graphs covers the dispatch-overhead ground more effectively.
- **Could revisit if**: CUDA Graphs lands and residual compile-able regions remain.

### 2. Token-bucket dynamic batching as a separate technique
- **Status**: DEMOTED via S62 profile
- **Why demoted**: Same root cause (padding waste) as sequence packing (S64 Phase A in progress).
- **Could revisit if**: Sequence packing leaves residual padding waste.

### 3. Fix #1 Option A (worker-side aggregation in CIS)
- **Status**: SKIPPED in S62 pending demonstrated need
- **Why demoted**: Option B already delivered -27% collect at prod; could recover ~13% additional gap between Phase C -41% and prod -27%, but HIGH risk.
- **Could revisit if**: Phase 2 needs more collect-side headroom.

### 4. Fix #1 Option C (shared-mem ring buffer)
- **Status**: SKIPPED in S62 pending demonstrated need
- **Why demoted**: High risk for marginal additional gain.
- **Could revisit if**: Both A and B exhausted.

### 5. Secondary fixes from PROFILE_BOTTLENECKS_REPORT §5 (CIS forward compile, tokenizer, CUDA graphs as compile)
- **Status**: DEMOTED pending data
- **Could revisit if**: Current optimization arc completes and update wall is still binding.

---

## DEFERRED — Blocked on circumstance change

### 1. FlashAttention-3, FP8, MXFP4
- **Blocker**: Hopper-class GPU (we are on Ampere A100)
- **Unblocks**: Hardware upgrade

### 2. Multi-gen-specific infra (D4-D6, BC v11, E1-E3)
- **Blocker**: Phase 2 must confirm gen-9-OU is elite first
- **Unblocks**: Phase 2 post-eval shows we hit elite

### 3. Doubles / Triples / VGC / AG / in-game format support
- **Blocker**: Multi-gen launch must complete first
- **Unblocks**: Multi-gen launch finishes; format work then proceeds independently

### 4. Wave training, fictitious play, league of opponents
- **Blocker**: Phase 2 baseline established first
- **Unblocks**: Phase 2 + multi-gen establish baselines; advanced training schemes are tuning-on-top

---

## How to add new entries

When refuting a technique:
1. Add to the appropriate category section (REFUTED / DEMOTED / DEFERRED)
2. Include: name, refuted-session, evidence (data with numbers, not vibes), why-not-retry, circumstances-that-would-warrant-revisit
3. Link to the memory file with full evidence (these are summaries)
4. Update `last updated` line at top of file
5. Update `memory/project_optimization_tracker.md` §2 if optimization-arc-relevant
6. Commit to master with `docs(refuted): <technique>` style message

When promoting a DEMOTED item back to active:
1. Move from DEMOTED to a new "TRIED AGAIN" or active section
2. State the new evidence that justifies the revisit
3. Document the outcome (re-refuted or shipped)

When un-deferring (circumstance has changed):
1. Same as promoting from DEMOTED
2. Note the circumstance change (e.g., "hardware upgraded to H100, FA3 now applicable")

---

## Cross-references

- **`memory/project_optimization_tracker.md` §2** — optimization-arc refutations log (memory, updated session-by-session)
- **`memory/project_session_boot_protocol.md` §5** — long-term "Don't reopen" list (memory, project-eternal layer)
- **`docs/PROFILE_BOTTLENECKS_REPORT.md`** — bottleneck data that motivated many of these decisions
- **`docs/PHASE1_POSTMORTEM.md`** — `--lr 3e-5` regression root cause analysis
- **`docs/PHASE1_DIAGNOSIS_REPORT.md`** — lr ablation matrix
- **`docs/CENTRALIZED_INFERENCE_DESIGN.md`** — CIS architecture (REBIND_WORKER, dual-slot, async-dispatch context)
- **`docs/SESSION_BOOT_PROTOCOL.md`** — standing orders + dead-paths section
