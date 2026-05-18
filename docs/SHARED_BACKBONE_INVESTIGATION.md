# Shared Backbone CIS — Investigation State (Track B)

**Branch**: `perf/multi-process-cis-mps` (kept on this branch for continuity with Phase A bench artifacts; can fork to `perf/shared-backbone` when committing to impl)
**Authored**: S66 (2026-05-18) — wrap of session
**Status**: INVESTIGATION IN FLIGHT — S1 partial, S2-S5 pending for next session
**Sibling**: `docs/WAVE_BASED_CIS_INVESTIGATION.md` (Track A) — discovered end of S66, lower-risk simpler path. Next session should evaluate BOTH tracks; Track A may suffice. This memo (Track B) remains the longer-term architecturally-pure path.
**Supersedes**: `docs/MULTI_PROCESS_CIS_DESIGN.md` (which has its own STATUS UPDATE at top)

---

## §0. Why this exists

S66 measured that multi-process CIS via CUDA MPS doesn't scale past N=2 for our workload (refuted at N=6 with 0.74-0.90× throughput). User constraint: model quality cannot be sacrificed for speed. After exploring CPU (insufficient throughput) and Triton (IPC + export issues), the only remaining architecture that hits both quality + pool-scaling goals is **shared backbone**: tokenizer + spatial encoder shared across all snapshots, temporal stack + heads specialized per snapshot.

This memo anchors the investigation. See `memory/project_s66_collect_arch_findings.md` for the data + decisions that got us here.

## §1. The proposed architecture (Option C)

**Backbone (frozen during PPO, single instance, shared across all snapshots)**:
- `tokenizer` (1.0M params)
- `spatial` SpatialTransformer (4.7M params)
- `summary_to_temporal` Linear (0.5M params)
- Subtotal: ~6.2M params (31% of total model)

**Specialized (per-snapshot, PPO-trained as today)**:
- `temporal` TemporalTransformer (12.7M params)
- `action_head` (0.5M)
- `value_head` (0.4M)
- `switch_encoder` (0.07M)
- Subtotal: ~13.7M params (69% of total model)

**Inference flow at CIS**:
1. Mega-batch all incoming requests (across all opp slots + player slot)
2. Run BACKBONE once → produces (spatial_out, summary) per request
3. Route per request to its snapshot's SPECIALIZED forward → action_logits + value
4. Send results back to workers

**Why this works** (predicted, not yet measured):
- Tokenizer is 64% of forward_spatial wall time AND near-fixed across batch (7.17ms at B=4 vs 7.31ms at B=16). Running it once on a mega-batch instead of N times per-slot saves big.
- Spatial similar (3.80ms vs 3.82ms).
- At pool=8 with mega-batch: ~11ms backbone + 8 × ~7ms specialized = ~67ms vs current ~8 × 18ms = 144ms → ~54% reduction.

**Why this might fail quality**:
- Freezing the spatial encoder during PPO limits what PPO can adapt
- BC v10's spatial features must be sufficient for the policy task
- The S57 erosion data (`project_phase1_v3_diagnosis.md`) suggests PPO drift in the spatial layer is actually BAD — freezing might HELP quality, not hurt. But this is plausible, not measured.

## §2. Required investigation steps (next session)

Per tasks #16-21. In order:

**S2 — Training-side mechanism choice**
- Read `train_rl.py` + `ppo.py` to find where parameters are aggregated for optimizer.step
- Identify exactly how to freeze the spatial encoder cleanly: set `param.requires_grad = False` on `model.tokenizer`, `model.spatial`, `model.summary_to_temporal`, OR use param-group exclusion in the optimizer
- Compare to LoRA approach: requires adding LoRA layers, more invasive
- Verify the BC anchor mechanism is compatible with frozen-spatial (BC anchor computes reference logits via full model forward — should still work since BC v10 is the reference and matches the frozen spatial weights)
- Output: concrete code change spec for freeze-spatial

**S3 — External opp adapter compatibility check**
- Read `external_adapters.py` + `external_opponent_manager.py`
- Read `metamon_accept_serve.py` + `foul_play_accept_serve.py`
- Confirm: external opps use SEPARATE inference paths (NOT CIS). They have their own model architectures.
- Document the integration boundary. Shared-backbone CIS only affects CIS path; external opps are unaffected.
- Output: confirmation + brief integration doc

**S4 — Precedent survey (brief, web)**
- Does VGC-Bench use shared backbones for self-play snapshots?
- Does any RL framework offer multi-snapshot inference with shared layers? (Look at RLlib, AMago, etc.)
- LLM serving: multi-LoRA on a frozen base is a mature pattern — relevant analogy
- Output: brief notes; don't go deep unless something surprising emerges

**S5 — Quality validation experiment design**
- Two PPO runs: A (current full-finetune) vs B (frozen-spatial)
- Same init (BC v10), same hyperparameters, same eval cadence
- Run length: 20-30 iters (matches BC anchor stability check from S57)
- Tracking: smart_avg, win rate, kl, bc_kl, eval registry results
- Gate: variant B must match A within noise (define "within noise" as ±X% on smart_avg — calibrate from S58/S64 baseline variance)
- Cost estimate: ~$10-30 pod time per variant
- **CRITICAL**: this is the GATE. Don't ship architecture without passing.

**S6 — Implementation design (after S2-S5 done + validation passes)**
- New file: `mp_centralized_collect_v2.py` OR major refactor of existing
- Routing: pool_slot_map already maintained — extend to route per-request to specialized weights
- Storage: snapshots store only specialized weights (not full model) — saves disk + load time
- Per-slot reload becomes "load specialized only"
- mp.send refactor: widx-batched (Stage 1 fix folded in)

## §3. Open questions for next session

1. **Where exactly does PPO update spatial encoder weights?** Find in `ppo.py` or `train_rl.py`. Need to know to implement freeze cleanly.

2. **Does the BC anchor mechanism need adjustment?** BC anchor computes reference logits from a full-model forward against the BC v10 checkpoint. If we freeze spatial in the trainee, the reference spatial output IS the BC v10 spatial output → bc_kl distance becomes purely about temporal+heads. This is probably fine but verify.

3. **What's the right "match within noise" tolerance for the quality gate?** Look at past run-to-run variance (e.g., S58 reruns, S60 runs) to calibrate.

4. **Should specialized weights also be batched at inference (vmap)?** Only matters if pool > 8 is needed. Defer until S5 results show whether shared-backbone-alone hits the threshold.

5. **What does the snapshot file format become?** Today: full model state_dict. Proposed: specialized-only state_dict + reference to BC v10 hash. Risk: stale BC v10 → snapshot inconsistency.

## §4. What's NOT to revisit

Per `docs/REFUTED_LOG.md` and `memory/project_s66_collect_arch_findings.md` §3.1:
- Multi-process CIS at N>2 via MPS (Phase A refuted)
- CPU opp inference at our scale (S65 + S66 both confirm insufficient throughput for temporal-heavy model)
- Triton Inference Server (gRPC + export pain)
- BC retrain (user explicit — long-term goal must not be hurt)
- Pool size cap (user explicit, except 6-8 is fine for self-play)

## §5. Branch + artifact map

On `perf/multi-process-cis-mps` (current branch):
- `docs/MULTI_PROCESS_CIS_DESIGN.md` — original design memo with STATUS UPDATE; preserved for decision-log value
- `docs/SHARED_BACKBONE_INVESTIGATION.md` — THIS FILE
- `pokemon-ai-starter/pokemon-ai/src/bench_mps_inference.py` — Phase A bench (run + results in `data/s65_artifacts/`)
- `memory/project_s66_collect_arch_findings.md` (in user memory, not in git) — comprehensive S66 findings

Bench logs:
- `pokemon-ai-starter/pokemon-ai/src/data/s65_artifacts/bench_mps_n1_2_6.log` — Phase A MPS bench
- `pokemon-ai-starter/pokemon-ai/src/data/s65_artifacts/stage_a_test.log` — S65 adaptive batching test (CIS-STATS)
- `pokemon-ai-starter/pokemon-ai/src/data/s65_artifacts/battle_server_diag/runA_15w_retest.log` — S65 baseline (CIS-STATS)

## §6. Pool sizing context (S66 user clarification)

Working assumption:
- **Self-play only**: pool 6-8 likely fine (each opp needs enough training signal; too few = insufficient diversity, too many = each opp underexposed)
- **Self-play + external opps**: tight, unclear yet — may need more capacity
- **Goal**: low wall time enables EXPERIMENTING with pool size — that's the user's stated motivation for fixing this

Design implications:
- Don't over-engineer for pool=33+
- Shared backbone alone (no vmap batching) should be sufficient at pool ≤ 8
- If pool > 8 ever needed: add vmap/bmm batching of specialized forward as a follow-on technique
