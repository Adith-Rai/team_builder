# Session 51 outcomes — durable record

This file is the canonical "what shipped Session 51" reference. The
next-prompt.txt has the FORWARD-looking handoff; this file is the
BACKWARD-looking record.

Session dates: 2026-05-07 / 2026-05-08
Branch: master
Commits (chronological): see `git log --oneline 2c7d97fc..f3728dcc`

---

## What shipped

### Tier 1 (production-critical)

**Per-submodule torch.compile (Path 2)** — `d5f500b8`, `251cd14a`
- Compiles `tokenizer + spatial + temporal + action_head + value_head`
  separately with `mode="default", dynamic=True`. Coverage ~90% vs Path 1's
  ~40-60% (single-method).
- `_dynamo.config.suppress_errors=True` for B=1 dynamic-shape edge.
- Compile happens AFTER `_resume_from_checkpoint` (caught + fixed bug
  where compile-before-resume produced state_dict mismatch).
- `_resume_from_checkpoint` strips `_orig_mod.` prefix on load.
- Validation: synthetic test `scripts/diag/test_compile_new_arch.py`
  shows 1.12-1.25× full forward, 2.49× cudagraph-cached forward_spatial.
- Production validation: iter 14 (uncompiled) 4423s vs iter 15 (compiled)
  3083s. **30.3% per-iter speedup.**
- Doc: cookbook §3i (rewrote stale "AttributeError" claim with actual
  failure mode = torch+triton version mismatch on pod).

**Fused AdamW** — `d5f500b8`
- `torch.optim.AdamW(fused=True)` autodetected on Ampere+ via
  `torch.cuda.get_device_capability()[0] >= 8`. Falls back cleanly otherwise.
- 3-7% on optimizer.step.

**`--bf16` flag** — `d5f500b8`
- Mutex with `--fp16`. New `precision_config.py` with `set_amp_dtype()` +
  `autocast_ctx()` global helper.
- Plumbed through mp workers via cmd dict (`amp_dtype` field, str → dtype
  via `parse_amp_dtype`). Workers call `set_amp_dtype()` on receipt.
- Affects collect-path autocast only (InferenceBatcher, battle_agent,
  rl_pipeline, mp_collect_v3). PPO update is fp32.
- Validation: 1-iter `--bf16` smoke at games=4 conc=2 N=1, AMP dtype: bf16,
  W/L 50%, kl=0.0285, no NaN.
- Production NOT yet using `--bf16`; held for next relaunch.

### Heartbeat mitigations — `fd88d552`

After Phase 1 v3 hung 7+ hours at iter 17 (8/8 workers stale-heartbeat
during pool-growth), three fixes in `mp_disk_collect.py`:
1. `HEARTBEAT_TIMEOUT_S` 300s → 600s
2. `time.sleep(0.25)` between worker cmd sends (sync + bg paths) — spreads
   the disk-load thundering herd
3. `_liveness_loop` thread fires heartbeats too — decouples liveness from
   asyncio loop being responsive (the broken assumption)

Validated in production: iter 16 of compiled run (first iter with pool=3
post-fix) ran cleanly: 8/8 workers, 1598/1600 trajs.

Memory: `feedback_iter17_heartbeat_hang.md`. Cookbook §3k.

### CIS (Centralized Inference Server)

**Phase 1 (skeleton)** — `50d4e80a`
- New `mp_centralized_collect.py`. CISClient/CISServer/CISClientHandle.
- Single CIS subprocess with model on GPU. Numpy IPC across mp.Pipe.
- `torch_dict_to_numpy` / `numpy_dict_to_torch` helpers handling nested
  dicts (field_banks, transition_ids, active_move_banks).
- Test: `test_cis_phase1.py` 100 batches B=8 fp16. **0.0 max abs diff
  (bit-exact)** vs main-process forward.

**Phase 2 (multi-worker batching)** — `50d4e80a`
- `_cis_main_multi` with cross-worker batching: accumulate pending
  requests until min_batch=8 OR timeout=15ms, then fire one batched
  forward, dispatch per-request slices.
- mp.connection.wait multiplexes N worker pipes.
- Test: `test_cis_phase2.py` 4 workers × 30 reqs. fp32 control max abs
  diff 5.72e-6 (numerical noise); fp16 7.81e-3 (1 fp16 ULP at log scale,
  expected from cross-worker batching).

**Phase 3 (weight reload)** — `50d4e80a`
- `reload` command in CIS protocol. Atomic file write + signal pattern
  mirroring mp_disk's. **Caught + fixed silent `_orig_mod.` strip bug**:
  without the strip, `strict=False` masked compile-state-dict key
  mismatches → half-loaded model.
- Test: `test_cis_phase3.py` perturb-save-reload-and-verify-back round
  trip, all stages bit-exact (0.0 diff).

**Phase 4.1/4.2 (production wiring)** — `8128beaa`, `21484fcc`
- `CISInferenceBatcher`: drop-in for `InferenceBatcher`. Same submit API;
  internally pipes to CIS via the worker's CISClientHandle.
- `_cis_worker_main` + `_do_collect_iter_cis` + `_run_collect_in_worker_cis`:
  worker entrypoint mirroring mp_disk's. POKE_LOOP rule applies (await
  asyncio.sleep(1.5) after _cancel_listener).
- `mp_centralized_collect_sync`: orchestrator. Lazy-spawns CIS server +
  N worker procs, signals weight reload at iter boundary, dispatches
  collect_iter cmds with 0.25s stagger, aggregates traj files.
- `--cis` flag in train_rl.py + dispatch in `_collect_data` + mutex
  vs `--mp` + launch banner shows "Collect: CIS|MP|SYNC".
- **Low-priority CUDA stream** (priority=-1) for CIS forwards. Main's
  optimizer.step on default-priority stream wins arbitration; CIS fills
  gaps. Falls back cleanly if priority creation fails.

**NOT YET DONE — Phase 4.3 deferred**:
- PFSP per-opp model swap. Current cut: CIS holds ONE model. Workers'
  player AND opp inference both go through same handle → fake PFSP.
  Critical correctness issue.
- Bg overlap re-enable for `--cis --pipeline`. Need CISBgCollector
  wrapper + sustained validation that the low-priority stream actually
  arbitrates as expected.
- 6-test pattern validation (logits identity vs `--mp` baseline,
  5-iter sustained, failure recovery).

### Multi-gen architectural foundation

**D1 gen-id token** — `fd878a9b`
- `nn.Embedding(10, d_model)` for gens 0-9 in TransformerBattlePolicy.
- Concat into spatial sequence at index 14. `N_BATTLE_STATE` 14→15,
  `N_TOKEN_TYPES` 28→29, `N_BATTLE_STATE_SLOTS` 12→13. Adds `TT_GEN`
  + `PS_SLOT_GEN`.
- Default fallback: cfg.format_config.gen if `batch["gen_id"]` missing.
- Test: `test_d1_gen_token.py` cross-gen output diff (gen 9 vs gen 6
  action_logits 9.15e-4, summary 8.03e-3 — small but real conditioning).
- **Architecture incompatible with BC v10 ckpt** — BC v11 retrain plan
  per MULTIGEN_FEASIBILITY.md handles this.

**D2 gen-aware features** — `17e2d111`
- `make_features` returns `gen: int` (read from battle.gen, fallback 9).
- `build_turn_batch` sets `batch["gen_id"]` (B=1,) long tensor.
- Per-gen feature gating is implicit via poke-env's API (False/0 for
  unavailable mechanics). No explicit code-level gating needed (and
  shouldn't be added — see next-prompt §B10).
- Test: `test_d2_gen_aware_features.py` source + behavioral checks.

**D3 per-gen procedural teambuilder** — `e6b0406c`
- team_generator.py was ALREADY gen-aware (`_UBERS_BY_GEN`,
  `_default_tiers`, `load_pokemon_pool`, `ProceduralTeambuilder`).
- D3's value is the validation that it actually works for gens 6/7/8.
- Test: `test_d3_per_gen_teambuilder.py` 4 gens × 5 teams each PASS.
  Pool sizes: gen 6: 442, gen 7: 246, gen 8: 212, gen 9: 545.

### Sanity checks (no code changes)

**E2 per-gen smart bots**: 16/16 (4 gens × 4 bots) construction-level PASS.
SimpleHeuristicsPlayer / SmartDamagePlayer / TacticalPlayer / StrategicPlayer
all instantiate cleanly for `battle_format=gen{6,7,8,9}ou`.

**D5 partial**: `_parse_gen_from_format` works for all gens 6-9, defaults
gracefully to 9. Full pipeline blocked on D4 (HuggingFace replay corpus pull).

---

## Gains in numbers

- 30.3% per-iter speedup from torch.compile (production iter 14 vs 15)
- ~$50-75 saved over remaining 185 iters at $1.50/hr
- 1.12-1.25× full forward speedup synthetic
- 2.49× cudagraph-cached `forward_spatial` synthetic
- 8/8 workers responding through pool=3 boundary (mitigations work)
- First production iter to hit 50.2% W/L (iter 16 compiled)
- 0.0 max abs diff (bit-exact) on 100-batch CIS logits identity
- 1.23% relative grad-norm diff CIS backward equivalence

---

## Failures + recoveries

1. **Iter 17 hang**: 7+ hour 0% CPU hang at `sys.exit(2) → shutdown_workers()`.
   Diagnosed via process state inspection + log scrub for stale heartbeat.
   Three mitigations shipped + validated.

2. **First compiled relaunch crash**: state_dict mismatch from
   compile-before-resume order. Caught at launch banner. Fixed in
   `251cd14a` (compile after resume + `_orig_mod` strip).

3. **CIS Phase 3 silent half-load**: stage 6 reload-back-to-original
   diff was 6.64e-1 (huge). Caught by the test asserting bit-exact
   round-trip. Root cause was `strict=False` masking `_orig_mod` key
   mismatches. Same fix pattern as #2.

Pattern: **every failure was caught by a validation step**. The cost of
running validation is much less than the cost of NOT running it.

---

## Dead paths investigated + rejected this session

- `--compile` "AttributeError on new arch" claim — was wrong. Method
  exists. Real blocker was triton version mismatch.
- `mode="reduce-overhead"` for full multi-module forward — cudagraph
  aliasing breaks. Use `mode="default"`.
- Explicit per-gen feature gating in `make_features` — poke-env handles
  it via API.

---

## What Session 52 starts with

- Production: Phase 1 v3 compiled run live, ~170 iters remaining
- Pod git HEAD: `251cd14a` (compile-after-resume fix)
- Latest committed master: `f3728dcc` (next-prompt for Session 52)
- All Tier 1 + heartbeat fixes + CIS infra + multi-gen arch foundation
  shipped + tested
- Phase 4.3 (CIS production-readiness) is the headline remaining infra
  work; multi-gen data pipeline (D4-D6) is the headline project work
- Production stays on `--mp --compile`; `--cis` is gated and not for
  production until 4.3 lands
