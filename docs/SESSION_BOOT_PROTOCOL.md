---
name: Session boot protocol — standing orders for every session, every time
description: PROJECT-ETERNAL. Mandatory first-read every session. Philosophy + sequencing + DO/DO-NOT + locked decisions + plan rationale. The user will NOT repeat this — internalize it on every load. If unsure between rule and request, surface the conflict; do not silently reconcile.
type: project
originSessionId: 6e4e7261-a0cf-4003-8f1a-3206900c7ce2
---
# SESSION BOOT PROTOCOL — read first, every session, no exceptions

**Last refreshed**: S64 Phase B wrap, 2026-05-15 (sequence packing SHIPPED + merged to master). Sections §2, §3, §6, §9, §11 are project-eternal and largely unchanged since S55. Sections §1, §4, §5, §7, §8, §10 refreshed to current state (post-S64 Phase B).

This file consolidates the project's standing orders. **The user will NOT repeat any of this each session.** They have explicitly asked that you internalize it permanently. Apply on every decision. If a user request seems to conflict with a rule below, surface the conflict — don't silently reconcile.

This is the project's project-eternal layer. `next-prompt.txt` is the session-specific layer (current production state, in-flight work, immediate priorities). Both must be read before responding to anything.

**Companion docs (added S64 Phase A — read these together for full context)**:
- `docs/STATUS.md` — current state + era history (where we've been, where we are)
- `docs/REFUTED_LOG.md` — techniques tried and refuted, with evidence + revisit conditions
- `docs/PROFILE_BOTTLENECKS_REPORT.md` — bottleneck data + current optimization arc state

---

## §1. Session start — exact order

Before responding to ANY user request, in this exact order:

1. **Read `MEMORY.md`** (auto-loaded; index of all memory pointers — has the "First action" entry pointing to current in-flight work)
2. **Read this file** (project_session_boot_protocol.md / docs/SESSION_BOOT_PROTOCOL.md) — standing orders
3. **Read `feedback_engineering_standard.md`** — quality standard (no shortcuts, ship correctly first time, read code + docs before touching)
4. **Read `project_optimization_tracker.md`** — current optimization arc roadmap (S63+), session protocols (§5.1/§5.2), failure-mode reference (§6)
5. **Read `next-prompt.txt`** at project root — session-specific deep state (most recent wrap at top, older wraps preserved below)
6. **Read the in-flight technique's results memo** if mid-flight on a specific technique (e.g., `project_s64_phase_a_results.md` if continuing Phase B)
7. **Re-read CONSUMER CODE before accepting any wrap memo's spec** (S64 Phase A learning — spec memos drift; consumer code is truth. Re-read what produces input to AND consumes output from whatever you're changing.)
8. **Verify pod state** with the batched SSH command in `project_optimization_tracker.md` §5.1 step 6 — pgrep, nvidia-smi, ss, git rev-parse, df
9. **Verify local branch state** — if technique code is on a `perf/*` branch (e.g., `perf/seq-packing`), checkout that branch before any edits. Confirm `git rev-parse <branch>` matches the wrap memo's stated HEAD.

If any of these files conflict, surface the conflict. Don't silently reconcile. **Newer findings override older speculation** — when memory and code disagree, trust the code AND update the memory.

If the user invokes a clear destructive action (kill, reset --hard, force push, etc.), pause and confirm before executing — even if the conversation seems to imply approval. (`feedback_ask_when_uncertain.md`)

**Historical context docs** (read when relevant, not always required):
- `docs/PHASE1_V3_OBSERVATIONS.md` — Phase 1 v3 trajectory + Phase 2 design implications
- `docs/PHASE1_POSTMORTEM.md` + `docs/PHASE1_DIAGNOSIS_REPORT.md` — S50 lr regression analysis
- `docs/CENTRALIZED_INFERENCE_DESIGN.md` + `project_cis_4_6_design.md` — CIS architecture (S53-S54)
- `project_strategic_frame.md` — locked sequencing (vertical → horizontal)

---

## §2. Project mission (immutable — locked across sessions)

**Goal**: Build a Pokemon battle AI that competes at an elite level — among the strongest human players, the strongest published bots (Metamon's Kakuna, SmallRL, Minikazam), and the strongest other engines.

**Scope (load-bearing, do not narrow)**:
- All legal Pokemon formats: singles, doubles, triples, 4v4, 6v6, all tiers (OU, Ubers, AG, UU, RU, etc.), random battles (varied levels), in-game/cartridge battles
- All gens (4-9+, expanding as new gens release)
- Plus team building, not just play

**Deployment scope**: massively scale on paid cloud for throughput-heavy training AND remain runnable locally on a 6 GB consumer GPU for development + smoke tests. **Same code, both environments. Not parallel codebases.**

**V1 baseline**: gen 9 OU singles. Multi-gen (gen 6+) is Phase 3+. Pre-gen-6 explicitly out of scope per `docs/MULTIGEN_FEASIBILITY.md`.

**Elite play means**: long-horizon planning, opponent modeling, risk-adjusted decision making, squeezing fractional advantage from type/STAB/effectiveness/speed/items/abilities/positioning.

---

## §3. Operational philosophy (immutable — never compromise)

These are the user's explicit standards. They override convenience, speed, or apparent simplicity.

1. **Quality > speed.** Every solution must be the proper, complete, honest one. The "good enough for now" version is the version that needs to be touched again.

2. **Cost discipline = thoroughness, not haste.** The pod is paid time. Investing the time to ship correctly the first time is cheaper than touching-again work. Don't run up the bill on re-investigated settled questions.

3. **Anything shipped should not need to be touched again.** "Done" means: code clean (no dead branches, no half-completions, no "TODO: fix later"), validation gate met (Tests 1-3 + smoke), docs that reference the area updated, next-session handoff captures it so future-me doesn't re-investigate.

4. **Long-term focus on every change.** Every change is in service of: gen 9 elite (vertical) → multi-gen launch (horizontal) → broader formats. Don't make changes that look good locally but conflict with where the project is going.

5. **Honest opinions, including pushback.** If the user's framing has a precision-bug or premise-bug, surface it gently and clarify before acting. "Yes, *and here's the precise version*" is better than executing the literal-but-wrong version. The user has explicitly rewarded this pattern (S52, S53, S54).

6. **Ask when uncertain on load-bearing or destructive actions.** Cost of waiting 60s for user response is tiny vs cost of irreversible action on wrong premise. (See `feedback_ask_when_uncertain.md` for canonical examples.)

7. **Read before touching.** Before any non-trivial edit: read the file, read its callers, read the docs that reference the area, check if the change has been tried + abandoned (`§K dead paths` in next-prompt + below in §6). Skipping the read step is how regressions ship.

8. **Verify before recommending from memory.** Memory snapshots are point-in-time, not current state. Grep for functions/flags before promising they work. Believing memory over the code wasted hours in past sessions.

---

## §4. Locked sequencing (do not deviate; reasoning below each)

The project has a strict order. Items LATER must not start before items EARLIER complete or are explicitly approved as parallel.

**Why locked**: vertical-first (gen 9 ceiling) protects the multi-gen launch from infra problems mid-flight (5-7 wk / $400-600+ runs are expensive to debug). The user has explicitly aligned on this in S52 — `project_strategic_frame.md` is the canonical reference.

### Order (current state at S64 Phase A — see `docs/STATUS.md` for full era history)

1. **Phase 1 v3 — DONE.** S57 ran to ~iter 90 then was diagnosed as failing via type-knowledge erosion (3-stage collapse: exploration → type-knowledge erosion → strategic collapse). See `memory/project_phase1_v3_diagnosis.md`. **BC anchor mechanism** designed S57 to prevent the erosion in Phase 2. The Phase 1 v3 baseline establishes the floor; Phase 2 must work first time.

2. **Tier 3 sequence-batched PPO — SHIPPED.** C1-C4 + C5 (compile boundary) + train_rl.py wiring + BC anchor + `--tier3-minibatch-size N` (required at prod scale to avoid OOM) all SHIPPED S55-S57. Used by current canonical Phase 2 stack.

3. **CIS Phase 4.6 — SHIPPED.** Option B full-reset on CIS failure (`memory/project_cis_4_6_design.md`). Single-opp-per-worker post-S58 (`memory/feedback_asyncio_gather_at_scale.md`).

4. **S58 ghost-tie root cause + fixes — SHIPPED.** `asyncio.gather` of N heterogeneous opp coros per worker was starving small batches → wait_for cancels → `battle.won = None`. Replaced with one-opp-per-worker. See `memory/project_s58_session_narrative.md`.

5. **Multi-session UPDATE OPTIMIZATION SEQUENCE — IN FLIGHT (S62+).** Per `memory/project_optimization_tracker.md`:
   - S62 Fix #1 Option B SHIPPED at prod (-27% collect)
   - S62 Fix #2 (--compile) REFUTED at prod (8% slower) — dropped from canonical Phase 2 stack
   - S62 update profile revealed orchestration-bound update (CUDA kernels = only 8% of update wall)
   - S63 free wins SHIPPED at -4.2% wall (set_to_none + .item() defer)
   - S64 step-back: sequence packing prioritized; ARCH (drop temporal stack) refuted as priority
   - S64 Phase 1 PASSED: torch 2.5.1 venv-isolation
   - S64 Phase A SHIPPED: `collate_episodes_packed` on `perf/seq-packing`. 11/11 equivalence tests pass.
   - **S64 Phase B SHIPPED + MERGED**: full `--packed` pipeline. Master at `ba2ced64`. **Measured -7.3% update wall / -5.3% overall at prod** (1600g/200conc). 5/5 bit-equiv gates passed. Canonical Phase 2 launch now `./launch_rl.sh ... --packed ...`. See `memory/project_s64_phase_b_results.md`.
   - **Phase B wrap surfaced finding**: update phase is ~4× super-linear in B (62× wall for 16× games; collect 15.4× ✓). Update is orchestration-bound (S62 profile: 92% Python/CPU). #2 CUDA Graphs projection REVISED UP to **1.5-3×** (was 1.3-2×) — attacks the per-chunk Python loop that drives the super-linearity.
   - Then: CUDA Graphs (#2 in tracker, revised projection), `cudaMemcpyAsync` investigation (#3, opportunistic).

6. **Phase 2 prep — DEFERRED until optimization sequence completes (or hits diminishing returns).** Per user. When sequence completes:
   - Source 100-150 real elite gen-9-OU teams (Smogon archive / ladder replays / tournament posts — copy verbatim, NEVER mix with the 16 mm-competitive eval set)
   - lr scheduling impl + re-ablate post-optimization-arc to confirm `--lr 1e-5` still optimal
   - External opps in PFSP pool (mcts-fast + Kakuna + SmallRL + Minikazam)
   - H2H gauntlet wiring
   - Canonical Phase 2 launch script with the validated stack

7. **Phase 2 launch** — canonical stack (post-S62/S63 refutations): `--cis --pipeline --bf16 --tier3 --tier3-minibatch-size 16 --bc-anchor-ckpt v10 --bc-anchor-coef 0.1 --cis-min-batch 32 --cis-timeout-ms 50` (NO --compile per S62 prod-refutation). Init from Phase 1 v3 final.pt or BC v10. Target ~5-15 min/iter (depends on optimization arc final speedup); ~$200-300 for 200 iters. **Where elite gen 9 gets measured.**

8. **Post-Phase-2 evaluation** — H2H gauntlet vs published bots tells us if we hit elite. If yes: proceed to multi-gen prep. If no: investigate + iterate (lr re-ablation, further training, etc.)

9. **Multi-gen-SPECIFIC prep (D4-D6, BC v11)** — only after #8 confirms gen 9 elite. ~$10-15 BC v11 + ~$5-300 corpus pulls + 5-7 day BC retrain.

10. **Multi-gen launch** — first multi-gen PPO run. ~5-7 wk / $400-600+. Should require zero infra changes (everything tested in vertical).

11. **Doubles / VGC / Triples / AG / in-game** — separate format support work post-multi-gen. Architecture is format-agnostic by construction; format-specific work is data + tokenizer extensions, not infra.

### Things explicitly NOT in this order until earlier items complete

- **Multi-gen-SPECIFIC infra (D4 corpus pull, D5 memmap, D6 BC v11, E1-E3 per-gen evals)** — JIT before #9 only. Per `project_strategic_frame.md`, multi-gen-SPECIFIC infra is JIT before launch, NOT shipped-and-cold. Do NOT pivot here while gen-9 vertical work is in flight.
- **Doubles / VGC / Triples / AG architecture work** — post-multi-gen. Tokenizer asserts n_active=1 (singles); multi-active = future redesign per REWRITE_DESIGN.md.
- **Network Volume migration** — explicitly deferred until multi-gen runs (per user). Container Disk OK for Phase 1/2 since user monitors and won't stop pod without informing.
- **Phase 2 launch** — DEFERRED until optimization sequence completes (or hits diminishing returns). Don't propose launching Phase 2 mid-arc.

---

## §5. DO NOT — with WHY for each

Each rule below has burned cycles in past sessions. Don't relearn them.

### Production safety

- **Don't pull new code on prod pod while a long run is in flight.** HEAD on prod is locked at a specific commit (currently `251cd14a` for Phase 1 v3). Newer commits change `state_dict` shapes (D1+D2 gen_embed, CIS 4.4/4.5/4.6 etc.). A respawned worker from new code crashes on `load_state_dict`.

- **Don't change variables mid-run.** A/B comparison only works if non-test variables are constant. Phase 1 v3 deliberately uses fp16 (not bf16) so the compile A/B against iter 13 baseline isn't confounded.

- **Don't take destructive actions (kill, reset --hard, force push, drop tables) without explicit confirmation.** Even if the user said "proceed" earlier, scope of approval is what was specified, not beyond. Cost of asking 60s for confirmation is trivial vs cost of irreversible mistake.

- **Don't skip hooks (--no-verify, --no-gpg-sign) unless user explicitly asks.** Investigate root cause if hook fails.

### Settled architectural decisions — don't reopen

These are documented dead paths. Each was investigated + rejected. Don't re-walk:

- **`--lr 3e-5`** caused regression on transformer arch (S50). lr=1e-5 is locked from 4-point ablation. (`docs/PHASE1_DIAGNOSIS_REPORT.md`)
- **REBIND_WORKER ctrl protocol** for CIS worker respawn — Connection-passing-via-mp.Pipe.send hangs in RunPod environment. Replaced with Option B full-reset (Phase 4.6, S54). (`project_cis_4_6_design.md`)
- **Dual-slot CIS** (slot 0 = player, slot 1 = active opp) — doesn't compose with asyncio.gather opp concurrency. Replaced with pool-mirror multi-slot (S53). (`project_cis_4_3_design.md`)
- **`--opponent-device cpu`** — tried S50, broke at production scale (CPU 20M-param forward × conc=200 exceeds WS keepalive). Stay on cuda. (`docs/PPO_CLOUD_COOKBOOK.md §3h`)
- **forkserver mp context** — SemLock race at N≥4 (CPython 3.11 bug). Use spawn.
- **Torch tensor IPC across mp.Pipe** — mmap explosion at production scale. Numpy IPC only.
- **`--compile` on legacy arch** — not supported (different forward shape). Only TransformerBattlePolicy supports compile.
- **mode="reduce-overhead" for full forward compile** — cudagraph aliasing breaks on multi-module patterns + recompile churn. Use "default" + dynamic=True.
- **Path 1 single-method compile** (forward_spatial only) — 40% coverage. Path 2 (per-submodule, all 5) is correct.
- **Standalone `grad_accum > 1`** — Session 31 showed silent quality regression. Tier 3 transition-level minibatching subsumes the benefit safely.
- **Explicit per-gen feature gating in code** — poke-env API already returns False/0 for unavailable mechanics per gen. Belt-and-suspenders gating is dead weight (S51 D2).
- **Removing CISClientHandle thread-safety** — both async-dispatch + sync lock paths are load-bearing. mp.Pipe.send not byte-atomic. (`feedback_cis_thread_safety.md`)
- **VGC-Bench as Tier 3 reference** — they're stateless per-turn (no temporal stack); their batching freedom doesn't transfer. Use Metamon's `metamon_to_amago.py`.
- **Gigantamax / Doubles / Triples in V1** — out of scope. Post-multi-gen format work.
- **Per-worker traj checkpointing on CIS failure** — REJECTED per Q3 cost analysis (<$3/run savings vs half-day dev). Don't reopen unless empirical failure rate exceeds 5/run. (`feedback_phase46_traj_checkpoint_rejected.md`)
- **`F.one_hot` inside torch.compile dynamic-shape region** — fails on torch 2.2.x with "Cannot call numel() on tensor with symbolic sizes/strides". Use broadcast-equality. (S56 `project_tier3_c5_design.md` Limit 1)
- **`nn.utils.clip_grad_norm_(model.parameters(), ...)` inside torch.compile** — graph-breaks on torch 2.2.x. Move to eager wrapper. (S56 Limit 2)
- **Single-graph optimizer.step on safety-mask-zeroed loss** — AdamW weight_decay drifts params even with zero gradients. Gate `optimizer.step()` on `step_mask.item() > 0.5` in eager wrapper. (S56 Limit 3)
- **`forward_ppo_sequence` with `L_max > cfg.temporal_context`** — outer indexing OOB. Pre-truncate via `collate_episodes(L_max=cfg.temporal_context, tail=True)`. (S57 `d1b101bb`)
- **Tier 3 `ppo_update_batched` mega-batching all episodes** — OOMs at 200+ games on A100 80GB. Use `--tier3-minibatch-size N`. (S57 `6821a907`)
- **`torch.inference_mode()` for BC ref forward** — CUDA OOB + CUBLAS_STATUS_EXECUTION_FAILED with autocast bf16. Use `torch.no_grad()`. (S57 `project_bc_anchor_design.md`)
- **Storing BC ref as `model._bc_ref` attribute** — state_dict pollution → mp_disk worker respawn fails. Store as local in `main()`. (S57)
- **`CUDA_LAUNCH_BLOCKING=1` for diagnosing PPO update crashes** — slows collect to a crawl. Use targeted instrumentation or `--n-iters 1`. (S57)
- **`asyncio.gather` of N heterogeneous opp coros per worker** — small batches starve, wait_for cancels → ghost ties. Use one-opp-per-worker (process-per-task pattern). (S58 `feedback_asyncio_gather_at_scale.md`)
- **`--compile` (whole-fn) at prod scale** — 8% SLOWER per step at 1600g/8w. Smoke 9× was anomaly. DROPPED from canonical Phase 2 stack. (S62 `project_s62_fix2_prod_validation.md`)
- **Vectorizing `collate_episodes` in Python** — V2 within noise, V3 strictly slower. C ops are bandwidth-bound. Superseded by S64 sequence packing. (S60 `project_s60_fix3_refuted.md`)
- **Liger-Kernel drop-in** — we use GELU + LayerNorm, NOT SwiGLU + RMSNorm. Doesn't apply. (S62)
- **Perm-at-eval** — measurement-fudging shortcut; masks invariance bugs by making eval easier. Other Pokemon PPO codebases (Metamon, ps-ppo) keep eval canonical. (S60 `feedback_perm_at_eval_wrong.md`)
- **Drop temporal stack as PRIORITY in optimization arc** — temporal stack <8% of update wall; dominant waste is in `collate_episodes` data prep. Sequence packing addresses it instead. May revisit post-Phase-2. (S64 step-back, `project_s64_arch_revisit_profile_findings.md`)

**Additional refutations + revisit conditions consolidated in `docs/REFUTED_LOG.md` (35 REFUTED items, 5 DEMOTED, 4 DEFERRED with evidence + memory cross-refs).**

### Training-correctness rules

- **Don't put the 16 metamon-competitive eval teams in training pool.** That's data leakage — model memorizes specific compositions, eval scores inflate artificially, eval signal lost. Phase 2 needs SEPARATE 100-150 real elite teams sourced from real competitive play (Smogon archive, Showdown ladder replays, tournament posts) — copy verbatim, NEVER generate or estimate.

- **Don't skip slot permutation.** Procedural teams + slot perm are LOAD-BEARING design choices for general Pokemon learning (not gen-9-OU memorization). The S50 postmortem critique was about train/eval distribution mismatch, not procedural-or-perm-being-wrong. The fix is MIX in real elite teams during training, NOT abandoning procedural.

- **Don't ship `grad_accum > 1` standalone.** Tier 3 minibatching is the correct path.

- **Don't reopen the lr question without a re-ablation.** lr=1e-5 was hard-won. Tier 3 may need re-ablation (changes effective batch size); do that as part of C6, not as a standalone change.

### Process rules

- **Don't add features beyond the task scope.** Gold-plating creates touched-again work. Solve the task in front of you completely, don't bundle extras. ("Don't make changes that look good locally but conflict with where the project is going" — `feedback_engineering_standard.md`)

- **Don't take shortcuts.** No `.float()` shims to paper over precision questions, no "fix the related thing later" (it never gets fixed), no hardcoding when a flag exists, no skipping smoke validation. The pod is paid time — every shortcut costs more later.

- **Don't pivot to multi-gen-SPECIFIC work mid-vertical.** D4-D6, BC v11, E1-E3 are JIT before multi-gen launch (item #9 in §4). They are NOT current work. If you find yourself starting them, stop — re-read §4 sequencing.

- **Don't skip end-of-session memory + docs + next-prompt updates.** Future-me reads what current-me writes. Skipping these creates the "what was that decision again?" pattern that costs hours next session.

---

## §6. DO — with WHY for each

### Before any non-trivial change

1. **Read the file you're about to change + its callers.** Don't assume you know how the system works from name + docstring.
2. **Read the docs that reference the area** (`PPO_CLOUD_COOKBOOK.md`, `MP_DISK_REDESIGN.md`, `CENTRALIZED_INFERENCE_DESIGN.md`, `PHASE1_V3_OBSERVATIONS.md`, relevant memory files).
3. **Check §5 dead paths above + `next-prompt §K`.** Has this been tried + abandoned?
4. **Check `project_strategic_frame.md`.** Does this align with strategy?
5. **If load-bearing or non-obvious: surface options + ask user before committing.**

### While coding

1. **Don't add code that's not actively needed.** Don't leave half-finished branches.
2. **Don't bypass safety checks** (`--no-verify`, `strict=False`, etc.) without strong justification documented.
3. **If a hook fails, fix the underlying issue — don't skip.**

### After coding, before declaring done

1. **Local smoke or cloud smoke if needed.** Engineering standard: validation gate met.
2. **Update docs that reference the area** (cookbook, design doc, memory files).
3. **Update memory if behavior or rationale changed.**
4. **Commit with full message explaining WHY.**
5. **Update `next-prompt.txt`** if the next session needs to know.

### Use of tools

- **Use TaskCreate for multi-step work.** Mark each task completed as soon as it's done; don't batch.
- **Use the dedicated tools** (Read, Edit, Write, Glob, Grep) — reserve Bash for shell-only ops.
- **Make all independent tool calls in parallel** when there are no dependencies.

---

## §7. Critical decisions already settled (with rationale links)

These are the load-bearing choices that have been made + validated. Don't relitigate without strong empirical reason.

| Decision | Source | Why |
|---|---|---|
| `--lr 1e-5` for transformer PPO | S50 4-point ablation | Higher lr causes drift on new arch (3.3× lower gradient sensitivity) |
| `--lam 0.95` (not 0.75) | S35 audit | Longer advantage horizon for the deeper arch |
| `--ent-coef 0.02` + adaptive entropy 0.65/0.95 | S29 + S35 | Stable exploration; starting entropy matters more than coef |
| `--warmup-iters 5` (not 20) | S50 cont. | v_loss converges by iter 3-5 |
| `--bf16` on PPO update (post-Phase-1-v3) | S52 commit `fb2127a3` | bf16 has fp32 range; fp16 backward without GradScaler underflows |
| `--compile` Path 2 (per-submodule, all 5 modules) | S51 | Path 1 covers only ~40% of forward |
| `--mp` workers each hold a GPU model | S50 design | --opp-cpu broke at scale; cuda is required |
| Pool-mirror CIS multi-slot (slots 1..K_max) | S53 (`project_cis_4_3_design.md`) | Dual-slot doesn't compose with asyncio.gather opp concurrency |
| Async-with-req_id-dispatch CIS handles | S53 (`feedback_cis_thread_safety.md`) | Per-handle Lock caps GPU at ~48% util |
| Dedicated CIS ctrl pipe (Phase 4.5) | S53 | Worker recv_loops race with parent reload reads at iter boundaries |
| Option B full-reset on CIS failure (Phase 4.6) | S54 (`project_cis_4_6_design.md`) | REBIND_WORKER hangs in RunPod env; full-reset is simpler + correct |
| Per-transition mean for Tier 3 batched loss | S54 | Metamon-style; B=1 case is bit-equivalent to per-episode mean |
| Procedural teams + slot perm in training | Project design intent | Teaches Pokemon mechanics broadly, not gen-9-OU memorization |
| 16 mm-competitive teams reserved for eval-only | S54 (`docs/PHASE1_V3_OBSERVATIONS.md`) | Mixing into training = data leakage; eval signal lost |
| Phase 2 to mix in 100-150 real elite teams (separate from eval set) | S54 (`docs/PHASE1_V3_OBSERVATIONS.md`) | Bridges train/eval distribution gap without leakage |
| Heartbeat from OS thread (not asyncio coroutine) | S51 iter-17 RCA | Blocking torch.load starves async heartbeat |
| spawn (not forkserver) mp context | S50 | CPython 3.11 SemLock race at N≥4 forkserver |
| Numpy-only IPC across mp.Pipe | S50 | torch tensor IPC mmap-explodes at scale |
| Tier 3 C5 compile boundary: forward+loss+masked-backward in graph; zero_grad/clip/optimizer.step eager | S56 | Maximum achievable on torch 2.2.x dynamo; three limits forced the split (one_hot, clip_grad iterator, weight_decay drift). 45% iter-2 update speedup measured on dev pod smoke. |
| `twohot_target` uses broadcast-equality not F.one_hot or scatter_ | S56 | F.one_hot fails dynamic compile; broadcast-equality is bit-equivalent + compile-safe |
| BC anchor uses `KL(BC ‖ model)` not `KL(model ‖ BC)`, coef 0.1 | S57 (`project_bc_anchor_design.md`) | Teacher-tells-student form. Bounded since BC's softmax mass is finite. Coef 0.1 preserves drift bound without capping improvement direction. |
| `collate_episodes` for forward_ppo_sequence callers must use `L_max=cfg.temporal_context, tail=True` | S57 (`d1b101bb`) | TemporalTransformer truncates internally; outer indexing requires L_max ≤ temporal_context. |
| Tier 3 `--tier3-minibatch-size N` REQUIRED at production scale (≥200 games) | S57 (`6821a907`) | Mega-batch OOMs on A100 80GB. Suggested: 16 for memory headroom. |
| BC ref stored as LOCAL variable, NOT `model._bc_ref` attribute | S57 | model attribute = state_dict pollution = mp_disk respawn fails. |
| One opp per worker (NOT N opps via `asyncio.gather`) | S58 | Small batches starve, wait_for cancels → ghost ties. Process-per-task is the industry standard. |
| Canonical Phase 2 stack: `--cis --pipeline --bf16 --tier3 --tier3-minibatch-size 16 --bc-anchor-ckpt v10 --bc-anchor-coef 0.1 --cis-min-batch 32 --cis-timeout-ms 50` (NO --compile) | S62 (`project_s62_fix1_b_results.md`, `project_s62_fix2_prod_validation.md`) | Option B SHIPPED at prod (-27% collect). --compile REFUTED at prod (8% slower per step at 1600g/8w). |
| `optimizer.zero_grad(set_to_none=True)` at all 12 ppo.py callsites | S63 (`project_s63_free_wins_results.md`) | Eliminates allocate-then-fill pattern. Numerically equivalent; combined with .item() defer = -4.2% update wall. |
| Defer chunk-stats `.item()` to epoch boundary in eager Tier3 path | S63 | Tensor accumulators across chunks; one `.item()` per epoch instead of ~3600. Bit-equivalent. |
| `collate_episodes_packed` (sister to `collate_episodes`) on `perf/seq-packing` branch — flat (sum_T, ...) + cu_seqlens | S64 Phase A (`project_s64_phase_a_results.md`) | Phase A of sequence packing arc. Pure additive; legacy untouched. Phase B wires it via `forward_ppo_sequence_packed` + flex_attention BlockMask. |

---

## §8. Plan trajectory — in flight + deferred (current at S64 Phase B)

### Recently completed (S64 — sequence packing optimization arc)

- S64 Phase A SHIPPED — `collate_episodes_packed` on `perf/seq-packing`. 11/11 equivalence tests pass.
- **S64 Phase B SHIPPED + MERGED TO MASTER** at `ba2ced64`. Full pipeline: `TemporalTransformer.forward_packed` (flex_attention + per-episode causal BlockMask), `forward_ppo_sequence_packed`, `_ppo_loss_packed_internal`, `--packed` flag, `./launch_rl.sh` wrapper. 5/5 bit-equiv gates passed (B.2-B.6), smoke + prod A/B at 1600g/200conc. **Measured -7.3% update wall / -5.3% overall at prod.** Numerical drift small (~-0.003 kl/bc_kl), bc_kl in favorable direction. See `memory/project_s64_phase_b_results.md` for full details + the surfaced super-linear update scaling finding.

### Next (user-decided)

- **CUDA Graphs over train_step** (#2 in tracker) — **projection REVISED UP to 1.5-3×** (was 1.3-2×) at Phase B wrap. Accounts for the per-chunk Python orchestration that drives the super-linear update scaling. 1-2 sessions.
- **`cudaMemcpyAsync` 62k @ 990μs root cause** — pending opportunistic investigation.

### Phase 2 prep (deferred until optimization arc completes)

- Source 100-150 real elite gen-9-OU teams (NEVER mix with 16 mm-competitive eval set)
- lr scheduling impl + re-ablation post-arc
- External opps PFSP scaling (S58 task #14)
- H2H gauntlet wiring (Kakuna/SmallRL/Minikazam adapters)
- Phase 2 launch script with validated canonical stack

### Phase 2 launch (canonical stack post-S62/S63)

```
--cis --pipeline --bf16 --tier3 --tier3-minibatch-size 16 \
--bc-anchor-ckpt v10 --bc-anchor-coef 0.1 \
--cis-min-batch 32 --cis-timeout-ms 50
```
**(NO --compile)** — REFUTED at prod S62 (8% slower per step). See REFUTED_LOG.md #1.

- Init from BC v10 or Phase 1 v3 final.pt (TBD based on type-knowledge erosion diagnosis + BC anchor protection)
- Team mix: 50-70% procedural / 30-50% NEW real elite teams (NOT the 16 mm-competitive eval set)
- Eval canonical (NO perm-at-eval — REFUTED S60, see REFUTED_LOG.md #17)
- External opps in PFSP pool (mcts-fast + Kakuna + SmallRL + Minikazam)
- Absolute early-stop threshold + KL gate

### Deferred — multi-gen-SPECIFIC (only after Phase 2 confirms gen 9 elite)

- D4 HuggingFace replay corpus pull (gens 6/7/8, ~$5)
- D5 multi-gen `replay_to_memmap` validation
- D6 BC v11 retrain on multi-gen (~$10-15, 5-7 days)
- E1 per-iter format selection
- E2 per-gen real-battle smart bot validation
- E3 per-gen evaluation harness
- Network Volume migration (per user: not needed until multi-gen)

### Much later (post-multi-gen)

- Wave training, fictitious play, league of opponents
- Doubles / Triples / VGC / AG / in-game format support
- Team building (separate model)

---

## §9. The user's explicit standards (verbatim quotes from past sessions)

These are direct user instructions. Apply on every decision.

- **"The pod is paid; pod time is the constraint, not session time."**
- **"No shortcuts — every solution must be the proper, complete, honest one."**
- **"Anything you ship should not need to be touched again."**
- **"Done means done — not 'works for now'."**
- **"Understand the codebase + docs THOROUGHLY before touching anything."**
- **"Cost discipline = thoroughness, not speed."**
- **"Don't run up the bill on re-investigated settled questions."**
- **"Ask me — don't guess"** (on destructive actions or load-bearing decisions)
- **"Honest opinion"** — surface actual reasoning including pushback on user when there's a premise bug
- **"Long term focus while working"**
- **"True full + most efficient + correct + stable + long-lasting solutions"**
- **"Don't half ass shit, don't forget shit, don't accidentally leave out shit"** (S55 — explicit standing order)

---

## §10. Cross-reference index

### Memory files (organized by topic)

**Operating standards (read every session)**:
- `feedback_engineering_standard.md` — quality standard
- `feedback_ask_when_uncertain.md` — when to ask vs guess
- `feedback_profile_driven_method.md` — S59-S64 measurement methodology (CONCRETE / STRONG INFERENCE / CONJECTURE labels)
- `feedback_priority_hierarchy.md` — quality > cost > calendar
- `feedback_premature_plateau_call.md` — don't call plateau on <20 iters
- `feedback_infra_first_means_first.md` — finish all infra before Phase 2 prep
- `feedback_perm_at_eval_wrong.md` — perm-at-eval is conceptually wrong
- `project_strategic_frame.md` — long-term sequencing
- `project_session_boot_protocol.md` — this file
- `project_optimization_tracker.md` — current optimization arc roadmap + session protocols (S63+)

**Current optimization arc (S63+ multi-session, read for in-flight context)**:
- `project_s62_update_profile_findings.md` — torch.profiler kernel breakdown (orchestration-bound finding)
- `project_s62_fix1_b_results.md` — Fix #1 Option B SHIPPED at prod (-27% collect)
- `project_s62_fix2_prod_validation.md` — Fix #2 (--compile) REFUTED at prod
- `project_s63_free_wins_results.md` — set_to_none + .item() defer SHIPPED (-4.2%)
- `project_s64_arch_revisit_profile_findings.md` — S64 step-back (ARCH refuted, seq-packing prioritized)
- `project_s64_phase1_torch_upgrade_results.md` — torch 2.5.1 venv-isolation PASSED
- `project_s64_phase_a_results.md` — Phase A SHIPPED + §4 detailed Phase B plan

**Phase 1 v3 + Phase 2 design context (read before Phase 2 design)**:
- `project_phase1_v3_diagnosis.md` — type-knowledge erosion (3-stage collapse)
- `project_bc_anchor_design.md` — BC anchor mechanism + 5 verified failure modes
- `project_s58_session_narrative.md` — ghost-tie root cause + asyncio.gather refutation

**Tier 3 + compile design (read before Tier 3 / compile-boundary work)**:
- `project_cis_4_6_design.md` — Phase 4.6 Option B full-reset
- `project_cis_4_3_design.md` — pool-mirror multi-slot
- `project_tier3_c5_design.md` — Tier 3 compile boundary + torch 2.2.x dynamo limits
- `feedback_cis_thread_safety.md` — pipe IPC + Lock + async-dispatch + ctrl pipe
- `feedback_cis_spawn_zombie_cascade.md` — failed CIS launches leave runaway children

**Diagnostic + behavioral patterns**:
- `feedback_iter17_heartbeat_hang.md` — heartbeat starvation RCA
- `feedback_diagnostic_signals.md` — FLOW > LIVE counter, server logs are truth
- `feedback_bf16_autocast_gating.md` — bf16-only autocast, flash via SDP
- `feedback_concurrency_vram.md` — InferenceBatcher shares model
- `feedback_asyncio_gather_at_scale.md` — don't fan out N heterogeneous coros in one loop

**Project context**:
- `project_vision_scope.md` — load-bearing project goals
- `project_session50_mp_redesign.md` — mp_disk redesign + POKE_LOOP
- `project_metamon_learnings.md` — S37 Metamon study
- `project_external_adapters.md` — external opp adapters validated
- `project_not_needed.md` — features the model learns; don't hand-engineer
- `reference_cloud_pods_usage.md` — pod connections, file paths, launch patterns

**Rejected / dead-path documentation** (don't relitigate):
- `feedback_phase46_traj_checkpoint_rejected.md` — Q3 deferral with cost math
- `project_s60_fix3_refuted.md` — vectorize collate_episodes REFUTED

### Project docs (in git, `docs/`)

- `STATUS.md` — current state + era history (read for "where are we / where have we been")
- `REFUTED_LOG.md` — durable record of refuted/demoted/deferred techniques with evidence + revisit conditions
- `PROFILE_BOTTLENECKS_REPORT.md` — bottleneck data, current optimization arc state, CONCRETE/CONJECTURE audit trail
- `PPO_CLOUD_COOKBOOK.md` — single ref for cloud PPO (§3i compile, §3j POKE_LOOP, §3k heartbeat, §5 validation tests)
- `CENTRALIZED_INFERENCE_DESIGN.md` — CIS architecture
- `MULTIGEN_FEASIBILITY.md` — multi-gen scope (gens 6+)
- `MP_DISK_REDESIGN.md` — mp_disk architecture
- `PHASE1_POSTMORTEM.md` — S50 catastrophic regression RCA (lr=3e-5 root cause)
- `PHASE1_DIAGNOSIS_REPORT.md` — S50 diagnostic report (lr ablation matrix)
- `PHASE1_INVESTIGATION_PLAN.md` — S50 investigation methodology
- `PHASE1_V3_OBSERVATIONS.md` — Phase 1 v3 trajectory + Phase 2 design implications
- `SESSION_BOOT_PROTOCOL.md` — git-mirror of this file (this is the authoritative version)

### Code orientation

- `pokemon-ai-starter/pokemon-ai/src/train_rl.py` — main training entry point
- `pokemon-ai-starter/pokemon-ai/src/ppo.py` — Trajectory, GAE, ppo_update, ppo_update_batched (Tier 3)
- `pokemon-ai-starter/pokemon-ai/src/model_transformer.py` — TransformerBattlePolicy + TemporalTransformer (causal mask + return_all_positions for Tier 3)
- `pokemon-ai-starter/pokemon-ai/src/mp_disk_collect.py` — `--mp` production path
- `pokemon-ai-starter/pokemon-ai/src/mp_centralized_collect.py` — `--cis` path (Phase 4.6 SHIPPED)
- `pokemon-ai-starter/pokemon-ai/src/features.py` — `make_features` + `build_turn_batch`
- `pokemon-ai-starter/pokemon-ai/src/eval_diag.py` — production eval

---

## §11. The "find this when" reference

- **"What's running right now?"** → next-prompt.txt §C + production status check
- **"Should I do X?"** → §4 sequencing + §5 DO NOT + `project_strategic_frame.md`
- **"Has this been tried?"** → §5 dead paths + next-prompt §K + `docs/PHASE1_*.md`
- **"What's the right hyperparameter?"** → §7 settled decisions
- **"Is this expected behavior or a bug?"** → check the relevant feedback memory + cookbook + DOC
- **"Should I commit + push?"** → engineering standard: validation passes, docs/memory updated, then yes
- **"User asked for something destructive"** → ASK first (`feedback_ask_when_uncertain.md`)
- **"Memory says X but I'm not sure if it's still true"** → grep / read code to verify before recommending

---

This file is the project's standing-order layer. Internalize it on every session. The user has explicitly asked that you not need them to repeat any of this.

End of boot protocol.
