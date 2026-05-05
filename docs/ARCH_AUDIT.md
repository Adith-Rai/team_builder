# ARCH_AUDIT.md — Architecture-Awareness Audit Across Pipelines

**STATUS (2026-05-04, Session 50): PHASE 1 BLOCKERS RESOLVED.** All four critical
gaps (PPO-1..PPO-4) and BC-2 are fixed; both arches smoke cleanly at small scale.
See `## Session 50 — implementation outcome` at the bottom of this document for the
landed approach and verification trail. The audit body below is preserved verbatim
as historical context for the design decisions.

**Created:** Session 49 (2026-05-04). After cloud BC v10 e3 hit Elo 1135.9 (#1 all-time), a smoke
test of PPO Phase 1 from `epoch_003.pt` revealed that the trainer-side compute paths are
hardcoded to the legacy `PokeTransformer` decomposition and crash on the new
`TransformerBattlePolicy`. This document is the systematic audit that grounds the
arch-dispatch refactor needed to launch Phase 1.

**Read this first:** Session 50 will refactor based on the punch list at the bottom of
this document. Don't trust grep-based "around line X" estimates — every gap has a file:line
citation pulled from a thorough Explore-agent pass.

---

## TL;DR — Phase 1 readiness by pipeline

| Pipeline | Works on transformer ckpt? | Blockers for Phase 1 |
|----------|---------------------------|----------------------|
| **BC training** (`train_bc.py`) | ✓ Yes (proven by 3-epoch cloud run) | None. Already shipped + validated. |
| **Eval — production (smart_avg, gauntlet, ladder)** | ✓ Yes | None. Dispatch via `is_transformer_checkpoint()` already in place. |
| **Eval — legacy (`eval_h2h_v8`, `eval_report_v8`, `eval_vs_external_pool`)** | ✗ Crashes | None for Phase 1 specifically (Phase 1 uses smart_avg + ladder, not these). Future cleanup. |
| **In-loop bot eval during PPO** (called every `--eval-interval`) | ✓ Works for **legacy** init; ✗ would crash on transformer init | Need to make `train_bc.eval_vs_bots` arch-aware (hardcodes `BattleAgent`). |
| **PPO trainer collection** (`InferenceBatcher`, `V9RLPlayer`) | ✗ Crashes | **`InferenceBatcher._gpu_forward` calls `model.action_encoder` which transformer doesn't have.** |
| **PPO trainer update step** (`ppo.py:165-198`) | ✗ Crashes | Same staged decomposition (`forward_spatial` / `action_encoder` / `policy_head`) only legacy supports. |
| **PPO multiprocess paths** (`mp_collect_v2`, `mp_collect_v3`, `rl_pipeline`) | ✗ Crashes | Same hardcoded legacy `InferenceServer.forward` pattern in all three. Only matters under `--mp` / `--pipeline` flags; Phase 1 single-process spec doesn't use these but they will need fixing for any future scaling. |
| **PFSP opponent loading (self-play)** | ✓ Yes | **Already fixed in Session 49** via `make_self_play_opponent()` factory in `rl_player.py`. |

**Phase 1 unblock = 6-8 hours of focused refactor**, dominated by InferenceBatcher (2 hr) +
ppo.py update step (1 hr) + smoke + regression test on a legacy ckpt (2-3 hr) +
one or two surprise gaps caught during testing.

---

## 1. BC Pipeline Audit

### Data flow

| Step | File:line | Function | Arch-aware | Notes |
|------|-----------|----------|------------|-------|
| 1. Arg parse | `train_bc.py:386` | `--use-transformer` flag | ✓ | Explicit boolean flag drives all dispatch |
| 2. Model factory | `train_bc.py:430-461` | `if args.use_transformer:` | ✓ | Builds `TransformerBattlePolicy` or `PokeTransformer` |
| 3. Compile | `train_bc.py:446-453` | `torch.compile(spatial)` + `torch.compile(temporal)` | **PARTIAL** | Compile block lives **inside** the `if args.use_transformer` branch — legacy arch never gets compiled even with `--compile` set |
| 4. Force eval=0 for transformer | `train_bc.py:454-458` | `if args.use_transformer: args.eval_games = 0` | ✓ | Intentional — `eval_vs_bots` is legacy-only (see §3) |
| 5. DataLoader / MemmapDataset | `dataset.py:29-103, 167-285` | `MemmapDataset`, `collate_seq` | ✓ | Arch-agnostic; same memmap layout for both arches |
| 6. Forward (training) | `train_bc.py:110` | `model.forward_sequence(collated, device)` | ✓ | Both arches expose same signature + return dict |
| 7. Loss | `train_bc.py:125-137` | `masked_policy_ce`, `model.twohot_target` | ✓ | Pure functions; both models implement `twohot_target` identically |
| 8. Backward + step | `train_bc.py:149-161` | Standard PyTorch | ✓ | Arch-agnostic |
| 9. Save ckpt | `train_bc.py:560, 608, 635` | `torch.save({"arch": arch, ...})` | ✓ | All three save sites tag arch |
| 10. Resume | `train_bc.py:506-535` | `_state_dict_is_transformer` + arch-mismatch raise | ✓ | Validates arch tag matches `--use-transformer`; strips `_orig_mod.` (compile artifact) |
| 11. Eval-during-BC | `train_bc.py:603-631` → `eval_vs_bots:267-350` | `eval_vs_bots(temp_ckpt, ...)` | **NO** | Hardcodes `from battle_agent import BattleAgent` (line 288/321). Currently safe-by-design because step 4 forces eval=0 for transformer; if that guard is removed, silent crash on transformer ckpt. |

### BC arch-discriminator (verified identical in two places)

```python
# train_bc.py:38-45
def _state_dict_is_transformer(state_dict):
    return any(k.startswith(("tokenizer.", "switch_encoder.", "action_head."))
               for k in state.keys())

# battle_agent_transformer.py:187-190 (inside is_transformer_checkpoint)
return any(
    k.startswith(("tokenizer.", "switch_encoder.", "action_head."))
    for k in state_keys
)
```

These three prefixes are unique to `TransformerBattlePolicy` — legacy arch uses `move_net.`,
`switch_mlp.`, `action_encoder.`, `policy_head.` instead.

### BC gaps

| # | Severity | File:line | Problem | Fix |
|---|----------|-----------|---------|-----|
| BC-1 | **HIGH** | `train_bc.py:446-453` | `--compile` only applied to transformer arch (block is inside `if args.use_transformer:`). Legacy users miss 10-25% speedup, no warning. | Move compile block outside the if/else — both arches expose `model.spatial` + `model.temporal` |
| BC-2 | MEDIUM | `train_bc.py:288, 321` | `eval_vs_bots` imports + uses `BattleAgent` unconditionally. Safe today (forced eval=0 for transformer) but fragile. | Add `is_transformer_checkpoint()` check + dispatch to `BattleAgentTransformer` (~15 lines) |
| BC-3 | LOW | `train_bc.py:403`, `model.py:988-1011` | `add_model_args` defines only legacy CLI flags (`--d-model`, `--n-heads`, etc.). Transformer config built from disk vocabs, ignores these. User confusion only — no crash. | Add `--use-transformer` early-detection so flag set is arch-appropriate, or document |
| BC-4 | DESIGN | `model.py:818`, `model_transformer.py:2054` | Both `forward_sequence` methods wrap spatial+temporal in a Python loop, so `torch.compile` only catches the inner modules — not a bug, expected. | Document in REWRITE_DESIGN.md if not already |

**BC pipeline is in good shape for Phase 1.** None of these gaps block PPO Phase 1 from launching.

---

## 2. PPO Pipeline Audit (the big one)

### Data flow

| Step | File:line | Function | Arch-aware | Notes |
|------|-----------|----------|------------|-------|
| 1. Arg parse | `train_rl.py:47-165` | `parse_args` | **NO `--use-transformer` flag** | Discovered: train_rl.py has no equivalent of `train_bc.py:386`. Arch is detected only via `load_checkpoint`'s state-dict inference. |
| 2. Init load | `train_rl.py:574` → `ppo.py:358-379` | `load_checkpoint(path, device)` | ✓ | Reads `ckpt["arch"]` or infers from state-dict keys; dispatches to right config + class |
| 3. add_model_args | `train_rl.py:164` → `model.py:988-1011` | Legacy-only flags | NO | Same as BC-3. Cosmetic for resumes (config restored from ckpt) but blocks training-from-scratch on new arch via CLI |
| 4. dim expansion | `train_rl.py:195-208`, `ppo.py:386-401` | Pad `move_net.mlp.0.weight`, `switch_mlp.0.weight` | PARTIAL | Skips harmlessly for transformer (keys don't exist) but logic is duplicated between train_rl.py and ppo.py |
| 5. Snapshot pool seed | `train_rl.py:597, 759` | `[args.init_from]` + protected paths | ✓ | Pool storage is path-based; arch-orthogonal |
| 6. PFSP sampling | `rl_collection.py:165-175` (`pfsp_sample`) | `(1-wr)²` weighting | ✓ | Pure math; arch-orthogonal |
| 7. **Trainer player** | `rl_player.py:25-264` (`V9RLPlayer`) | `Player.choose_move` → `InferenceBatcher.submit` | NO directly, but **calls into broken InferenceBatcher** | V9RLPlayer itself doesn't call model directly — it goes through the batcher |
| 8. **Inference batcher** | `inference_batcher.py:108-209` (`_gpu_forward`) | Staged forward | **NO** | **MAJOR BLOCKER.** See full body below. |
| 9. **Self-play opponent** | `rl_player.py:267+` (`SelfPlayOpponent`, `SelfPlayOpponentTransformer`, `make_self_play_opponent`) | Factory dispatches via `is_transformer_checkpoint` | ✓ | **Fixed in Session 49.** `rl_collection.py:274` now uses the factory. |
| 10. **PPO update** | `ppo.py:154-202` | Staged forward inside per-episode loop | **NO** | **MAJOR BLOCKER.** Same `forward_spatial` / `action_encoder` / `policy_head` pattern as InferenceBatcher. |
| 11. Save ckpt | `train_rl.py:407, 818` → `ppo.py:406-422` | `save_checkpoint` tags arch from `type(cfg).__name__` | ✓ | Detects `PokeTransformerConfig` vs `TransformerConfig` |
| 12. In-loop bot eval | `train_rl.py:438-492` (`_maybe_eval`) | `from train_bc import eval_vs_bots` (line 454) | NO (delegates to broken legacy path) | Works for legacy init (Phase 1 OK), would crash on transformer init |
| 13. mp_collect_v2 | `mp_collect_v2.py:57+` (`InferenceServer`) | Same staged forward pattern, separate process | **NO** | Active under `--mp` flag (`train_rl.py:271-289`). Phase 1 doesn't use it but future scaling will |
| 14. mp_collect_v3 | `mp_collect_v3.py` | Variant of v2 | **NO** | Need to check which is the active one (probably v3 supersedes v2; v2 may be legacy) |
| 15. rl_pipeline | `rl_pipeline.py:28+` (`MPPipelineCollector`) | Same staged pattern | **NO** | Active under `--pipeline` flag (`train_rl.py:319-333`). Phase 1 spec doesn't enable; future runs may |

### The core blocker — `InferenceBatcher._gpu_forward` (lines 108-209)

This is the critical path during PPO collection. Every turn × every concurrent battle hits this
function. It's hardwired to the legacy 4-stage decomposition:

```python
# inference_batcher.py:121-184 (abbreviated to highlight the blockers)
with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.fp16):
    # PHASE 1: works on both arches (both have forward_spatial)
    spatial_out, summaries = model.forward_spatial(mega)
    # ^^ legacy returns (N, 16, D); transformer returns (N, ~220, d_model). Token 0/1
    #    is still actor/critic for both, so spatial_out[:, 0, :] survives — BUT see
    #    phase 4 below where the assumption breaks because no action_ctx.

    # PHASE 2: BLOCKER — only legacy has action_encoder
    action_ctx = model.action_encoder(                   # <-- crashes on transformer
        mega["active_move_ids"], mega["active_move_banks"],
        mega["active_move_cont"], mega["switch_ids"], mega["switch_cont"],
    )

    # PHASE 3: works on both arches (both have temporal)
    temporal_ctx = model.temporal(all_summaries.float(), seq_lens_t).to(summaries.dtype)

    # PHASE 4: BLOCKER — signature mismatch
    actor_out = spatial_out[:, 0, :]
    at = torch.cat([actor_out, temporal_ctx], dim=-1)
    at_exp = at.unsqueeze(1).expand(-1, 9, -1)
    pi_input = torch.cat([at_exp, action_ctx], dim=-1)
    logits = model.policy_head(pi_input).squeeze(-1)     # <-- legacy takes (N,9,3D)
    # Transformer's ActionHead.forward(actor_out, temporal_ctx, action_ctx, legal_mask=...)
    # takes separate args, not a concat tensor.
```

**Transformer-equivalent path** the new branch should call:

```python
# Transformer doesn't expose action_encoder; equivalent context is built inside
# its forward(). For batched inference we need to either (a) call the full
# model.forward(batch, history, history_lens) and accept it does forward_spatial
# internally a second time, or (b) have it expose _per_action_context as a public
# method and call it with the spatial_out we already have.
#
# Cleanest: option (b). Add a public method to TransformerBattlePolicy that wraps
# the existing _per_action_context (model_transformer.py:1897-1958), or just rename
# it from _per_action_context → action_encoder so the name matches.
```

### The other big block — PPO update step (`ppo.py:154-202`)

Same staged pattern, used for PPO's policy gradient + value loss computation. Per-episode
loop builds a `mega` dict over T turns, calls the same 4 stages:

```python
# ppo.py:165-193 (abbreviated)
mega = {k: _stack_field(k) for k in ep["feat_batches"][0].keys()}
spatial_out, all_summaries = model.forward_spatial(mega)               # works
action_ctx = model.action_encoder(mega["active_move_ids"], ...)        # crashes on transformer
legal_all = mega["legal_mask"]

for t in range(T):
    s = all_summaries[t:t+1].unsqueeze(0)
    summary_buf = torch.cat([summary_buf, s], dim=1)
    if summary_buf.shape[1] > 200:
        summary_buf = summary_buf[:, -200:]
    temporal_ctx = model.temporal(summary_buf)                          # works
    actor_out = spatial_out[t, 0, :]
    critic_out = spatial_out[t, 1, :]
    act_ctx = action_ctx[t]
    at = torch.cat([actor_out, temporal_ctx.squeeze(0)], dim=-1)
    at_exp = at.unsqueeze(0).expand(9, -1)
    pi_input = torch.cat([at_exp, act_ctx], dim=-1)
    logits = model.policy_head(pi_input).squeeze(-1)                    # signature mismatch
    ...
    vl = model.value_head(vi.unsqueeze(0)).squeeze(0)                   # works
```

### PPO gaps — full list

| # | Severity | File:line | Problem | Fix |
|---|----------|-----------|---------|-----|
| PPO-1 | **CRITICAL** | `inference_batcher.py:131` | `model.action_encoder()` doesn't exist on transformer. Crashes every turn. | Branch on `hasattr(model, "action_encoder")` or arch flag. Transformer path: call `model._per_action_context(spatial_out, ...)` (or expose it publicly) |
| PPO-2 | **CRITICAL** | `inference_batcher.py:184` | `model.policy_head(pi_input)` signature mismatch on transformer (legacy takes concat tensor, transformer's `ActionHead.forward` takes separate args + legal_mask kw) | Same branch. Transformer path: `model.policy_head(actor_out, temporal_ctx, action_ctx, legal_mask=mega["legal_mask"])` |
| PPO-3 | **CRITICAL** | `ppo.py:166` | Same as PPO-1 in PPO update step | Same fix |
| PPO-4 | **CRITICAL** | `ppo.py:193` | Same as PPO-2 in PPO update step | Same fix |
| PPO-5 | HIGH | `mp_collect_v2.py:57+` | Hardcoded `PokeTransformer` import + same staged forward in `InferenceServer` | Apply PPO-1/PPO-2 fix to mp_collect_v2's forward path |
| PPO-6 | HIGH | `mp_collect_v3.py` | Same as PPO-5 (need to confirm whether v3 supersedes v2 or both are active under different flags) | Same fix |
| PPO-7 | HIGH | `rl_pipeline.py:28+` | Same as PPO-5 in pipelined collector | Same fix |
| PPO-8 | MEDIUM | `train_rl.py` (no `--use-transformer` flag exists) | Cannot train transformer arch from scratch — only via `--init-from <transformer_ckpt>`. Phase 1 spec uses init-from so this is OK for Phase 1 but limits future flexibility | Add `--use-transformer` flag mirroring `train_bc.py:386`; add factory branch around `model = ... cfg = ...` construction |
| PPO-9 | MEDIUM | `train_rl.py:195-208` + `ppo.py:386-401` | Dim-expansion logic duplicated in both files. If a new field is added, only one place may be updated. | Consolidate: `_resume_from_checkpoint` should call `load_checkpoint` instead of reimplementing |
| PPO-10 | MEDIUM | `inference_batcher.py:28` | Type hint `model: PokeTransformer` (cosmetic — duck-typed at runtime) | Update to `Union[PokeTransformer, TransformerBattlePolicy]` after fixing PPO-1/PPO-2 |
| PPO-11 | MEDIUM | `rl_collection.py:27, 179` | Type hint hardcoded to `PokeTransformer` | Same as PPO-10 |
| PPO-12 | LOW | `train_rl.py:581` (compile call) | `torch.compile(model.forward_spatial)` — works on both arches but no arch-aware comment | Add comment + verify compile path |
| PPO-13 | LOW | `train_rl.py:454-457` (in-loop eval) | Calls `train_bc.eval_vs_bots` which hardcodes `BattleAgent` (BC-2). Works for Phase 1 (legacy init) but blocks transformer init | Fix BC-2; cascades automatically |
| PPO-14 | LOW | `inference_batcher.py:206` | Returns `summaries[i].float()` as the per-turn history vector. Works for both arches IF `d_temporal` consistent — verify on transformer | Verify with smoke test post-refactor |
| PPO-15 | LOW | `inference_batcher.py:147` | `model.temporal.temporal_context` — both arches expose this attr (verified) | None |

### Key insight

The "right" abstraction would have been a `ModelInterface` Protocol with required methods.
Pragmatic fix without that: add an `is_transformer_arch(model)` helper that checks
`isinstance(model, TransformerBattlePolicy)`, then branch the 2-3 critical hotspots
(InferenceBatcher, ppo.py update, mp_collect/rl_pipeline). Cleaner long-term: make
`TransformerBattlePolicy` expose legacy-compatible `action_encoder()` and `policy_head()`
adapter methods that internally call its native forms — then NO branching needed in callers,
and the new arch quacks like the old one.

**Recommendation:** ship the adapter-method approach. ~30 lines on `TransformerBattlePolicy`,
zero changes in the 4 critical hotspots beyond a sanity check. Side-benefit: future
arch swaps don't need to retouch InferenceBatcher / ppo.py / mp_collect_*.

```python
# In model_transformer.py, add to TransformerBattlePolicy:

def action_encoder(self, active_move_ids, active_move_banks, active_move_cont,
                   switch_ids, switch_cont, *,
                   spatial_out=None, our_pokemon_move_ids=None,
                   our_pokemon_species_ids=None):
    """Legacy-compatible adapter for InferenceBatcher / ppo.py.

    Caller passes the same 5 positional args legacy expects, plus the spatial_out
    + permutation context kwargs that the new arch needs (these are computed in
    the spatial pass and held by the caller). Returns (N, 9, d_model) action context
    matching legacy's ActionSlotEncoder output shape.
    """
    return self._per_action_context(
        spatial_out=spatial_out,
        our_pokemon_move_ids=our_pokemon_move_ids,
        active_move_ids=active_move_ids,
        switch_ids=switch_ids,
        our_pokemon_species_ids=our_pokemon_species_ids,
        switch_cont=switch_cont,
    )

def policy_head_compat(self, pi_input, *, legal_mask=None):
    """Legacy-compatible policy head: takes concatenated (N, 9, 3D) tensor.

    Internally splits the concat back into (actor, temporal, action_ctx) and
    calls ActionHead.forward(...).
    """
    actor_out, temporal_ctx, action_ctx = self._split_concat(pi_input)
    return self.action_head(actor_out, temporal_ctx, action_ctx, legal_mask=legal_mask)
```

Then InferenceBatcher and ppo.py only need to handle:
1. Computing `our_pokemon_move_ids` + `our_pokemon_species_ids` from the batch (one helper)
2. Routing those plus `spatial_out` into `model.action_encoder(...)` via kwargs

This is 20-30 lines of changes vs. 200-300 lines of branching.

---

## 3. Eval Pipeline Audit

### Eval scripts inventory

| Script | Purpose | Arch-aware | Phase 1 uses it? |
|--------|---------|-----------|-------------------|
| `eval_metamon_competitive.py` | smart_avg vs 4 rule bots × 16 Metamon teams | ✓ | YES (in-loop every 20 iters via `train_bc.eval_vs_bots`) |
| `eval_h2h_gauntlet.py` | Champion-vs-N opponents H2H | ✓ | NO (used for ad-hoc post-hoc) |
| `eval_elo_ladder.py` | Bradley-Terry MLE Elo ladder | ✓ | YES (final 500g add-to at end of Phase 1) |
| `eval_vs_external_pool.py` | Eval vs FP/MM/mcts adapters | ✗ | NO |
| `eval_h2h_v8.py` | Legacy round-robin H2H | ✗ | NO |
| `eval_report_v8.py` | Comprehensive eval report | ✗ | NO |
| `analyze_eval.py` | Replay-only playstyle analysis | N/A (no model calls) | Maybe (post-hoc) |

### Dispatch implementation in production scripts

All three production-grade eval scripts dispatch identically: load the ckpt dict once,
call `is_transformer_checkpoint(ckpt)`, instantiate the right Player class with `_cached_ckpt`
to skip re-reading the file:

```python
# Pattern (simplified)
ckpt = torch.load(path, map_location=device, weights_only=False)
AgentClass = BattleAgentTransformer if is_transformer_checkpoint(ckpt) else BattleAgent
player = AgentClass(checkpoint_path=path, _cached_ckpt=ckpt, ...)
```

Specific call sites:
- **`eval_metamon_competitive.py:137`** — dispatch in main loop
- **`eval_h2h_gauntlet.py:56-71`** (`make_player()` factory) — dispatch in helper
- **`eval_elo_ladder.py:267-268`** — dispatch in `PlayerPool._make_snapshot()`

### Eval gaps

| # | Severity | File:line | Problem | Phase 1 impact |
|---|----------|-----------|---------|----------------|
| EV-1 | LOW (no Phase 1 impact) | `eval_vs_external_pool.py:137-144` | Hardcoded `BattleAgent`. Crashes on transformer ckpt. | None — Phase 1 doesn't use external opponents in eval |
| EV-2 | LOW (no Phase 1 impact) | `eval_h2h_v8.py:81-91` | Hardcoded `BattleAgent` x2. Legacy script. | None — Phase 1 uses gauntlet (which dispatches), not v8 |
| EV-3 | LOW (no Phase 1 impact) | `eval_report_v8.py:87-92` | Hardcoded `BattleAgent`. Legacy script. | None |

### Eval verdict for Phase 1

**No eval-side blockers for Phase 1.** The smart_avg in-loop eval works because
Phase 1 inits from a transformer ckpt but the eval still loads it via `BattleAgent` —
WAIT, no. Reread BC-2: `train_bc.eval_vs_bots` hardcodes legacy `BattleAgent`.
Will it crash on a transformer checkpoint?

**Yes — this IS a Phase 1 issue.** The in-loop eval at `train_rl.py:_maybe_eval` saves
a temp ckpt with arch=transformer, then calls `eval_vs_bots(temp_ckpt, ...)` which
instantiates `BattleAgent(temp_ckpt)` and crashes. **Need to fix BC-2 before Phase 1.**

Updated verdict: **BC-2 IS a Phase 1 blocker** — needs same dispatch as the production
eval scripts. Add to Phase 1 unblock list.

---

## 4. Phase 1 unblock punch list (the actual work)

Ordered by execution; each estimate includes write + smoke-test + regression-on-legacy.

| Step | Effort | Files touched | What gets done |
|------|--------|---------------|----------------|
| **1. Adapter methods on TransformerBattlePolicy** | 1.5 hr | `model_transformer.py` | Add `action_encoder()` + `policy_head_compat()` methods that internally dispatch to `_per_action_context` + `ActionHead`. Net: transformer "quacks like" legacy for InferenceBatcher and ppo.py. |
| **2. Wire helper kwargs through InferenceBatcher** | 1 hr | `inference_batcher.py` | Compute `our_pokemon_move_ids` + `our_pokemon_species_ids` from `mega` (these are already in the batch dict — verify field names). Pass to `model.action_encoder(..., spatial_out=..., our_pokemon_move_ids=..., our_pokemon_species_ids=...)`. Single arch-conditional `kwargs` block; no branching of phase logic. |
| **3. Same wiring in ppo.py update step** | 1 hr | `ppo.py:154-202` | Identical fix as step 2 but in PPO update path. |
| **4. Fix `train_bc.eval_vs_bots` dispatch (BC-2)** | 30 min | `train_bc.py:267-350` | Add `is_transformer_checkpoint(temp_ckpt)` + `BattleAgent` vs `BattleAgentTransformer` dispatch (mirror `eval_metamon_competitive.py` pattern). Required for in-loop eval during Phase 1. |
| **5. Smoke: 1-iter PPO end-to-end on transformer ckpt** | 1 hr | — | `train_rl.py --init-from epoch_003.pt --n-iters 1 --games-per-iter 20 --max-concurrent 20` should complete cleanly. Watch for crashes in collection (InferenceBatcher) and update (ppo.py). Catch any surprise gaps. |
| **6. Smoke: 1-iter PPO regression on legacy ckpt** | 1 hr | — | Same command but `--init-from sp_0229.pt` to confirm legacy didn't break. |
| **7. Smoke: in-loop eval transformer** | 30 min | — | Run `train_rl.py` for 1 iter with `--eval-interval 1 --eval-games 20` from a transformer ckpt; confirm `_maybe_eval → eval_vs_bots` dispatches correctly. |
| **8. Update CLOUD_RUNBOOK + NEXT_SESSION** | 30 min | `docs/*` | Document the fixes + delete this audit's "blocker" status. |
| **(Optional) mp_collect / rl_pipeline fixes** | 2-3 hr | `mp_collect_v2.py`, `mp_collect_v3.py`, `rl_pipeline.py` | Phase 1 spec doesn't use `--mp` or `--pipeline` so deferrable. Apply same adapter-method dispatch when scaling. |

**Total: ~6.5 hours of focused work.** Could squeeze to 5 hr if smoke catches no surprises.

After step 8: Phase 1 is launchable on transformer arch. Original 210-iter spec from
`PPO_PHASED_TRAINING.md` becomes valid.

---

## 5. Test plan for the refactor (don't skip)

The legacy arch is currently the only validated PPO path (Sessions 35-39 + Session 43).
The refactor must NOT break legacy. Required regression tests after the changes above:

1. **1-iter PPO on `sp_0229.pt`** at `--max-concurrent 20 --games-per-iter 20`. Expected:
   completes without errors, snapshot saved, smart_avg eval doesn't crash.
2. **1-iter PPO on `epoch_003.pt`** at the same scale. Expected: same.
3. **Inspect `n_won_battles` / `n_tied_battles`** non-zero — i.e. games actually completed,
   not silently failed.
4. **Inspect ppo update** — `kl_div` finite, `pi_loss` finite, `value_loss` finite, no NaN.
5. **Verify in-loop eval**: smart_avg returned, JSON saved to disk.
6. **Verify ckpt dispatch**: load the saved snapshot back via `ppo.load_checkpoint` —
   round-trips correctly.

If all 6 pass on both arches, ship Phase 1.

---

## 6. Open design questions deferred to next session

1. **Anchor sample-weight floor** (raised + decided in Session 49: skip for Phase 1, watch
   smart_avg for drift, build floor in Phase 2 if drift detected). No code change needed
   for Phase 1.
2. **Whether to keep mp_collect_v2 if v3 exists** — file inspection during the refactor
   should clarify which is active. Sunset the unused one to reduce surface area.
3. **`add_model_args` refactor** — current approach (only legacy flags) is fine for
   resumes/init-from but blocks training-from-scratch on new arch. Defer until needed.
4. **Eval scripts cleanup** (EV-1/2/3) — non-blocking; address opportunistically.

---

**End of audit.** Next session: walk down §4 punch list top-to-bottom. Tests are the
exit criterion, not "looks like it should work."

---

## Session 50 — implementation outcome (2026-05-04)

**Approach landed.** §4 steps 1-7 done; §4 step 8 (this update) covers documentation.
Diverged from the audit's "adapter methods that mimic the legacy signature" recommendation
in favor of a thinner option that keeps the new arch's native API clean:

1. **`TransformerBattlePolicy.action_encoder_from_spatial(batch, spatial_out)`** —
   single new method that derives spatial-order ids via the tokenizer and dispatches
   to `_per_action_context`. Reuses an already-computed `spatial_out` rather than
   re-running the spatial pass. ~10 lines.
2. **`TransformerBattlePolicy.d_temporal = cfg.d_temporal`** in `__init__` — one
   line, mirrors the legacy `PokeTransformer` convention so trainer-side helpers
   (`InferenceBatcher`, `mp_collect_v2`, `rl_pipeline`) can size temporal-history
   buffers via `getattr(model, "d_temporal", model.cfg.d_model)`. **This was a
   silent bug the audit missed:** without it, the InferenceBatcher pre-allocates
   `all_summaries` at `d_model=256` while `forward_spatial` returns summaries at
   `d_temporal=512`, causing every collection to crash mid-game.
3. **New `arch_compat.py`** with 4 helpers (`call_action_encoder`,
   `call_policy_logits`, `call_value_logits`, `get_v_support`). Duck-typed dispatch
   on `hasattr(model, "_per_action_context")` and `hasattr(model, "tokenizer")`.
   InferenceBatcher and ppo.py both import these. ~70 lines including docs.
4. **Audit-missed gaps fixed via the same helpers:** PPO-15-equivalents in
   `inference_batcher.py:196` and `ppo.py:198` (`model.value_head(vi)` returns
   `(v_logits, value)` tuple on transformer, not the bare `v_logits` legacy returns)
   and `model.v_support` access (top-level on legacy, nested in `value_head` on
   transformer). Both were latent landmines the audit didn't enumerate.
5. **BC-2 fix in `train_bc.py:eval_vs_bots`** — load ckpt once, dispatch via
   `is_transformer_checkpoint`, mirror `eval_metamon_competitive.py:137` pattern.
   ~10 lines.

**Files touched:**
- `pokemon-ai-starter/pokemon-ai/src/model_transformer.py` — adapter method + `d_temporal` attr
- `pokemon-ai-starter/pokemon-ai/src/arch_compat.py` — NEW
- `pokemon-ai-starter/pokemon-ai/src/inference_batcher.py` — 4 lines call helpers
- `pokemon-ai-starter/pokemon-ai/src/ppo.py` — 4 lines call helpers
- `pokemon-ai-starter/pokemon-ai/src/train_bc.py` — BC-2 dispatch

**Verification (§5 test plan):**

| Test | Result |
|------|--------|
| 1. 1-iter PPO on `sp_0229.pt` (legacy) | ✓ W/L/T=8/12/0, pi=-0.0104 v=2.4270 ent=0.858 kl=0.0344, snapshot saved |
| 2. 1-iter PPO on `epoch_003.pt` (transformer) | ✓ W/L/T=6/14/0, pi=0.0715 v=9.1691 ent=1.125 kl=0.0534, snapshot saved |
| 3. Real games completed | ✓ Both arches: non-zero W/L counts, real trajectories |
| 4. pi/v/kl finite (no NaN) | ✓ Both arches |
| 5. In-loop eval on transformer init | ✓ smart_avg=71% (SH=65 SmartDmg=80 Tactical=75 Strategic=65) at 20g/bot |
| 6. Round-trip via `ppo.load_checkpoint` | ✓ Both saved snapshots load to correct class with correct cfg |

**Deferred (audit §6, no Phase 1 impact):**
- `mp_collect_v2.py`, `mp_collect_v3.py`, `rl_pipeline.py` arch dispatch (Phase 1
  doesn't use `--mp` or `--pipeline`; the same helpers in `arch_compat.py` will
  apply when those paths are needed for scaling).
- `eval_h2h_v8.py`, `eval_report_v8.py`, `eval_vs_external_pool.py` legacy script
  cleanup (production eval already arch-aware).
- `train_rl.py --use-transformer` flag for from-scratch transformer training (Phase
  1 uses `--init-from`, not from scratch).
- BC-1 (`--compile` in legacy branch), BC-3 (`add_model_args` cosmetic), BC-4 (doc).

**Ready for Phase 1 launch.** See `next-prompt.txt` operational reference for the
final command. Note that the spec's `--max-concurrent 200` is sized for cloud or a
high-VRAM workstation; on the 6 GB consumer GPU used for this session's smoke
testing, scale conc down to ~30-50 (or run on cloud).
