# NEXT_SESSION.md — Project Handover

**Last updated: 2026-05-02 (Session 48 — Week 3 plumbing + eval pipeline shipped. Three commits: (1) `1f3ec01` `--use-transformer` plumbed through `train_bc.py` + `ppo.py::load_checkpoint`; (2) `a6e6b33` `bench_bc_step.py` + Postscript I documenting the B=8 memory cliff at 5 GB peak on the 6 GB RTX 3060 (per-turn 5 ms → 145 ms) and choice of B=4 as local operating point; (3) `5f04380` `BattleAgentTransformer` (mirror of legacy `BattleAgent` for the new arch) + arch dispatch in `eval_metamon_competitive.py`, smoke-validated end-to-end (20 random-init games in 15 s). Plumbing smoke (CPU B=4, 40 batches): loss 1.66→1.61 clean, no NaN, 19,994,924 params confirmed. CUDA fp16 bench at B=4: 6-11 ms/turn, peak ≤2.74 GB, 8/8 batches stable. Throughput projection: ~9.7 hr/epoch local → 5 epochs ≈ 2 days, 10 epochs ≈ 4 days (right at the cloud-trigger boundary). All 17/17 tokenizer + 9/9 policy tests still pass. Session 49 task: launch the full 5-epoch BC at B=4 fp16 (`--workers 2 --eval-games 0 --val-ratio 0.05 --use-transformer`), run `eval_metamon_competitive.py` post-epoch on each saved checkpoint. Eval pipeline is wired and validated. See `next-prompt.txt`.)**

This is the canonical reference for resuming work on this project. It's self-contained —
read this top-to-bottom and you should have full context to execute every pending task.

Supporting documents:
- `docs/REWRITE_DESIGN.md` — **READ THIS FIRST for Sessions 46+** the architecture rewrite design (pure transformer with attribute-level tokenization)
- `docs/EXTERNAL_OPPONENTS_PHASE2.md` — protocol-bug postmortem and reproducer
- `docs/METAMON_LEARNINGS.md` — Session 37 Metamon architecture study + recommendations
- `docs/RESEARCH.md` — architecture research, published system comparisons, experiment order
- `docs/STATUS.md` — full historical narrative if deep context needed (long, usually skippable)
- `docs/CLOUD_DEPLOY.md` — cloud migration plan

---

## Session 48 final status — TL;DR for new readers

**Session 48 plumbed Week 3 sub-task 1** (REWRITE_DESIGN.md §7 Week 3,
Postscript I): `--use-transformer` flag end-to-end through `train_bc.py`
and `ppo.py::load_checkpoint`. Both arches now coexist behind a single
flag; saved checkpoints are arch-tagged so resume can't silently load
the wrong class.

**Files changed (committed at `1f3ec01`):**
- `pokemon-ai-starter/pokemon-ai/src/train_bc.py` — `+66/-10` lines.
  Adds `--use-transformer` argparse, factory branch (legacy MLP vs
  `TransformerBattlePolicy`), resume-arch guard with state-dict-key
  inference for legacy ckpts (no `arch` field), forces `--eval-games 0`
  with notice when transformer is on (option (a) per §6b: defer
  `BattleAgentTransformer` to Week 5), `arch` field added to all save
  paths (`epoch_NNN.pt`, `step_NNN.pt`, `best.pt`, `mid_step*`,
  `_eval_temp.pt`, `config.json`).
- `pokemon-ai-starter/pokemon-ai/src/ppo.py` — `+32/-1` lines.
  `load_checkpoint` reads `ckpt["arch"]` (or infers from state-dict keys)
  and dispatches between `PokeTransformer` (legacy + dim-expansion
  preserved) and `TransformerBattlePolicy` (no expansion needed).
  `save_checkpoint` adds `arch` automatically by detecting cfg class.
  Same `(model, cfg, ckpt)` return signature — `train_rl.py` doesn't
  need to change.
- `pokemon-ai-starter/pokemon-ai/src/bench_bc_step.py` — new file (~110
  lines). Standalone bench: pre-loads N batches, runs N consecutive
  forward_sequence + backward + AdamW.step(), reports per-turn ms +
  peak GPU memory. Used to find the B=8 cliff. Committable for future
  throughput regression testing.
- `pokemon-ai-starter/pokemon-ai/src/battle_agent_transformer.py` — new
  file (~165 lines, commit `5f04380`). Mirror of `battle_agent.py` for
  the new arch (REWRITE_DESIGN.md §6b.2). Same Player interface, same
  `make_features` / `build_turn_batch` / `action_to_order` flow, same
  per-battle history buffer. Loads `TransformerConfig` +
  `TransformerBattlePolicy` + `move_flag_lookup`. Refuses non-transformer
  ckpts loudly. Exports `is_transformer_checkpoint(ckpt)` as the
  dispatch oracle.
- `pokemon-ai-starter/pokemon-ai/src/eval_metamon_competitive.py` —
  `+2 lines`. `eval_ckpt_vs_bot` picks `BattleAgentTransformer` vs
  `BattleAgent` per `is_transformer_checkpoint(cached)`. Smoke: 20 games
  on a random-init transformer ckpt at concurrency 2, 15 s wall-clock,
  smart_avg 0.0% (random net vs smart bots — expected). All 4 matchups
  load + complete without exception.

**Plumbing smoke (CPU B=4, 40 batches, killed manually):**
- 19,994,924 params confirmed at training start.
- Loss decreasing: 1.66 (batch 20) → 1.61 (batch 40), accuracy ~0.32.
- No NaN, no Inf. Print + flush working. Scheduler + AdamW running.

**CUDA fp16 throughput bench (`bench_bc_step.py`):**

| B | dt (ms) by batch | per-turn | peak_mem | verdict |
|---|------------------|----------|----------|---------|
| 8 | 1263, 753, 1092, **37557, 40762** | 6.6 → **145** ms | 3.5 → **5.12 GB** | cliff |
| 4 | 855, 700, 502, 480, 730, 636, 1084, 1057 | 6-11 ms | 1.5 → 2.74 GB | stable |

The 6 GB RTX 3060 Laptop hits a memory cliff at ~5 GB peak: cuda
allocator fragmentation + spillover synchronization once the cache
nears the device ceiling. Not OOM (max 5.12/6.14), but a **30× per-turn
slowdown** that sticks for the rest of the run. B=4 stays comfortably
below the cliff with 8/8 batches healthy and loss decreasing.

**Throughput projection (B=4 fp16 local):**
- Per-turn: ~7 ms median.
- Per-epoch: 7 ms × 5M turns ≈ **9.7 hr/epoch**.
- 5 epochs (legacy convergence point): ~50 hr ≈ **2.1 days**.
- 10 epochs: ~100 hr ≈ **4.2 days**. Right at the 5-day cloud-trigger.

**Verification (post-Session-48):**
- 17/17 `test_tokenizer.py` tests still pass.
- 9/9 `test_policy.py` tests still pass.
- 50/50 strict + STAB-only `verify_move_lookup.py` still passes.
- Plumbing smoke: 40 batches CPU clean, B=4 CUDA bench stable.

**Session 49 task = continue Week 3 of the roadmap**: launch the full
5-epoch BC training at B=4 fp16 locally (or pivot to cloud per
CLOUD_DEPLOY.md if early throughput is worse than projected). Monitor
val_loss + grad-norm. Run `eval_metamon_competitive.py` on each saved
epoch checkpoint to track smart_avg. Target: ≥45% smart_avg sustained
for ≥1 epoch (matches legacy v8 BC's S39 mark of 45.1%).
See `next-prompt.txt` for the runbook.

---

## Session 47 final status — TL;DR for new readers

**Session 47 shipped Week 2 of the rewrite** (REWRITE_DESIGN.md §7 Week 2).
New code (all in `pokemon-ai-starter/pokemon-ai/src/`):

- `model_transformer.py` (added at the bottom, before the CLI section):
  - `SpatialTransformer` — 6 layers × 8 heads × d_model=256, ff_mult=4,
    dropout=0.05, pre-norm. Poke-Mask built once from the Tokenizer's
    precomputed `type_ids` buffer (decision tokens isolated from state +
    summary; actor↔critic blocked).
  - `TemporalTransformer` — 4 layers × 8 heads × d_temporal=256,
    causal, learnable position embed (200-cap). Mirrors model.py:437-511.
  - `ActionHead` — actor + temporal_ctx + per-action context → 9 logits;
    legal_mask sets illegal actions to -100.0.
  - `ValueHead` — critic + temporal_ctx → 51-bin twohot, scalar
    expectation under softmax.
  - `SwitchActionEncoder` — `(d_model + 2)→d_model` MLP. Per Postscript C3:
    consumes `(bench_pool, switch_cont[..., -2:])` for switch slots 4-8.
  - `TransformerBattlePolicy` — top-level wrapper. Forward signature
    matches `PokeTransformer.forward(batch, history, history_lens)`;
    output dict has `action_logits / value / v_logits / summary /
    spatial_output` for interface compat with battle_agent.py.
  - `forward_sequence(collated, device)` — BC-training mega-batch path
    (mirrors model.py:818-978). Flattens all valid (b, t) turns through
    the heavy spatial pass once, loops temporal+heads per-turn.
- `test_policy.py` — 9 tests on real `human_v8_100k` data, including a
  synthetic PPO step (forward_sequence + fake advantages → backward →
  AdamW step). All pass.

**Critical implementation choices** (REWRITE_DESIGN.md Postscript H):
- **Capacity bumped to 20M.** Initial Week-2 implementation came in at
  9.5M (architecturally consistent with §4 spec; estimate was high).
  After reviewing METAMON_LEARNINGS.md §1 + REWRITE_DESIGN.md §9, bumped
  `d_temporal` 256→512 and `n_summary_tokens` (K) 2→4 to match Metamon
  Small's recipe. Lifts T:S ratio 1.0×→2.0× and total to 19.99M
  (~33% above Small's 15M, as design §9 targeted). Forward speed
  essentially unchanged at single-turn (36.45 ms vs 35.70 ms prior;
  temporal at T=1 has trivial attention). Headroom for further bumps
  if Week 4 PPO plateaus: d_model 256→384, n_spatial_layers 6→8, or
  d_temporal 512→768 (Metamon Medium recipe).
- **Per-action context permutation by species / move-id matching.** The
  prompt's formula `concat(mean_pool(bench_pokemon_j_tokens), switch_cont
  [:, j, -2:])` implicitly assumed both arrays use the same ordering. They
  don't (bench is alphabetical-by-species, switch_cont is poke-env's
  `available_switches` order). Same issue for active moves vs Pokemon-order
  moves. Implementation builds a permutation on the fly via id matching;
  illegal positions (no match) default to argmax-of-all-equal=0 and the
  legal_mask masks the output.
- **Side-mask skipped.** Memmap maps unrevealed opp info to learnable
  "unknown" embeddings, so the side-mask is a no-op on top of the
  Poke-Mask.
- **Real-PPO 1-iter smoke deferred to Week 3.** Synthetic PPO step
  (`test_synthetic_ppo_step`) covers the Week-2 milestone per §7's
  alternative path: real BC batch → forward_sequence → fake advantages →
  backward → AdamW. 246/246 params updated, grad-norm 1.57, loss finite.

**Verification (post-Postscript-H bump)**:
- 9/9 `test_policy.py` tests pass at 20M params.
- 17/17 `test_tokenizer.py` tests still pass (no regressions).
- Bench: **36.45 ms** median for B=32 single-turn full-policy forward
  on RTX 3060 Laptop CUDA fp32 (target <80 ms; per-turn ~1.14 ms).
- forward_sequence + backward at B=8 (190 valid turns): ~4.8 s.
- Param breakdown: Tokenizer 1.01M / Spatial 4.74M / Temporal **12.71M** /
  Action 525K / Value 420K / Switch 67K / Summary→Temporal proj 525K
  = **19.99M total**.

**Session 48 task = Week 3 of the roadmap** (REWRITE_DESIGN.md §7
Week 3): BC training to convergence on `human_v8_100k`. Target
smart_avg ≥45% on Metamon competitive teams (matching legacy v8 BC's
mark). Plumb `--use-transformer` into `train_bc.py`. Run 1-epoch smoke
on 1k episodes, then full BC (5-10 epochs). ETA 3-7 days. Watchpoint:
if smart_avg <40% after 10 epochs → debug (tokenizer bug? MLP collapse
on novel attribute combos?). If BC takes >5 days → trigger cloud
migration (CLOUD_DEPLOY.md).

### BC vs PPO infrastructure split (read this before touching either)

**They share a model class and feature/vocab modules. Most everything
else is different.** Carrying confusion between them here in past sessions
caused real bugs, so the table is canonical.

| Component                | BC (`train_bc.py`)                               | PPO (`train_rl.py`)                                                     |
|--------------------------|--------------------------------------------------|-------------------------------------------------------------------------|
| Data source              | `MemmapDataset` (offline ~5M-turn `human_v8_100k`)  | Live trajectories from `battle_server.js` collected via `rl_collection.py` |
| Battle servers (port 9000) | Eval-only (epoch end)                          | Continuous, often `--servers 9000,9000,9000,9000` (4-wave parallelism)  |
| Opponents                | None during training; eval = SH/SmartDmg/Tactical/Strategic | PFSP pool of self-play snapshots + MCTS + FoulPlay subprocesses + Metamon subprocesses |
| External manager         | Not used                                         | `external_opponent_manager.py` + 5 defense layers (S43-S44)             |
| Loss                     | Cross-entropy on action + value-distillation     | PPO clip + KL early-stop + adaptive entropy + grad accum                |
| Forward path             | `model.forward_sequence(collated, device)` mega-batch | `model.forward(batch, history)` per-turn during collect; `forward_sequence` on update |
| Throughput cap           | GPU-bound (~25 ms/turn fp32 at 20M params)       | I/O-bound (battle wave latency; ~22 min/iter at 4-slot)                 |
| Reward / value target    | Distilled from memmap `result` field             | Reward shaper (`reward_style=terminal/sparse/dense`)                    |
| Eval methodology         | `eval_vs_bots(...)` at each epoch end (200 games × 4 bots) | `eval_metamon_competitive.py` every N iters (500 games × 4 bots)        |
| Resume                   | `--resume <ckpt>`; `--lr-restart` resets optimizer | `--init-from <ckpt>` for fresh start, `--resume <ckpt>` for full state  |
| Output                   | `data/models/bc/<run_name>/{best,epoch_N}.pt`    | `data/models/rl_v9/selfplay_v9_<ts>/{snapshot_N,final}.pt`              |

**Shared modules** (touched by both):
- `model.py` (legacy MLP) **or** `model_transformer.py` (new, Session 47).
  After Week 3 plumbing, the model class is selected by `--use-transformer`.
- `features.py` — `make_features(battle)` extracts structured features for
  live play (used by `BattleAgent` during eval and by RL collection).
- `vocab.py`, `format_config.py`, `dataset.py` (BC only — PPO collects in memory).
- `ppo.py::load_checkpoint / save_checkpoint` — both use these for I/O.
  Session 48 made these arch-aware: `load_checkpoint` reads `ckpt["arch"]`
  (or infers from state-dict key prefixes for legacy ckpts) and dispatches;
  `save_checkpoint` tags by cfg class.
- `eval_metamon_competitive.py` — standalone eval, both architectures
  use it via the `BattleAgent` / `BattleAgentTransformer` wrapper.
- `battle_server.js` — Node.js Showdown emulator on port 9000. BC eval
  connects intermittently; PPO uses continuously.

**Where the new architecture changes things in Week 3:**
1. `train_bc.py:394` constructs `cfg = config_from_args(args)` then
   `model = PokeTransformer(cfg)`. Need a factory that returns
   `TransformerBattlePolicy(TransformerConfig.with_vocab_sizes_from_disk(),
   load_move_flag_lookup(...))` when `--use-transformer` is passed.
2. `train_bc.py:441` does `model.load_state_dict(...)`. State-dict
   keys differ between architectures; the load path must check the
   checkpoint's `model_config` field to pick the right class.
3. `train_bc.py::eval_vs_bots` constructs `BattleAgent(checkpoint_path,
   ...)`. For the new arch this needs `BattleAgentTransformer` (per
   REWRITE_DESIGN.md §6b.2). **Either build it in Week 3 or use a flag
   to skip eval on transformer runs and rely on `eval_metamon_competitive.py`
   manually after each epoch.**
4. `dataset.py` is unchanged — `Tokenizer` slices the existing memmap
   format, so no regen needed.
5. `ppo.py::load_checkpoint` (line 347-371) hardcodes `PokeTransformerConfig`
   and `PokeTransformer`. Same dispatch fix as in (1) and (2).

**Files NOT to touch in Week 3** (per REWRITE_DESIGN.md §6b.5):
- `model.py`, `features.py`, `battle_agent.py`, `vocab.py`, `format_config.py`
- The existing memmap (`data/datasets/human_v8_100k`)
- `data/lookup/move_flags_v1.pt` (schema v4)

**Files Week 3 will create / edit:**
- Edit: `train_bc.py` (add `--use-transformer`, factory, save/load
  dispatch; ~50-80 lines of additions).
- Edit: `ppo.py::load_checkpoint` (add model-class dispatch; ~30 lines).
- Create: `battle_agent_transformer.py` if eval-during-BC is needed
  (~250 lines per §6b.2). Otherwise defer to Week 5.

**Week-3 cloud-trigger calculus** (revisit after Week 3 smoke):
- Local BC throughput at 20M fp32: ~25 ms/turn × 5M turns = ~35 hr/epoch.
- With `--fp16` AMP: roughly halve, ~18 hr/epoch.
- Two epochs (legacy v9 BC's convergence point at 45.1%) ≈ 36 hr local.
- Five epochs ≈ 90 hr ≈ 3.7 days. Right at the cloud-trigger threshold.
- Decision: run smoke first; pick local vs cloud based on actual
  per-batch wall-clock at the chosen `--batch-size --workers --fp16`.

---

## Session 46 final status — TL;DR for new readers

**Session 46 went through 4 audit passes after the initial Week 1
implementation.** Each pass uncovered real signal/architecture gaps and
fixed them while still inside the refactor window. Net result: a
significantly richer Tokenizer than the original §3.1 design.

**Reading order for Session 47+ (don't skim):**
1. The Postscripts (A-G) in `docs/REWRITE_DESIGN.md` — these document
   what's actually in the code, not what §3.1 says.
2. The actual `model_transformer.py` source — read top-to-bottom.
3. `test_tokenizer.py` — 17 tests that pin down the contract.

**The four audit passes (commits in order):**
- `be36415` — Week 1 baseline: Tokenizer + MoveTokenizer + 107-dim lookup.
- `77e20f7` — Cleanup: type-sort bug fix, weight init, FormatConfig
  threading, lookup schema versioning. Postscript B.
- `b4b8004` — Signal recovery: restored active/fainted/combat/toxic/
  future_sight/visibility/level/weight/height to status_token; opp threat
  computed from damage chart instead of zero parameter. Postscript C.
- `3114d10` — Active-move flag override: real PP/disabled/STAB recovered
  for our 4 active moves (closing the regression vs legacy). Postscript D.
- `dec76c0` — Field-token split into 9 thematic tokens (weather/terrain/
  our_haz/opp_haz/our_screens/opp_screens/speed_field/mechanics/progression).
  Postscript E.
- `f55576d` — Move-flag enrichment 107 → 119 dim with 12 structural flags
  (slicing/bullet/bypasssub/pulse/charge/futuremove/ignore_defensive/
  use_target_offensive/thaws_target/reflectable/gravity/sleep_usable).
  Postscript F.
- `3334e97` — ItemTokenizer + AbilityTokenizer with structural feature
  dicts parsed from Showdown items.ts / abilities.ts. Postscript G.

**The architecture as it stands (Session 47 starting state):**
- 220 tokens per turn:
  - 14 battle-state tokens (actor/critic + transition + 2 active threats
    + 9 thematic field tokens)
  - 12 Pokemon × 17 attribute tokens = 204
  - 2 summary scratch
- d_model = 256, ~1.47M params for the Tokenizer alone
- Lookup schema v4: 119-dim move flags, 4-col banks, (20,20) damage chart,
  (n_items, 8) structural feats, (n_abilities, 7) structural feats
- Bench: 30 ms median for B=32 turns. Tokenizer alone: ~5 ms.
- 17/17 tests pass. Verifier 50/50 strict + 50/50 STAB-only divergence.

**The user's philosophy (carried forward):** derive features from true
state — Showdown data + poke-env structural attributes + the recorded
memmap. NO hand-curation. If items/abilities aren't sufficiently learned
from id_embed + structural feats + 5M training samples, the principled
escalation is text-encoder over Showdown shortDesc (Metamon style), not
hand-curated effect dicts.

**Switch defensive_eff/offensive_eff (the #1 ranked feature by Session
30 weight analysis) is currently DEFERRED to the Week 2 action head**,
where the legacy ActionSlotEncoder also placed it. If you're starting
Week 2: this signal lives in `switch_cont[..., -2:]` (last 2 dims of
the 30-dim switch_cont per bench Pokemon). Feed it directly into the
per-action context for switch slots 4-8. Postscript C3 explains why
this is better than per-Pokemon spatial-token attachment.

**Session 47 task = Week 2 of the roadmap**: Spatial transformer (6L,
8H, K=2, Poke-Mask + side-mask) + Temporal transformer (4L, 8H,
d_temporal=256, causal) + Action head + Value head + 1-iter PPO smoke.
ETA 3-5 days. See `next-prompt.txt` for orienting context — it tells
the new session to read docs + code with reasoning, and to evaluate
every decision against the elite-play goal.

---

## Session 46 mid-state status — TL;DR (kept for context)

**Session 46 implemented Week 1 of the rewrite** (REWRITE_DESIGN.md §7).
New files in `pokemon-ai-starter/pokemon-ai/src/`:

- `model_transformer.py` — `TransformerConfig`, `MoveTokenizer`,
  `Tokenizer`, plus `build_move_flag_lookup` / `save_move_flag_lookup`.
  1.42M params for the Tokenizer alone (move tokenizer 152K). Imports
  `NumericalBank` from `model.py` (read-only — legacy MLP arch is
  untouched per §6b guard rail).
- `verify_move_lookup.py` — sample 50 moves from the lookup vs
  `_project_move_flags`. 50/50 strict exact match; active-path call
  diverges only at the documented STAB index (cont[12]).
- `test_tokenizer.py` — 5 integration tests on real `human_v8_100k`
  data: shape (B, 212, 256), no NaN/inf, 20 token types present,
  multi-turn forward, opp unrevealed-moves OK, type_id and
  pokemon_slot embeddings demonstrably contribute.
- `bench_tokenizer.py` — RTX 3060 Laptop benchmark: tokenizer alone
  3.42 ms median for B=32, tokenizer + dummy 6-layer spatial 28.36 ms
  median. Per-turn ~0.89 ms.
- `data/lookup/move_flags_v1.pt` — built from poke-env Move(name, gen=9)
  for 952/952 named moves, ~840 KB.

**Critical implementation choices (postscript A in REWRITE_DESIGN.md):**
- All 4 move tokens (active + team) use the lookup, not active-only
  banks. Model loses current_pp / disabled / STAB signal in the move
  token; it can still recover STAB via attention to type_token. Watch
  for BC underfitting on PP-aware play in Week 3.
- Opp active-threat token is a learnable "unknown" parameter (memmap
  doesn't store opp's active-move continuous). Our side uses the
  trailing 2 dims of `active_move_cont`.
- Type token: single MLP from 2 type embeds (resolves §10 open question).
- Position IDs (type_id, pokemon_slot, move_slot) are precomputed in
  buffers — Week 2 spatial code shouldn't recompute them.

**The 200-iter MLP PPO run completed at 21:48 on 2026-05-01.** Final
checkpoint `data/models/rl_v9_curated_pool/selfplay_v9_20260501_011537/final.pt`.
This is the "best-of-MLP-arch" deliverable per the S44 plan; new arch
will eventually run direct H2H against snapshots from this run for the
§6b.3.1 measurement.

**Session 47 task = Week 2 of the roadmap** (REWRITE_DESIGN.md §7
Week 2): SpatialTransformer (6L, 8H, d_model=256, K=2 scratch tokens,
Poke-Mask + side-mask) + TemporalTransformer (4L, 8H, d_model=256,
causal) + action head + value head, all stitched together as
`TransformerBattlePolicy`. Then 1-iter PPO smoke. ETA 3-5 days.
See `next-prompt.txt` for sub-tasks.

---

## Session 45 final status — TL;DR for new readers (kept for context)

**Session 45 produced `docs/REWRITE_DESIGN.md` (1027 lines)** — the design
document for the pure-transformer-with-attribute-tokenization rewrite of
the model architecture. All 9 required sections present: goals/non-goals,
high-level diagram, tokenization scheme (212 tokens per turn, 12 Pokemon
× 17 attribute tokens + 6 battle-state tokens), attention architecture
(6 spatial layers × 4 temporal layers × d_model=256, K=2 summary scratch
tokens), action+value heads (preserved), training pipeline (no memmap
regen needed for V1), heterogeneous opponent support (legacy BattleAgent
stays alive), 6-week implementation roadmap with explicit decision points,
12-entry risk register, Metamon comparison.

**Headline design decisions:**
- ~212 tokens per turn, ONE rich token per move (not sub-decomposed)
- d_model=256, 6 spatial / 4 temporal layers, 8 heads, K=2 summary scratch
- ~25M params target (vs current 14M), comparable to Metamon Small (15M)
- **Memmap unchanged** — tokenizer slices existing 104 GB `human_v8_100k`
  at training time (saves 1-2 weeks of regen). Move bank values for team
  moves via `(n_moves, 107)` lookup table built from poke-env at init
- New code in 3 new files: `model_transformer.py`, `features_transformer.py`,
  `battle_agent_transformer.py`. Legacy `model.py`/`features.py`/`battle_agent.py`
  stay untouched for backward compat (sp_0229, sp_2979 etc. still play)
- Direct head-to-head Elo measurement (new arch vs sp_0229) becomes a
  first-class eval method via the heterogeneous-opponent infrastructure

**Session 46 task: implement Week 1 of the roadmap** — the Tokenizer
module + integration tests. See `next-prompt.txt` for detailed
instructions. Estimated effort: 3-5 days of focused implementation.
Milestone: tokenizer runs, token shapes match spec, no NaN, forward-pass
benchmark <50ms/batch on RTX 3060 Laptop.

**Open decision points carried forward to Sessions 46-51:**
- Optimal K (summary scratch tokens) — start at 2, possibly bump to 4
- Type-token decomposition (1 vs 2 tokens) — decide in Week 1 unit tests
- Cloud BC training trigger — Week 3, if local takes >5 days
- The dispositive Week 4-5 trigger: PPO trajectory mirroring attempt-11
  (peak 64-66% then plateau) means the architectural fix wasn't decisive
  → pivot to data scaling / TTA / larger model

---

## Session 44 final status — historical TL;DR (kept for context)

**Session 44 finalized with a pivot: skip A+B (and B+ and D), commit
directly to architecture rewrite.** The careful analysis in S44 narrowed
A+B's expected impact from "real architectural fix" to "modest accuracy/PP
awareness for team moves" (1-3pt smart_avg gain at best, plausibly less,
within measurement noise). Combined with the diagnostic value being
limited (we already have strong evidence the ceiling is structural — see
"Architectural insight: MLP encoding is a fundamental compositional
bottleneck" section below), **A+B is not worth the session time. Roll its
concerns into the rewrite.**

**Session 45 task: produce REWRITE_DESIGN.md** — the design document for
the architecture rewrite (pure transformer with attribute-level
tokenization). See "The architecture rewrite plan" section below for the
high-level approach. Session 45 is design, not implementation. See
`next-prompt.txt` for detailed instructions.

**A 200-iter PPO training run completed on 2026-05-01** (resumed from
`snapshot_0024.pt` of the prior 100-iter run; both runs at attempt-11
config with Layers 1-5 defenses). Final snapshot is the "best-of-MLP-
architecture" deliverable. Smart_avg trajectory peaked at iter 119 = 66%
on Metamon competitive (sp_0229 baseline 67.8%). NOT a breakthrough —
confirms the architectural ceiling is real. Best snapshot for ladder
submission: sp_0179 or sp_0199 (smart_avg=66% peaks).

**To check the run's health:**

```bash
tail -f /c/Users/raiad/OneDrive/Desktop/team_builder/logs/external/training_curated.log
```

Watch for: iter lines (every ~7 min), `[WARN] mm-X stalled` (handled by Layer 4
auto-respawn), `EVAL:` lines (every 20 iters at 500 games × 4 bots on Metamon
competitive teams), `[watchdog]` heartbeat lines (every 60s per stuck slot — proves
poll loop alive even when an opp is stalling).

**The five defense layers active in this run** (each addresses a bug class
found in attempts 1-10 of the curated-pool work):

| Layer | What it protects | Implementation |
|---|---|---|
| 1. Forfeit filter | Spurious +1 reward from server-flipped finishes | `V9RLPlayer._finish_looks_real` requires team-fully-fainted before accepting `won=True`; spurious finishes drop the trajectory, decrement W/L, recorded as `[+Nfft]` in iter line |
| 2. Queue-restart resilience | Crashed subprocess respawn losing the trainer's enqueued teams | `--clean-on-init false` on respawn (fp/mm `accept_serve.py`); only first-spawn wipes stale `.team` files |
| 3. Dispatch watchdog | Subprocess "alive but silent" (Popen running, no battles starting) | `rl_collection.py:_play_one_opponent` wraps `send_challenges` as a task; polls `n_won+n_lost+n_tied` every 15s; cancels at 5-min stall, prints `[WARN]`, skips remaining games for that opp |
| 4. Auto-respawn on stall | Manual kill+respawn babysitting | When watchdog cancels stall, calls `external_manager.restart_subprocess(name)` → kills the stuck PID → Layer 2 monitor respawns clean for next iter |
| 5. Pool curation | S43/S44 dirty-pool dilution (auto-discovery flooding pool with current-run snapshots → cycling/specialization regression) | New flags `--pool-anchors PATH[,PATH...]` and `--pool-max-current-run N`; defaults preserve old behavior |

---

## Quick-start: how to interact with the live run

### Resume the run if you killed it (use `--resume`, all flags must match)

```bash
cd /c/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src

nohup python -u train_rl.py \
  --resume <LATEST_SNAPSHOT_FROM_RUN_DIR>.pt \
  --pool-anchors C:/Users/raiad/OneDrive/Desktop/team_builder/data/models/_archived_pre_peak/rl_v9_full_pool/selfplay_v9_20260428_030636/snapshot_0114.pt \
  --pool-max-current-run 2 \
  --eval-team-set metamon-competitive \
  --target-kl 0.02 \
  --adaptive-entropy --adaptive-entropy-high 0.85 --adaptive-entropy-min 0.005 \
  --eval-interval 20 --eval-games 500 --early-stop --early-stop-patience 3 \
  --warmup-iters 0 --n-iters <REMAINING> \
  --device cuda --servers 9000,9000,9000,9000 --fp16 --pipeline \
  --games-per-iter 200 --max-concurrent 6 \
  --reward-style terminal --lam 0.95 --ent-coef 0.01 \
  --grad-accum 1 --lr 3e-5 --win-rate-mode ema \
  --procedural-teams C:/Users/raiad/OneDrive/Desktop/team_builder/raw_data/pokemon_usage/2024-04 \
  --external-adapters external_adapters_curated.yaml \
  --out-dir data/models/rl_v9_curated_pool \
  > /c/Users/raiad/OneDrive/Desktop/team_builder/logs/external/training_curated.log 2>&1 &
```

**Critical flags that MUST be re-passed on resume** (CLI args reset every launch):
- `--pool-anchors` — without it, sp_0114 isn't protected (won't be pruned, but disappears from the iter-line output)
- `--pool-max-current-run 2` — without it, defaults to -1 (unbounded) → dilution returns
- `--eval-team-set metamon-competitive` — without it, evals revert to the noisy 70-team pool

**Resume safety verified:** model + optimizer state restore correctly; `win_rates.json`
loads from prior run dir; pool comes from saved checkpoint metadata; disk-scan
glob (`data/models/rl_v9/selfplay_v9_*`) does NOT touch `data/models/rl_v9_curated_pool/`
so we don't double-pollute. **One-shot resume safe; multi-resume accumulates ~3
stale entries per resume in pool.**

### Re-eval any checkpoint vs the 4 heuristic bots on Metamon competitive teams

```bash
cd /c/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src

python -u eval_metamon_competitive.py \
  --checkpoints \
    label1=path1.pt \
    label2=path2.pt \
  --servers 9000 --n-games 200 --concurrency 8 --device cuda \
  --out-json data/eval/<NAME>.json
```

Standalone, no trainer interaction needed. Each checkpoint × 4 bots × 200 games
= ~5 min. Battle_server must be running on port 9000.

### Spin up battle_server (the JS Showdown emulator)

```bash
C:/Users/raiad/OneDrive/Desktop/team_builder/tools/node-v20.18.1-win-x64/node.exe \
  C:/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src/battle_server.js \
  --port 9000 \
  2>&1 | tee /c/Users/raiad/OneDrive/Desktop/team_builder/logs/external/battle_server_curated2.log
```

Battle_server stays up across trainer restarts; only relaunch if its log file is
missing or it crashed. Check status: `curl -s http://127.0.0.1:9000/`.

### Per-opponent W/L analysis on a finished run

```bash
cd /c/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src
python _analyze_run_log.py /c/Users/raiad/OneDrive/Desktop/team_builder/logs/external/training_curated.log
```

Prints per-opp W/L by 10-iter decile + overall trend. Use this to diagnose
"is the model winning/losing to specific opponents?" — needed to distinguish
real cycling (consistent <50% vs past selves) from noise.

---

## What to watch for in the live run (success / failure criteria)

**Run config** (current, attempt 11 = "resumed"):
- Init: `snapshot_0024.pt` (start_iter=25; the 100-iter prior run's last snapshot)
- 200 more iters → finishes at iter 224
- `--target-kl 0.02`, `--adaptive-entropy-high 0.85`, `--ent-coef 0.01`
- `--eval-interval 20 --eval-games 500 --eval-team-set metamon-competitive`
- Pool: 16 entries (init sp_0229 + anchor sp_0114 + 5 disk-scanned peak-era + 3 leftover from prior session + 7 externals); Layer 5 prune caps current-run additions at 2

**Success criteria (we've meaningfully improved on sp_0229):**
- smart_avg ≥ 71% sustained across 3+ evals (sp_0229 baseline = 67.8% on Metamon
  competitive at 200×4 games; ±3.5pt noise floor; ≥3pt above baseline = real signal)
- W/L vs sp_0114 in iter line consistently ≥50% (we're keeping pace with the
  S43 ladder champion)
- W/L vs peak-era sp_2979/2999/3009/3029 climbing past 60% (showing real strength
  vs older snapshots)
- KL stable around 0.02 (the new target); not spiking past 0.04
- Entropy stable in [0.70, 0.85]; not drifting up unchecked like attempt 9 did

**Failure criteria (stop and re-investigate):**
- smart_avg drops by ≥3pt sustained across 2 consecutive evals from baseline
  (early-stop with patience=3 should fire at the 3rd; will halt training)
- W/L vs *own past selves from this run* (sp0014/0019/0024/...) consistently <50%
  for 3+ iters → cycling pattern, same as attempt 9
- KL averages > 0.035 multiple iters (drift exceeding our new tighter target)
- Layer 4 fires repeatedly on same MM (>3 stalls in 10 iters) → underlying
  subprocess issue not just cold-start race

**Things that are NORMAL (do not fail-stop):**
- KL ≈ 0.029-0.031 (right at 1.5× target threshold). PPO's KL early-stop
  fires often at epoch 0; that's intentional — epoch 0 fully processes before
  the gate, just prevents epoch 1+. Standard ps-ppo behavior.
- `[WARN] mm-X stalled at 0/N for 301s` ONCE per MM during cold-start
  (Layer 4 respawns; that MM gets full samples next iter)
- Per-iter W/L bouncing 45-55% — within ±3.6pt same-policy noise
- Pool size growing to ~17 over the run as Layer 5 admits 2 rolling current-run
  snapshots after each save (init + anchor + 5 disk-scan + 2 rolling + 7 ext + leftover)

---

## Session 44 history (10 attempts, what each taught us)

**Each attempt ran from a clean kill of the prior, with code/config tweaks
between launches.** The first 10 produced the iter-0-24 baseline checkpoint
that the current resumed run continues from.

### Why drop FP at this stage

**Wall-clock cost is the issue, not gradient signal.** Per S42 measurements,
an FP battle takes ~19s (100ms MCTS × ~30-60 turns + Smogon set-guessing
parsing + IPC) vs ~1.7s for MM-Minikazam. With 4 FPs at weight 0.4 each
(group total 1.6 of pool ~5.1 ≈ 32% slot share), most waves have at least
one FP slot, and a wave finishes only when the slowest slot does. Drop FPs
and wave time snaps from ~19s to ~3s. Empirically: 22 min/iter → expect
13-15 min/iter, ~35-40% wall-clock saved.

**The gradient loss is near-zero.** PFSP weights opponents by `(1-wr)²`. At
~88% loss to FP, that's `(0.12)² = 0.014` — PFSP is *already* effectively
skipping FP. We pay FP's wall-clock cost without much training payoff at
sp_0229's skill level. Dropping FP formalizes what PFSP is doing anyway.

**MCTS-style gradient diversity is preserved** by the in-process
`mcts-fast` (80ms) and `mcts-medium` (200ms) — same poke-engine MCTS
family minus FP's specific Smogon set-guessing layer.

### The single-FP detour and what it taught us (deadlock postmortem)

**Attempt 3 tried 1 FP for wall-clock savings. It deadlocked at iter 0.**
Root cause: PFSP samples slots independently with replacement; in any 6-slot
wave it can land 2+ slots on the same FoulPlayBot1 subprocess. FP's
accept-serve loop processes one battle's frames in `pokemon_battle()` —
that loop swallows the second incoming `|pm|`/challenge during frame
consumption, so the 2nd challenge never gets accepted. The 2nd
V9RLPlayer waits forever; the wave never closes. Same shape as S42 bug
#7 (cleanupBattle re-emit fix), one floor lower in the protocol stack.

**With 4 FPs, the same collision distributes across 4 different
usernames → each FP subprocess only ever sees 1 simultaneous challenge.**
Attempt 4 confirmed this works (3 iters clean before scrap). So 1 FP is
not a viable simplification for FP specifically. **MM doesn't have this
problem** — Metamon's amago wrapper has `parallel_actors=N` built in;
one MM subprocess handles many concurrent battles natively. **This means
we can scale MM count freely while needing 4 FPs minimum if FPs are in
the pool at all.** Recorded as a constraint for any future FP re-add.

### Phased curriculum design (Phase 1 launching now, Phases 2/3 documented)

The framing: as the model gets stronger, gate harder opponents in over
phases. PFSP self-corrects in the long run, but per-iter wall-clock waste
on opponents we lose 90%+ to is real.

**Phase 1 (now → ~40-50% W/L vs each MM):** drop all 4 FPs, keep 2 mcts +
4 existing MMs, add 3 new MMs for size/style diversity.
- Add **MediumIL** (BC-style, medium scale; SmallIL+MediumRL family but BC instead of RL)
- Add **MediumRL_Aug** (augmented training data; similar strength to MediumRL but different failure modes)
- Add **LargeRL** (paper, ~50M params, gen 1-4 strong)
- ~~SmallRLGen9Beta~~ — DROPPED post-smoke. Hard-codes FlashAttention via
  `gin_overrides`; flash_attn package isn't available on Windows. Failure
  is `AssertionError: Missing flash attention 2 install (pip install
  amago[flash])` on every spawn. Defer to cloud or post-flash-attn-install.

YAML group total weights: 7 MMs × 0.4 = 2.8; 2 mcts (1.0+0.5) = 1.5;
sp pool ≈ 0.5. Externals ~5:1 over self-play, no FP wall-clock tax.

**Phase 2 (when ≥40-45% W/L on each Phase 1 MM):** swap *out* 1-2 weakest
or most-redundant Phase 1 entries (probably SmallRL/SmallIL given they
overlap with MediumIL/MediumRL_Aug). Add 1-2 from `Abra`, `Kadabra2`,
`Alakazam` (~50M each, multitask gen9-trained, paper-level on gen9OU).
**Re-add 1 FP at low weight** (0.2) for MCTS-anti-search learning at a
level where we can plausibly close to ≥30% WR. PFSP will weight it
correctly given empirical W/L.

**Phase 3 (chasing ceiling — currently NOT FEASIBLE on this hardware):**
add `Superkazam` and/or `Kakuna` (~140M params each). On RTX 3060 Laptop
6 GB, this is **physically infeasible** — Kakuna alone needs ~1.5-2 GB
inference, plus current 8 MMs (~3.4 GB) + trainer (~770 MB) puts total
over the 6 GB cap. Phase 3 would require either (a) GPU upgrade to ≥10 GB
VRAM, (b) cloud deployment with bigger GPUs (already-planned per
CLOUD_DEPLOY.md), or (c) CPU offload of the big MM (10-15 min/battle =
brutal wall-clock cost).

**Mathematically, Phase 3 is also marginal even if feasible.** At Kakuna's
+409 ladder-pt gap, expected ~95% loss → PFSP weight ~0.0025 → trainer
effectively skips it. Trajectories at that loss rate are low-information
("everything I tried lost"). Cleanest path: hit Phase 2 strength locally,
graduate to cloud for Phase 3.

### VRAM budget (RTX 3060 Laptop, 6 GB total)

Baseline trainer: ~770 MB (14M-param transformer at fp16 + activations + optimizer).
Per-MM measurements (S42 logs + extrapolation by `*_agent.gin` size class):

| MM | Approx size | Notes |
|---|---|---|
| Minikazam | 173 MB | RNN, 4.7M params |
| SmallRL | 218 MB | small_agent.gin, ~10M |
| SmallIL | 215 MB | small_agent.gin, ~10M |
| MediumRL | 530 MB | medium_agent.gin, ~25M |
| MediumIL (new) | ~530 MB | medium_agent.gin |
| MediumRL_Aug (new) | ~530 MB | medium_agent.gin |
| LargeRL (new) | ~1000 MB | large_agent.gin, ~50M |
| ~~SmallRLGen9Beta~~ | ~~~250 MB~~ | DROPPED — flash_attn dependency, see above |

**Phase 1 total** (smoke-validated 7 MMs): ~3.86 GB GPU during 7-MM
smoke = ~550 MB/MM avg + Python overhead. Add 0.77 GB trainer = ~4.6 GB
under load. Buffer ~1.4 GB. Workable.
**Phase 2 total** (swap small for medium-multitask): would push to 5-5.5 GB. Need to drop 1-2 Phase 1 entries.
**Phase 3 total**: 6.5+ GB. Over cap.

### Why fresh restart from sp_0229 (not resume from attempt-4 iter 14)

Drift contamination concern. 14 iters on the FP-heavy pool would have
moved the policy slightly (KL accumulated ~0.04, ~3% of capacity), but
more importantly Adam's running moment estimates would be calibrated
against FP-flavored gradients. Resuming with the new pool would mean
5-10 iters of optimizer momentum bias before re-aligning. Cost ≈ same as
the WARMUP we'd skip by resuming, but in the form of contamination
instead of clean wait. **Clean trajectory + properly-WARMUP-calibrated
PFSP for the new MMs > 1.5 hr saved on a 30-hr run.**

### Layer 3 — dispatch resilience watchdog (added attempt 7)

Phase 1 attempts 5-6 surfaced a new failure mode: trainer-side
`send_challenges()` to MM blocks indefinitely if MM is in some
intermediate state where /pms get accepted into poke-env but
`_challenge_queue` isn't yet bound, OR when MM crashes mid-iter, OR
under the bug-B login race. None of Layer 1 (forfeit filter) or
Layer 2 (queue restart) catch this — they protect the trainer's *win
counts* and *post-crash respawn*, not the *original dispatch*.

**Fix in `rl_collection.py:_play_one_opponent`** (subprocess opp path):
wrap `send_challenges` as a task; poll `n_won + n_lost + n_tied` every
15s. If no progress for 5 min OR total dispatch time hits 30 min,
cancel the task, log a `[WARN]`, and skip remaining games for that
opponent in this iter. PFSP just gets fewer games for that opp this
iter, which is correct (low-info data isn't useful anyway). Trajectories
were never created, so no discard needed on the trajectory side.

The previous 30s post-`wait_until_ready` GUARD sleep (attempt 6) was
removed — the watchdog catches the same case more gracefully and also
catches mid-iter stalls that the GUARD couldn't.

### Layer 5 — pool curation flags (added attempt 10)

Session 44 attempt 9 ran 100 iters cleanly under the watchdog/Layer-4
defenses but **regressed on smart_avg** (peak 63% at iter 39 → 57% at
iter 99) — same dirty-pool dilution pattern as S43. The per-iter,
per-opponent table generated post-run (`_analyze_run_log.py`) showed:

- W/L vs externals: mostly flat or down (mcts-medium dropped -14pt over
  the run, the worst case).
- W/L vs *own past selves* from earlier iters: ~36-47% in the last
  decile — the model was *losing to its own past versions*, classic
  self-play cycling/drift.
- Pool grew from 8 → 27 over the run (auto-discovery added every saved
  snapshot to the pool); by iter 99, ~64% of slot share routed to
  self-play snapshots vs ~36% to externals.

**Two new CLI flags address this without breaking back-compat:**

```
--pool-anchors <path[,path...]>
    Fixed checkpoints kept in the PFSP pool throughout training (e.g.
    peak-era references). Always present; never pruned. Use absolute
    paths — relative resolves from the trainer's CWD (src/).
    Default empty = old behavior.

--pool-max-current-run N
    Cap on # self-play snapshots from the CURRENT run kept in the pool.
    When N>=0 and the run has produced more than N snapshots, oldest
    are dropped from the pool (still saved on disk). Anchors and the
    init checkpoint are not affected.
    Default -1 = unbounded (old behavior — caused S43/S44 dilution).
```

**Recommended values for any future PPO run from sp_0229 baseline:**

```
--pool-anchors C:/Users/raiad/OneDrive/Desktop/team_builder/data/models/_archived_pre_peak/rl_v9_full_pool/selfplay_v9_20260428_030636/snapshot_0114.pt
--pool-max-current-run 2
```

This produces a pool of ~9-10 entries: 1 init (sp_0229) + 1 anchor
(sp_0114) + 2 rolling current-run + 7 externals. Externals dominate
slot share ~70%, the run's gradient signal is concentrated on opponents
that matter, and self-play has stable gravity wells (init + anchor)
to prevent cycling.

**Pre-S39 snapshots (sp_2979/2999/3009/3029/3179) DO load successfully** as
SelfPlayOpponent. They have ~13.4M params vs the current 14.44M, but
`battle_agent.py:load_state_dict` has dim-expansion logic (zero-pads new
feature columns added between sp_2979 era and now) that handles the
mismatch automatically. They've been in the disk-scan pool of every PPO run
since S35.

**The S39 change was NOT an obs-feature change** (type-effectiveness slots
were already there pre-S39). It was a *weight redistribution* between the
spatial and temporal transformers: pre-S39 was 384d/384d (1:1 = 50/50);
post-S39 is 256d/512d (2:1 = 33/66). Same total ~14M params, redistributed
toward Metamon's heavier-temporal recipe (their Large is 8:1).

So sp_2979 IS a valid anchor — and the disk-scan
glob (`data/models/rl_v9/selfplay_v9_*` filtered by `MIN_SNAPSHOT_ITER=260`)
already pulls it (and 4 other peak-era snapshots) into every resume's pool
automatically. Explicit `--pool-anchors` is only needed for snapshots
NOT in `data/models/rl_v9/` (e.g. sp_0114 lives in `_archived_pre_peak/`).

**Companion hyperparameter changes for any future PPO run** (validated
on attempt 10):
- `--target-kl 0.02` (down from 0.03; OAI-Five regime, tighter drift
  bound; observed avg KL was 0.04+ at 0.03 target)
- `--adaptive-entropy-high 0.85` (down from 0.95; catches drift when
  entropy crosses 0.85, not 0.95 — attempt 9's entropy went 0.77→0.91
  without ever triggering the 0.95 default)
- `--adaptive-entropy-min 0.005` (up from 0.003; floor too low)
- `--eval-interval 5` (down from 20; faster regression detection;
  early-stop patience 3 fires by iter ~15 of regression)

### Layer 4 — automatic stuck-subprocess respawn (added attempt 8)

Layer 3 alone produced an annoying recurring pattern: once an MM
subprocess hit the stuck state, it stayed stuck for *every* subsequent
iter. Watchdog skipped it each iter, burning the 5-min stall threshold,
~5 min/iter wasted per stuck opponent. Manual `Stop-Process` of the
stuck PID worked (Layer 2's monitor respawned the pair clean), but
required babysitting.

**Fix**: when the watchdog cancels dispatch on a stall, it also calls
`external_manager.restart_subprocess(entry.key)`. The manager kills
the subprocess; Layer 2's monitor sees `rc != None` next poll cycle
and respawns clean. Next iter that opponent is fresh.

`ExternalOpponentManager.restart_subprocess(name)` is the new public
method — kills the named opponent's Popen and lets the existing
auto-restart machinery handle the respawn. Returns True/False for
the trainer's logging.

Threading: `external_manager` is now a parameter of `collect_v9`,
`_collect_data` (in train_rl.py), `_start_background_collection`,
and `BackgroundCollector.start`. Optional everywhere — callers that
don't supply it get the Layer 3 behavior (skip + WARN, no respawn).

**Bug class this catches**: any failure mode where a subprocess is
"alive but silent" — Popen running, log mtime fresh from heartbeats,
but never accepting challenges. The original symptom was MM's bug B
(`_challenge_queue` not bound on first cold start) but the fix is
generic to any flavor of subprocess wedge.

### Attempts 1-10 timeline (the chronicle that produced the resumed run)

| # | Config | Outcome | Lesson |
|---|---|---|---|
| 1 | curated YAML, lr=1e-4 default | KL discards every iter | Drop default lr to 3e-5 — bug A from S43 reconfirmed |
| 2 | 4 FP + 4 MM, lr=3e-5 | 3 iters clean, killed by user for FP-iter-time | Reverted to attempt-2 config later as known-working baseline |
| 3 | 1 FP + 4 MM | Deadlock at iter 0 | PFSP collisions on single FP subprocess; need ≥4 FPs if FPs are in the pool at all (MM is OK with N=1 because of `parallel_actors`) |
| 4 | 4 FP + 4 MM revert | 9 iters clean WARMUP; scrapped to redesign curriculum | Confirmed attempt-2 path stable, but FP wall-clock cost too high |
| 5 | 8 MMs (no FPs) — Phase 1 introduced | Hung at iter 0 | "Login race": MM logs in 10s after PFSP-ready signal, challenges arrive too early, `_challenge_queue` not bound; |pms dropped silently |
| 6 | 5 MMs + 30s GUARD sleep | Same hang at iter 0 (slot 5 stuck on MM-SmallIL) | GUARD doesn't catch the same issue mid-run; need watchdog approach |
| 7 | + Layer 3 watchdog (poll task progress, cancel at 5-min stall) | iter 0 cleared with mm-largerl skipped + WARN; Minikazam cold-start still stuck across iters | Watchdog correctly skips one iter, but doesn't recover the MM for the next |
| 8 | + Layer 4 (auto-respawn on stall WARN) | First Layer 4 fire on Minikazam → respawned; mm-largerl crashed mid-iter1 (orphaned battle); subsequent slot 5 hang | `asyncio.shield(task)` had subtle interactions; switched to plain `asyncio.sleep` + done() check |
| 9 | + simplified watchdog + Layer 4 | **Ran 100 iters cleanly.** WARN=1 (just iter 0 cold-start). But smart_avg regressed 63% iter 39 → 57% iter 99 | Run is end-to-end stable, but pool dilution still hurt smart_avg — same S43 dirty-pool pattern. → motivated Layer 5 |
| 10 | + Layer 5 (--pool-anchors, --pool-max-current-run); 2 launches (10a relative path bug, 10b absolute) | Ran 24 iters before user requested mid-run eval | Per-iter pool stayed at 11; prune fired correctly at iter 14, 19, 24 dropping oldest current-run snapshot. Smart_avg trend: 59→60→63 (improving) — but post-eval analysis showed this is within noise floor |
| **11** | **Resume from sp_0024** with 200 more iters, eval switched to Metamon competitive teams | **In progress as of 2026-05-01 ~01:15** | TBD — see "Success criteria" above |

### Smart_avg measurement: Metamon competitive teams (replaces 70-team pool)

**Why we switched:** the 70-team eval pool has a **51-pt smart_avg spread** between
TEAM_AX (81.5% on sp_0229) and TEAM_AR (30.5%). Random-team eval is dominated by
team-draw noise. Metamon's `competitive` set is 16 human-made Smogon teams used
by Kakuna/Abra (50% GXE on the human ladder) — much tighter team-quality variance.

**Same-policy variance reference (measured):** at 200 games × 4 bots, smart_avg
has ±3.6pt noise floor (sp_0229 vs sp_warmup_0009 swing was 3.6pt; same policy,
different RNG). Per-bot 7-12pt swings within noise. **Bumped to 500 games × 4 bots
in the resumed run for ±2.2pt CI** (cleaner regression detection).

**5-checkpoint baseline (from `data/eval/metamon_competitive_eval.json`):**

| Ckpt | smart_avg | Note |
|---|---|---|
| sp_0229 | 67.8% | The init/baseline |
| sp_0114 | 64.5% | S43 ladder champion (1463 SR) |
| sp_warmup_0009 | 71.4% | Same policy as sp_0229 (warmup-only); +3.6pt = pure noise |
| sp_post_0019 | 68.0% | First post-WARMUP from prior run |
| sp_best_0024 | 65.1% | Prior run's best by training-time eval (63%); ±3.5pt = within noise of baseline |

**All 5 checkpoints are statistically indistinguishable at 200×4 games.** None
clearly improved on sp_0229's 67.8%. The "regression" we attributed to attempt 9
was likely 50-70% team-noise, not policy drift. The per-opp self-play data
(losing to past selves at 36-47%) was a real cycling signal but the smart_avg
trend on the 70-team pool was overstated.

**Switch is committed**: training-time evals (`--eval-team-set metamon-competitive`)
now run on the 16-team set; standalone re-evals use `eval_metamon_competitive.py`.
All future smart_avg numbers comparable to each other; not directly comparable to
historical 70-team pool numbers.

### Phase 1 launch state (deprecated — superseded by attempt 11 resumed run)

The attempt 9 fresh-from-sp_0229 run completed 100 iters. Final snapshot
`data/models/rl_v9_curated_pool/selfplay_v9_20260429_224058/snapshot_0099.pt`
exists but is not the deliverable (smart_avg regressed). The attempt-11 resumed
run picks up from `selfplay_v9_20260430_201427/snapshot_0024.pt`, which on
Metamon competitive scored within noise of sp_0229.

---

## Architecture audit (Session 44 finalization, 2026-05-01)

**This audit was triggered by the observation that we lose 28% vs Minikazam
(4.7M params) despite our 14M params. A 3× larger model losing to a
specialized smaller one is strong evidence of implementation issues, not
just scale disadvantage.** What follows is the verified-from-code reading
of where our model takes shortcuts vs where it does things properly.

### What our model does properly (verified, no fix needed)

- **Active-move encoding has full info per move.** `ActionEncoder` (model.py:540-571)
  feeds each of the 4 currently-pickable moves through `MoveNet` individually,
  with the full 109-dim continuous block (`_project_move_flags` returns 107 +
  type_eff + opp_threat = 109 dims) AND real bank values (BP, accuracy, PP,
  priority embeddings). Each move becomes its own attention token in the
  9-token action context (4 moves + 5 switches). This is "doing things
  properly" — when picking a move, the model sees full per-move detail.
- **Per-Pokemon entity tokens for both teams.** Spatial transformer attends
  over 14 tokens: actor + critic + field + transition + 6 ours + 6 opp.
  Opp Pokemon tokens contain their full state (species, item, ability,
  HP%, types, stats) including 4 compact-encoded revealed moves.
- **Type effectiveness is computed and surfaced.** Both `_compute_type_effectiveness`
  (move vs opp active) and `_max_opp_threat` (opp moves vs us) are explicit
  inputs to the active-move encoding.

### Real shortcuts identified (need fixing)

**A. Single pooled temporal summary** (`n_summary_tokens=0` default)

When `n_summary_tokens=0` (the default we've always used), spatial transformer's
14 entity outputs collapse via attention pooling to **ONE 384-dim vector** per
turn → goes to temporal transformer. Per `METAMON_LEARNINGS.md`:

> "Metamon keeps multiple summary vectors per turn (they flatten, not pool).
>  Our pooled summary may be a silent bottleneck."

The CLI flag `--n-summary-tokens K` for K>0 is supported in `model.py` but we've
**never set it in production training**. With K=2 or K=3, K extra learnable
"scratch tokens" added to spatial input, processed through self-attention with
the 14 entities, and **bypass pooling** — temporal transformer gets K vectors
per turn instead of 1. Documented as a fix for the bottleneck.

**Cost to apply**: NO obs/data change. Model adds K × 384 new learnable
parameters (initial values of scratch tokens). Resume from current best checkpoint
+ `--n-summary-tokens 3` + `--warmup-iters 12-20` to let new params settle
before policy updates. NO BC retrain needed. ~25hr PPO continuation.

**B. Move bank embeddings zeroed for team-level moves** (model.py:701-702)

The MoveNet has 4 bank embeddings (BP, accuracy, PP, priority) using `NumericalBank`
to map integer move-property values to learnable vectors. Active-move path
(model.py:558) feeds REAL values from `_project_move_flags`. **Team-move path
(model.py:701-702) feeds `torch.zeros(...)` for ALL FOUR banks.**

What's actually lost (since the 23-dim continuous channel still carries type +
BP + category + priority for team moves):
- **Accuracy** (huge — Stone Edge 80% vs Earthquake 100% plays totally different)
- **PP** (late-game mattering)
- **Drain/recoil/heal coefficients**
- **Multihit count**
- **Move flags** (sound/punch/contact/bite/powder)
- **Crit ratio**

These features are *only* exposed through the bank/non-compact path. Currently
the model is blind to them for any move not in our currently-active 4. This is
a real loss for tactical reasoning ("opp's Pokemon X has Stone Edge, which often
misses, vs Y has Earthquake, which is reliable").

**Cost to apply**: NO obs shape change. The bank values are deterministic from
move_id (each move has fixed BP/acc/PP/prio in poke-env's data). Compute them
inline at model forward time (or in the data loader) and feed to MoveNet's
existing bank inputs. NO BC retrain needed. Model parameters unchanged in count;
just feeding non-zero values into already-existing embedding tables. PPO
continuation needed (model has to relearn how to use the suddenly-non-zero
banks; ~5-10 iters of mild adaptation expected).

**Combined A + B**: one PPO continuation run, both fixes applied together,
warmup 12-20 iters to settle the new feature regime. Resume from attempt-11's
final checkpoint. Total: ~25hr.

### Architectural insight: shared parameters generalize understanding across our/opp

Important: PokemonNet and MoveNet have SHARED PARAMETERS across our 6 and
opp 6 Pokemon (model.py:683-687 — both go through ONE pokemon_net call).
The same MLP weights process both sides. This means:

- The model's "understanding" of a Pokemon (e.g. "high attack + STAB
  super-effective move = threat") generalizes from ours to opp by design
- The model's "understanding" of a move (Earthquake's properties) is the
  same regardless of who has it

What blocks full generalization is **inconsistent input data across
contexts**, not the architecture. Active-path moves get full 109-dim
continuous + real bank values. Team-path moves (both ours and opp's) get
23-dim compact + ZEROED banks. The same MoveNet sees Earthquake-with-real-
banks (active) and Earthquake-with-zero-banks (team) as different inputs
producing different 128-dim outputs. The model is forced to learn TWO
representations of the same move depending on context.

**B's purpose isn't adding new info — it's restoring architectural
consistency the design always intended.** The MoveNet bank embedding
tables exist for a reason; we're just feeding zero where real values
should go.

### Architectural insight: MLP encoding is a fundamental compositional bottleneck

PokemonNet's pipeline `attributes → MLP → 384-dim token` must compress a
combinatorially-explosive input space (species × items × abilities × stat
spreads × movesets) into a fixed-dim representation. MLPs handle this
through **memorization** (common patterns learned via gradient descent) +
**smooth interpolation** (combinations near common patterns work via
function approximation) + **unpredictable behavior far from training
distribution** (rare/novel combinations have no guarantees).

This is the "**beginner's luck — unusual set wrecks us**" failure mode.
It is intrinsic to MLP-based encoding, not just a data scale issue. More
training data shifts which combinations are "common enough" but never
solves the open-ended combinatorial explosion.

**This insight reframes the experiment ordering**:
- **A+B** (Session 45): cheap, no retrain. Tests if architectural
  *consistency* (bank zeros fixed) unlocks improvement within current
  architecture. **Do this regardless** — it's almost free.
- **B+ and D**: incremental fixes WITHIN MLP architecture. Both are
  **subsumed by an architecture rewrite** (every attribute tokenized →
  every move feature is a token, per-move attention is automatic).
  **Skip if architecture rewrite is the next major project.**
- **Architecture rewrite**: the principled answer to the wider novel-set
  generalization problem. See "## The architecture rewrite plan" below.

### The architecture rewrite plan (post-A+B)

**Approach: pure transformer with attribute-level tokenization.** Each
attribute of the battle state becomes its own token; attention over all
tokens computes interactions at runtime rather than memorizing them in
MLP weights.

**Why this approach (vs other rewrites considered):**

| Rewrite option | Verdict | Why |
|---|---|---|
| **Pure transformer w/ attribute tokenization** | ✓ chosen | Natural extension of current arch (we already use transformers); mature tooling; attention generalizes compositionally; tractable compute at battle-state size |
| LLM approach (battle state as text) | rejected | Gimmicky; expensive inference per decision; clunky text representation when we have structured features |
| Graph neural network (GNN) | rejected | Attention does everything edges do, simpler code, better tooling. Tedious and compute-heavy for marginal gain over attention. |
| Test-time adaptation | rejected | Online weight updates DURING play. Complex, error-prone, niche research direction. |

**What changes:**
- `MoveNet`/`PokemonNet`/`FieldNet`/`ActionEncoder` → tokenizers that emit
  per-attribute tokens (not per-entity)
- Pokemon goes from 1 token containing "everything about this Pokemon" to
  ~10-15 tokens (species token + item token + ability token + 6 stat
  tokens + 4 move tokens + status token + ...)
- Battle state goes from 14 entity tokens → ~80-120 attribute tokens
- Spatial/temporal transformers operate on attribute tokens directly
- Move's BP, accuracy, flags, drain, etc. each become their own token (or
  features within move tokens) — B+ subsumed
- Per-move attention across Pokemon happens automatically — D subsumed

**What this looks like in practice (Metamon's approach):**

Metamon's modern models use Perceiver-style cross-attention for
tokenization. Looking at `metamon_ref/metamon/rl/pretrained.py`:
- Minikazam uses `MetamonPerceiverTstepEncoder` with custom tokenizer
- Larger variants (Kakuna 140M, Superkazam) use the same paradigm at
  scale + 18M training trajectories

Their 71% GXE on gen9OU human ladder comes from this approach combined
with data scale and large models. Architecture alone doesn't get them
there — but it's a necessary part of the recipe.

**Cost estimate:**
- 3-6 weeks of careful design + implementation
- New BC training run from scratch (obs space changes; old BC checkpoints
  incompatible)
- New PPO run (need to verify the new model trains stably)
- Likely needs cloud compute for the BC run at this scale

**Decision point: when to commit to rewrite**

Run A+B (Session 45). Evaluate smart_avg + per-opp trends honestly:

- **If A+B unlocks ≥3pt smart_avg improvement past sp_0229's 67.8% baseline
  sustained over 3+ evals**: ceiling was implementation, not architecture.
  Ship best snapshot to ladder. Reconsider if rewrite is worth the cost.
- **If A+B is flat (within ±2pt of baseline)**: ceiling is structural.
  Skip B+ and D. Plan the architecture rewrite as the next major project.
- **If A+B regresses**: something went wrong with the implementation;
  diagnose before any further work.

### Deferred — not needed (or only if A+B insufficient)

**C-as-cosmetic-cleanup. Drop the 84-dim padding** (split MoveNet by path or
dynamic shape).

The team-move path pads its 23-dim continuous features to 109-dim with zeros
to share one MoveNet with the active-move path (109 dims). The first
`Linear(189 → 128)` in MoveNet has 86 input columns that are always zero on
team paths. **Cost: ~11K dead parameters out of 14M = 0.08% of model.**

Verdict: cosmetic only. Skip.

**B+ (or C-as-feature-expansion). Populate the 86 padding dims with REAL features
from `_project_move_flags` for team moves.**

After B, team moves still LACK these tactical features (currently only
exposed via active-path's 107-dim `_project_move_flags`):
- drain / recoil / heal coefficients (Giga Drain heals; Brave Bird hurts you)
- multihit count (Scale Shot defeats Substitute/Sash)
- contact flag (Rocky Helmet / Static / Flame Body / Effect Spore triggers)
- protect_blocked flag (Feint, etc.)
- sound flag (pierces Substitute, blocked by Soundproof)
- punch flag (Iron Fist; Punching Glove)
- bite flag (Strong Jaw)
- powder flag (blocked by Grass types / Overcoat / Safety Goggles)
- crit ratio (Stone Edge etc. crit much more than baseline)
- secondary effect probabilities (30% paralyze from Body Slam, etc.)

The model can recover some of this via move_id embedding learning, but
that's making it memorize ~600 moves' properties from training data
instead of being given them explicitly. Wasteful and incomplete
generalization.

**B+ extends team-path encoding from 23 → 109 dims by computing
`_project_move_flags` for all team moves (not just active 4).** This
populates the 86-dim padding with real features.

Cost: features.py change to compute these per team move; BC memmap regen
(obs shape changes); BC retrain + PPO retrain.

Verdict: **natural follow-up to A+B if results don't break ceiling.**
Completes the team/active feature parity that B starts.

**D. Per-move opp tokens in spatial transformer**

Currently each opp Pokemon is ONE token containing 4 concatenated move
encodings. After B (and B+), the model has consistent rich move info inside
Pokemon tokens. PokemonNet's MLP can encode "this Pokemon with these specific
moves" patterns into the token, and Pokemon-level spatial attention can
operate over those. Move-vs-move strategic patterns emerge through
PokemonNet's MLP **memorizing** them (rather than attention **generalizing**
them).

D would let attention compute things like "Sucker Punch on opp (priority +1,
only works on attacking targets) vs my setup move (non-attacking)" directly.
Plausibly better generalization for novel cross-move interactions because
attention is permutation-equivariant and pairwise.

Cost: spatial input grows 14 → ~38 tokens, attention compute ~7× (quadratic),
implementation complexity, BC + PPO retrain.

**Verdict — D is a tweak within MLP architecture, not THE solution to
novel-set generalization.**

The deeper problem is that MLP-based Pokemon encoding (`PokemonNet`) is a
**fundamental compositional bottleneck**. The pipeline `attributes → MLP →
384-dim token` must compress combinatorially-explosive input space (species
× items × abilities × stat spreads × movesets) into fixed-dim. MLPs handle
this through memorization and smooth interpolation; behavior is
unpredictable for combinations far from training distribution. The
"beginner's luck — unusual set wrecks us" failure mode is intrinsic to MLP
encoding, not just a data scale issue.

D addresses ONE slice of this (cross-move interaction generalization). It
does NOT help with:
- Unusual ability + stat + item combinations (per-Pokemon, not cross-move)
- Novel Pokemon-Pokemon role matchups
- Rare item / EV spread / ability interactions

True solutions to the wider novel-set problem are architecturally larger:
- **Foundation model approach**: train an LLM on Pokemon battles as text;
  massive prior knowledge, native compositional generalization
- **Graph neural networks**: each attribute is a node, learned edge
  functions compute interactions
- **Pure transformer over entity-level state**: every feature is a token,
  attention does all composition
- **Test-time adaptation**: model updates understanding on-the-fly from
  early-turn observations of an unfamiliar opp

These are paradigm shifts requiring weeks-to-months of work and likely
cloud compute. Held in reserve as Session 50+ material.

**For now**: D is a "tweak within MLP architecture." Worth trying because
it's much cheaper than architectural rewrite and might capture some of the
gain. But it's a slice of compositional generalization, not the full
answer. **Reserve for "A+B+B+ done, ceiling still real, before going for
the architectural rewrite."**

### What we're doing next (Session 45 plan) — UPDATED 2026-05-01

**Pivoted from "implement A+B" to "design the architecture rewrite."**

The S44 analysis narrowed A+B's expected impact from "real architectural
fix" to "modest accuracy/PP awareness for team moves" (1-3pt smart_avg
gain at best). Combined with limited diagnostic value (we already strongly
suspect the ceiling is structural; A+B's flat result wouldn't tell us
anything new), **A+B isn't worth the session time. Roll its concerns into
the rewrite.**

**Session 45 produces `docs/REWRITE_DESIGN.md`** — a design document for
the pure-transformer-with-attribute-tokenization rewrite. This is design,
not implementation. The document should specify:

1. **Tokenization scheme**: every token type, its dim, what it represents.
   Decompose Pokemon into ~10-15 attribute tokens (species, item, ability,
   6 stat tokens, 4 move tokens, status, types). Total battle-state
   tokens ~80-120.
2. **Attention architecture**: layer count, head count, attention masking
   (e.g., player-mask similar to current PokeMask), positional/type
   embeddings to disambiguate "this is a stat token" vs "this is a move
   token".
3. **Training pipeline**: BC dataset format (does memmap need regen? probably
   yes — obs format is fundamentally different), BC training strategy,
   PPO continuation strategy, evaluation methodology.
4. **Implementation roadmap**: file-by-file what changes, in what order,
   with milestones and decision points.
5. **Risk identification**: what could go wrong (e.g., BC dataset regeneration
   takes too long, attention scaling at 80-120 tokens has issues, training
   instability with new architecture). Mitigation plans.
6. **Reference comparison**: how our design compares to Metamon's
   Perceiver-based approach. What we adopt, what we adapt, what we differ.

Session 45 is **roughly half a day to a full day of focused design work**,
not a multi-day implementation push. The output is a document that future
sessions (46, 47, ...) can implement against.

**The current attempt-11 PPO run continues until ~02:00 on 2026-05-02.** Final
snapshot is the "best-of-MLP-arch" deliverable. Best by smart_avg: sp_0179 or
sp_0199 (both 66% on Metamon competitive). Optionally ship one to PokeAgent
ladder for historical record before pivoting fully to the rewrite.

### KNOWN WEAK POINT: per-update batch size (`--games-per-iter`)

**This deserves explicit attention from any future session.**

Our `--games-per-iter 200` setting is **likely the single biggest lever we
haven't pulled** for self-play stability. Quick comparison:

| System | Trajectories per update | Env steps per update |
|---|---|---|
| Us (current) | 200 | ~6,000 |
| Us at 400 (compromise) | 400 | ~12,000 |
| Us at 1000 (cloud-scale) | 1,000 | ~30,000 |
| OpenAI Five | — | ~131,000 |
| AlphaStar | — | similarly large |
| Metamon (offline RL) | replay buffer 16M+ | not directly comparable |

**Why it matters: gradient variance scales as 1/√N.** Going 200→1000 reduces
per-update gradient noise by ~55% (factor of √5 ≈ 2.2×). All published
self-play systems use 10-100× our per-update batch size *specifically* for
stability — high gradient variance is a known cycling driver.

**Concrete symptoms of under-sampling we've observed:**

- Smart_avg variance at 200×4 games is ±3.6pt (measured: sp_0229 vs
  sp_warmup_0009 swing). This is the *eval-time* echo of the *train-time*
  variance issue.
- Per-opp W/L in iter line has only ~14 games per opp → ±13% std error.
  Can't reliably distinguish cycling (sustained <50% vs past selves) from
  noise without averaging many iters of data.
- Attempts 9 and 10's "smart_avg regression" was probably 50-70%
  measurement noise, not real policy drift — but we couldn't tell from
  inside the run because per-iter sample size is too small to detect
  small-but-real signals.

**The cost-benefit at fixed wall-clock budget:**

| Choice | Iters in 24 hr | Total games | Per-update sharpness | Per-iter sharpness |
|---|---|---|---|---|
| 200/iter @ ~7 min | ~200 | 40k | low | low |
| 400/iter @ ~14 min | ~100 | 40k | medium | better |
| 1000/iter @ ~35 min | ~40 | 40k | high | best |

**Same total games, different distribution.** With 200/iter we get many
small noisy updates; with 1000/iter we get fewer, sharper updates. The
literature (Metamon S6.4, OAI Five paper, AlphaStar) is clear that **fewer-sharper
beats many-noisier** at this scale. We've been doing many-noisier.

**Recommended next-session experiments (in order):**

1. **First: let the current attempt-11 run finish.** 200 iters at 200 games
   = 40k total games, comparable budget to the recommended alternatives.
   We'll know after this whether cycling persists at the new pool config.
2. **If cycling persists**: launch attempt-12 at `--games-per-iter 400`
   from the resumed-run's best snapshot. 2× wall-clock per iter, ~½ the
   gradient variance, sharper per-opp signal.
3. **If 400 still cycles or smart_avg flat**: this is the trigger to
   evaluate moving to cloud (per CLOUD_DEPLOY.md). Local RTX 3060 Laptop
   6 GB is the wrong compute regime for 1000+/iter experiments.

**Why we haven't already pulled this lever**: every prior PPO run inherited
`--games-per-iter 200` from the v8/v9 lineage as a wall-clock-friendly
choice when each iter took 15-30 minutes anyway (FPs in pool). With the
Phase 1 pool (no FPs) we run at ~7 min/iter, so we *could* afford bigger
batches and we've just been carrying forward the historical default. **It's
a config inertia issue, not a deliberate choice.**

**Honest framing for the new reader**: we're training "thin" (fast
iteration, noisy updates) when the published recipes say "fat" (slow
iteration, sharp updates). Worth a deliberate experiment, with cost
explicitly accounted for. The architectural ceiling at Elo 1058 (per
S35 measurements) is one ceiling, but per-update batch size is plausibly
*another* ceiling we're hitting before the architectural one — we
genuinely don't know without running the experiment.

### Other open questions / known weak points

- **`max_concurrent_battles=6`** (per-V9RLPlayer in-flight battles): historic
  self-play-only runs used 100-200. Can be bumped without triggering
  Bug B (which is a wave-slot count limit, not a per-pair count limit).
  Helps in-process opponents (self-play, mcts) but bottlenecked at
  `parallel_actors=1` for MM subprocesses. Probable 5-10% iter time
  reduction at most without changes to MM side. Low-priority.
- **`parallel_actors=1` on MM subprocesses**: we never bumped this. Set
  in `metamon_accept_serve.py` to a default `1`. Setting to `N` would
  let one MM process N concurrent challenges. Risk: more
  `_challenge_queue` race surface. Untested at scale.
- **No popart on value head**: per METAMON_LEARNINGS.md "value-scale drift
  is correlated with collapse" and popart is a cheap bolt-on. We haven't
  done it. Would help if value targets drift during long runs (Exp 4
  collapse pattern). Implementation cost: small. Untried.
- **No explicit head-to-head Elo benchmark in training loop**. All eval is
  vs heuristic bots. Real ladder rating (e.g., on PokeAgent ladder) is the
  ground truth, but it requires a separate workflow (per memory, the
  pokeBot_rescale/newopp/bc_ppo agents). Worth periodic ladder
  submissions of current best snapshot.

---

## Session 43 status (READ THIS FIRST)

**100-iter PPO training run completed end-to-end; final checkpoint
`snapshot_0114.pt` available.** Run trajectory was rocky (three relaunches
because of bugs we surfaced in production) but the final 60-iter post-
resume span ran to completion with zero subprocess crashes.

### Key training run results

Final run dir: `data/models/rl_v9_full_pool/selfplay_v9_20260428_030636/`
- Started: `--resume sp_0054.pt --lr 3e-5 --ent-coef 0.01 --warmup-iters 0`
- 60 iters complete (iter 55 → 114), final iter WR 55%
- Three EVAL points: smart_avg 58 → 60 → 59 (regressed from sp_0229's 64% baseline)
- Per-opponent class trends (early → late):
  - **FP group: 7% → 14% (+7pt, doubled WR)** — real anti-search learning
  - **MM-SmallRL: 53% → 68% (+14pt)** against peer-size 13.9M-param model
  - **MM-Minikazam: 21% → 23%** (almost no movement, sample-starved)
  - **vs sp_0229 (init): 42 → 53 → 44%** (peaked mid, did NOT durably exceed)
  - **mcts-fast: 51% → 47%** (slight decline)
- **Specialization regression on bot-eval**: lost ~5pt smart_avg vs sp_0229
- **Real gain on MCTS-search-style opponents (FP)**: doubled WR

### The trade-off this run revealed

**Different metrics tell different stories.** sp_0229 is still strongest by
smart_avg (64%). snapshot_0114 is likely stronger on the **PokeAgent live
ladder** (where MCTS bots like Foul Play are common opponents) because of
the FP/MCTS-anti-search learning. The smart_avg eval bots don't use search,
so our anti-search learning doesn't show in that metric.

**Honest verdict**: this isn't a "PPO didn't work" run, it's a "we trained
for a different skill set than the eval measures" run. Real Elo gain
(public ladder, head-to-head vs peers) should be measured directly, not
inferred from smart_avg.

### Bug fixes committed this session (validated end-to-end)

Five S43 bugs surfaced + fixed in production:

1. **Forfeit-finish filter (Layer 1, training correctness).** When any
   opponent — subprocess WS drop, in-process poke-engine panic, network
   blip — causes the local battle_server to emit `|win|RL_user` with their
   team still alive, our trajectory used to gain a spurious +1 terminal
   reward after only 1-3 turns of real play, AND the spurious win inflated
   the opponent's PFSP cumulative win rate, dragging its `(1-wr)²` weight
   below the YAML target. `V9RLPlayer._finish_looks_real` now requires one
   team to be fully fainted (or self-initiated forfeit at turn cap) before
   accepting the finish; otherwise the trajectory is dropped and the W/L
   credit is excluded from PFSP via `n_forfeit_wins/losses` counters that
   `rl_collection.py` subtracts. Generic — covers any abrupt termination,
   not just the specific FP cascading-restart path. 11 unit tests in
   `test_forfeit_filter.py`.

2. **Queue-restart resilience (Layer 2, throughput protection).** When a
   subprocess opponent crashes mid-iter, `ExternalOpponentManager` auto-
   restarts it. Pre-fix, the restarted subprocess called
   `QueueTeambuilder(clean_on_init=True)` and wiped any teams the trainer
   had pre-enqueued during the crash window — leaving the new instance
   waiting forever and the trainer's `wait_for` firing a 5-min timeout.
   Fix: `foul_play_accept_serve.py` and `metamon_accept_serve.py` accept a
   `--clean-on-init` flag (default `true`); `_spawn` injects
   `--clean-on-init false` only when `n_restarts > 0`. First start still
   wipes stale `.team` files; respawn preserves the in-flight queue.

3. **Restart-log dump (instrumentation, no behavior change).**
   `ExternalOpponentManager._monitor_loop` now emits the last 30 lines of
   the dying subprocess's log alongside the exit warning. Free
   post-mortem data if/when Bug A's still-not-root-caused
   `ConnectionClosedError` actually fires during the long run.

**The 1-iter smoke after S43's changes** (`external_adapters_all3_smoke.yaml`,
9 games, 3 concurrent) completed cleanly: `Iter 0: W/L/T=2/7/0 (22.2%),
290 steps, collect=84s, update=4s, vs sp0229=1/3 mcts-fast=1/2
foulplay-100ms=0/2 metamon-minikazam=0/2 pool=4`. Zero `[FORFEIT]`, zero
`[WARN]`, zero tracebacks, no subprocess restarts. FP and MM both logged
`queue clean_on_init=True` on first start (correct default). Layer 1 + 2
are ship-ready.

4. **MM cascade fix — three coordinated changes.** S43 attempt 3 in
   production hit a different cascade pattern: MMs sat idle (correctly
   under-sampled by PFSP after we mastered them) for >1 hour, hit
   `_INIT_RETRIES * _TIME_BETWEEN_RETRIES` timeout, raised `RuntimeError`
   inside amago's `evaluate_test`, but amago swallowed the exception
   leaving the process alive-but-silent. Manager's `Popen.poll()` returned
   None forever — never detected the dead MM. Three changes fix this:
   (a) `_INIT_RETRIES = 7200 → 86400` (12-hour idle tolerance) in
   `metamon_accept_serve.py`'s `AcceptChallengesOnLocal` class.
   (b) Heartbeat threads in both FP and MM serve scripts (1-min
   interval, daemon) print `[heartbeat HH:MM:SS]` regardless of main-loop
   state, keeping log mtime fresh whenever scheduler is alive. Eliminates
   false-positive ZOMBIE flags on legitimate idle.
   (c) Manager's `_LIVENESS_MTIME_THRESHOLD_S = 5400 → 600` (90 min →
   10 min). With heartbeats keeping logs fresh, ZOMBIE check can be
   tighter; only fires on truly hung processes.
   Validated: zero subprocess crashes across the 60-iter post-resume
   training span.

5. **Pool-curation hazard documented** — when `--resume` runs, the
   trainer's default snapshot-pool scan re-discovers every snapshot
   under `data/models/rl_v9/*` (not just current run + YAML opponents).
   Old runs' mid-training snapshots dilute the configured external-
   opponent emphasis AND introduce architecture-mismatch artifacts
   (13.38M → 14.28M).  In our run, the resume jumped pool 18→33,
   bringing in 25+ historic snapshots. This caused per-iter wr to LOOK
   like it improved (more easy self-play in the mix) while smart_avg
   regressed 6pts (specialization to the dirty pool). The fix for next
   run: physically move old run dirs out (`mv data/models/rl_v9/selfplay_v9_2026[0-3]*
   data/models/_archived_old_runs/`) before `--resume`, OR use
   `--init-from` and explicitly --warmup-iters 10. Pool curation should
   be a first-class hyperparameter.

### TL;DR for next session

The S43 production run is **DONE**. Three useful checkpoints to choose from:

| Checkpoint | Strength | When to prefer |
|---|---|---|
| `sp_0229` | smart_avg 64%, project peak (TEAM_AX 81.5%) | bot-eval scenarios, strong heuristic opponents, **best confirmed by full 70-team scan** |
| `snapshot_0114` (this run final) | FP wr 14% (+7pt), MM-SmallRL 68% (+14pt), TEAM_AK 80.0% | live ladder vs MCTS-style opponents |
| `snapshot_0099` | mid-run, partial gains | balanced fallback if 0114 has issues |

### Full 70-team scan results (S43 retrospective, 2026-04-29)

`team_selection.py` was run on all three checkpoints. Architecture-specific
team preferences are dramatic:

```
                sp_2979 peak    sp_0229 peak    sp_0114 peak
                TEAM_AU 78.5%   TEAM_AX 81.5%   TEAM_AK 80.0%
                                ★ NEW PROJECT
                                  PEAK
```

Top-5 mean savg: sp_0229=77.5%, sp_0114=76.6%, sp_2979=74.3%. **Both new-arch
checkpoints converge on TEAM_AX, TEAM_P, TEAM_AK, TEAM_G as their preferred
teams.** sp_2979's TEAM_AU dropped from 78.5% (its #1) → 59.0% on sp_0229
(rank #9, -19.5pt) — confirming team selection is architecture-specific
and old top-10 was filtered through sp_2979's preferences.

Result files:
- `data/eval/team_sel_sp_0229_full70.json` — sp_0229 full 70-team
- `data/eval/team_sel_sp_0114_full70.json` — sp_0114 full 70-team
- `team_selection_results.json` (project root) — sp_2979 full 70-team (S36)

**Highest-leverage next steps** (priority order):

1. **PokeAgent ladder submissions IN PROGRESS** (S43-end, 2026-04-29):
   - `pokeBot_rescale` running `sp_0229 + TEAM_AX` (500 games, ~17 hr)
   - `pokeBot_newopp` running `sp_0114 + TEAM_AK` (500 games, ~17 hr)
   - Both use the same account password (verified — agents share account creds)
   - Compare to S36 baseline: `sp_2979 + TEAM_T` at SR 1444 (rank #12)
   - Expected: sp_0229+TEAM_AX SR ~1470-1490 if smart_avg→Elo correlation holds

2. **Curated-pool training restart** (READY TO LAUNCH after ladder finishes):
   See "Curated-pool restart from sp_0229" section below for the validated
   recipe. Avoids the S43 dirty-pool regression (where 30+ self-play
   snapshots drowned out FP/MM signal, costing 5pt smart_avg).

3. **Architectural levers** (from prior sessions, untouched):
   - Capacity reshape further toward Metamon's 5-8:1 temporal:spatial ratio
   - BC scaling (multi-gen data, larger model)
   - Search at inference (MCTS on top of NN)

4. **Cloud deployment** — separate project. Multi-node battle_servers gives
   true 10× throughput, not the 1.3-2× local-only optimizations.

### Curated-pool restart from sp_0229 (recipe — RUN AFTER LADDER FINISHES)

Goal: train sp_0229 against an externally-dominant pool to actually beat
sp_0229's own peak by FP/MM head-to-head, without the S43 dirty-pool
specialization regression that hurt smart_avg.

**Pre-flight (must run BEFORE training launch — never with ladder running, GPU/CPU contention):**

```bash
# 1. Wait for ladder runs to fully exit
powershell.exe -Command "Get-Process python -EA SilentlyContinue"
# Should show 0 processes. If pokeBot_rescale/newopp still running, let them finish.

# 2. Move pre-peak old run snapshot dirs out of the trainer's auto-discovery path
cd C:/Users/raiad/OneDrive/Desktop/team_builder
mkdir -p data/models/_archived_pre_peak
mv pokemon-ai-starter/pokemon-ai/src/data/models/rl_v9/selfplay_v9_20260331_* data/models/_archived_pre_peak/ 2>/dev/null || true
mv pokemon-ai-starter/pokemon-ai/src/data/models/rl_v9/selfplay_v9_20260402_* data/models/_archived_pre_peak/ 2>/dev/null || true
mv pokemon-ai-starter/pokemon-ai/src/data/models/rl_v9/selfplay_v9_20260408_* data/models/_archived_pre_peak/ 2>/dev/null || true
mv pokemon-ai-starter/pokemon-ai/src/data/models/rl_v9/selfplay_v9_20260410_* data/models/_archived_pre_peak/ 2>/dev/null || true

# Move S43 prior run dir aside too (don't want sp_0114 / sp_0054 era snapshots
# in the new pool — they're the destination not the start)
mv pokemon-ai-starter/pokemon-ai/src/data/models/rl_v9_full_pool data/models/_archived_pre_peak/ 2>/dev/null || true

# 3. KEEP these (peak-era and current init):
#    pokemon-ai-starter/pokemon-ai/src/data/models/rl_v9/selfplay_v9_20260413_061236  (sp_2979 era)
#    pokemon-ai-starter/pokemon-ai/src/data/models/rl_v9/selfplay_v9_20260411_115905  (sp_2299 era)
#    pokemon-ai-starter/pokemon-ai/src/data/models/rl_v9/selfplay_v9_20260425_062416  (sp_0229 init)

# 4. Verify what'll be auto-discovered
ls pokemon-ai-starter/pokemon-ai/src/data/models/rl_v9/

# 5. Clean any external_team_queue stragglers
rm -rf data/external_team_queue/*

# 6. Confirm full curated YAML exists
ls pokemon-ai-starter/pokemon-ai/src/external_adapters_curated.yaml
```

**Terminal 1 — battle_server:**

```bash
C:/Users/raiad/OneDrive/Desktop/team_builder/tools/node-v20.18.1-win-x64/node.exe \
  C:/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src/battle_server.js \
  --port 9000 \
  2>&1 | tee /c/Users/raiad/OneDrive/Desktop/team_builder/logs/external/battle_server_curated.log
```

**Terminal 2 — curated-pool training:**

```bash
cd /c/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src
python -u train_rl.py \
  --init-from data/models/rl_v9/selfplay_v9_20260425_062416/snapshot_0229.pt \
  --device cuda --servers 9000,9000,9000,9000 --fp16 --pipeline \
  --games-per-iter 200 --max-concurrent 6 --n-iters 100 \
  --warmup-iters 12 \
  --reward-style terminal --lam 0.95 --ent-coef 0.01 --grad-accum 1 \
  --adaptive-entropy --early-stop --win-rate-mode ema \
  --adaptive-entropy-min 0.003 \
  --eval-interval 20 \
  --out-dir data/models/rl_v9_curated_pool \
  --procedural-teams /c/Users/raiad/OneDrive/Desktop/team_builder/raw_data/pokemon_usage/2024-04 \
  --external-adapters external_adapters_curated.yaml \
  2>&1 | tee /c/Users/raiad/OneDrive/Desktop/team_builder/logs/external/training_curated.log
```

**Why each delta from S43's full-pool run:**

| Param | Old | New | Why |
|---|---|---|---|
| `--external-adapters` | `external_adapters_full_pool.yaml` (MM=0.25 each) | `external_adapters_curated.yaml` (MM=0.5 each, FP=0.4) | Boost MM exposure; PFSP under-sampled mastered MMs in S43 |
| `--warmup-iters` | 5 (S43 first attempt), 0 (S43 resumed) | **12** | New pool composition + new MM emphasis = bigger value-head recalibration window. Bumped from 10 → 12 since smaller pool = higher per-opp gradient variance early. |
| `--init-from` | sp_0229 (S43) | sp_0229 (same) | Confirmed best baseline by 70-team scan; new arch peaks here. |
| `--lr` | (default 3e-5 since S43 commit) | (still default 3e-5) | Validated. |
| Pool composition | 9 externals + 30+ auto-discovered self-play | 9 externals + ~5 peak-era self-play (curated via mv) | Stop the dirty-pool dilution that hurt S43's smart_avg. |
| `--out-dir` | `rl_v9_full_pool` | `rl_v9_curated_pool` | Isolate experiment. |

**Validation step before letting it run for 30+ hr:**

After kicking off the long run, verify iter 0 looks healthy:
- Per-iter line shows MM samples for at least 2 of 4 MMs (vs S43's 0.8/iter average)
- `pool=` count is in the ~13-16 range (curated, not 30+)
- Heartbeats firing on all subprocesses (no ZOMBIE detection)
- `[WARMUP]` marker on iter 0 (confirms warmup mode)

If iter 0 looks wrong (e.g. pool > 20, or no MMs sampled), kill and re-investigate
the pre-flight cleanup. Don't let a misconfigured run consume 30 hours.

### Known unfixed bugs (deferred — DO NOT FORGET)

| ID | Symptom | Why deferred | Mitigation |
|---|---|---|---|
| **A** | FP `ConnectionClosedError: no close frame received or sent` mid-handshake at 6+ slots, prior session believed kick-on-relogin path but **unverified** | Only fires above the 4-slot validated ceiling. Generic Layer 1+2 defenses protect training correctness regardless of root cause. | Layer 2 keeps queue alive on respawn; Layer 1 drops any spurious +1 terminal; restart-log dump captures forensic data if it does fire. |
| **B** | MM `_challenge_queue` AttributeError at 6+ slots; `_handle_challenge_request` runs before MM's accept-loop binds the attribute | Lives in metamon's poke-env 0.8.3.3 fork. Fix needs vendoring a patched poke-env in `metamon_venv` or monkey-patching at MM startup — both have ongoing maintenance cost. Only fires past 4-slot. | Stay at 4-slot. ~1.3-2× speedup not worth the upstream-fork burden for a local-only optimization. |
| **C** | poke-engine `PanicException` in mcts-fast (~1% of mcts battles, e.g. "Encore should not be active when last used move is not a move") | Bug is in poke-engine's Rust validator. Upstream patch needed; rare enough not to block training. | Layer 1 drops the partial trajectory if a panic produces an abrupt finish; trainer's `wait_for` catches hung battles and moves on. |
| **D** | poke-env's `_challenge_queue` register-from-both-frames behavior was the cause of bug #8 (now fixed by sending only `|pm|`, not `|updatechallenges|`) | n/a (fixed) | n/a |

**Crucially, A and B are LOCAL-MACHINE-ONLY throughput limits.** Cloud
deployment runs N independent 4-slot nodes in parallel for N× throughput
without ever needing to fix A or B. Direct that energy at cloud planning
instead of local optimization.

---

## Session 42 status (kept for context)

**External-opponent integration is now WORKING.** All four opponent paths
play a complete battle to completion against a poke-env 0.10 sender on
the local battle_server, validated with `diag_cross_venv.py`:

| Path | Adapter | First battle clean? |
|---|---|---|
| Self-play | `SelfPlayOpponent` (in-process) | ✓ |
| Foul Play MCTS core | `mcts` / `pokeengine` (in-process via poke-engine) | ✓ |
| Real Foul Play | `foulplay` subprocess in `foul_play_venv` | ✓ (19s, 100ms MCTS) |
| Metamon (Minikazam, 4.7M) | `metamon` subprocess in `metamon_venv` | ✓ (1.7s) |

The Session 39 conclusion ("subprocess design hit a wall, need to rewrite
as in-process Players") was **wrong**. The wall was server-side protocol
bugs in `battle_server.js`, not architectural. Once fixed (Session 42),
the original subprocess design completes battles cleanly. The skeleton
we built (process manager, QueueTeambuilder, MultiSourceTeambuilder,
PoolEntry) is all in production use.

**Nine protocol bugs fixed this session — see `docs/EXTERNAL_OPPONENTS_PHASE2.md`
for the postmortem and a 5-min reproducer recipe.** Short version:
1. Per-recipient framing in `pumpPlayer` (5-frame Showdown-faithful for
   FP/MM, bundled for poke-env 0.10).
2. `/choose`, `/team`, `/switch`, `/move` all strip trailing `|<rqid>`
   before forwarding to BattleStream (real-Showdown clients always append
   it; BattleStream rejects).
3. `isShowdownFaithful` checks display name (with dash) since toId form
   strips the `MM-` dash.
4. `_factory_metamon` sets `TORCHDYNAMO_DISABLE=1` on Windows (Triton has
   no Windows wheels; Metamon's `torch.compile` crashes otherwise).
5. (Session 39, already committed) `/challenge` PM uses Showdown-standard
   8-pipe / 9-split-field format.
6. **Multi-battle: `/leave <battle-tag>` echoes `>tag\n|deinit`** so FP's
   `leave_battle` returns. Without this FP hangs after every battle. FP
   sends /leave as a global command (`|/leave battle-tag`, empty room
   prefix) — fix is in the global /leave branch, not per-battle one.
7. **Multi-battle: `cleanupBattle` re-emits pending `|pm|/challenge`** to
   users who just became idle. Pre-fix, /pms sent during a battle were
   silently consumed by `pokemon_battle` and never reached `accept_challenge`.
8. **Multi-battle: drop `|updatechallenges|`, send only `|pm|/challenge`**.
   poke-env (in metamon's 0.8.3.3 fork) registers the challenger on
   `_challenge_queue` from BOTH frames, double-populating the queue.
   Metamon's iter 1 consumed one and /accepted; iter 2 consumed the
   duplicate and sent a stale /accept that battle_server rejected with
   "No pending challenge". FP only reads `|pm|`, so it's fine without
   `|updatechallenges|`.
9. **Multi-battle: bump Metamon's openai_api 50s idle timeout to 1 hour**
   via class attribute override (`_INIT_RETRIES = 7200`,
   `_TIME_BETWEEN_RETRIES = 0.5`) on `AcceptChallengesOnLocal`. Default
   raised `RuntimeError("Agent is not challenging")` 50s after each battle
   ended if the next /challenge didn't arrive in time — PFSP gaps blew
   that. Override picked up via MRO when openai_api does
   `self._INIT_RETRIES`.

**Full PPO smoke validated.** `train_rl.py --external-adapters
external_adapters_all3_smoke.yaml --games-per-iter 9 --max-concurrent 3
--n-iters 1 --init-from <sp0229.pt>` completes cleanly:
`Iter 0: W/L/T=3/6/0 (33.3%), collect=65s, update=2s, vs sp0229=2/3
mcts-fast=1/2 foulplay-100ms=0/2 metamon-minikazam=0/2`. All 4 routes
(self-play, in-process mcts, FP subprocess, MM subprocess) drove the
real PFSP collection loop end-to-end.

**Bonus:** `battle_server.js` now prefixes log lines with
`HH:MM:SS.mmm` timestamps for future protocol debugging.

### Session 42 — Throughput pass (Task #19, partially complete)

**Full opponent pool YAML** at `external_adapters_full_pool.yaml`:
- `mcts-fast` (in-process MCTS via poke-engine, weight=1.0)
- `foulplay-100ms-1..4` (4× FP subprocesses, each weight=0.25, sums to 1.0)
- `mm-minikazam` / `mm-smallrl` / `mm-smallil` / `mm-mediumrl` (4 different
  Metamon variants, each weight=0.25). Diverse policies: small RL, medium
  RL, BC-only, gen9-tuned RL.

**VRAM measured (Session 42, replacing earlier 870MB-per-MM estimate that
was actually working-set RAM, not GPU):**
- Minikazam (4.76M params): **173 MB GPU**
- SmallRL (13.9M): **218 MB**
- SmallIL (13.7M): **215 MB**
- MediumRL (50.5M): **530 MB**
- Total 4 MMs: **~1.14 GB** (comfortable on 6GB GPU alongside 3-4GB trainer)

**Wave parallelism finding (THE key knob): `--servers 9000,9000,9000,9000`**
creates 4 server-pool entries all pointing to the same battle_server.
`rl_collection.py` processes opponents in waves of `n_servers`, so this
yields 4× wave parallelism on a single battle_server with **zero code
changes**. Validated 1-iter smoke: 18 games in **124s** (vs 236s baseline
with single-slot wave) → **1.9× speedup**. PROF wave 0 hit batch=1.4
peak=5 → InferenceBatcher actually batching now.

**6-slot and 10-slot still fail (after one fix).** Diagnosis:

1. **Login-time |pm|/challenge race (FIXED, bug #10).** When the trainer
   issues N concurrent /challenges to N opponents, FP/MM subprocesses
   may not have finished their `/trn` handshake yet. battle_server
   silently dropped the |pm| (no ws to send to). Fix: in the /trn
   handler, after registering the user, iterate `pendingChallenges` for
   any entries targeting this user and emit |pm| directly. Validated at
   4-slot (the existing ceiling), still safe — fires only when there's
   a pending challenge.

2. **6+ slot status:**
   - **FP cascading restart starvation (Bug A — FIXED, Session 43).**
     `foul_play_accept_serve.py` and `metamon_accept_serve.py` now accept
     a `--clean-on-init` flag (default `true`). `ExternalOpponentManager._spawn`
     injects `--clean-on-init false` only when `n_restarts > 0`. Result: the
     trainer's pre-enqueued teams from the crash window survive a respawn,
     so the restarted subprocess picks up the next team and the iter
     completes without a 5-min `wait_for` timeout. Also covers the
     additional training-correctness side of the same incident — see Bug 11
     below — so the iter doesn't ingest a 1–3 turn forfeit-win trajectory
     even on the very rare crash that sneaks past the queue fix.
   - **MM `_challenge_queue` AttributeError (NOT FIXED, deferred):**
     `AttributeError: 'AcceptChallengesOnLocal' object has no attribute
     '_challenge_queue'` from poke_env's `_handle_challenge_request`.
     The login-time |pm| arrives during MM setup before the agent's
     `_challenge_queue` is bound, and the handler crashes. This is in
     metamon's poke-env fork (0.8.3.3) — fix would require monkey-patching
     poke-env in `metamon_venv` or vendoring a patched fork. Only fires at
     6+ slots, and 4-slot is the validated production ceiling, so this
     is left deferred. Revisit if 4-slot throughput becomes the limiter.

11. **Forfeit-finish filter on V9RLPlayer (Session 43, training correctness).**
    When any opponent (subprocess WS drop, in-process panic, network blip)
    causes the server to emit `|win|<RL_username>` with an alive opposing
    team, poke-env flips `battle.won = True` and our trajectory ends with
    a spurious +1 terminal after only 1–3 real turns of play. PPO trained
    on those, and PFSP `(1-wr)²` weights drifted because the spurious wins
    inflated the opponent's cumulative win rate. Fix: `V9RLPlayer._finish_looks_real`
    in `rl_player.py` checks the team-fainted counts; a finish is only
    treated as real if `opp_fainted >= team_size or my_fainted >= team_size`.
    Forfeit finishes drop the trajectory from `completed_trajectories` and
    increment `n_forfeit_wins` / `n_forfeit_losses`. `rl_collection._play_one_opponent`
    subtracts those from the W/L counts that flow into `total_wins`,
    `total_losses`, and `opp_records` (PFSP). The iter summary surfaces
    forfeits as `<opp>=<W>/<G>[+Nfft]`. Generic — covers ANY abrupt
    termination, not just the FP cascading-restart path.

**4-slot remains the validated production ceiling.** 1.9× speedup over
single-slot, all 9 external entries play battles cleanly. Smoke result:
`Iter 0: W/L/T=2/16/0 (11.1%), 18 games, collect=141s` (with new login-
resend code in place — slightly slower than the 124s pre-fix run, within
PFSP noise).

### 5-iter PPO smoke (Session 42, full pool stability validation)

To confirm the full pool isn't just one-shot stable, ran 5 consecutive
iters with `external_adapters_full_pool.yaml` + `--servers 9000,9000,9000,9000`
+ `--games-per-iter 18 --max-concurrent 6 --eval-interval 999` (skip evals).
All 5 iters reached PPO update; total wall-clock ~13 min:

| Iter | W/L (%)      | Collect | Update | Notes                          |
|------|--------------|---------|--------|--------------------------------|
| 0    | 8/10 (44%)   | 123s    | 6s     | clean                          |
| 1    | 4/14 (22%)   | 123s    | 7s     | clean                          |
| 2    | 7/11 (39%)   | 162s    | 8s     | clean                          |
| 3    | 8/10 (44%)   | 125s    | 6s     | clean                          |
| 4    | 4/13 (24%)   | 235s    | 6s     | poke-engine PanicException     |

**Iter 4's outlier**: one `mcts-fast` battle hit
`PanicException('Encore should not be active when last used move is not
a move')` — a niche edge case in the poke-engine Rust library validating
illegal sim states. Trainer's `wait_for` caught it, logged
`[WARN] Timed out vs mcts-fast after 2 games`, moved on. Lost 1 game
out of ~90 (1% failure rate). Recovery is automatic — no manual
intervention needed.

**Stability evidence:**
- Subprocess-side: no FP/MM crashes or auto-restarts across all 5 iters
- Coordinator-side: 2 login-time resends (early iter), 20 cleanup-time
  resends (~4/iter, normal for the multi-battle flow)
- Memory: stable, no leaks across iters
- PFSP win-rate evolution: visible across iters (e.g. mm-smallrl was
  hard early, sampled more later by `(1-wr)^2` weighting)

**Conclusion: full pool config is production-ready for long PPO runs.**
Per-iter cost extrapolation: 200 games/iter ≈ 200/18 × 130s avg = ~24
min/iter. A 50-iter run = ~20 hr. A 100-iter run = ~40 hr. Plan
accordingly.

**Caveat: SmallRLGen9Beta and other FlashAttention-required variants
crash on Windows** (Triton-less). The 4 variants in the YAML use
VanillaAttention fallback successfully. Adding more variants requires
checking they don't hard-require flash attention.

### Production runbook (validated end of Session 43)

Full PPO training run with the multi-opponent pool, Layer 1+2 defenses
active. Two terminals, copy-pasteable.

**Pre-flight (run once before each fresh training run):**

```bash
# 1. Kill any stale processes from previous runs
powershell.exe -Command "Get-Process node, python -ErrorAction SilentlyContinue | Stop-Process -Force"

# 2. Clean ALL external opponent team queues. The broad glob below catches
#    legacy 'foulplay/' and 'metamon/' dirs from older runbooks too — without
#    this, leftover .team files from aborted runs would be consumed by the
#    new run's first iter (subprocesses use clean_on_init=true on first start
#    by default, so this is belt-and-suspenders).
rm -rf C:/Users/raiad/OneDrive/Desktop/team_builder/data/external_team_queue/*

# 3. Confirm the init checkpoint and full-pool YAML exist
ls C:/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src/data/models/rl_v9/selfplay_v9_20260425_062416/snapshot_0229.pt
ls C:/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src/external_adapters_full_pool.yaml

# 4. Confirm GPU is free (no other python processes)
nvidia-smi | head -25

# 5. (Optional but recommended) Run the forfeit-filter unit tests so a
#    future code change to _finish_looks_real doesn't silently regress
#    training-data correctness.
cd C:/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src
python test_forfeit_filter.py
```

**Terminal 1 — battle_server (single instance handles 4-slot wave):**

```bash
C:/Users/raiad/OneDrive/Desktop/team_builder/tools/node-v20.18.1-win-x64/node.exe \
  C:/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src/battle_server.js \
  --port 9000 \
  2>&1 | tee C:/Users/raiad/OneDrive/Desktop/team_builder/logs/external/battle_server.log
```

Wait for `[battle_server HH:MM:SS.mmm] Listening on port 9000` then leave it running.

**Terminal 2 — production training run:**

```bash
cd C:/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src
python -u train_rl.py \
  --init-from data/models/rl_v9/selfplay_v9_20260425_062416/snapshot_0229.pt \
  --device cuda --servers 9000,9000,9000,9000 --fp16 --pipeline \
  --games-per-iter 200 --max-concurrent 6 --n-iters 100 --warmup-iters 10 \
  --lr 3e-5 \
  --reward-style terminal --lam 0.95 --ent-coef 0.02 --grad-accum 1 \
  --adaptive-entropy --early-stop --win-rate-mode ema \
  --eval-interval 20 \
  --out-dir data/models/rl_v9_full_pool \
  --procedural-teams C:/Users/raiad/OneDrive/Desktop/team_builder/raw_data/pokemon_usage/2024-04 \
  --external-adapters external_adapters_full_pool.yaml \
  2>&1 | tee /c/Users/raiad/OneDrive/Desktop/team_builder/logs/external/training.log
```

> **CRITICAL — `--lr 3e-5` is mandatory, NOT optional.** The trainer's
> default is `1e-4`. From a sharp PPO checkpoint (sp_0229) against this
> new pool, `lr=1e-4` produces: KL early-stop firing on every iter at
> epoch 0, 10-11% per-episode KL discards, win rate drift downward
> (40% → 34% over 5 iters in the S43 first attempt), entropy collapse
> requiring adaptive entropy intervention. `lr=3e-5` is the same value
> that produced the S39 smart_avg-64% record. Do not omit this flag.

**Why `--warmup-iters 10` for this run** (was `0` for legacy `--resume`,
was `5` for the S43 first attempt):
The legacy commands use `--resume` which continues an existing PPO run
with its already-calibrated value head. Our run uses `--init-from`
against a brand-new opponent pool (9 external entries the model has
never seen). Warmup freezes the backbone+policy and trains only the
value head for the first N iters (`train_rl.py:676-684`), letting the
value function recalibrate to the new state distribution against
mcts-fast / FP / 4× MM variants before the policy starts drifting from
sp_0229. The S43 first attempt used 5 warmup iters and observed
v_loss stuck at 2.3-2.5 by iter 11 — value head not fully recalibrated.
Bumping to 10 gives the value head ~2× the runway. ~10% time overhead
(~4 hr out of 40), dramatically reduces early-iter KL spike risk
against MediumRL (4× our params). Per-iter line shows `[WARMUP]`
while frozen.

The trainer auto-spawns FP and MM subprocesses via `ExternalOpponentManager`
based on the YAML, so no separate launch step. Spawn takes ~30s
(Metamon model loads dominate; trainer waits via `wait_until_ready`).

**Why each flag:**

| Flag | Purpose |
|------|---------|
| `--init-from <pt>` | Fresh PPO state from this checkpoint (separate optimizer state). Use `--resume <pt>` instead to continue an interrupted run with optimizer state preserved. |
| **`--lr 3e-5`** | **MANDATORY for this pool — NOT the trainer default of 1e-4.** See critical callout above the table. S43 first attempt at default `1e-4` triggered 10%+ KL discards, every-iter KL early-stop, win-rate drift, and entropy collapse. `3e-5` is the lr S39 used to set the smart_avg-64% record. |
| `--servers 9000,9000,9000,9000` | THE throughput knob. 4 server-pool slots all pointing at the same battle_server → 4× wave parallelism. Tested up to 4. **6+ stalls** (see deferred bugs above). |
| `--fp16` | Mixed precision on inference + PPO. ~2× speedup, no quality regression measured. |
| `--pipeline` | Background collector overlaps next iter's collection with current iter's PPO update. Saves the update wall-time on every iter (60–80s at 200 games). Costs ~1GB extra RAM (model copy on CPU). **Recommended on for production runs.** |
| `--games-per-iter 200` | Standard for our PPO scale. Smaller = faster iters but noisier gradients. |
| `--max-concurrent 6` | Per-opponent V9RLPlayer concurrent battles. With 4-slot wave × 6 = up to 24 concurrent battles. Higher works on bigger GPUs but doesn't help here. |
| `--n-iters 100` | 100-iter run ≈ 40 hr at the measured ~24 min/iter. Adjust to budget. |
| `--warmup-iters 10` | Value-head only training for first 10 iters (frozen backbone+policy). Recalibrates v_loss to the new pool's state distribution before the policy starts drifting. S43 first attempt used 5; observed v_loss stuck at 2.3-2.5 by iter 11. Bumped to 10 for ~2× runway. |
| `--lam 0.95 --ent-coef 0.02` | Validated hyperparams from Session 39 (the smart_avg-64% record). |
| `--adaptive-entropy --early-stop` | Safeguards from Session 35. Prevent entropy collapse. **Always on** for long runs. |
| `--win-rate-mode ema` | EMA over last 50 games per opponent for PFSP. Better than cumulative for non-stationary policies. |
| `--eval-interval 20` | Eval against the 4 fixed eval bots every 20 iters. Set to 999 for smokes (skip evals entirely). |
| `--external-adapters <yaml>` | Wires the 9 external opponents into the snapshot pool with PFSP weights from the YAML. |

**What to watch in `training.log` — full visibility map:**

Every failure mode below either prints to stdout/stderr (which `tee` captures)
or routes through Python's `logging` lastResort handler (which surfaces WARNING+
to stderr). Nothing is silent in the trainer log EXCEPT a battle_server.js
crash — that lives in Terminal 1's separate log (see "Anomalies" below).

- **Healthy iter line** (one per iter, ~24 min apart):
  `[HH:MM:SS] Iter N: W/L/T=W/L/0 (X%), N steps, collect=Ts, update=Ts, pi=... v=... ent=... kl=... vs=<per-opp> pool=10`
- **Per-opponent W/L with a forfeit suffix** (Layer 1 — abrupt finish exclusion):
  `vs ... foulplay-100ms-2=3/5[+1fft] ...` — the `[+1fft]` says one battle
  ended via abrupt WS drop / poke-engine panic / network blip and was
  excluded from training AND from PFSP weight updates. **Real coverage** of
  any failure mode that ends a game with one team still alive — generic.
  0–1 per iter at 4-slot is expected. Many per iter = subprocess instability.
- **Forfeit log line (Layer 1 trigger, one per dropped trajectory):**
  `[FORFEIT] battle-gen9ou-… (T turns, won=True, opp_fainted=N, my_fainted=M) — likely WS drop, dropping trajectory + W/L credit`
- **Subprocess crash → respawn (Layer 2 + restart-log dump):**
  ```
  <opp> exited rc=N after Xs (total restarts=Y)
    --- last 30 lines of /path/to/<opp>.log ---
    [last 30 lines of FP/MM stdout, including any traceback]
    --- end tail ---
  Spawning <opp> (user=..., restarts=Y+1)
  ```
  Examine the tail to diagnose the crash — this is the only forensic data
  we capture for the still-not-root-caused Bug A.
- **MCTS-fast (in-process pokeengine) PanicException (~1% of mcts battles):**
  `WARNING:pokeengine_player:PokeEngine MCTS failed for battle-…: <err> (PanicException)` + traceback.
  pokeengine_player catches `BaseException`, logs it with `exc_info=True`,
  and returns `choose_random_move`. Battle continues from a bad state and
  usually ends abruptly — Layer 1 then drops the trajectory.
- **Battle hangs (no crash):**
  `[WARN] Timed out vs <opp> after N games`. trainer's `wait_for` cap
  is 5 min/game; firing means the opponent stopped responding without
  crashing. 1–2/iter is OK; >5/iter = real problem.
- **Other per-opponent errors:**
  `[ERROR] vs <opp>: <e>` (caught in `_play_one_opponent`),
  `[ERROR] Wave opponent failed: <e>` (caught in `asyncio.gather`).
- **poke-env's own listener errors:**
  `RL...rXXX - CRITICAL - Listen interrupted by ...` — normal at
  end-of-iter cleanup; only alarming if it fires mid-iter.
- **Resends from battle_server (normal, ~4/iter at 4-slot):**
  `[battle_server HH:MM:SS.mmm] Resent pending challenge X -> Y after battle cleanup`
- **Anomalies — investigate:**
  - `Traceback` / `FATAL` anywhere NOT inside a `--- last 30 lines ---` block
  - `KL early stop: epoch 0` on every iter (batch too small or LR too high)
  - `[FORFEIT]` firing more than ~1/iter consistently → real subprocess instability
  - All opponents simultaneously timing out → **battle_server.js may have
    crashed in Terminal 1.** Check that terminal's tee'd
    `logs/external/battle_server.log` for the proximate cause; restart it
    if needed (the trainer doesn't auto-detect this).
  - Every iter ending with the same opponent at `=0/N[+Nfft]` → that subprocess
    is in a crash loop. Restart-tail will tell you why.

**Resume an interrupted run:**

```bash
# Find latest snapshot in the run dir
ls -t data/models/rl_v9_full_pool/selfplay_v9_*/snapshot_*.pt | head -1
# Then change --init-from to --resume in the Terminal 2 command
```

`--resume` preserves PPO optimizer state (Adam momentum/variance), the
PFSP pool with its win-rate stats, the run dir. `--init-from` gives a
fresh start.

**Stop cleanly:**

`Ctrl-C` the trainer in Terminal 2. The `ExternalOpponentManager` will
SIGTERM all FP/MM subprocesses on shutdown. Final checkpoint saves to
`<out_dir>/.../final.pt`. Battle_server in Terminal 1 keeps running
(no state to clean up); Ctrl-C separately or leave for next run.

**Throughput levers in priority order if 24 min/iter is too slow:**

1. `--pipeline` (already in command above; saves ~60s/iter, free).
2. Lower FP `search_time_ms` from 100 to 50 in `external_adapters_full_pool.yaml` (FP stays strong, ~2× faster per battle).
3. Replace 2 of 4 `foulplay-100ms-N` entries with additional `mcts-fast` entries — in-process, no subprocess serialization.
4. `--mp` flag for multiprocess collection (untested with current setup, exists in code).
5. **DO NOT** push `--servers` past 4 slots (cascading FP restart + MM `_challenge_queue` race; see "Known unfixed bugs" near top of file). Layer 1+2 protect correctness if you do try, but throughput regresses from cascading timeouts.

### TL;DR for Session 42 (superseded — see Session 43 TL;DR at top of file)

**Two pending items**, in priority order:

1. **Real long PPO run with the validated full-pool YAML.** Use the
   command in section above with `--servers 9000,9000,9000,9000
   --external-adapters external_adapters_full_pool.yaml`. Extrapolating
   from 18 games / 124s, a 200-game iter ≈ 23 min. Tight but workable.
   Optionally: `--pipeline` to overlap PPO update with next collection
   (saves ~5s/iter). Tune `--games-per-iter` if collection time is
   prohibitive — 100 games/iter gives ~12 min iters.

2. **Optional**: investigate the 6+ slot wave stall. Likely either
   (a) `_play_one_opponent`'s send_challenges call concurrency causing
   /pm crossings on battle_server; (b) battle_server's pendingChallenges
   map being keyed by (challenger,) so simultaneous challenges from N
   different RL players to the same opponent overwrite each other (no
   — different challengers, different keys); (c) ws backpressure on the
   single battle_server from N concurrent senders. Diagnostic: enable
   `BS_TRACE_USER=foulplaybot1` and watch where the deadlock forms.

3. **Incremental Elo on the 6-snapshot "Useful" set** (~2.5 hr,
   orthogonal): New BC + sp_0029 + sp_0099 + sp_0159 + sp_0219 + sp_0229.
   Adds incrementally to `data/eval/elo_exp5_FINAL.json`. Anchors on
   sp2979 (Elo 1058, still on disk). Settles whether the Session 39
   smart_avg gain (64% all-time peak) is a real Elo gain.

### PPO run on the human-data BC — DONE

Run dir: `data/models/rl_v9/selfplay_v9_20260425_062416/` (resumed from
the prev run's snapshot_0029). 200 iters total (iters 30 through 229).
Settings: `lr=3e-5`, `lam=0.95`, `ent-coef=0.02`, `--adaptive-entropy`,
`--win-rate-mode ema`, procedural teams. **Note: lr=1e-4 caused 80%+
KL discard in the original attempt; lr=3e-5 is the right starting point
from a 45% BC.** All eleven 20-iter evals plus the manual final eval:

| Iter | SH | SmDmg | Tact | Strat | smart_avg |
|---|---|---|---|---|---|
| 19 | 52 | 56 | 48 | 55 | 53.0 |
| 39 | 70 | 58 | 62 | 58 | 62.0 |
| 59 | 57 | 54 | 60 | 56 | 56.0 |
| 79 | 66 | 66 | 56 | 56 | 61.0 |
| 99 | 63 | 56 | 57 | 60 | 59.0 |
| 119 | 61 | 63 | 48 | 60 | 58.0 |
| 139 | 59 | 61 | 60 | 60 | 60.0 |
| 159 | 60 | 68 | 60 | 65 | 63.0 |
| 179 | 62 | 60 | 59 | 64 | 61.0 |
| 199 | 62 | 57 | 57 | 54 | 58.0 |
| **219** | **61** | **68** | **60** | **66** | **64.0** ← all-time peak |
| 229 (final, manual) | 67 | 61 | 56 | 69 | 63.4 |

Best snapshots: `sp_0219.pt` (peak 64%) and `sp_0229.pt` (final 63.4%).
Both are statistically tied; either could be best in actual Elo terms.

**Internal PFSP win rates** (after fixing my misread of the EMA file
format — `[wr*eff_games, eff_games]`, divide by 50 for %):
- vs **sp2979** (the historical Elo 1058 ceiling): we win **63%**
- vs **BC_base** (our new 45% BC): 86%
- vs hardest recent self-play (sp_0189): 35%
- vs prev-run snapshots (sp_0004…sp_0019): 75-76%
**i.e. we are solidly past sp2979 in actual head-to-head play, even
though smart_avg vs the bot anchors plateaued at ~60%.** Smart_avg has
likely saturated against the eval bots; the policy is genuinely
stronger than the metric resolves.

### External opponents — SKELETON DONE, ADAPTER WORK PENDING

What's committed (~5 commits, see git log):
- `external_opponent_manager.py` — process manager + YAML config (still
  useful if/when a bot doesn't have a clean Python entrypoint)
- `external_opponents_example.yaml` — config schema
- `metamon_local.yaml` — agents config for `metamon.rl.self_play.serve_model`
- `team_generator.py` — added `MultiSourceTeambuilder` + `StaticTeamPool`
- `battle_server.js` — `e01a37f` patches the /challenge PM to standard
  Showdown 9-field format (was 5-field inline; broke Foul Play's parser)

What we *learned* and *did not* commit upstream:
- `foul_play_ref/fp/websocket_client.py` needs a 1-line patch to do
  case-insensitive `_to_id`-style username comparison. Documented in
  `docs/EXTERNAL_OPPONENTS_PHASE2.md` step 1. Lives only in the local
  clone of foul_play_ref/, since that's an upstream repo we don't
  vendor.
- `poke-engine` (FoulPlay's MCTS dependency) builds via PEP 517 + Rust
  toolchain transparently; no manual Rust install needed.

What's broken / re-thought:
- Subprocess-bot + send_challenges design doesn't drive battles to
  completion. Phase 2 runbook now leads with the recommended adapter
  approach (Python Player wrapping `poke-engine` directly).

### Analysis tooling — DONE

`analyze_eval.py` extended with 5 features (3 commits this session):
- `--iter-trajectory <run_dir>` — per-iter playstyle table across
  `replays_iter*/` subdirs
- `--by-opponent` — split trajectory rows by opponent bot
- `--team-usage` — lead mon, send-out frequency, faint order
- `--decision-quality` — attacks-into-immune, switches-into-SE,
  setup-at-low-HP, recovery-at-full-HP rates
- `--human-baseline N` — stream N HF replays at >=1500 Elo, compute
  same playstyle profile, add as comparison column

Findings from running these on the current PPO run (iter 39 → iter 179):
- Voluntary switches halved (12% → 6.5%) — model committing more
- SE rate +4.5pts (49 → 53.5%) — better type targeting
- Setup-at-low-HP −5pts (19 → 14%) — fewer wasted setups
- **Surprise:** immune-attack rate went UP (11.4% → 13.5%). Worth
  flagging — possibly because the policy converged on a smaller set of
  preferred moves and some of those happen to hit immunities. Doesn't
  block training but is a real failure mode to investigate.

### Pipeline + infrastructure fixes (committed)

- `replay_to_memmap.py`: stale `make_obs_mask_and_slots` import removed;
  `MemmapV8Writer.finalize()` now actually trims raw .npy files instead
  of just claiming to. Reclaimed 63.67 GB from the existing
  `human_v8_100k/` memmap (174→111 GB).
- BC reshape on `human_v8_100k` produced `best.pt` at smart_avg 45.1%
  (epoch 2, 14.28M params, 2:1 temporal:spatial reshape). This is the
  BC base for the PPO run above.

---

## Session 38 status (kept for context)

**Session 38 completed two workstreams: (a) BC reshape trained to completion on the old
bot-generated memmap, reaching smart_avg 21.2% at epoch 4 (below the historical 22-26%
ceiling — a small regression); (b) regenerated the training data from 100k ≥1500-Elo
human Showdown replays, with current 109/30 features (type-effectiveness dims now present).
The reshape hypothesis is untested against the better data source, and that's what next
session should do.** Full details below.

### TL;DR for next session
1. Launch BC reshape training on the new human memmap (`data/datasets/human_v8_100k/`)
   — same reshape config, same 10 epochs. See "Next concrete step" below.
2. If `best.pt` beats 22-26% smart_avg, the capacity reshape + human data combination
   works. If flat at ~21%, we're at the BC ceiling and PPO is the next lever.

### BC reshape run on bot data — DONE (final result)

Run dir: `data/models/bc/v8_bc_20260423_124909/` (restarted after the mid-run BSOD kill).
All 10 epochs completed. Best at **epoch 4: smart_avg 21.2%** (SH=21, SmartDmg=20,
Tactical=22, Strategic=22). Per-epoch curve showed classic BC overfit signature:
val_acc climbed monotonically 0.671→0.713 while smart_avg plateaued at ~20%.

| Epoch | SH | SmDmg | Tact | Strat | avg | val_acc |
|---|---|---|---|---|---|---|
| 0 | 21 | 22 | 18 | 18 | 20.0 | 0.671 |
| 1 | 18 | 24 | 20 | 18 | 19.6 | 0.675 |
| 2 | 27 | 23 | 16 | 16 | 20.6 | 0.683 |
| 3 | 20 | 19 | 20 | 20 | 19.8 | 0.699 |
| **4** | **21** | **20** | **22** | **22** | **21.2** | **0.706** ← best.pt |
| 5 | 16 | 26 | 22 | 19 | 20.6 | 0.706 |
| 6 | 15 | 17 | 22 | 12 | 16.4 | 0.710 |
| 7 | 21 | 18 | 20 | 20 | 20.0 | 0.713 |
| 8 | 26 | 20 | 16 | 20 | 20.5 | 0.712 |
| 9 | 22 | 17 | 18 | 23 | 20.2 | 0.713 |

Interpretation: this was on the STALE memmap (107/28 dims, 4 missing type_eff scalars
silently zero-padded). Reshape underperformed old BC_base's 22-26% ceiling, but the gap
is within 200-game eval noise (±2-3% per bot, ±1.5% on the 4-bot mean). Not a verdict
on the reshape itself — the real test is on the human memmap with full 109/30 features.

### New human memmap — DONE

`data/datasets/human_v8_100k/` — generated via `replay_to_memmap.py` streaming from HF
dataset `jakegrigsby/metamon-raw-replays`, filtered to gen9ou, min_rating=1500.

- **100,000 replays** accepted (from 457,739 streamed; 78% skipped by rating filter)
- **199,919 episodes** (both perspectives via `--log-both=True`)
- **5,084,603 records** (14× the old bot memmap's 361k)
- **move_cont_dim=109, switch_cont_dim=30** — full type-effectiveness features present
  (the old bot memmap is 107/28 with zero-padded type_eff)
- **Size on disk: 104 GB** (post-trim; pre-trim was 163 GB due to preallocation slack)
- **Zero errors, zero validation failures** during 2h5min of streaming

Integrity verified: file sizes match N rows exactly, boundary row unchanged pre/post
trim (sum_abs=92.67), no NaN across 100k random sample, every legal-mask row has ≥1
active action, episode_index end row matches num_records exactly.

### Pipeline bug fixes (commit `61c665f`)

Two bugs blocked the regen and forced a manual fix mid-session:

1. `replay_parser.py:37` imported `make_obs_mask_and_slots` — a function removed in the
   Session 34 refactor. Top-level import failed, so `replay_to_memmap.py` couldn't even
   start. Fix: dropped the import (function was only used by the legacy `_parse_perspective`
   path, not called externally).

2. `MemmapV8Writer.finalize()` claimed to "trim memmaps to actual size" but only wrote
   `metadata.num_records` — the raw `.npy` files stayed at `max_rows`, wasting ~60 GB of
   preallocation. Fix: added `_trim_files_to_n_rows()` module-level helper that
   `os.truncate()`s each raw-memmap file to `N * per_row_bytes`, and called it from
   `finalize()` after releasing memmap handles (`mm._mmap.close()` + `gc.collect()`).

The fix was verified on the live data: reclaimed 63.67 GB, no integrity regression.

### Crashes earlier in session (context)
**Three BSODs on 2026-04-23 (all bugcheck `0x0000019C` KERNEL_AUTO_BOOST_INVALID_LOCK_RELEASE, Arg1=0x50):**
- 02:26 AM (killed BC training mid-epoch-2)
- 08:36 AM (after user rebooted)
- 11:12 AM (dump saved to `C:\Windows\MEMORY.DMP`)

Same bugcheck + same Arg1 three times = specific driver synchronization bug. **Not** GPU,
**not** fp16, **not** training code. This is the same cluster pattern flagged in
"Known machine-level issues" below and in earlier Session 31/35 BSOD notes.

**Remediation applied this session:**
1. `model.py` got a small AMP dtype fix (scatter sources cast to dest dtype — commit `cf38cf2`).
   Harmless/good regardless of the crashes; committed before investigation.
2. **NVIDIA driver updated** from 551.23 (Jan 2024, 14 months old) → **Studio 595.79**
   (Mar 10, 2026). Clean install via the installer's "Custom → Perform a clean installation".
   Verify post-reboot with `nvidia-smi` — driver_version should read `595.79`.

**Still NOT done (user deferred — flag in next session):**
- MSI bloatware is still running and is the prime suspect per `NEXT_SESSION.md` machine-
  level warning. Active services on this box right now:
  `MSI Foundation Service`, `MSI NBFoundation Service`, `MSI_Central_Service`,
  `MSI_Companion_Service`, `MSI_VoiceControl_Service`, `NahimicService`.
  If crashes resume after the driver update, uninstall MSI Center / Dragon Center /
  NBFoundation / Nahimic next.
- VirtualBox `VBoxNetLwf` errors on every boot (uninstall VirtualBox if not needed).

### Next concrete step — BC reshape on HUMAN data
1. `nvidia-smi` to confirm driver 595.79 and no lingering processes.
2. Start battle servers (commands in "Quick-reference commands" below).
3. Launch BC reshape training on the new human memmap:
   ```bash
   cd pokemon-ai-starter/pokemon-ai/src
   python -u train_bc.py \
     --memmap-dir data/datasets/human_v8_100k \
     --device cuda --fp16 \
     --d-spatial 256 --d-temporal 512 \
     --n-spatial-layers 3 --n-temporal-layers 3 \
     --n-summary-tokens 4 --dropout 0.05 \
     --lr 1e-4 --weight-decay 1e-4 --grad-clip 2.0 \
     --batch-size 16 --epochs 10 --sched cosine --warmup-steps 200 \
     --eval-games 200 2>&1 | tee bc_reshape_human.log
   ```
4. If another BSOD hits, stop, uninstall MSI bloatware (list below), reboot, retry.

**Expected runtime**: 14× more records than the bot memmap → ~14× epoch time, so ~3 hours
per epoch vs. the old ~13 min/epoch → **~30 hours total for 10 epochs**. Longer than one
session. If time-constrained, start with `--epochs 3` to see initial trajectory, then
continue with `--resume` after reviewing epoch-0/1/2 smart_avg.

### Uncommitted files in the tree (as of this session end)
- `watch.ps1` (project root) — PowerShell system monitor (GPU/VRAM/CPU/RAM/disk every 5s
  to `system_watch.log`). Created this session to catch the next BSOD; not committed yet.
  Safe to keep or commit as-is.

### Commits this session
- `cf38cf2` model.py: cast scatter sources to destination dtype in forward
- `61c665f` Fix replay_to_memmap regen: drop stale import, actually trim files

---

## Current position

**Best checkpoint: `selfplay_v9_20260413_061236/snapshot_2979.pt` at Elo 1058**
(confirmed by `data/eval/elo_exp5_FINAL.json`, 62 players, 1891 matches).

- +38 Elo above strongest heuristic bot (Tactical=1019)
- +30 Elo above previous session-top (sp1984=1028)
- +241 Elo above BC base (817)

Near-tied: `snapshot_2999.pt` (Elo 1055, CI overlaps). Top 7 snapshots are all 1048-1058.

**The architectural ceiling is CONFIRMED at ~1058 Elo.** Exp 5 (200 iters with all
safeguards: adaptive entropy + early stop + EMA PFSP + 400 games/iter) produced no
checkpoint that exceeded sp2979. The plateau is real, measured across 62 players and
1891 matchups. Further gains require architectural changes.

**Team selection test in progress:** testing sp2979 with each of 70 handcrafted OU teams
individually against bots (14,000 games). Early results show savg 51-77% depending
on team — massive variance. Fixed-team play is MUCH stronger than random-from-pool.
Results will be in `team_selection_results.json` when complete.

**Session 37 status — TWO major prep milestones done:**

1. **Metamon study complete** (`docs/METAMON_LEARNINGS.md`). Confirmed from direct config
   read that every Metamon size variant uses 5-8× temporal:spatial d_model — strongest
   evidence yet that Option A (capacity reallocation) is the right next architectural lever.

2. **Multi-gen prep DONE** (commit `18e965c`). Previous sessions had already implemented
   90%: vocab loops gens 1-9, features has Mega/Z-move/Dynamax/Tera flags, team_generator
   has per-gen ban lists, format_from_str parses gen. Session 37 fixed two real bugs —
   `ProceduralTeambuilder` wasn't passing `gen` to `load_pokemon_pool` (always loaded
   gen9 tiers), and OU rating threshold was hardcoded `1695` (gen9 only; gen6-8 use `1760`
   in 2024-04 data). Verified gen6/7/8/9 teams now generate era-appropriate mons
   (gen6 HP Ice Thundurus, gen7 Z-crystal Dragonite, gen8 Double Iron Bash Melmetal).

3. **Capacity reshape implemented** (commit `76acca8`). Added `d_spatial`, `d_temporal`,
   `n_summary_tokens` to PokeTransformerConfig. Legacy defaults (None/0) preserve exact
   sp2979 param count (13,382,572, strict=True load verified). New reshape config
   (256/512/3L/3L/K=4/dropout=0.05) instantiates at 14.28M, forward + multi-turn works
   cleanly, FP16 smoke-tested on CUDA. See `docs/METAMON_LEARNINGS.md` §5 for design rationale.

**Next concrete step:** Launch BC retrain with reshape config. Command below.

---

## Quick-reference commands

### Resume PPO from current peak (sp_0219, the all-time-best smart_avg)

```bash
cd pokemon-ai-starter/pokemon-ai/src
python -u train_rl.py \
  --init-from data/models/bc/v8_bc_20260423_195603/best.pt \
  --resume data/models/rl_v9/selfplay_v9_20260425_062416/snapshot_0219.pt \
  --device cuda --servers 9000,9001,9002 --fp16 --pipeline \
  --games-per-iter 200 --max-concurrent 200 --n-iters 200 \
  --warmup-iters 0 \
  --lr 3e-5 \
  --reward-style terminal --lam 0.95 --ent-coef 0.02 --grad-accum 1 \
  --adaptive-entropy --win-rate-mode ema \
  --procedural-teams C:/Users/raiad/OneDrive/Desktop/team_builder/raw_data/pokemon_usage/2024-04 \
  2>&1 | tee -a ppo_human_bc_lr3e5.log
```

**Critical: `--lr 3e-5` not `1e-4`.** From a 45% BC, lr=1e-4 caused 80%+
KL discard and the policy diverged in 3 iters. lr=3e-5 produces clean
multi-epoch updates.

### BC reshape on human data — done, checkpoint locked

Run dir: `data/models/bc/v8_bc_20260423_195603/`. `best.pt` is epoch 2
weights at smart_avg 45.1% (14.28M params, 256/512 spatial/temporal,
3L/3L, K=4). Use this as `--init-from` for PPO. Per-epoch curve:

| Epoch | smart_avg | val_acc |
|---|---|---|
| 0 | 31.1 | 0.671 |
| 1 | 44.4 | 0.683 |
| **2** | **45.1** | **0.706** ← best.pt |
| 3 | 39.9 | 0.706 |
| 4-9 | 16-22 | 0.71-0.713 |

Classic BC overfit after epoch 2 (val_acc keeps climbing, smart_avg
plateaus then regresses). The reshape + human data + restored type_eff
combo broke the old 22-26% BC ceiling decisively.

### Old BC reshape run on bot data — reference only

Run dir: `data/models/bc/v8_bc_20260423_124909/`. `best.pt` at smart_avg
21.2%. Trained on stale `memmap_v8` (107/28 zero-padded). Don't use as
go-forward base.

### Start 3 battle servers (Windows)
```bash
NODE=/c/Users/raiad/OneDrive/Desktop/team_builder/tools/node-v20.18.1-win-x64/node.exe
BS=/c/Users/raiad/OneDrive/Desktop/team_builder/pokemon-ai-starter/pokemon-ai/src/battle_server.js
$NODE "$BS" --port 9000 &
$NODE "$BS" --port 9001 &
$NODE "$BS" --port 9002 &
# Verify:
curl -s -o /dev/null -w "9000:%{http_code} " http://127.0.0.1:9000/action.php?
curl -s -o /dev/null -w "9001:%{http_code} " http://127.0.0.1:9001/action.php?
curl -s -o /dev/null -w "9002:%{http_code}\n" http://127.0.0.1:9002/action.php?
```

### Incremental Elo measurement — the "Useful" set we agreed on (Session 39)

5 snapshots: new BC + early/mid/peak/final from this run. Anchors on
sp2979 (Elo 1058) which is still on disk. Note that ~38 of the 52 old
snapshots in `elo_exp5_FINAL.json` were pruned and won't have real
matches against ours — but sp2979, sp2999, sp0284, sp0839, sp1199, etc.
do exist, so the comparison is meaningful.

```bash
cd pokemon-ai-starter/pokemon-ai/src

# Single shard (~2.5 hr for 5 snapshots × ~12 surviving opponents × ~30 min)
python -u eval_elo_ladder.py --add-to data/eval/elo_exp5_FINAL.json \
  --snapshots \
    data/models/bc/v8_bc_20260423_195603/best.pt \
    data/models/rl_v9/selfplay_v9_20260424_213428/snapshot_0029.pt \
    data/models/rl_v9/selfplay_v9_20260425_062416/snapshot_0099.pt \
    data/models/rl_v9/selfplay_v9_20260425_062416/snapshot_0159.pt \
    data/models/rl_v9/selfplay_v9_20260425_062416/snapshot_0219.pt \
    data/models/rl_v9/selfplay_v9_20260425_062416/snapshot_0229.pt \
  --names new_bc sp_pre_29 sp_0099 sp_0159 sp_0219 sp_0229 \
  --n-games 100 --concurrency 70 --device cuda \
  --server ws://127.0.0.1:9000/showdown/websocket \
  --out-json data/eval/elo_session39.json
```

What we expect to find: sp_0219 should land near or above sp2979 (1058)
in Elo terms — internal PFSP wr says we beat sp2979 63%. If Elo confirms,
real strength gain is real. If sp_0219 sits below 1058, the smart_avg
gain is mostly bot-eval-specific noise.

For 3-shard parallel (faster), use `--shard 0/3` etc. on different
servers as in the old eval_elo_ladder.py docs.

### Monitor training (PowerShell)
```powershell
Get-Content "path\to\log.log" -Wait | Select-String "\] Iter|EVAL:|Snapshot|KL early|FATAL|ERROR|PFSP|EARLY STOP|ENT"
```

---

## What's been tried and what works (consolidated)

### What's validated — don't change these
- **BC → PPO pipeline** (standard, works)
- **Terminal-only reward** (tried shaping, always hurt)
- **Entity tokenization** (the architectural breakthrough — preserved at any spatial dim)
- **Distributional value head** (51-bin categorical, not scalar)
- **Temporal model for OU** (needed for 30-60 turn games)
- **gamma=0.9999, clip=0.2, lam=0.95** (PPO standards, confirmed)
- **lam=0.95** (not 0.75 — that was our outlier, fixed in Exp 1 for +25 Elo)

### What's been tested and learned

| Experiment | Change | Result | Conclusion |
|-----------|--------|--------|------------|
| Lambda fix | 0.75 → 0.95 | +25 Elo | **Keep.** |
| Entropy tuning | ent 0.04 → 0.02 → 0.03 → 0.02 | 0.02 from healthy base is sweet spot | **Keep 0.02.** |
| Slot permutation | bench/move shuffle per turn | Small impact, but helped stability | **Keep.** |
| PFSP (vs uniform) | (1-wr)² weighted | +11 Elo, but 12% stale-rating waste | **Keep. Consider EMA to fix staleness.** |
| LR refinement | 1e-4 → 3e-5 | Hit peak sp2999 (Elo 1055), then collapsed | **Lower LR is risky without safeguards.** |
| Games/iter 400 | paired with low LR | More data per update | Secondary — paired with LR choice. |
| **Exp 5 safeguards** | adaptive ent + early stop + EMA | 200 iters, savg=55.3% mean, no collapse | **Ceiling confirmed at ~1058. Safeguards work.** |

### What causes collapse (learned the hard way)

Exp 4 collapsed from Elo 1055 (sp2999) to 785 (sp3054) in 55 iters. Root causes:

1. **Weak ent_coef** (0.02) + consistent "improvement" gradients → slow policy sharpening
2. **Stale PFSP cumulative ratings** → 12% of training on opponents we'd already mastered →
   reinforces narrow strategies → over-specialization
3. **Entropy dropped below 0.65 without triggering safeguards** (old adaptive threshold was 0.55)

This is now fixed — see safeguards below.

### Safeguards available (Session 36 additions)

All configurable, defaults preserve old behavior. Enable explicitly for safety.

**Adaptive entropy (`--adaptive-entropy`):**
Auto-raises ent_coef when entropy drops, auto-lowers when too exploratory. Defaults:
- `--adaptive-entropy-low 0.65` (raise ent_coef below this; was 0.55)
- `--adaptive-entropy-high 0.95` (lower ent_coef above this)
- `--adaptive-entropy-step 0.1` (±10% per iter; was ±5%)
- `--adaptive-entropy-max 0.08` (cap)
- `--adaptive-entropy-min 0.01` (floor)

Would have caught Exp 4 collapse at iter 3020 with ent_coef rising to ~0.05.

**Composite early stopping (`--early-stop`):**
Stops training if `patience` (default 3) consecutive RAW evals show BOTH savg regression
AND 3+ of 4 bots regressing, OR savg drops >2× threshold (handles specialization).
- Based on rolling-3 "best" baseline, raw-eval comparison for current
- Would have stopped Exp 4 at iter 3055 with only ~3 Elo loss from peak

Parameters: `--early-stop-patience 3`, `--early-stop-savg-threshold 2.0`,
`--early-stop-bot-threshold 3.0`, `--early-stop-bot-count 3`, `--early-stop-min-evals 5`.

9 unit tests in `test_early_stop.py` validate all trigger conditions including Exp 4 replay.

**EMA PFSP (`--win-rate-mode ema`):**
Replaces cumulative win-rate tracking with exponential moving average. Fixes the 12%
stale-oversampling problem. Old cumulative data naturally fades; current policy strength
reflected in rating.
- `--win-rate-ema-alpha 0.3` (blend weight for new batches)
- `--win-rate-ema-window 50` (cap on effective_games to prevent unbounded growth)

Default `cumulative` preserves exact old behavior. 8 unit tests in `test_ema_pfsp.py`.

---

## Code architecture — key files

### Training pipeline
```
train_rl.py           Main training loop + argparse. Calls collect → PPO → eval → save
 ├─ rl_collection.py    collect_v9 async collection + BackgroundCollector (pipeline)
 │   └─ pfsp_sample()   Prioritized opponent sampling — cumulative or EMA
 ├─ rl_player.py        V9RLPlayer + SelfPlayOpponent (shared build_turn_batch)
 ├─ rl_pipeline.py      Multiprocess variants (InferenceServer, MPRLPlayer)
 ├─ inference_batcher.py Async batched GPU forward — key for concurrency speedup
 ├─ ppo.py              Trajectory, GAE, ppo_update, save_checkpoint (atomic writes)
 ├─ model.py            PokeTransformer 13.38M: spatial 384d/4L + temporal 384d/2L
 └─ features.py         Entity tokenization, build_turn_batch, slot permutation
```

### Evaluation / analysis
```
eval_elo_ladder.py     Full Bradley-Terry ladder + --add-to incremental + --shard parallel
battle_agent.py        Inference player (eval, no training augmentation)
analyze_experiments.py Cross-experiment comparison (per-bot stats, correlations)
analyze_pfsp.py        PFSP opponent tracking analysis (staleness, concentration)
analyze_pfsp_timing.py Opponent rotation / encounter timing analysis
analyze_elo_trajectory.py  Elo plot + era aggregates from ladder JSON + eras.json
build_registry.py      Rebuild persistent registry from scratch
```

### Data / registry
```
data/eval/registry/
├── runs.jsonl         Training run configs (auto-logged by train_rl.py)
├── evals.jsonl        Per-iter bot eval results (auto-logged every 20 iters)
└── elos.jsonl         Per-snapshot Elo measurements (auto-logged by eval_elo_ladder.py)

data/eval/
├── elo_exp5_FINAL.json    Canonical Elo ladder (55 players)
├── eras.json                     Training era definitions for trajectory plots
└── trajectory_session36.png      Latest trajectory plot
```

### Tests
```
test_exp2_exp3.py      Slot permutation + PFSP correctness (14 tests)
test_early_stop.py     Composite early stopping trigger logic (9 tests)
test_ema_pfsp.py       EMA win-rate tracking (8 tests)
```

---

## What to do next — MULTI-GEN IS THE PRIORITY

The hyperparameter refinement lever is exhausted (confirmed by Exp 5). Multi-gen
prep comes FIRST before any model size changes, so all future work builds on a
multi-gen-ready foundation.

### IMMEDIATE NEXT: Study Metamon architecture + apply learnings

**PokeAgent ladder results (Session 36):**
Our model (13.38M) submitted to the PokeAgent Challenge live ladder.
- **Skill Rating 1444±30 with TEAM_T** — rank #12, ABOVE MM-Minikazam (4.7M, 1429)
- **Skill Rating 1376±24 with TEAM_AU** — rank #15
- Team choice moved us 68 points (more than all hyperparameter experiments combined)
- Gap to MM-SmallRLG9 (15M, same size as us): 73 points — they use params more efficiently
- Gap to MM-Kakuna (142M, best): 409 points
- We beat ALL heuristic baselines (BH tier: 1185-1240) by 200+ points

**Key insight: Metamon's 4.7M model nearly matches our 13.38M.** The gap is architecture
and methodology, not model size. Metamon is open source — study their design, apply to ours.

**Metamon reference repo cloned:** `metamon_ref/` (shallow clone, read-only reference).
Study these files (DO NOT modify our code to copy theirs — learn principles, apply to our design):
- Architecture: temporal:spatial ratio, entity token handling, layer configs
- BC data pipeline: preprocessing, quality filters, data scale
- Offline RL recipe: Binary+MaxQ methodology (different from our PPO)
- Minikazam (4.7M) vs Kakuna (142M): what scales and what doesn't

**After understanding Metamon's design, THEN do:**

### Multi-gen + gen-agnostic prep

**Why first:** User direction (Session 33, reconfirmed Session 36): multi-gen before
BC scaling or capacity reallocation. Reasoning: if we change model size or retrain BC
BEFORE multi-gen, we'd redo all that work when we add multi-gen later. Do it once.

**What needs to change (the architecture is ALREADY gen-agnostic):**
- `vocab.py`: verify/expand species/move/ability/item tables for gens 6-9
- `features.py`: add gen-specific volatile effects (Mega Evolution for gen6, Z-moves for gen7)
- `team_generator.py`: per-gen team generation + ban lists (currently gen9 only)
- `format_config.py`: add gen6/7/8 FormatConfig entries (n_actions, team_size, etc.)
- `replay_to_memmap_v8.py`: gen filter for scraping gen6/7/8 OU replays
- `battle_server.js`: verify it handles gen6/7/8 formats

**What does NOT change:** model.py, ppo.py, train_rl.py, rl_collection.py, rl_player.py
(all already parameterized via FormatConfig + --format flag).

**Gen scope:** Start with gens 6-9 (fully compatible features). Gens 4-5 need minor
adjustments. Skip gens 1-3 for now (missing abilities/items, different mechanics).

**Estimated effort:** 1-2 sessions for vocab + features + team gen. Multi-gen replay
scraping can run in background.

**Validation:** Train a 1-iter BC sanity check with expanded vocab on existing gen9
data → confirm no breakage. Then scrape gen6-8 replays and train multi-gen BC.

### AFTER multi-gen: architectural experiments

### Option A — Capacity reallocation (Exp 5 from old plan)

**What:** shrink spatial encoder, grow temporal encoder. Current ratio 1:1 (384d/4L spatial,
384d/2L temporal). Metamon uses 5-8:1 temporal:spatial across all model sizes. Proposed:
spatial 256d/3L, temporal 512d/3L (same param count, redistributed).

**Why:** Our temporal model is undersized for OU's 30-60 turn games with heavy hidden info.
The inverted ratio is our biggest architectural anomaly vs published systems.

**Cost:** Requires BC retrain from scratch (~1-2 days compute) + PPO from new BC (~1 week).

**Decision criterion:** If new model hits Elo 1100+ on existing ladder, ratio matters.
If flat at ~1058, capacity allocation isn't the bottleneck.

**Implementation:** change config values in `model.py` (d_model and n_spatial/temporal_layers).
The architecture supports any combination. Main work is BC retrain, not code.

### Option B — BC scaling (multi-gen path)

**What:** scale BC base from 13.4M to 30M+ params. Train on multi-gen data (gens 6-9 OU, not
just gen9). Per `memory/project_multigen_plan.md` this is mostly data-pipeline work since
architecture is already gen-agnostic.

**Why:** Our BC_base is Elo 817. Metamon's base is 1500+. PPO can only climb from where BC
starts — if BC ceiling is low, PPO ceiling is bounded. Bigger BC + more diverse data
= higher starting floor.

**Cost:** Several weeks. Multi-gen vocab prep (1-2 weeks), multi-gen replay scrape (1-2
weeks, mostly automated), 30M BC training (3-7 days), PPO from new base (1-2 weeks).

**Local limit:** 30M is realistic max on 6GB VRAM. For 50M+, need cloud.

**Decision criterion:** 30M BC at Elo 900+ → scaling helps → continue. 30M BC at ~810 → pivot.

**Steps in detail:**
1. Multi-gen vocab expansion (`vocab.py`, `features.py`)
2. Multi-gen replay memmap (`replay_to_memmap_v8.py` + gen filter)
3. 30M BC training (existing `train_bc.py` with larger config)
4. PPO from new base (same recipe, different checkpoint)

### Option C — Search at inference (MCTS)

**What:** add MCTS on top of learned policy/value. Use NN as prior/evaluator,
search for better moves. MIT thesis finding: "RL alone plateaus but RL+search breaks
through." Foul Play (PokeAgent Challenge co-winner) uses pure MCTS with no NN.

**Why:** Our policy has hit its representational ceiling. Search can compensate by
computing forward through difficult positions instead of relying on amortized NN output.

**Cost:** ~1-2 weeks implementation. Needs opponent modeling (opponent team is partially
hidden) + forward simulation via battle_server.

**Risk:** Inference latency. PokeChamp's LLM search times out 1/3 of games. Our search
would be faster (NN forward vs LLM call) but still adds per-move overhead.

**Expected gain:** MIT thesis achieved 1693 Elo with PPO+MCTS. Gen9 OU is harder than
Gen4 Random, but +200-400 Elo over the current policy is plausible.

### Option D — PokeAgent Challenge submission (READY TO GO)

**What:** Submit our model to the PokeAgent Challenge live benchmark. Script written
(`pokeagent_submit.py`), top teams identified (TEAM_AU at 78.5% savg).

**Why:** Ground-truth Elo against Metamon, Foul Play, and other agents. Our internal
"1058" is bot-anchored; external Elo could be very different.

**How to submit (exact steps):**

1. Go to https://battling.pokeagentchallenge.com
2. Click Login -> Create Team (creates an organization account)
3. Open "My Team" -> create a named AI agent (gets username + password credentials)
4. Run:
```bash
cd pokemon-ai-starter/pokemon-ai/src
python pokeagent_submit.py \
    --checkpoint data/models/rl_v9/selfplay_v9_20260413_061236/snapshot_2979.pt \
    --username <agent_username_from_step_3> \
    --password <agent_password_from_step_3> \
    --team TEAM_AU \
    --n-games 50 \
    --device cuda
```
5. Check rating at https://battling.pokeagentchallenge.com/ladder

Server: `wss://battling.pokeagentchallenge.com/showdown/websocket`
Discord for support: https://discord.gg/E2DuX5FWF7
Formats supported: Gen9 OU (our format), Gen1-4 OU, Gen9 VGC

**Cost:** ~1-2 hours for 50 games. No code changes needed. Uses existing BattleAgent.
Top team (TEAM_AU) scores 78.5% vs smart bots locally.

**Team selection results** in `team_selection_results.json`. Top 5: TEAM_AU (78.5%),
TEAM_T (77.5%), TEAM_G (77.0%), TEAM_B (75.0%), TEAM_C (73.5%). Use `--team random`
to rotate among top 10.

### Decision recommendation

**If you want decisive answers fast:** do D (PokeAgent submission) first — gives us an
external benchmark in days, informs A/B/C choice.

**If you want to push forward:** do A (capacity reallocation) before B (BC scaling).
A is cheaper (no new data) and tests the architectural hypothesis directly.

**If you want the highest ceiling:** do C (search). Highest-variance outcome but best
potential gain.

**For cloud deployment:** B + C + D path together. Scale up BC, add search, submit to
benchmark. Requires significant compute budget.

---

## Research context

See `docs/RESEARCH.md` §0 for full comparison tables. Key landscape facts (as of Apr 2026):

| System | Format | Elo | Model | Notes |
|--------|--------|-----|-------|-------|
| **Ours (sp2979)** | Gen9 OU | 1058 internal (bot-anchored) | 13.38M | Current peak |
| ps-ppo | Gen9 Random | 1900+ public ladder | ~15M | Random is easier than OU |
| VGC-Bench BCFP | VGC Doubles | 1768 internal | ~10M | Different format/scale |
| Metamon (Gen1OU) | Gen1 OU | 1500+ public | 15-200M | Published best for public ladder |
| Metamon Kakuna | Gen9OU | Unknown | 57M | Post-paper, no public Elo |
| PokeChamp | Gen9 OU | 1300-1500 | GPT-4 (no training) | LLM minimax search |
| PokeAgent PA-Agent | Gen9 OU | Unknown | RL-based | Co-won PokeAgent Track 1 |
| Foul Play | Multi-gen OU | Unknown | No NN (pure MCTS) | Co-won PokeAgent Track 1 |

**No pure RL agent has demonstrated >1800 Gen9 OU on public ladder** as of Apr 2026.
Top human Gen9 OU peaks: 2030-2115 Elo.

**Validated techniques from literature:**
- Entity tokenization (ps-ppo, VGC-Bench) — we have this
- Distributional value head (ps-ppo, Metamon via two-hot) — we have this
- Transformer with long context (Metamon, VGC-Bench) — we have this
- BC → PPO pipeline (all systems) — we have this
- PFSP opponent sampling (AlphaStar, VGC-Bench Double Oracle) — we have this
- Adaptive/constant entropy (EPO 2025) — we have this
- Slot permutation augmentation (no Pokemon paper; E2GN2 theory) — we have this
- Search + RL (MIT thesis, Foul Play, PokeChamp) — **we don't have this**

---

## Reference data

### Confirmed Elo (from `elo_exp5_FINAL.json`)

Top 10 snapshots:
```
 #1  sp2979  1058  [1044, 1070]   ← current all-time best
 #2  sp2999  1055  [1040, 1068]   ← tied (CI overlap)
 #3  sp3179  1053  [1039, 1065]
 #4  sp2299  1049  [1036, 1060]
 #5  sp2039  1047  [1034, 1058]
 #6  sp3199  1045  [1031, 1058]
 #7  sp2799  1042  [1027, 1056]
 #8  sp2359  1041  [1029, 1055]
 #9  sp2139  1040  [1026, 1051]
 #10 sp3019  1039  [1024, 1049]
```

Bot anchors:
```
  Tactical        1019  (top bot)
  Strategic       1018
  SmartDmg/SH     1000  (SH is anchor)
  Random          462   (floor)
```

Collapse reference:
```
  sp3054          785   ← Exp 4 post-collapse (below BC_base!)
  BC_base         817
```

### Critical checkpoints

- `data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt` — BC base (Elo 817), init-from for all PPO
- `selfplay_v9_20260413_061236/snapshot_2979.pt` — current all-time best (Elo 1058)
- `selfplay_v9_20260415_083340/emergency_iter_3055.pt` — Exp 4 collapse (DO NOT USE)

### Data files / registry
- `data/eval/elo_exp5_FINAL.json` — canonical Elo ladder (55 players, ~1400 matches)
- `data/eval/eras.json` — training era definitions (E0-E12)
- `data/eval/registry/runs.jsonl` — all training run configs
- `data/eval/registry/evals.jsonl` — per-iter bot evals (~200+ entries)
- `data/eval/registry/elos.jsonl` — all Elo measurements

### Infrastructure
- **Device:** RTX 3060 Laptop 6GB VRAM, 16GB RAM, Win10, Python 3.11, CUDA 12.1
- **Battle servers:** portable Node 20 at `tools/node-v20.18.1-win-x64/node.exe`
- **Use `127.0.0.1` not `localhost`** (Windows DNS quirk)

### Stale memmaps warning
`human_v8/memmap_v8` have `move_cont_dim=107, switch_cont_dim=28`. Current code expects
109/30. `dataset.py` auto-pads, but **regenerate memmaps before any BC training** to
get full type_eff features. Regeneration via `replay_to_memmap_v8.py`.

---

## Session handover checklist

For starting a new session:

1. **Read this file top-to-bottom.** It's self-contained.
2. **Check git status** (`git log --oneline -10`) for any un-handed-over work.
3. **Check for running processes** (`nvidia-smi | grep python`) — anything still training?
4. **Check latest Elo:** `data/eval/elo_exp5_FINAL.json` → confirm sp2979=1058 is still the best.
5. **If training didn't finish cleanly, check the emergency checkpoint** in the run dir.
6. **Pick a next step** from "What to do next" above based on time budget.
7. **If running any PPO training, enable safeguards**: `--adaptive-entropy --early-stop`.
8. **For cloud runs especially**: also `--win-rate-mode ema` (fixes staleness, saves compute).

### Known machine-level issues
- BSODs observed Mar 14-Apr 12 (multiple). Pattern stopped after Apr 12. If they resume,
  update NVIDIA drivers (nvlddmkm errors in Event Log) and uninstall MSI bloatware
  (Dragon Center/NBFoundation — crashed on reboot previously, implicated in WMI bugchecks).

---

## Memory file index

User-level memories in `memory/`:
- `MEMORY.md` — compact project summary loaded every session
- `feedback_*.md` — decisions/lessons that should persist
- `project_*.md` — project state notes
- `reference_*.md` — device/landscape references
