# ARCHITECTURE_REVIEW — TransformerBattlePolicy (2026-06-07)

> ⚠️ **GRAIN OF SALT — READ BEFORE ACTING.** This review is a point-in-time
> analysis based on reading `pokemon-ai-starter/pokemon-ai/src/model_transformer.py`
> on 2026-06-07. Code evolves. Before acting on any recommendation here:
>
> 1. **Re-read the actual current code.** File paths, line numbers, class
>    signatures, even architectural choices may have drifted since this was
>    written. The conclusions are only as valid as the snapshot.
> 2. **Run tests on the current version** to confirm the described behavior
>    still holds.
> 3. **Check `git log` since this commit** for any architecture-touching
>    changes that may have already addressed or invalidated points here.
>
> Treat this doc as a **starting point for thinking**, not as authoritative
> truth about the current codebase.

---

## Purpose

Aesthetic + design review of the `TransformerBattlePolicy` architecture.
Distinct from `ARCH_AUDIT.md` (which focused on PPO-correctness gaps in
Session 49/50). This review asks: **given the current implementation is
correct, are the design choices good, defensible, or worth revisiting?**

Intended trigger for action: **before the next major architectural phase**
(multi-gen, doubles/triples, VGC, etc.) where large rewrites and feature
additions are happening anyway. Some of these revisions could ride along
those bigger changes at low marginal cost; doing them standalone now would
have higher overhead vs reward.

## TL;DR

The architecture is well-thought-out, with each decision having recorded
rationale in REWRITE_DESIGN / METAMON_LEARNINGS / Postscript notes. No
obvious flaws. Several decisions stand out as **clearly right**. A few are
**defensible but worth ablating**. **One** is interesting enough to actually
prioritize before the next phase.

## What's clearly right (CONFIRMED design wins)

### 1. Spatial/temporal split with asymmetric widths

- **Choice**: separate `SpatialTransformer` (6L × 8H × d=256) and
  `TemporalTransformer` (4L × 8H × d=512). 2× temporal-to-spatial width.
- **Why right**: matches problem structure (within-turn tactical reasoning
  vs across-turn strategic reasoning). Evidence-based ratio choice
  (Metamon's "shift capacity to temporal" finding cited in code comments).
- **Empirical**: per memory `project_s66_collect_arch_findings`, 64% of
  params live in the temporal stack — strategic context is the lion's share
  of model capacity.
- **Verdict**: keep as-is.

### 2. Poke-Mask actor/critic separation at spatial layer

- **Choice**: spatial attention mask blocks state/summary tokens from
  attending to actor/critic tokens, and blocks actor↔critic.
- **Why right**: prevents the "actor knows what the critic was about to
  predict" leakage that affects many shared-trunk PPO setups. Cheap
  (one-time mask construction) and correct (zero-attention is the standard
  separation primitive).
- **Subtlety to flag**: separation is ONE-LAYER-DEEP. Both heads consume
  the same `temporal_ctx`, which derives from K=4 spatial summary tokens
  that DID see the spatial output. So critic-relevant info can theoretically
  flow to actor via the temporal compression path. Whether this matters
  depends on training dynamics — but it's worth a clarifying comment in the
  code (the design comment claims full separation; reality is "spatial-layer
  separation only"). REVIEW WHEN: training instability that looks like
  policy/value coupling.
- **Verdict**: keep, but document the limitation in `SpatialTransformer`
  docstring.

### 3. Per-action context for policy head

- **Choice**: `ActionHead` consumes `[actor_out ⊕ temporal_ctx ⊕ action_ctx]`
  where `action_ctx` has shape (B, 9, d_model) with per-action features
  (move token for moves, encoded switch slot for switches).
- **Why right**: grounds each logit in concrete action semantics. Without
  this, the model would have to learn `action_index → action_meaning`
  internally. With it, that mapping is free.
- **Verdict**: keep as-is.

### 4. Distributional value head (51-bin twohot)

- **Choice**: `ValueHead` outputs categorical distribution over 51 bins in
  [-1.6, 1.6], reduced to scalar via expectation. Distributional
  cross-entropy loss during training.
- **Why right**: scalar regression on variance-heavy returns is unstable.
  Distributional captures uncertainty and is more stable. Standard from
  C51 / MuZero / Atari literature.
- **Verdict**: keep as-is.

### 5. K=4 summary scratch tokens (spatial → temporal bridge)

- **Choice**: 4 trainable "scratch" tokens at the end of the spatial
  sequence; their outputs flatten and project to `d_temporal`.
- **Why right**: lets the spatial stack decide what to forward to temporal,
  via attention to these tokens. Better than hand-designed pooling.
- **Verdict**: keep as-is.

## What's defensible-but-questionable (worth ablating someday)

### A. Move vs switch action representation asymmetry

**THIS IS THE ONE TO PRIORITIZE.** See §"Recommended pre-next-phase action."

- **Choice**: move action contexts = 4 active-Pokemon move tokens read
  directly from spatial output (rich, attention-derived). Switch action
  contexts = mean-pool of 17 attribute tokens per bench Pokemon →
  `SwitchActionEncoder` → d_model (statistical aggregate, not attention).
- **Why questionable**: switch decisions are arguably the strategically
  HARDER decisions in competitive Pokemon (when to pivot, sac a mon,
  preserve a sweeper, etc.). Yet they get the THINNER representation.
- **Pragmatic justification (CONFIRMED-by-code-comments)**: only 4 moves
  per active vs up to 5 bench × 17 = 85 tokens; mean-pool is the cheap
  default.
- **Predicted asymmetry in failures (CONJECTURE)**: model should be
  relatively better at picking moves than picking switches. Could be
  validated by error-pattern analysis on eval data.
- **REVIEW WHEN**: doing the multi-gen / doubles rewrite (action space
  changes anyway) OR if eval analysis confirms switch-decision weakness.

### B. Causal mask in temporal training

- **Choice**: temporal stack uses causal attention (only past visible).
- **Why correct for inference**: at battle-time, the model only sees past
  turns. Causal mask matches.
- **Why questionable for training**: during training the trajectory is
  fully observed. Non-causal training with inference-time causal
  restriction is the standard BERT-style trick — can yield richer
  representations. Unclear if this was deliberately chosen vs default.
- **REVIEW WHEN**: training data efficiency or sample complexity becomes
  the bottleneck.

### C. Action logit masking with -100 (not -inf, not -1e4)

- **Choice**: illegal action logits set to -100.
- **Why mostly fine**: in fp32, `softmax(-100 vs +5)` is ~`exp(-105) ≈ 0`.
  Effectively masked.
- **Why slightly worrying**: in bf16 with adjacent legal logits at
  magnitudes 2-3, the masking is still numerically zero but the gradient
  through bf16 softmax with that magnitude difference could be subtly off.
  Standard mitigation is `-1e4` or `float('-inf')` (clamped) for more
  defensive behavior.
- **REVIEW WHEN**: investigating any policy-update numerical instability,
  or any future move to fp16 (NOT bf16). For bf16, current behavior is
  empirically fine.

### D. switch_cont scalars injected separately

- **Choice**: type effectiveness scalars (defensive_eff, offensive_eff)
  are pre-computed and concatenated with the mean-pooled bench-Pokemon
  rep before SwitchActionEncoder.
- **Why questionable**: model could theoretically derive these from the
  type tokens — letting the architecture "discover" effectiveness might
  generalize better to non-canonical type interactions (Tera-Type,
  Knock-Off + plate, etc.).
- **Why current choice is OK**: cheaper, more direct, no learned
  approximation error.
- **REVIEW WHEN**: Tera-Type / Knock-Off interaction modeling is a known
  weakness in the policy (would need ladder analysis).

### E. dropout=0.05 (very low)

- Standard for transformer RL where data is plentiful and overfitting is
  less of a concern than sample efficiency. Reasonable default.
- REVIEW WHEN: BC train/eval gap shows obvious overfitting.

### F. gradient_checkpoint=False by default

- 20M params, small enough to not need checkpoint by default.
- REVIEW WHEN: per-worker memory budget tightens or scaling up params.

## What's the actual scale question

### G. Model size (~20M params)

Metamon Small uses similar capacity. AlphaStar / OpenAI Five used 100M+
for harder games. Pokemon is information-rich but discrete and bounded.

20M might be the right operating point, OR might be the EASY ceiling — the
cheapest model that "just works" given current Tier 3 minibatch=64
constraints. Worth knowing whether scaling tests have been done.

REVIEW WHEN: training shows clear capacity-bottlenecked plateau OR before
committing to multi-gen architecture (capacity needs may change with
larger format coverage).

## Recommended pre-next-phase action

### Priority 1: switch-action representation ablation (Item A)

**The ask**: replace mean-pool-of-17-attribute-tokens with per-bench-Pokemon
attention summary tokens (similar pattern to K=4 spatial summary trick, but
per-Pokemon).

**Concrete proposal**:
- Add 5 dedicated bench-Pokemon summary scratch tokens to the spatial
  sequence (one per bench slot).
- During spatial attention, each can attend to the 17 attribute tokens
  for its corresponding bench Pokemon (or use full spatial attention with
  a slot-aware mask if simpler).
- Switch action context = these 5 spatial output vectors (one per bench
  Pokemon), permuted via species match to `available_switches` order,
  concatenated with `switch_cont` effectiveness scalars.
- Removes the mean-pool bottleneck; gives switch decisions the same
  attention-derived richness as moves.

**Cost**:
- ~5 extra scratch tokens × current spatial attention complexity (negligible)
- Touches: `Tokenizer` (add bench-Pokemon scratch tokens), possibly
  `SpatialTransformer` (mask refinement if slot-aware), `TransformerBattlePolicy`
  (per_action_context assembly for switch slots)
- ~50-100 LoC

**Expected benefit (CONJECTURE)**:
- Better switch-decision quality, particularly in setup-vs-sweep
  trade-offs and pivot moves
- If switch-decision-quality is the limiter for the 70-74 smart_avg
  plateau, could yield +1-3pp
- Even if no measurable lift, removes an architectural asymmetry that
  could confound future analysis

**Verification plan**:
- Pre-ablation: run error-pattern analysis on Run #5 / Run #6 eval data
  to check if switch-decision errors are over-represented vs move errors
- If pre-evidence is weak, lower priority
- If pre-evidence is strong, prioritize as the FIRST architecture change
  in the multi-gen rewrite

### Priority 2: validate Poke-Mask information flow assumption (Item 2 subtlety)

**The ask**: empirically verify whether `temporal_ctx` provides a
critic-to-actor information leak path in practice. Could be done with:
- Synthetic probe: train a tiny classifier on `temporal_ctx` to predict
  value-relevant signals; see if those signals also affect policy output
- Architectural alternative: separate temporal stacks for actor and
  critic — adds params but eliminates the question

**Cost**: ~half day of experimentation. Doesn't require code changes if
the diagnostic version is the conclusion.

**Expected benefit**: design clarity, possible bug fix if the leak is
real and harmful.

### Priority 3: causal vs non-causal training comparison (Item B)

**The ask**: train a small model with non-causal attention during BC,
then enforce causal at inference; compare downstream RL behavior.

**Cost**: 1 BC training run + 1 short RL run = ~$30-50.

**Expected benefit**: marginal at best for our context (battles cap at
~300 turns; causal is the "right" semantic for online play). Worth doing
only if other improvements have been exhausted.

## Updates / amendments to this review

When future sessions act on any item here, update this doc with:
- Date acted
- What was tried
- Result (CONFIRMED / REFUTED / INCONCLUSIVE)
- Whether the recommendation stands or has been superseded

## Cross-references

- `pokemon-ai-starter/pokemon-ai/src/model_transformer.py` — the code
  being reviewed
- `docs/ARCH_AUDIT.md` — prior architecture audit (Session 49/50,
  PPO-correctness focus)
- `docs/REWRITE_DESIGN.md` — original architecture design doc with
  rationale for current choices
- `memory/project_architecture_review_2026_06_07_todo.md` — TODO pointer
- `memory/project_s66_collect_arch_findings.md` — param distribution data
  (64% temporal)
