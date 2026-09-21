# REWRITE_DESIGN.md — Pure-Transformer Pokemon Battle AI

**Status:** design (Session 45, 2026-05-01). Implementation begins Session 46+.
**Replaces:** the MLP-encoder architecture in `pokemon-ai-starter/pokemon-ai/src/model.py`
**Companion docs:**
- `docs/NEXT_SESSION.md` — S44 architecture audit, the pivot rationale, the live PPO run
- `docs/METAMON_LEARNINGS.md` — Metamon's published recipes for inspiration
- `docs/EXTERNAL_OPPONENTS_PHASE2.md` — protocol bugs unrelated to the rewrite, kept fixed

This is a living document. Future implementation sessions update it as reality
diverges from spec — don't silently drift. Section headers are stable; details
within sections may shift.

---

## 0. Reading guide

If you're picking this up cold:

1. Read `docs/NEXT_SESSION.md` §"Architecture audit (Session 44 finalization)"
   first — that's where the *why* lives. This document is the *what*.
2. Then read this top-to-bottom. Every claim referencing existing code points
   at `file:line`. If you can't find that line, the file changed since 2026-05-01
   and the design needs an update.
3. Don't start coding before §7 (Implementation roadmap). The earlier sections
   build the picture; the roadmap turns it into discrete tasks.

Verification discipline (carried forward from S44): when you implement against
this design and reality disagrees, **update the design** before continuing.
Silent drift is how good designs become bad codebases.

---

## 1. Goals and non-goals

### Goals

- **G1. Address the MLP compositional bottleneck.** `PokemonNet` (model.py:164-240)
  compresses a combinatorially-explosive input (species × items × abilities ×
  stat spreads × movesets) into one 256-dim token via a 2-layer MLP. This is
  the structural ceiling diagnosed in `docs/NEXT_SESSION.md`'s
  "Architectural insight: MLP encoding is a fundamental compositional
  bottleneck" section. Every attribute gets its own token instead.
- **G2. Match or exceed sp_0229's 67.8% smart_avg baseline** on Metamon
  competitive teams (16-team set, 500 games × 4 bots, ±2.2pt CI). The MLP-arch
  ceiling sat at 64-66% smart_avg over a 200-iter PPO run; the rewrite must
  break that decisively (≥3pt sustained over 3+ evals = ≥71%).
- **G3. Provide a foundation that scales with more data/compute.** The new
  architecture should benefit from cloud BC training (more replay data,
  larger model) and longer PPO runs without hitting another structural
  ceiling at the next scale tier.
- **G4. Restore the architectural consistency that S44 §"Real shortcuts
  identified" found broken** — the team-move bank zeros (model.py:701-702),
  the 86-dim move-cont padding for team moves (model.py:233-234), and the
  single attention-pooled summary token (model.py:420-426 with
  `n_summary_tokens=0`). All three are subsumed by per-attribute tokens.

### Non-goals

- **N1. Outperform Metamon's Kakuna (140M params, 18M training trajectories).**
  Beyond reach without their compute budget. The realistic target is
  Minikazam (4.7M params, RNN-based) and SmallRL (15M params).
- **N2. Implement test-time adaptation.** Held in reserve. Listed in
  `NEXT_SESSION.md` as a paradigm shift for "Session 50+ material."
- **N3. Foundation-model approach (battle state as text, LLM backbone).**
  Rejected per S44 — gimmicky inference cost, clunky representation when
  we already have structured features.
- **N4. Rewrite the training pipeline (`train_rl.py`, `rl_collection.py`,
  `external_opponent_manager.py`, the 5 defense layers).** All of those
  remain. The rewrite changes the *model* and the *obs/feature path*; the
  PFSP pool, PPO loop, and external-opponent infrastructure stay.
- **N5. Replace `BattleAgent` (battle_agent.py:23).** New arch lives
  *alongside* the legacy MLP-arch player, not in place of. See §6b.
- **N6. Multi-gen support in V1.** Singles gen 9 only. Multi-gen is a
  natural extension (the tokenizer is gen-agnostic by design) but not
  in this rewrite's scope.

---

## 2. High-level architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Battle State (per turn)                      │
│   12 Pokemon × ~17 attributes  +  field  +  transition  +  legal    │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Tokenizer (replaces MLP encoders)                 │
│  Each attribute → one token (d_model). Type-ID embedding added so    │
│  attention can disambiguate "this is HP" vs "this is Move 3 BP."     │
│  Move tokens are rich (BP/acc/PP/prio/flags baked in via small MLP). │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
                    ~210 tokens (vs 14 entity tokens today)
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      Spatial Transformer                             │
│  6 layers × 8 heads × d_model=256, with Poke-Mask (kept) +           │
│  side-mask (decision tokens for player A can't attend tokens marked  │
│  as private to player B's Pokemon-level info).                       │
│  K=2 learnable summary scratch tokens collect per-turn output.       │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
                    K × d_model summary per turn
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      Temporal Transformer                            │
│  4 layers × 8 heads × d_model=256, causal mask, 200-turn context.    │
│  Same as model.py:437-511 in spirit; just larger.                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│       Action heads (9 logits)  +  Value head (51-bin twohot)        │
│  Action head consumes per-action tokens (kept from model.py:540-571) │
│  + actor + temporal_ctx; value head consumes critic + temporal_ctx.  │
└─────────────────────────────────────────────────────────────────────┘
```

### What's preserved from the current architecture

- **Temporal transformer with causal mask** (model.py:437-511). Multi-turn
  attention happens in a separate stack; tokens are not unified across
  turns. Rationale in §4.
- **Decision-token convention** (model.py:312-430). The actor and critic
  are special tokens; Poke-Mask blocks state→decision attention.
- **Action slot encoder** (model.py:522-571). The 9-action context (4
  moves + 5 switches) feeds the policy head per-action. Internally the
  encoder switches from `MoveNet` MLP to per-attribute move tokens, but
  the *interface* (`(B, 9, d_model)` → policy head) is the same.
- **Value head structure** (model.py:642-651). Distributional twohot over
  51 bins, range `[-1.6, 1.6]`. Twohot encoding (model.py:805-816) stays.
- **PFSP pool, 5 defense layers, training loop.** All non-model
  infrastructure (rl_collection.py, train_rl.py, external_opponent_manager.py)
  is unchanged.

### What's replaced

- **`MoveNet` (model.py:124-157), `PokemonNet` (model.py:164-240),
  `FieldNet` (model.py:247-275), `TransitionNet` (model.py:282-305).**
  All four are MLPs that compress structured features into one token per
  entity. They become a single `Tokenizer` module that emits ~17 tokens
  per Pokemon, ~1-2 tokens for field, ~1 token for transition.
- **Spatial transformer's 14 entity tokens** (model.py:312-430) → ~204
  attribute tokens.
- **Single attention-pooled summary** (model.py:420-426 with K=0) → K=2
  scratch summary tokens (already supported as a CLI flag in current
  arch but never used in production; this rewrite makes K≥2 the default
  rather than an opt-in).

---

## 3. Tokenization scheme — the heart of the design

### 3.1. Token budget

Per turn, per side (ours = full info; opp = revealed-only with learnable
"unknown" embedding for hidden fields):

**Per Pokemon (17 tokens × 12 Pokemon = 204):**

| # | Token name | Source | Dim | Type ID |
|---|---|---|---|---|
| 1 | `species_token` | embedding lookup `n_species=1548` | d_model | 0 |
| 2 | `item_token` | embedding lookup `n_items=2340`; ID 0 = unknown | d_model | 1 |
| 3 | `ability_token` | embedding lookup `n_abilities=314`; ID 0 = unknown | d_model | 2 |
| 4 | `type_token` | small MLP from 2× type embedding (19 types each) | d_model | 3 |
| 5 | `status_token` | small MLP from status one-hot (7) + volatiles bits (38) + paradox (7) + tera (20) flags | d_model | 4 |
| 6 | `hp_pct_token` | `NumericalBank(101)` lookup of HP% bucket (0-100) | d_model | 5 |
| 7 | `boosts_token` | small MLP from 7 boost values (atk/def/spa/spd/spe/acc/eva, range -6..+6) | d_model | 6 |
| 8 | `stat_hp_token` | `NumericalBank(256)` of base HP stat | d_model | 7 |
| 9 | `stat_atk_token` | `NumericalBank(256)` of base atk | d_model | 8 |
| 10 | `stat_def_token` | `NumericalBank(256)` of base def | d_model | 9 |
| 11 | `stat_spa_token` | `NumericalBank(256)` of base spa | d_model | 10 |
| 12 | `stat_spd_token` | `NumericalBank(256)` of base spd | d_model | 11 |
| 13 | `stat_spe_token` | `NumericalBank(256)` of base spe | d_model | 12 |
| 14 | `move_token[0]` | rich move encoder (see §3.2) | d_model | 13 |
| 15 | `move_token[1]` | rich move encoder | d_model | 13 |
| 16 | `move_token[2]` | rich move encoder | d_model | 13 |
| 17 | `move_token[3]` | rich move encoder | d_model | 13 |

The 4 move tokens share type ID 13; they're distinguished by *position
within Pokemon* (a separate `slot_id` embedding, see §3.3).

**Battle-state-level (6 tokens):**

| # | Token name | Source | Dim | Type ID |
|---|---|---|---|---|
| 1 | `actor_token` | `nn.Parameter` (learnable) | d_model | 14 |
| 2 | `critic_token` | `nn.Parameter` (learnable) | d_model | 15 |
| 3 | `field_token` | small MLP from `FIELD_CONT_DIM=52` (features.py:653) + `NumericalBank` of turn/weather_dur/terrain_dur/tr_dur | d_model | 16 |
| 4 | `transition_token` | small MLP from `TRANSITION_CONT_DIM=51` (features.py:957) + 2× action embed | d_model | 17 |
| 5 | `summary_token[0]` | `nn.Parameter` (learnable scratch) | d_model | 18 |
| 6 | `summary_token[1]` | `nn.Parameter` (learnable scratch) | d_model | 18 |

**Total per turn: 210 tokens** (12 × 17 + 6).

### 3.2. The move token (richest non-trivial token)

Each move's token is built from a small MLP — `MoveTokenizer` — that takes
the same inputs `MoveNet` (model.py:124-157) does today, but keeps them
*per-token* instead of mixing into a Pokemon-level concat:

**Inputs:**
- `move_id` embedding (`n_moves=953` × `entity_embed_dim=32`)
- `bp_bank` (`NumericalBank(256, bank_dim=16)`)
- `acc_bank` (`NumericalBank(101, bank_dim=16)`)
- `pp_bank` (`NumericalBank(65, bank_dim_small=8)`)
- `prio_bank` (`NumericalBank(13, bank_dim_small=8)`)
- 107-dim continuous feature vector from `_project_move_flags`
  (features.py:1005-1263), which already contains: drain, recoil, heal,
  multihit, contact, sound, punch, bite, powder, protect-blocked, crit
  ratio, secondary effect probabilities, hazards/weather/terrain setup,
  type one-hot, category one-hot, target one-hot, and 7-status applied
  one-hot. Verified in §3 of the features.py audit.

**Module:**
```python
class MoveTokenizer(nn.Module):
    # ~190-dim concat → 2-layer MLP → d_model
    in_dim = 32 + 16 + 16 + 8 + 8 + 107 = 187
    Linear(187, d_model) → GELU → LayerNorm → Linear(d_model, d_model) → GELU → LayerNorm
```

**Critical design point — solves S44's "B fix":** the `MoveTokenizer` is
called for *all* 4 moves on *all* 12 Pokemon (active and team), with the
same real bank values fed in everywhere. No more `torch.zeros(...)` for
team moves like model.py:701-702. The bank values are deterministic from
move_id (each move has fixed BP/acc/PP/prio in poke-env's data), so we
look them up at tokenization time. Implementation tip: precompute a
`(n_moves, 4)` lookup table at model init from poke-env's `Move` data
class; this avoids per-forward dict lookups.

The 107-dim flag vector is also computed for *all* 4 moves (not just
the active 4), unlike the current code where features.py only fills
this for the active set and uses 23-dim compact for team moves
(features.py:431-454, the `_encode_move_compact`). This subsumes both
S44's "B" fix (banks for team moves) AND the deferred "B+" expansion
(real flags for team moves), at no additional architectural cost since
attention scales with token count, not feature dim.

**Per-Pokemon active-vs-team distinction:** the move's tactical context
(type effectiveness vs current opponent, opponent threat back) varies
by who's active. Today this is dim 107-108 of the active-move encoding
only (features.py:1318-1320). In the new arch, we add 2 specialized
tokens per *active* Pokemon (1 per side):

| Token | When | Computed |
|---|---|---|
| `our_active_threat_token` | only for our active | `_compute_type_effectiveness` per move + `_max_opp_threat` over all 4 moves |
| `opp_active_threat_token` | only for opp active | symmetric |

That's 2 extra tokens (1 per side) per turn. New total: **212 tokens**.

### 3.3. Positional / type / slot disambiguation

The transformer has no inherent notion of "this is Pokemon 3, slot HP"
vs "this is Pokemon 4, slot HP." Three embeddings combine into the
input to layer 0:

1. **`type_id_embed[token_type]`** — 19-vocab lookup (types 0-18 above).
   Tells the model "this is a stat_atk token" vs "this is a move token."
2. **`pokemon_slot_embed[which_pokemon_0_to_11]`** — 12-vocab lookup.
   Pokemon 0-5 are ours, 6-11 are opp. Decision tokens (actor, critic,
   field, transition, summary) get a separate slot embedding (12-15).
3. **`move_slot_embed[which_move_0_to_3]`** — 4-vocab lookup, applied
   *only* to move tokens. Distinguishes "this is move slot 0 of this
   Pokemon" from "move slot 1." Non-move tokens get the zero embedding.

All three are learnable; total ID-embedding overhead: `(19 + 16 + 4) ×
d_model ≈ 39 × 256 = 10K parameters`. Trivial.

### 3.4. Side disambiguation and information leakage

Token type 12 (Pokemon slot) implicitly encodes side: slots 0-5 are ours,
6-11 are opp. But we want *attention masking* to also enforce that
private info stays private:

- For tokens marked "our hidden info" (e.g., our team's full ability,
  item, exact stats, all 4 moves' raw bank values), the *opponent's*
  active threat token shouldn't attend to them. We don't enforce this
  in our model since we ARE one side — but for the symmetry of training
  data (BC dataset has both sides as samples), we use a **side mask**:

  Side mask construction:
  - All tokens have a `side_id` (0 = ours, 1 = opp, 2 = neutral/field)
  - Decision tokens are side 0 (we're always playing as side 0 from our
    perspective; the BC dataset is normalized to "our perspective" already).
  - The poke-mask (model.py:372-384) blocks state-to-decision attention.
    The new side_mask additionally blocks our decision tokens from
    attending to opp's hidden info — but for opp-revealed info, attend
    freely. This matches features.py's revealed-only logic for opp:
    item=0 if not revealed, ability=0 if not revealed, moves use
    `move_id=0` for unrevealed. So in practice, the opp tokens for
    unrevealed attributes are already "unknown" embeddings, and side
    masking adds defensive consistency without changing observed
    behavior.

The full mask matrix is conceptually:

```
              actor critic state_us state_opp_revealed state_opp_hidden summary
actor          ✓     ✗      ✓        ✓                 (none)          ✓
critic         ✗     ✓      ✓        ✓                 (none)          ✓
state_us       ✗     ✗      ✓        ✓                 (none)          ✓
state_opp_rev  ✗     ✗      ✓        ✓                 (none)          ✓
summary        ✗     ✗      ✓        ✓                 (none)          ✓
```

The "state_opp_hidden" column is empty by construction — features.py
yields zero/unknown embeddings for unrevealed opp info, so there are
no "hidden" tokens in our forward. We don't need an explicit mask;
the design just ensures we never *create* hidden tokens.

### 3.5. Token count rationale (the alternative considered)

Per the prompt, the big trade-off is: ONE rich move token per move
(48 move tokens battle-wide) vs SEVERAL sub-tokens per move (move_id,
move_bp, move_acc as separate tokens, ~192-384 move-related tokens).

**Decision: ONE rich token per move.** Rationale:

- **Compute scaling.** Attention is O(N²). Going from 48 move tokens to
  192-384 increases total tokens from 210 to ~370-560. Squared cost
  goes 4-7×. With FlashAttention this is still tractable on a 6 GB GPU
  (we measured 14M-param current arch at ~7 min/iter; new arch at 25M
  params and 210 tokens with FA should be ~10-12 min/iter; at 370+
  tokens it would push to 20-30 min/iter, doubling our wall-clock for
  the run).
- **Move attribute redundancy.** A move's properties (BP, acc, type,
  flags) move together — they don't independently interact with anything
  in the battle state in a way that benefits from separate attention.
  Splitting move into 4-8 sub-tokens would mostly waste compute on
  attention pairs that learn near-zero cross-attention.
- **Compositional bottleneck only at move level.** The MLP bottleneck
  diagnosed in S44 was at the *Pokemon* level (everything mashed into
  one token). Moves were already sub-encoded by `MoveNet` separately
  (model.py:124-157), so per-move MLP is well-precedented and not the
  bottleneck. The fix is to surface those move tokens to spatial
  attention directly, not to further sub-decompose them.

If a future iteration finds movewise compositional gaps (e.g., "the
model can't compose Stone Edge's accuracy with the user's Compound
Eyes ability"), we can add sub-token decomposition at that point.
This rewrite leaves room: `MoveTokenizer` is one module, and replacing
its emit-1-token logic with emit-N-tokens is a localized change.

---

## 4. Attention architecture

### 4.1. Spatial transformer

- **Layers:** 6 (vs 4 in current model.py:346-347)
- **Heads:** 8 (vs 4 in current model.py:336-344). 8 heads at d_model=256
  gives 32d per head, matching common-practice scaling.
- **d_model:** 256 (vs current `d_spatial=256` per S39 capacity reshape;
  unchanged at the spatial side).
- **Feed-forward:** 4× expansion (matches current `ff_mult=4`,
  model.py:339).
- **Dropout:** 0.05 (matches Metamon default + current code, model.py:45).
- **Norm:** pre-norm (matches current `norm_first=True`, model.py:343).
- **Total tokens at input:** 212 (210 base + 2 active threat tokens).
- **Mask:** Poke-Mask + side-mask described in §3.4. Same -inf additive
  mask convention as model.py:372-384.
- **Output:** `(B, 212, d_model)`.

### 4.2. Summary scratch tokens (collection)

After spatial pass:
- Decision tokens (actor, critic) are read out at indices [0:2].
- Summary scratch tokens [last 2 indices] become the per-turn
  representation fed to the temporal transformer.
- Two scratch tokens × 256 dim flattened = 512-dim per-turn summary.

This matches Metamon Small's recipe (4 latents × 100 dim = 400-dim
per-turn summary; we use 2 × 256 = 512 — similar order of magnitude).
The current arch uses K=0 (single attention pool) — the scratch-token
path already exists (model.py:401-403, 428-430) and is just enabled
by setting `n_summary_tokens=2`. The key change is *flatten + project*
to d_temporal happens to a richer 512-dim input rather than 256-dim.

### 4.3. Temporal transformer

- **Layers:** 4 (vs 2 in current model.py:464). Metamon's recipes use
  more capacity here (5-8× temporal:spatial in d_model), and
  METAMON_LEARNINGS.md flags this as a key under-investment in our
  current arch.
- **d_temporal:** 256 (down from current 512). The spatial summary is
  already richer (2 scratch × 256 = 512), so the per-step input to
  temporal is wider; the temporal stack itself stays narrow. This
  approximates Metamon's "wide spatial output + narrow temporal" rather
  than current's "narrow spatial output + wide temporal."
- **Heads:** 8.
- **FF:** 4×.
- **Causal mask:** same as model.py:489-490.
- **Position embedding:** learnable, 200-position cap (matches current
  model.py:451). Stays the same.
- **Padding mask:** for variable-length sequences (matches current
  model.py:493-495).
- **Output:** the last valid timestep's d_temporal vector
  (matches current model.py:504-509).

### 4.4. Why keep temporal *separate* (not unify across turns × tokens)

The prompt explicitly recommends keeping spatial and temporal
transformers separate:

> "Unified attention over many turns × many tokens explodes compute."

Concretely: 200 turns × 212 tokens = 42,400 tokens in one unified
attention. That's `42400² ≈ 1.8B pairs` — even FlashAttention can't
make this fit on a 6 GB GPU. Keeping the 200×512 temporal sequence
separate reduces it to `200² + 212² ≈ 45K + 45K = 90K pairs` per
turn, which is fine.

The cost is some lost cross-temporal attribute attention ("our atk
boost from turn 5 + opp's lowered def from turn 7 + this turn's STAB
move"). In practice the 2 summary scratch tokens per turn carry enough
of this signal forward that the explicit cross-temporal attribute
attention isn't needed — Metamon's models work this way and reach
71% GXE on gen9OU (Kakuna, in their testing).

### 4.5. Total attention compute estimate

Per turn forward:
- Spatial: 212 tokens × 6 layers × 8 heads × 32 dim/head + FFN ≈ 25M ops
- Temporal: 200 turns × 4 layers × 8 heads × 32 dim/head + FFN ≈ 5M ops
- Action encoder (per-action context for 9 actions): negligible (4M ops)
- Total: ~35M ops per turn forward.

With FlashAttention 2 at fp16, this should run in ~1.5-2ms per turn
forward on RTX 3060 Laptop. At 60 turns/battle, ~120ms/battle, ~6
samples/sec collected. Roughly 2-3× slower than current arch but well
within the wall-clock budget for a multi-week training run.

For training: BC training over a 100k-replay dataset at batch=64 turns
should run at ~200-300 turns/sec on the same hardware (memory-bound by
the 12 Pokemon × 17 tokens reshape). ~10 hours per BC epoch (5M turns
total). Multi-day BC training for ~5-10 epochs.

---

## 5. Action heads + value head

### 5.1. Action head (preserved with minor changes)

The current `ActionSlotEncoder` (model.py:522-571) consumes 4 active
moves (with their full `_project_move_flags` 109-dim vectors and real
bank values) and 5 switch slots, projects each through MoveNet/switch
MLP, and emits `(B, 9, d_model)` per-action context.

In the rewrite:
- The 4 active moves are *already* tokenized by `MoveTokenizer` as part
  of the spatial input (move tokens 14-17 of our active Pokemon).
  Reading those 4 tokens out of the spatial transformer's output gives
  the per-action context "for free" — they've already been refined by
  6 layers of attention against the rest of the battle state.
- The 5 switch slots (other Pokemon on our team) similarly: their
  species_token, hp_pct_token, type_token, status_token, etc. have
  already been computed. We aggregate per-Pokemon: take the 17 tokens
  for that Pokemon and pool them (mean or attention-pooled with a
  learnable query) to get a single (B, d_model) vector. 5 such
  per-Pokemon vectors stack into `(B, 5, d_model)`.

Output: `(B, 9, d_model)` per-action context — same shape as today.

The policy head (model.py:633-637) consumes:
- actor token output: `(B, d_spatial)` from spatial transformer index 0
- temporal_ctx: `(B, d_temporal)` from temporal transformer
- action context: `(B, 9, d_spatial)` per above

Compute: `Linear(2*d_spatial + d_temporal, max(d_spatial, d_temporal))
→ GELU → Linear(..., 1)`. Reshape to `(B, 9)` logits. Same as today.

Legal mask applied identically (model.py:787-789).

### 5.2. Value head (preserved exactly)

`Linear(d_spatial + d_temporal, max(d_spatial, d_temporal)) → GELU →
Linear(..., 51)` with twohot encoding (model.py:805-816). No changes.

The critic token output is at spatial transformer index 1, exactly like
today.

### 5.3. Popart on the value head (small bolt-on, recommended)

Per `METAMON_LEARNINGS.md` and S44 deferred items, Metamon uses popart
normalization on the value head for value-scale stability across long
runs. We've never enabled it. This rewrite is the right time to add it
since we're rebuilding the model anyway.

Popart adds: a learned `(scale, shift)` pair that normalizes the value
target to unit-ish variance, and the corresponding inverse mapping at
inference. Implementation: ~30 lines of code, called from PPO update
in `ppo.py`. **Implement in §7 Week 5+ as a non-blocker but include in
V1.**

---

## 6. Training pipeline

### 6.1. BC dataset

**KEY DECISION: do not regenerate the memmap.**

The existing memmap at `data/datasets/human_v8_100k/` (104 GB, 5.08M
records, 199.9K episodes; details from dataset.py audit) already stores
all the structured features we need:

- `our_pokemon_ids` (B, 6, 7): species, item, ability, 4× move_id
- `our_pokemon_banks` (B, 6, 10): hp%, level, weight, height, 6× stats
- `our_pokemon_cont` (B, 6, 285): types(19) + status(7) + boosts(91) +
  flags(2) + volatiles(38) + paradox(7) + tera(20) + combat(5) +
  toxic(1) + future_sight(1) + visibility(2) + 4× move_compact(92)
- `our_pokemon_mcont` (B, 6, 4, 23): per-Pokemon per-move compact
  features (type one-hot + BP + category + priority)
- ... and the symmetric opp fields, plus field/transition/active-move/
  switch/legal/action/result.

The new tokenizer can be implemented as **slicing operations + small
MLPs at training time** that consume these structured tensors. Concrete
plan:

| Token | Built from | Memmap fields |
|---|---|---|
| `species_token` | embedding lookup | `our_pokemon_ids[:, :, :, 0]` |
| `item_token` | embedding lookup | `our_pokemon_ids[:, :, :, 1]` |
| `ability_token` | embedding lookup | `our_pokemon_ids[:, :, :, 2]` |
| `type_token` | MLP from 2 type embeds | slice `our_pokemon_cont[:, :, :, 0:19]` (the type multi-hot; recover top-2 indices) |
| `status_token` | MLP | slice `our_pokemon_cont[:, :, :, 19:26]` (status) + `:, :, :, 119:157]` (volatiles) + `:, :, :, 157:164]` (paradox) + `:, :, :, 164:184]` (tera) |
| `hp_pct_token` | NumericalBank | `our_pokemon_banks[:, :, :, 0]` |
| `boosts_token` | MLP from 7 boost values | slice `our_pokemon_cont[:, :, :, 26:117]` (the 91-dim 7×13 one-hot; recover boost values via argmax-and-rescale) |
| 6× `stat_token` | NumericalBank | `our_pokemon_banks[:, :, :, 4:10]` |
| `move_token[i]` | rich MoveTokenizer | `our_pokemon_ids[:, :, :, 3+i]` (move_id) + `our_pokemon_mcont[:, :, :, i, :]` (23-dim) + lookup of bp/acc/pp/prio from move_id (deterministic) + computed 107-dim `_project_move_flags` |

The only piece that requires the *full* 107-dim move flag vector for
team moves (not just active) is the new `MoveTokenizer`. Two options:

**Option A (preferred for V1): compute flags at training time from move_id.**
poke-env's `Move` data class (already used by features.py) has all the
flag info; the 107-dim projection is a deterministic function of
move_id alone (independent of battle state, except for type
effectiveness which is *not* needed for team moves — that's only for
active). Implement as a `(n_moves, 107)` lookup table built once at
model init from poke-env's move data, then indexed by move_id. **Cost:
zero per-forward overhead beyond a tensor index. Memmap unchanged.**

**Option B (later, if A's lookup table feels brittle): regenerate
memmap with explicit per-move 109-dim feature vectors stored.** Adds
~6 × 4 × 109 × 4 bytes = ~10 KB per turn, scales the 104 GB memmap to
~115 GB. Days of regeneration time. Defer unless A's lookup turns out
to miss some battle-state-dependent info.

Going with Option A. **Memmap: unchanged. BC training data: 100k
human replays, already there.**

### 6.2. BC training

- **Loss:** unchanged from train_bc.py:30-119. Cross-entropy over
  legal actions + 0.5× value loss (categorical cross-entropy via
  twohot encoding).
- **Batch size:** start at 32 (current default; train_bc.py defaults
  vary). Tune based on GPU memory. The new arch is ~25M params (rough
  estimate, see §4.5 capacity calculation; refine after Week 1
  forward-pass validation), vs current 14M. Memory budget on RTX 3060
  Laptop should support batch ≥ 32.
- **Optimizer:** AdamW, same as today (train_bc.py:408-409).
- **LR:** 3e-4 for BC (same as current per train_bc.py defaults).
  Warmup 200 steps, cosine decay over the run.
- **Epochs to convergence:** unknown until we run it. Current MLP arch
  converged at epoch 2 on `human_v8_100k` (smart_avg 45.1% per S39).
  New arch with more parameters and richer attention may need 5-10
  epochs to converge. Plan for ~3-5 days of BC training before PPO.
- **Mixed precision:** fp16 (train_bc.py supports `--fp16`).
- **AMP:** keep summary attention in fp32 (matches current
  model.py:421-426 with `torch.amp.autocast("cuda", enabled=False)`).

### 6.3. PPO continuation

Identical to current pipeline. The model class swap is the only change
to `train_rl.py`:
- `--init-from <new_arch_bc_checkpoint>.pt` instead of the v9 BC.
- All 5 defense layers (forfeit filter, queue restart, dispatch
  watchdog, auto-respawn, pool curation) work unchanged.
- `--external-adapters external_adapters_curated.yaml` — same external
  pool.
- PFSP, EMA win rate tracking, adaptive entropy, early stop — all
  unchanged.

### 6.4. Training teams (PROJECT-WIDE STANDARD)

ALL training MUST use procedurally-generated teams from Smogon usage
stats with `--random-team-pct 0.05`:

```
--procedural-teams C:/Users/raiad/OneDrive/Desktop/team_builder/raw_data/pokemon_usage/2024-04
--random-team-pct 0.05
```

This is enforced by `train_rl.py`'s pre-flight guard (which raises if
`--procedural-teams` is missing). The 70-team pool is **not** for
training (caused thousands of iters of overfitting in S35-S38 history).
Metamon competitive teams are eval-only.

### 6.5. Eval methodology (PROJECT-WIDE STANDARD)

ALL evals MUST use Metamon competitive teams via `--eval-team-set
metamon-competitive`. The 70-team pool's 51-pt smart_avg spread (TEAM_AX
81.5% vs TEAM_AR 30.5% per S43 retrospective) makes per-checkpoint
comparisons noisy enough that eval signals can't be distinguished from
team-draw variance. The Metamon competitive 16-team set has tight
variance and ladder-validated provenance.

**Eval-time team source ≠ training-time team source.** Mixing them
contaminates results. Keep separate.

### 6.6. Smart_avg target and significance threshold

- **Baseline:** sp_0229 = 67.8% smart_avg on Metamon competitive.
- **Same-policy noise floor:** ±2.2pt at 500 games × 4 bots (per S44
  measurement bumped from S43's 200×4 ±3.6pt).
- **Significance bar:** ≥3pt above baseline = ≥70.8% smart_avg
  sustained over 3+ consecutive evals = real improvement.
- **Failure threshold:** ≥3pt below baseline = ≤64.8% sustained over 2+
  consecutive evals = stop and re-investigate.

These match the current PPO run's success/failure criteria
(NEXT_SESSION.md "What to watch for"). We're carrying the same
measurement discipline forward.

---

## 6b. Heterogeneous opponent support — keep the legacy player alive

### 6b.1. Constraint

The new architecture and the existing MLP architecture must coexist as
poke-env Players that can play battles against each other through
`battle_server.js`. Both implement the poke-env `Player` interface;
neither shares internal state. The WebSocket protocol is the only
contract.

### 6b.2. Implementation

**Keep `battle_agent.py:23` (`class BattleAgent(Player)`) as-is.** Do
not delete, refactor, or renamerefactor for the duration of the rewrite.
It's the legacy player loader — it knows how to load
`PokeTransformerConfig` checkpoints, run the v9 obs pipeline (via
features.py:make_features), and play battles. It also has the
dim-expansion checkpoint-loading logic at battle_agent.py:53-71 that
handles pre-S39 checkpoints (sp_2979 era) by zero-padding new feature
columns.

`battle_agent.py` also depends on:
- `model.py` (PokeTransformer, PokeTransformerConfig)
- `features.py` (make_features, MOVE_SLOT_CONT_DIM, SWITCH_SLOT_CONT_DIM)
- `vocab.py` (no changes needed)

**None of these dependencies change.** model.py and features.py are
both kept around (they're used by BattleAgent's checkpoint loader).
The new arch's model/features live in *new modules*, not as edits to
the existing ones. Specifically:

- **New file `model_transformer.py`** — contains `TransformerBattlePolicy`,
  `TransformerConfig`, `MoveTokenizer`, `Tokenizer`, etc. ~600-800
  lines (similar to current model.py at 1033 lines, smaller because
  much of model.py is the obsolete MLP encoders).
- **New file `features_transformer.py`** — contains
  `make_features_transformer(battle)`, the per-token feature extractor.
  Initially this is mostly a thin wrapper around the existing
  `features.py:make_features` (since the memmap is unchanged) plus the
  per-token tensor reshaping, ~300-500 lines.
- **New file `battle_agent_transformer.py`** — contains
  `class BattleAgentTransformer(Player)`. Mirror of battle_agent.py
  but uses the new model + new feature extractor. ~250 lines (similar
  to existing battle_agent.py at ~150-200 lines).

**The new files import from the new model/features modules; legacy
files keep their existing imports. No circular dependencies, no
shared mutable state.** The only shared module is `vocab.py` (vocab
sizes are constant across architectures).

### 6b.3. Heterogeneous opponent use cases unlocked

1. **Direct head-to-head Elo measurement.** Spawn
   `BattleAgent('sp_0229.pt')` and `BattleAgentTransformer('new_ckpt.pt')`
   in the same battle pair. Run 200+ games. Get a direct measurement
   of "did the rewrite beat the MLP-arch champion?" without needing a
   proxy metric. (S44 didn't have this option clean; doing this
   first-class avoids future awkwardness.)

2. **Optional legacy opponents in PFSP pool during PPO of the new
   arch.** In `external_adapters_curated.yaml`, add entries like:
   ```yaml
   - name: legacy-sp_0229
     adapter: legacy_self_play
     checkpoint: data/models/.../sp_0229.pt
     weight: 0.3
   ```
   `external_opponent_manager.py` already supports heterogeneous
   opponent types (mcts, foulplay subprocess, metamon subprocess,
   self-play); adding a `legacy_self_play` type is incremental — same
   in-process Player path as `SelfPlayOpponent` (rl_player.py:267-282)
   but instantiates `BattleAgent(sp_0229.pt)` instead of using the
   trainer's current model.

3. **Backward-compatible eval.** Any future analysis of MLP-arch
   lineage (Elo trajectories, per-team analysis, ladder rerating)
   stays possible. Old checkpoints don't go to the dust bin.

### 6b.4. Cost

~600-800 lines of legacy code (battle_agent.py + model.py +
features.py-the-MLP-parts) maintained as read-only "vendor in our own
repo." Acceptable. Pure additive complexity. The trade-off is
clear-eyed: we don't do this and we lose the ability to measure new
arch vs old arch directly, which is the strongest signal we'll get.

### 6b.5. Guard rail in the implementation roadmap

> **DO NOT delete or substantially refactor `battle_agent.py`,
> `model.py`, `features.py` during the rewrite.** New arch lives
> alongside, not in place of. If you find yourself touching these for
> any reason other than adding `BatleAgentTransformer` to the
> `external_opponent_manager.py` adapter table, stop and reconsider.

---

## 7. Implementation roadmap

Six weeks of focused work. Each week ends with a measurable milestone;
if the milestone slips, that's the trigger to update the design.

### Week 1: Tokenizer module + integration tests — DONE (Session 46)

**Deliverable:** `model_transformer.py` exports `Tokenizer` that
consumes the existing memmap batch format and emits a `(B, 212,
d_model)` tensor + a slot/type ID tensor for positional embeddings.

**Tasks:**
- [x] Create `model_transformer.py` skeleton: `TransformerConfig`,
  `Tokenizer`, `MoveTokenizer` modules.
- [x] Implement `MoveTokenizer` per §3.2.
- [x] Implement `Tokenizer` that unpacks the memmap's per-Pokemon tensors
  into per-attribute tokens per §6.1.
- [x] Build the `(n_moves, 107)` flag lookup table from poke-env Move
  data. Confirm it matches `_project_move_flags` (features.py:1005-1263)
  on a sample of 50 moves.
- [x] Write unit tests: `test_tokenizer.py` with 5 sample battles
  (loaded from existing memmap), assert token count = 212, no NaNs,
  type_id embeddings line up with the spec table.
- [x] Forward-pass benchmark: time `Tokenizer + dummy spatial` on a
  single batch of 32 turns. Should be <50ms on RTX 3060 Laptop.

**Milestone:** Tokenizer runs. Token shapes match spec. No silent NaN.

**Session 46 results:**
- 952/952 valid moves built into `data/lookup/move_flags_v1.pt` (move_id 0
  reserved for pad/unknown). 50/50 sampled moves pass strict re-call exact
  match against `_project_move_flags(move)`. Active-path call with
  `poke_types=(move.type,)` diverges only at the documented STAB index — no
  unexpected battle-state-dependent components.
- Tokenizer on RTX 3060 Laptop, B=32 turns: **3.42 ms median for tokenizer
  alone, 28.36 ms median for tokenizer + dummy 6-layer spatial transformer**.
  Well under the 50 ms budget; ~0.89 ms/turn end-to-end.
- 5/5 unit tests pass on real `human_v8_100k` memmap data: shape =
  (B, 212, 256), no NaN/inf, all 20 token types present, multi-turn forward
  works, opp Pokemon with unrevealed move_id=0 produces finite output, type_id
  embeddings demonstrably contribute (zeroing them changes output by
  max |Δ|=3.85), pokemon_slot_embed disambiguates same-attribute tokens
  across slots.
- New files: `model_transformer.py` (1.42M params for Tokenizer +
  MoveTokenizer), `verify_move_lookup.py`, `test_tokenizer.py`,
  `bench_tokenizer.py`. Lookup at `data/lookup/move_flags_v1.pt`
  (~840 KB on disk). Total Week 1 footprint stays inside the legacy-untouched
  guard rail (no edits to `model.py` / `features.py` / `battle_agent.py`).

### Week 2: Spatial + temporal stack, replace model architecture — DONE (Session 47)

**Deliverable:** `TransformerBattlePolicy` runs forward end-to-end,
produces sane action_logits and value scalar on dummy memmap data.
PPO integration test.

**Tasks:**
- [x] Implement `SpatialTransformer` (6L, 8H, d_model=256, K=2 scratch
  tokens, Poke-Mask + side-mask).
- [x] Implement `TemporalTransformer` (4L, 8H, d_model=256, causal,
  pos embeddings).
- [x] Implement action head (per §5.1) and value head (per §5.2,
  identical to current).
- [x] Implement `TransformerBattlePolicy.forward(batch, history)`:
  same interface as `PokeTransformer.forward` (model.py:739-803), so
  it slots into existing collection/training code.
- [x] Implement `TransformerBattlePolicy.forward_sequence(collated,
  device)` for BC training (mirror of model.py:818-978).
- [x] Sanity check: load a sample BC batch, forward through new model,
  assert logits shape = (B, T, 9), value shape = (B, T), no NaN,
  legal-action masking works.
- [x] PPO smoke (synthetic; real-PPO integration deferred to Week 3 per
  the design's stated alternative). `test_policy.py::test_synthetic_ppo_step`
  runs `forward_sequence` on a real `human_v8_100k` BC batch, computes
  fake advantages, takes one AdamW step. All 246 parameters update;
  total grad-norm ≈ 1.57 (finite, far from explosion).

**Milestone:** TransformerBattlePolicy runs end-to-end. 9/9 policy
tests + 17/17 tokenizer tests pass. Synthetic PPO step proves the
model trains stably.

**Session 47 results (after Postscript H capacity bump):**
- Total params: **19,994,924** — d_temporal 256 → 512 and K 2 → 4 to
  match Metamon Small's recipe and lift T:S ratio from 1.0× to 2.0×.
  See Postscript H for the data + math.
- Param breakdown: Tokenizer 1.01M / Spatial 4.74M / Temporal 12.71M /
  Action head 525K / Value head 420K / Switch encoder 67K /
  Summary→Temporal proj 525K.
- Single-turn full-policy forward, B=32, RTX 3060 Laptop CUDA fp32:
  **36.45 ms median (per-turn 1.14 ms)**. Target <80 ms; only +0.75 ms
  vs the 9.5M version because temporal at T=1 has trivial attention.
- forward_sequence + backward (B=8, 190 valid turns): **~4.8 s**
  (~25 ms per valid turn). Same quadratic-T temporal recomputation
  shape as legacy v9.
- `test_policy.py`: 9/9 tests pass (single-turn forward, multi-turn
  history accumulation, forward_sequence shapes, param breakdown,
  value-head softmax + twohot round-trip, switch-context species match,
  move-context permutation, gradient flow, synthetic PPO step).
- `test_tokenizer.py`: 17/17 still pass (no regressions).

### Week 3: BC training to BC convergence

**Deliverable:** New-arch BC checkpoint at smart_avg comparable to
existing v8 BC's 45.1% (matched within ±2pt).

**Tasks:**
- [ ] Add `--use-transformer` flag to `train_bc.py` that swaps
  `PokeTransformer` → `TransformerBattlePolicy` and uses the new
  collate path.
- [ ] Run BC smoke: 1 epoch on first 1k episodes. Verify loss decreases,
  no NaN, gradient norm sane. If slow, profile and optimize.
- [ ] Run full BC: 5-10 epochs on `human_v8_100k`. Monitor val_loss,
  val_acc, periodic smart_avg eval (every epoch end).
- [ ] Pick best epoch by smart_avg (Metamon competitive, 200×4 games).
  Record as `bc_v10_<timestamp>/best.pt`.

**Milestone:** New arch BC reaches ≥45% smart_avg. If <40%, debug; if
50%+, that's a positive surprise and we proceed with extra confidence.

**Cloud option:** if local BC training takes >5 days at full epoch
count (likely on RTX 3060 Laptop), trigger cloud BC training (see
CLOUD_DEPLOY.md). Probable cost: $50-150 for BC convergence on a
single A100.

### Week 4: PPO from BC, eval methodology validated

**Deliverable:** First new-arch PPO checkpoint that beats 67.8% smart_avg
on Metamon competitive.

**Tasks:**
- [ ] Launch PPO from new BC checkpoint with the same recipe as the
  attempt-11 run (NEXT_SESSION.md section "Resume the run if you
  killed it"). Adjust `--init-from` and add `--use-transformer`. Pool
  composition: same curated YAML.
- [ ] Run 50 iters with frequent eval (`--eval-interval 5`). Monitor
  smart_avg trajectory. Compare to MLP arch's iter 0-50 trajectory
  from attempt 11 (will have to look this up in the run log).
- [ ] Mid-run sanity checks: per-iter pool size sane, KL drift below
  0.04, entropy in [0.6, 0.85], no Layer 1/2/3/4/5 fires beyond cold
  start.

**Milestone:** Iter ≥30 hits ≥70% smart_avg sustained over 2+ evals.
If the trajectory looks like attempt-11 (peak 64-66% then plateau),
that's a clear signal the architectural fix wasn't decisive — proceed
to deeper analysis (see Week 6).

### Week 5: Long PPO run + popart + final eval

**Deliverable:** Final new-arch checkpoint. Smart_avg comparison vs
sp_0229 baseline. Direct head-to-head Elo measurement via the new
heterogeneous-opponent infrastructure (§6b.3.1).

**Tasks:**
- [ ] Implement popart on the value head per §5.3. Add as a flag
  (`--popart`) that defaults to True for new arch.
- [ ] Resume PPO from Week 4 best, run another 100-150 iters with
  popart on. Monitor for value-scale stability gain (the diagnostic is
  whether avg KL stays tighter and whether smart_avg trajectory has
  fewer dips).
- [ ] At final iter or every 20 iters, run direct H2H eval:
  `BattleAgentTransformer(new.pt)` vs `BattleAgent(sp_0229.pt)` 200
  games. Goal: ≥55% WR (clearly better than the 50% noise floor).

**Milestone:** Final checkpoint at ≥71% smart_avg sustained over 3+
evals AND ≥55% WR vs sp_0229 in direct H2H. If both hit, the rewrite
delivered.

### Week 6: Analysis, retrospective, documentation

**Deliverable:** Updated NEXT_SESSION.md. Per-opp analysis of new arch
vs MLP arch. Decision on whether to scale up further (cloud, larger
model) or call the rewrite done.

**Tasks:**
- [ ] Run `_analyze_run_log.py` on the long PPO run. Per-opp W/L
  trajectory: where did new arch gain over MLP arch? where did it
  plateau?
- [ ] Compare attribute-level token activations on a sample of 20
  battles (use forward hooks). Sanity-check: are token-type embeddings
  distinct? Do unusual sets activate "novel" attention patterns vs
  common sets activate familiar ones?
- [ ] Update REWRITE_DESIGN.md (this doc) with all reality-vs-spec
  divergences encountered during implementation. Don't edit history;
  add postscript sections explaining what changed and why.
- [ ] Update NEXT_SESSION.md with the new arch's status, baseline
  smart_avg, and recommended next directions (more BC data? larger
  model? test-time adaptation?).

**Milestone:** Project state updated. Next direction chosen.

### Decision points / off-ramps

| Week | Trigger | Action |
|---|---|---|
| 1 | Tokenizer benchmarks >100ms/batch | Profile, optimize, possibly defer K=2 → K=4; otherwise simplify token count |
| 2 | New model fails to forward / produces NaN | Debug; if architectural issue, reconsider d_model / layer count |
| 3 | BC <40% smart_avg after 10 epochs | Investigate: tokenizer bug? MLP collapse on novel attribute combos? Possibly fall back to smaller token count and retest |
| 3 | BC training takes >5 days local | Trigger cloud migration |
| 4 | PPO trajectory mirrors attempt-11 (peak 64-66% then plateau) | New arch isn't the fix. Pivot to data scaling (more BC replays, multi-gen) OR test-time adaptation OR larger model. The rewrite is then a building block, not the answer. |
| 5 | Final smart_avg <70% | Same as Week 4: data/scale/TTA pivot |
| 5 | Final smart_avg ≥75% | Big win. Plan a ladder submission run + larger model + more BC data |
| 6 | Direct H2H WR <50% vs sp_0229 | Strange: smart_avg up but H2H down? Likely a Metamon-competitive teams effect; investigate |

---

## 8. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | BC training too slow on local 6 GB GPU (new arch ~25M params, 212 tokens, 5M turn dataset) | medium | medium | Cloud option budgeted (Week 3). Can also reduce `n_summary_tokens` from 2 to 1, drop temporal layers from 4 to 3, or batch-size down |
| 2 | New arch trains unstably (NaN, exploding gradients, exploding KL) | medium | high | LR sweep (Week 2-3), grad clip 2.0, layer norm everywhere, fp16 AMP with master fp32 weights for sensitive ops |
| 3 | Tokenizer slicing logic has a subtle off-by-one or wrong-side-of-mask bug | medium | high | Unit tests in Week 1, integration tests in Week 2. Compare token-by-token output to a reference implementation that uses for-loops instead of vectorized slicing |
| 4 | Memmap-with-no-regen approach hits a feature gap (some new attribute we want isn't sliceable from existing fields) | low | medium | Fall back to Option B: regen with explicit per-token features. ~2-3 days additional cost in Week 1 |
| 5 | Eval methodology drift (Metamon competitive teams change, or some new bot replaces an existing eval bot) | low | medium | Pin the Metamon competitive set version to git commit hash. Bots are SimpleHeuristics, SmartDamage, Tactical, Strategic — all in-tree |
| 6 | Heterogeneous opponent infrastructure (BattleAgent vs BattleAgentTransformer in the same battle) has a protocol incompatibility | low | medium | Test in Week 5 with `diag_cross_venv.py` style sanity check before PPO uses it. The existing battle_server already handles 4 separate Player types; one more shouldn't break it |
| 7 | New arch overfits to the BC dataset and doesn't generalize ("memorizes the 100k human replays") | low | high | Same risk as MLP arch had; smart_avg eval on Metamon competitive teams is the early warning. Dropout 0.05, weight decay matching current |
| 8 | New arch wins at smart_avg but loses on PokeAgent ladder (different distribution of opponents) | medium | low | Ladder submission is post-Week 6. Acceptable risk; the smart_avg measurement is the project-wide standard |
| 9 | Compute budget overrun: cloud BC + cloud PPO + Storage > $300 | medium | low | Cap individual run cost at $200; if cloud BC alone exceeds that, reduce model size before scaling further |
| 10 | Scope creep: implementing test-time adaptation / multi-gen / larger model in the rewrite session | medium | high | Hard guard: this design is V1. New scope = new design doc, post-Week 6 |
| 11 | The architectural rewrite turns out to be insufficient (the ceiling really IS data scale, not architecture) | medium | high | This is the central uncertainty. Week 4-5 evals are the dispositive test. If we hit this, the rewrite is still a building block; pivot to data scaling |
| 12 | Compute estimates in §4.5 are wrong, and the real model is 3-5× slower than projected | medium | medium | Profile in Week 2 forward benchmarks. Adjust K, layers, d_model accordingly |

---

## 9. Reference comparison to Metamon

Per `metamon_ref/metamon/rl/metamon_to_amago.py:534-589`, Metamon's
production tokenizer is `MetamonPerceiverTstepEncoder`. Their config for
Minikazam (4.7M params): 5 latent tokens × 64 d_model = 320-dim per-turn
output. For Small (15M params): 4 scratch × 100 = 400-dim. For Large
(140M params): 11 scratch × 160 = 1760-dim.

Their tokenization differs from ours in **two structural ways**:

1. **Text + numerical multimodal input.** Metamon's per-turn input is
   ~87 text tokens (words like "charizard", "leftovers", "blaze",
   "fire", "flying") + 3-6 numerical tokens (HP%, level, stats). The
   structure is *implicit in word ordering* — the model learns that
   the 5th word is the species, the 6th is the item, etc. They use a
   single 2541-token vocabulary covering gens 1-9.

2. **Perceiver cross-attention to compress.** The 90-93 input tokens
   cross-attend to K learnable latent tokens (5 for Minikazam, 11 for
   Large), then those latents self-attend. Output = K × d_model
   flattened.

**Our design differs**:

- **Explicit per-attribute tokens** (~212 of them) instead of implicit
  word-positional structure. We have more inductive bias (the model
  starts knowing "this is a stat token" via type_id_embed) at the cost
  of less flexibility for multi-gen extension.
- **Direct self-attention** instead of Perceiver cross-attention. We
  have ~212 input tokens, not 90 — but with FlashAttention, this is
  still tractable on consumer GPUs at d_model=256.

**What we adopt:**
- **Multi-summary scratch tokens** (K=2, like Minikazam's 5 latents).
  Replaces our current single attention-pooled summary
  (model.py:420-426), which METAMON_LEARNINGS.md flagged as a silent
  bottleneck.
- **Higher temporal-to-spatial ratio.** Metamon: 5-8:1 in d_model
  (small=5×100 spatial → 512 temporal hidden). Ours rewrite: 2:1 in
  per-turn output (2×256=512 spatial output → 256 temporal d_model),
  with 4 temporal layers vs 2. Closer to Metamon's recipe than current.
- **Popart on value head.** Metamon Small/Large/Kakuna all use it.
  We've never enabled it. Including in V1.
- **Dropout 0.05.** Already matches current.

**What we adapt (different from Metamon):**
- **Token count higher.** 212 vs Metamon's ~93. We trade compute for
  explicit compositional structure, betting that attention over
  attribute-level tokens generalizes better to novel sets than text
  ordering does.
- **Move tokens NOT split into sub-tokens.** Metamon doesn't have
  separate move-bp / move-acc tokens; they encode move text + bundled
  numerical features together. We similarly bundle (rich move token).
  But we do it explicitly via `MoveTokenizer` MLP, not by text-position
  in the input sequence. The explicit-vs-implicit distinction is the
  main divergence.
- **Spatial / temporal split kept.** Metamon Small/Large use a single
  trajectory transformer over (turns × latents) flattened. We keep
  spatial and temporal separate (§4.4) because at our token count
  unifying would explode compute.

**What we differ on (deliberately):**
- **Single-format singles gen 9 only in V1.** Metamon's tokenizer is
  multi-gen (gens 1-9). Ours is gen-9-only because we don't need
  multi-gen support yet (per §1 N6) and the gen-specific tokenization
  is more compact.
- **Our PFSP pool, our defense layers.** Metamon's training infra is
  AMAGO-based (different framework, different eval methodology).
  We keep ours — they work, and migrating to theirs is a separate
  project (not a rewrite of the model).

**Param target:**
- Minikazam at 4.7M: too small for our purposes; their RNN temporal is
  efficient but ours is transformer.
- Small at 15M: comparable to current (14M). Reasonable target for V1.
- Large at 140M: requires cloud, requires more BC data than we have.
  Future work.

Our V1 target: ~25M params (rough estimate, see §4.5; refine after
Week 1 forward-pass validation). Slightly larger than Small to
compensate for our lack of their 18M-trajectory dataset (we have ~5M).

---

## 10. Open questions / things deliberately left unresolved

- **Optimal K (summary scratch tokens).** Going from current K=0 to K=2
  is a clear improvement; K=4 (Metamon Small) might be better. Decide
  in Week 2 forward benchmarks based on what fits in compute budget.
- **Type-token decomposition.** §3.1 has 1 type token per Pokemon
  (compressing both types via small MLP from 2 type embeddings). Could
  alternatively be 2 separate tokens (type1, type2). Trade-off:
  consolidation saves 12 tokens (=12×12 attention pairs); separation
  enables attention to compose "Fire/Flying weakness to Stealth Rock"
  more directly. Decide in Week 1 unit tests.
- **Move-flag projection table size.** The `(n_moves, 107)` lookup is
  ~400 KB in fp32. If we want per-format generalization later, we'll
  need to either (a) regenerate per-format, (b) store sparse per-format
  diffs. V1: gen-9-only, ignore. Multi-gen extension is post-V1.
- **Whether to publish a `transformer_v1` BC dataset.** If memmap stays
  unchanged (Option A in §6.1), we don't need to. If we go to Option
  B (explicit per-token memmap regen), the new dataset is ~115 GB and
  worth tagging/versioning.
- **Whether to rerun full ladder eval on new arch.** PokeAgent ladder
  submission is mentioned in NEXT_SESSION.md as a post-Week 6 option.
  Cost: account management + days of ladder games + monitoring.
  Diagnostic value: validated public Elo. Decide after Week 5 results.

---

## 11. Document changelog

- **2026-05-01 (Session 45):** initial design. Covers V1 of the rewrite,
  spans Sessions 46-51 of implementation effort.
- **2026-05-01 (Session 46):** Week 1 complete. See §7 Week 1 results
  block for measurements. Postscript A below.
- **2026-05-01 (Session 46, late):** Self-audit pass — Postscript B
  below covers tier-A bugs fixed (type ordering, weight init), tier-B
  format-extensibility hooks (FormatConfig threading, vocab-driven defaults,
  lookup schema versioning), tier-C hardcoding cleanup, tier-D robustness
  lessons brought in from the legacy painful-fixes record (atomic write,
  gradient_checkpoint plumbed, logging instead of print, lookup version
  check), and tier-G recovery of the active-move real-banks signal that
  the V1 design had collapsed into the lookup.
- **2026-05-01 (Session 46, signal-recovery):** Second audit asked
  "is everything derived from source-of-truth?" Answer was no — the
  §3.1 token tables omitted several memmap-recorded signals
  (active/fainted/combat/toxic/future_sight/visibility/level/weight/
  height) and the opp threat token was a zero-init parameter.
  Postscript C documents the restoration with empirical references
  (Session 30 weight analysis: defensive_eff and type_eff are the
  #1 ranked features by 5-pt margins).
- **2026-05-01 (Session 46, active-move flag fix):** Third audit asked
  "the PP thing shows wrong info — what about acc?" Re-reading
  `_project_move_flags` confirmed: accuracy is static (no fix needed),
  but `current_pp` (dim 9), `disabled` (dim 10), and `stab` (dim 12)
  are dynamic. For OUR active 4 moves these are silently overridden
  by the lookup's static defaults, regressing vs legacy. Postscript D
  adds an `active_real_flags` override path mirroring Tier G's banks
  override, restoring real per-turn PP/disabled/STAB for the moves
  the model is choosing among.
- **2026-05-02 (Session 46, structural enrichment pass):** Fourth
  audit asked "what about hazards/weather/screens/move effects/
  items/abilities — anything else we're missing?" With user's "no
  hand-curation, derive from true state" philosophy:
  - Postscript E: split `field_token` into 9 thematic tokens
    (weather/terrain/our_haz/opp_haz/our_scr/opp_scr/speed_field/
    mechanics/progression). N_TOKENS 212 → 220.
  - Postscript F: enrich move flag lookup 107 → 119 dim with 12
    structurally-derivable flags poke-env exposes that
    `_project_move_flags` drops (slicing/bullet/bypasssub/pulse/
    charge/futuremove/ignore_defensive/use_target_offensive/
    thaws_target/reflectable/gravity/sleep_usable). Schema v3.
  - Postscript G: parse Showdown items.ts / abilities.ts at
    lookup-build time. Add `(n_items, 8)` and `(n_abilities, 7)`
    structural feature tables. New `ItemTokenizer` /
    `AbilityTokenizer` (id_embed + features → MLP → d_model),
    parallel to MoveTokenizer. Schema v4.
- **2026-05-02 (Session 47):** Week 2 complete. SpatialTransformer (6L,
  8H, d=256, Poke-Mask), TemporalTransformer (4L, 8H, d_temporal=256,
  causal), ActionHead, ValueHead, SwitchActionEncoder, TransformerBattlePolicy
  wrapper, forward_sequence mega-batch path, test_policy.py with 9
  tests including a synthetic PPO step. See Postscript H below for
  divergences (param count 9.5M vs 20-30M estimate, side-mask skipped,
  per-action context permutation by species/move-id matching, real-PPO
  smoke deferred to Week 3 in favor of synthetic-PPO).
- **2026-05-02 (Session 48):** Week 3 plumbing landed.
  `train_bc.py:--use-transformer` factory + arch-tagged checkpoints +
  arch dispatch in `ppo.py::load_checkpoint`. Plumbing smoke (CPU B=4,
  40 batches): loss 1.66 → 1.61, no NaN, 19,994,924 params confirmed.
  CUDA fp16 throughput bench (`bench_bc_step.py`) found a memory cliff
  at B=8 on the 6 GB RTX 3060: per-turn jumps from 5 ms → 145 ms once
  peak memory crosses ~5 GB (batch 3 of 5 spike). B=4 is stable —
  per-turn 6-11 ms fp16, peak ≤2.74 GB across 8 batches, loss decreasing
  cleanly. Postscript I documents the cliff + chosen operating point.
  Throughput: ~9.7 hr/epoch local fp16 → 5 epochs ≈ 2 days; 10 epochs
  ≈ 4 days. Right at the cloud-trigger boundary.

Future updates as implementation reveals divergences. Add a postscript
section per change; do not silently rewrite history.

---

## Postscript A — Session 46 Week 1 implementation notes

These items deviated mildly from §3 / §6.1 during implementation. None
require a redesign; they're carried forward into Week 2 work.

1. **All 4 move tokens (active + team) use the same `(n_moves, 107)`
   flag lookup AND the same `(n_moves, 4)` bank lookup.** This is
   explicit in §3.2 ("same real bank values fed in everywhere") but
   deserves restating. Concretely, the active 4 moves *no longer
   surface* current_pp, disabled, or STAB to the model via the move
   token — those are now uniformly the lookup's no-battle-state values
   (max pp, not disabled, no STAB). The model can still recover STAB
   via attention between move_token and the active Pokemon's
   type_token; current_pp / disabled signals are lost in V1. If Week
   3 BC convergence stalls and ablation suggests current_pp matters,
   add a per-active "pp_remaining_token" of 4 dims fed from
   `active_move_banks` (cheap, +4 tokens).

2. **Opp active-threat token is a learnable "unknown" parameter,
   not a symmetric MLP from opp's active-move flags.** Reason: the
   memmap doesn't carry an opp-active-move feature table. Our side
   uses the 2 trailing dims of `active_move_cont` (type_eff +
   opp_threat, the 109-dim active-only encoding). For symmetric
   training-data parity per §3.4, V1 just provides a learnable token
   for the opp slot — the model can attend to opp's exposed moves and
   types directly, so no information is lost; only the explicit
   pre-computed threat scalars are unavailable. Acceptable for V1.

3. **Type token uses ONE token, not 2 (open question §10 resolved).**
   Picked single-token aggregation via 2 type embeds → MLP → d_model.
   Saves 12 attention pairs per turn vs splitting; no measurable
   downside in unit tests. Re-examine if Week 3 BC underfits on
   dual-typing-dependent matchups.

4. **Position-ID layout is stored as buffers, not regenerated per
   forward.** Tokenizer.\_build\_position\_ids runs once at module
   init; all 212 positions get static type_id, pokemon_slot_id,
   move_slot_id tensors. The forward just adds three embedding lookups.
   Trivial, but explicit so Week 2's spatial / mask code doesn't try
   to recompute them.

5. **Lookup verification is stricter than spec.** §7 said "matches to
   within fp32 noise"; the actual verification at
   `verify_move_lookup.py` asserts bit-exact match (max |Δ| ≤ 1e-6) on
   the 105 battle-state-INDEPENDENT dims, plus only-STAB-differs on
   the active-path call. Documented: cont[12]=stab, cont[9]=current_pp,
   cont[10]=disabled are the three battle-state-dependent components
   not in the lookup.

6. **Imported `NumericalBank` from `model.py` rather than duplicating.**
   ~~The §6b guard rail says don't *modify* `model.py`; importing a small
   utility is fine~~. **Reverted in Postscript B**: now duplicated inline
   in `model_transformer.py` so we have zero outbound deps on the legacy
   module.

---

## Postscript B — Session 46 self-audit cleanup

Triggered by an explicit "what shortcuts / hardcoding / quirks?" review.
Items below are improvements *to Week 1 code as committed in be36415*; the
spec contract is unchanged (still ~212 tokens, still d_model=256, still
the lookup table approach).

### B1. Bug fixes (correctness)

**Type-token ordering canonicalized.** `_encode_pokemon_block` previously
did `topk(2)` on the type multi-hot to recover the two types. For 2-type
Pokemon, `topk` tie-breaking was implementation-dependent — Water/Ice
could come back as `(2, 5)` or `(5, 2)`, and the type MLP saw two
different inputs for the same Pokemon. Now: `topi.sort(dim=-1)` after
the absent-type fill, deterministic by index. Test 6 in
`test_tokenizer.py` asserts identical type tokens across slots with
identical types.

**Weight init matches legacy.** `init_module_(self, std=cfg.init_std=0.02)`
applied to MoveTokenizer + Tokenizer at construction; mirrors
`PokeTransformer._init_weights` (model.py:655-666). Sanity check (test 7):
9 Linear + 19 Embedding weights all have std within `[0.4×, 1.6×]× 0.02`.
Pre-fix max-|delta| from zeroing type_id was 3.85; post-fix is 0.085 —
activation magnitudes are now well-controlled at init.

### B2. Format extensibility (multi-format goal)

> 📋 **BEFORE doing the doubles/VGC/triples rewrite** that lifts
> `n_active != 1` — read `docs/ARCHITECTURE_REVIEW_2026_06_07.md`
> + memo `memory/project_architecture_review_2026_06_07_todo.md`. That
> review flagged the move-vs-switch action representation asymmetry
> (Priority 1) as worth fixing during action-space rewrites for low
> marginal cost. Doubles/VGC will reshape the action head dramatically
> (4+5=9 actions → up to 4×2+5=13 actions for doubles with target selection);
> bundling the switch-rep ablation here is the natural opportunity.
> ⚠️ **GRAIN OF SALT** — re-verify against current code first; review is a
> 2026-06-07 snapshot (commit `f421b7c7`).

Even though V1 is gen-9 singles only, the project goal is "all formats,
all gens 4+" (NEXT_SESSION.md). Cheap hooks added now to avoid retrofit
later:

- `TransformerConfig.format_config: FormatConfig` field with default
  `FORMAT_SINGLES`. All shape decisions (Pokemon count, Pokemon-slot
  vocab, type count, move count per Pokemon, threat-MLP fan-in) derive
  from `cfg.format_config`. `total_tokens(fmt)` and
  `n_pokemon_slot_vocab(fmt)` are pure functions of the format.
- **`Tokenizer.__init__` raises `NotImplementedError` for `n_active != 1`.**
  We're not pretending doubles works; we're failing loudly with a pointer
  to §1 N6. Test 10 asserts this.
- `TransformerConfig.with_vocab_sizes_from_disk()` constructor pulls
  `n_species/n_moves/n_items/n_abilities` from `Vocab.load()` so a vocab
  regen flows through automatically. Old defaults are still on the field
  (so checkpoint loading via `from_dict` continues to work).
- `Vocab.id_to_name(kind, id)` and `id_to_name_map(kind)` added as public
  helpers — no more `v._move` private-attribute access in lookup builder.

### B3. Hardcoding cleanup

- Slice offsets `_SL_*` are now derived from `N_TYPES`, `N_STATUS`,
  `N_VOLATILE`, `N_PARADOX`, and the explicit boosts/tera widths.
  Asserts at module load that the layout sums to `POKEMON_CONT_DIM`
  exactly — a features.py change that bumps any slice will fail at
  import, not silently shift offsets.
- `boosts_mlp` in_dim derived from `_SL_BOOSTS[1] - _SL_BOOSTS[0]`,
  `status_mlp` in_dim from the status/volatile/paradox/tera slice
  widths. The brittle `assert in_dim == 187 / 72` literals are gone.
- `threat_mlp` in_dim is `fmt.n_moves * 2`. The assertion that
  `MOVE_SLOT_CONT_DIM == MOVE_FLAG_DIM + 2` documents the contract
  with features.py.
- Battle-state Pokemon-slot IDs (`PS_SLOT_DECISION`, `PS_SLOT_FIELD`,
  `PS_SLOT_TRANSITION`, `PS_SLOT_SUMMARY`) are named, indexed off the
  per-format `2 * team_size` base.
- `Tokenizer.count_parameters()` added (matches legacy convenience).
- `_mlp_2_layer` / `_mlp_1_layer` factor the MLP-with-norm pattern that
  was repeated 5+ times.

### B4. Robustness lessons from legacy painful-fixes

The legacy MLP arch accumulated stability fixes over many iterations
(atomic-checkpoint write Session 35, AMP attention Session 35, etc.).
Most live in the *training pipeline* and are preserved by §N4. The ones
that apply to the Tokenizer:

- **Atomic lookup save:** `save_move_flag_lookup` now writes to `.tmp`
  then `.replace()`. Mirrors the Session 35 atomic-checkpoint fix.
- **`gradient_checkpoint: bool = False`** field plumbed in
  `TransformerConfig` so Week 2 spatial/temporal stack can opt in
  without a config schema change.
- **`logging.getLogger("pokemon_ai")`** replaces `print` in the lookup
  builder, matching the convention used by `dataset.py`,
  `train_rl.py`, etc. Verbose CLI usage gets a default StreamHandler;
  library callers control verbosity through normal logging config.
- **Lookup schema versioning:** the `.pt` file now embeds a `meta` dict
  with `schema_version`, `gen`, `vocab_n_moves`, `move_flag_dim`,
  `bank_fields`. `load_move_flag_lookup` rejects mismatches loudly
  (raises with rebuild instructions). `expected_n_moves` parameter
  catches vocab/lookup desync — if `Vocab.load().n_moves` != lookup's
  `n_moves`, training won't silently mis-tokenize moves with new IDs.

### B5. Brittleness fixes

- `_features_project_move_flags` shim isolates the one private import
  from features.py to a single line. If features.py renames the
  function, only this shim breaks (loud import failure at lookup-build
  time, not a silent training-time desync).
- `NumericalBank` inlined into `model_transformer.py`. Zero outbound
  deps on legacy `model.py`. Reverses Postscript A item 6.

### B6. Tier G — active-move real banks (recovers signal that V1 design lost)

**Spec deviation, not bug fix.** REWRITE_DESIGN.md §3.2 said
"`MoveTokenizer` is called for *all* 4 moves on *all* 12 Pokemon ...
with the same real bank values fed in everywhere." Strict reading:
active moves use the lookup, like team moves. That collapses
`current_pp`, `disabled`, and STAB into the lookup's no-battle-state
defaults — losing real signal the legacy MLP arch captured via
`active_move_banks` (model.py:780).

**B6 walks that back, partially.** The Tokenizer's `forward` now
optionally consumes `batch["active_move_banks"]` (already provided by
`dataset.unpack_turn_batch` line 336). When present, slot-0's 4 move
banks on our side are overridden with those real per-turn values
(current_pp / acc / etc.). The 107-dim flag vector still uses the
lookup uniformly — battle-state-dependent flag dims (#9 current_pp,
#10 disabled, #12 stab) remain frozen. So:

- Active moves on our side: REAL banks, lookup flags
- Team moves on our side: lookup banks, lookup flags
- All opp moves: lookup banks, lookup flags

Test 9 in `test_tokenizer.py` asserts the override changes only the 4
active-move tokens (slot 0 of our side) and leaves the other 208
tokens bit-identical.

This is asymmetric (only OUR active gets real banks) but correct: the
memmap reflects what we know. The model's value head can now reason
about "I'm at low PP on my best move; I should pivot before running
dry" — the legacy MLP arch had that signal; the strict V1 design
silently dropped it. We don't pay any compute cost; the lookup path
stays as the default, and the override is a single tensor index op.

### B7. What's still deferred (no change)

- Doubles / triples token layout — needs design (§1 N6)
- Multi-gen lookup structure — single-gen is V1
- `current_pp` / `disabled` / `stab` in the *flag* vector for active
  moves — deferred; 107-dim still lookup-only. If Week 3 BC underfits
  on PP-aware play, revisit by overriding flags[active] from
  `active_move_cont[..., :MOVE_FLAG_DIM]`.
- Sub-token move decomposition (§3.5 of design) — defer
- Test-time adaptation (§N2) — defer

### B8. Verification status post-cleanup (snapshot before Postscript C)

- Lookup rebuild: 952/952 valid moves in 953-vocab. Atomic write OK.
- Verifier: 50/50 strict match + 50/50 active-path with only-STAB-
  divergence — same as before B1-B7.
- 10/10 test_tokenizer.py tests pass on real `human_v8_100k` data.
  New tests (6, 7, 8, 9, 10) cover the B1, B1-init, B6, and B2
  rejection paths.
- Benchmark RTX 3060 Laptop, B=32 turns: tokenizer 3.61 ms,
  tokenizer+dummy spatial 27.60 ms (was 28.36 ms pre-cleanup, within
  noise — no perf regression from added abstractions).
- Param count unchanged at 1,420,680 — refactor introduced no new
  parameters.

---

## Postscript C — Session 46 signal-recovery pass

Triggered by: "is everything derived from the recorded obs vector and
poke-env, or are we making assumptions, simplifications, omissions?"
Audit found significant signal omissions vs both the legacy MLP arch
and the memmap. The original §3.1 token tables specified 17
per-Pokemon attribute tokens and a `status_token` from "status +
volatiles + paradox + tera flags." Several memmap-recorded fields
weren't allocated to any token. This section restores them.

### C1. Empirical evidence from the docs

The agent dive into STATUS.md / V8_PLAN.md / Session 30 notes
returned hard numbers, not speculation:

- **`defensive_eff` (switch_cont) ranks #1 in switch-decision weight
  analysis at 23.7 — 1.7× more important than HP%.** Adding it
  produced **+5 win-rate points** in Session 30 (47% → 52%).
- **Type effectiveness in active_move_cont ranks #1 overall at weight
  34.2 — 5.2× more important than Base Power.** The team explicitly
  tried "let the model learn it from entity embeddings"; it didn't
  work. The hard fix was to compute it explicitly.
- **Item embedding is 25× oversaturated** (800 items, H_ITEM=64 dims):
  Choice Scarf and Leftovers hash to nearly identical embeddings.
  Visibility flags (`item_known`, `ability_known`) are the partial
  mitigation that distinguishes "no item" from "we don't know yet."
- combat / toxic / future_sight were added in the Session 18 audit as
  "decision-relevant by next action" — non-trivial battle state with
  no other proxy (e.g., a Pokemon charging Solar Beam looks identical
  to a Pokemon in a no-op turn unless `preparing` is exposed).

V8_PLAN.md early on argued "ps-ppo learns this; remove pre-computed
features." Session 30's empirical results overruled that. We defer
to the empirical outcome.

### C2. What was restored

**Per-Pokemon `status_token` widened.** Same token name (API stable),
new inputs:

| Field | Slice | Prev | Now |
|---|---|---|---|
| status one-hot | cont[19:26] | ✓ | ✓ |
| volatile flags | cont[119:157] | ✓ | ✓ |
| paradox encoding | cont[157:164] | ✓ | ✓ |
| tera flags | cont[164:184] | ✓ | ✓ |
| **active flag** | cont[117:118] | dropped | ✓ |
| **fainted flag** | cont[118:119] | dropped | ✓ |
| **combat state (5 dims)** | cont[184:189] | dropped | ✓ |
| **toxic counter** | cont[189:190] | dropped | ✓ |
| **future_sight pending** | cont[190:191] | dropped | ✓ |
| **visibility (2 dims)** | cont[191:193] | dropped | ✓ |
| **level bank embed** | banks[1] via `NumericalBank(100, 8)` | dropped | ✓ |
| **weight bank embed** | banks[2] via `NumericalBank(201, 8)` | dropped | ✓ |
| **height bank embed** | banks[3] via `NumericalBank(41, 8)` | dropped | ✓ |

`status_mlp` first Linear: `83 → 256` cont dims + `24` bank-embed dims
= `107 → 256`. ~9K added params here.

**Battle-state opp threat is computed, not a zero parameter.**
Previously `opp_threat_unknown` was a single learnable d_model vector
(zero-init), which gave the opp threat token no input signal — the
model had to learn what "opp threat" means from absolutely nothing.
Now:

- The lookup `.pt` file embeds an `(N_TYPES+1, N_TYPES+1)` damage
  chart from `poke_env.data.GenData(gen).type_chart`. Schema bumped
  to v2; loader rejects v1 / missing-chart files.
- `Tokenizer._encode_opp_threat` extracts opp-active's 4 move types
  from the lookup (type one-hot lives at flag dims 81-99), looks up
  effectiveness against our active's two types (sorted canonical), and
  feeds the 4-dim multiplier vector through `opp_threat_mlp`. Same
  normalization (mult/4.0, floor 0.01) as `features.py:_compute_type_effectiveness`.
- Mirrors `our_threat_mlp` in spirit, but with 4-dim input (one
  multiplier per opp move) vs 8-dim input (4 our moves × {type_eff,
  opp_threat_back}). Asymmetric because the memmap doesn't carry
  opp-active-move continuous; the chart-driven version is the best we
  can do without expanding the memmap.

### C3. What was deferred (with rationale)

**Switch defensive_eff / offensive_eff (switch_cont last 2 dims) for
the 5 bench Pokemon — DEFERRED to Week 2 action head.** Reasons:

1. They're per-action-target signals, not per-Pokemon state. The
   legacy `ActionSlotEncoder` (model.py:522-571) consumed them
   directly when building the per-action context for switch slots
   4-8. That's where they belong architecturally.
2. Attaching them to the per-Pokemon `status_token` requires
   permutation logic to match memmap orderings: `switch_cont` is in
   poke-env's `available_switches` order; `our_pokemon_cont`'s bench
   is alphabetical-by-species. They don't align without a join on
   species_id, costing ~40 lines of permutation code per forward.
3. The signal is **preserved**, just routed correctly. Week 2's
   action head will pass `switch_cont[..., -2:]` directly into the
   per-action context for the 5 switch slots.

If Week 2 finds the action-head path insufficient (e.g., the value
head also needs the signal in the spatial transformer, not just the
policy head), revisit by adding a "switch_eval_token" per bench
Pokemon with the 2 eff dims as input.

### C4. What's NOT restored (and why)

- **Active move 107-dim flag vector** for active moves. We use the
  lookup uniformly. Lost: `current_pp` (recovered via Tier-G banks
  override), `disabled` (rare), `stab` (recoverable via attention to
  type_token). Acceptable.
- **Per-move `_compute_type_effectiveness` for non-active moves
  (team moves' active-only dims).** Memmap doesn't carry these; the
  legacy MLP arch zeroed them too. No regression.
- **Switch_cont's 19+1+7+1 dims that duplicate `our_pokemon_cont`.**
  Genuinely redundant — same Pokemon's types/hp/status/weight already
  in the per-Pokemon block.

### C5. Real shortcuts I'm taking (flagged for the user)

- **`status_token` is now an everything-bag** (~107-dim input).
  Functionally correct, but in attention probing later "what does the
  status token represent?" will be muddled. Alternative: split into
  3-4 narrower tokens (status, condition_flags, physical_meta,
  visibility) at the cost of +30 tokens per turn. I picked one wide
  token for V1; flag this for revisit if attention diagnostics get
  confusing.
- **`opp_threat_mlp` input is 4 dims (one mult per opp move).**
  `our_threat_mlp` input is 8 dims (4 moves × 2). Different shapes,
  different MLPs. Not a problem in the model, but worth noting that
  opp threat is structurally lower-rank than our threat — the model
  can attend to other tokens for the missing direction.
- **`opp_threat` mask for unknown moves uses 0.25 ("neutral") as the
  default.** This matches `features.py:_compute_type_effectiveness`'s
  no-info fallback. Real value: the model can't distinguish
  "unrevealed move would be 1.0× effective" from "unrevealed move
  could be 4× effective." Acceptable for V1; if Week 4 PPO struggles
  on early-turn opp-threat assessment, add a per-opp-move "is this
  revealed?" bit.

### C6. What the model now sees (vs legacy MLP arch)

Going through every memmap field:

| Field | Legacy uses it? | New tokenizer uses it? |
|---|---|---|
| `our_pokemon_ids[species/item/ability/4×move_id]` | ✓ | ✓ (3 entity tokens + 4 move tokens) |
| `our_pokemon_banks[hp_pct]` | ✓ | ✓ (`hp_pct_token`) |
| `our_pokemon_banks[level/weight/height]` | ✓ | ✓ (status_token bank embeds) |
| `our_pokemon_banks[6 stats]` | ✓ | ✓ (6 stat tokens) |
| `our_pokemon_cont[types]` | ✓ | ✓ (`type_token`) |
| `our_pokemon_cont[status/volatile/paradox/tera]` | ✓ | ✓ (`status_token`) |
| `our_pokemon_cont[boosts]` | ✓ | ✓ (`boosts_token`) |
| `our_pokemon_cont[active flag]` | ✓ | ✓ (status_token, restored) |
| `our_pokemon_cont[fainted flag]` | ✓ | ✓ (status_token, restored) |
| `our_pokemon_cont[combat state]` | ✓ | ✓ (status_token, restored) |
| `our_pokemon_cont[toxic counter]` | ✓ | ✓ (status_token, restored) |
| `our_pokemon_cont[future_sight]` | ✓ | ✓ (status_token, restored) |
| `our_pokemon_cont[visibility]` | ✓ | ✓ (status_token, restored) |
| `our_pokemon_cont[4× move_compact]` | ✓ | replaced by lookup (richer) |
| `our_pokemon_mcont[4×23]` | ✓ | replaced by lookup (richer) |
| `field_banks[4]` | ✓ | ✓ (`field_token`) |
| `field_cont[52]` | ✓ | ✓ (`field_token`) |
| `trans_ids[2]` | ✓ | ✓ (`transition_token`) |
| `trans_cont[51]` | ✓ | ✓ (`transition_token`) |
| `active_move_ids[4]` | ✓ | ✓ (move tokens at our active slot 0) |
| `active_move_banks[4×4]` | ✓ | ✓ (Tier G override on slot 0) |
| `active_move_cont[4×109]` | ✓ | partial — last 2 dims for `our_threat_token`; first 107 use lookup |
| `switch_ids[5]` / `switch_cont[5×30]` | ✓ | **DEFERRED to Week 2 action head**: 19+1+7+1 redundant; defensive/offensive_eff (last 2) are per-action-target signal that goes in the action context, not the spatial token sequence |
| `legal_mask[9]` | ✓ (action head) | ✓ (action head, Week 2) |

Symmetric audit for opp side:
- All `opp_pokemon_*` fields used the same way.
- Opp active threat now computed (Postscript C2) instead of being a
  zero-init parameter.

Everything in the memmap is now reaching the model except the 5×switch
deferred items, all of which are routed into the action head in
Week 2 — not lost, just architecturally placed where the legacy code
also placed them.

### C7. Verification status post-Postscript-C

- Lookup .pt v2: 952/952 valid moves, damage chart shape (20, 20),
  spot-checks Fire→Water=0.5, Ghost→Normal=0.0.
- `verify_move_lookup.py`: 50/50 strict match + 50/50 active-path
  STAB-only divergence. Schema v2 validation in load.
- `test_tokenizer.py`: 13/13 tests pass on real `human_v8_100k`. New
  tests: `test_opp_threat_uses_chart` (perturbing chart changes
  opp_threat token only), `test_restored_signals_reach_status_token`
  (each restored cont-slice perturbs status_token without leaking),
  `test_physical_banks_reach_status_token` (level/weight/height bank
  embeds reach status_token).
- Bench: tokenizer 4.29 ms median (was 3.61), tokenizer + dummy
  spatial 29.06 ms median (was 27.60). +1.5 ms from wider status MLP
  + opp threat computation; well under 50 ms budget.
- Param count: 1,433,912 (was 1,420,680). +13,232 params from level/
  weight/height banks, wider status_mlp, opp_threat_mlp.

### C8. How a future contributor verifies "no signal lost"

Run `python test_tokenizer.py`. Tests 11 + 12 perturb each restored
slice and assert it reaches `status_token`. Test 8b perturbs the
damage chart and asserts only `opp_threat_token` changes. If a future
edit accidentally drops a slice from `status_cont` cat, the
corresponding test 11/12 sub-test fails immediately with the slice
name in the error message.

---

## Postscript D — Session 46 active-move flag override

Triggered by: "isn't the PP thing also a big thing — it shows wrong
info? what about acc and stuff?" + "read the A/B/B+/C/D thread."

### D1. The exact problem

The 107-dim move-flag vector from `_project_move_flags` (features.py:
1005-1263) has three battle-state-dependent dims, classified by
re-reading the function:

| Dim | Field | Source | Static? |
|---|---|---|---|
| 0 | bp01 | `m.base_power / 250` | Static ✓ |
| 1 | acc01 | `m.accuracy` (1.0 if always-hit) | **Static** ✓ — accuracy boosts/drops live on the *Pokemon's* boost stat, not the move |
| 2 | prio_n | `m.priority / 3` | Static ✓ |
| 3-8 | drain/recoil/heal/multihit/flinch/crit_ratio | move properties | Static ✓ |
| **9** | **`current_pp/64`** | `m.current_pp` | **DYNAMIC** — varies per turn |
| **10** | **`disabled`** | `m.disabled` | **DYNAMIC** — Disable / Encore / Choice-lock |
| 11 | recharge | `m.recharge` | Static ✓ |
| **12** | **`stab`** | depends on `poke_types` arg | **DYNAMIC** — flips on STAB-able moves vs Pokemon's types |
| 13-106 | all the boolean / one-hot flags | move properties | Static ✓ |

So `accuracy` does NOT change at runtime — that one is fine in the
lookup. The three actually-dynamic dims are 9 / 10 / 12.

Previously (Postscript A item 1, then re-confirmed in Postscript B):
"all 4 move tokens (active + team) use the lookup uniformly." For
team and opp moves this is **strictly better** than the legacy MLP
arch (which zeroed those dims entirely). For OUR ACTIVE 4 MOVES this
is a **silent regression**: the legacy arch had real per-turn PP /
disabled / STAB via the `active_move_cont` memmap path; we threw that
away.

### D2. The A/B/B+ thread context

The whole point of the A/B/B+/D-and-rewrite design (NEXT_SESSION.md
L689-870) was: *the model should see real per-move features for every
move, not zeros.* Specifically B+ argued for "extend team-path
encoding from 23 → 109 dims." We did better than B+ for team moves
(lookup carries the static 104 of 107 dims, beating zeros). We did
worse than legacy for active moves (lookup overrides real per-turn
values with static defaults on 3 dims). Net result before this fix:
team improved, active regressed. This fix addresses the active
regression.

### D3. The fix

`_encode_pokemon_block` accepts a new optional `active_real_flags:
(B, n_moves, MOVE_FLAG_DIM)` argument. When provided, slot-0's 107-dim
flags are overridden with the real per-turn vector. The Tokenizer's
`forward()` slices `active_move_cont[..., :MOVE_FLAG_DIM]` (the
memmap stores the same 109-dim that features.py's active path
produces, with first 107 dims being the same shape as the lookup
but with battle-state-correct values; the trailing 2 dims are the
threat scalars we already use).

For our 4 active moves: real PP, real disabled, real STAB. For team
moves and opp moves: lookup unchanged (no battle-state info available
in the memmap for those positions anyway).

### D4. On items being learnable (the user's pushback)

The Session 30 docs noted "Choice Scarf = Leftovers to the model" but
the embedding was 64-dim against ~800 items (~12.5× oversaturation).
Our config has `d_model=256` for entity tokens (3.1× oversaturation),
4× more capacity per item. Combined with 5M training samples and
transformer attention composing item_token with stat / type / ability
tokens, items are within learnable range. This is consistent with the
legacy MLP arch reaching 67.8% smart_avg with 32-dim item embeds —
i.e., even at 73× oversaturation the model captured *something*.
Pre-baking item-effect feature dicts (Choice/Orb/immunity flags etc.)
is **deferred** to a possible Week 3 follow-up if BC ablation shows
weak item learning. Not a Week 1 task; YAGNI for now.

### D5. Verification

- Test 9b in `test_tokenizer.py`: `test_active_flag_override_recovers_dynamic_dims`.
  - Loads 2 sample episodes at t=0; reads `active_move_cont[..., :107]`
    vs `move_flags_lookup[move_ids]` for our 4 active moves.
  - At t=0, dim 9 (current_pp) and dim 10 (disabled) happen to agree
    (full PP, nothing disabled). Dim 12 (STAB) cumulative diff = 4.0
    (STAB flipped on at least 4 active moves across 2 episodes —
    exactly what we'd expect for STAB-able move/Pokemon pairings).
  - Move-token output diff = 5.28 across the 4 active moves with vs
    without the override path enabled.
- Test 9 (`test_active_real_banks_override`) still passes — the banks
  override is independent of the flag override.
- Bench: 27.13 ms median for B=32 turns (was 29.06 pre-fix; within
  noise, no perf regression).

### D6. What this still doesn't recover

- **Team-move PP**: no per-turn PP for our team's 5 bench Pokemon. The
  memmap doesn't carry it (only active_move_cont covers our 4 active).
  Acceptable: knowing "Wish PP at 2/16" on a Pokemon you're not using
  this turn is third-order info; legacy didn't have it either.
- **Opp-move PP / disabled / STAB**: same — memmap doesn't carry an
  opp-active-move-cont table. Lookup-only for opp.
- **`current_pp` and `disabled` flag dims for opp moves** stay at the
  lookup defaults forever. If Week 4 PPO struggles on PP-aware play
  against opp (e.g., model fails to recognize opp's Choice-locked
  state), revisit by deriving disabled from transition events
  (turn-by-turn Choice lock detection in features.py).

---

## Postscript E — field-token split into 9 thematic tokens

**Triggered by:** "for elite play this matters — sun + Solar Power
composes; opp screens halve our damage; tailwind alters speed tier; with
everything mashed into one MLP token, attention can't specialize."

The legacy / V1-pre-refactor code had ONE `field_token` from a single MLP
over all 52 cont dims + 4 banks. That collapsed all field state into
one attention slot. Replaced with 9 thematic tokens:

| Token | Source dims | Why split |
|---|---|---|
| `weather`     | weather one-hot (5) + dur bank | Weather sets up Sand Force / Solar Power / Slush Rush / Swift Swim / Sand Rush — each composes with specific abilities. Attention path needs to be its own. |
| `terrain`     | terrain one-hot (5) + dur bank | Same: Surge Surfer + Electric Terrain, Quark Drive + Electric Terrain, Grassy Terrain healing, Misty Terrain status block |
| `our_hazards` | SR / spikes (1-3) / tspikes (1-2) / web (4 dims) | The legacy `defensive_eff` signal is fundamentally about "does this Pokemon survive entry to opp's hazards" — needs its own attention path |
| `opp_hazards` | symmetric (4) | hazards I just set up; matters for "should I push for SR + Spikes vs swap to attacking" |
| `our_screens` | reflect / ls / av × {presence, dur} = 6 | Screens I have up: opp damage halved. Composes with my walls' bulk |
| `opp_screens` | symmetric (6) | Screens opp has: my damage halved. Composes with my sweeper's offense |
| `speed_field` | tailwind us/opp + trick room + tr_dur (3 + bank) | Speed-tier modifiers — flips outspeed calculations |
| `mechanics`   | can_tera/mega/z/dmax + used_* + dmax_turns + trapped + force_switch + opp_revealed_frac (17) | One-time-use resources + battle-state flags |
| `progression` | alive_us/alive_opp + turn bank | Game-progression — endgame counting, win-condition timing |

**Bookkeeping:**
- `N_BATTLE_STATE: 6 → 14`
- `N_TOKENS: 212 → 220` (+8 tokens, +3.7% attention compute at the
  spatial transformer; well under any meaningful budget)
- `N_TOKEN_TYPES: 20 → 28` (one per thematic token + threat/decision/etc.)
- `N_BATTLE_STATE_SLOTS: 4 → 12`

**Test 13 (`test_field_split_isolation`):** perturbing each cont slice
shows bit-perfect isolation — every other thematic token's output is
0.00e+00 different. The architectural separation is enforced at the
input level.

**Bench:** 30.04 ms median for B=32 turns (was 27.13 pre-split). +3 ms
from the added MLPs and tokens. Well under 50 ms budget.

---

## Postscript F — move flag lookup enrichment (107 → 119 dim)

**Triggered by:** "the model has to memorize 'this is Body Press, uses
Defense' from move_id alone; that's exactly the brittleness the rewrite
was supposed to fix."

`features.py:_project_move_flags` extracts 107 dims, dropping ~12
structurally-derivable flags poke-env's `Move` object exposes directly.
Adding them as a *Postscript-F* extension to the lookup, NOT modifying
features.py (guard-railed). Layout: `[0:107]` from `_project_move_flags`
(bit-exact); `[107:119]` from `_extra_move_flags(move)`.

| Flag | Source | Used by ability/move |
|---|---|---|
| `slicing` | `m.flags["slicing"]` | Sharpness boost |
| `bullet` | `m.flags["bullet"]` | Bulletproof immunity |
| `bypasssub` | `m.flags["bypasssub"]` | Sound + others bypass Sub |
| `pulse` | `m.flags["pulse"]` | Mega Launcher boost |
| `charge` | `m.flags["charge"]` | 2-turn moves (Solar Beam, Fly, ...) |
| `futuremove` | `m.flags["futuremove"]` | Future Sight, Doom Desire |
| `ignore_defensive` | `m.ignore_defensive` | Body Press / Foul Play category |
| `use_target_offensive` | `m.use_target_offensive` | Foul Play |
| `thaws_target` | `m.thaws_target` | Scald-style anti-freeze |
| `reflectable` | `m.flags["reflectable"]` | Magic Bounce target |
| `gravity` | `m.flags["gravity"]` | gravity-suppressed |
| `sleep_usable` | `m.sleep_usable` | Sleep Talk, Snore |

`MOVE_FLAG_DIM_BASE = 107`, `MOVE_FLAG_DIM = 119`. Schema bumped to v3.

**Active-move flag override path:** still reads
`active_move_cont[..., :MOVE_FLAG_DIM_BASE]` from the memmap (per-turn
ground truth on dims 9/10/12). The 12 extras are move-static and stay
from the lookup. No memmap shape change required.

**Test 14 (`test_extra_move_flags_in_lookup`):** 11/11 canonical
move-flag pairs verified — psychocut→slicing, eggbomb→bullet,
boomburst→bypasssub, aurasphere→pulse, solarbeam→charge,
futuresight→futuremove, foulplay→use_target_offensive,
scald→thaws_target, toxic→reflectable, bounce→gravity,
sleeptalk→sleep_usable.

---

## Postscript G — item + ability structural feature lookups

**Triggered by:** "items are tokenized — won't they be learned?" + "is
there a reason we can't tokenize items?" — the answer is "yes, with
sufficient capacity / data, embeddings learn effects" (256-dim is way
past legacy's 32/64-dim oversaturation), but giving the model a small
head start with structurally-derivable features is principled and
free, IF we don't hand-curate.

**Approach:** parse Showdown's `items.ts` and `abilities.ts` at
lookup-build time. Pure regex/pattern-based extraction of structurally-
present fields. NO interpretation of effect callbacks (those are JS
functions, untouchable as code without writing a JS interpreter, which
would be hand-curation by another name).

### G1. Item features (8 dim per item)

Parsed from items.ts `_split_top_level_blocks`:

| Field | Source pattern | Examples |
|---|---|---|
| `is_berry` | `isBerry: true` | Sitrus Berry, Lum Berry, Salac Berry |
| `is_gem` | `isGem: true` | Normal Gem, Fire Gem (gen 5/6) |
| `is_pokeball` | `isPokeball: true` | (battle-irrelevant; included for completeness) |
| `is_mega_stone` | `megaStone: { ... }` present | Garchompite, Charizardite Y |
| `is_z_crystal` | `zMove:` or `zMoveType:` present | Normalium Z, Firium Z |
| `ignore_klutz` | `ignoreKlutz: true` | Ability Shield |
| `fling_bp_norm` | `fling.basePower / 130` | Iron Ball (130), Hard Stone (100) |
| `natural_gift_bp_norm` | `naturalGift.basePower / 100` | berries used with Natural Gift |

Build result: 515 / 2339 items have at least one flag set (the rest
are battle-irrelevant or have only callback effects — Choice
Band/Specs/Scarf, Life Orb, Leftovers fall into the "callback only"
bucket and rely on the embedding learning their effects).

### G2. Ability features (7 dim per ability)

Parsed from abilities.ts:

| Field | Source pattern | Examples |
|---|---|---|
| `breakable` | `flags: { breakable: 1 }` | Aroma Veil, Armor Tail, Sturdy, Battle Armor |
| `cantsuppress` | `flags: { cantsuppress: 1 }` | Multitype, Stance Change |
| `notrace` | `flags: { notrace: 1 }` | Forecast, Flower Gift, Imposter |
| `notransform` | `flags: { notransform: 1 }` | Disguise, Battle Bond |
| `no_skill_swap` | `flags: { noskillswap: 1 }` | Wonder Guard, Multitype |
| `is_permanent` | `isPermanent: true` (top-level) | As One (Glastrier/Spectrier) |
| `suppress_weather` | `suppressWeather: true` | Air Lock, Cloud Nine |

Build result: 116 / 313 abilities have at least one flag set.

### G3. Tokenizer integration

- `ItemTokenizer`: `id_embed (entity_embed_dim=32) + ITEM_FEAT_DIM=8`
  → 2-layer MLP → `d_model`. Replaces direct `item_embed(item_id)` in
  `_encode_pokemon_block`.
- `AbilityTokenizer`: same pattern, `id_embed + ABILITY_FEAT_DIM=7` →
  MLP → `d_model`.
- `MoveTokenizer` already follows this pattern (id + banks + 119-dim
  flags → MLP → d_model). The two new tokenizers are deliberate
  parallels.

### G4. What this gives the model

For Sitrus Berry: the embedding starts knowing "this is a berry,"
which composes via attention with `cont[..., _SL_VOLATILE]` (the
volatiles slice in the per-Pokemon block, which includes the
`STOCKPILE`/`SUBSTITUTE`/etc. flags) and the HP bank. The "consumed
when below 50%" effect still has to be learned, but the categorical
"this is berry-shaped" prior is given.

For Air Lock: ability_token starts knowing "suppress_weather=1," which
the model can compose with the weather_token's value to predict that
weather effects don't fire. Faster convergence than learning purely
from outcomes.

For Choice Band: NO structural feature gives "1.5× phys atk" — that's
in the JS callback. The id embedding still has to learn this. With
256-dim and 5M training samples this is well within learnable range
(see Postscript A on items being learnable). The structural features
are a head start, not the main mechanism.

### G5. Schema bump and lookup file

`LOOKUP_SCHEMA_VERSION: 3 → 4`. The .pt blob now has:
- `flags` (n_moves, 119)
- `banks` (n_moves, 4)
- `valid` (n_moves,)
- `damage_chart` (N_TYPES+1, N_TYPES+1)
- `item_features` (n_items, 8)               — NEW v4
- `ability_features` (n_abilities, 7)         — NEW v4
- `meta` dict with all schema fields

Loader rejects v3 files with a "rebuild" error.

### G6. Verification status

- `verify_move_lookup.py`: 50/50 strict + 50/50 STAB-only-divergence.
- `test_tokenizer.py`: 17/17 pass. New test 15:
  - 4/5 canonical item-flag pairs verified (sitrusberry/garchompite/
    normaliumz/abilityshield; rockygem absent from gen-9 vocab — ok)
  - 4/4 canonical ability-flag pairs verified (aromaveil/armortail
    breakable; airlock/cloudnine suppress_weather)
  - Gradient-flow check: zeroing `item_features_lookup` changes the
    item_token output by 1.78 (nontrivial; structural features
    reach the token).
- Bench: 30.46 ms median for B=32 (was 30.04). +0.4 ms; within noise.
- Param count: 1,447,224 → ~1,470,000 (+~25K from the two new
  tokenizers' first Linears).

---

## Postscript H — Session 47 Week 2 implementation notes

Triggered by completing Week 2 (spatial + temporal stack + heads + PPO
smoke). Items below document deviations from §4 / §5 / §7-Week-2-prompt.

### H1. Param count: bumped from 9.5M to 20.0M (d_temporal 256 → 512, K 2 → 4)

The Week 2 prompt called for 20-30M params and design §4.5 flagged
"~25M params target." Initial Week 2 implementation came in at
**9,495,596** — far below the target — because attention at d=256 is
more compact than the §4.5 estimate assumed. The spec exactly matched
§4 (d_model=256, 6 spatial × 4 temporal layers, 8 heads, K=2,
d_temporal=256), so the estimate, not the implementation, was off.

**Direct evidence the initial 9.5M was undersized**, from
METAMON_LEARNINGS.md §1.1-§1.3 + REWRITE_DESIGN.md §9:

| Model              | Params | Spatial d | Temporal d | T:S ratio | K  | Per-turn output |
|--------------------|--------|-----------|------------|-----------|----|-----------------|
| Minikazam (GRU)    | 4.7M   | 64        | 400        | 6.25×     | 5  | 320             |
| **Small**          | **15M**| 100       | **512**    | **5.12×** | 4  | 400             |
| Medium             | 50M    | 100       | 768        | 7.68×     | 6  | 600             |
| Large              | 200M   | 160       | 1280       | 8.00×     | 11 | 1760            |
| **Initial Week-2** | 9.5M   | 256       | **256**    | **1.00×** | 2  | 512             |

Two structural anomalies vs every published Metamon size:
1. T:S ratio = 1.0× vs their 5-8× — METAMON_LEARNINGS.md §1.2
   identifies this as the "highest-leverage candidate experiment
   (before any BC scaling): shift capacity from spatial to temporal."
2. Total 9.5M < Metamon Small's 15M — REWRITE_DESIGN.md §9 explicitly
   targeted "slightly larger than Small to compensate for our lack of
   their 18M-trajectory dataset (we have ~5M)."

**Decision:** bump **d_temporal 256 → 512** + **K 2 → 4** before
shipping Week 2. New total **19,994,924** params. Doesn't fix the T:S
ratio fully (2.0× vs Metamon's 5-8×; bounded by our wider d_model=256
which is itself a deliberate choice for compositional structure), but
hits all the doc targets:

| Module                | Was          | Now (d_t=512, K=4) | Δ        |
|-----------------------|--------------|---------------------|----------|
| Tokenizer             | 1,007,096    | 1,007,608           | +512     |
| SpatialTransformer    | 4,738,560    | 4,738,560           | 0        |
| TemporalTransformer   | 3,210,240    | **12,711,936**      | +9.5M    |
| Summary→Temporal proj | 131,328      | 524,800             | +394K    |
| ActionHead            | 197,121      | 525,313             | +328K    |
| ValueHead             | 144,435      | 419,891             | +276K    |
| SwitchActionEncoder   | 66,816       | 66,816              | 0        |
| **TOTAL**             | **9,495,596**| **19,994,924**      | +10.5M   |

- **20.0M lands ~33% above Metamon Small's 15M** — matches design §9
  intent.
- **Per-turn output: 4 × 256 = 1024-dim** (between Small's 400 and
  Large's 1760).
- **VRAM:** at fp32, params ~80 MB; AdamW state ~160 MB; BC training
  activations B=32 / T=200 fp16 AMP ~250 MB. Comfortably <1.5 GB on
  6 GB RTX 3060 Laptop.
- **Forward speed (B=32, single-turn):** 36.45 ms median (was 35.70 ms
  at 9.5M). +0.75 ms — negligible because temporal at T=1 has trivial
  attention compute. Still well under the <80 ms target.
- **forward_sequence + backward (B=8, ~190 valid turns):** ~4.8 s
  (~25 ms/turn). Same shape of cost as legacy v9; quadratic-in-T
  temporal recomputation is preserved from model.py:937-949 (a known
  trade-off, not a Week-2 issue).

Headroom for further bumps if Week 4 PPO plateaus: bump d_model 256 →
384 (~+7M cheaply); bump n_spatial_layers 6 → 8; or lift d_temporal
to 768 to hit Metamon Medium's recipe.

### H2. Side-mask: skipped for V1

Design §3.4 specified Poke-Mask + side-mask; the latter is a no-op when
opp's hidden info is mapped to learnable "unknown" embeddings (which is
what the Tokenizer does for unrevealed items / abilities / moves).
Implementation skipped per the prompt's allowance ("can disable later").
Re-add when adding TTA or symmetric BC training. No test asserts its
presence; the Poke-Mask test in `test_policy.py` does cover the
decision-token isolation that matters in practice.

### H3. Action head context: species-match permutation, not by-index

The prompt described the per-action context for switch slot j as
`concat(mean_pool(bench_pokemon_j_tokens), switch_cont[:, j, -2:])` —
implicitly assuming `bench_pokemon_j` matches `switch_slot_j`. The two
orderings differ: our bench is alphabetical-by-species (features.py:471
sorts), `switch_ids` / `switch_cont` are in poke-env's
`available_switches` order (game order — fainted excluded, etc.).
Implementation: build a permutation on the fly via species-id matching
(`switch_ids[:, j] == bench_species[:, k]` → gather index). Same logic
applied to active moves: `our_pokemon_move_ids[:, 0, :]` is in
`pokemon.moves` order; `active_move_ids` is in `available_moves` order.
Permute via move-id match.

`test_policy.py::test_switch_context_species_match` and
`::test_move_context_permutation` verify every legal switch / active
move has a corresponding bench / move slot. For empty / illegal
positions the legal mask masks the output, so the gather index doesn't
matter (defaults to argmax-of-all-equal = 0).

### H4. PPO smoke: synthetic, not real

Design §7 Week 2 listed both real-PPO 1-iter smoke and synthetic-PPO as
acceptable. Implemented the synthetic (`test_synthetic_ppo_step`):
forward_sequence on a real human_v8_100k BC batch (B=2 episodes,
T_max=21 turns), legal-mask-aware action sampling, fake advantages,
fake twohot value targets, AdamW lr=1e-4. 246/246 parameters updated;
grad-norm L2 = 1.57; loss finite. Real-PPO integration deferred to
Week 3 (BC training comes first; PPO is Week 4).

### H5. Future-warning silenced (`mismatched src_key_padding_mask and mask`)

PyTorch warns that the float causal mask + bool padding mask have
mismatched dtypes in `TemporalTransformer.forward`. The legacy
`TemporalTransformer` (model.py:495-502) has the same warning and has
been ignored for a year. Same here. If a future PyTorch version
deprecates this combination, cast both to bool (or both to float -inf)
in one place.

### H6. forward_sequence: Python loop over (b, t)

The mega-batch path mirrors model.py:842-849 — a Python `for b in range(B):
for t in range(L):` loop builds the (b, t) index list, then a vectorized
gather. For B=64 / T=200 this is 12,800 iterations of pure Python — a
few ms per BC step. Same as legacy. If BC training profiling shows it
matters, replace with `torch.nonzero(mask)` (fully tensor-based). Not
worth it for V1.

### H7. Verification status post-Week-2 (after H1 capacity bump)

- `test_tokenizer.py`: 17/17 pass (no regressions from Week 2 or
  the H1 bump).
- `test_policy.py`: 9/9 pass at 20M params.
- Param count breakdown: 19,994,924 — see H1 table.
- B=32 single-turn full-policy forward: **36.45 ms median** on RTX 3060
  Laptop CUDA fp32. Target <80 ms.
- forward_sequence + backward at B=8 (190 valid turns total): ~4.8 s
  (~25 ms per valid turn).
- Synthetic PPO step: pi_loss=-0.13, v_loss=3.94, total grad-norm 3.24,
  246 params updated by AdamW step.

---

## Postscript I — Session 48 Week 3 plumbing + B=8 memory cliff

Triggered by completing Week 3 sub-task 1 (the `--use-transformer` flag
plumbed end-to-end through `train_bc.py` and `ppo.py::load_checkpoint`)
and running the first throughput bench against the 20M model.

### I1. Plumbing changes

- `train_bc.py`:
  - `--use-transformer` flag (default off — legacy MLP unchanged).
  - When on: factory builds `TransformerBattlePolicy(TransformerConfig.with_vocab_sizes_from_disk(),
    move_flag_lookup=load_move_flag_lookup(...))`. Forces `--eval-games 0`
    with a printed notice (option (a) per next-prompt.txt: defer
    `BattleAgentTransformer` to Week 5).
  - Resume guard: if `--resume <ckpt>` and ckpt's `arch` field disagrees
    with the run's `--use-transformer` flag, raise a clear `SystemExit`
    instead of letting `load_state_dict` produce a KeyError storm. Old
    checkpoints without an `arch` field are inferred from state-dict key
    prefixes (`tokenizer.`, `spatial.`, etc. → transformer; everything
    else → mlp).
  - All saved checkpoints (`epoch_NNN.pt`, `step_NNN.pt`, `best.pt`,
    `mid_step*`, `_eval_temp.pt`) get an `arch: "transformer" | "mlp"`
    field. `config.json` gets the same field for human-readable inspection.
- `ppo.py::load_checkpoint`:
  - Reads `ckpt["arch"]` (falling back to state-dict-key inference) and
    dispatches between `PokeTransformer` (legacy + dim-expansion logic
    preserved) and `TransformerBattlePolicy` (no dim-expansion needed —
    transformer ckpts can't predate the spec). Same `(model, cfg, ckpt)`
    return signature.
  - `save_checkpoint` adds the `arch` field automatically by detecting
    `cfg`'s class name. Callers (`train_rl.py`) don't need to change.

### I2. Plumbing smoke (CPU B=4)

Forty batches on CPU before manual termination. Validated:
- 19,994,924 params confirmed at training start.
- Loss decreasing: 1.66 (batch 20) → 1.61 (batch 40), accuracy ~0.32.
- No NaN, no Inf, scheduler + AdamW running cleanly.
- Print path uses `flush=True`; reports every 20 batches.

### I3. Throughput bench: the B=8 memory cliff

`bench_bc_step.py` (added) measures per-batch wall-clock for the BC step
(forward_sequence + loss + backward + AdamW) on 5-8 pre-loaded real
batches. Standalone — no DataLoader/workers — to isolate model-side
throughput from data pipeline.

**B=8 fp16 CUDA results (5 batches):**

| Batch | Valid turns | dt (ms) | per-turn (ms) | peak_mem (GB) |
|-------|-------------|---------|---------------|---------------|
| 0     | 190         | 1263    | 6.6           | 3.50          |
| 1     | 146         | 753     | 5.2           | 3.50          |
| 2     | 222         | 1092    | 4.9           | 4.28          |
| 3     | 258         | **37557**| **145.6**    | **5.12**      |
| 4     | 248         | 40762   | 164.4         | 5.12          |

Batches 0-2 are healthy at 5-7 ms/turn fp16. **Batch 3 spikes 30× when
peak memory crosses ~5 GB**, and stays slow afterward. The cliff is
not OOM (we top out at 5.12 / 6.14 GB) but a thrash: cuda allocator
fragmentation + spillover synchronization once the cache nears the
device ceiling.

The dominant memory consumer in `forward_sequence` is the spatial pass
mega-batch over `N_valid` turns × 220 tokens × d_model=256 with 6
layer's worth of activations, plus the `(B, T, 220, 256)` fp32
`spatial_grid` allocated as workspace. At B=8 with avg_T=26 and
occasional T=66 episodes, N_valid hits 250+ and memory peaks above the
cliff.

**B=4 fp16 CUDA results (8 batches):** all stable at 6-11 ms/turn,
peak memory 1.5 → 2.74 GB. No spikes. Loss decreasing across batches
(pi: 1.74 → 1.71 ; v: 1.94 → 0.42).

### I4. Decision: B=4 is the local operating point

- Per-turn: ~7 ms fp16 median (call it 7-10 ms with margin).
- Epoch wall-clock: 7 ms × 5M turns ≈ 9.7 hr → call it ~10 hr/epoch.
- 5 epochs ≈ 50 hr ≈ 2.1 days. 10 epochs ≈ 100 hr ≈ 4.2 days.
- Cloud-trigger threshold (REWRITE_DESIGN.md §7 Week 3): >5 days local.
  Local at B=4 is right at the boundary — if there's any extra slowdown
  (Windows sleep, thermals, validation overhead, eval fires), cross it.

**Recommendation for Session 49**: launch a 5-epoch BC at B=4 fp16
locally, with `--workers 2 --eval-games 0 --val-ratio 0.05`, monitoring
val_loss for early-stopping after epoch 2-3 (legacy converged at epoch 2
for smart_avg 45.1%). If after 24 hr the wall-clock projects past 5
days, kill and migrate to cloud per CLOUD_DEPLOY.md.

### I5. Why not lower T_max instead of lowering B?

Could clamp episodes to T ≤ 100 in the dataset, which reduces grid
sizes and N_valid spike. But:
- Episodes with T > 100 are rare (avg_T ≈ 26, max in spot-checks 66-200).
- Truncating loses long-game training signal — exactly what BC needs
  to learn endgame play, hazard tracking, set prediction.
- B=4 already fits comfortably without truncation.
Stick with B=4 for V1.

### I6. Headroom if B=4 throughput is unacceptable

- **Cloud A100 40 GB**: 7× memory headroom → B=32 trivially fits, ~10×
  faster per epoch (per Postscript H estimate). 5 epochs in ~12 hr cloud
  vs 50 hr local. ~$50-150 budget per CLOUD_DEPLOY.md.
- **Truncate `forward_sequence` to chunks**: process B in mini-batches
  of 2 episodes at a time inside the function, accumulating gradients.
  ~2× memory savings, but adds Python overhead and complicates BC's
  backward pass. Not worth it unless cloud is unavailable.
- **Drop fp32 `spatial_grid`**: cast to fp16 with `torch.zeros(...,
  dtype=torch.float16)`. Saves ~half the workspace memory but risks
  precision loss on the action_head context input. Worth testing if
  time pressure forces it.

### I7. Verification status post-Session-48

- All 17/17 tokenizer + 9/9 policy + 50/50 lookup tests still pass.
- Plumbing smoke: 40 batches CPU clean. CUDA bench: B=4 stable, B=8 cliff.
- Both checkpoints (legacy MLP + transformer) load round-trip via the
  new `load_checkpoint` dispatch (legacy verified by existing PPO ckpts;
  transformer verified manually via plumbing smoke save → torch.load).
- `bench_bc_step.py` committed for future profiling.
