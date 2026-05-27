# Multi-Gen Support — Feasibility & Scope (Gen 6+)

**Created:** Session 50 continuation (2026-05-06)
**Scope decided:** Gen 6, 7, 8, 9, and future gens as they ship. **Pre-gen-6 explicitly out of scope.**
**Status:** Architectural scoping — not a Phase 3 commitment yet, but a clean estimate of cost/risk if we go.

---

## Why "Gen 6+" is the right scope

The big mechanical changes are all in gens 1-5; gen 6 onwards is largely additive:

| Gen | Big change | Why pre-gen-6 is messy |
|---|---|---|
| 1 | Original mechanics | Different damage formula (1/255 stat divisor), critical hit math via Speed, 15 types, no Special/Physical split |
| 2 | +Dark, +Steel types | New types break gen-1 type chart |
| 3 | Abilities introduced, modern stat formula | Pre-gen-3 has no abilities at all |
| 4 | Physical/Special split (per-MOVE not per-TYPE) | Damage calc fundamentally different from gens 1-3 |
| 5 | Hidden Abilities, Eternal Weather, dream world | More items + abilities, but mechanics start stabilizing |
| 6+ | **Fairy type, Mega Evolution; mechanics stable from here on** | (this is where we start) |
| 7 | Z-moves (additive) | |
| 8 | Dynamax/Gigantamax (additive) | |
| 9 | Terastallization (additive) | |

**From gen 6 onward, the damage formula, stat formula, type chart, and core mechanics are stable.** The new gimmicks (Mega/Z/Dynamax/Tera) are additive features the model can condition on via gen-id, not fundamental mechanic rewrites.

This makes "gen 6+" a vastly cleaner scope than "all gens 1-9" — type chart is identical (18 types since gen 6), damage formula identical, status mechanics identical. We're really just dealing with:
- Different species pools (move/ability/item availability per gen)
- Different gimmicks (Mega/Z/Dynamax/Tera per gen)
- Different metagames (gen-specific viable strategies)

---

## What's already multi-gen-ready in the workspace

Audited Session 50 continuation:

| Layer | Status | Source |
|---|---|---|
| **Pokedex/move/item/ability data** | ✅ Complete for all gens | `raw_data/pokemon/{1to8,ScarletViolet_PBS}/`, `raw_data/items/items.csv` (with `gen_added`), `raw_data/movesets/moves.csv` (with `gen_added`) |
| **Smogon usage stats** | ✅ Gen 5-9 monthly dumps from 2014-11 → 2024-04 | `raw_data/pokemon_usage/<YYYY-MM>/gen{N}ou-*.txt` |
| **Vocab** | ✅ Already iterates `for gen in range(1, 10)` and unifies all species/moves into one ID space | `vocab.py:152-162` |
| **Model embeddings** | ✅ Sized to full multi-gen vocab (n_species=1548, n_moves=953, n_items=2340, n_abilities=314). Gen 6-9 entries exist; just need training signal. | `model_transformer.py:339-342` |
| **FormatConfig** | ✅ Has `gen` field; `format_from_str()` parses gen from format string. | `format_config.py:39, 73-77` |
| **poke-env / battle_server.js** | ✅ Already gen-aware | poke-env's `Battle(gen=N)` constructor |

**Conclusion: ~80% of the data and infrastructure is in place.** The remaining work is mostly model conditioning + data assembly.

---

## 🚨 PHASE 2 PREREQUISITE — encoding completeness audit (S67-EXT, 2026-05-27)

**MUST be done before BC v11 retrain.** Schema changes are free during the BC retrain window; ~10× cost if added later (would invalidate v11 + all multi-gen memmap).

Game state elements currently NOT encoded (or poorly encoded) that should be added:

### Tier 1 — Critical (no proxy, model can't learn from observation)
- **Substitute HP** — sub absorbs damage with no HP delta on the mon behind it; stateful damage-calc accumulation NNs are weakest at
- **Nature** — only resulting stats encoded; can't predict "Adamant boost +Atk/-SpA" structurally; needed for team-building too
- **EVs spread** — only final stats; can't distinguish 252/252/4 from 244/0/252 spreads; critical for team-building
- **IVs spread** — needed for gens 1-7 Hidden Power type derivation; also for 0-IV trick room mons
- **Tera type as own token** — currently bundled in status MLP; should be its own dedicated token

### Tier 2 — Duration counters (currently flag-only)
- Encore (3), Taunt (3), Disable (4), Yawn (1), Tailwind (3-4), Heal Block (5), Wish (1-2), Sleep (1-3), Future Sight (3)
- Model has indirect signal via temporal stack but explicit counters save attention capacity

### Tier 2 — State attributes not exposed
- Choice lock specific move (which move is locked, not just that one is)
- Wish HP/turn counter
- HP exact value (own side) in addition to hp_pct

### Tier 3 — Pre-computed knowledge (consider)
- Type effectiveness matrix per active mon
- Speed tier comparison explicit
- Damage calc baselines per move-vs-target

**Full audit + investigation methodology + cost estimate**: see [`project_encoding_audit_phase2_todo.md`](../../.claude/projects/C--Users-raiad-OneDrive-Desktop-team-builder/memory/project_encoding_audit_phase2_todo.md) in memory.

**Why this matters for team building**: future team-building work needs nature + EVs visibility to reason about spreads. Without it, structural blind spot.

**Decision flow at Phase 2 prep window**:
1. Enumerate ALL Tier 1 + 2 gaps from audit memo
2. Decide encoding format for each (dedicated token / MLP / bank)
3. Update features.py + dataset.py + model_transformer.py + bump LOOKUP_SCHEMA_VERSION (4 → 5)
4. Re-encode multi-gen replay corpus with full v5 schema
5. BC v11 trains on v5 directly — no later retrofitting needed

---

## What needs to be built

### A. Architectural changes (small, ~1 week)

#### A1. Gen-id token in model input (~30-50 lines)
Add a learned embedding table for gen IDs:
```python
# In TransformerBattlePolicy.__init__
self.gen_embed = nn.Embedding(10, cfg.d_model)  # gens 0-9 (0 = unknown/multi)

# In forward:
gen_id = batch.get("gen_id")  # (B,) long, default to cfg.format_config.gen
gen_token = self.gen_embed(gen_id).unsqueeze(1)  # (B, 1, d)

# Concat at the front of the spatial sequence
seq = torch.cat([actor_t, critic_t, trans_t, gen_token, ...], dim=1)
```

The gen token tells the model "expect Mega Evolution / expect Tera type" etc. Everything else flows through.

#### A2. Gen-aware feature pipeline
- `make_features(battle)` already takes a `Battle` object that knows its gen via `battle.gen` (poke-env)
- Need: include `gen_id` in the batch dict
- Need: thread gen-specific feature gates (Tera type field only meaningful in gen 9; Z-moves only in gen 7; etc.) — these can be zeroed out for non-applicable gens via legal-mask logic

#### A3. Per-gen procedural teambuilder
- `team_generator.py:477` already has `self.gen = gen`
- Need: filter species/movesets/items by `gen_added <= gen`
- Need: gen-specific viability filters (e.g., don't include Mega stones in gen 8+)

### B. Data assembly (medium, ~3-5 days of compute + processing)

#### B1. Per-gen replay corpus
Source: HuggingFace `jakegrigsby/metamon-raw-replays` already filters by gen + rating.

| Gen | Estimated replay count @ ≥1500 ELO |
|---|---|
| Gen 6 (ORAS) | Plentiful — popular era, ~50k-100k replays |
| Gen 7 (SUMO/USUM) | Plentiful — ~50k-150k replays |
| Gen 8 (SS) | Moderate — ~30k-80k replays |
| Gen 9 (SV) | We already have 100k cleaned |

Should be enough for solid BC training across all 4 gens. Total: ~250k-400k replays.

#### B2. Multi-gen memmap
- Modify `replay_to_memmap.py` to accept multi-gen replay stream (already gen-aware via `_parse_gen_from_format`)
- Single memmap with mixed gens, gen-id field per sample
- Estimated size: ~3-4x current 104 GB human_v8_100k = ~300-400 GB on cloud

#### B3. Re-train BC v11 on multi-gen
- Same architecture, same hyperparameters
- 5 epochs at B=48 fp16 on A100 80GB → ~5-7 days compute (vs gen 9 alone's 5 epochs / ~24 hr)
- Cost: probably 4-7x current BC training compute

### C. PPO multi-gen (small, ~2-3 days)

#### C1. Per-iter format selection
Currently `--battle-format gen9ou` hardcodes the gen for the entire run. Need to either:
- Train one PPO run per gen separately, OR
- **Mix gens within one PPO run** — sample a gen per game, pass gen-id to the model

Mixing is more efficient but requires the BC base to be multi-gen first (which B3 provides).

#### C2. Per-gen eval
Smart-bot eval (SH/SmartDmg/Tactical/Strategic) needs to work across gens. Current bots are gen-9 specific in their move-pool assumptions but the heuristic logic should generalize. Worth a quick smoke test per gen.

### D. Per-gen mechanic gating (small but fiddly, ~1-2 days)

Gen-specific gimmicks need legal-action gating:
- Mega Evolution: only legal in gen 6-7 with Mega stone
- Z-moves: only legal in gen 7 with Z-crystal
- Dynamax/Gigantamax: only legal in gen 8
- Tera: only legal in gen 9

Most of this is already handled by Pokemon Showdown's server (it won't accept illegal actions). But the model's `legal_mask` needs to reflect what's actually legal in the current state, which poke-env's Battle object should already provide.

---

## Effort estimate

**Path A (minimum viable multi-gen):**
- Gen-id token + gen-aware feature gate: ~1 week
- Multi-gen BC retrain: ~1-2 weeks (data assembly + cloud compute + validation)
- Per-gen PPO smoke + eval: ~3-5 days
- **Total: ~3-4 weeks of focused work**

**Path B (production-quality multi-gen):**
- Path A + per-gen specific tuning (move flag lookups per gen for accurate item/ability info)
- Per-gen eval bots properly tuned
- Multi-gen evaluation harness (smart_avg-per-gen tracking)
- **Total: ~5-7 weeks**

---

## Risks / open questions

1. **BC quality across gens may be uneven.** Gen 9 has the most replay data and the best ladder activity. Gen 6-7-8 metas are smaller. Multi-gen BC might be strong on gen 9, weaker on gen 6. Tracking smart_avg per gen separately is essential.

2. **Capacity allocation in the model.** A 20M-param model trained on gen 9 alone learns gen-9-specific patterns. Adding 3 more gens worth of strategy might require **bigger model**. Worth ablating: same 20M param model trained on multi-gen, vs scaling to 40M+ for multi-gen.

3. **Procedural teambuilder needs gen-aware filtering.** Smogon usage stats are per-gen, so this is straightforward — just point it at the right month's gen data. But verifying the procedural samples are valid in the chosen gen is necessary.

4. **Self-play distribution shift across gens.** PPO trains against past snapshots. If snapshots came from playing gen 9 but we're now playing gen 7, trajectory distribution differs. Mitigation: per-gen snapshot pool (separate PFSP pools per gen). Adds complexity.

5. **Eval bots may not work in older gens.** SmartDamagePlayer's pivot detection, TacticalPlayer's hazard play, etc. all assume gen-9-ish item/move pools. Need to test or stub each gen.

6. **Tera type encoding.** In gen 9 each pokemon has a `tera_type` field that's 0 (unset) for non-gen-9 plays. Already present in our feature space. Pre-gen-9 just gets 0 (typeless). Should be handled cleanly.

---

## Recommended phasing

If we ever pursue multi-gen, suggested order:

1. **Phase 3a (Foundation):** Gen-id token in model. Train a small BC on gen 8 + gen 9 data (just two gens). Validate the gen-id signal actually conditions the model. ~1-1.5 weeks.

2. **Phase 3b (Data scale):** Pull gen 6, 7 replay data. Re-do BC with all 4 gens. Validate per-gen smart_avg holds. ~2 weeks compute + analysis.

3. **Phase 3c (PPO multi-gen):** Single PPO run with gen sampled per game. Mixed-gen self-play. Per-gen eval reporting. ~1-2 weeks.

4. **Phase 3d (Production quality):** Per-gen bot tuning, per-gen procedural team validation, deeper eval harness. ~1-2 weeks.

Total: 4-7 weeks of focused work to get a production-quality multi-gen system.

---

## Honest take

**Multi-gen at gen 6+ scope is genuinely tractable.** The architecture was forward-designed for it (vocab is multi-gen, FormatConfig has `gen`, data exists), the mechanic stability from gen 6 onward removes the hardest parts, and we can sequence it incrementally.

**Biggest unknown is whether 20M params is enough for multi-gen.** Gen-9 alone has used about half the model's capacity; squeezing 4 gens worth of strategy into the same 20M might require scaling. Cloud BC retrain at, say, 40M-60M would be the natural counter-experiment if multi-gen-20M underperforms.

**Don't pursue this until Phase 1 v2 stabilizes** — until we know the new arch + lr=1e-5 (or whatever wins the ablation) actually produces a strong gen-9 PPO model, expanding to 4 gens is premature. But once we have a strong gen-9 baseline, multi-gen is **the natural next architectural lever** alongside (or instead of) BC scaling.

---

## Files referenced

- `pokemon-ai-starter/pokemon-ai/src/vocab.py:152-162` — vocab already multi-gen
- `pokemon-ai-starter/pokemon-ai/src/format_config.py:39, 73-77` — FormatConfig with gen field
- `pokemon-ai-starter/pokemon-ai/src/model_transformer.py:339-342` — vocab-sized embeddings
- `pokemon-ai-starter/pokemon-ai/src/team_generator.py:477` — gen field present
- `raw_data/pokemon/`, `raw_data/items/items.csv`, `raw_data/movesets/moves.csv` — source data with `gen_added` columns
- `raw_data/pokemon_usage/<YYYY-MM>/gen{N}ou-*.txt` — per-gen Smogon usage stats
- HuggingFace `jakegrigsby/metamon-raw-replays` — multi-gen replay source (per `next-prompt.txt` references)
