# CURRENT_ARCH_PATHWAY.md — Live Data Flow & Architecture Reference

**Created:** 2026-05-23 (S67, mid Phase-2 Stage-1 anneal experiment)
**Scope:** `TransformerBattlePolicy` (Session 46+ rewrite). Documents what the LIVE model actually consumes vs. what's encoded-but-dead, all lookup tables, and the architectural data path end-to-end. Sibling docs (`REWRITE_DESIGN.md`, `ARCH_AUDIT.md`) cover *design intent* and *trainer-side dispatch*; this one is the model-internal pathway.

**Confidence labels:** HIGH (read code both sides), MEDIUM (inferred from comments+structure), LOW (best guess).

---

## TL;DR — what this doc settled

1. **Dead bytes confirmed**: `_SL_MOVE_COMPACT` (92 dims per Pokemon × 12 = 1104 dims per turn) is encoded into the memmap but NEVER read by the model. Same applies to the bench-Pokemon entries in `pokemon_move_cont`. ~82MB wasted per 50k-turn dataset.
2. **Live move pathway**: Lookup-table based — `move_flags_lookup` (n_moves × 119) + `move_banks_lookup` (n_moves × 4) populated at init from poke-env move metadata. Accuracy/PP ARE included.
3. **Live data path**: features.py → memmap → dataset.unpack_turn_batch → Tokenizer.forward → spatial transformer (6 layers, 220 tokens) → temporal transformer → action heads (9-way: 4 moves + 5 switches).
4. **Active-mon override pathway** ("Postscript D"): slot-0 moves get real per-turn PP/disabled/STAB values via `active_real_banks` + `active_real_flags`, overriding the static lookup. This is the only place dynamic move state enters the model.
5. **All lookups built at init** from Showdown source (`items.ts`, `abilities.ts`) and poke-env (`Move`, `Pokedex`) — saved in a `.pt` file with `LOOKUP_SCHEMA_VERSION` for invalidation. Current version: v4.

---

## §1 Data Flow Diagram (end-to-end)

```
┌─────────────────────────────────────────────────────────────────────┐
│ features.py                                                          │
│   _encode_pokemon → pokemon_cont (285 dims), pokemon_banks, ids     │
│   _encode_field   → field_cont (52 dims), field_banks               │
│   _parse_turn_events / _encode_transition → trans_cont (51 dims), ids│
│   _encode_move_compact → 92 dims per pokemon ⚠️ DEAD                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ memmap files (dataset.py:51-99)                                      │
│   our_pokemon_{ids, banks, cont, mcont}.npy                          │
│   opp_pokemon_{ids, banks, cont, mcont}.npy                          │
│   field_{banks, cont}.npy                                            │
│   trans_{ids, cont}.npy                                              │
│   move_{ids, banks, cont}.npy   ← active mon's 4 moves               │
│   switch_{ids, cont}.npy        ← bench mons for switch action       │
│   legal.npy (9-dim mask), action.npy (BC label)                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ dataset.unpack_turn_batch → batch dict consumed by Tokenizer        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ model_transformer.py: Tokenizer.forward                              │
│   _encode_pokemon_block  → 17 tokens × 12 mons = 204 tokens          │
│   _encode_field_tokens   → 9 tokens                                  │
│   _encode_transition     → 1 token                                   │
│   _encode_our_threat     → 1 token                                   │
│   actor/critic/opp_threat/gen → 4 tokens                             │
│   summary scratch         → K=4 learnable                            │
│   TOTAL: 220 tokens per timestep                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Spatial transformer (6 layers, 8 heads, d_model=256, pre-norm)       │
│   Poke-mask: actor/critic isolation                                  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Temporal transformer (causal over [history + current])               │
│   Input: K summary tokens flattened → d_temporal                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Action heads (9-way: 4 moves + 5 switches) + value head              │
│   Legal mask applied at logit level (illegal → -100)                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## §2 Per-Pokemon Block (17 Tokens) [HIGH]

Stacked at `model_transformer.py:1320-1327`. Each Pokemon (12 total: 6 ours + 6 opp) becomes 17 attribute tokens of `d_model` dim each.

| # | Token | Inputs | Source field | Tokenizer |
|---|---|---|---|---|
| 1 | **species** | id only | `ids[..., 0]` | nn.Embedding(n_species, d) |
| 2 | **item** | id + 8-dim structural | `ids[..., 1]`, `item_features_lookup` | ItemTokenizer |
| 3 | **ability** | id + 7-dim structural | `ids[..., 2]`, `ability_features_lookup` | AbilityTokenizer |
| 4 | **type** | top-2 types from 19-dim multi-hot, sorted | `cont[_SL_TYPES]` | type_embed × 2 → type_mlp |
| 5 | **status** | 83 cont dims + 3 banks (96 dim) = 179 input | many slices (see below) | status_mlp |
| 6 | **hp_pct** | bank(101 buckets) | `banks[..., 0]` | hp_bank → bank_proj |
| 7-12 | **6 stat tokens** | bank(256) each | `banks[..., 4:10]` | stat_bank → bank_proj |
| 13-16 | **4 move tokens** | id + 4 banks + 119 flags | `ids[..., 3:7]`, lookup tables | MoveTokenizer |

### §2.1 Status token deep-dive (the catch-all)

Inputs concatenated at `model_transformer.py:1266-1280` (83 cont + 96 bank-embed = 179):
- status one-hot (7), volatile bits (38), paradox state (7), tera state (20)
- active flag (1), fainted flag (1), combat (5), toxic counter (1), future_sight (1), visibility (2)
- level_bank(32) + weight_bank(32) + height_bank(32)

**Note**: many of the active-only fields (volatile, paradox, combat) are zero for bench Pokemon. They share the same status MLP — the MLP learns to recognize zero patterns. Not strictly redundant but worth noting.

### §2.2 Move tokens — the LIVE move pathway [HIGH]

`MoveTokenizer.forward` (model_transformer.py:885-900) consumes:

| Input | Source | Dim | Notes |
|---|---|---|---|
| `move_id` | `ids[..., 3:7]` | int | learned move embedding |
| `bp` | `move_banks_lookup[..., 0]` | bank(256, bank_dim) | base power |
| **`acc`** | `move_banks_lookup[..., 1]` | **bank(101, bank_dim)** | **accuracy (0-100%)** |
| **`pp`** | `move_banks_lookup[..., 2]` | **bank(65, bank_dim_small)** | **Power Points** |
| `prio` | `move_banks_lookup[..., 3]` | bank(13, bank_dim_small) | priority |
| `flags` | `move_flags_lookup[move_id]` | 119 | structural (see §6) |

Concat → MLP → `d_model` token. **Accuracy and PP are part of the live pathway** — the user's recollection of "old arch missed accuracy" applies to legacy/v5 only.

### §2.3 ⚠️ Dead path: `_SL_MOVE_COMPACT` (92 dims per Pokemon) [HIGH]

`features.py:_encode_move_compact` produces 4 × 23 dims (type one-hot + bp + category + priority) appended to `pokemon_cont`. The slice `_SL_MOVE_COMPACT` is **defined** at `model_transformer.py:141` (structural assertion only) but **never read** by any `cont[..., slice]` call. Verified by grep across the file.

**Status**: dead. Storage waste: 92 × 12 mons × 50k turns ≈ **55 MB per 50k-turn dataset**.

**Why it exists**: Pre-Session-46 architecture consumed this directly. Lookup-table pathway (move_flags_lookup + MoveTokenizer) replaced it. The encoding was kept in features.py for backward-compatibility with old memmap layouts but is never consumed by the new model.

**Cleanup opportunity**: remove `_encode_move_compact` call from `_encode_pokemon` + drop `our_pokemon_mcont` / `opp_pokemon_mcont` from memmap layout + drop the `_SL_MOVE_COMPACT` assertion. Saves ~82 MB per dataset (includes opp side). NOT urgent — purely a hygiene cleanup.

---

## §3 Field Tokens (9 Thematic) [HIGH]

`Tokenizer._encode_field_tokens` (model_transformer.py:1331-1375). All 9 inputs are populated and consumed.

| Token | Cont slice | Bank | MLP input |
|---|---|---|---|
| weather | `_FL_WEATHER_OH` (5) | weather_dur (9) | 37 |
| terrain | `_FL_TERRAIN_OH` (5) | terrain_dur (6) | 37 |
| our_hazards | `_FL_OUR_HAZARDS` (4) | — | 4 (SR, spikes/3, tspikes/2, web) |
| opp_hazards | `_FL_OPP_HAZARDS` (4) | — | 4 |
| our_screens | `_FL_OUR_SCREENS` (6) | — | 6 (3 screens × {presence, dur}) |
| opp_screens | `_FL_OPP_SCREENS` (6) | — | 6 |
| speed_field | `_FL_TAILWIND` (2) + `_FL_TRICK_ROOM` (1) | tr_dur (6) | 35 |
| mechanics | `_FL_MECHANICS` (17) | — | 17 (tera/mega/z/dmax avail + used + dmax_turns + trapped + force_switch + opp_revealed_frac) |
| progression | `_FL_ALIVE` (2) | turn (201) | 66 (our_alive/6 + opp_alive/6 + turn embed) |

**All field slices are LIVE**. Zero dead bytes here.

---

## §4 Transition Token + Active Overrides [HIGH]

### §4.1 `_encode_transition` (1377-1387)

Inputs:
- `trans_ids["our_action"]` → action embed
- `trans_ids["opp_action"]` → action embed
- `trans_cont` (51 dims, see breakdown below)

Concat → trans_mlp → `d_model` token.

**trans_cont layout** (features.py:921-953, all consumed):
- Action kinds: our (3) + opp (3) one-hot
- Who moved first (3)
- Effectiveness outcomes: our (6) + opp (6)
- Crits (2), flinch (2), confusion (2)
- Statuses applied: to us (7) + to opp (7)
- Stat changes: 4 (our_gained, our_lost, opp_gained, opp_lost)
- KOs (2), entry hazard damage (2), field changes (2)

### §4.2 Active overrides ("Postscript D") [HIGH]

`_encode_pokemon_block:1298-1314`: slot-0 (active) move banks/flags get **real per-turn** values from `active_move_banks` / `active_move_cont` memmap fields, overriding the static lookup:

- **active_real_banks** → bp, acc, pp, prio per-turn values. Recovers current_pp (decremented), accuracy modifiers (sand, gravity, evasion boosts) for the action selection.
- **active_real_flags** → first 107 dims of the move flags vector, recovering 3 dynamic dims:
  - `cont[9]` = current_pp / 64
  - `cont[10]` = disabled flag
  - `cont[12]` = STAB (depends on user's current types — matters for Tera)
- The 12 extra structural flags at `[MOVE_FLAG_DIM_BASE:]` (slicing/bullet/charge/etc.) are static — kept from lookup.

**Why this matters**: without overrides, the model would see lookup-default PP/disabled/STAB for the moves it's about to choose between. The override patches this for the active mon only.

### §4.3 our_threat token (1389-1393)

Uses last 2 dims of `active_move_cont` per move (4 moves × 2 = 8 dims):
- type_eff_vs_opp (our move's type effectiveness against active opp)
- opp_threat_back (reciprocal — opp's expected damage)

Flattened → our_threat_mlp → `d_model` token. Encodes the matchup snapshot for action selection.

---

## §5 Lookup Tables (built at init, stored in `.pt`) [HIGH]

All tables loaded once at model construction from a single `.pt` file. Schema version `LOOKUP_SCHEMA_VERSION = 4` at model_transformer.py:71. Loader rejects mismatches loudly.

| Table | Shape | Built from | Consumed by |
|---|---|---|---|
| `move_flags_lookup` | (n_moves, 119) | poke-env `_project_move_flags(Move)` + 12 structural extras (slicing/bullet/bypasssub/pulse/charge/futuremove/ignore_defensive/use_target_offensive/thaws_target/reflectable/gravity/sleep_usable) | MoveTokenizer flags input, _encode_opp_threat |
| `move_banks_lookup` | (n_moves, 4) | poke-env Move → [bp_int, acc_int, pp_int, priority_int] | MoveTokenizer banks input |
| `damage_chart` | (20, 20) | (N_TYPES+1) × (N_TYPES+1) effectiveness matrix (the +1 is "no type") | _encode_opp_threat for type matchup |
| `item_features_lookup` | (n_items, 8) | `parse_showdown_items()` from Showdown `items.ts` — [is_berry, is_gem, is_pokeball, is_mega_stone, is_z_crystal, ignore_klutz, fling_bp_norm, natural_gift_bp_norm] | ItemTokenizer |
| `ability_features_lookup` | (n_abilities, 7) | `parse_showdown_abilities()` from Showdown `abilities.ts` — [breakable, cantsuppress, notrace, notransform, no_skill_swap, is_permanent, suppress_weather] | AbilityTokenizer |

**Build scripts**: `build_move_flag_lookup()` (model_transformer.py:602-716), `load_move_flag_lookup()` (729-781), `parse_showdown_items()` (465-497), `parse_showdown_abilities()` (500-528).

**Schema version history** (per model_transformer.py:60-70):
- v1: initial (flags + banks + valid + meta)
- v2: + damage_chart
- v3: MOVE_FLAG_DIM 107 → 119 (added 12 structural extras)
- v4: + item_features + ability_features (Session 46 Postscript G)

**Bumping the schema**: edit `LOOKUP_SCHEMA_VERSION` and regenerate the `.pt` file via the build script. The loader will refuse mismatched versions.

---

## §6 Move Flags Lookup (119 dims) [HIGH]

The richest move feature representation. Constructed once at init.

**107 base dims** from poke-env's `_project_move_flags(move)["continuous"]` — slice 81-99 is the type one-hot (19 dims). The remaining 88 dims encode all `Move` static properties:
- bp norm, accuracy, priority (also redundantly in banks)
- secondary effect chance + stat-drop chance + status chance
- target type, multi-hit, recoil, drain, crit_ratio
- contact, sound, makes_contact, ignore_defensive, etc.
- Dynamic at runtime: `cont[9]` = current_pp/64, `cont[10]` = disabled, `cont[12]` = stab — overridden per-turn for active mon (see §4.2)

**12 structural extras** appended at v3:
- slicing (Sharpness boost), bullet (Bulletproof immune), bypasssub (Sound), pulse (Mega Launcher boost)
- charge (2-turn moves: Solar Beam, Fly, Dive), futuremove (Future Sight, Doom Desire)
- ignore_defensive, use_target_offensive (Foul Play family)
- thaws_target (anti-freeze), reflectable (Magic Bounce target), gravity (suppressed under Gravity), sleep_usable (Sleep Talk, Snore)

---

## §7 Spatial + Temporal Stacks [HIGH for spatial, MEDIUM for temporal config]

### §7.1 Spatial Transformer

- **Layers**: 6 (cfg.n_spatial_layers)
- **Heads**: 8 (cfg.n_heads, but check config — sometimes 4)
- **d_model**: 256 (cfg.d_model) — current default; was 384 in some experiments
- **d_ff**: 4 × d_model = 1024 (cfg.ff_mult=4)
- **Activation**: GELU
- **Dropout**: 0.05
- **Norm**: pre-norm (norm_first=True)
- **Mask**: Poke-mask (model_transformer.py:1628-1649). Actor and critic CANNOT see each other; non-decision tokens blocked from actor/critic keys. Preserves policy/value separation.

**Token count for V1 singles**: 220 tokens
- Battle-state head: actor + critic + transition + our_threat + opp_threat + gen = 6
- Per-Pokemon: 17 × 12 = 204
- Summary scratch (learnable): 4
- Field tokens: 9 (note: actually folded into the 220 differently — check `total_tokens(fmt)` for exact)

### §7.2 Temporal Transformer

- **Context window**: cfg.temporal_context (200 turns default)
- **Mode**: "summary" — K summary scratch tokens (K=4) per spatial output, flattened → d_temporal
- **Layers**: cfg.n_temporal_layers (2 default)
- **Causal mask**: over [history turns + current turn]
- **Output**: (B, d_temporal) → action/value heads

### §7.3 Action heads [HIGH]

- **9 actions**: 4 move slots + 5 switch slots
- **Legal mask** applied at logit level (illegal → -100.0) at `model_transformer.py:2012`
- **Value head**: 51-bin distributional (cfg.v_bins=51)
- **Move/switch discrimination**: separate head MLPs share the d_model context but emit slot-specific logits

---

## §8 torch.compile Boundary [MEDIUM]

- **What's compiled (Tier 3 C5 / S64 era)**: a single bundled "train_step" wrapping the entire forward + loss compute. See `project_tier3_c5_design` memory + S64 Phase B memos.
- **What stays eager**: optimizer, BC anchor KL (CPU side), profiling/.item() syncs.
- **Why**: torch 2.2.x dynamo + AOTAutograd had subtle issues with in-place ops (`scatter_`) and Python-side dynamic shapes. The compile boundary was tuned in S64 to maximize wall-time savings without breaking autograd.
- **Current canonical**: NOT using `--compile` (S62 refuted it as 8% slower at prod scale). See `project_s62_fix2_prod_validation` memory.

---

## §9 Confirmed Dead Bytes [HIGH]

These ARE produced by features.py + stored in memmap + delivered to the model in the batch dict, but the model NEVER reads them.

| Field | Dims | Source | Status |
|---|---|---|---|
| `_SL_MOVE_COMPACT` per-Pokemon | 92 × 12 = 1104/turn | features.py:375-382 `_encode_move_compact` | DEAD — slice defined at model_transformer.py:141 but never read |
| `our_pokemon_move_cont` / `opp_pokemon_move_cont` in batch dict | 552/turn | dataset.py:317-318 unpack | DEAD — passed to model but no key lookup in `Tokenizer.forward` |
| Bench Pokemon move data (subset of above) | 460 of the 552 | encoded but bench mons never use moves | DEAD |

**Total waste**: ~1104 dims per turn × 50k turns × 4 bytes (float32) ≈ **220 MB per 50k-turn dataset memmap**, of which we only "lose" the dead portion (~82 MB). The rest is consumed by the live `_SL_TYPES`/`_SL_BOOSTS`/etc.

**Cleanup plan** (post-Phase-2, NOT urgent):
1. Remove `_encode_move_compact` calls from `features.py:_encode_pokemon`
2. Drop `our_pokemon_mcont` / `opp_pokemon_mcont` from memmap schema in `dataset.py`
3. Remove `_SL_MOVE_COMPACT` and its assertion from `model_transformer.py:141-144`
4. Bump memmap schema version + rebuild datasets (or invalidate via meta version check)
5. Verify model output unchanged

**Why not now**: mid-experiment. Would invalidate all existing memmap data + require rebuilding training datasets. Save for an infra cleanup window between Phase 2 and multi-gen.

---

## §10 Open Questions / Suspected Gaps [MEDIUM-LOW]

These didn't fully resolve during audit — flagged for future investigation:

1. **Boosts one-hot vs. floats** [LOW]: `_SL_BOOSTS` is 91 dims (7 stats × 13 buckets one-hot). Per comment at features.py:176, "ps-ppo finding: one-hot is better than normalized float." The MLP could in principle learn from 7 float dims. May be vestigial from a now-defunct comparison. Worth re-validating if storage gets tight.

2. **Switch action encoder** [MEDIUM]: `switch_cont` / `switch_ids` fed via separate `SwitchActionEncoder` pathway (outside `Tokenizer` scope — handled at `TransformerBattlePolicy.forward` level). Audit didn't fully trace this. Likely live based on action-space depending on it. Worth a second pass.

3. **n_summary_tokens=0 in run config** [LOW]: Inspection of running anneal `config.json` showed `"n_summary_tokens": 0` despite K=4 documented. Possible defaults differ from training. Verify cfg pathway: maybe summary tokens are produced internally rather than configured externally.

4. **`damage_chart` lookup** [MEDIUM]: built into the lookup `.pt` but only one consumer (_encode_opp_threat). Possibly redundant with the in-features type one-hot encoding the model could learn type effectiveness from. Verify whether ablation actually hurts.

5. **Some `_encode_pokemon` continuous fields might be zero for bench mons but still encoded**: volatile, paradox, combat — zero-padding wastes ~50 dims × 5 bench mons × 2 sides = 500 dims per turn. Could conditionally skip for bench. Minor optimization.

---

## §11 How to verify a slice is live (recipe for future audits)

1. Find the slice definition: `grep -n "_SL_FOO" model_transformer.py` (or `_FL_FOO` for field)
2. Find all reads: `grep -n "cont\[.*_SL_FOO" model_transformer.py` (or `field_cont[.*_FL_FOO`)
3. If only the definition + structural assertion appears (no `cont[..., _SL_FOO[0]:_SL_FOO[1]]` in any forward/encoder function), it's DEAD.
4. Cross-check `dataset.py` to confirm the field IS being unpacked into the batch dict (so we know it's reaching the model boundary).
5. Cross-check `features.py` to confirm the encoding code is running (some may be conditional).

**This recipe caught `_SL_MOVE_COMPACT` in 30 seconds.** Apply during future code reviews.

---

## §12 Cross-references

- **Design intent**: `docs/REWRITE_DESIGN.md` (110KB, Session 46 design)
- **Trainer-side dispatch**: `docs/ARCH_AUDIT.md` (Session 49/50 PPO/BC arch-discrimination)
- **Phase 2 design** (uses this model): `memory/project_phase2_launch_plan.md`
- **Inflection point checkpoint** (S67 era): `memory/project_iter89_inflection_point.md`
- **Move/type analysis tools**: `memory/reference_analyze_eval_tool.md`

---

## Maintenance

When code changes in any of these files, update this doc:
- `features.py` (encoding) — §1, §2, §3, §4, §9
- `model_transformer.py` (consumption) — all sections
- `dataset.py` (memmap schema) — §1, §9
- Build scripts for lookup tables (`build_move_flag_lookup`, `parse_showdown_*`) — §5, §6
- `LOOKUP_SCHEMA_VERSION` bumps — §5

Quick verification recipe in §11. The Explore agent prompt that produced this audit is preserved in the git history of this commit's PR description (or in the S67 session memory).
