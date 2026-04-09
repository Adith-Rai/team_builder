# V8 Architecture Overhaul — Complete Implementation Spec

## Why V8?

V7 proved the training pipeline works (BC -> self-play PPO with population diversity) but the model
plateaus at 20-23% vs smart bots due to architectural limitations. Comparison with ps-ppo (>1900 Elo
on RTX 3090) and Metamon (top 10%) revealed critical gaps:

1. **Flat vector -> MLP bottleneck** crushes entity relationships before the transformer sees them
2. **No temporal feedback** -- model doesn't know what it did last turn or what happened
3. **Scalar value head** gives weak gradient signal vs distributional (used by both ps-ppo and Metamon)
4. **Limited volatile tracking** -- only 17 of 238 poke-env effects tracked
5. **Pre-computed features** (48 dims of damage estimates, matchup scores) are noisy and prevent the
   model from learning the underlying mechanics through entity-level attention

None of the successful approaches inject type effectiveness or damage estimates -- they learn these
from entity embeddings + entity-level attention. The architecture enables this, not the features.

## Reference Implementations

- **ps-ppo** (github.com/Nebraskinator/ps-ppo): >1900 Elo, RTX 3090, 2 days. PokeTransformer with
  entity tokens, Poke-Mask, distributional value, numerical embedding banks. Our primary reference.
- **Metamon** (github.com/UT-Austin-RPL/metamon): Top 10%, text + numerical obs, two-hot value
  classification, AMAGO framework. 142M params, 22M trajectories.
- **VGC-Bench** (github.com/cameronangliss/vgc-bench): Dual transformer (spatial + temporal),
  entity embeddings, PPO + population methods.

## What Carries Over from V7

- **Training pipeline**: BC -> self-play PPO (validated)
- **Self-play infrastructure**: snapshot pool, hall of fame, uniform historical sampling, lineage tracking
- **Battle infrastructure**: battle_server.js, poke-env 0.10.0, teams_ou.py (70 teams)
- **Eval infrastructure**: eval_bc_vs_bots.py, eval_head_to_head.py, analyze_eval.py
- **Bots**: 9 rule-based opponents (policy_rulebots.py, policy_smartbots.py)
- **Human data**: human_v3_memmap (10.1M records, compressed to 2.2 GB tar.gz)
- **Reward shaping**: ko_delta + hp_delta + terminal (rewards.py)
- **observer.py**, **direct_player.py**, **replay_parser.py** -- need adaptation for new features
- **Vocab**: vocab.py with species/move/item/ability ID mappings
- **v7 backup**: `backups/v7_source_backup/` (716 MB)

## What Changes

### Files to create (new):
- **features_v8.py** -- structured per-entity feature output (replaces flat vector)
- **policy_heads_v8.py** -- PokeTransformer architecture (replaces MLP+transformer)

### Files to modify:
- **bc_train.py** -- new collation and dataset class for structured features
- **bc_policy_player.py** -- new inference with temporal summary buffer
- **rl_train.py** -- distributional value loss, new feature format, temporal handling
- **observer.py** -- emit new feature format (call features_v8 instead of features)
- **convert_jsonl_to_memmap.py** -- new structured memmap format
- **replay_parser.py** -- adapt for new feature format

### Files unchanged:
- battle_server.js, battle_worker.js, direct_player.py
- policy_rulebots.py, policy_smartbots.py, policy_random.py
- rewards.py, teams_ou.py, env_wrapper.py, vocab.py
- eval_bc_vs_bots.py, eval_head_to_head.py, analyze_eval.py (minor arg changes only)

---

## Architecture

### Overview

```
Battle State (poke-env Battle object)
    |
    v  features_v8.py
    |
    +-- 6x our_pokemon dicts --> PokemonNet --> 6 tokens (384-dim each)
    +-- 6x opp_pokemon dicts --> PokemonNet --> 6 tokens
    +-- field dict ------------> FieldNet -----> 1 token
    +-- transition dict -------> TransitionNet -> 1 token
    +-- actor (learnable) ----------------------> 1 token
    +-- critic (learnable) ---------------------> 1 token
    +-- legal_mask (9-dim)
    |
    v  Spatial Transformer (4 layers, 4 heads, d_model=384)
    |  17 tokens with Poke-Mask attention
    |
    +-- Attention pool 17 tokens --> 1 turn summary (384-dim)
    |
    v  Temporal Transformer (2 layers, 4 heads, d_model=384)
    |  Last 200 turn summaries with causal mask
    |
    +-- Actor output --> Policy head --> 9 action logits
    +-- Critic output -> Value head --> 51-bin distribution
```

### Token Structure (17 tokens)

```
[0]  Actor decision token (learnable parameter, d_model)
[1]  Critic decision token (learnable parameter, d_model)
[2]  Field token (FieldNet output)
[3]  Transition token (TransitionNet output)
[4-9]   Our 6 Pokemon tokens (PokemonNet output, active first)
[10-15] Opponent 6 Pokemon tokens (PokemonNet output, active first, unrevealed=zeros)
```

### Poke-Mask Attention

```python
def build_poke_mask(n_tokens=17):
    """State tokens can't see decision tokens. Actor can't see Critic."""
    mask = torch.zeros(n_tokens, n_tokens)
    mask[2:, 0:2] = float('-inf')  # state can't attend to actor/critic
    mask[0, 1] = float('-inf')      # actor can't see critic
    mask[1, 0] = float('-inf')      # critic can't see actor
    return mask  # additive mask for nn.MultiheadAttention
```

Forces the state representation to be "pure" -- not warped by policy/value gradients.
The battle state is encoded objectively, then actor and critic independently read from it.

### Sub-Networks

#### MoveNet (per-move encoder, output: 128-dim)

```
Input:
  move_id: int --> nn.Embedding(n_moves, 32) --> 32-dim
  bp: int 0-255 --> NumericalBank(256, 16) --> 16-dim
  accuracy: int 0-100 --> NumericalBank(101, 16) --> 16-dim
  pp: int 0-64 --> NumericalBank(65, 8) --> 8-dim
  priority: int --> NumericalBank(13, 8) --> 8-dim  (mapped from -6..+6 to 0..12)
  type_onehot: 19-dim
  category_onehot: 3-dim (physical/special/status)
  flags: ~35 binary (contact, sound, punch, bite, powder, phaze, pivot, trap,
         hazard/weather/terrain/screen setters, clears_hazards, protect, stalling,
         breaks_protect, self_destruct, fixed_damage, ignore_ability, ignore_immunity,
         recharge, disabled, stab)
  continuous: drain, recoil, heal, multihit, flinch, crit_boost, secondary chances (5)
  self_boost: 7-dim, target_boost: 7-dim
  target_onehot: 6-dim
  status_to_onehot: 7-dim

  Total raw input: ~150 dims (32 emb + 48 bank + 19 + 3 + 35 + ~15)
  --> 2-layer MLP with GELU + LayerNorm --> 128-dim output
```

#### PokemonNet (per-pokemon encoder, output: d_model=384)

```
Input:
  species_id: int --> nn.Embedding(n_species, 32) --> 32-dim
  item_id: int --> nn.Embedding(n_items, 32) --> 32-dim
  ability_id: int --> nn.Embedding(n_abilities, 32) --> 32-dim
  4x MoveNet outputs: 4 * 128 = 512-dim
  hp_pct: int 0-100 --> NumericalBank(101, 16) --> 16-dim
  base_stats: 6x int 0-255 --> NumericalBank(256, 16) --> 6 * 16 = 96-dim
  level: int 1-100 --> NumericalBank(100, 8) --> 8-dim
  boosts: 7 stats x one-hot(13) = 91-dim  (maps -6..+6 to 0..12)
  types: multi-hot 19-dim
  status: one-hot 7-dim
  is_active: 1-dim
  is_fainted: 1-dim
  weight: int 0-200 --> NumericalBank(201, 8) --> 8-dim (weight_kg/5, clamped)
  height: int 0-40 --> NumericalBank(41, 8) --> 8-dim (height_m*2, clamped)

  Active-only features (zeros for bench):
    volatile_status: 38 binary flags (expanded list, see below)
    paradox_boost: 7-dim (proto_active, quark_active, boosted_stat_5)
    tera_state: 1 + 19 = 20-dim (is_terastallized + tera_type_onehot)
    combat_state: 5-dim (first_turn, must_recharge, preparing, protect_counter, status_counter)
    toxic_fraction: 1-dim
    future_sight: 1-dim

  Opponent-only flags:
    ability_revealed: 1-dim
    item_revealed: 1-dim (UNKNOWN_ITEM sentinel check)

  Total raw input: ~920 dims (96 emb + 512 moves + 16+96+8 banks + 91+19+7+2+8+8 + ~73 active)
  --> 2-layer MLP with GELU + LayerNorm --> 384-dim output
```

#### FieldNet (field state encoder, output: d_model=384)

```
Input:
  turn: int 0-200 --> NumericalBank(201, 16) --> 16-dim
  weather: one-hot 5-dim (none/sun/rain/sand/snow) + duration NumericalBank(9, 8)
  terrain: one-hot 5-dim (none/elec/grass/psychic/misty) + duration NumericalBank(6, 8)
  trick_room: 1-dim + duration
  our_hazards: sr(1) + spikes_layers(0-3) + tspikes_layers(0-2) + web(1) = 4 dims
  opp_hazards: same 4 dims
  our_screens: reflect(1) + light_screen(1) + aurora_veil(1) + durations(3) = 6 dims
  opp_screens: same 6 dims
  tailwind: us(1) + opp(1) + durations(2) = 4 dims
  mechanics: can_tera, can_mega, can_z, can_dmax, used_tera_us/opp, used_mega_us/opp,
             used_z_us/opp, used_dmax_us/opp, dmax_turns_us/opp, trapped, opp_trapped,
             force_switch, opp_revealed_frac = 18 dims
  alive_counts: our_alive/6, opp_alive/6 = 2 dims

  Total raw input: ~100 dims
  --> 2-layer MLP with GELU + LayerNorm --> 384-dim output
```

#### TransitionNet (previous-turn events, output: d_model=384)

```
Input:
  our_last_action_kind: 3-dim one-hot (NONE/MOVE/SWITCH)
  our_last_action_id: int --> nn.Embedding (move or species) --> 32-dim
  opp_last_action_kind: 3-dim one-hot
  opp_last_action_id: int --> nn.Embedding --> 32-dim
  who_moved_first: 3-dim one-hot (us/them/unknown)

  Our move effectiveness: 6 flags (immune/barely/NVE/neutral/SE/ultra)
  Opp move effectiveness: 6 flags
  Our move crit: 1 flag
  Opp move crit: 1 flag
  Our mon flinched: 1 flag
  Opp mon flinched: 1 flag

  Status applied to us this turn: 7-dim (none/brn/par/psn/tox/slp/frz)
  Status applied to opp this turn: 7-dim
  Confusion applied to us: 1 flag
  Confusion applied to opp: 1 flag

  Stat changes this turn: 4 floats (our_boosts, our_drops, opp_boosts, opp_drops)
  KO events: 2 flags (we_kod_them, they_kod_us)
  Entry hazard damage: 2 floats (our_entry_dmg_frac, opp_entry_dmg_frac)
  Weather changed: 1 flag
  Terrain changed: 1 flag

  Total: ~57 float features + 2 entity embeddings (64 dims)
  --> 2-layer MLP with GELU + LayerNorm --> 384-dim output

  Source: battle.observations[prev_turn].events -- parse Showdown protocol tags:
    |-supereffective| (count 1=2x, 2=4x)
    |-resisted| (count 1=0.5x, 2=0.25x)
    |-immune|
    |-crit|
    |cant|...|flinch
    |-status|
    |-boost| / |-unboost|
    |faint|
    |-damage|...|[from] Stealth Rock
    |-weather| / |-fieldstart|
    |move| (action taken)
    |switch| (switch action)
```

#### NumericalBank

```python
class NumericalBank(nn.Module):
    """Learned embedding for quantized continuous values.
    Replaces raw floats which cause gradient instability (ps-ppo finding)."""
    def __init__(self, num_values: int, bank_dim: int):
        self.embedding = nn.Embedding(num_values, bank_dim)
    def forward(self, x: Tensor) -> Tensor:  # x: int tensor
        return self.embedding(x.clamp(0, self.num_values - 1))

# Banks used:
#   HP:       NumericalBank(101, 16)   -- 0-100%
#   Stats:    NumericalBank(256, 16)   -- base stats 0-255
#   BP:       NumericalBank(256, 16)   -- move base power 0-255
#   Accuracy: NumericalBank(101, 16)   -- 0-100%
#   PP:       NumericalBank(65, 8)     -- 0-64
#   Priority: NumericalBank(13, 8)     -- -6 to +6 mapped to 0-12
#   Turn:     NumericalBank(201, 16)   -- turn 0-200
#   Level:    NumericalBank(100, 8)    -- level 1-100
#   Weight:   NumericalBank(201, 8)    -- weight_kg/5 clamped 0-200
#   Height:   NumericalBank(41, 8)     -- height_m*2 clamped 0-40
#   Weather/Terrain duration: NumericalBank(9, 8) / NumericalBank(6, 8)
```

### Temporal Transformer

Two modes controlled by `--temporal-mode`:

**Summary mode (default, local GPU):**
```
1. Spatial transformer processes current turn's 17 tokens --> 17 output vectors
2. Attention pooling: learnable query attends over 17 output tokens --> 1 summary (384-dim)
3. Summary appended to history buffer (max 200 turns, oldest dropped)
4. Temporal transformer (2L, 4H, causal mask) processes all summaries
5. Last temporal output feeds into policy/value heads

Per-turn memory: 384 dims * 4 bytes = 1.5 KB
200 turns: 300 KB per battle. Negligible.
Temporal attention matrix: 200x200 = 40K elements. Negligible.
```

**Frames mode (cloud GPU, ideal):**
```
1. Spatial transformer processes current turn --> 17 output vectors
2. All 17 vectors stored in history (max N turns, configurable)
3. Temporal transformer processes N*17 tokens with causal mask
4. Decision tokens from last frame feed into policy/value heads

N=32: 544 tokens. Attention: 544x544 = ~300K per head. Fits on 24GB.
N=64: 1088 tokens. Fits on 40GB+.
```

Summary mode enables 200-turn context on 6GB. Frames mode is strictly more powerful
(can attend to individual entities from past turns) but requires cloud GPU.

### Distributional Value Head

```python
# 51-bin categorical distribution over value support
v_bins = 51
v_support = torch.linspace(-1.6, 1.6, 51)  # bin centers
value_head = nn.Linear(d_model, 51)

def get_value(critic_output):
    logits = value_head(critic_output)
    probs = F.softmax(logits, dim=-1)
    return (probs * v_support).sum(-1)  # expected value for inference

def twohot_target(value):
    """Two-hot encoding for value regression targets."""
    value = value.clamp(v_support[0], v_support[-1])
    bin_width = v_support[1] - v_support[0]
    idx = (value - v_support[0]) / bin_width
    lo = idx.floor().long()
    hi = (lo + 1).clamp(max=v_bins - 1)
    weight_hi = idx - lo.float()
    target = torch.zeros(..., v_bins)
    target.scatter_(-1, lo.unsqueeze(-1), (1 - weight_hi).unsqueeze(-1))
    target.scatter_(-1, hi.unsqueeze(-1), weight_hi.unsqueeze(-1))
    return target

# PPO value loss: F.cross_entropy(v_logits, twohot_target(returns))
# Replaces: MSE(value, returns)
```

### Expanded Volatile Status List (38 + 7 paradox)

**38 binary flags per active Pokemon:**

```python
_VOLATILE_EFFECTS = [
    # Original 17 (from v7):
    Effect.CONFUSION, Effect.ENCORE, Effect.TAUNT, Effect.DISABLE,
    Effect.SUBSTITUTE, Effect.LEECH_SEED, Effect.YAWN, Effect.CURSE,
    Effect.TORMENT, Effect.PROTECT, Effect.FOCUS_ENERGY,
    Effect.TRAPPED, Effect.PARTIALLY_TRAPPED,
    Effect.PERISH0, Effect.PERISH1, Effect.PERISH2, Effect.PERISH3,

    # NEW high-impact (14):
    Effect.MAGNET_RISE,     # Ground immunity for 5 turns
    Effect.FLASH_FIRE,      # Fire immunity + 1.5x Fire boost
    Effect.SMACK_DOWN,      # Grounded (loses Flying Ground-immunity)
    Effect.HEAL_BLOCK,      # Can't use recovery moves
    Effect.DESTINY_BOND,    # Mutual KO if we faint them
    Effect.IMPRISON,        # Opponent can't use shared moves
    Effect.GLAIVE_RUSH,     # Takes double damage next turn
    Effect.TAR_SHOT,        # Doubles Fire damage + grounded
    Effect.GASTRO_ACID,     # Ability suppressed
    Effect.NO_RETREAT,      # Can't switch + all stats boosted
    Effect.INGRAIN,         # Can't switch + heals 1/16/turn
    Effect.SALT_CURE,       # Residual damage (1/4 Water/Steel, 1/8 other)
    Effect.ENDURE,          # Survives at 1 HP this turn
    Effect.LOCKED_MOVE,     # Choice-locked or Outrage-locked

    # NEW medium-impact (7):
    Effect.AQUA_RING,       # Free 1/16 HP healing per turn
    Effect.SYRUP_BOMB,      # Speed drops over 3 turns (Gen 9)
    Effect.THROAT_CHOP,     # Can't use sound moves for 2 turns
    Effect.STOCKPILE1,      # Spit Up/Swallow scaling
    Effect.STOCKPILE2,
    Effect.STOCKPILE3,
    Effect.LASER_FOCUS,     # Next move guaranteed crit
]
N_VOLATILE = 38

# Paradox boosts (7-dim encoding: [proto_active, quark_active, boosted_stat x5])
_PARADOX_EFFECTS = {
    Effect.PROTOSYNTHESISATK: 0, Effect.PROTOSYNTHESISDEF: 1,
    Effect.PROTOSYNTHESISSPA: 2, Effect.PROTOSYNTHESISSPD: 3,
    Effect.PROTOSYNTHESISSPE: 4,
    Effect.QUARKDRIVEATK: 0, Effect.QUARKDRIVEDEF: 1,
    Effect.QUARKDRIVESPA: 2, Effect.QUARKDRIVESPD: 3,
    Effect.QUARKDRIVESPE: 4,
}
```

**Not tracked (ends-on-turn, not decision-relevant by our next action):**
FLINCH, ROOST, QUICK_GUARD, WIDE_GUARD, MAT_BLOCK, BANEFUL_BUNKER,
KINGS_SHIELD, SILK_TRAP, BURNING_BULWARK, OBSTRUCT, SPIKY_SHIELD
(All end before our next decision. Flinch captured in transition token instead.)

---

## Feature Audit: What Was Removed from V7

### Removed (91 dims of pre-computed features)

| Feature | Dims | Why removed |
|---------|------|-------------|
| `_compute_v4_features()` | 48 | Pre-computed matchup scores, damage estimates, KO checks, bench analysis. Uses incomplete hand-rolled type chart. ps-ppo proves entity attention learns this. Noisy estimates create shortcuts. |
| `matchup_scalars` | 3 | `_has_revealed_se_now`, `_opp_has_revealed_se_into_us`, `_count_revealed_se_in_party`. Uses incomplete type chart. Entity attention replaces this. |
| `move_type_histograms` | 38 | Revealed move type presence per active. Redundant with per-pokemon move embeddings via MoveNet. v1 legacy. |
| `_type_effectiveness()` chart | -- | ~100 lines of hardcoded type matchups. Incomplete (no abilities, no tera). Only used by removed features. poke-env's `Pokemon.damage_multiplier()` is correct. |

### Removed (ctx_extra as separate concept)

| Feature | Dims | Why removed |
|---------|------|-------------|
| `derive_ctx_extra_live()` | 26 | Exact duplicate of hazard/screen/weather/terrain bits in `_board_bits()`. Absorbed into field token. |
| `encode_opp_last_ctx()` | 15 | Opponent last action payload. Absorbed into transition token. |
| ctx_extra total | 41 | No longer a separate projection pathway. |

### Removed (duplicates)

- `moved_first_bits` (3 dims in obs) + speed tier (3 dims in v4_computed) -- both encode "who goes first". In v8: speed derivable from entity tokens, who moved first goes in transition token.
- Opp last action encoded in 3 places (ctx_extra + entity_ids[80-81] + revealed moves). Consolidated into transition token.

### Kept (all per-entity raw state features)

All "what IS" features are kept: types, HP, status, stats, boosts, volatiles, tera, weight/height,
combat state, move features (107-dim slots), switch features (28-dim slots), board state, mechanics.
These are raw observations, not pre-computed interpretations.

---

## Architecture Scaling (CLI flags)

All architecture params are CLI flags with defaults for local GPU:

```
--d-model 384              # hidden dimension (scale: 512, 768, 1024)
--n-spatial-layers 4       # spatial transformer layers (scale: 6, 8)
--n-temporal-layers 2      # temporal transformer layers (scale: 4, 6)
--n-heads 4                # attention heads (scale: 8, 16)
--temporal-context 200     # turns of history (no hard clamp on episode length)
--temporal-mode summary    # or: frames (for cloud GPU)
--v-bins 51                # distributional value bins
--move-dim 128             # MoveNet output dimension
--bank-dim 16              # NumericalBank embedding dimension
--ff-mult 4                # feedforward expansion multiplier
```

### Scaling Table

| Config | d_model | Spatial | Temporal | Heads | Temporal Mode | Params | VRAM (PPO) | Target |
|--------|---------|---------|----------|-------|---------------|--------|------------|--------|
| Local (RTX 3060 6GB) | 384 | 4L | 2L | 4 | summary, 200 turns | ~12-15M | ~2.5-3.5 GB | Development, proof of concept |
| Mid (RTX 3090 24GB) | 512 | 6L | 3L | 8 | frames, 32 turns | ~30-40M | ~10-14 GB | Serious training |
| Cloud (A100 40GB) | 768 | 6L | 4L | 8 | frames, 64 turns | ~60-80M | ~25-35 GB | Production |
| Cloud Large (A100 80GB) | 1024 | 8L | 6L | 16 | frames, 100 turns | ~120-150M | ~50-65 GB | SOTA attempt |

---

## Data Pipeline

### Structured Memmap Format (v8)

Each memmap directory contains:

```
metadata.json              -- version, dims, vocab sizes, feature descriptions
episode_index.npy          -- (E, 3) int64: [start_idx, length, hash]

# Per-Pokemon features (our team)
our_pokemon_ids.npy        -- (N, 6, 4) int32: [species_id, item_id, ability_id, level]
our_pokemon_move_ids.npy   -- (N, 6, 4) int32: move IDs per mon
our_pokemon_cont.npy       -- (N, 6, D_poke_cont) float32: HP, stats, types, status, boosts, etc.
our_pokemon_volatile.npy   -- (N, 6, 45) float32: 38 volatile + 7 paradox (active only, bench=0)

# Per-Pokemon features (opponent team)
opp_pokemon_ids.npy        -- (N, 6, 4) int32
opp_pokemon_move_ids.npy   -- (N, 6, 4) int32
opp_pokemon_cont.npy       -- (N, 6, D_poke_cont) float32
opp_pokemon_volatile.npy   -- (N, 6, 45) float32

# Per-move features (active moves for action selection)
active_move_ids.npy        -- (N, 4) int32: current available move IDs
active_move_cont.npy       -- (N, 4, D_move_cont) float32: 107-dim slot features

# Switch slot features
switch_ids.npy             -- (N, 5) int32: available switch species IDs
switch_cont.npy            -- (N, 5, 28) float32: types + hp + status + weight

# Field state
field.npy                  -- (N, D_field) float32

# Transition features
transition_ids.npy         -- (N, 2) int32: [our_last_action_id, opp_last_action_id]
transition_cont.npy        -- (N, D_trans) float32: ~57 dims

# Action / result
action.npy                 -- (N,) int32: 0-8
legal.npy                  -- (N, 9) float32
result.npy                 -- (N,) float32: 1.0/0.0/0.5
```

### Data Generation Strategy

1. **Bot data (proof of concept)**: Run observer.py with updated features_v8, generate ~500K records
   with 9 bots. ~2 hours. Used for initial BC training to validate architecture.

2. **Human data (background)**: Update replay_parser.py for new features, re-scrape/convert
   human_v3_memmap data. Run overnight. Higher quality data for BC.

3. **Self-play PPO**: Features generated live during battles. No data pipeline needed.

---

## Robustness Measures

### Dimension Safety
- Every features_v8 output dict has shape assertions against named constants
- `_clamp_int(val, lo, hi)` helper used everywhere NumericalBanks are fed
- metadata.json sidecar in memmap dirs records all dimensions for validation

### Checkpoint Safety
- `PolicyConfig` dict saved in every checkpoint with all dims and architecture params
- `checkpoint["v8_version"] = "8.0"` -- increment on any breaking change
- `load_state_dict(strict=True)` by default -- shape mismatch = loud error
- No silent weight pruning (v7's `_prune_state_dict_to_model` removed)

### Event Parsing Safety
- All transition feature event parsing wrapped in try/except with graceful fallback
- Unknown events produce zero features, never crash
- Unit tests for every event type against real battle logs

### VRAM Safety
- Gradient checkpointing via `--gradient-checkpoint` flag (halves VRAM, ~30% slower)
- Dynamic temporal truncation: temporal context is a sliding window, not a hard episode clamp
- Episode length is NOT clamped -- battles run to natural completion. Only the temporal
  attention window is limited (200 turns default, oldest summaries dropped).

### Training Safety
- Validation script reads N random memmap samples, verifies shapes, value ranges, no NaN/inf
- poke-env native API used everywhere (Pokemon.damage_multiplier, battle.observations, etc.)
- No hand-rolled type charts or damage calculations

---

## Implementation Order

### Phase A: Architecture (this session)
1. Write `features_v8.py` -- structured per-entity output + transition features
2. Write `policy_heads_v8.py` -- PokeTransformer + temporal + distributional value
3. Smoke test: forward pass with random data, verify shapes, check VRAM

### Phase B: Training Pipeline
4. Update `convert_jsonl_to_memmap.py` for new structured memmap format
5. Update `bc_train.py` -- new collation, new dataset class
6. Update `observer.py` -- emit new feature format
7. Generate bot-data proof of concept (~500K records)
8. Train BC (5-10 epochs), eval vs bots

### Phase C: Human Data (can run in parallel with Phase B)
9. Update `replay_parser.py` for new features
10. Re-scrape/convert human replay data

### Phase D: RL + Self-Play
11. Update `rl_train.py` -- distributional value loss, new feature format
12. Update `bc_policy_player.py` -- new inference with temporal buffer
13. BC -> self-play PPO with population diversity

### Phase E: Scaling (future)
14. Cloud training with larger model (d_model=512+, frames mode)
15. Search at inference time (MIT thesis approach)
16. Multi-format training (Metamon approach)

### Phase F: Multi-Format / Multi-Gen (future)

**What's reusable across ALL formats:**
- PokeTransformer (spatial + temporal) — attention over N tokens works for any N
- PokemonNet, MoveNet, FieldNet, TransitionNet, NumericalBanks — format-agnostic
- Entity embeddings (species/moves/items/abilities) — vocab covers all gens
- PPO training loop, self-play infrastructure, reward shaping
- Temporal context (summaries work regardless of format)

**Gen 6-9 Singles (LOW effort):**
- Same action space (4 moves + 5 switches = 9)
- Feature changes: remove tera (gen 6-8), add mega (gen 6-7), Z-moves (gen 7), dynamax (gen 8)
  Already have flags for all of these in features_v8.py
- Need: gen-specific human replays for BC, gen-specific teams
- Optional: gen embedding token so one model handles all gens (Metamon approach, 16 formats)

**Gen 9 other tiers — UU, Ubers, AG, Randoms (TRIVIAL):**
- No code changes. Different teams + replays only.

**3v3 Singles (TRIVIAL):**
- Empty bench slots auto-pad to zero. 3 pokemon tokens instead of 6, rest are zeros.
- Action space unchanged (still 4 moves + up to 2 switches)

**Doubles / VGC (HIGH effort — separate project):**
- Action space: joint action for 2 active mons. Options:
  A) Flat: (mon1_action × mon2_action) = ~81 options (9×9) — too large, most illegal
  B) Hierarchical: pick mon1 action (9), then mon2 action (9) = 2 sequential decisions
  C) Two separate heads: actor_1 and actor_2 tokens in spatial transformer
- Move targeting: some moves need target selection (which opponent). Extra action dimension.
- New features: partner Pokemon state, spread move flags, Follow Me/Helping Hand, positioning
- Entity tokens: 2+2 active + 4+4 bench = 12 pokemon (same count, different active/bench split)
- Everything else reusable: transformers, embeddings, banks, temporal, PPO

**VGC specifically:** Requires team preview decision (pick 4 from 6 = C(6,4)=15 options)
before battle starts. Separate decision head or model. VGC-Bench handles this.
Should be done AFTER team building phase (Phase 7 in master plan) is complete.

**Triples (VERY HIGH effort, low priority):**
- 3 active per side, positioning (left/center/right) affects targeting
- Very niche format, few players/replays available

### Phase G: Team Builder (future — requires battler at 1600+ Elo)

**Goal:** Given 0-6 user-selected Pokemon, build a complete competitive team with optimal
species, items, abilities, natures, moves, and EV spreads for the target format.

**Architecture — Two Models:**

**Model 1: Team Composer (new)**
- Autoregressive transformer that picks Pokemon + sets one at a time
- Input: format rules, user constraints (locked Pokemon slots), metagame context
- Output per slot: species_id, item_id, ability_id, nature_id, 4×move_id, 6×EV_value
- Autoregressive because each pick depends on previous (need type coverage, synergy,
  role balance). Pick Pokemon 1 → conditioned on 1, pick Pokemon 2 → etc.
- Legal constraints enforced: learnset validation (can this Pokemon learn this move?),
  item clause (no duplicate items), species clause, format banlists
- EV generation: constrained to 508 total, 252 max per stat. Could use a small MLP
  that outputs 6 EV values normalized to sum=508.

**Model 2: Battle Evaluator (existing PokeTransformer)**
- Plays N games (e.g., 50) with the generated team against diverse opponents
- Returns win rate as the reward signal for Team Composer
- Must be strong enough (1600+ Elo) to accurately judge team quality
- Weak battler = noisy reward = Composer learns bad teams

**Training Loop:**
```
1. User provides constraints (0-6 locked Pokemon with optional locked sets)
2. Team Composer fills remaining slots autoregressively
3. Battle Evaluator plays 50 games with the generated team
4. Win rate → reward for Team Composer
5. Team Composer updates via REINFORCE/PPO on the reward
6. Repeat for thousands of team generations
```

**User Constraint Handling:**
- 6 "slots" in the input, each marked LOCKED or OPEN
- LOCKED slots: species/item/ability/moves/EVs are fixed input features
- OPEN slots: model generates these autoregressively
- Supports all use cases:
  - 0 locked: "Build me the best team for OU" → model picks all 6
  - 1 locked: "Build around Garchomp" → model picks 5 teammates + all sets
  - 3 locked: "I want Garchomp, Corviknight, Heatran" → model picks 3 + optimizes sets
  - 6 locked: "Optimize this team's sets" → model picks best moves/EVs/items/abilities
  - Partial locks: "Garchomp with Swords Dance + Earthquake" → model fills remaining moves + EVs

**What We Already Have:**
- `raw_data/` (27 GB): Smogon usage stats 2014-2024, move/learnset data, Pokemon stats
- Entity embeddings from battler: species/move/item/ability already have learned representations
- vocab.py: complete ID mappings for all entities
- Battle infrastructure: poke-env, battle_server.js, eval pipeline
- Strong battler (in progress): evaluates team quality through actual games

**What We'd Need to Build:**
- Team Composer model (autoregressive transformer, ~5-10M params)
- Legal constraint validator (learnset, item clause, format banlists — from poke-env GenData)
- EV optimization head (constrained generation)
- Reward pipeline: team → battle eval → win rate → gradient
- User interface for constraint specification

**Chicken-and-Egg Solution:**
1. First: train battler to 1600+ Elo on human teams (current phase)
2. Then: train Composer using battler as evaluator (Composer generates, battler judges)
3. Optional: jointly fine-tune (battler adapts to Composer teams, Composer to better battler)
4. Bootstrap: seed Composer with Smogon usage data (most popular sets as starting point)

**Research Context:**
- Metamon used ~1K procedurally generated "Variety Teams" but no learned team builder
- No published work on learned team building for competitive Pokemon
- This would be novel research if successful

**Key Challenge: Search Space**
~1500 species × ~300 abilities × ~2000 items × C(~950,4) moves × EV spreads × 25 natures
= astronomically large per Pokemon, times 6 Pokemon per team.
Autoregressive generation + RL from battle outcomes is the most tractable approach.
Alternative: evolutionary search (generate many teams, battle-test, keep best, mutate)
could work as a simpler baseline before the full learned Composer.

---

## Success Criteria

- **Phase A**: Forward pass works, VRAM fits, shapes correct
- **Phase B**: BC trains, eval vs bots shows sane behavior
- **Phase C**: Human data pipeline produces valid memmap
- **Phase D**: Smart avg breaks 30% (vs 20-23% v7 ceiling)
- **Phase D+**: Smart avg >40% with extended self-play
- **Ultimate**: >50% vs smart bots, competitive on Pokemon Showdown ladder

---

## Hyperparameter Changes from V7

Based on ps-ppo's validated config and our findings:

```
gamma: 0.99 -> 0.9999     (value winning regardless of game length)
lambda: 0.95 -> 0.8       (less variance with better value head)
entropy_coef: 0.01 -> 0.02 (more exploration with larger action understanding)
lr: 1e-4                   (same, validated for transformers at bs=32)
clip_eps: 0.2              (same)
ppo_epochs: 4              (same)
batch_size: 32             (same, may increase with gradient accumulation)
```

## Key Design Decisions Log

1. **Summary temporal over frame stacking**: 200-turn context on 6GB vs 8-32 frames on 24GB+.
   Summary compresses 17 tokens to 1 per turn via attention pooling. Ideal is frames mode on cloud.
2. **Remove pre-computed features**: ps-ppo proves entity attention learns matchups, damage, speed.
   Our pre-computed features used incomplete type chart and created shortcuts.
3. **38 volatile effects**: Expanded from 17 to cover all strategically relevant Gen 9 effects.
   Flinch goes in transition token (ends-on-turn).
4. **Boosts as one-hot(13)**: ps-ppo finding -- better than normalized floats for discrete values.
5. **No episode length clamp**: Battles run to completion. Temporal window is 200 turns (sliding).
6. **Clean break from v7**: No backward compat with v7 checkpoints. Simpler code.
7. **All poke-env native API**: No hand-rolled type charts, damage calcs, or protocol parsing.
   Use Pokemon.damage_multiplier, battle.observations, Move attributes, Effect enum.
