# src/features_v8.py
# v8 structured feature extractor for PokeTransformer.
#
# Outputs per-entity feature dicts (not flat vectors) for:
#   - 6 our Pokemon tokens + 6 opponent Pokemon tokens
#   - 1 field token
#   - 1 transition token (previous-turn events)
#   - Legal mask + active move/switch slot features
#
# All continuous values intended for NumericalBanks are output as clamped ints.
# All categorical IDs use vocab.py integer lookups.
# All data sourced from poke-env 0.10.0 native API (no hand-rolled type charts).
#
# Public API:
#   make_features(battle, prev_events=None) -> dict
#
# The returned dict has keys:
#   "our_pokemon": list of 6 dicts (active first, then bench sorted by species)
#   "opp_pokemon": list of 6 dicts (active first, then revealed bench, unrevealed=empty)
#   "field": dict
#   "transition": dict
#   "legal_mask": np.ndarray (9,) float32
#   "active_moves": list of 4 dicts (move slot features for action selection)
#   "switch_slots": list of 5 dicts (switch target features)

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from poke_env.battle import SideCondition, Weather, Field, Status, MoveCategory, Target
from poke_env.battle.effect import Effect

# =============================
# Constants
# =============================

from format_config import FormatConfig, FORMAT_SINGLES

N_TYPES = FORMAT_SINGLES.n_types
MAX_BENCH = FORMAT_SINGLES.n_bench
MAX_MOVES = FORMAT_SINGLES.n_moves
N_STATS = FORMAT_SINGLES.n_stats
_STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]
_BOOST_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

# v8 expanded volatile effects (38 total)
_VOLATILE_EFFECTS = [
    # Original 17 (from v7):
    Effect.CONFUSION, Effect.ENCORE, Effect.TAUNT, Effect.DISABLE,
    Effect.SUBSTITUTE, Effect.LEECH_SEED, Effect.YAWN, Effect.CURSE,
    Effect.TORMENT, Effect.PROTECT, Effect.FOCUS_ENERGY,
    Effect.TRAPPED, Effect.PARTIALLY_TRAPPED,
    Effect.PERISH0, Effect.PERISH1, Effect.PERISH2, Effect.PERISH3,
    # NEW high-impact (14):
    Effect.MAGNET_RISE, Effect.FLASH_FIRE, Effect.SMACK_DOWN,
    Effect.HEAL_BLOCK, Effect.DESTINY_BOND, Effect.IMPRISON,
    Effect.GLAIVE_RUSH, Effect.TAR_SHOT, Effect.GASTRO_ACID,
    Effect.NO_RETREAT, Effect.INGRAIN, Effect.SALT_CURE,
    Effect.ENDURE, Effect.LOCKED_MOVE,
    # NEW medium-impact (7):
    Effect.AQUA_RING, Effect.SYRUP_BOMB, Effect.THROAT_CHOP,
    Effect.STOCKPILE1, Effect.STOCKPILE2, Effect.STOCKPILE3,
    Effect.LASER_FOCUS,
]
N_VOLATILE = len(_VOLATILE_EFFECTS)  # 38
assert N_VOLATILE == 38, f"Expected 38 volatiles, got {N_VOLATILE}"

# Paradox boost effects (Proto/Quark per stat)
_PARADOX_PROTO = {
    Effect.PROTOSYNTHESISATK: 0, Effect.PROTOSYNTHESISDEF: 1,
    Effect.PROTOSYNTHESISSPA: 2, Effect.PROTOSYNTHESISSPD: 3,
    Effect.PROTOSYNTHESISSPE: 4,
}
_PARADOX_QUARK = {
    Effect.QUARKDRIVEATK: 0, Effect.QUARKDRIVEDEF: 1,
    Effect.QUARKDRIVESPA: 2, Effect.QUARKDRIVESPD: 3,
    Effect.QUARKDRIVESPE: 4,
}
N_PARADOX = 7  # [proto_active, quark_active, boosted_stat x5]

_TYPES = [
    "NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE", "FIGHTING",
    "POISON", "GROUND", "FLYING", "PSYCHIC", "BUG", "ROCK", "GHOST",
    "DRAGON", "DARK", "STEEL", "FAIRY", "???"
]
_TYPE_TO_IDX = {t: i for i, t in enumerate(_TYPES)}

_STATUS_LIST = ["NONE", "BRN", "PAR", "PSN", "TOX", "SLP", "FRZ"]
_STATUS_TO_IDX = {s: i for i, s in enumerate(_STATUS_LIST)}
N_STATUS = len(_STATUS_LIST)  # 7

# Move target one-hot mapping (enum keys, module-level constant)
_TARGET_MAP = {
    Target.SELF: 0, Target.NORMAL: 1, Target.ALL_ADJACENT: 2,
    Target.ALL_ADJACENT_FOES: 2, Target.FOE_SIDE: 3, Target.ALLY_SIDE: 4,
}

# Self-targeting target enums (for boost routing)
_SELF_TARGETS = (Target.SELF, Target.ADJACENT_ALLY_OR_SELF, Target.ALLY_SIDE, Target.ALLY_TEAM, Target.ALLIES)

# poke-env UNKNOWN_ITEM sentinel
_UNKNOWN_ITEM = "unknown_item"

# =============================
# Vocab (lazy-loaded)
# =============================

_vocab = None

def _get_vocab():
    global _vocab
    if _vocab is None:
        from vocab import Vocab
        _vocab = Vocab.load()
    return _vocab

# =============================
# Small helpers
# =============================

def _clamp_int(val, lo: int, hi: int) -> int:
    """Clamp a value to [lo, hi] as int. Safe for None/NaN."""
    try:
        return max(lo, min(hi, int(val)))
    except (TypeError, ValueError):
        return lo


def _clamp_float(val, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float to [lo, hi]. Safe for None."""
    try:
        return max(lo, min(hi, float(val)))
    except (TypeError, ValueError):
        return lo


def _hp_pct_int(poke) -> int:
    """HP as integer 0-100 for NumericalBank."""
    try:
        frac = poke.current_hp_fraction
        if frac is None:
            return 0
        return _clamp_int(frac * 100, 0, 100)
    except Exception:
        return 0


def _types_multihot(types) -> List[float]:
    """19-dim multi-hot type encoding."""
    v = [0.0] * N_TYPES
    for t in (types or []):
        name = t.name if hasattr(t, "name") else str(t).upper()
        idx = _TYPE_TO_IDX.get(name)
        if idx is not None:
            v[idx] = 1.0
    return v


def _status_onehot(status) -> List[float]:
    """7-dim one-hot status encoding."""
    v = [0.0] * N_STATUS
    if status is None:
        v[0] = 1.0
        return v
    name = status.name if hasattr(status, "name") else str(status).upper()
    if name.startswith("TOX"):
        name = "TOX"
    elif name.startswith("PSN"):
        name = "PSN"
    idx = _STATUS_TO_IDX.get(name, 0)
    v[idx] = 1.0
    return v


def _boosts_onehot13(poke) -> List[float]:
    """7 stats x 13-dim one-hot (maps -6..+6 to indices 0..12). Total: 91 dims.
    ps-ppo finding: one-hot is better than normalized float for discrete boost values."""
    result = []
    boosts = getattr(poke, "boosts", {}) or {}
    for k in _BOOST_KEYS:
        val = _clamp_int(boosts.get(k, 0), -6, 6)
        oh = [0.0] * 13
        oh[val + 6] = 1.0
        result.extend(oh)
    return result


def _volatile_bits(poke) -> List[float]:
    """38-dim binary flags for expanded volatile effects."""
    bits = [0.0] * N_VOLATILE
    if poke is None:
        return bits
    try:
        effects = poke.effects
        if not effects:
            return bits
        for i, eff in enumerate(_VOLATILE_EFFECTS):
            if eff in effects:
                bits[i] = 1.0
    except Exception:
        pass
    return bits


def _paradox_encoding(poke) -> List[float]:
    """7-dim: [proto_active, quark_active, boosted_stat x5]."""
    vec = [0.0] * N_PARADOX
    if poke is None:
        return vec
    try:
        effects = poke.effects or {}
        for eff, stat_idx in _PARADOX_PROTO.items():
            if eff in effects:
                vec[0] = 1.0  # proto active
                vec[2 + stat_idx] = 1.0
        for eff, stat_idx in _PARADOX_QUARK.items():
            if eff in effects:
                vec[1] = 1.0  # quark active
                vec[2 + stat_idx] = 1.0
    except Exception:
        pass
    return vec


def _tera_encoding(poke) -> List[float]:
    """20-dim: [is_terastallized] + tera_type one-hot (19)."""
    vec = [0.0] * (1 + N_TYPES)
    if poke is None:
        return vec
    try:
        if poke.is_terastallized:
            vec[0] = 1.0
            tt = poke.tera_type
            if tt is not None:
                name = tt.name if hasattr(tt, "name") else str(tt).upper()
                idx = _TYPE_TO_IDX.get(name)
                if idx is not None:
                    vec[1 + idx] = 1.0
    except Exception:
        pass
    return vec


def _combat_state(poke) -> List[float]:
    """5-dim: first_turn, must_recharge, preparing, protect_counter/4, status_counter/16."""
    if poke is None:
        return [0.0] * 5
    try:
        return [
            1.0 if getattr(poke, "first_turn", False) else 0.0,
            1.0 if getattr(poke, "must_recharge", False) else 0.0,
            1.0 if getattr(poke, "preparing", False) else 0.0,
            _clamp_float(float(getattr(poke, "protect_counter", 0) or 0) / 4.0),
            _clamp_float(float(getattr(poke, "status_counter", 0) or 0) / 16.0),
        ]
    except Exception:
        return [0.0] * 5


def _toxic_fraction(poke) -> float:
    """Toxic escalation: status_counter/16 when TOX, else 0."""
    try:
        if poke and poke.status == Status.TOX:
            return _clamp_float(float(getattr(poke, "status_counter", 0) or 0) / 16.0)
    except Exception:
        pass
    return 0.0


def _future_sight_bit(poke) -> float:
    """1.0 if Future Sight pending."""
    try:
        if poke and poke.effects and Effect.FUTURE_SIGHT in poke.effects:
            return 1.0
    except Exception:
        pass
    return 0.0


# =============================
# Per-Pokemon feature extraction
# =============================

def _encode_pokemon(poke, is_active: bool, is_opponent: bool) -> dict:
    """Extract structured features for one Pokemon token.

    Returns dict with:
      ids: dict of int entity IDs for embeddings
      banks: dict of int values for NumericalBanks
      continuous: list of float features (concatenated in model)
    """
    v = _get_vocab()

    if poke is None:
        # Unrevealed opponent slot or empty
        return {
            "ids": {"species": 0, "item": 0, "ability": 0,
                    "move0": 0, "move1": 0, "move2": 0, "move3": 0},
            "banks": {"hp_pct": 0, "level": 0, "weight": 0, "height": 0,
                      "stat_hp": 0, "stat_atk": 0, "stat_def": 0,
                      "stat_spa": 0, "stat_spd": 0, "stat_spe": 0},
            "continuous": [0.0] * _pokemon_cont_dim(),
            "is_empty": True,
        }

    # --- Entity IDs ---
    species_id = v.species(poke.species)
    item_raw = getattr(poke, "item", None)
    item_known = item_raw is not None and item_raw != _UNKNOWN_ITEM
    item_id = v.item(item_raw) if item_known else 0
    ability_raw = poke.ability or (
        poke.possible_abilities[0] if poke.possible_abilities else None)
    ability_known = poke.ability is not None
    ability_id = v.ability(ability_raw) if ability_known or not is_opponent else 0

    # Move IDs (up to 4)
    moves_list = list((poke.moves or {}).keys())
    move_ids = [v.move(moves_list[i]) if i < len(moves_list) else 0
                for i in range(MAX_MOVES)]

    # --- NumericalBank values (clamped ints) ---
    hp_pct = _hp_pct_int(poke)
    level = _clamp_int(getattr(poke, "level", 100), 1, 100)

    # Weight in kg / 5, clamped 0-200 (covers 0.1 to 999.9 kg)
    weight_raw = getattr(poke, "weight", 0) or 0
    weight_int = _clamp_int(float(weight_raw) / 5.0, 0, 200)

    # Height in m * 2, clamped 0-40 (covers 0.1 to 20.0 m)
    height_raw = getattr(poke, "height", 0) or 0
    height_int = _clamp_int(float(height_raw) * 2.0, 0, 40)

    # Stats: use actual stats for own mons, base_stats for opponents
    if not is_opponent and isinstance(getattr(poke, "stats", None), dict):
        stats_dict = poke.stats
    else:
        stats_dict = poke.base_stats if poke.base_stats else {}
    stat_ints = {f"stat_{k}": _clamp_int(stats_dict.get(k, 0) or 0, 0, 255)
                 for k in _STAT_KEYS}

    # --- Continuous features ---
    cont = []

    # Types (19 multi-hot)
    cont.extend(_types_multihot(poke.types))

    # Status (7 one-hot)
    cont.extend(_status_onehot(poke.status))

    # Boosts (7 x 13 = 91 one-hot)
    cont.extend(_boosts_onehot13(poke))

    # Active/fainted flags
    cont.append(1.0 if is_active else 0.0)
    cont.append(1.0 if poke.fainted else 0.0)

    # Active-only features (zeros for bench)
    if is_active:
        cont.extend(_volatile_bits(poke))       # 38
        cont.extend(_paradox_encoding(poke))     # 7
        cont.extend(_tera_encoding(poke))        # 20
        cont.extend(_combat_state(poke))         # 5
        cont.append(_toxic_fraction(poke))       # 1
        cont.append(_future_sight_bit(poke))     # 1
    else:
        cont.extend([0.0] * (N_VOLATILE + N_PARADOX + 20 + 5 + 1 + 1))

    # Opponent visibility flags
    if is_opponent:
        cont.append(1.0 if ability_known else 0.0)
        cont.append(1.0 if item_known else 0.0)
    else:
        cont.append(1.0)  # own ability always known
        cont.append(1.0)  # own item always known

    # Move compact encodings (4 x 23 = 92 dims)
    # type_onehot(19) + bp_norm(1) + category(2) + priority_norm(1)
    moves_vals = list((poke.moves or {}).values())
    for i in range(MAX_MOVES):
        if i < len(moves_vals):
            cont.extend(_encode_move_compact(moves_vals[i]))
        else:
            cont.extend([0.0] * 23)

    return {
        "ids": {
            "species": species_id, "item": item_id, "ability": ability_id,
            "move0": move_ids[0], "move1": move_ids[1],
            "move2": move_ids[2], "move3": move_ids[3],
        },
        "banks": {
            "hp_pct": hp_pct, "level": level,
            "weight": weight_int, "height": height_int,
            **stat_ints,
        },
        "continuous": cont,
        "is_empty": False,
    }


def _pokemon_cont_dim() -> int:
    """Dimension of the continuous part of a Pokemon token."""
    # types(19) + status(7) + boosts(91) + active/fainted(2)
    # + volatile(38) + paradox(7) + tera(20) + combat(5) + toxic(1) + future_sight(1)
    # + ability_revealed(1) + item_revealed(1)
    # + move_compact(4 * 23 = 92)
    return 19 + 7 + 91 + 2 + 38 + 7 + 20 + 5 + 1 + 1 + 1 + 1 + 92


POKEMON_CONT_DIM = _pokemon_cont_dim()  # 285

# Compact move encoding: last 4*23=92 dims of pokemon_cont are per-move features.
# Use these helpers instead of hardcoding offsets.
MOVE_CONT_PER_SLOT = 23
N_MOVE_SLOTS = MAX_MOVES  # 4


def extract_move_cont(pokemon_cont):
    """Extract 4×23 compact move features from end of pokemon continuous vector.

    Args:
        pokemon_cont: list or array of length POKEMON_CONT_DIM (285)
    Returns:
        list of 4 sublists, each length MOVE_CONT_PER_SLOT (23)
    """
    n = N_MOVE_SLOTS * MOVE_CONT_PER_SLOT  # 92
    base = len(pokemon_cont) - n
    return [pokemon_cont[base + i * MOVE_CONT_PER_SLOT: base + (i + 1) * MOVE_CONT_PER_SLOT]
            for i in range(N_MOVE_SLOTS)]


def _encode_move_compact(move) -> List[float]:
    """23-dim compact move encoding: type_onehot(19) + bp(1) + category(2) + priority(1)."""
    if move is None:
        return [0.0] * 23
    try:
        # Type one-hot
        type_oh = [0.0] * N_TYPES
        mt = getattr(move, "type", None)
        if mt:
            name = mt.name if hasattr(mt, "name") else str(mt).upper()
            idx = _TYPE_TO_IDX.get(name)
            if idx is not None:
                type_oh[idx] = 1.0
        # Base power normalized
        bp = _clamp_float(float(getattr(move, "base_power", 0) or 0) / 250.0)
        # Category
        cat = getattr(move, "category", None)
        cat_phys = 1.0 if cat == MoveCategory.PHYSICAL else 0.0
        cat_spec = 1.0 if cat == MoveCategory.SPECIAL else 0.0
        # Priority normalized
        prio = _clamp_float(float(getattr(move, "priority", 0) or 0) / 3.0, -1.0, 1.0)
        return type_oh + [bp, cat_phys, cat_spec, prio]
    except Exception:
        return [0.0] * 23


# =============================
# Team extraction
# =============================

def _get_sorted_bench(team) -> list:
    """Non-active bench Pokemon sorted by species name."""
    try:
        pokes = list(team.values())
        active = None
        for p in pokes:
            if p.active:
                active = p
                break
        bench = [p for p in pokes if p is not active]
        bench.sort(key=lambda x: x.species or "")
        return bench
    except Exception:
        return []


def _encode_team(battle, is_opponent: bool) -> List[dict]:
    """Encode a full team as 6 Pokemon dicts. Active first, then bench (sorted)."""
    team = battle.opponent_team if is_opponent else battle.team
    active = battle.opponent_active_pokemon if is_opponent else battle.active_pokemon

    result = []

    # Active Pokemon (slot 0)
    result.append(_encode_pokemon(active, is_active=True, is_opponent=is_opponent))

    # Bench Pokemon (slots 1-5)
    bench = _get_sorted_bench(team)
    for i in range(MAX_BENCH):
        if i < len(bench):
            result.append(_encode_pokemon(bench[i], is_active=False, is_opponent=is_opponent))
        else:
            result.append(_encode_pokemon(None, is_active=False, is_opponent=is_opponent))

    assert len(result) == 6, f"Team should have 6 slots, got {len(result)}"
    return result


# =============================
# Field token
# =============================

def _encode_field(battle) -> dict:
    """Extract field state features."""
    our_sc = battle.side_conditions or {}
    opp_sc = battle.opponent_side_conditions or {}
    weather = battle.weather or {}
    fields = battle.fields or {}
    turn = battle.turn or 0

    def _sc_val(sc, cond, default=0):
        return int(sc.get(cond, default)) if cond in sc else 0

    def _sc_has(sc, cond):
        return 1.0 if cond in sc else 0.0

    # Weather categorical (0=none, 1=sun, 2=rain, 3=sand, 4=snow)
    weather_id = 0
    weather_dur = 0
    if Weather.SUNNYDAY in weather or Weather.DESOLATELAND in weather:
        weather_id = 1
        weather_dur = weather.get(Weather.SUNNYDAY, weather.get(Weather.DESOLATELAND, 0))
    elif Weather.RAINDANCE in weather or Weather.PRIMORDIALSEA in weather:
        weather_id = 2
        weather_dur = weather.get(Weather.RAINDANCE, weather.get(Weather.PRIMORDIALSEA, 0))
    elif Weather.SANDSTORM in weather:
        weather_id = 3
        weather_dur = weather.get(Weather.SANDSTORM, 0)
    elif Weather.SNOWSCAPE in weather or Weather.HAIL in weather:
        weather_id = 4
        weather_dur = weather.get(Weather.SNOWSCAPE, weather.get(Weather.HAIL, 0))

    # Terrain categorical (0=none, 1=electric, 2=grassy, 3=psychic, 4=misty)
    terrain_id = 0
    terrain_dur = 0
    if Field.ELECTRIC_TERRAIN in fields:
        terrain_id = 1
        terrain_dur = fields.get(Field.ELECTRIC_TERRAIN, 0)
    elif Field.GRASSY_TERRAIN in fields:
        terrain_id = 2
        terrain_dur = fields.get(Field.GRASSY_TERRAIN, 0)
    elif Field.PSYCHIC_TERRAIN in fields:
        terrain_id = 3
        terrain_dur = fields.get(Field.PSYCHIC_TERRAIN, 0)
    elif Field.MISTY_TERRAIN in fields:
        terrain_id = 4
        terrain_dur = fields.get(Field.MISTY_TERRAIN, 0)

    trick_room = 1.0 if Field.TRICK_ROOM in fields else 0.0
    tr_dur = _clamp_int(fields.get(Field.TRICK_ROOM, 0), 0, 5)

    # Hazards
    our_sr = _sc_has(our_sc, SideCondition.STEALTH_ROCK)
    our_spikes = _clamp_int(_sc_val(our_sc, SideCondition.SPIKES), 0, 3)
    our_tspikes = _clamp_int(_sc_val(our_sc, SideCondition.TOXIC_SPIKES), 0, 2)
    our_web = _sc_has(our_sc, SideCondition.STICKY_WEB)

    opp_sr = _sc_has(opp_sc, SideCondition.STEALTH_ROCK)
    opp_spikes = _clamp_int(_sc_val(opp_sc, SideCondition.SPIKES), 0, 3)
    opp_tspikes = _clamp_int(_sc_val(opp_sc, SideCondition.TOXIC_SPIKES), 0, 2)
    opp_web = _sc_has(opp_sc, SideCondition.STICKY_WEB)

    # Screens (presence + duration)
    def _screen_dur(sc, cond, max_dur=8):
        if cond not in sc:
            return 0.0, 0
        dur = _clamp_int(turn - int(sc.get(cond, turn)), 0, max_dur)
        remaining = max(0, max_dur - dur)
        return 1.0, remaining

    our_reflect, our_reflect_dur = _screen_dur(our_sc, SideCondition.REFLECT)
    our_ls, our_ls_dur = _screen_dur(our_sc, SideCondition.LIGHT_SCREEN)
    our_av, our_av_dur = _screen_dur(our_sc, SideCondition.AURORA_VEIL)
    opp_reflect, opp_reflect_dur = _screen_dur(opp_sc, SideCondition.REFLECT)
    opp_ls, opp_ls_dur = _screen_dur(opp_sc, SideCondition.LIGHT_SCREEN)
    opp_av, opp_av_dur = _screen_dur(opp_sc, SideCondition.AURORA_VEIL)

    # Tailwind
    our_tailwind = _sc_has(our_sc, SideCondition.TAILWIND)
    opp_tailwind = _sc_has(opp_sc, SideCondition.TAILWIND)

    # Mechanics
    can_tera = 1.0 if battle.can_tera else 0.0
    can_mega = 1.0 if battle.can_mega_evolve else 0.0
    can_z = 1.0 if battle.can_z_move else 0.0
    can_dmax = 1.0 if getattr(battle, "can_dynamax", False) else 0.0
    used_tera_us = 1.0 if battle.used_tera else 0.0
    used_tera_opp = 1.0 if battle.opponent_used_tera else 0.0
    used_mega_us = 1.0 if battle.used_mega_evolve else 0.0
    used_mega_opp = 1.0 if battle.opponent_used_mega_evolve else 0.0
    used_z_us = 1.0 if battle.used_z_move else 0.0
    used_z_opp = 1.0 if battle.opponent_used_z_move else 0.0
    used_dmax_us = 1.0 if getattr(battle, "used_dynamax", False) else 0.0
    used_dmax_opp = 1.0 if getattr(battle, "opponent_used_dynamax", False) else 0.0
    dmax_turns_us = _clamp_float(float(getattr(battle, "dynamax_turns_left", 0) or 0) / 3.0)
    dmax_turns_opp = _clamp_float(float(getattr(battle, "opponent_dynamax_turns_left", 0) or 0) / 3.0)

    trapped = 1.0 if (battle.trapped or battle.maybe_trapped) else 0.0
    force_switch = 1.0 if battle.force_switch else 0.0

    # Opponent revealed fraction
    opp_revealed = sum(1 for p in battle.opponent_team.values()
                       if getattr(p, "revealed", True))
    opp_revealed_frac = _clamp_float(opp_revealed / 6.0)

    # Alive counts
    our_alive = sum(1 for p in battle.team.values() if not p.fainted)
    opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)
    opp_alive = 6 - opp_fainted

    return {
        "banks": {
            "turn": _clamp_int(turn, 0, 200),
            "weather_dur": _clamp_int(weather_dur, 0, 8),
            "terrain_dur": _clamp_int(terrain_dur, 0, 5),
            "tr_dur": tr_dur,
        },
        "continuous": [
            # Weather one-hot (5: none/sun/rain/sand/snow)
            *([1.0 if weather_id == i else 0.0 for i in range(5)]),
            # Terrain one-hot (5: none/elec/grass/psychic/misty)
            *([1.0 if terrain_id == i else 0.0 for i in range(5)]),
            # Trick room
            trick_room,
            # Our hazards
            our_sr, our_spikes / 3.0, our_tspikes / 2.0, our_web,
            # Opp hazards
            opp_sr, opp_spikes / 3.0, opp_tspikes / 2.0, opp_web,
            # Our screens (presence + duration normalized)
            our_reflect, our_reflect_dur / 8.0,
            our_ls, our_ls_dur / 8.0,
            our_av, our_av_dur / 8.0,
            # Opp screens
            opp_reflect, opp_reflect_dur / 8.0,
            opp_ls, opp_ls_dur / 8.0,
            opp_av, opp_av_dur / 8.0,
            # Tailwind
            our_tailwind, opp_tailwind,
            # Mechanics
            can_tera, can_mega, can_z, can_dmax,
            used_tera_us, used_tera_opp,
            used_mega_us, used_mega_opp,
            used_z_us, used_z_opp,
            used_dmax_us, used_dmax_opp,
            dmax_turns_us, dmax_turns_opp,
            trapped, force_switch, opp_revealed_frac,
            # Alive
            our_alive / 6.0, opp_alive / 6.0,
        ],
    }


FIELD_CONT_DIM = 5 + 5 + 1 + 4 + 4 + 6 + 6 + 2 + 17 + 2  # = 52


# =============================
# Transition token
# =============================

def _parse_turn_events(battle) -> dict:
    """Parse previous turn's battle events for the transition token.
    Uses battle.observations[prev_turn].events (poke-env protocol messages)."""

    empty = {
        "our_action_kind": 0,  # 0=NONE, 1=MOVE, 2=SWITCH
        "our_action_id": 0,
        "opp_action_kind": 0,
        "opp_action_id": 0,
        "who_moved_first": 2,  # 0=us, 1=them, 2=unknown
        "our_eff": [0.0] * 6,  # immune/barely/NVE/neutral/SE/ultra
        "opp_eff": [0.0] * 6,
        "our_crit": 0.0,
        "opp_crit": 0.0,
        "our_flinched": 0.0,
        "opp_flinched": 0.0,
        "status_to_us": [0.0] * N_STATUS,
        "status_to_opp": [0.0] * N_STATUS,
        "confused_us": 0.0,
        "confused_opp": 0.0,
        "our_boosts_gained": 0.0,
        "our_boosts_lost": 0.0,
        "opp_boosts_gained": 0.0,
        "opp_boosts_lost": 0.0,
        "we_kod_them": 0.0,
        "they_kod_us": 0.0,
        "our_entry_dmg": 0.0,
        "opp_entry_dmg": 0.0,
        "weather_changed": 0.0,
        "terrain_changed": 0.0,
    }

    try:
        prev_turn = battle.turn - 1
        if prev_turn < 1 or prev_turn not in battle.observations:
            return empty

        events = battle.observations[prev_turn].events
        role = battle.player_role
        if not role:
            return empty
        opp_role = "p2" if role == "p1" else "p1"

        v = _get_vocab()
        result = dict(empty)  # copy defaults

        our_se_count = 0
        our_resist_count = 0
        opp_se_count = 0
        opp_resist_count = 0
        first_mover = None  # track who moved first

        for event in events:
            if len(event) < 2:
                continue

            tag = event[1] if len(event) > 1 else ""
            # Identify which side the event applies to
            who = ""
            if len(event) > 2:
                who = event[2].split(":")[0].strip() if ":" in event[2] else event[2].strip()

            is_our = who.startswith(role)
            is_opp = who.startswith(opp_role)

            # --- Actions ---
            if tag == "move":
                if first_mover is None:
                    first_mover = "us" if is_our else "them"
                mv_name = event[3].strip().lower().replace(" ", "") if len(event) > 3 else ""
                if is_our:
                    result["our_action_kind"] = 1
                    result["our_action_id"] = v.move(mv_name)
                elif is_opp:
                    result["opp_action_kind"] = 1
                    result["opp_action_id"] = v.move(mv_name)

            elif tag == "switch" or tag == "drag":
                species = ""
                if len(event) > 3:
                    # event[3] is like "Garchomp, L100, M" or "Garchomp, L100"
                    species = event[3].split(",")[0].strip().lower().replace(" ", "")
                if is_our:
                    if result["our_action_kind"] == 0:  # don't overwrite move
                        result["our_action_kind"] = 2
                        result["our_action_id"] = v.species(species)
                elif is_opp:
                    if result["opp_action_kind"] == 0:
                        result["opp_action_kind"] = 2
                        result["opp_action_id"] = v.species(species)

            # --- Effectiveness ---
            elif tag == "-supereffective":
                if is_opp:  # our move was SE against opponent
                    our_se_count += 1
                elif is_our:  # opp move was SE against us
                    opp_se_count += 1

            elif tag == "-resisted":
                if is_opp:
                    our_resist_count += 1
                elif is_our:
                    opp_resist_count += 1

            elif tag == "-immune":
                if is_opp:  # our move was immune
                    result["our_eff"][0] = 1.0  # immune
                elif is_our:
                    result["opp_eff"][0] = 1.0

            # --- Crits ---
            elif tag == "-crit":
                if is_opp:  # we crit the opponent
                    result["our_crit"] = 1.0
                elif is_our:
                    result["opp_crit"] = 1.0

            # --- Flinch ---
            elif tag == "cant":
                reason = event[3] if len(event) > 3 else ""
                if "flinch" in reason.lower():
                    if is_our:
                        result["our_flinched"] = 1.0
                    elif is_opp:
                        result["opp_flinched"] = 1.0

            # --- Status applied ---
            elif tag == "-status":
                status_str = event[3].upper() if len(event) > 3 else ""
                if status_str.startswith("TOX"):
                    status_str = "TOX"
                elif status_str.startswith("PSN"):
                    status_str = "PSN"
                status_idx = _STATUS_TO_IDX.get(status_str, 0)
                if is_our:
                    result["status_to_us"][status_idx] = 1.0
                elif is_opp:
                    result["status_to_opp"][status_idx] = 1.0

            # --- Confusion ---
            elif tag == "-start":
                vol = event[3] if len(event) > 3 else ""
                if "confusion" in vol.lower():
                    if is_our:
                        result["confused_us"] = 1.0
                    elif is_opp:
                        result["confused_opp"] = 1.0

            # --- Stat changes ---
            elif tag == "-boost":
                amount = _clamp_int(event[4] if len(event) > 4 else 1, 0, 6)
                if is_our:
                    result["our_boosts_gained"] += amount / 6.0
                elif is_opp:
                    result["opp_boosts_gained"] += amount / 6.0

            elif tag == "-unboost":
                amount = _clamp_int(event[4] if len(event) > 4 else 1, 0, 6)
                if is_our:
                    result["our_boosts_lost"] += amount / 6.0
                elif is_opp:
                    result["opp_boosts_lost"] += amount / 6.0

            # --- KO ---
            elif tag == "faint":
                if is_our:
                    result["they_kod_us"] = 1.0
                elif is_opp:
                    result["we_kod_them"] = 1.0

            # --- Entry hazard damage ---
            elif tag == "-damage":
                source = ""
                for part in event[3:]:
                    if "[from]" in str(part):
                        source = str(part).lower()
                        break
                if any(h in source for h in ["stealth rock", "spikes", "g-max steelsurge"]):
                    # Parse HP fraction from damage event
                    # Format: "p1a: Mon|85/100" or similar
                    try:
                        hp_text = event[3] if len(event) > 3 else ""
                        if "/" in hp_text:
                            cur, mx = hp_text.split("/")
                            dmg_frac = 1.0 - float(cur) / float(mx)
                        else:
                            dmg_frac = 0.125  # default SR damage
                    except (ValueError, ZeroDivisionError):
                        dmg_frac = 0.125
                    if is_our:
                        result["our_entry_dmg"] = _clamp_float(dmg_frac, 0.0, 0.5)
                    elif is_opp:
                        result["opp_entry_dmg"] = _clamp_float(dmg_frac, 0.0, 0.5)

            # --- Weather/terrain changes ---
            elif tag == "-weather":
                weather_name = event[2] if len(event) > 2 else ""
                if weather_name.lower() != "none":
                    result["weather_changed"] = 1.0

            elif tag == "-fieldstart":
                result["terrain_changed"] = 1.0

        # Resolve effectiveness from SE/resist counts
        # Our move effectiveness
        if result["our_eff"][0] < 0.5:  # not immune
            if our_se_count >= 2:
                result["our_eff"][5] = 1.0  # ultra effective (4x)
            elif our_se_count == 1 and our_resist_count == 0:
                result["our_eff"][4] = 1.0  # super effective (2x)
            elif our_se_count == 1 and our_resist_count == 1:
                result["our_eff"][3] = 1.0  # neutral (SE + resist = 1x)
            elif our_resist_count >= 2:
                result["our_eff"][1] = 1.0  # barely effective (0.25x)
            elif our_resist_count == 1:
                result["our_eff"][2] = 1.0  # not very effective (0.5x)
            elif our_se_count == 0 and our_resist_count == 0:
                result["our_eff"][3] = 1.0  # neutral (1x)

        # Opp move effectiveness
        if result["opp_eff"][0] < 0.5:
            if opp_se_count >= 2:
                result["opp_eff"][5] = 1.0
            elif opp_se_count == 1 and opp_resist_count == 0:
                result["opp_eff"][4] = 1.0
            elif opp_se_count == 1 and opp_resist_count == 1:
                result["opp_eff"][3] = 1.0
            elif opp_resist_count >= 2:
                result["opp_eff"][1] = 1.0
            elif opp_resist_count == 1:
                result["opp_eff"][2] = 1.0
            elif opp_se_count == 0 and opp_resist_count == 0:
                result["opp_eff"][3] = 1.0

        # Who moved first
        if first_mover == "us":
            result["who_moved_first"] = 0
        elif first_mover == "them":
            result["who_moved_first"] = 1

        # Clamp accumulated stat change floats
        result["our_boosts_gained"] = _clamp_float(result["our_boosts_gained"])
        result["our_boosts_lost"] = _clamp_float(result["our_boosts_lost"])
        result["opp_boosts_gained"] = _clamp_float(result["opp_boosts_gained"])
        result["opp_boosts_lost"] = _clamp_float(result["opp_boosts_lost"])

        return result

    except Exception:
        return empty


def _encode_transition(battle) -> dict:
    """Build the transition token from previous turn events."""
    ev = _parse_turn_events(battle)

    return {
        "ids": {
            "our_action": ev["our_action_id"],
            "opp_action": ev["opp_action_id"],
        },
        "continuous": [
            # Action kind one-hots (3 each)
            1.0 if ev["our_action_kind"] == 0 else 0.0,
            1.0 if ev["our_action_kind"] == 1 else 0.0,
            1.0 if ev["our_action_kind"] == 2 else 0.0,
            1.0 if ev["opp_action_kind"] == 0 else 0.0,
            1.0 if ev["opp_action_kind"] == 1 else 0.0,
            1.0 if ev["opp_action_kind"] == 2 else 0.0,
            # Who moved first (3 one-hot)
            1.0 if ev["who_moved_first"] == 0 else 0.0,
            1.0 if ev["who_moved_first"] == 1 else 0.0,
            1.0 if ev["who_moved_first"] == 2 else 0.0,
            # Our effectiveness (6)
            *ev["our_eff"],
            # Opp effectiveness (6)
            *ev["opp_eff"],
            # Crits (2)
            ev["our_crit"], ev["opp_crit"],
            # Flinch (2)
            ev["our_flinched"], ev["opp_flinched"],
            # Status applied (7 + 7 + 2)
            *ev["status_to_us"], *ev["status_to_opp"],
            ev["confused_us"], ev["confused_opp"],
            # Stat changes (4)
            ev["our_boosts_gained"], ev["our_boosts_lost"],
            ev["opp_boosts_gained"], ev["opp_boosts_lost"],
            # KO (2)
            ev["we_kod_them"], ev["they_kod_us"],
            # Entry hazard damage (2)
            ev["our_entry_dmg"], ev["opp_entry_dmg"],
            # Field changes (2)
            ev["weather_changed"], ev["terrain_changed"],
        ],
    }


TRANSITION_CONT_DIM = 3 + 3 + 3 + 6 + 6 + 2 + 2 + 7 + 7 + 2 + 4 + 2 + 2 + 2  # = 51
# Plus the 6 action kind dims are included above. Let me recount:
# action_kind(3+3) + moved_first(3) + eff(6+6) + crit(2) + flinch(2)
# + status(7+7) + confused(2) + stat_changes(4) + ko(2) + entry(2) + field(2)
# = 6 + 3 + 12 + 2 + 2 + 14 + 2 + 4 + 2 + 2 + 2 = 51


# =============================
# Move slot features (for action selection)
# =============================

def _coerce_bool(x) -> bool:
    try:
        return bool(x)
    except Exception:
        return False


def _coerce_float(x, default=0.0) -> float:
    try:
        return float(x if x is not None else default)
    except Exception:
        return default


def _coerce_int(x, default=0) -> int:
    try:
        return int(x if x is not None else default)
    except Exception:
        return default


_safe_getattr_warned = set()

def _safe_getattr(obj, name, default=None):
    """Like getattr but catches KeyError/ValueError from poke-env properties
    that access self.entry[key] internally (fails for unknown/incomplete moves)."""
    try:
        return getattr(obj, name, default)
    except (KeyError, ValueError, TypeError):
        obj_id = getattr(obj, "id", getattr(obj, "_id", "?"))
        key = f"{obj_id}.{name}"
        if key not in _safe_getattr_warned:
            _safe_getattr_warned.add(key)
            print(f"  [WARN] _safe_getattr: {key} failed, using default={default}", flush=True)
        return default


def _project_move_flags(m, poke_types=None) -> dict:
    """Convert a poke-env Move object into a model-friendly dict.
    Uses poke-env Move attributes only — no hardcoded dex tables.
    Identical to v7 _project_move_flags (well-tested, 107-dim base output, +2 from action slots)."""

    # Accuracy
    acc_raw = _safe_getattr(m, "accuracy", None)
    if acc_raw is True or acc_raw is None:
        acc01 = 1.0
    else:
        acc01 = max(0.0, min(1.0, float(acc_raw)))

    prio = _coerce_int(_safe_getattr(m, "priority", 0), default=0)
    prio_n = max(-3, min(3, prio))

    drain = _coerce_float(_safe_getattr(m, "drain", 0), default=0.0)
    recoil = _coerce_float(_safe_getattr(m, "recoil", 0), default=0.0)
    heal = _coerce_float(_safe_getattr(m, "heal", 0), default=0.0)
    recharge = _coerce_bool(_safe_getattr(m, "recharge", _safe_getattr(m, "recharge_turn", False)))

    multihit = _safe_getattr(m, "multihit", None)
    if isinstance(multihit, (tuple, list)) and len(multihit) == 2:
        multihit_est = float((int(multihit[0]) + int(multihit[1])) / 2.0)
    else:
        multihit_est = float(_coerce_int(multihit or 1, default=1))

    _flags = _safe_getattr(m, "flags", set()) or set()
    contact = "contact" in _flags
    protect_blocked = "protect" in _flags
    sound = "sound" in _flags
    punch = "punch" in _flags
    bite = "bite" in _flags
    powder = "powder" in _flags
    crit_ratio = _coerce_int(_safe_getattr(m, "crit_ratio", 0), default=0)

    status_to = None
    st = _safe_getattr(m, "status", None)
    if st:
        status_to = st.name if hasattr(st, "name") else str(st).upper()

    # Flinch from secondary effects
    flinch01 = 0.0
    _sec = _safe_getattr(m, "secondary", None) or []
    _sec_list = _sec if isinstance(_sec, list) else [_sec]
    for _s in _sec_list:
        if isinstance(_s, dict) and "flinch" in str(_s.get("volatileStatus", "")):
            flinch01 = max(flinch01, _coerce_float(_s.get("chance", 100), 100.0) / 100.0)

    phaze = _coerce_bool(_safe_getattr(m, "force_switch", _safe_getattr(m, "phaze", False)))
    pivot = _coerce_bool(_safe_getattr(m, "self_switch", _safe_getattr(m, "pivot", False)))

    mid = str(_safe_getattr(m, "id", "") or "").lower()

    vs = _safe_getattr(m, "volatile_status", None)
    # Trapping: volatile_status=None for meanlook/block/etc — must stay hardcoded
    trap = (vs in (Effect.PARTIALLY_TRAPPED, Effect.NO_RETREAT) or
            mid in ("meanlook", "block", "spiderweb", "anchorshot", "spiritshackle",
                    "thousandwaves", "jawlock"))

    # Hazards, weather, terrain — poke-env enum comparisons
    _sc = _safe_getattr(m, "side_condition", None)
    sets_sr = _sc == SideCondition.STEALTH_ROCK
    sets_spikes = _sc == SideCondition.SPIKES
    sets_tspikes = _sc == SideCondition.TOXIC_SPIKES
    sets_web = _sc == SideCondition.STICKY_WEB

    _weather = _safe_getattr(m, "weather", None)
    sets_sun = _weather in (Weather.SUNNYDAY, Weather.DESOLATELAND)
    sets_rain = _weather in (Weather.RAINDANCE, Weather.PRIMORDIALSEA)
    sets_sand = _weather == Weather.SANDSTORM
    sets_snow = _weather in (Weather.SNOWSCAPE, Weather.HAIL)

    _terrain = _safe_getattr(m, "terrain", None)
    sets_electric = _terrain == Field.ELECTRIC_TERRAIN
    sets_grassy = _terrain == Field.GRASSY_TERRAIN
    sets_psychic = _terrain == Field.PSYCHIC_TERRAIN
    sets_misty = _terrain == Field.MISTY_TERRAIN

    # Hazard clearing — no structured property in poke-env, must stay hardcoded
    clears_hazards = mid in ("defog", "rapidspin", "courtchange", "mortalspin", "tidyup")

    cat = _safe_getattr(m, "category", None)

    # STAB
    stab = False
    if poke_types and _safe_getattr(m, "type", None):
        move_type_name = m.type.name if hasattr(m.type, "name") else str(m.type).upper()
        for pt in poke_types:
            if pt and (pt.name if hasattr(pt, "name") else str(pt).upper()) == move_type_name:
                stab = True
                break

    # Boosts
    _BOOST_KEYS_M = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
    self_boost_raw = dict(_safe_getattr(m, "self_boost", None) or {})
    boosts_raw = dict(_safe_getattr(m, "boosts", None) or {})
    _tgt = _safe_getattr(m, "target", None)
    if _tgt in _SELF_TARGETS and boosts_raw:
        for k, bv in boosts_raw.items():
            self_boost_raw[k] = self_boost_raw.get(k, 0) + bv
        boosts_raw = {}
    # Protect / stalling / breaks protect
    is_protect = _coerce_bool(_safe_getattr(m, "is_protect_move", False))
    is_stalling = _coerce_bool(_safe_getattr(m, "stalling_move", False))
    breaks_prot = _coerce_bool(_safe_getattr(m, "breaks_protect", False))

    # Self-destruct, fixed damage
    self_destruct_val = _coerce_bool(_safe_getattr(m, "self_destruct", False))
    raw_damage = _safe_getattr(m, "damage", 0)
    if isinstance(raw_damage, str) and raw_damage.lower() == "level":
        has_fixed_damage = True
    else:
        has_fixed_damage = (_coerce_int(raw_damage, 0) > 0)

    # Secondary effect chances — parse ALL fields from the secondary dict
    secondary = _safe_getattr(m, "secondary", None) or []
    sec_burn = sec_freeze = sec_para = sec_poison = sec_flinch = 0.0
    sec_confusion = 0.0
    sec_target_boosts = {}  # stat -> weighted boost change
    sec_self_boosts = {}
    sec_list = secondary if isinstance(secondary, list) else [secondary]
    for sec in sec_list:
        if not isinstance(sec, dict):
            continue
        chance = _coerce_float(sec.get("chance", 100), 100.0) / 100.0
        # Status (raw string from Showdown data, not enum)
        sec_status = sec.get("status", "")
        if hasattr(sec_status, "name"):
            sec_status = sec_status.name
        sec_status = str(sec_status).upper()
        if "BRN" in sec_status: sec_burn = max(sec_burn, chance)
        if "FRZ" in sec_status: sec_freeze = max(sec_freeze, chance)
        if "PAR" in sec_status: sec_para = max(sec_para, chance)
        if "PSN" in sec_status or "TOX" in sec_status: sec_poison = max(sec_poison, chance)
        # Volatile status
        sec_vs = str(sec.get("volatileStatus", "")).lower()
        if "flinch" in sec_vs: sec_flinch = max(sec_flinch, chance)
        if "confusion" in sec_vs: sec_confusion = max(sec_confusion, chance)
        # Target stat drops (e.g. Moonblast 30% SpA-1, Crunch 20% Def-1)
        sec_boosts = sec.get("boosts")
        if isinstance(sec_boosts, dict):
            for stat, val in sec_boosts.items():
                sec_target_boosts[stat] = sec_target_boosts.get(stat, 0) + chance * val
        # Self stat boosts (e.g. Ancient Power 10% omniboost)
        sec_self = sec.get("self")
        if isinstance(sec_self, dict):
            sec_self_b = sec_self.get("boosts")
            if isinstance(sec_self_b, dict):
                for stat, val in sec_self_b.items():
                    sec_self_boosts[stat] = sec_self_boosts.get(stat, 0) + chance * val

    # Merge secondary boosts into primary boost dicts before normalization
    for k in _BOOST_KEYS_M:
        if k in sec_self_boosts:
            self_boost_raw[k] = self_boost_raw.get(k, 0) + sec_self_boosts[k]
        if k in sec_target_boosts:
            boosts_raw[k] = boosts_raw.get(k, 0) + sec_target_boosts[k]

    has_self_boosts = bool(self_boost_raw) or bool(sec_self_boosts)
    has_target_boosts = bool(boosts_raw) or bool(sec_target_boosts)
    norm_boost = lambda d: [max(0.0, min(1.0, (float(d.get(k, 0)) + 3.0) / 6.0))
                            for k in _BOOST_KEYS_M]
    self_boost_vals = norm_boost(self_boost_raw) if has_self_boosts else [0.5] * 7
    target_boost_vals = norm_boost(boosts_raw) if has_target_boosts else [0.5] * 7

    # Target one-hot (uses module-level _TARGET_MAP with enum keys)
    target_raw = _safe_getattr(m, "target", None)
    target_idx = _TARGET_MAP.get(target_raw, 5)
    target_oh = [0.0] * 6
    target_oh[target_idx] = 1.0

    # Ignore ability / immunity
    ignore_ability = _coerce_bool(_safe_getattr(m, "ignore_ability", False))
    ignore_immunity_raw = _safe_getattr(m, "ignore_immunity", False)
    ignore_immunity = bool(ignore_immunity_raw and ignore_immunity_raw is not False)

    # Type one-hot
    type_oh = [0.0] * N_TYPES
    if _safe_getattr(m, "type", None):
        tn = m.type.name if hasattr(m.type, "name") else str(m.type).upper()
        tidx = _TYPE_TO_IDX.get(tn)
        if tidx is not None:
            type_oh[tidx] = 1.0

    # Status to one-hot
    status_to_oh = [0.0] * N_STATUS
    if status_to:
        sidx = _STATUS_TO_IDX.get(status_to, 0)
        status_to_oh[sidx] = 1.0

    bp01 = _clamp_float(float(_coerce_int(_safe_getattr(m, "base_power", 0), 0)) / 250.0)

    return {
        "id": _safe_getattr(m, "id", None),
        "move_id": _get_vocab().move(_safe_getattr(m, "id", None)),
        "bp_int": _clamp_int(_safe_getattr(m, "base_power", 0), 0, 255),
        "acc_int": _clamp_int(acc01 * 100, 0, 100),
        "pp_int": _clamp_int(_safe_getattr(m, "current_pp", _safe_getattr(m, "pp", 0)), 0, 64),
        "priority_int": _clamp_int(prio + 6, 0, 12),  # map -6..+6 to 0..12
        "continuous": [
            bp01, acc01, prio_n / 3.0,
            max(0.0, min(1.0, drain)), max(0.0, min(1.0, abs(recoil))),  # recoil magnitude (poke-env stores positive)
            max(0.0, min(1.0, heal)), multihit_est / 5.0, flinch01,
            min(1.0, crit_ratio / 3.0),
            _coerce_float(_safe_getattr(m, "current_pp", _safe_getattr(m, "pp", 0)), 0) / 64.0,
            1.0 if _coerce_bool(_safe_getattr(m, "disabled", False)) else 0.0,
            1.0 if recharge else 0.0,
            1.0 if stab else 0.0,
            1.0 if contact else 0.0,
            1.0 if protect_blocked else 0.0,
            1.0 if sound else 0.0,
            1.0 if punch else 0.0,
            1.0 if bite else 0.0,
            1.0 if powder else 0.0,
            1.0 if phaze else 0.0,
            1.0 if pivot else 0.0,
            1.0 if trap else 0.0,
            1.0 if sets_sr else 0.0,
            1.0 if sets_spikes else 0.0,
            1.0 if sets_tspikes else 0.0,
            1.0 if sets_web else 0.0,
            1.0 if sets_sun else 0.0,
            1.0 if sets_rain else 0.0,
            1.0 if sets_sand else 0.0,
            1.0 if sets_snow else 0.0,
            1.0 if sets_electric else 0.0,
            1.0 if sets_grassy else 0.0,
            1.0 if sets_psychic else 0.0,
            1.0 if sets_misty else 0.0,
            1.0 if clears_hazards else 0.0,
            1.0 if cat == MoveCategory.PHYSICAL else 0.0,
            1.0 if cat == MoveCategory.SPECIAL else 0.0,
            1.0 if cat == MoveCategory.STATUS else 0.0,
            1.0 if _sc == SideCondition.REFLECT else 0.0,
            1.0 if _sc == SideCondition.LIGHT_SCREEN else 0.0,
            1.0 if _sc == SideCondition.AURORA_VEIL else 0.0,
            sec_confusion if sec_confusion > 0 else (1.0 if vs == Effect.CONFUSION else 0.0),
            1.0 if vs == Effect.TAUNT else 0.0,
            1.0 if vs == Effect.ENCORE else 0.0,
            1.0 if vs == Effect.DISABLE else 0.0,
            1.0 if vs == Effect.LEECH_SEED else 0.0,
            1.0 if vs == Effect.YAWN else 0.0,
            1.0 if vs == Effect.SUBSTITUTE else 0.0,
            1.0 if vs == Effect.TORMENT else 0.0,
            1.0 if is_protect else 0.0,
            1.0 if is_stalling else 0.0,
            1.0 if breaks_prot else 0.0,
            1.0 if self_destruct_val else 0.0,
            1.0 if has_fixed_damage else 0.0,
            sec_burn, sec_freeze, sec_para, sec_poison, sec_flinch,
            1.0 if ignore_ability else 0.0,
            1.0 if ignore_immunity else 0.0,
            *self_boost_vals,  # 7
            *target_boost_vals,  # 7
            *target_oh,  # 6
            *type_oh,  # 19
            *status_to_oh,  # 7
        ],
    }


MOVE_SLOT_CONT_DIM = 109  # 107 base + 1 type_effectiveness + 1 opp_threat


# =============================
# Legal mask + action slots
# =============================

def _compute_type_effectiveness(move, opp_pokemon) -> float:
    """Compute damage multiplier of move vs opponent active pokemon.
    Returns normalized float via mult/4.0:
    0x(immune)->0.0, 0.25x->0.0625, 0.5x->0.125, 1x->0.25, 2x->0.5, 4x->1.0
    Uses poke-env Pokemon.damage_multiplier() — no hardcoded type chart."""
    if opp_pokemon is None or move is None:
        return 0.25  # neutral default (1.0/4.0)
    move_type = _safe_getattr(move, "type", None)
    if move_type is None:
        return 0.25
    try:
        mult = opp_pokemon.damage_multiplier(move_type)
    except Exception:
        return 0.25
    return min(1.0, max(0.0, mult / 4.0))


def _opp_type_threat(defender, opp_active) -> float:
    """Max type effectiveness of opponent's known moves (or STAB types) vs defender.
    Returns 0-1 scale via mult/4.0 (0.25=neutral). Higher = more danger.
    Uses poke-env damage_multiplier() throughout."""
    if defender is None or opp_active is None:
        return 0.25  # neutral when unknown

    max_eff = 0.01  # floor at 0.01 (not 0.0) to avoid FP16 edge cases
    # Check opponent's revealed moves
    opp_moves = getattr(opp_active, "moves", {})
    for move in opp_moves.values():
        eff = _compute_type_effectiveness(move, defender)
        max_eff = max(max_eff, eff)

    # If no moves revealed, use opponent STAB types as proxy
    if not opp_moves and opp_active.types:
        for opp_type in opp_active.types:
            if opp_type is None:
                continue
            try:
                mult = defender.damage_multiplier(opp_type)
            except Exception:
                continue
            max_eff = max(max_eff, min(1.0, max(0.01, mult / 4.0)))

    return max_eff


def _max_opp_threat(battle) -> float:
    """Max opponent threat vs our active pokemon."""
    return _opp_type_threat(battle.active_pokemon, battle.opponent_active_pokemon)


def _switch_defensive_effectiveness(switch_target, opp_active) -> float:
    """Max opponent threat vs a potential switch target. LOWER = safer to switch into."""
    return _opp_type_threat(switch_target, opp_active)


def _switch_offensive_effectiveness(switch_target, opp_active) -> float:
    """Max STAB effectiveness of switch target vs opponent active.
    Uses switch target's types as STAB proxy. HIGHER = better attacker if switched in.
    Uses poke-env damage_multiplier()."""
    if switch_target is None or opp_active is None:
        return 0.25  # neutral when unknown
    max_eff = 0.01  # floor at 0.01 (not 0.0) to avoid FP16 edge cases
    for stab_type in switch_target.types:
        if stab_type is None:
            continue
        try:
            mult = opp_active.damage_multiplier(stab_type)
        except Exception:
            continue
        max_eff = max(max_eff, min(1.0, max(0.01, mult / 4.0)))
    return max_eff


def _encode_action_slots(battle) -> Tuple[np.ndarray, List[dict], List[dict]]:
    """Returns (legal_mask[9], active_move_dicts[4], switch_dicts[5])."""
    mask = np.zeros(9, dtype=np.float32)
    move_dicts = []
    switch_dicts = []

    # Moves
    moves = list(battle.available_moves or [])
    active_types = battle.active_pokemon.types if battle.active_pokemon else None
    opp_active = battle.opponent_active_pokemon
    opp_threat = _max_opp_threat(battle)
    for i in range(4):
        if i < len(moves):
            mask[i] = 1.0
            md = _project_move_flags(moves[i], poke_types=active_types)
            # Append type effectiveness vs opponent active + opponent threat to us
            md["continuous"].append(_compute_type_effectiveness(moves[i], opp_active))
            md["continuous"].append(opp_threat)
            move_dicts.append(md)
        else:
            move_dicts.append(None)

    # Switches
    switches = list(battle.available_switches or [])
    v = _get_vocab()
    for j in range(5):
        if j < len(switches):
            p = switches[j]
            mask[4 + j] = 1.0
            switch_dicts.append({
                "species_id": v.species(p.species),
                "continuous": (
                    _types_multihot(p.types) +  # 19
                    [_clamp_float(p.current_hp_fraction or 0)] +  # 1
                    _status_onehot(p.status) +  # 7
                    [_clamp_float(float(getattr(p, "weight", 0) or 0) / 1000.0)] +  # 1
                    [_switch_defensive_effectiveness(p, opp_active)] +  # 1
                    [_switch_offensive_effectiveness(p, opp_active)]  # 1
                ),
            })
        else:
            switch_dicts.append(None)

    # Safety: ensure at least one legal action
    if mask.sum() <= 0.0:
        if len(moves) > 0:
            mask[0] = 1.0
        elif len(switches) > 0:
            mask[4] = 1.0

    return mask, move_dicts, switch_dicts


SWITCH_SLOT_CONT_DIM = 30  # 19 + 1 + 7 + 1 + 1 (defensive_eff) + 1 (offensive_eff)


# =============================
# Public API
# =============================

def make_features(battle) -> dict:
    """Extract all v8 structured features from a poke-env Battle object.

    Returns dict with keys:
      our_pokemon: list of 6 dicts
      opp_pokemon: list of 6 dicts
      field: dict
      transition: dict
      legal_mask: np.ndarray (9,) float32
      active_moves: list of 4 dicts (or None for empty slots)
      switch_slots: list of 5 dicts (or None for empty slots)
    """
    our_pokemon = _encode_team(battle, is_opponent=False)
    opp_pokemon = _encode_team(battle, is_opponent=True)
    field = _encode_field(battle)
    transition = _encode_transition(battle)
    legal_mask, active_moves, switch_slots = _encode_action_slots(battle)

    return {
        "our_pokemon": our_pokemon,
        "opp_pokemon": opp_pokemon,
        "field": field,
        "transition": transition,
        "legal_mask": legal_mask,
        "active_moves": active_moves,
        "switch_slots": switch_slots,
    }


# =============================
# Dimension constants (for validation and memmap)
# =============================

DIMS = {
    "pokemon_cont": POKEMON_CONT_DIM,      # 285
    "pokemon_ids": 7,                        # species, item, ability, move0-3
    "pokemon_banks": 10,                     # hp_pct, level, weight, height, stat_hp/atk/def/spa/spd/spe
    "field_cont": FIELD_CONT_DIM,           # 52
    "field_banks": 4,                        # turn, weather_dur, terrain_dur, tr_dur
    "transition_cont": TRANSITION_CONT_DIM, # 51
    "transition_ids": 2,                     # our_action, opp_action
    "move_slot_cont": MOVE_SLOT_CONT_DIM,  # 109
    "move_slot_ids": 1,                      # move_id
    "move_slot_banks": 4,                    # bp_int, acc_int, pp_int, priority_int
    "switch_slot_cont": SWITCH_SLOT_CONT_DIM, # 30
    "switch_slot_ids": 1,                    # species_id
    "legal_mask": 9,
    "n_volatile": N_VOLATILE,               # 38
    "n_paradox": N_PARADOX,                  # 7
}


# =============================
# Shared batch-building utilities
# =============================
# Used by battle_agent.py, rl_player.py, rl_pipeline.py to convert
# make_features() output into the dict PokeTransformer.forward() expects.
# Centralizing here eliminates 300+ lines of duplication.


def _poke_ids(p: dict) -> list:
    """Extract [species, item, ability] from a pokemon feature dict."""
    ids = p["ids"]
    return [ids["species"], ids["item"], ids["ability"]]


def _poke_banks(p: dict) -> list:
    """Extract 10 bank values from a pokemon feature dict."""
    b = p["banks"]
    return [b["hp_pct"], b["level"], b["weight"], b["height"],
            b["stat_hp"], b["stat_atk"], b["stat_def"],
            b["stat_spa"], b["stat_spd"], b["stat_spe"]]


def _poke_move_ids(p: dict) -> list:
    """Extract [move0, move1, move2, move3] IDs from a pokemon feature dict."""
    ids = p["ids"]
    return [ids["move0"], ids["move1"], ids["move2"], ids["move3"]]


def _permute_team(team: list) -> list:
    """Randomly permute bench slots (1-5) of a 6-pokemon team list.

    Slot 0 (active) is never moved. Returns a new list (does not mutate input).
    Also randomly permutes the 4 move features within each pokemon's continuous
    vector (the last 92 dims = 4 moves × 23 features each).
    """
    import random

    import copy

    # Permute bench order (shallow copy team, deep copy each pokemon dict)
    perm = [0] + random.sample(range(1, 6), 5)
    team = [copy.deepcopy(team[i]) for i in perm]

    # Permute move order within each pokemon's entity features
    for p in team:
        cont = p.get("continuous")
        move_ids_key = "move_ids"
        if cont is None:
            continue
        # Move features are the last N_MOVE_SLOTS * MOVE_CONT_PER_SLOT dims of continuous
        n_move_dims = N_MOVE_SLOTS * MOVE_CONT_PER_SLOT  # 4 * 23 = 92
        base = len(cont) - n_move_dims
        if base < 0:
            continue
        move_perm = random.sample(range(N_MOVE_SLOTS), N_MOVE_SLOTS)
        # Permute continuous move features
        new_cont = list(cont[:base])
        for mi in move_perm:
            src = base + mi * MOVE_CONT_PER_SLOT
            new_cont.extend(cont[src: src + MOVE_CONT_PER_SLOT])
        p["continuous"] = new_cont
        # Permute categorical move IDs (stored as ids["move0"]..ids["move3"])
        ids = p.get("ids")
        if ids and "move0" in ids:
            old = [ids[f"move{i}"] for i in range(N_MOVE_SLOTS)]
            for i, mi in enumerate(move_perm):
                ids[f"move{i}"] = old[mi]

    return team


def build_turn_batch(feat: dict, device=None, training: bool = False) -> dict:
    """Convert make_features() output to PokeTransformer batch dict.

    Args:
        feat: dict from make_features(battle)
        device: torch device (None = CPU tensors, 'cuda' = GPU tensors)
        training: if True, apply slot permutation augmentation (bench + moves)

    Returns:
        dict with all keys expected by PokeTransformer.forward()
    """
    import torch

    def _t(data, dtype=torch.long):
        t = torch.tensor(data, dtype=dtype)
        return t.to(device, non_blocking=True) if device else t

    def _tf(data):
        return _t(data, dtype=torch.float32)

    our, opp = feat["our_pokemon"], feat["opp_pokemon"]

    if training:
        our = _permute_team(list(our))
        opp = _permute_team(list(opp))

    batch = {
        # Pokemon IDs, banks, continuous, move IDs, move continuous
        "our_pokemon_ids": _t([[_poke_ids(p) for p in our]]),
        "our_pokemon_banks": _t([[_poke_banks(p) for p in our]]),
        "our_pokemon_cont": _tf([[p["continuous"] for p in our]]),
        "our_pokemon_move_ids": _t([[_poke_move_ids(p) for p in our]]),
        "our_pokemon_move_cont": _tf([[extract_move_cont(p["continuous"]) for p in our]]),
        "opp_pokemon_ids": _t([[_poke_ids(p) for p in opp]]),
        "opp_pokemon_banks": _t([[_poke_banks(p) for p in opp]]),
        "opp_pokemon_cont": _tf([[p["continuous"] for p in opp]]),
        "opp_pokemon_move_ids": _t([[_poke_move_ids(p) for p in opp]]),
        "opp_pokemon_move_cont": _tf([[extract_move_cont(p["continuous"]) for p in opp]]),
        # Field
        "field_cont": _tf([feat["field"]["continuous"]]),
        # Transition
        "transition_cont": _tf([feat["transition"]["continuous"]]),
        # Legal mask
        "legal_mask": _tf(feat["legal_mask"].reshape(1, -1).tolist()),
    }

    # Field banks (dict of tensors)
    fb = feat["field"]["banks"]
    batch["field_banks"] = {k: _t([fb[k]]) for k in fb}

    # Transition IDs (dict of tensors)
    ti = feat["transition"]["ids"]
    batch["transition_ids"] = {k: _t([ti[k]]) for k in ti}

    # Active moves
    mids, mbp, mac, mpp, mpr, mco = [], [], [], [], [], []
    for m in feat["active_moves"]:
        if m is None:
            mids.append(0); mbp.append(0); mac.append(0)
            mpp.append(0); mpr.append(6)
            mco.append([0.0] * MOVE_SLOT_CONT_DIM)
        else:
            mids.append(m["move_id"]); mbp.append(m["bp_int"])
            mac.append(m["acc_int"]); mpp.append(m["pp_int"])
            mpr.append(m["priority_int"]); mco.append(m["continuous"])
    batch["active_move_ids"] = _t([mids])
    batch["active_move_cont"] = _tf([mco])
    batch["active_move_banks"] = {
        "bp": _t([mbp]), "acc": _t([mac]),
        "pp": _t([mpp]), "prio": _t([mpr]),
    }

    # Switch slots
    sids, sco = [], []
    for s in feat["switch_slots"]:
        if s is None:
            sids.append(0); sco.append([0.0] * SWITCH_SLOT_CONT_DIM)
        else:
            sids.append(s["species_id"]); sco.append(s["continuous"])
    batch["switch_ids"] = _t([sids])
    batch["switch_cont"] = _tf([sco])

    return batch


def action_to_order(player, battle, action_idx: int):
    """Convert action index (0-8) to a poke-env BattleOrder.

    Args:
        player: poke-env Player instance (has create_order method)
        battle: poke-env Battle instance (has available_moves/switches)
        action_idx: 0-3 = move, 4-8 = switch

    Falls back to first legal move/switch if index is out of range.
    Returns None only if no legal actions exist.
    """
    if action_idx < 4:
        moves = list(battle.available_moves or [])
        if action_idx < len(moves):
            return player.create_order(moves[action_idx])
    else:
        switches = list(battle.available_switches or [])
        si = action_idx - 4
        if si < len(switches):
            return player.create_order(switches[si])
    # Fallback
    if battle.available_moves:
        return player.create_order(battle.available_moves[0])
    if battle.available_switches:
        return player.create_order(battle.available_switches[0])
    return None
