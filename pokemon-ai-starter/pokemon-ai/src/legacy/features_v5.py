# src/features.py
# v5 feature encoder for singles.
# - Continuous obs vector (np.float32) + integer entity_ids (np.int32) for nn.Embedding
# - No hash bucketing — categorical IDs go to entity_ids side-channel
# - Action mask side-channel: 9-way legality (4 moves + 5 switches)
# - Revealed-only matchup bits (no omniscience)
#
# Public API:
#   featurize(battle, ...) -> (obs: np.ndarray[float32], entity_ids: np.ndarray[int32])
#   action_mask(battle) -> (mask, moves_meta, switches_meta)
#   make_obs_mask_and_slots(battle, ...) -> (obs, mask, ctx, move_slots, switch_slots, entity_ids, move_ids, switch_ids)

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from poke_env.battle import SideCondition, Weather, Field, Status
from poke_env.battle.effect import Effect

# =========================
# Config / dimensions
# =========================

N_TYPES = 19  # poke-env uses modern type set; keep 19 for gen-agnostic encoding

# Bench slots per side (singles)
MAX_BENCH = 5

# v2 observation additions
N_STATS = 6                    # HP, Atk, Def, SpA, SpD, Spe
STAT_NORM = 255.0              # normalization divisor for stats (max pokemon stat ~255)
MOVE_COMPACT_DIM = 23          # type(19) + bp(1) + cat(2) + prio(1)
MAX_MOVES = 4                  # max moves per pokemon

# --- Opponent last-turn ctx packing (fixed sizes) ---
OPP_LAST_KIND_DIM = 3           # NONE, MOVE, SWITCH
OPP_LAST_PAYLOAD_DIM = 12       # v5: no hashes. move: bp+acc+prio+flinch+stab+status(7)=12, switch: hp+status(7)+pad(4)=12
OPP_LAST_CTX_DIM = OPP_LAST_KIND_DIM + OPP_LAST_PAYLOAD_DIM

# v5 entity IDs layout (integer side-channel for nn.Embedding)
# [0-2]   our_active: species, item, ability
# [3-5]   opp_active: species, item, ability
# [6-10]  our_bench_species (5)
# [11-15] opp_bench_species (5)
# [16-20] our_bench_items (5)
# [21-25] opp_bench_items (5)
# [26-30] our_bench_abilities (5)
# [31-35] opp_bench_abilities (5)
# [36-39] opp_active_revealed_moves (4)
# [40-59] opp_bench_revealed_moves (5x4=20)
# [60-79] our_bench_moves (5x4=20)
# [80]    opp_last_move_id
# [81]    opp_last_switch_species
# [82]    our_preparing_move_id
# [83]    opp_preparing_move_id
N_ENTITY_IDS = 84

# --- Vocab (lazy-loaded) ---
_vocab = None
def _get_vocab():
    global _vocab
    if _vocab is None:
        from vocab import Vocab
        _vocab = Vocab.load()
    return _vocab

# =========================
# Small helpers
# =========================

_TYPES = [
    "NORMAL","FIRE","WATER","ELECTRIC","GRASS","ICE","FIGHTING","POISON","GROUND","FLYING",
    "PSYCHIC","BUG","ROCK","GHOST","DRAGON","DARK","STEEL","FAIRY","???"
]
_TYPE_TO_IDX = {t:i for i,t in enumerate(_TYPES)}

_STATUS = ["NONE", "BRN", "PAR", "PSN", "TOX", "SLP", "FRZ"]
_STATUS_TO_IDX = {s:i for i,s in enumerate(_STATUS)}

def _one_hot(idx: Optional[int], size: int) -> List[float]:
    v = [0.0]*size
    if idx is not None and 0 <= idx < size:
        v[idx] = 1.0
    return v

def _norm01(x: float, lo: float, hi: float) -> float:
    if hi <= lo: return 0.0
    return float(max(0.0, min(1.0, (x - lo) / (hi - lo))))

def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _status_one_hot(st) -> List[float]:
    if st is None:
        return _one_hot(0, len(_STATUS))  # NONE
    # Handle poke-env Status enum
    name = st.name if hasattr(st, "name") else str(st).upper()
    if name.startswith("TOX"):
        name = "TOX"
    elif name.startswith("PSN"):
        name = "PSN"
    return _one_hot(_STATUS_TO_IDX.get(name, 0), len(_STATUS))

def _types_one_hot(types) -> List[float]:
    v = [0.0]*N_TYPES
    for t in types or []:
        name = t.name if hasattr(t, "name") else str(t).upper()
        idx = _TYPE_TO_IDX.get(name, None)
        if idx is not None:
            v[idx] = 1.0
    return v
    
def _revealed_move_type_hist(poke, n_types=N_TYPES) -> List[float]:
    """Histogram over visible move TYPES (not ids) for this Pokemon."""
    hist = [0.0]*n_types
    try:
        moves = getattr(poke, "moves", None)
        if not moves: return hist
        for mv in moves.values():
            mt = getattr(mv, "type", None)
            name = mt.name if hasattr(mt, "name") else str(mt).upper()
            if name in _TYPE_TO_IDX:
                hist[_TYPE_TO_IDX[name]] = 1.0
    except Exception:
        pass
    return hist

# Key volatile statuses that change battle strategy (15 effects per Pokemon)
_VOLATILE_EFFECTS = [
    Effect.CONFUSION, Effect.ENCORE, Effect.TAUNT, Effect.DISABLE,
    Effect.SUBSTITUTE, Effect.LEECH_SEED, Effect.YAWN, Effect.CURSE,
    Effect.TORMENT, Effect.PROTECT, Effect.FOCUS_ENERGY,
    Effect.TRAPPED, Effect.PARTIALLY_TRAPPED,
    Effect.PERISH0, Effect.PERISH1, Effect.PERISH2, Effect.PERISH3,
]
N_VOLATILE = len(_VOLATILE_EFFECTS)  # 17 (was 15, added PERISH0 + PERISH3 as separate slots)

def _volatile_status_bits(poke) -> List[float]:
    """Encode key volatile statuses as presence bits. Returns N_VOLATILE floats."""
    bits = [0.0] * N_VOLATILE
    if poke is None:
        return bits
    try:
        effects = poke.effects  # Dict[Effect, int]
        if not effects:
            return bits
        for i, eff in enumerate(_VOLATILE_EFFECTS):
            if eff in effects:
                bits[i] = 1.0
    except Exception:
        pass
    return bits

def _tera_type_encoding(poke) -> List[float]:
    """Encode tera state: [is_terastallized] + tera_type one-hot (N_TYPES). Returns N_TYPES+1 floats."""
    vec = [0.0] * (N_TYPES + 1)
    if poke is None:
        return vec
    try:
        if poke.is_terastallized:
            vec[0] = 1.0
            tt = poke.tera_type
            if tt is not None:
                name = tt.name if hasattr(tt, "name") else str(tt).upper()
                idx = _TYPE_TO_IDX.get(name, None)
                if idx is not None:
                    vec[1 + idx] = 1.0
    except Exception:
        pass
    return vec

def _active_combat_state(poke) -> List[float]:
    """Encode combat-relevant state: first_turn, must_recharge, preparing, protect_counter, status_counter.
    Returns 5 floats."""
    if poke is None:
        return [0.0] * 5
    try:
        first_turn = 1.0 if getattr(poke, "first_turn", False) else 0.0
        must_recharge = 1.0 if getattr(poke, "must_recharge", False) else 0.0
        preparing = 1.0 if getattr(poke, "preparing", False) else 0.0
        protect_ctr = _norm01(float(getattr(poke, "protect_counter", 0) or 0), 0.0, 4.0)
        status_ctr = _norm01(float(getattr(poke, "status_counter", 0) or 0), 0.0, 16.0)
        return [first_turn, must_recharge, preparing, protect_ctr, status_ctr]
    except Exception:
        return [0.0] * 5

N_COMBAT_STATE = 5  # first_turn, must_recharge, preparing, protect_counter, status_counter

def _boosts_block(poke) -> List[float]:
    # attack, defense, spa, spd, speed, accuracy, evasion in [-6..+6] -> [0..1]
    names = ["atk","def","spa","spd","spe","accuracy","evasion"]
    out = []
    boosts = getattr(poke, "boosts", {}) or {}
    for k in names:
        v = float(boosts.get(k, 0))
        out.append(_norm01(v, -6.0, 6.0))
    return out

def _hp_frac(poke) -> float:
    try:
        cond = getattr(poke, "current_hp_fraction", None)
        if cond is None:
            # poke.condition like "123/321" — poke-env often gives current_hp_fraction, else derive
            c = str(getattr(poke, "condition", ""))
            if "/" in c:
                num, den = c.split("/", 1)
                num = float(num) if num.isdigit() else 0.0
                den = float(den) if den.isdigit() and float(den) > 0 else 1.0
                return _clip01(num/den)
            return 0.0
        return _clip01(float(cond))
    except Exception:
        return 0.0

def _one_hot_len(idx: int, size: int) -> List[float]:
    v = [0.0]*size
    if 0 <= idx < size: v[idx] = 1.0
    return v

def _mini_move_payload_from_meta(meta: dict) -> List[float]:
    """Compress move meta into a 12-dim mini vector (v5: no hashes, IDs via entity_ids)."""
    bp01   = _norm01(float(meta.get("base_power", 0.0)), 0.0, 250.0)
    acc01  = float(meta.get("accuracy01", 1.0))
    prio_n = float(meta.get("priority_norm", 0.0))
    flinch01 = float(meta.get("flinch01", 0.0))
    stab   = 1.0 if meta.get("stab", False) else 0.0
    status_to = _status_to_onehot_name(meta.get("status_to"))
    vec = [bp01, acc01, prio_n, flinch01, stab] + status_to
    # 5 + 7 = 12
    assert len(vec) == OPP_LAST_PAYLOAD_DIM
    return vec

def _mini_move_id_from_meta(meta: dict) -> int:
    """Extract move integer ID from meta (for entity_ids)."""
    return _get_vocab().move(meta.get("id"))

def _mini_switch_id_from_meta(meta: dict) -> int:
    """Extract species integer ID from switch meta (for entity_ids)."""
    return _get_vocab().species(meta.get("species"))

def _mini_switch_payload_from_meta(meta: dict) -> List[float]:
    """Switch meta into 12-dim vector (v5: no hash, species ID via entity_ids)."""
    hp_frac = float(meta.get("hp_frac", 0.0))
    status7 = _status_to_onehot_name(meta.get("status"))
    vec8 = [hp_frac] + status7  # 1 + 7 = 8
    # pad to 12
    vec = vec8 + [0.0] * (OPP_LAST_PAYLOAD_DIM - len(vec8))
    return vec

def _opp_last_action_from_logs(battle) -> tuple[str, dict]:
    """
    Returns ("NONE"|"MOVE"|"SWITCH", meta_dict) for opponent's last action.
    Uses poke-env observations API to access previous turn events.
    """
    try:
        prev_turn = battle.turn - 1
        if prev_turn < 1 or prev_turn not in battle.observations:
            return "NONE", {}

        events = battle.observations[prev_turn].events
        role = battle.player_role
        if not role:
            return "NONE", {}
        opp_role = "p2" if role == "p1" else "p1"

        for event in events:
            if len(event) < 3:
                continue
            who = event[2].split(":")[0].strip()
            if not who.startswith(opp_role):
                continue

            if event[1] == "move":
                mv_name = event[3].strip().lower().replace(" ", "") if len(event) > 3 else ""
                move_obj = None
                opp = battle.opponent_active_pokemon
                if opp and opp.moves:
                    for mv in opp.moves.values():
                        if mv.id == mv_name or mv.id.replace("-", "") == mv_name:
                            move_obj = mv
                            break
                if move_obj is not None:
                    meta = _project_move_flags(move_obj)
                else:
                    meta = {"id": mv_name, "type": None, "base_power": 0, "accuracy01": 1.0, "priority_norm": 0.0}
                return "MOVE", meta

            if event[1] == "switch":
                p = battle.opponent_active_pokemon
                meta = {
                    "species": p.species if p else None,
                    "types": [t.name if hasattr(t, "name") else str(t).upper() for t in (p.types if p else [])],
                    "hp_frac": _hp_frac(p),
                    "status": p.status.name if p and p.status else None,
                }
                return "SWITCH", meta

        return "NONE", {}
    except Exception:
        return "NONE", {}

# =========================
# v2: Stats, revealed moves, bench items/abilities, alive counts
# =========================

_STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]

def _get_sorted_bench(team) -> list:
    """Extract non-active bench pokemon sorted by species name.
    Shared helper to guarantee consistent ordering across all bench encodings."""
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

def _encode_stats(poke, use_base: bool = False) -> List[float]:
    """6 dims: HP/Atk/Def/SpA/SpD/Spe normalized by STAT_NORM.
    use_base=True reads base_stats (for opponents), False reads actual stats."""
    if poke is None:
        return [0.0] * N_STATS
    try:
        stats = poke.base_stats if use_base else poke.stats
        if not isinstance(stats, dict):
            return [0.0] * N_STATS
        return [min(float(stats.get(k, 0) or 0) / STAT_NORM, 1.5) for k in _STAT_KEYS]
    except Exception:
        return [0.0] * N_STATS

def _encode_move_compact(move) -> List[float]:
    """23-dim compact encoding: type_one_hot(19) + bp(1) + category(2) + priority(1)."""
    if move is None:
        return [0.0] * MOVE_COMPACT_DIM
    try:
        # type one-hot (19)
        type_oh = [0.0] * N_TYPES
        mt = getattr(move, "type", None)
        if mt:
            name = mt.name if hasattr(mt, "name") else str(mt).upper()
            idx = _TYPE_TO_IDX.get(name, None)
            if idx is not None:
                type_oh[idx] = 1.0
        # base power normalized
        bp = _norm01(float(getattr(move, "base_power", 0) or 0), 0.0, 250.0)
        # category (physical / special)
        cat = getattr(move, "category", None)
        cat_name = cat.name if hasattr(cat, "name") else str(cat or "").upper()
        cat_phys = 1.0 if cat_name == "PHYSICAL" else 0.0
        cat_spec = 1.0 if cat_name == "SPECIAL" else 0.0
        # priority normalized to [-1, 1]
        prio = max(-1.0, min(1.0, float(getattr(move, "priority", 0) or 0) / 3.0))
        return type_oh + [bp, cat_phys, cat_spec, prio]
    except Exception:
        return [0.0] * MOVE_COMPACT_DIM

def _encode_pokemon_moves_compact(poke) -> List[float]:
    """4 move slots × 23 dims = 92 dims. Encodes all known moves for a pokemon.
    Unrevealed/empty slots are zeros."""
    result = []
    moves_list = []
    if poke is not None:
        try:
            moves_list = list((poke.moves or {}).values())
        except Exception:
            pass
    for i in range(MAX_MOVES):
        if i < len(moves_list):
            result.extend(_encode_move_compact(moves_list[i]))
        else:
            result.extend([0.0] * MOVE_COMPACT_DIM)
    return result

def _encode_bench_stats(bench: list, use_base: bool = False) -> List[float]:
    """5 bench × 6 stats = 30 dims."""
    result = []
    for i in range(MAX_BENCH):
        if i < len(bench):
            result.extend(_encode_stats(bench[i], use_base=use_base))
        else:
            result.extend([0.0] * N_STATS)
    return result

def _encode_bench_moves(bench: list) -> List[float]:
    """5 bench × 4 moves × 23 dims = 460 dims."""
    result = []
    for i in range(MAX_BENCH):
        if i < len(bench):
            result.extend(_encode_pokemon_moves_compact(bench[i]))
        else:
            result.extend([0.0] * (MAX_MOVES * MOVE_COMPACT_DIM))
    return result

def _alive_counts(battle) -> List[float]:
    """[our_alive/6, opp_alive/6]. For opponents, unrevealed mons are assumed alive."""
    try:
        our_alive = sum(1 for p in battle.team.values() if not p.fainted)
        opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)
        opp_alive = 6 - opp_fainted  # revealed alive + unrevealed (assumed alive)
        return [our_alive / 6.0, opp_alive / 6.0]
    except Exception:
        return [0.5, 0.5]

def encode_opp_last_ctx(kind: str, meta: dict) -> List[float]:
    kind_map = {"NONE":0, "MOVE":1, "SWITCH":2}
    k = _one_hot_len(kind_map.get(kind, 0), OPP_LAST_KIND_DIM)
    if kind == "MOVE":
        payload = _mini_move_payload_from_meta(meta)
    elif kind == "SWITCH":
        payload = _mini_switch_payload_from_meta(meta)
    else:
        payload = [0.0]*OPP_LAST_PAYLOAD_DIM
    return k + payload

# === Slot encoders (put near other helpers) ===

def _status_to_onehot_name(name) -> list[float]:
    """Convert a status name (str or Status enum) to one-hot."""
    if name is None:
        s = "NONE"
    elif hasattr(name, "name"):
        s = name.name
    else:
        s = str(name).upper()
    if s.startswith("TOX"): s = "TOX"
    if s.startswith("PSN"): s = "PSN"
    v = [0.0]*len(_STATUS)
    try:
        v[_STATUS.index(s)] = 1.0
    except Exception:
        v[0] = 1.0
    return v

def encode_move_slot_vector(meta: dict) -> list[float]:
    # Numbers (normalized where needed)
    base_power = float(meta.get("base_power", 0.0))
    acc01      = float(meta.get("accuracy01", 1.0))
    prio_n     = float(meta.get("priority_norm", 0.0))  # [-1..1]
    drain01    = float(meta.get("drain01", 0.0))
    recoil01   = float(meta.get("recoil01", 0.0))
    heal01     = float(meta.get("heal01", 0.0))
    multihit   = float(meta.get("multihit_est", 1.0))
    flinch01   = float(meta.get("flinch01", 0.0))
    crit_boost = float(meta.get("crit_boost", 0.0))
    pp         = float(meta.get("pp", 0.0))
    disabled   = 1.0 if meta.get("disabled", False) else 0.0
    recharge   = 1.0 if meta.get("recharge", False) else 0.0
    stab       = 1.0 if meta.get("stab", False) else 0.0

    # Booleans → bits
    contact = 1.0 if meta.get("contact") else 0.0
    protect_blocked = 1.0 if meta.get("protect_blocked") else 0.0
    sound   = 1.0 if meta.get("sound") else 0.0
    punch   = 1.0 if meta.get("punch") else 0.0
    bite    = 1.0 if meta.get("bite") else 0.0
    powder  = 1.0 if meta.get("powder") else 0.0
    phaze   = 1.0 if meta.get("phaze") else 0.0
    pivot   = 1.0 if meta.get("pivot") else 0.0
    trap    = 1.0 if meta.get("trap") else 0.0

    # Field setters
    set_sr = 1.0 if meta.get("set_sr") else 0.0
    set_spikes = 1.0 if meta.get("set_spikes") else 0.0
    set_tspikes = 1.0 if meta.get("set_tspikes") else 0.0
    set_web = 1.0 if meta.get("set_web") else 0.0
    set_sun = 1.0 if meta.get("set_sun") else 0.0
    set_rain = 1.0 if meta.get("set_rain") else 0.0
    set_sand = 1.0 if meta.get("set_sand") else 0.0
    set_snow = 1.0 if meta.get("set_snow") else 0.0
    set_terrain_el = 1.0 if meta.get("set_terrain_el") else 0.0
    set_terrain_gr = 1.0 if meta.get("set_terrain_gr") else 0.0
    set_terrain_ps = 1.0 if meta.get("set_terrain_ps") else 0.0
    set_terrain_ms = 1.0 if meta.get("set_terrain_ms") else 0.0

    # Categoricals (v5: no move ID hash — ID goes to entity_ids)
    type_oh   = _types_one_hot([meta.get("type")] if meta.get("type") else [])
    status_to = _status_to_onehot_name(meta.get("status_to"))

    # Base power normalization: 0..250 → [0..1] (v5: raised cap for Explosion etc.)
    bp01 = _norm01(base_power, 0.0, 250.0)

    # NEW: Move category (physical/special/status)
    cat_phys = 1.0 if meta.get("cat_physical") else 0.0
    cat_spec = 1.0 if meta.get("cat_special") else 0.0
    cat_stat = 1.0 if meta.get("cat_status") else 0.0

    # NEW: Self-boost and target boost (7 stats each)
    self_boost_v = meta.get("self_boost", [0.5]*7)
    if not isinstance(self_boost_v, list) or len(self_boost_v) != 7:
        self_boost_v = [0.5]*7
    target_boost_v = meta.get("target_boost", [0.5]*7)
    if not isinstance(target_boost_v, list) or len(target_boost_v) != 7:
        target_boost_v = [0.5]*7

    # NEW: Screen setters
    set_reflect = 1.0 if meta.get("set_reflect") else 0.0
    set_lscreen = 1.0 if meta.get("set_lscreen") else 0.0
    set_aveil   = 1.0 if meta.get("set_aveil") else 0.0

    # NEW: Volatile status inflicted
    vs_confuse   = 1.0 if meta.get("vs_confuse") else 0.0
    vs_taunt     = 1.0 if meta.get("vs_taunt") else 0.0
    vs_encore    = 1.0 if meta.get("vs_encore") else 0.0
    vs_disable   = 1.0 if meta.get("vs_disable") else 0.0
    vs_leechseed = 1.0 if meta.get("vs_leechseed") else 0.0
    vs_yawn      = 1.0 if meta.get("vs_yawn") else 0.0
    vs_sub       = 1.0 if meta.get("vs_sub") else 0.0
    vs_torment   = 1.0 if meta.get("vs_torment") else 0.0

    # NEW: Protect / breaks protect / self-destruct / fixed damage
    is_protect_m  = 1.0 if meta.get("is_protect") else 0.0
    is_stalling_m = 1.0 if meta.get("is_stalling") else 0.0
    breaks_prot_m = 1.0 if meta.get("breaks_protect") else 0.0
    self_destr    = 1.0 if meta.get("self_destruct") else 0.0
    fixed_dmg     = 1.0 if meta.get("has_fixed_damage") else 0.0

    # NEW: Secondary effect chances
    sec_burn_v   = float(meta.get("sec_burn", 0.0))
    sec_freeze_v = float(meta.get("sec_freeze", 0.0))
    sec_para_v   = float(meta.get("sec_para", 0.0))
    sec_poison_v = float(meta.get("sec_poison", 0.0))
    sec_flinch_v = float(meta.get("sec_flinch", 0.0))

    # Hazard removal
    clears_haz   = 1.0 if meta.get("clears_hazards") else 0.0

    # Move target (6-way one-hot)
    target_oh_v = meta.get("target_oh", [0.0]*6)
    if not isinstance(target_oh_v, list) or len(target_oh_v) != 6:
        target_oh_v = [0.0]*6

    # Ignore ability / immunity
    ign_abil = 1.0 if meta.get("ignore_ability") else 0.0
    ign_immu = 1.0 if meta.get("ignore_immunity") else 0.0

    return [
        bp01, acc01, prio_n, drain01, recoil01, heal01, multihit, flinch01, crit_boost,
        pp/40.0, disabled, recharge, stab,
        contact, protect_blocked, sound, punch, bite, powder,
        phaze, pivot, trap,
        set_sr, set_spikes, set_tspikes, set_web,
        set_sun, set_rain, set_sand, set_snow,
        set_terrain_el, set_terrain_gr, set_terrain_ps, set_terrain_ms,
        clears_haz,
        cat_phys, cat_spec, cat_stat,
        set_reflect, set_lscreen, set_aveil,
        vs_confuse, vs_taunt, vs_encore, vs_disable, vs_leechseed, vs_yawn, vs_sub, vs_torment,
        is_protect_m, is_stalling_m, breaks_prot_m,
        self_destr, fixed_dmg,
        sec_burn_v, sec_freeze_v, sec_para_v, sec_poison_v, sec_flinch_v,
        ign_abil, ign_immu,
    ] + self_boost_v + target_boost_v + target_oh_v + type_oh + status_to

def encode_switch_slot_vector(meta: dict) -> list[float]:
    """v5: no species hash — species ID goes to entity_ids."""
    types   = meta.get("types") or []
    hp_frac = float(meta.get("hp_frac", 0.0))
    status  = meta.get("status")
    types_oh = _types_one_hot(types)
    weight_n = float(meta.get("weight_norm", 0.0))
    return types_oh + [hp_frac] + _status_to_onehot_name(status) + [weight_n]

SWITCH_SLOT_DIM = N_TYPES + 1 + len(_STATUS) + 1  # 19 + 1 + 7 + 1 = 28

def _move_slot_dim() -> int:
    """Compute expected move slot vector dimension (v5: no id_hash)."""
    # 35 original (incl clears_hazards) + 24 new flags + 14 boosts
    # + 2 ignore (ability/immunity) + 6 target one-hot + N_TYPES + len(_STATUS)
    return 35 + 24 + 14 + 2 + 6 + N_TYPES + len(_STATUS)

def encode_move_and_switch_slots(moves_meta: list[dict], switches_meta: list[dict]) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Returns (move_slots, switch_slots, move_ids, switch_ids).
    move_ids/switch_ids are integer IDs for nn.Embedding."""
    v = _get_vocab()
    _mdim = _move_slot_dim()
    mv = [encode_move_slot_vector(m) if m else [0.0] * _mdim for m in moves_meta]
    sw = [encode_switch_slot_vector(s) if s else [0.0] * SWITCH_SLOT_DIM for s in switches_meta]
    # Extract IDs
    move_ids = [v.move(m.get("id")) if m else 0 for m in moves_meta]
    switch_ids = [v.species(s.get("species")) if s else 0 for s in switches_meta]
    # ensure fixed 4/5 slots
    while len(mv) < 4: mv.append([0.0] * _mdim)
    while len(sw) < 5: sw.append([0.0] * SWITCH_SLOT_DIM)
    while len(move_ids) < 4: move_ids.append(0)
    while len(switch_ids) < 5: switch_ids.append(0)
    return np.asarray(mv, dtype=np.float32), np.asarray(sw, dtype=np.float32), move_ids, switch_ids

# =========================
# Move projection helpers
# =========================

def _coerce_bool(x) -> bool:
    try:
        return bool(x)
    except Exception:
        return False

def _coerce_int(x, default=0) -> int:
    try:
        return int(x if x is not None else default)
    except Exception:
        return default

def _coerce_float(x, default=0.0) -> float:
    try:
        return float(x if x is not None else default)
    except Exception:
        return default

def _compute_stab(m, poke_types) -> bool:
    """Check if move gets STAB (Same Type Attack Bonus) from the user's types."""
    if poke_types is None or m.type is None:
        return False
    move_type_name = m.type.name if hasattr(m.type, "name") else str(m.type).upper()
    for pt in poke_types:
        if pt is None:
            continue
        pt_name = pt.name if hasattr(pt, "name") else str(pt).upper()
        if move_type_name == pt_name:
            return True
    return False

def _project_move_flags(m, poke_types=None) -> dict:
    """
    Convert a poke-env Move object into a compact, model-friendly dict.
    Only uses attributes visible in the object; no hard-coded dex table.
    All values are primitives (int/float/bool/str).
    poke_types: optional list of PokemonType for the user (to compute STAB).
    """
    # Basic numeric
    # poke-env Move.accuracy already returns 0-1 (e.g., 0.7 for 70%, 1.0 for 100%)
    acc_raw = getattr(m, "accuracy", None)
    if acc_raw is True or acc_raw is None:
        acc01 = 1.0  # never-miss moves
    else:
        acc01 = max(0.0, min(1.0, float(acc_raw)))
    prio = _coerce_int(getattr(m, "priority", 0), default=0)
    prio_n = max(-3, min(3, prio))  # clamp; your model can learn sign/magnitude

    # End-of-turn resource-y stuff
    # poke-env returns drain/recoil/heal as floats 0-1 (e.g., 0.5 for Giga Drain)
    drain = _coerce_float(getattr(m, "drain", 0), default=0.0)
    recoil = _coerce_float(getattr(m, "recoil", 0), default=0.0)
    heal = _coerce_float(getattr(m, "heal", 0), default=0.0)
    recharge = _coerce_bool(getattr(m, "recharge", getattr(m, "recharge_turn", False)))
    multihit = getattr(m, "multihit", None)
    if isinstance(multihit, (tuple, list)) and len(multihit) == 2:
        multi_lo, multi_hi = int(multihit[0]), int(multihit[1])
        multihit_est = float((multi_lo + multi_hi) / 2.0)
    else:
        multihit_est = float(_coerce_int(multihit or 1, default=1))

    # Contact / protect / sound / punch / bite / powder — check Move.flags (a set)
    _flags = getattr(m, "flags", set()) or set()
    contact = "contact" in _flags
    protect_blocked = "protect" in _flags
    sound = "sound" in _flags
    punch = "punch" in _flags
    bite  = "bite" in _flags
    powder = "powder" in _flags
    crit_ratio = _coerce_int(getattr(m, "crit_ratio", 0), default=0)  # +crit stages

    # Status / boosts effects — presence only (we don't need exact magnitude to start)
    status_to = None
    st = getattr(m, "status", None)
    if st:
        status_to = str(st).upper()

    # Flinch: extract from secondary effects (Move.flinch_chance doesn't exist in poke-env)
    flinch01 = 0.0
    _sec = getattr(m, "secondary", None) or []
    _sec_list = _sec if isinstance(_sec, list) else [_sec]
    for _s in _sec_list:
        if isinstance(_s, dict) and ("flinch" in str(_s.get("volatileStatus", ""))):
            flinch01 = max(flinch01, _coerce_float(_s.get("chance", 100), 100.0) / 100.0)

    # Phaze / pivot / trap
    phaze  = _coerce_bool(getattr(m, "force_switch", getattr(m, "phaze", False)))
    pivot  = _coerce_bool(getattr(m, "self_switch", getattr(m, "pivot", False)))

    # Field setters: hazards / weather / terrain (string matching on move name/id/type fields we do see)
    mid = str(getattr(m, "id", "") or "").lower()
    mname = str(getattr(m, "name", "") or "").lower()

    # poke-env Move has no "trap" attr; trapping moves set volatile_status to "partiallytrapped" etc.
    _vs_raw = getattr(m, "volatile_status", None)
    _vs_str = (_vs_raw.name if hasattr(_vs_raw, "name") else str(_vs_raw or "")).lower()
    trap = _vs_str in ("partiallytrapped", "no_retreat", "octolock") or mid in ("meanlook", "block", "spiderweb", "anchorshot", "spiritshackle", "thousandwaves", "jawlock")

    sets_sr     = ("stealthrock" in (mid or mname))
    sets_spikes = (mid == "spikes" or mname == "spikes")
    sets_tspikes= (mid == "toxicspikes" or mname == "toxic spikes")
    sets_web    = (mid == "stickyweb" or mname == "sticky web")

    sets_sun  = (mid == "sunnyday" or mname == "sunny day")
    sets_rain = (mid == "raindance" or mname == "rain dance")
    sets_sand = (mid == "sandstorm" or mname == "sandstorm")
    sets_snow = (mid == "snowscape" or mname == "hail" or mname == "snowscape")

    sets_electric = (mid == "electricterrain" or mname == "electric terrain")
    sets_grassy   = (mid == "grassyterrain" or mname == "grassy terrain")
    sets_psychic  = (mid == "psychicterrain" or mname == "psychic terrain")
    sets_misty    = (mid == "mistyterrain" or mname == "misty terrain")

    # Hazard removal moves
    clears_hazards = mid in ("defog", "rapidspin", "courtchange", "mortalspin", "tidyup")

    # === NEW: Move category (physical / special / status) ===
    cat = getattr(m, "category", None)
    cat_name = cat.name if hasattr(cat, "name") else str(cat or "").upper()
    cat_physical = 1.0 if cat_name == "PHYSICAL" else 0.0
    cat_special  = 1.0 if cat_name == "SPECIAL" else 0.0
    cat_status   = 1.0 if cat_name == "STATUS" else 0.0

    # === Self-boost vs target-boost ===
    # poke-env semantics:
    #   m.self_boost: for attack-and-self-boost moves (Close Combat's def/spd drops, PUP via secondary)
    #   m.boosts: for pure boost moves (Swords Dance, Charm)
    #   m.target: SELF means the boosts apply to the user, NORMAL/etc means target
    _BOOST_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

    # Gather all boost sources
    self_boost_raw = dict(getattr(m, "self_boost", None) or {})
    boosts_raw = dict(getattr(m, "boosts", None) or {})
    target_name = str(getattr(m, "target", "")).upper()

    # If move targets self (Swords Dance, Dragon Dance, etc.), boosts go to self
    if "SELF" in target_name and boosts_raw:
        for k, v in boosts_raw.items():
            self_boost_raw[k] = self_boost_raw.get(k, 0) + v
        boosts_raw = {}  # clear so they don't also go to target

    self_boost_vals = [_norm01(float(self_boost_raw.get(k, 0)), -3.0, 3.0) for k in _BOOST_KEYS] if self_boost_raw else [0.5] * 7
    target_boost_vals = [_norm01(float(boosts_raw.get(k, 0)), -3.0, 3.0) for k in _BOOST_KEYS] if boosts_raw else [0.5] * 7

    # === NEW: Screen-setting (side_condition property) ===
    sc = getattr(m, "side_condition", None)
    sc_name = sc.name if hasattr(sc, "name") else str(sc or "").upper()
    sets_reflect = 1.0 if "REFLECT" in sc_name else 0.0
    sets_lscreen = 1.0 if "LIGHT_SCREEN" in sc_name else 0.0
    sets_aveil   = 1.0 if "AURORA_VEIL" in sc_name else 0.0

    # === NEW: Volatile status the move inflicts ===
    vs = getattr(m, "volatile_status", None)
    vs_name = vs.name if hasattr(vs, "name") else str(vs or "").upper()
    vs_confuse = 1.0 if "CONFUSION" in vs_name else 0.0
    vs_taunt   = 1.0 if mid == "taunt" else 0.0
    vs_encore  = 1.0 if mid == "encore" else 0.0
    vs_disable = 1.0 if mid == "disable" else 0.0
    vs_leechseed = 1.0 if mid == "leechseed" else 0.0
    vs_yawn    = 1.0 if mid == "yawn" else 0.0
    vs_sub     = 1.0 if mid == "substitute" else 0.0
    vs_torment = 1.0 if mid == "torment" else 0.0

    # === NEW: Protect / stalling / breaks protect ===
    is_protect = _coerce_bool(getattr(m, "is_protect_move", False))
    is_stalling = _coerce_bool(getattr(m, "stalling_move", False))
    breaks_prot = _coerce_bool(getattr(m, "breaks_protect", False))

    # === NEW: Self-destruct, fixed damage ===
    self_destruct_val = _coerce_bool(getattr(m, "self_destruct", False))
    raw_damage = getattr(m, "damage", 0)
    # Moves like Seismic Toss / Night Shade have damage="level" (string, not int)
    if isinstance(raw_damage, str) and raw_damage.lower() == "level":
        has_fixed_damage = 1.0
    else:
        fixed_damage = _coerce_int(raw_damage, default=0)
        has_fixed_damage = 1.0 if (isinstance(fixed_damage, (int, float)) and fixed_damage > 0) else 0.0

    # === NEW: Secondary effect chances (burn/freeze/para from secondary) ===
    secondary = getattr(m, "secondary", None) or []
    sec_burn = 0.0; sec_freeze = 0.0; sec_para = 0.0; sec_poison = 0.0; sec_flinch = 0.0
    sec_list = secondary if isinstance(secondary, list) else [secondary]
    for sec in sec_list:
        if not isinstance(sec, dict):
            continue
        chance = _coerce_float(sec.get("chance", 100), 100.0) / 100.0
        sec_status = sec.get("status", "")
        if hasattr(sec_status, "name"):
            sec_status = sec_status.name
        sec_status = str(sec_status).upper()
        if "BRN" in sec_status: sec_burn = max(sec_burn, chance)
        if "FRZ" in sec_status: sec_freeze = max(sec_freeze, chance)
        if "PAR" in sec_status: sec_para = max(sec_para, chance)
        if "PSN" in sec_status or "TOX" in sec_status: sec_poison = max(sec_poison, chance)
        if sec.get("volatileStatus") == "flinch" or "flinch" in str(sec.get("volatileStatus", "")):
            sec_flinch = max(sec_flinch, chance)

    # === Move target category (one-hot: SELF, NORMAL, ALL_ADJACENT, FOE_SIDE, ALLY_SIDE, OTHER) ===
    _TARGET_MAP = {"SELF": 0, "NORMAL": 1, "ALL_ADJACENT": 2, "ALL_ADJACENT_FOES": 2,
                   "FOE_SIDE": 3, "ALLY_SIDE": 4}
    target_raw = getattr(m, "target", None)
    target_str = target_raw.name if hasattr(target_raw, "name") else str(target_raw or "").upper()
    target_idx = _TARGET_MAP.get(target_str, 5)  # 5 = OTHER
    target_oh = [0.0] * 6
    target_oh[target_idx] = 1.0

    # === ignore_ability / ignore_immunity ===
    ignore_ability = 1.0 if _coerce_bool(getattr(m, "ignore_ability", False)) else 0.0
    ignore_immunity_raw = getattr(m, "ignore_immunity", False)
    ignore_immunity = 1.0 if (ignore_immunity_raw and ignore_immunity_raw is not False) else 0.0

    return {
        # Keep what you already log
        "id": getattr(m, "id", None),
        "base_power": _coerce_int(getattr(m, "base_power", 0), default=0),
        "stab": _compute_stab(m, poke_types),
        "priority": prio,
        "priority_norm": prio_n / 3.0,     # [-1..+1]
        "pp": _coerce_int(getattr(m, "current_pp", getattr(m, "pp", 0)), default=0),
        "disabled": _coerce_bool(getattr(m, "disabled", False)),
        "type": (m.type.name if hasattr(m.type, "name") else str(m.type).upper()) if m.type else None,

        # Newly added dynamics
        "accuracy01": acc01,
        "contact": contact,
        "protect_blocked": protect_blocked,
        "sound": sound, "punch": punch, "bite": bite, "powder": powder,
        "crit_boost": crit_ratio,          # 0..n
        "drain01": max(0.0, min(1.0, drain)),
        "recoil01": max(0.0, min(1.0, abs(recoil))),
        "heal01": max(0.0, min(1.0, heal)),
        "recharge": recharge,
        "multihit_est": float(multihit_est),
        "flinch01": flinch01,
        "status_to": status_to,            # e.g., BRN / PAR / PSN / TOX / SLP / FRZ

        # Tactical field effects
        "phaze": phaze,
        "pivot": pivot,
        "trap": trap,

        # Setters (presence bits)
        "set_sr": sets_sr, "set_spikes": sets_spikes, "set_tspikes": sets_tspikes, "set_web": sets_web,
        "set_sun": sets_sun, "set_rain": sets_rain, "set_sand": sets_sand, "set_snow": sets_snow,
        "set_terrain_el": sets_electric, "set_terrain_gr": sets_grassy,
        "set_terrain_ps": sets_psychic,  "set_terrain_ms": sets_misty,
        "clears_hazards": clears_hazards,

        # Move category
        "cat_physical": cat_physical, "cat_special": cat_special, "cat_status": cat_status,
        # Self-boost (7 stats normalized)
        "self_boost": self_boost_vals,
        # Target boosts (7 stats normalized)
        "target_boost": target_boost_vals,
        # Screen setters
        "set_reflect": sets_reflect, "set_lscreen": sets_lscreen, "set_aveil": sets_aveil,
        # Volatile status inflicted
        "vs_confuse": vs_confuse, "vs_taunt": vs_taunt, "vs_encore": vs_encore,
        "vs_disable": vs_disable, "vs_leechseed": vs_leechseed, "vs_yawn": vs_yawn,
        "vs_sub": vs_sub, "vs_torment": vs_torment,
        # Protect / breaks protect
        "is_protect": is_protect, "is_stalling": is_stalling, "breaks_protect": breaks_prot,
        # Self-destruct, fixed damage
        "self_destruct": self_destruct_val, "has_fixed_damage": has_fixed_damage,
        # Secondary effect chances
        "sec_burn": sec_burn, "sec_freeze": sec_freeze, "sec_para": sec_para,
        "sec_poison": sec_poison, "sec_flinch": sec_flinch,
        # Move target (6-way one-hot)
        "target_oh": target_oh,
        # Ignore ability / immunity
        "ignore_ability": ignore_ability, "ignore_immunity": ignore_immunity,
    }

# =========================
# Heuristic moved-first (fallback)
# =========================

def _eff_speed_with_conditions(poke, side_conditions) -> Optional[float]:
    """Best-effort effective speed from visible info: stats.spe, Tailwind, PAR."""
    try:
        stats = poke.stats
        if not isinstance(stats, dict):
            return None
        sp = stats.get("spe", None)
        if sp is None:
            return None
        if side_conditions and SideCondition.TAILWIND in side_conditions:
            sp *= 2
        if poke.status == Status.PAR:
            sp *= 0.5
        return float(sp)
    except Exception:
        return None

def moved_first_heuristic_bits(battle) -> List[float]:
    """One-hot [we_first, they_first, unknown]. Only use visible cues; else unknown."""
    try:
        our = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if our is None or opp is None:
            return [0.0, 0.0, 1.0]
        our_sp = _eff_speed_with_conditions(our, battle.side_conditions)
        opp_sp = _eff_speed_with_conditions(opp, battle.opponent_side_conditions)
        if our_sp is None or opp_sp is None or our_sp == opp_sp:
            return [0.0, 0.0, 1.0]
        trick_room = Field.TRICK_ROOM in battle.fields
        we_first = our_sp > opp_sp
        if trick_room:
            we_first = not we_first
        return [1.0, 0.0, 0.0] if we_first else [0.0, 1.0, 0.0]
    except Exception:
        return [0.0, 0.0, 1.0]

# =========================
# Revealed-only matchup bits
# =========================

def _type_effectiveness(attacking_type, defending_types) -> float:
    """Tiny hand-rolled chart for SE/NE/NORM checks. Only needs 0, 0.5, 1, 2 granularity."""
    # Minimal matrix — include only what's commonly needed; unknown -> 1.0
    # For brevity, not the full matrix; we only need >1 or <1 flags.
    # You can expand if desired.
    se = {
        "FIRE": {"GRASS","ICE","BUG","STEEL"},
        "WATER": {"FIRE","GROUND","ROCK"},
        "ELECTRIC": {"WATER","FLYING"},
        "GRASS": {"WATER","GROUND","ROCK"},
        "ICE": {"GRASS","GROUND","FLYING","DRAGON"},
        "FIGHTING": {"NORMAL","ICE","ROCK","DARK","STEEL"},
        "POISON": {"GRASS","FAIRY"},
        "GROUND": {"FIRE","ELECTRIC","POISON","ROCK","STEEL"},
        "FLYING": {"GRASS","FIGHTING","BUG"},
        "PSYCHIC": {"FIGHTING","POISON"},
        "BUG": {"GRASS","PSYCHIC","DARK"},
        "ROCK": {"FIRE","ICE","FLYING","BUG"},
        "GHOST": {"PSYCHIC","GHOST"},
        "DRAGON": {"DRAGON"},
        "DARK": {"PSYCHIC","GHOST"},
        "STEEL": {"ICE","ROCK","FAIRY"},
        "FAIRY": {"FIGHTING","DRAGON","DARK"},
    }
    ne = {
        "NORMAL": {"ROCK","STEEL"},
        "FIRE": {"FIRE","WATER","ROCK","DRAGON"},
        "WATER": {"WATER","GRASS","DRAGON"},
        "ELECTRIC": {"ELECTRIC","GRASS","DRAGON"},
        "GRASS": {"FIRE","GRASS","POISON","FLYING","BUG","DRAGON","STEEL"},
        "ICE": {"FIRE","WATER","ICE","STEEL"},
        "FIGHTING": {"POISON","FLYING","PSYCHIC","BUG","FAIRY"},
        "POISON": {"POISON","GROUND","ROCK","GHOST"},
        "GROUND": {"GRASS","BUG"},
        "FLYING": {"ELECTRIC","ROCK","STEEL"},
        "PSYCHIC": {"PSYCHIC","STEEL"},
        "BUG": {"FIRE","FIGHTING","POISON","FLYING","GHOST","STEEL","FAIRY"},
        "ROCK": {"FIGHTING","GROUND","STEEL"},
        "GHOST": {"DARK"},
        "DRAGON": {"STEEL"},
        "DARK": {"FIGHTING","DARK","FAIRY"},
        "STEEL": {"FIRE","WATER","ELECTRIC","STEEL"},
        "FAIRY": {"FIRE","POISON","STEEL"},
    }
    imm = {
        "ELECTRIC": {"GROUND"},
        "GROUND": {"FLYING"},
        "GHOST": {"NORMAL"},
        "NORMAL": {"GHOST"},
        "FIGHTING": {"GHOST"},
        "PSYCHIC": {"DARK"},
        "DRAGON": {"FAIRY"},
        "POISON": {"STEEL"},
    }
    atk = attacking_type.name if hasattr(attacking_type, "name") else str(attacking_type).upper()
    if atk not in _TYPE_TO_IDX:
        return 1.0
    dtypes = {(t.name if hasattr(t, "name") else str(t).upper()) for t in (defending_types or [])}
    # Immunities first
    if atk in imm and any(t in imm[atk] for t in dtypes):
        return 0.0
    # Compute effectiveness per defending type for correct dual-type handling
    eff = 1.0
    for t in dtypes:
        if atk in se and t in se[atk]:
            eff *= 2.0
        elif atk in ne and t in ne[atk]:
            eff *= 0.5
    return eff

def _has_revealed_se_now(our_poke, opp_poke) -> float:
    """Do we currently have a revealed move that is SE into opp's visible types?"""
    try:
        if not our_poke or not our_poke.moves:
            return 0.0
        opp_types = list(opp_poke.types) if opp_poke else []
        for mv in our_poke.moves.values():
            if _type_effectiveness(mv.type, opp_types) > 1.0:
                return 1.0
        return 0.0
    except Exception:
        return 0.0

def _opp_has_revealed_se_into_us(our_poke, opp_poke) -> float:
    try:
        if not opp_poke or not opp_poke.moves:
            return 0.0
        our_types = list(our_poke.types) if our_poke else []
        for mv in opp_poke.moves.values():
            if _type_effectiveness(mv.type, our_types) > 1.0:
                return 1.0
        return 0.0
    except Exception:
        return 0.0

def _count_revealed_se_in_party(our_team, opp_active_types) -> float:
    """Fraction of teammates (0..1) that have a revealed SE move vs opp active."""
    try:
        c = 0
        total = 0
        for poke in our_team.values():
            total += 1
            if not poke.moves:
                continue
            for mv in poke.moves.values():
                if _type_effectiveness(mv.type, opp_active_types) > 1.0:
                    c += 1
                    break
        if total == 0:
            return 0.0
        return float(c) / float(total)
    except Exception:
        return 0.0

# =========================
# v4: Computed battle features (matchup scores, damage estimates, speed tiers)
# These replicate what SimpleHeuristics/SmartBots compute internally,
# giving the model pre-computed "Pokemon math" instead of raw numbers.
# =========================

# Constants matching SimpleHeuristicsPlayer
_SH_SPEED_TIER_COEF = 0.1
_SH_HP_FRAC_COEF = 0.4

def _stat_estimation(poke, stat: str, use_base: bool = False) -> float:
    """Boost-adjusted stat estimate matching SimpleHeuristics._stat_estimation.
    For opponents, use_base=True (stats dict is None)."""
    try:
        if poke is None:
            return 1.0
        if use_base:
            base = poke.base_stats.get(stat, 80) if poke.base_stats else 80
        else:
            stats = poke.stats
            if isinstance(stats, dict) and stats.get(stat) is not None:
                base_val = float(stats[stat])
                # stats already includes EVs/IVs/nature, just apply boost
                boost_val = poke.boosts.get(stat, 0) if poke.boosts else 0
                if boost_val > 0:
                    boost = (2 + boost_val) / 2
                else:
                    boost = 2 / (2 - boost_val)
                return max(1.0, base_val * boost)
            else:
                base = poke.base_stats.get(stat, 80) if poke.base_stats else 80
        # Fallback: SH formula with base stats
        boost_val = poke.boosts.get(stat, 0) if poke.boosts else 0
        if boost_val > 0:
            boost = (2 + boost_val) / 2
        else:
            boost = 2 / (2 - boost_val)
        return max(1.0, ((2 * base + 31) + 5) * boost)
    except Exception:
        return 1.0

def _estimate_matchup(mon, opponent) -> float:
    """Matchup score matching SimpleHeuristicsPlayer._estimate_matchup.
    Works with revealed info only (base_stats for opponent)."""
    try:
        if mon is None or opponent is None:
            return 0.0
        # Offensive type advantage: how well do we hit them
        our_off = max(
            (opponent.damage_multiplier(t) for t in mon.types if t is not None),
            default=1.0
        )
        # Defensive disadvantage: how well do they hit us
        opp_off = max(
            (mon.damage_multiplier(t) for t in opponent.types if t is not None),
            default=1.0
        )
        score = our_off - opp_off
        # Speed tier
        our_spe = mon.base_stats.get("spe", 80) if mon.base_stats else 80
        opp_spe = opponent.base_stats.get("spe", 80) if opponent.base_stats else 80
        if our_spe > opp_spe:
            score += _SH_SPEED_TIER_COEF
        elif opp_spe > our_spe:
            score -= _SH_SPEED_TIER_COEF
        # HP fraction
        score += (mon.current_hp_fraction or 0) * _SH_HP_FRAC_COEF
        score -= (opponent.current_hp_fraction or 0) * _SH_HP_FRAC_COEF
        return score
    except Exception:
        return 0.0

def _move_damage_score(move, active, opponent) -> float:
    """Damage scoring formula matching SimpleHeuristics/SmartBots.
    Returns raw score (typically 0-500+)."""
    try:
        if move is None or active is None or opponent is None:
            return 0.0
        bp = float(getattr(move, "base_power", 0) or 0)
        if bp <= 0:
            return 0.0
        cat = getattr(move, "category", None)
        cat_name = cat.name if hasattr(cat, "name") else str(cat or "").upper()
        is_opp = not (isinstance(active.stats, dict) and active.stats.get("atk") is not None)
        if cat_name == "PHYSICAL":
            ratio = _stat_estimation(active, "atk", use_base=is_opp) / max(1.0, _stat_estimation(opponent, "def", use_base=True))
        elif cat_name == "SPECIAL":
            ratio = _stat_estimation(active, "spa", use_base=is_opp) / max(1.0, _stat_estimation(opponent, "spd", use_base=True))
        else:
            return 0.0
        stab = 1.5 if (hasattr(move, "type") and move.type in active.types) else 1.0
        acc = float(getattr(move, "accuracy", 1.0) or 1.0)
        hits = float(getattr(move, "expected_hits", 1.0) or 1.0)
        eff = float(opponent.damage_multiplier(move))
        return bp * stab * ratio * acc * hits * eff
    except Exception:
        return 0.0

def _eff_speed(poke, side_conds, fields, use_base: bool = False) -> float:
    """Effective speed accounting for boosts, Tailwind, Paralysis, Trick Room."""
    try:
        if poke is None:
            return 0.0
        sp = _stat_estimation(poke, "spe", use_base=use_base)
        if side_conds and SideCondition.TAILWIND in side_conds:
            sp *= 2
        if poke.status == Status.PAR:
            sp *= 0.5
        return sp
    except Exception:
        return 0.0

def _sr_damage_frac(mon) -> float:
    """Stealth Rock damage fraction for a mon based on its types."""
    try:
        if mon is None:
            return 0.125
        from poke_env.battle.pokemon_type import PokemonType
        mult = 1.0
        for t in mon.types:
            if t is not None:
                mult *= PokemonType.ROCK.damage_multiplier(t)
        return 0.125 * mult
    except Exception:
        return 0.125

def _compute_v4_features(battle) -> List[float]:
    """Compute all v4 battle features. Returns flat list of floats.

    Layout (48 dims total):
    Group A - Active vs Active (16):
      matchup_score(1), our_off_type(1), opp_off_type(1),
      speed_tier(3), phys_ratio(1), spec_ratio(1),
      opp_phys_ratio(1), opp_spec_ratio(1),
      best_move_score(1), can_ko(1), opp_best_revealed(1), opp_can_ko(1),
      have_priority(1), priority_can_ko(1)
    Group B - Our bench vs opp active (10):
      bench_matchup(5), bench_resist(5)
    Group C - Opp bench vs our active (5):
      opp_bench_threat(5)
    Group D - Aggregate bench signals (5):
      best_bench_matchup(1), best_bench_resist(1), n_positive_matchup(1),
      best_switch_w_hazards(1), hazard_entry_cost(1)
    Group E - Game context signals (12):
      endgame(1), opp_boost_total(1), our_boost_total(1),
      n_remaining(1), n_opp_remaining(1),
      our_hp_adv(1), opp_has_status(1), we_have_status(1),
      weather_benefit_us(1), weather_benefit_opp(1),
      best_move_is_stab(1), opp_active_threatened_by_bench(1)
    """
    feat: List[float] = []
    our = battle.active_pokemon
    opp = battle.opponent_active_pokemon
    our_team = battle.team
    opp_team = battle.opponent_team
    our_sc = battle.side_conditions
    opp_sc = battle.opponent_side_conditions
    fields = battle.fields

    # ===== Group A: Active vs Active (16 dims) =====

    # Matchup score (normalized to ~[-1,1] via clip/3)
    matchup = _estimate_matchup(our, opp)
    feat.append(max(-1.0, min(1.0, matchup / 3.0)))

    # Offensive type advantage: how well our types hit opp
    try:
        our_off = max((opp.damage_multiplier(t) for t in our.types if t is not None), default=1.0) if our and opp else 1.0
    except Exception:
        our_off = 1.0
    feat.append(min(1.0, our_off / 4.0))

    # Opponent offensive type advantage
    try:
        opp_off = max((our.damage_multiplier(t) for t in opp.types if t is not None), default=1.0) if our and opp else 1.0
    except Exception:
        opp_off = 1.0
    feat.append(min(1.0, opp_off / 4.0))

    # Speed tier: [we_faster, they_faster, tied]
    try:
        our_sp = _eff_speed(our, our_sc, fields, use_base=False)
        opp_sp = _eff_speed(opp, opp_sc, fields, use_base=True)
        trick_room = Field.TRICK_ROOM in fields if fields else False
        if our_sp > opp_sp:
            sp_tier = [1.0, 0.0, 0.0] if not trick_room else [0.0, 1.0, 0.0]
        elif opp_sp > our_sp:
            sp_tier = [0.0, 1.0, 0.0] if not trick_room else [1.0, 0.0, 0.0]
        else:
            sp_tier = [0.0, 0.0, 1.0]
    except Exception:
        sp_tier = [0.0, 0.0, 1.0]
    feat.extend(sp_tier)

    # Physical and special ratios (our attacking into opp defending)
    is_opp_base = True  # opponent always uses base_stats
    is_our_base = not (isinstance(getattr(our, 'stats', None), dict) and
                       getattr(our, 'stats', {}).get('atk') is not None) if our else True
    try:
        phys_r = _stat_estimation(our, "atk", is_our_base) / max(1.0, _stat_estimation(opp, "def", is_opp_base))
        spec_r = _stat_estimation(our, "spa", is_our_base) / max(1.0, _stat_estimation(opp, "spd", is_opp_base))
    except Exception:
        phys_r, spec_r = 1.0, 1.0
    feat.append(min(1.0, phys_r / 2.0))
    feat.append(min(1.0, spec_r / 2.0))

    # Opponent ratios (opp attacking into us defending)
    try:
        opp_phys = _stat_estimation(opp, "atk", is_opp_base) / max(1.0, _stat_estimation(our, "def", is_our_base))
        opp_spec = _stat_estimation(opp, "spa", is_opp_base) / max(1.0, _stat_estimation(our, "spd", is_our_base))
    except Exception:
        opp_phys, opp_spec = 1.0, 1.0
    feat.append(min(1.0, opp_phys / 2.0))
    feat.append(min(1.0, opp_spec / 2.0))

    # Best move damage score + KO check
    best_score = 0.0
    have_priority = 0.0
    priority_ko = 0.0
    try:
        for m in (battle.available_moves or []):
            sc = _move_damage_score(m, our, opp)
            if sc > best_score:
                best_score = sc
            prio = getattr(m, "priority", 0) or 0
            bp = getattr(m, "base_power", 0) or 0
            if prio > 0 and bp > 0:
                have_priority = 1.0
                if sc > 50 and opp and (opp.current_hp_fraction or 1.0) < 0.3:
                    priority_ko = 1.0
    except Exception:
        pass
    feat.append(min(1.0, best_score / 500.0))
    can_ko = 1.0 if (best_score > 150 and opp and (opp.current_hp_fraction or 1.0) < 0.4) else 0.0
    feat.append(can_ko)

    # Opponent's best revealed move into us
    opp_best = 0.0
    try:
        if opp and opp.moves:
            for m in opp.moves.values():
                sc = _move_damage_score(m, opp, our)
                if sc > opp_best:
                    opp_best = sc
    except Exception:
        pass
    feat.append(min(1.0, opp_best / 500.0))
    opp_can_ko = 1.0 if (opp_best > 150 and our and (our.current_hp_fraction or 1.0) < 0.4) else 0.0
    feat.append(opp_can_ko)

    # Priority signals
    feat.append(have_priority)
    feat.append(priority_ko)

    # ===== Group B: Our bench vs opponent active (10 dims) =====
    our_bench = _get_sorted_bench(our_team)
    bench_matchups = []
    bench_resists = []
    for i in range(MAX_BENCH):
        if i < len(our_bench) and not our_bench[i].fainted:
            bmon = our_bench[i]
            bm = _estimate_matchup(bmon, opp)
            bench_matchups.append(max(-1.0, min(1.0, bm / 3.0)))
            # Defensive resist: how poorly opp STAB types hit this bench mon
            try:
                opp_stab_mult = max(
                    (bmon.damage_multiplier(t) for t in opp.types if t is not None),
                    default=1.0
                ) if opp else 1.0
                bench_resists.append(min(1.0, opp_stab_mult / 4.0))
            except Exception:
                bench_resists.append(0.25)
        else:
            bench_matchups.append(0.0)
            bench_resists.append(0.0)
    feat.extend(bench_matchups)
    feat.extend(bench_resists)

    # ===== Group C: Opponent bench vs our active (5 dims) =====
    opp_bench = _get_sorted_bench(opp_team)
    for i in range(MAX_BENCH):
        if i < len(opp_bench) and not opp_bench[i].fainted:
            threat = _estimate_matchup(opp_bench[i], our)
            feat.append(max(-1.0, min(1.0, threat / 3.0)))
        else:
            feat.append(0.0)

    # ===== Group D: Aggregate bench signals (5 dims) =====
    # Best bench matchup vs opp active
    valid_matchups = [m for m, bm in zip(bench_matchups, our_bench[:MAX_BENCH])
                      if not getattr(bm, 'fainted', True)] if our_bench else bench_matchups
    feat.append(max(valid_matchups) if valid_matchups else 0.0)

    # Best bench resist (lowest opp STAB mult = best wall)
    feat.append(min(bench_resists) if bench_resists else 0.0)

    # Count bench with positive matchup
    n_pos = sum(1 for m in valid_matchups if m > 0)
    feat.append(float(n_pos) / MAX_BENCH)

    # Best switch accounting for hazard damage
    best_sw_score = -10.0
    try:
        has_sr = SideCondition.STEALTH_ROCK in our_sc if our_sc else False
        has_spikes = SideCondition.SPIKES in our_sc if our_sc else False
        for i, bm_score in enumerate(bench_matchups):
            if i < len(our_bench) and not our_bench[i].fainted:
                penalty = 0.0
                if has_sr:
                    penalty += _sr_damage_frac(our_bench[i]) * 2
                if has_spikes:
                    penalty += 0.1
                sw = bm_score - penalty
                if sw > best_sw_score:
                    best_sw_score = sw
    except Exception:
        pass
    feat.append(max(-1.0, min(1.0, best_sw_score)) if best_sw_score > -10.0 else 0.0)

    # Hazard entry cost for current active's types (for switch-in considerations)
    feat.append(_sr_damage_frac(our) if our else 0.125)

    # ===== Group E: Game context signals (12 dims) =====

    # Endgame flag
    try:
        n_remaining = len([m for m in our_team.values() if not m.fainted]) if our_team else 6
        n_opp_remaining = 6 - len([m for m in opp_team.values() if m.fainted]) if opp_team else 6
    except Exception:
        n_remaining, n_opp_remaining = 3, 3
    endgame = 1.0 if (n_remaining <= 2 and n_opp_remaining <= 2) else 0.0
    feat.append(endgame)

    # Opponent total boosts
    try:
        opp_boosts = sum(max(0, v) for v in opp.boosts.values()) if opp and opp.boosts else 0
    except Exception:
        opp_boosts = 0
    feat.append(min(1.0, opp_boosts / 12.0))

    # Our total boosts
    try:
        our_boosts = sum(max(0, v) for v in our.boosts.values()) if our and our.boosts else 0
    except Exception:
        our_boosts = 0
    feat.append(min(1.0, our_boosts / 12.0))

    # Remaining counts (normalized)
    feat.append(float(n_remaining) / 6.0)
    feat.append(float(n_opp_remaining) / 6.0)

    # HP advantage (our team HP% - opp team HP%)
    try:
        our_hp_sum = sum(m.current_hp_fraction for m in our_team.values() if not m.fainted) if our_team else 0
        opp_hp_sum = sum(m.current_hp_fraction for m in opp_team.values() if not m.fainted) if opp_team else 0
        hp_adv = (our_hp_sum / max(1, n_remaining)) - (opp_hp_sum / max(1, n_opp_remaining))
    except Exception:
        hp_adv = 0.0
    feat.append(max(-1.0, min(1.0, hp_adv)))

    # Status flags
    feat.append(1.0 if (opp and opp.status) else 0.0)
    feat.append(1.0 if (our and our.status) else 0.0)

    # Weather benefit signals
    try:
        weather = battle.weather or {}
        sun = Weather.SUNNYDAY in weather or Weather.DESOLATELAND in weather
        rain = Weather.RAINDANCE in weather or Weather.PRIMORDIALSEA in weather
        our_types = set(t.name for t in our.types if t is not None) if our else set()
        opp_types = set(t.name for t in opp.types if t is not None) if opp else set()
        w_us = 0.0
        w_opp = 0.0
        if sun:
            w_us += (1.0 if "FIRE" in our_types else 0.0) - (0.5 if "WATER" in our_types else 0.0)
            w_opp += (1.0 if "FIRE" in opp_types else 0.0) - (0.5 if "WATER" in opp_types else 0.0)
        if rain:
            w_us += (1.0 if "WATER" in our_types else 0.0) - (0.5 if "FIRE" in our_types else 0.0)
            w_opp += (1.0 if "WATER" in opp_types else 0.0) - (0.5 if "FIRE" in opp_types else 0.0)
    except Exception:
        w_us, w_opp = 0.0, 0.0
    feat.append(max(-1.0, min(1.0, w_us)))
    feat.append(max(-1.0, min(1.0, w_opp)))

    # Best move is STAB
    try:
        best_is_stab = 0.0
        if our and battle.available_moves:
            best_m = max(battle.available_moves, key=lambda m: _move_damage_score(m, our, opp))
            if hasattr(best_m, "type") and best_m.type in our.types:
                best_is_stab = 1.0
    except Exception:
        best_is_stab = 0.0
    feat.append(best_is_stab)

    # Opp active threatened by any bench mon (SE move exists)
    try:
        bench_threatens = 0.0
        for bmon in our_bench:
            if bmon.fainted:
                continue
            if bmon.moves:
                for mv in bmon.moves.values():
                    if opp and _type_effectiveness(mv.type, list(opp.types)) > 1.0:
                        bench_threatens = 1.0
                        break
            if bench_threatens > 0:
                break
    except Exception:
        bench_threatens = 0.0
    feat.append(bench_threatens)

    return feat  # 48 dims total

# v4 computed feature dimension
V4_COMPUTED_DIM = 48

# =========================
# Per-side blocks
# =========================

def _weight_height(poke) -> List[float]:
    """Return [weight_norm, height_norm] for a Pokemon. Important for Low Kick, Heavy Slam, etc."""
    if poke is None:
        return [0.0, 0.0]
    w = _coerce_float(getattr(poke, "weight", 0), default=0.0)
    h = _coerce_float(getattr(poke, "height", 0), default=0.0)
    return [min(1.0, w / 1000.0), min(1.0, h / 20.0)]

def _preparing_bits(poke) -> List[float]:
    """Return [is_preparing]. Preparing move ID goes to entity_ids separately."""
    if poke is None:
        return [0.0]
    return [1.0 if getattr(poke, "preparing", False) else 0.0]

def _toxic_fraction(poke) -> List[float]:
    """Toxic-specific escalation: status_counter/16 when status is TOX, else 0.
    Distinct from generic status_counter — gives cleaner signal for toxic damage ramp.
    Returns [toxic_frac]."""
    if poke is None:
        return [0.0]
    try:
        if poke.status == Status.TOX:
            return [min(1.0, float(getattr(poke, "status_counter", 0) or 0) / 16.0)]
    except Exception:
        pass
    return [0.0]

def _future_sight_bit(poke) -> List[float]:
    """1.0 if Future Sight is pending on this Pokemon's position. Returns [bit]."""
    if poke is None:
        return [0.0]
    try:
        effects = getattr(poke, "effects", {}) or {}
        return [1.0 if Effect.FUTURE_SIGHT in effects else 0.0]
    except Exception:
        return [0.0]

def _encode_active(poke) -> List[float]:
    """Encode active pokemon continuous features (v5: no hashes, IDs via entity_ids)."""
    if poke is None:
        return (
            _types_one_hot([])
            + [0.0]
            + _status_one_hot(None)
            + [_norm01(0, -6, 6)] * 7
            + _volatile_status_bits(None)
            + _tera_type_encoding(None)
            + _active_combat_state(None)
            + _weight_height(None)
            + _preparing_bits(None)
            + _toxic_fraction(None)
            + _future_sight_bit(None)
        )
    return (
        _types_one_hot(poke.types)
        + [_hp_frac(poke)]
        + _status_one_hot(poke.status)
        + _boosts_block(poke)
        + _volatile_status_bits(poke)
        + _tera_type_encoding(poke)
        + _active_combat_state(poke)
        + _weight_height(poke)
        + _preparing_bits(poke)
        + _toxic_fraction(poke)
        + _future_sight_bit(poke)
    )

# v5 active continuous dim: types(19) + hp(1) + status(7) + boosts(7) + volatile(17) + tera(20) + combat(5) + weight_height(2) + preparing(1) + toxic_frac(1) + future_sight(1) = 81
ACTIVE_DIM = 19 + 1 + 7 + 7 + N_VOLATILE + (N_TYPES + 1) + N_COMBAT_STATE + 2 + 1 + 1 + 1

def _extract_active_ids(poke) -> List[int]:
    """Extract [species_id, item_id, ability_id] for entity_ids."""
    v = _get_vocab()
    if poke is None:
        return [0, 0, 0]
    ability = poke.ability or (poke.possible_abilities[0] if poke.possible_abilities else None)
    return [v.species(poke.species), v.item(poke.item), v.ability(ability)]

def _encode_bench(team: Dict[str, Any], dropout_prob: float = 0.0, is_opp: bool = False) -> List[float]:
    """Up to MAX_BENCH slots. Each slot: [hp_frac, status_onehot(7)] = 8 dims (v5: no hash)."""
    slots: List[List[float]] = []
    bench = _get_sorted_bench(team)
    for i in range(MAX_BENCH):
        if i < len(bench):
            p = bench[i]
            if is_opp and dropout_prob > 0.0 and np.random.rand() < dropout_prob:
                slots.append([0.0] + _status_one_hot(None))
                continue
            v = [_hp_frac(p)] + _status_one_hot(p.status)
            slots.append(v)
        else:
            slots.append([0.0] + _status_one_hot(None))
    flat: List[float] = []
    for s in slots:
        flat.extend(s)
    return flat

# v5 bench slot dim: hp(1) + status(7) = 8
BENCH_SLOT_DIM = 1 + 7

def _extract_bench_ids(team: Dict[str, Any], is_opp: bool = False) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Extract (species_ids[5], item_ids[5], ability_ids[5], move_ids[20]) from bench."""
    v = _get_vocab()
    bench = _get_sorted_bench(team)
    species_ids = []
    item_ids = []
    ability_ids = []
    move_ids = []
    for i in range(MAX_BENCH):
        if i < len(bench):
            p = bench[i]
            species_ids.append(v.species(p.species))
            item_ids.append(v.item(p.item) if not is_opp or p.item else 0)
            ab = p.ability or (p.possible_abilities[0] if p.possible_abilities else None)
            ability_ids.append(v.ability(ab) if not is_opp or p.ability else 0)
            # Moves: up to 4 per pokemon
            moves = list(p.moves.keys()) if p.moves else []
            for j in range(MAX_MOVES):
                move_ids.append(v.move(moves[j]) if j < len(moves) else 0)
        else:
            species_ids.append(0)
            item_ids.append(0)
            ability_ids.append(0)
            move_ids.extend([0] * MAX_MOVES)
    return species_ids, item_ids, ability_ids, move_ids

def _extract_active_move_ids(poke) -> List[int]:
    """Extract revealed move IDs for active pokemon (4 slots)."""
    v = _get_vocab()
    if poke is None or not poke.moves:
        return [0] * MAX_MOVES
    moves = list(poke.moves.keys())
    return [v.move(moves[j]) if j < len(moves) else 0 for j in range(MAX_MOVES)]

def _norm_layers(val, lo, hi) -> float:
    try:
        return _norm01(float(val or 0), float(lo), float(hi))
    except Exception:
        return 0.0

def _side_cond_presence_and_layers(sc) -> Dict[str, float]:
    """Return presence + normalized layer counts for SR/Spikes/TSpikes/Web/Tailwind.
    sc is Dict[SideCondition, int] from poke-env."""
    out = dict(
        sr=0.0, spikes=0.0, spikes_layers=0.0, tspikes=0.0, tspikes_layers=0.0,
        web=0.0, tailwind=0.0
    )
    if not sc:
        return out
    out["sr"]       = 1.0 if SideCondition.STEALTH_ROCK in sc else 0.0
    out["spikes"]   = 1.0 if SideCondition.SPIKES in sc else 0.0
    out["tspikes"]  = 1.0 if SideCondition.TOXIC_SPIKES in sc else 0.0
    out["web"]      = 1.0 if SideCondition.STICKY_WEB in sc else 0.0
    out["tailwind"] = 1.0 if SideCondition.TAILWIND in sc else 0.0
    out["spikes_layers"]  = _norm_layers(sc.get(SideCondition.SPIKES, 0), 0, 3)
    out["tspikes_layers"] = _norm_layers(sc.get(SideCondition.TOXIC_SPIKES, 0), 0, 2)
    return out

def _screen_veil_bits(ours_sc, opps_sc) -> List[float]:
    """Per-side presence of Reflect/Light Screen/Aurora Veil.
    sc is Dict[SideCondition, int] from poke-env."""
    def _sc_has(sc, cond):
        return 1.0 if sc and cond in sc else 0.0
    return [
        _sc_has(ours_sc, SideCondition.REFLECT),
        _sc_has(ours_sc, SideCondition.LIGHT_SCREEN),
        _sc_has(ours_sc, SideCondition.AURORA_VEIL),
        _sc_has(opps_sc, SideCondition.REFLECT),
        _sc_has(opps_sc, SideCondition.LIGHT_SCREEN),
        _sc_has(opps_sc, SideCondition.AURORA_VEIL),
    ]

def _weather_onehot(w) -> List[float]:
    """Encode weather as one-hot + turn counter. w is Dict[Weather, int] from poke-env.
    Returns 6 floats: [sun, rain, sand, snow, any_weather, turns_remaining/8]."""
    if not w:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    sun_key = Weather.SUNNYDAY if Weather.SUNNYDAY in w else (Weather.DESOLATELAND if Weather.DESOLATELAND in w else None)
    rain_key = Weather.RAINDANCE if Weather.RAINDANCE in w else (Weather.PRIMORDIALSEA if Weather.PRIMORDIALSEA in w else None)
    sand_key = Weather.SANDSTORM if Weather.SANDSTORM in w else None
    snow_key = Weather.SNOWSCAPE if Weather.SNOWSCAPE in w else (Weather.HAIL if Weather.HAIL in w else None)
    sun  = 1.0 if sun_key else 0.0
    rain = 1.0 if rain_key else 0.0
    sand = 1.0 if sand_key else 0.0
    snow = 1.0 if snow_key else 0.0
    anyw = 1.0 if (sun or rain or sand or snow) else 0.0
    # Turn counter: poke-env stores turns elapsed; weather lasts 5 turns (8 with rock/icy rock)
    # Normalize by 8 (max possible duration)
    active_key = sun_key or rain_key or sand_key or snow_key
    turns = min(8, int(w.get(active_key, 0) or 0)) / 8.0 if active_key else 0.0
    return [sun, rain, sand, snow, anyw, turns]

def _terrain_onehot(fields) -> List[float]:
    """Encode terrain as one-hot + turn counter. fields is Dict[Field, int] from poke-env.
    Returns 6 floats: [electric, grassy, psychic, misty, any_terrain, turns_remaining/5]."""
    if not fields:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    el_key = Field.ELECTRIC_TERRAIN if Field.ELECTRIC_TERRAIN in fields else None
    gr_key = Field.GRASSY_TERRAIN if Field.GRASSY_TERRAIN in fields else None
    ps_key = Field.PSYCHIC_TERRAIN if Field.PSYCHIC_TERRAIN in fields else None
    ms_key = Field.MISTY_TERRAIN if Field.MISTY_TERRAIN in fields else None
    el = 1.0 if el_key else 0.0
    gr = 1.0 if gr_key else 0.0
    ps = 1.0 if ps_key else 0.0
    ms = 1.0 if ms_key else 0.0
    anyt = 1.0 if (el or gr or ps or ms) else 0.0
    active_key = el_key or gr_key or ps_key or ms_key
    turns = min(5, int(fields.get(active_key, 0) or 0)) / 5.0 if active_key else 0.0
    return [el, gr, ps, ms, anyt, turns]

def _mechanic_flags(battle) -> List[float]:
    """Per-battle mechanics: can/used tera/mega/z for both sides, plus trapped/force-switch flags."""
    can_tera      = 1.0 if battle.can_tera else 0.0
    can_mega      = 1.0 if battle.can_mega_evolve else 0.0
    can_zmove     = 1.0 if battle.can_z_move else 0.0
    used_tera_us  = 1.0 if battle.used_tera else 0.0
    used_tera_opp = 1.0 if battle.opponent_used_tera else 0.0
    used_mega_us  = 1.0 if battle.used_mega_evolve else 0.0
    used_mega_opp = 1.0 if battle.opponent_used_mega_evolve else 0.0
    used_z_us     = 1.0 if battle.used_z_move else 0.0
    used_z_opp    = 1.0 if battle.opponent_used_z_move else 0.0
    trapped       = 1.0 if (battle.trapped or battle.maybe_trapped) else 0.0
    # v5 fix: detect if opponent is trapped (via our ability or their volatile status)
    opp_trapped   = 0.0
    opp = battle.opponent_active_pokemon
    our = battle.active_pokemon
    if opp is not None:
        opp_effects = getattr(opp, "effects", {}) or {}
        if Effect.TRAPPED in opp_effects or Effect.PARTIALLY_TRAPPED in opp_effects:
            opp_trapped = 1.0
    if our is not None and opp_trapped < 0.5:
        our_ability = (getattr(our, "ability", None) or "").lower()
        if our_ability in ("arenatrap", "shadowtag", "magnetpull"):
            opp_trapped = 1.0
    force_sw      = 1.0 if battle.force_switch else 0.0
    # v5: Dynamax tracking
    can_dmax      = 1.0 if getattr(battle, "can_dynamax", False) else 0.0
    used_dmax_us  = 1.0 if getattr(battle, "used_dynamax", False) else 0.0
    used_dmax_opp = 1.0 if getattr(battle, "opponent_used_dynamax", False) else 0.0
    dmax_turns_us  = min(3, getattr(battle, "dynamax_turns_left", 0) or 0) / 3.0
    dmax_turns_opp = min(3, getattr(battle, "opponent_dynamax_turns_left", 0) or 0) / 3.0
    # v5: Opponent revealed count (how many opponent mons have been seen)
    opp_revealed = 0
    for p in battle.opponent_team.values():
        if getattr(p, "revealed", True):
            opp_revealed += 1
    opp_revealed_frac = opp_revealed / 6.0
    return [can_tera, can_mega, can_zmove, can_dmax,
            used_tera_us, used_tera_opp, used_mega_us, used_mega_opp, used_z_us, used_z_opp,
            used_dmax_us, used_dmax_opp, dmax_turns_us, dmax_turns_opp,
            trapped, opp_trapped, force_sw, opp_revealed_frac]

def _cond_turns_remaining(sc, cond, current_turn, max_duration) -> float:
    """Normalized turns remaining for a timed (non-stackable) side condition.
    poke-env stores set_turn for non-stackable conditions."""
    if not sc or cond not in sc:
        return 0.0
    try:
        set_turn = int(sc[cond])
        elapsed = current_turn - set_turn
        remaining = max(0, max_duration - elapsed)
        return min(1.0, remaining / max_duration)
    except Exception:
        return 0.0

def _board_bits(battle) -> List[float]:
    """Expanded board state: hazards (presence + layers), screens/veil per side, TR, weather, terrain, mechanics, turn counters."""
    ours_sc = battle.side_conditions
    opps_sc = battle.opponent_side_conditions
    turn = battle.turn or 0

    ours = _side_cond_presence_and_layers(ours_sc)
    opps = _side_cond_presence_and_layers(opps_sc)

    fields = battle.fields  # Dict[Field, int]
    trick = 1.0 if Field.TRICK_ROOM in fields else 0.0
    weather = battle.weather  # Dict[Weather, int]

    return [
        # ours hazards
        ours["sr"], ours["spikes"], ours["spikes_layers"], ours["tspikes"], ours["tspikes_layers"], ours["web"], ours["tailwind"],
        # opp hazards
        opps["sr"], opps["spikes"], opps["spikes_layers"], opps["tspikes"], opps["tspikes_layers"], opps["web"], opps["tailwind"],
        # per-side screens/veil
        *_screen_veil_bits(ours_sc, opps_sc),
        # field controls
        trick,
        *_weather_onehot(weather),
        *_terrain_onehot(fields),
        # mechanics + traps + force switch
        *_mechanic_flags(battle),
        # Timed side condition turn counters (8 dims)
        # Tailwind lasts 4 turns
        _cond_turns_remaining(ours_sc, SideCondition.TAILWIND, turn, 4),
        _cond_turns_remaining(opps_sc, SideCondition.TAILWIND, turn, 4),
        # Screens last 5 turns (8 with Light Clay); normalize by 8
        _cond_turns_remaining(ours_sc, SideCondition.REFLECT, turn, 8),
        _cond_turns_remaining(ours_sc, SideCondition.LIGHT_SCREEN, turn, 8),
        _cond_turns_remaining(ours_sc, SideCondition.AURORA_VEIL, turn, 8),
        _cond_turns_remaining(opps_sc, SideCondition.REFLECT, turn, 8),
        _cond_turns_remaining(opps_sc, SideCondition.LIGHT_SCREEN, turn, 8),
        _cond_turns_remaining(opps_sc, SideCondition.AURORA_VEIL, turn, 8),
    ]

# =========================
# Public: featurize
# =========================

def featurize(
    battle,
    *,
    bench_dropout_prob_opp: float = 0.0,
    moved_first_bits: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode battle to (obs_continuous: float32[], entity_ids: int32[]).

    v5: categorical identities (species/move/item/ability) are integer IDs
    in entity_ids for nn.Embedding. Continuous obs has no hash buckets.
    """
    feat: List[float] = []
    ids: List[int] = []

    # Global scalar (normalized turn)
    turn = float(battle.turn or 0)
    feat.append(_norm01(turn, 0.0, 100.0))

    # Active mons (ours, theirs) — continuous only
    our = battle.active_pokemon
    opp = battle.opponent_active_pokemon
    feat.extend(_encode_active(our))      # 81 dims
    feat.extend(_encode_active(opp))      # 81 dims

    # Active entity IDs: [species, item, ability] × 2
    ids.extend(_extract_active_ids(our))   # IDs [0-2]
    ids.extend(_extract_active_ids(opp))   # IDs [3-5]

    # Add revealed move-type presence histograms for both actives
    feat.extend(_revealed_move_type_hist(our))   # 19
    feat.extend(_revealed_move_type_hist(opp))   # 19

    # Bench (ours, theirs) — continuous: hp + status only
    our_team = battle.team
    opp_team = battle.opponent_team
    feat.extend(_encode_bench(our_team, dropout_prob=0.0, is_opp=False))     # 40
    feat.extend(_encode_bench(opp_team, dropout_prob=bench_dropout_prob_opp, is_opp=True))  # 40

    # Bench entity IDs
    our_b_sp, our_b_it, our_b_ab, our_b_mv = _extract_bench_ids(our_team, is_opp=False)
    opp_b_sp, opp_b_it, opp_b_ab, opp_b_mv = _extract_bench_ids(opp_team, is_opp=True)
    ids.extend(our_b_sp)   # IDs [6-10]
    ids.extend(opp_b_sp)   # IDs [11-15]
    ids.extend(our_b_it)   # IDs [16-20]
    ids.extend(opp_b_it)   # IDs [21-25]
    ids.extend(our_b_ab)   # IDs [26-30]
    ids.extend(opp_b_ab)   # IDs [31-35]

    # Expanded board bits
    feat.extend(_board_bits(battle))  # 59

    # Revealed-only matchup scalars
    opp_types = list(opp.types) if opp else []
    feat.append(_has_revealed_se_now(our, opp))
    feat.append(_opp_has_revealed_se_into_us(our, opp))
    feat.append(_count_revealed_se_in_party(our_team, opp_types))

    # Moved-first bits
    if moved_first_bits is not None:
        mfb = [float(x) for x in moved_first_bits]
        if len(mfb) != 3:
            mfb = [0.0, 0.0, 1.0]
    else:
        mfb = moved_first_heuristic_bits(battle)
    feat.extend(mfb)

    # === v2 additions (stats + moves, NO items/abilities hash) ===

    our_bench = _get_sorted_bench(our_team)
    opp_bench = _get_sorted_bench(opp_team)

    # Active stats: our actual (6) + opponent base (6) = 12
    feat.extend(_encode_stats(our, use_base=False))
    feat.extend(_encode_stats(opp, use_base=True))

    # Bench stats: our actual (30) + opponent base (30) = 60
    feat.extend(_encode_bench_stats(our_bench, use_base=False))
    feat.extend(_encode_bench_stats(opp_bench, use_base=True))

    # Opponent active revealed moves: 4 × 23 = 92 (continuous, no hash)
    feat.extend(_encode_pokemon_moves_compact(opp))

    # Opponent bench revealed moves: 5 × 4 × 23 = 460
    feat.extend(_encode_bench_moves(opp_bench))

    # Our bench moves: 5 × 4 × 23 = 460
    feat.extend(_encode_bench_moves(our_bench))

    # v5: bench items/abilities are entity IDs only (no continuous encoding)
    # (IDs already extracted above via _extract_bench_ids)

    # Alive counts: [our/6, opp/6] = 2
    feat.extend(_alive_counts(battle))

    # Move IDs for revealed moves
    ids.extend(_extract_active_move_ids(opp))    # IDs [36-39]
    ids.extend(opp_b_mv)                          # IDs [40-59]
    ids.extend(our_b_mv)                          # IDs [60-79]

    # === v4 additions: computed battle features (48 dims) ===
    feat.extend(_compute_v4_features(battle))

    # Opponent last-action IDs (populated later in make_obs_mask_and_slots)
    ids.extend([0, 0])  # IDs [80-81]: opp_last_move, opp_last_switch_species

    # Preparing move IDs (two-turn moves like Fly, Dig, Solar Beam)
    v = _get_vocab()
    our_prep_move = getattr(our, "preparing_move", None) if our else None
    opp_prep_move = getattr(opp, "preparing_move", None) if opp else None
    ids.append(v.move(our_prep_move.id) if our_prep_move and hasattr(our_prep_move, "id") else 0)  # ID [82]
    ids.append(v.move(opp_prep_move.id) if opp_prep_move and hasattr(opp_prep_move, "id") else 0)  # ID [83]

    assert len(ids) == N_ENTITY_IDS, f"entity_ids len {len(ids)} != {N_ENTITY_IDS}"

    return np.asarray(feat, dtype=np.float32), np.asarray(ids, dtype=np.int32)

# =========================
# Public: action mask + metadata
# =========================

def action_mask(battle) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return 9-way mask (4 moves + 5 switches) and compact per-slot metadata."""
    mask = np.zeros((9,), dtype=np.float32)
    moves_meta: List[Dict[str, Any]] = []
    switches_meta: List[Dict[str, Any]] = []

    # Moves (up to 4)
    moves = list(battle.available_moves or [])
    for i in range(4):
        if i < len(moves):
            m = moves[i]
            mask[i] = 1.0
            # Rich move projection with STAB from active Pokemon types
            active_types = None
            if battle.active_pokemon:
                active_types = battle.active_pokemon.types
            meta = _project_move_flags(m, poke_types=active_types)
            moves_meta.append(meta)
        else:
            moves_meta.append({})
    # Switches (up to 5)
    switches = list(battle.available_switches or [])
    for j in range(5):
        if j < len(switches):
            p = switches[j]
            mask[4 + j] = 1.0
            switches_meta.append({
                "species": p.species,
                "types": [t.name if hasattr(t, "name") else str(t).upper() for t in p.types],
                "hp_frac": _hp_frac(p),
                "status": p.status.name if p.status else None,
                "weight_norm": min(1.0, _coerce_float(getattr(p, "weight", 0), 0.0) / 1000.0),
            })
        else:
            switches_meta.append({})
    mask = np.asarray(mask, dtype=np.float32)
    # Safety: ensure at least one legal bit (prevents NaNs in inference)
    if float(mask.sum()) <= 0.0:
        # Prefer a move slot; else first switch; else leave as zeros
        print("[Features] [warn] No legal action chosen, falling back to heuristic default")
        if len(moves) > 0:
            mask[0] = 1.0
        elif len(switches) > 0:
            mask[4] = 1.0
    return mask, moves_meta, switches_meta

def make_obs_and_masks(battle, moved_first_bits=None):
    """
    Returns (obs_vec, legal_mask, ctx_extra, entity_ids)
    v5: entity_ids is a list of ints for nn.Embedding.
    """
    # 1) obs + entity_ids
    obs_arr, ids_arr = featurize(
        battle,
        bench_dropout_prob_opp=0.0,
        moved_first_bits=moved_first_bits
    )
    obs_vec = obs_arr.tolist()
    entity_ids = ids_arr.tolist()

    # 2) legal
    mask, _m, _s = action_mask(battle)
    legal_mask = mask.astype("float32").tolist()

    # 3) ctx — always fixed-size (26 base + 15 opp_last = 41 dims)
    # derive_ctx_extra_live returns 26 dims or None on error; fall back to zeros
    base_ctx = derive_ctx_extra_live(battle)
    if base_ctx is None:
        print("[Features] [warn] derive_ctx_extra_live failed — filling ctx with zeros (data is in main obs)")
        base_ctx = [0.0] * 26  # zeros if extraction fails — keeps dim stable
    kind, last_meta = _opp_last_action_from_logs(battle)
    opp_last = encode_opp_last_ctx(kind, last_meta)

    # Fill opp last-action entity IDs
    if kind == "MOVE":
        entity_ids[80] = _mini_move_id_from_meta(last_meta)
    elif kind == "SWITCH":
        entity_ids[81] = _mini_switch_id_from_meta(last_meta)

    ctx = (base_ctx + opp_last)  # always 41 dims

    return obs_vec, legal_mask, ctx, entity_ids

# === Full output including slot tensors ===
def make_obs_mask_and_slots(battle, moved_first_bits=None):
    """Returns (obs_vec, legal_mask, ctx, move_slots, switch_slots, entity_ids, move_ids, switch_ids)."""
    # 1) obs + entity_ids
    obs_arr, ids_arr = featurize(
        battle,
        bench_dropout_prob_opp=0.0,
        moved_first_bits=moved_first_bits
    )
    obs_vec = obs_arr.tolist()
    entity_ids = ids_arr.tolist()

    # 2) legal mask + slot meta (single action_mask call)
    mask, moves_meta, switches_meta = action_mask(battle)
    legal_mask = mask.astype("float32").tolist()

    # 3) ctx — always fixed-size (26 base + 15 opp_last = 41 dims)
    base_ctx = derive_ctx_extra_live(battle)
    if base_ctx is None:
        print("[Features] [warn] derive_ctx_extra_live failed — filling ctx with zeros (data is in main obs)")
        base_ctx = [0.0] * 26
    kind, last_meta = _opp_last_action_from_logs(battle)
    opp_last = encode_opp_last_ctx(kind, last_meta)

    # Fill opp last-action entity IDs
    if kind == "MOVE":
        entity_ids[80] = _mini_move_id_from_meta(last_meta)
    elif kind == "SWITCH":
        entity_ids[81] = _mini_switch_id_from_meta(last_meta)

    ctx = (base_ctx + opp_last)  # always 41 dims

    # 4) slot encodings
    mv, sw, move_ids, switch_ids = encode_move_and_switch_slots(moves_meta, switches_meta)
    return obs_vec, legal_mask, ctx, mv.astype("float32").tolist(), sw.astype("float32").tolist(), entity_ids, move_ids, switch_ids

# =====================================
# Step-type & hazard/context helpers
# =====================================

def step_type_from_pos(t: int, T: int, bins: int = 3) -> int:
    """
    Relative step bucket in [0..bins-1]. Defaults: 0=early,1=mid,2=late.
    If T<=0, returns 0.
    """
    if bins <= 1 or T <= 0: 
        return 0
    r = max(0.0, min(1.0, float(t) / float(max(1, T-1))))
    # equal-width bins
    return min(bins - 1, int(r * bins))

def step_type_from_abs_t(t: int, bins: int = 3, cap: int = 50) -> int:
    """
    Absolute fallback when episode length is unknown (row-mode).
    Buckets 0..bins-1 by clamping t into [0, cap] and uniform-binning.
    For legacy behavior (3 buckets, <=7/<=32/>32), pass bins=3, cap=32.
    """
    try:
        t = int(max(0, t))
        bins = int(max(1, bins))
        cap = int(max(1, cap))
        # Uniform bucket over [0, cap] inclusive; mirror inference path
        idx = int((t / float(cap)) * bins)
        return min(bins - 1, idx)
    except Exception:
        # ultra-safe: legacy fallback
        if t <= 10: return 0
        if t <= 26: return 1
        return 2
    
# --- Environment extractors (screens/TR/weather/terrain) ---

def _has_side_cond(side_conds, cond: SideCondition) -> bool:
    """Check if a SideCondition is present."""
    return side_conds is not None and cond in side_conds

def _has_field(fields, field: Field) -> bool:
    """Check if a Field condition is present."""
    return fields is not None and field in fields

# --- Shared ctx helper used by observer/train/inference ---
def derive_ctx_extra_live(battle):
    """
    Compact ctx: decisive board presences only (no layers), stable across gens.
    Order (26):
      ours:   [sr, spikes, tspikes, web, tailwind, reflect, lightscreen, veil]
      opp:    [sr, spikes, tspikes, web, tailwind, reflect, lightscreen, veil]
      field:  [trickroom]
      mech:   [trapped_or_forced]
      extras: [weather_sun, weather_rain, weather_sand, weather_snow, terr_electric, terr_psychic, terr_grassy, terr_misty]
    """
    try:
        ours_sc = battle.side_conditions
        opps_sc = battle.opponent_side_conditions

        def _has(sc, cond):
            return 1.0 if sc and cond in sc else 0.0

        ours = [
            _has(ours_sc, SideCondition.STEALTH_ROCK), _has(ours_sc, SideCondition.SPIKES),
            _has(ours_sc, SideCondition.TOXIC_SPIKES), _has(ours_sc, SideCondition.STICKY_WEB),
            _has(ours_sc, SideCondition.TAILWIND),
            _has(ours_sc, SideCondition.REFLECT), _has(ours_sc, SideCondition.LIGHT_SCREEN),
            _has(ours_sc, SideCondition.AURORA_VEIL),
        ]
        opps = [
            _has(opps_sc, SideCondition.STEALTH_ROCK), _has(opps_sc, SideCondition.SPIKES),
            _has(opps_sc, SideCondition.TOXIC_SPIKES), _has(opps_sc, SideCondition.STICKY_WEB),
            _has(opps_sc, SideCondition.TAILWIND),
            _has(opps_sc, SideCondition.REFLECT), _has(opps_sc, SideCondition.LIGHT_SCREEN),
            _has(opps_sc, SideCondition.AURORA_VEIL),
        ]

        fields = battle.fields  # Dict[Field, int]
        weather = battle.weather  # Dict[Weather, int]
        tr = 1.0 if Field.TRICK_ROOM in fields else 0.0
        trapped_or_forced = 1.0 if (battle.trapped or battle.maybe_trapped or battle.force_switch) else 0.0

        weather_sun  = 1.0 if (Weather.SUNNYDAY in weather or Weather.DESOLATELAND in weather) else 0.0
        weather_rain = 1.0 if (Weather.RAINDANCE in weather or Weather.PRIMORDIALSEA in weather) else 0.0
        weather_sand = 1.0 if Weather.SANDSTORM in weather else 0.0
        weather_snow = 1.0 if (Weather.SNOWSCAPE in weather or Weather.HAIL in weather) else 0.0

        terr_electric = 1.0 if Field.ELECTRIC_TERRAIN in fields else 0.0
        terr_psychic  = 1.0 if Field.PSYCHIC_TERRAIN in fields else 0.0
        terr_grassy   = 1.0 if Field.GRASSY_TERRAIN in fields else 0.0
        terr_misty    = 1.0 if Field.MISTY_TERRAIN in fields else 0.0

        vec = ours + opps + [tr, trapped_or_forced, weather_sun, weather_rain, weather_sand, weather_snow, terr_electric, terr_psychic, terr_grassy, terr_misty]
        return [float(x) for x in vec]
    except Exception:
        return None

def describe_obs_layout_runtime(battle=None) -> dict:
    """Build a slice map for the v5 continuous obs vector."""
    active = ACTIVE_DIM  # 81 (types+hp+status+boosts+volatile+tera+combat+weight_height+preparing+toxic_frac+future_sight)
    bench_slot = 1 + 7  # v5: hp + status only (no species hash)
    bench = bench_slot * MAX_BENCH

    hist = 2 * N_TYPES
    # board: hazards(7+7) + screens(6) + trick(1) + weather(6) + terrain(6) + mechanics(18) + turn_counters(8)
    board = 7 + 7 + 6 + 1 + 6 + 6 + 18 + 8  # 59
    scalars = 3 + 3  # matchup(3) + moved_first(3)

    # v2 additions (no bench items/abilities — those are entity IDs now)
    active_stats = N_STATS
    bench_stats = N_STATS * MAX_BENCH
    active_revealed_moves = MAX_MOVES * MOVE_COMPACT_DIM
    bench_moves = MAX_BENCH * MAX_MOVES * MOVE_COMPACT_DIM
    alive = 2
    v4_computed = 48  # v4 computed battle features

    idx = 0
    def put(name, length):
        nonlocal idx
        start, idx2 = idx, idx + length
        idx = idx2
        return name, (start, idx2 - 1)

    out = dict([
        put("turn_norm", 1),
        put("our_active", active),
        put("opp_active", active),
        put("move_type_histograms", hist),
        put("our_bench", bench),
        put("opp_bench", bench),
        put("board_bits", board),
        put("matchup_scalars_plus_moved_first", scalars),
        put("our_active_stats", active_stats),
        put("opp_active_stats", active_stats),
        put("our_bench_stats", bench_stats),
        put("opp_bench_stats", bench_stats),
        put("opp_active_revealed_moves", active_revealed_moves),
        put("opp_bench_revealed_moves", bench_moves),
        put("our_bench_moves", bench_moves),
        put("alive_counts", alive),
        put("v4_computed_features", v4_computed),
    ])
    out["total"] = idx
    out["entity_ids"] = N_ENTITY_IDS
    out["move_slot_dim"] = _move_slot_dim()
    out["switch_slot_dim"] = SWITCH_SLOT_DIM
    return out
