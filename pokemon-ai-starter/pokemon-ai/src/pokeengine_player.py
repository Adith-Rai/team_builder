"""pokeengine_player.py — Foul Play-style MCTS adapter wrapping poke-engine.

Drives Foul Play's MCTS Rust library (`poke-engine`) directly from a poke-env
Battle, in-process. No subprocess, no separate Showdown client. The adapter is
a normal poke-env Player subclass, so it slots into our existing PPO collection
(`battle_against` works as-is, PFSP weighting works as-is via the same display
name everywhere).

State conversion is "Foul Play lite" — for our side we use the request-derived
exact stats / nature / EVs (poke-env exposes these via Battle); for the
opponent we fall back to base-stats × neutral nature × 85 EVs because true
opponent set info is hidden. Foul Play does the same kind of guessing via its
SmogonSets/TeamDatasets — we just don't pull in that pkmn_sets stack.

Dependencies:
- `poke_engine` Python package (PEP 517 + Rust toolchain via `pip install poke-engine`).
- Foul Play's pokedex.json (read from `foul_play_ref/data/`) for weight + base
  types of opponent pokemon when not yet revealed.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
from pathlib import Path
from typing import Optional

from poke_env.player import Player
from poke_env import AccountConfiguration
from poke_env.battle import Status, SideCondition, Weather, Field

import poke_engine as pe

try:
    from policy_smartbots import TacticalPlayer as _FallbackBotCls
except ImportError:
    try:
        from .policy_smartbots import TacticalPlayer as _FallbackBotCls
    except ImportError:
        _FallbackBotCls = None

logger = logging.getLogger(__name__)


# ── Foul Play data (lazy-loaded JSON, no Python deps) ─────────────
_FOUL_PLAY_DATA_DIR = Path(
    os.environ.get(
        "FOUL_PLAY_DATA_DIR",
        Path(__file__).resolve().parents[3] / "foul_play_ref" / "data",
    )
)
_POKEDEX: Optional[dict] = None


def _load_pokedex() -> dict:
    global _POKEDEX
    if _POKEDEX is None:
        path = _FOUL_PLAY_DATA_DIR / "pokedex.json"
        with open(path) as f:
            _POKEDEX = json.load(f)
    return _POKEDEX


def _norm(s) -> str:
    if s is None:
        return ""
    return "".join(c for c in str(s).lower() if c.isalnum())


_STATUS_MAP = {
    Status.SLP: "Sleep",
    Status.BRN: "Burn",
    Status.FRZ: "Freeze",
    Status.PAR: "Paralyze",
    Status.PSN: "Poison",
    Status.TOX: "Toxic",
    Status.FNT: "None",
    None: "None",
}


def _opponent_stat(base: int, level: int, is_hp: bool = False) -> int:
    """Approximate a stat for an opponent pokemon: 31 IV, 85 EV, neutral nature."""
    if is_hp:
        return ((2 * base + 31 + 21) * level // 100) + level + 10
    return ((2 * base + 31 + 21) * level // 100) + 5


def _pokemon_to_pe(poke, is_opponent: bool) -> pe.Pokemon:
    """Convert poke-env Pokemon -> poke_engine.Pokemon.

    For our side, we use exact stats from request_json (poke-env populates
    poke.stats in that case). For the opponent, we approximate from base
    stats with neutral nature, 85 EVs, 31 IVs.
    """
    if poke is None:
        return pe.Pokemon.create_fainted()

    pokedex = _load_pokedex()
    species_id = _norm(poke.species)
    dex_entry = pokedex.get(species_id, {})
    level = int(poke.level or 100)

    # Types (current) — poke-env may have already swapped these for Tera
    types = poke.types or [None, None]
    t0 = _norm(types[0].name) if (len(types) > 0 and types[0]) else "typeless"
    t1 = _norm(types[1].name) if (len(types) > 1 and types[1]) else "typeless"

    # Base types (pre-Tera) — pull from pokedex if available
    dex_types = dex_entry.get("types", [])
    bt0 = _norm(dex_types[0]) if len(dex_types) >= 1 else t0
    bt1 = _norm(dex_types[1]) if len(dex_types) >= 2 else "typeless"

    # HP — for our side max_hp is real; for opponents poke-env reports % out of 100
    if is_opponent:
        # Compute real max_hp from base stats so damage scales correctly
        base_hp = dex_entry.get("baseStats", {}).get("hp", 100)
        real_max_hp = _opponent_stat(base_hp, level, is_hp=True)
        frac = float(poke.current_hp_fraction or 0)
        if poke.fainted:
            real_max_hp = real_max_hp or 1
            hp = 0
        else:
            hp = int(round(real_max_hp * frac))
        max_hp = real_max_hp or 1
    else:
        max_hp = int(poke.max_hp or 1) or 1
        hp = int(poke.current_hp or 0) if not poke.fainted else 0

    # Ability — guess first option for unrevealed opponents
    if poke.ability:
        ability = _norm(poke.ability)
    elif is_opponent and poke.possible_abilities:
        first = next(iter(poke.possible_abilities), "")
        ability = _norm(first)
    else:
        ability = "none"

    # Item — poke-env uses None / "unknown_item" / a string
    item_raw = poke.item
    if item_raw is None or item_raw == "":
        item = "none"
    elif item_raw == "unknown_item":
        item = "unknown_item"
    else:
        item = _norm(item_raw)

    # Stats: real values for our side, derived for the opponent
    stats = poke.stats or {}
    if is_opponent or not stats or any(stats.get(k) is None for k in ("atk", "def", "spa", "spd", "spe")):
        base_stats = dex_entry.get("baseStats", {})
        atk = _opponent_stat(base_stats.get("atk", base_stats.get("attack", 100)), level)
        df = _opponent_stat(base_stats.get("def", base_stats.get("defense", 100)), level)
        spa = _opponent_stat(base_stats.get("spa", base_stats.get("special-attack", 100)), level)
        spd = _opponent_stat(base_stats.get("spd", base_stats.get("special-defense", 100)), level)
        spe = _opponent_stat(base_stats.get("spe", base_stats.get("speed", 100)), level)
    else:
        atk = int(stats.get("atk") or 100)
        df = int(stats.get("def") or 100)
        spa = int(stats.get("spa") or 100)
        spd = int(stats.get("spd") or 100)
        spe = int(stats.get("spe") or 100)

    status = _STATUS_MAP.get(poke.status, "None")

    # Moves — at most 4. Disabled isn't reliably exposed by poke-env on the
    # active pokemon (request_json carries it), so we don't try to mark
    # disabled here. poke-engine's MCTS still won't pick a move with 0 PP.
    moves: list[pe.Move] = []
    for mid, move_obj in list((poke.moves or {}).items())[:4]:
        moves.append(
            pe.Move(
                id=_norm(mid),
                disabled=False,
                pp=int(getattr(move_obj, "current_pp", 16) or 16),
            )
        )
    while len(moves) < 4:
        moves.append(pe.Move(id="none", disabled=True, pp=0))

    # Tera
    tera_type = "typeless"
    if poke.tera_type is not None:
        try:
            tera_type = _norm(poke.tera_type.name)
        except AttributeError:
            tera_type = _norm(poke.tera_type)
    terastallized = bool(getattr(poke, "is_terastallized", False) or getattr(poke, "terastallized", False))

    weight_kg = float(dex_entry.get("weightkg", 0) or 0)

    return pe.Pokemon(
        id=species_id or "pikachu",
        level=level,
        types=(t0, t1),
        base_types=(bt0, bt1),
        hp=hp,
        maxhp=max_hp,
        ability=ability,
        base_ability=ability,
        item=item,
        nature="serious",
        evs=(85, 85, 85, 85, 85, 85),
        attack=atk,
        defense=df,
        special_attack=spa,
        special_defense=spd,
        speed=spe,
        status=status,
        rest_turns=0,
        sleep_turns=int(poke.status_counter or 0) if poke.status == Status.SLP else 0,
        weight_kg=weight_kg,
        moves=moves,
        tera_type=tera_type,
        terastallized=terastallized,
    )


def _side_conditions_to_pe(sc: dict) -> pe.SideConditions:
    sc = sc or {}

    def _has(key) -> int:
        return 1 if key in sc else 0

    def _val(key) -> int:
        return int(sc.get(key, 0)) if key in sc else 0

    return pe.SideConditions(
        spikes=_val(SideCondition.SPIKES),
        toxic_spikes=_val(SideCondition.TOXIC_SPIKES),
        stealth_rock=_has(SideCondition.STEALTH_ROCK),
        sticky_web=_has(SideCondition.STICKY_WEB),
        tailwind=_has(SideCondition.TAILWIND),
        reflect=_has(SideCondition.REFLECT),
        light_screen=_has(SideCondition.LIGHT_SCREEN),
        aurora_veil=_has(SideCondition.AURORA_VEIL),
        safeguard=_has(SideCondition.SAFEGUARD),
        mist=_has(SideCondition.MIST),
    )


_BOOST_KEYS = {
    "atk": "attack_boost",
    "def": "defense_boost",
    "spa": "special_attack_boost",
    "spd": "special_defense_boost",
    "spe": "speed_boost",
    "accuracy": "accuracy_boost",
    "evasion": "evasion_boost",
}


def _boosts_kwargs(active) -> dict:
    boosts = (active.boosts if active else {}) or {}
    return {pe_key: int(boosts.get(env_key, 0) or 0) for env_key, pe_key in _BOOST_KEYS.items()}


def _build_side(battle, is_opponent: bool, force_switch: bool) -> pe.Side:
    if is_opponent:
        active = battle.opponent_active_pokemon
        team = battle.opponent_team or {}
        sc = battle.opponent_side_conditions
    else:
        active = battle.active_pokemon
        team = battle.team or {}
        sc = battle.side_conditions

    # Active occupies index 0; reserves fill 1-5. Sort reserves: alive then fainted.
    pokemon: list[pe.Pokemon] = [_pokemon_to_pe(active, is_opponent)]
    bench = [p for p in team.values() if p is not active]
    bench.sort(key=lambda p: (p.fainted, p.species or ""))
    for p in bench:
        if len(pokemon) >= 6:
            break
        pokemon.append(_pokemon_to_pe(p, is_opponent))
    while len(pokemon) < 6:
        pokemon.append(pe.Pokemon.create_fainted())

    # Volatile statuses come from the active pokemon's effects dict
    volatile: set[str] = set()
    if active and getattr(active, "effects", None):
        for eff in active.effects.keys():
            name = getattr(eff, "name", None) or str(eff)
            volatile.add(_norm(name))

    return pe.Side(
        pokemon=pokemon,
        active_index=pe.PokemonIndex.P0,
        baton_passing=False,
        shed_tailing=False,
        wish=(0, 0),
        future_sight=(0, "0"),
        force_switch=force_switch,
        force_trapped=False,  # poke-env exposes this only on our side and inconsistently
        slow_uturn_move=False,
        volatile_statuses=volatile,
        substitute_health=0,
        last_used_move="move:none",
        switch_out_move_second_saved_move="none",
        side_conditions=_side_conditions_to_pe(sc),
        **_boosts_kwargs(active),
    )


def _battle_to_pe_state(battle) -> pe.State:
    weather_map = [
        (Weather.SUNNYDAY, "sun"),
        (Weather.DESOLATELAND, "harshsun"),
        (Weather.RAINDANCE, "rain"),
        (Weather.PRIMORDIALSEA, "heavyrain"),
        (Weather.SANDSTORM, "sand"),
        (Weather.SNOWSCAPE, "snow"),
        (Weather.HAIL, "hail"),
    ]
    weather = "none"
    weather_turns = 0
    bw = battle.weather or {}
    for w, name in weather_map:
        if w in bw:
            weather = name
            weather_turns = int(bw.get(w, 0) or 0)
            break

    fields = battle.fields or {}
    terrain_map = [
        (Field.ELECTRIC_TERRAIN, "electricterrain"),
        (Field.GRASSY_TERRAIN, "grassyterrain"),
        (Field.MISTY_TERRAIN, "mistyterrain"),
        (Field.PSYCHIC_TERRAIN, "psychicterrain"),
    ]
    terrain = "none"
    terrain_turns = 0
    for f, name in terrain_map:
        if f in fields:
            terrain = name
            terrain_turns = int(fields.get(f, 0) or 0)
            break

    trick_room = Field.TRICK_ROOM in fields
    tr_turns = int(fields.get(Field.TRICK_ROOM, 0) or 0) if trick_room else 0

    side_one = _build_side(battle, is_opponent=False, force_switch=bool(battle.force_switch))
    side_two = _build_side(battle, is_opponent=True, force_switch=False)

    return pe.State(
        side_one=side_one,
        side_two=side_two,
        weather=weather,
        weather_turns_remaining=weather_turns,
        terrain=terrain,
        terrain_turns_remaining=terrain_turns,
        trick_room=trick_room,
        trick_room_turns_remaining=tr_turns,
        team_preview=False,
    )


def _select_choice_from_mcts(mcts_result) -> Optional[str]:
    if not mcts_result.side_one:
        return None
    best = max(mcts_result.side_one, key=lambda r: r.visits)
    return best.move_choice


def _choice_to_order(player: Player, battle, choice: Optional[str]):
    """Map a poke-engine move_choice string back to a poke-env BattleOrder.

    poke-engine returns:
      "<move_id>"           — use this move
      "<move_id>-tera"      — use this move + terastallize
      "<move_id>-mega"      — use this move + mega evolve
      "switch <species>"    — switch to this pokemon by species id

    The chosen move/switch is validated against battle.available_moves and
    battle.available_switches; if MCTS picks something poke-env considers
    illegal (disabled by Choice/Taunt/Encore, 0 PP, fainted target, or just
    an unknown id), we fall back to choose_random_move. Without this check
    the server rejects the order silently and poke-env loops forever.
    """
    if not choice:
        return player.choose_random_move(battle)

    tera = mega = False
    if choice.endswith("-tera"):
        choice = choice[: -len("-tera")]
        tera = True
    elif choice.endswith("-mega"):
        choice = choice[: -len("-mega")]
        mega = True

    if choice.startswith("switch "):
        target = _norm(choice.split(" ", 1)[1])
        for p in (battle.available_switches or []):
            if _norm(p.species) == target or _norm(getattr(p, "base_species", "")) == target:
                return player.create_order(p)
        return player.choose_random_move(battle)

    move_id = _norm(choice)
    available = battle.available_moves or []
    matched = None
    for m in available:
        if _norm(getattr(m, "id", "") or getattr(m, "_id", "")) == move_id:
            matched = m
            break
    if matched is None:
        return player.choose_random_move(battle)

    if tera and not battle.can_tera:
        tera = False
    if mega and not battle.can_mega_evolve:
        mega = False
    try:
        return player.create_order(matched, terastallize=tera, mega=mega)
    except TypeError:
        return player.create_order(matched)


class PokeEnginePlayer(Player):
    """poke-env Player driving Foul Play's MCTS via poke-engine in-process.

    Each choose_move call serializes the live Battle into a poke_engine.State
    and runs MCTS for `search_time_ms` milliseconds. The most-visited move
    from the MCTS root is chosen. MCTS releases the GIL during search, so we
    run it in a thread pool to keep the asyncio event loop responsive.
    """

    # Hard cap on consecutive choose_move calls for the same (battle_tag, turn).
    # If MCTS keeps producing an invalid order Showdown rejects, we have to
    # break the loop or battle_against hangs forever.
    _MAX_RETRIES_PER_TURN = 4

    def __init__(self, search_time_ms: int = 200, **kwargs):
        super().__init__(**kwargs)
        self.search_time_ms = int(search_time_ms)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"pe-mcts-{id(self):x}"
        )
        # {battle_tag: (last_turn, count_of_calls_at_this_turn)}
        self._turn_call_count: dict[str, tuple[int, int]] = {}
        # Never-connected smart bot used as panic fallback. Falls back to
        # random move if smart-bot init fails or its choose_move raises.
        self._fallback_bot = None
        if _FallbackBotCls is not None:
            try:
                self._fallback_bot = _FallbackBotCls(
                    account_configuration=AccountConfiguration(
                        f"pefb-{id(self):x}"[:18], None
                    ),
                    start_listening=False,
                )
            except BaseException as e:
                logger.warning("PokeEngine fallback-bot init failed (%s): falling back to random",
                               type(e).__name__)
                self._fallback_bot = None

    def _smart_fallback(self, battle):
        if self._fallback_bot is None:
            return self.choose_random_move(battle)
        try:
            return self._fallback_bot.choose_move(battle)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            logger.warning("PokeEngine smart-fallback raised for %s: %s (%s); using random",
                           battle.battle_tag, e, type(e).__name__)
            return self.choose_random_move(battle)

    def _bump_turn_counter(self, battle) -> int:
        last_turn, n = self._turn_call_count.get(battle.battle_tag, (-1, 0))
        if battle.turn == last_turn:
            n += 1
        else:
            n = 1
        self._turn_call_count[battle.battle_tag] = (battle.turn, n)
        return n

    async def choose_move(self, battle):
        n_calls_this_turn = self._bump_turn_counter(battle)

        # Stuck-turn fallback: after N retries on the same turn, hand off to
        # poke-env's random/default order. Showdown will then unstick us.
        if n_calls_this_turn > self._MAX_RETRIES_PER_TURN:
            return self.choose_random_move(battle)

        # poke-engine raises pyo3 PanicException (a BaseException subclass,
        # NOT a regular Exception) on niche edge cases like
        # `'Encore should not be active when last used move is not a move'`
        # (Run #7 pattern) or `'InvalidWeight'` from mcts.rs:112 when all
        # generate_instructions_from_move_pair percentages are 0 (Run #9
        # pattern — see docs/TODO_MCTS_RUN9.md). Plain `except Exception`
        # doesn't catch BaseException, so the panic propagates up through
        # poke-env's _handle_battle_request and crashes the listener task —
        # battle hangs, send_challenges times out (~600s), iter loses one
        # game. Catching BaseException with explicit re-raise for
        # KeyboardInterrupt / SystemExit lets us fall back to random move
        # cleanly while preserving Ctrl-C semantics.
        # S68 2026-06-10: exc_info=False on panic warnings — formatting the
        # full Rust→Python traceback per turn was a significant chunk of
        # per-panic recovery cost. The single-line warning (battle_tag +
        # exc type + str(e)) is sufficient for diagnosis; full tracebacks
        # add no info since the panic origin is always inside poke-engine.
        try:
            state = _battle_to_pe_state(battle)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            logger.warning("PokeEngine state build failed for %s: %s (%s)",
                           battle.battle_tag, e, type(e).__name__)
            return self._smart_fallback(battle)

        loop = asyncio.get_event_loop()
        try:
            mcts_result = await loop.run_in_executor(
                self._executor,
                pe.monte_carlo_tree_search,
                state,
                self.search_time_ms,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            logger.warning("PokeEngine MCTS failed for %s: %s (%s)",
                           battle.battle_tag, e, type(e).__name__)
            return self._smart_fallback(battle)

        try:
            choice = _select_choice_from_mcts(mcts_result)
            return _choice_to_order(self, battle, choice)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            logger.warning("PokeEngine choice translation failed for %s: %s (%s)",
                           battle.battle_tag, e, type(e).__name__)
            return self._smart_fallback(battle)

    def _battle_finished_callback(self, battle):
        self._turn_call_count.pop(battle.battle_tag, None)
        super()._battle_finished_callback(battle)

    def __del__(self):
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
