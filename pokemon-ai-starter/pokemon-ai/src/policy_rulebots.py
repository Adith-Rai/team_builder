# src/policy_rulebots.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
from poke_env.player import Player
from poke_env.battle import SideCondition

# ---------- Helpers & scoring ----------

PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot"}
SETUP_MOVES = {
    "swordsdance", "dragondance", "nastyplot", "calmmind", "quiverdance",
    "bulkup", "shiftgear", "trailblaze", "agility", "rockpolish", "coil",
    "substitute"
}
HAZARD_MOVES = {"stealthrock", "spikes", "toxicspikes"}
REMOVAL_MOVES = {"rapidspin", "defog"}


def _type_multiplier(move, opp_types):
    """Approx damage multiplier based on typing."""
    try:
        mult = 1.0
        for t in (opp_types or []):
            mult *= move.type.damage_multiplier(t)
        return float(mult)
    except Exception:
        return 1.0


def _effective_power(move, active_pokemon=None):
    """Base power plus small bumps for priority and STAB."""
    bp = float(getattr(move, "base_power", 0) or 0)
    acc = move.accuracy if hasattr(move, "accuracy") else True
    acc = 1.0 if acc is True else float(acc) / 100.0 if acc > 1.0 else float(acc)
    stab = 1.5 if (active_pokemon and move.type in active_pokemon.types) else 1.0
    prio = float(getattr(move, "priority", 0) or 0)
    return bp * stab * acc * (1.2 if prio > 0 else 1.0)


def _best_attacking_move(battle):
    legal = [m for m in battle.available_moves if m is not None]
    if not legal:
        return None
    active = battle.active_pokemon
    opp = battle.opponent_active_pokemon
    opp_types = opp.types if opp else []
    scored = []
    for m in legal:
        eff = _type_multiplier(m, opp_types)
        ep = _effective_power(m, active)
        scored.append((eff, ep, m))
    # sort by effectiveness first, then effective power
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


def _best_pivot_move(battle):
    legal = {m.id: m for m in battle.available_moves if m is not None}
    for mid in PIVOT_MOVES:
        if mid in legal:
            return legal[mid]
    return None


def _best_switch_candidate(battle):
    """Pick a switch that best resists the opponent's STAB types."""
    switches = [p for p in battle.available_switches if p is not None]
    if not switches:
        return None

    opp = battle.opponent_active_pokemon
    opp_types = [t for t in (opp.types if opp else []) if t is not None]
    if not opp_types:
        return np.random.choice(switches)

    # For each opponent STAB type, compute the product of its effectiveness
    # against ALL of the defender's types (correct type-chart math:
    # e.g. Fire vs Water/Ground = Fire->Water * Fire->Ground).
    # Then take the MAX across opponent STAB types (worst case for defender).
    best = None
    for p in switches:
        worst_mult = 0.0
        try:
            for opp_t in opp_types:
                type_mult = 1.0
                for my_t in (p.types or []):
                    if my_t is not None:
                        type_mult *= opp_t.damage_multiplier(my_t)
                worst_mult = max(worst_mult, type_mult)
        except Exception:
            worst_mult = 1.0
        score = -worst_mult  # lower mult is better (less damage taken)
        if best is None or score > best[0]:
            best = (score, p)
    return best[1] if best else np.random.choice(switches)


# ---------- Bots ----------

class GreedySEPlayer(Player):
    """Greedy attacker: prefers SE hits, ties by effective power."""
    def choose_move(self, battle):
        best = _best_attacking_move(battle)
        if best is not None:
            return self.create_order(best)
        sw = _best_switch_candidate(battle)
        if sw is not None:
            return self.create_order(sw)
        return self.choose_random_move(battle)


class HazardSensePlayer(Player):
    """Hazards/utility first; otherwise attack, else switch to resist."""
    def choose_move(self, battle):
        if battle.available_moves:
            moves_by_id = {m.id: m for m in battle.available_moves if m is not None}

            # Place SR if not up
            if "stealthrock" in moves_by_id and SideCondition.STEALTH_ROCK not in battle.opponent_side_conditions:
                return self.create_order(moves_by_id["stealthrock"])

            # Clear our hazards if possible
            if any(x in moves_by_id for x in REMOVAL_MOVES):
                if any(k in battle.side_conditions for k in (SideCondition.STEALTH_ROCK, SideCondition.SPIKES, SideCondition.TOXIC_SPIKES, SideCondition.STICKY_WEB)):
                    if "rapidspin" in moves_by_id:
                        return self.create_order(moves_by_id["rapidspin"])
                    if "defog" in moves_by_id:
                        return self.create_order(moves_by_id["defog"])

        best = _best_attacking_move(battle)
        if best is not None:
            return self.create_order(best)

        sw = _best_switch_candidate(battle)
        if sw is not None:
            return self.create_order(sw)

        return self.choose_random_move(battle)


class SwitchAwareEscapePlayer(Player):
    """
    Tries to pivot/escape poor positions, then attack or switch intelligently.
    """
    def choose_move(self, battle):
        legal_moves = [m for m in battle.available_moves if m is not None]
        legal_switches = [p for p in battle.available_switches if p is not None]

        # Pivot if matchup looks bad
        pivot = _best_pivot_move(battle)
        if pivot is not None:
            best = _best_attacking_move(battle)
            if best is None:
                return self.create_order(pivot)
            opp = battle.opponent_active_pokemon
            eff = _type_multiplier(best, (opp.types if opp else []))
            if eff < 1.0:
                return self.create_order(pivot)

        # If effectively "locked" into one weak move, try to switch
        if len(legal_moves) == 1 and legal_switches:
            the_move = legal_moves[0]
            opp = battle.opponent_active_pokemon
            eff = _type_multiplier(the_move, (opp.types if opp else []))
            if _effective_power(the_move, battle.active_pokemon) <= 0 or eff < 1.0:
                sw = _best_switch_candidate(battle)
                if sw is not None:
                    return self.create_order(sw)

        # Attack normally
        best = _best_attacking_move(battle)
        if best is not None:
            return self.create_order(best)

        # If walled, switch to a resist
        sw = _best_switch_candidate(battle)
        if sw is not None:
            return self.create_order(sw)

        return self.choose_random_move(battle)


class SetupThenSweepPlayer(Player):
    """
    Boost first if safe & useful, then attack; switch if totally walled.
    """
    HEALTH_THRESH = 0.70

    def choose_move(self, battle):
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        legal_moves = [m for m in battle.available_moves if m is not None]
        if legal_moves:
            moves_by_id = {m.id: m for m in legal_moves}

            best_attack = _best_attacking_move(battle)
            best_eff = _type_multiplier(best_attack, (opp.types if opp else [])) if best_attack else 0.0
            hp_ok = (me.current_hp_fraction if me else 1.0) >= self.HEALTH_THRESH

            have_setup = any(mid in moves_by_id for mid in SETUP_MOVES)
            # Setup is safe when HP is healthy AND our best attack is at least neutral
            # (we can threaten the opponent after boosting)
            setup_safe = hp_ok and (best_eff >= 1.0)

            # Don't waste turns setting up if already at max boosts
            if me:
                atk_boost = me.boosts.get("atk", 0)
                spa_boost = me.boosts.get("spa", 0)
                max_offensive_boost = max(atk_boost, spa_boost)
                if max_offensive_boost >= 6:
                    have_setup = False

            if have_setup and setup_safe:
                for prefer in ("dragondance", "swordsdance", "nastyplot", "calmmind", "quiverdance"):
                    if prefer in moves_by_id:
                        return self.create_order(moves_by_id[prefer])
                for mid in SETUP_MOVES:
                    if mid in moves_by_id:
                        return self.create_order(moves_by_id[mid])

            if best_attack is not None:
                return self.create_order(best_attack)

        sw = _best_switch_candidate(battle)
        if sw is not None:
            return self.create_order(sw)

        return self.choose_random_move(battle)
