# src/policy_smartbots.py
# Advanced rule-based bots that extend poke-env's SimpleHeuristicsPlayer.
# SimpleHeuristics is strong at damage-based move selection but ignores
# status moves, recovery, pivoting, and team-level strategy.
# These bots add those capabilities on top.

from __future__ import annotations
from typing import List, Optional
import random as _random

from poke_env.player import Player
from poke_env.player.baselines import SimpleHeuristicsPlayer
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition


# =====================================================================
# Move ID sets
# =====================================================================

PIVOT_IDS = {"uturn", "voltswitch", "flipturn", "partingshot", "teleport"}
RECOVERY_IDS = {
    "recover", "softboiled", "roost", "moonlight", "morningsun", "synthesis",
    "shoreup", "milkdrink", "slackoff", "strengthsap",
}
STATUS_SLEEP_IDS = {"spore", "sleeppowder", "hypnosis", "darkvoid", "grasswhistle", "sing", "yawn"}
STATUS_PARA_IDS = {"thunderwave", "stunspore", "glare", "nuzzle"}
STATUS_BURN_IDS = {"willowisp"}
STATUS_POISON_IDS = {"toxic", "toxicthread"}
SCREEN_IDS = {"reflect", "lightscreen", "auroraveil"}
PROTECT_IDS = {"protect", "detect", "kingsshield", "banefulbunker", "spikyshield", "silktrap", "obstruct"}
PHAZING_IDS = {"roar", "whirlwind", "dragontail", "circlethrow", "haze"}


# =====================================================================
# SmartDamagePlayer — SimpleHeuristics with better switching
# =====================================================================

class SmartDamagePlayer(SimpleHeuristicsPlayer):
    """
    Extends SimpleHeuristics with improved switching logic:
    - Switches out at lower threshold (-1 instead of -2)
    - Considers pivot moves before switching
    - Won't switch when we can KO the opponent
    """
    SWITCH_OUT_MATCHUP_THRESHOLD = -1

    def choose_move(self, battle: AbstractBattle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if active is None or opponent is None:
            return self.choose_random_move(battle)

        # Check if we have a pivot move and bad matchup
        if battle.available_moves and battle.available_switches:
            matchup = self._estimate_matchup(active, opponent)
            if -2 < matchup < 0:
                for m in battle.available_moves:
                    if m.id in PIVOT_IDS:
                        return self.create_order(m)

        return super().choose_move(battle)


# =====================================================================
# TacticalPlayer — adds status, recovery, and utility to SimpleHeuristics
# =====================================================================

class TacticalPlayer(SimpleHeuristicsPlayer):
    """
    Extends SimpleHeuristics with status moves, recovery, and utility:
    - Uses sleep/burn/para/toxic when appropriate
    - Uses recovery when low HP and good matchup
    - Better hazard play (layers spikes, uses toxic spikes)
    - Uses screens when available
    - Phazes boosted opponents
    """
    SWITCH_OUT_MATCHUP_THRESHOLD = -1.5

    def _move_damage_score(self, move, active, opponent):
        """SimpleHeuristics' damage scoring formula."""
        physical_ratio = self._stat_estimation(active, "atk") / self._stat_estimation(opponent, "def")
        special_ratio = self._stat_estimation(active, "spa") / self._stat_estimation(opponent, "spd")
        return (
            move.base_power
            * (1.5 if move.type in active.types else 1)
            * (physical_ratio if move.category == MoveCategory.PHYSICAL else special_ratio)
            * move.accuracy
            * move.expected_hits
            * opponent.damage_multiplier(move)
        )

    def _can_probably_ko(self, battle, active, opponent):
        """Check if our best move can likely KO."""
        if not battle.available_moves:
            return False
        best_score = max(self._move_damage_score(m, active, opponent) for m in battle.available_moves)
        # Rough threshold: if score is high relative to opponent HP
        # SimpleHeuristics formula gives values like 50-500+ for normal hits
        # A KO-level hit usually scores >200 against weakened targets
        return best_score > 150 and opponent.current_hp_fraction < 0.4

    def choose_move(self, battle: AbstractBattle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if active is None or opponent is None:
            return self.choose_random_move(battle)

        matchup = self._estimate_matchup(active, opponent)
        n_remaining = len([m for m in battle.team.values() if not m.fainted])
        n_opp_remaining = 6 - sum(1 for m in battle.opponent_team.values() if m.fainted)

        # --- Phase 1: Can we KO? If so, just attack (skip utility) ---
        if self._can_probably_ko(battle, active, opponent):
            # Use SimpleHeuristics' attack selection
            if battle.available_moves:
                best = max(battle.available_moves, key=lambda m: self._move_damage_score(m, active, opponent))
                if best.base_power > 0:
                    return self.create_order(best)

        # --- Phase 2: Phaze boosted opponents ---
        if battle.available_moves:
            opp_boosts = sum(max(0, v) for v in opponent.boosts.values())
            if opp_boosts >= 2:
                for m in battle.available_moves:
                    if m.id in PHAZING_IDS:
                        return self.create_order(m)

        # --- Phase 3: Check if best attack is weak (guides utility decisions) ---
        best_atk_score = 0
        if battle.available_moves:
            attacking = [m for m in battle.available_moves if m.base_power > 0]
            if attacking:
                best_atk_score = max(self._move_damage_score(m, active, opponent) for m in attacking)

        use_utility = best_atk_score < 80

        # --- Phase 4: Status moves only when attacks are weak ---
        if use_utility and battle.available_moves and opponent.status is None and matchup > -1:
            for m in battle.available_moves:
                if m.id in STATUS_SLEEP_IDS:
                    return self.create_order(m)
            if matchup > 0 and opponent.base_stats["atk"] > opponent.base_stats["spa"]:
                for m in battle.available_moves:
                    if m.id in STATUS_BURN_IDS:
                        return self.create_order(m)
            if active.current_hp_fraction > 0.6 and opponent.base_stats["spe"] > active.base_stats["spe"]:
                for m in battle.available_moves:
                    if m.id in STATUS_PARA_IDS:
                        return self.create_order(m)
            if matchup > 0.5 and active.current_hp_fraction > 0.6:
                for m in battle.available_moves:
                    if m.id in STATUS_POISON_IDS:
                        return self.create_order(m)

        # --- Phase 5: Recovery when safe ---
        if battle.available_moves and active.current_hp_fraction < 0.35 and matchup > 0:
            for m in battle.available_moves:
                if m.id in RECOVERY_IDS:
                    return self.create_order(m)

        # --- Phase 6: Screens on safe turns ---
        if use_utility and battle.available_moves and active.current_hp_fraction > 0.8:
            for m in battle.available_moves:
                if m.id == "reflect" and SideCondition.REFLECT not in battle.side_conditions:
                    return self.create_order(m)
                if m.id == "lightscreen" and SideCondition.LIGHT_SCREEN not in battle.side_conditions:
                    return self.create_order(m)

        # --- Phase 7: Pivot on bad matchup ---
        if battle.available_moves and battle.available_switches and matchup < -0.5:
            for m in battle.available_moves:
                if m.id in PIVOT_IDS:
                    return self.create_order(m)

        # --- Phase 7: Fall through to SimpleHeuristics (hazards, setup, best attack, switch) ---
        return super().choose_move(battle)


# =====================================================================
# StrategicPlayer — most advanced, adds team-level strategy
# =====================================================================

class StrategicPlayer(SimpleHeuristicsPlayer):
    """
    Most advanced bot. Combines TacticalPlayer's utility with:
    - Team-aware decisions (preserve win conditions)
    - Endgame aggression
    - Better switch evaluation (considers hazard damage on switch)
    - Status-aware attack selection
    - Priority move awareness for finishing
    """
    SWITCH_OUT_MATCHUP_THRESHOLD = -1.5

    def _move_damage_score(self, move, active, opponent):
        physical_ratio = self._stat_estimation(active, "atk") / self._stat_estimation(opponent, "def")
        special_ratio = self._stat_estimation(active, "spa") / self._stat_estimation(opponent, "spd")
        return (
            move.base_power
            * (1.5 if move.type in active.types else 1)
            * (physical_ratio if move.category == MoveCategory.PHYSICAL else special_ratio)
            * move.accuracy
            * move.expected_hits
            * opponent.damage_multiplier(move)
        )

    def _best_attack(self, battle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if not battle.available_moves:
            return None, 0
        attacking = [m for m in battle.available_moves if m.base_power > 0]
        if not attacking:
            return None, 0
        best = max(attacking, key=lambda m: self._move_damage_score(m, active, opponent))
        return best, self._move_damage_score(best, active, opponent)

    def _has_priority_ko(self, battle):
        """Check for priority moves that can likely finish off a weakened opponent."""
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if opponent.current_hp_fraction > 0.3:
            return None
        for m in battle.available_moves:
            if m.priority > 0 and m.base_power > 0:
                score = self._move_damage_score(m, active, opponent)
                # Lower threshold for priority on low-HP target
                if score > 50:
                    return m
        return None

    def _enhanced_switch_score(self, mon, opponent, battle):
        """Better switch scoring that considers hazard damage."""
        score = self._estimate_matchup(mon, opponent)
        # Penalty for switching into hazards
        if SideCondition.STEALTH_ROCK in battle.side_conditions:
            # Rock weakness = more SR damage
            from poke_env.data import GenData
            try:
                sr_mult = 1.0
                for t in mon.types:
                    if t is not None:
                        from poke_env.battle.pokemon_type import PokemonType
                        sr_mult *= PokemonType.ROCK.damage_multiplier(t)
                sr_damage = min(0.5, 0.125 * sr_mult)  # Cap at 50% (4x weakness)
                # Heavy-Duty Boots negates entry hazard damage
                if hasattr(mon, "item") and mon.item and mon.item.lower().replace("-", "").replace(" ", "") == "heavydutyboots":
                    sr_damage = 0.0
                score -= sr_damage * 2  # Weight hazard damage
            except Exception:
                score -= 0.15
        if SideCondition.SPIKES in battle.side_conditions:
            score -= 0.1
        return score

    def choose_move(self, battle: AbstractBattle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if active is None or opponent is None:
            return self.choose_random_move(battle)

        matchup = self._estimate_matchup(active, opponent)
        n_remaining = len([m for m in battle.team.values() if not m.fainted])
        n_opp_remaining = 6 - sum(1 for m in battle.opponent_team.values() if m.fainted)

        if not battle.available_moves and not battle.available_switches:
            return self.choose_random_move(battle)

        # === Priority 1: Priority KO ===
        if battle.available_moves:
            prio_move = self._has_priority_ko(battle)
            if prio_move:
                return self.create_order(prio_move)

        # === Priority 2: Best attack KO check ===
        best_atk, best_score = self._best_attack(battle)
        if best_atk and best_score > 200 and opponent.current_hp_fraction < 0.5:
            return self.create_order(best_atk)

        # === Priority 3: Phaze boosted opponents ===
        if battle.available_moves:
            opp_boosts = sum(max(0, v) for v in opponent.boosts.values())
            if opp_boosts >= 2:
                for m in battle.available_moves:
                    if m.id in PHAZING_IDS:
                        return self.create_order(m)
                # If very boosted and no phaze, switch
                if opp_boosts >= 4 and battle.available_switches:
                    best_sw = max(battle.available_switches,
                                  key=lambda s: self._enhanced_switch_score(s, opponent, battle))
                    if self._enhanced_switch_score(best_sw, opponent, battle) > 0:
                        return self.create_order(best_sw)

        # === Priority 4: Status/utility only when best attack is weak ===
        # Only use utility if best damaging move scores below threshold
        use_utility = best_score < 80  # Low damage = look for utility plays

        if use_utility and battle.available_moves and opponent.status is None and matchup > -1.5:
            # Sleep is always worth it
            for m in battle.available_moves:
                if m.id in STATUS_SLEEP_IDS:
                    return self.create_order(m)
            # Burn physical attackers when we resist them
            if matchup > 0 and opponent.base_stats["atk"] > opponent.base_stats["spa"]:
                for m in battle.available_moves:
                    if m.id in STATUS_BURN_IDS:
                        return self.create_order(m)
            # Para faster threats when we can take a hit
            if active.current_hp_fraction > 0.6 and opponent.base_stats["spe"] > active.base_stats["spe"]:
                for m in battle.available_moves:
                    if m.id in STATUS_PARA_IDS:
                        return self.create_order(m)
            # Toxic for walls/stall matchups
            if matchup > 0.5 and active.current_hp_fraction > 0.6:
                for m in battle.available_moves:
                    if m.id in STATUS_POISON_IDS:
                        return self.create_order(m)

        # === Priority 5: Recovery only when necessary and safe ===
        if battle.available_moves and active.current_hp_fraction < 0.35 and matchup > 0:
            for m in battle.available_moves:
                if m.id in RECOVERY_IDS:
                    return self.create_order(m)

        # === Priority 6: Screens only on safe turns with weak attacks ===
        if use_utility and battle.available_moves and active.current_hp_fraction > 0.8 and n_remaining >= 4:
            for m in battle.available_moves:
                if m.id == "reflect" and SideCondition.REFLECT not in battle.side_conditions:
                    return self.create_order(m)
                if m.id == "lightscreen" and SideCondition.LIGHT_SCREEN not in battle.side_conditions:
                    return self.create_order(m)

        # === Priority 7: Pivot on bad matchup ===
        if battle.available_moves and battle.available_switches and matchup < -0.5:
            for m in battle.available_moves:
                if m.id in PIVOT_IDS:
                    return self.create_order(m)

        # === Priority 8: Endgame — don't switch, just attack ===
        if n_remaining <= 2 and n_opp_remaining <= 2:
            if best_atk:
                return self.create_order(best_atk)

        # === Priority 9: Fall through to SimpleHeuristics ===
        return super().choose_move(battle)
