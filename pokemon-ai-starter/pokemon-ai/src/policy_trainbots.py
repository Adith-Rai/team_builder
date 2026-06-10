# src/policy_trainbots.py
# Bots designed for TRAINING use (NOT eval) — distinct styles + targeted
# failure-mode coverage that the eval bots (SH / SmartDmg / Tactical /
# Strategic in policy_smartbots.py) don't currently provide.
#
# Built S68 (2026-06-09) after the Run #7 decision-pattern analysis confirmed
# the no-anchor model develops a setup-spam playstyle hard-punished by
# deterministic damage maximizers. See
# `memory/project_s68_run7_decision_pattern_findings_2026_06_09.md` for the
# evidence.
#
# Bots in this file MUST stay out of the eval set to preserve eval validity.

from __future__ import annotations
from typing import Optional

from poke_env.player import Player
from poke_env.player.baselines import SimpleHeuristicsPlayer
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.side_condition import SideCondition

from policy_rulebots import (
    _best_attacking_move,
    _best_pivot_move,
    _best_switch_candidate,
    _type_multiplier,
    _effective_power,
)


# Move-id sets reused from policy_smartbots.py — keep separate here so
# this file can be loaded without that import.
PIVOT_IDS = {"uturn", "voltswitch", "flipturn", "partingshot", "teleport"}
SLEEP_IDS = {"spore", "sleeppowder", "hypnosis", "darkvoid", "grasswhistle", "sing", "yawn"}
PARA_IDS = {"thunderwave", "stunspore", "glare", "nuzzle"}
BURN_IDS = {"willowisp"}
PHAZE_IDS = {"roar", "whirlwind", "dragontail", "circlethrow", "haze", "clearsmog"}
TAUNT_IDS = {"taunt"}
SETUP_IDS = {
    "swordsdance", "dragondance", "nastyplot", "calmmind", "quiverdance",
    "bulkup", "shiftgear", "trailblaze", "agility", "rockpolish", "coil",
    "growth", "tailglow", "geomancy", "bellydrum", "shellsmash", "filletaway",
    "irondefense", "cosmicpower", "cottonguard", "stockpile", "acidarmor",
}


# ====================================================================
# AntiSetupBot — designed to maximally punish setup-spam playstyles
# ====================================================================

class AntiSetupBot(SimpleHeuristicsPlayer):
    """Maximally punishes opponents who use setup moves.

    Priority order in choose_move:
      1. Sleep status (most setup-disabling status) when available
      2. Phaze (Roar/Whirlwind/Dragon Tail/Haze/Clear Smog) when opp has
         ANY positive boost (lower threshold than Strategic's +2)
      3. Taunt when opp has any boost OR matchup is favorable
         (preempt future setups + recovery)
      4. Priority damage when opp has boost (chip before sweep)
      5. Switch to best resist when opp has heavy boost (+2 cumulative)
         and we have a switch that resists their STAB
      6. Burn physical attackers / paralyze fast threats when matchup OK
      7. Otherwise max-damage attack (NEVER sets up itself)

    Inherits _estimate_matchup, _stat_estimation, ENTRY_HAZARDS,
    ANTI_HAZARDS_MOVES from SimpleHeuristicsPlayer for general play heuristics.

    Note: this bot intentionally does NOT use setup moves itself, so the
    training pool gets a "consistent punisher" rather than a "setup vs setup"
    mirror match.
    """

    SWITCH_OUT_MATCHUP_THRESHOLD = -1.5

    def _opp_boost_sum(self, opponent) -> int:
        """Sum of positive opponent boosts (atk, spa, spe, etc.)."""
        try:
            return sum(max(0, v) for v in opponent.boosts.values())
        except Exception:
            return 0

    def _move_damage_score(self, move, active, opponent):
        """SH-style damage score (mirror of policy_smartbots formula)."""
        try:
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
        except Exception:
            return 0.0

    def _best_attack(self, battle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if not battle.available_moves or opponent is None:
            return None
        attacking = [m for m in battle.available_moves if m.base_power > 0]
        if not attacking:
            return None
        return max(attacking, key=lambda m: self._move_damage_score(m, active, opponent))

    def _best_resist_switch(self, battle, opponent):
        """Best switch by type-resist to opponent's STAB types."""
        if not battle.available_switches or opponent is None:
            return None
        opp_types = [t for t in (opponent.types or []) if t is not None]
        if not opp_types:
            return None
        best = None
        for mon in battle.available_switches:
            if mon is None:
                continue
            worst = 0.0
            try:
                for ot in opp_types:
                    mult = 1.0
                    for mt in (mon.types or []):
                        if mt is not None:
                            mult *= ot.damage_multiplier(mt)
                    worst = max(worst, mult)
            except Exception:
                worst = 1.0
            # Lower mult = better resist; prefer mons that take <=1.0 from STAB
            if best is None or worst < best[0]:
                best = (worst, mon)
        if best is None or best[0] > 1.0:
            return None  # No switch actually resists
        return best[1]

    def choose_move(self, battle: AbstractBattle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if active is None or opponent is None:
            return self.choose_random_move(battle)

        if not battle.available_moves and not battle.available_switches:
            return self.choose_random_move(battle)

        moves_by_id = {m.id: m for m in battle.available_moves if m is not None}
        opp_boost_sum = self._opp_boost_sum(opponent)
        matchup = self._estimate_matchup(active, opponent)

        # --- 1. Sleep — strongest setup disabler ---
        # Apply when opp has no status and matchup isn't terrible.
        if opp_boost_sum >= 1 and opponent.status is None and matchup > -1.5:
            for mid in SLEEP_IDS:
                if mid in moves_by_id:
                    return self.create_order(moves_by_id[mid])

        # --- 2. Phaze — clear opponent boosts ---
        # Lower threshold than Strategic (which fires at boost >= 2).
        if opp_boost_sum >= 1:
            for mid in PHAZE_IDS:
                if mid in moves_by_id:
                    return self.create_order(moves_by_id[mid])

        # --- 3. Taunt — preempt future setups / recovery ---
        # Use when matchup is OK (we'll live to benefit) and opp is healthy
        # enough to potentially setup again.
        if "taunt" in moves_by_id and matchup > -0.5 and opponent.current_hp_fraction > 0.5:
            return self.create_order(moves_by_id["taunt"])

        # --- 4. Priority damage when opp boosted ---
        # Chip them down before they sweep. Worth even if not lethal.
        if opp_boost_sum >= 1 and battle.available_moves:
            prio_attacks = [m for m in battle.available_moves
                            if m.priority > 0 and m.base_power > 0]
            if prio_attacks:
                best_prio = max(prio_attacks,
                                key=lambda m: self._move_damage_score(m, active, opponent))
                return self.create_order(best_prio)

        # --- 5. Switch to resist when opp is heavily boosted ---
        if opp_boost_sum >= 2 and battle.available_switches:
            sw = self._best_resist_switch(battle, opponent)
            if sw is not None:
                return self.create_order(sw)

        # --- 6. Cripple status (burn/para) when matchup is positive ---
        if opponent.status is None and matchup > 0 and battle.available_moves:
            # Burn physical attackers
            if opponent.base_stats.get("atk", 0) > opponent.base_stats.get("spa", 0):
                for mid in BURN_IDS:
                    if mid in moves_by_id:
                        return self.create_order(moves_by_id[mid])
            # Para fast threats (when we'd otherwise be outsped)
            if opponent.base_stats.get("spe", 0) > active.base_stats.get("spe", 0):
                for mid in PARA_IDS:
                    if mid in moves_by_id:
                        return self.create_order(moves_by_id[mid])

        # --- 7. Max-damage attack ---
        # NEVER set up ourselves. If we have available attacks, use the best
        # one. Otherwise switch (poke-env handles via SH _should_switch_out).
        best = self._best_attack(battle)
        if best is not None:
            return self.create_order(best)

        # --- 8. Fall-through: pivot or switch ---
        if battle.available_moves:
            for m in battle.available_moves:
                if m.id in PIVOT_IDS:
                    return self.create_order(m)
        if battle.available_switches:
            sw = self._best_resist_switch(battle, opponent)
            if sw is not None:
                return self.create_order(sw)

        return self.choose_random_move(battle)


# ====================================================================
# StrategicV2 — Strategic with loosened conditions for training only
# ====================================================================

class StrategicV2(SimpleHeuristicsPlayer):
    """Strategic-style decision logic with loosened firing conditions so
    its specialist branches actually fire.

    Differences from policy_smartbots.StrategicPlayer:
      - Phaze threshold lowered: +1 boost (was +2)
      - KO check loosened: score > 150 (was 200) OR opp HP < 60% (was 50%)
      - Pivot threshold loosened: matchup < -0.2 (was -0.5)
      - Endgame engages earlier: n <= 3 each side (was 2)
      - Utility window widened: best_score < 110 (was 80)
      - Priority KO threshold loosened: opp HP < 0.4 (was 0.3), score > 40 (was 50)

    Does NOT use setup moves itself (deliberate — same as AntiSetupBot, keeps
    training-pool diverse from setup-spam).

    Intentionally NOT used as an eval bot — preserves policy_smartbots.py's
    Strategic as the validation baseline.
    """

    SWITCH_OUT_MATCHUP_THRESHOLD = -1.5
    KO_SCORE_THRESH = 150
    KO_HP_THRESH = 0.6
    PRIO_KO_SCORE = 40
    PRIO_KO_HP = 0.4
    PHAZE_BOOST_THRESH = 1
    PIVOT_MATCHUP_THRESH = -0.2
    UTILITY_WINDOW = 110
    ENDGAME_N = 3

    def _move_damage_score(self, move, active, opponent):
        try:
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
        except Exception:
            return 0.0

    def _best_attack(self, battle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if not battle.available_moves or opponent is None:
            return None, 0.0
        attacking = [m for m in battle.available_moves if m.base_power > 0]
        if not attacking:
            return None, 0.0
        best = max(attacking, key=lambda m: self._move_damage_score(m, active, opponent))
        return best, self._move_damage_score(best, active, opponent)

    def _has_priority_ko(self, battle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if opponent is None or opponent.current_hp_fraction > self.PRIO_KO_HP:
            return None
        for m in battle.available_moves:
            if m.priority > 0 and m.base_power > 0:
                if self._move_damage_score(m, active, opponent) > self.PRIO_KO_SCORE:
                    return m
        return None

    def choose_move(self, battle: AbstractBattle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if active is None or opponent is None:
            return self.choose_random_move(battle)
        if not battle.available_moves and not battle.available_switches:
            return self.choose_random_move(battle)

        moves_by_id = {m.id: m for m in battle.available_moves if m is not None}
        matchup = self._estimate_matchup(active, opponent)
        n_remaining = len([m for m in battle.team.values() if not m.fainted])
        n_opp_remaining = 6 - sum(1 for m in battle.opponent_team.values() if m.fainted)
        opp_boost_sum = sum(max(0, v) for v in opponent.boosts.values())

        # === 1. Priority KO (loosened) ===
        if battle.available_moves:
            prio = self._has_priority_ko(battle)
            if prio is not None:
                return self.create_order(prio)

        # === 2. Best-attack KO (loosened) ===
        best_atk, best_score = self._best_attack(battle)
        if best_atk and best_score > self.KO_SCORE_THRESH and opponent.current_hp_fraction < self.KO_HP_THRESH:
            return self.create_order(best_atk)

        # === 3. Phaze (loosened from +2 to +1) ===
        if opp_boost_sum >= self.PHAZE_BOOST_THRESH:
            for mid in PHAZE_IDS:
                if mid in moves_by_id:
                    return self.create_order(moves_by_id[mid])

        # === 4. Taunt — added vs original Strategic ===
        if ("taunt" in moves_by_id and matchup > -0.5
                and opponent.current_hp_fraction > 0.5
                and opp_boost_sum == 0):
            # Preempt future setups
            return self.create_order(moves_by_id["taunt"])

        # === 5. Utility (widened window) ===
        use_utility = best_score < self.UTILITY_WINDOW

        if use_utility and opponent.status is None and matchup > -1.0:
            # Sleep
            for mid in SLEEP_IDS:
                if mid in moves_by_id:
                    return self.create_order(moves_by_id[mid])
            # Burn physical attackers
            if matchup > 0 and opponent.base_stats.get("atk", 0) > opponent.base_stats.get("spa", 0):
                for mid in BURN_IDS:
                    if mid in moves_by_id:
                        return self.create_order(moves_by_id[mid])
            # Para fast threats
            if active.current_hp_fraction > 0.6 and opponent.base_stats.get("spe", 0) > active.base_stats.get("spe", 0):
                for mid in PARA_IDS:
                    if mid in moves_by_id:
                        return self.create_order(moves_by_id[mid])

        # === 6. Recovery (unchanged) ===
        from poke_env.battle.move import Move  # avoid hard import at top
        if active.current_hp_fraction < 0.35 and matchup > 0:
            RECOVERY_IDS = {"recover", "softboiled", "roost", "moonlight", "morningsun",
                            "synthesis", "shoreup", "milkdrink", "slackoff", "strengthsap"}
            for mid in RECOVERY_IDS:
                if mid in moves_by_id:
                    return self.create_order(moves_by_id[mid])

        # === 7. Pivot (loosened from -0.5 to -0.2) ===
        if battle.available_switches and matchup < self.PIVOT_MATCHUP_THRESH:
            for m in battle.available_moves:
                if m.id in PIVOT_IDS:
                    return self.create_order(m)

        # === 8. Endgame (earlier — engages at 3 mons each side, was 2) ===
        if n_remaining <= self.ENDGAME_N and n_opp_remaining <= self.ENDGAME_N:
            if best_atk is not None:
                return self.create_order(best_atk)

        # === 9. Best attack (still no self-setup) ===
        if best_atk is not None:
            return self.create_order(best_atk)

        # === 10. Switch fallback ===
        if battle.available_switches:
            opp_types = [t for t in (opponent.types or []) if t is not None]
            if opp_types:
                best_sw = None
                for mon in battle.available_switches:
                    if mon is None:
                        continue
                    worst = 0.0
                    try:
                        for ot in opp_types:
                            mult = 1.0
                            for mt in (mon.types or []):
                                if mt is not None:
                                    mult *= ot.damage_multiplier(mt)
                            worst = max(worst, mult)
                    except Exception:
                        worst = 1.0
                    if best_sw is None or worst < best_sw[0]:
                        best_sw = (worst, mon)
                if best_sw is not None:
                    return self.create_order(best_sw[1])

        return self.choose_random_move(battle)


# ====================================================================
# GreedySEv2 — SimpleHeuristics base + greedy super-effective preference
# ====================================================================

class GreedySEv2(SimpleHeuristicsPlayer):
    """SimpleHeuristics base + 'always prefer super-effective' move preference.

    Style: greedy attacker. Picks the most super-effective attacking move
    available, breaking ties by effective power. Uses SH's strong default
    logic for switching, hazards, and fallback decisions.

    Differs from policy_rulebots.GreedySEPlayer (raw Player base, ~790 Elo):
    inherits SH's full decision tree instead of falling back to choose_random_move
    when no attacks fit the greedy-SE template. Preserves the "always SE first"
    identity in the move scorer.

    Differs from SH itself in move selection: SH picks max(bp * stab * ratio *
    accuracy * type_mult) — a smooth damage maximizer. GreedySEv2 sorts
    lexicographically by (type_mult, effective_power) — strictly prefers a 4x
    move over a 1x move even if BP is lower. This preserves the distinctive
    "greedy SE" identity.

    Does NOT setup itself (greedy attackers don't setup) — overrides SH's
    setup branch by checking matchup first and trying to attack.
    """

    def _best_se_attack(self, battle, active, opponent):
        """Greedy super-effective move selection.

        Sorts attacking moves by (type_effectiveness, effective_power) lexicographic,
        matching policy_rulebots.GreedySEPlayer's "always pick SE first" semantic.
        """
        legal = [m for m in battle.available_moves if m is not None and m.base_power > 0]
        if not legal:
            return None
        try:
            physical_ratio = self._stat_estimation(active, "atk") / self._stat_estimation(opponent, "def")
            special_ratio = self._stat_estimation(active, "spa") / self._stat_estimation(opponent, "spd")
        except Exception:
            physical_ratio = special_ratio = 1.0

        scored = []
        for m in legal:
            try:
                type_mult = opponent.damage_multiplier(m)
            except Exception:
                type_mult = 1.0
            stab = 1.5 if m.type in active.types else 1.0
            cat_ratio = physical_ratio if m.category == MoveCategory.PHYSICAL else special_ratio
            effective_power = m.base_power * stab * cat_ratio * m.accuracy
            scored.append((type_mult, effective_power, m))

        # Sort by (type_mult, effective_power) descending — original GreedySE semantic
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return scored[0][2]

    def choose_move(self, battle: AbstractBattle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if active is None or opponent is None:
            return self.choose_random_move(battle)

        # Check: should we switch out per SH logic?
        # Same gate SH uses: if we have moves AND (not should_switch OR no switches available)
        if battle.available_moves and (
            not self._should_switch_out(battle) or not battle.available_switches
        ):
            # Try greedy-SE attack first (overrides SH's smooth damage maximizer)
            best = self._best_se_attack(battle, active, opponent)
            if best is not None:
                return self.create_order(best)

        # Fall through to SH (handles switch-out, hazards, etc.)
        # Note: SH's setup branch could fire here, but only if active HP=100% AND
        # matchup>0 AND we passed the moves+!should_switch check above. In that
        # case we'd have returned _best_se_attack already (a damaging move exists),
        # so setup branch only fires when no damage move is available — fine.
        return super().choose_move(battle)


# ====================================================================
# HazardSensev2 — SimpleHeuristics base + aggressive hazard prioritization
# ====================================================================

HAZARD_LAY_IDS = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
HAZARD_REMOVAL_IDS = {"rapidspin", "defog", "courtchange", "tidyup", "mortalspin"}


class HazardSensev2(SimpleHeuristicsPlayer):
    """SimpleHeuristics base + aggressive hazard play.

    Style: hazard-first specialist. Prioritizes hazard placement and removal
    over damage when reasonable. Uses SH's strong default logic for everything
    else.

    Differs from policy_rulebots.HazardSensePlayer (raw Player base, ~735 Elo):
    inherits SH's full decision tree instead of choose_random_move fallback.

    Differs from SH itself:
    - SH lays SR only if n_opp_remaining >= 3. HazardSensev2 lays at >= 2
      (more aggressive — accepts marginal value when 1 KO already happened).
    - HazardSensev2 ALSO lays Spikes/Toxic Spikes/Sticky Web if available AND
      target hazard isn't already up (SH only handles SR).
    - HazardSensev2 always uses Rapid Spin / Defog if hazards on our side
      and we have >= 2 mons (more aggressive than SH's similar check).

    Does NOT setup itself.
    """

    def choose_move(self, battle: AbstractBattle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if active is None or opponent is None:
            return self.choose_random_move(battle)

        if not battle.available_moves:
            return super().choose_move(battle)

        # Count remaining mons
        try:
            n_remaining = sum(1 for m in battle.team.values() if not m.fainted)
            n_opp_remaining = 6 - sum(1 for m in battle.opponent_team.values() if m.fainted)
        except Exception:
            n_remaining = n_opp_remaining = 6

        moves_by_id = {m.id: m for m in battle.available_moves if m is not None}
        own_side = battle.side_conditions
        opp_side = battle.opponent_side_conditions

        # === Priority 1: Lay Stealth Rock (more aggressive than SH's n_opp >= 3) ===
        if n_opp_remaining >= 2 and "stealthrock" in moves_by_id:
            if SideCondition.STEALTH_ROCK not in opp_side:
                return self.create_order(moves_by_id["stealthrock"])

        # === Priority 2: Lay Spikes / Toxic Spikes / Sticky Web ===
        if n_opp_remaining >= 2:
            if "spikes" in moves_by_id and SideCondition.SPIKES not in opp_side:
                return self.create_order(moves_by_id["spikes"])
            if "toxicspikes" in moves_by_id and SideCondition.TOXIC_SPIKES not in opp_side:
                return self.create_order(moves_by_id["toxicspikes"])
            if "stickyweb" in moves_by_id and SideCondition.STICKY_WEB not in opp_side:
                return self.create_order(moves_by_id["stickyweb"])

        # === Priority 3: Remove hazards on our side (Rapid Spin / Defog) ===
        any_own_hazards = any(c in own_side for c in (
            SideCondition.STEALTH_ROCK, SideCondition.SPIKES,
            SideCondition.TOXIC_SPIKES, SideCondition.STICKY_WEB))
        if any_own_hazards and n_remaining >= 2:
            for mid in HAZARD_REMOVAL_IDS:
                if mid in moves_by_id:
                    return self.create_order(moves_by_id[mid])

        # === Fall through to SH default (handles attack/switch/setup) ===
        return super().choose_move(battle)


# ====================================================================
# SwitchAwareEscapev3 — SimpleHeuristics base + offensive pivot preference
# ====================================================================

class SwitchAwareEscapev3(SimpleHeuristicsPlayer):
    """SimpleHeuristics base + 'use pivot moves to escape bad matchups'.

    Style: pivot specialist. When matchup is unfavorable, uses an offensive
    pivot move (U-turn, Volt Switch, Flip Turn, Parting Shot, Teleport) to
    deal damage while switching out, instead of a raw switch.

    Differs from policy_rulebots.SwitchAwareEscapePlayer (raw Player, ~735 Elo)
    and SwitchAwareEscapeV2 (raw Player, ~776 Elo): inherits SH's full decision
    tree instead of choose_random_move fallback.

    Differs from SH itself:
    - SH's _should_switch_out triggers a normal switch via choose_switch.
    - SwitchAwareEscapev3 PREFERS to switch via a pivot MOVE instead (deals
      damage as it switches), if one is available.
    - Lowers the switch trigger from SH's matchup < -2 to matchup < -0.5
      (more aggressive escape — uses the pivot's damage to make it worth).

    Does NOT setup itself.
    """

    PIVOT_MATCHUP_THRESH = -0.5

    def choose_move(self, battle: AbstractBattle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if active is None or opponent is None:
            return self.choose_random_move(battle)

        # Find available pivot move (if any)
        pivot_move = None
        if battle.available_moves:
            for m in battle.available_moves:
                if m is not None and m.id in PIVOT_IDS:
                    pivot_move = m
                    break

        if pivot_move is not None and battle.available_switches:
            # Aggressive pivot: use pivot when matchup is bad (looser than SH's switch threshold)
            try:
                matchup = self._estimate_matchup(active, opponent)
            except Exception:
                matchup = 0.0

            # Trigger 1: SH would switch out
            should_pivot = False
            try:
                if self._should_switch_out(battle):
                    should_pivot = True
            except Exception:
                pass

            # Trigger 2: mild matchup disadvantage (use pivot's damage)
            if not should_pivot and matchup < self.PIVOT_MATCHUP_THRESH:
                should_pivot = True

            if should_pivot:
                return self.create_order(pivot_move)

        # Fall through to SH default (handles attack/switch/setup/hazards)
        return super().choose_move(battle)


# ====================================================================
# SwitchAwareEscapeV2 — original pivot trigger + stat-disadvantage trigger
# ====================================================================

def _is_stat_disadvantage(active, opponent, attacking_move):
    """Return True if we're outsped AND opp's likely strongest offensive stat
    exceeds our matching defensive stat by a meaningful margin.

    Heuristic threshold: opp_offense - our_defense > 20 base points.
    """
    if active is None or opponent is None:
        return False
    try:
        active_spe = active.base_stats.get("spe", 0) or 0
        opp_spe = opponent.base_stats.get("spe", 0) or 0
        outsped = opp_spe > active_spe
        if not outsped:
            return False
        # Opp's best offensive stat (whichever is higher)
        opp_atk = opponent.base_stats.get("atk", 0) or 0
        opp_spa = opponent.base_stats.get("spa", 0) or 0
        opp_offense = max(opp_atk, opp_spa)
        # Our defensive stat — match to attacker side if known,
        # otherwise take the lower of the two (worst case).
        our_def = active.base_stats.get("def", 0) or 0
        our_spd = active.base_stats.get("spd", 0) or 0
        # If opp clearly leans physical or special, match that
        if opp_atk > opp_spa + 20:
            our_defense = our_def
        elif opp_spa > opp_atk + 20:
            our_defense = our_spd
        else:
            our_defense = min(our_def, our_spd)
        return (opp_offense - our_defense) > 20
    except Exception:
        return False


class SwitchAwareEscapeV2(Player):
    """Enhanced SwitchAwareEscape — pivots on type-eff < 1.0 OR stat disadvantage.

    Differs from policy_rulebots.SwitchAwareEscapePlayer only in the pivot
    trigger condition. Still uses offensive pivot moves (U-turn / Volt Switch
    etc.) as its signature mechanic — the broader trigger just catches more
    bad-matchup scenarios.

    Original SwitchAwareEscapePlayer is intentionally left untouched so both
    flavors live in the training pool with different "voice".
    """

    def choose_move(self, battle):
        legal_moves = [m for m in battle.available_moves if m is not None]
        legal_switches = [p for p in battle.available_switches if p is not None]
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        # === Pivot trigger: type OR stat disadvantage ===
        pivot = _best_pivot_move(battle)
        if pivot is not None:
            best = _best_attacking_move(battle)
            should_pivot = False
            if best is None:
                should_pivot = True
            else:
                # Trigger 1 (original): type-resisted by opponent
                eff = _type_multiplier(best, (opponent.types if opponent else []))
                if eff < 1.0:
                    should_pivot = True
                # Trigger 2 (new): outsped + outclassed offensively
                elif _is_stat_disadvantage(active, opponent, best):
                    should_pivot = True
            if should_pivot:
                return self.create_order(pivot)

        # === If "locked" into one weak move, try to switch ===
        if len(legal_moves) == 1 and legal_switches:
            the_move = legal_moves[0]
            eff = _type_multiplier(the_move, (opponent.types if opponent else []))
            if _effective_power(the_move, active) <= 0 or eff < 1.0:
                sw = _best_switch_candidate(battle)
                if sw is not None:
                    return self.create_order(sw)

        # === Attack normally ===
        best = _best_attacking_move(battle)
        if best is not None:
            return self.create_order(best)

        # === If walled, switch to a resist ===
        sw = _best_switch_candidate(battle)
        if sw is not None:
            return self.create_order(sw)

        return self.choose_random_move(battle)
