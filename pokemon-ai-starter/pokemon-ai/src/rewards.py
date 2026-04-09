# rewards.py
# Unified dense reward shaping for both online RL (PPO) and offline RL (IQL).
#
# Design principles (v5):
#   - Only reward OUTCOMES (KOs, HP, winning), never specific actions
#   - Terminal reward dominates (~1.0 vs ~0.3-0.4 total shaping)
#   - No tempo tax (penalizes strategic play like hazards/status/setup)
#   - No potential-based framing (was never truly potential-based anyway)
#   - Same formula as iql_train.py:compute_shaped_reward for consistency
#
# Formula:
#   reward[t] = ko_coef * KO_delta[t] + hp_coef * HP_delta[t]
#   reward[terminal] += terminal_coef * result
#
# Where:
#   KO_delta = (opp_mons_lost - our_mons_lost) since last step
#   HP_delta = change in (our_team_hp% - opp_team_hp%) since last step
#
# Coefficients chosen so terminal >> shaping:
#   ko_coef=0.05  →  6 KOs × 0.05 = 0.30 max shaped reward from KOs
#   hp_coef=0.02  →  ~0.10 total HP signal in a typical game
#   terminal=1.0  →  dominates at 1.0

from __future__ import annotations
from typing import Dict, Any


class RewardShaper:
    """Step-by-step dense reward shaping for online RL (PPO).

    Matches iql_train.py:compute_shaped_reward but operates on live
    battle objects instead of batched tensors.

    Args:
        ko_coef:       reward per net KO delta (positive = we KO'd, negative = we fainted)
        hp_coef:       reward per unit change in HP advantage (our_hp% - opp_hp%)
        terminal_coef: multiplier on terminal result (+1 win, -1 loss)
        clip_abs:      clamp per-step shaped reward (before terminal) into [-clip_abs, +clip_abs]
    """

    def __init__(
        self,
        ko_coef: float = 0.05,
        hp_coef: float = 0.02,
        terminal_coef: float = 1.0,
        clip_abs: float = 2.0,
        immune_penalty: float = 0.0,
        # Legacy params accepted for config compat but UNUSED:
        gamma: float = 0.99,
        alpha: float = 0.5,
        beta: float = 0.1,
        tempo_tax: float = 0.0,
        ko_bonus: float = 0.0,
        faint_penalty: float = 0.0,
        dense_events: bool = True,
    ):
        self.ko_coef = ko_coef
        self.hp_coef = hp_coef
        self.terminal_coef = terminal_coef
        self.clip_abs = clip_abs
        self.immune_penalty = immune_penalty
        # State
        self._prev_our_fainted = 0
        self._prev_opp_fainted = 0
        self._prev_hp_adv = None
        self._acc = 0.0

    @staticmethod
    def _team_hp_frac(team_dict: Dict[str, Any]) -> float:
        vals = []
        for p in (team_dict or {}).values():
            hp = getattr(p, "current_hp_fraction", None)
            if hp is not None:
                vals.append(max(0.0, min(1.0, float(hp))))
        # Assume unseen mons (not in team_dict) are at 100% HP
        from format_config import FORMAT_SINGLES
        ts = FORMAT_SINGLES.team_size
        return (sum(vals) + max(0, ts - len(team_dict or {}))) / float(ts)

    @staticmethod
    def _count_fainted(team_dict: Dict[str, Any]) -> int:
        c = 0
        for p in (team_dict or {}).values():
            if getattr(p, "fainted", False):
                c += 1
        return c

    def step(self, battle, our_move_immune: bool = False) -> float:
        """Compute shaped reward for the transition that just happened.

        Call ONCE per decision. Returns the dense reward for the previous step
        (the state change between last call and this call).

        Args:
            battle: poke-env Battle object
            our_move_immune: True if our previous move hit an immunity
        """
        our_team = getattr(battle, "team", {})
        opp_team = getattr(battle, "opponent_team", {})

        our_f = self._count_fainted(our_team)
        opp_f = self._count_fainted(opp_team)
        hp_adv = self._team_hp_frac(our_team) - self._team_hp_frac(opp_team)

        if self._prev_hp_adv is None:
            # First step: no delta to compute yet
            self._prev_our_fainted = our_f
            self._prev_opp_fainted = opp_f
            self._prev_hp_adv = hp_adv
            return 0.0

        # KO delta: positive when we KO, negative when we faint
        ko_delta = (opp_f - self._prev_opp_fainted) - (our_f - self._prev_our_fainted)
        hp_delta = hp_adv - self._prev_hp_adv

        r = self.ko_coef * ko_delta + self.hp_coef * hp_delta

        # Immune penalty: penalize when our move hit an immunity
        if our_move_immune and self.immune_penalty > 0:
            r -= self.immune_penalty

        r = max(-self.clip_abs, min(self.clip_abs, r))

        self._prev_our_fainted = our_f
        self._prev_opp_fainted = opp_f
        self._prev_hp_adv = hp_adv
        self._acc += r
        return r

    def end_episode(self) -> float:
        """Call at episode end. Returns accumulated shaping reward. Resets state."""
        total = self._acc
        self._prev_our_fainted = 0
        self._prev_opp_fainted = 0
        self._prev_hp_adv = None
        self._acc = 0.0
        return total


def terminal_sparse(win: bool) -> float:
    """Terminal reward: +1 for win, -1 for loss."""
    return 1.0 if win else -1.0
