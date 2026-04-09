# src/policy_random.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from poke_env.player import Player


class RandomPolicy(Player):
    """
    Uniform random among legal moves/switches.
    Returns BattleOrder objects directly (compatible with observer).
    """
    def choose_move(self, battle):
        moves = [m for m in battle.available_moves if m is not None]
        switches = [p for p in battle.available_switches if p is not None]

        if not moves and not switches:
            return self.choose_random_move(battle)

        total = len(moves) + len(switches)
        idx = int(np.random.randint(total))
        if idx < len(moves):
            return self.create_order(moves[idx])
        else:
            return self.create_order(switches[idx - len(moves)])
