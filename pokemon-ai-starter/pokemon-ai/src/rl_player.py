# rl_player.py — RL player classes for self-play PPO training.
#
# Extracted from rl_train_v9.py during Session 34 refactor.
# V9RLPlayer: async choose_move that submits to InferenceBatcher
# SelfPlayOpponent: opponent with temperature randomization

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from poke_env.player import Player

from features import make_features, MOVE_SLOT_CONT_DIM, SWITCH_SLOT_CONT_DIM
from battle_agent import BattleAgent
from battle_agent_transformer import BattleAgentTransformer, is_transformer_checkpoint
from rewards import RewardShaper
from ppo import Trajectory
from inference_batcher import InferenceBatcher


class V9RLPlayer(Player):
    """PPO player with async batched inference."""

    def __init__(self, batcher: InferenceBatcher, device: torch.device,
                 reward_shaper_cfg: Optional[dict] = None,
                 temperature: float = 1.0, turn_cap: int = 300, **kwargs):
        super().__init__(**kwargs)
        self.batcher = batcher
        self.device = device
        self._rs_cfg = reward_shaper_cfg or {"ko_coef": 0.05, "hp_coef": 0.02, "clip_abs": 2.0}
        self.temperature = temperature
        self.turn_cap = turn_cap
        self._history: Dict[str, torch.Tensor] = {}
        self._trajectories: Dict[str, Trajectory] = {}
        self._reward_shapers: Dict[str, RewardShaper] = {}
        self.completed_trajectories: List[Trajectory] = []
        self._tainted: set = set()  # battle tags with NaN — discard trajectory
        # Battles where WE called forfeit_battle ourselves (turn-cap path).
        # Those are legitimate terminations even though both teams are still
        # mostly alive — the -1 terminal IS real and the trajectory IS real
        # play, so we keep them out of the abrupt-disconnect filter below.
        self._self_forfeited: set = set()
        # Forfeit-finish counters: incremented when a battle ends with a |win|/|lose|
        # frame but neither team is fully fainted (= opponent or our subprocess
        # crashed and battle_server flipped the result on WS drop). rl_collection
        # subtracts these from the poke-env W/L totals so PFSP weights and the
        # iter summary reflect real games only. Raw spurious +1 terminal rewards
        # are also dropped from completed_trajectories.
        self.n_forfeit_wins: int = 0
        self.n_forfeit_losses: int = 0

    def _get_shaper(self, btag):
        if btag not in self._reward_shapers:
            self._reward_shapers[btag] = RewardShaper(**self._rs_cfg)
        return self._reward_shapers[btag]

    def _get_traj(self, btag):
        if btag not in self._trajectories:
            self._trajectories[btag] = Trajectory()
        return self._trajectories[btag]

    def _build_turn_batch(self, feat: dict) -> dict:
        """Convert make_features() output to PokeTransformer batch dict on self.device."""
        from features import build_turn_batch
        return build_turn_batch(feat, device=self.device, training=True)

    def _to_cpu(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.cpu()
            elif isinstance(v, dict):
                out[k] = {kk: vv.cpu() if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
            else:
                out[k] = v
        return out

    async def choose_move(self, battle):
        """Async choose_move — submits to batcher, awaits result."""
        btag = battle.battle_tag
        traj = self._get_traj(btag)
        shaper = self._get_shaper(btag)

        # Turn cap
        if len(traj) >= self.turn_cap:
            print(f"  [TURN CAP] {btag} hit {self.turn_cap} turns, forfeiting", flush=True)
            # Mark as self-initiated so the abrupt-disconnect filter doesn't
            # drop the resulting trajectory as a forfeit-loss. We DID play
            # turn_cap real turns; the -1 terminal is an honest signal.
            self._self_forfeited.add(btag)
            try:
                self.forfeit_battle(battle)
            except Exception:
                pass
            return self.choose_random_move(battle)

        # Feature extraction (CPU, ~1ms)
        feat = make_features(battle)

        # Dense reward for previous step (with immune detection from transition)
        if len(traj.rewards) > 0:
            # our_eff[0] = immune flag, at index 9 in transition continuous
            # (6 action_kind + 3 moved_first = 9 offset)
            our_move_immune = feat["transition"]["continuous"][9] > 0.5
            traj.rewards[-1] += shaper.step(battle, our_move_immune=our_move_immune)
        batch = self._build_turn_batch(feat)
        history = self._history.get(btag)
        h_len = history.shape[1] if history is not None else 0

        # Submit to batcher and await (yields to event loop!)
        try:
            result = await self.batcher.submit(batch, history, h_len)
        except Exception as e:
            print(f"  [ERROR] Batcher failed for {btag}: {e}", flush=True)
            self._tainted.add(btag)
            return self.choose_random_move(battle)

        # Update temporal history (always float32)
        summary = result["summary"].unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        if history is None:
            self._history[btag] = summary
        else:
            # Pre-slice before cat to avoid temporary OOM on long battles
            if history.shape[1] >= 200:
                history = history[:, -199:]
            self._history[btag] = torch.cat([history, summary], dim=1)

        # Sample action with temperature
        logits = result["action_logits"]  # (9,)
        if self.temperature != 1.0:
            scaled = logits / self.temperature
        else:
            scaled = logits
        # Guard against NaN/inf from FP16 overflow — taint entire battle
        if torch.isnan(scaled).any() or torch.isinf(scaled).any():
            self._tainted.add(btag)
            return self.choose_random_move(battle)
        probs = F.softmax(scaled, dim=-1)
        if torch.isnan(probs).any() or (probs < 0).any():
            self._tainted.add(btag)
            return self.choose_random_move(battle)
        action_idx = torch.multinomial(probs, 1).item()

        # Store UNSCALED log_prob
        log_prob = F.log_softmax(logits, dim=-1)[action_idx].item()
        value = result["value"].item()

        # Store trajectory (CPU)
        traj.feat_batches.append(self._to_cpu(batch))
        traj.actions.append(action_idx)
        traj.log_probs.append(log_prob)
        traj.values.append(value)
        traj.rewards.append(0.0)
        traj.dones.append(False)
        traj.action_masks.append(feat["legal_mask"].copy())

        return self._action_to_order(battle, action_idx)

    def _action_to_order(self, battle, idx):
        from features import action_to_order
        order = action_to_order(self, battle, idx)
        return order if order is not None else self.choose_random_move(battle)

    def _finish_looks_real(self, battle) -> bool:
        """True iff this battle ended via a natural OU termination.

        Three outcomes count as real:
          1. We KO'd all of opponent's mons (opp_fainted >= team_size).
          2. Opponent KO'd all of ours (my_fainted >= team_size).
          3. We called `forfeit_battle` ourselves (turn_cap path) — the
             trajectory is real play and the -1 terminal is honest.

        Anything else is suspicious — when a subprocess opponent's WS
        drops mid-battle, our local battle_server emits `|win|<our_username>`
        abruptly and poke-env sets `battle.won = True` even though most of
        their team is alive. Without this filter, those finishes contribute
        (a) 1-3 turn trajectories with spurious ±1 terminal rewards to PPO
        and (b) inflated W/L counts to PFSP weight updates.

        team_size resolution: prefer `battle.max_team_size` (format-correct,
        e.g. 6 for OU). If that's None, fall back to the larger of the two
        team dicts — handles both "we brought 4 mons" and turn-0 disconnects
        where opponent_team is empty. Final fallback to 6 (OU default).

        Errs on the side of trusting the finish: any introspection error
        returns True. Better to keep one rare miscount than discard a real
        game.
        """
        try:
            if battle.battle_tag in self._self_forfeited:
                return True
            opp_team = battle.opponent_team or {}
            my_team = battle.team or {}
            team_size = battle.max_team_size
            if not team_size:
                # Format didn't set max_team_size — infer from the actual team
                # dicts. `, 1` floor ensures a turn-0 disconnect with empty
                # team dicts isn't trivially "real" (0 >= 0).
                team_size = max(len(my_team), len(opp_team), 1)
            opp_fainted = sum(1 for m in opp_team.values() if m and m.fainted)
            my_fainted = sum(1 for m in my_team.values() if m and m.fainted)
            return opp_fainted >= team_size or my_fainted >= team_size
        except Exception:
            return True

    def _battle_finished_callback(self, battle):
        btag = battle.battle_tag
        traj = self._trajectories.get(btag)
        is_tainted = btag in self._tainted
        is_real = self._finish_looks_real(battle)

        if is_tainted:
            if traj and len(traj) > 0:
                print(f"  [TAINTED] Discarding {btag} ({len(traj)} turns) — NaN detected", flush=True)
        elif not is_real:
            # WS drop on one side. Track for W/L correction in rl_collection;
            # drop the trajectory (whether empty or partial) so the spurious
            # ±1 terminal doesn't reach PPO.
            if battle.won:
                self.n_forfeit_wins += 1
            elif battle.lost:
                self.n_forfeit_losses += 1
            try:
                opp_fainted = sum(1 for m in (battle.opponent_team or {}).values() if m and m.fainted)
                my_fainted = sum(1 for m in (battle.team or {}).values() if m and m.fainted)
            except Exception:
                opp_fainted = my_fainted = -1
            traj_turns = len(traj) if traj else 0
            print(f"  [FORFEIT] {btag} ({traj_turns} turns, won={battle.won}, "
                  f"opp_fainted={opp_fainted}, my_fainted={my_fainted}) — "
                  f"likely WS drop, dropping trajectory + W/L credit", flush=True)
        elif traj and len(traj) > 0:
            shaper = self._get_shaper(btag)
            # Terminal step — no feature extraction available, immune irrelevant
            # (terminal ±1.0 dominates)
            traj.rewards[-1] += shaper.step(battle, our_move_immune=False)
            if battle.won:
                traj.rewards[-1] += 1.0
            elif battle.lost:
                traj.rewards[-1] -= 1.0
            else:
                # Tied (battle.won=False AND battle.lost=False). Showdown emits
                # `|tie` on simultaneous KO, Endless Battle Clause, OR — most
                # common in our setup — both agents hitting turn_cap=300 on the
                # same turn (both self-forfeit, Showdown can't pick a winner).
                # S58 finding: dev pod 200g learned stall play because the
                # previous tie=0 reward made it RATIONAL to convert losses
                # into ties when winning was uncertain (EV(stall)=0 > EV(lose)=-1).
                # tie=-0.5 would still leave a stall incentive in most P(loss)
                # scenarios; tie=-1.0 (same as loss) forces the model to commit
                # to a win attempt. The competitive cost of "tie a stall match"
                # is genuinely equivalent to a loss in competitive Pokemon
                # (the goal is to win, not avoid losing).
                traj.rewards[-1] -= 1.0
            traj.dones[-1] = True
            self.completed_trajectories.append(traj)

        self._tainted.discard(btag)
        self._self_forfeited.discard(btag)
        self._trajectories.pop(btag, None)
        self._history.pop(btag, None)
        self._reward_shapers.pop(btag, None)
        super()._battle_finished_callback(battle)

    def reset_battles(self):
        self._history.clear()
        self._trajectories.clear()
        self._reward_shapers.clear()
        self._tainted.clear()
        self._self_forfeited.clear()
        self.completed_trajectories.clear()
        self.n_forfeit_wins = 0
        self.n_forfeit_losses = 0
        super().reset_battles()


class SelfPlayOpponent(BattleAgent):
    """Legacy-arch opponent with temperature randomization for self-play diversity.

    Use `make_self_play_opponent()` factory below for arch-aware dispatch — it
    picks this class for legacy ckpts and `SelfPlayOpponentTransformer` for
    new-arch ckpts.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda",
                 temp_range: Tuple[float, float] = (1.0, 2.25), **kwargs):
        temp = random.uniform(*temp_range)
        # fp16=False for opponent to avoid any FP16 numerical issues
        super().__init__(checkpoint_path=checkpoint_path, device=device,
                         temperature=temp, fp16=False, **kwargs)
        self._temp_range = temp_range

    def _battle_finished_callback(self, battle):
        """Re-randomize temperature for next game."""
        super()._battle_finished_callback(battle)
        self.temperature = random.uniform(*self._temp_range)


class SelfPlayOpponentTransformer(BattleAgentTransformer):
    """New-arch opponent with temperature randomization (mirror of SelfPlayOpponent)."""

    def __init__(self, checkpoint_path: str, device: str = "cuda",
                 temp_range: Tuple[float, float] = (1.0, 2.25), **kwargs):
        temp = random.uniform(*temp_range)
        super().__init__(checkpoint_path=checkpoint_path, device=device,
                         temperature=temp, fp16=False, **kwargs)
        self._temp_range = temp_range

    def _battle_finished_callback(self, battle):
        super()._battle_finished_callback(battle)
        self.temperature = random.uniform(*self._temp_range)


def make_self_play_opponent(checkpoint_path: str, device: str = "cuda",
                            temp_range: Tuple[float, float] = (1.0, 2.25),
                            _cached_ckpt: Optional[dict] = None, **kwargs):
    """Arch-aware factory: picks SelfPlayOpponent (legacy) or SelfPlayOpponentTransformer
    (new arch) based on the checkpoint's state-dict keys.

    `_cached_ckpt` lets repeated callers (e.g. PFSP pool spawns) avoid re-reading
    the file from disk. If not provided, the file is opened once for arch detection
    and the dict is forwarded to the chosen class so it doesn't re-read either.
    """
    if _cached_ckpt is None:
        _cached_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cls = SelfPlayOpponentTransformer if is_transformer_checkpoint(_cached_ckpt) else SelfPlayOpponent
    return cls(checkpoint_path=checkpoint_path, device=device, temp_range=temp_range,
               _cached_ckpt=_cached_ckpt, **kwargs)
