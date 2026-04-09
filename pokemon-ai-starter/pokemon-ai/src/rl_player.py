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

    def _get_shaper(self, btag):
        if btag not in self._reward_shapers:
            self._reward_shapers[btag] = RewardShaper(**self._rs_cfg)
        return self._reward_shapers[btag]

    def _get_traj(self, btag):
        if btag not in self._trajectories:
            self._trajectories[btag] = Trajectory()
        return self._trajectories[btag]

    def _build_turn_batch(self, feat: dict) -> dict:
        """Convert feature output to model batch dict on self.device."""
        dev = self.device

        def _pi(p):
            i = p["ids"]
            return [i["species"], i["item"], i["ability"]]
        def _pb(p):
            b = p["banks"]
            return [b["hp_pct"], b["level"], b["weight"], b["height"],
                    b["stat_hp"], b["stat_atk"], b["stat_def"],
                    b["stat_spa"], b["stat_spd"], b["stat_spe"]]
        def _pmi(p):
            i = p["ids"]
            return [i["move0"], i["move1"], i["move2"], i["move3"]]
        def _pmc(p):
            from features import extract_move_cont
            return extract_move_cont(p["continuous"])

        our, opp = feat["our_pokemon"], feat["opp_pokemon"]
        int_arrays = {
            "our_pokemon_ids": np.array([[_pi(p) for p in our]], dtype=np.int64),
            "our_pokemon_banks": np.array([[_pb(p) for p in our]], dtype=np.int64),
            "our_pokemon_move_ids": np.array([[_pmi(p) for p in our]], dtype=np.int64),
            "opp_pokemon_ids": np.array([[_pi(p) for p in opp]], dtype=np.int64),
            "opp_pokemon_banks": np.array([[_pb(p) for p in opp]], dtype=np.int64),
            "opp_pokemon_move_ids": np.array([[_pmi(p) for p in opp]], dtype=np.int64),
        }
        float_arrays = {
            "our_pokemon_cont": np.array([[p["continuous"] for p in our]], dtype=np.float32),
            "our_pokemon_move_cont": np.array([[_pmc(p) for p in our]], dtype=np.float32),
            "opp_pokemon_cont": np.array([[p["continuous"] for p in opp]], dtype=np.float32),
            "opp_pokemon_move_cont": np.array([[_pmc(p) for p in opp]], dtype=np.float32),
            "field_cont": np.array([feat["field"]["continuous"]], dtype=np.float32),
            "transition_cont": np.array([feat["transition"]["continuous"]], dtype=np.float32),
            "legal_mask": feat["legal_mask"].reshape(1, 9).astype(np.float32),
        }

        mids, mbp, mac, mpp, mpr, mco = [], [], [], [], [], []
        for m in feat["active_moves"]:
            if m is None:
                mids.append(0); mbp.append(0); mac.append(0); mpp.append(0); mpr.append(6)
                mco.append([0.0]*MOVE_SLOT_CONT_DIM)
            else:
                mids.append(m["move_id"]); mbp.append(m["bp_int"]); mac.append(m["acc_int"])
                mpp.append(m["pp_int"]); mpr.append(m["priority_int"]); mco.append(m["continuous"])
        int_arrays["active_move_ids"] = np.array([mids], dtype=np.int64)
        float_arrays["active_move_cont"] = np.array([mco], dtype=np.float32)

        sids, sco = [], []
        for s in feat["switch_slots"]:
            if s is None:
                sids.append(0); sco.append([0.0]*SWITCH_SLOT_CONT_DIM)
            else:
                sids.append(s["species_id"]); sco.append(s["continuous"])
        int_arrays["switch_ids"] = np.array([sids], dtype=np.int64)
        float_arrays["switch_cont"] = np.array([sco], dtype=np.float32)

        batch = {}
        for k, arr in int_arrays.items():
            batch[k] = torch.from_numpy(arr).to(dev, non_blocking=True)
        for k, arr in float_arrays.items():
            batch[k] = torch.from_numpy(arr).to(dev, non_blocking=True)

        fb = feat["field"]["banks"]
        batch["field_banks"] = {k: torch.tensor([fb[k]], dtype=torch.long, device=dev) for k in fb}
        ti = feat["transition"]["ids"]
        batch["transition_ids"] = {k: torch.tensor([ti[k]], dtype=torch.long, device=dev) for k in ti}
        batch["active_move_banks"] = {
            "bp": torch.tensor([mbp], dtype=torch.long, device=dev),
            "acc": torch.tensor([mac], dtype=torch.long, device=dev),
            "pp": torch.tensor([mpp], dtype=torch.long, device=dev),
            "prio": torch.tensor([mpr], dtype=torch.long, device=dev),
        }
        return batch

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
            self._history[btag] = torch.cat([history, summary], dim=1)
            if self._history[btag].shape[1] > 200:
                self._history[btag] = self._history[btag][:, -200:]

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
        if idx < 4:
            moves = list(battle.available_moves or [])
            if idx < len(moves):
                return self.create_order(moves[idx])
        else:
            sw = list(battle.available_switches or [])
            si = idx - 4
            if si < len(sw):
                return self.create_order(sw[si])
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])
        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        btag = battle.battle_tag
        traj = self._trajectories.get(btag)
        if traj and len(traj) > 0:
            if btag in self._tainted:
                print(f"  [TAINTED] Discarding {btag} ({len(traj)} turns) — NaN detected", flush=True)
                self._tainted.discard(btag)
            else:
                shaper = self._get_shaper(btag)
                # Terminal step — no feature extraction available, immune irrelevant
                # (terminal ±1.0 dominates)
                traj.rewards[-1] += shaper.step(battle, our_move_immune=False)
                if battle.won:
                    traj.rewards[-1] += 1.0
                elif battle.lost:
                    traj.rewards[-1] -= 1.0
                traj.dones[-1] = True
                self.completed_trajectories.append(traj)

        self._tainted.discard(btag)
        self._trajectories.pop(btag, None)
        self._history.pop(btag, None)
        self._reward_shapers.pop(btag, None)
        super()._battle_finished_callback(battle)

    def reset_battles(self):
        self._history.clear()
        self._trajectories.clear()
        self._reward_shapers.clear()
        self._tainted.clear()
        self.completed_trajectories.clear()
        super().reset_battles()


class SelfPlayOpponent(BattleAgent):
    """Opponent with temperature randomization for self-play diversity."""

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
