#!/usr/bin/env python3
# battle_agent.py — Inference wrapper for PokeTransformer v8.
#
# BattleAgent is a poke-env Player subclass that:
#   1. Loads a v8 checkpoint (PokeTransformer)
#   2. Extracts features via features_v8.make_features()
#   3. Manages per-battle temporal history (summary buffer)
#   4. Picks the highest-scoring legal action
#
# Drop-in compatible with eval_bc_vs_bots.py and self-play infrastructure.

from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn.functional as F
import numpy as np

from poke_env.player import Player
from features import make_features
from model import PokeTransformer, PokeTransformerConfig


class BattleAgent(Player):
    """V8 PokeTransformer inference player for live battles."""

    def __init__(self, checkpoint_path: str = None, device: str = "cpu",
                 temperature: float = 0.0, fp16: bool = False,
                 turn_cap: int = 300, _cached_ckpt: dict = None, **kwargs):
        """V8 PokeTransformer inference player.

        For repeated instantiation (e.g. tournament/Elo evaluation), pass an
        already-loaded checkpoint dict via `_cached_ckpt` to bypass the disk
        read entirely. The `checkpoint_path` is then only used for logging.
        """
        super().__init__(**kwargs)
        self.device = torch.device(device)
        self.temperature = temperature
        self.fp16 = fp16 and self.device.type == "cuda"
        self.turn_cap = turn_cap
        self._turn_counts: Dict[str, int] = {}

        # Load checkpoint (or use pre-loaded cache)
        if _cached_ckpt is not None:
            ckpt = _cached_ckpt
        else:
            if checkpoint_path is None:
                raise ValueError("Either checkpoint_path or _cached_ckpt is required")
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg_dict = ckpt.get("model_config", {})
        self.cfg = PokeTransformerConfig.from_dict(cfg_dict)
        self.model = PokeTransformer(self.cfg).to(self.device)

        # Handle dim expansion for type effectiveness features (zero-init new columns)
        state = ckpt["model_state_dict"]
        _expand_targets = ["move_net.mlp.0.weight", "switch_mlp.0.weight"]
        for key in list(state.keys()):
            if any(key.endswith(t) for t in _expand_targets):
                old_w = state[key]
                parts = key.split(".")
                mod = self.model
                for p in parts[:-1]:
                    mod = getattr(mod, p) if not p.isdigit() else mod[int(p)]
                expected_in = mod.in_features
                if old_w.shape[1] < expected_in:
                    pad = expected_in - old_w.shape[1]
                    state[key] = torch.cat([old_w, torch.zeros(old_w.shape[0], pad, device=old_w.device)], dim=1)
                    import logging
                    logging.getLogger("pokemon_ai").warning(
                        f"Dim expansion: {key} padded {old_w.shape[1]} -> {expected_in} (+{pad} zero-init cols). "
                        f"If this is unexpected, the checkpoint may not match the current feature set."
                    )

        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        # Per-battle temporal history: battle_tag -> (B=1, T, D) tensor
        self._history: Dict[str, torch.Tensor] = {}

        print(f"[BattleAgent] Loaded {checkpoint_path} "
              f"({self.model.count_parameters():,} params, device={device})")

    def _get_history(self, btag: str) -> Optional[torch.Tensor]:
        return self._history.get(btag)

    def _build_turn_batch(self, feat: dict) -> dict:
        """Convert features_v8 output to the batch dict PokeTransformer.forward() expects.
        All tensors are (B=1, ...) on self.device."""
        dev = self.device

        def _poke_ids(p):
            ids = p["ids"]
            return [ids["species"], ids["item"], ids["ability"]]

        def _poke_banks(p):
            b = p["banks"]
            return [b["hp_pct"], b["level"], b["weight"], b["height"],
                    b["stat_hp"], b["stat_atk"], b["stat_def"],
                    b["stat_spa"], b["stat_spd"], b["stat_spe"]]

        def _poke_move_ids(p):
            ids = p["ids"]
            return [ids["move0"], ids["move1"], ids["move2"], ids["move3"]]

        def _poke_move_cont(p):
            from features import extract_move_cont
            return extract_move_cont(p["continuous"])

        # Build (1, 6, ...) tensors for pokemon
        our_ids = torch.tensor([[_poke_ids(p) for p in feat["our_pokemon"]]], dtype=torch.long, device=dev)
        our_banks = torch.tensor([[_poke_banks(p) for p in feat["our_pokemon"]]], dtype=torch.long, device=dev)
        our_cont = torch.tensor([[p["continuous"] for p in feat["our_pokemon"]]], dtype=torch.float32, device=dev)
        our_move_ids = torch.tensor([[_poke_move_ids(p) for p in feat["our_pokemon"]]], dtype=torch.long, device=dev)
        our_mcont = torch.tensor([[_poke_move_cont(p) for p in feat["our_pokemon"]]], dtype=torch.float32, device=dev)

        opp_ids = torch.tensor([[_poke_ids(p) for p in feat["opp_pokemon"]]], dtype=torch.long, device=dev)
        opp_banks = torch.tensor([[_poke_banks(p) for p in feat["opp_pokemon"]]], dtype=torch.long, device=dev)
        opp_cont = torch.tensor([[p["continuous"] for p in feat["opp_pokemon"]]], dtype=torch.float32, device=dev)
        opp_move_ids = torch.tensor([[_poke_move_ids(p) for p in feat["opp_pokemon"]]], dtype=torch.long, device=dev)
        opp_mcont = torch.tensor([[_poke_move_cont(p) for p in feat["opp_pokemon"]]], dtype=torch.float32, device=dev)

        # Field
        fb = feat["field"]["banks"]
        field_banks = {
            "turn": torch.tensor([fb["turn"]], dtype=torch.long, device=dev),
            "weather_dur": torch.tensor([fb["weather_dur"]], dtype=torch.long, device=dev),
            "terrain_dur": torch.tensor([fb["terrain_dur"]], dtype=torch.long, device=dev),
            "tr_dur": torch.tensor([fb["tr_dur"]], dtype=torch.long, device=dev),
        }
        field_cont = torch.tensor([feat["field"]["continuous"]], dtype=torch.float32, device=dev)

        # Transition
        ti = feat["transition"]["ids"]
        transition_ids = {
            "our_action": torch.tensor([ti["our_action"]], dtype=torch.long, device=dev),
            "opp_action": torch.tensor([ti["opp_action"]], dtype=torch.long, device=dev),
        }
        transition_cont = torch.tensor([feat["transition"]["continuous"]], dtype=torch.float32, device=dev)

        # Active moves
        active_moves = feat["active_moves"]
        move_ids_list = []
        move_banks_bp, move_banks_acc, move_banks_pp, move_banks_prio = [], [], [], []
        move_cont_list = []
        for i in range(4):
            m = active_moves[i]
            if m is None:
                move_ids_list.append(0)
                move_banks_bp.append(0); move_banks_acc.append(0)
                move_banks_pp.append(0); move_banks_prio.append(6)
                from features import MOVE_SLOT_CONT_DIM
                move_cont_list.append([0.0] * MOVE_SLOT_CONT_DIM)
            else:
                move_ids_list.append(m["move_id"])
                move_banks_bp.append(m["bp_int"]); move_banks_acc.append(m["acc_int"])
                move_banks_pp.append(m["pp_int"]); move_banks_prio.append(m["priority_int"])
                move_cont_list.append(m["continuous"])

        active_move_ids = torch.tensor([move_ids_list], dtype=torch.long, device=dev)
        active_move_banks = {
            "bp": torch.tensor([move_banks_bp], dtype=torch.long, device=dev),
            "acc": torch.tensor([move_banks_acc], dtype=torch.long, device=dev),
            "pp": torch.tensor([move_banks_pp], dtype=torch.long, device=dev),
            "prio": torch.tensor([move_banks_prio], dtype=torch.long, device=dev),
        }
        active_move_cont = torch.tensor([move_cont_list], dtype=torch.float32, device=dev)

        # Switches
        switch_slots = feat["switch_slots"]
        sw_ids = []
        sw_cont = []
        for i in range(5):
            s = switch_slots[i]
            if s is None:
                sw_ids.append(0)
                from features import SWITCH_SLOT_CONT_DIM
                sw_cont.append([0.0] * SWITCH_SLOT_CONT_DIM)
            else:
                sw_ids.append(s["species_id"])
                sw_cont.append(s["continuous"])

        switch_ids = torch.tensor([sw_ids], dtype=torch.long, device=dev)
        switch_cont = torch.tensor([sw_cont], dtype=torch.float32, device=dev)

        # Legal mask
        legal_mask = torch.tensor([feat["legal_mask"].tolist()], dtype=torch.float32, device=dev)

        return {
            "our_pokemon_ids": our_ids,
            "our_pokemon_banks": our_banks,
            "our_pokemon_cont": our_cont,
            "our_pokemon_move_ids": our_move_ids,
            "our_pokemon_move_cont": our_mcont,
            "opp_pokemon_ids": opp_ids,
            "opp_pokemon_banks": opp_banks,
            "opp_pokemon_cont": opp_cont,
            "opp_pokemon_move_ids": opp_move_ids,
            "opp_pokemon_move_cont": opp_mcont,
            "field_banks": field_banks,
            "field_cont": field_cont,
            "transition_ids": transition_ids,
            "transition_cont": transition_cont,
            "active_move_ids": active_move_ids,
            "active_move_banks": active_move_banks,
            "active_move_cont": active_move_cont,
            "switch_ids": switch_ids,
            "switch_cont": switch_cont,
            "legal_mask": legal_mask,
        }

    def choose_move(self, battle):
        btag = battle.battle_tag

        # Turn cap — forfeit if battle runs too long
        self._turn_counts[btag] = self._turn_counts.get(btag, 0) + 1
        if self._turn_counts[btag] >= self.turn_cap:
            try:
                self.forfeit_battle(battle)
            except Exception:
                pass
            return self.choose_random_move(battle)

        # Extract features
        feat = make_features(battle)
        batch = self._build_turn_batch(feat)
        history = self._get_history(btag)

        # Forward pass
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.fp16):
            out = self.model(batch, history=history)

        # Update history (always float32 for stable accumulation)
        summary = out["summary"].float().unsqueeze(1)  # (1, 1, D)
        if history is None:
            self._history[btag] = summary
        else:
            self._history[btag] = torch.cat([history, summary], dim=1)
            # Trim to temporal context limit
            if self._history[btag].shape[1] > self.cfg.temporal_context:
                self._history[btag] = self._history[btag][:, -self.cfg.temporal_context:]

        # Select action
        logits = out["action_logits"][0]  # (9,)
        if self.temperature > 0:
            probs = F.softmax(logits / self.temperature, dim=-1)
            action_idx = torch.multinomial(probs, 1).item()
        else:
            action_idx = logits.argmax().item()

        # Map to poke-env order
        return self._action_to_order(battle, action_idx)

    def _action_to_order(self, battle, action_idx: int):
        """Convert action index 0-8 to a poke-env BattleOrder."""
        if action_idx < 4:
            moves = list(battle.available_moves or [])
            if action_idx < len(moves):
                return self.create_order(moves[action_idx])
        else:
            switches = list(battle.available_switches or [])
            sw_idx = action_idx - 4
            if sw_idx < len(switches):
                return self.create_order(switches[sw_idx])

        # Fallback: pick first legal action
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])
        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        """Clean up per-battle state."""
        btag = battle.battle_tag
        self._history.pop(btag, None)
        self._turn_counts.pop(btag, None)
        super()._battle_finished_callback(battle)

    def reset_battles(self):
        """Clear all battle state."""
        self._history.clear()
        super().reset_battles()
