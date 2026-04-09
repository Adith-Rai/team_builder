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
        """Convert make_features() output to PokeTransformer batch dict on self.device."""
        from features import build_turn_batch
        return build_turn_batch(feat, device=self.device)

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
        from features import action_to_order
        order = action_to_order(self, battle, action_idx)
        return order if order is not None else self.choose_random_move(battle)

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
