#!/usr/bin/env python3
# battle_agent_transformer.py — Inference wrapper for TransformerBattlePolicy.
#
# Mirror of `battle_agent.py` but loads the new-arch checkpoints
# (REWRITE_DESIGN.md §6b.2 — "keep legacy alive, new arch lives alongside").
# The legacy `BattleAgent` is unchanged; this is an additive sibling.
#
# Differences from BattleAgent:
#   - Builds `TransformerBattlePolicy(TransformerConfig, move_flag_lookup)`
#     from `data/lookup/move_flags_v1.pt`.
#   - No dim-expansion path (transformer ckpts can't predate the spec).
#   - Otherwise: same `Player` interface, same `make_features` /
#     `build_turn_batch` / `action_to_order` flow, same per-battle history
#     buffer, same temperature-or-argmax action selection.

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from poke_env.player import Player
from features import make_features
from model_transformer import (
    TransformerBattlePolicy, TransformerConfig, load_move_flag_lookup,
)


DEFAULT_LOOKUP_PATH = Path("data/lookup/move_flags_v1.pt")


class BattleAgentTransformer(Player):
    """TransformerBattlePolicy inference player for live battles.

    Drop-in replacement for `BattleAgent` when evaluating new-arch checkpoints.
    Use `is_transformer_checkpoint(path)` (below) for arch dispatch in
    higher-level eval scripts.
    """

    def __init__(self, checkpoint_path: str = None, device: str = "cpu",
                 temperature: float = 0.0, fp16: bool = False,
                 turn_cap: int = 300, _cached_ckpt: dict = None,
                 lookup_path: Path = DEFAULT_LOOKUP_PATH, **kwargs):
        """TransformerBattlePolicy inference player.

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

        # Load checkpoint (or use pre-loaded cache).
        if _cached_ckpt is not None:
            ckpt = _cached_ckpt
        else:
            if checkpoint_path is None:
                raise ValueError("Either checkpoint_path or _cached_ckpt is required")
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Sanity check: arch must be transformer or inferable as such.
        arch = ckpt.get("arch")
        if arch is None:
            # Legacy ckpts predate the `arch` field; fall back to state-dict prefix sniffing.
            state_keys = ckpt.get("model_state_dict", {}).keys()
            arch = "transformer" if any(
                k.startswith(("tokenizer.", "spatial.", "temporal.", "action_head.",
                              "value_head.", "switch_encoder.", "summary_to_temporal."))
                for k in state_keys
            ) else "mlp"
        if arch != "transformer":
            raise ValueError(
                f"BattleAgentTransformer cannot load arch={arch!r}. "
                f"Use BattleAgent for legacy MLP checkpoints."
            )

        cfg_dict = ckpt.get("model_config", {})
        self.cfg = TransformerConfig.from_dict(cfg_dict)

        lookup = load_move_flag_lookup(Path(lookup_path), expected_n_moves=self.cfg.n_moves)
        self.model = TransformerBattlePolicy(self.cfg, move_flag_lookup=lookup).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()

        # Per-battle temporal history: battle_tag -> (B=1, T, d_temporal) tensor.
        self._history: Dict[str, torch.Tensor] = {}

        print(f"[BattleAgentTransformer] Loaded {checkpoint_path} "
              f"({self.model.count_parameters():,} params, device={device})")

    def _get_history(self, btag: str) -> Optional[torch.Tensor]:
        return self._history.get(btag)

    def _build_turn_batch(self, feat: dict) -> dict:
        """Convert make_features() output to the batch dict on self.device."""
        from features import build_turn_batch
        return build_turn_batch(feat, device=self.device)

    def choose_move(self, battle):
        btag = battle.battle_tag

        # Turn cap — forfeit if battle runs too long.
        self._turn_counts[btag] = self._turn_counts.get(btag, 0) + 1
        if self._turn_counts[btag] >= self.turn_cap:
            try:
                self.forfeit_battle(battle)
            except Exception:
                pass
            return self.choose_random_move(battle)

        # Extract features.
        feat = make_features(battle)
        batch = self._build_turn_batch(feat)
        history = self._get_history(btag)

        # Forward pass.
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.fp16):
            out = self.model(batch, history=history)

        # Update history (always float32 for stable accumulation).
        summary = out["summary"].float().unsqueeze(1)  # (1, 1, d_temporal)
        if history is None:
            self._history[btag] = summary
        else:
            # Pre-slice before cat to avoid temporary OOM on long battles
            # (preserves the legacy fix at battle_agent.py:117).
            ctx = self.cfg.temporal_context
            if history.shape[1] >= ctx:
                history = history[:, -(ctx - 1):]
            self._history[btag] = torch.cat([history, summary], dim=1)

        # Select action.
        logits = out["action_logits"][0]  # (n_actions,)
        if self.temperature > 0:
            probs = F.softmax(logits / self.temperature, dim=-1)
            action_idx = torch.multinomial(probs, 1).item()
        else:
            action_idx = logits.argmax().item()

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


def is_transformer_checkpoint(ckpt_or_path) -> bool:
    """Inspect a checkpoint (loaded dict or file path) and return True iff
    it's a transformer-arch checkpoint. Used by eval scripts to dispatch
    between BattleAgent and BattleAgentTransformer.

    Reads `ckpt["arch"]` if present (Session 48 onward) or falls back to
    state-dict key prefix inference for legacy ckpts.
    """
    if isinstance(ckpt_or_path, (str, Path)):
        ckpt = torch.load(ckpt_or_path, map_location="cpu", weights_only=False)
    else:
        ckpt = ckpt_or_path
    arch = ckpt.get("arch")
    if arch is not None:
        return arch == "transformer"
    keys = ckpt.get("model_state_dict", {}).keys()
    return any(
        k.startswith(("tokenizer.", "spatial.", "temporal.", "action_head.",
                      "value_head.", "switch_encoder.", "summary_to_temporal."))
        for k in keys
    )
