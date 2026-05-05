# arch_compat.py — Arch-aware dispatch helpers for trainer-side compute paths.
#
# Background: legacy PokeTransformer (model.py) decomposes its forward into
# (forward_spatial, action_encoder, policy_head, value_head, v_support) modules
# that InferenceBatcher and ppo.ppo_update call individually so they can share
# a single spatial pass across N concurrent battles / T per-episode turns.
#
# TransformerBattlePolicy (model_transformer.py) was designed without that
# decomposition exposed — its native API is `forward(batch, history)`. To keep
# InferenceBatcher / ppo.ppo_update arch-agnostic without touching the legacy
# model class or polluting the transformer with legacy-shaped methods, the four
# helpers below dispatch on duck-typed arch detection.
#
# See docs/ARCH_AUDIT.md §2 for the full audit + rationale.

from __future__ import annotations

import torch


def _is_transformer(model) -> bool:
    """Duck-type check. Both arches have `spatial.*`/`temporal.*` so we key
    off attrs unique to TransformerBattlePolicy. Mirrors the discriminator
    in `battle_agent_transformer.is_transformer_checkpoint`."""
    return hasattr(model, "tokenizer") and hasattr(model, "_per_action_context")


def call_action_encoder(model, mega: dict, spatial_out: torch.Tensor) -> torch.Tensor:
    """Compute (B, 9, d_model) per-action context.

    Legacy: calls `model.action_encoder(move_ids, banks, cont, sw_ids, sw_cont)`.
    Transformer: calls `model.action_encoder_from_spatial(mega, spatial_out)`,
    which derives spatial-order ids via the tokenizer and dispatches to
    `_per_action_context` (reusing the already-computed spatial pass).
    """
    if _is_transformer(model):
        return model.action_encoder_from_spatial(mega, spatial_out)
    return model.action_encoder(
        mega["active_move_ids"], mega["active_move_banks"],
        mega["active_move_cont"], mega["switch_ids"], mega["switch_cont"],
    )


def call_policy_logits(model, pi_input: torch.Tensor) -> torch.Tensor:
    """Run the policy MLP on a pre-concatenated (B, 9, 2D+Dt) tensor.

    Returns (B, 9) — squeeze trailing dim to match legacy convention.
    Caller is responsible for legal-mask application (matches legacy site).
    """
    if _is_transformer(model):
        return model.action_head.mlp(pi_input).squeeze(-1)
    return model.policy_head(pi_input).squeeze(-1)


def call_value_logits(model, vi: torch.Tensor) -> torch.Tensor:
    """Run the value MLP on a pre-concatenated (B, D+Dt) tensor.

    Returns v_logits (B, v_bins). Caller computes the scalar value via
    softmax × v_support — matches legacy convention so the post-MLP
    arithmetic in InferenceBatcher / ppo.ppo_update is unchanged.
    """
    if _is_transformer(model):
        return model.value_head.mlp(vi)
    return model.value_head(vi)


def get_v_support(model) -> torch.Tensor:
    """Return the value-bin centers buffer.

    Legacy registers it at the model root; transformer registers it inside
    `value_head`. Callers in ppo.ppo_update / InferenceBatcher use it to
    compute `(softmax(v_logits) * v_support).sum(-1)`.
    """
    if _is_transformer(model):
        return model.value_head.v_support
    return model.v_support
