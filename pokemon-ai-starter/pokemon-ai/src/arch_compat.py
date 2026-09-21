# arch_compat.py — adapter between the batched call pattern and the model API.
#
# InferenceBatcher and ppo.ppo_update do not call `model.forward(...)`. They
# run one shared spatial pass across N concurrent battles / T per-episode
# turns, then invoke the action-encoder, policy head and value head on that
# shared output. TransformerBattlePolicy's native API is `forward(batch,
# history)`, so these four helpers are the single place that knows how to
# drive its pieces individually.
#
# HISTORY: this module was originally an arch DISPATCHER, bridging the legacy
# PokeTransformer (which exposed forward_spatial / action_encoder /
# policy_head / value_head as separate modules) and TransformerBattlePolicy
# (which does not). The legacy arch was retired in S69, so the dispatch and
# every legacy branch are gone. The module is kept because the adapter role
# is still real — deleting it would duplicate this logic across ppo.py,
# inference_batcher.py and mp_centralized_collect.py.
#
# See docs/CURRENT_STATE.md section 2 (step 2) for the retirement.

from __future__ import annotations

import torch


def call_action_encoder(model, mega: dict, spatial_out: torch.Tensor) -> torch.Tensor:
    """Compute (B, n_actions, d_model) per-action context.

    Calls `model.action_encoder_from_spatial(mega, spatial_out)`, which derives
    spatial-order ids via the tokenizer and dispatches to
    `_per_action_context`, reusing the already-computed spatial pass.
    """
    return model.action_encoder_from_spatial(mega, spatial_out)


def call_policy_logits(model, pi_input: torch.Tensor) -> torch.Tensor:
    """Run the policy MLP on a pre-concatenated (B, n_actions, 2D+Dt) tensor.

    Returns (B, n_actions). The caller applies the legal mask.
    """
    return model.action_head.mlp(pi_input).squeeze(-1)


def call_value_logits(model, vi: torch.Tensor) -> torch.Tensor:
    """Run the value MLP on a pre-concatenated (B, D+Dt) tensor.

    Returns v_logits (B, v_bins). The caller computes the scalar value as
    `(softmax(v_logits) * v_support).sum(-1)`.
    """
    return model.value_head.mlp(vi)


def get_v_support(model) -> torch.Tensor:
    """Return the value-bin centers buffer, registered inside `value_head`."""
    return model.value_head.v_support
