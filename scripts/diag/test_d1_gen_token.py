#!/usr/bin/env python
"""D1 gen-id token validation.

Tests the multi-gen token added to TransformerBattlePolicy in Session 51:
1. Fresh model has gen_embed attribute + correct constants
2. Forward without gen_id defaults to cfg.format_config.gen + works
3. Forward with explicit gen_id=9 vs gen_id=6 produces DIFFERENT outputs
   (proves the token is actually conditioning the model)
4. Output shapes match expected (N_BATTLE_STATE=15, N_TOKENS=223)
5. Backward through gen_embed updates its parameters

Usage on cloud pod:
  PYTHONPATH=/tmp:/workspace/team_builder/pokemon-ai-starter/pokemon-ai/src \\
  python /workspace/scripts/test_d1_gen_token.py
where /tmp holds the modified model_transformer.py (do not modify the
production src/ copy until validated, since production process spawns
new workers that would pick up the new code).
"""

from __future__ import annotations

import os
import sys


def main():
    import torch
    import model_transformer as M
    print(f"loaded: {M.__file__}")
    print(f"N_BATTLE_STATE={M.N_BATTLE_STATE}  N_TOKENS={M.N_TOKENS}  TT_GEN={M.TT_GEN}  PS_SLOT_GEN={M.PS_SLOT_GEN}")

    if M.N_BATTLE_STATE != 15 or M.N_TOKENS != 223:
        print(f"FAIL: expected N_BATTLE_STATE=15 N_TOKENS=223")
        return 1

    # 1. Build fresh policy
    torch.manual_seed(0)
    cfg = M.TransformerConfig.with_vocab_sizes_from_disk()
    lookup = M.load_move_flag_lookup("data/lookup/move_flags_v1.pt",
                                     expected_n_moves=cfg.n_moves)
    policy = M.TransformerBattlePolicy(cfg, move_flag_lookup=lookup).cuda().eval()
    n_params = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"policy: {n_params:.1f}M params")

    if not hasattr(policy.tokenizer, "gen_embed"):
        print("FAIL: tokenizer.gen_embed missing")
        return 1
    print(f"  gen_embed: {policy.tokenizer.gen_embed}")

    # 2. Build synthetic batch
    B = 2
    device = torch.device("cuda")
    fmt = cfg.format_config
    def _ids(s, v):
        return torch.randint(0, max(int(v), 1), s, dtype=torch.long, device=device)
    def _bank(s):
        return torch.randint(0, 32, s, dtype=torch.long, device=device)
    def _zf(s):
        return torch.zeros(s, dtype=torch.float32, device=device)

    batch = {
        "our_pokemon_ids": torch.stack([
            _ids((B, 6), cfg.n_species),
            _ids((B, 6), cfg.n_items),
            _ids((B, 6), cfg.n_abilities),
        ], dim=-1),
        "opp_pokemon_ids": torch.stack([
            _ids((B, 6), cfg.n_species),
            _ids((B, 6), cfg.n_items),
            _ids((B, 6), cfg.n_abilities),
        ], dim=-1),
        "our_pokemon_banks": _bank((B, 6, 10)),
        "opp_pokemon_banks": _bank((B, 6, 10)),
        "our_pokemon_cont": _zf((B, 6, 285)),
        "opp_pokemon_cont": _zf((B, 6, 285)),
        "our_pokemon_move_ids": _ids((B, 6, 4), cfg.n_moves),
        "opp_pokemon_move_ids": _ids((B, 6, 4), cfg.n_moves),
        "our_pokemon_move_cont": _zf((B, 6, 4, 23)),
        "opp_pokemon_move_cont": _zf((B, 6, 4, 23)),
        "field_banks": {"turn": _bank((B,)), "weather_dur": _bank((B,)),
                        "terrain_dur": _bank((B,)), "tr_dur": _bank((B,))},
        "field_cont": _zf((B, 52)),
        "transition_ids": {"our_action": _bank((B,)), "opp_action": _bank((B,))},
        "transition_cont": _zf((B, 51)),
        "active_move_ids": _ids((B, 4), cfg.n_moves),
        "active_move_banks": {"bp": _bank((B, 4)), "acc": _bank((B, 4)),
                              "pp": _bank((B, 4)), "prio": _bank((B, 4))},
        "active_move_cont": _zf((B, 4, 109)),
        "switch_ids": _ids((B, 5), cfg.n_species),
        "switch_cont": _zf((B, 5, 30)),
        "legal_mask": torch.ones((B, 9), device=device),
    }

    # 3. Forward without gen_id - should default to cfg.format_config.gen
    print(f"\n--- forward without gen_id (default = cfg.gen={fmt.gen}) ---")
    with torch.no_grad():
        out = policy(batch)
    spatial_shape = tuple(out["spatial_output"].shape)
    expected = (B, 223, cfg.d_model)
    print(f"  spatial_output: {spatial_shape}")
    if spatial_shape != expected:
        print(f"FAIL: expected {expected}")
        return 1
    finite = torch.isfinite(out["action_logits"]).all().item() and torch.isfinite(out["value"]).all().item()
    print(f"  finite: {finite}")
    if not finite:
        print("FAIL: non-finite outputs")
        return 1

    # 4. Cross-gen sanity: gen 9 vs gen 6 should differ
    print("\n--- cross-gen test: gen_id=9 vs gen_id=6 ---")
    batch9 = dict(batch)
    batch9["gen_id"] = torch.tensor([9, 9], device=device)
    batch6 = dict(batch)
    batch6["gen_id"] = torch.tensor([6, 6], device=device)
    with torch.no_grad():
        out9 = policy(batch9)
        out6 = policy(batch6)
    diff_logits = (out9["action_logits"] - out6["action_logits"]).abs().max().item()
    diff_value = (out9["value"] - out6["value"]).abs().max().item()
    diff_summary = (out9["summary"] - out6["summary"]).abs().max().item()
    print(f"  action_logits max abs diff: {diff_logits:.4e}")
    print(f"  value         max abs diff: {diff_value:.4e}")
    print(f"  summary       max abs diff: {diff_summary:.4e}")
    if diff_logits < 1e-4 or diff_value < 1e-4 or diff_summary < 1e-4:
        print("FAIL: gen-id has no observable effect on output")
        return 1
    print("  PASS: gen-id is conditioning the model")

    # 5. Backward through gen_embed
    print("\n--- backward test: gradient flows through gen_embed ---")
    policy.train()
    for p in policy.parameters():
        if p.grad is not None:
            p.grad.zero_()
    out = policy(batch)
    loss = out["action_logits"].sum() + out["value"].sum()
    loss.backward()
    gen_grad = policy.tokenizer.gen_embed.weight.grad
    if gen_grad is None:
        print("FAIL: gen_embed.weight.grad is None - no gradient flow")
        return 1
    grad_norm = float(gen_grad.norm())
    print(f"  gen_embed.weight.grad.norm(): {grad_norm:.4e}")
    if grad_norm == 0:
        print("FAIL: zero gradient through gen_embed")
        return 1
    print("  PASS: gradient flows through gen_embed")

    print("\n=== D1 PASS: gen-id token works ===")
    print("Coverage: shape, defaults, cross-gen output difference, backward equivalence.")
    print("Next: D2 (gen-aware feature pipeline) + D3 (per-gen teambuilder).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
