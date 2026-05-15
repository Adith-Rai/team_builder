"""S64 Phase B.3+B.4 bundled bit-equivalence test.

Builds synthetic episodes with real-shape feat_batches (matching the contract
the model actually consumes — 15+ keys with correct dims per features.py
build_turn_batch and DIMS dict). Runs BOTH:
  - legacy: collate_episodes → forward_ppo_sequence → (B, L_max, ...) outputs
  - packed: collate_episodes_packed → forward_ppo_sequence_packed → (sum_T, ...) outputs

Asserts:
  (i) shape correctness on the packed outputs
  (ii) numerical bit-equiv at all valid (b, t) ↔ idx positions, fp32 eval mode

Tolerance: rtol=1e-4 atol=1e-5 (rationale in B.2 results memo — flex_attention
Triton kernel vs SDPA-slow-path cuDNN math cannot be bit-identical at fp32,
~7e-6 absolute drift is expected from 4-layer cross-kernel compose).

Run on prod:
  cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
  export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
  python test_forward_ppo_sequence_packed.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from ppo import load_checkpoint, collate_episodes, collate_episodes_packed


# Tolerance bands. fp32: cross-kernel reorder (flex_attention/Triton vs
# SDPA-slow-path/cuDNN) unavoidable at ~7e-6 max-abs after 4 layers; real
# bugs show as max_abs >= 1e-3. bf16: 7-bit mantissa = ~6e-3 representable
# precision; reasonable transformer-output spread (~1.0) means absolute
# diff up to ~1e-2 is expected with no semantic bug. Tolerance bands
# match the Phase A memo §4-R.2 spec for B.4.
FP32_RTOL = 1e-4
FP32_ATOL = 1e-5
BF16_RTOL = 1e-2
BF16_ATOL = 1e-2


def _make_synthetic_turn(rng: np.random.RandomState, A: int = 9) -> dict:
    """Build one per-turn feat dict matching features.build_turn_batch output.

    All keys/shapes per features.py DIMS dict + build_turn_batch (line 1542-1629).
    Content is random — the model doesn't crash on out-of-vocab ids since
    every embedding layer has nn.Embedding's normal random-init weights at
    every index (and BC v10's embeds have been trained on the actual vocab,
    but synthetic ids in-range still produce valid forward values).
    """
    g = torch.Generator().manual_seed(int(rng.randint(0, 2**31 - 1)))

    # Bounds chosen safely inside known vocab ranges to avoid embed OOB.
    # Actual production vocabs are larger; these are conservative.
    SPECIES = 100; ITEM = 50; ABILITY = 50; MOVE = 100; GEN = 9

    def _ri(low, high, shape):
        return torch.randint(low, high, shape, generator=g, dtype=torch.long)

    def _rf(shape):
        return torch.randn(shape, generator=g, dtype=torch.float32)

    turn = {
        # Pokemon ids — (1, 6, 3) = [species, item, ability] per pokemon
        "our_pokemon_ids":     _ri(1, SPECIES, (1, 6, 3)),
        "opp_pokemon_ids":     _ri(1, SPECIES, (1, 6, 3)),
        # Pokemon banks — (1, 6, 10) = hp_pct, level, weight, height, 6 stats
        "our_pokemon_banks":   _ri(0, 100, (1, 6, 10)),
        "opp_pokemon_banks":   _ri(0, 100, (1, 6, 10)),
        # Pokemon continuous — (1, 6, 285)
        "our_pokemon_cont":    _rf((1, 6, 285)),
        "opp_pokemon_cont":    _rf((1, 6, 285)),
        # Pokemon move ids — (1, 6, 4) = move0-3 per pokemon
        "our_pokemon_move_ids": _ri(1, MOVE, (1, 6, 4)),
        "opp_pokemon_move_ids": _ri(1, MOVE, (1, 6, 4)),
        # Pokemon move cont — (1, 6, 4, 23) = 4 moves × 23 cont dims per pokemon
        "our_pokemon_move_cont": _rf((1, 6, 4, 23)),
        "opp_pokemon_move_cont": _rf((1, 6, 4, 23)),
        # Field — (1, 52) cont + 4 bank scalars
        "field_cont": _rf((1, 52)),
        "field_banks": {
            "turn":         _ri(0, 50, (1,)),
            "weather_dur":  _ri(0, 8, (1,)),
            "terrain_dur":  _ri(0, 8, (1,)),
            "tr_dur":       _ri(0, 5, (1,)),
        },
        # Transition — (1, 51) cont + 2 ids
        "transition_cont": _rf((1, 51)),
        "transition_ids": {
            "our_action": _ri(0, 9, (1,)),
            "opp_action": _ri(0, 9, (1,)),
        },
        # Active moves — (1, 4) ids + (1, 4, 109) cont + 4 banks (1, 4)
        "active_move_ids":  _ri(1, MOVE, (1, 4)),
        "active_move_cont": _rf((1, 4, 109)),
        "active_move_banks": {
            "bp":   _ri(0, 200, (1, 4)),
            "acc":  _ri(0, 100, (1, 4)),
            "pp":   _ri(0, 50, (1, 4)),
            "prio": _ri(0, 7, (1, 4)),  # priority -6..+6 → +6 shift
        },
        # Switch slots — (1, 5) ids + (1, 5, 30) cont
        "switch_ids":  _ri(1, SPECIES, (1, 5)),
        "switch_cont": _rf((1, 5, 30)),
        # Legal mask — (1, 9), at least 2 actions legal to avoid degenerate softmax
        "legal_mask": (_rf((1, A)) > 0.0).float() + 0.5,  # always > 0.5 ish
        # Gen id
        "gen_id": torch.tensor([GEN], dtype=torch.long),
    }
    # Ensure legal_mask has at least 2 legal actions to avoid -100 collapse
    turn["legal_mask"] = torch.ones(1, A, dtype=torch.float32)

    return turn


def _make_synthetic_episode(T: int, A: int = 9, seed: int = 0) -> dict:
    """Build a full episode dict matching build_ppo_episodes output."""
    rng = np.random.RandomState(seed)
    feat_batches = [_make_synthetic_turn(rng, A=A) for _ in range(T)]
    return {
        "feat_batches": feat_batches,
        "actions":      rng.randint(0, A, size=T).tolist(),
        "old_logp":     rng.randn(T).astype(np.float32).tolist(),
        "advantages":   rng.randn(T).astype(np.float32).tolist(),
        "returns":      rng.randn(T).astype(np.float32).tolist(),
        "action_masks": [[1.0] * A for _ in range(T)],  # all legal
    }


def _check_forward_equiv(legacy_out: dict, packed_out: dict, T_list, cu_seqlens,
                          rtol: float, atol: float, label: str) -> bool:
    """Compare legacy (B, L_max, ...) outputs vs packed (sum_T, ...) outputs
    at every valid (b, t) ↔ idx position. Three output fields: action_logits,
    value, v_logits."""
    B = len(T_list)
    cu = cu_seqlens.long()
    all_pass = True

    for field in ["action_logits", "value", "v_logits"]:
        legacy_t = legacy_out[field]
        packed_t = packed_out[field]
        max_abs_overall = 0.0
        for b in range(B):
            start, stop = int(cu[b].item()), int(cu[b + 1].item())
            T_b = T_list[b]
            legacy_slice = legacy_t[b, :T_b]
            packed_slice = packed_t[start:stop]

            # Action logits use -100.0 fill at illegal positions; both sides
            # mask the same way so the diff there is exact zero. Compare raw.
            diff = (legacy_slice - packed_slice).abs()
            max_abs = diff.max().item()
            max_abs_overall = max(max_abs_overall, max_abs)

        passed = max_abs_overall <= atol + rtol * legacy_t.abs().max().item()
        marker = "OK" if passed else "FAIL"
        print(f"  [{label}] {field:15s}: max_abs={max_abs_overall:.2e}  [{marker}]")
        if not passed:
            all_pass = False
    return all_pass


def test_forward_ppo_sequence_packed():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, torch={torch.__version__}")

    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
        print(f"mha fastpath disabled: {not torch.backends.mha.get_fastpath_enabled()}")

    print("Loading BC v10...")
    model, cfg, _ = load_checkpoint("data/models/bc/v10_padded_for_cis_dev.pt", device)
    model.eval()
    print(f"BC v10 loaded. n_actions={cfg.format_config.n_actions}, "
          f"v_bins={cfg.v_bins}, temporal_context={cfg.temporal_context}")
    print()

    # Three cases mirroring test_temporal_packed.py: B=1, B=3 varied, B=5 edge.
    # Episode T values kept modest (max 50) to keep test wall low — the
    # B.2 gate already validated the temporal stack at T=200.
    # Each case runs at both fp32 (no autocast) and bf16 (autocast — the
    # production path under --bf16). Different tolerance bands per dtype.
    cases = [
        ("B=1",         [25]),
        ("B=3 varied",  [10, 30, 20]),
        ("B=5 mixed",   [5, 15, 1, 40, 25]),
    ]
    dtypes = [
        ("fp32", None,                FP32_RTOL, FP32_ATOL),
        ("bf16", torch.bfloat16,      BF16_RTOL, BF16_ATOL),
    ]

    all_pass = True
    for label, T_list in cases:
      for dtype_label, autocast_dtype, rtol, atol in dtypes:
        full_label = f"{label}/{dtype_label}"
        print(f"Case {full_label}: T_list={T_list}")
        episodes = [_make_synthetic_episode(T, A=cfg.format_config.n_actions, seed=i)
                    for i, T in enumerate(T_list)]

        # Build BOTH collated forms from the SAME episodes
        legacy_collated = collate_episodes(
            episodes, L_max=cfg.temporal_context, device=device, tail=True,
        )
        packed_collated = collate_episodes_packed(
            episodes, max_seqlen=cfg.temporal_context, device=device, tail=True,
        )

        # Sanity: both should report B equal
        assert legacy_collated["B"] == packed_collated["B"] == len(T_list), \
            f"B mismatch: legacy={legacy_collated['B']} packed={packed_collated['B']}"

        # Run both forwards — eval + no_grad. fp32: no autocast. bf16: with
        # autocast (matches production ppo_update_batched _update_amp_ctx path).
        from contextlib import nullcontext
        autocast_ctx = (torch.autocast("cuda", dtype=autocast_dtype)
                         if autocast_dtype is not None else nullcontext())
        with torch.no_grad(), autocast_ctx:
            legacy_out = model.forward_ppo_sequence(legacy_collated, device)
            packed_out = model.forward_ppo_sequence_packed(packed_collated, device)

        # Shape checks (B.3-equivalent gate). legacy collate with L_max=
        # cfg.temporal_context pads to 200, NOT max(T_list) — matches what
        # ppo_update_batched does in production (ppo.py:1266-1268).
        sum_T = sum(T_list)
        legacy_L = legacy_collated["L_max"]
        B = len(T_list)
        n_actions = cfg.format_config.n_actions
        v_bins = cfg.v_bins
        assert legacy_out["action_logits"].shape == (B, legacy_L, n_actions), \
            f"legacy action_logits shape: {legacy_out['action_logits'].shape}"
        assert packed_out["action_logits"].shape == (sum_T, n_actions), \
            f"packed action_logits shape: {packed_out['action_logits'].shape}"
        assert packed_out["value"].shape == (sum_T,), \
            f"packed value shape: {packed_out['value'].shape}"
        assert packed_out["v_logits"].shape == (sum_T, v_bins), \
            f"packed v_logits shape: {packed_out['v_logits'].shape}"
        print(f"  shapes OK: legacy=(B={B}, L={legacy_L}, ...) packed=(sum_T={sum_T}, ...)")

        # Bit-equiv check at valid positions (B.4 gate)
        passed = _check_forward_equiv(
            legacy_out, packed_out, T_list, packed_collated["cu_seqlens"],
            rtol=rtol, atol=atol, label=full_label,
        )
        if not passed:
            all_pass = False
        print()

    print("=" * 60)
    if all_pass:
        print("B.3+B.4 GATE: PASS")
        print(f"  forward_ppo_sequence_packed bit-equivalent to forward_ppo_sequence")
        print(f"  at fp32 eval mode, BC v10 weights, across B=1/B=3/B=5 cases.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("B.3+B.4 GATE: FAIL — diagnose before proceeding to B.5")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    test_forward_ppo_sequence_packed()
