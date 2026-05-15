"""S64 Phase B.5 loss-equivalence test.

Runs both legacy and packed PPO loss paths on matching collated inputs +
forward outputs (the latter validated by B.3/B.4 bit-equiv gate). Compares
total_loss + each component (pi_loss, entropy, v_loss, approx_kl,
ratio_clip_frac, bc_kl) between the two paths.

Mathematical contract: at matching valid positions, the legacy aggregation
`(per_pos * pad_mask).sum() / n_valid` is exactly `per_pos.mean()` over
the packed flat tensor. Any difference is floating-point reorder noise +
the small upstream forward-output drift (B.4 measured ~1e-6 fp32, ~1e-2 bf16).

Tolerance bands:
  fp32: rtol=1e-3 atol=1e-4 — slightly wider than B.4's forward gate because
        losses involve more reductions (sum-then-divide).
  bf16: rtol=5e-2 atol=5e-3 — accommodates compounded bf16 reductions.

Three coverage axes:
  - Three batch shapes (B=1, B=3 varied, B=5 mixed) — reuses B.3/B.4 fixtures
  - fp32 + bf16 (matches production --bf16 autocast)
  - BC anchor OFF + BC anchor ON (production uses ON with coef=0.1)

Run on prod:
  cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
  export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
  python test_loss_packed.py
"""
import os
import sys
from contextlib import nullcontext

import torch

sys.path.insert(0, os.path.dirname(__file__))

from ppo import (
    load_checkpoint, collate_episodes, collate_episodes_packed,
    _ppo_loss_batched_internal, _ppo_loss_packed_internal,
)
# Reuse the B.3/B.4 synthetic episode builder
from test_forward_ppo_sequence_packed import _make_synthetic_episode


FP32_RTOL = 1e-3
FP32_ATOL = 1e-4
BF16_RTOL = 5e-2
BF16_ATOL = 5e-3

# Loss components to A/B (all scalar tensors)
LOSS_KEYS = ["total_loss", "pi_loss", "entropy", "v_loss",
             "approx_kl", "ratio_clip_frac", "bc_kl"]


def _compare_loss_dicts(legacy: dict, packed: dict, rtol: float, atol: float,
                         label: str) -> bool:
    """Compare loss-dict scalars between legacy + packed paths."""
    all_pass = True
    for k in LOSS_KEYS:
        l = legacy[k].item() if hasattr(legacy[k], "item") else float(legacy[k])
        p = packed[k].item() if hasattr(packed[k], "item") else float(packed[k])
        diff = abs(l - p)
        threshold = atol + rtol * max(abs(l), abs(p), 1e-12)
        passed = diff <= threshold
        marker = "OK" if passed else "FAIL"
        print(f"  [{label}] {k:18s}: legacy={l:.6e}  packed={p:.6e}  "
              f"diff={diff:.2e}  [{marker}]")
        if not passed:
            all_pass = False
    return all_pass


def test_loss_packed():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, torch={torch.__version__}")

    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
        print(f"mha fastpath disabled: {not torch.backends.mha.get_fastpath_enabled()}")

    print("Loading BC v10...")
    model, cfg, _ = load_checkpoint("data/models/bc/v10_padded_for_cis_dev.pt", device)
    model.eval()
    print(f"BC v10 loaded. v_bins={cfg.v_bins}, v_min={cfg.v_min}, v_max={cfg.v_max}")
    print()

    cases = [
        ("B=1",         [25]),
        ("B=3 varied",  [10, 30, 20]),
        ("B=5 mixed",   [5, 15, 1, 40, 25]),
    ]
    dtypes = [
        ("fp32", None,           FP32_RTOL, FP32_ATOL),
        ("bf16", torch.bfloat16, BF16_RTOL, BF16_ATOL),
    ]
    bc_settings = [
        ("noBC", False),
        ("BC=0.1", True),
    ]

    # Hyperparams matching production canonical Phase 2 stack
    ent_coef, vf_coef, clip_eps = 0.02, 0.5, 0.2
    bc_anchor_coef = 0.1

    all_pass = True
    for case_label, T_list in cases:
      episodes = [_make_synthetic_episode(T, A=cfg.format_config.n_actions, seed=i)
                  for i, T in enumerate(T_list)]
      for dtype_label, autocast_dtype, rtol, atol in dtypes:
        for bc_label, bc_on in bc_settings:
          full_label = f"{case_label}/{dtype_label}/{bc_label}"
          print(f"Case {full_label}: T_list={T_list}")

          legacy_collated = collate_episodes(
              episodes, L_max=cfg.temporal_context, device=device, tail=True,
          )
          packed_collated = collate_episodes_packed(
              episodes, max_seqlen=cfg.temporal_context, device=device, tail=True,
          )

          autocast_ctx = (torch.autocast("cuda", dtype=autocast_dtype)
                           if autocast_dtype is not None else nullcontext())

          with torch.no_grad(), autocast_ctx:
              legacy_fwd = model.forward_ppo_sequence(legacy_collated, device)
              packed_fwd = model.forward_ppo_sequence_packed(packed_collated, device)

              # BC ref logits: use the model itself as BC ref (KL exactly 0
              # against trainable model when forward outputs are bit-identical;
              # both paths get the same bc_logits format as their forward
              # output). Detached, no grad. This is a STRUCTURAL test: legacy
              # and packed paths must compute bc_kl identically given matching
              # bc_logits shapes.
              if bc_on:
                  legacy_bc = legacy_fwd["action_logits"].detach()
                  packed_bc = packed_fwd["action_logits"].detach()
              else:
                  legacy_bc = None
                  packed_bc = None

              legacy_loss = _ppo_loss_batched_internal(
                  legacy_collated, legacy_fwd, model, cfg,
                  ent_coef=ent_coef, vf_coef=vf_coef, clip_eps=clip_eps,
                  normalize_advantages=False,
                  bc_logits=legacy_bc, bc_anchor_coef=bc_anchor_coef if bc_on else 0.0,
              )
              packed_loss = _ppo_loss_packed_internal(
                  packed_collated, packed_fwd, model, cfg,
                  ent_coef=ent_coef, vf_coef=vf_coef, clip_eps=clip_eps,
                  normalize_advantages=False,
                  bc_logits=packed_bc, bc_anchor_coef=bc_anchor_coef if bc_on else 0.0,
              )

          passed = _compare_loss_dicts(
              legacy_loss, packed_loss, rtol=rtol, atol=atol, label=full_label,
          )
          if not passed:
              all_pass = False
          print()

    print("=" * 60)
    if all_pass:
        print("B.5 GATE: PASS")
        print("  _ppo_loss_packed_internal numerically equivalent to legacy")
        print("  across B=1/3/5 × fp32/bf16 × noBC/BC=0.1 (12 combinations).")
        print("=" * 60)
        sys.exit(0)
    else:
        print("B.5 GATE: FAIL")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    test_loss_packed()
