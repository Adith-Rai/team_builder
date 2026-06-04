#!/usr/bin/env python
# Verification: compute_awr_loss(packed=False) vs compute_awr_loss(packed=True)
# should give matching loss + advantage stats on the same replay batch.
#
# The two paths use different forwards (forward_ppo_sequence vs
# forward_ppo_sequence_packed) and different aggregation layouts ((B,L_max,...)
# vs (sum_T,...)). They MUST agree at the math level if we're going to claim
# AWR with --packed is safe.
#
# Tolerances: 1e-3 (matches forward_paths_equivalence test). Different
# floating-point reduction order between padded and packed CAN produce
# bit-level differences; we accept ~fp32 noise.
#
# Usage:
#   python test_awr_packed_equivalence.py \
#     --ckpt data/models/rl_v10/lr8e5_v1_flash/.../snapshot_0139.pt \
#     --memmap data/datasets/human_v8_5k

from __future__ import annotations
import argparse
import sys

import torch

from awr_replay import AWRReplayBuffer, compute_awr_loss
from train_rl import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--memmap", default="data/datasets/human_v8_5k")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tol", type=float, default=1e-3)
    args = p.parse_args()

    device = torch.device(args.device)

    print(f"=== AWR padded vs packed equivalence test ===")
    print(f"  memmap: {args.memmap}")
    print(f"  ckpt:   {args.ckpt}")
    print(f"  device: {device}")
    print(f"  tol:    {args.tol}")
    print()

    buf = AWRReplayBuffer(args.memmap, rng_seed=42)
    model, cfg, _ = load_checkpoint(args.ckpt, device)
    model.eval()

    batch = buf.sample(args.batch_size, device)
    print(f"sampled: B={batch['mask'].shape[0]}, T_max={batch['mask'].shape[1]}, "
          f"n_valid={int(batch['mask'].sum().item())}, "
          f"sum_T={int(batch['seq_lens'].sum().item())}")
    print()

    with torch.no_grad():
        out_padded = compute_awr_loss(model, batch, device,
                                       awr_binary=True, detach_value=True,
                                       packed=False)
        out_packed = compute_awr_loss(model, batch, device,
                                       awr_binary=True, detach_value=True,
                                       packed=True)

    print("=== compare AWR loss + diagnostics ===")
    all_pass = True
    for key in ["loss", "advantage_mean", "advantage_pos_frac",
                "weight_max", "weight_mean", "n_valid", "n_total_valid_pad"]:
        a = out_padded[key].item()
        b = out_packed[key].item()
        diff = abs(a - b)
        pass_str = "PASS" if diff < args.tol else "FAIL"
        if diff >= args.tol:
            all_pass = False
        print(f"  {key:24s}: padded={a:>12.6f}  packed={b:>12.6f}  diff={diff:.2e}  [{pass_str}]")

    # The n_valid + n_total_valid_pad checks: in padded format, n_valid counts
    # non-pad AND non-NaN-R positions; in packed format, n_valid counts non-NaN-R.
    # n_total_valid_pad: padded uses sum of pad_mask; packed uses sum_T.
    # If batch has no NaN terminal_result, both should equal sum_T.

    print()
    if all_pass:
        print(f"VERDICT: PADDED AWR ≈ PACKED AWR (all diffs < {args.tol}).")
        print("        Safe to use --packed with AWR.")
        sys.exit(0)
    else:
        print(f"VERDICT: PADDED AWR DIVERGES FROM PACKED AWR (some diffs > {args.tol}).")
        print("        DO NOT enable --packed with AWR until investigated.")
        sys.exit(2)


if __name__ == "__main__":
    main()
