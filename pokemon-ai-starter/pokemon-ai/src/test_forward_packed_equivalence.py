#!/usr/bin/env python
# Verification: forward_ppo_sequence (padded) vs forward_ppo_sequence_packed
# at corresponding valid positions should give bit-equivalent outputs
# (or at least within fp32 reduction-order noise).
#
# This is the analog of test_forward_paths_equivalence.py (which tested
# forward_sequence vs forward_ppo_sequence and found ~1.9e-3 divergence).
# We need the same check for padded vs packed PPO forwards because AWR
# is now switchable between them via --packed.
#
# Both forwards SHOULD produce identical outputs at matching positions
# (per model_transformer.py docstrings), but it was never explicitly
# verified. If they diverge meaningfully, AWR loss matches across the
# two paths only because of fortuitous reduction-order averaging.
#
# Compare:
#   padded output: (B, L_max, n_actions) at valid (b, t) positions
#   packed output: (sum_T, n_actions) at packed indices
# Map: packed[cu_seqlens[b] : cu_seqlens[b+1]] ↔ padded[b, :seq_lens[b]]

from __future__ import annotations
import argparse
import sys

import torch

from awr_replay import (
    AWRReplayBuffer,
    bc_to_ppo_format,
    bc_to_ppo_packed_format,
)
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

    print(f"=== forward_ppo_sequence (PADDED) vs forward_ppo_sequence_packed equivalence ===")
    print(f"  memmap: {args.memmap}")
    print(f"  ckpt:   {args.ckpt}")
    print(f"  device: {device}")
    print(f"  tol:    {args.tol}")
    print()

    buf = AWRReplayBuffer(args.memmap, rng_seed=42)
    model, cfg, _ = load_checkpoint(args.ckpt, device)
    model.eval()

    bc_batch = buf.sample(args.batch_size, device)
    B = bc_batch["mask"].shape[0]
    seq_lens = bc_batch["seq_lens"].tolist()
    n_valid = sum(seq_lens)
    print(f"sampled: B={B}, T_max={bc_batch['mask'].shape[1]}, n_valid (sum_T)={n_valid}, seq_lens={seq_lens}")
    print()

    with torch.no_grad():
        # Padded path
        padded_batch = bc_to_ppo_format(bc_batch)
        out_padded = model.forward_ppo_sequence(padded_batch, device)
        # Packed path
        packed_batch = bc_to_ppo_packed_format(bc_batch)
        out_packed = model.forward_ppo_sequence_packed(packed_batch, device)

    print(f"padded shapes:")
    print(f"  action_logits: {tuple(out_padded['action_logits'].shape)}")
    print(f"  value:         {tuple(out_padded['value'].shape)}")
    print(f"  v_logits:      {tuple(out_padded['v_logits'].shape)}")
    print(f"packed shapes:")
    print(f"  action_logits: {tuple(out_packed['action_logits'].shape)}")
    print(f"  value:         {tuple(out_packed['value'].shape)}")
    print(f"  v_logits:      {tuple(out_packed['v_logits'].shape)}")

    # Extract padded's valid positions into a flat (sum_T, ...) ordering
    # that matches packed's layout. Order: for b in range(B), take padded[b, :seq_lens[b]]
    # then concat.
    def _flatten_padded(padded_out: torch.Tensor) -> torch.Tensor:
        parts = []
        for b in range(B):
            parts.append(padded_out[b, :seq_lens[b]])
        return torch.cat(parts, dim=0)

    print()
    print(f"=== compare at {n_valid} matching positions ===")
    all_pass = True
    for key in ["action_logits", "value", "v_logits"]:
        padded_flat = _flatten_padded(out_padded[key].float())
        packed_flat = out_packed[key].float()
        assert padded_flat.shape == packed_flat.shape, \
            f"shape mismatch on {key}: padded_flat={padded_flat.shape}, packed={packed_flat.shape}"
        diff = (padded_flat - packed_flat).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        pass_str = "PASS" if max_diff < args.tol else "FAIL"
        if max_diff >= args.tol:
            all_pass = False
        print(f"  {key:14s}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e} [{pass_str}]")

    print()
    if all_pass:
        a_max = (_flatten_padded(out_padded["action_logits"]) - out_packed["action_logits"]).abs().max().item()
        v_max = (_flatten_padded(out_padded["value"]) - out_packed["value"]).abs().max().item()
        print(f"VERDICT: PADDED ≈ PACKED (max action_logits {a_max:.2e}, max value {v_max:.2e} < {args.tol}).")
        print("        Safe: AWR using either forward path will give same outputs as PPO using either path.")
        sys.exit(0)
    else:
        print(f"VERDICT: PADDED vs PACKED DIVERGE > {args.tol}.")
        print("        AWR must match PPO's choice (--packed) by construction → already correct in our impl.")
        print("        But the divergence itself documents another path-equivalence issue worth noting.")
        sys.exit(2)


if __name__ == "__main__":
    main()
