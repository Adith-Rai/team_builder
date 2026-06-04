#!/usr/bin/env python
# RIGOROUS verification: bc_to_ppo_format vs bc_to_ppo_packed_format
# must produce EQUIVALENT tensors at the input level (before any model
# forward). If inputs match exactly, the only divergence between
# forward_ppo_sequence and forward_ppo_sequence_packed outputs comes
# from the model's temporal attention implementation (varlen vs padded
# causal-masked), NOT from a packing bug.
#
# This isolates "adapter correctness" from "model forward equivalence".
# If THIS test passes, training pollution from a packing bug is
# eliminated. If THIS test FAILS, there's a real adapter bug to fix
# before any AWR training.
#
# Comparison method: for each feature key, flatten padded
# tensor[b, :seq_lens[b]] over all b in batch order, then compare
# element-wise to the packed flat tensor. Must be bit-equivalent
# (zero tolerance — these are tensor views/slices of the same source,
# no arithmetic ops involved).
#
# Usage:
#   python test_packed_adapter_inputs.py --memmap data/datasets/human_v8_5k

from __future__ import annotations
import argparse
import sys
import torch

from awr_replay import (
    AWRReplayBuffer,
    bc_to_ppo_format,
    bc_to_ppo_packed_format,
)


def _flatten_padded(tensor: torch.Tensor, seq_lens: list) -> torch.Tensor:
    """Flatten (B, L_max, ...) padded -> (sum_T, ...) by taking valid slices."""
    B = tensor.shape[0]
    parts = []
    for b in range(B):
        parts.append(tensor[b, :seq_lens[b]])
    return torch.cat(parts, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--memmap", default="data/datasets/human_v8_5k")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cpu",
                   help="cpu works; we're testing CPU tensor equivalence")
    args = p.parse_args()

    device = torch.device(args.device)

    print(f"=== bc_to_ppo_format vs bc_to_ppo_packed_format input equivalence ===")
    print(f"  memmap: {args.memmap}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  device: {device}")
    print()

    buf = AWRReplayBuffer(args.memmap, rng_seed=42)
    bc_batch = buf.sample(args.batch_size, device)

    seq_lens = bc_batch["seq_lens"].tolist()
    B = bc_batch["mask"].shape[0]
    sum_T = sum(seq_lens)
    print(f"sampled: B={B}, T_max={bc_batch['mask'].shape[1]}, sum_T={sum_T}, seq_lens={seq_lens}")
    print()

    padded = bc_to_ppo_format(bc_batch)
    packed = bc_to_ppo_packed_format(bc_batch)

    # Sanity: top-level structure
    print("=== top-level shapes ===")
    print(f"  padded: feat_batches keys={len(padded['feat_batches'])}, "
          f"pad_mask={tuple(padded['pad_mask'].shape)}, "
          f"seq_lens={tuple(padded['seq_lens'].shape)}, B={padded['B']}, L_max={padded['L_max']}")
    print(f"  packed: flat_feat_batches keys={len(packed['flat_feat_batches'])}, "
          f"cu_seqlens={tuple(packed['cu_seqlens'].shape)}, "
          f"seq_lens={tuple(packed['seq_lens'].shape)}, B={packed['B']}, max_seqlen={packed['max_seqlen']}")
    print()

    # cu_seqlens correctness
    expected_cu = torch.tensor([0] + [sum(seq_lens[:i+1]) for i in range(B)], dtype=torch.int32)
    cu_match = torch.equal(packed["cu_seqlens"].cpu(), expected_cu)
    print(f"cu_seqlens correct: {cu_match} (expected {expected_cu.tolist()}, got {packed['cu_seqlens'].tolist()})")
    print()

    # Compare each tensor in feat_batches
    padded_fb = padded["feat_batches"]
    packed_fb = packed["flat_feat_batches"]

    all_keys = set(padded_fb.keys()) | set(packed_fb.keys())
    print(f"=== per-feature input comparison ({len(all_keys)} keys) ===")
    all_pass = True
    mismatches = []

    for key in sorted(all_keys):
        if key not in padded_fb:
            print(f"  {key:24s}: MISSING in padded")
            all_pass = False
            continue
        if key not in packed_fb:
            print(f"  {key:24s}: MISSING in packed")
            all_pass = False
            continue

        padded_val = padded_fb[key]
        packed_val = packed_fb[key]

        if isinstance(padded_val, dict):
            # Nested dict (field_banks, transition_ids, active_move_banks).
            if not isinstance(packed_val, dict):
                print(f"  {key:24s}: TYPE MISMATCH (padded=dict, packed={type(packed_val).__name__})")
                all_pass = False
                continue
            for inner_key in sorted(set(padded_val.keys()) | set(packed_val.keys())):
                if inner_key not in padded_val or inner_key not in packed_val:
                    print(f"  {key}.{inner_key:18s}: KEY MISSING")
                    all_pass = False
                    continue
                p_inner = padded_val[inner_key]
                k_inner = packed_val[inner_key]
                p_flat = _flatten_padded(p_inner, seq_lens)
                if p_flat.shape != k_inner.shape:
                    print(f"  {key}.{inner_key:18s}: SHAPE MISMATCH "
                          f"(padded_flat={p_flat.shape}, packed={k_inner.shape})")
                    all_pass = False
                    mismatches.append(f"{key}.{inner_key}")
                    continue
                eq = torch.equal(p_flat, k_inner)
                pass_str = "PASS" if eq else "FAIL"
                if not eq:
                    diff = (p_flat.float() - k_inner.float()).abs().max().item()
                    print(f"  {key}.{inner_key:18s}: {pass_str} (max diff {diff:.3e})")
                    all_pass = False
                    mismatches.append(f"{key}.{inner_key}")
                else:
                    print(f"  {key}.{inner_key:18s}: {pass_str}")
        else:
            # Flat tensor.
            if key == "gen_id":
                # Both produced via torch.full; padded is (B, L_max), packed is (sum_T,).
                # Flatten padded then compare.
                p_flat = _flatten_padded(padded_val, seq_lens)
            else:
                p_flat = _flatten_padded(padded_val, seq_lens)
            if p_flat.shape != packed_val.shape:
                print(f"  {key:24s}: SHAPE MISMATCH "
                      f"(padded_flat={p_flat.shape}, packed={packed_val.shape})")
                all_pass = False
                mismatches.append(key)
                continue
            eq = torch.equal(p_flat, packed_val)
            pass_str = "PASS" if eq else "FAIL"
            if not eq:
                diff = (p_flat.float() - packed_val.float()).abs().max().item()
                print(f"  {key:24s}: {pass_str} (max diff {diff:.3e})")
                all_pass = False
                mismatches.append(key)
            else:
                print(f"  {key:24s}: {pass_str}")

    print()
    if all_pass and cu_match:
        print("VERDICT: PACKED ADAPTER CORRECTLY MIRRORS PADDED ADAPTER.")
        print("        Any forward-output divergence is from model's temporal attention,")
        print("        NOT from a packing bug. Adapter is safe for training.")
        sys.exit(0)
    else:
        print(f"VERDICT: PACKED ADAPTER HAS BUGS at {len(mismatches)} feature(s):")
        for m in mismatches:
            print(f"  - {m}")
        if not cu_match:
            print(f"  - cu_seqlens incorrect")
        print()
        print("DO NOT use packed AWR until adapter fixed (would cause training pollution).")
        sys.exit(2)


if __name__ == "__main__":
    main()
