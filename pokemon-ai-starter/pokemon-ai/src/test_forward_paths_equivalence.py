#!/usr/bin/env python
# Verification: forward_sequence (BC path) vs forward_ppo_sequence (PPO/Tier 3 path)
# should produce identical action_logits + value at every valid (b, t) position.
#
# Per model_transformer.py:2473 docstring on forward_ppo_sequence:
#   "Equivalence guarantee: at every valid (b, t) position ... outputs are
#    bit-equivalent (within fp32) to running forward_sequence on the same data."
#
# This script:
#   1. Loads model + samples a real batch from AWRReplayBuffer
#   2. Forward through forward_sequence (BC path) -> (B, T, n_actions) logits + value
#   3. Convert BC-format batch to PPO-format
#   4. Forward through forward_ppo_sequence (PPO path) -> same shape
#   5. Compare outputs at valid positions, report max abs diff
#
# If max diff < 1e-3 (fp32 numerical noise + reduction reorder) -> equivalent ->
# AWR using forward_sequence is mathematically safe.
# If max diff is large -> paths diverge -> AWR MUST switch to forward_ppo_sequence.

from __future__ import annotations
import argparse
import sys

import torch

from awr_replay import AWRReplayBuffer
from train_rl import load_checkpoint


def bc_to_ppo_format(bc_collated: dict) -> dict:
    """Convert dataset.collate_seq output to the dict format that
    forward_ppo_sequence expects (output of ppo.collate_episodes).

    Specifically:
      - flat field_banks_raw (B,L,4) -> nested dict of 4 (B,L) tensors
      - flat trans_ids_raw (B,L,2)  -> dict of 2 (B,L) tensors
      - flat active_move_banks_raw (B,L,4,4) -> dict of 4 (B,L,4) tensors
      - rename *_raw keys to non-raw
      - add pad_mask, B, L_max, seq_lens at top level
      - add gen_id default 9 (gen9ou)
    """
    B, L = bc_collated["our_pokemon_ids"].shape[:2]
    device = bc_collated["our_pokemon_ids"].device
    feat_batches = {
        "our_pokemon_ids":      bc_collated["our_pokemon_ids"],
        "our_pokemon_banks":    bc_collated["our_pokemon_banks"],
        "our_pokemon_cont":     bc_collated["our_pokemon_cont"],
        "our_pokemon_move_ids": bc_collated["our_pokemon_move_ids"],
        "our_pokemon_move_cont": bc_collated["our_pokemon_move_cont"],
        "opp_pokemon_ids":      bc_collated["opp_pokemon_ids"],
        "opp_pokemon_banks":    bc_collated["opp_pokemon_banks"],
        "opp_pokemon_cont":     bc_collated["opp_pokemon_cont"],
        "opp_pokemon_move_ids": bc_collated["opp_pokemon_move_ids"],
        "opp_pokemon_move_cont": bc_collated["opp_pokemon_move_cont"],
        "field_cont":           bc_collated["field_cont_raw"],
        "transition_cont":      bc_collated["trans_cont_raw"],
        "active_move_ids":      bc_collated["active_move_ids_raw"],
        "active_move_cont":     bc_collated["active_move_cont_raw"],
        "switch_ids":           bc_collated["switch_ids_raw"],
        "switch_cont":          bc_collated["switch_cont_raw"],
        "legal_mask":           bc_collated["legal_mask_raw"],
    }
    fb = bc_collated["field_banks_raw"]
    feat_batches["field_banks"] = {
        "turn":        fb[:, :, 0],
        "weather_dur": fb[:, :, 1],
        "terrain_dur": fb[:, :, 2],
        "tr_dur":      fb[:, :, 3],
    }
    ti = bc_collated["trans_ids_raw"]
    feat_batches["transition_ids"] = {
        "our_action": ti[:, :, 0],
        "opp_action": ti[:, :, 1],
    }
    amb = bc_collated["active_move_banks_raw"]
    feat_batches["active_move_banks"] = {
        "bp":   amb[:, :, :, 0],
        "acc":  amb[:, :, :, 1],
        "pp":   amb[:, :, :, 2],
        "prio": amb[:, :, :, 3],
    }
    feat_batches["gen_id"] = torch.full(
        (B, L), 9, dtype=torch.long, device=device
    )

    mask = bc_collated["mask"]
    pad_mask = (mask > 0.5)

    return {
        "feat_batches": feat_batches,
        "pad_mask": pad_mask,
        "seq_lens": bc_collated["seq_lens"],
        "B": B,
        "L_max": L,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--memmap", default="data/datasets/human_v8_5k")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="max abs-diff tolerance for equivalence pass")
    args = p.parse_args()

    device = torch.device(args.device)

    print(f"=== Forward-path equivalence test ===")
    print(f"  memmap: {args.memmap}")
    print(f"  ckpt:   {args.ckpt}")
    print(f"  device: {device}")
    print(f"  tol:    {args.tol}")
    print()

    buf = AWRReplayBuffer(args.memmap, rng_seed=42)
    model, cfg, _ = load_checkpoint(args.ckpt, device)
    model.eval()  # eliminate dropout-induced variance

    batch_bc = buf.sample(args.batch_size, device)
    print(f"sampled batch: B={batch_bc['mask'].shape[0]}, T={batch_bc['mask'].shape[1]}, "
          f"n_valid={int(batch_bc['mask'].sum().item())}")

    with torch.no_grad():
        # Path 1: forward_sequence (BC path, what AWR currently uses)
        out_bc = model.forward_sequence(batch_bc, device)
        print(f"forward_sequence outputs:")
        print(f"  action_logits: {tuple(out_bc['action_logits'].shape)}, dtype={out_bc['action_logits'].dtype}")
        print(f"  value:         {tuple(out_bc['value'].shape)}, dtype={out_bc['value'].dtype}")
        print(f"  v_logits:      {tuple(out_bc['v_logits'].shape)}, dtype={out_bc['v_logits'].dtype}")

        # Path 2: forward_ppo_sequence (Tier 3 / production PPO path)
        batch_ppo = bc_to_ppo_format(batch_bc)
        out_ppo = model.forward_ppo_sequence(batch_ppo, device)
        print(f"forward_ppo_sequence outputs:")
        print(f"  action_logits: {tuple(out_ppo['action_logits'].shape)}")
        print(f"  value:         {tuple(out_ppo['value'].shape)}")
        print(f"  v_logits:      {tuple(out_ppo['v_logits'].shape)}")

    # Compare at valid positions only (padding positions are different fillers
    # in the two paths and unrelated to model output).
    mask = batch_bc["mask"].bool()  # (B, T)
    n_valid_pos = int(mask.sum().item())

    print()
    print(f"=== compare at {n_valid_pos} valid positions ===")

    for key in ["action_logits", "value", "v_logits"]:
        a = out_bc[key].float()
        b = out_ppo[key].float()
        # Slice to valid positions (works for both 2D and 3D outputs)
        if a.dim() == 3:
            a_valid = a[mask]
            b_valid = b[mask]
        else:
            a_valid = a[mask]
            b_valid = b[mask]
        diff = (a_valid - b_valid).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        pass_str = "PASS" if max_diff < args.tol else "FAIL"
        print(f"  {key:14s}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e} [{pass_str}]")

    # Headline verdict
    print()
    a_max = (out_bc["action_logits"][mask] - out_ppo["action_logits"][mask]).abs().max().item()
    v_max = (out_bc["value"][mask] - out_ppo["value"][mask]).abs().max().item()
    if a_max < args.tol and v_max < args.tol:
        print(f"VERDICT: PATHS EQUIVALENT (max action_logits diff {a_max:.2e}, "
              f"max value diff {v_max:.2e} < tol {args.tol}).")
        print("        AWR using forward_sequence is mathematically safe vs PPO's forward_ppo_sequence.")
        sys.exit(0)
    else:
        print(f"VERDICT: PATHS DIVERGE (action_logits max {a_max:.2e}, value max {v_max:.2e} > tol {args.tol}).")
        print("        AWR MUST switch to forward_ppo_sequence to match PPO training distribution.")
        sys.exit(2)


if __name__ == "__main__":
    main()
