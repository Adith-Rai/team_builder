"""bench_bc_step.py — measure per-batch wall-clock for the BC training step.

Standalone, no DataLoader. Loads N episodes once, then runs M consecutive
forward_sequence + backward + AdamW.step() iterations on CUDA fp16. Used to
diagnose Session 48 BC throughput without the DataLoader / workers stack.

Usage:
    python bench_bc_step.py --batch-size 8 --n-batches 5 --device cuda --fp16
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import MemmapDataset, collate_seq
from model_transformer import (
    TransformerBattlePolicy, TransformerConfig, load_move_flag_lookup,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--memmap-dir", default="data/datasets/human_v8_100k")
    p.add_argument("--lookup-path", default="data/lookup/move_flags_v1.pt")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-batches", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[bench] device={device} fp16={args.fp16} B={args.batch_size} n_batches={args.n_batches}")

    print("[bench] Loading dataset metadata + sampling 1 batch...")
    t0 = time.time()
    ds = MemmapDataset(args.memmap_dir, split="train", val_ratio=0.05)
    print(f"  episodes: {len(ds)}  ({time.time()-t0:.1f}s)")

    # Pre-load N batches' worth of episodes synchronously.
    print(f"[bench] Materializing {args.n_batches} batches × {args.batch_size} episodes...")
    t0 = time.time()
    batches = []
    idx = 0
    for _ in range(args.n_batches):
        ep_list = [ds[i] for i in range(idx, idx + args.batch_size)]
        idx += args.batch_size
        batches.append(collate_seq(ep_list))
    t_load = time.time() - t0
    print(f"  loaded {args.n_batches} batches in {t_load:.1f}s ({t_load/args.n_batches:.2f}s/batch)")

    # Report turn statistics on this slice.
    total_turns = 0
    max_T = 0
    for b in batches:
        total_turns += int(b["seq_lens"].sum().item())
        max_T = max(max_T, int(b["seq_lens"].max().item()))
    print(f"  total valid turns: {total_turns}  max_T: {max_T}  avg_T: {total_turns/(args.n_batches*args.batch_size):.1f}")

    # Build model.
    print("[bench] Building TransformerBattlePolicy...")
    t0 = time.time()
    cfg = TransformerConfig.with_vocab_sizes_from_disk()
    lookup = load_move_flag_lookup(Path(args.lookup_path), expected_n_moves=cfg.n_moves)
    model = TransformerBattlePolicy(cfg, move_flag_lookup=lookup).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if args.fp16 and device.type == "cuda" else None
    print(f"  params: {model.count_parameters():,}  ({time.time()-t0:.1f}s)")

    print("[bench] Step timings (forward_sequence + loss + backward + opt.step)...")
    for i, collated in enumerate(batches):
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()

        mask = collated["mask"].to(device)
        actions = collated["action"].to(device)
        results = collated["result"].to(device)

        with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
            out = model.forward_sequence(collated, device)
            logits = out["action_logits"]
            v_logits = out["v_logits"]

            valid = (mask > 0.5) & (actions >= 0)
            flat_logits = logits[valid].float()
            flat_actions = actions[valid]
            flat_legal = collated["legal_mask_raw"].to(device)[valid]
            masked = flat_logits.masked_fill(flat_legal < 0.5, -100.0)
            pi_loss = F.cross_entropy(masked, flat_actions)

            v_valid = results[valid] >= 0
            if v_valid.any():
                vl = v_logits[valid][v_valid].float()
                vt = results[valid][v_valid]
                twohot = model.twohot_target(vt)
                v_loss = F.cross_entropy(vl, twohot) * 0.5
            else:
                v_loss = torch.tensor(0.0, device=device)
            loss = pi_loss + v_loss

        opt.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        torch.cuda.synchronize() if device.type == "cuda" else None
        dt = time.time() - t0
        n_valid = int(valid.sum().item())
        mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
        print(f"  [{i}] dt={dt*1000:.0f}ms  pi={pi_loss.item():.3f}  v={v_loss.item():.3f}  "
              f"valid_turns={n_valid}  per_turn={dt*1000/max(1,n_valid):.1f}ms  "
              f"peak_mem={mem:.2f}GB", flush=True)

    print("[bench] done.")


if __name__ == "__main__":
    main()
