#!/usr/bin/env python
"""Pad BC v10 ckpt to current (post-D1+D2) arch for use as warm init in
CIS dev tests where battle quality matters (e.g., wall-time A/B at prod
scale where random init causes turn-cap forfeits and inflated collect
time).

Strategy:
- Load BC v10 state_dict (14-token N_BATTLE_STATE, type_id=28, pokemon_slot=24,
  no gen_embed).
- Construct a fresh post-D1+D2 model with current arch (15-token, type_id=29,
  pokemon_slot=25, with gen_embed.weight).
- Copy BC v10 keys that exist in both. For shape mismatches, zero-pad the
  missing rows (so new rows contribute nothing initially → identity to BC v10
  behavior in gen 9 only).
- Leave gen_embed.weight at the fresh model's init (it will get zero-padded
  to act as "no gen info" baseline if we explicitly zero it).
- Save as a new ckpt usable by load_checkpoint at HEAD.

Usage:
  cd pokemon-ai-starter/pokemon-ai/src
  python scripts/diag/pad_bc_v10_to_d1d2_arch.py \\
      --src data/models/bc/v10_cloud_gen9/epoch_003.pt \\
      --dst data/models/bc/v10_padded_for_cis_dev.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True,
                   help="BC v10 ckpt path (pre-D1+D2 arch)")
    p.add_argument("--dst", required=True,
                   help="Output: padded ckpt loadable by HEAD's load_checkpoint")
    p.add_argument("--zero-gen-embed", action="store_true", default=True,
                   help="Force gen_embed.weight to zeros (default: True). "
                        "Zero gen_embed = 'no gen info' baseline, equivalent "
                        "to BC v10 behavior since BC v10 didn't see gen ids.")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(here, "..", "..",
                                           "pokemon-ai-starter", "pokemon-ai", "src"))
    if os.path.isdir(src_dir):
        os.chdir(src_dir)
        sys.path.insert(0, src_dir)
    else:
        sys.path.insert(0, ".")

    from model_transformer import (
        TransformerBattlePolicy, TransformerConfig, load_move_flag_lookup,
    )

    print(f"=== BC v10 → current arch padding ===")
    print(f"src: {args.src}")
    print(f"dst: {args.dst}")
    print()

    # 1. Load source ckpt + strip _orig_mod. prefix (BC v10 was saved while
    # torch.compile-wrapped; the wrapped state_dict has an extra ._orig_mod.
    # segment that the unwrapped model doesn't know about. ppo.py:load_checkpoint
    # does the same strip at line 406).
    src_ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    src_sd_raw = src_ckpt["model_state_dict"]
    src_sd = {k.replace("._orig_mod.", "."): v for k, v in src_sd_raw.items()}
    n_stripped = sum(1 for k in src_sd_raw if "_orig_mod." in k)
    print(f"Source state_dict: {len(src_sd)} keys "
          f"({n_stripped} had _orig_mod. prefix stripped)")

    # 2. Construct fresh current-arch model
    cfg = TransformerConfig.from_dict({})
    lookup = load_move_flag_lookup(
        Path("data/lookup/move_flags_v1.pt"), expected_n_moves=cfg.n_moves,
    )
    model = TransformerBattlePolicy(cfg, move_flag_lookup=lookup)
    fresh_sd = model.state_dict()
    print(f"Fresh model state_dict: {len(fresh_sd)} keys, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print()

    # 3. Build padded state_dict: start with fresh, overwrite from src where
    # shapes match; zero-pad rows where shapes differ in dim 0; warn on others.
    padded_sd = {}
    n_copied = n_padded = n_dropped = n_kept_fresh = 0
    n_shape_warned = 0

    for key, fresh_v in fresh_sd.items():
        if key not in src_sd:
            # New key (e.g., gen_embed.weight). Zero-fill if requested.
            if args.zero_gen_embed and "gen_embed" in key:
                padded_sd[key] = torch.zeros_like(fresh_v)
                n_padded += 1
                print(f"  PAD (zero) {key}: shape={tuple(fresh_v.shape)} "
                      f"(new in current arch)")
            else:
                padded_sd[key] = fresh_v.clone()
                n_kept_fresh += 1
                print(f"  KEEP fresh {key}: shape={tuple(fresh_v.shape)} "
                      f"(new key, kept fresh init)")
            continue

        src_v = src_sd[key]
        if src_v.shape == fresh_v.shape:
            # Exact match, copy.
            padded_sd[key] = src_v.clone()
            n_copied += 1
            continue

        # Shape mismatch. Try to pad along dim 0 (typical for embedding tables).
        if src_v.dim() == fresh_v.dim() and src_v.shape[1:] == fresh_v.shape[1:]:
            # Same shape after dim 0; pad new rows with zeros.
            extra_rows = fresh_v.shape[0] - src_v.shape[0]
            if extra_rows < 0:
                # Source has MORE rows than fresh — drop extras (rare, would be
                # arch shrinking which we don't expect).
                print(f"  WARN {key}: src has {src_v.shape[0]} rows, "
                      f"fresh has {fresh_v.shape[0]}; truncating src")
                padded_sd[key] = src_v[:fresh_v.shape[0]].clone()
                n_dropped += 1
                continue
            pad = torch.zeros((extra_rows,) + tuple(src_v.shape[1:]),
                              dtype=src_v.dtype)
            padded_sd[key] = torch.cat([src_v, pad], dim=0)
            n_padded += 1
            print(f"  PAD (rows) {key}: src {tuple(src_v.shape)} → "
                  f"target {tuple(fresh_v.shape)} (+{extra_rows} zero rows)")
        else:
            # Non-trivial shape mismatch. Drop and warn loudly.
            print(f"  WARN DROP {key}: src {tuple(src_v.shape)} vs "
                  f"fresh {tuple(fresh_v.shape)} — incompatible, using fresh init")
            padded_sd[key] = fresh_v.clone()
            n_shape_warned += 1

    print()
    print(f"Summary:")
    print(f"  copied (shape match):  {n_copied}")
    print(f"  padded (zero rows):    {n_padded}")
    print(f"  dropped from src:      {n_dropped}")
    print(f"  warned (shape diff):   {n_shape_warned}")
    print(f"  kept fresh init:       {n_kept_fresh}")
    print()

    # 4. Sanity: try loading into the model
    missing, unexpected = model.load_state_dict(padded_sd, strict=False)
    if missing or unexpected:
        print(f"  WARN strict=False load:")
        for k in missing.missing_keys[:5] if hasattr(missing, 'missing_keys') else missing[:5]:
            print(f"    missing: {k}")
        for k in missing.unexpected_keys[:5] if hasattr(missing, 'unexpected_keys') else unexpected[:5]:
            print(f"    unexpected: {k}")
    else:
        print(f"  load_state_dict strict=False: clean (no missing/unexpected)")

    # 5. Save with same shape as load_checkpoint expects
    out_ckpt = {
        "arch": "transformer",
        "model_state_dict": padded_sd,
        "model_config": cfg.to_dict(),
        "optimizer_state_dict": {},
        "iteration": 0,
        "metrics": {"note": f"BC v10 padded to current arch from {args.src}"},
        "v8_version": "8.0",
    }
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    tmp = args.dst + ".tmp"
    torch.save(out_ckpt, tmp)
    os.replace(tmp, args.dst)
    print(f"\nSaved padded ckpt to {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
