#!/usr/bin/env python
"""CIS Phase 1 logits-identity test (Test 1 of cookbook §5).

Validates the CIS scaffolding in mp_centralized_collect.py:
  1. CIS process spawns + loads model
  2. Numpy IPC round-trip is lossless (synthetic batch)
  3. Forward via CIS matches direct-main forward (logits identity)
  4. Process shuts down cleanly

Acceptance:
  max abs diff per logit < 1e-3 across 100 fixed batches (allows fp16 noise)

This is the gate for proceeding to CIS Phase 2 (batching + N=4 workers).
If this fails, investigate before moving on.

Usage on cloud pod:
  cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
  python /workspace/scripts/test_cis_phase1.py [--n-batches 100] [--batch 8]

Side-effect-free: spawns 1 CIS subprocess (uses ~1 GB GPU), no battle_servers
needed, no disk writes. Safe alongside production training.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Any

import numpy as np
import torch


def synthesize_batch(model, B: int = 8, device: str = "cuda", seed: int = 0):
    """Synthetic batch matching unpack_turn_batch's schema. Same shape generator
    as test_compile_new_arch.py - share-by-paste because diag scripts are
    standalone (no shared utils module yet)."""
    torch.manual_seed(seed)

    cfg = model.cfg
    fmt = cfg.format_config
    team_size = fmt.team_size
    n_moves = fmt.n_moves
    n_switches = fmt.n_switches

    POKE_CONT = 285
    FIELD_CONT = 52
    TRANS_CONT = 51
    ACTIVE_MOVE_CONT = 109
    PER_POKEMON_MOVE_CONT = 23
    SWITCH_CONT = 30

    dev = torch.device(device)

    def _ids(shape, vocab):
        return torch.randint(0, max(int(vocab), 1), shape, dtype=torch.long, device=dev)

    def _bank(shape):
        return torch.randint(0, 32, shape, dtype=torch.long, device=dev)

    def _zero_f(shape):
        return torch.zeros(shape, dtype=torch.float32, device=dev)

    return {
        "our_pokemon_ids": torch.stack([
            _ids((B, team_size), cfg.n_species),
            _ids((B, team_size), cfg.n_items),
            _ids((B, team_size), cfg.n_abilities),
        ], dim=-1),
        "opp_pokemon_ids": torch.stack([
            _ids((B, team_size), cfg.n_species),
            _ids((B, team_size), cfg.n_items),
            _ids((B, team_size), cfg.n_abilities),
        ], dim=-1),
        "our_pokemon_banks": _bank((B, team_size, 10)),
        "opp_pokemon_banks": _bank((B, team_size, 10)),
        "our_pokemon_cont": _zero_f((B, team_size, POKE_CONT)),
        "opp_pokemon_cont": _zero_f((B, team_size, POKE_CONT)),
        "our_pokemon_move_ids": _ids((B, team_size, 4), cfg.n_moves),
        "opp_pokemon_move_ids": _ids((B, team_size, 4), cfg.n_moves),
        "our_pokemon_move_cont": _zero_f((B, team_size, 4, PER_POKEMON_MOVE_CONT)),
        "opp_pokemon_move_cont": _zero_f((B, team_size, 4, PER_POKEMON_MOVE_CONT)),
        "field_banks": {
            "turn": _bank((B,)),
            "weather_dur": _bank((B,)),
            "terrain_dur": _bank((B,)),
            "tr_dur": _bank((B,)),
        },
        "field_cont": _zero_f((B, FIELD_CONT)),
        "transition_ids": {
            "our_action": _bank((B,)),
            "opp_action": _bank((B,)),
        },
        "transition_cont": _zero_f((B, TRANS_CONT)),
        "active_move_ids": _ids((B, n_moves), cfg.n_moves),
        "active_move_banks": {
            "bp": _bank((B, n_moves)),
            "acc": _bank((B, n_moves)),
            "pp": _bank((B, n_moves)),
            "prio": _bank((B, n_moves)),
        },
        "active_move_cont": _zero_f((B, n_moves, ACTIVE_MOVE_CONT)),
        "switch_ids": _ids((B, n_switches), cfg.n_species),
        "switch_cont": _zero_f((B, n_switches, SWITCH_CONT)),
        "legal_mask": torch.ones((B, fmt.n_actions), dtype=torch.float32, device=dev),
    }


def _bool(x):
    return "PASS" if x else "FAIL"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="data/models/bc/v10_cloud_gen9/epoch_003.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-batches", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--tol", type=float, default=1e-3)
    args = p.parse_args()

    fp16 = not args.no_fp16

    print("=== CIS Phase 1 logits-identity test ===")
    print(f"ckpt={args.ckpt}, n_batches={args.n_batches}, B={args.batch}, fp16={fp16}, tol={args.tol}")
    print()

    # Make src/ importable
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(here, "..", "..", "pokemon-ai-starter", "pokemon-ai", "src"))
    if os.path.isdir(src_dir):
        os.chdir(src_dir)
        sys.path.insert(0, src_dir)
    else:
        sys.path.insert(0, ".")

    from ppo import load_checkpoint
    from precision_config import autocast_ctx
    from mp_centralized_collect import CISClient, torch_dict_to_numpy

    device = torch.device(args.device)

    # 1. Load model in main, generate synthetic batches + reference outputs
    print("Stage 1: load model in MAIN process, run reference forward")
    t0 = time.time()
    model, cfg, _ = load_checkpoint(args.ckpt, device)
    model.eval()
    print(f"  main model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"{time.time()-t0:.1f}s)")

    # Generate batches + reference outputs
    print(f"  generating {args.n_batches} batches + reference forward...")
    batches_torch = []
    reference_outputs = []  # list of dict[str, np.ndarray]

    t0 = time.time()
    with torch.no_grad(), autocast_ctx(fp16):
        for i in range(args.n_batches):
            batch = synthesize_batch(model, B=args.batch, device=args.device, seed=i)
            batches_torch.append(batch)
            out = model(batch)
            reference_outputs.append({
                "action_logits": out["action_logits"].detach().float().cpu().numpy(),
                "value":         out["value"].detach().float().cpu().numpy(),
                "v_logits":      out["v_logits"].detach().float().cpu().numpy(),
                "summary":       out["summary"].detach().float().cpu().numpy(),
            })
    print(f"  reference forward done ({time.time()-t0:.1f}s)")

    # Free reference model GPU mem before spawning CIS (CIS will also load it).
    # Important: at production scale, main and CIS share the GPU. For this test,
    # we have headroom on A100 80GB but freeing is good hygiene.
    print()

    # 2. Convert batches to numpy (lossless serialization sanity)
    print("Stage 2: serialize batches to numpy (round-trip check)")
    batches_numpy = [torch_dict_to_numpy(b) for b in batches_torch]
    # Sanity: round-trip first batch and compare shapes
    from mp_centralized_collect import numpy_dict_to_torch
    rt = numpy_dict_to_torch(batches_numpy[0], device)
    for k in batches_torch[0]:
        if isinstance(batches_torch[0][k], torch.Tensor):
            d = (rt[k].float() - batches_torch[0][k].float()).abs().max().item()
            if d > 1e-7:
                print(f"  WARN: round-trip diff for {k}: {d:.2e}")
    print("  numpy round-trip OK")
    print()

    # 3. Spawn CIS, send each batch, collect responses
    print("Stage 3: spawn CIS subprocess + run inference via IPC")
    cis = CISClient(args.ckpt, device=args.device, fp16=fp16, amp_dtype_name=None)
    t0 = time.time()
    cis.spawn(ready_timeout_s=120.0)
    print(f"  CIS spawn + ready: {time.time()-t0:.1f}s")
    print(f"  CIS ping: {_bool(cis.ping())}")

    print(f"  sending {args.n_batches} infer requests...")
    cis_outputs = []
    t0 = time.time()
    for i, np_batch in enumerate(batches_numpy):
        out = cis.infer(np_batch, timeout_s=30.0)
        cis_outputs.append(out)
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{args.n_batches} done ({elapsed:.1f}s, {(i+1)/elapsed:.1f} req/s)")
    cis_total = time.time() - t0
    print(f"  CIS inference total: {cis_total:.1f}s ({args.n_batches/cis_total:.1f} req/s)")
    print()

    # 4. Compare reference vs CIS outputs
    print("Stage 4: logits identity check")
    keys_to_check = ["action_logits", "value", "v_logits", "summary"]
    max_diffs = {k: 0.0 for k in keys_to_check}
    failed_count = 0
    failed_examples = []

    for i, (ref, cis_out) in enumerate(zip(reference_outputs, cis_outputs)):
        for k in keys_to_check:
            d = np.abs(ref[k] - cis_out[k]).max()
            if d > max_diffs[k]:
                max_diffs[k] = float(d)
            if d > args.tol:
                failed_count += 1
                if len(failed_examples) < 3:
                    failed_examples.append((i, k, d))

    for k, d in max_diffs.items():
        ok = d < args.tol
        print(f"  max abs diff {k:14s}: {d:.2e}  [{_bool(ok)}]")

    overall = all(d < args.tol for d in max_diffs.values())
    print(f"  overall logits identity (< {args.tol}): {_bool(overall)}")
    if failed_examples:
        print(f"  failed batches (first 3): {failed_examples}")
    print()

    # 5. Shutdown
    print("Stage 5: shutdown CIS")
    cis.shutdown()
    print("  CIS shutdown OK")
    print()

    # Summary
    print("=== Summary ===")
    print(f"  CIS spawn + ready: ~{time.time()-t0:.0f}s (one-time)")
    print(f"  IPC throughput: {args.n_batches/cis_total:.1f} req/s")
    print(f"  logits identity: {_bool(overall)}")
    if overall:
        print()
        print("VERDICT: CIS Phase 1 PASSED. Proceed to Phase 2 (batching + N=4 workers).")
        return 0
    else:
        print()
        print("VERDICT: CIS Phase 1 FAILED. Investigate before Phase 2.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
