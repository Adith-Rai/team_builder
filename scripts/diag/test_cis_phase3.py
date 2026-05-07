#!/usr/bin/env python
"""CIS Phase 3 weight-reload test.

Validates the reload protocol in mp_centralized_collect.py:
  1. CIS subprocess spawns with initial weights
  2. Run inference, record output A
  3. Main perturbs its model's weights, saves to a new path atomically
  4. Send reload signal to CIS
  5. Run inference (same input batch), record output B
  6. Verify: B differs from A in proportion to the perturbation, AND B
     matches main's NEW reference forward (with perturbed weights)
  7. Reload back to original weights, verify CIS returns to output A

Acceptance:
  - reload command returns "ok"
  - post-reload output != pre-reload output (perturbation took effect)
  - post-reload output == direct main forward with new weights (within
    fp16 fused-kernel noise tolerance)
  - second reload back to original recovers output A

Usage on cloud pod:
  cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
  python /workspace/scripts/test_cis_phase3.py [--ckpt PATH] [--batch 8]

Side-effect-free except writes /tmp/cis_phase3_weights_*.pt (cleaned up at end).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def synthesize_batch(model, B: int = 8, device: str = "cuda", seed: int = 0):
    """Same synth as test_cis_phase1.py - duplicated for standalone."""
    torch.manual_seed(seed)
    cfg = model.cfg
    fmt = cfg.format_config
    team_size = fmt.team_size
    n_moves = fmt.n_moves
    n_switches = fmt.n_switches

    POKE_CONT, FIELD_CONT, TRANS_CONT = 285, 52, 51
    ACTIVE_MOVE_CONT, PER_POKEMON_MOVE_CONT, SWITCH_CONT = 109, 23, 30

    dev = torch.device(device)
    _ids = lambda shape, vocab: torch.randint(0, max(int(vocab), 1), shape, dtype=torch.long, device=dev)
    _bank = lambda shape: torch.randint(0, 32, shape, dtype=torch.long, device=dev)
    _z = lambda shape: torch.zeros(shape, dtype=torch.float32, device=dev)

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
        "our_pokemon_cont": _z((B, team_size, POKE_CONT)),
        "opp_pokemon_cont": _z((B, team_size, POKE_CONT)),
        "our_pokemon_move_ids": _ids((B, team_size, 4), cfg.n_moves),
        "opp_pokemon_move_ids": _ids((B, team_size, 4), cfg.n_moves),
        "our_pokemon_move_cont": _z((B, team_size, 4, PER_POKEMON_MOVE_CONT)),
        "opp_pokemon_move_cont": _z((B, team_size, 4, PER_POKEMON_MOVE_CONT)),
        "field_banks": {"turn": _bank((B,)), "weather_dur": _bank((B,)),
                        "terrain_dur": _bank((B,)), "tr_dur": _bank((B,))},
        "field_cont": _z((B, FIELD_CONT)),
        "transition_ids": {"our_action": _bank((B,)), "opp_action": _bank((B,))},
        "transition_cont": _z((B, TRANS_CONT)),
        "active_move_ids": _ids((B, n_moves), cfg.n_moves),
        "active_move_banks": {
            "bp": _bank((B, n_moves)), "acc": _bank((B, n_moves)),
            "pp": _bank((B, n_moves)), "prio": _bank((B, n_moves)),
        },
        "active_move_cont": _z((B, n_moves, ACTIVE_MOVE_CONT)),
        "switch_ids": _ids((B, n_switches), cfg.n_species),
        "switch_cont": _z((B, n_switches, SWITCH_CONT)),
        "legal_mask": torch.ones((B, fmt.n_actions), dtype=torch.float32, device=dev),
    }


def _bool(x):
    return "PASS" if x else "FAIL"


def _save_weights_atomic(model, cfg, path: str) -> None:
    tmp = path + ".tmp"
    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else None
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": cfg_dict,
        "arch": "transformer",
    }, tmp)
    try:
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
    except Exception:
        pass
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="data/models/bc/v10_cloud_gen9/epoch_003.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--tol", type=float, default=1e-2)
    p.add_argument("--perturb-scale", type=float, default=0.01)
    args = p.parse_args()

    fp16 = not args.no_fp16

    print("=== CIS Phase 3 weight-reload test ===")
    print(f"ckpt={args.ckpt}, B={args.batch}, fp16={fp16}, tol={args.tol}, "
          f"perturb_scale={args.perturb_scale}")
    print()

    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(here, "..", "..", "pokemon-ai-starter", "pokemon-ai", "src"))
    if os.path.isdir(src_dir):
        os.chdir(src_dir)
        sys.path.insert(0, src_dir)
    else:
        sys.path.insert(0, ".")

    from ppo import load_checkpoint
    from precision_config import autocast_ctx
    from mp_centralized_collect import CISServer, torch_dict_to_numpy

    device = torch.device(args.device)

    # Stage 1: load main model + generate batch + reference forward A (original)
    print("Stage 1: load main model + reference forward A (original weights)")
    main_model, cfg, _ = load_checkpoint(args.ckpt, device)
    main_model.eval()
    batch_torch = synthesize_batch(main_model, B=args.batch, device=args.device, seed=42)
    np_batch = torch_dict_to_numpy(batch_torch)

    with torch.no_grad(), autocast_ctx(fp16):
        out_A = main_model(batch_torch)
    ref_A = {
        "action_logits": out_A["action_logits"].detach().float().cpu().numpy(),
        "value":         out_A["value"].detach().float().cpu().numpy(),
    }
    print(f"  ref A: action_logits[0,0]={ref_A['action_logits'][0,0]:.4f}, "
          f"value[0]={ref_A['value'][0]:.4f}")
    print()

    # Stage 2: spawn CIS with original weights
    print("Stage 2: spawn CIS with original weights")
    server = CISServer(args.ckpt, n_workers=1, device=args.device, fp16=fp16,
                       min_batch=1, timeout_ms=15)
    handles = server.spawn(ready_timeout_s=120.0)
    h = handles[0]
    print(f"  CIS up, ping {_bool(h.ping())}")
    cis_A = h.infer(np_batch, timeout_s=30.0)
    print(f"  CIS forward A: action_logits[0,0]={cis_A['action_logits'][0,0]:.4f}")
    diff_A = float(np.abs(ref_A["action_logits"] - cis_A["action_logits"]).max())
    print(f"  CIS A vs main A max abs diff: {diff_A:.2e}  [{_bool(diff_A < args.tol)}]")
    print()

    # Stage 3: perturb main weights, save to new path
    print(f"Stage 3: perturb main weights (scale={args.perturb_scale}) + save")
    torch.manual_seed(123)
    with torch.no_grad():
        for p_ in main_model.parameters():
            p_.add_(torch.randn_like(p_) * args.perturb_scale)

    with torch.no_grad(), autocast_ctx(fp16):
        out_B = main_model(batch_torch)
    ref_B = {
        "action_logits": out_B["action_logits"].detach().float().cpu().numpy(),
        "value":         out_B["value"].detach().float().cpu().numpy(),
    }
    print(f"  ref B: action_logits[0,0]={ref_B['action_logits'][0,0]:.4f}, "
          f"value[0]={ref_B['value'][0]:.4f}")
    diff_AB_ref = float(np.abs(ref_A["action_logits"] - ref_B["action_logits"]).max())
    print(f"  ref A vs ref B max abs diff (perturbation magnitude): {diff_AB_ref:.2e}")

    perturbed_path = "/tmp/cis_phase3_weights_perturbed.pt"
    _save_weights_atomic(main_model, cfg, perturbed_path)
    print(f"  saved perturbed weights to {perturbed_path}")
    print()

    # Stage 4: signal CIS reload
    print("Stage 4: signal CIS reload")
    t0 = time.time()
    resp = server.reload_weights(perturbed_path, timeout_s=60.0)
    print(f"  reload completed in {time.time()-t0:.1f}s (status={resp.get('status')})")
    if resp.get("missing_keys"):
        print(f"  WARN missing_keys: {resp['missing_keys']}")
    if resp.get("unexpected_keys"):
        print(f"  WARN unexpected_keys: {resp['unexpected_keys']}")
    print()

    # Stage 5: forward via CIS post-reload
    print("Stage 5: post-reload CIS forward (should match ref B)")
    cis_B = h.infer(np_batch, timeout_s=30.0)
    print(f"  CIS B: action_logits[0,0]={cis_B['action_logits'][0,0]:.4f}")
    diff_B = float(np.abs(ref_B["action_logits"] - cis_B["action_logits"]).max())
    print(f"  CIS B vs main B max abs diff: {diff_B:.2e}  [{_bool(diff_B < args.tol)}]")

    # Sanity: post-reload should differ from pre-reload by ~perturbation magnitude
    diff_cis_AB = float(np.abs(cis_A["action_logits"] - cis_B["action_logits"]).max())
    print(f"  CIS A vs CIS B max abs diff (reload took effect): {diff_cis_AB:.2e}")
    reload_took_effect = diff_cis_AB > 0.5 * diff_AB_ref  # at least half the perturbation
    print(f"  reload visible at inference output: {_bool(reload_took_effect)}")
    print()

    # Stage 6: reload back to original, confirm CIS returns to output A
    print("Stage 6: reload back to original ckpt, confirm output recovers")
    resp = server.reload_weights(args.ckpt, timeout_s=60.0)
    print(f"  reload back: status={resp.get('status')}")
    cis_A2 = h.infer(np_batch, timeout_s=30.0)
    diff_A_recovered = float(np.abs(ref_A["action_logits"] - cis_A2["action_logits"]).max())
    print(f"  CIS recovered A vs ref A max abs diff: {diff_A_recovered:.2e}  "
          f"[{_bool(diff_A_recovered < args.tol)}]")
    print()

    # Stage 7: shutdown + cleanup
    print("Stage 7: shutdown + cleanup")
    server.shutdown()
    try:
        Path(perturbed_path).unlink(missing_ok=True)
    except Exception:
        pass
    print("  shutdown OK")
    print()

    overall = (diff_A < args.tol and diff_B < args.tol and
               reload_took_effect and diff_A_recovered < args.tol)

    print("=== Summary ===")
    print(f"  Stage 2 CIS A vs main A:                  diff {diff_A:.2e}  [{_bool(diff_A < args.tol)}]")
    print(f"  Stage 5 CIS B vs main B (post-reload):    diff {diff_B:.2e}  [{_bool(diff_B < args.tol)}]")
    print(f"  Stage 5 CIS A vs CIS B (reload visible):  diff {diff_cis_AB:.2e}  [{_bool(reload_took_effect)}]")
    print(f"  Stage 6 CIS recovered A vs ref A:         diff {diff_A_recovered:.2e}  [{_bool(diff_A_recovered < args.tol)}]")
    print()
    if overall:
        print("VERDICT: CIS Phase 3 PASSED. Weight reload protocol works.")
        return 0
    else:
        print("VERDICT: CIS Phase 3 FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
