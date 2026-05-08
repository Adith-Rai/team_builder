#!/usr/bin/env python
"""CIS Phase 4.3a slot-mechanism tests.

Three load-bearing correctness gates per project_cis_4_3_design.md.

Test 1 - Slot identity:
  Spawn CIS with K=2 slots [model_A, model_B]. For N fixed batches, send
  via slot 0 + slot 1. Verify slot 0 output ~= direct model_A(batch) and
  slot 1 output ~= direct model_B(batch). Acceptance: max diff < tol per slot.
  Catches: slot routing bug (request goes to wrong model).

Test 2 - Reload round-trip + slot isolation:
  Stage A: Spawn CIS with K=1 (slot 0 = A). Forward -> A1. Reload slot 0
  with B -> forward -> B1 (must differ from A1). Reload slot 0 back to A
  -> forward -> A2. Acceptance: A1 vs A2 = 0.0 bit-exact.
  Stage B: Spawn CIS with K=2 [A, B]. Forward slot 0 -> A_init. Reload
  slot 1 with C. Forward slot 0 -> A_after. Acceptance: A_init vs A_after
  = 0.0 bit-exact (reload of slot 1 must NOT perturb slot 0).
  Catches: half-loaded state bug + cross-slot reload contamination.

Test 3 - Batch partition (multi-worker):
  Spawn CIS with K=3 [A, B, C], n_workers=4. Each worker thread sends
  3 sequential infer requests (slots 0, 1, 2) using sync h.infer(..., slot=).
  Server cross-batches across workers within each slot when timing aligns.
  Each worker verifies its own responses match direct forward of the right
  model. Acceptance: max diff < tol across all 12 (4*3) requests.
  Catches: cross-worker slot routing bug, cross-worker batch dispatch bug,
  per-worker resp pipe contamination.

Usage on cloud pod:
  cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
  python /workspace/scripts/diag/test_cis_phase4_3_slots.py [--ckpt PATH]

Side-effect-free except writes /tmp/cis_phase43_weights_{B,C}.pt (cleaned at end).
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
    """Same synth as test_cis_phase{1,3}.py - duplicated for standalone."""
    torch.manual_seed(seed)
    cfg = model.cfg
    fmt = cfg.format_config
    team_size, n_moves, n_switches = fmt.team_size, fmt.n_moves, fmt.n_switches

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


def perturb_and_save(base_ckpt_path: str, out_path: str, scale: float, seed: int):
    """Load base ckpt on CPU, additively perturb floating-point weights,
    save atomically. Each call with a different seed produces an independent
    perturbed copy."""
    ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    torch.manual_seed(seed)
    for k in list(sd.keys()):
        v = sd[k]
        if torch.is_floating_point(v):
            sd[k] = v + torch.randn_like(v) * scale
    tmp = out_path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="data/models/bc/dummy_for_cis_dev.pt",
                   help="Base ckpt; B and C are perturbed copies")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-batches", type=int, default=20,
                   help="Test 1: batches per slot")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--perturb-scale", type=float, default=0.02)
    args = p.parse_args()

    fp16 = not args.no_fp16

    print("=== CIS Phase 4.3a slot-mechanism tests ===")
    print(f"ckpt={args.ckpt}, B={args.batch}, fp16={fp16}, tol={args.tol}, "
          f"perturb={args.perturb_scale}, n_batches={args.n_batches}")
    print()

    # Make src/ importable + chdir to src
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(here, "..", "..",
                                           "pokemon-ai-starter", "pokemon-ai", "src"))
    if os.path.isdir(src_dir):
        os.chdir(src_dir)
        sys.path.insert(0, src_dir)
    else:
        sys.path.insert(0, ".")

    from ppo import load_checkpoint
    from precision_config import autocast_ctx
    from mp_centralized_collect import CISServer, torch_dict_to_numpy

    device = torch.device(args.device)

    # ----- Setup: generate perturbed ckpts B and C from base A -----
    ckpt_A = args.ckpt
    ckpt_B = "/tmp/cis_phase43_weights_B.pt"
    ckpt_C = "/tmp/cis_phase43_weights_C.pt"
    print(f"Setup: generating perturbed ckpts B (seed=101) and C (seed=202)")
    perturb_and_save(ckpt_A, ckpt_B, args.perturb_scale, seed=101)
    perturb_and_save(ckpt_A, ckpt_C, args.perturb_scale, seed=202)
    print(f"  B={ckpt_B}, C={ckpt_C}")
    print()

    # Load all three on main for direct-forward references
    model_A, _, _ = load_checkpoint(ckpt_A, device); model_A.eval()
    model_B, _, _ = load_checkpoint(ckpt_B, device); model_B.eval()
    model_C, _, _ = load_checkpoint(ckpt_C, device); model_C.eval()
    print(f"  loaded A,B,C on main "
          f"({sum(p_.numel() for p_ in model_A.parameters())/1e6:.1f}M each)")
    print()

    results = {}

    # ===== TEST 1: SLOT IDENTITY =====
    print("=" * 60)
    print("TEST 1 — Slot identity (slot 0=A, slot 1=B, N batches)")
    print("=" * 60)
    server = CISServer([ckpt_A, ckpt_B], n_workers=1, device=args.device,
                       fp16=fp16, min_batch=1, timeout_ms=15)
    handles = server.spawn(ready_timeout_s=120.0)
    h = handles[0]

    max_diff_slot0 = 0.0
    max_diff_slot1 = 0.0
    n_pass = 0
    for i in range(args.n_batches):
        batch_torch = synthesize_batch(model_A, B=args.batch,
                                       device=args.device, seed=10000 + i)
        np_batch = torch_dict_to_numpy(batch_torch)

        # Direct refs
        with torch.no_grad(), autocast_ctx(fp16):
            ref0 = model_A(batch_torch)
            ref1 = model_B(batch_torch)
        ref0_np = ref0["action_logits"].detach().float().cpu().numpy()
        ref1_np = ref1["action_logits"].detach().float().cpu().numpy()

        # Via CIS
        out0 = h.infer(np_batch, slot=0, timeout_s=30.0)
        out1 = h.infer(np_batch, slot=1, timeout_s=30.0)

        d0 = float(np.abs(ref0_np - out0["action_logits"]).max())
        d1 = float(np.abs(ref1_np - out1["action_logits"]).max())
        max_diff_slot0 = max(max_diff_slot0, d0)
        max_diff_slot1 = max(max_diff_slot1, d1)
        if d0 < args.tol and d1 < args.tol:
            n_pass += 1

        # Sanity: slot 0 output should differ noticeably from slot 1 output
        # (else routing might be silently broken even if A==B in this batch)
        ab_diff = float(np.abs(out0["action_logits"]
                               - out1["action_logits"]).max())
        if i == 0:
            print(f"  batch 0: slot0 vs slot1 output diff = {ab_diff:.3f} "
                  f"(should be non-trivial; >0.01 means routing visible)")

    print(f"  slot 0 max abs diff vs model_A: {max_diff_slot0:.2e}  "
          f"[{_bool(max_diff_slot0 < args.tol)}]")
    print(f"  slot 1 max abs diff vs model_B: {max_diff_slot1:.2e}  "
          f"[{_bool(max_diff_slot1 < args.tol)}]")
    print(f"  passed batches: {n_pass}/{args.n_batches}")
    server.shutdown()
    test1_pass = (max_diff_slot0 < args.tol and max_diff_slot1 < args.tol
                  and n_pass == args.n_batches)
    results["test_1_slot_identity"] = test1_pass
    print(f"  TEST 1: {_bool(test1_pass)}")
    print()

    # ===== TEST 2: RELOAD ROUND-TRIP + SLOT ISOLATION =====
    print("=" * 60)
    print("TEST 2 — Reload round-trip + slot isolation")
    print("=" * 60)
    # --- Stage A: round-trip on a single-slot server ---
    print("Stage A: K=1, slot 0: A -> B -> A bit-exact")
    server = CISServer(ckpt_A, n_workers=1, device=args.device,
                       fp16=fp16, min_batch=1, timeout_ms=15)
    handles = server.spawn(ready_timeout_s=120.0)
    h = handles[0]

    batch_torch = synthesize_batch(model_A, B=args.batch,
                                   device=args.device, seed=42)
    np_batch = torch_dict_to_numpy(batch_torch)

    out_A1 = h.infer(np_batch, slot=0, timeout_s=30.0)
    print(f"  A1: action_logits[0,0]={out_A1['action_logits'][0,0]:.4f}")
    r = h.reload(ckpt_B, slot=0, timeout_s=60.0)
    out_B1 = h.infer(np_batch, slot=0, timeout_s=30.0)
    diff_AB = float(np.abs(out_A1["action_logits"]
                           - out_B1["action_logits"]).max())
    print(f"  B1: action_logits[0,0]={out_B1['action_logits'][0,0]:.4f}")
    print(f"  A1 vs B1 diff (reload visible at infer): {diff_AB:.3f}  "
          f"[{_bool(diff_AB > 0.01)}]")
    h.reload(ckpt_A, slot=0, timeout_s=60.0)
    out_A2 = h.infer(np_batch, slot=0, timeout_s=30.0)
    diff_A1A2 = float(np.abs(out_A1["action_logits"]
                             - out_A2["action_logits"]).max())
    print(f"  A1 vs A2 (round-trip bit-exact): {diff_A1A2:.2e}  "
          f"[{_bool(diff_A1A2 < 1e-6)}]")
    server.shutdown()
    stageA_pass = (diff_AB > 0.01 and diff_A1A2 < 1e-6)
    print(f"  Stage A: {_bool(stageA_pass)}")

    # --- Stage B: slot isolation on K=2 server ---
    # Reload of slot 1 must NOT perturb slot 0. Catches a bug where reload
    # accidentally writes into the wrong slot's parameters.
    print("Stage B: K=2 [A,B]; reload slot 1 -> C; verify slot 0 unchanged")
    server = CISServer([ckpt_A, ckpt_B], n_workers=1, device=args.device,
                       fp16=fp16, min_batch=1, timeout_ms=15)
    handles = server.spawn(ready_timeout_s=120.0)
    h = handles[0]

    out_slot0_init = h.infer(np_batch, slot=0, timeout_s=30.0)
    out_slot1_init = h.infer(np_batch, slot=1, timeout_s=30.0)
    # Reload ONLY slot 1, with a fresh perturbation (C)
    r = h.reload(ckpt_C, slot=1, timeout_s=60.0)
    print(f"  reload slot 1 -> C: status={r.get('status')}, slot={r.get('slot')}")
    out_slot0_after = h.infer(np_batch, slot=0, timeout_s=30.0)
    out_slot1_after = h.infer(np_batch, slot=1, timeout_s=30.0)

    diff_slot0 = float(np.abs(out_slot0_init["action_logits"]
                              - out_slot0_after["action_logits"]).max())
    diff_slot1_changed = float(np.abs(out_slot1_init["action_logits"]
                                      - out_slot1_after["action_logits"]).max())
    print(f"  slot 0 init vs after-reload-slot1 (must be 0): {diff_slot0:.2e}  "
          f"[{_bool(diff_slot0 < 1e-6)}]")
    print(f"  slot 1 init vs after-reload-slot1 (must change): "
          f"{diff_slot1_changed:.3f}  [{_bool(diff_slot1_changed > 0.01)}]")
    server.shutdown()
    stageB_pass = (diff_slot0 < 1e-6 and diff_slot1_changed > 0.01)
    print(f"  Stage B: {_bool(stageB_pass)}")

    test2_pass = stageA_pass and stageB_pass
    results["test_2_reload_round_trip"] = test2_pass
    print(f"  TEST 2: {_bool(test2_pass)}")
    print()

    # ===== TEST 3: BATCH PARTITION (multi-worker) =====
    print("=" * 60)
    print("TEST 3 — Batch partition (K=3, N=4 workers, sequential mixed-slot)")
    print("=" * 60)
    # Each of 4 workers sends 3 sync infers across slots 0/1/2 sequentially.
    # Cross-worker batching naturally happens at the server when timing
    # aligns (e.g., 4 workers each on slot 0 simultaneously -> server batches
    # 4 reqs into a B>=min_batch fire). Sequential per-worker sync sends
    # avoid the pipe-buffer deadlock that burst-send hit.
    import threading

    n_threads = 4
    server = CISServer([ckpt_A, ckpt_B, ckpt_C], n_workers=n_threads,
                       device=args.device, fp16=fp16,
                       min_batch=2, timeout_ms=50)
    handles = server.spawn(ready_timeout_s=120.0)

    slot_seq = [0, 1, 2]  # each thread sends one req per slot
    slot_to_model = {0: model_A, 1: model_B, 2: model_C}

    # Pre-compute refs for each (thread_id, step) to compare against.
    refs_t3 = {}
    for tid in range(n_threads):
        for step, slot in enumerate(slot_seq):
            seed = 30000 + tid * 100 + step
            b = synthesize_batch(model_A, B=args.batch,
                                 device=args.device, seed=seed)
            np_b = torch_dict_to_numpy(b)
            with torch.no_grad(), autocast_ctx(fp16):
                o = slot_to_model[slot](b)
            refs_t3[(tid, step)] = {
                "np_batch": np_b,
                "ref": o["action_logits"].detach().float().cpu().numpy(),
                "slot": slot,
            }

    # Each thread runs through slot_seq sequentially, recording outputs.
    thread_results: Dict[int, Dict[int, Any]] = {tid: {} for tid in range(n_threads)}
    thread_errors: Dict[int, str] = {}
    barrier = threading.Barrier(n_threads)

    def _worker_fn(tid: int):
        try:
            h_t = handles[tid]
            for step, slot in enumerate(slot_seq):
                # Barrier-synced sends so all threads hit the same slot at
                # roughly the same time -> server can cross-batch them.
                barrier.wait(timeout=10.0)
                np_b = refs_t3[(tid, step)]["np_batch"]
                out = h_t.infer(np_b, slot=slot, timeout_s=30.0)
                thread_results[tid][step] = out
        except Exception as e:
            thread_errors[tid] = f"{type(e).__name__}: {e}"

    threads = [threading.Thread(target=_worker_fn, args=(tid,))
               for tid in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120.0)

    if thread_errors:
        for tid, err in thread_errors.items():
            print(f"  thread {tid} error: {err}")

    # Cross-worker batched fp16 forwards can drift up to ~1-2e-3 from
    # direct-forward refs due to fused-kernel non-determinism (different
    # batch sizes -> different reduction orders). 5e-3 is well above that
    # noise floor and still 100× tighter than any real routing bug
    # (which would manifest as ~0.1-0.5 diff from wrong-slot output).
    tol_t3 = max(args.tol, 5e-3)
    max_diff_per_slot = {0: 0.0, 1: 0.0, 2: 0.0}
    n_pass_total = 0
    n_total = 0
    for tid in range(n_threads):
        for step, slot in enumerate(slot_seq):
            n_total += 1
            out = thread_results[tid].get(step)
            if out is None:
                continue
            cis_logits = out["action_logits"]
            ref_logits = refs_t3[(tid, step)]["ref"]
            d = float(np.abs(cis_logits - ref_logits).max())
            if d > max_diff_per_slot[slot]:
                max_diff_per_slot[slot] = d
            if d < tol_t3:
                n_pass_total += 1
            else:
                print(f"  FAIL tid={tid} step={step} slot={slot}: diff={d:.2e}")

    for s, d in max_diff_per_slot.items():
        print(f"  slot {s} max diff: {d:.2e}  [{_bool(d < tol_t3)}]")
    print(f"  passed: {n_pass_total}/{n_total} requests (tol={tol_t3:.0e})")
    server.shutdown()
    test3_pass = (n_pass_total == n_total and not thread_errors)
    results["test_3_batch_partition"] = test3_pass
    print(f"  TEST 3: {_bool(test3_pass)}")
    print()

    # ===== Cleanup + summary =====
    for p_ in (ckpt_B, ckpt_C):
        try:
            os.remove(p_)
        except OSError:
            pass

    print("=" * 60)
    print("=== Phase 4.3a slot-mechanism summary ===")
    for k, v in results.items():
        print(f"  {k:32s} {_bool(v)}")
    overall = all(results.values())
    print(f"  overall: {_bool(overall)}")
    print("=" * 60)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
