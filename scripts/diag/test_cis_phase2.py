#!/usr/bin/env python
"""CIS Phase 2 multi-worker batching test.

Validates `CISServer` + `CISClientHandle` in mp_centralized_collect.py:
  1. CIS subprocess spawns + accepts N worker pipes
  2. N threads each send M requests concurrently via their own handle
  3. CIS multiplexes via mp.connection.wait + cross-worker batches
  4. All NxM responses arrive correctly + match direct-main reference
  5. Throughput should beat Phase 1's single-threaded ~28 req/s by N (or more
     with batching - cross-worker batching is the headline win)

Acceptance:
  - All NxM responses received
  - Logits identity per response: max abs diff < 1e-3 (fp16 noise tolerance)
  - No NaN/inf in any response
  - Clean shutdown

Usage on cloud pod:
  cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
  python /workspace/scripts/test_cis_phase2.py [--n-workers 4] [--m 50] [--batch 8]

Side-effect-free: spawns 1 CIS subprocess (uses ~1 GB GPU). Safe alongside
production training.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import Dict, List, Tuple

import numpy as np
import torch


def synthesize_batch(model, B: int = 8, device: str = "cuda", seed: int = 0):
    """Same synth as test_cis_phase1.py - duplicated because diag scripts
    are standalone."""
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
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--m", type=int, default=50,
                   help="requests per worker")
    p.add_argument("--batch", type=int, default=8,
                   help="B per request")
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--min-batch", type=int, default=8)
    p.add_argument("--timeout-ms", type=int, default=15)
    args = p.parse_args()

    fp16 = not args.no_fp16

    print("=== CIS Phase 2 multi-worker batching test ===")
    print(f"ckpt={args.ckpt}, N_workers={args.n_workers}, M_per_worker={args.m}, "
          f"B={args.batch}, fp16={fp16}, tol={args.tol}, "
          f"min_batch={args.min_batch}, timeout_ms={args.timeout_ms}")
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
    from mp_centralized_collect import CISServer, torch_dict_to_numpy

    device = torch.device(args.device)

    # Stage 1: Reference forward in main
    print("Stage 1: load model in main + compute reference outputs")
    t0 = time.time()
    model, cfg, _ = load_checkpoint(args.ckpt, device)
    model.eval()
    n_total = args.n_workers * args.m
    print(f"  main model loaded ({time.time()-t0:.1f}s)")
    print(f"  generating {n_total} batches + reference forward...")

    # Same seed scheme: thread t request i uses seed (t * 10000 + i)
    batches_torch: Dict[Tuple[int, int], Dict] = {}
    reference: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    t0 = time.time()
    with torch.no_grad(), autocast_ctx(fp16):
        for t in range(args.n_workers):
            for i in range(args.m):
                seed = t * 10000 + i
                batch = synthesize_batch(model, B=args.batch, device=args.device, seed=seed)
                batches_torch[(t, i)] = batch
                out = model(batch)
                reference[(t, i)] = {
                    "action_logits": out["action_logits"].detach().float().cpu().numpy(),
                    "value":         out["value"].detach().float().cpu().numpy(),
                    "v_logits":      out["v_logits"].detach().float().cpu().numpy(),
                    "summary":       out["summary"].detach().float().cpu().numpy(),
                }
    print(f"  reference done ({time.time()-t0:.1f}s)")
    print()

    # Pre-serialize batches to numpy
    print("Stage 2: serialize batches to numpy")
    batches_numpy: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {
        k: torch_dict_to_numpy(v) for k, v in batches_torch.items()
    }
    print(f"  {n_total} batches converted")
    print()

    # Stage 3: spawn CIS server with N worker handles
    print(f"Stage 3: spawn CIS server (N={args.n_workers})")
    server = CISServer(args.ckpt, n_workers=args.n_workers,
                       device=args.device, fp16=fp16,
                       min_batch=args.min_batch, timeout_ms=args.timeout_ms)
    t0 = time.time()
    handles = server.spawn(ready_timeout_s=120.0)
    print(f"  CIS server up, {len(handles)} handles ready ({time.time()-t0:.1f}s)")
    for h in handles:
        ok = h.ping(timeout_s=5.0)
        print(f"    handle worker_idx={h.worker_idx}: ping {_bool(ok)}")
    print()

    # Stage 4: N threads each send M requests
    print(f"Stage 4: {args.n_workers} threads sending {args.m} reqs each")
    results: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    results_lock = threading.Lock()
    errors: List[str] = []

    def _worker_thread(handle, worker_idx: int):
        for i in range(args.m):
            np_batch = batches_numpy[(worker_idx, i)]
            try:
                out = handle.infer(np_batch, timeout_s=60.0)
            except Exception as e:
                with results_lock:
                    errors.append(f"worker {worker_idx} req {i}: {e}")
                return
            with results_lock:
                results[(worker_idx, i)] = out

    t0 = time.time()
    threads = [threading.Thread(target=_worker_thread, args=(handles[t], t),
                                 daemon=True)
               for t in range(args.n_workers)]
    for thr in threads:
        thr.start()
    for thr in threads:
        thr.join(timeout=600)
    elapsed = time.time() - t0
    print(f"  all threads done in {elapsed:.1f}s")
    total_throughput = n_total / elapsed
    print(f"  aggregate throughput: {total_throughput:.1f} req/s "
          f"(per-worker: {total_throughput/args.n_workers:.1f} req/s)")
    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for e in errors[:5]:
            print(f"    {e}")
    print()

    # Stage 5: validate all responses received + logits identity
    print("Stage 5: logits identity check across all responses")
    missing = [k for k in batches_numpy if k not in results]
    if missing:
        print(f"  MISSING {len(missing)} responses (first 3): {missing[:3]}")
    received = len(results)
    print(f"  received: {received}/{n_total}")

    keys = ["action_logits", "value", "v_logits", "summary"]
    max_diffs = {k: 0.0 for k in keys}
    nan_count = 0
    failed_count = 0
    failed_examples: List[str] = []

    for k_idx, (key, ref_out) in enumerate(reference.items()):
        if key not in results:
            continue
        cis_out = results[key]
        for fk in keys:
            if not np.isfinite(cis_out[fk]).all():
                nan_count += 1
                if len(failed_examples) < 3:
                    failed_examples.append(f"NaN in {fk} at {key}")
                continue
            d = float(np.abs(ref_out[fk] - cis_out[fk]).max())
            if d > max_diffs[fk]:
                max_diffs[fk] = d
            if d > args.tol:
                failed_count += 1
                if len(failed_examples) < 5:
                    failed_examples.append(f"{key} {fk} diff={d:.2e}")

    for k, d in max_diffs.items():
        ok = d < args.tol
        print(f"  max abs diff {k:14s}: {d:.2e}  [{_bool(ok)}]")
    print(f"  NaN/inf entries: {nan_count}")
    print(f"  failed (diff > tol): {failed_count}")
    if failed_examples:
        print(f"  examples: {failed_examples}")
    print()

    overall = (received == n_total and nan_count == 0 and failed_count == 0
               and not errors)

    # Stage 6: shutdown
    print("Stage 6: shutdown CIS server")
    server.shutdown()
    print("  shutdown OK")
    print()

    # Summary
    print("=== Summary ===")
    print(f"  N workers: {args.n_workers}")
    print(f"  M requests/worker: {args.m}")
    print(f"  total requests: {n_total}")
    print(f"  received responses: {received}/{n_total}")
    print(f"  aggregate throughput: {total_throughput:.1f} req/s")
    print(f"  max action_logits diff: {max_diffs['action_logits']:.2e}")
    print(f"  errors: {len(errors)}, NaN: {nan_count}, diff-fail: {failed_count}")
    print(f"  result: {_bool(overall)}")
    print()
    if overall:
        print("VERDICT: CIS Phase 2 PASSED. Cross-worker batching works.")
        return 0
    else:
        print("VERDICT: CIS Phase 2 FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
