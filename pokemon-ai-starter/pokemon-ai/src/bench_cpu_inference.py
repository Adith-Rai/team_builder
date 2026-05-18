"""CPU inference latency / throughput benchmark for the 20M
TransformerBattlePolicy.

Gate for task #49 (CPU opp inference service). No production code modified.

Measures:
  1. Single-process latency at batch sizes [1, 2, 4, 8, 16] (uses all cores)
  2. Multi-process aggregate throughput at worker counts [1, 4, 16, 32, 64]
     (each worker limited to 1 thread → linear scaling test)

Compare to GPU CIS baseline (from CIS-STATS at our prod config):
  - Single slot at batch=23: ~1150 inf/sec
  - 2 slots aggregate (pool=1): ~2300 inf/sec
  - 5-6 slots starving (pool=5): ~500 inf/sec aggregate

Decision criterion:
  - Multi-process aggregate >= ~2000 inf/sec at batch>=4 → centralized CPU
    service likely viable; task #49 worth building
  - Aggregate < 500 inf/sec → CPU service refuted; close task #49

Run on prod:
  cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
  export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
  python bench_cpu_inference.py \\
    --bc-ckpt data/models/bc/v10_padded_for_cis_dev.pt
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))


def _stack_batch_for_inference(per_turn_dicts, device):
    """Stack a list of B per-turn dicts into a single batch dict with B-leading
    tensor dims. Mirrors features.build_turn_batch's expected output for B
    games. Just torch.cat along dim 0 for each leaf, recursing into nested
    dicts (field_banks, transition_ids, active_move_banks)."""
    out = {}
    keys = per_turn_dicts[0].keys()
    for k in keys:
        v0 = per_turn_dicts[0][k]
        if isinstance(v0, dict):
            sub = {}
            for kk in v0.keys():
                sub[kk] = torch.cat(
                    [d[k][kk] for d in per_turn_dicts], dim=0
                ).to(device)
            out[k] = sub
        else:
            out[k] = torch.cat(
                [d[k] for d in per_turn_dicts], dim=0
            ).to(device)
    return out


def _build_batch(B: int, seed: int = 0, device: torch.device = torch.device("cpu")):
    """Build a single batch of B games' worth of input features."""
    from test_forward_ppo_sequence_packed import _make_synthetic_turn

    rng = np.random.RandomState(seed)
    turns = [_make_synthetic_turn(rng) for _ in range(B)]
    return _stack_batch_for_inference(turns, device)


def bench_single_process(model, batch_sizes, n_iters_per_batch=30, warmup=5):
    """Measure forward_spatial latency at various batch sizes, single process.
    Uses default torch thread count (typically all cores)."""
    print(f"\n{'=' * 70}")
    print(f"Single-process latency (torch threads = {torch.get_num_threads()})")
    print(f"{'=' * 70}")
    print(f"{'batch':>6} {'per_call_ms':>13} {'per_inf_ms':>11} {'throughput_inf/s':>18}")

    results = []
    for B in batch_sizes:
        batch = _build_batch(B)
        with torch.no_grad():
            # Warmup
            for _ in range(warmup):
                _ = model.forward_spatial(batch)
            # Timed
            t0 = time.perf_counter()
            for _ in range(n_iters_per_batch):
                _ = model.forward_spatial(batch)
            elapsed = time.perf_counter() - t0
        per_call_s = elapsed / n_iters_per_batch
        per_inf_s = per_call_s / B
        throughput = B / per_call_s
        print(f"{B:>6} {per_call_s * 1000:>13.2f} "
              f"{per_inf_s * 1000:>11.3f} {throughput:>18.1f}")
        results.append({"B": B, "per_call_ms": per_call_s * 1000,
                        "per_inf_ms": per_inf_s * 1000,
                        "throughput": throughput})
    return results


def _worker_run(ckpt_path: str, batch_size: int, n_iters: int,
                worker_idx: int, n_threads: int, conn) -> None:
    """Multi-process worker: load model on CPU, run inference loop, report timing."""
    try:
        import torch as _t
        _t.set_num_threads(n_threads)
        from ppo import load_checkpoint
        # Disable autograd globally for this process
        _t.set_grad_enabled(False)
        model, _, _ = load_checkpoint(ckpt_path, _t.device("cpu"))
        model.eval()
        batch = _build_batch(batch_size, seed=worker_idx)
        # Warmup
        for _ in range(3):
            _ = model.forward_spatial(batch)
        # Timed
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = model.forward_spatial(batch)
        elapsed = time.perf_counter() - t0
        conn.send({"worker_idx": worker_idx, "elapsed": elapsed,
                   "n_iters": n_iters, "batch_size": batch_size})
    except Exception as e:
        import traceback
        conn.send({"worker_idx": worker_idx, "error": str(e),
                   "tb": traceback.format_exc()})
    finally:
        conn.close()


def bench_multi_process(ckpt_path: str, n_workers: int, batch_size: int,
                        n_iters: int, threads_per_worker: int = 1):
    """Spawn N parallel inference workers, each does N inferences, aggregate."""
    ctx = mp.get_context("spawn")
    procs = []
    parent_conns = []
    for i in range(n_workers):
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        p = ctx.Process(
            target=_worker_run,
            args=(ckpt_path, batch_size, n_iters, i, threads_per_worker,
                  child_conn),
        )
        p.start()
        parent_conns.append(parent_conn)
        procs.append(p)

    results = []
    for c in parent_conns:
        try:
            r = c.recv()
            results.append(r)
        except Exception as e:
            results.append({"error": f"recv failed: {e}"})
    for p in procs:
        p.join(timeout=30.0)

    errors = [r for r in results if "error" in r]
    if errors:
        print(f"  ⚠️  {len(errors)}/{n_workers} workers errored")
        for e in errors[:2]:
            print(f"  err: {e.get('error', '?')}")
        return None

    total_inferences = sum(r["n_iters"] * r["batch_size"] for r in results)
    # Aggregate throughput = total work / max worker elapsed (bottleneck)
    max_elapsed = max(r["elapsed"] for r in results)
    min_elapsed = min(r["elapsed"] for r in results)
    aggregate_throughput = total_inferences / max_elapsed
    # Per-worker avg throughput
    per_worker_throughput = aggregate_throughput / n_workers
    return {
        "n_workers": n_workers,
        "batch_size": batch_size,
        "total_inferences": total_inferences,
        "max_elapsed_s": max_elapsed,
        "min_elapsed_s": min_elapsed,
        "aggregate_throughput": aggregate_throughput,
        "per_worker_throughput": per_worker_throughput,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bc-ckpt", required=True,
                   help="Path to BC v10 checkpoint (we benchmark its arch).")
    p.add_argument("--n-iters-single", type=int, default=30,
                   help="N forwards per batch size in single-process bench.")
    p.add_argument("--n-iters-multi", type=int, default=20,
                   help="N forwards per worker in multi-process bench.")
    p.add_argument("--batch-sizes", type=str, default="1,2,4,8,16",
                   help="Comma-separated batch sizes for single-process bench.")
    p.add_argument("--worker-counts", type=str, default="1,4,16,32,64",
                   help="Comma-separated worker counts for multi-process bench.")
    p.add_argument("--mp-batch-size", type=int, default=4,
                   help="Batch size used in multi-process bench (one value).")
    p.add_argument("--mp-threads", type=int, default=1,
                   help="Threads per worker in multi-process bench (default 1 "
                        "for clean linear-scaling test).")
    args = p.parse_args()

    print("=" * 70)
    print("CPU Inference Benchmark — gate for task #49 (CPU opp service)")
    print("=" * 70)

    import platform
    n_cpu = os.cpu_count()
    print(f"Host: {platform.node()}")
    print(f"CPUs visible: {n_cpu}")
    try:
        with open("/proc/cpuinfo") as f:
            model = next(
                (line.split(":")[1].strip() for line in f
                 if line.startswith("model name")), "?"
            )
        print(f"CPU model: {model}")
    except Exception:
        pass
    print(f"Torch version: {torch.__version__}")
    print(f"Initial torch.get_num_threads(): {torch.get_num_threads()}")

    # Load model once for the single-process bench
    from ppo import load_checkpoint
    print(f"\nLoading checkpoint: {args.bc_ckpt}")
    t0 = time.perf_counter()
    model, _cfg, _meta = load_checkpoint(args.bc_ckpt, torch.device("cpu"))
    model.eval()
    load_time = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {n_params / 1e6:.1f}M params on CPU in {load_time:.1f}s")

    # === Single-process latency ===
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    with torch.no_grad():
        single_results = bench_single_process(
            model, batch_sizes, n_iters_per_batch=args.n_iters_single
        )

    # Release model before spawning workers (each loads its own)
    del model

    # === Multi-process throughput ===
    worker_counts = [int(x) for x in args.worker_counts.split(",")]
    print(f"\n{'=' * 70}")
    print(f"Multi-process aggregate throughput "
          f"(batch={args.mp_batch_size}, threads/worker={args.mp_threads})")
    print(f"{'=' * 70}")
    print(f"{'n_workers':>10} {'agg_inf/s':>11} {'per_worker':>11} "
          f"{'max_elapsed_s':>14}")

    multi_results = []
    for nw in worker_counts:
        if nw > n_cpu and args.mp_threads >= 1:
            print(f"  [skip n_workers={nw}: exceeds visible CPUs {n_cpu}]")
            continue
        r = bench_multi_process(
            args.bc_ckpt, n_workers=nw, batch_size=args.mp_batch_size,
            n_iters=args.n_iters_multi, threads_per_worker=args.mp_threads,
        )
        if r is None:
            print(f"  [n_workers={nw}: failed]")
            continue
        print(f"{r['n_workers']:>10} {r['aggregate_throughput']:>11.1f} "
              f"{r['per_worker_throughput']:>11.1f} "
              f"{r['max_elapsed_s']:>14.2f}")
        multi_results.append(r)

    # === Summary + decision ===
    print(f"\n{'=' * 70}")
    print("Decision metrics for task #49 (CPU opp service)")
    print(f"{'=' * 70}")
    print(f"Single-process throughput at batch=4: "
          f"{[r['throughput'] for r in single_results if r['B'] == 4][0]:.1f} inf/s")
    if multi_results:
        best = max(multi_results, key=lambda r: r["aggregate_throughput"])
        print(f"Best multi-process aggregate: "
              f"{best['aggregate_throughput']:.1f} inf/s "
              f"@ n_workers={best['n_workers']} "
              f"batch={best['batch_size']}")
        print(f"\nComparison to GPU CIS at our prod config:")
        print(f"  GPU pool=1 (single slot batched): ~1150 inf/s per slot")
        print(f"  GPU pool=1 aggregate (2 slots): ~2300 inf/s")
        print(f"  GPU pool=5 starving (5-6 slots): ~500 inf/s aggregate")
        print(f"\n  Best CPU aggregate / GPU pool=1: "
              f"{best['aggregate_throughput'] / 2300 * 100:.0f}%")
        print(f"  Best CPU aggregate / GPU pool=5: "
              f"{best['aggregate_throughput'] / 500 * 100:.0f}%")
        print(f"\nDecision:")
        if best["aggregate_throughput"] >= 2000:
            print("  ✓ CPU aggregate >= 2000 inf/s → centralized CPU service "
                  "VIABLE for task #49")
        elif best["aggregate_throughput"] >= 500:
            print("  ~ CPU aggregate 500-2000 inf/s → marginal. May help at "
                  "high pool (vs GPU starving) but won't beat low pool. "
                  "Investigate latency profile before building.")
        else:
            print("  ✗ CPU aggregate < 500 inf/s → centralized CPU service "
                  "REFUTED at our scale. Close task #49.")


if __name__ == "__main__":
    sys.exit(main())
