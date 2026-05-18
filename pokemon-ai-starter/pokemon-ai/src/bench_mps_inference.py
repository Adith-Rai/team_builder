"""MPS-shared GPU inference benchmark for the 20M TransformerBattlePolicy.

Phase A gate for multi-process CIS + CUDA MPS architecture (see
docs/MULTI_PROCESS_CIS_DESIGN.md). Adapted from bench_cpu_inference.py.

Measures:
  1. Single-process GPU throughput at batch sizes [1, 4, 16] — baseline
  2. Multi-process aggregate throughput at N=[2, 6] processes sharing the
     GPU via CUDA MPS daemon

Key methodology notes:
  - Uses forward_spatial (matches CIS hot path at line 690 of
    mp_centralized_collect.py for history-bearing requests AND matches
    bench_cpu_inference.py for direct comparison)
  - bf16 autocast (matches production CIS dispatch)
  - GPU sync via .cpu() at end of each forward, so timings include the sync
  - MPS env vars (CUDA_MPS_PIPE_DIRECTORY, CUDA_MPS_LOG_DIRECTORY) inherited
    from the parent process; MPS daemon must be started BEFORE running this
    script.

Compare to:
  - GPU CIS at our prod config:
    - Single slot at batch=23: ~1150 inf/sec (per S65 memo)
    - 2 slots aggregate (pool=1): ~2300 inf/sec
  - Gate (per design memo §4.1 A.3): if N=6 aggregate >= 2x N=1 throughput
    at our typical batch sizes (4-16), MPS scaling is viable for Phase B
    impl. If gains diminish past N=2, our workload doesn't get MPS's full
    benefit at N=6.

Run on prod (after starting MPS daemon):
  cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
  export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
  python bench_mps_inference.py \\
    --bc-ckpt data/models/bc/v10_padded_for_cis_dev.pt \\
    --n-procs-list 1,2,6
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
    tensor dims. Mirrors features.build_turn_batch's expected output."""
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
    """Build a single batch of B games' worth of input features on the given device."""
    from test_forward_ppo_sequence_packed import _make_synthetic_turn

    rng = np.random.RandomState(seed)
    turns = [_make_synthetic_turn(rng) for _ in range(B)]
    return _stack_batch_for_inference(turns, device)


def _do_forward(model, batch, use_bf16=True):
    """Single forward pass via forward_spatial with optional bf16 autocast.
    Forces GPU sync via .cpu() on output to match CIS dispatch timing."""
    if use_bf16:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            spatial_out, summary = model.forward_spatial(batch)
    else:
        with torch.no_grad():
            spatial_out, summary = model.forward_spatial(batch)
    # Force sync (matches CIS .cpu().numpy() pattern at line 759-762)
    _ = spatial_out.detach().float().cpu().numpy()
    _ = summary.detach().float().cpu().numpy()


def bench_single_process(model, batch_sizes, n_iters_per_batch=50, warmup=10):
    """Measure forward_spatial latency at various batch sizes, single process on GPU."""
    print(f"\n{'=' * 70}")
    print(f"Single-process GPU throughput (bf16, forward_spatial + sync)")
    print(f"{'=' * 70}")
    print(f"{'batch':>6} {'per_call_ms':>13} {'per_inf_ms':>11} {'throughput_inf/s':>18}")

    device = next(model.parameters()).device
    results = []
    for B in batch_sizes:
        batch = _build_batch(B, device=device)
        # Warmup
        for _ in range(warmup):
            _do_forward(model, batch)
        torch.cuda.synchronize()
        # Timed
        t0 = time.perf_counter()
        for _ in range(n_iters_per_batch):
            _do_forward(model, batch)
        torch.cuda.synchronize()
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
                worker_idx: int, conn) -> None:
    """Multi-process worker: load model on GPU, run inference loop, report timing.
    MPS env vars must be set in parent shell for this to share GPU efficiently."""
    try:
        import torch as _t
        from ppo import load_checkpoint
        _t.set_grad_enabled(False)
        device = _t.device("cuda")
        # Each process gets its own CUDA context (shared via MPS daemon if running)
        model, _, _ = load_checkpoint(ckpt_path, device)
        model.eval()
        batch = _build_batch(batch_size, seed=worker_idx, device=device)

        # Warmup
        for _ in range(10):
            _do_forward(model, batch)
        _t.cuda.synchronize()

        # Timed — synchronized start across workers via wait-on-conn signal
        # but here we just do a barrier via small sleep + perf_counter
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _do_forward(model, batch)
        _t.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        # Report GPU memory used by this process
        mem_allocated_mib = _t.cuda.memory_allocated() / 1024**2
        mem_reserved_mib = _t.cuda.memory_reserved() / 1024**2

        conn.send({
            "worker_idx": worker_idx,
            "elapsed": elapsed,
            "n_iters": n_iters,
            "batch_size": batch_size,
            "mem_allocated_mib": mem_allocated_mib,
            "mem_reserved_mib": mem_reserved_mib,
        })
    except Exception as e:
        import traceback
        conn.send({"worker_idx": worker_idx, "error": str(e),
                   "tb": traceback.format_exc()})
    finally:
        conn.close()


def bench_multi_process(ckpt_path: str, n_workers: int, batch_size: int,
                        n_iters: int):
    """Spawn N parallel GPU inference workers via MPS, aggregate throughput."""
    ctx = mp.get_context("spawn")
    procs = []
    parent_conns = []
    for i in range(n_workers):
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        p = ctx.Process(
            target=_worker_run,
            args=(ckpt_path, batch_size, n_iters, i, child_conn),
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
        p.join(timeout=120.0)

    errors = [r for r in results if "error" in r]
    if errors:
        print(f"  WARN: {len(errors)}/{n_workers} workers errored")
        for e in errors[:2]:
            print(f"  err: {e.get('error', '?')}")
            print(f"  tb: {e.get('tb', '?')[:500]}")
        return None

    total_inferences = sum(r["n_iters"] * r["batch_size"] for r in results)
    max_elapsed = max(r["elapsed"] for r in results)
    min_elapsed = min(r["elapsed"] for r in results)
    aggregate_throughput = total_inferences / max_elapsed
    per_worker_throughput = aggregate_throughput / n_workers
    total_mem_allocated = sum(r["mem_allocated_mib"] for r in results)
    total_mem_reserved = sum(r["mem_reserved_mib"] for r in results)
    return {
        "n_workers": n_workers,
        "batch_size": batch_size,
        "total_inferences": total_inferences,
        "max_elapsed_s": max_elapsed,
        "min_elapsed_s": min_elapsed,
        "aggregate_throughput": aggregate_throughput,
        "per_worker_throughput": per_worker_throughput,
        "total_mem_allocated_mib": total_mem_allocated,
        "total_mem_reserved_mib": total_mem_reserved,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bc-ckpt", required=True,
                   help="Path to BC v10 checkpoint (we benchmark its arch).")
    p.add_argument("--n-iters-single", type=int, default=50,
                   help="N forwards per batch size in single-process bench.")
    p.add_argument("--n-iters-multi", type=int, default=100,
                   help="N forwards per worker in multi-process bench.")
    p.add_argument("--batch-sizes", type=str, default="1,4,16",
                   help="Comma-separated batch sizes for single-process bench.")
    p.add_argument("--n-procs-list", type=str, default="1,2,6",
                   help="Comma-separated process counts for multi-process bench.")
    p.add_argument("--mp-batch-sizes", type=str, default="4,16",
                   help="Comma-separated batch sizes for multi-process bench.")
    args = p.parse_args()

    print("=" * 70)
    print("MPS Multi-Process GPU Inference Benchmark")
    print("Gate for multi-process CIS architecture (see MULTI_PROCESS_CIS_DESIGN.md)")
    print("=" * 70)

    import platform
    print(f"Host: {platform.node()}")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA mem: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")

    mps_pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "(not set)")
    mps_log = os.environ.get("CUDA_MPS_LOG_DIRECTORY", "(not set)")
    print(f"CUDA_MPS_PIPE_DIRECTORY: {mps_pipe}")
    print(f"CUDA_MPS_LOG_DIRECTORY: {mps_log}")

    # Load model once for the single-process bench
    from ppo import load_checkpoint
    print(f"\nLoading checkpoint: {args.bc_ckpt}")
    device = torch.device("cuda")
    model, _, _ = load_checkpoint(args.bc_ckpt, device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {n_params/1e6:.1f}M params on {device}")

    # Single-process baseline
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    single_results = bench_single_process(model, batch_sizes, n_iters_per_batch=args.n_iters_single)

    # Free the model before spawning workers (each will load its own)
    del model
    torch.cuda.empty_cache()

    # Multi-process at various N
    n_procs_list = [int(n) for n in args.n_procs_list.split(",")]
    mp_batch_sizes = [int(b) for b in args.mp_batch_sizes.split(",")]

    print(f"\n{'=' * 70}")
    print(f"Multi-process aggregate throughput (MPS-shared GPU)")
    print(f"{'=' * 70}")
    header = f"{'n_procs':>8} {'batch':>6} {'agg_inf/s':>10} {'per_proc':>10} {'max_s':>8} {'mem_alloc_MiB':>14}"
    print(header)

    multi_results = []
    for n in n_procs_list:
        for B in mp_batch_sizes:
            print(f"  Running N={n} batch={B} ...", flush=True)
            r = bench_multi_process(
                ckpt_path=args.bc_ckpt,
                n_workers=n,
                batch_size=B,
                n_iters=args.n_iters_multi,
            )
            if r is None:
                print(f"{n:>8} {B:>6}  ERROR (see above)")
                continue
            print(f"{n:>8} {B:>6} {r['aggregate_throughput']:>10.1f} "
                  f"{r['per_worker_throughput']:>10.1f} "
                  f"{r['max_elapsed_s']:>8.2f} {r['total_mem_allocated_mib']:>14.0f}")
            multi_results.append(r)

    # Summary + gate evaluation
    print(f"\n{'=' * 70}")
    print(f"Gate evaluation (Phase A.3)")
    print(f"{'=' * 70}")
    for B in mp_batch_sizes:
        baseline = next((r for r in multi_results
                         if r["n_workers"] == 1 and r["batch_size"] == B), None)
        n6 = next((r for r in multi_results
                   if r["n_workers"] == 6 and r["batch_size"] == B), None)
        if baseline is None or n6 is None:
            print(f"batch={B}: missing data (need n=1 AND n=6)")
            continue
        baseline_thru = baseline["aggregate_throughput"]
        n6_thru = n6["aggregate_throughput"]
        ratio = n6_thru / baseline_thru
        verdict = "PASS" if ratio >= 2.0 else "FAIL"
        print(f"batch={B}: N=1 = {baseline_thru:.0f} inf/s, "
              f"N=6 = {n6_thru:.0f} inf/s, ratio = {ratio:.2f}x ({verdict})")

    print("\n=== BENCH DONE ===")


if __name__ == "__main__":
    main()
