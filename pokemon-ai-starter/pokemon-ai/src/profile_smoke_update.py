"""Standalone profile harness for smoke-scale ppo_update_batched.

Wraps ONE call to ppo_update_batched in torch.profiler. Saves chrome trace
+ prints top-N ops by self CPU time, self CUDA time, total CPU time. Goal:
compare the decomposition (CUDA vs Python/CPU share) to S62's prod profile
findings to validate or refute the "92% Python orchestration" claim that
underpins the post-wrap optimization plan.

Doesn't go through CIS collection — uses synthetic episodes from
test_packed_update_smoke.py. This is FINE for the question being asked:
"how is update wall split between CUDA and Python/CPU" is independent of
the data content.

Run on prod:
  cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
  export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
  python profile_smoke_update.py --packed --n-episodes 100 --minibatch-size 16
  python profile_smoke_update.py --packed --n-episodes 100 --minibatch-size 32  # for 2a experiment
"""
import argparse
import os
import sys

import torch
import torch.profiler as tp

sys.path.insert(0, os.path.dirname(__file__))

from ppo import load_checkpoint, ppo_update_batched
from test_forward_ppo_sequence_packed import _make_synthetic_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=100,
                   help="Number of synthetic episodes (default 100 = smoke scale)")
    p.add_argument("--minibatch-size", type=int, default=16,
                   help="--tier3-minibatch-size; chunks = ceil(n_episodes/this)")
    p.add_argument("--packed", action="store_true", help="Use --packed path")
    p.add_argument("--with-bc", action="store_true", default=True,
                   help="Enable BC anchor (matches canonical Phase 2 stack)")
    p.add_argument("--out-prefix", default="/tmp/profile_smoke",
                   help="Chrome trace + log will be <prefix>.json and <prefix>.log")
    p.add_argument("--with-stack", action="store_true",
                   help="Capture Python call stacks (slower; matches S64 step-back)")
    p.add_argument("--record-shapes", action="store_true",
                   help="Capture op input shapes")
    p.add_argument("--avg-T", type=int, default=48,
                   help="Average episode length (matches prod ~48 turns)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, torch={torch.__version__}")
    print(f"config: n_episodes={args.n_episodes}, minibatch={args.minibatch_size}, "
          f"packed={args.packed}, with_bc={args.with_bc}, "
          f"with_stack={args.with_stack}, record_shapes={args.record_shapes}")

    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)

    print("Loading BC v10...")
    model, cfg, _ = load_checkpoint(
        "data/models/bc/v10_padded_for_cis_dev.pt", device,
    )
    print(f"BC v10 loaded.")

    # Build synthetic episodes — avg T ~48 turns to match prod's 77000/1600 ratio
    print(f"Building {args.n_episodes} synthetic episodes (avg T={args.avg_T})...")
    import random as _random
    _random.seed(0)
    T_list = [max(10, args.avg_T + _random.randint(-20, 30))
              for _ in range(args.n_episodes)]
    episodes = [
        _make_synthetic_episode(T, A=cfg.format_config.n_actions, seed=i)
        for i, T in enumerate(T_list)
    ]
    print(f"  Total transitions: {sum(T_list)}")
    print(f"  Chunks per epoch: {(args.n_episodes + args.minibatch_size - 1) // args.minibatch_size}")
    print(f"  Total chunk passes (3 epochs): "
          f"{((args.n_episodes + args.minibatch_size - 1) // args.minibatch_size) * 3}")

    # BC ref = model itself (same as B.6 smoke test setup; identical inference cost)
    bc_ref = model if args.with_bc else None

    common = dict(
        epochs=3,
        clip_eps=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=100.0,  # disable gate to match production data flow
        normalize_advantages=False,
        compiled_step=None,
        bc_ref=bc_ref,
        bc_anchor_coef=0.1 if args.with_bc else 0.0,
        minibatch_size=args.minibatch_size,
        packed=args.packed,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Warmup: one untimed iter to bring CUDA + kernel caches hot
    print("Warmup run (untimed)...")
    _stats = ppo_update_batched(model, optim, list(episodes), device, cfg, **common)
    print(f"  warmup completed: n_succeeded={_stats['n_succeeded']}")

    # Profiled run
    print("Profiled run...")
    log_path = args.out_prefix + ".log"
    trace_path = args.out_prefix + ".json"

    activities = [tp.ProfilerActivity.CPU, tp.ProfilerActivity.CUDA]
    import time
    t0 = time.time()

    with tp.profile(
        activities=activities,
        record_shapes=args.record_shapes,
        with_stack=args.with_stack,
        profile_memory=False,
    ) as prof:
        stats = ppo_update_batched(model, optim, list(episodes), device, cfg, **common)
    wall = time.time() - t0
    print(f"  wall: {wall:.2f}s   n_succeeded={stats['n_succeeded']}")

    # Export chrome trace (Phase B/wrap convention)
    print(f"Exporting chrome trace: {trace_path}")
    prof.export_chrome_trace(trace_path)

    # Top-N ops summary, written to log + stdout
    sort_keys = [
        ("self_cuda_time_total", 30, "TOP 30 BY SELF CUDA TIME"),
        ("self_cpu_time_total",  30, "TOP 30 BY SELF CPU TIME"),
        ("cpu_time_total",       20, "TOP 20 BY TOTAL CPU TIME (includes subcalls)"),
    ]
    with open(log_path, "w") as f:
        def _w(s):
            print(s, flush=True)
            f.write(s + "\n")

        _w(f"=== profile_smoke_update.py ===")
        _w(f"config: n_episodes={args.n_episodes}, minibatch={args.minibatch_size}, "
           f"packed={args.packed}, with_bc={args.with_bc}, avg_T={args.avg_T}")
        _w(f"total transitions: {sum(T_list)}")
        _w(f"chunks per epoch: {(args.n_episodes + args.minibatch_size - 1) // args.minibatch_size}")
        _w(f"total chunk passes (3 epochs): "
           f"{((args.n_episodes + args.minibatch_size - 1) // args.minibatch_size) * 3}")
        _w(f"profiled wall: {wall:.2f}s")
        _w(f"n_succeeded: {stats['n_succeeded']}")
        _w("")

        for sort_key, row_limit, title in sort_keys:
            _w(f"=== {title} ===")
            _w(prof.key_averages().table(
                sort_by=sort_key,
                row_limit=row_limit,
            ))
            _w("")

    print(f"Log written to: {log_path}")
    print(f"Trace written to: {trace_path}")


if __name__ == "__main__":
    main()
