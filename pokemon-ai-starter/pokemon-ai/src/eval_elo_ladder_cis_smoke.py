#!/usr/bin/env python3
"""CIS-routed Elo eval — Phase 1 SMOKE only.

Per project_cis_elo_ladder_design memo. This script validates the CIS-routing
mechanism end-to-end on a minimal 2-player matchup BEFORE building the full
ladder eval.

What it does:
    1. Spawn CIS server subprocess (via _cis_main_multi) with N=2 slots,
       pre-loading the two specified checkpoints.
    2. Create 2 BattleAgentTransformerCIS players sharing one pipe pair,
       each bound to its respective slot.
    3. Play N games between them, identical to current eval_elo_ladder
       run_match logic.
    4. Print W/L/T result + per-game time.

Comparison vs current eval_elo_ladder.py:
    Both should produce statistically-equivalent results given same teams.
    Differences in individual game outcomes are expected (asyncio scheduling
    + Showdown's RNG). Aggregate W/L over enough games should match within
    binomial CI.

Usage:
    python eval_elo_ladder_cis_smoke.py \\
        --ckpt-a data/models/bc/v10_padded_for_cis_dev.pt \\
        --ckpt-b data/models/rl_v10/lr8e5_v1_flash/.../snapshot_0139.pt \\
        --n-games 10 \\
        --server ws://127.0.0.1:9020/showdown/websocket

Cost: ~5 min for 10 games on dev pod. Uses 2 model slots (~400 MB GPU) +
existing battle server infrastructure.
"""
import argparse
import asyncio
import gc
import os
import sys
import threading
import time
from pathlib import Path

import torch

# Reuse from existing code where possible
from poke_env.ps_client.account_configuration import AccountConfiguration
from ppo import load_checkpoint
from model_transformer import TransformerConfig
from battle_agent_transformer_cis import BattleAgentTransformerCIS
from eval_elo_ladder import (
    resolve_server,
    random_pool_teambuilder,
    _battle_pair,
)
from mp_centralized_collect import _cis_main_multi, _get_mp_ctx


def _get_cfg_from_ckpt(path: str) -> TransformerConfig:
    """Load TransformerConfig from a ckpt — needed for Player init (temporal_context etc.).

    Note: this does a torch.load() just to read the cfg dict. Cheap relative
    to the model load that CIS will do separately.
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg_dict = ckpt.get("model_config", {})
    return TransformerConfig.from_dict(cfg_dict)


def main():
    p = argparse.ArgumentParser(description="CIS Elo eval — Phase 1 smoke (2 players)")
    p.add_argument("--ckpt-a", required=True, help="Checkpoint A path")
    p.add_argument("--ckpt-b", required=True, help="Checkpoint B path")
    p.add_argument("--n-games", type=int, default=10, help="Games between A and B")
    p.add_argument("--server", default="ws://127.0.0.1:9020/showdown/websocket",
                   help="Battle server URL (use port not in fishbowl range)")
    p.add_argument("--device", default="cuda", help="Device for CIS slots")
    p.add_argument("--format", default="gen9ou", dest="battle_format")
    p.add_argument("--concurrency", type=int, default=4, help="Max concurrent battles per player")
    p.add_argument("--cis-min-batch", type=int, default=2,
                   help="CIS batch min size — set to 2 (we have 2 players sharing pipe)")
    p.add_argument("--cis-timeout-ms", type=int, default=15)
    args = p.parse_args()

    # Validate ckpts exist
    for label, path in (("A", args.ckpt_a), ("B", args.ckpt_b)):
        if not Path(path).exists():
            print(f"ERROR: ckpt {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    print(f"[smoke] Phase 1 CIS eval — {args.n_games} games")
    print(f"  Slot 0 (Player A): {args.ckpt_a}")
    print(f"  Slot 1 (Player B): {args.ckpt_b}")
    print(f"  Server: {args.server}")
    print()

    # Get cfg from one ckpt (assume both have compatible cfg — true for our use case)
    cfg = _get_cfg_from_ckpt(args.ckpt_a)
    print(f"  Loaded TransformerConfig from {args.ckpt_a}")
    print(f"    temporal_context={cfg.temporal_context}, n_moves={cfg.n_moves}")

    # Set up CIS subprocess
    # For smoke: 1 "worker" (this process) sharing one pipe pair.
    ctx = _get_mp_ctx()
    req_r, req_w = ctx.Pipe(duplex=False)   # main -> CIS direction
    resp_r, resp_w = ctx.Pipe(duplex=False)  # CIS -> main direction
    # Control pipes (per _cis_main_multi signature — unused for eval but required)
    ctrl_req_r, ctrl_req_w = ctx.Pipe(duplex=False)
    ctrl_resp_r, ctrl_resp_w = ctx.Pipe(duplex=False)

    print(f"  Spawning CIS server with 2 slots...")
    cis_proc = ctx.Process(
        target=_cis_main_multi,
        args=([req_r], [resp_w],
              [args.ckpt_a, args.ckpt_b],
              args.device,
              True,        # fp16
              None,        # amp_dtype_name (auto-detect bf16 on a100)
              args.cis_min_batch,
              args.cis_timeout_ms,
              ctrl_req_r, ctrl_resp_w),
        daemon=False,
    )
    cis_proc.start()
    # Close child-side ends in parent (held only by child)
    req_r.close()
    resp_w.close()
    ctrl_req_r.close()
    ctrl_resp_w.close()

    # Wait for CIS ready signal (it writes to resp pipe after model load)
    print(f"  Waiting for CIS server ready...")
    try:
        ready = resp_r.recv()
        print(f"    CIS ready: {ready}")
    except Exception as e:
        print(f"  ERROR: CIS server failed to signal ready: {e}")
        cis_proc.terminate()
        sys.exit(1)

    # Pipe lock — both Players share one pipe pair in this process
    pipe_lock = threading.Lock()

    # Set up two CIS-routed players
    server_cfg = resolve_server(args.server)
    _pid = os.getpid() % 10000

    player_a = BattleAgentTransformerCIS(
        cis_req_writer=req_w,
        cis_resp_reader=resp_r,
        cis_pipe_lock=pipe_lock,
        slot_id=0,
        cfg=cfg,
        device=args.device,
        battle_format=args.battle_format,
        max_concurrent_battles=args.concurrency,
        server_configuration=server_cfg,
        team=random_pool_teambuilder(),
        account_configuration=AccountConfiguration(f"E{_pid}smkA", None),
        checkpoint_path=args.ckpt_a,
    )
    player_b = BattleAgentTransformerCIS(
        cis_req_writer=req_w,
        cis_resp_reader=resp_r,
        cis_pipe_lock=pipe_lock,
        slot_id=1,
        cfg=cfg,
        device=args.device,
        battle_format=args.battle_format,
        max_concurrent_battles=args.concurrency,
        server_configuration=server_cfg,
        team=random_pool_teambuilder(),
        account_configuration=AccountConfiguration(f"E{_pid}smkB", None),
        checkpoint_path=args.ckpt_b,
    )

    print(f"  Players created. Starting {args.n_games}-game match...")
    t0 = time.time()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_battle_pair(
            player_a, player_b, args.n_games,
            timeout=max(600, args.n_games * 30),  # 30s/game cap for safety
        ))
    finally:
        loop.close()
    elapsed = time.time() - t0

    w_a, w_b = player_a.n_won_battles, player_b.n_won_battles
    ties = player_a.n_tied_battles
    total = w_a + w_b + ties

    print()
    print(f"[smoke] DONE in {elapsed:.1f}s ({elapsed/max(1,total):.1f}s/game)")
    print(f"  A wins:  {w_a}  ({w_a/max(1,total):.1%})")
    print(f"  B wins:  {w_b}  ({w_b/max(1,total):.1%})")
    print(f"  Ties:    {ties}")
    print(f"  Total:   {total}")

    # Teardown CIS subprocess
    print(f"\n  Shutting down CIS subprocess...")
    try:
        with pipe_lock:
            req_w.send({"cmd": "shutdown"})
    except Exception:
        pass
    cis_proc.join(timeout=10)
    if cis_proc.is_alive():
        cis_proc.terminate()
        cis_proc.join(timeout=5)
    print(f"  CIS exited (exitcode={cis_proc.exitcode})")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
