#!/usr/bin/env python3
"""CIS-routed Elo ladder eval — Phase 2.

Replaces eval_elo_ladder.py's per-pair model-load with a shared CIS server
that pre-loads all NN players' checkpoints into GPU slots. Bots run as
before (rule-based, no GPU). Reuses all of eval_elo_ladder.py's machinery
for matchup generation, Bradley-Terry Elo computation, output format.

Per project_cis_elo_ladder_design memo. Validates Phase 1 mechanism scales
to N-player ladder with bot inclusion.

Usage:
    # Quick 5-player smoke (NN BC + 2 lr8e5 + 2 bots, 5g/pair)
    python eval_elo_ladder_cis.py \\
        --snapshots data/models/bc/v10_padded_for_cis_dev.pt \\
        data/models/rl_v10/lr8e5_v1_flash/.../snapshot_0099.pt \\
        data/models/rl_v10/lr8e5_v1_flash/.../snapshot_0139.pt \\
        --names BC_v10 lr8e5_99 lr8e5_139 \\
        --bots SH SmartDmg \\
        --n-games 5 \\
        --server ws://127.0.0.1:9020/showdown/websocket \\
        --out-json data/eval/cis_smoke_5player.json

Compatible JSON output with eval_elo_ladder.py format (players, matches,
elos, config) so era4_chain accumulation + downstream tools work unchanged.
"""
import argparse
import asyncio
import gc
import json
import os
import sys
import threading
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from poke_env.ps_client.account_configuration import AccountConfiguration

from ppo import load_checkpoint
from model_transformer import TransformerConfig
from battle_agent_transformer_cis import BattleAgentTransformerCIS
from eval_elo_ladder import (
    PlayerSpec,
    ALL_BOTS,
    resolve_server,
    random_pool_teambuilder,
    _battle_pair,
    fit_bradley_terry,
)
from mp_centralized_collect import _cis_main_multi, _get_mp_ctx


def _get_cfg_from_ckpt(path: str) -> TransformerConfig:
    """Load TransformerConfig from a ckpt (cheap header peek)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg_dict = ckpt.get("model_config", {})
    return TransformerConfig.from_dict(cfg_dict)


def _build_player_specs(snapshots, names, bots) -> List[PlayerSpec]:
    """Build PlayerSpec list mixing snapshots + bots."""
    specs = []
    if names is None:
        names = [Path(p).stem for p in snapshots]
    assert len(names) == len(snapshots), "--names must match --snapshots count"
    for path, name in zip(snapshots, names):
        if not Path(path).exists():
            raise SystemExit(f"snapshot not found: {path}")
        specs.append(PlayerSpec(kind="snapshot", name=name, ckpt=path))
    for bot_name in (bots or []):
        if bot_name not in ALL_BOTS:
            raise SystemExit(
                f"bot {bot_name!r} not in ALL_BOTS registry. "
                f"Available: {sorted(ALL_BOTS.keys())}"
            )
        specs.append(PlayerSpec(kind="bot", name=bot_name, bot_cls=ALL_BOTS[bot_name]))
    return specs


def _spawn_cis_server(nn_ckpt_paths: List[str], device: str,
                     min_batch: int, timeout_ms: int):
    """Spawn CIS subprocess with N slots, one per NN ckpt. Returns
    (cis_proc, req_writer, resp_reader, ctrl_pipes) for parent to use.

    Single-worker model: parent process holds all Players, shares one pipe
    pair via lock. For ladder eval where we run matchups sequentially this
    is fine. (Multi-worker parallelism = Phase 3.)
    """
    ctx = _get_mp_ctx()
    req_r, req_w = ctx.Pipe(duplex=False)
    resp_r, resp_w = ctx.Pipe(duplex=False)
    ctrl_req_r, ctrl_req_w = ctx.Pipe(duplex=False)
    ctrl_resp_r, ctrl_resp_w = ctx.Pipe(duplex=False)

    cis_proc = ctx.Process(
        target=_cis_main_multi,
        args=([req_r], [resp_w],
              nn_ckpt_paths, device,
              True,        # fp16
              None,        # amp_dtype_name
              min_batch, timeout_ms,
              ctrl_req_r, ctrl_resp_w),
        daemon=False,
    )
    cis_proc.start()
    # Close child-side ends in parent
    req_r.close()
    resp_w.close()
    ctrl_req_r.close()
    ctrl_resp_w.close()

    # Wait for ready signal
    print(f"  Waiting for CIS ready (N slots={len(nn_ckpt_paths)})...", flush=True)
    ready = resp_r.recv()
    if ready.get("status") != "ready":
        cis_proc.terminate()
        raise SystemExit(f"CIS failed to come up: {ready}")
    print(f"  CIS ready.", flush=True)
    return cis_proc, req_w, resp_r, (ctrl_req_w, ctrl_resp_r)


def _make_player_for_spec(spec: PlayerSpec, slot_id: int, cis_req_w,
                          cis_resp_r, cis_lock, cfg, server_cfg,
                          account_name: str, device: str,
                          battle_format: str, concurrency: int):
    """Create a Player instance for one matchup side.

    NN players use BattleAgentTransformerCIS (CIS dispatch).
    Bots use their existing class (no model, no CIS).
    """
    common = dict(
        battle_format=battle_format,
        max_concurrent_battles=concurrency,
        server_configuration=server_cfg,
        team=random_pool_teambuilder(),
        account_configuration=AccountConfiguration(account_name, None),
    )
    if spec.kind == "snapshot":
        return BattleAgentTransformerCIS(
            cis_req_writer=cis_req_w,
            cis_resp_reader=cis_resp_r,
            cis_pipe_lock=cis_lock,
            slot_id=slot_id,
            cfg=cfg,
            device=device,  # logged only — Player itself runs on CPU
            checkpoint_path=spec.ckpt,
            **common,
        )
    elif spec.kind == "bot":
        return spec.bot_cls(**common)
    else:
        raise ValueError(spec.kind)


def run_match_cis(spec_a: PlayerSpec, spec_b: PlayerSpec,
                  n_games: int, slot_map: Dict[str, int],
                  cis_req_w, cis_resp_r, cis_lock, cfg,
                  server_cfg, device: str, battle_format: str,
                  concurrency: int, match_idx: int) -> dict:
    """Run n_games between spec_a and spec_b via CIS-routed players.

    Returns dict with same shape as eval_elo_ladder.py run_match output.
    """
    _pid = os.getpid() % 10000
    name_a = f"E{_pid}m{match_idx}a"
    name_b = f"E{_pid}m{match_idx}b"

    slot_a = slot_map.get(spec_a.name, -1)
    slot_b = slot_map.get(spec_b.name, -1)

    p1 = _make_player_for_spec(spec_a, slot_a, cis_req_w, cis_resp_r,
                                cis_lock, cfg, server_cfg, name_a,
                                device, battle_format, concurrency)
    p2 = _make_player_for_spec(spec_b, slot_b, cis_req_w, cis_resp_r,
                                cis_lock, cfg, server_cfg, name_b,
                                device, battle_format, concurrency)

    t0 = time.time()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_battle_pair(
            p1, p2, n_games, timeout=max(600, n_games * 30)
        ))
    finally:
        loop.close()
    elapsed = time.time() - t0

    w1, w2 = p1.n_won_battles, p2.n_won_battles
    ties = p1.n_tied_battles
    total = w1 + w2 + ties

    # Cleanup
    for p in (p1, p2):
        try:
            p.reset_battles()
        except Exception:
            pass
    del p1, p2

    return {
        "p1": spec_a.name, "p2": spec_b.name,
        "p1_kind": spec_a.kind, "p2_kind": spec_b.kind,
        "p1_wins": w1, "p2_wins": w2, "ties": ties, "total": total,
        "p1_wr": w1 / max(1, total),
        "elapsed": round(elapsed, 1),
    }


def compute_elos(player_specs: List[PlayerSpec],
                 matches: List[dict],
                 anchor: str = "SH",
                 anchor_elo: float = 1000.0) -> Dict[str, float]:
    """Fit BT model on match results, return per-player Elo with anchor calibration."""
    names = [s.name for s in player_specs]
    name_set = set(names)

    # Build wins / games dicts keyed by (name, name) per fit_bradley_terry signature
    wins: Dict[Tuple[str, str], float] = {}
    games: Dict[Tuple[str, str], int] = {}
    for m in matches:
        a, b = m["p1"], m["p2"]
        if a not in name_set or b not in name_set:
            continue
        w_ab = m["p1_wins"]
        w_ba = m["p2_wins"]
        ties = m["ties"]
        w_ab += ties * 0.5
        w_ba += ties * 0.5
        wins[(a, b)] = wins.get((a, b), 0) + w_ab
        wins[(b, a)] = wins.get((b, a), 0) + w_ba
        games[(a, b)] = games.get((a, b), 0) + m["total"]
        games[(b, a)] = games.get((b, a), 0) + m["total"]

    pis = fit_bradley_terry(names, wins, games)
    # Convert to Elo: elo_i = 400 * log10(pi_i) + offset; anchor = SH at 1000
    log_pis = {n: np.log10(p) if p > 0 else -10.0 for n, p in pis.items()}
    if anchor in log_pis:
        offset = anchor_elo - 400.0 * log_pis[anchor]
    else:
        offset = 1000.0
    elos = {n: 400.0 * log_pis[n] + offset for n in names}
    return elos


def main():
    p = argparse.ArgumentParser(
        description="CIS-routed Elo ladder eval (Phase 2 — N-player all-vs-all)"
    )
    p.add_argument("--snapshots", nargs="+", default=[],
                   help="NN checkpoint paths (slot 0, 1, ...)")
    p.add_argument("--names", nargs="+", default=None,
                   help="Names for snapshots (must match --snapshots count). Defaults to file stems.")
    p.add_argument("--bots", nargs="+", default=[],
                   help=f"Bot names from registry. Available: {sorted(ALL_BOTS.keys())}")
    p.add_argument("--n-games", type=int, default=10,
                   help="Games per matchup")
    p.add_argument("--server", default="ws://127.0.0.1:9020/showdown/websocket")
    p.add_argument("--device", default="cuda")
    p.add_argument("--format", default="gen9ou", dest="battle_format")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--cis-min-batch", type=int, default=2)
    p.add_argument("--cis-timeout-ms", type=int, default=15)
    p.add_argument("--anchor", default="SH", help="Bot name to anchor Elo")
    p.add_argument("--anchor-elo", type=float, default=1000.0)
    p.add_argument("--out-json", default=None,
                   help="Output JSON path. If None, prints to stdout only.")
    args = p.parse_args()

    if not args.snapshots and not args.bots:
        raise SystemExit("Need at least --snapshots OR --bots")

    print(f"=== CIS Elo ladder eval (Phase 2) ===")
    specs = _build_player_specs(args.snapshots, args.names, args.bots)
    print(f"  {len(specs)} players: "
          f"{sum(1 for s in specs if s.kind == 'snapshot')} NN + "
          f"{sum(1 for s in specs if s.kind == 'bot')} bots")
    for i, s in enumerate(specs):
        print(f"    [{i}] {s.kind:8s} {s.name}")

    nn_specs = [s for s in specs if s.kind == "snapshot"]
    slot_map = {s.name: i for i, s in enumerate(nn_specs)}

    # Generate matchups: all-vs-all
    n = len(specs)
    matchups = list(combinations(range(n), 2))
    print(f"  {len(matchups)} matchups × {args.n_games} games "
          f"= {len(matchups) * args.n_games} total games")

    # Get cfg from first NN ckpt
    if nn_specs:
        cfg = _get_cfg_from_ckpt(nn_specs[0].ckpt)
        print(f"  cfg from {nn_specs[0].name}: temporal_context={cfg.temporal_context}, "
              f"n_moves={cfg.n_moves}")
    else:
        cfg = None  # No NN players — pure bot ladder (rare)

    # Spawn CIS server (if we have NN ckpts)
    cis_proc = None
    req_w = resp_r = ctrl_pipes = None
    cis_lock = threading.Lock()
    if nn_specs:
        print(f"\n  Spawning CIS server with {len(nn_specs)} NN slot(s)...")
        cis_proc, req_w, resp_r, ctrl_pipes = _spawn_cis_server(
            [s.ckpt for s in nn_specs],
            args.device,
            args.cis_min_batch,
            args.cis_timeout_ms,
        )

    server_cfg = resolve_server(args.server)

    # Run matchups
    matches = []
    t_start = time.time()
    for mi, (i, j) in enumerate(matchups):
        spec_a, spec_b = specs[i], specs[j]
        print(f"\n  [{mi+1}/{len(matchups)}] {spec_a.name} vs {spec_b.name}...", flush=True)
        try:
            result = run_match_cis(
                spec_a, spec_b, args.n_games, slot_map,
                req_w, resp_r, cis_lock, cfg,
                server_cfg, args.device, args.battle_format,
                args.concurrency, match_idx=mi,
            )
            matches.append(result)
            print(f"    {result['p1_wins']}W/{result['p2_wins']}L/{result['ties']}T "
                  f"({result['p1_wr']:.0%} for {spec_a.name}, {result['elapsed']}s)")
        except Exception as e:
            print(f"    ERROR in matchup {spec_a.name} vs {spec_b.name}: {e}")
            import traceback
            traceback.print_exc()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    t_elapsed = time.time() - t_start
    print(f"\n=== {len(matches)}/{len(matchups)} matchups done in {t_elapsed:.0f}s ===")

    # Compute Elos
    if matches:
        print(f"\n=== Bradley-Terry Elo (anchor: {args.anchor}={args.anchor_elo}) ===")
        elos = compute_elos(specs, matches, args.anchor, args.anchor_elo)
        for name, elo in sorted(elos.items(), key=lambda x: -x[1]):
            print(f"  {name:30s} {elo:7.1f}")

        # Save JSON
        if args.out_json:
            out_path = Path(args.out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out = {
                "config": {
                    "n_games_per_pair": args.n_games,
                    "anchor": args.anchor,
                    "anchor_elo": args.anchor_elo,
                    "format": args.battle_format,
                    "via": "eval_elo_ladder_cis",
                    "timestamp": datetime.utcnow().isoformat(),
                },
                "players": [{"name": s.name, "kind": s.kind, "ckpt": s.ckpt}
                            for s in specs],
                "matches": matches,
                "elos": elos,
            }
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"\n  Saved: {out_path}")

    # Teardown CIS
    if cis_proc is not None:
        print(f"\n  Shutting down CIS subprocess...")
        try:
            with cis_lock:
                req_w.send({"cmd": "shutdown"})
        except Exception:
            pass
        cis_proc.join(timeout=10)
        if cis_proc.is_alive():
            cis_proc.terminate()
            cis_proc.join(timeout=5)
        print(f"  CIS exited (exitcode={cis_proc.exitcode})")


if __name__ == "__main__":
    main()
