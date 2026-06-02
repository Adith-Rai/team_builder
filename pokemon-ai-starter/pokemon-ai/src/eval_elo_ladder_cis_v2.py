#!/usr/bin/env python3
"""CIS-routed Elo ladder eval — v2 (per-worker process architecture).

Architectural rewrite of eval_elo_ladder_cis.py (v1). v1 ran all matchups
sequentially in ONE process with ONE (P1, P2) account pair per matchup,
which hits poke-env / Showdown per-account-pair serialization → effective
concurrency ~2-4 → CIS gets meanq=1, 75ms pipe IPC dominates 10ms forward,
throughput collapses.

v2 mirrors training's CIS collect: spawn W worker processes, each holding
its own (P1, P2) account pair, all sharing ONE CIS server. Showdown sees
W distinct account pairs → no per-pair serialization → CIS batches across
workers → throughput matches training (~400 games/min aggregate at W=90).

Reuses v1 helpers (arch classification, pad pipeline, --add-to,
incremental JSONL, --merge-shards) — only the matchup-dispatch path changes.

Usage:
    # Single-pod multi-worker (no --shard needed; workers ARE the parallelism)
    python eval_elo_ladder_cis_v2.py \\
        --snapshots S1 S2 S3 S4 --names N1 N2 N3 N4 \\
        --n-games 500 --workers 80 \\
        --add-to era4_chain_FINAL.json \\
        --out-jsonl era4_v2_workers.jsonl \\
        --out-json era4_chain_v2.json
"""
import argparse
import asyncio
import gc
import json
import math
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from poke_env.ps_client.account_configuration import AccountConfiguration

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

# Reuse v1 helpers (arch classification, pad, ladder I/O)
from eval_elo_ladder_cis import (
    _get_cfg_from_ckpt,
    _build_player_specs,
    _classify_ckpt_arch,
    _pad_legacy_transformer_ckpt,
    _load_existing_ladder,
    _compute_new_pairs,
    _save_match_jsonl,
    _load_jsonl,
    _merge_shard_jsonls,
    compute_elos,
)


def _spawn_cis_server_multiworker(nn_ckpt_paths: List[str], device: str,
                                  min_batch: int, timeout_ms: int,
                                  n_workers: int):
    """Spawn CIS subprocess with N slots and W per-worker pipe pairs.

    Returns (cis_proc, worker_pipes, ctrl_pipes). worker_pipes is a list of
    (req_writer, resp_reader) tuples — one pair per worker, exclusive to that
    worker so no Lock needed cross-process. CIS multiplexes via mp_wait across
    all worker req-pipes and demuxes responses to the originating worker's
    resp-pipe. Same protocol as training's CIS.
    """
    ctx = _get_mp_ctx()
    worker_req_readers = []   # CIS reads from these
    worker_resp_writers = []  # CIS writes to these
    worker_pipes = []         # what we return to main (worker keeps these ends)

    for _ in range(n_workers):
        req_r, req_w = ctx.Pipe(duplex=False)
        resp_r, resp_w = ctx.Pipe(duplex=False)
        worker_req_readers.append(req_r)
        worker_resp_writers.append(resp_w)
        worker_pipes.append((req_w, resp_r))

    ctrl_req_r, ctrl_req_w = ctx.Pipe(duplex=False)
    ctrl_resp_r, ctrl_resp_w = ctx.Pipe(duplex=False)

    cis_proc = ctx.Process(
        target=_cis_main_multi,
        args=(worker_req_readers, worker_resp_writers,
              nn_ckpt_paths, device,
              True,        # fp16
              None,        # amp_dtype_name
              min_batch, timeout_ms,
              ctrl_req_r, ctrl_resp_w),
        daemon=False,
    )
    cis_proc.start()
    # Close child-side ends in parent
    for r in worker_req_readers:
        r.close()
    for w in worker_resp_writers:
        w.close()
    ctrl_req_r.close()
    ctrl_resp_w.close()

    print(f"  Waiting for CIS ready (N slots={len(nn_ckpt_paths)}, "
          f"N workers={n_workers})...", flush=True)
    # _cis_main_multi sends "ready" on EACH worker's resp pipe.
    for i, (_, resp_r) in enumerate(worker_pipes):
        ready = resp_r.recv()
        if ready.get("status") != "ready":
            cis_proc.terminate()
            raise SystemExit(f"CIS failed to come up on worker {i}: {ready}")
    print(f"  CIS ready (all {n_workers} worker pipes signalled).", flush=True)
    return cis_proc, worker_pipes, (ctrl_req_w, ctrl_resp_r)


def _make_player_for_spec_v2(spec: PlayerSpec, slot_id: int, arch: str,
                              cis_req_w, cis_resp_r, cis_lock, cfg, server_cfg,
                              account_name: str, battle_format: str,
                              concurrency: int):
    """Worker-side player constructor. Workers always run on CPU
    (BattleAgentTransformerCIS sends numpy to CIS via pipe; MLP runs on CPU
    to avoid competing with CIS for GPU). Mirrors v1 dispatch but device='cpu'.
    """
    common = dict(
        battle_format=battle_format,
        max_concurrent_battles=concurrency,
        server_configuration=server_cfg,
        team=random_pool_teambuilder(),
        account_configuration=AccountConfiguration(account_name, None),
    )
    if spec.kind == "bot":
        return spec.bot_cls(**common)
    if spec.kind == "snapshot":
        if arch == "mlp":
            from battle_agent import BattleAgent
            return BattleAgent(
                checkpoint_path=spec.ckpt,
                device="cpu",
                **common,
            )
        elif arch in ("transformer_current", "transformer_pre_pad"):
            return BattleAgentTransformerCIS(
                cis_req_writer=cis_req_w,
                cis_resp_reader=cis_resp_r,
                cis_pipe_lock=cis_lock,
                slot_id=slot_id,
                cfg=cfg,
                device="cpu",
                checkpoint_path=spec.ckpt,
                **common,
            )
        else:
            raise ValueError(f"unknown arch {arch!r} for snapshot {spec.name}")
    raise ValueError(spec.kind)


def _run_matchup_in_worker(worker_id: int, matchup_idx: int,
                           spec_a: PlayerSpec, spec_b: PlayerSpec,
                           cis_req_w, cis_resp_r, cis_lock,
                           arch_map: Dict[str, str], slot_map: Dict[str, int],
                           cfg, server_url: str, battle_format: str,
                           n_games: int, concurrency: int) -> dict:
    """One matchup run inside a worker. Builds 2 Players, plays n_games via
    battle_against, returns result dict (same shape as v1)."""
    name_a = f"W{worker_id}m{matchup_idx}a"
    name_b = f"W{worker_id}m{matchup_idx}b"

    slot_a = slot_map.get(spec_a.name, -1)
    slot_b = slot_map.get(spec_b.name, -1)
    arch_a = arch_map.get(spec_a.name, "bot")
    arch_b = arch_map.get(spec_b.name, "bot")

    server_cfg = resolve_server(server_url)

    p1 = _make_player_for_spec_v2(spec_a, slot_a, arch_a, cis_req_w, cis_resp_r,
                                   cis_lock, cfg, server_cfg, name_a,
                                   battle_format, concurrency)
    p2 = _make_player_for_spec_v2(spec_b, slot_b, arch_b, cis_req_w, cis_resp_r,
                                   cis_lock, cfg, server_cfg, name_b,
                                   battle_format, concurrency)

    t0 = time.time()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(asyncio.wait_for(
            p1.battle_against(p2, n_battles=n_games),
            timeout=max(600, n_games * 30),
        ))
    finally:
        loop.close()
    elapsed = time.time() - t0

    w1, w2 = p1.n_won_battles, p2.n_won_battles
    ties = p1.n_tied_battles
    total = w1 + w2 + ties

    for p in (p1, p2):
        try:
            p.reset_battles()
        except Exception:
            pass
    del p1, p2
    gc.collect()

    return {
        "p1": spec_a.name, "p2": spec_b.name,
        "p1_kind": spec_a.kind, "p2_kind": spec_b.kind,
        "p1_wins": w1, "p2_wins": w2, "ties": ties, "total": total,
        "p1_wr": w1 / max(1, total),
        "elapsed": round(elapsed, 1),
        "matchup_idx": matchup_idx,
        "worker_id": worker_id,
    }


def _worker_main(worker_id: int, matchup_q, result_q,
                 cis_req_w, cis_resp_r,
                 arch_map: Dict[str, str], slot_map: Dict[str, int],
                 cfg, server_url: str, battle_format: str,
                 n_games: int, concurrency: int) -> None:
    """Long-lived worker. Pulls (matchup_idx, spec_a, spec_b) tuples from
    matchup_q, runs each, pushes result dict via result_q. Exits on None
    sentinel."""
    # Per-worker thread caps (training pattern S68): workers do tiny CPU
    # tensor ops; one thread per worker fits the cgroup pids.max ceiling.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    cis_lock = threading.Lock()  # serialize the 2 Players within this worker

    while True:
        try:
            item = matchup_q.get(timeout=600)
        except Exception:
            break
        if item is None:
            break
        matchup_idx, spec_a, spec_b = item
        try:
            result = _run_matchup_in_worker(
                worker_id, matchup_idx, spec_a, spec_b,
                cis_req_w, cis_resp_r, cis_lock,
                arch_map, slot_map, cfg,
                server_url, battle_format, n_games, concurrency,
            )
            result_q.put(result)
        except Exception as e:
            tb = traceback.format_exc()
            result_q.put({
                "matchup_idx": matchup_idx,
                "p1": spec_a.name, "p2": spec_b.name,
                "p1_kind": spec_a.kind, "p2_kind": spec_b.kind,
                "p1_wins": 0, "p2_wins": 0, "ties": 0, "total": 0, "p1_wr": 0.0,
                "elapsed": 0, "worker_id": worker_id,
                "error": str(e), "traceback": tb,
            })


def main():
    p = argparse.ArgumentParser(
        description="CIS-routed Elo ladder eval — v2 (per-worker process arch)"
    )
    p.add_argument("--snapshots", nargs="+", default=[])
    p.add_argument("--names", nargs="+", default=None)
    p.add_argument("--bots", nargs="+", default=[])
    p.add_argument("--n-games", type=int, default=500)
    p.add_argument("--workers", type=int, default=60,
                   help="Number of worker processes (matches training pattern).")
    p.add_argument("--servers", nargs="+",
                   default=["ws://127.0.0.1:9000/showdown/websocket"],
                   help="One or more showdown servers; workers round-robin across.")
    p.add_argument("--device", default="cuda",
                   help="CIS server device. Workers always run CPU.")
    p.add_argument("--format", default="gen9ou", dest="battle_format")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Battles per Player (within one matchup). 1 = serial within "
                        "matchup, parallelism comes from W workers.")
    p.add_argument("--cis-min-batch", type=int, default=8)
    p.add_argument("--cis-timeout-ms", type=int, default=15)
    p.add_argument("--result-timeout-sec", type=int, default=7200,
                   help="Max wait between matchup-completion results in main. "
                        "Default 7200 (2h). First-batch wall at high N is long; "
                        "later batches arrive faster. Set generous.")
    p.add_argument("--anchor", default="SH")
    p.add_argument("--anchor-elo", type=float, default=1000.0)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-jsonl", default=None,
                   help="Per-match incremental JSONL save (crash-resume).")
    p.add_argument("--add-to", default=None,
                   help="Path to existing ladder JSON; only NEW matchups run.")
    p.add_argument("--merge-shards", nargs="+", default=None,
                   help="Merge mode (delegates to v1 _merge_shard_jsonls).")
    args = p.parse_args()

    if args.merge_shards:
        if not args.add_to or not args.out_json:
            raise SystemExit("--merge-shards requires --add-to and --out-json")
        print(f"=== MERGE MODE (v2 delegates to v1 merge) ===")
        _merge_shard_jsonls(
            [Path(p) for p in args.merge_shards],
            args.add_to, args.out_json, args.anchor, args.anchor_elo,
        )
        return

    if not args.snapshots and not args.bots:
        raise SystemExit("Need at least --snapshots OR --bots")

    print(f"=== CIS-Elo v2 ({'ADD-TO' if args.add_to else 'NEW-LADDER'}) ===")
    new_specs = _build_player_specs(args.snapshots, args.names, args.bots)
    print(f"  NEW players: {len(new_specs)}")

    if args.add_to:
        print(f"  Loading existing ladder: {args.add_to}")
        existing_specs, existing_matches = _load_existing_ladder(args.add_to)
        print(f"    {len(existing_specs)} existing players, "
              f"{len(existing_matches)} existing matches")
        existing_name_set = {s.name for s in existing_specs}
        truly_new = [s for s in new_specs if s.name not in existing_name_set]
        if not truly_new:
            raise SystemExit("All --snapshots/--bots already in ladder.")
        specs = list(existing_specs) + truly_new
        new_names_set = {s.name for s in truly_new}
        matchups = _compute_new_pairs(specs, new_names_set)
        print(f"    {len(matchups)} NEW matchups (vs {math.comb(len(specs), 2)} all-vs-all)")
    else:
        specs = new_specs
        matchups = list(combinations(range(len(specs)), 2))

    # Arch classification + padding (mirror v1)
    nn_specs_all = [s for s in specs if s.kind == "snapshot"]
    arch_map: Dict[str, str] = {}
    for s in specs:
        if s.kind == "bot":
            arch_map[s.name] = "bot"
        elif s.kind == "snapshot":
            cls = _classify_ckpt_arch(s.ckpt)
            if cls == "unknown":
                raise SystemExit(f"Could not classify arch for {s.name}")
            arch_map[s.name] = cls
    n_cur = sum(1 for v in arch_map.values() if v == "transformer_current")
    n_pre = sum(1 for v in arch_map.values() if v == "transformer_pre_pad")
    n_mlp = sum(1 for v in arch_map.values() if v == "mlp")
    n_bot = sum(1 for v in arch_map.values() if v == "bot")
    print(f"  Arch: {n_cur} current + {n_pre} pre-pad + {n_mlp} mlp + {n_bot} bot")

    transformer_nn_specs = [s for s in nn_specs_all
                            if arch_map[s.name] in ("transformer_current",
                                                    "transformer_pre_pad")]
    ref_sd_for_pad = None
    if n_pre > 0:
        ref_path = next(s.ckpt for s in nn_specs_all
                        if arch_map[s.name] == "transformer_current")
        ref_ckpt = torch.load(ref_path, map_location='cpu', weights_only=False)
        ref_sd_for_pad = ref_ckpt.get('model_state_dict',
                                       ref_ckpt.get('state_dict', ref_ckpt))
        del ref_ckpt
    pad_cache_dir = Path("data/eval/_cis_padded_cache")
    cis_specs = []
    for s in transformer_nn_specs:
        if arch_map[s.name] == "transformer_pre_pad":
            padded = _pad_legacy_transformer_ckpt(s.ckpt, ref_sd_for_pad, pad_cache_dir)
            cis_specs.append((s, padded))
        else:
            cis_specs.append((s, s.ckpt))
    slot_map = {sp.name: i for i, (sp, _) in enumerate(cis_specs)}

    # cfg from a current-arch ckpt
    cfg = None
    if cis_specs:
        cur = [(s, pth) for (s, pth) in cis_specs
               if arch_map[s.name] == "transformer_current"]
        cfg_src, cfg_path = (cur[0] if cur else cis_specs[0])
        cfg = _get_cfg_from_ckpt(cfg_path)
        print(f"  cfg from {cfg_src.name}: temporal_context={cfg.temporal_context}, "
              f"n_moves={cfg.n_moves}")

    # Crash-resume from existing JSONL
    done_pairs = set()
    if args.out_jsonl:
        prior = _load_jsonl(Path(args.out_jsonl))
        if prior:
            done_pairs = {tuple(sorted([m["p1"], m["p2"]])) for m in prior}
            pre = len(matchups)
            matchups = [(i, j) for (i, j) in matchups
                        if tuple(sorted([specs[i].name, specs[j].name])) not in done_pairs]
            print(f"  RESUME: {pre - len(matchups)} already done, "
                  f"{len(matchups)} remaining")

    print(f"  Plan: {len(matchups)} matchups × {args.n_games} games "
          f"= {len(matchups) * args.n_games} games via {args.workers} workers "
          f"across {len(args.servers)} server(s)")

    # Spawn CIS multi-worker
    ctx = _get_mp_ctx()
    cis_proc = None
    worker_pipes = []
    ctrl_pipes = None
    if cis_specs:
        cis_proc, worker_pipes, ctrl_pipes = _spawn_cis_server_multiworker(
            [pth for (_, pth) in cis_specs],
            args.device, args.cis_min_batch, args.cis_timeout_ms,
            args.workers,
        )
    else:
        # No NN players: still need fake pipes for workers (they'll never call CIS)
        for _ in range(args.workers):
            req_r, req_w = ctx.Pipe(duplex=False)
            resp_r, resp_w = ctx.Pipe(duplex=False)
            worker_pipes.append((req_w, resp_r))

    # Build matchup queue
    matchup_q = ctx.Queue()
    for mi, (i, j) in enumerate(matchups):
        matchup_q.put((mi, specs[i], specs[j]))
    for _ in range(args.workers):
        matchup_q.put(None)  # poison pills
    result_q = ctx.Queue()

    # Spawn workers
    workers = []
    for wi in range(args.workers):
        req_w, resp_r = worker_pipes[wi]
        server_url = args.servers[wi % len(args.servers)]
        wp = ctx.Process(target=_worker_main,
                         args=(wi, matchup_q, result_q,
                               req_w, resp_r,
                               arch_map, slot_map, cfg,
                               server_url, args.battle_format,
                               args.n_games, args.concurrency))
        wp.start()
        workers.append(wp)

    print(f"\n  Spawned {len(workers)} workers. Collecting results...")

    # Collect results
    results = []
    t_start = time.time()
    while len(results) < len(matchups):
        try:
            result = result_q.get(timeout=args.result_timeout_sec)
        except Exception:
            print(f"  [!] result_q timeout ({args.result_timeout_sec}s) after "
                  f"{len(results)}/{len(matchups)} matchups; aborting wait")
            break
        results.append(result)
        if "error" in result:
            print(f"  [{len(results)}/{len(matchups)}] ERROR in {result['p1']} vs "
                  f"{result['p2']}: {result['error']}", flush=True)
        else:
            elapsed = time.time() - t_start
            pct = 100 * len(results) / len(matchups)
            avg = elapsed / max(1, len(results))
            eta_min = avg * (len(matchups) - len(results)) / 60
            print(f"  [{len(results)}/{len(matchups)}] {result['p1']} vs "
                  f"{result['p2']}: {result['p1_wins']}W/{result['p2_wins']}L/"
                  f"{result['ties']}T ({result['p1_wr']:.0%}, "
                  f"{result['elapsed']}s) [{pct:.0f}%, ETA {eta_min:.1f}min]",
                  flush=True)
            if args.out_jsonl:
                _save_match_jsonl(Path(args.out_jsonl), result)

    t_elapsed = time.time() - t_start
    n_ok = sum(1 for r in results if "error" not in r)
    print(f"\n=== {n_ok}/{len(matchups)} matchups OK in {t_elapsed:.0f}s "
          f"({t_elapsed/60:.1f}min) ===")

    # Drain workers
    for wp in workers:
        wp.join(timeout=10)
        if wp.is_alive():
            wp.terminate()

    new_results_ok = [r for r in results if "error" not in r]
    if args.add_to and len(new_results_ok) == 0:
        print("  [!] No NEW matchup results completed — refusing to compute BT "
              "(would produce degenerate Elos for new players). JSON NOT saved.")
        cis_proc and cis_proc.terminate()
        return

    # Compute final Elos (combining --add-to base if present)
    if args.add_to:
        all_matches = list(existing_matches) + new_results_ok
    else:
        all_matches = new_results_ok

    if all_matches:
        elos = compute_elos(specs, all_matches, args.anchor, args.anchor_elo)
        print(f"\n=== Bradley-Terry Elo (anchor: {args.anchor}={args.anchor_elo}) ===")
        new_names_set = set()
        if args.add_to:
            new_names_set = {s.name for s in truly_new}
        for name, elo in sorted(elos.items(), key=lambda x: -x[1]):
            marker = " <-- NEW" if name in new_names_set else ""
            print(f"  {name:35s} {elo:7.1f}{marker}")

        if args.out_json:
            out_path = Path(args.out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out = {
                "config": {
                    "n_games_per_pair": args.n_games,
                    "n_workers": args.workers,
                    "anchor": args.anchor,
                    "anchor_elo": args.anchor_elo,
                    "format": args.battle_format,
                    "via": "eval_elo_ladder_cis_v2 (per-worker arch)",
                    "add_to": args.add_to,
                    "timestamp": datetime.utcnow().isoformat(),
                },
                "players": [{"name": s.name, "kind": s.kind, "ckpt": s.ckpt}
                            for s in specs],
                "matches": all_matches,
                "elos": elos,
            }
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"\n  Saved: {out_path}")

    # Shutdown CIS
    if cis_proc is not None:
        print(f"\n  Shutting down CIS...")
        try:
            ctrl_w, ctrl_r = ctrl_pipes
            ctrl_w.send({"cmd": "shutdown"})
        except Exception:
            pass
        cis_proc.join(timeout=10)
        if cis_proc.is_alive():
            cis_proc.terminate()
            cis_proc.join(timeout=5)
        print(f"  CIS exited (exitcode={cis_proc.exitcode})")


if __name__ == "__main__":
    main()
