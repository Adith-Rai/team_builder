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


def _make_player_for_spec(spec: PlayerSpec, slot_id: int, arch: str,
                          cis_req_w, cis_resp_r, cis_lock, cfg, server_cfg,
                          account_name: str, device: str,
                          battle_format: str, concurrency: int):
    """Create a Player instance for one matchup side. Arch-aware dispatch:
        - bot                                       → spec.bot_cls (rule-based)
        - snapshot, arch in (transformer_current, transformer_pre_pad)
                                                    → BattleAgentTransformerCIS (CIS slot)
        - snapshot, arch == mlp                     → BattleAgent (local V9 MLP load)

    arch is the string from _classify_ckpt_arch; for bots it's ignored.
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
            # Legacy V9 MLP — no CIS slot, instantiate BattleAgent locally
            from battle_agent import BattleAgent
            return BattleAgent(
                checkpoint_path=spec.ckpt,
                device=device,
                **common,
            )
        elif arch in ("transformer_current", "transformer_pre_pad"):
            # CIS-routed (both current and pre-pad-padded share same TransformerBattlePolicy)
            return BattleAgentTransformerCIS(
                cis_req_writer=cis_req_w,
                cis_resp_reader=cis_resp_r,
                cis_pipe_lock=cis_lock,
                slot_id=slot_id,
                cfg=cfg,
                device=device,
                checkpoint_path=spec.ckpt,
                **common,
            )
        else:
            raise ValueError(f"unknown arch {arch!r} for snapshot {spec.name}")
    raise ValueError(spec.kind)


def run_match_cis(spec_a: PlayerSpec, spec_b: PlayerSpec,
                  n_games: int, slot_map: Dict[str, int],
                  arch_map: Dict[str, str],
                  cis_req_w, cis_resp_r, cis_lock, cfg,
                  server_cfg, device: str, battle_format: str,
                  concurrency: int, match_idx: int) -> dict:
    """Run n_games between spec_a and spec_b. Arch-aware dispatch via
    arch_map: transformer arch classes go through CIS slot_map; MLP arch
    runs as a local BattleAgent (no CIS slot); bots ignore both maps.

    Returns dict with same shape as eval_elo_ladder.py run_match output.
    """
    _pid = os.getpid() % 10000
    name_a = f"E{_pid}m{match_idx}a"
    name_b = f"E{_pid}m{match_idx}b"

    slot_a = slot_map.get(spec_a.name, -1)
    slot_b = slot_map.get(spec_b.name, -1)
    arch_a = arch_map.get(spec_a.name, "bot")  # 'bot' for spec.kind=='bot'
    arch_b = arch_map.get(spec_b.name, "bot")

    p1 = _make_player_for_spec(spec_a, slot_a, arch_a, cis_req_w, cis_resp_r,
                                cis_lock, cfg, server_cfg, name_a,
                                device, battle_format, concurrency)
    p2 = _make_player_for_spec(spec_b, slot_b, arch_b, cis_req_w, cis_resp_r,
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


def _classify_ckpt_arch(path: str) -> str:
    """Identify checkpoint arch class. Returns one of:
        'transformer_current': loads via current TransformerBattlePolicy directly (CIS slot)
        'transformer_pre_pad': needs tokenizer auto-pad (type=28→29, slots=24→25, +gen_embed)
        'mlp':                 V9 BattleAgent arch (NOT CIS; use subprocess Player)
        'unknown':             can't classify (skipped with warning)
    """
    try:
        sd = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        elif isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        # MLP signature: no tokenizer keys at all
        has_tokenizer = any(k.startswith('tokenizer.') for k in sd.keys())
        if not has_tokenizer:
            return 'mlp'
        type_w = sd.get('tokenizer.type_id_embed.weight')
        slot_w = sd.get('tokenizer.pokemon_slot_embed.weight')
        gen_w  = sd.get('tokenizer.gen_embed.weight')
        if type_w is None or slot_w is None:
            return 'unknown'
        t_shape = tuple(type_w.shape)
        s_shape = tuple(slot_w.shape)
        # Current arch: [29, *], [25, *], gen_embed present
        if t_shape[0] == 29 and s_shape[0] == 25 and gen_w is not None:
            return 'transformer_current'
        # Pre-pad: [28, *], [24, *], no gen_embed
        if t_shape[0] == 28 and s_shape[0] == 24 and gen_w is None:
            return 'transformer_pre_pad'
        return 'unknown'
    except Exception:
        return 'unknown'


def _pad_legacy_transformer_ckpt(in_path: str, ref_model_sd: dict, cache_dir: Path) -> str:
    """Auto-pad a pre-pad-transformer ckpt to current arch. Returns path to padded ckpt.

    Pads:
      tokenizer.type_id_embed.weight: [28, D] → [29, D]    (zero-init row 28)
      tokenizer.pokemon_slot_embed.weight: [24, D] → [25, D] (zero-init row 24)
      tokenizer.gen_embed.weight: missing → [N_gens, D_gen]  (zero-init, gens default)

    Caches result in cache_dir so re-pad is one-shot. Output ckpt is functionally
    equivalent for inference on inputs that don't use the new dims (which the
    pre-pad model never saw anyway).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    in_p = Path(in_path)
    cached = cache_dir / f"{in_p.stem}__padded.pt"
    if cached.exists():
        return str(cached)
    print(f"  [pad] padding pre-pad ckpt: {in_p.name} -> {cached.name}", flush=True)
    ckpt = torch.load(in_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    # Pad type embedding row 28
    if 'tokenizer.type_id_embed.weight' in sd:
        old = sd['tokenizer.type_id_embed.weight']
        if old.shape[0] == 28:
            sd['tokenizer.type_id_embed.weight'] = torch.cat(
                [old, torch.zeros(1, old.shape[1], dtype=old.dtype)], dim=0)
    # Pad slot embedding row 24
    if 'tokenizer.pokemon_slot_embed.weight' in sd:
        old = sd['tokenizer.pokemon_slot_embed.weight']
        if old.shape[0] == 24:
            sd['tokenizer.pokemon_slot_embed.weight'] = torch.cat(
                [old, torch.zeros(1, old.shape[1], dtype=old.dtype)], dim=0)
    # Add gen_embed if missing — derive shape from ref model
    if 'tokenizer.gen_embed.weight' not in sd and 'tokenizer.gen_embed.weight' in ref_model_sd:
        ref_shape = ref_model_sd['tokenizer.gen_embed.weight'].shape
        sd['tokenizer.gen_embed.weight'] = torch.zeros(ref_shape, dtype=ref_model_sd['tokenizer.gen_embed.weight'].dtype)
    # Save padded
    if 'model_state_dict' in ckpt:
        ckpt['model_state_dict'] = sd
    elif 'state_dict' in ckpt:
        ckpt['state_dict'] = sd
    else:
        ckpt = sd  # raw state_dict
    torch.save(ckpt, cached)
    return str(cached)


def _load_existing_ladder(path: str):
    """Load existing era4_chain JSON. Returns (existing_specs, existing_matches).

    Bot specs are reconstructed via ALL_BOTS registry lookup.
    Snapshot specs use the ckpt path stored in JSON.
    """
    with open(path) as f:
        data = json.load(f)
    specs = []
    for p in data["players"]:
        if p["kind"] == "snapshot":
            specs.append(PlayerSpec(kind="snapshot", name=p["name"], ckpt=p["ckpt"]))
        elif p["kind"] == "bot":
            bot_cls = ALL_BOTS.get(p["name"])
            if bot_cls is None:
                print(f"  [WARN] bot {p['name']!r} not in ALL_BOTS registry, skipping")
                continue
            specs.append(PlayerSpec(kind="bot", name=p["name"], bot_cls=bot_cls))
    return specs, data.get("matches", [])


def _compute_new_pairs(all_specs: List[PlayerSpec], new_names: set) -> List[Tuple[int, int]]:
    """Return list of (i,j) pairs where at least one of i,j is a new player.

    Pairs are canonicalized (i < j) and deduped. Used for --add-to mode:
    only NEW matchups need to be run.
    """
    name_to_idx = {s.name: i for i, s in enumerate(all_specs)}
    pairs_set = set()
    new_indices = [i for i, s in enumerate(all_specs) if s.name in new_names]
    for ni in new_indices:
        for j in range(len(all_specs)):
            if ni == j:
                continue
            a, b = min(ni, j), max(ni, j)
            pairs_set.add((a, b))
    return sorted(pairs_set)


def _apply_shard(pairs: List, shard_str: str) -> List:
    """Apply --shard i/N partition. Returns subset owned by shard i.

    Deterministic round-robin: pair k → shard (k % N). Matches
    eval_elo_ladder.py shard convention.
    """
    if not shard_str:
        return pairs
    try:
        i, n = (int(x) for x in shard_str.split("/"))
    except Exception:
        raise SystemExit(f"--shard must be 'i/N' format, got {shard_str!r}")
    if not (0 <= i < n):
        raise SystemExit(f"--shard i={i} must be in [0, N={n})")
    sub = [p for k, p in enumerate(pairs) if k % n == i]
    print(f"  [shard {i}/{n}]: {len(sub)} of {len(pairs)} matchups")
    return sub


def _save_match_jsonl(path: Path, match: dict):
    """Append one match dict as JSONL line for crash-resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(match) + "\n")


def _load_jsonl(path: Path) -> List[dict]:
    """Load JSONL of previously-run matches (for resume + shard merge)."""
    if not path.exists():
        return []
    matches = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                matches.append(json.loads(line))
            except Exception:
                continue
    return matches


def _merge_shard_jsonls(jsonl_paths: List[Path], add_to_path: str,
                       out_json: str, anchor: str, anchor_elo: float):
    """Merge N shard JSONLs into final combined JSON.

    Loads base ladder (from --add-to), appends all shard matches, refits BT,
    writes final JSON in eval_elo_ladder.py-compatible format.
    """
    base_specs, base_matches = _load_existing_ladder(add_to_path)
    all_matches = list(base_matches)
    seen_pairs = {tuple(sorted([m["p1"], m["p2"]])) for m in base_matches}

    for sp in jsonl_paths:
        loaded = _load_jsonl(sp)
        for m in loaded:
            key = tuple(sorted([m["p1"], m["p2"]]))
            if key in seen_pairs:
                continue  # dedupe — base already has this matchup
            all_matches.append(m)
            seen_pairs.add(key)
        print(f"  [merge] {sp.name}: {len(loaded)} matches loaded")

    print(f"  [merge] Total matches: {len(all_matches)} ({len(all_matches) - len(base_matches)} new)")

    # Need to know full player list — derived from matches + base specs
    all_names = set()
    for m in all_matches:
        all_names.add(m["p1"])
        all_names.add(m["p2"])
    # Reconstruct PlayerSpec list — base_specs + any inferred from matches
    base_name_set = {s.name for s in base_specs}
    final_specs = list(base_specs)
    # New players (from shard matches) need to be added — infer kind from match data
    for m in all_matches:
        for nm, kind_key in ((m["p1"], "p1_kind"), (m["p2"], "p2_kind")):
            if nm in base_name_set:
                continue
            kind = m.get(kind_key, "snapshot")
            # We don't have ckpt path in match — caller should pass --names+--snapshots
            # For now: just add with placeholder ckpt; user-provided --snapshots will be
            # passed separately to fix this.
            if not any(s.name == nm for s in final_specs):
                if kind == "bot":
                    bot_cls = ALL_BOTS.get(nm)
                    final_specs.append(PlayerSpec(kind="bot", name=nm, bot_cls=bot_cls))
                else:
                    final_specs.append(PlayerSpec(kind="snapshot", name=nm, ckpt=None))

    elos = compute_elos(final_specs, all_matches, anchor, anchor_elo)
    print(f"\n=== MERGED ELO LADDER (anchor: {anchor}={anchor_elo}) ===")
    for name, elo in sorted(elos.items(), key=lambda x: -x[1]):
        marker = " <-- NEW" if name not in base_name_set else ""
        print(f"  {name:30s} {elo:7.1f}{marker}")

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final = {
        "config": {
            "merged_from": add_to_path,
            "anchor": anchor,
            "anchor_elo": anchor_elo,
            "n_players": len(final_specs),
            "n_matches": len(all_matches),
            "via": "eval_elo_ladder_cis (Phase 3+4A merge)",
            "timestamp": datetime.utcnow().isoformat(),
        },
        "players": [{"name": s.name, "kind": s.kind, "ckpt": s.ckpt}
                    for s in final_specs],
        "matches": all_matches,
        "elos": elos,
    }
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n  Saved merged ladder: {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="CIS-routed Elo ladder eval (Phase 2 + 3 + 4A: --add-to + --shard)"
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
    # Phase 3: incremental add to existing ladder
    p.add_argument("--add-to", default=None,
                   help="Path to existing ladder JSON (e.g. era4_chain_FINAL.json). "
                        "Loads existing players + matches; runs ONLY new pairs "
                        "(new × all). Output JSON contains merged matches + refit Elos. "
                        "MUTUALLY EXCLUSIVE with --merge-shards.")
    # Phase 4A: sharding for parallel ladder eval
    p.add_argument("--shard", default=None,
                   help="Shard partition 'i/N' (i in [0,N)). Each shard handles "
                        "subset of matchups via deterministic round-robin (pair k → "
                        "shard k%%N). Per-shard JSONL incremental save to "
                        "<out_jsonl> for crash-resume + merging.")
    p.add_argument("--out-jsonl", default=None,
                   help="Per-shard incremental JSONL output (one match per line). "
                        "REQUIRED for --shard mode. Used by --merge-shards to combine.")
    # Phase 4A: merge shard outputs
    p.add_argument("--merge-shards", nargs="+", default=None,
                   help="Paths to N shard JSONL files. Merges all into final ladder, "
                        "refits BT, saves to --out-json. Requires --add-to (base ladder). "
                        "MUTUALLY EXCLUSIVE with --shard.")
    args = p.parse_args()

    # ── Mode handling: --merge-shards / --add-to / --shard ──
    if args.merge_shards and args.shard:
        raise SystemExit("--merge-shards and --shard are mutually exclusive")
    if args.merge_shards:
        if not args.add_to:
            raise SystemExit("--merge-shards requires --add-to (base ladder)")
        if not args.out_json:
            raise SystemExit("--merge-shards requires --out-json (merged output)")
        print(f"=== MERGE MODE ===")
        _merge_shard_jsonls(
            [Path(p) for p in args.merge_shards],
            args.add_to, args.out_json,
            args.anchor, args.anchor_elo,
        )
        return  # done — no CIS spawn needed

    if args.shard and not args.out_jsonl:
        raise SystemExit("--shard mode requires --out-jsonl (per-shard incremental save)")

    if not args.snapshots and not args.bots:
        raise SystemExit("Need at least --snapshots OR --bots")

    print(f"=== CIS Elo ladder eval ({'ADD-TO' if args.add_to else 'NEW-LADDER'}) ===")
    new_specs = _build_player_specs(args.snapshots, args.names, args.bots)
    print(f"  NEW players: {len(new_specs)} "
          f"({sum(1 for s in new_specs if s.kind == 'snapshot')} NN + "
          f"{sum(1 for s in new_specs if s.kind == 'bot')} bots)")
    for i, s in enumerate(new_specs):
        print(f"    [{i}] {s.kind:8s} {s.name}")

    # If --add-to: load existing, build combined specs, only run NEW pairs
    if args.add_to:
        print(f"\n  Loading existing ladder: {args.add_to}")
        existing_specs, existing_matches = _load_existing_ladder(args.add_to)
        print(f"    {len(existing_specs)} existing players, {len(existing_matches)} existing matches")
        existing_name_set = {s.name for s in existing_specs}
        new_name_set = {s.name for s in new_specs}
        # Dedup: drop new players already in existing
        truly_new = [s for s in new_specs if s.name not in existing_name_set]
        if len(truly_new) < len(new_specs):
            dropped = [s.name for s in new_specs if s.name in existing_name_set]
            print(f"    SKIP {len(dropped)} already-in-ladder: {dropped}")
        if not truly_new:
            raise SystemExit("All --snapshots / --bots already in ladder. Nothing to do.")
        # Combined spec list: existing + new
        specs = list(existing_specs) + truly_new
        # Build matchup queue: only NEW × ALL pairs
        new_names_set = {s.name for s in truly_new}
        matchups = _compute_new_pairs(specs, new_names_set)
        print(f"    {len(matchups)} NEW matchups (vs {len(combinations(range(len(specs)), 2))} all-vs-all)")
    else:
        specs = new_specs
        n = len(specs)
        matchups = list(combinations(range(n), 2))

    # ── Arch classification: snapshot ckpts → transformer_current / transformer_pre_pad / mlp ──
    # MLP arch can't go through CIS (different model class); pre-pad transformer can after padding.
    nn_specs_all = [s for s in specs if s.kind == "snapshot"]
    arch_map: Dict[str, str] = {}  # name → arch class ('transformer_current' | 'transformer_pre_pad' | 'mlp')
    for s in specs:
        if s.kind == "bot":
            arch_map[s.name] = "bot"
        elif s.kind == "snapshot":
            cls = _classify_ckpt_arch(s.ckpt)
            if cls == "unknown":
                raise SystemExit(f"Could not classify arch for {s.name} ({s.ckpt}). "
                                 f"Inspect ckpt manually.")
            arch_map[s.name] = cls
    n_cur = sum(1 for v in arch_map.values() if v == "transformer_current")
    n_pre = sum(1 for v in arch_map.values() if v == "transformer_pre_pad")
    n_mlp = sum(1 for v in arch_map.values() if v == "mlp")
    n_bot = sum(1 for v in arch_map.values() if v == "bot")
    print(f"  Arch classification: {n_cur} current + {n_pre} pre-pad-transformer "
          f"+ {n_mlp} mlp + {n_bot} bot")

    # Pre-pad legacy transformer ckpts. Need a reference state_dict for gen_embed shape.
    # Pick a current-arch ckpt as reference.
    transformer_nn_specs = [s for s in nn_specs_all
                            if arch_map[s.name] in ("transformer_current", "transformer_pre_pad")]
    ref_sd_for_pad = None
    if n_pre > 0:
        ref_path = next(s.ckpt for s in nn_specs_all if arch_map[s.name] == "transformer_current")
        ref_ckpt = torch.load(ref_path, map_location='cpu', weights_only=False)
        ref_sd_for_pad = ref_ckpt.get('model_state_dict', ref_ckpt.get('state_dict', ref_ckpt))
        del ref_ckpt
    pad_cache_dir = Path("data/eval/_cis_padded_cache")
    # Build CIS ckpt list (only transformer arch); pre-pad ckpts get padded path
    cis_specs = []  # list of (spec, ckpt_path_for_cis)
    for s in transformer_nn_specs:
        if arch_map[s.name] == "transformer_pre_pad":
            padded = _pad_legacy_transformer_ckpt(s.ckpt, ref_sd_for_pad, pad_cache_dir)
            cis_specs.append((s, padded))
        else:
            cis_specs.append((s, s.ckpt))

    # slot_map: only transformer-arch snapshots get a CIS slot. MLP arch = -1.
    slot_map = {sp.name: i for i, (sp, _) in enumerate(cis_specs)}

    # Apply --shard filter
    matchups = _apply_shard(matchups, args.shard)

    print(f"  {len(matchups)} matchups × {args.n_games} games "
          f"= {len(matchups) * args.n_games} total games this run")
    if args.shard:
        print(f"  Per-match JSONL: {args.out_jsonl}")
        # Crash-resume: skip already-done pairs from prior shard run
        prior = _load_jsonl(Path(args.out_jsonl))
        if prior:
            done_pairs = {tuple(sorted([m["p1"], m["p2"]])) for m in prior}
            name_to_idx = {s.name: k for k, s in enumerate(specs)}
            matchups_pre = matchups
            matchups = [(i, j) for (i, j) in matchups
                        if tuple(sorted([specs[i].name, specs[j].name])) not in done_pairs]
            print(f"  RESUME: {len(matchups_pre) - len(matchups)} matchups already done in prior run, "
                  f"{len(matchups)} remaining")

    # Get cfg from a CURRENT-arch transformer ckpt (prefer current; pre-pad cfg may
    # lack new tokenizer fields). Used by all transformer players for input tokenization.
    cfg = None
    cfg_src = None
    if cis_specs:
        cur_specs = [(s, p) for (s, p) in cis_specs if arch_map[s.name] == "transformer_current"]
        if cur_specs:
            cfg_src = cur_specs[0][0]
            cfg = _get_cfg_from_ckpt(cur_specs[0][1])
        else:
            cfg_src = cis_specs[0][0]
            cfg = _get_cfg_from_ckpt(cis_specs[0][1])
        print(f"  cfg from {cfg_src.name}: temporal_context={cfg.temporal_context}, "
              f"n_moves={cfg.n_moves}")

    # Spawn CIS server (only if any transformer-arch ckpts; MLP/bot-only ladders skip it)
    cis_proc = None
    req_w = resp_r = ctrl_pipes = None
    cis_lock = threading.Lock()
    if cis_specs:
        print(f"\n  Spawning CIS server with {len(cis_specs)} transformer slot(s) "
              f"({sum(1 for s,_ in cis_specs if arch_map[s.name]=='transformer_pre_pad')} pre-pad-padded)...")
        cis_proc, req_w, resp_r, ctrl_pipes = _spawn_cis_server(
            [p for (_, p) in cis_specs],
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
                spec_a, spec_b, args.n_games, slot_map, arch_map,
                req_w, resp_r, cis_lock, cfg,
                server_cfg, args.device, args.battle_format,
                args.concurrency, match_idx=mi,
            )
            matches.append(result)
            print(f"    {result['p1_wins']}W/{result['p2_wins']}L/{result['ties']}T "
                  f"({result['p1_wr']:.0%} for {spec_a.name}, {result['elapsed']}s)")
            # Incremental JSONL save for --shard mode (crash-resume + merge)
            if args.out_jsonl:
                _save_match_jsonl(Path(args.out_jsonl), result)
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
