#!/usr/bin/env python3
"""eval_mm_vs_smartbots.py — MM vs entities (smart-bots AND/OR our checkpoints).

Spawns each Metamon trained model as a subprocess (via metamon_accept_serve.py),
then runs each test entity (smart bot or our model checkpoint) against it for
N games. Outputs a WR matrix.

Two complementary use cases:
  (1) MM-vs-smart-bots: calibrate the smart-bot anchor vs known external models
      (is the 70-74% smart_avg ceiling really our model's plateau, or is it
      a smart-bot-side cap that any decent trained model hits?)
  (2) MM-vs-our-checkpoints: direct H2H of our models vs MM models on the
      SAME teams smart_avg uses (closes the BT triangle so we can place all
      three — smart_bots, MMs, our_models — on the same Elo scale)

Both sides use the same 16 metamon-competitive teams (random per battle per side),
matching smart_avg eval setup exactly so WRs are directly comparable.

Usage:
    # Original mode: 4 smart bots vs all MMs
    python eval_mm_vs_smartbots.py --n-games 500

    # Subset of MMs
    python eval_mm_vs_smartbots.py --n-games 500 --models LargeRL Minikazam

    # NEW: our model checkpoint(s) vs all MMs (snapshot LABEL=PATH)
    python eval_mm_vs_smartbots.py --n-games 500 --bots none \\
        --snapshots POST_INIT_iter139=data/models/.../snapshot_0139.pt \\
                    fbv2_iter149=data/models/.../snapshot_0149.pt

    # Both: bots AND snapshots vs all MMs
    python eval_mm_vs_smartbots.py --n-games 500 \\
        --snapshots POST_INIT_iter139=data/models/.../snapshot_0139.pt

Output:
    Console WR matrix + JSON saved to /tmp/mm_vs_smartbots_<timestamp>.json
"""
import argparse
import asyncio
import json
import os
import subprocess
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from poke_env.ps_client.account_configuration import AccountConfiguration

# Reuse bot registry + teambuilder + server resolver from eval_elo_ladder
from eval_elo_ladder import ALL_BOTS, random_pool_teambuilder, resolve_server

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # team_builder/
SRC_DIR = Path(__file__).resolve().parent
METAMON_CACHE = str(PROJECT_ROOT / "metamon_cache")

# Default MM set matches fishbowl_lr1e-4_v1.yaml (the 3 trained models) +
# Minikazam (default in external_adapters.py — small RNN baseline meant to be
# in the pool but not currently in the fishbowl yaml; eval-tier unknown).
DEFAULT_MMS = [
    ("LargeRL", "MMevalLargeRL", 9000),
    ("MediumRL_Aug", "MMevalMediumRLAug", 9001),
    ("SyntheticRLV2", "MMevalSyntheticRLV2", 9002),
    ("Minikazam", "MMevalMinikazam", 9003),
]

SMART_BOTS = ["SH", "SmartDmg", "Tactical", "Strategic"]


def spawn_mm(model: str, username: str, port: int,
             team_mode: str = "competitive",
             team_queue_dir: str = None) -> Tuple[subprocess.Popen, str]:
    """team_mode ∈ {'competitive', 'queue'}.
      'competitive': metamon's 16 curated teams (--team-set competitive)
      'queue': MM consumes packed teams from team_queue_dir (--team-queue);
               caller is responsible for keeping the queue populated.
    """
    venv_python = PROJECT_ROOT / "metamon_venv" / "bin" / "python"
    if not venv_python.exists():
        raise FileNotFoundError(f"metamon_venv missing at {venv_python}")
    serve_script = SRC_DIR / "metamon_accept_serve.py"
    if not serve_script.exists():
        raise FileNotFoundError(f"metamon_accept_serve.py missing at {serve_script}")
    cmd = [
        str(venv_python),
        str(serve_script),
        "--model", model,
        "--username", username,
        "--server-port", str(port),
        "--format", "gen9ou",
        "--num-battles", "100000",
        "--temperature", "0.01",
    ]
    if team_mode == "queue":
        if not team_queue_dir:
            raise ValueError("team_mode='queue' requires team_queue_dir")
        cmd += ["--team-queue", team_queue_dir]
    else:
        cmd += ["--team-set", "competitive"]
    env = {**os.environ, "METAMON_CACHE_DIR": METAMON_CACHE}
    log_path = f"/tmp/mm_eval_{username}.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc, log_path


# Cached teambuilders to avoid reloading procedural pool 60+ times
_TB_CACHE: Dict[str, object] = {}

def _get_teambuilder(team_set: str = "metamon-competitive",
                     procedural_teams_path: str = "/workspace/raw_data/pokemon_usage/2024-04"):
    """Get teambuilder for the bot/snapshot side.

    metamon-competitive: 16 curated teams (random per battle).
                         Matches smart_avg eval. Default.
    procedural:          ProceduralTeambuilder from Smogon usage data.
                         Fresh team per battle. Matches training-time
                         team distribution exactly. Use this to test
                         whether the training-vs-eval WR gap is teams.
    """
    cache_key = f"{team_set}::{procedural_teams_path}"
    if cache_key in _TB_CACHE:
        return _TB_CACHE[cache_key]
    if team_set == "metamon-competitive":
        from eval_metamon_competitive import MetamonCompetitiveTeambuilder
        tb = MetamonCompetitiveTeambuilder()
    elif team_set == "procedural":
        from team_generator import procedural_teambuilder
        tb = procedural_teambuilder(procedural_teams_path)
    else:
        raise ValueError(f"Unknown team_set {team_set!r}; "
                         f"use 'metamon-competitive' or 'procedural'")
    _TB_CACHE[cache_key] = tb
    return tb


def prefill_team_queue(team_queue_dir: str, n_teams: int,
                       procedural_teams_path: str) -> None:
    """Pre-populate MM's team queue with N packed procedural teams.

    MM consumes one team per battle via QueueTeambuilder. We pre-fill before
    sending challenges so MM never blocks waiting. n_teams should be >=
    n_games + small buffer.
    """
    from team_generator import enqueue_team
    Path(team_queue_dir).mkdir(parents=True, exist_ok=True)
    tb = _get_teambuilder("procedural", procedural_teams_path)
    for _ in range(n_teams):
        enqueue_team(team_queue_dir, tb.yield_team())


def make_entity_player(entity_name: str, entity_kind: str, ckpt_path: str,
                       cached_ckpt, port: int, account: str, concurrency: int,
                       device: str = "cuda",
                       team_set: str = "metamon-competitive",
                       procedural_teams_path: str = "/workspace/raw_data/pokemon_usage/2024-04"):
    """Build a Player for either a bot or a model snapshot.

    entity_kind ∈ {"bot", "snapshot"}.
    For bots: ckpt_path/cached_ckpt are ignored; entity_name is a key in ALL_BOTS.
    For snapshots: ckpt_path is the model file; cached_ckpt is the pre-loaded dict
                   (avoids redundant disk reads across MM matchups).
    """
    server_cfg = resolve_server(f"ws://127.0.0.1:{port}/showdown/websocket")
    common = dict(
        battle_format="gen9ou",
        max_concurrent_battles=concurrency,
        server_configuration=server_cfg,
        team=_get_teambuilder(team_set, procedural_teams_path),
        account_configuration=AccountConfiguration(account, None),
    )
    if entity_kind == "bot":
        cls = ALL_BOTS[entity_name]
        return cls(**common)
    elif entity_kind == "snapshot":
        from battle_agent_transformer import is_transformer_checkpoint, BattleAgentTransformer
        from battle_agent import BattleAgent
        AgentClass = BattleAgentTransformer if is_transformer_checkpoint(cached_ckpt) else BattleAgent
        return AgentClass(
            checkpoint_path=ckpt_path,
            _cached_ckpt=cached_ckpt,
            device=device,
            **common,
        )
    raise ValueError(f"Unknown entity_kind {entity_kind!r}")


# Backwards-compat shim for any caller (kept for the original "bot" code path)
def make_bot(bot_name: str, port: int, account: str, concurrency: int):
    return make_entity_player(bot_name, "bot", "", None, port, account, concurrency)


async def run_pair(entity_name: str, entity_kind: str, ckpt_path: str, cached_ckpt,
                   mm_username: str, port: int, n_games: int,
                   bot_concurrency: int, match_idx: int,
                   device: str = "cuda",
                   team_set: str = "metamon-competitive",
                   procedural_teams_path: str = "/workspace/raw_data/pokemon_usage/2024-04") -> dict:
    account = f"E{os.getpid() % 9999}m{match_idx}{entity_name[:4]}"
    bot = make_entity_player(entity_name, entity_kind, ckpt_path, cached_ckpt,
                              port, account, bot_concurrency, device,
                              team_set, procedural_teams_path)

    t0 = time.time()
    # send_challenges returns when all are queued/sent. Then poll for completion.
    await asyncio.wait_for(
        bot.send_challenges(mm_username, n_games), timeout=600
    )
    # Wait for all battles to finish (poll bot.n_finished_battles)
    deadline = time.time() + max(600, n_games * 30)
    while bot.n_finished_battles < n_games and time.time() < deadline:
        await asyncio.sleep(2.0)
    elapsed = time.time() - t0

    wins = bot.n_won_battles
    losses = bot.n_lost_battles
    ties = bot.n_tied_battles
    total = wins + losses + ties

    try:
        bot.reset_battles()
    except Exception:
        pass

    return {
        "bot": entity_name, "bot_kind": entity_kind, "mm": mm_username,
        "bot_wins": wins, "mm_wins": losses, "ties": ties, "total": total,
        "bot_wr": wins / max(1, total),
        "elapsed_s": round(elapsed, 1),
    }


async def main_async(n_games: int, bot_concurrency: int, mms: List[Tuple[str, str, int]],
                     entities: List[Tuple[str, str, str]], cached_ckpts: dict,
                     out_json: str, device: str = "cuda",
                     mm_startup_wait: int = 60,
                     team_set: str = "metamon-competitive",
                     procedural_teams_path: str = "/workspace/raw_data/pokemon_usage/2024-04") -> None:
    """entities is a list of (name, kind, path) tuples. kind ∈ {"bot", "snapshot"}.
    For bot: path is "" (unused; name is the ALL_BOTS key).
    For snapshot: path is the .pt file; cached_ckpts[name] is the pre-loaded dict.
    """
    results: List[dict] = []
    overall_t0 = time.time()
    match_idx = 0

    # Per-MM team queue dir (only used for team_set='procedural')
    n_entities = len(entities)
    teams_needed_per_mm = n_games * n_entities + 50  # buffer

    for mm_model, mm_username, mm_port in mms:
        print(f"\n=== Spawning Metamon {mm_model} as {mm_username} on port {mm_port} ===",
              flush=True)
        team_queue_dir = None
        if team_set == "procedural":
            team_queue_dir = f"/tmp/mm_eval_queue_{mm_username}"
            # Clean stale queue from prior runs
            import shutil
            shutil.rmtree(team_queue_dir, ignore_errors=True)
            Path(team_queue_dir).mkdir(parents=True, exist_ok=True)
            print(f"  [procedural] pre-filling queue {team_queue_dir} with "
                  f"{teams_needed_per_mm} teams...", flush=True)
            prefill_team_queue(team_queue_dir, teams_needed_per_mm, procedural_teams_path)
            print(f"  [procedural] queue filled", flush=True)
            proc, log_path = spawn_mm(mm_model, mm_username, mm_port,
                                       team_mode="queue", team_queue_dir=team_queue_dir)
        else:
            proc, log_path = spawn_mm(mm_model, mm_username, mm_port,
                                       team_mode="competitive")
        print(f"  PID={proc.pid}, log={log_path}", flush=True)
        # Wait for MM to log in and start accepting (longer = safer for big models)
        # SyntheticRLV2 is 200M params and needs ~30-60s; 15s timed out the first run.
        await asyncio.sleep(mm_startup_wait)
        if proc.poll() is not None:
            print(f"  [!] {mm_model} subprocess died early (exitcode={proc.returncode}); skip",
                  flush=True)
            try:
                with open(log_path) as f:
                    print("  --- log tail ---")
                    for line in f.readlines()[-15:]:
                        print(f"    {line.rstrip()}")
            except Exception:
                pass
            continue

        try:
            for entity_name, entity_kind, ckpt_path in entities:
                match_idx += 1
                cached = cached_ckpts.get(entity_name) if entity_kind == "snapshot" else None
                kind_tag = "[bot]" if entity_kind == "bot" else "[snap]"
                print(f"  {kind_tag} [{entity_name} vs {mm_model}] starting (n_games={n_games})...",
                      flush=True)
                try:
                    r = await run_pair(entity_name, entity_kind, ckpt_path, cached,
                                       mm_username, mm_port,
                                       n_games, bot_concurrency, match_idx, device,
                                       team_set, procedural_teams_path)
                    results.append(r)
                    print(f"    {entity_name} {r['bot_wins']}W/{r['mm_wins']}L/"
                          f"{r['ties']}T ({100*r['bot_wr']:.1f}% for {entity_name}, "
                          f"{r['elapsed_s']}s)", flush=True)
                    # Incremental save (crash resume)
                    with open(out_json, "w") as f:
                        json.dump({"results": results,
                                   "timestamp": datetime.utcnow().isoformat()},
                                  f, indent=2)
                except Exception as e:
                    print(f"    ERROR: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
        finally:
            print(f"  Terminating {mm_model}...", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    total_wall = time.time() - overall_t0
    print(f"\n=== TOTAL WALL: {total_wall:.0f}s ({total_wall/60:.1f} min) ===")

    # Print WR matrix — rows are entities, columns are MMs
    print(f"\n=== WR MATRIX  (% entity wins, n_games={n_games}) ===")
    header = f"{'ENTITY':22s}"
    for mm_model, _, _ in mms:
        header += f" {mm_model[:14]:>16s}"
    print(header)
    for entity_name, entity_kind, _ in entities:
        tag = "[B]" if entity_kind == "bot" else "[S]"
        row = f"{tag} {entity_name:18s}"
        for mm_model, mm_username, _ in mms:
            found = next((r for r in results
                          if r["bot"] == entity_name and r["mm"] == mm_username), None)
            if found and found["total"] > 0:
                wr = 100 * found["bot_wr"]
                row += f" {wr:>14.1f}% "
            else:
                row += f" {'N/A':>15s} "
        print(row)
    print()
    print(f"Saved JSON: {out_json}")


def _parse_snapshot_arg(s: str):
    """LABEL=PATH or just PATH (label = stem)."""
    if "=" in s:
        label, path = s.split("=", 1)
        return label.strip(), path.strip()
    return Path(s).stem, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=500)
    ap.add_argument("--bot-concurrency", type=int, default=8,
                    help="Battles in flight per entity-MM pair (Showdown caps per-pair)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Subset of MMs to test (e.g. LargeRL SyntheticRLV2)")
    ap.add_argument("--bots", nargs="+", default=SMART_BOTS,
                    help=f"Smart bots to test. Default: {SMART_BOTS}. Pass 'none' "
                         f"to skip bots entirely (snapshot-only mode).")
    ap.add_argument("--snapshots", nargs="+", default=[],
                    help="Model checkpoints to test alongside (or instead of) bots. "
                         "Format: 'LABEL=PATH' or just 'PATH'. Each runs vs all MMs.")
    ap.add_argument("--device", default="cuda",
                    help="Device for loading our checkpoint models. MMs run in their own "
                         "subprocess and ignore this.")
    ap.add_argument("--team-set", choices=["metamon-competitive", "procedural"],
                    default="metamon-competitive",
                    help="Team distribution for BOTH sides. 'metamon-competitive' "
                         "(default) = 16 curated teams, matches smart_avg eval. "
                         "'procedural' = ProceduralTeambuilder from Smogon usage data, "
                         "matches TRAINING-TIME teams. MM side is fed packed teams "
                         "via --team-queue when procedural is selected.")
    ap.add_argument("--procedural-teams-path", default="/workspace/raw_data/pokemon_usage/2024-04",
                    help="Path to Smogon usage data for ProceduralTeambuilder. "
                         "Only used when --team-set procedural.")
    ap.add_argument("--mm-startup-wait", type=int, default=60,
                    help="Seconds to wait after spawning each MM subprocess before "
                         "sending challenges. SyntheticRLV2 (200M params) needs ~30-60s; "
                         "smaller MMs need ~10-15s. Default 60 (safe for all).")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    mms = DEFAULT_MMS
    if args.models:
        mms = [m for m in DEFAULT_MMS if m[0] in args.models]
        if not mms:
            raise SystemExit(f"No MMs matched {args.models}. Available: "
                             f"{[m[0] for m in DEFAULT_MMS]}")

    # Build entity list: bots + snapshots
    entities: List[Tuple[str, str, str]] = []
    bot_list = [] if args.bots == ["none"] else args.bots
    for b in bot_list:
        if b not in ALL_BOTS:
            raise SystemExit(f"Unknown bot {b!r}. Available: {sorted(ALL_BOTS.keys())}")
        entities.append((b, "bot", ""))

    cached_ckpts: Dict[str, object] = {}
    if args.snapshots:
        import torch
        for s in args.snapshots:
            label, path = _parse_snapshot_arg(s)
            if not Path(path).exists():
                raise SystemExit(f"Snapshot not found: {path}")
            print(f"  [preload] {label} <- {path}", flush=True)
            cached_ckpts[label] = torch.load(path, map_location=args.device,
                                              weights_only=False)
            entities.append((label, "snapshot", path))

    if not entities:
        raise SystemExit("Need at least one bot or snapshot to test. "
                         "Either keep default --bots OR pass --snapshots.")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_json = args.out_json or f"/tmp/mm_vs_smartbots_{ts}.json"

    print(f"=== MM vs Entities eval ===")
    print(f"  N games: {args.n_games}, per-pair concurrency: {args.bot_concurrency}")
    print(f"  MMs: {[m[0] for m in mms]}")
    print(f"  Entities: {[(e[0], e[1]) for e in entities]}")
    print(f"  Team set: {args.team_set!r} (both sides)")
    if args.team_set == "procedural":
        print(f"  Procedural teams source: {args.procedural_teams_path}")
    print(f"  Output: {out_json}")

    asyncio.run(main_async(args.n_games, args.bot_concurrency, mms, entities,
                            cached_ckpts, out_json, args.device,
                            args.mm_startup_wait, args.team_set,
                            args.procedural_teams_path))


if __name__ == "__main__":
    main()
