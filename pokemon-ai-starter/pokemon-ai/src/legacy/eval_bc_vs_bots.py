#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, glob, time, json, csv, shutil, asyncio, random
from pathlib import Path
from typing import Dict, Any, List

import torch
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from bc_policy_player import BCPolicyPlayer
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from policy_rulebots import (
    GreedySEPlayer,
    HazardSensePlayer,
    SwitchAwareEscapePlayer,
    SetupThenSweepPlayer,
)
from policy_smartbots import SmartDamagePlayer, TacticalPlayer, StrategicPlayer
from policy_random import RandomPolicy
from teams_ou import random_teambuilder, random_pool_teambuilder

# ------------------------------
# Bot registry + common aliases
# ------------------------------
BOTS = {
    "MaxBasePower": MaxBasePowerPlayer,        # aka "MaxDamage"
    "SimpleHeuristics": SimpleHeuristicsPlayer,
    "Random": RandomPolicy,
    "GreedySE": GreedySEPlayer,
    "HazardSense": HazardSensePlayer,
    "SwitchAwareEscape": SwitchAwareEscapePlayer,
    "SetupThenSweep": SetupThenSweepPlayer,
    "SmartDamage": SmartDamagePlayer,
    "Tactical": TacticalPlayer,
    "Strategic": StrategicPlayer,
}

ALIASES = {
    "MaxDamage": "MaxBasePower",
    "MaxBP": "MaxBasePower",
    "Simple": "SimpleHeuristics",
}

# ------------------------------
# CLI
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # checkpoint selection
    p.add_argument("--checkpoint", default=None, help="Path to a single .pt")
    p.add_argument("--ckpt-glob", default=None, help="Glob like data/models/bc/<run>/epoch_*.pt")
    p.add_argument("--epochs", default="last", help="'all' or 'last' or '1,3,5'")
    # opponents / games
    p.add_argument("--bots", default="Random,MaxDamage,SimpleHeuristics",
                   help="comma list from: " + ",".join(sorted(BOTS.keys())))
    p.add_argument("--n-battles", type=int, default=100)
    p.add_argument("--games", type=int, default=None, help="alias for --n-battles")
    # env / model
    p.add_argument("--format", default="gen9ou")
    p.add_argument("--device", default="cuda")
    p.add_argument("--run-name", default=None)
    # server + concurrency
    _host = os.environ.get("SHOWDOWN_HOST", "127.0.0.1")
    _port = os.environ.get("SHOWDOWN_PORT", "8000")
    p.add_argument("--server", dest="server_url",
                   default=f"ws://{_host}:{_port}/showdown/websocket",
                   help="WebSocket URL for Showdown (set SHOWDOWN_HOST env for Docker)")
    p.add_argument("--max-concurrent", dest="concurrency", type=int, default=2)
    # optional inference overrides
    p.add_argument("--use-lstm", default=None, type=lambda x: None if x.lower() == "none" else x.lower() in ("1","true","yes"),
                   nargs="?", const=True, help="Force LSTM on/off; omit to use checkpoint setting")
    p.add_argument("--lstm-hidden", type=int, default=None)
    p.add_argument("--mlp-hidden", type=int, default=None)
    # replays / outputs
    p.add_argument("--save-replays", action="store_true", help="Enable replay saving in players")
    p.add_argument("--replays-root", default="data/replays/replays_eval")
    p.add_argument("--out-csv", default=None, help="If set, also write a flat CSV of all rows here")
    p.add_argument("--out-jsonl-battles", default=None, help="If set, write per-battle JSONL here")
    # bookkeeping
    p.add_argument("--stage", default="training_bc")
    p.add_argument("--seed", type=int, default=1337)
    # extras: Elo + speed
    p.add_argument("--elo-baseline", type=int, default=1500)
    p.add_argument("--elo-k", type=float, default=32.0)
    p.add_argument("--direct", action="store_true",
                   help="Use direct BattleStream transport (no websockets/Docker)")

    args = p.parse_args()
    if args.games is not None:
        args.n_battles = args.games  # backward compatibility alias
    return args

# ------------------------------
# Helpers
# ------------------------------
def infer_run_name(ckpt_path: str) -> str:
    # data/models/bc/<run>/epoch_XXX.pt
    p = Path(ckpt_path).parts
    return p[p.index("bc")+1] if "bc" in p else Path(ckpt_path).parent.name

def make_dir(root, stage, run_name, epoch, bot):
    d = Path(root)/stage/run_name/f"epoch_{int(epoch):03d}"/f"vs_{bot}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def resolve_server(ws_url: str) -> ServerConfiguration:
    ws = ws_url.rstrip("/")
    http = ws.replace("wss://", "https://").replace("ws://", "http://")
    if ws.endswith("/showdown/websocket"):
        http = http[: http.rfind("/showdown/websocket")] + "/action.php?"
    return ServerConfiguration(ws, http)

def snapshot_replays() -> set:
    root = Path("replays")
    if not root.exists():
        return set()
    return set(str(p) for p in root.rglob("*.html"))

def move_new_replays(before_set: set, dest_dir: Path, tag: str) -> List[str]:
    root = Path("replays")
    if not root.exists():
        return []
    added = [p for p in root.rglob("*.html") if str(p) not in before_set]
    moved_paths = []
    for i, src in enumerate(sorted(added)):
        dst = dest_dir / f"{tag}_{i:04d}.html"
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(src), str(dst))
        except Exception:
            try:
                shutil.copy2(str(src), str(dst))
            except Exception:
                continue
        moved_paths.append(str(dst))
    return moved_paths

async def _battle_with_retry(our, opp, n_battles: int, retries: int = 1):
    for attempt in range(retries + 1):
        try:
            return await our.battle_against(opp, n_battles=n_battles)
        except asyncio.TimeoutError:
            if attempt >= retries:
                raise
            await asyncio.sleep(1.0)

def _elo_update(rating: float, opp_rating: float, score: float, k: float) -> float:
    # score in {1, 0.5, 0}
    expected = 1.0 / (1.0 + 10 ** ((opp_rating - rating)/400.0))
    return rating + k * (score - expected)

def _battle_rows_for_jsonl(our_player, bot_name: str, run_name: str, epoch: int,
                           format_str: str, moved_paths: List[str]) -> List[Dict[str, Any]]:
    rows = []
    # We do heuristic mapping: enumerate battles; attach a replay path if index < moved files
    moved_by_idx = {i: moved_paths[i] for i in range(len(moved_paths))}
    for i, (battle_id, battle) in enumerate(sorted(our_player.battles.items())):
        # Determine outcome
        if battle.won:
            outcome = "win"
        elif battle.lost:
            outcome = "loss"
        else:
            outcome = "tie"
        rows.append({
            "battle_id": battle_id,
            "bot": bot_name,
            "run_name": run_name,
            "epoch": int(epoch),
            "format": format_str,
            "turns": int(getattr(battle, "turn", 0) or 0),
            "outcome": outcome,
            "replay_html": moved_by_idx.get(i),
            "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
        })
    return rows

# ------------------------------
# One eval (one ckpt × one bot)
# ------------------------------
def eval_one(ckpt_path, epoch, bot_name, args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = args.run_name or infer_run_name(ckpt_path)
    out_dir = make_dir(args.replays_root, args.stage, run_name, epoch, bot_name)
    use_direct = getattr(args, "direct", False)

    before = snapshot_replays()

    # Common kwargs; when --direct, skip websocket listener
    extra_kwargs = {}
    if use_direct:
        extra_kwargs["start_listening"] = False
    else:
        server = resolve_server(args.server_url)
        extra_kwargs["server_configuration"] = server

    our = BCPolicyPlayer(
        checkpoint_path=ckpt_path,
        device=args.device,
        battle_format=args.format,
        team=random_pool_teambuilder(),
        save_replays=args.save_replays,
        max_concurrent_battles=max(1, int(args.concurrency)),
        use_lstm=args.use_lstm,
        lstm_hidden=args.lstm_hidden,
        mlp_hidden=args.mlp_hidden,
        # v5: let BCPolicyPlayer infer these from checkpoint's policy_cfg
        # (explicitly passing None means "use checkpoint value")
        **extra_kwargs,
    )
    OppClass = BOTS[bot_name]
    opp = OppClass(
        battle_format=args.format,
        team=random_pool_teambuilder(),
        save_replays=args.save_replays,
        max_concurrent_battles=max(1, int(args.concurrency)),
        **extra_kwargs,
    )

    # Patch players for direct transport if --direct
    if use_direct:
        from direct_player import patch_to_direct, direct_battle_against
        patch_to_direct(our)
        patch_to_direct(opp)

    # Run battles + rough timing
    t0 = time.time()
    if use_direct:
        from poke_env.concurrency import POKE_LOOP
        import concurrent.futures
        future = asyncio.run_coroutine_threadsafe(
            direct_battle_against(our, opp, n_battles=args.n_battles),
            POKE_LOOP,
        )
        future.result(timeout=args.n_battles * 60)  # generous timeout
    else:
        loop = asyncio.get_event_loop_policy().new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_battle_with_retry(our, opp, n_battles=args.n_battles, retries=1))
        finally:
            loop.close()
    t1 = time.time()

    # Sweep replays into labeled folder
    tag = f"{bot_name}_epoch{int(epoch):03d}"
    moved_paths = move_new_replays(before, out_dir, tag)

    # Stats
    wins, losses, ties = our.n_won_battles, our.n_lost_battles, our.n_tied_battles
    total = wins + losses + ties
    winrate = wins / total if total else 0.0
    total_turns = sum(int(getattr(b, "turn", 0) or 0) for b in our.battles.values())
    wall_secs = max(1e-6, t1 - t0)
    turns_per_sec = total_turns / wall_secs if total_turns else 0.0
    decisions_per_sec = (total * 1.0) / wall_secs if total else 0.0  # coarse

    meta = {
        "run_name": run_name, "epoch": int(epoch), "checkpoint": ckpt_path,
        "bot": bot_name, "format": args.format, "n_battles": args.n_battles,
        "wins": wins, "losses": losses, "ties": ties, "winrate": winrate,
        "replay_dir": str(out_dir), "replays_moved": len(moved_paths),
        "turns_total": total_turns, "turns_per_sec": turns_per_sec,
        "decisions_per_sec": decisions_per_sec,
        "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(meta, f, indent=2)

    # TensorBoard (best-effort)
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=str(Path("data/logs/tb/eval") /
                            f"{run_name}_epoch{int(epoch):03d}_{time.strftime('%Y%m%d_%H%M%S')}"))
        tb.add_scalar(f"winrate/{bot_name}", winrate, int(epoch))
        tb.add_scalar(f"counts/{bot_name}_wins", wins, int(epoch))
        tb.add_scalar(f"counts/{bot_name}_ties", ties, int(epoch))
        tb.add_scalar(f"counts/{bot_name}_losses", losses, int(epoch))
        tb.add_scalar(f"speed/{bot_name}_turns_per_sec", turns_per_sec, int(epoch))
        tb.add_scalar(f"speed/{bot_name}_decisions_per_sec", decisions_per_sec, int(epoch))
        tb.close()
    except Exception:
        pass

    # Optional per-battle JSONL rows for this (ckpt, bot)
    battle_rows = _battle_rows_for_jsonl(our, bot_name, run_name, epoch, args.format, moved_paths)

    print(f"[EVAL] {run_name} epoch {epoch} vs {bot_name}: winrate={winrate:.3f} "
          f"(W{wins}/L{losses}/T{ties})  replays={out_dir} (moved {len(moved_paths)})  "
          f"turns/s={turns_per_sec:.2f} decisions/s={decisions_per_sec:.2f}")

    return meta, battle_rows

# ------------------------------
# Main
# ------------------------------
def main():
    args = parse_args()

    # Resolve checkpoints
    ckpts = []
    if args.checkpoint:
        ckpts = [args.checkpoint]
    elif args.ckpt_glob:
        ckpts = sorted(glob.glob(args.ckpt_glob))
    if not ckpts:
        raise SystemExit("No checkpoint(s) provided. Use --checkpoint or --ckpt-glob.")

    # Epoch filtering
    if args.epochs == "last":
        ckpts = ckpts[-1:]
    elif args.epochs != "all":
        want = {int(x) for x in args.epochs.split(",") if x.strip()}
        ckpts = [p for p in ckpts if any(p.endswith(f"epoch_{e:03d}.pt") for e in want)]

    # Parse bots and map aliases
    bots = [b.strip() for b in args.bots.split(",") if b.strip()]
    bots = [ALIASES.get(b, b) for b in bots]
    unknown = [b for b in bots if b not in BOTS]
    if unknown:
        raise SystemExit(f"Unknown bot(s): {unknown}. Valid: {', '.join(sorted(BOTS.keys()))}")

    # Output buckets
    ts = time.strftime("%Y%m%d_%H%M%S")
    first_run = args.run_name or (infer_run_name(ckpts[0]) if ckpts else "unknown")
    run_bucket = first_run if all(infer_run_name(c) == first_run for c in ckpts) else "multi"
    eval_dir = Path("data/evaluations") / args.stage / run_bucket / ts
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Global collectors
    all_meta_rows: List[Dict[str, Any]] = []
    all_battle_rows: List[Dict[str, Any]] = []

    # Approximate Elo across all games (vs each bot treated as 1500 unless you change it)
    elo = float(args.elo_baseline)
    for ck in ckpts:
        name = os.path.basename(ck)
        try:
            ep = int(name.split("_")[-1].split(".")[0])
        except Exception:
            ep = -1
        for bot in bots:
            meta, battle_rows = eval_one(ck, ep, bot, args)
            all_meta_rows.append(meta)
            all_battle_rows.extend(battle_rows)

            # Elo update: treat the bot as static 1500 for a rough skill proxy
            total = meta["wins"] + meta["losses"] + meta["ties"]
            # score = (W + 0.5*T) / N
            score = (meta["wins"] + 0.5 * meta["ties"]) / total if total else 0.0
            elo = _elo_update(elo, float(args.elo_baseline), score, float(args.elo_k))
            meta["elo_estimate"] = round(elo, 2)

    # Write consolidated CSV (default path if not specified)
    if not all_meta_rows:
        print("[eval] No evaluation results to write.", flush=True)
        return
    out_csv = Path(args.out_csv) if args.out_csv else (eval_dir / "results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_meta_rows[0].keys()))
        w.writeheader()
        for r in all_meta_rows:
            w.writerow(r)

    # Optional JSONL per-battle
    if all_battle_rows and args.out_jsonl_battles:
        out_jsonl = Path(args.out_jsonl_battles)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in all_battle_rows:
                f.write(json.dumps(r) + "\n")

    # Save invocation args
    with open(eval_dir / "args.json", "w") as f:
        json.dump({
            "ckpt_glob": args.ckpt_glob,
            "epochs": args.epochs,
            "bots": bots,
            "n_battles": args.n_battles,
            "format": args.format,
            "device": args.device,
            "run_name": args.run_name,
            "replays_root": args.replays_root,
            "stage": args.stage,
            "server_url": args.server_url,
            "concurrency": args.concurrency,
            "timestamp": ts,
            "elo_estimate": round(elo, 2),
            "elo_baseline": args.elo_baseline,
            "elo_k": args.elo_k,
        }, f, indent=2)

    print(f"[EVAL] Wrote consolidated results to {out_csv}")
    if all_battle_rows and args.out_jsonl_battles:
        print(f"[EVAL] Per-battle JSONL: {args.out_jsonl_battles}")
    print(f"[EVAL] Eval pack (args.json + results.csv) in {eval_dir}")
    print(f"[EVAL] Elo estimate across all eval games: {elo:.2f}")

if __name__ == "__main__":
    main()
    # Force exit when using --direct to avoid hang from poke-env background threads
    if "--direct" in sys.argv:
        os._exit(0)
