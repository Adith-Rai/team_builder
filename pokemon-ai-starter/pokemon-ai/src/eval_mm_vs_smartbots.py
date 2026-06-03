#!/usr/bin/env python3
"""eval_mm_vs_smartbots.py — quick MM vs smart-bot WR check.

Spawns each Metamon trained model as a subprocess (via metamon_accept_serve.py),
then runs each of our 4 smart bots (SH, SmartDmg, Tactical, Strategic) against
it for N games. Outputs a WR matrix.

Goal: calibrate where our smart-bot anchor sits relative to known external
benchmarks. Specifically: is the 70-74% smart_avg ceiling that our trained
models hit "clearly beatable" by metamon trained models, or do they cluster
around it too?

Usage:
    python eval_mm_vs_smartbots.py --n-games 500
    # Optional: --bot-concurrency 8  (battles in flight per bot-MM pair)
    # Optional: --models LargeRL SyntheticRLV2  (subset)

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

# Default MM set matches fishbowl_lr1e-4_v1.yaml (the 3 trained models)
DEFAULT_MMS = [
    ("LargeRL", "MMevalLargeRL", 9000),
    ("MediumRL_Aug", "MMevalMediumRLAug", 9001),
    ("SyntheticRLV2", "MMevalSyntheticRLV2", 9002),
]

SMART_BOTS = ["SH", "SmartDmg", "Tactical", "Strategic"]


def spawn_mm(model: str, username: str, port: int) -> Tuple[subprocess.Popen, str]:
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
        "--team-set", "competitive",
    ]
    env = {**os.environ, "METAMON_CACHE_DIR": METAMON_CACHE}
    log_path = f"/tmp/mm_eval_{username}.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc, log_path


def _get_teambuilder():
    """Returns MetamonCompetitiveTeambuilder for bot side.

    Bot uses the same 16 metamon-competitive teams that our smart_avg
    eval uses, so MM-vs-bot WRs are directly comparable to our
    model-vs-smart-bot smart_avg numbers. MM side already uses
    --team-set competitive (the same 16 teams) per spawn_mm config.
    """
    from eval_metamon_competitive import MetamonCompetitiveTeambuilder
    return MetamonCompetitiveTeambuilder()


def make_bot(bot_name: str, port: int, account: str, concurrency: int):
    cls = ALL_BOTS[bot_name]
    server_cfg = resolve_server(f"ws://127.0.0.1:{port}/showdown/websocket")
    return cls(
        battle_format="gen9ou",
        max_concurrent_battles=concurrency,
        server_configuration=server_cfg,
        team=_get_teambuilder(),
        account_configuration=AccountConfiguration(account, None),
    )


async def run_pair(bot_name: str, mm_username: str, port: int, n_games: int,
                   bot_concurrency: int, match_idx: int) -> dict:
    account = f"E{os.getpid() % 9999}m{match_idx}{bot_name[:4]}"
    bot = make_bot(bot_name, port, account, bot_concurrency)

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
        "bot": bot_name, "mm": mm_username,
        "bot_wins": wins, "mm_wins": losses, "ties": ties, "total": total,
        "bot_wr": wins / max(1, total),
        "elapsed_s": round(elapsed, 1),
    }


async def main_async(n_games: int, bot_concurrency: int, mms: List[Tuple[str, str, int]],
                     bots: List[str], out_json: str) -> None:
    results: List[dict] = []
    overall_t0 = time.time()
    match_idx = 0

    for mm_model, mm_username, mm_port in mms:
        print(f"\n=== Spawning Metamon {mm_model} as {mm_username} on port {mm_port} ===",
              flush=True)
        proc, log_path = spawn_mm(mm_model, mm_username, mm_port)
        print(f"  PID={proc.pid}, log={log_path}", flush=True)
        # Wait for MM to log in and start accepting (15s typically enough)
        await asyncio.sleep(15)
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
            for bot_name in bots:
                match_idx += 1
                print(f"  [{bot_name} vs {mm_model}] starting (n_games={n_games})...",
                      flush=True)
                try:
                    r = await run_pair(bot_name, mm_username, mm_port,
                                       n_games, bot_concurrency, match_idx)
                    results.append(r)
                    print(f"    {bot_name} {r['bot_wins']}W/{r['mm_wins']}L/"
                          f"{r['ties']}T ({100*r['bot_wr']:.1f}% for {bot_name}, "
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

    # Print WR matrix
    print(f"\n=== WR MATRIX  (% bot wins, n_games={n_games}) ===")
    header = f"{'BOT':14s}"
    for mm_model, _, _ in mms:
        header += f" {mm_model[:14]:>16s}"
    print(header)
    for bot_name in bots:
        row = f"{bot_name:14s}"
        for mm_model, _, _ in mms:
            found = next((r for r in results if r["bot"] == bot_name
                          and r["mm"].endswith(mm_model.replace("_", ""))
                          or r["mm"].endswith(mm_model)), None)
            # Match by reconstructing search
            for r in results:
                if r["bot"] == bot_name and mm_model.replace("_", "") in r["mm"]:
                    found = r
                    break
            if found and found["total"] > 0:
                wr = 100 * found["bot_wr"]
                row += f" {wr:>14.1f}% "
            else:
                row += f" {'N/A':>15s} "
        print(row)
    print()
    print(f"Saved JSON: {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=500)
    ap.add_argument("--bot-concurrency", type=int, default=8,
                    help="Battles in flight per bot-MM pair (Showdown caps per-pair)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Subset of MMs to test (e.g. LargeRL SyntheticRLV2)")
    ap.add_argument("--bots", nargs="+", default=SMART_BOTS,
                    help=f"Smart bots to test. Default: {SMART_BOTS}")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    mms = DEFAULT_MMS
    if args.models:
        mms = [m for m in DEFAULT_MMS if m[0] in args.models]
        if not mms:
            raise SystemExit(f"No MMs matched {args.models}. Available: "
                             f"{[m[0] for m in DEFAULT_MMS]}")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_json = args.out_json or f"/tmp/mm_vs_smartbots_{ts}.json"

    print(f"=== MM vs SmartBots eval ===")
    print(f"  N games: {args.n_games}, bot concurrency: {args.bot_concurrency}")
    print(f"  MMs: {[m[0] for m in mms]}")
    print(f"  Bots: {args.bots}")
    print(f"  Output: {out_json}")

    asyncio.run(main_async(args.n_games, args.bot_concurrency, mms, args.bots, out_json))


if __name__ == "__main__":
    main()
