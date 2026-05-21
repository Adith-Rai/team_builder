#!/usr/bin/env python3
"""Phase 2 Elo poller — V1.

Polls a prod pod for new training snapshots, pulls each via scp, runs a
3-server sharded Elo eval against the ladder, appends results to a per-run
ladder JSON, persists state. Sleeps when caught up.

Designed to run alongside a Phase 2 prod training run. ~1-1.5 hr per
snapshot at 500 g/opp on local RTX 3060 with 3-shard parallelism.

USAGE:
    python elo_poller.py --run-name phase2_stage1_v1

REQUIRES:
    - 3 battle servers running locally on ports 9000-9002
    - SSH key at ~/.ssh/id_ed25519 with access to prod
    - eval_elo_ladder.py in current directory
    - data/eval/registry/elo_v10_500g_focused_plus_iter49.json (default base
      ladder) — or pass --ladder-base to override

STATE:
    Persists per-run state at data/eval/registry/<run>_poller_state.json.
    Resumable across restarts. Already-evaluated iters skipped.

LADDER CHAIN:
    Each eval uses --add-to with the *latest* ladder JSON (i.e. the output
    of the previous eval). New snapshot's BT estimate stays consistent with
    all prior snapshots in the run + the curated base anchors.

V1 SCOPE (intentional):
    - No automated dip detection / backtrack (manual review of trajectory)
    - No state-file recovery on corruption (delete + restart)
    - No retry of failed evals (warn + skip; rerun manually)
    Add these in V2 once the basic flow is validated.

See docs/PHASE2_LAUNCH_PLAN.md §4 + §9.6 for design + the equivalent
manual 3-shard command.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Config — adjust if pod connection or paths change
# ──────────────────────────────────────────────────────────────────────────────
SSH_KEY = "~/.ssh/id_ed25519"
SSH_HOST = "root@195.26.233.30"
SSH_PORT = "47913"
PROD_RUN_DIR_TEMPLATE = (
    "/workspace/team_builder/pokemon-ai-starter/pokemon-ai/src/"
    "data/models/rl_v10/{run_name}"
)

BASE_LADDER_DEFAULT = Path("data/eval/registry/elo_v10_500g_focused_plus_iter49.json")
LOCAL_DATA_DIR = Path("data/cloud_runs")
LOCAL_LADDER_DIR = Path("data/eval/registry")
EVAL_SCRIPT = "eval_elo_ladder.py"

N_GAMES_PER_OPP = "500"
CONCURRENCY_PER_SHARD = "50"
BS_PORTS = [9000, 9001, 9002]  # local battle server ports for sharding

POLL_INTERVAL_DEFAULT_S = 300


# ──────────────────────────────────────────────────────────────────────────────
# SSH / scp helpers
# ──────────────────────────────────────────────────────────────────────────────
def ssh(cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "ssh", "-i", SSH_KEY, "-p", SSH_PORT,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=15",
            SSH_HOST, cmd,
        ],
        capture_output=True, text=True, timeout=timeout,
    )


def scp_pull(remote: str, local: Path, timeout: int = 180) -> None:
    subprocess.run(
        [
            "scp", "-i", SSH_KEY, "-P", SSH_PORT,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=30",
            f"{SSH_HOST}:{remote}", str(local),
        ],
        check=True, timeout=timeout,
    )


def list_prod_snapshots(run_name: str) -> list[tuple[int, str]]:
    """Return [(iter_n, prod_path), ...] sorted by iter ascending."""
    pattern = (
        f"{PROD_RUN_DIR_TEMPLATE.format(run_name=run_name)}/"
        "selfplay_v9_*/iter_*.pt"
    )
    result = ssh(f"ls {pattern} 2>/dev/null")
    snaps: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        m = re.search(r"iter_(\d+)\.pt$", line)
        if m:
            snaps.append((int(m.group(1)), line))
    snaps.sort(key=lambda t: t[0])
    return snaps


# ──────────────────────────────────────────────────────────────────────────────
# Eval orchestration
# ──────────────────────────────────────────────────────────────────────────────
def run_sharded_eval(
    local_ckpt: Path,
    eval_name: str,
    ladder_in: Path,
    ladder_out_base: str,
) -> Path:
    """Spawn 3 shards in parallel, wait, combine, return final JSON path."""
    procs: list[subprocess.Popen] = []
    for shard_idx, port in enumerate(BS_PORTS):
        out = f"{ladder_out_base}_shard{shard_idx}.json"
        cmd = [
            sys.executable, "-u", EVAL_SCRIPT,
            "--add-to", str(ladder_in),
            "--snapshots", str(local_ckpt),
            "--names", eval_name,
            "--n-games", N_GAMES_PER_OPP,
            "--concurrency", CONCURRENCY_PER_SHARD,
            "--device", "cuda",
            "--shard", f"{shard_idx}/{len(BS_PORTS)}",
            "--server", f"ws://127.0.0.1:{port}/showdown/websocket",
            "--out-json", out,
        ]
        logging.info(f"  Launching shard {shard_idx} (port {port}) -> {out}")
        procs.append(subprocess.Popen(cmd))

    for i, p in enumerate(procs):
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"Shard {i} failed with rc={rc}")

    out_final = f"{ladder_out_base}_FINAL.json"
    combine_cmd = [
        sys.executable, "-u", EVAL_SCRIPT, "--combine",
        *[f"{ladder_out_base}_shard{i}.json" for i in range(len(BS_PORTS))],
        "--out-json", out_final,
    ]
    logging.info(f"  Combining -> {out_final}")
    subprocess.run(combine_cmd, check=True)
    return Path(out_final)


def _extract_elo(elo_raw) -> Optional[tuple[float, float, float]]:
    """Normalize elo result across the three ladder formats observed in the wild.

    V1 bug (2026-05-21): assumed list [med, lo, hi]. Actual `eval_elo_ladder.py`
    output is a DICT {"median": ..., "lo95": ..., "hi95": ...}. Some older ladders
    store just a float median. Handle all three gracefully.
    """
    if elo_raw is None:
        return None
    if isinstance(elo_raw, dict):
        med = elo_raw.get("median")
        lo = elo_raw.get("lo95", med)
        hi = elo_raw.get("hi95", med)
        if med is None:
            return None
        return (float(med), float(lo), float(hi))
    if isinstance(elo_raw, (list, tuple)) and len(elo_raw) == 3:
        return (float(elo_raw[0]), float(elo_raw[1]), float(elo_raw[2]))
    if isinstance(elo_raw, (int, float)):
        # Older ladders stored just the median
        med = float(elo_raw)
        return (med, med, med)
    return None


def evaluate_snapshot(
    run_name: str,
    iter_n: int,
    prod_path: str,
    local_dir: Path,
    ladder_in: Path,
) -> tuple[Path, Optional[tuple[float, float, float]]]:
    """Pull + 3-shard eval one snapshot. Return (final_ladder, elo_triple_or_None)."""
    local_name = f"{run_name}_iter_{iter_n:04d}.pt"
    local_ckpt = local_dir / local_name

    if not local_ckpt.exists():
        logging.info(f"Pulling iter {iter_n}: {prod_path} -> {local_ckpt}")
        scp_pull(prod_path, local_ckpt)
    else:
        logging.info(f"Iter {iter_n}: local copy already exists at {local_ckpt}")

    eval_name = f"{run_name}_iter{iter_n}"
    ladder_out_base = str(LOCAL_LADDER_DIR / f"elo_{eval_name}")

    logging.info(f"Running 3-shard eval for {eval_name} against {ladder_in}")
    t0 = time.time()
    final_ladder = run_sharded_eval(local_ckpt, eval_name, ladder_in, ladder_out_base)
    duration_min = (time.time() - t0) / 60
    logging.info(f"  Eval complete in {duration_min:.1f} min -> {final_ladder}")

    # Extract Elo for this snapshot — robust to dict/list/float formats
    elo = None
    try:
        final = json.loads(final_ladder.read_text())
        elo_raw = final.get("elos", {}).get(eval_name)
        elo = _extract_elo(elo_raw)
        if elo:
            med, lo, hi = elo
            logging.info(f"  Iter {iter_n} Elo: {med:.1f} [{lo:.0f}, {hi:.0f}]")
        else:
            keys = list(final.get("elos", {}).keys())[:5]
            logging.warning(
                f"  Iter {iter_n} Elo not extractable. Sample keys: {keys}. "
                f"elo_raw type: {type(elo_raw).__name__}"
            )
    except Exception as e:
        # Eval succeeded (ladder written) but extraction failed. Don't re-evaluate
        # — log and move on. State will still mark iter as done.
        logging.warning(f"  Iter {iter_n} Elo extraction failed: {e}")

    return final_ladder, elo


# ──────────────────────────────────────────────────────────────────────────────
# State persistence
# ──────────────────────────────────────────────────────────────────────────────
def load_state(state_file: Path, base_ladder: Path) -> dict:
    if state_file.exists():
        return json.loads(state_file.read_text())
    return {
        "evaluated_iters": [],
        "latest_ladder": str(base_ladder),
        "latest_eval": None,
    }


def save_state(state_file: Path, state: dict) -> None:
    state_file.write_text(json.dumps(state, indent=2))


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description="Phase 2 Elo poller (V1)")
    p.add_argument("--run-name", required=True,
                   help="Prod run name (matches /workspace/.../rl_v10/<run>)")
    p.add_argument("--poll-interval", type=int, default=POLL_INTERVAL_DEFAULT_S,
                   help="Seconds between polls when caught up (default 300)")
    p.add_argument("--ladder-base", type=Path, default=BASE_LADDER_DEFAULT,
                   help=f"Starting ladder JSON (default {BASE_LADDER_DEFAULT})")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.ladder_base.exists():
        logging.error(f"Base ladder not found: {args.ladder_base}")
        return 1

    local_dir = LOCAL_DATA_DIR / args.run_name
    local_dir.mkdir(parents=True, exist_ok=True)
    LOCAL_LADDER_DIR.mkdir(parents=True, exist_ok=True)

    state_file = LOCAL_LADDER_DIR / f"{args.run_name}_poller_state.json"
    state = load_state(state_file, args.ladder_base)

    logging.info("=== Phase 2 Elo poller (V1) ===")
    logging.info(f"Run name:       {args.run_name}")
    logging.info(f"Local dir:      {local_dir}")
    logging.info(f"Base ladder:    {args.ladder_base}")
    logging.info(f"State file:     {state_file}")
    logging.info(f"Latest ladder:  {state['latest_ladder']}")
    logging.info(f"Already done:   iters {state['evaluated_iters']}")
    logging.info(f"Poll interval:  {args.poll_interval}s")
    logging.info("")

    while True:
        try:
            snaps = list_prod_snapshots(args.run_name)
            new = [(i, path) for i, path in snaps if i not in state["evaluated_iters"]]

            if not new:
                latest = snaps[-1][0] if snaps else "none"
                logging.info(
                    f"No new snapshots (latest prod iter: {latest}), "
                    f"sleeping {args.poll_interval}s"
                )
                time.sleep(args.poll_interval)
                continue

            logging.info(f"Found {len(new)} new snapshots: iters {[i for i, _ in new]}")

            for iter_n, prod_path in new:
                logging.info(f"=== Evaluating iter {iter_n} ===")
                ladder_in = Path(state["latest_ladder"])
                try:
                    final_ladder, elo = evaluate_snapshot(
                        args.run_name, iter_n, prod_path, local_dir, ladder_in,
                    )
                except Exception as e:
                    logging.error(f"Eval failed for iter {iter_n}: {e}", exc_info=True)
                    logging.warning(
                        f"Skipping iter {iter_n} — rerun manually if needed. "
                        "Continuing to next snapshot."
                    )
                    continue

                state["evaluated_iters"].append(iter_n)
                state["evaluated_iters"].sort()
                state["latest_ladder"] = str(final_ladder)
                state["latest_eval"] = {
                    "iter": iter_n,
                    "elo": elo,
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_state(state_file, state)

        except KeyboardInterrupt:
            logging.info("Caught Ctrl-C, exiting cleanly.")
            return 0
        except subprocess.TimeoutExpired as e:
            logging.error(f"SSH/scp timed out: {e}. Sleeping {args.poll_interval}s.")
            time.sleep(args.poll_interval)
        except Exception as e:
            logging.error(f"Unexpected error in poll cycle: {e}", exc_info=True)
            logging.info(f"Sleeping {args.poll_interval}s before retry.")
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())
