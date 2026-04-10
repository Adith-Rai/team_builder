#!/usr/bin/env python
"""Build/update the persistent data registry from existing training runs.

Scans all training run directories, TensorBoard logs, and Elo JSONs to produce
a unified set of JSONL files that serve as the single source of truth for:
  - runs.jsonl: training run configs (iter range, hyperparams, run dir)
  - evals.jsonl: per-iter bot evaluation results (SH, SmartDmg, Tactical, Strategic, smart_avg)
  - elos.jsonl: per-snapshot Elo measurements from all ladder runs

Usage:
    python build_registry.py                    # rebuild from scratch
    python build_registry.py --update           # add new data only
    python build_registry.py --print-summary    # show summary stats
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path


REGISTRY_DIR = Path("data/eval/registry")
RUNS_FILE = REGISTRY_DIR / "runs.jsonl"
EVALS_FILE = REGISTRY_DIR / "evals.jsonl"
ELOS_FILE = REGISTRY_DIR / "elos.jsonl"


def _scan_runs():
    """Scan all training run directories for config.json files."""
    runs = []
    for config_path in sorted(glob.glob("data/models/rl_v9/selfplay_v9_*/config.json")):
        run_dir = str(Path(config_path).parent)
        with open(config_path) as f:
            config = json.load(f)

        # Find iter range from snapshots
        snapshots = sorted(glob.glob(os.path.join(run_dir, "snapshot_*.pt")))
        if not snapshots:
            continue
        iter_nums = []
        for s in snapshots:
            m = re.search(r'snapshot_(\d+)\.pt', s)
            if m:
                iter_nums.append(int(m.group(1)))

        if not iter_nums:
            continue

        run = {
            "run_dir": run_dir,
            "iter_lo": min(iter_nums),
            "iter_hi": max(iter_nums),
            "n_snapshots": len(iter_nums),
            "lam": config.get("lam"),
            "ent_coef": config.get("ent_coef"),
            "gamma": config.get("gamma"),
            "lr": config.get("lr"),
            "clip_eps": config.get("clip_eps"),
            "ppo_epochs": config.get("ppo_epochs"),
            "games_per_iter": config.get("games_per_iter"),
            "max_concurrent": config.get("max_concurrent"),
            "reward_style": config.get("reward_style"),
            "format": config.get("format", "gen9ou"),
            "fp16": config.get("fp16", False),
            "pipeline": config.get("pipeline", False),
            "resume_from": config.get("resume"),
        }
        runs.append(run)
    return runs


def _scan_evals():
    """Extract eval results from all TensorBoard logs."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    evals = []
    eval_tags = ["eval/SH", "eval/SmartDmg", "eval/Tactical", "eval/Strategic", "eval/smart_avg"]

    for tb_dir in sorted(glob.glob("data/models/rl_v9/selfplay_v9_*/tb")):
        run_dir = str(Path(tb_dir).parent)
        ea = EventAccumulator(tb_dir)
        ea.Reload()
        available = ea.Tags().get("scalars", [])

        # Collect all eval steps for this run
        step_data = {}
        for tag in eval_tags:
            if tag not in available:
                continue
            col = tag.split("/")[-1]
            for event in ea.Scalars(tag):
                step = event.step
                if step not in step_data:
                    step_data[step] = {"iter": step, "run_dir": run_dir}
                step_data[step][col] = round(event.value, 4)

        for step in sorted(step_data):
            row = step_data[step]
            if "smart_avg" in row:
                evals.append(row)

    return evals


def _scan_elos():
    """Extract per-snapshot Elo ratings from all ladder JSON files."""
    elos = []
    bot_names = {'SH', 'Random', 'MaxBasePower', 'MaxBP', 'GreedySE', 'HazardSense',
                 'SwitchAwareEscape', 'SetupThenSweep', 'SmartDmg', 'Tactical', 'Strategic'}

    for elo_json in sorted(glob.glob("data/eval/elo_session*.json")):
        # Skip shard/partial files
        basename = Path(elo_json).stem
        if 'shard' in basename or 'partial' in basename or 'smoke' in basename:
            continue

        with open(elo_json) as f:
            data = json.load(f)

        ladder_name = basename
        config = data.get("config", {})
        n_games = config.get("n_games", config.get("n_matchups", "?"))

        for name, elo in data.get("elos", {}).items():
            if name in bot_names:
                continue

            ci = data.get("cis", {}).get(name, {})
            lo95 = ci.get("lo95", elo)
            hi95 = ci.get("hi95", elo)

            # Extract iter number
            if name == "BC_base":
                it = 0
            else:
                m = re.search(r'(\d{3,4})', name)
                it = int(m.group(1)) if m else -1

            # Find checkpoint path
            ckpt = None
            for p in data.get("players", []):
                if p.get("name") == name:
                    ckpt = p.get("ckpt")
                    break

            elos.append({
                "iter": it,
                "name": name,
                "elo": round(elo, 1),
                "ci_lo": round(lo95, 1),
                "ci_hi": round(hi95, 1),
                "ladder": ladder_name,
                "n_games": n_games,
                "ckpt": ckpt,
            })

    return elos


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Wrote {len(records)} records to {path}")


def build(update_only=False):
    print("Building registry...")

    runs = _scan_runs()
    _write_jsonl(RUNS_FILE, runs)

    print("  Scanning TensorBoard for evals (this may take a minute)...")
    evals = _scan_evals()
    _write_jsonl(EVALS_FILE, evals)

    elos = _scan_elos()
    _write_jsonl(ELOS_FILE, elos)

    print(f"\nRegistry built: {len(runs)} runs, {len(evals)} eval points, {len(elos)} Elo measurements")
    print(f"Location: {REGISTRY_DIR}/")


def print_summary():
    if not EVALS_FILE.exists():
        print("Registry not built yet. Run: python build_registry.py")
        return

    evals = []
    with open(EVALS_FILE) as f:
        for line in f:
            evals.append(json.loads(line))

    elos = []
    with open(ELOS_FILE) as f:
        for line in f:
            elos.append(json.loads(line))

    runs = []
    with open(RUNS_FILE) as f:
        for line in f:
            runs.append(json.loads(line))

    print(f"Registry: {len(runs)} runs, {len(evals)} evals, {len(elos)} Elo measurements")
    print(f"\nIter range: {min(e['iter'] for e in evals)} - {max(e['iter'] for e in evals)}")
    print(f"Elo ladders: {len(set(e['ladder'] for e in elos))}")

    # Latest eval
    latest = max(evals, key=lambda x: x['iter'])
    print(f"\nLatest eval (iter {latest['iter']}):")
    for k in ['SH', 'SmartDmg', 'Tactical', 'Strategic', 'smart_avg']:
        if k in latest:
            print(f"  {k}: {latest[k]:.1f}%")

    # Latest Elo
    latest_elo = max(elos, key=lambda x: x['iter'])
    print(f"\nLatest Elo (iter {latest_elo['iter']}, {latest_elo['ladder']}):")
    print(f"  {latest_elo['name']}: Elo {latest_elo['elo']} [{latest_elo['ci_lo']}-{latest_elo['ci_hi']}]")

    # Best Elo ever
    best = max(elos, key=lambda x: x['elo'])
    print(f"\nBest Elo ever: {best['name']} Elo {best['elo']} [{best['ci_lo']}-{best['ci_hi']}] ({best['ladder']})")


def main():
    parser = argparse.ArgumentParser(description="Build/update persistent data registry")
    parser.add_argument("--update", action="store_true", help="Add new data only")
    parser.add_argument("--print-summary", action="store_true", help="Print summary stats")
    args = parser.parse_args()

    if args.print_summary:
        print_summary()
    else:
        build(update_only=args.update)
        print_summary()


if __name__ == "__main__":
    main()
