"""Persistent data registry — fire-and-forget append helpers.

All functions are safe to call from training loops: if the write fails,
a warning is printed and execution continues. Training never crashes
because of a registry issue.

Files:
    data/eval/registry/runs.jsonl   — one line per training run start
    data/eval/registry/evals.jsonl  — one line per bot evaluation
    data/eval/registry/elos.jsonl   — one line per Elo measurement
"""
import json
from pathlib import Path

REGISTRY_DIR = Path("data/eval/registry")


def _append(path: Path, record: dict):
    """Append a JSON line to a file. Fire-and-forget."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"  [WARN] Registry append to {path} failed: {e}", flush=True)


def log_run(run_dir: str, config: dict, iter_lo: int, iter_hi: int):
    """Log a training run start. Call once when training begins."""
    _append(REGISTRY_DIR / "runs.jsonl", {
        "run_dir": run_dir,
        "iter_lo": iter_lo,
        "iter_hi": iter_hi,
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
    })


def log_eval(iter_num: int, run_dir: str, sh: float, smartdmg: float,
             tactical: float, strategic: float, smart_avg: float):
    """Log a bot evaluation result. Call after each eval during training."""
    _append(REGISTRY_DIR / "evals.jsonl", {
        "iter": iter_num,
        "run_dir": run_dir,
        "SH": round(sh, 4),
        "SmartDmg": round(smartdmg, 4),
        "Tactical": round(tactical, 4),
        "Strategic": round(strategic, 4),
        "smart_avg": round(smart_avg, 4),
    })


def log_elo(iter_num: int, name: str, elo: float, ci_lo: float, ci_hi: float,
            ladder: str, n_games: int = 0, ckpt: str = None):
    """Log an Elo measurement. Call after fitting BT in the Elo ladder."""
    _append(REGISTRY_DIR / "elos.jsonl", {
        "iter": iter_num,
        "name": name,
        "elo": round(elo, 1),
        "ci_lo": round(ci_lo, 1),
        "ci_hi": round(ci_hi, 1),
        "ladder": ladder,
        "n_games": n_games,
        "ckpt": ckpt,
    })
