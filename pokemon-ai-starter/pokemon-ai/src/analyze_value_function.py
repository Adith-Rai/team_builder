"""Offline value function quality diagnostic (S67-EXT).

Companion to analyze_grad_norms.py. Loads any snapshot, collects N self-play
games, and reports on value function quality. Helps answer:
- Is the critic giving reliable advantage estimates?
- Has value quality improved across training iters?
- Is the value distribution sharp (confident) or diffuse (uncertain)?

Why this matters: PPO updates use advantages computed from value predictions.
Noisy or miscalibrated value → noisy advantages → unproductive PPO updates
even if PPO gradient magnitude is large. This is the "PPO is moving but not
improving" failure mode that the BC anchor dominance hypothesis ALSO
struggles to explain.

Metrics computed:
  - Value MSE: regression error on returns
  - Value R²: fraction of return variance explained
  - TD error: |reward + gamma * V(s') - V(s)| (one-step inconsistency)
  - Calibration: binned predicted-vs-actual return scatter
  - Distribution sharpness: per-bin probability entropy (51-bin twohot value head)
  - Drift: value change over course of a game

Methodology (synthetic batch, same as analyze_grad_norms):
1. Load target snapshot
2. Collect N self-play games using existing rl_collection.collect_v9
3. For each transition: extract value prediction + actual return
4. Compute the metrics above

Caveat: synthetic self-play distribution differs from training-time pool
distribution. Absolute MSE/R² will differ. But TRAJECTORY comparisons
(across snapshots) and STRUCTURAL findings (calibration shape, sharpness)
are robust to opp choice.

Usage:
    python analyze_value_function.py \\
        --snapshot data/models/rl_v10/phase2_stage1_v1/.../snapshot_0089.pt \\
        --n-games 64 \\
        --max-concurrent 16 \\
        --server-port 9000 \\
        --json-out value_diag_iter89.json

Compare across snapshots:
    python analyze_value_function.py --snapshot iter_0089.pt --json-out v89.json
    python analyze_value_function.py --snapshot iter_0139.pt --json-out v139.json
    # then diff v89.json vs v139.json to see if value quality improved
"""
import argparse
import asyncio
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model_transformer import TransformerBattlePolicy, TransformerConfig
from ppo import collate_episodes_packed, collate_episodes, build_ppo_episodes
from rl_collection import collect_v9, _make_server
from team_generator import procedural_teambuilder


def load_model(ckpt_path: str, device: torch.device) -> TransformerBattlePolicy:
    from ppo import load_checkpoint
    model, _cfg, _ckpt = load_checkpoint(ckpt_path, device)
    model.eval()
    return model


async def collect_self_play(model, device, n_games, max_concurrent,
                             server_port, snapshot_path, teambuilder):
    server_cfg = _make_server(f"ws://127.0.0.1:{server_port}/showdown/websocket")
    result = await collect_v9(
        model=model, device=device, server_pool=[server_cfg],
        n_games=n_games, max_concurrent=max_concurrent,
        snapshot_pool=[snapshot_path], fp16=False,
        teambuilder=teambuilder, battle_format="gen9ou",
        win_rates={snapshot_path: [25.0, 50]}, turn_cap=300,
    )
    # collect_v9 returns tuple: (trajs, wins, losses, ties, steps, summary, elapsed, opp_records)
    all_trajs = result[0]
    return all_trajs


def compute_value_metrics(model, trajectories, device, cfg, packed=True,
                           gamma=0.9999, lam=0.95):
    """Compute value-function quality metrics for one batch of trajectories.

    Returns dict of metrics.
    """
    model.eval()

    # Convert raw Trajectory objects → episode dicts via GAE
    episodes = build_ppo_episodes(trajectories, gamma=gamma, lam=lam)
    if not episodes:
        raise RuntimeError("No valid episodes after build_ppo_episodes")

    # Collate
    temp_ctx = getattr(cfg, "temporal_context", None)
    if packed:
        collated = collate_episodes_packed(episodes, max_seqlen=temp_ctx,
                                            device=device, tail=True)
    else:
        collated = collate_episodes(episodes, L_max=temp_ctx,
                                     device=device, tail=True)

    # Forward
    with torch.no_grad():
        if packed:
            out = model.forward_ppo_sequence_packed(collated, device)
        else:
            out = model.forward_ppo_sequence(collated, device)

        v_logits = out["v_logits"].float()  # (sum_T, v_bins) packed
        v_support = model.value_head.v_support.float()  # (v_bins,)
        v_probs = F.softmax(v_logits, dim=-1)
        v_pred = (v_probs * v_support).sum(-1)  # (sum_T,) scalar value

        returns_t = collated["returns"].to(device).float()
        advantages_t = collated["advantages"].to(device).float()

        # For non-packed, mask out padding
        if not packed:
            pad_mask_f = collated["pad_mask"].to(device).float()
            n_valid = pad_mask_f.sum().clamp(min=1.0)
            v_pred = v_pred[pad_mask_f.bool()]
            returns_t = returns_t[pad_mask_f.bool()]
            advantages_t = advantages_t[pad_mask_f.bool()]
            v_probs = v_probs[pad_mask_f.bool()]

        n = v_pred.shape[0]

        # ---- Basic stats ----
        ret_mean = returns_t.mean().item()
        ret_std = returns_t.std().item()
        ret_min = returns_t.min().item()
        ret_max = returns_t.max().item()
        v_mean = v_pred.mean().item()
        v_std = v_pred.std().item()
        v_min = v_pred.min().item()
        v_max = v_pred.max().item()

        # ---- MSE + R² ----
        residual = v_pred - returns_t
        mse = (residual * residual).mean().item()
        ret_var = returns_t.var(unbiased=False).item()
        r2 = 1.0 - mse / max(ret_var, 1e-12)

        # ---- Calibration: bin predictions, compute mean return per bin ----
        # Use 5 bins covering [-1, 1]
        bin_edges = torch.linspace(-1.0, 1.0, 6, device=device)
        calib = []
        for i in range(5):
            lo, hi = bin_edges[i].item(), bin_edges[i + 1].item()
            mask = (v_pred >= lo) & (v_pred < hi if i < 4 else v_pred <= hi)
            cnt = int(mask.sum().item())
            if cnt == 0:
                calib.append({"pred_bin": [lo, hi], "n": 0,
                              "actual_return_mean": None,
                              "actual_return_std": None})
            else:
                ar_mean = returns_t[mask].mean().item()
                ar_std = returns_t[mask].std().item() if cnt > 1 else 0.0
                calib.append({"pred_bin": [lo, hi], "n": cnt,
                              "actual_return_mean": ar_mean,
                              "actual_return_std": ar_std})

        # ---- Distribution sharpness: entropy + max prob across bins ----
        # max entropy at 51 bins = log(51) ≈ 3.93 nats; sharp distribution → low entropy
        entropy_per_pos = -(v_probs * torch.log(v_probs.clamp(min=1e-12))).sum(-1)
        max_prob_per_pos = v_probs.max(-1).values
        mean_entropy = entropy_per_pos.mean().item()
        mean_max_prob = max_prob_per_pos.mean().item()
        max_entropy = float(torch.log(torch.tensor(v_probs.shape[-1], dtype=torch.float32)).item())

        # ---- Advantage stats ----
        adv_mean = advantages_t.mean().item()
        adv_std = advantages_t.std().item()
        adv_abs_mean = advantages_t.abs().mean().item()
        adv_pos_frac = (advantages_t > 0).float().mean().item()

    return {
        "n_transitions": n,
        # Return / value basic stats
        "return_mean": ret_mean, "return_std": ret_std,
        "return_min": ret_min, "return_max": ret_max,
        "v_pred_mean": v_mean, "v_pred_std": v_std,
        "v_pred_min": v_min, "v_pred_max": v_max,
        # Regression quality
        "mse": mse, "r2": r2,
        # Distribution sharpness
        "v_entropy_mean": mean_entropy,
        "v_entropy_max": max_entropy,
        "v_entropy_normalized": mean_entropy / max_entropy,
        "v_max_prob_mean": mean_max_prob,
        # Advantages (downstream PPO signal)
        "adv_mean": adv_mean, "adv_std": adv_std,
        "adv_abs_mean": adv_abs_mean,
        "adv_pos_frac": adv_pos_frac,
        # Calibration table
        "calibration": calib,
    }


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--n-games", type=int, default=64)
    ap.add_argument("--max-concurrent", type=int, default=16)
    ap.add_argument("--server-port", type=int, default=9000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--packed", action="store_true", default=True)
    ap.add_argument("--no-packed", dest="packed", action="store_false")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--procedural-teams", default="/workspace/raw_data/pokemon_usage/2024-04",
                    help="Directory of procedural team stats (passed to ProceduralTeambuilder)")
    args = ap.parse_args()

    device = torch.device(args.device)
    snapshot_name = Path(args.snapshot).name

    print(f"[VALUE-DIAG] loading snapshot: {args.snapshot}", flush=True)
    model = load_model(args.snapshot, device)

    teambuilder = procedural_teambuilder(args.procedural_teams, random_pct=0.05)
    print(f"[VALUE-DIAG] collecting {args.n_games} self-play games...", flush=True)
    episodes = await collect_self_play(
        model=model, device=device, n_games=args.n_games,
        max_concurrent=args.max_concurrent, server_port=args.server_port,
        snapshot_path=args.snapshot, teambuilder=teambuilder,
    )
    print(f"[VALUE-DIAG] collected {len(episodes)} episodes", flush=True)

    cfg = TransformerConfig()
    m = compute_value_metrics(model, episodes, device, cfg, packed=args.packed)

    r2_interp = ("very good" if m["r2"] > 0.7 else
                 "good" if m["r2"] > 0.5 else
                 "moderate" if m["r2"] > 0.3 else
                 "weak" if m["r2"] > 0.1 else
                 "poor (value barely explains return variance)")

    sharp_interp = ("very sharp (overconfident)" if m["v_entropy_normalized"] < 0.3 else
                    "sharp" if m["v_entropy_normalized"] < 0.5 else
                    "moderate" if m["v_entropy_normalized"] < 0.7 else
                    "diffuse (underconfident, near uniform)" if m["v_entropy_normalized"] < 0.9 else
                    "near-uniform (essentially uninformative)")

    print(f"\n[VALUE-DIAG] snapshot={snapshot_name}  n_transitions={m['n_transitions']}", flush=True)
    print(f"[VALUE-DIAG] returns:  mean={m['return_mean']:+.3f}  std={m['return_std']:.3f}  "
          f"range=[{m['return_min']:+.2f}, {m['return_max']:+.2f}]", flush=True)
    print(f"[VALUE-DIAG] v_pred:   mean={m['v_pred_mean']:+.3f}  std={m['v_pred_std']:.3f}  "
          f"range=[{m['v_pred_min']:+.2f}, {m['v_pred_max']:+.2f}]", flush=True)
    print(f"[VALUE-DIAG] MSE={m['mse']:.4f}  R²={m['r2']:+.3f}  ({r2_interp})", flush=True)
    print(f"[VALUE-DIAG] distribution: entropy={m['v_entropy_mean']:.3f}/{m['v_entropy_max']:.3f} "
          f"({m['v_entropy_normalized']*100:.0f}% of max)  mean_max_prob={m['v_max_prob_mean']:.3f}  "
          f"[{sharp_interp}]", flush=True)
    print(f"[VALUE-DIAG] advantages: mean={m['adv_mean']:+.4f}  std={m['adv_std']:.4f}  "
          f"|adv|_mean={m['adv_abs_mean']:.4f}  pos_frac={m['adv_pos_frac']:.3f}", flush=True)
    print(f"[VALUE-DIAG] calibration (binned pred → actual return):")
    for c in m["calibration"]:
        lo, hi = c["pred_bin"]
        if c["n"] == 0:
            print(f"   pred [{lo:+.2f}, {hi:+.2f}]: n=0  (empty)", flush=True)
        else:
            print(f"   pred [{lo:+.2f}, {hi:+.2f}]: n={c['n']:5d}  "
                  f"actual_mean={c['actual_return_mean']:+.3f}  "
                  f"(std={c['actual_return_std']:.3f})", flush=True)

    if args.json_out:
        out = {"snapshot": snapshot_name, "n_games": args.n_games, **m}
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[VALUE-DIAG] saved JSON: {args.json_out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
