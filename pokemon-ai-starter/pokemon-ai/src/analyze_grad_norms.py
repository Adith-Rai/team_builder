"""Offline grad-norm decomposition analyzer (S67-EXT).

Companion to the live --diag-grad-norms flag in train_rl.py / ppo.py. Runs
the same PPO-vs-BC grad-norm decomp on ANY snapshot retroactively.

Use case: retrospective trajectory analysis. Run on snapshots iter 9, 19,
..., 89 of Stage 1 (and lr3e5 / diversity_v1 snaps) to see HOW the BC/PPO
balance evolved across training. Live diag only gives forward-going data;
this fills in the trajectory.

Methodology (synthetic batch):
1. Load target snapshot + BC reference
2. Collect N self-play games (model vs itself) using existing rl_collection
3. Compute GAE advantages on those episodes
4. Forward pass through model + BC ref
5. Compute PPO loss components (pi, v, ent) + BC kl loss
6. Backward each separately via linearity trick
7. Report grad norms + cosine similarity

Caveat: synthetic batch uses self-play, NOT the pool composition the
snapshot was trained against. So the absolute grad norms differ from
iter-time. BUT the RATIO (bc/ppo) and cosine are largely robust to opp
choice — they're properties of the model+BC pair, not the opp.

Usage:
    python analyze_grad_norms.py \\
        --snapshot data/models/rl_v10/phase2_stage1_v1/.../snapshot_0089.pt \\
        --bc-anchor data/models/bc/v10_padded_for_cis_dev.pt \\
        --bc-anchor-coef 0.10 \\
        --n-games 64 \\
        --max-concurrent 16 \\
        --server-port 9000

Output:
    [GRAD-OFFLINE] snapshot=snapshot_0089.pt n_games=64 n_transitions=4823
    [GRAD-OFFLINE] ppo_norm=0.034  bc_norm=0.083  bc/ppo=2.44x (BC-DOMINATED)  cos=-0.22
"""
import argparse
import asyncio
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Local imports — match train_rl.py setup
from model_transformer import TransformerBattlePolicy
from agent_v9 import V9RLPlayer
from features import TransformerConfig
from ppo import (
    collate_episodes, collate_episodes_packed,
    _ppo_loss_batched_internal, _ppo_loss_packed_internal,
    forward_ppo_sequence,
)
from rl_collection import collect_v9
from poke_env.player import LocalhostServerConfiguration


def load_model(ckpt_path: str, device: torch.device) -> TransformerBattlePolicy:
    """Load a TransformerBattlePolicy from a checkpoint."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        sd = state["model_state_dict"]
        cfg = state.get("model_config") or state.get("cfg")
    elif isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
        cfg = state.get("config") or state.get("cfg")
    else:
        sd = state
        cfg = None

    if cfg is None:
        # Fall back to default config; check shapes via load_state_dict
        cfg = TransformerConfig()

    model = TransformerBattlePolicy(cfg).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [load] missing={len(missing)} unexpected={len(unexpected)} "
              f"keys (likely OK if BC vs RL diff)", flush=True)
    model.eval()
    return model


async def collect_self_play(
    model: TransformerBattlePolicy,
    device: torch.device,
    n_games: int,
    max_concurrent: int,
    server_port: int,
    snapshot_path: str,
) -> list:
    """Collect N self-play games. Returns list of episode dicts."""
    server_cfg = LocalhostServerConfiguration._replace(
        websocket_url=f"ws://localhost:{server_port}/showdown/websocket"
    )

    # Self-play: snapshot_pool with just the snapshot itself
    # Use random teambuilder (avoid needing procedural)
    episodes = await collect_v9(
        model=model,
        device=device,
        server_pool=[server_cfg],
        n_games=n_games,
        max_concurrent=max_concurrent,
        snapshot_pool=[snapshot_path],
        fp16=False,
        teambuilder=None,  # use default
        battle_format="gen9ou",
        win_rates={snapshot_path: [25.0, 50]},  # 50/50 placeholder
        turn_cap=300,
    )
    return episodes


def analyze_grad_norms(
    model: TransformerBattlePolicy,
    bc_ref: TransformerBattlePolicy,
    episodes: list,
    device: torch.device,
    cfg,
    bc_anchor_coef: float,
    ent_coef: float = 0.02,
    vf_coef: float = 0.5,
    clip_eps: float = 0.2,
    packed: bool = True,
) -> dict:
    """Compute PPO/BC grad-norm decomp on one batch of episodes.

    Returns: dict with ppo_norm, bc_norm, cos, ppo/bc ratio.
    Mirrors the ppo_update_batched diag_grad_norms path.
    """
    model.train()
    bc_ref.eval()

    # Collate
    temp_ctx = getattr(cfg, "temporal_context", None)
    if packed:
        collated = collate_episodes_packed(episodes, max_seqlen=temp_ctx,
                                            device=device, tail=True)
    else:
        collated = collate_episodes(episodes, L_max=temp_ctx,
                                     device=device, tail=True)

    # Forward both
    with torch.no_grad():
        if packed:
            bc_out = bc_ref.forward_ppo_sequence_packed(collated, device)
        else:
            bc_out = bc_ref.forward_ppo_sequence(collated, device)
        bc_logits = bc_out["action_logits"].detach()

    if packed:
        forward_out = model.forward_ppo_sequence_packed(collated, device)
    else:
        forward_out = model.forward_ppo_sequence(collated, device)

    # Compute losses (uses the SAME internal as ppo_update_batched)
    _loss_fn = (_ppo_loss_packed_internal if packed
                else _ppo_loss_batched_internal)
    loss_dict = _loss_fn(
        collated, forward_out, model, cfg,
        ent_coef=ent_coef, vf_coef=vf_coef, clip_eps=clip_eps,
        normalize_advantages=False,
        bc_logits=bc_logits, bc_anchor_coef=bc_anchor_coef,
    )

    pi_t = loss_dict["pi_loss"]
    ent_t = loss_dict["entropy"]
    v_t = loss_dict["v_loss"]
    bc_t = loss_dict["bc_kl"]

    ppo_part = pi_t - ent_coef * ent_t + vf_coef * v_t
    bc_part = bc_anchor_coef * bc_t

    # Grad decomp via linearity (same as live diag)
    model.zero_grad(set_to_none=True)
    ppo_part.backward(retain_graph=True)
    ppo_grad_sqs = []
    ppo_grad_clones = []
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().float()
            ppo_grad_sqs.append((g * g).sum())
            ppo_grad_clones.append((p, g.clone()))
    ppo_norm = float(torch.sqrt(torch.stack(ppo_grad_sqs).sum()).item())

    bc_part.backward(retain_graph=False)
    bc_grad_sqs = []
    dot_sum = torch.zeros((), device=device, dtype=torch.float32)
    ppo_sq_sum = torch.zeros((), device=device, dtype=torch.float32)
    bc_sq_sum = torch.zeros((), device=device, dtype=torch.float32)
    for (p, ppo_g) in ppo_grad_clones:
        if p.grad is None:
            continue
        bc_g = p.grad.detach().float() - ppo_g
        bc_grad_sqs.append((bc_g * bc_g).sum())
        dot_sum += (ppo_g * bc_g).sum()
        ppo_sq_sum += (ppo_g * ppo_g).sum()
        bc_sq_sum += (bc_g * bc_g).sum()
    bc_norm = float(torch.sqrt(torch.stack(bc_grad_sqs).sum()).item())
    denom = (torch.sqrt(ppo_sq_sum) * torch.sqrt(bc_sq_sum)).clamp(min=1e-12)
    cos = float((dot_sum / denom).item())

    return {
        "ppo_norm": ppo_norm,
        "bc_norm": bc_norm,
        "bc_ppo_ratio": bc_norm / max(ppo_norm, 1e-12),
        "cos": cos,
        "pi_loss": float(pi_t.item()),
        "bc_kl": float(bc_t.item()),
        "v_loss": float(v_t.item()),
        "entropy": float(ent_t.item()),
        "n_transitions": (collated["advantages"].shape[0] if packed
                          else int(collated["pad_mask"].sum().item())),
    }


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", required=True, help="Path to model snapshot .pt")
    ap.add_argument("--bc-anchor", required=True, help="Path to BC reference .pt")
    ap.add_argument("--bc-anchor-coef", type=float, default=0.10)
    ap.add_argument("--n-games", type=int, default=64,
                    help="Number of self-play games for batch (default 64)")
    ap.add_argument("--max-concurrent", type=int, default=16)
    ap.add_argument("--server-port", type=int, default=9000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ent-coef", type=float, default=0.02)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--packed", action="store_true", default=True)
    ap.add_argument("--no-packed", dest="packed", action="store_false")
    ap.add_argument("--json-out", default=None,
                    help="Optional path to write results JSON")
    args = ap.parse_args()

    device = torch.device(args.device)
    snapshot_name = Path(args.snapshot).name

    print(f"[GRAD-OFFLINE] loading snapshot: {args.snapshot}", flush=True)
    model = load_model(args.snapshot, device)

    print(f"[GRAD-OFFLINE] loading BC ref: {args.bc_anchor}", flush=True)
    bc_ref = load_model(args.bc_anchor, device)

    print(f"[GRAD-OFFLINE] collecting {args.n_games} self-play games "
          f"(max_concurrent={args.max_concurrent}, port={args.server_port})...",
          flush=True)
    episodes = await collect_self_play(
        model=model, device=device,
        n_games=args.n_games,
        max_concurrent=args.max_concurrent,
        server_port=args.server_port,
        snapshot_path=args.snapshot,
    )
    print(f"[GRAD-OFFLINE] collected {len(episodes)} episodes", flush=True)

    cfg = TransformerConfig()
    result = analyze_grad_norms(
        model=model, bc_ref=bc_ref, episodes=episodes,
        device=device, cfg=cfg,
        bc_anchor_coef=args.bc_anchor_coef,
        ent_coef=args.ent_coef, vf_coef=args.vf_coef,
        clip_eps=args.clip_eps, packed=args.packed,
    )

    ratio = result["bc_ppo_ratio"]
    dom = ("BC-DOMINATED" if ratio > 1.5 else
           "PPO-dominated" if ratio < 0.67 else
           "balanced")
    print(f"\n[GRAD-OFFLINE] snapshot={snapshot_name}  "
          f"n_games={args.n_games}  n_transitions={result['n_transitions']}",
          flush=True)
    print(f"[GRAD-OFFLINE] pi_loss={result['pi_loss']:+.4f}  "
          f"bc_kl={result['bc_kl']:.4f}  "
          f"v_loss={result['v_loss']:.4f}  "
          f"ent={result['entropy']:.3f}", flush=True)
    print(f"[GRAD-OFFLINE] ppo_norm={result['ppo_norm']:.4f}  "
          f"bc_norm={result['bc_norm']:.4f}  "
          f"bc/ppo={ratio:.2f}x ({dom})  "
          f"cos={result['cos']:+.3f}", flush=True)

    if args.json_out:
        import json
        out = {"snapshot": snapshot_name, "n_games": args.n_games, **result}
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[GRAD-OFFLINE] saved JSON: {args.json_out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
