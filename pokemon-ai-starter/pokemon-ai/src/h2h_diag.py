# h2h_diag.py — head-to-head between two ckpts with replay saving.
#
# Session 50 throwaway. Used to compare iter-79 (degraded) vs BC anchors
# (epoch_003 / snapshot_0019) head-to-head, with replays for analyze_eval.py.
#
# Usage:
#   python h2h_diag.py --p1-ckpt <path> --p2-ckpt <path> \
#       --p1-label epoch3 --p2-label sp79 \
#       --n-battles 200 --replay-dir data/replays/h2h_diag/epoch3_vs_sp79

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

import torch
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from battle_agent import BattleAgent
from battle_agent_transformer import BattleAgentTransformer, is_transformer_checkpoint


def make_player(label: str, ckpt_path: str, ckpt: dict, device: str,
                server: ServerConfiguration, tb, save_replays: str = None,
                concurrency: int = 20, perm: bool = False, fp16: bool = False):
    AgentClass = BattleAgentTransformer if is_transformer_checkpoint(ckpt) else BattleAgent
    kwargs = {}
    if isinstance(AgentClass, type) and AgentClass is BattleAgentTransformer:
        kwargs["fp16"] = fp16
    player = AgentClass(
        ckpt_path, device=device, _cached_ckpt=ckpt,
        account_configuration=AccountConfiguration.generate(label[:6], rand=True),
        battle_format="gen9ou", max_concurrent_battles=concurrency,
        server_configuration=server, team=tb,
        save_replays=save_replays if save_replays else False,
        **kwargs,
    )

    # Per-instance training-mode patch (perm features at eval).
    # Replaces this player's `_build_turn_batch` method only.
    if perm:
        from features import build_turn_batch as _btb
        def _build_perm(self, feat: dict) -> dict:
            return _btb(feat, device=self.device, training=True)
        # Bind as method on the instance.
        player._build_turn_batch = _build_perm.__get__(player, type(player))
        print(f"  [DIAG] {label}: training=True (perm features)", flush=True)

    return player


def make_teambuilder(team_set: str):
    if team_set == "metamon-competitive":
        from eval_metamon_competitive import MetamonCompetitiveTeambuilder
        return MetamonCompetitiveTeambuilder()
    elif team_set == "pool":
        from teams_ou import random_pool_teambuilder
        return random_pool_teambuilder()
    elif team_set == "procedural":
        from team_generator import procedural_teambuilder
        from pathlib import Path as _P
        canon = _P(__file__).resolve().parents[3] / "raw_data" / "pokemon_usage" / "2024-04"
        return procedural_teambuilder(str(canon), random_pct=0.05)
    else:
        raise ValueError(f"unknown --team-set {team_set!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--p1-ckpt", required=True)
    p.add_argument("--p1-label", required=True)
    p.add_argument("--p2-ckpt", required=True)
    p.add_argument("--p2-label", required=True)
    p.add_argument("--n-battles", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--server", default="ws://127.0.0.1:9000/showdown/websocket")
    p.add_argument("--replay-dir", default=None,
                   help="If set, saves replays for both sides under <dir>/<label>/")
    p.add_argument("--p1-perm", action="store_true", help="p1 uses training=True features")
    p.add_argument("--p2-perm", action="store_true", help="p2 uses training=True features")
    p.add_argument("--p1-fp16", action="store_true", help="p1 uses fp16 forward")
    p.add_argument("--p2-fp16", action="store_true", help="p2 uses fp16 forward")
    p.add_argument("--team-set", default="metamon-competitive",
                   choices=["metamon-competitive", "pool", "procedural"])
    args = p.parse_args()

    # Pre-load both ckpts.
    p1_ckpt = torch.load(args.p1_ckpt, map_location=args.device, weights_only=False)
    p2_ckpt = torch.load(args.p2_ckpt, map_location=args.device, weights_only=False)

    # Replay dirs.
    rd1 = rd2 = None
    if args.replay_dir:
        rd1 = os.path.join(args.replay_dir, args.p1_label)
        rd2 = os.path.join(args.replay_dir, args.p2_label)
        os.makedirs(rd1, exist_ok=True)
        os.makedirs(rd2, exist_ok=True)
        print(f"[h2h] saving replays under {args.replay_dir}", flush=True)

    server = ServerConfiguration(args.server, None)
    tb = make_teambuilder(args.team_set)

    async def _run():
        p1 = make_player(args.p1_label, args.p1_ckpt, p1_ckpt, args.device,
                         server, tb, save_replays=rd1, concurrency=args.concurrency,
                         perm=args.p1_perm, fp16=args.p1_fp16)
        p2 = make_player(args.p2_label, args.p2_ckpt, p2_ckpt, args.device,
                         server, tb, save_replays=rd2, concurrency=args.concurrency,
                         perm=args.p2_perm, fp16=args.p2_fp16)

        t0 = time.time()
        await p1.battle_against(p2, n_battles=args.n_battles)
        elapsed = time.time() - t0

        w1 = p1.n_won_battles
        l1 = p1.n_lost_battles
        t1 = p1.n_tied_battles
        total = w1 + l1 + t1
        wr = 100 * w1 / max(1, total)
        return w1, l1, t1, total, elapsed, wr

    flags = []
    if args.p1_perm: flags.append(f"{args.p1_label}=perm")
    if args.p2_perm: flags.append(f"{args.p2_label}=perm")
    if args.p1_fp16: flags.append(f"{args.p1_label}=fp16")
    if args.p2_fp16: flags.append(f"{args.p2_label}=fp16")
    flags_str = " | ".join(flags) if flags else "default (canonical, fp32)"
    print(f"\n=== H2H: {args.p1_label} vs {args.p2_label}  team_set={args.team_set}  [{flags_str}] ===", flush=True)
    print(f"  p1 = {args.p1_ckpt}", flush=True)
    print(f"  p2 = {args.p2_ckpt}", flush=True)
    print(f"  n_battles={args.n_battles}, concurrency={args.concurrency}", flush=True)

    w, l, t, total, elapsed, wr = asyncio.run(_run())
    print(f"\n  RESULT: {args.p1_label} W/L/T = {w}/{l}/{t} of {total}, "
          f"wr = {wr:.1f}%  ({elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
