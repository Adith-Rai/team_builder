# eval_diag.py — minimal smart_avg eval with overridable fp16 + training flags.
#
# Session 50 throwaway: diagnose the Phase 1 fall-off
# (smart_avg 67 → 35 over 79 iters). Tests two hypotheses:
#   1. Training feeds permuted features (training=True) but eval feeds canonical
#      → eval-time distribution shift. `--diag-training-mode` passes training=True
#      through BattleAgentTransformer at eval time.
#   2. Training is fp16, eval is fp32 → numerical drift. `--fp16` runs eval in fp16.
#
# Usage:
#   python eval_diag.py --ckpt <path> --n-games 100 [--fp16] [--diag-training-mode]

from __future__ import annotations

import argparse
import asyncio
import gc
import sys
import time
from pathlib import Path

import torch
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.player.baselines import SimpleHeuristicsPlayer

from policy_smartbots import SmartDamagePlayer, TacticalPlayer, StrategicPlayer
from battle_agent_transformer import BattleAgentTransformer
from features import build_turn_batch as _build_turn_batch_orig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-games", type=int, default=100)
    p.add_argument("--max-conc", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--server", default="ws://127.0.0.1:9000/showdown/websocket")
    p.add_argument("--fp16", action="store_true", help="Force fp16 forward in eval")
    p.add_argument("--diag-training-mode", action="store_true",
                   help="Patch _build_turn_batch to pass training=True (perm features at eval)")
    p.add_argument("--team-set", default="metamon-competitive")
    p.add_argument("--label", default="diag")
    args = p.parse_args()

    if args.diag_training_mode:
        # Monkey-patch BattleAgentTransformer's batch builder to pass training=True.
        # This is a one-line override of `battle_agent_transformer.py:104-106`.
        def _build_turn_batch_training(self, feat: dict) -> dict:
            return _build_turn_batch_orig(feat, device=self.device, training=True)
        BattleAgentTransformer._build_turn_batch = _build_turn_batch_training
        print(f"[DIAG] Patched _build_turn_batch -> training=True (permuted features)")

    if args.team_set == "metamon-competitive":
        from eval_metamon_competitive import MetamonCompetitiveTeambuilder
        _shared_tb = MetamonCompetitiveTeambuilder()
        def _make_tb():
            return _shared_tb
    elif args.team_set == "pool":
        from teams_ou import random_pool_teambuilder
        def _make_tb():
            return random_pool_teambuilder()
    else:
        raise SystemExit(f"unknown team_set={args.team_set}")

    server = ServerConfiguration(args.server, None)
    opponents = [
        (SimpleHeuristicsPlayer, "SH"),
        (SmartDamagePlayer,      "SmartDmg"),
        (TacticalPlayer,         "Tactical"),
        (StrategicPlayer,        "Strategic"),
    ]

    # Pre-load ckpt once.
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)

    async def _run():
        results = {}
        for opp_cls, opp_name in opponents:
            t0 = time.time()
            p1 = BattleAgentTransformer(
                args.ckpt, device=args.device, _cached_ckpt=ckpt, fp16=args.fp16,
                account_configuration=AccountConfiguration.generate(f"D{args.label[:5]}", rand=True),
                battle_format="gen9ou", max_concurrent_battles=args.max_conc,
                server_configuration=server, team=_make_tb(),
            )
            p2 = opp_cls(
                account_configuration=AccountConfiguration.generate(opp_name, rand=True),
                battle_format="gen9ou", max_concurrent_battles=args.max_conc,
                server_configuration=server, team=_make_tb(),
            )
            await p1.battle_against(p2, n_battles=args.n_games)
            wr = p1.n_won_battles / args.n_games * 100
            results[opp_name] = wr
            elapsed = time.time() - t0
            print(f"  vs {opp_name:12s}: {p1.n_won_battles}/{args.n_games} = {wr:5.1f}%  ({elapsed:.0f}s)", flush=True)
            try: p1.reset_battles()
            except Exception: pass
            try: p2.reset_battles()
            except Exception: pass
            del p1, p2
            gc.collect(); torch.cuda.empty_cache()
        results["smart_avg"] = sum(results[k] for k in ["SH", "SmartDmg", "Tactical", "Strategic"]) / 4
        return results

    flags = []
    if args.fp16: flags.append("fp16=True")
    if args.diag_training_mode: flags.append("training=True")
    flags_str = ", ".join(flags) or "default (fp32, training=False)"
    print(f"\n=== eval_diag {args.label}  ckpt={Path(args.ckpt).name}  {flags_str} ===", flush=True)
    print(f"team_set={args.team_set}, n_games={args.n_games} ({args.n_games*4} total)", flush=True)

    results = asyncio.run(_run())
    print(f"\n  RESULT [{args.label}]:  SH={results['SH']:.0f}%  SmartDmg={results['SmartDmg']:.0f}%  "
          f"Tactical={results['Tactical']:.0f}%  Strategic={results['Strategic']:.0f}%  "
          f"smart_avg={results['smart_avg']:.1f}%", flush=True)


if __name__ == "__main__":
    main()
