"""Stress-test heuristic bots vs snapshot at high concurrency.

Reproduces conditions similar to CIS iter 0 — high concurrency (50-100
concurrent battles), syn paired teambuilder, repeated rounds — to surface
the intermittent hang seen in Run #9 iter 1.

Instruments bot.choose_move to catch silent exceptions and log them.
Identifies stuck battles by tracking choose_move call timestamps per battle.

Usage:
  python stress_test_heur_bots.py \
    --snapshot path/to/snap.pt \
    --bots GreedySEv2,HazardSensev2,SwitchAwareEscapev3,AntiSetupBot,\\
           SetupThenSweepv2,GreedySEPlayer,HazardSensePlayer,\\
           SwitchAwareEscapePlayer,SetupThenSweepPlayer,RandomPlayer,\\
           MaxBasePowerPlayer,StrategicV2,SwitchAwareEscapeV2 \
    --n-games 100 --concurrency 50 \
    --server ws://127.0.0.1:9000/showdown/websocket
"""

import argparse
import asyncio
import os
import time
import traceback
from collections import defaultdict, Counter
from pathlib import Path

from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.ps_client.account_configuration import AccountConfiguration

from battle_agent_transformer import BattleAgentTransformer, is_transformer_checkpoint
from battle_agent import BattleAgent
from eval_elo_ladder import ALL_BOTS, resolve_server
from external_adapters import _resolve_heuristic_class
import torch


# Instrument bot.choose_move to log exceptions + call timestamps
_exceptions_per_bot = defaultdict(list)
_choose_move_call_times = defaultdict(dict)


def _instrument_bot(bot, bot_name):
    """Wrap bot.choose_move to log exceptions + track per-battle call times."""
    original_choose_move = bot.choose_move

    def wrapped(battle):
        battle_tag = getattr(battle, "battle_tag", id(battle))
        _choose_move_call_times[bot_name][battle_tag] = time.time()
        try:
            result = original_choose_move(battle)
            if result is None:
                _exceptions_per_bot[bot_name].append(
                    f"battle={battle_tag} returned None (battle.available_moves="
                    f"{[m.id for m in (battle.available_moves or [])]}, "
                    f"available_switches={[s.species for s in (battle.available_switches or [])]})")
            return result
        except Exception as e:
            tb = traceback.format_exc()
            _exceptions_per_bot[bot_name].append(
                f"battle={battle_tag} {type(e).__name__}: {e}\n{tb[:500]}")
            raise

    bot.choose_move = wrapped


def _get_team_set():
    """Use procedural teambuilder (gen9ou). Skip syn for simplicity in stress test."""
    # Just use random pool teambuilder — same as Run #7 procedural side
    from teams_ou import random_pool_teambuilder
    return random_pool_teambuilder()


async def run_one_bot(snap_path, snap_cached, bot_name, n_games, concurrency, server_cfg, teambuilder, match_idx):
    """Run one bot vs snap for n_games, log progress every 30s + final stats."""
    print(f"\n=== {bot_name} (match {match_idx}) — n={n_games}, concurrency={concurrency} ===", flush=True)
    suffix = f"{os.getpid() % 999}m{match_idx}"
    snap_account = AccountConfiguration(f"St{suffix}", None)
    bot_account = AccountConfiguration(f"B{bot_name[:5]}{suffix}", None)

    common = dict(
        battle_format="gen9ou",
        max_concurrent_battles=concurrency,
        server_configuration=server_cfg,
        team=teambuilder,
    )

    AgentClass = BattleAgentTransformer if is_transformer_checkpoint(snap_cached) else BattleAgent
    snap = AgentClass(
        checkpoint_path=snap_path, _cached_ckpt=snap_cached, device="cuda",
        save_replays=False, account_configuration=snap_account, **common,
    )
    bot_cls = ALL_BOTS[bot_name]
    bot = bot_cls(save_replays=False, account_configuration=bot_account, **common)
    _instrument_bot(bot, bot_name)

    # Race the battle against a progress monitor
    t0 = time.time()
    done_event = asyncio.Event()

    async def progress_monitor():
        last_finished = 0
        while not done_event.is_set():
            await asyncio.sleep(30.0)
            finished = bot.n_finished_battles
            elapsed = time.time() - t0
            # Find battles whose choose_move hasn't been called recently
            now = time.time()
            stale_battles = []
            for btag, last_call in _choose_move_call_times[bot_name].items():
                if now - last_call > 60:  # 60s since last move
                    stale_battles.append((btag, now - last_call))
            stale_msg = f", {len(stale_battles)} STALE>60s" if stale_battles else ""
            print(f"  [{bot_name} +{elapsed:.0f}s] {finished}/{n_games} done"
                  f" (+{finished-last_finished} since last check)"
                  f"{stale_msg}", flush=True)
            last_finished = finished

    async def main_battle():
        try:
            await asyncio.wait_for(
                bot.battle_against(snap, n_battles=n_games),
                timeout=600.0  # 10 min hard cap per bot
            )
        except asyncio.TimeoutError:
            print(f"  [{bot_name}] TIMEOUT at 600s. Finished={bot.n_finished_battles}/{n_games}", flush=True)
        finally:
            done_event.set()

    await asyncio.gather(main_battle(), progress_monitor())

    elapsed = time.time() - t0
    finished = bot.n_finished_battles
    snap_wins = snap.n_won_battles
    bot_wins = bot.n_won_battles
    ties = snap.n_tied_battles

    n_exceptions = len(_exceptions_per_bot[bot_name])
    print(f"  [{bot_name}] DONE: {finished}/{n_games} battles in {elapsed:.0f}s "
          f"({snap_wins} snap / {bot_wins} bot / {ties} ties), {n_exceptions} exceptions", flush=True)
    if _exceptions_per_bot[bot_name]:
        print(f"  [{bot_name}] EXCEPTIONS (first 5):", flush=True)
        for exc in _exceptions_per_bot[bot_name][:5]:
            print(f"    {exc}", flush=True)

    return {
        "bot": bot_name, "finished": finished, "n_games": n_games,
        "elapsed_s": round(elapsed, 1),
        "snap_wins": snap_wins, "bot_wins": bot_wins, "ties": ties,
        "n_exceptions": n_exceptions,
    }


async def main_async(snap_path, bots, n_games, concurrency, server_url):
    print(f"Loading snap: {snap_path}", flush=True)
    snap_cached = torch.load(snap_path, map_location="cuda", weights_only=False)
    print(f"Snap loaded.", flush=True)

    server_cfg = resolve_server(server_url)
    teambuilder = _get_team_set()

    results = []
    for idx, bot_name in enumerate(bots):
        # Verify bot is loadable
        try:
            if bot_name not in ALL_BOTS:
                # try heuristic factory
                _resolve_heuristic_class(bot_name)
        except Exception as e:
            print(f"[{bot_name}] SKIPPED: not loadable: {e}", flush=True)
            continue
        r = await run_one_bot(snap_path, snap_cached, bot_name, n_games, concurrency,
                              server_cfg, teambuilder, idx)
        results.append(r)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'Bot':<30} {'Done':>10} {'Wall':>8} {'Errs':>5}")
    for r in results:
        flag = " ⚠" if (r["finished"] < r["n_games"] or r["n_exceptions"] > 0) else ""
        print(f"{r['bot']:<30} {r['finished']:>4}/{r['n_games']:<5} {r['elapsed_s']:>7.0f}s {r['n_exceptions']:>5}{flag}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--bots", required=True)
    p.add_argument("--n-games", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=30)
    p.add_argument("--server", default="ws://127.0.0.1:9000/showdown/websocket")
    args = p.parse_args()

    bots = [b.strip() for b in args.bots.split(",")]
    asyncio.run(main_async(args.snapshot, bots, args.n_games, args.concurrency, args.server))


if __name__ == "__main__":
    main()
