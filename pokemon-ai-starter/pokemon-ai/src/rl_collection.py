# rl_collection.py — Self-play data collection for PPO training.
#
# Extracted from rl_train_v9.py during Session 34 refactor.
# collect_v9: async collection against uniform snapshot pool
# BackgroundCollector: pipelined collection in background thread

from __future__ import annotations

import asyncio
import gc
import random
import time
import traceback
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from poke_env.player import Player
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from model import PokeTransformer
from ppo import Trajectory, _cancel_listener
# teams_ou.random_pool_teambuilder (the 70 hand-curated eval teams) is intentionally
# NOT imported here — it's eval-only. Training callers must pass a procedural
# teambuilder via the teambuilder= kwarg below; we raise if not.
from inference_batcher import InferenceBatcher
from rl_player import V9RLPlayer, SelfPlayOpponent

import os
_pid_tag = os.getpid() % 10000
_collect_round = 0


@dataclass
class PoolEntry:
    """Unified PFSP pool entry — local snapshot, in-process adapter, or external user.

    `key` is what gets stored in the win_rates dict. For local entries that's
    the checkpoint path (preserves backward compat with existing keys); for
    external adapters it's a stable display name like "foulplay".

    Three flavors of external entry:

    - **In-process adapter** (`factory` set): the factory builds a poke-env
      Player in our process. We face it via `player.battle_against(opp, n)`.
      Used by PokeEnginePlayer.

    - **External Showdown user** (`showdown_username` set, no factory): the
      opponent is a separate process (e.g. a Metamon subprocess running
      `metamon_accept_serve.py`) connected to the same Showdown server. We
      face it via `player.send_challenges(username, n)`. Used by Metamon
      because its dep stack (torch>=2.6, poke-env fork) conflicts with ours.

    - **Local snapshot** (`path` set, kind='local'): the existing
      SelfPlayOpponent path.
    """
    kind: str  # 'local' or 'external'
    key: str
    path: Optional[str] = None                       # local: .pt file
    factory: Optional[Callable[..., Player]] = None  # external in-process
    factory_kwargs: dict = field(default_factory=dict)
    showdown_username: Optional[str] = None          # external subprocess
    weight: float = 1.0


def _coerce_entry(item: Union[str, "PoolEntry"]) -> "PoolEntry":
    """Wrap a bare path string as a local PoolEntry — preserves backward
    compatibility with code that passes `snapshot_pool: List[str]`."""
    if isinstance(item, PoolEntry):
        return item
    return PoolEntry(kind="local", path=item, key=item)


def _entry_key(item) -> str:
    return _coerce_entry(item).key


def _make_server(ws_url: str) -> ServerConfiguration:
    ws = ws_url.strip().rstrip("/")
    if ws.isdigit():
        ws = f"ws://127.0.0.1:{ws}/showdown/websocket"
    elif not ws.endswith("/showdown/websocket"):
        ws += "/showdown/websocket"
    if not ws.startswith("ws://"):
        ws = "ws://" + ws
    http = ws.replace("ws://", "http://").replace("/showdown/websocket", "/action.php?")
    return ServerConfiguration(ws, http)


def pfsp_sample(
    snapshot_pool: List[Union[str, PoolEntry]],
    win_rates: Dict[str, list],
    n_opponents: int = 15,
    uniform_frac: float = 0.15,
    latest_snapshot: Optional[str] = None,
) -> List[Union[str, PoolEntry]]:
    """Select opponents using Prioritized Fictitious Self-Play (PFSP).

    Weights each entry by (1 - win_rate)^2 × entry.weight: harder opponents
    are sampled more often. A fraction of slots are filled by uniform random
    sampling for anti-forgetting (re-tests opponents with stale ratings).

    Args:
        snapshot_pool: list of paths (legacy) and/or PoolEntry objects
        win_rates: {entry_key: [wins, games]} — missing = default 0.5
        n_opponents: total opponents to select
        uniform_frac: fraction of slots for uniform random (anti-forgetting)
        latest_snapshot: always include this checkpoint path if provided

    Returns:
        list of selected pool items (originals, not coerced — preserves type)
    """
    pool_size = len(snapshot_pool)
    if pool_size <= n_opponents:
        return list(snapshot_pool)

    # Compute PFSP weights: (1 - win_rate)^2 × entry.weight
    weights = np.empty(pool_size, dtype=np.float64)
    for i, item in enumerate(snapshot_pool):
        entry = _coerce_entry(item)
        wr_data = win_rates.get(entry.key)
        if wr_data and wr_data[1] > 0:
            wr = wr_data[0] / wr_data[1]
        else:
            wr = 0.5  # unknown opponent — assume even match
        weights[i] = ((1.0 - wr) ** 2) * float(entry.weight or 1.0)

    # Prevent all-zero weights (if model wins 100% vs everything)
    if weights.sum() < 1e-12:
        weights[:] = 1.0

    probs = weights / weights.sum()

    # Split between PFSP-weighted and uniform
    n_uniform = max(1, int(n_opponents * uniform_frac))
    n_pfsp = n_opponents - n_uniform

    # Reserve a slot for latest if needed
    has_latest = latest_snapshot is not None
    if has_latest:
        n_pfsp = max(0, n_pfsp - 1)

    # PFSP weighted sample (without replacement)
    pfsp_indices = np.random.choice(pool_size, size=min(n_pfsp, pool_size),
                                    replace=False, p=probs)
    selected_set = set(pfsp_indices)

    # Uniform random sample from remaining pool
    remaining = [i for i in range(pool_size) if i not in selected_set]
    if remaining and n_uniform > 0:
        uniform_indices = random.sample(remaining, min(n_uniform, len(remaining)))
        selected_set.update(uniform_indices)

    # Always include latest (matched by key)
    if has_latest:
        latest_idx = None
        for i, item in enumerate(snapshot_pool):
            if _entry_key(item) == latest_snapshot:
                latest_idx = i
                break
        if latest_idx is not None:
            selected_set.add(latest_idx)

    return [snapshot_pool[i] for i in selected_set]


async def collect_v9(
    model: PokeTransformer, device: torch.device,
    server_pool: List[ServerConfiguration],
    n_games: int = 200, max_concurrent: int = 20,
    snapshot_pool: List[Union[str, PoolEntry]] = None, fp16: bool = True,
    reward_shaper_cfg: Optional[dict] = None,
    temp_range: Tuple[float, float] = (1.0, 2.25),
    opponent_device: str = "cuda",
    latest_snapshot: Optional[str] = None,
    teambuilder=None,
    battle_format: str = "gen9ou",
    win_rates: Optional[Dict[str, list]] = None,
):
    """Pure self-play collection with batched inference.
    Plays against MULTIPLE opponents per iteration (uniform from pool, max 15).
    Latest snapshot gets temp randomization; historical play at full strength."""
    global _collect_round
    _collect_round += 1
    rid = _collect_round

    if not snapshot_pool:
        raise ValueError("snapshot_pool must contain at least one checkpoint")

    # Select opponents via PFSP (prioritized) or uniform fallback.
    # 15 balances diversity (more opponents = broader training signal) against
    # GPU memory (each opponent loads a separate model copy for inference).
    max_opponents = 15
    if len(snapshot_pool) <= max_opponents:
        selected = list(snapshot_pool)
    elif win_rates is not None:
        selected = pfsp_sample(snapshot_pool, win_rates, max_opponents,
                               uniform_frac=0.15, latest_snapshot=latest_snapshot)
    else:
        selected = random.sample(snapshot_pool, max_opponents)
        if latest_snapshot and not any(_entry_key(s) == latest_snapshot for s in selected):
            selected[-1] = next((s for s in snapshot_pool if _entry_key(s) == latest_snapshot),
                                selected[-1])

    # Distribute games across opponents (roughly equal)
    games_per_opp = max(1, n_games // len(selected))
    remainder = n_games - games_per_opp * len(selected)

    rs_cfg = reward_shaper_cfg or {"ko_coef": 0.05, "hp_coef": 0.02, "clip_abs": 2.0}
    all_trajs = []
    total_wins, total_losses, total_ties, total_steps = 0, 0, 0, 0
    opp_results = []
    opp_records = {}  # {checkpoint_path: [wins, games]} for PFSP update
    t0 = time.time()

    # --- Parallel opponent collection ---
    n_servers = len(server_pool)
    conc_per_pair = max_concurrent

    async def _play_one_opponent(oi, opp_item, n_battles, batcher, srv, batch_id):
        """Play n_battles against one opponent. Returns (trajs, wins, losses, ties,
        short_name, opp_key). `opp_item` is either a checkpoint path string (legacy)
        or a PoolEntry — for external entries we instantiate via entry.factory."""
        entry = _coerce_entry(opp_item)
        opp_name = Path(entry.path).stem if entry.kind == "local" and entry.path else entry.key

        # Training MUST be passed a procedural teambuilder. The previous silent
        # fallback to random_pool_teambuilder() (= the 70 hand-curated eval teams)
        # caused thousands of iters of training on the same teams. The 70-team
        # pool is for eval only; if you reach this branch with teambuilder=None,
        # the call site is misconfigured.
        if teambuilder is None:
            raise RuntimeError(
                "rl_collection.collect_v9 requires teambuilder=. The previous "
                "fallback to random_pool_teambuilder() (the 70 static eval teams) "
                "is removed; use procedural_teambuilder(stats_dir) instead. "
                "If you really mean to use the eval teams, pass them explicitly."
            )
        tb = teambuilder
        player = V9RLPlayer(
            batcher=batcher, device=device,
            reward_shaper_cfg=rs_cfg,
            temperature=1.0,
            turn_cap=300,
            battle_format=battle_format,
            team=tb,
            max_concurrent_battles=conc_per_pair,
            account_configuration=AccountConfiguration(f"RL{_pid_tag}r{batch_id}", None),
            server_configuration=srv,
        )

        opponent = None  # only set for local + in-process external paths
        if entry.kind == "local":
            is_latest = (latest_snapshot is not None and entry.key == latest_snapshot)
            if len(snapshot_pool) > 15 or not is_latest:
                opp_temp_range = (1.0, 1.0)
            else:
                opp_temp_range = temp_range

            # Both sides share THE SAME teambuilder instance — each .yield_team()
            # call samples independently from the same procedural source.
            opponent = SelfPlayOpponent(
                checkpoint_path=entry.path,
                device=opponent_device,
                temp_range=opp_temp_range,
                battle_format=battle_format,
                team=teambuilder,
                max_concurrent_battles=conc_per_pair,
                account_configuration=AccountConfiguration(f"Op{_pid_tag}r{batch_id}", None),
                server_configuration=srv,
            )
        elif entry.factory is not None:
            # External in-process adapter (e.g. PokeEnginePlayer). Same matched
            # teambuilder so both sides draw from the same procedural source.
            opp_tb = teambuilder
            try:
                opponent = entry.factory(
                    server_configuration=srv,
                    account_configuration=AccountConfiguration(f"Op{_pid_tag}r{batch_id}", None),
                    team=opp_tb,
                    battle_format=battle_format,
                    max_concurrent_battles=conc_per_pair,
                    **(entry.factory_kwargs or {}),
                )
            except Exception as e:
                print(f"  [ERROR] external factory for {entry.key} failed: {e}", flush=True)
                _cancel_listener(player)
                del player
                return [], 0, 0, 0, f"{entry.key}=0/0(factory)", entry.key
        elif entry.showdown_username is not None:
            # External Showdown user (subprocess) — challenge by username, no in-process opponent
            pass
        else:
            print(f"  [ERROR] PoolEntry {entry.key} has neither factory nor showdown_username; skipping",
                  flush=True)
            _cancel_listener(player)
            del player
            return [], 0, 0, 0, f"{entry.key}=0/0(misconfigured)", entry.key

        try:
            if opponent is not None:
                await asyncio.wait_for(
                    player.battle_against(opponent, n_battles=n_battles),
                    timeout=max(180, n_battles * 25),
                )
            else:
                # Subprocess opponent — challenge their username and play out n battles.
                # Throughput is capped by the subprocess's parallel_actors (typically 1
                # for a Metamon agent on a CPU-bound transformer step), so allow lots
                # of wall time. Each Metamon turn is 100-500ms even on GPU; a battle
                # of 30-60 turns × parallel_actors=1 is ~minutes per game. Generous
                # default; PFSP weight controls how often a slow opponent gets sampled.
                await asyncio.wait_for(
                    player.send_challenges(entry.showdown_username, n_challenges=n_battles),
                    timeout=max(900, n_battles * 600),
                )
        except asyncio.TimeoutError:
            print(f"  [WARN] Timed out vs {opp_name} after {n_battles} games", flush=True)
        except Exception as e:
            print(f"  [ERROR] vs {opp_name}: {e}", flush=True)

        w, l = player.n_won_battles, player.n_lost_battles
        trajs = list(player.completed_trajectories)
        ties = player.n_tied_battles
        short = opp_name.replace("snapshot_", "sp").replace("BEST_PPO_iter80_h2h_52.8pct", "init")

        try:
            player.reset_battles()
        except EnvironmentError:
            pass
        if opponent is not None:
            try:
                opponent.reset_battles()
            except EnvironmentError:
                pass
        _cancel_listener(player)
        if opponent is not None:
            _cancel_listener(opponent)
        del player
        if opponent is not None:
            del opponent

        return trajs, w, l, ties, f"{short}={w}/{w+l}", entry.key

    # Build opponent tasks
    opp_tasks = []
    for oi, opp_item in enumerate(selected):
        n = games_per_opp + (1 if oi < remainder else 0)
        if n <= 0:
            continue
        opp_tasks.append((oi, opp_item, n))

    # Process in waves of n_servers (parallel within wave, sequential across waves)
    for wave_start in range(0, len(opp_tasks), n_servers):
        wave = opp_tasks[wave_start:wave_start + n_servers]

        # One shared batcher for the wave
        batcher = InferenceBatcher(
            model, device, fp16=fp16,
            min_batch=min(8, conc_per_pair * len(wave)),
            timeout_ms=15,
        )

        coros = []
        for wi, (oi, opp_item, n) in enumerate(wave):
            batch_id = rid * 100 + oi
            srv = server_pool[wi % n_servers]
            coros.append(_play_one_opponent(oi, opp_item, n, batcher, srv, batch_id))

        wave_results = await asyncio.gather(*coros, return_exceptions=True)

        for result in wave_results:
            if isinstance(result, Exception):
                print(f"  [ERROR] Wave opponent failed: {result}", flush=True)
                continue
            trajs, w, l, ties, summary, opp_key = result
            all_trajs.extend(trajs)
            total_wins += w
            total_losses += l
            total_ties += ties
            opp_results.append(summary)
            # Track per-opponent results for PFSP win rate updates
            games = w + l
            if games > 0:
                rec = opp_records.get(opp_key, [0, 0])
                rec[0] += w
                rec[1] += games
                opp_records[opp_key] = rec

        # Print batcher profiling for this wave
        prof = batcher.prof_summary()
        wave_idx = wave_start // n_servers
        print(f"  [PROF] wave {wave_idx}: {prof}", flush=True)
        del batcher

    elapsed = time.time() - t0
    total_steps = sum(len(t) for t in all_trajs)
    opp_summary = " ".join(opp_results)
    gc.collect()

    return all_trajs, total_wins, total_losses, total_ties, total_steps, opp_summary, elapsed, opp_records


class BackgroundCollector:
    """Runs collection in a background thread with a model copy.
    Allows PPO update and collection to overlap on GPU.

    With cpu_inference=True, the background model runs on CPU to avoid
    GPU contention with PPO. This frees the GPU entirely for training.
    """

    def __init__(self, cpu_inference: bool = False):
        self._thread: Optional[threading.Thread] = None
        self._result = None
        self._error = None
        self.cpu_inference = cpu_inference

    def start(self, model, device, server_pool, snapshot_pool, args_dict, win_rates=None):
        """Start background collection with a deepcopy of the model."""
        self._win_rates = win_rates
        collect_model = deepcopy(model)

        # CPU inference: move model copy to CPU, zero GPU contention with PPO
        if self.cpu_inference:
            collect_device = torch.device("cpu")
            collect_model = collect_model.to(collect_device)
            collect_fp16 = False  # no FP16 on CPU
            collect_opp_device = "cpu"
        else:
            collect_device = device
            collect_fp16 = args_dict["fp16"]
            collect_opp_device = args_dict["opponent_device"]

        collect_model.eval()

        self._result = None
        self._error = None
        self._thread = threading.Thread(
            target=self._run,
            args=(collect_model, collect_device, collect_fp16, collect_opp_device,
                  server_pool, snapshot_pool, args_dict, self._win_rates),
            daemon=True,
        )
        self._thread.start()

    def _run(self, collect_model, device, fp16, opp_device, server_pool, snapshot_pool, a, win_rates):
        try:
            loop = asyncio.new_event_loop()
            latest_sp = _entry_key(snapshot_pool[-1]) if len(snapshot_pool) > 1 else None
            self._result = loop.run_until_complete(
                collect_v9(
                    collect_model, device, server_pool,
                    n_games=a["games_per_iter"],
                    max_concurrent=a["max_concurrent"],
                    snapshot_pool=snapshot_pool,
                    fp16=fp16,
                    reward_shaper_cfg=a["rs_cfg"],
                    temp_range=a["temp_range"],
                    opponent_device=opp_device,
                    latest_snapshot=latest_sp,
                    teambuilder=a.get("teambuilder"),
                    win_rates=win_rates,
                )
            )
            loop.close()
        except Exception as e:
            self._error = e
            traceback.print_exc()
        finally:
            del collect_model
            gc.collect()

    def join(self):
        """Wait for background collection to finish. Returns result tuple or None."""
        if self._thread is None:
            return None
        self._thread.join()
        self._thread = None
        if self._error:
            print(f"  [ERROR] Background collection failed: {self._error}", flush=True)
            return None
        return self._result

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()
