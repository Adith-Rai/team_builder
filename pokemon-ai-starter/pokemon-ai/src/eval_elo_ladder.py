#!/usr/bin/env python3
"""Snapshot Elo ladder — round-robin across N snapshots + smart bot anchors.

Why this exists (Session 33):
  smart_avg has been our primary metric since Phase E. Session 29 documented that
  it does NOT predict H2H strength. We literally don't know our actual Elo. The
  cloud burst decision is gated on knowing whether we're at ~1700 (close to
  VGC-Bench BCFP's published 1768 at our compute scale) or at ~1500 (real local
  gap to close before scaling). This script answers that question.

Method:
  - Round-robin tournament: every player plays every other player N games
  - Players: arbitrary mix of v8 checkpoints + heuristic bots (anchors)
  - Each game uses random_pool_teambuilder() (handcrafted 70 OU teams) for
    LOWER variance per game vs procedural — better signal for measurement
  - Win matrix → Bradley-Terry MLE → Elo with bootstrap 95% CI
  - Bots act as absolute anchors; SH is fixed at 1000 by default so the scale
    is interpretable

Usage:
  # Quickest sanity run: 10 bots only, no snapshots (~12 min, validates everything works)
  python -u eval_elo_ladder.py --bots all --n-games 50 \
    --out-json data/eval/elo_bots_only.json

  # Standard run: 20 evenly-spaced snapshots + all 10 bot anchors (~2-3 hours, 50g/match)
  python -u eval_elo_ladder.py \
    --snapshot-glob "data/models/rl_v9/selfplay_v9_*/snapshot_*.pt" \
    --sample-n 20 --bots all --n-games 50 \
    --out-json data/eval/elo_ladder_v9.json

  # Faster (looser CIs): same but 30 games/match (~70 min)
  python -u eval_elo_ladder.py \
    --snapshot-glob "data/models/rl_v9/selfplay_v9_*/snapshot_*.pt" \
    --sample-n 20 --bots all --n-games 30 \
    --out-json data/eval/elo_ladder_quick.json

  # Specific checkpoints, smart bots only (much faster)
  python -u eval_elo_ladder.py \
    --snapshots data/models/rl_v9/.../snapshot_0699.pt data/models/rl_v9/.../snapshot_1724.pt \
    --names peak_57pct latest \
    --bots smart --n-games 100

  # PARALLEL: 3 shards across 3 battle servers (~3x speedup)
  # Run these in 3 separate terminals SIMULTANEOUSLY:
  python -u eval_elo_ladder.py --snapshot-glob "data/models/rl_v9/selfplay_v9_*/snapshot_*.pt" \
    --sample-n 20 --bots all --n-games 50 \
    --server ws://127.0.0.1:9000/showdown/websocket \
    --shard 0/3 --out-json data/eval/elo_shard0.json
  python -u eval_elo_ladder.py --snapshot-glob "data/models/rl_v9/selfplay_v9_*/snapshot_*.pt" \
    --sample-n 20 --bots all --n-games 50 \
    --server ws://127.0.0.1:9001/showdown/websocket \
    --shard 1/3 --out-json data/eval/elo_shard1.json
  python -u eval_elo_ladder.py --snapshot-glob "data/models/rl_v9/selfplay_v9_*/snapshot_*.pt" \
    --sample-n 20 --bots all --n-games 50 \
    --server ws://127.0.0.1:9002/showdown/websocket \
    --shard 2/3 --out-json data/eval/elo_shard2.json
  # Then combine into final ladder:
  python -u eval_elo_ladder.py --combine \
    data/eval/elo_shard0.json data/eval/elo_shard1.json data/eval/elo_shard2.json \
    --out-json data/eval/elo_ladder_final.json

Notes:
  - Defaults: 50 games/matchup, all 10 bots, BayesElo via Bradley-Terry MLE +
    bootstrap CI. SH anchored at 1000 Elo.
  - With 30 players (20 snapshots + 10 bots), 50 games/matchup, expect ~2-3 hours
    on a single battle server at concurrency 10. Bump to 9000,9001,9002 by running
    multiple instances on partitioned matchup lists (not auto-parallelized).
  - All 10 bots span ~75% win-rate spread (Random ~5% to Tactical ~82% in the
    Session 23 round-robin) → ~400 Elo of dynamic range. Snapshots interpolate
    between these anchors.

How Elo is computed:
  1. Run all matchups, build win matrix W[i][j] (i won this many vs j) and game
     count matrix N[i][j].
  2. Fit a Bradley-Terry model via Hunter (2004) MM algorithm:
        P(i beats j) = π_i / (π_i + π_j)
     where π_i is the BT strength parameter. Iterative update:
        π_i_new = W_i / Σ_j (N[i][j] / (π_i + π_j))
     (W_i = total wins by i across all opponents). Converges in ~50-200 iters
     for our scale. Identifiable up to multiplicative scale.
  3. Convert BT strengths to Elo:
        elo_i = 400 * log10(π_i) + offset
     The 400/log10(10) constant is the standard Elo definition: a 400-point
     advantage means 10× the BT strength, which corresponds to ~91% expected
     win rate (1/(1+10^-1)).
  4. Anchor scale: pick a reference player (default SH = 1000 Elo). Compute
     offset = 1000 - elo[SH], add to all players. Now the Elo numbers have
     absolute meaning relative to a known baseline.
  5. Bootstrap 95% CI: for B=200 iterations, resample each matchup's wins
     binomially from observed (P_win, N_total), refit BT, recompute Elos.
     Take 2.5/97.5 percentiles per player. This captures sampling noise from
     the finite N games per matchup.

Why Bradley-Terry over per-pair win rate:
  Pairwise win rates ignore the strength of opponents — beating a strong player
  60% is much more impressive than beating a weak one 60%. BT solves the entire
  league simultaneously and produces a self-consistent strength estimate where
  each player's number reflects who they beat, weighted by who those opponents
  also beat. Same approach used by Elo, Glicko, BayesElo, TrueSkill.
"""

import argparse
import asyncio
import gc
import glob
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.player.baselines import (
    RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer,
)

from battle_agent import BattleAgent
from policy_smartbots import SmartDamagePlayer, TacticalPlayer, StrategicPlayer
from policy_trainbots import (
    AntiSetupBot, StrategicV2, SwitchAwareEscapeV2, GreedySEv2,
    HazardSensev2, SwitchAwareEscapev3,
)
from policy_rulebots import (
    GreedySEPlayer, HazardSensePlayer, SwitchAwareEscapePlayer, SetupThenSweepPlayer,
)
from teams_ou import random_pool_teambuilder


_pid = os.getpid() % 10000
_match_id = 0


# =============================
# Bot registry
# =============================

# Smart bot anchors (Session 28 — same set as eval_vs_bots in train_bc.py).
# Also serves as the floor anchor scale.
SMART_BOTS = {
    "SH":        SimpleHeuristicsPlayer,    # baseline strong heuristic
    "SmartDmg":  SmartDamagePlayer,         # damage-aware extension of SH
    "Tactical":  TacticalPlayer,            # tactical extension of SH
    "Strategic": StrategicPlayer,           # strategic extension of SH
}

# Rule bots — basic heuristics, mid-tier strength
RULE_BOTS = {
    "GreedySE":          GreedySEPlayer,
    "HazardSense":       HazardSensePlayer,
    "SwitchAwareEscape": SwitchAwareEscapePlayer,
    "SetupThenSweep":    SetupThenSweepPlayer,
}

# Floor / random bots — extends Elo scale downward to give a wider anchor range
FLOOR_BOTS = {
    "Random":      RandomPlayer,
    "MaxBasePower": MaxBasePowerPlayer,
}

# Training-only bots (S68, 2026-06-09) — designed for training pool, NOT eval.
# Included in ALL_BOTS so they can be measured/ranked in Elo round-robins,
# but should NOT appear in the production eval pipeline (smart_avg, MM eval).
# See policy_trainbots.py docstrings + project_s68_run7_decision_pattern_findings_2026_06_09.md.
TRAIN_BOTS = {
    "AntiSetupBot":      AntiSetupBot,
    "StrategicV2":       StrategicV2,
    "SwitchAwareEscapeV2": SwitchAwareEscapeV2,
    "GreedySEv2":        GreedySEv2,
    "HazardSensev2":     HazardSensev2,
    "SwitchAwareEscapev3": SwitchAwareEscapev3,
}

# All 10 + 3 bots — full anchor set. Per Session 23 round-robin (50 games × 45 matchups):
#   Tactical 81.8% > Strategic 81.3% > SmartDmg 78.0% > SH 67.8% > SetupThenSweep 44.7%
#   > GreedySE 38.4% > HazardSense 36.7% > MaxBP 33.3% > SwitchAwareEscape 32.7% > Random 5.3%
# This gives a ~75% win-rate spread across the bot anchors → ~400 Elo of dynamic range,
# which is wide enough to interpolate snapshot Elos meaningfully.
# Training-only bots (AntiSetupBot, StrategicV2, SwitchAwareEscapeV2) added S68
# 2026-06-09 to enable round-robin ranking; their training-only status is enforced
# at training pipeline level (which bots get challenge-served), not here.
ALL_BOTS = {**FLOOR_BOTS, **RULE_BOTS, **SMART_BOTS, **TRAIN_BOTS}


# =============================
# Checkpoint cache + PlayerPool (Session 33 permanent fix)
# =============================
#
# The original eval_elo_ladder created fresh BattleAgent instances per
# matchup, paying ~30s of disk load + CUDA init for every single matchup. With
# 100+ matchups per shard this dominated wall time and caused progressive CUDA
# allocator fragmentation that eventually thrashed the GPU.
#
# The permanent fix has three parts:
#   1. _CKPT_CACHE — module-level dict mapping checkpoint paths to pre-loaded
#      state_dicts (CPU). Each checkpoint is loaded from disk EXACTLY ONCE per
#      shard process. Subsequent player creations use the cached dict.
#   2. PlayerPool — persistent pool of live BattleAgent/bot instances.
#      Bots are kept forever (zero GPU cost). Snapshots are managed via LRU
#      cache with configurable size (default 5 = ~3 GB VRAM at concurrency 15).
#      Players are reused across matchups; reset_battles() is called between.
#   3. Incremental JSONL save + resume — every completed match is appended to
#      a per-shard JSONL file as it finishes. On restart, existing JSONL is
#      replayed and already-completed matchups are skipped.
#
# Combined effect: per-matchup time drops from ~80s (best case) / ~250s (after
# fragmentation) to ~30s (steady state). For a 30-player tournament, wall time
# drops from ~14h projected (current dying run) to ~50min.

_CKPT_CACHE: Dict[str, dict] = {}


def load_ckpt_cached(path: str) -> dict:
    """Load a v8 checkpoint to CPU and cache it for repeated use.

    Returns a dict with at minimum 'model_state_dict' and 'model_config'.
    Strips optimizer state, snapshot pool, etc. to keep cache lean.
    """
    if path not in _CKPT_CACHE:
        full = torch.load(path, map_location='cpu', weights_only=False)
        # Keep only what BattleAgent.__init__ needs
        _CKPT_CACHE[path] = {
            'model_state_dict': full['model_state_dict'],
            'model_config': full.get('model_config', {}),
        }
        del full
        gc.collect()
    return _CKPT_CACHE[path]


class PlayerPool:
    """Persistent pool of live players (bots + LRU snapshot cache).

    Bots are instantiated once and never evicted (CPU only).
    Snapshot players are kept in an LRU cache; on cache miss, the oldest
    snapshot player is torn down and a new one is created using the cached
    state_dict (no disk read).
    """

    def __init__(self, max_snapshots: int, server_cfg, device: str,
                 concurrency: int, fp16: bool = False, battle_format: str = "gen9ou",
                 teambuilder_factory=None):
        self.max_snapshots = max_snapshots
        self.server_cfg = server_cfg
        self.device = device
        self.concurrency = concurrency
        self.fp16 = fp16
        self.battle_format = battle_format
        # teambuilder_factory: zero-arg callable returning a Teambuilder instance
        # per player. Default is the legacy 70-team random pool. For Metamon
        # competitive eval consistency, pass MetamonCompetitiveTeambuilder().
        if teambuilder_factory is None:
            teambuilder_factory = random_pool_teambuilder
        self.teambuilder_factory = teambuilder_factory
        self._bot_pool: Dict[str, Any] = {}
        self._snapshot_pool: "OrderedDict[str, Any]" = OrderedDict()
        self._account_counter = 0
        self._pid = os.getpid() % 10000

    def _next_account_name(self) -> str:
        self._account_counter += 1
        # Showdown account names are limited; keep short and alphanumeric
        return f"E{self._pid}p{self._account_counter}"

    def _make_bot(self, spec: 'PlayerSpec'):
        return spec.bot_cls(
            battle_format=self.battle_format,
            max_concurrent_battles=self.concurrency,
            server_configuration=self.server_cfg,
            team=self.teambuilder_factory(),
            account_configuration=AccountConfiguration(self._next_account_name(), None),
        )

    def _make_snapshot(self, spec: 'PlayerSpec'):
        # Get the cached checkpoint (load from disk only on first request)
        ckpt = load_ckpt_cached(spec.ckpt)
        # Arch dispatch: legacy MLP -> BattleAgent, new transformer -> BattleAgentTransformer.
        # Imports are lazy so older sessions that haven't built the new arch still load.
        from battle_agent_transformer import is_transformer_checkpoint, BattleAgentTransformer
        AgentClass = BattleAgentTransformer if is_transformer_checkpoint(ckpt) else BattleAgent
        return AgentClass(
            checkpoint_path=spec.ckpt,
            _cached_ckpt=ckpt,
            device=self.device,
            battle_format=self.battle_format,
            max_concurrent_battles=self.concurrency,
            server_configuration=self.server_cfg,
            team=self.teambuilder_factory(),
            account_configuration=AccountConfiguration(self._next_account_name(), None),
        )

    def get(self, spec: 'PlayerSpec'):
        """Get a live player for this spec, creating or evicting as needed."""
        if spec.kind == "bot":
            if spec.name not in self._bot_pool:
                self._bot_pool[spec.name] = self._make_bot(spec)
            return self._bot_pool[spec.name]

        # Snapshot path: LRU cache
        if spec.name in self._snapshot_pool:
            self._snapshot_pool.move_to_end(spec.name)  # mark as MRU
            return self._snapshot_pool[spec.name]

        # Cache miss — evict LRU if at capacity
        while len(self._snapshot_pool) >= self.max_snapshots:
            evict_name, evict_player = self._snapshot_pool.popitem(last=False)
            self._teardown_player(evict_player)

        # Create the new player
        player = self._make_snapshot(spec)
        self._snapshot_pool[spec.name] = player
        return player

    def reset_for_matchup(self, *players):
        """Reset per-matchup state on a set of players. Call before each matchup."""
        for p in players:
            try:
                p.reset_battles()
            except Exception:
                pass
            # BattleAgent also keeps per-battle temporal state — clear it
            if hasattr(p, '_history'):
                p._history.clear()
            if hasattr(p, '_turn_counts'):
                p._turn_counts.clear()

    def _teardown_player(self, player):
        try:
            player.reset_battles()
        except Exception:
            pass
        # Cancel websocket listener if present (per Session 32 hygiene)
        try:
            ps = getattr(player, "ps_client", None) or getattr(player, "_ps_client", None)
            if ps and hasattr(ps, "_listening_coroutine"):
                ps._listening_coroutine.cancel()
        except Exception:
            pass
        del player

    def cleanup(self):
        """Tear down all live players. Call at end of tournament."""
        for p in list(self._bot_pool.values()):
            self._teardown_player(p)
        self._bot_pool.clear()
        for p in list(self._snapshot_pool.values()):
            self._teardown_player(p)
        self._snapshot_pool.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def stats(self) -> str:
        return (f"bots:{len(self._bot_pool)} snapshots_in_pool:{len(self._snapshot_pool)}"
                f"/{self.max_snapshots} ckpt_cache:{len(_CKPT_CACHE)}")


# =============================
# Incremental JSONL save + resume
# =============================

def jsonl_path_for(out_json: Optional[str]) -> Optional[Path]:
    """Return the JSONL path for incremental saves alongside an --out-json target."""
    if not out_json:
        return None
    return Path(out_json).with_suffix(".jsonl")


def load_existing_matches(jsonl_path: Optional[Path]) -> Tuple[List[dict], Set[Tuple[str, str]]]:
    """Read previously-completed matches from JSONL for resume."""
    if jsonl_path is None or not jsonl_path.exists():
        return [], set()
    matches: List[dict] = []
    done: Set[Tuple[str, str]] = set()
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
                matches.append(m)
                done.add(tuple(sorted([m['p1'], m['p2']])))
            except json.JSONDecodeError:
                continue
    return matches, done


def append_match_to_jsonl(jsonl_path: Optional[Path], match: dict):
    if jsonl_path is None:
        return
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, 'a') as f:
        f.write(json.dumps(match) + "\n")


# =============================
# Server config
# =============================

def resolve_server(ws_url: str) -> ServerConfiguration:
    ws = ws_url.rstrip("/")
    http = ws.replace("wss://", "https://").replace("ws://", "http://")
    if ws.endswith("/showdown/websocket"):
        http = http[: http.rfind("/showdown/websocket")] + "/action.php?"
    return ServerConfiguration(ws, http)


# =============================
# Player factory
# =============================

class PlayerSpec:
    """Lightweight description of a tournament participant."""
    def __init__(self, kind: str, name: str, ckpt: Optional[str] = None,
                 bot_cls: Optional[type] = None):
        assert kind in ("snapshot", "bot")
        self.kind = kind
        self.name = name
        self.ckpt = ckpt
        self.bot_cls = bot_cls

    def __repr__(self):
        return f"PlayerSpec({self.kind}:{self.name})"


def make_player(spec: PlayerSpec, account_name: str, server_cfg, device: str,
                concurrency: int, battle_format: str = "gen9ou"):
    """Instantiate a poke-env Player for one match."""
    common = dict(
        battle_format=battle_format,
        max_concurrent_battles=concurrency,
        server_configuration=server_cfg,
        team=random_pool_teambuilder(),
        account_configuration=AccountConfiguration(account_name, None),
    )
    if spec.kind == "snapshot":
        return BattleAgent(
            checkpoint_path=spec.ckpt,
            device=device,
            **common,
        )
    elif spec.kind == "bot":
        return spec.bot_cls(**common)
    else:
        raise ValueError(spec.kind)


# =============================
# Match runner
# =============================

async def _battle_pair(p1, p2, n_games, timeout):
    # poke-env 0.10.0 quirk: must use n_battles= keyword. Positional binds wrong
    # and produces "'int' object has no attribute 'username'" deep inside.
    await asyncio.wait_for(
        p1.battle_against(p2, n_battles=n_games),
        timeout=timeout,
    )


def run_match(spec_a: PlayerSpec, spec_b: PlayerSpec, n_games: int,
              server: str, device: str, concurrency: int = 10) -> dict:
    """Run n_games between two players. Returns result dict."""
    global _match_id
    _match_id += 1
    mid = _match_id

    server_cfg = resolve_server(server)

    # Short, alphanumeric account names — Showdown rejects long/funky names
    name_a = f"E{_pid}m{mid}a"
    name_b = f"E{_pid}m{mid}b"

    p1 = make_player(spec_a, name_a, server_cfg, device, concurrency)
    p2 = make_player(spec_b, name_b, server_cfg, device, concurrency)

    t0 = time.time()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_battle_pair(
            p1, p2, n_games, timeout=max(600, n_games * 12)
        ))
    finally:
        loop.close()
    elapsed = time.time() - t0

    w1, w2 = p1.n_won_battles, p2.n_won_battles
    ties = p1.n_tied_battles
    total = w1 + w2 + ties

    # Cleanup
    for p in (p1, p2):
        try:
            p.reset_battles()
        except Exception:
            pass
    del p1, p2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "p1": spec_a.name, "p2": spec_b.name,
        "p1_kind": spec_a.kind, "p2_kind": spec_b.kind,
        "p1_wins": w1, "p2_wins": w2, "ties": ties, "total": total,
        "p1_wr": w1 / max(1, total),
        "elapsed": round(elapsed, 1),
    }


# =============================
# Bradley-Terry MLE for Elo
# =============================

def fit_bradley_terry(names: List[str], wins: Dict[Tuple[str, str], int],
                      games: Dict[Tuple[str, str], int],
                      max_iter: int = 1000, tol: float = 1e-7) -> Dict[str, float]:
    """Fit BT model via Hunter (2004) MM algorithm.

    Returns dict {name: pi} where pi is the BT strength parameter.
    Convert to Elo via: elo = 400 * log10(pi) + offset.

    wins[(i, j)] = number of times i beat j (over the n_games matches between i and j)
    games[(i, j)] = total games between i and j (= games[(j, i)])
    """
    n = len(names)
    name_to_idx = {n: i for i, n in enumerate(names)}
    pi = np.ones(n)

    # Total wins per player W_i
    W = np.zeros(n)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                continue
            W[i] += wins.get((ni, nj), 0)

    # n_ij matrix (symmetric)
    N = np.zeros((n, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i < j:
                g = games.get((ni, nj), 0) + games.get((nj, ni), 0)
                N[i, j] = g
                N[j, i] = g

    for it in range(max_iter):
        new_pi = np.zeros(n)
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i == j or N[i, j] == 0:
                    continue
                denom += N[i, j] / (pi[i] + pi[j])
            if denom > 0:
                new_pi[i] = W[i] / denom
            else:
                new_pi[i] = pi[i]
        # Normalize (BT is identifiable up to scale)
        if new_pi.sum() > 0:
            new_pi = new_pi / new_pi.sum() * n

        if np.allclose(new_pi, pi, atol=tol, rtol=tol):
            pi = new_pi
            break
        pi = new_pi

    return {n: float(pi[i]) for i, n in enumerate(names)}


def pi_to_elo(pi: Dict[str, float], anchor_name: str = None,
              anchor_elo: float = 1000.0) -> Dict[str, float]:
    """Convert BT strengths to Elo. Anchor one player at fixed Elo for scale."""
    elos = {n: 400.0 * math.log10(max(p, 1e-12)) for n, p in pi.items()}
    if anchor_name is not None and anchor_name in elos:
        offset = anchor_elo - elos[anchor_name]
        elos = {n: e + offset for n, e in elos.items()}
    return elos


def bootstrap_elo_cis(matches: List[dict], names: List[str],
                      n_bootstrap: int = 200, anchor_name: str = None,
                      anchor_elo: float = 1000.0) -> Dict[str, Tuple[float, float, float]]:
    """Bootstrap 95% CIs on Elo by resampling individual game outcomes per matchup.

    Returns dict {name: (median, lo95, hi95)}.
    """
    rng = np.random.default_rng(42)
    samples: Dict[str, List[float]] = {n: [] for n in names}

    for b in range(n_bootstrap):
        # Resample wins per matchup binomially
        wins_b: Dict[Tuple[str, str], int] = {}
        games_b: Dict[Tuple[str, str], int] = {}
        for m in matches:
            total = m["total"]
            if total == 0:
                continue
            p_win = m["p1_wins"] / total
            new_p1_wins = int(rng.binomial(total, p_win))
            new_p2_wins = total - new_p1_wins  # treat ties as losses for BT (rare)
            wins_b[(m["p1"], m["p2"])] = new_p1_wins
            wins_b[(m["p2"], m["p1"])] = new_p2_wins
            games_b[(m["p1"], m["p2"])] = total
            games_b[(m["p2"], m["p1"])] = total

        try:
            pi_b = fit_bradley_terry(names, wins_b, games_b, max_iter=300)
            elos_b = pi_to_elo(pi_b, anchor_name=anchor_name, anchor_elo=anchor_elo)
            for n in names:
                samples[n].append(elos_b[n])
        except Exception as e:
            print(f"  [WARN] bootstrap iter {b} failed: {e}", flush=True)

    cis = {}
    for n in names:
        arr = np.array(samples[n]) if samples[n] else np.array([0.0])
        cis[n] = (
            float(np.median(arr)),
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)),
        )
    return cis


# =============================
# Snapshot sampling
# =============================

def sample_snapshots(glob_pattern: str, sample_n: int = None,
                     sample_stride: int = None) -> List[str]:
    """Sample N snapshots from a glob pattern, evenly spaced if sample_n given."""
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        return []
    if sample_stride:
        return paths[::sample_stride]
    if sample_n and sample_n < len(paths):
        idxs = np.linspace(0, len(paths) - 1, sample_n).astype(int)
        return [paths[i] for i in idxs]
    return paths


# =============================
# Main
# =============================

def combine_shards(shard_paths: List[str], anchor_bot: str, anchor_elo: float,
                   n_bootstrap: int, out_json: Optional[str]):
    """Merge multiple shard JSONs into a single Elo ladder.

    Each shard JSON has the format produced by main(): {config, players, matches, ...}.
    We collect all matches, dedupe by (p1, p2) pair, refit BT, recompute CIs, print.
    """
    all_matches: List[dict] = []
    all_players: List[dict] = []
    seen_pairs: set = set()
    seen_players: set = set()

    for path in shard_paths:
        with open(path) as f:
            data = json.load(f)
        for sp in data["players"]:
            if sp["name"] not in seen_players:
                all_players.append(sp)
                seen_players.add(sp["name"])
        for m in data["matches"]:
            key = tuple(sorted([m["p1"], m["p2"]]))
            if key in seen_pairs:
                # Duplicate pair across shards — sum the games
                # (this can happen if shards overlap; usually they shouldn't)
                continue
            seen_pairs.add(key)
            all_matches.append(m)

    print(f"Combined {len(shard_paths)} shards: {len(all_players)} players, "
          f"{len(all_matches)} unique matches", flush=True)

    names = [sp["name"] for sp in all_players]
    wins: Dict[Tuple[str, str], int] = {}
    games: Dict[Tuple[str, str], int] = {}
    for m in all_matches:
        wins[(m["p1"], m["p2"])] = m["p1_wins"]
        wins[(m["p2"], m["p1"])] = m["p2_wins"]
        games[(m["p1"], m["p2"])] = m["total"]
        games[(m["p2"], m["p1"])] = m["total"]

    print(f"Fitting Bradley-Terry MLE on {len(all_matches)} matches...", flush=True)
    pi_hat = fit_bradley_terry(names, wins, games)
    elos = pi_to_elo(pi_hat, anchor_name=anchor_bot, anchor_elo=anchor_elo)

    print(f"Bootstrapping CIs ({n_bootstrap} iters)...", flush=True)
    cis = bootstrap_elo_cis(all_matches, names, n_bootstrap=n_bootstrap,
                            anchor_name=anchor_bot, anchor_elo=anchor_elo)

    # Print
    print(f"\n{'='*72}")
    print(f"  ELO LADDER (combined from {len(shard_paths)} shards, "
          f"anchored to {anchor_bot}={anchor_elo:.0f})")
    print(f"{'='*72}\n")
    rows = sorted(elos.items(), key=lambda x: -x[1])
    print(f"{'Rank':>5}  {'Player':<28}  {'Elo':>6}  {'95% CI':>14}  {'Type':<5}")
    print("-" * 72)
    for rank, (name, elo) in enumerate(rows, 1):
        ci = cis.get(name, (elo, elo, elo))
        spec = next(sp for sp in all_players if sp["name"] == name)
        kind = "snap" if spec["kind"] == "snapshot" else "bot"
        print(f"  #{rank:>2}  {name:<28}  {elo:>6.0f}  [{ci[1]:>4.0f}, {ci[2]:>4.0f}]  {kind}")
    print()

    if out_json:
        out = {
            "config": {"combined_from": shard_paths, "anchor_bot": anchor_bot,
                       "anchor_elo": anchor_elo, "n_bootstrap": n_bootstrap,
                       "n_matchups": len(all_matches), "n_players": len(all_players)},
            "players": all_players,
            "elos": {n: round(e, 1) for n, e in elos.items()},
            "cis": {n: {"median": round(c[0], 1), "lo95": round(c[1], 1), "hi95": round(c[2], 1)}
                    for n, c in cis.items()},
            "matches": all_matches,
        }
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved combined results to {out_json}")


def _add_to_existing(args):
    """Add new player(s) to an existing Elo ladder JSON.

    Loads all previous matches from the existing JSON, adds the new snapshot(s)
    specified by --snapshots/--names, runs ONLY the matchups involving new players
    vs all existing players, then refits BT on the combined match set.

    Usage:
        python eval_elo_ladder.py --add-to data/eval/elo_session35_exp1.json \
            --snapshots path/to/new_snapshot.pt --names sp2099 \
            --n-games 100 --concurrency 100 --device cuda \
            --server ws://127.0.0.1:9000/showdown/websocket \
            --out-json data/eval/elo_updated.json
    """
    import time
    from datetime import datetime as _dt
    from itertools import combinations

    # Load existing ladder
    with open(args.add_to) as f:
        existing = json.load(f)

    existing_players = existing["players"]
    existing_matches = existing["matches"]
    existing_names = {p["name"] for p in existing_players}

    print(f"Loaded existing ladder: {len(existing_players)} players, "
          f"{len(existing_matches)} matches from {args.add_to}", flush=True)

    # Build new player specs from --snapshots / --names
    if not args.snapshots:
        print("ERROR: --add-to requires --snapshots (the new player(s) to add)", file=sys.stderr)
        sys.exit(1)

    new_names = args.names if args.names else [Path(p).stem for p in args.snapshots]
    assert len(new_names) == len(args.snapshots), "--names must match --snapshots count"

    new_players = []
    for path, name in zip(args.snapshots, new_names):
        if name in existing_names:
            print(f"  SKIP: {name} already in ladder", flush=True)
            continue
        new_players.append(PlayerSpec(name=name, kind="snapshot", ckpt=path))
        print(f"  NEW: {name} ({path})", flush=True)

    if not new_players:
        print("No new players to add. Exiting.", flush=True)
        return

    # Build combined player list
    all_player_specs = []
    for p in existing_players:
        if p["kind"] == "snapshot":
            all_player_specs.append(PlayerSpec(name=p["name"], kind="snapshot", ckpt=p["ckpt"]))
        else:
            # Look up bot class from registry — needed for instantiation
            bot_cls = ALL_BOTS.get(p["name"])
            if bot_cls is None:
                print(f"  WARN: bot {p['name']} not in ALL_BOTS registry, skipping", flush=True)
                continue
            all_player_specs.append(PlayerSpec(name=p["name"], kind="bot", ckpt=None, bot_cls=bot_cls))
    all_player_specs.extend(new_players)

    name_to_idx = {sp.name: i for i, sp in enumerate(all_player_specs)}

    # Only run matchups involving new players vs all existing players
    new_pairs = []
    for new_p in new_players:
        ni = name_to_idx[new_p.name]
        for existing_p in all_player_specs:
            if existing_p.name == new_p.name:
                continue
            ei = name_to_idx[existing_p.name]
            i, j = min(ni, ei), max(ni, ei)
            new_pairs.append((i, j))

    # Also add pairs between new players if multiple
    if len(new_players) > 1:
        for a, b in combinations([name_to_idx[p.name] for p in new_players], 2):
            new_pairs.append((min(a, b), max(a, b)))

    full_pair_count = len(new_pairs)
    print(f"\nNew matchups to run: {full_pair_count} "
          f"({len(new_players)} new player(s) vs {len(all_player_specs) - len(new_players)} existing)",
          flush=True)

    # Optional sharding for parallel runs across multiple servers
    if args.shard:
        try:
            shard_idx, shard_total = (int(x) for x in args.shard.split("/"))
            assert 0 <= shard_idx < shard_total
        except Exception:
            print(f"--shard must be 'i/N' with 0 <= i < N, got {args.shard!r}", file=sys.stderr)
            sys.exit(1)
        # Deterministic round-robin partition: pair k goes to shard (k % N)
        new_pairs = [(i, j) for k, (i, j) in enumerate(new_pairs) if k % shard_total == shard_idx]
        print(f"Shard {shard_idx}/{shard_total}: {len(new_pairs)} of {full_pair_count} matchups",
              flush=True)

    # Set up player pool and run matches
    server_cfg = resolve_server(args.server)
    # Build the teambuilder factory based on --team-set.
    if getattr(args, "team_set", "metamon-competitive") == "metamon-competitive":
        from eval_metamon_competitive import MetamonCompetitiveTeambuilder
        # Single shared instance — both sides + bots all sample from the same 16 teams.
        _shared_tb = MetamonCompetitiveTeambuilder()
        teambuilder_factory = lambda: _shared_tb
    else:
        teambuilder_factory = random_pool_teambuilder

    pool = PlayerPool(
        max_snapshots=args.max_snapshots_in_pool,
        server_cfg=server_cfg,
        device=args.device,
        concurrency=args.concurrency,
        battle_format=args.format,
        teambuilder_factory=teambuilder_factory,
    )

    # Start with existing matches
    matches = list(existing_matches)
    wins = {}
    game_counts = {}
    for m in existing_matches:
        wins[(m['p1'], m['p2'])] = m['p1_wins']
        wins[(m['p2'], m['p1'])] = m['p2_wins']
        game_counts[(m['p1'], m['p2'])] = m['total']
        game_counts[(m['p2'], m['p1'])] = m['total']

    # JSONL for incremental saves
    jsonl_path = jsonl_path_for(args.out_json) if args.out_json else None
    already_done = set()
    if jsonl_path and jsonl_path.exists():
        prev, already_done = load_existing_matches(jsonl_path)
        if prev:
            print(f"  RESUME: {len(prev)} matches from {jsonl_path}", flush=True)
            for m in prev:
                key = tuple(sorted([m['p1'], m['p2']]))
                if key not in {tuple(sorted([em['p1'], em['p2']])) for em in existing_matches}:
                    matches.append(m)
                    wins[(m['p1'], m['p2'])] = m['p1_wins']
                    wins[(m['p2'], m['p1'])] = m['p2_wins']
                    game_counts[(m['p1'], m['p2'])] = m['total']
                    game_counts[(m['p2'], m['p1'])] = m['total']

    def _log(msg):
        print(f"[{_dt.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    t_start = time.time()
    pairs_to_run = [(i, j) for (i, j) in new_pairs
                    if tuple(sorted([all_player_specs[i].name, all_player_specs[j].name]))
                    not in already_done]
    _log(f"=== Running {len(pairs_to_run)} new matchups ({len(new_pairs) - len(pairs_to_run)} already done) ===")

    completed = 0
    try:
        for pi, (i, j) in enumerate(pairs_to_run):
            a, b = all_player_specs[i], all_player_specs[j]
            elapsed = time.time() - t_start
            eta = ((elapsed / max(completed, 1)) * (len(pairs_to_run) - pi)) if completed > 0 else 0
            _log(f"[{pi+1}/{len(pairs_to_run)}] {a.name} vs {b.name}  "
                 f"(elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m)")
            try:
                p_a = pool.get(a)
                p_b = pool.get(b)
                pool.reset_for_matchup(p_a, p_b)

                t_match = time.time()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(_battle_pair(
                        p_a, p_b, args.n_games,
                        timeout=max(600, args.n_games * 12)
                    ))
                finally:
                    loop.close()
                match_elapsed = time.time() - t_match

                w1, w2 = p_a.n_won_battles, p_b.n_won_battles
                ties = p_a.n_tied_battles
                total = w1 + w2 + ties
                r = {
                    "p1": a.name, "p2": b.name,
                    "p1_kind": a.kind, "p2_kind": b.kind,
                    "p1_wins": w1, "p2_wins": w2, "ties": ties, "total": total,
                    "p1_wr": w1 / max(1, total),
                    "elapsed": round(match_elapsed, 1),
                }
            except Exception as e:
                _log(f"  [ERROR] match failed: {e}")
                continue

            matches.append(r)
            wins[(a.name, b.name)] = r["p1_wins"]
            wins[(b.name, a.name)] = r["p2_wins"]
            game_counts[(a.name, b.name)] = r["total"]
            game_counts[(b.name, a.name)] = r["total"]
            completed += 1
            # S67-EXT readability: label matchup explicitly so interleaved
            # shard logs are unambiguous. Format: "[A vs B] A_wins-B_wins (ties) | A WR=X% | elapsed"
            _log(f"  -> [{a.name} vs {b.name}] {r['p1_wins']}-{r['p2_wins']} (ties:{r['ties']}) "
                 f"| {a.name} WR={r['p1_wr']:.0%} | {r['elapsed']:.0f}s")
            append_match_to_jsonl(jsonl_path, r)
    finally:
        _log(f"=== Done: {completed} new matchups in {(time.time()-t_start)/60:.1f}m ===")
        pool.cleanup()

    # Refit BT on ALL matches (existing + new)
    all_names = [sp.name for sp in all_player_specs]
    print(f"\nFitting Bradley-Terry on {len(matches)} total matches "
          f"({len(existing_matches)} existing + {completed} new)...", flush=True)
    pi_hat = fit_bradley_terry(all_names, wins, game_counts)
    elos_result = pi_to_elo(pi_hat, anchor_name=args.anchor_bot, anchor_elo=args.anchor_elo)

    print(f"Bootstrapping CIs ({args.n_bootstrap} iters)...", flush=True)
    cis_result = bootstrap_elo_cis(matches, all_names, n_bootstrap=args.n_bootstrap,
                                   anchor_name=args.anchor_bot, anchor_elo=args.anchor_elo)

    # Print results
    print(f"\n{'='*72}")
    print(f"  ELO LADDER  (updated with {len(new_players)} new player(s), "
          f"anchored to {args.anchor_bot}={args.anchor_elo:.0f})")
    print(f"{'='*72}\n")

    rows = sorted(elos_result.items(), key=lambda x: -x[1])
    print(f"{'Rank':>5}  {'Player':<28}  {'Elo':>6}  {'95% CI':>14}  {'Type':<5}  {'New'}")
    print("-" * 80)
    new_name_set = {p.name for p in new_players}
    for rank, (name, elo) in enumerate(rows, 1):
        ci = cis_result.get(name, (elo, elo, elo))
        spec = next(sp for sp in all_player_specs if sp.name == name)
        kind = "snap" if spec.kind == "snapshot" else "bot"
        marker = " <-- NEW" if name in new_name_set else ""
        print(f"  #{rank:>2}  {name:<28}  {elo:>6.0f}  [{ci[1]:>4.0f}, {ci[2]:>4.0f}]  {kind}  {marker}")
    print()

    # Save
    if args.out_json:
        out = {
            "config": {
                "added_to": args.add_to,
                "n_games": args.n_games,
                "anchor_bot": args.anchor_bot,
                "anchor_elo": args.anchor_elo,
                "n_bootstrap": args.n_bootstrap,
                "n_matchups": len(matches),
                "n_players": len(all_player_specs),
                "new_players": [p.name for p in new_players],
            },
            "players": [
                {"name": sp.name, "kind": sp.kind,
                 "ckpt": sp.ckpt if sp.kind == "snapshot" else None}
                for sp in all_player_specs
            ],
            "elos": {n: round(e, 1) for n, e in elos_result.items()},
            "cis": {n: {"median": round(c[0], 1), "lo95": round(c[1], 1), "hi95": round(c[2], 1)}
                    for n, c in cis_result.items()},
            "matches": matches,
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved updated results to {args.out_json}")

    # Persist Elo results to registry (fire-and-forget)
    try:
        from registry import log_elo
        import re as _re
        bot_set = {sp.name for sp in all_player_specs if sp.kind == "bot"}
        ladder_name = Path(args.out_json).stem if args.out_json else "add_to"
        for name, elo_val in elos_result.items():
            if name in bot_set:
                continue
            ci = cis_result.get(name, (elo_val, elo_val, elo_val))
            m = _re.search(r'(\d{3,4})', name)
            it = int(m.group(1)) if m else (0 if name == "BC_base" else -1)
            ckpt = next((sp.ckpt for sp in all_player_specs if sp.name == name), None)
            log_elo(it, name, elo_val, ci[1], ci[2], ladder_name,
                    n_games=args.n_games, ckpt=ckpt)
    except Exception as e:
        print(f"  [WARN] Registry Elo logging failed: {e}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Snapshot Elo ladder + bot anchors")
    p.add_argument("--snapshots", nargs="+", default=None,
                   help="Explicit snapshot paths")
    p.add_argument("--names", nargs="+", default=None,
                   help="Display names matching --snapshots (else use stem)")
    p.add_argument("--snapshot-glob", default=None,
                   help="Glob pattern for snapshot sampling")
    p.add_argument("--sample-n", type=int, default=None,
                   help="Sample N evenly-spaced snapshots from --snapshot-glob")
    p.add_argument("--sample-stride", type=int, default=None,
                   help="Sample every Nth snapshot from --snapshot-glob")
    p.add_argument("--bots", choices=["none", "smart", "all"], default="all",
                   help="Bot anchors: 'smart' = 4 strong (SH/SmartDmg/Tactical/Strategic), "
                        "'all' = 10 (adds Random, MaxBP, GreedySE, HazardSense, "
                        "SwitchAwareEscape, SetupThenSweep — wider Elo range), "
                        "'none' = snapshots only (internal-only Elo, NOT recommended)")
    p.add_argument("--include-bots", action="store_true",
                   help="DEPRECATED: same as --bots smart. Kept for backwards compat.")
    p.add_argument("--anchor-bot", default="SH",
                   help="Bot to anchor at fixed Elo (default SH @ 1000). Other bots and "
                        "snapshots get Elos relative to this anchor.")
    p.add_argument("--anchor-elo", type=float, default=1000.0)
    p.add_argument("--n-games", type=int, default=50,
                   help="Games per matchup (50 = ~+/-50 Elo CI on tight pairs)")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--team-set", choices=["pool", "metamon-competitive"], default="metamon-competitive",
                   help="Team source. 'pool' = legacy 70-team random pool (Session 23-era convention); "
                        "'metamon-competitive' = 16 curated Smogon teams matching our smart_avg/H2H "
                        "eval methodology (default, recommended).")
    p.add_argument("--format", default="gen9ou", help="Battle format (gen9ou, gen8ou, etc.)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--server", default="ws://127.0.0.1:9000/showdown/websocket")
    p.add_argument("--n-bootstrap", type=int, default=200)
    p.add_argument("--out-json", default=None)
    p.add_argument("--match-list", default=None,
                   help="Optional JSON file listing (i,j) pairs to run; useful for "
                        "manually parallelizing across multiple server ports")
    p.add_argument("--shard", default=None,
                   help="Auto-partition matchups across N parallel shards. Format: i/N "
                        "(e.g. '0/3', '1/3', '2/3'). Each shard runs a deterministic "
                        "1/N slice of the full matchup list. Use --combine afterward "
                        "to merge shard JSONs into a final ladder. Mutually exclusive "
                        "with --match-list.")
    p.add_argument("--max-snapshots-in-pool", type=int, default=5,
                   help="LRU cache size for live snapshot players (default 5 = ~3 GB "
                        "VRAM at concurrency 15). Larger pool = fewer evictions = "
                        "faster, but more GPU memory pressure.")
    p.add_argument("--no-resume", action="store_true",
                   help="Don't read existing JSONL on startup (re-run from scratch). "
                        "Default: resume from JSONL if present.")
    p.add_argument("--combine", nargs="+", default=None,
                   help="Combine multiple shard JSONs into a final Elo ladder. "
                        "Pass shard JSON paths as arguments. Skips tournament; "
                        "just refits BT on the merged win matrix.")
    p.add_argument("--add-to", default=None,
                   help="Add new player(s) to an existing Elo JSON. Loads all previous "
                        "matches, adds --snapshots as new players, runs ONLY matchups "
                        "involving the new player(s) vs all existing players, refits BT. "
                        "Much faster than a full round-robin for incremental measurement.")
    args = p.parse_args()

    # ---- Combine mode: merge shard JSONs and exit ----
    if args.combine:
        combine_shards(args.combine, args.anchor_bot, args.anchor_elo,
                       args.n_bootstrap, args.out_json)
        return

    # ---- Add-to mode: add new player(s) to existing ladder ----
    if args.add_to:
        _add_to_existing(args)
        return

    # ---- Build player list ----
    players: List[PlayerSpec] = []

    # Snapshots — additive: both --snapshots and --snapshot-glob can be passed.
    # Order: explicit --snapshots first (with --names if provided), then sampled
    # from glob (auto-named from stem), dedup'd by absolute path.
    snap_paths: List[str] = []
    snap_names_built: List[str] = []
    seen_paths: set = set()
    stem_counts: Dict[str, int] = {}

    def _auto_name(path: str) -> str:
        stem = Path(path).stem
        if stem in stem_counts:
            stem_counts[stem] += 1
            return f"{stem}_{stem_counts[stem]}"
        stem_counts[stem] = 0
        return stem

    # Step 1: explicit --snapshots (with optional --names)
    if args.snapshots:
        if args.names:
            assert len(args.names) == len(args.snapshots), (
                f"--names count ({len(args.names)}) must match --snapshots count "
                f"({len(args.snapshots)}). --names ONLY labels --snapshots; the "
                f"--snapshot-glob results auto-name from stem."
            )
            explicit_names = list(args.names)
        else:
            explicit_names = [Path(p).stem for p in args.snapshots]
        for p, name in zip(args.snapshots, explicit_names):
            ap = str(Path(p).resolve())
            if ap not in seen_paths:
                snap_paths.append(p)
                snap_names_built.append(name)
                seen_paths.add(ap)
                # Track this stem so glob-sampled duplicates get suffixed
                stem_counts[Path(p).stem] = 0

    # Step 2: sampled from glob (auto-named, dedup'd against explicit)
    if args.snapshot_glob:
        sampled = sample_snapshots(args.snapshot_glob,
                                   sample_n=args.sample_n,
                                   sample_stride=args.sample_stride)
        for p in sampled:
            ap = str(Path(p).resolve())
            if ap not in seen_paths:
                snap_paths.append(p)
                snap_names_built.append(_auto_name(p))
                seen_paths.add(ap)

    for path, name in zip(snap_paths, snap_names_built):
        players.append(PlayerSpec("snapshot", name, ckpt=path))

    # Bots — backwards compat: --include-bots == --bots smart
    bot_set_name = args.bots
    if args.include_bots and args.bots == "all":
        # User passed legacy flag; if they didn't override --bots, treat as 'smart'
        bot_set_name = "smart"

    if bot_set_name == "smart":
        bot_set = SMART_BOTS
    elif bot_set_name == "all":
        bot_set = ALL_BOTS
    else:
        bot_set = {}

    for name, cls in bot_set.items():
        players.append(PlayerSpec("bot", name, bot_cls=cls))

    if len(players) < 2:
        print("Need at least 2 players. Pass --snapshots and/or --include-bots.", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Snapshot Elo Ladder ===")
    print(f"Players: {len(players)} ({sum(1 for p in players if p.kind == 'snapshot')} snapshots, "
          f"{sum(1 for p in players if p.kind == 'bot')} bots)")
    for sp in players:
        kind = "snap" if sp.kind == "snapshot" else "bot "
        print(f"  [{kind}] {sp.name}")
    print(f"Games per matchup: {args.n_games}")
    print(f"Server: {args.server}")
    print(f"Anchor: {args.anchor_bot} @ {args.anchor_elo} Elo")
    n_matchups = len(players) * (len(players) - 1) // 2
    print(f"Total matchups: {n_matchups}")
    print(f"Estimated time: ~{n_matchups * args.n_games * 0.7 / 60:.0f} min "
          f"(at ~0.7s/game on 1 server)\n", flush=True)

    # ---- Round robin ----
    pairs = list(combinations(range(len(players)), 2))
    full_pair_count = len(pairs)

    # Auto-shard partitioning (--shard i/N)
    if args.shard:
        try:
            shard_idx, shard_total = (int(x) for x in args.shard.split("/"))
            assert 0 <= shard_idx < shard_total
        except Exception:
            print(f"--shard must be 'i/N' with 0 <= i < N, got {args.shard!r}", file=sys.stderr)
            sys.exit(1)
        # Deterministic round-robin partition: pair k goes to shard (k % N)
        # This balances load AND ensures each shard sees a representative slice
        # of player matchups (rather than e.g. shard 0 only doing snapshot-vs-snapshot).
        pairs = [(i, j) for k, (i, j) in enumerate(pairs) if k % shard_total == shard_idx]
        print(f"Shard {shard_idx}/{shard_total}: {len(pairs)} of {full_pair_count} matchups\n",
              flush=True)

    # Optional manual partition for parallel runs (mutually exclusive with --shard)
    elif args.match_list:
        with open(args.match_list) as f:
            partition = json.load(f)
        pairs = [(int(i), int(j)) for i, j in partition]
        print(f"Loaded {len(pairs)} matchups from {args.match_list}\n", flush=True)

    matches: List[dict] = []
    wins: Dict[Tuple[str, str], int] = {}
    games: Dict[Tuple[str, str], int] = {}

    from datetime import datetime as _dt
    def _log(msg):
        print(f"[{_dt.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    # ---- Resume from JSONL if it exists ----
    jsonl_path = jsonl_path_for(args.out_json)
    already_done: Set[Tuple[str, str]] = set()
    if jsonl_path is not None and not args.no_resume:
        existing, already_done = load_existing_matches(jsonl_path)
        if existing:
            _log(f"RESUME: loaded {len(existing)} previously-completed matches "
                 f"from {jsonl_path}")
            matches.extend(existing)
            for m in existing:
                wins[(m['p1'], m['p2'])] = m['p1_wins']
                wins[(m['p2'], m['p1'])] = m['p2_wins']
                games[(m['p1'], m['p2'])] = m['total']
                games[(m['p2'], m['p1'])] = m['total']

    # ---- Build PlayerPool (persistent player cache) ----
    server_cfg = resolve_server(args.server)
    # Build the teambuilder factory based on --team-set.
    if getattr(args, "team_set", "metamon-competitive") == "metamon-competitive":
        from eval_metamon_competitive import MetamonCompetitiveTeambuilder
        # Single shared instance — both sides + bots all sample from the same 16 teams.
        _shared_tb = MetamonCompetitiveTeambuilder()
        teambuilder_factory = lambda: _shared_tb
    else:
        teambuilder_factory = random_pool_teambuilder

    pool = PlayerPool(
        max_snapshots=args.max_snapshots_in_pool,
        server_cfg=server_cfg,
        device=args.device,
        concurrency=args.concurrency,
        battle_format=args.format,
        teambuilder_factory=teambuilder_factory,
    )
    _log(f"PlayerPool created (max_snapshots={args.max_snapshots_in_pool}, "
         f"concurrency={args.concurrency})")

    t_start = time.time()
    pairs_to_run = [(i, j) for (i, j) in pairs
                    if tuple(sorted([players[i].name, players[j].name])) not in already_done]
    _log(f"=== Starting tournament: {len(pairs_to_run)} matchups to run "
         f"({len(pairs) - len(pairs_to_run)} already done) ===")

    completed_this_session = 0
    try:
        for pi, (i, j) in enumerate(pairs_to_run):
            a, b = players[i], players[j]
            elapsed_so_far = time.time() - t_start
            eta_s = ((elapsed_so_far / max(completed_this_session, 1))
                     * (len(pairs_to_run) - pi)) if completed_this_session > 0 else 0
            _log(f"[{pi+1}/{len(pairs_to_run)}] {a.name} vs {b.name}  "
                 f"(elapsed {elapsed_so_far/60:.1f}m, ETA {eta_s/60:.1f}m, "
                 f"pool: {pool.stats()})")
            try:
                # Get or create players from pool (persistent across matchups)
                p_a = pool.get(a)
                p_b = pool.get(b)

                # Reset per-matchup state on both players
                pool.reset_for_matchup(p_a, p_b)

                # Run the games
                t_match = time.time()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(_battle_pair(
                        p_a, p_b, args.n_games,
                        timeout=max(600, args.n_games * 12)
                    ))
                finally:
                    loop.close()
                elapsed = time.time() - t_match

                w1, w2 = p_a.n_won_battles, p_b.n_won_battles
                ties = p_a.n_tied_battles
                total = w1 + w2 + ties
                r = {
                    "p1": a.name, "p2": b.name,
                    "p1_kind": a.kind, "p2_kind": b.kind,
                    "p1_wins": w1, "p2_wins": w2, "ties": ties, "total": total,
                    "p1_wr": w1 / max(1, total),
                    "elapsed": round(elapsed, 1),
                }
            except Exception as e:
                _log(f"  [ERROR] match failed: {e}")
                continue

            matches.append(r)
            wins[(a.name, b.name)] = r["p1_wins"]
            wins[(b.name, a.name)] = r["p2_wins"]
            games[(a.name, b.name)] = r["total"]
            games[(b.name, a.name)] = r["total"]
            completed_this_session += 1
            # S67-EXT readability: label matchup explicitly so interleaved
            # shard logs are unambiguous. Format: "[A vs B] A_wins-B_wins (ties) | A WR=X% | elapsed"
            _log(f"  -> [{a.name} vs {b.name}] {r['p1_wins']}-{r['p2_wins']} (ties:{r['ties']}) "
                 f"| {a.name} WR={r['p1_wr']:.0%} | {r['elapsed']:.0f}s")

            # Incremental save — append to JSONL immediately
            append_match_to_jsonl(jsonl_path, r)
    finally:
        _log(f"=== Tournament done: {len(matches)} total matches "
             f"({completed_this_session} this session), "
             f"total wall {(time.time()-t_start)/60:.1f}m ===")
        pool.cleanup()

    # ---- Fit Bradley-Terry ----
    names = [sp.name for sp in players]
    is_shard = args.shard is not None

    if is_shard:
        # Skip BT fit — sparse matrix from one shard would produce confused Elos.
        # Just save the matches JSON; user runs --combine afterward.
        print(f"\n[shard mode] Skipping BT fit. Save and run --combine on all shards.",
              flush=True)
        pi_hat = {n: 1.0 for n in names}
        elos = {n: 0.0 for n in names}
        cis = {n: (0.0, 0.0, 0.0) for n in names}
    else:
        print(f"\nFitting Bradley-Terry MLE on {len(matches)} matches...", flush=True)
        pi_hat = fit_bradley_terry(names, wins, games)
        elos = pi_to_elo(pi_hat, anchor_name=args.anchor_bot, anchor_elo=args.anchor_elo)

        print(f"Bootstrapping CIs ({args.n_bootstrap} iters)...", flush=True)
        cis = bootstrap_elo_cis(matches, names, n_bootstrap=args.n_bootstrap,
                                anchor_name=args.anchor_bot, anchor_elo=args.anchor_elo)

    # ---- Print results (skip in shard mode — combine produces the real ladder) ----
    if not is_shard:
        print(f"\n{'='*72}")
        print(f"  ELO LADDER  ({args.n_games} games/matchup, "
              f"anchored to {args.anchor_bot}={args.anchor_elo:.0f})")
        print(f"{'='*72}\n")

        rows = sorted(elos.items(), key=lambda x: -x[1])
        print(f"{'Rank':>5}  {'Player':<28}  {'Elo':>6}  {'95% CI':>14}  {'Type':<5}")
        print("-" * 72)
        for rank, (name, elo) in enumerate(rows, 1):
            ci = cis.get(name, (elo, elo, elo))
            spec = next(sp for sp in players if sp.name == name)
            kind = "snap" if spec.kind == "snapshot" else "bot"
            print(f"  #{rank:>2}  {name:<28}  {elo:>6.0f}  [{ci[1]:>4.0f}, {ci[2]:>4.0f}]  {kind}")
        print()
    else:
        print(f"\n[shard mode] Saved {len(matches)} match results. "
              f"Run --combine on all shard JSONs to produce the final ladder.\n", flush=True)

    # ---- Save JSON ----
    if args.out_json:
        out = {
            "config": {
                "n_games": args.n_games,
                "anchor_bot": args.anchor_bot,
                "anchor_elo": args.anchor_elo,
                "n_bootstrap": args.n_bootstrap,
                "n_matchups": len(matches),
                "n_players": len(players),
            },
            "players": [
                {"name": sp.name, "kind": sp.kind,
                 "ckpt": sp.ckpt if sp.kind == "snapshot" else None}
                for sp in players
            ],
            "elos": {n: round(e, 1) for n, e in elos.items()},
            "cis": {n: {"median": round(c[0], 1), "lo95": round(c[1], 1), "hi95": round(c[2], 1)}
                    for n, c in cis.items()},
            "matches": matches,
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved results to {args.out_json}")

    # Persist Elo results to registry (fire-and-forget)
    if not is_shard:
        try:
            from registry import log_elo
            import re as _re
            bot_names_set = {sp.name for sp in players if sp.kind == "bot"}
            ladder_name = Path(args.out_json).stem if args.out_json else "unnamed"
            for name, elo_val in elos.items():
                if name in bot_names_set:
                    continue
                ci = cis.get(name, (elo_val, elo_val, elo_val))
                m = _re.search(r'(\d{3,4})', name)
                it = int(m.group(1)) if m else (0 if name == "BC_base" else -1)
                ckpt = next((sp.ckpt for sp in players if sp.name == name), None)
                log_elo(it, name, elo_val, ci[1], ci[2], ladder_name,
                        n_games=args.n_games, ckpt=ckpt)
        except Exception as e:
            print(f"  [WARN] Registry Elo logging failed: {e}", flush=True)


if __name__ == "__main__":
    main()
