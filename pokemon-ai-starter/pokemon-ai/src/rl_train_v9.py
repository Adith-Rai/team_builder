#!/usr/bin/env python
# rl_train_v9.py — Pure self-play PPO with batched GPU inference
#
# Key changes from v8:
#   - InferenceBatcher: async batched GPU forward for concurrent battles
#   - V9RLPlayer: async choose_move that submits to batcher
#   - Pure self-play: no bots, opponent from uniform snapshot pool
#   - Temperature randomization [1.0, 2.25] for opponent diversity
#   - Imports PPO update, GAE, trajectory, checkpoint I/O from rl_train_v8
#
# Usage:
#   python -u rl_train_v9.py \
#     --init-from data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt \
#     --device cuda --servers 9000,9001 --fp16 \
#     --games-per-iter 200 --max-concurrent 20 --n-iters 500

from __future__ import annotations
import argparse
import asyncio
import gc
import json
import os
import random
import sys
import threading
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from poke_env.player import Player
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from features import make_features
from model import PokeTransformer, PokeTransformerConfig
from battle_agent import BattleAgent
from rewards import RewardShaper
from teams_ou import random_pool_teambuilder
from team_generator import ProceduralTeambuilder, procedural_teambuilder

# Reuse v8 internals — no copies
from ppo import (
    Trajectory, compute_gae, build_ppo_episodes, ppo_update,
    load_checkpoint, save_checkpoint, _cancel_listener,
)

# For model arg parsing
from model import add_model_args

_pid_tag = os.getpid() % 10000
_collect_round = 0


# =============================
# InferenceBatcher
# =============================

class InferenceBatcher:
    """Async batch collector for GPU inference.

    Concurrent battles submit (features, history) and await results.
    When min_batch requests accumulate (or timeout fires), ONE batched
    model.forward() processes them all.
    """

    def __init__(self, model: PokeTransformer, device: torch.device,
                 fp16: bool = False, min_batch: int = 8, timeout_ms: int = 20):
        self.model = model
        self.device = device
        self.fp16 = fp16 and device.type == "cuda"
        self.min_batch = min_batch
        self.timeout_s = timeout_ms / 1000.0
        self._pending: List[Tuple[dict, Optional[torch.Tensor], int, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._d_model = model.cfg.d_model
        # Profiling counters (reset per collection)
        self._prof_batch_sizes = []
        self._prof_gpu_times = []
        self._prof_timeout_fires = 0
        self._prof_normal_fires = 0

    async def submit(self, batch_dict: dict, history: Optional[torch.Tensor],
                     history_len: int) -> dict:
        """Submit inference request. Returns dict with action_logits, value, summary."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending.append((batch_dict, history, history_len, future))

        if len(self._pending) >= self.min_batch:
            # Yield to event loop FIRST so other pending websocket messages
            # can be processed and more battles can submit requests.
            # This dramatically increases batch sizes with many concurrent battles.
            await asyncio.sleep(0)
            self._prof_normal_fires += 1
            await self._fire_batch()
        else:
            # Schedule timeout to fire partial batch
            loop.call_later(self.timeout_s, self._timeout_fire)

        return await future

    def _timeout_fire(self):
        if self._pending:
            self._prof_timeout_fires += 1
            asyncio.ensure_future(self._fire_batch())

    async def _fire_batch(self):
        async with self._lock:
            if not self._pending:
                return
            requests = self._pending[:]
            self._pending.clear()

        try:
            t0 = time.time()
            # Run GPU forward directly on event loop. Blocks for ~14ms but
            # run_in_executor is slower due to GIL contention with CUDA ops.
            results = self._gpu_forward(requests)
            gpu_ms = (time.time() - t0) * 1000
            self._prof_batch_sizes.append(len(requests))
            self._prof_gpu_times.append(gpu_ms)
            for i, (_, _, _, future) in enumerate(requests):
                if not future.done():
                    future.set_result(results[i])
        except Exception as e:
            for _, _, _, future in requests:
                if not future.done():
                    future.set_exception(e)

    def prof_summary(self) -> str:
        """Return profiling summary string and reset counters."""
        if not self._prof_batch_sizes:
            return "no batches"
        import numpy as np
        sizes = np.array(self._prof_batch_sizes)
        times = np.array(self._prof_gpu_times)
        s = (f"batches={len(sizes)}, size={sizes.mean():.1f}avg/{sizes.min()}-{sizes.max()} "
             f"gpu={times.mean():.1f}ms avg/{times.sum()/1000:.1f}s total "
             f"fires=normal:{self._prof_normal_fires}/timeout:{self._prof_timeout_fires}")
        self._prof_batch_sizes.clear()
        self._prof_gpu_times.clear()
        self._prof_timeout_fires = 0
        self._prof_normal_fires = 0
        return s

    def _gpu_forward(self, requests) -> List[dict]:
        """Fully batched forward: spatial + temporal + heads all batched.

        All N requests are processed in parallel. Variable-length temporal
        histories are left-aligned and padded, with seq_lens for masking.
        """
        N = len(requests)
        model = self.model
        D = self._d_model

        # Stack batch dicts: (1, ...) -> (N, ...)
        mega = self._stack_batches([r[0] for r in requests])

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.fp16):
            # Phase 1: Batched spatial
            spatial_out, summaries = model.forward_spatial(mega)  # (N, 16, D), (N, D)

            if spatial_out.isnan().any():
                print(f"  [NaN-DIAG] spatial_out has NaN", flush=True)
            if summaries.isnan().any():
                print(f"  [NaN-DIAG] summaries has NaN", flush=True)

            # Phase 2: Batched action encoding
            action_ctx = model.action_encoder(
                mega["active_move_ids"], mega["active_move_banks"],
                mega["active_move_cont"], mega["switch_ids"], mega["switch_cont"],
            )  # (N, 9, D)

            if action_ctx.isnan().any():
                print(f"  [NaN-DIAG] action_ctx has NaN", flush=True)

            # Phase 3: Batched temporal
            # Build left-aligned padded history tensor + current summary
            # Each item: [hist_0, hist_1, ..., hist_len-1, current_summary, PAD, ...]
            seq_lens = []
            for i in range(N):
                history = requests[i][1]
                h_len = history.shape[1] if history is not None and history.shape[1] > 0 else 0
                seq_lens.append(h_len + 1)  # +1 for current summary

            max_T = min(max(seq_lens), model.temporal.temporal_context)
            seq_lens_t = torch.tensor(seq_lens, device=self.device, dtype=torch.long).clamp(max=max_T)

            # Pre-allocate padded tensor (zeros for padding)
            all_summaries = torch.zeros(N, max_T, D, device=self.device, dtype=summaries.dtype)

            for i in range(N):
                history = requests[i][1]
                summary_i = summaries[i]  # (D,)

                if history is not None and history.shape[1] > 0:
                    h = history.to(self.device).squeeze(0)  # (T_i, D)
                    # Truncate from left if history + 1 > max_T
                    if h.shape[0] + 1 > max_T:
                        h = h[-(max_T - 1):]
                    h_len = h.shape[0]
                    all_summaries[i, :h_len] = h
                    all_summaries[i, h_len] = summary_i
                else:
                    all_summaries[i, 0] = summary_i

            # FP32 for temporal (matches per-item behavior — summary_attn is fp32)
            temporal_ctx = model.temporal(
                all_summaries.float(), seq_lens_t
            ).to(summaries.dtype)  # (N, D)

            if temporal_ctx.isnan().any():
                nan_mask = temporal_ctx.isnan().any(dim=-1)
                for i in range(N):
                    if nan_mask[i]:
                        print(f"  [NaN-DIAG] temporal has NaN (item {i}, history_len={seq_lens[i]})", flush=True)

            # Phase 4: Batched policy head
            actor_out = spatial_out[:, 0, :]  # (N, D)
            at = torch.cat([actor_out, temporal_ctx], dim=-1)  # (N, 2D)
            at_exp = at.unsqueeze(1).expand(-1, 9, -1)  # (N, 9, 2D)
            pi_input = torch.cat([at_exp, action_ctx], dim=-1)  # (N, 9, 3D)
            logits = model.policy_head(pi_input).squeeze(-1)  # (N, 9)

            if logits.isnan().any():
                print(f"  [NaN-DIAG] policy_head has NaN", flush=True)

            # Apply legal masks
            if "legal_mask" in mega:
                logits = logits.float().masked_fill(mega["legal_mask"] < 0.5, -100.0)

            # Phase 5: Batched value head
            critic_out = spatial_out[:, 1, :]  # (N, D)
            vi = torch.cat([critic_out, temporal_ctx], dim=-1)  # (N, 2D)
            v_logits = model.value_head(vi)  # (N, 51)
            v_probs = F.softmax(v_logits, dim=-1)
            values = (v_probs * model.v_support).sum(-1)  # (N,)

            # Unpack results
            results = []
            for i in range(N):
                results.append({
                    "action_logits": logits[i],        # (9,)
                    "value": values[i],                 # scalar
                    "summary": summaries[i].float(),    # (D,) always fp32
                })

        return results

    def _stack_batches(self, batch_list: List[dict]) -> dict:
        """Stack N batch dicts (each B=1) into mega-batch (B=N)."""
        mega = {}
        ref = batch_list[0]
        for key in ref:
            if isinstance(ref[key], torch.Tensor):
                mega[key] = torch.cat([b[key] for b in batch_list], dim=0)
            elif isinstance(ref[key], dict):
                mega[key] = {
                    k: torch.cat([b[key][k] for b in batch_list], dim=0)
                    for k in ref[key]
                }
            else:
                mega[key] = ref[key]  # non-tensor, just pass through
        return mega

    # Note: as of Session 32, temporal + heads are now fully batched alongside spatial.
    # Variable-length histories are left-aligned and padded with seq_lens masking.
    # Previous per-item approach was a workaround for a padding direction bug.


# =============================
# V9RLPlayer (async choose_move)
# =============================

class V9RLPlayer(Player):
    """PPO player with async batched inference."""

    def __init__(self, batcher: InferenceBatcher, device: torch.device,
                 reward_shaper_cfg: Optional[dict] = None,
                 temperature: float = 1.0, turn_cap: int = 300, **kwargs):
        super().__init__(**kwargs)
        self.batcher = batcher
        self.device = device
        self._rs_cfg = reward_shaper_cfg or {"ko_coef": 0.05, "hp_coef": 0.02, "clip_abs": 2.0}
        self.temperature = temperature
        self.turn_cap = turn_cap
        self._history: Dict[str, torch.Tensor] = {}
        self._trajectories: Dict[str, Trajectory] = {}
        self._reward_shapers: Dict[str, RewardShaper] = {}
        self.completed_trajectories: List[Trajectory] = []
        self._tainted: set = set()  # battle tags with NaN — discard trajectory

    def _get_shaper(self, btag):
        if btag not in self._reward_shapers:
            self._reward_shapers[btag] = RewardShaper(**self._rs_cfg)
        return self._reward_shapers[btag]

    def _get_traj(self, btag):
        if btag not in self._trajectories:
            self._trajectories[btag] = Trajectory()
        return self._trajectories[btag]

    def _build_turn_batch(self, feat: dict) -> dict:
        """Convert features_v8 output to model batch dict on self.device."""
        dev = self.device

        def _pi(p):
            i = p["ids"]
            return [i["species"], i["item"], i["ability"]]
        def _pb(p):
            b = p["banks"]
            return [b["hp_pct"], b["level"], b["weight"], b["height"],
                    b["stat_hp"], b["stat_atk"], b["stat_def"],
                    b["stat_spa"], b["stat_spd"], b["stat_spe"]]
        def _pmi(p):
            i = p["ids"]
            return [i["move0"], i["move1"], i["move2"], i["move3"]]
        def _pmc(p):
            c = p["continuous"]
            b = len(c) - 92
            return [c[b+i*23:b+(i+1)*23] for i in range(4)]

        our, opp = feat["our_pokemon"], feat["opp_pokemon"]
        int_arrays = {
            "our_pokemon_ids": np.array([[_pi(p) for p in our]], dtype=np.int64),
            "our_pokemon_banks": np.array([[_pb(p) for p in our]], dtype=np.int64),
            "our_pokemon_move_ids": np.array([[_pmi(p) for p in our]], dtype=np.int64),
            "opp_pokemon_ids": np.array([[_pi(p) for p in opp]], dtype=np.int64),
            "opp_pokemon_banks": np.array([[_pb(p) for p in opp]], dtype=np.int64),
            "opp_pokemon_move_ids": np.array([[_pmi(p) for p in opp]], dtype=np.int64),
        }
        float_arrays = {
            "our_pokemon_cont": np.array([[p["continuous"] for p in our]], dtype=np.float32),
            "our_pokemon_move_cont": np.array([[_pmc(p) for p in our]], dtype=np.float32),
            "opp_pokemon_cont": np.array([[p["continuous"] for p in opp]], dtype=np.float32),
            "opp_pokemon_move_cont": np.array([[_pmc(p) for p in opp]], dtype=np.float32),
            "field_cont": np.array([feat["field"]["continuous"]], dtype=np.float32),
            "transition_cont": np.array([feat["transition"]["continuous"]], dtype=np.float32),
            "legal_mask": feat["legal_mask"].reshape(1, 9).astype(np.float32),
        }

        mids, mbp, mac, mpp, mpr, mco = [], [], [], [], [], []
        for m in feat["active_moves"]:
            if m is None:
                mids.append(0); mbp.append(0); mac.append(0); mpp.append(0); mpr.append(6)
                from features import MOVE_SLOT_CONT_DIM
                mco.append([0.0]*MOVE_SLOT_CONT_DIM)
            else:
                mids.append(m["move_id"]); mbp.append(m["bp_int"]); mac.append(m["acc_int"])
                mpp.append(m["pp_int"]); mpr.append(m["priority_int"]); mco.append(m["continuous"])
        int_arrays["active_move_ids"] = np.array([mids], dtype=np.int64)
        float_arrays["active_move_cont"] = np.array([mco], dtype=np.float32)

        sids, sco = [], []
        for s in feat["switch_slots"]:
            if s is None:
                from features import SWITCH_SLOT_CONT_DIM
                sids.append(0); sco.append([0.0]*SWITCH_SLOT_CONT_DIM)
            else:
                sids.append(s["species_id"]); sco.append(s["continuous"])
        int_arrays["switch_ids"] = np.array([sids], dtype=np.int64)
        float_arrays["switch_cont"] = np.array([sco], dtype=np.float32)

        batch = {}
        for k, arr in int_arrays.items():
            batch[k] = torch.from_numpy(arr).to(dev, non_blocking=True)
        for k, arr in float_arrays.items():
            batch[k] = torch.from_numpy(arr).to(dev, non_blocking=True)

        fb = feat["field"]["banks"]
        batch["field_banks"] = {k: torch.tensor([fb[k]], dtype=torch.long, device=dev) for k in fb}
        ti = feat["transition"]["ids"]
        batch["transition_ids"] = {k: torch.tensor([ti[k]], dtype=torch.long, device=dev) for k in ti}
        batch["active_move_banks"] = {
            "bp": torch.tensor([mbp], dtype=torch.long, device=dev),
            "acc": torch.tensor([mac], dtype=torch.long, device=dev),
            "pp": torch.tensor([mpp], dtype=torch.long, device=dev),
            "prio": torch.tensor([mpr], dtype=torch.long, device=dev),
        }
        return batch

    def _to_cpu(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.cpu()
            elif isinstance(v, dict):
                out[k] = {kk: vv.cpu() if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
            else:
                out[k] = v
        return out

    async def choose_move(self, battle):
        """Async choose_move — submits to batcher, awaits result."""
        btag = battle.battle_tag
        traj = self._get_traj(btag)
        shaper = self._get_shaper(btag)

        # Turn cap
        if len(traj) >= self.turn_cap:
            print(f"  [TURN CAP] {btag} hit {self.turn_cap} turns, forfeiting", flush=True)
            try:
                self.forfeit_battle(battle)
            except Exception:
                pass
            return self.choose_random_move(battle)

        # Feature extraction (CPU, ~1ms)
        feat = make_features(battle)

        # Dense reward for previous step (with immune detection from transition)
        if len(traj.rewards) > 0:
            # our_eff[0] = immune flag, at index 9 in transition continuous
            # (6 action_kind + 3 moved_first = 9 offset)
            our_move_immune = feat["transition"]["continuous"][9] > 0.5
            traj.rewards[-1] += shaper.step(battle, our_move_immune=our_move_immune)
        batch = self._build_turn_batch(feat)
        history = self._history.get(btag)
        h_len = history.shape[1] if history is not None else 0

        # Submit to batcher and await (yields to event loop!)
        try:
            result = await self.batcher.submit(batch, history, h_len)
        except Exception as e:
            print(f"  [ERROR] Batcher failed for {btag}: {e}", flush=True)
            self._tainted.add(btag)
            return self.choose_random_move(battle)

        # Update temporal history (always float32)
        summary = result["summary"].unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        if history is None:
            self._history[btag] = summary
        else:
            self._history[btag] = torch.cat([history, summary], dim=1)
            if self._history[btag].shape[1] > 200:
                self._history[btag] = self._history[btag][:, -200:]

        # Sample action with temperature
        logits = result["action_logits"]  # (9,)
        if self.temperature != 1.0:
            scaled = logits / self.temperature
        else:
            scaled = logits
        # Guard against NaN/inf from FP16 overflow — taint entire battle
        if torch.isnan(scaled).any() or torch.isinf(scaled).any():
            self._tainted.add(btag)
            return self.choose_random_move(battle)
        probs = F.softmax(scaled, dim=-1)
        if torch.isnan(probs).any() or (probs < 0).any():
            self._tainted.add(btag)
            return self.choose_random_move(battle)
        action_idx = torch.multinomial(probs, 1).item()

        # Store UNSCALED log_prob
        log_prob = F.log_softmax(logits, dim=-1)[action_idx].item()
        value = result["value"].item()

        # Store trajectory (CPU)
        traj.feat_batches.append(self._to_cpu(batch))
        traj.actions.append(action_idx)
        traj.log_probs.append(log_prob)
        traj.values.append(value)
        traj.rewards.append(0.0)
        traj.dones.append(False)
        traj.action_masks.append(feat["legal_mask"].copy())

        return self._action_to_order(battle, action_idx)

    def _action_to_order(self, battle, idx):
        if idx < 4:
            moves = list(battle.available_moves or [])
            if idx < len(moves):
                return self.create_order(moves[idx])
        else:
            sw = list(battle.available_switches or [])
            si = idx - 4
            if si < len(sw):
                return self.create_order(sw[si])
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])
        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        btag = battle.battle_tag
        traj = self._trajectories.get(btag)
        if traj and len(traj) > 0:
            if btag in self._tainted:
                print(f"  [TAINTED] Discarding {btag} ({len(traj)} turns) — NaN detected", flush=True)
                self._tainted.discard(btag)
            else:
                shaper = self._get_shaper(btag)
                # Terminal step — no feature extraction available, immune irrelevant
                # (terminal ±1.0 dominates)
                traj.rewards[-1] += shaper.step(battle, our_move_immune=False)
                if battle.won:
                    traj.rewards[-1] += 1.0
                elif battle.lost:
                    traj.rewards[-1] -= 1.0
                traj.dones[-1] = True
                self.completed_trajectories.append(traj)

        self._tainted.discard(btag)
        self._trajectories.pop(btag, None)
        self._history.pop(btag, None)
        self._reward_shapers.pop(btag, None)
        super()._battle_finished_callback(battle)

    def reset_battles(self):
        self._history.clear()
        self._trajectories.clear()
        self._reward_shapers.clear()
        self._tainted.clear()
        self.completed_trajectories.clear()
        super().reset_battles()


# =============================
# SelfPlayOpponent
# =============================

class SelfPlayOpponent(BattleAgent):
    """Opponent with temperature randomization for self-play diversity."""

    def __init__(self, checkpoint_path: str, device: str = "cuda",
                 temp_range: Tuple[float, float] = (1.0, 2.25), **kwargs):
        temp = random.uniform(*temp_range)
        # fp16=False for opponent to avoid any FP16 numerical issues
        super().__init__(checkpoint_path=checkpoint_path, device=device,
                         temperature=temp, fp16=False, **kwargs)
        self._temp_range = temp_range

    def _battle_finished_callback(self, battle):
        """Re-randomize temperature for next game."""
        super()._battle_finished_callback(battle)
        self.temperature = random.uniform(*self._temp_range)


# =============================
# Collection
# =============================

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


async def collect_v9(
    model: PokeTransformer, device: torch.device,
    server_pool: List[ServerConfiguration],
    n_games: int = 200, max_concurrent: int = 20,
    snapshot_pool: List[str] = None, fp16: bool = True,
    reward_shaper_cfg: Optional[dict] = None,
    temp_range: Tuple[float, float] = (1.0, 2.25),
    opponent_device: str = "cuda",
    latest_snapshot: Optional[str] = None,
    teambuilder=None,
):
    """Pure self-play collection with batched inference.
    Plays against MULTIPLE opponents per iteration (uniform from pool, max 15).
    Latest snapshot gets temp randomization; historical play at full strength."""
    global _collect_round
    _collect_round += 1
    rid = _collect_round

    if not snapshot_pool:
        raise ValueError("snapshot_pool must contain at least one checkpoint")

    # Select opponents: use ALL if pool is small, sample up to 15 if large
    max_opponents = 15
    if len(snapshot_pool) <= max_opponents:
        selected = list(snapshot_pool)
    else:
        selected = random.sample(snapshot_pool, max_opponents)
        # Always include the latest snapshot
        if latest_snapshot and latest_snapshot not in selected:
            selected[-1] = latest_snapshot

    # Distribute games across opponents (roughly equal)
    games_per_opp = max(1, n_games // len(selected))
    remainder = n_games - games_per_opp * len(selected)

    rs_cfg = reward_shaper_cfg or {"ko_coef": 0.05, "hp_coef": 0.02, "clip_abs": 2.0}
    all_trajs = []
    total_wins, total_losses, total_ties, total_steps = 0, 0, 0, 0
    opp_results = []
    t0 = time.time()

    # --- Parallel opponent collection ---
    # Group opponents into waves of len(server_pool), run each wave concurrently.
    # All RL players in a wave share ONE InferenceBatcher for GPU batching efficiency.
    # Each opponent pair gets FULL max_concurrent — total concurrent = max_concurrent × n_parallel.
    # This is safe because each pair is on a different server and they share a GPU batcher.
    n_servers = len(server_pool)
    conc_per_pair = max_concurrent

    async def _play_one_opponent(oi, opp_ckpt, n_battles, batcher, srv, batch_id):
        """Play n_battles against one opponent. Returns (trajs, wins, losses, ties, short_name)."""
        opp_name = Path(opp_ckpt).stem
        tb = teambuilder or random_pool_teambuilder()
        player = V9RLPlayer(
            batcher=batcher, device=device,
            reward_shaper_cfg=rs_cfg,
            temperature=1.0,
            turn_cap=300,
            battle_format="gen9ou",
            team=tb,
            max_concurrent_battles=conc_per_pair,
            account_configuration=AccountConfiguration(f"RL{_pid_tag}r{batch_id}", None),
            server_configuration=srv,
        )

        is_latest = (latest_snapshot is not None and opp_ckpt == latest_snapshot)
        if len(snapshot_pool) > 15 or not is_latest:
            opp_temp_range = (1.0, 1.0)
        else:
            opp_temp_range = temp_range

        opp_tb = teambuilder or random_pool_teambuilder()
        opponent = SelfPlayOpponent(
            checkpoint_path=opp_ckpt,
            device=opponent_device,
            temp_range=opp_temp_range,
            battle_format="gen9ou",
            team=opp_tb,
            max_concurrent_battles=conc_per_pair,
            account_configuration=AccountConfiguration(f"Op{_pid_tag}r{batch_id}", None),
            server_configuration=srv,
        )

        try:
            await asyncio.wait_for(
                player.battle_against(opponent, n_battles=n_battles),
                timeout=max(180, n_battles * 25),
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
        try:
            opponent.reset_battles()
        except EnvironmentError:
            pass
        _cancel_listener(player)
        _cancel_listener(opponent)
        del player, opponent

        return trajs, w, l, ties, f"{short}={w}/{w+l}"

    # Build opponent tasks: (oi, checkpoint, n_games, server)
    opp_tasks = []
    for oi, opp_ckpt in enumerate(selected):
        n = games_per_opp + (1 if oi < remainder else 0)
        if n <= 0:
            continue
        opp_tasks.append((oi, opp_ckpt, n))

    # Process in waves of n_servers (parallel within wave, sequential across waves)
    for wave_start in range(0, len(opp_tasks), n_servers):
        wave = opp_tasks[wave_start:wave_start + n_servers]

        # One shared batcher for the wave — all RL players submit to it
        batcher = InferenceBatcher(
            model, device, fp16=fp16,
            min_batch=min(8, conc_per_pair * len(wave)),
            timeout_ms=15,
        )

        coros = []
        for wi, (oi, opp_ckpt, n) in enumerate(wave):
            batch_id = rid * 100 + oi
            srv = server_pool[wi % n_servers]
            coros.append(_play_one_opponent(oi, opp_ckpt, n, batcher, srv, batch_id))

        wave_results = await asyncio.gather(*coros, return_exceptions=True)

        for result in wave_results:
            if isinstance(result, Exception):
                print(f"  [ERROR] Wave opponent failed: {result}", flush=True)
                continue
            trajs, w, l, ties, summary = result
            all_trajs.extend(trajs)
            total_wins += w
            total_losses += l
            total_ties += ties
            opp_results.append(summary)

        # Print batcher profiling for this wave
        prof = batcher.prof_summary()
        wave_idx = wave_start // n_servers
        print(f"  [PROF] wave {wave_idx}: {prof}", flush=True)
        del batcher

    elapsed = time.time() - t0
    total_steps = sum(len(t) for t in all_trajs)
    opp_summary = " ".join(opp_results)
    gc.collect()

    return all_trajs, total_wins, total_losses, total_ties, total_steps, opp_summary, elapsed


# =============================
# Pipelined Collection (background thread)
# =============================

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

    def start(self, model, device, server_pool, snapshot_pool, args_dict):
        """Start background collection with a deepcopy of the model."""
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
                  server_pool, snapshot_pool, args_dict),
            daemon=True,
        )
        self._thread.start()

    def _run(self, collect_model, device, fp16, opp_device, server_pool, snapshot_pool, a):
        try:
            loop = asyncio.new_event_loop()
            latest_sp = snapshot_pool[-1] if len(snapshot_pool) > 1 else None
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


# =============================
# Multiprocess Collection (Session 32)
# See docs/MULTIPROCESS_COLLECTION.md for design and invariants.
# =============================

import torch.multiprocessing as mp_mod
from multiprocessing import Queue as MPQueue

# Must use 'spawn' for CUDA in child processes (not 'fork')
try:
    mp_mod.set_start_method('spawn', force=False)
except RuntimeError:
    pass  # already set

# Message types for request queue
_MSG_INFER = 0   # inference request: (worker_id, btag, obs_dict_cpu)
_MSG_CLEAR = 1   # clear history: (worker_id, btag)
_MSG_TRAJ = 2    # completed trajectory: (worker_id, trajectory)
_MSG_DONE = 3    # worker finished: (worker_id,)


class InferenceServer:
    """Main-process GPU inference server for multiprocess collection.

    Reads requests from shared queue, batches them, runs GPU forward,
    returns results to per-worker result queues. Manages temporal history.
    """

    def __init__(self, model: PokeTransformer, device: torch.device,
                 request_queue: MPQueue, result_queues: Dict[int, MPQueue],
                 fp16: bool = False, batch_timeout_ms: float = 20,
                 min_batch: int = 4):
        self.model = model
        self.device = device
        self.fp16 = fp16 and device.type == "cuda"
        self.request_queue = request_queue
        self.result_queues = result_queues
        self.batch_timeout = batch_timeout_ms / 1000.0
        self.min_batch = min_batch
        self.D = model.cfg.d_model
        self.max_temporal = model.temporal.temporal_context

        # History store: (worker_id, btag) -> (1, T, D) tensor on device
        self.history: Dict[Tuple[int, str], torch.Tensor] = {}

        # Collected trajectories
        self.trajectories: List[Trajectory] = []

        # Profiling
        self._prof_batch_sizes = []
        self._prof_gpu_times = []
        self._prof_total_requests = 0

    def run_until_workers_done(self, n_workers: int):
        """Process requests until all workers signal DONE."""
        workers_done = set()
        self.model.eval()

        while len(workers_done) < n_workers:
            # Drain queue: collect inference requests greedily
            infer_requests = []  # [(worker_id, btag, obs_dict)]
            deadline = time.time() + self.batch_timeout

            while time.time() < deadline:
                try:
                    msg = self.request_queue.get(timeout=max(0.001, deadline - time.time()))
                except Exception:
                    break

                msg_type = msg[0]
                if msg_type == _MSG_INFER:
                    _, wid, btag, obs_dict = msg
                    infer_requests.append((wid, btag, obs_dict))
                    if len(infer_requests) >= self.min_batch * 2:
                        break  # batch is big enough, process now
                elif msg_type == _MSG_CLEAR:
                    _, wid, btag = msg
                    self.history.pop((wid, btag), None)
                elif msg_type == _MSG_TRAJ:
                    _, wid, traj = msg
                    self.trajectories.append(traj)
                elif msg_type == _MSG_DONE:
                    _, wid = msg
                    workers_done.add(wid)

            if infer_requests:
                self._process_batch(infer_requests)

        # Drain any remaining messages
        while not self.request_queue.empty():
            try:
                msg = self.request_queue.get_nowait()
                if msg[0] == _MSG_TRAJ:
                    self.trajectories.append(msg[2])
                elif msg[0] == _MSG_CLEAR:
                    self.history.pop((msg[1], msg[2]), None)
            except Exception:
                break

    def _process_batch(self, infer_requests):
        """Run batched GPU forward on N inference requests."""
        N = len(infer_requests)
        self._prof_total_requests += N
        D = self.D
        model = self.model
        device = self.device

        # Move obs dicts to GPU and stack into mega batch
        mega = self._stack_obs_to_device([r[2] for r in infer_requests])

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.fp16):
            t0 = time.time()

            # Phase 1: Batched spatial
            spatial_out, summaries = model.forward_spatial(mega)  # (N, 16, D), (N, D)

            # Phase 2: Batched action encoding
            action_ctx = model.action_encoder(
                mega["active_move_ids"], mega["active_move_banks"],
                mega["active_move_cont"], mega["switch_ids"], mega["switch_cont"],
            )  # (N, 9, D)

            # Phase 3: Batched temporal — build padded history tensor
            seq_lens = []
            for i, (wid, btag, _) in enumerate(infer_requests):
                key = (wid, btag)
                h = self.history.get(key)
                h_len = h.shape[1] if h is not None else 0
                seq_lens.append(h_len + 1)

            max_T = min(max(seq_lens), self.max_temporal)
            seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.long).clamp(max=max_T)
            all_summaries = torch.zeros(N, max_T, D, device=device, dtype=summaries.dtype)

            for i, (wid, btag, _) in enumerate(infer_requests):
                key = (wid, btag)
                h = self.history.get(key)
                summary_i = summaries[i]

                if h is not None and h.shape[1] > 0:
                    hh = h.squeeze(0)  # (T_i, D)
                    if hh.shape[0] + 1 > max_T:
                        hh = hh[-(max_T - 1):]
                    h_len = hh.shape[0]
                    all_summaries[i, :h_len] = hh
                    all_summaries[i, h_len] = summary_i
                else:
                    all_summaries[i, 0] = summary_i

            temporal_ctx = model.temporal(
                all_summaries.float(), seq_lens_t
            ).to(summaries.dtype)

            # Phase 4: Batched policy head
            actor_out = spatial_out[:, 0, :]
            at = torch.cat([actor_out, temporal_ctx], dim=-1)
            at_exp = at.unsqueeze(1).expand(-1, 9, -1)
            pi_input = torch.cat([at_exp, action_ctx], dim=-1)
            logits = model.policy_head(pi_input).squeeze(-1)  # (N, 9)

            if "legal_mask" in mega:
                logits = logits.float().masked_fill(mega["legal_mask"] < 0.5, -100.0)

            # Phase 5: Batched value head
            critic_out = spatial_out[:, 1, :]
            vi = torch.cat([critic_out, temporal_ctx], dim=-1)
            v_logits = model.value_head(vi)
            v_probs = F.softmax(v_logits, dim=-1)
            values = (v_probs * model.v_support).sum(-1)

            gpu_ms = (time.time() - t0) * 1000
            self._prof_batch_sizes.append(N)
            self._prof_gpu_times.append(gpu_ms)

        # Update histories and dispatch results
        for i, (wid, btag, _) in enumerate(infer_requests):
            key = (wid, btag)
            summary_f32 = summaries[i].float().unsqueeze(0).unsqueeze(0)  # (1, 1, D)
            h = self.history.get(key)
            if h is None:
                self.history[key] = summary_f32
            else:
                self.history[key] = torch.cat([h, summary_f32], dim=1)
                if self.history[key].shape[1] > self.max_temporal:
                    self.history[key] = self.history[key][:, -self.max_temporal:]

            # Send result to worker's queue
            result = {
                "action_logits": logits[i].cpu(),
                "value": values[i].cpu(),
            }
            self.result_queues[wid].put(result)

    def _stack_obs_to_device(self, obs_list: List[dict]) -> dict:
        """Stack N CPU obs dicts into one GPU mega-batch."""
        mega = {}
        ref = obs_list[0]
        dev = self.device
        for key in ref:
            if isinstance(ref[key], torch.Tensor):
                mega[key] = torch.cat([b[key] for b in obs_list], dim=0).to(dev, non_blocking=True)
            elif isinstance(ref[key], dict):
                mega[key] = {
                    k: torch.cat([b[key][k] for b in obs_list], dim=0).to(dev, non_blocking=True)
                    for k in ref[key]
                }
            else:
                mega[key] = ref[key]
        return mega

    def prof_summary(self) -> str:
        if not self._prof_batch_sizes:
            return "no batches"
        import numpy as _np
        sizes = _np.array(self._prof_batch_sizes)
        times = _np.array(self._prof_gpu_times)
        s = (f"batches={len(sizes)}, size={sizes.mean():.1f}avg/{sizes.min()}-{sizes.max()} "
             f"gpu={times.mean():.1f}ms avg/{times.sum()/1000:.1f}s total "
             f"requests={self._prof_total_requests}")
        return s


class MPRLPlayer(Player):
    """Worker-side RL player for multiprocess collection.

    Same logic as V9RLPlayer but sends obs to main process via queue
    instead of calling InferenceBatcher directly. No GPU access.
    """

    def __init__(self, worker_id: int, request_queue: MPQueue, result_queue: MPQueue,
                 reward_shaper_cfg: Optional[dict] = None,
                 temperature: float = 1.0, turn_cap: int = 300, **kwargs):
        super().__init__(**kwargs)
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.result_queue = result_queue
        self._rs_cfg = reward_shaper_cfg or {"ko_coef": 0.05, "hp_coef": 0.02, "clip_abs": 2.0}
        self.temperature = temperature
        self.turn_cap = turn_cap
        self._trajectories: Dict[str, Trajectory] = {}
        self._reward_shapers: Dict[str, RewardShaper] = {}
        self.completed_trajectories: List[Trajectory] = []
        self._tainted: set = set()
        self._request_id = 0

    def _get_shaper(self, btag):
        if btag not in self._reward_shapers:
            self._reward_shapers[btag] = RewardShaper(**self._rs_cfg)
        return self._reward_shapers[btag]

    def _get_traj(self, btag):
        if btag not in self._trajectories:
            self._trajectories[btag] = Trajectory()
        return self._trajectories[btag]

    def _build_turn_batch_cpu(self, feat: dict) -> dict:
        """Convert features_v8 output to model batch dict on CPU."""
        # Same as V9RLPlayer._build_turn_batch but tensors stay on CPU
        def _pi(p):
            i = p["ids"]
            return [i["species"], i["item"], i["ability"]]
        def _pb(p):
            b = p["banks"]
            return [b["hp_pct"], b["level"], b["weight"], b["height"],
                    b["stat_hp"], b["stat_atk"], b["stat_def"],
                    b["stat_spa"], b["stat_spd"], b["stat_spe"]]
        def _pmi(p):
            i = p["ids"]
            return [i["move0"], i["move1"], i["move2"], i["move3"]]
        def _pmc(p):
            c = p["continuous"]
            b = len(c) - 92
            return [c[b+i*23:b+(i+1)*23] for i in range(4)]

        our, opp = feat["our_pokemon"], feat["opp_pokemon"]
        int_arrays = {
            "our_pokemon_ids": np.array([[_pi(p) for p in our]], dtype=np.int64),
            "our_pokemon_banks": np.array([[_pb(p) for p in our]], dtype=np.int64),
            "our_pokemon_move_ids": np.array([[_pmi(p) for p in our]], dtype=np.int64),
            "opp_pokemon_ids": np.array([[_pi(p) for p in opp]], dtype=np.int64),
            "opp_pokemon_banks": np.array([[_pb(p) for p in opp]], dtype=np.int64),
            "opp_pokemon_move_ids": np.array([[_pmi(p) for p in opp]], dtype=np.int64),
        }
        float_arrays = {
            "our_pokemon_cont": np.array([[p["continuous"] for p in our]], dtype=np.float32),
            "our_pokemon_move_cont": np.array([[_pmc(p) for p in our]], dtype=np.float32),
            "opp_pokemon_cont": np.array([[p["continuous"] for p in opp]], dtype=np.float32),
            "opp_pokemon_move_cont": np.array([[_pmc(p) for p in opp]], dtype=np.float32),
            "field_cont": np.array([feat["field"]["continuous"]], dtype=np.float32),
            "transition_cont": np.array([feat["transition"]["continuous"]], dtype=np.float32),
            "legal_mask": feat["legal_mask"].reshape(1, 9).astype(np.float32),
        }

        mids, mbp, mac, mpp, mpr, mco = [], [], [], [], [], []
        for m in feat["active_moves"]:
            if m is None:
                mids.append(0); mbp.append(0); mac.append(0); mpp.append(0); mpr.append(6)
                from features import MOVE_SLOT_CONT_DIM
                mco.append([0.0]*MOVE_SLOT_CONT_DIM)
            else:
                mids.append(m["move_id"]); mbp.append(m["bp_int"]); mac.append(m["acc_int"])
                mpp.append(m["pp_int"]); mpr.append(m["priority_int"]); mco.append(m["continuous"])
        int_arrays["active_move_ids"] = np.array([mids], dtype=np.int64)
        float_arrays["active_move_cont"] = np.array([mco], dtype=np.float32)

        sids, sco = [], []
        for s in feat["switch_slots"]:
            if s is None:
                from features import SWITCH_SLOT_CONT_DIM
                sids.append(0); sco.append([0.0]*SWITCH_SLOT_CONT_DIM)
            else:
                sids.append(s["species_id"]); sco.append(s["continuous"])
        int_arrays["switch_ids"] = np.array([sids], dtype=np.int64)
        float_arrays["switch_cont"] = np.array([sco], dtype=np.float32)

        batch = {}
        for k, arr in int_arrays.items():
            batch[k] = torch.from_numpy(arr)  # CPU tensor
        for k, arr in float_arrays.items():
            batch[k] = torch.from_numpy(arr)  # CPU tensor

        fb = feat["field"]["banks"]
        batch["field_banks"] = {k: torch.tensor([fb[k]], dtype=torch.long) for k in fb}
        ti = feat["transition"]["ids"]
        batch["transition_ids"] = {k: torch.tensor([ti[k]], dtype=torch.long) for k in ti}
        batch["active_move_banks"] = {
            "bp": torch.tensor([mbp], dtype=torch.long),
            "acc": torch.tensor([mac], dtype=torch.long),
            "pp": torch.tensor([mpp], dtype=torch.long),
            "prio": torch.tensor([mpr], dtype=torch.long),
        }
        return batch

    async def choose_move(self, battle):
        btag = battle.battle_tag
        traj = self._get_traj(btag)
        shaper = self._get_shaper(btag)

        if len(traj) >= self.turn_cap:
            print(f"  [TURN CAP] {btag} hit {self.turn_cap} turns, forfeiting", flush=True)
            try:
                self.forfeit_battle(battle)
            except Exception:
                pass
            return self.choose_random_move(battle)

        feat = make_features(battle)

        if len(traj.rewards) > 0:
            our_move_immune = feat["transition"]["continuous"][9] > 0.5
            traj.rewards[-1] += shaper.step(battle, our_move_immune=our_move_immune)

        batch_cpu = self._build_turn_batch_cpu(feat)

        # Send inference request to main process
        self._request_id += 1
        self.request_queue.put((_MSG_INFER, self.worker_id, btag, batch_cpu))

        # Await result — use asyncio to yield control while waiting
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self.result_queue.get, True, 30.0)

        logits = result["action_logits"]  # (9,) CPU tensor
        value_t = result["value"]

        # Sample action
        if self.temperature != 1.0:
            scaled = logits / self.temperature
        else:
            scaled = logits

        if torch.isnan(scaled).any() or torch.isinf(scaled).any():
            self._tainted.add(btag)
            return self.choose_random_move(battle)

        probs = F.softmax(scaled, dim=-1)
        if torch.isnan(probs).any() or (probs < 0).any():
            self._tainted.add(btag)
            return self.choose_random_move(battle)

        action_idx = torch.multinomial(probs, 1).item()
        log_prob = F.log_softmax(logits, dim=-1)[action_idx].item()
        value = value_t.item()

        # Store trajectory (all CPU)
        traj.feat_batches.append(batch_cpu)
        traj.actions.append(action_idx)
        traj.log_probs.append(log_prob)
        traj.values.append(value)
        traj.rewards.append(0.0)
        traj.dones.append(False)
        traj.action_masks.append(feat["legal_mask"].copy())

        return self._action_to_order(battle, action_idx)

    def _action_to_order(self, battle, idx):
        if idx < 4:
            moves = list(battle.available_moves or [])
            if idx < len(moves):
                return self.create_order(moves[idx])
        else:
            sw = list(battle.available_switches or [])
            si = idx - 4
            if si < len(sw):
                return self.create_order(sw[si])
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])
        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        btag = battle.battle_tag
        traj = self._trajectories.get(btag)
        if traj and len(traj) > 0:
            if btag in self._tainted:
                print(f"  [TAINTED] w{self.worker_id} discarding {btag}", flush=True)
                self._tainted.discard(btag)
            else:
                shaper = self._get_shaper(btag)
                traj.rewards[-1] += shaper.step(battle, our_move_immune=False)
                if battle.won:
                    traj.rewards[-1] += 1.0
                elif battle.lost:
                    traj.rewards[-1] -= 1.0
                traj.dones[-1] = True
                # Send completed trajectory to main process
                self.request_queue.put((_MSG_TRAJ, self.worker_id, traj))
                self.completed_trajectories.append(traj)

        # Signal main to clear history for this battle
        self.request_queue.put((_MSG_CLEAR, self.worker_id, btag))

        self._tainted.discard(btag)
        self._trajectories.pop(btag, None)
        self._reward_shapers.pop(btag, None)
        super()._battle_finished_callback(battle)


def _mp_worker(
    worker_id: int,
    request_queue: MPQueue,
    result_queue: MPQueue,
    server_url: str,
    opponent_checkpoints: List[Tuple[str, int]],  # [(ckpt_path, n_games), ...]
    max_concurrent: int,
    rs_cfg: dict,
    temp_range: Tuple[float, float],
    snapshot_pool_size: int,
    teambuilder_path: Optional[str],
):
    """Worker process entry point. Runs battles, sends obs to main for inference."""
    import warnings
    warnings.filterwarnings("ignore")

    # Build teambuilder in worker process
    if teambuilder_path:
        tb = procedural_teambuilder(teambuilder_path)
    else:
        tb = None

    srv = _make_server(server_url)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    wins = losses = ties = 0

    for opp_ckpt, n_games in opponent_checkpoints:
        opp_name = Path(opp_ckpt).stem
        batch_id = worker_id * 1000 + hash(opp_name) % 1000

        player = MPRLPlayer(
            worker_id=worker_id,
            request_queue=request_queue,
            result_queue=result_queue,
            reward_shaper_cfg=rs_cfg,
            temperature=1.0,
            turn_cap=300,
            battle_format="gen9ou",
            team=tb or random_pool_teambuilder(),
            max_concurrent_battles=max_concurrent,
            account_configuration=AccountConfiguration(f"MPw{worker_id}r{batch_id}", None),
            server_configuration=srv,
        )

        # Opponent: CPU inference in this worker process
        is_latest = False  # simplified — temperature logic handled below
        if snapshot_pool_size > 15:
            opp_temp_range = (1.0, 1.0)
        else:
            opp_temp_range = temp_range

        opp_tb = tb or random_pool_teambuilder()
        opponent = SelfPlayOpponent(
            checkpoint_path=opp_ckpt,
            device="cpu",  # CPU in worker for cloud; locally use single-process mode instead
            temp_range=opp_temp_range,
            battle_format="gen9ou",
            team=opp_tb,
            max_concurrent_battles=max_concurrent,
            account_configuration=AccountConfiguration(f"MPo{worker_id}r{batch_id}", None),
            server_configuration=srv,
        )

        try:
            loop.run_until_complete(asyncio.wait_for(
                player.battle_against(opponent, n_battles=n_games),
                timeout=max(300, n_games * 30),
            ))
        except asyncio.TimeoutError:
            print(f"  [WARN] w{worker_id} timed out vs {opp_name}", flush=True)
        except Exception as e:
            print(f"  [ERROR] w{worker_id} vs {opp_name}: {e}", flush=True)

        wins += player.n_won_battles
        losses += player.n_lost_battles
        ties += player.n_tied_battles

        try:
            player.reset_battles()
        except EnvironmentError:
            pass
        try:
            opponent.reset_battles()
        except EnvironmentError:
            pass
        _cancel_listener(player)
        _cancel_listener(opponent)
        del player, opponent

    loop.close()

    # Signal done
    request_queue.put((_MSG_DONE, worker_id))
    return wins, losses, ties


def mp_collect_v9(
    model: PokeTransformer, device: torch.device,
    server_pool: List[ServerConfiguration],
    n_games: int = 200, max_concurrent: int = 10,
    snapshot_pool: List[str] = None, fp16: bool = True,
    reward_shaper_cfg: Optional[dict] = None,
    temp_range: Tuple[float, float] = (1.0, 2.25),
    latest_snapshot: Optional[str] = None,
    teambuilder_path: Optional[str] = None,
):
    """Multiprocess self-play collection. Workers handle battles on CPU,
    main process handles GPU inference via InferenceServer."""
    if not snapshot_pool:
        raise ValueError("snapshot_pool must contain at least one checkpoint")

    n_workers = len(server_pool)
    rs_cfg = reward_shaper_cfg or {"ko_coef": 0.05, "hp_coef": 0.02, "clip_abs": 2.0}

    # Select opponents
    max_opponents = 15
    if len(snapshot_pool) <= max_opponents:
        selected = list(snapshot_pool)
    else:
        selected = random.sample(snapshot_pool, max_opponents)
        if latest_snapshot and latest_snapshot not in selected:
            selected[-1] = latest_snapshot

    # Distribute opponents across workers (round-robin)
    games_per_opp = max(1, n_games // len(selected))
    remainder = n_games - games_per_opp * len(selected)

    worker_assignments: Dict[int, List[Tuple[str, int]]] = {i: [] for i in range(n_workers)}
    for oi, opp_ckpt in enumerate(selected):
        n = games_per_opp + (1 if oi < remainder else 0)
        if n <= 0:
            continue
        wid = oi % n_workers
        worker_assignments[wid].append((opp_ckpt, n))

    # Create queues
    request_queue = MPQueue()
    result_queues = {i: MPQueue() for i in range(n_workers)}

    # Create inference server
    server = InferenceServer(
        model, device, request_queue, result_queues,
        fp16=fp16, batch_timeout_ms=20,
        min_batch=max(2, max_concurrent // 2),
    )

    t0 = time.time()

    # Spawn worker processes
    processes = []
    for wid in range(n_workers):
        srv_url = server_pool[wid].websocket_url
        p = mp_mod.Process(
            target=_mp_worker,
            args=(
                wid, request_queue, result_queues[wid],
                srv_url, worker_assignments[wid],
                max_concurrent, rs_cfg, temp_range,
                len(snapshot_pool),
                teambuilder_path,
            ),
            daemon=True,
        )
        p.start()
        processes.append(p)

    # Run inference server until all workers done
    server.run_until_workers_done(n_workers)

    # Wait for worker processes to exit
    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    elapsed = time.time() - t0

    # Collect results
    all_trajs = server.trajectories
    total_steps = sum(len(t) for t in all_trajs)
    total_wins = sum(1 for t in all_trajs if t.dones and t.dones[-1] and t.rewards[-1] > 0)
    total_losses = sum(1 for t in all_trajs if t.dones and t.dones[-1] and t.rewards[-1] < 0)

    # Build opponent summary from trajectory count per snapshot
    opp_summary = f"mp_{n_workers}w_{len(all_trajs)}ep"

    prof = server.prof_summary()
    print(f"  [MP-PROF] {prof}", flush=True)

    # Verify: check no stale histories remain
    if server.history:
        print(f"  [MP-WARN] {len(server.history)} stale histories after collection", flush=True)
        server.history.clear()

    gc.collect()
    return all_trajs, total_wins, total_losses, 0, total_steps, opp_summary, elapsed


# =============================
# CLI
# =============================

def parse_args():
    p = argparse.ArgumentParser(description="V9 Pure Self-Play PPO with Batched Inference")
    p.add_argument("--init-from", required=True, help="Init checkpoint (e.g. iter80)")
    p.add_argument("--resume", default=None, help="Resume from v9 checkpoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--opponent-device", default="cuda")
    p.add_argument("--servers", default="9000", help="Comma-separated ports")
    p.add_argument("--games-per-iter", type=int, default=200)
    p.add_argument("--max-concurrent", type=int, default=20)
    p.add_argument("--n-iters", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.9999)
    p.add_argument("--lam", type=float, default=0.75)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=5)
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--adaptive-entropy", action="store_true",
                   help="Auto-adjust ent_coef to keep entropy in [0.55, 0.80] range")
    p.add_argument("--vf-coef", type=float, default=1.0)
    p.add_argument("--target-kl", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--grad-accum", type=int, default=10,
                   help="Accumulate gradients over N episodes before each optimizer step")
    p.add_argument("--warmup-iters", type=int, default=5)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--ko-coef", type=float, default=0.05)
    p.add_argument("--hp-coef", type=float, default=0.02)
    p.add_argument("--reward-clip", type=float, default=2.0)
    p.add_argument("--temp-min", type=float, default=1.0, help="Opponent temp range min")
    p.add_argument("--temp-max", type=float, default=2.25, help="Opponent temp range max")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile spatial encoder (Linux only, ~1.5-2x spatial speedup)")
    p.add_argument("--pipeline", action="store_true",
                   help="Pipeline collection and PPO update (overlap on GPU, ~1.7x speedup)")
    p.add_argument("--snapshot-interval", type=int, default=5, help="Save snapshot every N iters")
    p.add_argument("--eval-interval", type=int, default=20)
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--out-dir", default="data/models/rl_v9")
    p.add_argument("--immune-penalty", type=float, default=0.0,
                   help="Per-step penalty when our move hits immunity (0=off, 0.01 recommended)")
    p.add_argument("--procedural-teams", default=None,
                   help="Path to Smogon usage stats dir for procedural team generation "
                        "(e.g. raw_data/pokemon_usage/2024-04). If unset, uses 70 handcrafted teams.")
    p.add_argument("--random-team-pct", type=float, default=0.05,
                   help="Fraction of procedural teams with uniform weights (default 0.05)")
    p.add_argument("--lr-restart", action="store_true",
                   help="Reset optimizer on resume (use when dims changed or hyperparams changed)")
    p.add_argument("--mp", action="store_true",
                   help="Use multiprocess collection (workers on CPU, GPU inference centralized)")
    p.add_argument("--batch-timeout-ms", type=float, default=15,
                   help="InferenceBatcher batch timeout in ms (lower = faster fire, smaller batches)")
    p.add_argument("--reward-style", choices=["dense", "sparse", "terminal"], default="dense",
                   help="Reward shaping style. "
                        "dense: KO+HP shaping + immune + terminal (default). "
                        "sparse: immune penalty + terminal only (no KO/HP). "
                        "terminal: win/loss only (no shaping).")
    add_model_args(p)
    return p.parse_args()


# =============================
# Main
# =============================

def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load init checkpoint
    model, cfg, init_ckpt = load_checkpoint(args.init_from, device)
    model.to(device)

    # torch.compile (Linux/cloud only — ~1.5-2x spatial speedup)
    compiled = False
    if args.compile:
        try:
            model.forward_spatial = torch.compile(model.forward_spatial, mode="reduce-overhead")
            compiled = True
            print("torch.compile: spatial encoder compiled successfully", flush=True)
        except Exception as e:
            print(f"torch.compile: SKIPPED ({e})", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Run directory
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / f"selfplay_v9_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    # Server pool
    server_pool = [_make_server(s.strip()) for s in args.servers.split(",")]

    # Snapshot pool — start with init checkpoint
    snapshot_pool = [args.init_from]

    # Reward shaper config — three styles via --reward-style flag
    #   dense (default): KO + HP shaping + immune penalty + terminal (current behavior)
    #   sparse:          immune penalty + terminal only (no KO/HP shaping)
    #   terminal:        terminal win/loss only (no shaping at all)
    reward_style = getattr(args, 'reward_style', 'dense')
    if reward_style == 'dense':
        rs_cfg = {"ko_coef": args.ko_coef, "hp_coef": args.hp_coef,
                  "clip_abs": args.reward_clip, "immune_penalty": args.immune_penalty}
    elif reward_style == 'sparse':
        rs_cfg = {"ko_coef": 0.0, "hp_coef": 0.0,
                  "clip_abs": args.reward_clip, "immune_penalty": args.immune_penalty}
    elif reward_style == 'terminal':
        rs_cfg = {"ko_coef": 0.0, "hp_coef": 0.0,
                  "clip_abs": args.reward_clip, "immune_penalty": 0.0}
    else:
        raise ValueError(f"Unknown reward_style: {reward_style}")
    print(f"Reward style: {reward_style} ({rs_cfg})", flush=True)
    reward_shaper = RewardShaper(**rs_cfg)

    # Team builder — procedural for training, handcrafted for eval
    if args.procedural_teams:
        train_teambuilder = procedural_teambuilder(args.procedural_teams,
                                                    random_pct=args.random_team_pct)
    else:
        train_teambuilder = None  # will use random_pool_teambuilder() per-player

    # Save config
    config = vars(args)
    config["run_dir"] = str(run_dir)
    config["init_checkpoint"] = args.init_from
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Resume support
    start_iter = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # Handle dim expansion on resume (move 107->109, switch 28->29)
        resume_state = ckpt["model_state_dict"]
        _expand_targets = ["move_net.mlp.0.weight", "switch_mlp.0.weight"]
        for key in list(resume_state.keys()):
            if any(key.endswith(t) for t in _expand_targets):
                old_w = resume_state[key]
                parts = key.split(".")
                mod = model
                for p in parts[:-1]:
                    mod = getattr(mod, p) if not p.isdigit() else mod[int(p)]
                expected_in = mod.in_features
                if old_w.shape[1] < expected_in:
                    pad = expected_in - old_w.shape[1]
                    resume_state[key] = torch.cat([old_w, torch.zeros(old_w.shape[0], pad, device=old_w.device)], dim=1)
                    print(f"  [INFO] Expanding {key}: {old_w.shape[1]} -> {expected_in} (+{pad} dims, zero-init)")
        model.load_state_dict(resume_state)
        if args.lr_restart:
            print("  [INFO] --lr-restart: optimizer reset (fresh Adam state)")
        else:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iter = ckpt.get("iteration", 0) + 1
        snapshot_pool = ckpt.get("metrics", {}).get("snapshot_pool", snapshot_pool)

        # Scan disk for ALL existing snapshots and add to pool (rebuilds full history)
        # Filter pre-sp260: first 47% eval was around iter 259 (training_type_eff.log).
        # Snapshots before that are from pre-competent era (eval 25-44%), provide minimal
        # training signal and can corrupt value function with suboptimal play patterns.
        import glob as _glob, re as _re
        MIN_SNAPSHOT_ITER = 260
        all_disk_snapshots = sorted(set(_glob.glob("data/models/rl_v9/selfplay_v9_*/snapshot_*.pt")))
        def _snap_iter(path):
            m = _re.search(r'snapshot_(\d+)\.pt$', path)
            return int(m.group(1)) if m else 0
        all_disk_snapshots = [s for s in all_disk_snapshots if _snap_iter(s) >= MIN_SNAPSHOT_ITER]
        existing = set(snapshot_pool)
        new_snapshots = [s for s in all_disk_snapshots if s not in existing]
        added = len(new_snapshots)
        if new_snapshots:
            snapshot_pool = new_snapshots + snapshot_pool
        print(f"Resumed from {args.resume}, starting at iter {start_iter}, "
              f"pool: {len(snapshot_pool)} checkpoints (+{added} from disk scan, "
              f"filtered sp<{MIN_SNAPSHOT_ITER})", flush=True)

    opp_device = args.opponent_device
    loop = asyncio.new_event_loop()

    print(f"\n=== V9 Pure Self-Play PPO ===")
    print(f"Init: {args.init_from}")
    print(f"Run dir: {run_dir}")
    print(f"Iters: {args.n_iters}, Games/iter: {args.games_per_iter}, Concurrent: {args.max_concurrent}")
    print(f"gamma={args.gamma}, lam={args.lam}, ent={args.ent_coef}, target_kl={args.target_kl}, grad_accum={args.grad_accum}")
    print(f"Opponent temp range: [{args.temp_min}, {args.temp_max}]")
    print(f"FP16: {'ON' if args.fp16 else 'OFF'}, Compile: {'ON' if compiled else 'OFF'}, "
          f"Pipeline: {'ON' if args.pipeline else 'OFF'}, Device: {device}, Opp device: {opp_device}")
    print(f"Snapshot interval: every {args.snapshot_interval} iters (keep all)")
    print(f"Value warmup: {args.warmup_iters} iters")
    print(f"Immune penalty: {args.immune_penalty}")
    print(f"Teams: {'procedural (' + args.procedural_teams + ')' if args.procedural_teams else 'handcrafted (70 OU)'}")
    print(f"Servers: {[s.websocket_url for s in server_pool]}")
    print(f"Snapshot pool: {len(snapshot_pool)} checkpoints\n", flush=True)

    best_eval_wr = 0.0
    ent_coef = args.ent_coef  # mutable — adapted each iter to keep entropy in safe range
    bg_collector = BackgroundCollector(cpu_inference=False) if args.pipeline else None
    collect_args = {
        "games_per_iter": args.games_per_iter,
        "max_concurrent": args.max_concurrent,
        "fp16": args.fp16,
        "rs_cfg": rs_cfg,
        "temp_range": (args.temp_min, args.temp_max),
        "opponent_device": opp_device,
        "teambuilder": train_teambuilder,
    }
    pending_collection = None  # pre-collected result from background
    mp_bg_collector = None  # for mp+pipeline combined mode

    for it in range(start_iter, start_iter + args.n_iters):
        t0 = time.time()

        # Value warmup (freeze backbone+policy, train only value head)
        in_warmup = (it - start_iter) < args.warmup_iters
        if in_warmup:
            for name, param in model.named_parameters():
                param.requires_grad = "value_head" in name
        elif (it - start_iter) == args.warmup_iters:
            for param in model.parameters():
                param.requires_grad = True
            print(f"  Value warmup complete, unfreezing all parameters", flush=True)

        # ---- Collection ----
        from datetime import datetime as _dt
        _flow_t0 = time.time()
        def _flow(msg):
            elapsed = time.time() - _flow_t0
            print(f"  [FLOW {_dt.now().strftime('%H:%M:%S')} +{elapsed:6.1f}s] {msg}", flush=True)
        _flow("iter start")

        if pending_collection is not None:
            # Use pre-collected data from background/mp pipeline
            _flow("using pre-collected data from background")
            trajs, wins, losses, ties, steps, opp_name, collect_time = pending_collection
            pending_collection = None
            _flow(f"unpacked pre-collected: {len(trajs)} trajs, {steps} steps")
        elif getattr(args, 'mp', False):
            # Multiprocess collection v2
            from mp_collect_v2 import mp_collect_v2
            model.eval()
            latest_sp = snapshot_pool[-1] if len(snapshot_pool) > 1 else None
            tb_path = getattr(args, 'procedural_teams', None)
            trajs, wins, losses, ties, steps, opp_name, collect_time = mp_collect_v2(
                model, device, server_pool,
                n_games=args.games_per_iter,
                max_concurrent=args.max_concurrent,
                snapshot_pool=snapshot_pool,
                fp16=args.fp16,
                reward_shaper_cfg=rs_cfg,
                temp_range=(args.temp_min, args.temp_max),
                latest_snapshot=latest_sp,
                teambuilder_path=tb_path,
                opponent_device=opp_device,
                batch_timeout_ms=args.batch_timeout_ms,
            )
        else:
            # Single-process collection
            _flow("starting SYNC collection (no pre-collected available)")
            model.eval()
            latest_sp = snapshot_pool[-1] if len(snapshot_pool) > 1 else None
            trajs, wins, losses, ties, steps, opp_name, collect_time = loop.run_until_complete(
                collect_v9(
                    model, device, server_pool,
                    n_games=args.games_per_iter,
                    max_concurrent=args.max_concurrent,
                    snapshot_pool=snapshot_pool,
                    fp16=args.fp16,
                    reward_shaper_cfg=rs_cfg,
                    temp_range=(args.temp_min, args.temp_max),
                    opponent_device=opp_device,
                    latest_snapshot=latest_sp,
                    teambuilder=train_teambuilder,
                )
            )
            _flow(f"sync collection done: collect_time={collect_time:.0f}s, {len(trajs)} trajs")

        total_games = wins + losses + ties
        wr = wins / max(1, total_games)

        # ---- Start background collection for NEXT iter (pipeline) ----
        if args.mp and args.pipeline and not in_warmup:
            # MP + pipeline combined: multiprocess collection in background thread
            from mp_collect_v2 import MPPipelineCollector
            if mp_bg_collector is None:
                mp_bg_collector = MPPipelineCollector()
            mp_collect_args = {
                "games_per_iter": args.games_per_iter,
                "max_concurrent": args.max_concurrent,
                "fp16": args.fp16,
                "rs_cfg": rs_cfg,
                "temp_range": (args.temp_min, args.temp_max),
                "teambuilder_path": getattr(args, 'procedural_teams', None),
                "opponent_device": opp_device,
                "batch_timeout_ms": args.batch_timeout_ms,
            }
            mp_bg_collector.start(model, device, server_pool, snapshot_pool, mp_collect_args)
        elif bg_collector and not in_warmup and not args.mp:
            # Single-process pipeline
            _flow("starting BACKGROUND collection thread for next iter")
            bg_collector.start(model, device, server_pool, snapshot_pool, collect_args)
            _flow("background thread spawned")

        # ---- PPO Update ----
        _flow("building PPO episodes")
        episodes = build_ppo_episodes(trajs, gamma=args.gamma, lam=args.lam)
        _flow(f"PPO episodes built: {len(episodes)} episodes")

        model.train()
        if in_warmup:
            for name, param in model.named_parameters():
                param.requires_grad = "value_head" in name

        _flow("starting PPO update")
        t_update = time.time()
        loss_info = ppo_update(
            model, optimizer, episodes, device, cfg,
            epochs=args.ppo_epochs,
            clip_eps=args.clip_eps,
            ent_coef=ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            grad_accum=args.grad_accum,
        )
        update_time = time.time() - t_update
        _flow(f"PPO update DONE: {update_time:.0f}s")

        # ---- Catastrophic-failure guard (added Session 33 post-script) ----
        # If 0 episodes succeeded in PPO, the catch loop swallowed every exception
        # (most likely a wedged CUDA context after a driver TDR). Returning here
        # would log a "successful" iter with all-zero stats, optionally write a
        # tainted snapshot, and try an eval on a dead GPU. Save an emergency
        # checkpoint and exit so the next session can resume cleanly.
        if loss_info.get("n_succeeded", 1) == 0:
            n_failed_str = loss_info.get("n_failed", "?")
            print(f"  [FATAL] PPO update produced 0 successful episodes "
                  f"(n_failed={n_failed_str}, n_episodes={len(episodes)}). "
                  f"Likely CUDA context loss. Saving emergency checkpoint and exiting.",
                  flush=True)
            try:
                emerg_path = str(run_dir / f"emergency_iter_{it:04d}.pt")
                save_checkpoint(emerg_path, model, cfg, optimizer, it, metrics={
                    "win_rate": wr, "best_eval_wr": best_eval_wr,
                    "snapshot_pool": snapshot_pool[-500:],
                })
                print(f"  [FATAL] Emergency checkpoint saved: {emerg_path}", flush=True)
            except Exception as e:
                print(f"  [FATAL] Emergency save failed (GPU likely dead): {e}", flush=True)
            try:
                writer.close()
            except Exception:
                pass
            sys.exit(2)

        # ---- Wait for background collection if running ----
        if mp_bg_collector is not None and mp_bg_collector.running:
            _flow("waiting for MP background collection (join)")
            pending_collection = mp_bg_collector.join()
            _flow(f"MP background join done, result={'OK' if pending_collection else 'NONE'}")
            if pending_collection is None:
                pass  # failed, will fall back to sync next iter
        elif bg_collector and bg_collector.running:
            _flow("waiting for background collection (join)")
            pending_collection = bg_collector.join()
            _flow(f"background join done, result={'OK' if pending_collection else 'NONE'}")
            if pending_collection is None:
                pass
        elif bg_collector and not bg_collector.running and bg_collector._result is not None:
            _flow("background collection ALREADY DONE before join (good overlap!)")
            pending_collection = bg_collector.join()

        # ---- Logging ----
        kl_str = f" kl={loss_info['kl']:.4f}" if 'kl' in loss_info else ""
        warmup_str = " [WARMUP]" if in_warmup else ""
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Iter {it}: W/L/T={wins}/{losses}/{ties} ({wr:.1%}), {steps} steps, "
              f"collect={collect_time:.0f}s, update={update_time:.0f}s, "
              f"pi={loss_info['pi']:.4f} v={loss_info['v']:.4f} "
              f"ent={loss_info['ent']:.4f}{kl_str}{warmup_str} "
              f"vs={opp_name} pool={len(snapshot_pool)}",
              flush=True)

        # TensorBoard
        writer.add_scalar("train/win_rate", wr, it)
        writer.add_scalar("train/pi_loss", loss_info["pi"], it)
        writer.add_scalar("train/v_loss", loss_info["v"], it)
        writer.add_scalar("train/entropy", loss_info["ent"], it)
        if "kl" in loss_info:
            writer.add_scalar("train/kl", loss_info["kl"], it)
        writer.add_scalar("train/collect_time", collect_time, it)
        writer.add_scalar("train/update_time", update_time, it)
        writer.add_scalar("train/steps", steps, it)
        writer.add_scalar("train/pool_size", len(snapshot_pool), it)

        # ---- Adaptive entropy (optional): keep entropy in target range ----
        if args.adaptive_entropy and loss_info["ent"] > 0.01:
            if loss_info["ent"] < 0.55:
                ent_coef = min(ent_coef * 1.05, 0.06)
                print(f"  [ENT] Entropy low ({loss_info['ent']:.3f}), raising ent_coef to {ent_coef:.4f}")
            elif loss_info["ent"] > 0.80:
                ent_coef = max(ent_coef * 0.95, 0.01)
                print(f"  [ENT] Entropy high ({loss_info['ent']:.3f}), lowering ent_coef to {ent_coef:.4f}")

        # ---- Snapshot (every N iters, keep ALL) ----
        if (it + 1) % args.snapshot_interval == 0:
            # Skip saving if no usable steps were collected (collapsed model)
            if steps < 100:
                print(f"  Snapshot SKIPPED: only {steps} steps (min 100 required)", flush=True)
            elif loss_info.get("n_succeeded", 1) == 0:
                # Belt-and-suspenders: zero-success PPO would normally trigger the
                # FATAL guard above. Keep this so a tainted snapshot never enters
                # the pool even if that guard is later loosened.
                print(f"  Snapshot SKIPPED: 0 PPO episodes succeeded (tainted iter)", flush=True)
            else:
                sp_path = str(run_dir / f"snapshot_{it:04d}.pt")
                save_checkpoint(sp_path, model, cfg, optimizer, it, metrics={
                    "win_rate": wr, "best_eval_wr": best_eval_wr,
                    "snapshot_pool": snapshot_pool[-500:],
                })
                snapshot_pool.append(sp_path)
                print(f"  Snapshot saved: {sp_path} (pool={len(snapshot_pool)})", flush=True)

        # ---- Bot Eval (every N iters) ----
        if (it + 1) % args.eval_interval == 0:
            try:
                # Save temp checkpoint for eval
                tmp = str(run_dir / f"iter_{it:04d}.pt")
                save_checkpoint(tmp, model, cfg, optimizer, it)

                from train_bc import eval_vs_bots
                srv_url = f"ws://127.0.0.1:{args.servers.split(',')[0].strip()}/showdown/websocket"
                replay_path = str(run_dir / f"replays_iter{it:04d}")
                results = eval_vs_bots(tmp, device=str(device), n_battles=args.eval_games,
                                       server_url=srv_url, replay_dir=replay_path)
                sh = results.get("SH", 0)
                smd = results.get("SmartDmg", results.get("SmD", 0))
                tac = results.get("Tactical", results.get("Tac", 0))
                stra = results.get("Strategic", results.get("Str", 0))
                smart_avg = (sh + smd + tac + stra) / 4

                print(f"  EVAL: SH={sh:.0f}%, SmartDmg={smd:.0f}%, Tactical={tac:.0f}%, "
                      f"Strategic={stra:.0f}%, smart_avg={smart_avg:.0f}%", flush=True)

                writer.add_scalar("eval/smart_avg", smart_avg, it)
                writer.add_scalar("eval/SH", sh, it)
                writer.add_scalar("eval/SmartDmg", smd, it)
                writer.add_scalar("eval/Tactical", tac, it)
                writer.add_scalar("eval/Strategic", stra, it)

                if smart_avg > best_eval_wr:
                    best_eval_wr = smart_avg
            except Exception as e:
                print(f"  [ERROR] Eval failed: {e}", flush=True)

        # Memory cleanup
        del trajs, episodes
        gc.collect()
        torch.cuda.empty_cache()

    # Final save
    final_path = str(run_dir / "final.pt")
    save_checkpoint(final_path, model, cfg, optimizer, start_iter + args.n_iters - 1,
                       metrics={"best_eval_wr": best_eval_wr, "snapshot_pool": snapshot_pool[-500:]})
    print(f"\nTraining complete. Final checkpoint: {final_path}", flush=True)
    writer.close()
    loop.close()


if __name__ == "__main__":
    main()
