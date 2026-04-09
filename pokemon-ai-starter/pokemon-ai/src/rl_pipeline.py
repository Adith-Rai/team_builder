# rl_pipeline.py — Multiprocess collection infrastructure for cloud/parallel training.
#
# Extracted from rl_train_v9.py during Session 34 refactor.
# InferenceServer: main-process GPU server for multiprocess collection
# MPRLPlayer: worker-side RL player (CPU, sends obs to main for GPU inference)
# mp_collect_v9: orchestrates multiprocess collection

from __future__ import annotations

import asyncio
import gc
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp_mod
from multiprocessing import Queue as MPQueue

from poke_env.player import Player
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from features import make_features, MOVE_SLOT_CONT_DIM, SWITCH_SLOT_CONT_DIM
from model import PokeTransformer
from ppo import Trajectory, _cancel_listener
from rewards import RewardShaper
from teams_ou import random_pool_teambuilder
from team_generator import procedural_teambuilder
from rl_player import SelfPlayOpponent
from rl_collection import _make_server

# Must use 'spawn' for CUDA in child processes (not 'fork')
try:
    mp_mod.set_start_method('spawn', force=False)
except RuntimeError:
    pass  # already set

# Message types for request queue
MSG_INFER = 0   # inference request: (worker_id, btag, obs_dict_cpu)
MSG_CLEAR = 1   # clear history: (worker_id, btag)
MSG_TRAJ = 2    # completed trajectory: (worker_id, trajectory)
MSG_DONE = 3    # worker finished: (worker_id,)

# Backward-compat aliases (used by test_mp_collection.py)
_MSG_INFER = MSG_INFER
_MSG_CLEAR = MSG_CLEAR
_MSG_TRAJ = MSG_TRAJ
_MSG_DONE = MSG_DONE


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
            infer_requests = []
            deadline = time.time() + self.batch_timeout

            while time.time() < deadline:
                try:
                    msg = self.request_queue.get(timeout=max(0.001, deadline - time.time()))
                except Exception:
                    break

                msg_type = msg[0]
                if msg_type == MSG_INFER:
                    _, wid, btag, obs_dict = msg
                    infer_requests.append((wid, btag, obs_dict))
                    if len(infer_requests) >= self.min_batch * 2:
                        break
                elif msg_type == MSG_CLEAR:
                    _, wid, btag = msg
                    self.history.pop((wid, btag), None)
                elif msg_type == MSG_TRAJ:
                    _, wid, traj = msg
                    self.trajectories.append(traj)
                elif msg_type == MSG_DONE:
                    _, wid = msg
                    workers_done.add(wid)

            if infer_requests:
                self._process_batch(infer_requests)

        # Drain remaining messages
        while not self.request_queue.empty():
            try:
                msg = self.request_queue.get_nowait()
                if msg[0] == MSG_TRAJ:
                    self.trajectories.append(msg[2])
                elif msg[0] == MSG_CLEAR:
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

        mega = self._stack_obs_to_device([r[2] for r in infer_requests])

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.fp16):
            t0 = time.time()

            spatial_out, summaries = model.forward_spatial(mega)
            action_ctx = model.action_encoder(
                mega["active_move_ids"], mega["active_move_banks"],
                mega["active_move_cont"], mega["switch_ids"], mega["switch_cont"],
            )

            # Batched temporal
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
                    hh = h.squeeze(0)
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

            actor_out = spatial_out[:, 0, :]
            at = torch.cat([actor_out, temporal_ctx], dim=-1)
            at_exp = at.unsqueeze(1).expand(-1, 9, -1)
            pi_input = torch.cat([at_exp, action_ctx], dim=-1)
            logits = model.policy_head(pi_input).squeeze(-1)

            if "legal_mask" in mega:
                logits = logits.float().masked_fill(mega["legal_mask"] < 0.5, -100.0)

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
            summary_f32 = summaries[i].float().unsqueeze(0).unsqueeze(0)
            h = self.history.get(key)
            if h is None:
                self.history[key] = summary_f32
            else:
                self.history[key] = torch.cat([h, summary_f32], dim=1)
                if self.history[key].shape[1] > self.max_temporal:
                    self.history[key] = self.history[key][:, -self.max_temporal:]

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
        sizes = np.array(self._prof_batch_sizes)
        times = np.array(self._prof_gpu_times)
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
        """Convert feature output to model batch dict on CPU."""
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
                mco.append([0.0]*MOVE_SLOT_CONT_DIM)
            else:
                mids.append(m["move_id"]); mbp.append(m["bp_int"]); mac.append(m["acc_int"])
                mpp.append(m["pp_int"]); mpr.append(m["priority_int"]); mco.append(m["continuous"])
        int_arrays["active_move_ids"] = np.array([mids], dtype=np.int64)
        float_arrays["active_move_cont"] = np.array([mco], dtype=np.float32)

        sids, sco = [], []
        for s in feat["switch_slots"]:
            if s is None:
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
        self.request_queue.put((MSG_INFER, self.worker_id, btag, batch_cpu))

        # Await result
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self.result_queue.get, True, 30.0)

        logits = result["action_logits"]
        value_t = result["value"]

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
                self.request_queue.put((MSG_TRAJ, self.worker_id, traj))
                self.completed_trajectories.append(traj)

        self.request_queue.put((MSG_CLEAR, self.worker_id, btag))
        self._tainted.discard(btag)
        self._trajectories.pop(btag, None)
        self._reward_shapers.pop(btag, None)
        super()._battle_finished_callback(battle)


def _mp_worker(
    worker_id: int,
    request_queue: MPQueue,
    result_queue: MPQueue,
    server_url: str,
    opponent_checkpoints: List[Tuple[str, int]],
    max_concurrent: int,
    rs_cfg: dict,
    temp_range: Tuple[float, float],
    snapshot_pool_size: int,
    teambuilder_path: Optional[str],
    battle_format: str = "gen9ou",
):
    """Worker process entry point. Runs battles, sends obs to main for inference."""
    import warnings
    warnings.filterwarnings("ignore")

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
            battle_format=battle_format,
            team=tb or random_pool_teambuilder(),
            max_concurrent_battles=max_concurrent,
            account_configuration=AccountConfiguration(f"MPw{worker_id}r{batch_id}", None),
            server_configuration=srv,
        )

        if snapshot_pool_size > 15:
            opp_temp_range = (1.0, 1.0)
        else:
            opp_temp_range = temp_range

        opp_tb = tb or random_pool_teambuilder()
        opponent = SelfPlayOpponent(
            checkpoint_path=opp_ckpt,
            device="cpu",
            temp_range=opp_temp_range,
            battle_format=battle_format,
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

    request_queue.put((MSG_DONE, worker_id))
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
    battle_format: str = "gen9ou",
):
    """Multiprocess self-play collection. Workers handle battles on CPU,
    main process handles GPU inference via InferenceServer."""
    if not snapshot_pool:
        raise ValueError("snapshot_pool must contain at least one checkpoint")

    n_workers = len(server_pool)
    rs_cfg = reward_shaper_cfg or {"ko_coef": 0.05, "hp_coef": 0.02, "clip_abs": 2.0}

    max_opponents = 15
    if len(snapshot_pool) <= max_opponents:
        selected = list(snapshot_pool)
    else:
        selected = random.sample(snapshot_pool, max_opponents)
        if latest_snapshot and latest_snapshot not in selected:
            selected[-1] = latest_snapshot

    games_per_opp = max(1, n_games // len(selected))
    remainder = n_games - games_per_opp * len(selected)

    worker_assignments: Dict[int, List[Tuple[str, int]]] = {i: [] for i in range(n_workers)}
    for oi, opp_ckpt in enumerate(selected):
        n = games_per_opp + (1 if oi < remainder else 0)
        if n <= 0:
            continue
        wid = oi % n_workers
        worker_assignments[wid].append((opp_ckpt, n))

    request_queue = MPQueue()
    result_queues = {i: MPQueue() for i in range(n_workers)}

    server = InferenceServer(
        model, device, request_queue, result_queues,
        fp16=fp16, batch_timeout_ms=20,
        min_batch=max(2, max_concurrent // 2),
    )

    t0 = time.time()

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
                battle_format,
            ),
            daemon=True,
        )
        p.start()
        processes.append(p)

    server.run_until_workers_done(n_workers)

    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    elapsed = time.time() - t0

    all_trajs = server.trajectories
    total_steps = sum(len(t) for t in all_trajs)
    total_wins = sum(1 for t in all_trajs if t.dones and t.dones[-1] and t.rewards[-1] > 0)
    total_losses = sum(1 for t in all_trajs if t.dones and t.dones[-1] and t.rewards[-1] < 0)

    opp_summary = f"mp_{n_workers}w_{len(all_trajs)}ep"

    prof = server.prof_summary()
    print(f"  [MP-PROF] {prof}", flush=True)

    if server.history:
        print(f"  [MP-WARN] {len(server.history)} stale histories after collection", flush=True)
        server.history.clear()

    gc.collect()
    return all_trajs, total_wins, total_losses, 0, total_steps, opp_summary, elapsed
