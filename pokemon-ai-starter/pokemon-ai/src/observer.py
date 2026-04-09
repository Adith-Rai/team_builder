# observer.py - Battle observation recorder for JSONL dataset generation
# Records complete battle state from poke-env self-play for behavioral cloning.
#
# Key design:
#   - RecorderMixin wraps any Player subclass via multiple inheritance
#   - Records the ACTUAL bot's decisions (not random actions)
#   - Uses poke-env 0.10.0 observations API for protocol history
#   - Writes JSONL with episode_id / t / obs / legal / action / mods / done

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import json
import os
import sys
import random
import signal
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player.battle_order import BattleOrder
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.player.baselines import RandomPlayer as PokeRandomPlayer

from policy_rulebots import (
    GreedySEPlayer,
    HazardSensePlayer,
    SwitchAwareEscapePlayer,
    SetupThenSweepPlayer,
)
from policy_smartbots import SmartDamagePlayer, TacticalPlayer, StrategicPlayer
from teams_ou import random_teambuilder, random_pool_teambuilder
from features import make_obs_mask_and_slots
from features import make_features
from bc_policy_player import BCPolicyPlayer

def _default_server() -> ServerConfiguration:
    """Build default server config from env vars (SHOWDOWN_HOST, SHOWDOWN_PORT)."""
    host = os.environ.get("SHOWDOWN_HOST", "127.0.0.1")
    port = os.environ.get("SHOWDOWN_PORT", "8000")
    return ServerConfiguration(
        f"ws://{host}:{port}/showdown/websocket",
        f"http://{host}:{port}/action.php?",
    )

DEFAULT_SERVER = _default_server()


# ---------------------------------------------------------------------------
# JSONL Writer
# ---------------------------------------------------------------------------

class JSONLRecorder:
    def __init__(self, path: str, fsync_every: int = 512):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a", buffering=1, encoding="utf-8")
        self._n = 0
        self._fsync_every = fsync_every

    def write(self, obj: Dict[str, Any]) -> None:
        def _json_default(o):
            try:
                import numpy as _np
                if isinstance(o, _np.generic):
                    return o.item()
                if isinstance(o, _np.ndarray):
                    return o.tolist()
            except Exception:
                pass
            try:
                import torch as _torch
                if isinstance(o, _torch.Tensor):
                    return o.detach().cpu().tolist()
            except Exception:
                pass
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        self._f.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")
        self._n += 1
        if self._fsync_every and (self._n % self._fsync_every == 0):
            try:
                self._f.flush()
                os.fsync(self._f.fileno())
            except Exception:
                pass

    def close(self) -> None:
        try:
            self._f.flush()
            os.fsync(self._f.fileno())
        except Exception:
            pass
        try:
            self._f.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# RecorderMixin - wraps any Player subclass to record observations
# ---------------------------------------------------------------------------

class RecorderMixin:
    """Mixin that intercepts choose_move to record battle observations.

    Used via multiple inheritance: type("Logged_X", (RecorderMixin, XPlayer), {})
    Records the ACTUAL bot's decisions by calling super().choose_move().
    """

    def _recorder_init(self, fmt_id: str, recorder: JSONLRecorder,
                       turn_cap: int, step_idle_timeout: float,
                       use_v8: bool = False):
        self._fmt_id = fmt_id
        self._recorder = recorder
        self._turn_cap = max(1, int(turn_cap))
        self._step_idle_timeout = float(step_idle_timeout)
        self._episode_buffer: List[Dict[str, Any]] = []
        self._t = 0
        self._role = getattr(self, "_role", "our")
        self._use_v8 = use_v8

    # --- Episode lifecycle ---

    def _episode_init(self) -> None:
        self._episode_buffer = []
        self._t = 0
        self._wins_before = int(getattr(self, "n_won_battles", 0))
        self._losses_before = int(getattr(self, "n_lost_battles", 0))
        self._finished_before = int(getattr(self, "n_finished_battles", 0))
        self._flushed_in_callback = False

    def _ensure_episode_started(self, battle=None) -> None:
        if not hasattr(self, "_episode_buffer") or self._episode_buffer is None:
            self._episode_init()
        if not hasattr(self, "_wins_before"):
            self._wins_before = int(getattr(self, "n_won_battles", 0))
            self._losses_before = int(getattr(self, "n_lost_battles", 0))
            self._finished_before = int(getattr(self, "n_finished_battles", 0))
        if not hasattr(self, "_flushed_in_callback"):
            self._flushed_in_callback = False

    def _episode_id(self, battle) -> str:
        return f"{battle.battle_tag}::{self._role}"

    def _battle_meta(self, battle) -> Dict[str, Any]:
        return {
            "fmt_id": self._fmt_id,
            "p1": str(battle.player_username or ""),
            "opp": str(battle.opponent_username or ""),
            "perspective": self._role,
        }

    def _flush_episode(self, terminal_reason: Optional[str] = None) -> None:
        if not self._episode_buffer:
            return
        last = self._episode_buffer[-1]
        if not last.get("done", False):
            last["done"] = True
            # Try to determine actual result from poke-env counters
            # before falling back to unknown/loss
            wins_after = int(getattr(self, "n_won_battles", 0))
            losses_after = int(getattr(self, "n_lost_battles", 0))
            won_delta = wins_after - getattr(self, "_wins_before", 0)
            lost_delta = losses_after - getattr(self, "_losses_before", 0)
            if won_delta > 0:
                last.setdefault("winner", "our")
                last.setdefault("result", 1)
            elif lost_delta > 0:
                last.setdefault("winner", "opp")
                last.setdefault("result", 0)
            else:
                print("[observer][warn] battle ended for unknown reason — forcing flush", flush=True)
                last.setdefault("winner", "opp_forced_unknown")
                last.setdefault("result", 0)
            if terminal_reason:
                last["terminal_reason"] = terminal_reason
        for row in self._episode_buffer:
            self._recorder.write(row)
        self._episode_buffer.clear()
        self._t = 0

    def _stamp_from_counters(self, last: dict, terminal_reason: str) -> None:
        wins_after = int(getattr(self, "n_won_battles", 0))
        losses_after = int(getattr(self, "n_lost_battles", 0))
        finished_after = int(getattr(self, "n_finished_battles", 0))

        won_delta = wins_after - getattr(self, "_wins_before", 0)
        lost_delta = losses_after - getattr(self, "_losses_before", 0)
        fin_delta = finished_after - getattr(self, "_finished_before", 0)

        last["done"] = True
        if won_delta > 0:
            last["winner"] = "our"
            last["result"] = 1
        elif lost_delta > 0:
            last["winner"] = "opp"
            last["result"] = 0
        elif fin_delta > 0:
            last["winner"] = "tie"
            last["result"] = 0.5
        else:
            print("[observer][warn] battle ended for unknown reason — forcing forfeit", flush=True)
            last.setdefault("winner", "opp_forced_unknown")
            last.setdefault("result", 0)
        last["terminal_reason"] = terminal_reason

    def _finalize_and_flush(self, terminal_reason: str = "finished") -> None:
        try:
            if self._episode_buffer:
                self._stamp_from_counters(self._episode_buffer[-1], terminal_reason)
            self._flush_episode(terminal_reason=terminal_reason)
            self._flushed_in_callback = True
        except Exception:
            print("[observer][err] finalize+flush failed:\n" + traceback.format_exc(), flush=True)

    def _timeout_finalize_and_flush(self) -> None:
        print("[observer][warn] timeout cap reached — forcing forfeit & flush", flush=True)
        try:
            if self._episode_buffer:
                last = self._episode_buffer[-1]
                last["done"] = True
                last["winner"] = "opp_forced_timeout"
                last["result"] = 0
                last["terminal_reason"] = "battle_timeout"
            self._flush_episode(terminal_reason="battle_timeout")
            self._flushed_in_callback = True
        except Exception:
            print("[observer][err] timeout finalize+flush failed:\n" + traceback.format_exc(), flush=True)

    # --- Battle state extraction using poke-env observations API ---

    def _who_moved_first_last_turn(self, battle) -> Optional[Tuple[float, float, float]]:
        """Parse who moved first on the previous turn from poke-env observations.

        Returns (ours_first, opp_first, unknown) as one-hot floats.
        Uses battle.observations[turn].events which stores pre-split protocol messages.
        """
        try:
            prev_turn = battle.turn - 1
            if prev_turn < 1 or prev_turn not in battle.observations:
                return None

            events = battle.observations[prev_turn].events
            role = battle.player_role  # "p1" or "p2"
            if not role:
                return None

            for event in events:
                if len(event) < 3:
                    continue
                etype = event[1]
                if etype in ("move", "switch"):
                    who = event[2].split(":")[0].strip()  # e.g. "p1a" or "p2a"
                    ours_first = float(who.startswith(role))
                    opp_first = float(not who.startswith(role))
                    return (ours_first, opp_first, 0.0)

            return None
        except Exception:
            return None

    @staticmethod
    def _extract_mods_from_order(order) -> Dict[str, int]:
        """Extract modifier flags from a BattleOrder."""
        if not isinstance(order, BattleOrder):
            return {"tera": 0, "dmax": 0, "zmove": 0, "mega": 0}
        return {
            "tera": 1 if getattr(order, "terastallize", False) else 0,
            "dmax": 1 if getattr(order, "dynamax", False) else 0,
            "zmove": 1 if getattr(order, "z_move", False) else 0,
            "mega": 1 if getattr(order, "mega", False) else 0,
        }

    # --- Core recording ---

    def _order_to_index(self, battle, order) -> int:
        """Map a BattleOrder back to an action index (0-3: moves, 4-8: switches)."""
        if not isinstance(order, BattleOrder) or order.order is None:
            return self._first_legal_index(battle)

        action = order.order

        # Move action
        if isinstance(action, Move):
            for i, m in enumerate(battle.available_moves):
                if m is action or m.id == action.id:
                    return i

        # Switch action
        elif isinstance(action, Pokemon):
            # Try identity first (handles duplicate species correctly)
            for j, p in enumerate(battle.available_switches):
                if p is action:
                    return 4 + j
            # Fallback to species match
            for j, p in enumerate(battle.available_switches):
                if p.species == action.species:
                    return 4 + j

        return self._first_legal_index(battle)

    @staticmethod
    def _first_legal_index(battle) -> int:
        # Prefer switches over move-slot-0 when no moves are available
        # (avoids mapping Struggle to action 0 with an all-zero move mask)
        if battle.available_moves:
            return 0
        if battle.available_switches:
            return 4
        return 0  # truly no legal action (shouldn't happen)

    def choose_move(self, battle):
        """Record observation, delegate to parent bot for the actual decision."""
        self._ensure_episode_started(battle)

        # Turn cap: forfeit and flush
        if self._t >= self._turn_cap:
            print("[observer][warn] turn cap reached — forcing forfeit & flush", flush=True)
            if self._episode_buffer:
                last = self._episode_buffer[-1]
                last["done"] = True
                last.setdefault("winner", "opp_forced_turncap")
                last.setdefault("result", 0)
                last["terminal_reason"] = "turn_cap"
            try:
                self.forfeit_battle(battle)
            except Exception:
                pass
            self._flush_episode(terminal_reason="turn_cap")
            # Prevent ghost episode: set buffer to None so any further
            # choose_move calls before forfeit takes effect don't start
            # a new episode with stale counters.
            self._episode_buffer = None
            self._flushed_in_callback = True
            return self.choose_random_move(battle)

        # Get the ACTUAL bot's decision by calling the parent's choose_move
        order = super().choose_move(battle)

        # Reverse-map the order to an action index
        action_idx = self._order_to_index(battle, order)

        # Extract modifiers from the order itself (not from protocol logs)
        mods = self._extract_mods_from_order(order)

        # Build JSONL row
        ep_id = self._episode_id(battle)
        t = int(self._t)
        battle_turn = int(getattr(battle, "turn", t + 1))
        phase = 0 if t <= 3 else (1 if t <= 15 else 2)

        if getattr(self, "_use_v8", False):
            row = self._build_v8_row(battle, ep_id, t, battle_turn, phase,
                                     action_idx, mods)
        else:
            # v7 feature extraction
            mfb = self._who_moved_first_last_turn(battle)
            obs_vec, legal_mask, ctx_extra, move_slots, switch_slots, entity_ids, move_ids, switch_ids = make_obs_mask_and_slots(
                battle, moved_first_bits=mfb
            )
            row = {
                "episode_id": ep_id,
                "t": t,
                "battle_turn": battle_turn,
                "obs": obs_vec,
                "legal": legal_mask,
                "action": int(action_idx),
                "mods": mods,
                "done": False,
                "meta": self._battle_meta(battle),
                "phase": phase,
                "move_slots": move_slots,
                "switch_slots": switch_slots,
                "entity_ids": entity_ids,
                "move_ids": move_ids,
                "switch_ids": switch_ids,
            }
            if ctx_extra is not None:
                row["ctx_extra"] = ctx_extra

        self._episode_buffer.append(row)
        self._t += 1

        return order  # Return the original bot's order

    # --- v8 feature serialization ---

    def _build_v8_row(self, battle, ep_id, t, battle_turn, phase,
                      action_idx, mods) -> dict:
        """Build a v8-format JSONL row from structured features."""
        feat = make_features(battle)

        def _poke_ids(poke_dict):
            ids = poke_dict["ids"]
            return [ids["species"], ids["item"], ids["ability"],
                    ids["move0"], ids["move1"], ids["move2"], ids["move3"]]

        def _poke_banks(poke_dict):
            b = poke_dict["banks"]
            return [b["hp_pct"], b["level"], b["weight"], b["height"],
                    b["stat_hp"], b["stat_atk"], b["stat_def"],
                    b["stat_spa"], b["stat_spd"], b["stat_spe"]]

        def _poke_move_cont(poke_dict):
            """Extract 4x23 compact move encoding from pokemon continuous features."""
            from features import extract_move_cont
            return extract_move_cont(poke_dict["continuous"])

        def _move_slot(m):
            if m is None:
                from features import MOVE_SLOT_CONT_DIM
                return {"id": 0, "bp": 0, "acc": 0, "pp": 0, "prio": 6, "cont": [0.0] * MOVE_SLOT_CONT_DIM}
            return {"id": m["move_id"], "bp": m["bp_int"], "acc": m["acc_int"],
                    "pp": m["pp_int"], "prio": m["priority_int"], "cont": m["continuous"]}

        def _switch_slot(s):
            if s is None:
                from features import SWITCH_SLOT_CONT_DIM
                return {"id": 0, "cont": [0.0] * SWITCH_SLOT_CONT_DIM}
            return {"id": s["species_id"], "cont": s["continuous"]}

        move_slots = [_move_slot(m) for m in feat["active_moves"]]
        switch_slots = [_switch_slot(s) for s in feat["switch_slots"]]

        return {
            "v8": True,
            "episode_id": ep_id,
            "t": t,
            "battle_turn": battle_turn,
            "action": int(action_idx),
            "mods": mods,
            "done": False,
            "meta": self._battle_meta(battle),
            "phase": phase,
            "legal": feat["legal_mask"].tolist(),
            # Pokemon (6 per side)
            "our_poke_ids": [_poke_ids(p) for p in feat["our_pokemon"]],
            "our_poke_banks": [_poke_banks(p) for p in feat["our_pokemon"]],
            "our_poke_cont": [p["continuous"] for p in feat["our_pokemon"]],
            "our_poke_mcont": [_poke_move_cont(p) for p in feat["our_pokemon"]],
            "opp_poke_ids": [_poke_ids(p) for p in feat["opp_pokemon"]],
            "opp_poke_banks": [_poke_banks(p) for p in feat["opp_pokemon"]],
            "opp_poke_cont": [p["continuous"] for p in feat["opp_pokemon"]],
            "opp_poke_mcont": [_poke_move_cont(p) for p in feat["opp_pokemon"]],
            # Field
            "field_banks": [feat["field"]["banks"]["turn"],
                            feat["field"]["banks"]["weather_dur"],
                            feat["field"]["banks"]["terrain_dur"],
                            feat["field"]["banks"]["tr_dur"]],
            "field_cont": feat["field"]["continuous"],
            # Transition
            "trans_ids": [feat["transition"]["ids"]["our_action"],
                          feat["transition"]["ids"]["opp_action"]],
            "trans_cont": feat["transition"]["continuous"],
            # Active move slots
            "move_ids": [ms["id"] for ms in move_slots],
            "move_banks": [[ms["bp"], ms["acc"], ms["pp"], ms["prio"]] for ms in move_slots],
            "move_cont": [ms["cont"] for ms in move_slots],
            # Switch slots
            "switch_ids": [ss["id"] for ss in switch_slots],
            "switch_cont": [ss["cont"] for ss in switch_slots],
        }

    # --- poke-env callback (single correct one) ---

    def _battle_finished_callback(self, battle) -> None:
        self._finalize_and_flush("finished")
        super()._battle_finished_callback(battle)


# ---------------------------------------------------------------------------
# Bot registry
# ---------------------------------------------------------------------------

LOGGED = {
    "MaxDamage":         type("LoggedMaxBP",       (RecorderMixin, MaxBasePowerPlayer), {}),
    "MaxBasePower":      type("LoggedMaxBasePower", (RecorderMixin, MaxBasePowerPlayer), {}),
    "SimpleHeuristics":  type("LoggedSimple",       (RecorderMixin, SimpleHeuristicsPlayer), {}),
    "GreedySE":          type("LoggedGreedySE",     (RecorderMixin, GreedySEPlayer), {}),
    "HazardSense":       type("LoggedHazard",       (RecorderMixin, HazardSensePlayer), {}),
    "SwitchAwareEscape": type("LoggedSwitchAware",  (RecorderMixin, SwitchAwareEscapePlayer), {}),
    "SetupThenSweep":    type("LoggedSetupSweep",   (RecorderMixin, SetupThenSweepPlayer), {}),
    "Random":            type("LoggedRandom",       (RecorderMixin, PokeRandomPlayer), {}),
    "SmartDamage":       type("LoggedSmartDmg",     (RecorderMixin, SmartDamagePlayer), {}),
    "Tactical":          type("LoggedTactical",     (RecorderMixin, TacticalPlayer), {}),
    "Strategic":         type("LoggedStrategic",    (RecorderMixin, StrategicPlayer), {}),
}

BOT_REGISTRY = {
    "MaxDamage":         MaxBasePowerPlayer,
    "MaxBasePower":      MaxBasePowerPlayer,
    "SimpleHeuristics":  SimpleHeuristicsPlayer,
    "GreedySE":          GreedySEPlayer,
    "HazardSense":       HazardSensePlayer,
    "SwitchAwareEscape": SwitchAwareEscapePlayer,
    "SetupThenSweep":    SetupThenSweepPlayer,
    "Random":            PokeRandomPlayer,
    "SmartDamage":       SmartDamagePlayer,
    "Tactical":          TacticalPlayer,
    "Strategic":         StrategicPlayer,
}


# ---------------------------------------------------------------------------
# Model player support for self-play
# ---------------------------------------------------------------------------

def _make_model_logged_class(checkpoint_path: str, device: str = "cpu"):
    """Create a RecorderMixin + BCPolicyPlayer class for a model checkpoint.

    Returns (LoggedClass, UnloggedClass) where LoggedClass has RecorderMixin
    and UnloggedClass is plain BCPolicyPlayer. Both bind the checkpoint path
    and device so they can be instantiated with standard player kwargs.
    """
    class ModelPlayer(BCPolicyPlayer):
        """BCPolicyPlayer bound to a specific checkpoint."""
        def __init__(self, **kwargs):
            kwargs.pop("replay_folder", None)
            super().__init__(checkpoint_path=checkpoint_path, device=device, **kwargs)

    class LoggedModelPlayer(RecorderMixin, BCPolicyPlayer):
        """RecorderMixin + BCPolicyPlayer bound to a specific checkpoint."""
        def __init__(self, **kwargs):
            kwargs.pop("replay_folder", None)
            super().__init__(checkpoint_path=checkpoint_path, device=device, **kwargs)

    tag = Path(checkpoint_path).stem
    ModelPlayer.__name__ = f"Model_{tag}"
    LoggedModelPlayer.__name__ = f"LoggedModel_{tag}"
    return LoggedModelPlayer, ModelPlayer


def register_model_bot(name: str, checkpoint_path: str, device: str = "cpu"):
    """Register a model checkpoint as a bot in LOGGED/BOT_REGISTRY."""
    logged, unlogged = _make_model_logged_class(checkpoint_path, device)
    LOGGED[name] = logged
    BOT_REGISTRY[name] = unlogged


def pick_logged(name: str):
    name = name.strip()
    if name not in LOGGED:
        raise SystemExit(f"Unknown bot '{name}'. Valid: {', '.join(sorted(LOGGED))}")
    return LOGGED[name]


def pick_unlogged(name: str):
    name = name.strip()
    if name == "MaxDamage":
        name = "MaxBasePower"
    if name not in BOT_REGISTRY:
        raise SystemExit(f"Unknown bot '{name}'. Valid: {', '.join(sorted(BOT_REGISTRY))}")
    return BOT_REGISTRY[name]


def _canonical_display(name: str) -> str:
    return "MaxDamage" if name == "MaxBasePower" else name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate JSONL observation datasets from self-play")
    p.add_argument("--format", default="gen9ou")
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--bots", type=str, default="SimpleHeuristics,MaxDamage")
    p.add_argument("--all-mode", choices=["all_pairings", "all_vs_anchor"], default="all_pairings")
    p.add_argument("--anchor-bot", type=str, default="MaxDamage")
    p.add_argument("--max-concurrent", type=int, default=8)
    p.add_argument("--turn-cap", type=int, default=int(os.getenv("TURN_CAP", "300")))
    p.add_argument("--battle-timeout", type=int, default=int(os.getenv("BATTLE_TIMEOUT_S", "600")))
    p.add_argument("--step-idle-timeout", type=float, default=float(os.getenv("STEP_IDLE_TIMEOUT_S", "120")))
    p.add_argument("--server", type=str, default=None)
    p.add_argument("--log-both", action="store_true")
    p.add_argument("--parallel-pairings", type=int, default=3,
                   help="How many bot pairings to run concurrently (for --bots all)")
    p.add_argument("--batch-per-worker", type=int, default=5,
                   help="Games per player object (reuses ws connection within batch)")
    p.add_argument("--direct", action="store_true",
                   help="Use direct BattleStream transport (no websockets/Docker)")
    p.add_argument("--model", type=str, action="append", default=[],
                   help="Register model checkpoint as bot. Format: name:path "
                        "(e.g. --model iql_ep10:data/models/iql/.../epoch_010_policy.pt). "
                        "Can be repeated. Use name in --bots.")
    p.add_argument("--model-device", type=str, default="cpu",
                   help="Device for model inference during self-play (default: cpu)")
    p.add_argument("--v8", action="store_true",
                   help="Use v8 structured features (entity tokenization) instead of v7 flat vector")
    return p.parse_args()


def resolve_server(args) -> ServerConfiguration:
    if args.server:
        ws = args.server.rstrip("/")
        # If the full websocket path is given, use it as-is
        # Otherwise append /showdown/websocket
        if not ws.endswith("/showdown/websocket"):
            ws = f"{ws}/showdown/websocket"
        http = ws.replace("wss://", "https://").replace("ws://", "http://")
        http = http[:http.rfind("/showdown/websocket")] + "/action.php?"
        return ServerConfiguration(ws, http)
    return DEFAULT_SERVER


def needs_team(fmt_id: str) -> bool:
    return "random" not in fmt_id


# ---------------------------------------------------------------------------
# Async battle execution
# ---------------------------------------------------------------------------

_SHOULD_STOP = False


def _install_signal_handlers(rec: JSONLRecorder):
    def _handler(signum, frame):
        global _SHOULD_STOP
        _SHOULD_STOP = True
        try:
            rec.close()
        except Exception:
            pass
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


async def play_batch(OurCls, OppBase, OppLogged,
                     fmt_id: str, server: ServerConfiguration,
                     recorder: JSONLRecorder, args,
                     n_games: int = 5) -> int:
    """Play a batch of games reusing the same player objects.
    Returns number of games completed. Team is randomized per-battle via teambuilder."""
    use_tb = needs_team(fmt_id)
    use_direct = getattr(args, "direct", False)

    # Common kwargs; when --direct, skip websocket listener
    extra_kwargs = {}
    if use_direct:
        extra_kwargs["start_listening"] = False
    else:
        extra_kwargs["server_configuration"] = server

    our = OurCls(
        account_configuration=AccountConfiguration.generate(OurCls.__name__, rand=True),
        battle_format=fmt_id,
        team=(random_pool_teambuilder() if use_tb else None),
        max_concurrent_battles=1,
        save_replays=False,
        **extra_kwargs,
    )
    setattr(our, "_role", "our")
    if hasattr(our, "_recorder_init"):
        our._recorder_init(fmt_id=fmt_id, recorder=recorder,
                           turn_cap=args.turn_cap, step_idle_timeout=args.step_idle_timeout,
                           use_v8=getattr(args, "v8", False))

    if getattr(args, "log_both", False):
        opp = OppLogged(
            account_configuration=AccountConfiguration.generate(OppLogged.__name__, rand=True),
            battle_format=fmt_id,
            team=(random_pool_teambuilder() if use_tb else None),
            max_concurrent_battles=1,
            save_replays=False,
            **extra_kwargs,
        )
        setattr(opp, "_role", "opp")
        if hasattr(opp, "_recorder_init"):
            opp._recorder_init(fmt_id=fmt_id, recorder=recorder,
                               turn_cap=args.turn_cap, step_idle_timeout=args.step_idle_timeout,
                               use_v8=getattr(args, "v8", False))
    else:
        opp = OppBase(
            account_configuration=AccountConfiguration.generate(OppBase.__name__, rand=True),
            battle_format=fmt_id,
            team=(random_pool_teambuilder() if use_tb else None),
            max_concurrent_battles=1,
            save_replays=False,
            **extra_kwargs,
        )

    # Patch players for direct transport if --direct
    if use_direct:
        from direct_player import patch_to_direct, direct_battle_against
        patch_to_direct(our)
        patch_to_direct(opp)

    completed = 0
    try:
        for g in range(n_games):
            our._ensure_episode_started()
            flushed = False
            try:
                if use_direct:
                    from poke_env.concurrency import POKE_LOOP
                    fut = asyncio.run_coroutine_threadsafe(
                        direct_battle_against(our, opp, n_battles=1),
                        POKE_LOOP,
                    )
                    await asyncio.wait_for(
                        asyncio.wrap_future(fut),
                        timeout=int(args.battle_timeout),
                    )
                else:
                    await asyncio.wait_for(
                        our.battle_against(opp, n_battles=1),
                        timeout=int(args.battle_timeout)
                    )
            except asyncio.TimeoutError:
                our._timeout_finalize_and_flush()
                flushed = True
                break  # Timeout likely means connection issues, stop this batch
            else:
                if getattr(our, "_flushed_in_callback", False):
                    pass
                elif our._episode_buffer:
                    our._finalize_and_flush("finished")
                flushed = True
            finally:
                if not flushed and hasattr(our, "_flush_episode"):
                    try:
                        our._flush_episode(terminal_reason="finished")
                    except Exception:
                        pass

            # Ensure opponent is flushed too when --log-both
            if hasattr(opp, "_episode_buffer"):
                if opp._episode_buffer and not getattr(opp, "_flushed_in_callback", False):
                    try:
                        opp._finalize_and_flush("finished")
                    except Exception:
                        pass
                # ALWAYS reset opp state for next game, even if callback already flushed.
                # _flush_episode uses .clear() which leaves [] not None,
                # so we must explicitly set None to trigger _episode_init() next game.
                opp._flushed_in_callback = False
                opp._episode_buffer = None
                opp._t = 0

            completed += 1
            # Reset for next game (keep ws connection alive)
            # CRITICAL: set to None (not []) so _ensure_episode_started() calls
            # _episode_init() which resets _wins_before/_losses_before counters.
            # Using [] caused counters to persist across games in a batch,
            # making ALL games after a first win also report result=1 (bug).
            our._flushed_in_callback = False
            our._episode_buffer = None
            our._t = 0
    finally:
        # Clean up player objects — close websockets to free Showdown usernames
        if use_direct:
            from direct_player import shutdown_worker
            # Worker cleanup is handled globally, not per-batch
        else:
            for player in (our, opp):
                try:
                    if hasattr(player, "_stop_listening"):
                        await player._stop_listening()
                except Exception:
                    pass
                # Cancel the listening coroutine on POKE_LOOP to prevent
                # zombie listeners accumulating. PSClient.__init__ creates
                # _listening_coroutine via run_coroutine_threadsafe but
                # _stop_listening only closes the websocket — the coroutine
                # future itself is never cancelled, leaking memory over
                # thousands of batches.
                try:
                    ps = getattr(player, "ps_client", None) or getattr(player, "_ps_client", None)
                    if ps and hasattr(ps, "_listening_coroutine"):
                        ps._listening_coroutine.cancel()
                except Exception:
                    pass
        for player in (our, opp):
            try:
                if hasattr(player, "reset_battles"):
                    player.reset_battles()
            except Exception:
                pass
        del our, opp
        gc.collect()

    return completed


async def play_one(OurCls, OppBase, OppLogged,
                   fmt_id: str, server: ServerConfiguration,
                   recorder: JSONLRecorder, args) -> None:
    """Play a single game (backward compat wrapper)."""
    await play_batch(OurCls, OppBase, OppLogged,
                     fmt_id, server, recorder, args, n_games=1)


async def _run_for_pair(args, fmt_id: str, server: ServerConfiguration,
                        our_name: str, opp_name: str) -> None:
    OurCls = pick_logged(our_name)
    OppBase = pick_unlogged(opp_name)
    OppLogged = pick_logged(opp_name)

    ts = time.strftime("%Y%m%d_%H%M%S")
    obs_subdir = "obs_v8" if getattr(args, "v8", False) else "obs"
    out_base = getattr(args, "out", None) or f"data/datasets/{obs_subdir}"
    out = f"{out_base}/obs_{fmt_id}_{_canonical_display(our_name)}-vs-{_canonical_display(opp_name)}_{ts}.jsonl"
    recorder = JSONLRecorder(out, fsync_every=512)
    _install_signal_handlers(recorder)

    total = int(args.games)
    concurrency = max(1, int(args.max_concurrent))
    batch_per_worker = max(1, getattr(args, "batch_per_worker", 5))

    print(f"[observer] pairing: {_canonical_display(our_name)} vs {_canonical_display(opp_name)} "
          f"-> {out} (games={total}, concurrent={concurrency}, batch={batch_per_worker})", flush=True)

    q: asyncio.Queue[int] = asyncio.Queue()
    for i in range(total):
        q.put_nowait(i)

    progress_last = time.time()

    async def worker(wid: int):
        nonlocal progress_last
        while not q.empty() and not _SHOULD_STOP:
            # Grab a batch of games from the queue
            batch_count = 0
            for _ in range(batch_per_worker):
                try:
                    _ = await asyncio.wait_for(q.get(), timeout=0.1)
                    batch_count += 1
                except asyncio.TimeoutError:
                    break
            if batch_count == 0:
                break
            try:
                completed = await play_batch(
                    OurCls, OppBase, OppLogged, fmt_id, server, recorder, args,
                    n_games=batch_count,
                )
            finally:
                for _ in range(batch_count):
                    q.task_done()

            now = time.time()
            if wid == 0 and (now - progress_last) > 10:
                progress_last = now
                print(f"[observer] progress: remaining={q.qsize()}/{total}", flush=True)

    try:
        workers = [asyncio.create_task(worker(w)) for w in range(concurrency)]
        await q.join()
        for w in workers:
            w.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*workers)
    finally:
        try:
            recorder.close()
        except Exception:
            pass
        print(f"[observer] finished. wrote={recorder._n} rows to {out}", flush=True)


async def run(args):
    # Register model checkpoints as bots (--model name:path)
    for spec in getattr(args, "model", []):
        if ":" not in spec:
            raise SystemExit(f"--model must be name:path, got '{spec}'")
        name, path = spec.split(":", 1)
        if not os.path.isfile(path):
            raise SystemExit(f"Model checkpoint not found: {path}")
        register_model_bot(name, path, device=getattr(args, "model_device", "cpu"))
        print(f"[observer] Registered model bot '{name}' from {path}", flush=True)

    fmt_id = args.format
    server = resolve_server(args)
    bots_arg = args.bots.strip().lower()

    if bots_arg == "all":
        names = sorted(set(LOGGED.keys()))
        if args.all_mode == "all_vs_anchor":
            anchor = args.anchor_bot
            if anchor not in LOGGED:
                raise SystemExit(f"--anchor-bot '{anchor}' unknown. Valid: {', '.join(sorted(LOGGED.keys()))}")
            pairings = [(anchor, other) for other in names]
        else:
            pairings = [(a, b) for a in names for b in names]

        # Run pairings in parallel batches (--parallel-pairings at a time)
        batch_size = getattr(args, "parallel_pairings", 3)
        for i in range(0, len(pairings), batch_size):
            batch = pairings[i:i+batch_size]
            await asyncio.gather(
                *[_run_for_pair(args, fmt_id, server, a, b) for a, b in batch]
            )
        return

    bots = [b for b in (x.strip() for x in args.bots.split(",")) if b]
    if len(bots) == 1:
        bots = [bots[0], bots[0]]
    if len(bots) != 2:
        raise SystemExit("--bots must resolve to exactly two names (or one for mirror), or use --bots all")

    await _run_for_pair(args, fmt_id, server, bots[0], bots[1])


def main():
    args = parse_args()
    random.seed(1337 + int(time.time()) % 100000)
    if args.bots.strip().lower() != "all":
        for b in [x.strip() for x in args.bots.split(",") if x.strip()]:
            if b not in LOGGED:
                raise SystemExit(f"Unknown bot '{b}'. Valid: {', '.join(sorted(LOGGED.keys()))}  (or use --bots all)")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
    if "--direct" in sys.argv:
        os._exit(0)
