#!/usr/bin/env python3
"""CIS-routed BattleAgentTransformer for shared-inference Elo eval.

Drop-in replacement for BattleAgentTransformer that does NOT load a model
locally. Instead routes every inference call through a shared CIS server
(per project_cis_elo_ladder_design memo, S68 Jun 1).

Why: enables N-player ladder eval to share GPU memory + batch inference
across all matches concurrently. Current eval_elo_ladder.py LRU-caches
N model instances which serializes per-pair model loads + holds N × ~200MB
GPU. CIS path pre-loads all into one server, dispatches to slots, batches
across workers.

Usage:
    # Caller spawns CIS server first via _cis_main_multi (mp_centralized_collect)
    # then passes pipe ends + slot_id to each player it creates:
    player_a = BattleAgentTransformerCIS(
        cis_req_writer=req_writer,
        cis_resp_reader=resp_reader,
        slot_id=0,                # player A's slot
        cfg=transformer_cfg,
        ...standard Player kwargs...
    )

Protocol matches _cis_main_multi (mp_centralized_collect.py:445-451):
    Out: {"cmd": "infer", "slot": slot_id, "batch": dict,
          "history": tensor|None, "req_id": int}
    In:  per-request response dict with action_logits / summary
"""
from typing import Optional, Dict
from pathlib import Path

import torch
import torch.nn.functional as F
from poke_env.player import Player

# Reuse parent's helpers — fp16 autocast, features, action mapping.
from battle_agent_transformer import (
    BattleAgentTransformer,
    autocast_ctx,
    make_features,
    DEFAULT_LOOKUP_PATH,
)
from model_transformer import TransformerConfig


class BattleAgentTransformerCIS(BattleAgentTransformer):
    """CIS-routed inference Player. No local model load."""

    def __init__(self,
                 cis_req_writer,
                 cis_resp_reader,
                 cis_pipe_lock,
                 slot_id: int,
                 cfg: TransformerConfig,
                 device: str = "cpu",
                 temperature: float = 0.0,
                 fp16: bool = False,
                 turn_cap: int = 300,
                 checkpoint_path: str = "",  # for logging only
                 **kwargs):
        """CIS-routed Player. Does NOT load a model — sends inference requests
        to the CIS server via pipes.

        Args:
            cis_req_writer:   mp.Pipe writer end to CIS server (this worker's req pipe).
            cis_resp_reader:  mp.Pipe reader end from CIS server (this worker's resp pipe).
            cis_pipe_lock:    threading.Lock to serialize pipe I/O if multiple
                              Players share one pipe pair in the same process.
                              Each player's choose_move acquires the lock for
                              its req+recv round-trip, releases after. Prevents
                              interleaved response reads on shared pipes.
            slot_id:          index into the CIS server's model_slots that this
                              Player should dispatch to (= this player's
                              checkpoint slot).
            cfg:              TransformerConfig — only used for temporal_context
                              + n_moves shape. NO model loaded.
            device:           CPU/CUDA — kept for tensor dispatch on local-side
                              tensors only (history); model lives in CIS.
            temperature:      sampling temperature (0 = argmax).
            fp16:             autocast for any local tensor ops (history accum).
            turn_cap:         forfeit if battle exceeds this many turns.
            checkpoint_path:  for logging only.
            **kwargs:         passed to Player base (battle_format, server_configuration,
                              max_concurrent_battles, team, account_configuration, etc.).
        """
        # Bypass BattleAgentTransformer's __init__ (which loads model).
        # Call Player.__init__ directly with kwargs.
        Player.__init__(self, **kwargs)
        # FORCE local device to CPU — CIS handles GPU work in its subprocess.
        # We need CPU tensors over the pipe (cuda tensors fail numpy conversion
        # at receive end). Local-side ops (history accum) are tiny — CPU is fine.
        # The `device` arg is retained for API compat but used only for logging.
        self.device = torch.device('cpu')
        self._cis_device_label = device  # log-only
        self.temperature = temperature
        self.fp16 = False  # CPU-side ops never need fp16; CIS handles its own
        self.turn_cap = turn_cap
        self._turn_counts: Dict[str, int] = {}

        self._cis_req = cis_req_writer
        self._cis_resp = cis_resp_reader
        self._cis_lock = cis_pipe_lock
        self.slot_id = slot_id
        self.cfg = cfg

        # Per-battle temporal history: battle_tag -> (B=1, T, d_temporal) tensor.
        # IMPORTANT: held locally; CIS doesn't track per-battle state.
        self._history: Dict[str, torch.Tensor] = {}

        # Local request ID counter — for sanity matching req↔resp under shared pipes.
        self._req_id_counter = 0

        # NO self.model. NO load_state_dict. NO move_flag_lookup.
        # All inference goes through CIS slot.
        self._ckpt_label = checkpoint_path or f"cis_slot_{slot_id}"
        print(f"[BattleAgentTransformerCIS] slot {slot_id} bound (no local model). "
              f"ckpt_label={self._ckpt_label}")

    def _build_turn_batch(self, feat: dict) -> dict:
        """Build batch on CPU (not self.device).

        CIS expects CPU tensors over the pipe — it handles its own device
        placement internally. Sending CUDA tensors causes pickle/numpy
        conversion errors at CIS receive side.
        """
        from features import build_turn_batch
        return build_turn_batch(feat, device='cpu')

    def _cis_infer(self, batch: dict, history) -> dict:
        """Send inference request to CIS server, return response dict with
        torch tensors keyed by action_logits / summary / value / v_logits.

        Blocks on pipe round-trip. Lock-guarded so multiple Players sharing
        a pipe pair don't interleave reads.

        CIS protocol (per _cis_main_multi line 778):
            Server sends {"status": "ok", "out": <numpy_dict>, "req_id": int}
            numpy_dict has action_logits / value / v_logits / summary
            (slice indices [0..batch_size]; for B=1 inference, slice is [0..1])
        """
        self._req_id_counter += 1
        req_id = self._req_id_counter
        # Convert torch tensors to numpy — CIS does its own numpy→torch→device.
        # Pattern matches training workers (mp_centralized_collect.py:668).
        np_batch = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()}
        np_history = history.cpu().numpy() if isinstance(history, torch.Tensor) else None
        msg = {
            "cmd": "infer",
            "slot": self.slot_id,
            "batch": np_batch,
            "history": np_history,  # may be None
            "req_id": req_id,
        }
        with self._cis_lock:
            self._cis_req.send(msg)
            resp = self._cis_resp.recv()
        if not isinstance(resp, dict):
            raise RuntimeError(f"CIS responded with non-dict: {type(resp)}")
        # CIS server includes req_id in response for matching.
        if "req_id" in resp and resp["req_id"] != req_id:
            raise RuntimeError(
                f"CIS resp req_id={resp['req_id']} != sent req_id={req_id} — "
                f"pipe ordering broken (lock not held?)"
            )
        if resp.get("status") != "ok":
            raise RuntimeError(
                f"CIS error: status={resp.get('status')!r} "
                f"msg={resp.get('exc_msg', '')[:200]}"
            )
        np_out = resp["out"]
        # Convert numpy arrays to torch tensors on local device.
        # Each key is shape (B=1, ...) — we keep the batch dim for caller.
        out = {}
        for k, v in np_out.items():
            t = torch.from_numpy(v).to(self.device)
            out[k] = t
        return out

    def choose_move(self, battle):
        """Identical to BattleAgentTransformer.choose_move except inference
        routes through CIS instead of self.model.
        """
        btag = battle.battle_tag

        # Turn cap — forfeit if battle runs too long.
        self._turn_counts[btag] = self._turn_counts.get(btag, 0) + 1
        if self._turn_counts[btag] >= self.turn_cap:
            try:
                self.forfeit_battle(battle)
            except Exception:
                pass
            return self.choose_random_move(battle)

        # Extract features (same as parent).
        feat = make_features(battle)
        batch = self._build_turn_batch(feat)
        history = self._get_history(btag)

        # Inference via CIS instead of local model.
        out = self._cis_infer(batch, history)

        # Update history (always float32 — same logic as parent).
        summary = out["summary"].float().unsqueeze(1)  # (1, 1, d_temporal)
        if history is None:
            self._history[btag] = summary
        else:
            ctx = self.cfg.temporal_context
            if history.shape[1] >= ctx:
                history = history[:, -(ctx - 1):]
            self._history[btag] = torch.cat([history, summary], dim=1)

        # Select action — same as parent.
        logits = out["action_logits"][0]  # (n_actions,)
        if self.temperature > 0:
            probs = F.softmax(logits / self.temperature, dim=-1)
            action_idx = torch.multinomial(probs, 1).item()
        else:
            action_idx = logits.argmax().item()

        return self._action_to_order(battle, action_idx)
