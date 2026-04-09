#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from poke_env.player import Player

from features import featurize, make_obs_and_masks, make_obs_mask_and_slots, step_type_from_abs_t
from policy_heads import BattlePolicy, PolicyConfig, ModifierSpec


# ---------- inference helpers ----------

def _infer_use_lstm_from_state_dict(sd: dict) -> bool:
    return any("lstm.weight_ih_l0" in k for k in sd.keys())

def _infer_modifiers_from_state_dict(sd: dict) -> list[str]:
    names = set()
    for k in sd.keys():
        if k.startswith("mod_heads.") and k.count(".") >= 2:
            names.add(k.split(".")[1])
    return sorted(names)

def _infer_core_dim_from_sd(sd: dict) -> Optional[int]:
    # read head input width (core width)
    for head_name in ("action_head.weight", "value_head.weight", "aux_move_first.weight"):
        W = sd.get(head_name, None)
        if isinstance(W, torch.Tensor) and W.ndim == 2:
            return int(W.shape[1])
    for k, W in sd.items():
        if k.startswith("mod_heads.") and k.endswith(".weight") and isinstance(W, torch.Tensor) and W.ndim == 2:
            return int(W.shape[1])
    return None

def _infer_obs_in_features_from_sd(sd: dict) -> Optional[int]:
    # first Linear in obs encoder
    for k, W in sd.items():
        if k.startswith("obs_encoder.") and k.endswith(".weight") and isinstance(W, torch.Tensor) and W.ndim == 2:
            return int(W.shape[1])
    return None

def _prune_state_dict_to_model(sd: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Keep only keys that exist in model with identical shapes."""
    keep = {}
    msd = model.state_dict()
    for k, v in sd.items():
        if k in msd and isinstance(v, torch.Tensor) and isinstance(msd[k], torch.Tensor):
            if tuple(v.shape) == tuple(msd[k].shape):
                keep[k] = v
    return keep

def _extract_action_logits_from_forward(out):
    """
    Accept either:
      - a flat tensor of shape [B, A] (already-ready action logits), or
      - a dict of heads and compose into [B, A].

    We handle common keys:
      - 'logits' or 'joint'  -> already joint
      - ('move_logits', 'switch_logits', optional specials like 'tera_logit','mega_logit','zmove_logit','dmax_logit')
    Returns a tensor [B, A].
    Raises ValueError if we cannot figure it out.
    """
    import torch

    # Already a tensor?
    if torch.is_tensor(out):
        return out

    if not isinstance(out, dict):
        raise ValueError(f"Unsupported policy forward output type: {type(out)}")

    # Common single-key cases
    for k in ("logits", "joint", "action_logits"):
        if k in out and torch.is_tensor(out[k]):
            return out[k]

    # Multi-head: moves + (optional specials) + switches
    parts = []
    move = out.get("move_logits")
    switch = out.get("switch_logits")

    # Optional specials (0/1-dim heads) – include only if present
    specials = []
    for k in ("tera_logit", "mega_logit", "zmove_logit", "dmax_logit", "special_logits"):
        v = out.get(k)
        if v is not None:
            specials.append(v)

    # Sanity and concatenation order must match your action encoding
    # Default ordering here is: [moves][specials][switches]
    # (This matches the common mask layout in many PS/Poke-Env encoders:
    #  moves first, then optional one-hot specials, then switches.)
    if move is None and switch is None and not specials:
        raise ValueError("Could not find known logits in dict output. Keys: " + str(list(out.keys())))

    for p in (move, *(specials or []), switch):
        if p is None:
            continue
        if p.ndim == 1:
            p = p.unsqueeze(0)
        parts.append(p)

    if not parts:
        raise ValueError("No logits parts available to concatenate.")

    logits = torch.cat(parts, dim=-1)

    return logits

# ---------- poke-env order wrapper ----------

# _StringOrder removed: BattleOrder.message already produces the /choose string


# ---------- main player ----------

class BCPolicyPlayer(Player):
    """
    Run a BC checkpoint in poke-env with strong shape-compatibility:

    - Forces core width (MLP/LSTM hidden) to the checkpoint's width when available.
    - Prunes checkpoint tensors to matching shapes before load_state_dict.
    - step_type index + ctx_extra are passed as separate tensors; the model embeds/projects internally.
    - Pads/truncates runtime obs to the exact width expected by the model.
    """

    def __init__(
        self,
        checkpoint_path: str,
        obs_dim: int | None = None,
        device: str = "cuda",
        # optional overrides
        use_lstm: bool | None = None,
        lstm_hidden: int | None = None,
        mlp_hidden: int | None = None,
        modifiers=None,
        # NEW: keep inference in lockstep with training
        ctx_extra_dim: int | None = None,
        step_type_bins: int | None = None,
        # player kwargs (battle_format, server_configuration, team, etc.)
        **player_kwargs,
    ):
        player_kwargs = dict(player_kwargs)
        player_kwargs.pop("replay_folder", None)  # tolerate older poke-env
        super().__init__(**player_kwargs)

        self.device = torch.device(
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ---- Load checkpoint & infer shapes
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        sd = ckpt.get("model", {})
        if not sd:
            raise ValueError(f"Checkpoint {checkpoint_path} missing 'model' state_dict")

        pcfg = ckpt.get("policy_cfg")
        ckpt_obs_dim = ckpt.get("obs_dim", None)
        inferred_obs_in = _infer_obs_in_features_from_sd(sd)
        inferred_core = _infer_core_dim_from_sd(sd)
        inferred_use_lstm = _infer_use_lstm_from_state_dict(sd)

        # Determine the obs width the model should expect
        target_obs_dim = (
            obs_dim if obs_dim is not None else
            (pcfg.get("obs_dim") if isinstance(pcfg, dict) and "obs_dim" in pcfg else
             (ckpt_obs_dim if ckpt_obs_dim is not None else inferred_obs_in))
        )
        if target_obs_dim is None:
            raise ValueError("obs_dim not found; pass --obs-dim or save obs_dim during training")

        # ---- Build PolicyConfig that MATCHES the checkpoint (strict by default)
        pcfg = ckpt.get("policy_cfg")
        ckpt_obs_dim = ckpt.get("obs_dim", None)

        if not isinstance(pcfg, dict):
            # Fallback to your existing inference if really no cfg was saved:
            inferred_obs_in = _infer_obs_in_features_from_sd(sd)
            inferred_core   = _infer_core_dim_from_sd(sd)
            inferred_use_lstm = _infer_use_lstm_from_state_dict(sd)
            core = inferred_core if inferred_core is not None else (lstm_hidden or mlp_hidden or 256)
            cfg = PolicyConfig(
                obs_dim=int(obs_dim if obs_dim is not None else (ckpt_obs_dim if ckpt_obs_dim is not None else inferred_obs_in)),
                action_dim=9,
                use_lstm=bool(use_lstm if use_lstm is not None else inferred_use_lstm),
                lstm_hidden=int(lstm_hidden if lstm_hidden is not None else core),
                mlp_hidden=int(mlp_hidden if mlp_hidden is not None else core),
                modifiers=[ModifierSpec(name=m) for m in (_infer_modifiers_from_state_dict(sd) or [])] or None,
                # keep defaults for ctx / step bins if truly missing
                step_type_bins=int(step_type_bins if step_type_bins is not None else 0),
                ctx_extra_dim=int(ctx_extra_dim if ctx_extra_dim is not None else 0),
            )
        else:
            # Respect the saved training cfg (preferred path)
            if obs_dim is not None:
                pcfg["obs_dim"] = int(obs_dim)  # only override if the caller insists
            # Allow selective runtime toggles if you passed them on purpose
            if use_lstm is not None:    pcfg["use_lstm"]    = bool(use_lstm)
            if lstm_hidden is not None: pcfg["lstm_hidden"] = int(lstm_hidden)
            if mlp_hidden is not None:  pcfg["mlp_hidden"]  = int(mlp_hidden)
            # Only override ctx/step if caller explicitly passed non-None values
            if step_type_bins is not None:
                pcfg["step_type_bins"] = int(step_type_bins)
            if ctx_extra_dim is not None:
                pcfg["ctx_extra_dim"]  = int(ctx_extra_dim)

            # Recreate modifiers from checkpoint unless caller passed a custom list
            if modifiers is not None:
                pcfg["modifiers"] = [ModifierSpec(name=m) for m in modifiers] if modifiers else None
            elif "modifiers" not in pcfg:
                mods = _infer_modifiers_from_state_dict(sd)
                pcfg["modifiers"] = [ModifierSpec(name=m) for m in mods] if mods else None

            cfg = PolicyConfig(**pcfg)

        # Instantiate and load
        self.model = BattlePolicy(cfg).to(self.device)
        pruned = _prune_state_dict_to_model(sd, self.model)
        self.model.load_state_dict(pruned, strict=False)
        self.model.eval()

        # meta
        self.obs_dim_required = int(cfg.obs_dim)
        self.ckpt_obs_dim = int(ckpt_obs_dim) if ckpt_obs_dim is not None else None
        self._battle_hidden = {}  # per-battle LSTM hidden state: {battle_tag: hidden}
        self._battle_history = {}  # per-battle transformer history: {battle_tag: list of input dicts}
        self._is_transformer = getattr(self.model, 'transformer_core', None) is not None

        # Keep for convenience; the model’s cfg is the source of truth at call time
        self.ctx_extra_dim = int(cfg.ctx_extra_dim)
        self.step_type_bins = int(cfg.step_type_bins)

    # Order normalization helpers removed: BattleOrder from create_order() is
    # the correct return type for choose_move() in poke-env.

    # ----------- main policy hook -----------

    def choose_move(self, battle):
        """
        Featurize -> build (optional) ctx_extra & step-type -> forward (no obs width changes) -> masked argmax.
        Always returns an object with `.message: str`.
        """
        btag = getattr(battle, "battle_tag", None) or f"battle-{id(battle)}"
        turn = int(getattr(battle, "turn", 1))

        # --- Per-battle state management ---
        # For LSTM: hidden state (h, c) tuple
        # For Transformer: buffer of past (obs, mask, step_type, ctx, eids, mids, sids, ms, ss) tensors
        if turn <= 1:
            self._battle_hidden.pop(btag, None)
            self._battle_history.pop(btag, None)

        # LSTM path
        h0 = self._battle_hidden.get(btag, None)
        # --- End: per-battle state ---
        
        # 1) canonical features (same as observer & train)
        mfb = None  # (optional) moved-first bits if you decide to track them per battle
        obs_vec, legal_mask, ctx_list, move_slots, switch_slots, entity_ids, move_ids, switch_ids = make_obs_mask_and_slots(battle, moved_first_bits=mfb)

        obs_t  = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)     # [1, F]
        mask_t = torch.tensor(legal_mask, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, 9]

        if mask_t.sum().item() <= 0:
            # No legal actions; fall back safely
            return self.choose_random_move(battle)

        # 2) step-type (bins identical to training)
        step_t = None
        K = int(getattr(self.model.cfg, "step_type_bins", 0) or 0)
        if K > 0:
            t_abs = max(0, int(getattr(battle, "turn", 1)) - 1)
            # Use the shared helper; pass a conservative cap for inference-only bucketing
            bin_idx = step_type_from_abs_t(t_abs, bins=K, cap=50)
            step_t = torch.tensor([bin_idx], dtype=torch.long, device=self.device)

        # 3) ctx tensor (zeros if model expects but ctx_list is None; pad/truncate to match D)
        ctx_t = None
        D = int(getattr(self.model.cfg, "ctx_extra_dim", 0) or 0)
        if D > 0:
            if ctx_list is None:
                ctx_t = torch.zeros((1, D), dtype=torch.float32, device=self.device)
            else:
                # Pad or truncate to exact model dimension to avoid shape mismatch
                ctx_flat = list(ctx_list)
                if len(ctx_flat) != D:
                    print(f"[BCPolicy] [warn] ctx dim {len(ctx_flat)} != model expects {D}, padding/truncating")
                if len(ctx_flat) < D:
                    ctx_flat.extend([0.0] * (D - len(ctx_flat)))
                elif len(ctx_flat) > D:
                    ctx_flat = ctx_flat[:D]
                ctx_t = torch.tensor(ctx_flat, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, D]
        
        mv = torch.tensor(move_slots, dtype=torch.float32, device=self.device).unsqueeze(0)   # [1,4,M]
        sw = torch.tensor(switch_slots, dtype=torch.float32, device=self.device).unsqueeze(0) # [1,5,S]

        # v5 entity IDs for embeddings
        eids_t = None
        mids_t = None
        sids_t = None
        if entity_ids is not None and len(entity_ids) > 0:
            eids_t = torch.tensor(entity_ids, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, N]
        if move_ids is not None and len(move_ids) > 0:
            mids_t = torch.tensor(move_ids, dtype=torch.long, device=self.device).unsqueeze(0)    # [1, 4]
        if switch_ids is not None and len(switch_ids) > 0:
            sids_t = torch.tensor(switch_ids, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, 5]

        # 4) forward (do NOT append step/ctx into obs)
        with torch.no_grad():
            if self._is_transformer:
                # Transformer: accumulate history on CPU (save GPU memory) and pass full sequence
                hist = self._battle_history.get(btag, [])
                # Store on CPU to avoid GPU memory bloat (~40-50 MB per long battle otherwise)
                hist.append({
                    "obs": obs_t.cpu(), "mask": mask_t.cpu(),
                    "step_type": step_t.cpu() if step_t is not None else None,
                    "ctx_extra": ctx_t.cpu() if ctx_t is not None else None,
                    "entity_ids": eids_t.cpu() if eids_t is not None else None,
                    "move_ids": mids_t.cpu() if mids_t is not None else None,
                    "switch_ids": sids_t.cpu() if sids_t is not None else None,
                    "move_slots": mv.cpu(), "switch_slots": sw.cpu(),
                })
                ctx_len = getattr(self.model.cfg, "context_length", 128)
                if len(hist) > ctx_len:
                    hist = hist[-ctx_len:]
                self._battle_history[btag] = hist

                # Stack history into sequence tensors [1, T, ...] and move to device
                T = len(hist)
                dev = self.device

                def _cat_field(key):
                    """Concat field across history, handling None consistently."""
                    vals = [h[key] for h in hist]
                    if vals[0] is None:
                        return None
                    return torch.cat(vals, dim=0).unsqueeze(0).to(dev)

                obs_seq = _cat_field("obs")         # [1,T,F]
                mask_seq = _cat_field("mask")        # [1,T,9]
                st_seq = _cat_field("step_type")
                ctx_seq = _cat_field("ctx_extra")
                eid_seq = _cat_field("entity_ids")
                mid_seq = _cat_field("move_ids")
                sid_seq = _cat_field("switch_ids")
                ms_seq = _cat_field("move_slots")    # [1,T,4,M]
                ss_seq = _cat_field("switch_slots")  # [1,T,5,S]
                sl = torch.tensor([T], device=dev)

                out = self.model(obs_seq, action_mask=mask_seq, step_type=st_seq, ctx_extra=ctx_seq,
                                 move_slots=ms_seq, switch_slots=ss_seq, seq_lens=sl,
                                 entity_ids=eid_seq, move_ids=mid_seq, switch_ids=sid_seq)

                # Extract logits for the LAST timestep only
                logits = _extract_action_logits_from_forward(out)
                if logits.dim() == 3:
                    logits = logits[:, -1, :]
                mask_t = mask_seq[:, -1, :]
            else:
                # LSTM path: single-step with hidden state
                out = self.model(obs_t, action_mask=mask_t, step_type=step_t, ctx_extra=ctx_t,
                                 move_slots=mv, switch_slots=sw, h0=h0,
                                 entity_ids=eids_t, move_ids=mids_t, switch_ids=sids_t)
                logits = _extract_action_logits_from_forward(out)

            # Safety: align device & mask
            logits = logits.to(mask_t.device, dtype=torch.float32)

            # Mask out illegal actions before sampling/argmax
            # (Assumes mask_t is 1.0 for legal, 0.0 for illegal)
            illegal = (mask_t <= 0)
            logits = logits.masked_fill(illegal, -1e9)

        # LSTM hidden threading
        if isinstance(out, dict):
            # be generous with key names for forward/backward compat
            if logits is None:
                logits = out.get("joint", out.get("policy"))
            new_hidden = out.get("h_n", out.get("hidden", None))
        elif isinstance(out, (list, tuple)) and len(out) == 2:
            # Old-style: (logits, hidden)
            logits, new_hidden = out
        else:
            # Oldest-style: raw logits only
            logits = out
            new_hidden = None

        # Store hidden state for this specific battle
        if new_hidden is not None:
            self._battle_hidden[btag] = new_hidden

        # Clean up finished battles to prevent memory leak
        if battle.finished:
            self._battle_hidden.pop(btag, None)
            self._battle_history.pop(btag, None)

        if logits is None:
            raise RuntimeError("Policy forward did not return logits (got: %r)" % (type(out),))

        # 5) masked argmax to choose action index
        #    (safe, differentiable not needed at inference; tie-breaker by first max)
        masked = logits.clone()
        masked[mask_t <= 0] = -1e9
        idx = int(torch.argmax(masked, dim=-1).item())

        # 6) map index -> legal order
        # action_mask returns metadata dicts; we need the actual Move/Pokemon objects
        avail_moves = list(battle.available_moves or [])
        avail_switches = list(battle.available_switches or [])
        num_moves = min(4, len(avail_moves))
        if idx < num_moves:
            return self.create_order(avail_moves[idx])
        sidx = idx - 4
        if 0 <= sidx < len(avail_switches):
            return self.create_order(avail_switches[sidx])
        print("[BC][player][warn] could Not find legal move or switch - fallback to Random.", flush=True)
        return self.choose_random_move(battle)

