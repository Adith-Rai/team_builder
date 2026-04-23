# src/policy_heads_v8.py
# v8 PokeTransformer architecture for Pokemon battle AI.
#
# Architecture:
#   17 entity tokens -> Spatial Transformer (Poke-Mask) -> Turn Summary
#   Turn summaries -> Temporal Transformer (causal) -> Policy/Value heads
#
# Sub-networks: PokemonNet, MoveNet, FieldNet, TransitionNet, NumericalBank
# Value head: 51-bin distributional (two-hot encoding)
#
# All architecture params are configurable via PokeTransformerConfig dataclass.
# CLI flags defined in add_model_args().

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import (
    DIMS, N_TYPES, N_STATUS, N_VOLATILE, N_PARADOX,
    POKEMON_CONT_DIM, FIELD_CONT_DIM, TRANSITION_CONT_DIM,
    MOVE_SLOT_CONT_DIM, SWITCH_SLOT_CONT_DIM,
)


# =============================
# Config
# =============================

@dataclass
class PokeTransformerConfig:
    """All architecture hyperparameters. Saved in checkpoints for reproducibility."""

    # Core dimensions
    d_model: int = 384
    n_spatial_layers: int = 4
    n_temporal_layers: int = 2
    n_heads: int = 4
    ff_mult: int = 4          # feedforward expansion
    # Dropout 0.05 matches Metamon's IL + RL configs. Old default was 0.1;
    # loading old checkpoints preserves the saved 0.1 via from_dict.
    dropout: float = 0.05

    # --- Capacity reallocation (Session 37, Metamon-inspired) ---
    # When d_spatial / d_temporal are None, fall back to d_model (old behavior).
    # When n_summary_tokens == 0, use legacy single attention-pooled summary.
    # When >= 1, use K learnable scratch tokens (Metamon-style) + projection to d_temporal.
    d_spatial: Optional[int] = None
    d_temporal: Optional[int] = None
    n_summary_tokens: int = 0

    # Temporal
    temporal_context: int = 200  # max turns of history
    temporal_mode: str = "summary"  # "summary" or "frames"

    # Sub-network dims
    move_dim: int = 128        # MoveNet output dim

    # NumericalBank embedding dims
    bank_dim: int = 16         # default bank embedding size
    bank_dim_small: int = 8    # smaller banks (PP, priority, level, weight, height)

    # Entity embedding dims
    entity_embed_dim: int = 32

    # Vocab sizes (from vocab.py, with pad=0)
    n_species: int = 1548
    n_moves: int = 953
    n_items: int = 2340
    n_abilities: int = 314

    # Value head
    v_bins: int = 51
    v_min: float = -1.6
    v_max: float = 1.6

    # Action space (from FormatConfig — 4 moves + 5 switches for singles)
    n_actions: int = 9

    # Gradient checkpointing (saves VRAM, ~30% slower)
    gradient_checkpoint: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PokeTransformerConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        dropped = [k for k in d.keys() if k not in valid]
        if dropped:
            import logging
            logging.getLogger("pokemon_ai").warning(
                "PokeTransformerConfig.from_dict: dropping unknown keys %s "
                "(checkpoint has fields not present in current code)", dropped,
            )
        return cls(**{k: v for k, v in d.items() if k in valid})


# =============================
# NumericalBank
# =============================

class NumericalBank(nn.Module):
    """Learned embedding for quantized continuous values.
    Replaces raw floats which cause gradient instability (ps-ppo finding)."""

    def __init__(self, num_values: int, bank_dim: int):
        super().__init__()
        self.num_values = num_values
        self.embedding = nn.Embedding(num_values, bank_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: int tensor of any shape. Returns: (*x.shape, bank_dim)."""
        return self.embedding(x.clamp(0, self.num_values - 1))


# =============================
# MoveNet
# =============================

class MoveNet(nn.Module):
    """Encodes one move into a fixed-dim vector.

    Input: move_id (int), bp/acc/pp/priority (ints for banks), continuous (109-dim)
    Output: (move_dim,) tensor
    """

    def __init__(self, cfg: PokeTransformerConfig):
        super().__init__()
        self.move_embed = nn.Embedding(cfg.n_moves, cfg.entity_embed_dim)
        self.bp_bank = NumericalBank(256, cfg.bank_dim)
        self.acc_bank = NumericalBank(101, cfg.bank_dim)
        self.pp_bank = NumericalBank(65, cfg.bank_dim_small)
        self.prio_bank = NumericalBank(13, cfg.bank_dim_small)

        # Total input: embed(32) + banks(16+16+8+8=48) + continuous(109) = 189
        in_dim = cfg.entity_embed_dim + cfg.bank_dim * 2 + cfg.bank_dim_small * 2 + MOVE_SLOT_CONT_DIM
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.move_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.move_dim),
            nn.Linear(cfg.move_dim, cfg.move_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.move_dim),
        )

    def forward(self, move_id: torch.Tensor, bp: torch.Tensor, acc: torch.Tensor,
                pp: torch.Tensor, prio: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
        """All inputs: (...) shaped. cont: (..., 109). Returns: (..., move_dim)."""
        e = self.move_embed(move_id)
        b = torch.cat([self.bp_bank(bp), self.acc_bank(acc),
                        self.pp_bank(pp), self.prio_bank(prio)], dim=-1)
        x = torch.cat([e, b, cont], dim=-1)
        return self.mlp(x)


# =============================
# PokemonNet
# =============================

class PokemonNet(nn.Module):
    """Encodes one Pokemon into a d_spatial-dim token.

    Processes entity embeddings + NumericalBanks + continuous features + 4 MoveNet outputs.
    """

    def __init__(self, cfg: PokeTransformerConfig, move_net: MoveNet, d_out: int):
        super().__init__()
        self.move_net = move_net  # shared across all Pokemon

        self.species_embed = nn.Embedding(cfg.n_species, cfg.entity_embed_dim)
        self.item_embed = nn.Embedding(cfg.n_items, cfg.entity_embed_dim)
        self.ability_embed = nn.Embedding(cfg.n_abilities, cfg.entity_embed_dim)

        self.hp_bank = NumericalBank(101, cfg.bank_dim)
        self.level_bank = NumericalBank(100, cfg.bank_dim_small)
        self.weight_bank = NumericalBank(201, cfg.bank_dim_small)
        self.height_bank = NumericalBank(41, cfg.bank_dim_small)
        # 6 stat banks (shared embedding)
        self.stat_bank = NumericalBank(256, cfg.bank_dim)

        # Input: 3 entity embeds(3*32=96) + banks(16 + 8*3 + 16*6 = 136) + 4*move_dim(512) + cont(285)
        entity_dim = cfg.entity_embed_dim * 3
        bank_total = cfg.bank_dim + cfg.bank_dim_small * 3 + cfg.bank_dim * 6  # hp + level/weight/height + 6 stats
        move_total = 4 * cfg.move_dim
        in_dim = entity_dim + bank_total + move_total + POKEMON_CONT_DIM

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_out),
            nn.GELU(),
            nn.LayerNorm(d_out),
            nn.Linear(d_out, d_out),
            nn.GELU(),
            nn.LayerNorm(d_out),
        )

    def forward(self, ids: Dict[str, torch.Tensor], banks: Dict[str, torch.Tensor],
                cont: torch.Tensor, move_ids: torch.Tensor,
                move_banks: Dict[str, torch.Tensor], move_cont: torch.Tensor) -> torch.Tensor:
        """
        Supports arbitrary batch dims. All inputs share the same leading dims.
        ids: species/item/ability int tensors (...,)
        banks: hp_pct/level/weight/height/stat_* int tensors (...,)
        cont: (..., POKEMON_CONT_DIM) float
        move_ids: (..., 4) int
        move_banks: bp/acc/pp/prio (..., 4) int tensors
        move_cont: (..., 4, 23) float

        Returns: (..., d_model)
        """
        # Entity embeddings
        sp = self.species_embed(ids["species"])
        it = self.item_embed(ids["item"])
        ab = self.ability_embed(ids["ability"])

        # Banks
        hp = self.hp_bank(banks["hp_pct"])
        lv = self.level_bank(banks["level"])
        wt = self.weight_bank(banks["weight"])
        ht = self.height_bank(banks["height"])
        stat_embeds = [self.stat_bank(banks[k]) for k in
                       ["stat_hp", "stat_atk", "stat_def", "stat_spa", "stat_spd", "stat_spe"]]
        stats = torch.cat(stat_embeds, dim=-1)

        # Batched move encoding: flatten (..., 4) → (...*4) for one MoveNet call
        leading = move_ids.shape[:-1]  # everything except the 4
        flat_mid = move_ids.reshape(-1)  # (N*4,)
        flat_zero = torch.zeros_like(flat_mid)
        flat_mc = move_cont.reshape(-1, 23)  # (N*4, 23)
        flat_pad = torch.zeros(flat_mc.shape[0], MOVE_SLOT_CONT_DIM - 23, device=flat_mc.device)
        flat_mc_padded = torch.cat([flat_mc, flat_pad], dim=-1)  # (N*4, MOVE_SLOT_CONT_DIM)
        flat_mv = self.move_net(flat_mid, flat_zero, flat_zero, flat_zero, flat_zero, flat_mc_padded)
        # Reshape back: (N*4, move_dim) → (..., 4*move_dim)
        moves = flat_mv.reshape(*leading, -1)  # (..., 4*move_dim)

        x = torch.cat([sp, it, ab, hp, lv, wt, ht, stats, moves, cont], dim=-1)
        return self.mlp(x)


# =============================
# FieldNet
# =============================

class FieldNet(nn.Module):
    """Encodes field state into a d_spatial-dim token."""

    def __init__(self, cfg: PokeTransformerConfig, d_out: int):
        super().__init__()
        self.turn_bank = NumericalBank(201, cfg.bank_dim)
        self.weather_dur_bank = NumericalBank(9, cfg.bank_dim_small)
        self.terrain_dur_bank = NumericalBank(6, cfg.bank_dim_small)
        self.tr_dur_bank = NumericalBank(6, cfg.bank_dim_small)

        bank_total = cfg.bank_dim + cfg.bank_dim_small * 3
        in_dim = bank_total + FIELD_CONT_DIM

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_out),
            nn.GELU(),
            nn.LayerNorm(d_out),
            nn.Linear(d_out, d_out),
            nn.GELU(),
            nn.LayerNorm(d_out),
        )

    def forward(self, banks: Dict[str, torch.Tensor], cont: torch.Tensor) -> torch.Tensor:
        t = self.turn_bank(banks["turn"])
        wd = self.weather_dur_bank(banks["weather_dur"])
        td = self.terrain_dur_bank(banks["terrain_dur"])
        trd = self.tr_dur_bank(banks["tr_dur"])
        x = torch.cat([t, wd, td, trd, cont], dim=-1)
        return self.mlp(x)


# =============================
# TransitionNet
# =============================

class TransitionNet(nn.Module):
    """Encodes previous-turn events into a d_spatial-dim token."""

    def __init__(self, cfg: PokeTransformerConfig, d_out: int):
        super().__init__()
        # Action IDs can be either move or species — use a combined embedding
        # (move and species IDs don't overlap much, and unknown=0 handles both)
        self.action_embed = nn.Embedding(max(cfg.n_moves, cfg.n_species) + 1, cfg.entity_embed_dim)

        in_dim = cfg.entity_embed_dim * 2 + TRANSITION_CONT_DIM
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_out),
            nn.GELU(),
            nn.LayerNorm(d_out),
            nn.Linear(d_out, d_out),
            nn.GELU(),
            nn.LayerNorm(d_out),
        )

    def forward(self, ids: Dict[str, torch.Tensor], cont: torch.Tensor) -> torch.Tensor:
        our_act = self.action_embed(ids["our_action"])
        opp_act = self.action_embed(ids["opp_action"])
        x = torch.cat([our_act, opp_act, cont], dim=-1)
        return self.mlp(x)


# =============================
# Spatial Transformer with Poke-Mask
# =============================

class SpatialTransformer(nn.Module):
    """Processes entity tokens with Poke-Mask attention.

    Token layout (legacy K=0):
      [actor, critic, field, transition, 6 our mons, 6 opp mons] = 16 tokens
      Summary: single attention-pooled vector.

    Token layout (K>=1, Metamon-inspired):
      [actor, critic, field, transition, 6 our mons, 6 opp mons, K scratch] = 16+K tokens
      Summaries: K scratch token outputs (no pooling).

    Poke-Mask: non-decision tokens ([2:]) can't attend to decision tokens [0:2].
    Actor [0] can't see critic [1] and vice versa.
    """

    def __init__(self, cfg: PokeTransformerConfig, d_spatial: int):
        super().__init__()
        self.d_spatial = d_spatial
        self.gradient_checkpoint = cfg.gradient_checkpoint
        self.n_summary_tokens = cfg.n_summary_tokens
        # 16 base tokens + K summary scratch tokens (0 in legacy mode)
        self.n_base = 16
        self.n_tokens = self.n_base + self.n_summary_tokens

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_spatial,
            nhead=cfg.n_heads,
            dim_feedforward=d_spatial * cfg.ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_spatial_layers,
        )

        # Learnable decision tokens
        self.actor_token = nn.Parameter(torch.randn(d_spatial) * 0.02)
        self.critic_token = nn.Parameter(torch.randn(d_spatial) * 0.02)

        # Token type embedding. Legacy (K=0) has 6 types; new (K≥1) adds type 6=summary.
        # Sizing conditionally preserves exact state-dict shape for loading legacy checkpoints.
        n_types_embed = 6 if self.n_summary_tokens == 0 else 7
        self.token_type_embed = nn.Embedding(n_types_embed, d_spatial)

        if self.n_summary_tokens == 0:
            # Legacy path: attention pooling yields a single summary vector.
            self.summary_query = nn.Parameter(torch.randn(d_spatial) * 0.02)
            self.summary_attn = nn.MultiheadAttention(
                d_spatial, num_heads=cfg.n_heads, dropout=cfg.dropout, batch_first=True,
            )
            self.summary_norm = nn.LayerNorm(d_spatial)
        else:
            # New path: K learnable scratch tokens participate in self-attention.
            # Flatten output (B, K, d_spatial) → (B, K*d_spatial); projection to temporal happens outside.
            self.summary_scratch = nn.Parameter(torch.randn(self.n_summary_tokens, d_spatial) * 0.02)

        self.register_buffer("poke_mask", self._build_poke_mask())

    def _build_poke_mask(self) -> torch.Tensor:
        """Additive attention mask. -inf blocks attention.

        All non-decision tokens (state + summary) are blocked from attending to
        decision tokens [0:2]. This preserves the Poke-Mask guarantee that the
        policy/value representation at the actor/critic tokens cannot be
        indirectly contaminated via summary-token relays.
        """
        mask = torch.zeros(self.n_tokens, self.n_tokens)
        mask[2:, 0:2] = float("-inf")    # state + summary tokens can't see decision
        mask[0, 1] = float("-inf")       # actor can't see critic
        mask[1, 0] = float("-inf")       # critic can't see actor
        return mask

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        tokens: (B, 14, d_spatial) -- field + transition + 6 our + 6 opp = 14 entity tokens

        Returns:
            full_output: (B, n_tokens, d_spatial) -- all token outputs
            summaries: legacy: (B, d_spatial) single pooled summary
                       new:    (B, K, d_spatial) K scratch token outputs
        """
        B = tokens.shape[0]
        device = tokens.device

        # Prepend decision tokens; append K summary scratch tokens if enabled
        actor = self.actor_token.unsqueeze(0).expand(B, -1).unsqueeze(1)   # (B, 1, D)
        critic = self.critic_token.unsqueeze(0).expand(B, -1).unsqueeze(1)  # (B, 1, D)
        if self.n_summary_tokens > 0:
            scratch = self.summary_scratch.unsqueeze(0).expand(B, -1, -1)   # (B, K, D)
            x = torch.cat([actor, critic, tokens, scratch], dim=1)          # (B, 16+K, D)
        else:
            x = torch.cat([actor, critic, tokens], dim=1)                   # (B, 16, D)

        # Token type IDs
        type_list = [0, 1, 2, 3] + [4] * 6 + [5] * 6 + [6] * self.n_summary_tokens
        type_ids = torch.tensor(type_list, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        assert type_ids.shape[1] == self.n_tokens, f"type_ids {type_ids.shape[1]} != n_tokens {self.n_tokens}"
        x = x + self.token_type_embed(type_ids)

        if self.gradient_checkpoint and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self.transformer, x, self.poke_mask, use_reentrant=False,
            )
        else:
            x = self.transformer(x, mask=self.poke_mask)

        if self.n_summary_tokens == 0:
            # Legacy attention-pool summary (FP32 to prevent FP16 overflow in attention scores)
            query = self.summary_query.unsqueeze(0).unsqueeze(1).expand(B, -1, -1)  # (B, 1, D)
            with torch.amp.autocast("cuda", enabled=False):
                summary, _ = self.summary_attn(query.float(), x.float(), x.float())
                summary = self.summary_norm(summary.squeeze(1))  # (B, D)
            return x, summary
        else:
            # Scratch-token summaries: (B, K, D)
            summaries = x[:, self.n_base:, :]
            return x, summaries


# =============================
# Temporal Transformer
# =============================

class TemporalTransformer(nn.Module):
    """Processes turn summaries across time with causal attention.

    Operates in d_temporal, which may differ from d_spatial. When differ, a
    projection layer in PokeTransformer maps the flattened spatial summary
    (K * d_spatial) to d_temporal before this module sees the sequence.
    """

    def __init__(self, cfg: PokeTransformerConfig, d_temporal: int):
        super().__init__()
        self.d_temporal = d_temporal
        self.temporal_context = cfg.temporal_context
        self.temporal_mode = cfg.temporal_mode
        self.gradient_checkpoint = cfg.gradient_checkpoint

        self.pos_embed = nn.Embedding(cfg.temporal_context, d_temporal)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_temporal,
            nhead=cfg.n_heads,
            dim_feedforward=d_temporal * cfg.ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_temporal_layers,
        )

    def forward(self, summaries: torch.Tensor, seq_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Full recompute forward (used during training / PPO update).

        summaries: (B, T, d_temporal) -- T turn summaries (most recent last)
        seq_lens: (B,) optional -- actual lengths per batch item

        Returns: (B, d_temporal) -- temporal context from the last turn
        """
        B, T, D = summaries.shape
        device = summaries.device

        # Truncate to temporal_context
        if T > self.temporal_context:
            summaries = summaries[:, -self.temporal_context:, :]
            T = self.temporal_context
            if seq_lens is not None:
                seq_lens = seq_lens.clamp(max=T)

        # Add positional embeddings
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = summaries + self.pos_embed(positions)

        # Causal mask (each turn can only attend to past turns + itself)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        # Padding mask for variable-length sequences
        padding_mask = None
        if seq_lens is not None:
            padding_mask = torch.arange(T, device=device).unsqueeze(0) >= seq_lens.unsqueeze(1)

        if self.gradient_checkpoint and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self.transformer, x, causal_mask, padding_mask, use_reentrant=False,
            )
        else:
            x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)

        # Extract last valid timestep per batch item
        if seq_lens is not None:
            idx = (seq_lens - 1).clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, D)
            output = x.gather(1, idx).squeeze(1)  # (B, D)
        else:
            output = x[:, -1, :]  # (B, D)

        return output

    # KV-cache was explored but temporal is only 14% of forward pass (~0.95ms).
    # Caching saves ~0.5ms at most — not worth the complexity.
    # Spatial (54%) + action_encoder (24%) are the real bottlenecks, already batched in v9.


# =============================
# Action Slot Encoder (for move/switch context)
# =============================

class ActionSlotEncoder(nn.Module):
    """Encodes active move slots and switch slots for action-aware policy head."""

    def __init__(self, cfg: PokeTransformerConfig, move_net: MoveNet, d_out: int):
        super().__init__()
        self.move_net = move_net  # shared MoveNet

        # Switch slot encoder
        self.switch_species_embed = nn.Embedding(cfg.n_species, cfg.entity_embed_dim)
        self.switch_mlp = nn.Sequential(
            nn.Linear(cfg.entity_embed_dim + SWITCH_SLOT_CONT_DIM, cfg.move_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.move_dim),
        )

        # Project 4 move vecs + 5 switch vecs to per-action context (d_spatial)
        self.action_proj = nn.Linear(cfg.move_dim, d_out)

    def forward(self, move_ids: torch.Tensor, move_banks: Dict[str, torch.Tensor],
                move_cont: torch.Tensor, switch_ids: torch.Tensor,
                switch_cont: torch.Tensor) -> torch.Tensor:
        """
        move_ids: (B, 4) int
        move_banks: bp/acc/pp/prio (B, 4) int each
        move_cont: (B, 4, 109) float
        switch_ids: (B, 5) int
        switch_cont: (B, 5, 30) float

        Returns: (B, 9, d_model) -- per-action context vectors
        """
        B = move_ids.shape[0]

        # Encode 4 moves through MoveNet
        move_vecs = []
        for i in range(4):
            mv = self.move_net(
                move_ids[:, i], move_banks["bp"][:, i], move_banks["acc"][:, i],
                move_banks["pp"][:, i], move_banks["prio"][:, i], move_cont[:, i],
            )
            move_vecs.append(mv)
        move_vecs = torch.stack(move_vecs, dim=1)  # (B, 4, move_dim)

        # Encode 5 switches
        sw_embeds = self.switch_species_embed(switch_ids)  # (B, 5, entity_embed_dim)
        sw_x = torch.cat([sw_embeds, switch_cont], dim=-1)  # (B, 5, embed+SWITCH_SLOT_CONT_DIM)
        sw_vecs = self.switch_mlp(sw_x)  # (B, 5, move_dim)

        # Combine: 4 moves + 5 switches = 9 actions
        all_vecs = torch.cat([move_vecs, sw_vecs], dim=1)  # (B, 9, move_dim)
        return self.action_proj(all_vecs)  # (B, 9, d_model)


# =============================
# Main Model: PokeTransformer
# =============================

class PokeTransformer(nn.Module):
    """Full v8 battle policy + value network.

    Forward flow:
    1. Encode entities via sub-networks -> 15 tokens
    2. Spatial transformer with Poke-Mask -> 17 tokens + turn summary
    3. Temporal transformer over turn summaries -> temporal context
    4. Policy head: actor output + action slot context -> 9 logits
    5. Value head: critic output -> 51-bin distribution
    """

    def __init__(self, cfg: PokeTransformerConfig):
        super().__init__()
        self.cfg = cfg

        # Resolve dim variants. When d_spatial/d_temporal are None, use d_model
        # (legacy behavior). When explicitly set, allows asymmetric shapes per
        # Metamon-style capacity allocation (temporal > spatial).
        self.d_spatial = cfg.d_spatial if cfg.d_spatial is not None else cfg.d_model
        self.d_temporal = cfg.d_temporal if cfg.d_temporal is not None else cfg.d_model
        self.n_summary_tokens = cfg.n_summary_tokens

        # Shared MoveNet (used by PokemonNet and ActionSlotEncoder)
        self.move_net = MoveNet(cfg)

        # Sub-networks — all operate in d_spatial
        self.pokemon_net = PokemonNet(cfg, self.move_net, self.d_spatial)
        self.field_net = FieldNet(cfg, self.d_spatial)
        self.transition_net = TransitionNet(cfg, self.d_spatial)

        # Spatial transformer (d_spatial) → summaries
        self.spatial = SpatialTransformer(cfg, self.d_spatial)

        # Summary → temporal projection. In legacy mode (K=0), summary is already
        # (B, d_spatial); project to d_temporal only if dims differ. In new mode
        # (K≥1), flatten K scratch tokens and project (B, K*d_spatial → d_temporal).
        if self.n_summary_tokens == 0:
            if self.d_spatial == self.d_temporal:
                self.summary_to_temporal = nn.Identity()
            else:
                self.summary_to_temporal = nn.Linear(self.d_spatial, self.d_temporal)
        else:
            self.summary_to_temporal = nn.Linear(
                self.n_summary_tokens * self.d_spatial, self.d_temporal,
            )

        # Temporal transformer (d_temporal)
        self.temporal = TemporalTransformer(cfg, self.d_temporal)

        # Action slot encoder — outputs per-action d_spatial (matches spatial tokens)
        self.action_encoder = ActionSlotEncoder(cfg, self.move_net, self.d_spatial)

        # Policy head: actor (d_spatial) + temporal_ctx (d_temporal) + action_ctx (d_spatial)
        policy_in = 2 * self.d_spatial + self.d_temporal
        policy_hidden = max(self.d_spatial, self.d_temporal)
        self.policy_head = nn.Sequential(
            nn.Linear(policy_in, policy_hidden),
            nn.GELU(),
            nn.Linear(policy_hidden, 1),
        )

        # Value head: critic (d_spatial) + temporal_ctx (d_temporal)
        value_in = self.d_spatial + self.d_temporal
        value_hidden = max(self.d_spatial, self.d_temporal)
        self.value_head = nn.Sequential(
            nn.Linear(value_in, value_hidden),
            nn.GELU(),
            nn.Linear(value_hidden, cfg.v_bins),
        )

        # Value support (bin centers)
        self.register_buffer(
            "v_support", torch.linspace(cfg.v_min, cfg.v_max, cfg.v_bins)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _encode_entities(self, batch: dict) -> torch.Tensor:
        """Encode all entities into 14 tokens (field + transition + 12 pokemon).

        Optimized: batches all 12 pokemon into ONE PokemonNet call (and all 48 moves
        into one MoveNet call internally). 62 small forwards → 3 batched forwards.

        batch: dict with structured feature tensors.
        Returns: (B, 14, d_model)
        """
        B = batch["field_cont"].shape[0]

        # Field + transition tokens (small, not worth batching)
        field_token = self.field_net(batch["field_banks"], batch["field_cont"])     # (B, D)
        trans_token = self.transition_net(batch["transition_ids"], batch["transition_cont"])  # (B, D)

        # Batch all 12 pokemon (6 our + 6 opp) in ONE PokemonNet call
        # Stack our + opp: (B, 6, ...) + (B, 6, ...) → (B, 12, ...)
        all_poke_ids = torch.cat([batch["our_pokemon_ids"], batch["opp_pokemon_ids"]], dim=1)      # (B, 12, 3)
        all_poke_banks = torch.cat([batch["our_pokemon_banks"], batch["opp_pokemon_banks"]], dim=1)  # (B, 12, 10)
        all_poke_cont = torch.cat([batch["our_pokemon_cont"], batch["opp_pokemon_cont"]], dim=1)    # (B, 12, 285)
        all_move_ids = torch.cat([batch["our_pokemon_move_ids"], batch["opp_pokemon_move_ids"]], dim=1)  # (B, 12, 4)
        all_move_cont = torch.cat([batch["our_pokemon_move_cont"], batch["opp_pokemon_move_cont"]], dim=1)  # (B, 12, 4, 23)

        # Reshape (B, 12, ...) → (B*12, ...) for batched PokemonNet
        B12 = B * 12
        _id_names = ["species", "item", "ability"]
        _bank_names = ["hp_pct", "level", "weight", "height",
                       "stat_hp", "stat_atk", "stat_def", "stat_spa", "stat_spd", "stat_spe"]

        flat_ids = {_id_names[j]: all_poke_ids[:, :, j].reshape(B12) for j in range(3)}
        flat_banks = {_bank_names[j]: all_poke_banks[:, :, j].reshape(B12) for j in range(10)}
        flat_cont = all_poke_cont.reshape(B12, -1)          # (B*12, 285)
        flat_move_ids = all_move_ids.reshape(B12, 4)         # (B*12, 4)
        flat_move_banks = {k: torch.zeros(B12, 4, dtype=torch.long, device=flat_cont.device)
                           for k in ["bp", "acc", "pp", "prio"]}
        flat_move_cont = all_move_cont.reshape(B12, 4, 23)   # (B*12, 4, 23)

        # ONE call for all 12 pokemon (internally batches 48 moves too)
        all_poke_tokens = self.pokemon_net(
            flat_ids, flat_banks, flat_cont, flat_move_ids, flat_move_banks, flat_move_cont,
        )  # (B*12, D)

        # Reshape back to (B, 12, D) → split into our (B, 6, D) and opp (B, 6, D)
        all_poke_tokens = all_poke_tokens.reshape(B, 12, -1)

        # Assemble: field + transition + 12 pokemon = 14 tokens
        tokens = torch.cat([
            field_token.unsqueeze(1),    # (B, 1, D)
            trans_token.unsqueeze(1),    # (B, 1, D)
            all_poke_tokens,             # (B, 12, D)
        ], dim=1)  # (B, 14, D)

        return tokens

    def forward_spatial(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run spatial transformer + project summaries to d_temporal.

        Returns:
            spatial_out: (B, n_tokens, d_spatial)
            summary_temporal: (B, d_temporal) — ready for temporal model
        """
        entity_tokens = self._encode_entities(batch)
        spatial_out, raw_summary = self.spatial(entity_tokens)
        # Legacy (K=0): raw_summary is (B, d_spatial). New (K≥1): (B, K, d_spatial).
        if self.n_summary_tokens == 0:
            summary_temporal = self.summary_to_temporal(raw_summary)  # (B, d_temporal)
        else:
            B = raw_summary.shape[0]
            summary_temporal = self.summary_to_temporal(raw_summary.reshape(B, -1))
        return spatial_out, summary_temporal

    def forward(self, batch: dict, history: Optional[torch.Tensor] = None,
                history_lens: Optional[torch.Tensor] = None) -> dict:
        """Full forward pass.

        batch: dict of tensors for current turn
        history: (B, T-1, d_model) past turn summaries (or None for first turn)
        history_lens: (B,) actual history lengths

        Returns dict:
            action_logits: (B, 9)
            value: (B,) expected value
            v_logits: (B, 51) distributional logits
            summary: (B, d_temporal) turn summary for temporal buffer (projected)
            spatial_output: (B, n_tokens, d_spatial) all spatial tokens
        """
        B = batch["field_cont"].shape[0]
        device = batch["field_cont"].device

        # 1. Spatial transformer (projects summary to d_temporal)
        spatial_out, summary = self.forward_spatial(batch)

        # 2. Temporal transformer (operates in d_temporal)
        if history is not None and history.shape[1] > 0:
            all_summaries = torch.cat([history, summary.unsqueeze(1)], dim=1)
            if history_lens is not None:
                temporal_lens = history_lens + 1
            else:
                temporal_lens = None
        else:
            all_summaries = summary.unsqueeze(1)  # (B, 1, d_temporal)
            temporal_lens = None

        temporal_ctx = self.temporal(all_summaries, temporal_lens)  # (B, d_temporal)

        # 3. Policy head — actor (d_spatial) + temporal_ctx (d_temporal) + action_ctx (d_spatial)
        actor_out = spatial_out[:, 0, :]   # (B, d_spatial)
        critic_out = spatial_out[:, 1, :]  # (B, d_spatial)

        action_ctx = self.action_encoder(
            batch["active_move_ids"], batch["active_move_banks"],
            batch["active_move_cont"], batch["switch_ids"], batch["switch_cont"],
        )  # (B, 9, d_spatial)

        actor_temporal = torch.cat([actor_out, temporal_ctx], dim=-1)  # (B, d_spatial + d_temporal)
        actor_temporal_exp = actor_temporal.unsqueeze(1).expand(-1, 9, -1)
        policy_input = torch.cat([actor_temporal_exp, action_ctx], dim=-1)  # (B, 9, 2*d_spatial + d_temporal)
        action_logits = self.policy_head(policy_input).squeeze(-1)  # (B, 9)

        if "legal_mask" in batch:
            mask = batch["legal_mask"]
            action_logits = action_logits.masked_fill(mask < 0.5, -100.0)

        # 4. Value head — critic (d_spatial) + temporal_ctx (d_temporal)
        value_input = torch.cat([critic_out, temporal_ctx], dim=-1)  # (B, d_spatial + d_temporal)
        v_logits = self.value_head(value_input)
        v_probs = F.softmax(v_logits, dim=-1)
        value = (v_probs * self.v_support).sum(-1)

        return {
            "action_logits": action_logits,
            "value": value,
            "v_logits": v_logits,
            "summary": summary.detach(),  # detach for temporal buffer
            "spatial_output": spatial_out,
        }

    def twohot_target(self, value: torch.Tensor) -> torch.Tensor:
        """Convert scalar value targets to two-hot encoding over v_bins."""
        value = value.clamp(self.cfg.v_min, self.cfg.v_max)
        bin_width = self.v_support[1] - self.v_support[0]
        idx = (value - self.v_support[0]) / bin_width
        lo = idx.floor().long().clamp(0, self.cfg.v_bins - 2)
        hi = (lo + 1).clamp(max=self.cfg.v_bins - 1)
        weight_hi = (idx - lo.float()).clamp(0, 1)
        target = torch.zeros(value.shape[0], self.cfg.v_bins, device=value.device)
        target.scatter_(1, lo.unsqueeze(1), (1 - weight_hi).unsqueeze(1))
        target.scatter_(1, hi.unsqueeze(1), weight_hi.unsqueeze(1))
        return target

    def forward_sequence(self, collated: dict, device: torch.device) -> dict:
        """Optimized forward for BC training over full episodes.

        Batches ALL turns' spatial processing in one GPU call (the expensive part),
        then runs temporal + heads per-turn (cheap since summaries are just vectors).

        collated: output of collate_seq with keys like our_pokemon_ids [B, T, 6, 7] etc.
        device: torch device

        Returns dict:
            action_logits: (B, T, 9)
            value: (B, T)
            v_logits: (B, T, 51)
        """
        from dataset import unpack_turn_batch

        B = collated["seq_lens"].shape[0]
        T = collated["mask"].shape[1]
        mask = collated["mask"]  # (B, T)

        # --- Phase 1: Batch all spatial processing ---
        # Flatten B*T turns into one big batch, run spatial once
        # Only process valid (non-padded) turns

        valid_indices = []  # list of (b, t) tuples
        for b in range(B):
            L = int(collated["seq_lens"][b].item())
            for t in range(L):
                valid_indices.append((b, t))

        if not valid_indices:
            return {
                "action_logits": torch.zeros(B, T, 9, device=device),
                "value": torch.zeros(B, T, device=device),
                "v_logits": torch.zeros(B, T, self.cfg.v_bins, device=device),
            }

        N_valid = len(valid_indices)
        bs_arr = [i[0] for i in valid_indices]
        ts_arr = [i[1] for i in valid_indices]

        # Build a mega-batch of all valid turns for spatial processing
        def _gather(key, indices_b, indices_t):
            """Gather [B, T, ...] -> [N, ...] using (b, t) index pairs."""
            tensor = collated[key]
            return tensor[indices_b, indices_t].to(device)

        # Build per-turn batches for spatial (entity encoding + spatial transformer)
        # We need to construct the batch dict that _encode_entities expects
        # with shape [N_valid, ...] instead of [B, ...]

        mega_batch = {
            "our_pokemon_ids": _gather("our_pokemon_ids", bs_arr, ts_arr)[:, :, :3],
            "our_pokemon_banks": _gather("our_pokemon_banks", bs_arr, ts_arr),
            "our_pokemon_cont": _gather("our_pokemon_cont", bs_arr, ts_arr),
            "our_pokemon_move_ids": _gather("our_pokemon_move_ids", bs_arr, ts_arr),
            "our_pokemon_move_cont": _gather("our_pokemon_move_cont", bs_arr, ts_arr),
            "opp_pokemon_ids": _gather("opp_pokemon_ids", bs_arr, ts_arr)[:, :, :3],
            "opp_pokemon_banks": _gather("opp_pokemon_banks", bs_arr, ts_arr),
            "opp_pokemon_cont": _gather("opp_pokemon_cont", bs_arr, ts_arr),
            "opp_pokemon_move_ids": _gather("opp_pokemon_move_ids", bs_arr, ts_arr),
            "opp_pokemon_move_cont": _gather("opp_pokemon_move_cont", bs_arr, ts_arr),
            "field_cont": collated["field_cont_raw"][bs_arr, ts_arr].to(device),
            "transition_cont": collated["trans_cont_raw"][bs_arr, ts_arr].to(device),
            "active_move_ids": collated["active_move_ids_raw"][bs_arr, ts_arr].to(device),
            "active_move_cont": collated["active_move_cont_raw"][bs_arr, ts_arr].to(device),
            "switch_ids": collated["switch_ids_raw"][bs_arr, ts_arr].to(device),
            "switch_cont": collated["switch_cont_raw"][bs_arr, ts_arr].to(device),
            "legal_mask": collated["legal_mask_raw"][bs_arr, ts_arr].to(device),
        }

        # Field banks -> dict
        fb = collated["field_banks_raw"][bs_arr, ts_arr].to(device)
        mega_batch["field_banks"] = {
            "turn": fb[:, 0], "weather_dur": fb[:, 1],
            "terrain_dur": fb[:, 2], "tr_dur": fb[:, 3],
        }

        # Transition ids -> dict
        ti = collated["trans_ids_raw"][bs_arr, ts_arr].to(device)
        mega_batch["transition_ids"] = {"our_action": ti[:, 0], "opp_action": ti[:, 1]}

        # Active move banks -> dict
        amb = collated["active_move_banks_raw"][bs_arr, ts_arr].to(device)
        mega_batch["active_move_banks"] = {
            "bp": amb[:, :, 0], "acc": amb[:, :, 1],
            "pp": amb[:, :, 2], "prio": amb[:, :, 3],
        }

        # Run spatial on entire mega-batch at once.
        # forward_spatial already projects summary to d_temporal.
        # spatial_out: (N_valid, n_tokens, d_spatial); all_summaries: (N_valid, d_temporal)
        spatial_out, all_summaries = self.forward_spatial(mega_batch)
        n_tokens = spatial_out.shape[1]

        # Action context from the mega-batch (d_spatial)
        action_ctx = self.action_encoder(
            mega_batch["active_move_ids"], mega_batch["active_move_banks"],
            mega_batch["active_move_cont"], mega_batch["switch_ids"],
            mega_batch["switch_cont"],
        )  # (N_valid, 9, d_spatial)

        # --- Phase 2: Scatter to (B, T, ...) grids ---
        summary_grid = torch.zeros(B, T, self.d_temporal, device=device)
        spatial_grid = torch.zeros(B, T, n_tokens, self.d_spatial, device=device)
        actctx_grid = torch.zeros(B, T, 9, self.d_spatial, device=device)
        legal_grid = torch.zeros(B, T, 9, device=device)
        for idx, (b, t) in enumerate(valid_indices):
            summary_grid[b, t] = all_summaries[idx]
            spatial_grid[b, t] = spatial_out[idx]
            actctx_grid[b, t] = action_ctx[idx]
            legal_grid[b, t] = mega_batch["legal_mask"][idx]

        out_logits = torch.full((B, T, 9), -100.0, device=device)
        out_value = torch.zeros(B, T, device=device)
        out_vlogits = torch.zeros(B, T, self.cfg.v_bins, device=device)

        seq_lens = collated["seq_lens"].to(device)  # (B,)

        # Process turns sequentially but batch across B items
        for t in range(T):
            # Which batch items are valid at turn t
            valid_mask_t = seq_lens > t  # (B,) bool
            if not valid_mask_t.any():
                break

            n_valid_t = int(valid_mask_t.sum().item())

            # Gather summaries for turns 0..t for valid batch items: (n_valid, t+1, D)
            temporal_input = summary_grid[valid_mask_t, :t+1, :]  # (n_valid, t+1, D)
            temporal_lens = torch.full((n_valid_t,), t + 1, dtype=torch.long, device=device)
            temporal_ctx = self.temporal(temporal_input, temporal_lens)  # (n_valid, D)

            # Gather actor/critic outputs for this turn
            actor_out = spatial_grid[valid_mask_t, t, 0, :]   # (n_valid, D)
            critic_out = spatial_grid[valid_mask_t, t, 1, :]   # (n_valid, D)
            act_ctx = actctx_grid[valid_mask_t, t]             # (n_valid, 9, D)
            legal_t = legal_grid[valid_mask_t, t]              # (n_valid, 9)

            # Policy: batched over all valid items at turn t
            at = torch.cat([actor_out, temporal_ctx], dim=-1)   # (n_valid, 2D)
            at_exp = at.unsqueeze(1).expand(-1, 9, -1)          # (n_valid, 9, 2D)
            pi_input = torch.cat([at_exp, act_ctx], dim=-1)     # (n_valid, 9, 3D)
            logits = self.policy_head(pi_input).squeeze(-1)     # (n_valid, 9)
            logits = logits.masked_fill(legal_t < 0.5, -100.0)

            # Value: batched
            vi = torch.cat([critic_out, temporal_ctx], dim=-1)  # (n_valid, 2D)
            vl = self.value_head(vi)                            # (n_valid, 51)
            val = (F.softmax(vl, dim=-1) * self.v_support).sum(-1)  # (n_valid,)

            # Scatter back
            out_logits[valid_mask_t, t] = logits
            out_vlogits[valid_mask_t, t] = vl
            out_value[valid_mask_t, t] = val

        return {
            "action_logits": out_logits,
            "value": out_value,
            "v_logits": out_vlogits,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================
# CLI argument helper
# =============================

def add_model_args(parser):
    """Add v8 model arguments to an argparse parser."""
    g = parser.add_argument_group("v8 Model Architecture")
    g.add_argument("--d-model", type=int, default=384)
    g.add_argument("--n-spatial-layers", type=int, default=4)
    g.add_argument("--n-temporal-layers", type=int, default=2)
    g.add_argument("--n-heads", type=int, default=4)
    g.add_argument("--ff-mult", type=int, default=4)
    g.add_argument("--dropout", type=float, default=0.05)
    g.add_argument("--temporal-context", type=int, default=200)
    g.add_argument("--temporal-mode", choices=["summary", "frames"], default="summary")
    g.add_argument("--move-dim", type=int, default=128)
    g.add_argument("--bank-dim", type=int, default=16)
    g.add_argument("--entity-embed-dim", type=int, default=32)
    g.add_argument("--v-bins", type=int, default=51)
    g.add_argument("--gradient-checkpoint", action="store_true")
    # Capacity reallocation (Session 37)
    g.add_argument("--d-spatial", type=int, default=None,
                   help="Spatial encoder d_model (defaults to --d-model)")
    g.add_argument("--d-temporal", type=int, default=None,
                   help="Temporal encoder d_model (defaults to --d-model)")
    g.add_argument("--n-summary-tokens", type=int, default=0,
                   help="K scratch summary tokens per turn (0 = legacy attention pool)")
    return parser


def config_from_args(args) -> PokeTransformerConfig:
    """Build PokeTransformerConfig from parsed argparse args."""
    return PokeTransformerConfig(
        d_model=args.d_model,
        n_spatial_layers=args.n_spatial_layers,
        n_temporal_layers=args.n_temporal_layers,
        n_heads=args.n_heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        temporal_context=args.temporal_context,
        temporal_mode=args.temporal_mode,
        move_dim=args.move_dim,
        bank_dim=args.bank_dim,
        entity_embed_dim=args.entity_embed_dim,
        v_bins=args.v_bins,
        gradient_checkpoint=args.gradient_checkpoint,
        d_spatial=getattr(args, "d_spatial", None),
        d_temporal=getattr(args, "d_temporal", None),
        n_summary_tokens=getattr(args, "n_summary_tokens", 0),
    )
