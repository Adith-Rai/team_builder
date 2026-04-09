# === FILE: policy_heads.py (FULL REPLACEMENT) ===
# Gen-agnostic multi-head policy with LSTM or Transformer core.
# Heads:
#   - action head: 9-way (4 moves + 5 switches)
#   - hierarchical option: (Move vs Switch) -> (which move / which bench), recombined into 9 logits
#   - modifier heads: any subset (tera/zmove/dmax/mega...), each is 1-logit (binary "use now?")
#   - value head (for PPO)
#   - auxiliary head: "will we move first next turn?"
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModifierSpec:
    name: str
    applies_to: str = "move"

@dataclass
class PolicyConfig:
    obs_dim: int
    action_dim: int = 9
    use_lstm: bool = False           # False = use transformer (new default)
    use_transformer: bool = True     # True = transformer core (new default)
    lstm_hidden: int = 256
    mlp_hidden: int = 256
    lstm_layers: int = 1
    mlp_layers: int = 2
    modifiers: List[ModifierSpec] = None
    # new toggles (default off)
    hierarchical: bool = False
    step_type_bins: int = 0          # 0=off; e.g., 3 for early/mid/late
    ctx_extra_dim: int = 0           # reserve small context vector concat (0=off)
    ctx_proj_dim: int = 32          # projection width for ctx
    move_slot_dim: int = 0
    switch_slot_dim: int = 0
    slot_hidden: int = 32      # small shared encoder size
    # v5 entity embeddings
    n_entity_ids: int = 0      # 0=off (legacy), 82=v5
    embed_dim: int = 32        # embedding vector size per entity
    n_species: int = 1548
    n_moves: int = 953
    n_items: int = 2340
    n_abilities: int = 314
    # Transformer config (only used when use_transformer=True)
    n_transformer_layers: int = 6
    n_heads: int = 4
    transformer_dropout: float = 0.1
    context_length: int = 128        # max turns to attend to

    def names(self) -> List[str]:
        return [m.name for m in (self.modifiers or [])]

class CausalTransformerCore(nn.Module):
    """Causal (decoder-only) transformer for sequence modeling over battle turns.

    Each turn can only attend to itself and previous turns (causal mask).
    Learned positional embeddings up to context_length.
    """
    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 context_length: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.pos_embed = nn.Embedding(context_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, seq_lens: Optional[torch.Tensor] = None):
        """
        Args:
            x: [B, T, D] encoded observations
            seq_lens: [B] actual lengths (optional, for padding mask)
        Returns:
            out: [B, T, D] transformer output
        """
        B, T, D = x.shape
        T_clamped = min(T, self.context_length)
        x = x[:, :T_clamped]

        # Learned positional embedding
        pos = torch.arange(T_clamped, device=x.device)
        x = x + self.pos_embed(pos).unsqueeze(0)

        # Causal mask: upper triangle = True (blocked)
        causal = torch.triu(torch.ones(T_clamped, T_clamped, device=x.device, dtype=torch.bool), diagonal=1)

        # Padding mask: True = ignore
        if seq_lens is not None:
            pad_mask = torch.arange(T_clamped, device=x.device).unsqueeze(0) >= seq_lens.unsqueeze(1).clamp(max=T_clamped)
        else:
            pad_mask = None

        out = self.transformer(x, mask=causal, src_key_padding_mask=pad_mask)
        out = self.ln(out)

        # Pad back to original T if we truncated
        if T > T_clamped:
            pad = torch.zeros(B, T - T_clamped, D, device=out.device, dtype=out.dtype)
            out = torch.cat([out, pad], dim=1)

        return out


class BattlePolicy(nn.Module):
    # v5 entity ID group definitions (indices into entity_ids tensor)
    # Each group: (start, end, embed_type)  where embed_type in {"species","item","ability","move"}
    _ENTITY_GROUPS = [
        # Active entities (per-entity: species + item + ability summed)
        ("our_active_species",  0,  1, "species"),
        ("our_active_item",     1,  2, "item"),
        ("our_active_ability",  2,  3, "ability"),
        ("opp_active_species",  3,  4, "species"),
        ("opp_active_item",     4,  5, "item"),
        ("opp_active_ability",  5,  6, "ability"),
        # Bench (per-side sum-pooled)
        ("our_bench_species",   6, 11, "species"),
        ("opp_bench_species",  11, 16, "species"),
        ("our_bench_items",    16, 21, "item"),
        ("opp_bench_items",    21, 26, "item"),
        ("our_bench_abilities",26, 31, "ability"),
        ("opp_bench_abilities",31, 36, "ability"),
        # Moves (per-group sum-pooled)
        ("opp_active_moves",   36, 40, "move"),
        ("opp_bench_moves",    40, 60, "move"),
        ("our_bench_moves",    60, 80, "move"),
        # Last action
        ("opp_last_move",      80, 81, "move"),
        ("opp_last_switch",    81, 82, "species"),
        # Preparing moves (two-turn moves)
        ("our_preparing_move", 82, 83, "move"),
        ("opp_preparing_move", 83, 84, "move"),
    ]
    N_EMBED_GROUPS = len(_ENTITY_GROUPS)  # 19 groups → 19 * embed_dim

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.mlp_hidden
        in_f = cfg.obs_dim

        # v5 entity embeddings
        self.use_embeddings = cfg.n_entity_ids > 0
        embed_contribution = 0
        if self.use_embeddings:
            E = cfg.embed_dim
            self.species_embed = nn.Embedding(cfg.n_species, E, padding_idx=0)
            self.move_embed = nn.Embedding(cfg.n_moves, E, padding_idx=0)
            self.item_embed = nn.Embedding(cfg.n_items, E, padding_idx=0)
            self.ability_embed = nn.Embedding(cfg.n_abilities, E, padding_idx=0)
            # Project pooled embeddings (19 groups * embed_dim → compressed)
            embed_contribution = self.N_EMBED_GROUPS * E
        else:
            self.species_embed = None
            self.move_embed = None
            self.item_embed = None
            self.ability_embed = None

        # Optional context slots (step-type embedding, hazard/context projection)
        ctx_in = 0
        if cfg.step_type_bins > 0:
            self.step_type_emb = nn.Embedding(cfg.step_type_bins, 8)
            ctx_in += 8
        else:
            self.step_type_emb = None

        if cfg.ctx_extra_dim > 0:
            self.ctx_proj = nn.Linear(cfg.ctx_extra_dim, cfg.ctx_proj_dim)
            ctx_in += cfg.ctx_proj_dim
        else:
            self.ctx_proj = None

        enc_in = in_f + ctx_in + embed_contribution

        # MLP encoder
        enc: List[nn.Module] = []
        for _ in range(max(0, cfg.mlp_layers)):
            enc += [nn.Linear(enc_in, H), nn.ReLU()]
            enc_in = H
        if cfg.mlp_layers == 0:
            enc = [nn.Linear(enc_in, H)]
        self.obs_encoder = nn.Sequential(*enc)

        # Sequence core: transformer, LSTM, or none (MLP only)
        self.transformer_core = None
        if cfg.use_transformer and not cfg.use_lstm:
            self.transformer_core = CausalTransformerCore(
                d_model=H,
                n_heads=cfg.n_heads,
                n_layers=cfg.n_transformer_layers,
                context_length=cfg.context_length,
                dropout=cfg.transformer_dropout,
            )
            self.lstm = None
            core_out = H
        elif cfg.use_lstm:
            self.lstm = nn.LSTM(
                input_size=H,
                hidden_size=cfg.lstm_hidden,
                num_layers=max(1, cfg.lstm_layers),
                batch_first=True
            )
            self.transformer_core = None
            core_out = cfg.lstm_hidden
        else:
            self.lstm = None
            core_out = H
        
        # === Slot encoders and scorers ===
        self.move_slot_dim = int(cfg.move_slot_dim or 0)
        self.switch_slot_dim = int(cfg.switch_slot_dim or 0)
        # v5: slot input includes embedding concat
        move_slot_in = self.move_slot_dim + (cfg.embed_dim if self.use_embeddings else 0)
        switch_slot_in = self.switch_slot_dim + (cfg.embed_dim if self.use_embeddings else 0)

        if self.move_slot_dim > 0:
            self.move_enc = nn.Sequential(
                nn.Linear(move_slot_in, cfg.slot_hidden),
                nn.ReLU(),
            )
            self.move_score = nn.Linear(cfg.slot_hidden + core_out, 1)  # per-slot
        else:
            self.move_enc = None
            self.move_score = None

        if self.switch_slot_dim > 0:
            self.switch_enc = nn.Sequential(
                nn.Linear(switch_slot_in, cfg.slot_hidden),
                nn.ReLU(),
            )
            self.switch_score = nn.Linear(cfg.slot_hidden + core_out, 1)
        else:
            self.switch_enc = None
            self.switch_score = None
        
        # Action heads
        if not cfg.hierarchical:
            self.action_head = nn.Linear(core_out, cfg.action_dim)   # flat 9
            self.mv_head = self.sw_head = None
        else:
            self.mvs_head = nn.Linear(core_out, 2)    # Move vs Switch
            self.mv_head = nn.Linear(core_out, 4)     # which move (0..3)
            self.sw_head = nn.Linear(core_out, 5)     # which bench (0..4)
            self.action_head = None

        # Value / Aux
        self.value_head  = nn.Linear(core_out, 1)
        self.aux_move_first = nn.Linear(core_out, 1)

        # Modifiers
        mods = []
        for m in (cfg.modifiers or []):
            if isinstance(m, dict):
                name = m.get("name")
            else:
                name = getattr(m, "name", None)
            if name:
                mods.append((name, nn.Linear(core_out, 1)))
        self.mod_heads = nn.ModuleDict(mods)

    def _embed_entity_ids(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """Process entity_ids [B,N] or [B,T,N] → pooled embedding vector [B,G*E] or [B,T,G*E].
        G = number of groups (19), E = embed_dim."""
        embed_tables = {
            "species": self.species_embed,
            "move": self.move_embed,
            "item": self.item_embed,
            "ability": self.ability_embed,
        }
        parts = []
        for _name, start, end, etype in self._ENTITY_GROUPS:
            ids = entity_ids[..., start:end].long()  # [..., n_ids_in_group]
            emb = embed_tables[etype](ids)           # [..., n_ids, E]
            # Sum-pool across IDs in this group (handles both single and multi-ID groups)
            pooled = emb.sum(dim=-2)                 # [..., E]
            parts.append(pooled)
        return torch.cat(parts, dim=-1)              # [..., G*E]

    def _embed_slot_ids(self, slot_ids: torch.Tensor, embed_type: str) -> torch.Tensor:
        """Embed slot IDs [B,K] or [B,T,K] → [B,K,E] or [B,T,K,E]."""
        embed_tables = {
            "species": self.species_embed,
            "move": self.move_embed,
        }
        return embed_tables[embed_type](slot_ids.long())

    def _concat_context(self, obs: torch.Tensor, step_type: Optional[torch.Tensor]=None,
                        ctx_extra: Optional[torch.Tensor]=None,
                        entity_ids: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        obs: [B,F] or [B,T,F]; step_type: [B] or [B,T] ints; ctx_extra: [B,?] or [B,T,?]
        Returns matching shape with extra channels concatenated.
        """
        x = obs
        add = []
        if self.step_type_emb is not None and step_type is not None:
            emb = self.step_type_emb(step_type.clamp_min(0).clamp_max(self.cfg.step_type_bins-1).long())
            add.append(emb)
        if self.ctx_proj is not None and ctx_extra is not None:
            proj = F.relu(self.ctx_proj(ctx_extra))
            add.append(proj)
        if self.use_embeddings and entity_ids is not None:
            embed_vec = self._embed_entity_ids(entity_ids)
            add.append(embed_vec)
        if add:
            return torch.cat([x] + add, dim=-1)
        return x

    def _prepare_slot_input(self, slots: torch.Tensor, slot_ids: Optional[torch.Tensor], embed_type: str) -> torch.Tensor:
        """Concat slot continuous features with slot ID embeddings if available."""
        if self.use_embeddings and slot_ids is not None:
            emb = self._embed_slot_ids(slot_ids, embed_type)  # [..., K, E]
            return torch.cat([slots, emb], dim=-1)            # [..., K, M+E]
        return slots

    def _core(self, x: torch.Tensor, seq_lens: Optional[torch.Tensor]=None,
              h0: Optional[Tuple[torch.Tensor,torch.Tensor]]=None):
        """Run MLP encoder + sequence core (transformer, LSTM, or none).

        Args:
            x: input tensor [B,F] or [B,T,F]
            seq_lens: optional [B] int tensor of actual sequence lengths per batch element.
                      If provided and LSTM is active, uses pack_padded_sequence /
                      pad_packed_sequence so the LSTM does not process padding tokens.
                      For transformer, used as padding mask.
                      If None, processes the full sequence (legacy behavior).
            h0: optional initial LSTM hidden state tuple (h, c). Ignored for transformer.
        """
        # Transformer path
        if self.transformer_core is not None:
            single = (x.dim() == 2)
            if single: x = x.unsqueeze(1)
            x = self.obs_encoder(x)
            out = self.transformer_core(x, seq_lens)
            if single: out = out[:, -1, :]
            return out, None

        # MLP-only path (no LSTM, no transformer)
        if self.lstm is None:
            out = self.obs_encoder(x)
            return out, None

        # LSTM path
        single = (x.dim() == 2)
        if single: x = x.unsqueeze(1)
        x = self.obs_encoder(x)
        if seq_lens is not None:
            # Pack sequences so LSTM skips padding timesteps
            seq_lens_cpu = seq_lens.cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, seq_lens_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, h_n = self.lstm(packed, h0)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, h_n = self.lstm(x, h0)
        if single: out = out[:, -1, :]
        return out, h_n

    def forward(self, obs: torch.Tensor,

                action_mask: Optional[torch.Tensor]=None,   # [B,A] or [B,T,A]
                mod_legal: Optional[Dict[str, torch.Tensor]]=None,
                seq_lens: Optional[torch.Tensor]=None,
                h0: Optional[Tuple[torch.Tensor,torch.Tensor]]=None,
                step_type: Optional[torch.Tensor]=None,
                ctx_extra: Optional[torch.Tensor]=None,
                move_slots: Optional[torch.Tensor]=None,    # [B,4,M] or [B,T,4,M]
                switch_slots: Optional[torch.Tensor]=None,  # [B,5,S] or [B,T,5,S]
                entity_ids: Optional[torch.Tensor]=None,    # [B,N] or [B,T,N] int
                move_ids: Optional[torch.Tensor]=None,      # [B,4] or [B,T,4] int
                switch_ids: Optional[torch.Tensor]=None,    # [B,5] or [B,T,5] int
    ):
        # Sanity: obs last-dim must match configured obs_dim
        assert obs.shape[-1] == self.cfg.obs_dim, f"obs_dim mismatch: got {obs.shape[-1]}, want {self.cfg.obs_dim}"
        # Add optional context channels + entity embeddings
        obs_plus = self._concat_context(obs, step_type=step_type, ctx_extra=ctx_extra, entity_ids=entity_ids)
        core, h_n = self._core(obs_plus, seq_lens, h0)

        if self.cfg.hierarchical:
            mv_logits = self.mvs_head(core)             # [B,2] or [B,T,2]

            if self.move_enc is not None and move_slots is not None:
                move_in = self._prepare_slot_input(move_slots, move_ids, "move")
                if core.dim() == 2:
                    h = core.unsqueeze(1).repeat(1, 4, 1)
                    ms = self.move_enc(move_in)
                    mv_detail = self.move_score(torch.cat([h, ms], dim=-1)).squeeze(-1)
                else:
                    h = core.unsqueeze(2).repeat(1, 1, 4, 1)
                    ms = self.move_enc(move_in)
                    mv_detail = self.move_score(torch.cat([h, ms], dim=-1)).squeeze(-1)
            else:
                mv_detail = self.mv_head(core)

            if self.switch_enc is not None and switch_slots is not None:
                switch_in = self._prepare_slot_input(switch_slots, switch_ids, "species")
                if core.dim() == 2:
                    h = core.unsqueeze(1).repeat(1, 5, 1)
                    ss = self.switch_enc(switch_in)
                    sw_detail = self.switch_score(torch.cat([h, ss], dim=-1)).squeeze(-1)
                else:
                    h = core.unsqueeze(2).repeat(1, 1, 5, 1)
                    ss = self.switch_enc(switch_in)
                    sw_detail = self.switch_score(torch.cat([h, ss], dim=-1)).squeeze(-1)
            else:
                sw_detail = self.sw_head(core)

            mv_bias_move = mv_logits[..., 0:1]
            mv_bias_sw   = mv_logits[..., 1:2]
            move_joint   = mv_detail + mv_bias_move
            switch_joint = sw_detail + mv_bias_sw
            action_logits = torch.cat([move_joint, switch_joint], dim=-1)
        else:
            # Flat action head: use slot encoders when available for per-action scoring.
            # Supports partial slot encoding: if only move_slots or only switch_slots
            # are provided, score those with the encoder and the rest with the linear head.
            have_move_enc = self.move_enc is not None and move_slots is not None
            have_switch_enc = self.switch_enc is not None and switch_slots is not None
            if have_move_enc or have_switch_enc:
                if have_move_enc:
                    move_in = self._prepare_slot_input(move_slots, move_ids, "move")
                    if core.dim() == 2:
                        h_m = core.unsqueeze(1).expand(-1, 4, -1)
                        ms = self.move_enc(move_in)
                        move_logits = self.move_score(torch.cat([h_m, ms], dim=-1)).squeeze(-1)
                    else:
                        h_m = core.unsqueeze(2).expand(-1, -1, 4, -1)
                        ms = self.move_enc(move_in)
                        move_logits = self.move_score(torch.cat([h_m, ms], dim=-1)).squeeze(-1)
                else:
                    # No move encoder: use first 4 logits from linear head
                    move_logits = self.action_head(core)[..., :4]

                if have_switch_enc:
                    switch_in = self._prepare_slot_input(switch_slots, switch_ids, "species")
                    if core.dim() == 2:
                        h_s = core.unsqueeze(1).expand(-1, 5, -1)
                        ss = self.switch_enc(switch_in)
                        sw_logits = self.switch_score(torch.cat([h_s, ss], dim=-1)).squeeze(-1)
                    else:
                        h_s = core.unsqueeze(2).expand(-1, -1, 5, -1)
                        ss = self.switch_enc(switch_in)
                        sw_logits = self.switch_score(torch.cat([h_s, ss], dim=-1)).squeeze(-1)
                else:
                    # No switch encoder: use last 5 logits from linear head
                    sw_logits = self.action_head(core)[..., 4:]

                action_logits = torch.cat([move_logits, sw_logits], dim=-1)
            else:
                action_logits = self.action_head(core)

        if action_mask is not None:
            # Use large finite negative instead of -inf to avoid NaN gradients
            # (-inf causes 0 * -inf = NaN in entropy backprop)
            # Use -6e4 instead of -1e9: safe for AMP float16 (max ~65504)
            MASK_VAL = -6e4
            action_logits = torch.where(action_mask > 0, action_logits, torch.full_like(action_logits, MASK_VAL))

        mods = {}
        for name, head in self.mod_heads.items():
            mlogit = head(core)
            if mod_legal is not None and name in mod_legal:
                legal = mod_legal[name]
                mlogit = torch.where(legal > 0, mlogit, torch.full_like(mlogit, -6e4))
            mods[name] = mlogit

        value = self.value_head(core)
        aux_move_first = self.aux_move_first(core)

        return {
            "action_logits": action_logits,
            "mod_logits": mods,
            "value": value,
            "aux_move_first": aux_move_first,
            "h_n": h_n,
        }

@torch.no_grad()
def greedy_decode(out: dict,
                  is_move_mask: torch.Tensor,
                  mod_legal: Dict[str, torch.Tensor]):
    action = out["action_logits"].argmax(dim=-1)
    chosen_mods = {}
    for name, logit in out["mod_logits"].items():
        p = torch.sigmoid(logit).squeeze(-1)
        legal = mod_legal.get(name, torch.ones_like(p))
        use = (p > 0.5) & (is_move_mask.squeeze(-1) > 0) & (legal.squeeze(-1) > 0)
        chosen_mods[name] = use
    return action, chosen_mods

def ppo_losses(
    out: dict,
    actions: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    action_mask: torch.Tensor,
    mod_labels: Dict[str, torch.Tensor],
    mod_masks: Dict[str, torch.Tensor],
    aux_move_first_label: Optional[torch.Tensor]=None,
    aux_mask: Optional[torch.Tensor]=None,
    clip_eps: float=0.2, ent_coef: float=0.01, vf_coef: float=0.5, aux_coef: float=0.05,
):
    logits = out["action_logits"]
    logp_all = torch.log_softmax(logits, dim=-1)
    logp_a   = logp_all.gather(-1, actions.view(-1,1)).squeeze(-1)
    ratio = (logp_a - old_logp).exp()
    unclipped = ratio * advantages
    clipped   = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
    pi_loss = -torch.mean(torch.min(unclipped, clipped))

    # Entropy over legal actions only: mask illegal slots before softmax
    masked_logits = logits.clone()
    masked_logits[action_mask == 0] = -6e4  # float16-safe (not -1e9 which overflows half)
    masked_probs = torch.softmax(masked_logits, dim=-1)
    masked_logp  = torch.log_softmax(masked_logits, dim=-1)
    ent_terms = masked_probs * masked_logp
    ent_terms = torch.where(torch.isnan(ent_terms), torch.zeros_like(ent_terms), ent_terms)
    ent = -torch.sum(ent_terms, dim=-1).mean()
    ent_loss = -ent_coef * ent

    v = out["value"].squeeze(-1)
    v_loss = vf_coef * F.mse_loss(v, returns)

    mod_loss = 0.0
    for name, mlogit in out["mod_logits"].items():
        y = mod_labels.get(name, None)
        m = mod_masks.get(name, None)
        if y is None or m is None: 
            continue
        if m.sum() > 0:
            mod_loss = mod_loss + F.binary_cross_entropy_with_logits(mlogit[m>0], y[m>0])

    aux_loss = 0.0
    if aux_move_first_label is not None and aux_mask is not None and aux_mask.sum() > 0:
        aux_loss = aux_coef * F.binary_cross_entropy_with_logits(
            out["aux_move_first"][aux_mask>0], aux_move_first_label[aux_mask>0]
        )

    total = pi_loss + ent_loss + v_loss + mod_loss + aux_loss
    comps = {"total": total, "pi": pi_loss, "ent": ent_loss, "v": v_loss, "mods": mod_loss, "aux": aux_loss}
    return total, comps
