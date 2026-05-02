# model_transformer.py
# Pure-transformer rewrite (REWRITE_DESIGN.md §1-§7). V1 = Week 1 deliverable:
# Tokenizer + MoveTokenizer that consume the existing v8 memmap batch format
# (dataset.py:167-285) and emit (B, 212, d_model) for the spatial transformer.
#
# Lives alongside the legacy MLP arch in model.py — does NOT modify it.
# Imports NumericalBank from model.py to avoid duplication; that's a read-only use.
#
# Token layout (212 per turn):
#   0     actor                        learnable
#   1     critic                       learnable
#   2     field                        MLP from 4 banks + 52-dim cont
#   3     transition                   MLP from 2 action embeds + 51-dim cont
#   4     our_active_threat            MLP from active-only summary stats
#   5     opp_active_threat            MLP from active-only summary stats
#   6-209 12 Pokemon × 17 attrs        per-attribute tokens (see below)
#   210   summary scratch 0            learnable
#   211   summary scratch 1            learnable
#
# Per-Pokemon 17-token layout (per slot p in 0..11):
#   0  species   (embed)
#   1  item      (embed)
#   2  ability   (embed)
#   3  type      (MLP from 2 type embeds)
#   4  status    (MLP from status+volatile+paradox+tera flags)
#   5  hp_pct    (NumericalBank)
#   6  boosts    (MLP from 7×13 one-hot)
#   7-12  6 stats (NumericalBank)
#   13-16 4 moves (MoveTokenizer: id + 4 banks + 107-dim flag lookup)

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from features import (
    POKEMON_CONT_DIM, FIELD_CONT_DIM, TRANSITION_CONT_DIM, MOVE_SLOT_CONT_DIM,
    N_TYPES, N_STATUS, N_VOLATILE, N_PARADOX,
)
from model import NumericalBank


# =============================
# Slice offsets in the 285-dim our_pokemon_cont vector
# =============================
# Verified against features.py:_encode_pokemon (lines 341-382):
#   types(19) + status(7) + boosts(91) + active_flag(1) + fainted_flag(1)
#   + volatile(38) + paradox(7) + tera(20) + combat(5) + toxic(1) + future_sight(1)
#   + ability_known(1) + item_known(1) + 4×move_compact(23 each) = 285
_SL_TYPES        = (0,    19)   # 19
_SL_STATUS       = (19,   26)   # 7
_SL_BOOSTS       = (26,   117)  # 91 = 7 stats × 13 buckets
_SL_ACTIVE_FLAG  = (117,  118)
_SL_FAINTED      = (118,  119)
_SL_VOLATILE     = (119,  157)  # 38
_SL_PARADOX      = (157,  164)  # 7
_SL_TERA         = (164,  184)  # 20
_SL_COMBAT       = (184,  189)  # 5
_SL_TOXIC        = (189,  190)
_SL_FUTURESIGHT  = (190,  191)
_SL_VISIBILITY   = (191,  193)  # ability_known, item_known
_SL_MOVE_COMPACT = (193,  285)  # 4 × 23

assert _SL_MOVE_COMPACT[1] == POKEMON_CONT_DIM, \
    f"Slice end {_SL_MOVE_COMPACT[1]} != POKEMON_CONT_DIM {POKEMON_CONT_DIM}"


# =============================
# Token type / slot ID conventions
# =============================
N_PER_POKEMON = 17  # tokens per Pokemon (see header)
N_POKEMON     = 12  # 6 ours + 6 opp
N_BATTLE_STATE = 6  # actor, critic, field, transition, 2 active threats
N_SUMMARY     = 2   # K=2 summary scratch tokens (per §3.2 / §4.2)
N_TOKENS      = N_BATTLE_STATE + N_PER_POKEMON * N_POKEMON + N_SUMMARY  # 6 + 204 + 2 = 212

# Token type IDs (19 types, used by type_id_embed). One per row in §3.1 tables.
TT_SPECIES   = 0
TT_ITEM      = 1
TT_ABILITY   = 2
TT_TYPE      = 3
TT_STATUS    = 4
TT_HP_PCT    = 5
TT_BOOSTS    = 6
TT_STAT_HP   = 7
TT_STAT_ATK  = 8
TT_STAT_DEF  = 9
TT_STAT_SPA  = 10
TT_STAT_SPD  = 11
TT_STAT_SPE  = 12
TT_MOVE      = 13      # all 4 move tokens share this; move_slot_embed disambiguates
TT_ACTOR     = 14
TT_CRITIC    = 15
TT_FIELD     = 16
TT_TRANSITION = 17
TT_SUMMARY   = 18      # both K=2 scratch tokens
TT_THREAT    = 19      # 2 active threat tokens (own + opp side)
N_TOKEN_TYPES = 20

# Pokemon-slot IDs (16 vocab: 0..5 ours, 6..11 opp, 12..15 battle-state). Move-slot
# IDs (4 vocab: 0..3 = which-of-4-moves; non-move tokens use 0 with no effect).
N_POKEMON_SLOTS = 16
N_MOVE_SLOTS    = 4

# Per-Pokemon: where each of the 17 token types maps in the 17-token block.
_PER_POKEMON_TT = [
    TT_SPECIES, TT_ITEM, TT_ABILITY, TT_TYPE, TT_STATUS, TT_HP_PCT, TT_BOOSTS,
    TT_STAT_HP, TT_STAT_ATK, TT_STAT_DEF, TT_STAT_SPA, TT_STAT_SPD, TT_STAT_SPE,
    TT_MOVE, TT_MOVE, TT_MOVE, TT_MOVE,
]
assert len(_PER_POKEMON_TT) == N_PER_POKEMON


# =============================
# Config
# =============================

@dataclass
class TransformerConfig:
    """Mirror of PokeTransformerConfig with new defaults per REWRITE_DESIGN.md §4."""
    # Core dims
    d_model: int = 256
    n_spatial_layers: int = 6
    n_temporal_layers: int = 4
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.05

    # Multi-summary scratch (K=2, design default)
    n_summary_tokens: int = 2

    # d_temporal — design says 256 (per §4.3), keep wider spatial output (K*d_model=512)
    d_temporal: int = 256
    temporal_context: int = 200

    # Bank embedding dims (match current arch)
    bank_dim: int = 16
    bank_dim_small: int = 8
    entity_embed_dim: int = 32

    # Vocab sizes (mirror PokeTransformerConfig defaults; checkpoint may override)
    n_species: int = 1548
    n_moves: int = 953
    n_items: int = 2340
    n_abilities: int = 314

    # Value head
    v_bins: int = 51
    v_min: float = -1.6
    v_max: float = 1.6
    n_actions: int = 9

    # Move flag lookup (set at model init; saved-or-loaded from disk per §6.1 Option A)
    move_flag_dim: int = 107  # output of _project_move_flags["continuous"]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


# =============================
# Move flag lookup builder
# =============================

def build_move_flag_lookup(
    n_moves: int,
    gen: int = 9,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    """Build the (n_moves, 107) flag table + (n_moves, 4) bank table from poke-env.

    Per REWRITE_DESIGN.md §6.1 Option A: deterministic function of move_id alone.
    Indexes the move's max-PP / no-disabled state — battle-state-dependent dims
    (current_pp, disabled, stab) are uniform in the lookup. The trainer feeds the
    same lookup at every call (active and team), so the model sees one canonical
    107-dim per move regardless of slot.

    Returns:
        {
            "flags": (n_moves, 107) float32,
            "banks": (n_moves, 4) int32, columns = bp_int, acc_int, pp_int, priority_int,
            "valid": (n_moves,) bool — True if Move(name, gen) instantiated successfully,
        }
    """
    from vocab import Vocab
    from features import _project_move_flags
    from poke_env.battle import Move

    v = Vocab.load()
    # Build reverse map move_id -> name
    id_to_name = {idx: name for name, idx in v._move.items()}

    flags = torch.zeros(n_moves, 107, dtype=torch.float32)
    banks = torch.zeros(n_moves, 4, dtype=torch.int32)
    valid = torch.zeros(n_moves, dtype=torch.bool)

    n_built = 0
    n_failed = 0
    for mid in range(n_moves):
        if mid == 0:
            # ID 0 = pad/unknown — leave as zeros
            continue
        name = id_to_name.get(mid)
        if name is None:
            continue
        try:
            move = Move(name, gen=gen)
            d = _project_move_flags(move)
        except Exception as e:
            n_failed += 1
            if verbose and n_failed <= 5:
                print(f"  [lookup] move_id={mid} name={name!r}: {type(e).__name__}: {e}")
            continue

        cont = d.get("continuous", [])
        if len(cont) != 107:
            if verbose:
                print(f"  [lookup] move_id={mid} name={name!r}: cont dim {len(cont)} != 107")
            continue
        flags[mid] = torch.tensor(cont, dtype=torch.float32)
        banks[mid, 0] = int(d.get("bp_int", 0))
        banks[mid, 1] = int(d.get("acc_int", 0))
        banks[mid, 2] = int(d.get("pp_int", 0))
        banks[mid, 3] = int(d.get("priority_int", 0))
        valid[mid] = True
        n_built += 1

    if verbose:
        print(f"  [lookup] built {n_built} / {n_moves - 1} moves; {n_failed} failed to instantiate")
    return {"flags": flags, "banks": banks, "valid": valid}


def save_move_flag_lookup(out_path: Path, lookup: Dict[str, torch.Tensor]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lookup, str(out_path))


def load_move_flag_lookup(path: Path) -> Dict[str, torch.Tensor]:
    return torch.load(str(path), weights_only=True, map_location="cpu")


# =============================
# MoveTokenizer
# =============================

class MoveTokenizer(nn.Module):
    """Per move (active or team): id + 4 banks + 107-dim flags → d_model token.

    Per §3.2:
        in_dim = 32 (id embed) + 16 + 16 + 8 + 8 (banks) + 107 (flags) = 187
        Linear(187, d) → GELU → LayerNorm → Linear(d, d) → GELU → LayerNorm
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.move_embed = nn.Embedding(cfg.n_moves, cfg.entity_embed_dim)
        self.bp_bank   = NumericalBank(256, cfg.bank_dim)
        self.acc_bank  = NumericalBank(101, cfg.bank_dim)
        self.pp_bank   = NumericalBank(65,  cfg.bank_dim_small)
        self.prio_bank = NumericalBank(13,  cfg.bank_dim_small)

        in_dim = (cfg.entity_embed_dim + 2 * cfg.bank_dim + 2 * cfg.bank_dim_small
                  + cfg.move_flag_dim)
        assert in_dim == 187, f"MoveTokenizer in_dim {in_dim} != 187"

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )

    def forward(
        self,
        move_id: torch.Tensor,    # (..., n_moves) long
        bp:      torch.Tensor,    # same shape int
        acc:     torch.Tensor,
        pp:      torch.Tensor,
        prio:    torch.Tensor,
        flags:   torch.Tensor,    # (..., n_moves, 107) float
    ) -> torch.Tensor:
        e = self.move_embed(move_id)
        b = torch.cat([
            self.bp_bank(bp), self.acc_bank(acc),
            self.pp_bank(pp), self.prio_bank(prio),
        ], dim=-1)
        x = torch.cat([e, b, flags], dim=-1)
        return self.mlp(x)


# =============================
# Tokenizer (212 tokens / turn)
# =============================

class Tokenizer(nn.Module):
    """Slices the v8 memmap into 212 per-turn tokens at d_model.

    Per REWRITE_DESIGN.md §3.1, §3.3, §6.1.

    The forward consumes a `batch` dict with the keys `_encode_entities`
    (model.py:668-720) consumes today, plus `field_banks` / `transition_ids`.
    See dataset.py:288-370 (`unpack_turn_batch`) for the canonical assembly.
    """

    def __init__(
        self,
        cfg: TransformerConfig,
        move_flag_lookup: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # Move token (shared across all 12 × 4 = 48 move slots)
        self.move_tokenizer = MoveTokenizer(cfg)

        # Entity embeddings → directly d_model (§3.1: token = embedding lookup)
        self.species_embed = nn.Embedding(cfg.n_species,   d)
        self.item_embed    = nn.Embedding(cfg.n_items,     d)
        self.ability_embed = nn.Embedding(cfg.n_abilities, d)

        # Pokemon-level small MLPs (§3.1 / §6.1)
        # type_token: 2 type embeds (cfg.entity_embed_dim each) → MLP → d
        self.type_embed = nn.Embedding(N_TYPES + 1, cfg.entity_embed_dim)  # +1 for "no second type"
        self.type_mlp = nn.Sequential(
            nn.Linear(2 * cfg.entity_embed_dim, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )

        # status_token in_dim = status(7) + volatile(38) + paradox(7) + tera(20) = 72
        status_in = N_STATUS + N_VOLATILE + N_PARADOX + (1 + N_TYPES)
        assert status_in == 72, f"status_in {status_in} != 72"
        self.status_mlp = nn.Sequential(
            nn.Linear(status_in, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )

        # boosts_token in_dim = 91 (7×13 one-hot)
        self.boosts_mlp = nn.Sequential(
            nn.Linear(91, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )

        # hp_pct + 6 stat banks → project bank_dim → d_model (small linear)
        self.hp_bank   = NumericalBank(101, cfg.bank_dim)
        self.stat_bank = NumericalBank(256, cfg.bank_dim)  # shared across 6 stats
        self.bank_proj = nn.Linear(cfg.bank_dim, d)

        # Field token: 4 banks (turn / weather_dur / terrain_dur / tr_dur) + 52-dim cont
        self.turn_bank        = NumericalBank(201, cfg.bank_dim)
        self.weather_dur_bank = NumericalBank(9,   cfg.bank_dim_small)
        self.terrain_dur_bank = NumericalBank(6,   cfg.bank_dim_small)
        self.tr_dur_bank      = NumericalBank(6,   cfg.bank_dim_small)
        field_in = (cfg.bank_dim + 3 * cfg.bank_dim_small + FIELD_CONT_DIM)
        self.field_mlp = nn.Sequential(
            nn.Linear(field_in, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )

        # Transition token: 2 action embeds + 51-dim cont
        # (Use the same combined embedding space as model.py:289 — moves and species can share)
        self.action_embed = nn.Embedding(max(cfg.n_moves, cfg.n_species) + 1, cfg.entity_embed_dim)
        trans_in = 2 * cfg.entity_embed_dim + TRANSITION_CONT_DIM
        self.trans_mlp = nn.Sequential(
            nn.Linear(trans_in, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )

        # Threat tokens (active-only, per §3.2 last paragraph). Each side gets 1.
        # Inputs: max move-type-effectiveness vs current opp + max opp-back threat.
        # In the BC memmap these are the last 2 dims of `active_move_cont` (§features.py
        # MOVE_SLOT_CONT_DIM = 109 = 107 base + 1 type_eff + 1 opp_threat). For our side
        # we have those 2 dims directly per active move; for opp side we don't (memmap
        # doesn't carry an opp-active-move table). Use a learnable "no info" embedding
        # as opp threat for now — design §3.2 acknowledges symmetric construction is
        # for training data parity only, not strictly needed for our policy.
        self.threat_mlp = nn.Sequential(
            nn.Linear(8, d),  # 4 moves × (type_eff, opp_threat) = 8
            nn.GELU(),
            nn.LayerNorm(d),
        )
        self.opp_threat_unknown = nn.Parameter(torch.zeros(d))

        # Battle-state learnable tokens (§3.1 second table)
        self.actor_token   = nn.Parameter(torch.randn(d) * 0.02)
        self.critic_token  = nn.Parameter(torch.randn(d) * 0.02)
        self.summary_scratch = nn.Parameter(torch.randn(N_SUMMARY, d) * 0.02)

        # Position / type / slot embeddings (§3.3)
        self.type_id_embed       = nn.Embedding(N_TOKEN_TYPES,    d)
        self.pokemon_slot_embed  = nn.Embedding(N_POKEMON_SLOTS,  d)
        self.move_slot_embed     = nn.Embedding(N_MOVE_SLOTS + 1, d)  # +1 for non-move (idx 0)

        # Move flag lookup: (n_moves, 107) buffer + (n_moves, 4) bank buffer
        if move_flag_lookup is None:
            flags = torch.zeros(cfg.n_moves, 107, dtype=torch.float32)
            mbanks = torch.zeros(cfg.n_moves, 4, dtype=torch.int32)
        else:
            flags = move_flag_lookup["flags"].float()
            mbanks = move_flag_lookup["banks"].int()
            assert flags.shape == (cfg.n_moves, 107), \
                f"lookup flags {flags.shape} != ({cfg.n_moves}, 107)"
            assert mbanks.shape == (cfg.n_moves, 4), \
                f"lookup banks {mbanks.shape} != ({cfg.n_moves}, 4)"
        self.register_buffer("move_flags_lookup", flags)
        self.register_buffer("move_banks_lookup", mbanks)

        # Precompute ID tensors for the 212-token sequence
        type_ids, poke_slots, move_slots = self._build_position_ids()
        self.register_buffer("type_ids",   type_ids,   persistent=False)
        self.register_buffer("poke_slots", poke_slots, persistent=False)
        self.register_buffer("move_slots", move_slots, persistent=False)

    # ----------------- position-ID layout -----------------

    def _build_position_ids(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build (type_ids, poke_slots, move_slots) of length N_TOKENS=212.

        Layout matches the forward()'s assembly order.
        """
        tt = []
        ps = []
        ms = []  # move_slot is 0 for non-move tokens; 1..4 for moves

        # 0,1: actor, critic — pokemon_slot 12 (battle-state)
        tt.extend([TT_ACTOR, TT_CRITIC])
        ps.extend([12, 12])
        ms.extend([0, 0])

        # 2: field. pokemon_slot 13.
        tt.append(TT_FIELD); ps.append(13); ms.append(0)
        # 3: transition. pokemon_slot 14.
        tt.append(TT_TRANSITION); ps.append(14); ms.append(0)
        # 4: our active threat. pokemon_slot 0 (active is slot 0 of ours).
        tt.append(TT_THREAT); ps.append(0); ms.append(0)
        # 5: opp active threat. pokemon_slot 6.
        tt.append(TT_THREAT); ps.append(6); ms.append(0)

        # 6..209: 12 Pokemon × 17 attribute tokens.
        for p in range(N_POKEMON):  # 0..5 ours, 6..11 opp
            for ti, t_id in enumerate(_PER_POKEMON_TT):
                tt.append(t_id)
                ps.append(p)
                if t_id == TT_MOVE:
                    move_slot = ti - 13  # tokens 13,14,15,16 → 0,1,2,3
                    ms.append(move_slot + 1)
                else:
                    ms.append(0)

        # 210, 211: summary scratch. pokemon_slot 15.
        tt.extend([TT_SUMMARY, TT_SUMMARY])
        ps.extend([15, 15])
        ms.extend([0, 0])

        assert len(tt) == N_TOKENS, f"{len(tt)} != {N_TOKENS}"
        return (
            torch.tensor(tt, dtype=torch.long),
            torch.tensor(ps, dtype=torch.long),
            torch.tensor(ms, dtype=torch.long),
        )

    # ----------------- per-Pokemon token block -----------------

    def _encode_pokemon_block(
        self,
        ids: torch.Tensor,    # (B, 6, 7) long: species, item, ability, 4×move_id
        banks: torch.Tensor,  # (B, 6, 10) long: hp_pct, level, weight, height, 6 stats
        cont: torch.Tensor,   # (B, 6, 285) float
    ) -> torch.Tensor:
        """Returns (B, 6, 17, d_model)."""
        B, P = ids.shape[:2]
        assert P == 6
        d = self.cfg.d_model

        # ---- 3 entity tokens (species/item/ability) ----
        species_tok = self.species_embed(ids[..., 0])  # (B, 6, d)
        item_tok    = self.item_embed   (ids[..., 1])
        ability_tok = self.ability_embed(ids[..., 2])

        # ---- type token: pull 2 type indices from cont[0:19] multi-hot ----
        # Take top-2 by value (multi-hot has 0..2 ones; 0-ones gives indices = 0,0).
        # That's fine: the type_embed at 0 ("NORMAL") with the same pad slot is a known
        # ambiguity for empty slots, but empty slots have ids=0 anyway → masked
        # downstream.
        types_oh = cont[..., _SL_TYPES[0]:_SL_TYPES[1]]  # (B, 6, 19)
        # top-2 by value gives (vals, idx); for slots with only 1 type, the second top-2
        # idx is whatever the next-highest is (usually 0 since rest are 0).
        topv, topi = types_oh.topk(2, dim=-1)
        # If second val is 0 (no second type), use the "no second type" pad index N_TYPES.
        topi = torch.where(topv > 0, topi, torch.full_like(topi, N_TYPES))
        # First slot: keep as-is (always >0 except for empty pokemon).
        topi[..., 0] = torch.where(topv[..., 0] > 0, topi[..., 0],
                                   torch.full_like(topi[..., 0], N_TYPES))
        type_e = self.type_embed(topi)  # (B, 6, 2, e)
        type_in = type_e.flatten(-2, -1)  # (B, 6, 2*e)
        type_tok = self.type_mlp(type_in)

        # ---- status token: status(7) + volatiles(38) + paradox(7) + tera(20) = 72 ----
        status_cont = torch.cat([
            cont[..., _SL_STATUS[0]:_SL_STATUS[1]],
            cont[..., _SL_VOLATILE[0]:_SL_VOLATILE[1]],
            cont[..., _SL_PARADOX[0]:_SL_PARADOX[1]],
            cont[..., _SL_TERA[0]:_SL_TERA[1]],
        ], dim=-1)
        status_tok = self.status_mlp(status_cont)

        # ---- hp_pct token (banks[..., 0]) ----
        hp_e = self.hp_bank(banks[..., 0])
        hp_tok = self.bank_proj(hp_e)

        # ---- boosts token (cont[26:117], 91-dim) ----
        boosts_cont = cont[..., _SL_BOOSTS[0]:_SL_BOOSTS[1]]
        boosts_tok = self.boosts_mlp(boosts_cont)

        # ---- 6 stat tokens (banks[..., 4:10]) ----
        stats_b = banks[..., 4:10]                 # (B, 6, 6) long
        stats_e = self.stat_bank(stats_b)          # (B, 6, 6, bank_dim)
        stat_toks = self.bank_proj(stats_e)        # (B, 6, 6, d)

        # ---- 4 move tokens via MoveTokenizer + lookup ----
        # ids[..., 3:7] is (B, 6, 4) long.
        move_ids = ids[..., 3:7]
        # Lookup: move_flags_lookup is (n_moves, 107) — index by move_id.
        flags = self.move_flags_lookup[move_ids]   # (B, 6, 4, 107)
        mb = self.move_banks_lookup[move_ids].long()  # (B, 6, 4, 4)
        bp_v   = mb[..., 0]
        acc_v  = mb[..., 1]
        pp_v   = mb[..., 2]
        prio_v = mb[..., 3]
        move_toks = self.move_tokenizer(move_ids, bp_v, acc_v, pp_v, prio_v, flags)
        # move_toks: (B, 6, 4, d)

        # Stack all 17 tokens per Pokemon
        out = torch.stack([
            species_tok, item_tok, ability_tok, type_tok, status_tok,
            hp_tok, boosts_tok,
            stat_toks[..., 0, :], stat_toks[..., 1, :], stat_toks[..., 2, :],
            stat_toks[..., 3, :], stat_toks[..., 4, :], stat_toks[..., 5, :],
            move_toks[..., 0, :], move_toks[..., 1, :],
            move_toks[..., 2, :], move_toks[..., 3, :],
        ], dim=-2)  # (B, 6, 17, d)
        return out

    # ----------------- battle-state tokens -----------------

    def _encode_field(
        self,
        field_banks: Dict[str, torch.Tensor],
        field_cont: torch.Tensor,  # (B, 52)
    ) -> torch.Tensor:
        t   = self.turn_bank       (field_banks["turn"])
        wd  = self.weather_dur_bank(field_banks["weather_dur"])
        td  = self.terrain_dur_bank(field_banks["terrain_dur"])
        trd = self.tr_dur_bank     (field_banks["tr_dur"])
        x = torch.cat([t, wd, td, trd, field_cont], dim=-1)
        return self.field_mlp(x)

    def _encode_transition(
        self,
        trans_ids: Dict[str, torch.Tensor],
        trans_cont: torch.Tensor,  # (B, 51)
    ) -> torch.Tensor:
        our = self.action_embed(trans_ids["our_action"])
        opp = self.action_embed(trans_ids["opp_action"])
        x = torch.cat([our, opp, trans_cont], dim=-1)
        return self.trans_mlp(x)

    def _encode_our_threat(self, active_move_cont: torch.Tensor) -> torch.Tensor:
        """active_move_cont is (B, 4, MOVE_SLOT_CONT_DIM=109).
        Last 2 dims (107,108) are type_eff + opp_threat. We use them — the design §3.2
        last paragraph specifies this exactly.
        """
        # Pull the 2 active-only dims for each of 4 moves, flatten → (B, 8)
        eff_threat = active_move_cont[..., -2:]  # (B, 4, 2)
        x = eff_threat.flatten(-2, -1)
        return self.threat_mlp(x)

    # ----------------- forward -----------------

    def forward(self, batch: dict) -> Dict[str, torch.Tensor]:
        """Tokenize one batch of turns into (B, 212, d_model) + position IDs.

        `batch` dict matches `unpack_turn_batch` (dataset.py:288-370). Required keys:
          our_pokemon_ids        (B, 6, 7) long
          our_pokemon_banks      (B, 6, 10) long
          our_pokemon_cont       (B, 6, 285) float
          opp_pokemon_ids        (B, 6, 7) long  (memmap key 'opp_pokemon_move_ids' gives 4-move slice)
          opp_pokemon_banks      (B, 6, 10) long
          opp_pokemon_cont       (B, 6, 285) float
          field_banks            dict[str -> (B,) long]
          field_cont             (B, 52) float
          transition_ids         dict[str -> (B,) long]
          transition_cont        (B, 51) float
          active_move_cont       (B, 4, 109) float — used for our active-threat token
        """
        # ---- 12 Pokemon × 17 attribute tokens = 204 tokens ----
        our = batch["our_pokemon_ids"]
        # The collate stores opp ids as (B, 6, 7) — memmap path. Some callers truncate
        # to (B, 6, 3) (model.py:870 mega_batch line). Handle both.
        opp_ids_full = batch["opp_pokemon_ids"]
        if opp_ids_full.shape[-1] == 3:
            # Old path: split out move_ids elsewhere and reassemble.
            opp_move_ids = batch.get("opp_pokemon_move_ids")
            assert opp_move_ids is not None, "opp_pokemon_ids has no moves; need opp_pokemon_move_ids"
            opp_ids_full = torch.cat([opp_ids_full, opp_move_ids], dim=-1)  # (B, 6, 7)
        if our.shape[-1] == 3:
            our_move_ids = batch.get("our_pokemon_move_ids")
            assert our_move_ids is not None, "our_pokemon_ids has no moves; need our_pokemon_move_ids"
            our = torch.cat([our, our_move_ids], dim=-1)

        our_block = self._encode_pokemon_block(
            our, batch["our_pokemon_banks"], batch["our_pokemon_cont"],
        )  # (B, 6, 17, d)
        opp_block = self._encode_pokemon_block(
            opp_ids_full, batch["opp_pokemon_banks"], batch["opp_pokemon_cont"],
        )  # (B, 6, 17, d)
        # Concat along Pokemon axis → (B, 12, 17, d) → flatten → (B, 204, d)
        all_poke = torch.cat([our_block, opp_block], dim=1).flatten(1, 2)
        B = all_poke.shape[0]
        d = self.cfg.d_model

        # ---- 4 battle-state tokens ----
        actor_t  = self.actor_token .unsqueeze(0).expand(B, d).unsqueeze(1)
        critic_t = self.critic_token.unsqueeze(0).expand(B, d).unsqueeze(1)
        field_t  = self._encode_field(batch["field_banks"], batch["field_cont"]).unsqueeze(1)
        trans_t  = self._encode_transition(batch["transition_ids"], batch["transition_cont"]).unsqueeze(1)

        # ---- 2 active-threat tokens ----
        our_threat_t = self._encode_our_threat(batch["active_move_cont"]).unsqueeze(1)
        # Opp side: memmap doesn't carry opp's active-move continuous. Use learnable
        # "unknown threat" embedding broadcast to (B, 1, d).
        opp_threat_t = self.opp_threat_unknown.unsqueeze(0).expand(B, d).unsqueeze(1)

        # ---- 2 summary scratch tokens ----
        scratch = self.summary_scratch.unsqueeze(0).expand(B, N_SUMMARY, d)

        # ---- Assemble sequence in the layout matching _build_position_ids ----
        seq = torch.cat([
            actor_t, critic_t, field_t, trans_t,
            our_threat_t, opp_threat_t,
            all_poke,
            scratch,
        ], dim=1)  # (B, 212, d)
        assert seq.shape == (B, N_TOKENS, d), \
            f"seq {seq.shape} != ({B}, {N_TOKENS}, {d})"

        # ---- Add positional embeddings ----
        type_e = self.type_id_embed     (self.type_ids  ).unsqueeze(0)  # (1, 212, d)
        poke_e = self.pokemon_slot_embed(self.poke_slots).unsqueeze(0)
        move_e = self.move_slot_embed   (self.move_slots).unsqueeze(0)
        seq = seq + type_e + poke_e + move_e

        return {
            "tokens":     seq,                  # (B, 212, d_model)
            "type_ids":   self.type_ids,        # (212,)
            "poke_slots": self.poke_slots,
            "move_slots": self.move_slots,
        }


# =============================
# CLI: build the lookup table
# =============================

def _cli_build_lookup(out_path: str, n_moves: int = 953, gen: int = 9, verbose: bool = True):
    print(f"Building (n_moves={n_moves}, 107) move flag lookup at gen={gen} -> {out_path}")
    lookup = build_move_flag_lookup(n_moves=n_moves, gen=gen, verbose=verbose)
    save_move_flag_lookup(Path(out_path), lookup)
    n_valid = int(lookup["valid"].sum().item())
    print(f"  Saved. {n_valid} / {n_moves - 1} moves built (id 0 reserved for pad).")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build (n_moves, 107) flag lookup table.")
    p.add_argument("--out", default="data/lookup/move_flags_v1.pt")
    p.add_argument("--n-moves", type=int, default=953)
    p.add_argument("--gen", type=int, default=9)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    _cli_build_lookup(args.out, args.n_moves, args.gen, verbose=not args.quiet)
