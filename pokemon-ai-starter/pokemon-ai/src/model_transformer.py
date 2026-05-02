# model_transformer.py
# Pure-transformer rewrite (REWRITE_DESIGN.md §1-§7). V1 = Week 1 deliverable:
# Tokenizer + MoveTokenizer that consume the existing v8 memmap batch format
# (dataset.py:167-285) and emit (B, N_TOKENS, d_model) for the spatial transformer.
#
# Lives alongside the legacy MLP arch in model.py — does NOT modify it.
# Self-contained: NumericalBank is inlined, _project_move_flags is wrapped via
# `_features_project_move_flags` (the shim isolates the one private import).
#
# V1 SCOPE (per design §1 N6): singles, gen 9. The architecture takes a
# FormatConfig and threads `team_size` / `n_active` / `n_types` / `n_stats`
# through, but doubles/triples (`n_active > 1`) needs further design and is
# explicitly rejected at construction time.
#
# Token layout (V1 singles, total = N_TOKENS):
#   battle-state (N_BATTLE_STATE = 6):
#     0  actor                      learnable
#     1  critic                     learnable
#     2  field                      MLP from 4 banks + FIELD_CONT_DIM cont
#     3  transition                 MLP from 2 action embeds + TRANSITION_CONT_DIM cont
#     4  our_active_threat          MLP from active-only summary stats
#     5  opp_active_threat          learnable "unknown" (memmap has no opp threat)
#   per-Pokemon (N_PER_POKEMON = 17, repeated for 2 * team_size = 12 Pokemon):
#     0  species   (embed)
#     1  item      (embed)
#     2  ability   (embed)
#     3  type      (MLP from 2 type embeds, sorted ascending for determinism)
#     4  status    (MLP from status+volatile+paradox+tera flags)
#     5  hp_pct    (NumericalBank)
#     6  boosts    (MLP from 7×13 one-hot)
#     7-12  6 stats (NumericalBank, shared across stats)
#     13-16 4 moves (MoveTokenizer: id + 4 banks + 107-dim flag lookup)
#   summary scratch (N_SUMMARY = 2): learnable scratch tokens

from __future__ import annotations
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from features import (
    POKEMON_CONT_DIM, FIELD_CONT_DIM, TRANSITION_CONT_DIM, MOVE_SLOT_CONT_DIM,
    N_TYPES, N_STATUS, N_VOLATILE, N_PARADOX,
)
from format_config import FormatConfig, FORMAT_SINGLES


_log = logging.getLogger("pokemon_ai")


# =============================
# Lookup schema
# =============================
# Bumped when the (n_moves, 107) contract changes (e.g., features.py adds a
# move-flag dim, gen support shifts). Saved into the .pt file; loader rejects
# mismatches loudly.
LOOKUP_SCHEMA_VERSION = 1
MOVE_FLAG_DIM = 107          # dim of `_project_move_flags(move)["continuous"]`
MOVE_BANK_FIELDS = ("bp_int", "acc_int", "pp_int", "priority_int")  # column order


# =============================
# Slice offsets in the 285-dim our_pokemon_cont vector
# =============================
# Verified against features.py:_encode_pokemon (lines 341-382):
#   types(N_TYPES=19) + status(N_STATUS=7) + boosts(91) + active(1) + fainted(1)
#   + volatile(N_VOLATILE=38) + paradox(N_PARADOX=7) + tera(1+N_TYPES=20)
#   + combat(5) + toxic(1) + future_sight(1) + visibility(2) + 4×move_compact(23) = 285
_BOOSTS_DIM = 91   # 7 stats * 13 buckets
_TERA_DIM   = 1 + N_TYPES  # is_tera flag + tera_type one-hot

_SL_TYPES        = (0,    N_TYPES)
_SL_STATUS       = (N_TYPES, N_TYPES + N_STATUS)
_SL_BOOSTS       = (_SL_STATUS[1], _SL_STATUS[1] + _BOOSTS_DIM)
_SL_ACTIVE_FLAG  = (_SL_BOOSTS[1], _SL_BOOSTS[1] + 1)
_SL_FAINTED      = (_SL_ACTIVE_FLAG[1], _SL_ACTIVE_FLAG[1] + 1)
_SL_VOLATILE     = (_SL_FAINTED[1], _SL_FAINTED[1] + N_VOLATILE)
_SL_PARADOX      = (_SL_VOLATILE[1], _SL_VOLATILE[1] + N_PARADOX)
_SL_TERA         = (_SL_PARADOX[1], _SL_PARADOX[1] + _TERA_DIM)
_SL_COMBAT       = (_SL_TERA[1], _SL_TERA[1] + 5)
_SL_TOXIC        = (_SL_COMBAT[1], _SL_COMBAT[1] + 1)
_SL_FUTURESIGHT  = (_SL_TOXIC[1], _SL_TOXIC[1] + 1)
_SL_VISIBILITY   = (_SL_FUTURESIGHT[1], _SL_FUTURESIGHT[1] + 2)
_SL_MOVE_COMPACT = (_SL_VISIBILITY[1], _SL_VISIBILITY[1] + 4 * 23)

assert _SL_MOVE_COMPACT[1] == POKEMON_CONT_DIM, (
    f"layout end {_SL_MOVE_COMPACT[1]} != POKEMON_CONT_DIM {POKEMON_CONT_DIM}; "
    "features.py changed shape — update slice offsets here"
)


# =============================
# Token type / slot / position constants
# =============================
N_PER_POKEMON  = 17
N_BATTLE_STATE = 6     # actor, critic, field, transition, 2 active threats
N_SUMMARY      = 2     # K=2 summary scratch tokens (per §3.2 / §4.2)
N_THREAT_SIDES = 2     # our + opp active-threat tokens
MAX_TYPES_PER_POKEMON = 2

# Token type IDs — one per row in §3.1 tables, plus the threat-token type.
TT_SPECIES, TT_ITEM, TT_ABILITY, TT_TYPE, TT_STATUS = 0, 1, 2, 3, 4
TT_HP_PCT, TT_BOOSTS = 5, 6
TT_STAT_HP, TT_STAT_ATK, TT_STAT_DEF, TT_STAT_SPA, TT_STAT_SPD, TT_STAT_SPE = 7, 8, 9, 10, 11, 12
TT_MOVE       = 13   # all 4 move tokens share this; move_slot_embed disambiguates
TT_ACTOR      = 14
TT_CRITIC     = 15
TT_FIELD      = 16
TT_TRANSITION = 17
TT_SUMMARY    = 18   # both K=2 scratch tokens
TT_THREAT     = 19   # our + opp active-threat tokens
N_TOKEN_TYPES = 20

# Per-Pokemon: where each of the 17 token types maps in the 17-token block.
_PER_POKEMON_TT = (
    TT_SPECIES, TT_ITEM, TT_ABILITY, TT_TYPE, TT_STATUS, TT_HP_PCT, TT_BOOSTS,
    TT_STAT_HP, TT_STAT_ATK, TT_STAT_DEF, TT_STAT_SPA, TT_STAT_SPD, TT_STAT_SPE,
    TT_MOVE, TT_MOVE, TT_MOVE, TT_MOVE,
)
_FIRST_MOVE_OFFSET = _PER_POKEMON_TT.index(TT_MOVE)
assert len(_PER_POKEMON_TT) == N_PER_POKEMON

# Pokemon-slot-ID values (vocabulary for `pokemon_slot_embed`):
#   0..team_size-1     : our Pokemon slots
#   team_size..2*team_size-1 : opp Pokemon slots
#   followed by named battle-state slots:
PS_SLOT_DECISION   = 0   # actor / critic — relative to "battle-state slot base"
PS_SLOT_FIELD      = 1
PS_SLOT_TRANSITION = 2
PS_SLOT_SUMMARY    = 3
N_BATTLE_STATE_SLOTS = 4


def n_pokemon(fmt: FormatConfig) -> int:
    """Total Pokemon tokens per turn = both sides combined (= 2 * team_size)."""
    return 2 * fmt.team_size


def n_pokemon_slot_vocab(fmt: FormatConfig) -> int:
    """Vocab size for pokemon_slot_embed: 2 * team_size + 4 (named battle-state slots)."""
    return n_pokemon(fmt) + N_BATTLE_STATE_SLOTS


def total_tokens(fmt: FormatConfig) -> int:
    """N_TOKENS = battle-state + per-pokemon block + summary scratch."""
    return N_BATTLE_STATE + N_PER_POKEMON * n_pokemon(fmt) + N_SUMMARY


# Singles default (kept as module-level for tests / readability).
N_TOKENS = total_tokens(FORMAT_SINGLES)


# =============================
# NumericalBank (inlined; mirrors model.py:106-117 to avoid outbound dep on
# legacy module).
# =============================

class NumericalBank(nn.Module):
    """Learned embedding for quantized continuous values. Replaces raw floats
    which cause gradient instability (ps-ppo finding, carried over from the
    legacy MLP architecture)."""

    def __init__(self, num_values: int, bank_dim: int):
        super().__init__()
        self.num_values = num_values
        self.embedding = nn.Embedding(num_values, bank_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x.clamp(0, self.num_values - 1))


# =============================
# Config
# =============================

def _default_format() -> FormatConfig:
    return FORMAT_SINGLES


@dataclass
class TransformerConfig:
    """Architecture hyperparameters. Saved in checkpoints for reproducibility.

    `format_config` carries team layout / type counts / gen. V1 supports only
    `n_active == 1` (singles); the constructor enforces this.
    """
    # Format (team layout, gen, action space)
    format_config: FormatConfig = field(default_factory=_default_format)

    # Core dims (REWRITE_DESIGN.md §4)
    d_model: int = 256
    n_spatial_layers: int = 6
    n_temporal_layers: int = 4
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.05

    # Multi-summary scratch (K=2 per §3.2 / §4.2)
    n_summary_tokens: int = N_SUMMARY

    # Temporal stack
    d_temporal: int = 256
    temporal_context: int = 200

    # Bank + entity embedding dims (mirror legacy)
    bank_dim: int = 16
    bank_dim_small: int = 8
    entity_embed_dim: int = 32

    # Vocab sizes — defaults loaded from `Vocab.load()` if not set explicitly.
    # Provided as fields so old checkpoints keep loading via from_dict.
    n_species: int = 1548
    n_moves: int = 953
    n_items: int = 2340
    n_abilities: int = 314

    # Value head
    v_bins: int = 51
    v_min: float = -1.6
    v_max: float = 1.6

    # Memory / runtime
    gradient_checkpoint: bool = False    # threaded for Week 2 spatial/temporal stack

    # Init std (matches legacy `_init_weights`)
    init_std: float = 0.02

    def to_dict(self) -> dict:
        d = asdict(self)
        # FormatConfig isn't JSON-friendly; expand the bits we need to reconstruct.
        fmt = self.format_config
        d["format_config"] = {
            "battle_format": fmt.battle_format,
            "team_size": fmt.team_size, "n_active": fmt.n_active, "n_bench": fmt.n_bench,
            "n_moves": fmt.n_moves, "n_switches": fmt.n_switches, "n_actions": fmt.n_actions,
            "n_types": fmt.n_types, "n_stats": fmt.n_stats, "gen": fmt.gen,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        out = {k: v for k, v in d.items() if k in valid}
        if isinstance(out.get("format_config"), dict):
            out["format_config"] = FormatConfig(**out["format_config"])
        return cls(**out)

    @classmethod
    def with_vocab_sizes_from_disk(cls, **overrides) -> "TransformerConfig":
        """Construct a config whose vocab sizes match the on-disk Vocab.

        Use this in new code — defaults won't drift if vocab.py is regenerated.
        """
        from vocab import Vocab
        v = Vocab.load()
        kwargs = dict(
            n_species=v.n_species, n_moves=v.n_moves,
            n_items=v.n_items, n_abilities=v.n_abilities,
        )
        kwargs.update(overrides)
        return cls(**kwargs)


# =============================
# Move flag lookup (§6.1 Option A)
# =============================

def _features_project_move_flags(move):
    """Shim for features.py:_project_move_flags. Isolates the one private
    import; if features.py renames the function, only this line breaks."""
    from features import _project_move_flags
    return _project_move_flags(move)


def build_move_flag_lookup(
    n_moves: int,
    gen: int = 9,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Build the (n_moves, 107) flag table + (n_moves, 4) bank table from poke-env.

    Returns a dict suitable for save_move_flag_lookup. Row 0 is reserved for
    pad/unknown and is left zero.
    """
    from vocab import Vocab
    from poke_env.battle import Move

    v = Vocab.load()
    id_to_name = v.id_to_name_map("move")

    flags = torch.zeros(n_moves, MOVE_FLAG_DIM, dtype=torch.float32)
    banks = torch.zeros(n_moves, len(MOVE_BANK_FIELDS), dtype=torch.int32)
    valid = torch.zeros(n_moves, dtype=torch.bool)

    n_built = 0
    n_failed = 0
    for mid in range(1, n_moves):                # 0 = pad
        name = id_to_name.get(mid)
        if name is None:
            continue
        try:
            move = Move(name, gen=gen)
            d = _features_project_move_flags(move)
        except Exception as e:                   # noqa: BLE001 — third-party may raise anything
            n_failed += 1
            if verbose and n_failed <= 5:
                _log.warning("[lookup] move_id=%d name=%r: %s: %s",
                             mid, name, type(e).__name__, e)
            continue

        cont = d.get("continuous", [])
        if len(cont) != MOVE_FLAG_DIM:
            _log.warning("[lookup] move_id=%d name=%r: cont dim %d != %d",
                         mid, name, len(cont), MOVE_FLAG_DIM)
            continue
        flags[mid] = torch.tensor(cont, dtype=torch.float32)
        for ci, field_name in enumerate(MOVE_BANK_FIELDS):
            banks[mid, ci] = int(d.get(field_name, 0))
        valid[mid] = True
        n_built += 1

    if verbose:
        _log.info("[lookup] built %d / %d moves; %d failed to instantiate",
                  n_built, n_moves - 1, n_failed)
    return {
        "flags": flags,
        "banks": banks,
        "valid": valid,
        "meta": {
            "schema_version": LOOKUP_SCHEMA_VERSION,
            "gen": int(gen),
            "vocab_n_moves": int(n_moves),
            "move_flag_dim": MOVE_FLAG_DIM,
            "bank_fields": list(MOVE_BANK_FIELDS),
        },
    }


def save_move_flag_lookup(out_path: Path, lookup: Dict) -> None:
    """Atomic write: tmp file then rename, mirrors the Session 35 atomic-checkpoint
    fix (avoids half-written files if the process crashes mid-save)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(lookup, str(tmp))
    tmp.replace(out_path)


def load_move_flag_lookup(path: Path, expected_n_moves: Optional[int] = None) -> Dict:
    """Load a lookup .pt and verify schema. Raises on schema mismatch.

    `expected_n_moves`: if given, asserts the lookup was built for the same
    vocab size. Caller passes `cfg.n_moves`.
    """
    blob = torch.load(str(path), weights_only=True, map_location="cpu")
    meta = blob.get("meta", {})
    sv = meta.get("schema_version")
    if sv != LOOKUP_SCHEMA_VERSION:
        raise RuntimeError(
            f"Move-flag lookup at {path} has schema_version={sv!r}, code expects "
            f"{LOOKUP_SCHEMA_VERSION}. Rebuild via `python model_transformer.py "
            f"--out {path}`."
        )
    if meta.get("move_flag_dim") != MOVE_FLAG_DIM:
        raise RuntimeError(
            f"Lookup move_flag_dim={meta.get('move_flag_dim')!r} != code {MOVE_FLAG_DIM}"
        )
    lookup_n = int(blob["flags"].shape[0])
    if expected_n_moves is not None and lookup_n != expected_n_moves:
        raise RuntimeError(
            f"Lookup at {path} has n_moves={lookup_n}, but TransformerConfig "
            f"has n_moves={expected_n_moves}. Vocabs are out of sync — rebuild."
        )
    return blob


# =============================
# Init helper
# =============================

def init_module_(module: nn.Module, std: float = 0.02) -> None:
    """In-place initializer that matches PokeTransformer._init_weights
    (model.py:655-666). Idempotent — safe to call after partial overrides."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=std)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


# =============================
# MLP block helper
# =============================

def _mlp_2_layer(in_dim: int, d: int) -> nn.Sequential:
    """Two-layer post-norm MLP used by every per-attribute small MLP."""
    return nn.Sequential(
        nn.Linear(in_dim, d), nn.GELU(), nn.LayerNorm(d),
        nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d),
    )


def _mlp_1_layer(in_dim: int, d: int) -> nn.Sequential:
    """Single-Linear-with-GELU-and-norm; used for trivial projections."""
    return nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.LayerNorm(d))


# =============================
# MoveTokenizer
# =============================

class MoveTokenizer(nn.Module):
    """Per move (active or team): id + 4 banks + 107-dim flags -> d_model token."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.move_embed = nn.Embedding(cfg.n_moves, cfg.entity_embed_dim)
        self.bp_bank   = NumericalBank(256, cfg.bank_dim)
        self.acc_bank  = NumericalBank(101, cfg.bank_dim)
        self.pp_bank   = NumericalBank(65,  cfg.bank_dim_small)
        self.prio_bank = NumericalBank(13,  cfg.bank_dim_small)

        in_dim = (cfg.entity_embed_dim
                  + 2 * cfg.bank_dim + 2 * cfg.bank_dim_small
                  + MOVE_FLAG_DIM)
        self.mlp = _mlp_2_layer(in_dim, cfg.d_model)
        init_module_(self, std=cfg.init_std)

    def forward(
        self,
        move_id: torch.Tensor,
        bp:      torch.Tensor,
        acc:     torch.Tensor,
        pp:      torch.Tensor,
        prio:    torch.Tensor,
        flags:   torch.Tensor,
    ) -> torch.Tensor:
        e = self.move_embed(move_id)
        b = torch.cat([
            self.bp_bank(bp), self.acc_bank(acc),
            self.pp_bank(pp), self.prio_bank(prio),
        ], dim=-1)
        x = torch.cat([e, b, flags], dim=-1)
        return self.mlp(x)


# =============================
# Tokenizer
# =============================

class Tokenizer(nn.Module):
    """Slices the v8 memmap into N_TOKENS per-turn tokens at d_model.

    Per REWRITE_DESIGN.md §3.1, §3.3, §6.1 + Postscript A/B for V1
    refinements. The forward consumes a `batch` dict matching
    `dataset.unpack_turn_batch` (dataset.py:288-370) — see `forward()`
    docstring for the required keys.
    """

    def __init__(
        self,
        cfg: TransformerConfig,
        move_flag_lookup: Optional[Dict] = None,
    ):
        super().__init__()
        fmt = cfg.format_config
        if fmt.n_active != 1:
            raise NotImplementedError(
                f"Tokenizer V1 supports n_active=1 (singles). Got n_active={fmt.n_active}. "
                "Doubles/triples need a redesign of the active-threat token, action head, "
                "and slot-id space - see REWRITE_DESIGN.md section 1 N6."
            )
        self.cfg = cfg
        d = cfg.d_model

        # ---- Move sub-tokenizer ----
        self.move_tokenizer = MoveTokenizer(cfg)

        # ---- 3 entity tokens: species / item / ability ----
        self.species_embed = nn.Embedding(cfg.n_species,   d)
        self.item_embed    = nn.Embedding(cfg.n_items,     d)
        self.ability_embed = nn.Embedding(cfg.n_abilities, d)

        # ---- Type token (MAX_TYPES_PER_POKEMON type embeds, sorted) ----
        # +1 vocab slot for "no second type" (used when a Pokemon has only one
        # type, or is empty). Distinct from the existing "???" type at idx
        # N_TYPES-1; the design wants a dedicated absent-second-type embedding.
        self.type_embed = nn.Embedding(fmt.n_types + 1, cfg.entity_embed_dim)
        self.type_mlp = _mlp_1_layer(MAX_TYPES_PER_POKEMON * cfg.entity_embed_dim, d)

        # ---- Status token ----
        # status(7) + volatile(38) + paradox(7) + tera(20). Derived from slice
        # widths so a features.py change is caught at construction, not runtime.
        status_in = (
            (_SL_STATUS[1]   - _SL_STATUS[0])
            + (_SL_VOLATILE[1] - _SL_VOLATILE[0])
            + (_SL_PARADOX[1]  - _SL_PARADOX[0])
            + (_SL_TERA[1]     - _SL_TERA[0])
        )
        self.status_mlp = _mlp_1_layer(status_in, d)

        # ---- Boosts token (7 stats × 13 buckets one-hot) ----
        boosts_in = _SL_BOOSTS[1] - _SL_BOOSTS[0]
        self.boosts_mlp = _mlp_1_layer(boosts_in, d)

        # ---- HP-pct + 6 stat banks share a projection bank_dim -> d_model ----
        self.hp_bank   = NumericalBank(101, cfg.bank_dim)
        self.stat_bank = NumericalBank(256, cfg.bank_dim)
        self.bank_proj = nn.Linear(cfg.bank_dim, d)

        # ---- Field token: 4 banks (turn / weather_dur / terrain_dur / tr_dur) + cont ----
        self.turn_bank        = NumericalBank(201, cfg.bank_dim)
        self.weather_dur_bank = NumericalBank(9,   cfg.bank_dim_small)
        self.terrain_dur_bank = NumericalBank(6,   cfg.bank_dim_small)
        self.tr_dur_bank      = NumericalBank(6,   cfg.bank_dim_small)
        field_in = cfg.bank_dim + 3 * cfg.bank_dim_small + FIELD_CONT_DIM
        self.field_mlp = _mlp_1_layer(field_in, d)

        # ---- Transition token: shared embedding for moves + species (legacy convention) ----
        self.action_embed = nn.Embedding(max(cfg.n_moves, cfg.n_species) + 1, cfg.entity_embed_dim)
        trans_in = 2 * cfg.entity_embed_dim + TRANSITION_CONT_DIM
        self.trans_mlp = _mlp_1_layer(trans_in, d)

        # ---- Threat tokens ----
        # Our side: features.py appends 2 active-only dims per move
        # (type_eff_vs_opp, opp_threat_back) to MOVE_SLOT_CONT_DIM = 109.
        # We pull those 2 dims for each of fmt.n_moves moves -> 2*n_moves-dim input.
        threat_in = fmt.n_moves * 2
        assert MOVE_SLOT_CONT_DIM == MOVE_FLAG_DIM + 2, (
            f"MOVE_SLOT_CONT_DIM ({MOVE_SLOT_CONT_DIM}) should equal "
            f"MOVE_FLAG_DIM ({MOVE_FLAG_DIM}) + 2 active-only dims; "
            "features.py changed and the threat token wiring needs review."
        )
        self.threat_mlp = _mlp_1_layer(threat_in, d)
        self.opp_threat_unknown = nn.Parameter(torch.zeros(d))

        # ---- Battle-state learnable tokens ----
        self.actor_token   = nn.Parameter(torch.randn(d) * cfg.init_std)
        self.critic_token  = nn.Parameter(torch.randn(d) * cfg.init_std)
        self.summary_scratch = nn.Parameter(torch.randn(N_SUMMARY, d) * cfg.init_std)

        # ---- Position / type / slot embeddings (§3.3) ----
        self.type_id_embed       = nn.Embedding(N_TOKEN_TYPES,                   d)
        self.pokemon_slot_embed  = nn.Embedding(n_pokemon_slot_vocab(fmt),       d)
        self.move_slot_embed     = nn.Embedding(fmt.n_moves + 1,                 d)  # +1 for non-move (idx 0)

        # ---- Move flag + bank lookup ----
        if move_flag_lookup is None:
            flags = torch.zeros(cfg.n_moves, MOVE_FLAG_DIM, dtype=torch.float32)
            mbanks = torch.zeros(cfg.n_moves, len(MOVE_BANK_FIELDS), dtype=torch.int32)
        else:
            flags = move_flag_lookup["flags"].float()
            mbanks = move_flag_lookup["banks"].int()
            if flags.shape != (cfg.n_moves, MOVE_FLAG_DIM):
                raise RuntimeError(
                    f"Lookup flags shape {tuple(flags.shape)} != "
                    f"({cfg.n_moves}, {MOVE_FLAG_DIM})"
                )
            if mbanks.shape != (cfg.n_moves, len(MOVE_BANK_FIELDS)):
                raise RuntimeError(
                    f"Lookup banks shape {tuple(mbanks.shape)} != "
                    f"({cfg.n_moves}, {len(MOVE_BANK_FIELDS)})"
                )
        self.register_buffer("move_flags_lookup", flags)
        self.register_buffer("move_banks_lookup", mbanks)

        # ---- Precomputed position-ID buffers ----
        type_ids, poke_slots, move_slots = self._build_position_ids(fmt)
        self.register_buffer("type_ids",   type_ids,   persistent=False)
        self.register_buffer("poke_slots", poke_slots, persistent=False)
        self.register_buffer("move_slots", move_slots, persistent=False)

        init_module_(self, std=cfg.init_std)

    # ---- Public utilities ----

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---- Position-ID layout ----

    def _build_position_ids(
        self, fmt: FormatConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build (type_ids, poke_slots, move_slots) buffers of length total_tokens(fmt).
        Layout matches the forward()'s assembly order — keep them in lock-step."""
        team = fmt.team_size
        bs_base = 2 * team   # battle-state slot IDs start after both sides' Pokemon slots
        slot_decision = bs_base + PS_SLOT_DECISION
        slot_field    = bs_base + PS_SLOT_FIELD
        slot_trans    = bs_base + PS_SLOT_TRANSITION
        slot_summary  = bs_base + PS_SLOT_SUMMARY

        tt: list[int] = []
        ps: list[int] = []
        ms: list[int] = []

        # 0,1: actor + critic
        tt.extend([TT_ACTOR, TT_CRITIC]); ps.extend([slot_decision] * 2); ms.extend([0, 0])
        # 2: field
        tt.append(TT_FIELD);      ps.append(slot_field);    ms.append(0)
        # 3: transition
        tt.append(TT_TRANSITION); ps.append(slot_trans);    ms.append(0)
        # 4: our active threat — pinned to our active Pokemon slot 0
        tt.append(TT_THREAT);     ps.append(0);             ms.append(0)
        # 5: opp active threat — pinned to opp active Pokemon slot (= team_size)
        tt.append(TT_THREAT);     ps.append(team);          ms.append(0)

        # 6 .. 6 + 2*team*N_PER_POKEMON: per-Pokemon attribute blocks
        for p in range(2 * team):
            for offset, t_id in enumerate(_PER_POKEMON_TT):
                tt.append(t_id)
                ps.append(p)
                if t_id == TT_MOVE:
                    move_slot = offset - _FIRST_MOVE_OFFSET   # 0..n_moves-1
                    ms.append(move_slot + 1)                  # +1: 0 reserved for non-move
                else:
                    ms.append(0)

        # K summary scratch tokens (last)
        tt.extend([TT_SUMMARY] * N_SUMMARY)
        ps.extend([slot_summary] * N_SUMMARY)
        ms.extend([0] * N_SUMMARY)

        expected = total_tokens(fmt)
        assert len(tt) == expected, f"position-ID length {len(tt)} != total_tokens {expected}"
        return (
            torch.tensor(tt, dtype=torch.long),
            torch.tensor(ps, dtype=torch.long),
            torch.tensor(ms, dtype=torch.long),
        )

    # ---- Per-Pokemon block ----

    def _encode_pokemon_block(
        self,
        ids: torch.Tensor,                                         # (B, T_size, 7)
        banks: torch.Tensor,                                       # (B, T_size, 10)
        cont: torch.Tensor,                                        # (B, T_size, POKEMON_CONT_DIM)
        active_real_banks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Returns (B, T_size, N_PER_POKEMON, d_model).

        `active_real_banks` (our side only): override slot 0's move banks with
        real per-turn values from the memmap's `active_move_banks`. Recovers
        the current_pp / disabled signal lost when team moves use the lookup
        (Postscript B). Pass None for opp side.
        """
        team_size = ids.shape[1]
        d = self.cfg.d_model

        # 3 entity tokens
        species_tok = self.species_embed(ids[..., 0])
        item_tok    = self.item_embed   (ids[..., 1])
        ability_tok = self.ability_embed(ids[..., 2])

        # Type token: top-2 indices from the multi-hot, sorted ascending so the
        # MLP sees a deterministic order regardless of topk tie-breaking.
        # Empty / zero-type slots map both indices to fmt.n_types ("absent").
        types_oh = cont[..., _SL_TYPES[0]:_SL_TYPES[1]]                          # (B, T, n_types)
        topv, topi = types_oh.topk(MAX_TYPES_PER_POKEMON, dim=-1)
        absent = torch.full_like(topi, self.cfg.format_config.n_types)
        topi = torch.where(topv > 0, topi, absent)
        topi, _ = topi.sort(dim=-1)                                              # canonical order
        type_e = self.type_embed(topi).flatten(-2, -1)                           # (B, T, k*e)
        type_tok = self.type_mlp(type_e)

        # Status token (status + volatile + paradox + tera)
        status_cont = torch.cat([
            cont[..., _SL_STATUS[0]:_SL_STATUS[1]],
            cont[..., _SL_VOLATILE[0]:_SL_VOLATILE[1]],
            cont[..., _SL_PARADOX[0]:_SL_PARADOX[1]],
            cont[..., _SL_TERA[0]:_SL_TERA[1]],
        ], dim=-1)
        status_tok = self.status_mlp(status_cont)

        # HP%
        hp_tok = self.bank_proj(self.hp_bank(banks[..., 0]))

        # Boosts
        boosts_tok = self.boosts_mlp(cont[..., _SL_BOOSTS[0]:_SL_BOOSTS[1]])

        # 6 stat tokens (banks columns 4..10)
        stats_e = self.stat_bank(banks[..., 4:10])                               # (B, T, 6, bank_dim)
        stat_toks = self.bank_proj(stats_e)                                      # (B, T, 6, d)

        # 4 move tokens via lookup (+ optional real-banks override on slot 0)
        move_ids = ids[..., 3:7]                                                 # (B, T, 4)
        flags = self.move_flags_lookup[move_ids]                                 # (B, T, 4, 107)
        mb = self.move_banks_lookup[move_ids].long()                             # (B, T, 4, 4)

        if active_real_banks is not None:
            # active_real_banks: {bp/acc/pp/prio: (B, n_moves) long}.
            # Replace slot-0 move banks with the trainer's per-turn observed values
            # (Postscript B — restores current_pp / disabled signal for our active).
            real = torch.stack([
                active_real_banks[k] for k in ("bp", "acc", "pp", "prio")
            ], dim=-1).long()                                                    # (B, n_moves, 4)
            mb = mb.clone()
            mb[:, 0, :, :] = real
        bp_v, acc_v, pp_v, prio_v = mb[..., 0], mb[..., 1], mb[..., 2], mb[..., 3]
        move_toks = self.move_tokenizer(move_ids, bp_v, acc_v, pp_v, prio_v, flags)

        # Stack the 17 attribute tokens per Pokemon
        return torch.stack([
            species_tok, item_tok, ability_tok, type_tok, status_tok,
            hp_tok, boosts_tok,
            stat_toks[..., 0, :], stat_toks[..., 1, :], stat_toks[..., 2, :],
            stat_toks[..., 3, :], stat_toks[..., 4, :], stat_toks[..., 5, :],
            move_toks[..., 0, :], move_toks[..., 1, :],
            move_toks[..., 2, :], move_toks[..., 3, :],
        ], dim=-2)

    # ---- Battle-state encoders ----

    def _encode_field(
        self,
        field_banks: Dict[str, torch.Tensor],
        field_cont: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([
            self.turn_bank       (field_banks["turn"]),
            self.weather_dur_bank(field_banks["weather_dur"]),
            self.terrain_dur_bank(field_banks["terrain_dur"]),
            self.tr_dur_bank     (field_banks["tr_dur"]),
            field_cont,
        ], dim=-1)
        return self.field_mlp(x)

    def _encode_transition(
        self,
        trans_ids: Dict[str, torch.Tensor],
        trans_cont: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([
            self.action_embed(trans_ids["our_action"]),
            self.action_embed(trans_ids["opp_action"]),
            trans_cont,
        ], dim=-1)
        return self.trans_mlp(x)

    def _encode_our_threat(self, active_move_cont: torch.Tensor) -> torch.Tensor:
        """`active_move_cont`: (B, n_moves, MOVE_SLOT_CONT_DIM). The trailing 2
        dims are `type_eff_vs_opp` and `opp_threat_back` (features.py:1362-1363)."""
        eff_threat = active_move_cont[..., -2:]                                  # (B, n_moves, 2)
        return self.threat_mlp(eff_threat.flatten(-2, -1))

    # ---- forward ----

    def forward(self, batch: dict) -> Dict[str, torch.Tensor]:
        """Tokenize one batch of turns into (B, N_TOKENS, d_model) + position IDs.

        Required keys in `batch` (matches `dataset.unpack_turn_batch`):
          our_pokemon_ids        (B, team_size, 7) long  — or (B, team_size, 3) + our_pokemon_move_ids
          our_pokemon_banks      (B, team_size, 10) long
          our_pokemon_cont       (B, team_size, POKEMON_CONT_DIM) float
          opp_pokemon_ids / banks / cont — symmetric
          field_banks            dict[str -> (B,) long]: turn / weather_dur / terrain_dur / tr_dur
          field_cont             (B, FIELD_CONT_DIM) float
          transition_ids         dict[str -> (B,) long]: our_action / opp_action
          transition_cont        (B, TRANSITION_CONT_DIM) float
          active_move_cont       (B, n_moves, MOVE_SLOT_CONT_DIM) float — our active 4 moves
        Optional:
          active_move_banks      dict[str -> (B, n_moves) long]: bp / acc / pp / prio.
                                 If present, slot-0 move banks of our side are
                                 overridden with these real-time values
                                 (current_pp + acc/etc.); see Postscript B.
        """
        our_ids = self._fix_ids(batch, "our")
        opp_ids = self._fix_ids(batch, "opp")

        active_real_banks = batch.get("active_move_banks")    # may be None

        our_block = self._encode_pokemon_block(
            our_ids, batch["our_pokemon_banks"], batch["our_pokemon_cont"],
            active_real_banks=active_real_banks,
        )
        opp_block = self._encode_pokemon_block(
            opp_ids, batch["opp_pokemon_banks"], batch["opp_pokemon_cont"],
        )
        all_poke = torch.cat([our_block, opp_block], dim=1).flatten(1, 2)        # (B, 2*team*17, d)
        B = all_poke.shape[0]
        d = self.cfg.d_model

        actor_t  = self.actor_token .unsqueeze(0).expand(B, d).unsqueeze(1)
        critic_t = self.critic_token.unsqueeze(0).expand(B, d).unsqueeze(1)
        field_t  = self._encode_field(batch["field_banks"], batch["field_cont"]).unsqueeze(1)
        trans_t  = self._encode_transition(batch["transition_ids"], batch["transition_cont"]).unsqueeze(1)

        our_threat_t = self._encode_our_threat(batch["active_move_cont"]).unsqueeze(1)
        opp_threat_t = self.opp_threat_unknown.unsqueeze(0).expand(B, d).unsqueeze(1)

        scratch = self.summary_scratch.unsqueeze(0).expand(B, N_SUMMARY, d)

        seq = torch.cat([
            actor_t, critic_t, field_t, trans_t,
            our_threat_t, opp_threat_t,
            all_poke,
            scratch,
        ], dim=1)
        expected_n = total_tokens(self.cfg.format_config)
        assert seq.shape == (B, expected_n, d), \
            f"seq {tuple(seq.shape)} != ({B}, {expected_n}, {d})"

        type_e = self.type_id_embed     (self.type_ids  ).unsqueeze(0)
        poke_e = self.pokemon_slot_embed(self.poke_slots).unsqueeze(0)
        move_e = self.move_slot_embed   (self.move_slots).unsqueeze(0)
        seq = seq + type_e + poke_e + move_e

        return {
            "tokens":     seq,
            "type_ids":   self.type_ids,
            "poke_slots": self.poke_slots,
            "move_slots": self.move_slots,
        }

    # ---- helpers ----

    def _fix_ids(self, batch: dict, side: str) -> torch.Tensor:
        """Return (B, team_size, 7) ids tensor regardless of caller convention.

        Some collate paths split `pokemon_ids[:, :3]` from `pokemon_move_ids[:, 3:7]`
        (model.py:870 mega_batch). We accept both."""
        ids = batch[f"{side}_pokemon_ids"]
        if ids.shape[-1] == 7:
            return ids
        if ids.shape[-1] == 3:
            move_ids = batch.get(f"{side}_pokemon_move_ids")
            if move_ids is None:
                raise KeyError(
                    f"{side}_pokemon_ids has shape {ids.shape}; need either width 7 "
                    f"OR width 3 + {side}_pokemon_move_ids"
                )
            return torch.cat([ids, move_ids], dim=-1)
        raise ValueError(f"{side}_pokemon_ids has unexpected last-dim {ids.shape[-1]}")


# =============================
# CLI: build the lookup table
# =============================

def _cli_build_lookup(out_path: str, n_moves: Optional[int], gen: int, verbose: bool):
    if n_moves is None:
        from vocab import Vocab
        n_moves = Vocab.load().n_moves
    if verbose and not _log.handlers:
        # CLI usage: send INFO+ to stderr if no handler is configured upstream.
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        _log.addHandler(h)
        _log.setLevel(logging.INFO)
    _log.info("Building (n_moves=%d, %d) move flag lookup at gen=%d -> %s",
              n_moves, MOVE_FLAG_DIM, gen, out_path)
    lookup = build_move_flag_lookup(n_moves=n_moves, gen=gen, verbose=verbose)
    save_move_flag_lookup(Path(out_path), lookup)
    _log.info("Saved %d / %d valid moves (id 0 reserved for pad).",
              int(lookup["valid"].sum().item()), n_moves - 1)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build (n_moves, 107) flag lookup table.")
    p.add_argument("--out", default="data/lookup/move_flags_v1.pt")
    p.add_argument("--n-moves", type=int, default=None,
                   help="Override Vocab.load().n_moves (default: pulled from disk vocab).")
    p.add_argument("--gen", type=int, default=9)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    _cli_build_lookup(args.out, args.n_moves, args.gen, verbose=not args.quiet)
