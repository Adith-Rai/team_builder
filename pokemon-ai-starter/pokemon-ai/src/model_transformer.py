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
#   summary scratch (N_SUMMARY = 4): learnable scratch tokens

from __future__ import annotations
import logging
import re
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
# Bumped when the (n_moves, 107) contract or accompanying chart changes.
# Saved into the .pt file; loader rejects mismatches loudly.
#   v1 (Session 46 initial): flags + banks + valid + meta
#   v2 (Session 46 signal-recovery): + (N_TYPES+1, N_TYPES+1) damage_chart for opp threat
#   v3 (Session 46 extra flags, Postscript F): MOVE_FLAG_DIM bumped from 107
#       to 119 by appending 12 structural flags poke-env exposes that
#       _project_move_flags drops (slicing/bullet/bypasssub/pulse/charge/
#       futuremove/ignore_defensive/use_target_offensive/thaws_target/
#       reflectable/gravity/sleep_usable).
#   v4 (Session 46, Postscript G): adds (n_items, ITEM_FEAT_DIM) and
#       (n_abilities, ABILITY_FEAT_DIM) feature lookups parsed from
#       Showdown's items.ts / abilities.ts. Pure structural extraction;
#       no curation. ItemTokenizer/AbilityTokenizer consume these.
LOOKUP_SCHEMA_VERSION = 4
MOVE_FLAG_DIM_BASE = 107     # the slice from `_project_move_flags(move)["continuous"]`
MOVE_FLAG_EXTRA = (
    "slicing",                  # Sharpness ability boosts these
    "bullet",                   # Bulletproof ability immunes these
    "bypasssub",                # Sound + others bypass Substitute
    "pulse",                    # Mega Launcher boosts these
    "charge",                   # 2-turn moves (Solar Beam, Fly, Dive, ...)
    "futuremove",               # Future Sight, Doom Desire — delayed damage
    "ignore_defensive",         # uses target's offensive stat (Foul Play)
    "use_target_offensive",     # alias path for Foul Play
    "thaws_target",             # Flame Wheel-style anti-freeze
    "reflectable",              # Magic Bounce target
    "gravity",                  # gravity-suppressed (Bounce, Fly, ...)
    "sleep_usable",             # Sleep Talk, Snore — usable while asleep
)
MOVE_FLAG_DIM = MOVE_FLAG_DIM_BASE + len(MOVE_FLAG_EXTRA)   # 119
MOVE_BANK_FIELDS = ("bp_int", "acc_int", "pp_int", "priority_int")  # column order

# Postscript G — structural feature dims for items and abilities. Columns
# are documented in `parse_showdown_items` / `parse_showdown_abilities`.
ITEM_FEAT_FIELDS = (
    "is_berry", "is_gem", "is_pokeball", "is_mega_stone", "is_z_crystal",
    "ignore_klutz", "fling_bp_norm", "natural_gift_bp_norm",
)
ITEM_FEAT_DIM = len(ITEM_FEAT_FIELDS)

ABILITY_FEAT_FIELDS = (
    "breakable",          # Mold Breaker / Teravolt etc. bypass these
    "cantsuppress",       # Gastro Acid / Skill Swap can't suppress
    "notrace",            # Trace can't copy
    "notransform",        # Transform/Imposter can't copy
    "no_skill_swap",      # Skill Swap can't move (some abilities)
    "is_permanent",       # isPermanent: true (As One, etc.)
    "suppress_weather",   # Air Lock, Cloud Nine
)
ABILITY_FEAT_DIM = len(ABILITY_FEAT_FIELDS)

# In the 107-dim flag vector, the type one-hot lives at this slice (indices
# 81..99 — see features.py:_project_move_flags lines 1257-1262, the order of
# elements appended to the `continuous` list).
_MOVE_TYPE_ONEHOT_SLICE = (81, 81 + 19)
assert _MOVE_TYPE_ONEHOT_SLICE[1] - _MOVE_TYPE_ONEHOT_SLICE[0] == N_TYPES, (
    "_MOVE_TYPE_ONEHOT_SLICE width must equal N_TYPES; if features.py reordered "
    "the continuous fields, recompute the slice from a fresh _project_move_flags call."
)


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
# Slice offsets in the FIELD_CONT_DIM=52 field cont vector.
# Verified against features.py:_encode_field (lines 618-649). Layout:
#   weather one-hot (5) + terrain one-hot (5) + trick_room (1)
#   + our hazards (4: SR, spikes/3, tspikes/2, web) + opp hazards (4)
#   + our screens (6: 3 × {presence, dur}) + opp screens (6)
#   + tailwind (2: us, opp)
#   + mechanics (17: tera/mega/z/dmax availability + used + dmax_turns
#       + trapped + force_switch + opp_revealed_frac)
#   + alive (2: our/6, opp/6)
# =============================
_FL_WEATHER_OH    = (0,    5)    # weather one-hot
_FL_TERRAIN_OH    = (5,    10)   # terrain one-hot
_FL_TRICK_ROOM    = (10,   11)
_FL_OUR_HAZARDS   = (11,   15)   # SR / spikes / tspikes / web
_FL_OPP_HAZARDS   = (15,   19)
_FL_OUR_SCREENS   = (19,   25)   # 3 screens × (presence, dur)
_FL_OPP_SCREENS   = (25,   31)
_FL_TAILWIND      = (31,   33)   # us, opp
_FL_MECHANICS     = (33,   50)   # 17 dims of one-time-use resources + flags
_FL_ALIVE         = (50,   52)   # our_alive/6, opp_alive/6

assert _FL_ALIVE[1] == FIELD_CONT_DIM, (
    f"field layout end {_FL_ALIVE[1]} != FIELD_CONT_DIM {FIELD_CONT_DIM}; "
    "features.py changed _encode_field — update _FL_* offsets here"
)


# =============================
# Token type / slot / position constants
# =============================
N_PER_POKEMON  = 17
# Battle-state tokens (sequence indices 0..N_BATTLE_STATE-1):
#  0  actor (learnable)
#  1  critic (learnable)
#  2  transition (last-turn events MLP)
#  3  our active threat (per-move type-eff vs opp + threat back, MLP)
#  4  opp active threat (computed from damage chart + lookup, MLP)
#  5  weather  (Postscript E field-token split)
#  6  terrain
#  7  our hazards
#  8  opp hazards
#  9  our screens
# 10  opp screens
# 11  speed-field (tailwind + trick room)
# 12  mechanics (tera/mega/z/dmax availability + trapped/force_switch/opp_revealed_frac)
# 13  progression (turn + alive counts)
N_BATTLE_STATE = 14
# K=4 summary scratch tokens (per §3.2 / §4.2; bumped from K=2 in Session 47
# Postscript H to match Metamon Small's recipe and lift per-turn output to
# 4×d_model=1024-dim feeding the temporal stack).
N_SUMMARY      = 4
N_THREAT_SIDES = 2     # our + opp active-threat tokens
MAX_TYPES_PER_POKEMON = 2

# Token type IDs — one per row in §3.1 tables + threat-token + Postscript E
# field-token split.
TT_SPECIES, TT_ITEM, TT_ABILITY, TT_TYPE, TT_STATUS = 0, 1, 2, 3, 4
TT_HP_PCT, TT_BOOSTS = 5, 6
TT_STAT_HP, TT_STAT_ATK, TT_STAT_DEF, TT_STAT_SPA, TT_STAT_SPD, TT_STAT_SPE = 7, 8, 9, 10, 11, 12
TT_MOVE       = 13   # all 4 move tokens share this; move_slot_embed disambiguates
TT_ACTOR      = 14
TT_CRITIC     = 15
TT_TRANSITION = 16
TT_SUMMARY    = 17   # all K=4 scratch tokens share this token type
TT_THREAT     = 18   # our + opp active-threat tokens
# Field-token split (Postscript E): the legacy single TT_FIELD becomes 9
# thematic tokens. This lets attention specialize — e.g., opp_screens_token
# can attend to our sweeper's stats independently of weather state.
TT_WEATHER     = 19
TT_TERRAIN     = 20
TT_OUR_HAZARDS = 21
TT_OPP_HAZARDS = 22
TT_OUR_SCREENS = 23
TT_OPP_SCREENS = 24
TT_SPEED_FIELD = 25  # tailwind us/opp + trick room (speed-tier modifiers)
TT_MECHANICS   = 26  # one-time-use resources: tera/mega/z/dmax + trapped/force_switch/opp_revealed
TT_PROGRESSION = 27  # turn count + alive counts (game-progression info)
N_TOKEN_TYPES = 28

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
PS_SLOT_DECISION       = 0   # actor / critic — relative to "battle-state slot base"
PS_SLOT_TRANSITION     = 1
PS_SLOT_WEATHER        = 2
PS_SLOT_TERRAIN        = 3
PS_SLOT_OUR_HAZARDS    = 4
PS_SLOT_OPP_HAZARDS    = 5
PS_SLOT_OUR_SCREENS    = 6
PS_SLOT_OPP_SCREENS    = 7
PS_SLOT_SPEED_FIELD    = 8
PS_SLOT_MECHANICS      = 9
PS_SLOT_PROGRESSION    = 10
PS_SLOT_SUMMARY        = 11
N_BATTLE_STATE_SLOTS = 12


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

    # Multi-summary scratch (K=4 per §3.2 / §4.2 + Postscript H bump).
    n_summary_tokens: int = N_SUMMARY

    # Temporal stack — d_temporal=512 matches Metamon Small's `TformerTrajEncoder.d_model`.
    # Bumped from 256 in Session 47 Postscript H per METAMON_LEARNINGS.md §1.2's
    # "shift capacity from spatial to temporal" recommendation. T:S width ratio
    # goes 1.0× → 2.0× (still below Metamon's 5-8× because we run a wider
    # spatial than they do — d_model=256 vs Small's 100).
    d_temporal: int = 512
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


def _extra_move_flags(move) -> list:
    """Postscript F: structural flags `_project_move_flags` drops. Pure
    extraction from poke-env Move attributes / m.flags set — no curation.
    Order MUST match MOVE_FLAG_EXTRA tuple."""
    flags = getattr(move, "flags", None) or set()
    return [
        1.0 if "slicing"               in flags else 0.0,
        1.0 if "bullet"                in flags else 0.0,
        1.0 if "bypasssub"             in flags else 0.0,
        1.0 if "pulse"                 in flags else 0.0,
        1.0 if "charge"                in flags else 0.0,
        1.0 if "futuremove"            in flags else 0.0,
        1.0 if bool(getattr(move, "ignore_defensive", False))     else 0.0,
        1.0 if bool(getattr(move, "use_target_offensive", False)) else 0.0,
        1.0 if bool(getattr(move, "thaws_target", False))         else 0.0,
        1.0 if "reflectable"           in flags else 0.0,
        1.0 if "gravity"               in flags else 0.0,
        1.0 if bool(getattr(move, "sleep_usable", False))         else 0.0,
    ]


# =============================
# Showdown items.ts / abilities.ts parsers (Postscript G)
# =============================
# Pure mechanical extraction of structural fields. No effect-logic curation —
# the actual effect callbacks are JS functions we don't try to interpret.

_SHOWDOWN_DATA_DIR = Path(__file__).parent.parent.parent / "showdown-reference" / "data"


def _split_top_level_blocks(text: str) -> Dict[str, str]:
    """Split a Showdown TS data file into {top-level-id: block-text} pairs.

    Items / abilities are tab-indented top-level keys followed by `{...}`. We
    find the matching closing brace by depth-counting.
    """
    out: Dict[str, str] = {}
    pattern = re.compile(r"^\t(\w+):\s*\{", re.MULTILINE)
    for m in pattern.finditer(text):
        key = m.group(1)
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            c = text[i]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            i += 1
        out[key] = text[start:i - 1]
    return out


def parse_showdown_items(items_ts: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    """Parse Showdown's items.ts into {item_id: {feature_name: value}}.

    Extracted features (per ITEM_FEAT_FIELDS, in order):
      is_berry / is_gem / is_pokeball: top-level bool literals
      is_mega_stone: derived from `megaStone:` field present
      is_z_crystal: derived from `zMove:` or `zMoveType:` field present
      ignore_klutz: bool literal
      fling_bp_norm: fling.basePower / 130 (max ~130)
      natural_gift_bp_norm: naturalGift.basePower / 100 (max 100)
    """
    p = items_ts or (_SHOWDOWN_DATA_DIR / "items.ts")
    text = p.read_text(encoding="utf-8")
    out = {}
    for item_id, block in _split_top_level_blocks(text).items():
        out[item_id] = {
            "is_berry":     1.0 if re.search(r"\bisBerry:\s*true",     block) else 0.0,
            "is_gem":       1.0 if re.search(r"\bisGem:\s*true",       block) else 0.0,
            "is_pokeball":  1.0 if re.search(r"\bisPokeball:\s*true",  block) else 0.0,
            "is_mega_stone": 1.0 if re.search(r"\bmegaStone:\s*\{",     block) else 0.0,
            "is_z_crystal":  1.0 if (re.search(r"\bzMove:\s*\{",        block)
                                     or re.search(r'\bzMoveType:\s*"', block)) else 0.0,
            "ignore_klutz": 1.0 if re.search(r"\bignoreKlutz:\s*true", block) else 0.0,
            "fling_bp_norm":        0.0,
            "natural_gift_bp_norm": 0.0,
        }
        m = re.search(r"\bfling:\s*\{[^}]*\bbasePower:\s*(\d+)", block, re.DOTALL)
        if m:
            out[item_id]["fling_bp_norm"] = min(1.0, int(m.group(1)) / 130.0)
        m = re.search(r"\bnaturalGift:\s*\{[^}]*\bbasePower:\s*(\d+)", block, re.DOTALL)
        if m:
            out[item_id]["natural_gift_bp_norm"] = min(1.0, int(m.group(1)) / 100.0)
    return out


def parse_showdown_abilities(abilities_ts: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    """Parse Showdown's abilities.ts into {ability_id: {feature_name: value}}.

    Extracted features (per ABILITY_FEAT_FIELDS):
      breakable / cantsuppress / notrace / notransform / no_skill_swap:
        bits inside the `flags: { ... }` object (presence of `flagname: 1`).
      is_permanent: top-level `isPermanent: true`
      suppress_weather: top-level `suppressWeather: true`
    """
    p = abilities_ts or (_SHOWDOWN_DATA_DIR / "abilities.ts")
    text = p.read_text(encoding="utf-8")
    out = {}
    for ability_id, block in _split_top_level_blocks(text).items():
        # Pull the flags inner object so we don't false-match field names that
        # appear in callback bodies elsewhere in the block.
        flags_block = ""
        m = re.search(r"\bflags:\s*\{([^}]*)\}", block, re.DOTALL)
        if m:
            flags_block = m.group(1)
        out[ability_id] = {
            "breakable":         1.0 if re.search(r"\bbreakable:\s*1",         flags_block) else 0.0,
            "cantsuppress":      1.0 if re.search(r"\bcantsuppress:\s*1",      flags_block) else 0.0,
            "notrace":           1.0 if re.search(r"\bnotrace:\s*1",           flags_block) else 0.0,
            "notransform":       1.0 if re.search(r"\bnotransform:\s*1",       flags_block) else 0.0,
            "no_skill_swap":     1.0 if re.search(r"\bnoskillswap:\s*1",       flags_block) else 0.0,
            "is_permanent":      1.0 if re.search(r"\bisPermanent:\s*true",    block) else 0.0,
            "suppress_weather":  1.0 if re.search(r"\bsuppressWeather:\s*true", block) else 0.0,
        }
    return out


def build_item_features(n_items: int, vocab) -> torch.Tensor:
    """Build (n_items, ITEM_FEAT_DIM) tensor indexed by Vocab item_id."""
    parsed = parse_showdown_items()
    out = torch.zeros(n_items, ITEM_FEAT_DIM, dtype=torch.float32)
    id_to_name = vocab.id_to_name_map("item")
    for iid in range(1, n_items):
        name = id_to_name.get(iid)
        if name is None:
            continue
        feats = parsed.get(name)
        if feats is None:
            continue
        for ci, fname in enumerate(ITEM_FEAT_FIELDS):
            out[iid, ci] = feats[fname]
    return out


def build_ability_features(n_abilities: int, vocab) -> torch.Tensor:
    """Build (n_abilities, ABILITY_FEAT_DIM) tensor indexed by Vocab ability_id."""
    parsed = parse_showdown_abilities()
    out = torch.zeros(n_abilities, ABILITY_FEAT_DIM, dtype=torch.float32)
    id_to_name = vocab.id_to_name_map("ability")
    for aid in range(1, n_abilities):
        name = id_to_name.get(aid)
        if name is None:
            continue
        feats = parsed.get(name)
        if feats is None:
            continue
        for ci, fname in enumerate(ABILITY_FEAT_FIELDS):
            out[aid, ci] = feats[fname]
    return out


def build_damage_chart(gen: int = 9) -> torch.Tensor:
    """Build a (N_TYPES+1, N_TYPES+1) damage chart from poke-env's GenData.

    Layout: `chart[attacker, defender]` = single-type effectiveness multiplier.
    The trailing row+col (index N_TYPES) is the "absent" type slot used when a
    Pokemon has only one type — multiplier 1.0 for any pairing (no-op factor
    in the t1*t2 product).

    Index mapping matches features.py's `_TYPES` list (NORMAL=0, ..., ???=18).
    Used for opp-side threat computation: `eff = chart[move_t, our_t1] * chart[move_t, our_t2]`.

    Saved into the lookup .pt; built from the same gen as the move flags.
    """
    from poke_env.battle import PokemonType
    from poke_env.data import GenData

    chart_dict = GenData.from_gen(gen).type_chart   # {defender_str: {attacker_str: mult}}
    n = N_TYPES + 1
    out = torch.ones(n, n, dtype=torch.float32)     # default 1.0 (covers ??? + absent)

    # features.py:_TYPES drives our index ↔ string mapping (line 80-84).
    # Rebuild the same list here to avoid re-importing a private constant.
    type_names = ["NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE", "FIGHTING",
                  "POISON", "GROUND", "FLYING", "PSYCHIC", "BUG", "ROCK", "GHOST",
                  "DRAGON", "DARK", "STEEL", "FAIRY", "???"]
    assert len(type_names) == N_TYPES, "type_names must match features.py:_TYPES order"

    for atk_idx, atk_name in enumerate(type_names):
        for def_idx, def_name in enumerate(type_names):
            # poke-env chart is keyed [defender][attacker]. ??? is not in the chart;
            # leave both rows/cols as 1.0 (the default).
            mult = chart_dict.get(def_name, {}).get(atk_name)
            if mult is not None:
                out[atk_idx, def_idx] = float(mult)
    return out


def build_move_flag_lookup(
    n_moves: int,
    gen: int = 9,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Build the lookup .pt blob: move flags + banks + damage chart + item /
    ability feature tables. Row 0 of each table is reserved for pad/unknown
    and stays zero.

    Returns a dict suitable for save_move_flag_lookup. Schema v4.
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
        if len(cont) != MOVE_FLAG_DIM_BASE:
            _log.warning("[lookup] move_id=%d name=%r: cont dim %d != %d",
                         mid, name, len(cont), MOVE_FLAG_DIM_BASE)
            continue
        # Concatenate base 107 dim from features.py with the 12 extra structural
        # flags poke-env exposes directly (Postscript F).
        flags[mid, :MOVE_FLAG_DIM_BASE] = torch.tensor(cont, dtype=torch.float32)
        flags[mid, MOVE_FLAG_DIM_BASE:] = torch.tensor(
            _extra_move_flags(move), dtype=torch.float32,
        )
        for ci, field_name in enumerate(MOVE_BANK_FIELDS):
            banks[mid, ci] = int(d.get(field_name, 0))
        valid[mid] = True
        n_built += 1

    if verbose:
        _log.info("[lookup] built %d / %d moves; %d failed to instantiate",
                  n_built, n_moves - 1, n_failed)

    chart = build_damage_chart(gen)
    if verbose:
        _log.info("[lookup] damage chart built: shape=%s, sample fire->water=%.1f, "
                  "ghost->normal=%.1f", tuple(chart.shape),
                  chart[1, 2].item(), chart[13, 0].item())

    # Sanity: verify a few well-known move types are correctly recoverable from
    # the lookup's type-onehot slice — catches a features.py reorder of the
    # 107-dim continuous fields.
    if verbose and n_built > 0:
        v = Vocab.load()
        for known_move, expected_type_idx in [
            ("earthquake", 8),    # GROUND
            ("flamethrower", 1),  # FIRE
            ("psychic", 10),      # PSYCHIC
        ]:
            mid = v.move(known_move)
            if mid > 0 and bool(valid[mid].item()):
                t_oh = flags[mid, _MOVE_TYPE_ONEHOT_SLICE[0]:_MOVE_TYPE_ONEHOT_SLICE[1]]
                got = int(t_oh.argmax().item())
                assert got == expected_type_idx, (
                    f"_MOVE_TYPE_ONEHOT_SLICE seems wrong: {known_move!r} mapped to "
                    f"type idx {got}, expected {expected_type_idx}. features.py reordered."
                )

    # Postscript G: parse Showdown items.ts / abilities.ts.
    item_features = build_item_features(v.n_items, v)
    ability_features = build_ability_features(v.n_abilities, v)
    if verbose:
        n_item_set    = int((item_features.sum(dim=-1) > 0).sum().item())
        n_ability_set = int((ability_features.sum(dim=-1) > 0).sum().item())
        _log.info("[lookup] parsed Showdown items.ts: %d / %d items have at least one structural flag set",
                  n_item_set, v.n_items - 1)
        _log.info("[lookup] parsed Showdown abilities.ts: %d / %d abilities have at least one structural flag set",
                  n_ability_set, v.n_abilities - 1)

    return {
        "flags": flags,
        "banks": banks,
        "valid": valid,
        "damage_chart": chart,
        "item_features": item_features,
        "ability_features": ability_features,
        "meta": {
            "schema_version": LOOKUP_SCHEMA_VERSION,
            "gen": int(gen),
            "vocab_n_moves": int(n_moves),
            "vocab_n_items": int(v.n_items),
            "vocab_n_abilities": int(v.n_abilities),
            "move_flag_dim": MOVE_FLAG_DIM,
            "item_feat_dim": ITEM_FEAT_DIM,
            "ability_feat_dim": ABILITY_FEAT_DIM,
            "bank_fields": list(MOVE_BANK_FIELDS),
            "item_feat_fields": list(ITEM_FEAT_FIELDS),
            "ability_feat_fields": list(ABILITY_FEAT_FIELDS),
            "type_onehot_slice": list(_MOVE_TYPE_ONEHOT_SLICE),
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
    # v2 schema requirements
    if "damage_chart" not in blob:
        raise RuntimeError(
            f"Lookup at {path} missing damage_chart (schema v2). Rebuild."
        )
    expected_chart_shape = (N_TYPES + 1, N_TYPES + 1)
    if tuple(blob["damage_chart"].shape) != expected_chart_shape:
        raise RuntimeError(
            f"damage_chart shape {tuple(blob['damage_chart'].shape)} != "
            f"{expected_chart_shape}"
        )
    # v4 schema requirements
    for k in ("item_features", "ability_features"):
        if k not in blob:
            raise RuntimeError(
                f"Lookup at {path} missing {k} (schema v4). Rebuild."
            )
    if meta.get("item_feat_dim") != ITEM_FEAT_DIM:
        raise RuntimeError(
            f"Lookup item_feat_dim={meta.get('item_feat_dim')!r} != code {ITEM_FEAT_DIM}. "
            "Rebuild lookup with current code."
        )
    if meta.get("ability_feat_dim") != ABILITY_FEAT_DIM:
        raise RuntimeError(
            f"Lookup ability_feat_dim={meta.get('ability_feat_dim')!r} != code {ABILITY_FEAT_DIM}. "
            "Rebuild lookup with current code."
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

class ItemTokenizer(nn.Module):
    """Per-Pokemon item token: id_embed + structural features -> d_model.

    Postscript G: parallel to MoveTokenizer in spirit. The id embedding
    learns item effects from gameplay outcomes; the structural features
    (is_berry/is_gem/is_pokeball/is_mega_stone/is_z_crystal/ignore_klutz/
    fling_bp_norm/natural_gift_bp_norm) are pure structural extraction
    from Showdown items.ts — no curation, derived from true state.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.id_embed = nn.Embedding(cfg.n_items, cfg.entity_embed_dim)
        in_dim = cfg.entity_embed_dim + ITEM_FEAT_DIM
        self.mlp = _mlp_2_layer(in_dim, cfg.d_model)
        init_module_(self, std=cfg.init_std)

    def forward(self, item_id: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        e = self.id_embed(item_id)
        x = torch.cat([e, features], dim=-1)
        return self.mlp(x)


class AbilityTokenizer(nn.Module):
    """Per-Pokemon ability token: id_embed + structural features -> d_model.
    Mirrors ItemTokenizer; structural features parsed from Showdown abilities.ts.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.id_embed = nn.Embedding(cfg.n_abilities, cfg.entity_embed_dim)
        in_dim = cfg.entity_embed_dim + ABILITY_FEAT_DIM
        self.mlp = _mlp_2_layer(in_dim, cfg.d_model)
        init_module_(self, std=cfg.init_std)

    def forward(self, ability_id: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        e = self.id_embed(ability_id)
        x = torch.cat([e, features], dim=-1)
        return self.mlp(x)


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

        # ---- Entity tokens ----
        # species: id-only embedding (no Showdown structural fields are
        # competitive-relevant — types/stats/abilities are already separate tokens)
        self.species_embed = nn.Embedding(cfg.n_species, d)
        # item / ability: id_embed + structural features (Postscript G).
        # Embedding learns effects from gameplay; features give a head start.
        self.item_tokenizer    = ItemTokenizer(cfg)
        self.ability_tokenizer = AbilityTokenizer(cfg)

        # ---- Type token (MAX_TYPES_PER_POKEMON type embeds, sorted) ----
        # +1 vocab slot for "no second type" (used when a Pokemon has only one
        # type, or is empty). Distinct from the existing "???" type at idx
        # N_TYPES-1; the design wants a dedicated absent-second-type embedding.
        self.type_embed = nn.Embedding(fmt.n_types + 1, cfg.entity_embed_dim)
        self.type_mlp = _mlp_1_layer(MAX_TYPES_PER_POKEMON * cfg.entity_embed_dim, d)

        # ---- Per-Pokemon misc-state token (kept name `status_token` for API
        # stability; it now carries all per-Pokemon scalar/flag state).
        # Restored signals per Postscript C — Session 30 weight analysis showed
        # several of these are top-ranked features that the model can't learn
        # implicitly. Inputs:
        #   - status one-hot               (_SL_STATUS, 7)
        #   - volatile flags               (_SL_VOLATILE, 38)
        #   - paradox encoding             (_SL_PARADOX, 7)
        #   - tera flags                   (_SL_TERA, 20)
        #   - active flag                  (1)            -- WAS DROPPED, restored
        #   - fainted flag                 (1)            -- WAS DROPPED, restored
        #   - combat state                 (5)            -- WAS DROPPED, restored
        #   - toxic-escalation counter     (1)            -- WAS DROPPED, restored
        #   - future_sight pending         (1)            -- WAS DROPPED, restored
        #   - visibility (ability/item)    (2)            -- WAS DROPPED, restored
        #   - level / weight / height bank embeds (3 × bank_dim_small)
        # See REWRITE_DESIGN.md Postscript C for rationale.
        status_cont_dims = (
            (_SL_STATUS[1]   - _SL_STATUS[0])
            + (_SL_VOLATILE[1] - _SL_VOLATILE[0])
            + (_SL_PARADOX[1]  - _SL_PARADOX[0])
            + (_SL_TERA[1]     - _SL_TERA[0])
            + 1                                    # active flag
            + 1                                    # fainted flag
            + (_SL_COMBAT[1]   - _SL_COMBAT[0])    # combat (5)
            + 1                                    # toxic
            + 1                                    # future_sight
            + (_SL_VISIBILITY[1] - _SL_VISIBILITY[0])  # visibility (2)
        )
        # Bank embeds for level/weight/height appended at forward time.
        status_in = status_cont_dims + 3 * cfg.bank_dim_small
        self.status_mlp = _mlp_1_layer(status_in, d)

        # Per-Pokemon physical banks (NumericalBank ranges from features.py:
        # level 1..100 -> 100 buckets; weight kg/5 clamped 0..200 -> 201;
        # height m*2 clamped 0..40 -> 41).
        self.level_bank  = NumericalBank(100, cfg.bank_dim_small)
        self.weight_bank = NumericalBank(201, cfg.bank_dim_small)
        self.height_bank = NumericalBank(41,  cfg.bank_dim_small)

        # ---- Boosts token (7 stats × 13 buckets one-hot) ----
        boosts_in = _SL_BOOSTS[1] - _SL_BOOSTS[0]
        self.boosts_mlp = _mlp_1_layer(boosts_in, d)

        # ---- HP-pct + 6 stat banks share a projection bank_dim -> d_model ----
        self.hp_bank   = NumericalBank(101, cfg.bank_dim)
        self.stat_bank = NumericalBank(256, cfg.bank_dim)
        self.bank_proj = nn.Linear(cfg.bank_dim, d)

        # ---- Field tokens (Postscript E split): 9 thematic tokens replacing 1.
        # Each thematic MLP takes a narrow slice of `field_cont` plus relevant
        # banks. Lets attention specialize per theme.
        self.turn_bank        = NumericalBank(201, cfg.bank_dim)
        self.weather_dur_bank = NumericalBank(9,   cfg.bank_dim_small)
        self.terrain_dur_bank = NumericalBank(6,   cfg.bank_dim_small)
        self.tr_dur_bank      = NumericalBank(6,   cfg.bank_dim_small)

        weather_in   = (_FL_WEATHER_OH[1] - _FL_WEATHER_OH[0]) + cfg.bank_dim_small
        terrain_in   = (_FL_TERRAIN_OH[1] - _FL_TERRAIN_OH[0]) + cfg.bank_dim_small
        our_haz_in   = (_FL_OUR_HAZARDS[1] - _FL_OUR_HAZARDS[0])
        opp_haz_in   = (_FL_OPP_HAZARDS[1] - _FL_OPP_HAZARDS[0])
        our_scr_in   = (_FL_OUR_SCREENS[1] - _FL_OUR_SCREENS[0])
        opp_scr_in   = (_FL_OPP_SCREENS[1] - _FL_OPP_SCREENS[0])
        # Speed-field token: tailwind (2) + trick_room (1) + tr_dur bank
        speed_field_in = (_FL_TAILWIND[1] - _FL_TAILWIND[0]) + (_FL_TRICK_ROOM[1] - _FL_TRICK_ROOM[0]) + cfg.bank_dim_small
        mech_in      = (_FL_MECHANICS[1] - _FL_MECHANICS[0])
        # Progression token: alive_us / alive_opp (2) + turn bank
        prog_in      = (_FL_ALIVE[1] - _FL_ALIVE[0]) + cfg.bank_dim

        self.weather_mlp     = _mlp_1_layer(weather_in,     d)
        self.terrain_mlp     = _mlp_1_layer(terrain_in,     d)
        self.our_hazards_mlp = _mlp_1_layer(our_haz_in,     d)
        self.opp_hazards_mlp = _mlp_1_layer(opp_haz_in,     d)
        self.our_screens_mlp = _mlp_1_layer(our_scr_in,     d)
        self.opp_screens_mlp = _mlp_1_layer(opp_scr_in,     d)
        self.speed_field_mlp = _mlp_1_layer(speed_field_in, d)
        self.mechanics_mlp   = _mlp_1_layer(mech_in,        d)
        self.progression_mlp = _mlp_1_layer(prog_in,        d)

        # ---- Transition token: shared embedding for moves + species (legacy convention) ----
        self.action_embed = nn.Embedding(max(cfg.n_moves, cfg.n_species) + 1, cfg.entity_embed_dim)
        trans_in = 2 * cfg.entity_embed_dim + TRANSITION_CONT_DIM
        self.trans_mlp = _mlp_1_layer(trans_in, d)

        # ---- Threat tokens (Postscript C: opp side computed, not zero param) ----
        # Our side: features.py appends 2 active-only dims per move
        # (type_eff_vs_opp, opp_threat_back) to MOVE_SLOT_CONT_DIM = 109.
        # We pull those 2 dims for each of fmt.n_moves moves -> 2*n_moves-dim input.
        assert MOVE_SLOT_CONT_DIM == MOVE_FLAG_DIM_BASE + 2, (
            f"MOVE_SLOT_CONT_DIM ({MOVE_SLOT_CONT_DIM}) should equal "
            f"MOVE_FLAG_DIM_BASE ({MOVE_FLAG_DIM_BASE}) + 2 active-only dims; "
            "features.py changed and the threat token wiring needs review. "
            f"(MOVE_FLAG_DIM={MOVE_FLAG_DIM} is the lookup width including "
            "the Postscript F structural extras.)"
        )
        our_threat_in = fmt.n_moves * 2
        self.our_threat_mlp = _mlp_1_layer(our_threat_in, d)

        # Opp side: we don't have features.py-precomputed threats in the memmap
        # (no opp-active-move table). Compute at forward time from the lookup's
        # type one-hot + the damage chart. Input is fmt.n_moves dims (one
        # effectiveness multiplier per opp move vs our active).
        self.opp_threat_mlp = _mlp_1_layer(fmt.n_moves, d)

        # ---- Battle-state learnable tokens ----
        self.actor_token   = nn.Parameter(torch.randn(d) * cfg.init_std)
        self.critic_token  = nn.Parameter(torch.randn(d) * cfg.init_std)
        self.summary_scratch = nn.Parameter(torch.randn(N_SUMMARY, d) * cfg.init_std)

        # ---- Position / type / slot embeddings (§3.3) ----
        self.type_id_embed       = nn.Embedding(N_TOKEN_TYPES,                   d)
        self.pokemon_slot_embed  = nn.Embedding(n_pokemon_slot_vocab(fmt),       d)
        self.move_slot_embed     = nn.Embedding(fmt.n_moves + 1,                 d)  # +1 for non-move (idx 0)

        # ---- Move + item + ability + chart lookups ----
        if move_flag_lookup is None:
            flags = torch.zeros(cfg.n_moves, MOVE_FLAG_DIM, dtype=torch.float32)
            mbanks = torch.zeros(cfg.n_moves, len(MOVE_BANK_FIELDS), dtype=torch.int32)
            chart = torch.ones(N_TYPES + 1, N_TYPES + 1, dtype=torch.float32)
            item_feats = torch.zeros(cfg.n_items, ITEM_FEAT_DIM, dtype=torch.float32)
            ability_feats = torch.zeros(cfg.n_abilities, ABILITY_FEAT_DIM, dtype=torch.float32)
        else:
            flags = move_flag_lookup["flags"].float()
            mbanks = move_flag_lookup["banks"].int()
            chart = move_flag_lookup["damage_chart"].float()
            item_feats = move_flag_lookup["item_features"].float()
            ability_feats = move_flag_lookup["ability_features"].float()
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
            if chart.shape != (N_TYPES + 1, N_TYPES + 1):
                raise RuntimeError(
                    f"Lookup damage_chart shape {tuple(chart.shape)} != "
                    f"({N_TYPES + 1}, {N_TYPES + 1})"
                )
            if item_feats.shape != (cfg.n_items, ITEM_FEAT_DIM):
                raise RuntimeError(
                    f"Lookup item_features shape {tuple(item_feats.shape)} != "
                    f"({cfg.n_items}, {ITEM_FEAT_DIM})"
                )
            if ability_feats.shape != (cfg.n_abilities, ABILITY_FEAT_DIM):
                raise RuntimeError(
                    f"Lookup ability_features shape {tuple(ability_feats.shape)} != "
                    f"({cfg.n_abilities}, {ABILITY_FEAT_DIM})"
                )
        self.register_buffer("move_flags_lookup",     flags)
        self.register_buffer("move_banks_lookup",     mbanks)
        self.register_buffer("damage_chart",          chart)
        self.register_buffer("item_features_lookup",  item_feats)
        self.register_buffer("ability_features_lookup", ability_feats)

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
        def bs(slot_offset: int) -> int:
            return bs_base + slot_offset

        tt: list[int] = []
        ps: list[int] = []
        ms: list[int] = []

        # Sequence indices 0..N_BATTLE_STATE-1 (must match forward()'s
        # assembly order). 14 battle-state tokens after the field-token split.
        # 0, 1: actor + critic
        tt.extend([TT_ACTOR, TT_CRITIC])
        ps.extend([bs(PS_SLOT_DECISION)] * 2)
        ms.extend([0, 0])
        # 2: transition
        tt.append(TT_TRANSITION)
        ps.append(bs(PS_SLOT_TRANSITION))
        ms.append(0)
        # 3: our active threat — pinned to our active Pokemon slot 0
        tt.append(TT_THREAT)
        ps.append(0)
        ms.append(0)
        # 4: opp active threat — pinned to opp active Pokemon slot (= team_size)
        tt.append(TT_THREAT)
        ps.append(team)
        ms.append(0)
        # 5..13: 9 thematic field tokens (Postscript E split)
        for t_id, p_slot in [
            (TT_WEATHER,     PS_SLOT_WEATHER),
            (TT_TERRAIN,     PS_SLOT_TERRAIN),
            (TT_OUR_HAZARDS, PS_SLOT_OUR_HAZARDS),
            (TT_OPP_HAZARDS, PS_SLOT_OPP_HAZARDS),
            (TT_OUR_SCREENS, PS_SLOT_OUR_SCREENS),
            (TT_OPP_SCREENS, PS_SLOT_OPP_SCREENS),
            (TT_SPEED_FIELD, PS_SLOT_SPEED_FIELD),
            (TT_MECHANICS,   PS_SLOT_MECHANICS),
            (TT_PROGRESSION, PS_SLOT_PROGRESSION),
        ]:
            tt.append(t_id)
            ps.append(bs(p_slot))
            ms.append(0)

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
        ps.extend([bs(PS_SLOT_SUMMARY)] * N_SUMMARY)
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
        active_real_flags: Optional[torch.Tensor] = None,           # (B, n_moves, MOVE_FLAG_DIM)
    ) -> torch.Tensor:
        """Returns (B, T_size, N_PER_POKEMON, d_model).

        Per-Pokemon block: 17 attribute tokens (3 entity + type + status +
        hp + boosts + 6 stats + 4 moves).

        `active_real_banks` (our side only): override slot 0's move banks with
        real per-turn values from the memmap's `active_move_banks`. Recovers
        current_pp / acc / etc. that the lookup uses static defaults for
        (Postscript B).

        `active_real_flags` (our side only): override slot 0's 107-dim move
        flag vector with `active_move_cont[..., :MOVE_FLAG_DIM]`. The memmap
        stores the *real per-turn* values for the 3 dynamic dims (cont[9]
        current_pp, cont[10] disabled, cont[12] stab — STAB depends on the
        user's types). Without this override the model would see stale info
        on those 3 dims for the 4 moves it's about to choose between, which
        is the exact failure mode the A/B/B+ thread (NEXT_SESSION.md L689-870)
        identified for team moves and that we accidentally re-introduced for
        active moves when collapsing to the lookup. Pass None for opp side.
        """
        team_size = ids.shape[1]
        d = self.cfg.d_model

        # 3 entity tokens. species: id-only; item/ability: id + structural flags.
        species_tok = self.species_embed(ids[..., 0])
        item_id     = ids[..., 1]
        ability_id  = ids[..., 2]
        item_feats    = self.item_features_lookup   [item_id]      # (B, T, ITEM_FEAT_DIM)
        ability_feats = self.ability_features_lookup[ability_id]
        item_tok    = self.item_tokenizer   (item_id,    item_feats)
        ability_tok = self.ability_tokenizer(ability_id, ability_feats)

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

        # Status token — Postscript C: all per-Pokemon misc state lives here.
        # Cont slices for status/volatile/paradox/tera + scalar flags
        # (active/fainted/combat/toxic/future_sight/visibility) +
        # bank embeds for level/weight/height.
        status_cont = torch.cat([
            cont[..., _SL_STATUS[0]:_SL_STATUS[1]],
            cont[..., _SL_VOLATILE[0]:_SL_VOLATILE[1]],
            cont[..., _SL_PARADOX[0]:_SL_PARADOX[1]],
            cont[..., _SL_TERA[0]:_SL_TERA[1]],
            cont[..., _SL_ACTIVE_FLAG[0]:_SL_ACTIVE_FLAG[1]],
            cont[..., _SL_FAINTED[0]:_SL_FAINTED[1]],
            cont[..., _SL_COMBAT[0]:_SL_COMBAT[1]],
            cont[..., _SL_TOXIC[0]:_SL_TOXIC[1]],
            cont[..., _SL_FUTURESIGHT[0]:_SL_FUTURESIGHT[1]],
            cont[..., _SL_VISIBILITY[0]:_SL_VISIBILITY[1]],
            self.level_bank (banks[..., 1]),
            self.weight_bank(banks[..., 2]),
            self.height_bank(banks[..., 3]),
        ], dim=-1)
        status_tok = self.status_mlp(status_cont)

        # HP%
        hp_tok = self.bank_proj(self.hp_bank(banks[..., 0]))

        # Boosts
        boosts_tok = self.boosts_mlp(cont[..., _SL_BOOSTS[0]:_SL_BOOSTS[1]])

        # 6 stat tokens (banks columns 4..10)
        stats_e = self.stat_bank(banks[..., 4:10])                               # (B, T, 6, bank_dim)
        stat_toks = self.bank_proj(stats_e)                                      # (B, T, 6, d)

        # 4 move tokens via lookup (+ optional real-banks/flags overrides on slot 0).
        move_ids = ids[..., 3:7]                                                 # (B, T, 4)
        flags = self.move_flags_lookup[move_ids]                                 # (B, T, 4, 107)
        mb = self.move_banks_lookup[move_ids].long()                             # (B, T, 4, 4)

        if active_real_banks is not None:
            # Replace slot-0 move banks with the trainer's per-turn observed values.
            real = torch.stack([
                active_real_banks[k] for k in ("bp", "acc", "pp", "prio")
            ], dim=-1).long()                                                    # (B, n_moves, 4)
            mb = mb.clone()
            mb[:, 0, :, :] = real
        if active_real_flags is not None:
            # Replace slot-0 first-107-dim flags (the `_project_move_flags` slice)
            # with the per-turn version. Static dims match the lookup; the 3
            # dynamic dims (cont[9] current_pp/64, cont[10] disabled, cont[12]
            # stab) get their real values back. The 12 extra structural flags
            # at [MOVE_FLAG_DIM_BASE:] are move-static and stay from the lookup.
            assert active_real_flags.shape[-1] == MOVE_FLAG_DIM_BASE, \
                f"active_real_flags last dim {active_real_flags.shape[-1]} != {MOVE_FLAG_DIM_BASE}"
            flags = flags.clone()
            flags[:, 0, :, :MOVE_FLAG_DIM_BASE] = active_real_flags

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

    def _encode_field_tokens(
        self,
        field_banks: Dict[str, torch.Tensor],
        field_cont: torch.Tensor,        # (B, FIELD_CONT_DIM)
    ) -> Dict[str, torch.Tensor]:
        """Postscript E: produce 9 thematic field tokens, each (B, d_model)."""
        weather_dur_e = self.weather_dur_bank(field_banks["weather_dur"])
        terrain_dur_e = self.terrain_dur_bank(field_banks["terrain_dur"])
        tr_dur_e      = self.tr_dur_bank     (field_banks["tr_dur"])
        turn_e        = self.turn_bank       (field_banks["turn"])

        weather_t = self.weather_mlp(torch.cat([
            field_cont[..., _FL_WEATHER_OH[0]:_FL_WEATHER_OH[1]],
            weather_dur_e,
        ], dim=-1))
        terrain_t = self.terrain_mlp(torch.cat([
            field_cont[..., _FL_TERRAIN_OH[0]:_FL_TERRAIN_OH[1]],
            terrain_dur_e,
        ], dim=-1))
        our_haz_t = self.our_hazards_mlp(field_cont[..., _FL_OUR_HAZARDS[0]:_FL_OUR_HAZARDS[1]])
        opp_haz_t = self.opp_hazards_mlp(field_cont[..., _FL_OPP_HAZARDS[0]:_FL_OPP_HAZARDS[1]])
        our_scr_t = self.our_screens_mlp(field_cont[..., _FL_OUR_SCREENS[0]:_FL_OUR_SCREENS[1]])
        opp_scr_t = self.opp_screens_mlp(field_cont[..., _FL_OPP_SCREENS[0]:_FL_OPP_SCREENS[1]])
        speed_field_t = self.speed_field_mlp(torch.cat([
            field_cont[..., _FL_TAILWIND[0]:_FL_TAILWIND[1]],
            field_cont[..., _FL_TRICK_ROOM[0]:_FL_TRICK_ROOM[1]],
            tr_dur_e,
        ], dim=-1))
        mech_t = self.mechanics_mlp(field_cont[..., _FL_MECHANICS[0]:_FL_MECHANICS[1]])
        prog_t = self.progression_mlp(torch.cat([
            field_cont[..., _FL_ALIVE[0]:_FL_ALIVE[1]],
            turn_e,
        ], dim=-1))

        return {
            "weather":     weather_t,
            "terrain":     terrain_t,
            "our_hazards": our_haz_t,
            "opp_hazards": opp_haz_t,
            "our_screens": our_scr_t,
            "opp_screens": opp_scr_t,
            "speed_field": speed_field_t,
            "mechanics":   mech_t,
            "progression": prog_t,
        }

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
        return self.our_threat_mlp(eff_threat.flatten(-2, -1))

    def _encode_opp_threat(
        self,
        opp_active_move_ids: torch.Tensor,        # (B, n_moves) long — opp slot 0's 4 moves
        our_active_cont: torch.Tensor,            # (B, POKEMON_CONT_DIM) — our slot 0's cont
    ) -> torch.Tensor:
        """Compute opp's max-effectiveness threat per move vs our active.

        Mirrors features.py:_compute_type_effectiveness on a per-move basis,
        but driven entirely by tensor ops on the lookup + damage chart so it
        works at training time when no live Battle object exists.

        Returns (B, d_model) — fed via `opp_threat_mlp`. Empty/unknown opp
        moves (move_id=0) contribute the chart's pad-row -> all-1.0 -> mult
        capped to 0.25 in the normalized range; functionally a "neutral" prior.
        """
        # 1) Opp move types from lookup's type-onehot slice.
        # opp_move_flags shape: (B, n_moves, MOVE_FLAG_DIM)
        opp_move_flags = self.move_flags_lookup[opp_active_move_ids]
        type_oh_slice = self.move_flags_lookup.new_zeros(0)  # silence linters
        a, b = _MOVE_TYPE_ONEHOT_SLICE
        opp_move_type = opp_move_flags[..., a:b].argmax(dim=-1)                  # (B, n_moves) long

        # 2) Our active types from cont multi-hot, sorted canonical (mirrors
        # the type-token logic). For an empty/typeless slot, both indices map
        # to N_TYPES (the chart's "absent" row, multiplier 1.0).
        types_oh = our_active_cont[..., _SL_TYPES[0]:_SL_TYPES[1]]               # (B, n_types)
        topv, topi = types_oh.topk(MAX_TYPES_PER_POKEMON, dim=-1)
        absent = torch.full_like(topi, self.cfg.format_config.n_types)
        topi = torch.where(topv > 0, topi, absent)
        topi, _ = topi.sort(dim=-1)
        our_t1 = topi[..., 0:1]                                                  # (B, 1)
        our_t2 = topi[..., 1:2]                                                  # (B, 1)

        # 3) Multiplier per opp move via chart[atk_type, def_type]. Defender is
        # our active's t1/t2 — multiply for combined typing.
        # damage_chart: (N_TYPES+1, N_TYPES+1). Index [opp_move_type, our_t*].
        chart = self.damage_chart                                                # (n+1, n+1)
        # Gather: chart row by opp_move_type → shape (B, n_moves, n+1).
        rows = chart[opp_move_type]                                              # (B, n_moves, n+1)
        # Pick our_t1 and our_t2 columns. Expand for gather.
        our_t1_exp = our_t1.unsqueeze(-1).expand(-1, opp_move_type.shape[-1], 1) # (B, n_moves, 1)
        our_t2_exp = our_t2.unsqueeze(-1).expand(-1, opp_move_type.shape[-1], 1)
        mult1 = rows.gather(-1, our_t1_exp).squeeze(-1)                          # (B, n_moves)
        mult2 = rows.gather(-1, our_t2_exp).squeeze(-1)
        mult  = mult1 * mult2                                                    # (B, n_moves)

        # 4) Normalize to [0, 1] like features.py:_compute_type_effectiveness
        # (mult/4.0, capped 0..1). Floor at 0.01 except for true-immune (0)
        # so FP16 doesn't lose the immune signal — matches features.py:1297.
        norm = (mult.clamp(0.0, 4.0) / 4.0)                                       # (B, n_moves)
        norm = torch.where(mult > 0, norm.clamp(min=0.01), norm)

        # 5) Mask unknown opp moves (move_id=0) to neutral (0.25 = 1.0 mult / 4).
        # This mirrors features.py's "0.25 default when move type unknown."
        unknown = (opp_active_move_ids == 0).float()
        norm = norm * (1 - unknown) + 0.25 * unknown

        return self.opp_threat_mlp(norm)

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
        # Postscript D: override slot-0's first 107 flag dims with per-turn
        # ground truth from `active_move_cont[..., :MOVE_FLAG_DIM_BASE]`. The
        # trailing 2 dims of `active_move_cont` are the active-only threat
        # scalars (consumed elsewhere); the first 107 dims match the
        # `_project_move_flags` shape with battle-state-correct values for
        # current_pp / disabled / STAB. The 12 extra structural flags at
        # MOVE_FLAG_DIM_BASE:MOVE_FLAG_DIM aren't in active_move_cont and stay
        # from the lookup (they're move-static anyway).
        amc = batch.get("active_move_cont")
        active_real_flags = amc[..., :MOVE_FLAG_DIM_BASE] if amc is not None else None

        our_block = self._encode_pokemon_block(
            our_ids, batch["our_pokemon_banks"], batch["our_pokemon_cont"],
            active_real_banks=active_real_banks,
            active_real_flags=active_real_flags,
        )
        opp_block = self._encode_pokemon_block(
            opp_ids, batch["opp_pokemon_banks"], batch["opp_pokemon_cont"],
        )
        all_poke = torch.cat([our_block, opp_block], dim=1).flatten(1, 2)        # (B, 2*team*17, d)
        B = all_poke.shape[0]
        d = self.cfg.d_model

        actor_t  = self.actor_token .unsqueeze(0).expand(B, d).unsqueeze(1)
        critic_t = self.critic_token.unsqueeze(0).expand(B, d).unsqueeze(1)
        trans_t  = self._encode_transition(batch["transition_ids"], batch["transition_cont"]).unsqueeze(1)

        our_threat_t = self._encode_our_threat(batch["active_move_cont"]).unsqueeze(1)
        # Postscript C: opp threat computed from lookup + damage chart.
        opp_active_move_ids = opp_ids[:, 0, 3:7]                                 # (B, n_moves)
        our_active_cont     = batch["our_pokemon_cont"][:, 0, :]                  # (B, POKEMON_CONT_DIM)
        opp_threat_t = self._encode_opp_threat(
            opp_active_move_ids, our_active_cont,
        ).unsqueeze(1)

        # Postscript E: 9 thematic field tokens.
        f = self._encode_field_tokens(batch["field_banks"], batch["field_cont"])
        field_seq = torch.stack([
            f["weather"], f["terrain"],
            f["our_hazards"], f["opp_hazards"],
            f["our_screens"], f["opp_screens"],
            f["speed_field"], f["mechanics"], f["progression"],
        ], dim=1)                                                                 # (B, 9, d)

        scratch = self.summary_scratch.unsqueeze(0).expand(B, N_SUMMARY, d)

        # Order MUST match _build_position_ids: actor, critic, transition,
        # our_threat, opp_threat, 9 field tokens, then per-Pokemon block, then
        # summary scratch.
        seq = torch.cat([
            actor_t, critic_t, trans_t,
            our_threat_t, opp_threat_t,
            field_seq,
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
# SpatialTransformer (REWRITE_DESIGN.md §4.1, Week 2 deliverable)
# =============================

class SpatialTransformer(nn.Module):
    """Self-attention over the 220 per-turn tokens emitted by Tokenizer.

    Layers / heads / dim per design §4.1: 6 layers × 8 heads × d_model=256,
    pre-norm, ff_mult=4, dropout=0.05.

    Mask: Poke-Mask (model.py:372-384 in spirit, rebuilt for the 220-token
    layout). State and summary tokens cannot attend to actor/critic; actor
    and critic cannot see each other. This preserves the design contract
    that the policy/value reps at the actor/critic positions are not
    contaminated via summary-token relays.

    Side-mask (§3.4): in V1 our memmap yields zero/unknown embeddings for
    unrevealed opp info, so there are NO genuinely "hidden" opp tokens —
    the side-mask would be a logical no-op on top of the Poke-Mask.
    Skipped for V1 to keep the mask matrix simple; revisit when adding
    test-time adaptation or symmetric BC training (per design §3.4).
    """

    def __init__(
        self,
        cfg: TransformerConfig,
        type_ids: torch.Tensor,    # (N,) — from Tokenizer
    ):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.d_model = d
        self.n_tokens = type_ids.shape[0]

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=d * cfg.ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_spatial_layers)

        # Build the Poke-Mask once from the position-IDs the Tokenizer published.
        # Decision tokens are TT_ACTOR / TT_CRITIC; everyone else is "state-or-summary".
        is_decision = (type_ids == TT_ACTOR) | (type_ids == TT_CRITIC)
        # actor / critic each appear exactly once in V1
        if int(is_decision.sum().item()) != 2:
            raise RuntimeError(
                f"SpatialTransformer Poke-Mask expects exactly 2 decision tokens "
                f"(actor + critic), found {int(is_decision.sum().item())}. "
                "Tokenizer position-ID layout drifted."
            )
        actor_idx  = int((type_ids == TT_ACTOR).nonzero(as_tuple=False)[0].item())
        critic_idx = int((type_ids == TT_CRITIC).nonzero(as_tuple=False)[0].item())
        N = self.n_tokens
        mask = torch.zeros(N, N)
        # All non-decision tokens (queries) blocked from decision keys.
        non_decision = ~is_decision
        mask[non_decision, actor_idx]  = float("-inf")
        mask[non_decision, critic_idx] = float("-inf")
        # Actor query can't see critic key, and vice versa.
        mask[actor_idx,  critic_idx] = float("-inf")
        mask[critic_idx, actor_idx]  = float("-inf")
        self.register_buffer("poke_mask", mask, persistent=False)

        init_module_(self, std=cfg.init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model). Returns: (B, N, d_model)."""
        if x.shape[1] != self.n_tokens:
            raise ValueError(
                f"SpatialTransformer expected N={self.n_tokens} tokens, got {x.shape[1]}"
            )
        if self.cfg.gradient_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.transformer, x, self.poke_mask, use_reentrant=False,
            )
        return self.transformer(x, mask=self.poke_mask)


# =============================
# TemporalTransformer (REWRITE_DESIGN.md §4.3, Week 2 deliverable)
# =============================

class TemporalTransformer(nn.Module):
    """Self-attention over (T, d_temporal) per-turn summaries with a causal mask.

    Layers / heads / dim per design §4.3: 4 × 8 × d_temporal=256, pre-norm,
    ff_mult=4, learnable position embedding capped at `temporal_context`.
    Mirrors the legacy `TemporalTransformer` (model.py:437-511) almost
    exactly; `d_temporal` is independent of `d_model` so the spatial summary
    can be wider (K * d_model = 1024 for K=4) while temporal width is 512.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.d_temporal = cfg.d_temporal
        self.temporal_context = cfg.temporal_context

        self.pos_embed = nn.Embedding(cfg.temporal_context, cfg.d_temporal)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_temporal,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_temporal * cfg.ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_temporal_layers)
        init_module_(self, std=cfg.init_std)

    def forward(
        self,
        summaries: torch.Tensor,          # (B, T, d_temporal)
        seq_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = summaries.shape
        device = summaries.device

        if T > self.temporal_context:
            summaries = summaries[:, -self.temporal_context:, :]
            T = self.temporal_context
            if seq_lens is not None:
                seq_lens = seq_lens.clamp(max=T)

        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = summaries + self.pos_embed(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        padding_mask = None
        if seq_lens is not None:
            padding_mask = (
                torch.arange(T, device=device).unsqueeze(0) >= seq_lens.unsqueeze(1)
            )

        if self.cfg.gradient_checkpoint and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self.transformer, x, causal_mask, padding_mask, use_reentrant=False,
            )
        else:
            x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)

        # Last valid timestep per batch item.
        if seq_lens is not None:
            idx = (seq_lens - 1).clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, D)
            return x.gather(1, idx).squeeze(1)
        return x[:, -1, :]


# =============================
# Switch context encoder (action head, switch slots 4-8)
# =============================

class SwitchActionEncoder(nn.Module):
    """Per-bench-Pokemon context for the 5 switch actions.

    Per REWRITE_DESIGN.md Postscript C3: feed `switch_cont[..., -2:]`
    (the per-action defensive_eff / offensive_eff signals — the #1 ranked
    feature in Session 30 weight analysis) DIRECTLY into the action head's
    per-action context. NOT into the spatial token sequence (the
    permutation logic between memmap orderings is fragile).

    For each switch slot j (j in 0..4), the per-action context is
        concat(spatial_pool_of_bench_Pokemon_with_species_id_match,
               switch_cont[:, j, -2:])
    permuted so that switch slot j (in `available_switches` order, used by
    legal_mask) maps to the spatial output of the same Pokemon. The
    permutation gathers via species_id matching; for empty/unavailable
    switches the legal_mask masks the output. See ActionHead.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        # Input: pooled bench token (d) + 2 eff dims = d + 2.
        self.proj = _mlp_1_layer(d + 2, d)
        init_module_(self, std=cfg.init_std)

    def forward(
        self,
        bench_pooled: torch.Tensor,     # (B, 5, d)  — our 5 bench Pokemon attribute pool
        eff: torch.Tensor,              # (B, 5, 2)  — defensive_eff, offensive_eff
    ) -> torch.Tensor:
        x = torch.cat([bench_pooled, eff], dim=-1)
        return self.proj(x)


# =============================
# ActionHead (REWRITE_DESIGN.md §5.1)
# =============================

class ActionHead(nn.Module):
    """9-action policy head: 4 active moves + 5 switches.

    Inputs:
      actor_out      (B, d_model)            — spatial output at actor token
      temporal_ctx   (B, d_temporal)         — temporal stack output
      action_ctx     (B, 9, d_model)         — per-action context (4 moves + 5 switches)
      legal_mask     (B, 9)                  — 1.0 if action legal, 0.0 else

    Output: (B, 9) logits with -100.0 on illegal actions. Identical shape
    convention to legacy `policy_head` (model.py:633-637).
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model
        d_temporal = cfg.d_temporal
        in_dim = 2 * d_model + d_temporal
        hidden = max(d_model, d_temporal)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        init_module_(self, std=cfg.init_std)

    def forward(
        self,
        actor_out: torch.Tensor,         # (B, d_model)
        temporal_ctx: torch.Tensor,      # (B, d_temporal)
        action_ctx: torch.Tensor,        # (B, 9, d_model)
        legal_mask: Optional[torch.Tensor] = None,  # (B, 9) optional
    ) -> torch.Tensor:
        at = torch.cat([actor_out, temporal_ctx], dim=-1)        # (B, d_model+d_temporal)
        at_exp = at.unsqueeze(1).expand(-1, action_ctx.shape[1], -1)
        x = torch.cat([at_exp, action_ctx], dim=-1)              # (B, 9, 2*d_model+d_temporal)
        logits = self.mlp(x).squeeze(-1)                          # (B, 9)
        if legal_mask is not None:
            logits = logits.masked_fill(legal_mask < 0.5, -100.0)
        return logits


# =============================
# ValueHead (REWRITE_DESIGN.md §5.2)
# =============================

class ValueHead(nn.Module):
    """Distributional value head: 51-bin twohot over [v_min, v_max].

    Inputs:
      critic_out   (B, d_model)
      temporal_ctx (B, d_temporal)

    Outputs:
      v_logits (B, v_bins) for distributional cross-entropy training
      value    (B,)        scalar expectation under softmax(v_logits) · v_support
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.d_model + cfg.d_temporal
        hidden = max(cfg.d_model, cfg.d_temporal)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.v_bins),
        )
        self.register_buffer(
            "v_support", torch.linspace(cfg.v_min, cfg.v_max, cfg.v_bins),
        )
        init_module_(self, std=cfg.init_std)

    def forward(
        self,
        critic_out: torch.Tensor,        # (B, d_model)
        temporal_ctx: torch.Tensor,      # (B, d_temporal)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([critic_out, temporal_ctx], dim=-1)
        v_logits = self.mlp(x)                                    # (B, v_bins)
        import torch.nn.functional as F
        v_probs = F.softmax(v_logits, dim=-1)
        value = (v_probs * self.v_support).sum(-1)
        return v_logits, value


# =============================
# TransformerBattlePolicy — top-level model
# =============================

class TransformerBattlePolicy(nn.Module):
    """End-to-end battle policy, interface-compatible with legacy `PokeTransformer`.

    Forward flow:
      1. `Tokenizer` slices the memmap batch into (B, N=220, d_model) tokens.
      2. `SpatialTransformer` self-attends with Poke-Mask -> (B, N, d_model).
      3. K=4 summary scratch token outputs flatten to (B, K*d_model=1024) and
         project to d_temporal=512.
      4. `TemporalTransformer` runs causal attention over [history + this turn]
         summaries and emits (B, d_temporal).
      5. `ActionHead` consumes actor + temporal_ctx + per-action context (4
         spatial-output move tokens permuted to legal_mask order + 5 bench
         pool tokens permuted via species match + switch_cont eff dims).
      6. `ValueHead` consumes critic + temporal_ctx -> 51-bin twohot.
    """

    def __init__(
        self,
        cfg: TransformerConfig,
        move_flag_lookup: Optional[Dict] = None,
    ):
        super().__init__()
        self.cfg = cfg
        # Top-level d_temporal mirrors legacy PokeTransformer convention
        # (model.py:597). Trainer-side helpers (InferenceBatcher) read it via
        # getattr to size the temporal-history buffer.
        self.d_temporal = cfg.d_temporal
        self.tokenizer = Tokenizer(cfg, move_flag_lookup=move_flag_lookup)
        self.spatial = SpatialTransformer(cfg, type_ids=self.tokenizer.type_ids)

        # K summary scratch tokens at the END of the spatial sequence.
        # Flatten K * d_model -> d_temporal.
        K = cfg.n_summary_tokens
        if K < 1:
            raise ValueError(
                f"TransformerBattlePolicy V1 requires n_summary_tokens >= 1; got {K}. "
                "Per design §4.2 + Postscript H, K=4 is the V1 baseline "
                "(matches Metamon Small's recipe)."
            )
        self.summary_to_temporal = nn.Linear(K * cfg.d_model, cfg.d_temporal)

        self.temporal = TemporalTransformer(cfg)
        self.switch_encoder = SwitchActionEncoder(cfg)
        self.action_head = ActionHead(cfg)
        self.value_head = ValueHead(cfg)

        init_module_(self.summary_to_temporal, std=cfg.init_std)

        # Cache the position constants we need at forward time.
        fmt = cfg.format_config
        self._n_tokens = total_tokens(fmt)
        self._our_active_move_start = N_BATTLE_STATE + _FIRST_MOVE_OFFSET   # 14+13=27
        self._our_bench_start = N_BATTLE_STATE + N_PER_POKEMON              # first bench Pokemon
        self._n_pokemon = n_pokemon(fmt)
        self._n_actions = fmt.n_actions
        self._n_moves = fmt.n_moves     # = 4 in V1
        self._n_switches = fmt.n_switches  # = 5 in V1

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---- Action context assembly ----

    def _per_action_context(
        self,
        spatial_out: torch.Tensor,              # (B, N, d_model)
        our_pokemon_move_ids: torch.Tensor,     # (B, team_size, 4) — spatial-order move ids
        active_move_ids: torch.Tensor,          # (B, 4) — legal_mask-order move ids
        switch_ids: torch.Tensor,               # (B, 5) — legal_mask-order species ids
        our_pokemon_species_ids: torch.Tensor,  # (B, team_size) — spatial-order species
        switch_cont: torch.Tensor,              # (B, 5, SWITCH_SLOT_CONT_DIM)
    ) -> torch.Tensor:
        """Produce (B, 9, d_model) per-action context.

        Move slots 0-3:
          - Read the 4 active-Pokemon move tokens from spatial output (positions
            27..30 in V1 layout). Permute to legal-mask order via move_id matching
            because `our_pokemon_move_ids[:, 0, :]` is poke-env's `pokemon.moves`
            order while `active_move_ids` is `available_moves` order.
        Switch slots 4-8 (per Postscript C3):
          - Mean-pool the 17 attribute tokens per bench Pokemon -> (B, 5, d).
          - Permute via species match (our bench is alphabetical-by-species;
            `switch_ids` is `available_switches` order).
          - Concat with `switch_cont[:, j, -2:]` (defensive_eff, offensive_eff).
          - Project through `SwitchActionEncoder` to d_model.
        """
        B, _, d = spatial_out.shape
        n_moves = self._n_moves
        n_switches = self._n_switches
        per_pokemon = N_PER_POKEMON

        # ---- Move tokens (active Pokemon's 4 spatial-output move tokens) ----
        # Indices in the spatial output: N_BATTLE_STATE + 0*17 + (13..16)
        first = self._our_active_move_start
        spatial_active_moves = spatial_out[:, first:first + n_moves, :]   # (B, 4, d)
        # spatial-order move ids for our active Pokemon
        spatial_move_ids = our_pokemon_move_ids[:, 0, :]                  # (B, 4)
        # Legal-mask-order ids:
        # match[b, j_legal, k_spatial] = 1 if active_move_ids[b, j] == spatial_move_ids[b, k]
        match = (active_move_ids.unsqueeze(-1) == spatial_move_ids.unsqueeze(-2))  # (B, 4, 4)
        # argmax over k. For empty slots (active_move_ids[b,j]==0) every entry can
        # match if any spatial id is also 0 — that's fine, illegal slots are masked.
        move_perm = match.long().argmax(dim=-1)                            # (B, 4)
        idx = move_perm.unsqueeze(-1).expand(-1, -1, d)                    # (B, 4, d)
        move_ctx = spatial_active_moves.gather(1, idx)                     # (B, 4, d)

        # ---- Switch tokens (5 bench Pokemon pooled, permuted via species) ----
        bench_start = self._our_bench_start
        n_bench = self._n_switches
        # Bench Pokemon at our slots 1..1+n_bench-1 (alphabetical-by-species).
        # 17 tokens each -> mean-pool over the 17 attrs.
        bench_seq = spatial_out[:, bench_start:bench_start + n_bench * per_pokemon, :]
        bench_seq = bench_seq.reshape(B, n_bench, per_pokemon, d)
        bench_pooled_spatial = bench_seq.mean(dim=2)                       # (B, 5, d)
        # spatial bench species ids (our slots 1..5)
        bench_species_spatial = our_pokemon_species_ids[:, 1:1 + n_bench]  # (B, 5)
        # Permute: switch_ids[:, j] (game order) -> bench_pooled_spatial[:, k_match]
        sw_match = (switch_ids.unsqueeze(-1) == bench_species_spatial.unsqueeze(-2))
        sw_perm = sw_match.long().argmax(dim=-1)                            # (B, 5)
        bench_idx = sw_perm.unsqueeze(-1).expand(-1, -1, d)                 # (B, 5, d)
        bench_pooled = bench_pooled_spatial.gather(1, bench_idx)            # (B, 5, d)
        eff = switch_cont[..., -2:]                                         # (B, 5, 2)
        switch_ctx = self.switch_encoder(bench_pooled, eff)                 # (B, 5, d)

        return torch.cat([move_ctx, switch_ctx], dim=1)                    # (B, 9, d)

    # ---- Trainer-side adapter (used by InferenceBatcher / ppo.ppo_update) ----

    def action_encoder_from_spatial(
        self,
        batch: dict,
        spatial_out: torch.Tensor,
    ) -> torch.Tensor:
        """(B, 9, d_model) action context from an already-computed spatial pass.

        InferenceBatcher and ppo.ppo_update have spatial_out in hand from a
        prior `forward_spatial` call and need per-action context next. This
        method derives the spatial-order ids from `batch` via the tokenizer
        and dispatches to `_per_action_context` — analogous to legacy
        `model.action_encoder(...)` but consuming the already-computed
        spatial output rather than re-running the spatial transformer.
        """
        our_full_ids = self.tokenizer._fix_ids(batch, "our")  # (B, team_size, 7)
        return self._per_action_context(
            spatial_out=spatial_out,
            our_pokemon_move_ids=our_full_ids[..., 3:7],
            active_move_ids=batch["active_move_ids"],
            switch_ids=batch["switch_ids"],
            our_pokemon_species_ids=our_full_ids[..., 0],
            switch_cont=batch["switch_cont"],
        )

    # ---- Spatial pass with summary projection ----

    def forward_spatial(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize + spatial transformer + summary projection.

        Returns:
            spatial_out: (B, N, d_model)
            summary_temporal: (B, d_temporal) — flattened K scratch outputs
        """
        tok_out = self.tokenizer(batch)
        x = tok_out["tokens"]
        spatial_out = self.spatial(x)                                       # (B, N, d_model)
        K = self.cfg.n_summary_tokens
        scratch = spatial_out[:, -K:, :]                                    # (B, K, d_model)
        summary_temporal = self.summary_to_temporal(scratch.reshape(scratch.shape[0], -1))
        return spatial_out, summary_temporal

    # ---- Full forward (single turn, for live play / collection) ----

    def forward(
        self,
        batch: dict,
        history: Optional[torch.Tensor] = None,           # (B, T-1, d_temporal)
        history_lens: Optional[torch.Tensor] = None,      # (B,) optional
    ) -> Dict[str, torch.Tensor]:
        """Single-turn forward, interface-compatible with `PokeTransformer.forward`.

        See model.py:739-803 for the legacy equivalent. Returns dict with
        keys: action_logits, value, v_logits, summary, spatial_output.
        """
        spatial_out, summary = self.forward_spatial(batch)

        # Temporal: cat with history then run causal attention.
        if history is not None and history.shape[1] > 0:
            all_summaries = torch.cat([history, summary.unsqueeze(1)], dim=1)
            temporal_lens = (history_lens + 1) if history_lens is not None else None
        else:
            all_summaries = summary.unsqueeze(1)
            temporal_lens = None
        temporal_ctx = self.temporal(all_summaries, temporal_lens)          # (B, d_temporal)

        # Heads.
        actor_out  = spatial_out[:, 0, :]                                   # (B, d_model)
        critic_out = spatial_out[:, 1, :]                                   # (B, d_model)

        # Need spatial-order move ids and species ids (same path the Tokenizer
        # used) to permute action context.
        our_full_ids = self.tokenizer._fix_ids(batch, "our")                # (B, team_size, 7)
        our_pokemon_move_ids = our_full_ids[..., 3:7]                       # (B, team_size, 4)
        our_pokemon_species_ids = our_full_ids[..., 0]                      # (B, team_size)

        action_ctx = self._per_action_context(
            spatial_out=spatial_out,
            our_pokemon_move_ids=our_pokemon_move_ids,
            active_move_ids=batch["active_move_ids"],
            switch_ids=batch["switch_ids"],
            our_pokemon_species_ids=our_pokemon_species_ids,
            switch_cont=batch["switch_cont"],
        )

        action_logits = self.action_head(
            actor_out, temporal_ctx, action_ctx,
            legal_mask=batch.get("legal_mask"),
        )
        v_logits, value = self.value_head(critic_out, temporal_ctx)

        return {
            "action_logits": action_logits,
            "value":         value,
            "v_logits":      v_logits,
            # Detached for the per-battle history buffer (legacy convention).
            "summary":       summary.detach(),
            "spatial_output": spatial_out,
        }

    # ---- Legacy convenience ----

    def twohot_target(self, value: torch.Tensor) -> torch.Tensor:
        """Convert scalar value targets to 2-hot encoding. Mirrors model.py:805-816."""
        cfg = self.cfg
        v_support = self.value_head.v_support
        value = value.clamp(cfg.v_min, cfg.v_max)
        bin_width = v_support[1] - v_support[0]
        idx = (value - v_support[0]) / bin_width
        lo = idx.floor().long().clamp(0, cfg.v_bins - 2)
        hi = (lo + 1).clamp(max=cfg.v_bins - 1)
        weight_hi = (idx - lo.float()).clamp(0, 1)
        target = torch.zeros(value.shape[0], cfg.v_bins, device=value.device)
        target.scatter_(1, lo.unsqueeze(1), (1 - weight_hi).unsqueeze(1))
        target.scatter_(1, hi.unsqueeze(1), weight_hi.unsqueeze(1))
        return target

    # ---- BC training mega-batch path ----

    def forward_sequence(self, collated: dict, device: torch.device) -> Dict[str, torch.Tensor]:
        """Optimized BC-training forward: mega-batch the spatial pass over all
        valid (b, t) turns, then loop temporal + heads per-turn.

        Mirrors `PokeTransformer.forward_sequence` (model.py:818-978). The
        big efficiency win is running the heavy attention layers once over
        N_valid turns instead of B*T separate forwards.

        `collated`: output of dataset.collate_seq.
        Returns dict: action_logits (B,T,9), value (B,T), v_logits (B,T,v_bins).
        """
        import torch.nn.functional as F  # noqa: F401 — used by ValueHead
        cfg = self.cfg
        B = collated["seq_lens"].shape[0]
        T = collated["mask"].shape[1]

        # Build a list of (b, t) pairs for valid turns (non-padded).
        valid_indices = []
        for b in range(B):
            L = int(collated["seq_lens"][b].item())
            for t in range(L):
                valid_indices.append((b, t))

        if not valid_indices:
            return {
                "action_logits": torch.zeros(B, T, cfg.format_config.n_actions, device=device),
                "value":         torch.zeros(B, T, device=device),
                "v_logits":      torch.zeros(B, T, cfg.v_bins, device=device),
            }

        bs_arr = [i[0] for i in valid_indices]
        ts_arr = [i[1] for i in valid_indices]

        def _gather(key):
            return collated[key][bs_arr, ts_arr].to(device)

        mega_batch = {
            "our_pokemon_ids":   _gather("our_pokemon_ids"),     # (N, 6, 7)
            "our_pokemon_banks": _gather("our_pokemon_banks"),
            "our_pokemon_cont":  _gather("our_pokemon_cont"),
            "opp_pokemon_ids":   _gather("opp_pokemon_ids"),
            "opp_pokemon_banks": _gather("opp_pokemon_banks"),
            "opp_pokemon_cont":  _gather("opp_pokemon_cont"),
            "field_cont":        _gather("field_cont_raw"),
            "transition_cont":   _gather("trans_cont_raw"),
            "active_move_ids":   _gather("active_move_ids_raw"),
            "active_move_cont":  _gather("active_move_cont_raw"),
            "switch_ids":        _gather("switch_ids_raw"),
            "switch_cont":       _gather("switch_cont_raw"),
            "legal_mask":        _gather("legal_mask_raw"),
        }
        # Field banks → dict
        fb = _gather("field_banks_raw")
        mega_batch["field_banks"] = {
            "turn":         fb[:, 0], "weather_dur": fb[:, 1],
            "terrain_dur":  fb[:, 2], "tr_dur":      fb[:, 3],
        }
        # Transition ids → dict
        ti = _gather("trans_ids_raw")
        mega_batch["transition_ids"] = {"our_action": ti[:, 0], "opp_action": ti[:, 1]}
        # Active-move banks → dict
        amb = _gather("active_move_banks_raw")
        mega_batch["active_move_banks"] = {
            "bp":   amb[:, :, 0], "acc":  amb[:, :, 1],
            "pp":   amb[:, :, 2], "prio": amb[:, :, 3],
        }

        # Phase 1: mega-batch spatial pass.
        spatial_out, all_summaries = self.forward_spatial(mega_batch)
        n_tokens = spatial_out.shape[1]

        # Action context (mega-batch). Tokenizer's `_fix_ids` accepts the
        # combined-width-7 form here directly.
        our_full_ids = self.tokenizer._fix_ids(mega_batch, "our")           # (N, 6, 7)
        action_ctx = self._per_action_context(
            spatial_out=spatial_out,
            our_pokemon_move_ids=our_full_ids[..., 3:7],
            active_move_ids=mega_batch["active_move_ids"],
            switch_ids=mega_batch["switch_ids"],
            our_pokemon_species_ids=our_full_ids[..., 0],
            switch_cont=mega_batch["switch_cont"],
        )

        # Phase 2: scatter mega-batch outputs back into (B, T, ...) grids.
        d_model = cfg.d_model
        d_temporal = cfg.d_temporal
        n_actions = cfg.format_config.n_actions
        summary_grid = torch.zeros(B, T, d_temporal, device=device)
        spatial_grid = torch.zeros(B, T, n_tokens, d_model, device=device)
        actctx_grid  = torch.zeros(B, T, n_actions, d_model, device=device)
        legal_grid   = torch.zeros(B, T, n_actions, device=device)
        for idx, (b, t) in enumerate(valid_indices):
            summary_grid[b, t] = all_summaries[idx]
            spatial_grid[b, t] = spatial_out[idx]
            actctx_grid[b, t]  = action_ctx[idx]
            legal_grid[b, t]   = mega_batch["legal_mask"][idx]

        out_logits  = torch.full((B, T, n_actions), -100.0, device=device)
        out_value   = torch.zeros(B, T, device=device)
        out_vlogits = torch.zeros(B, T, cfg.v_bins, device=device)

        seq_lens = collated["seq_lens"].to(device)

        # Phase 3: temporal + heads, batched across B at each turn.
        for t in range(T):
            valid_mask_t = seq_lens > t
            if not valid_mask_t.any():
                break
            n_valid_t = int(valid_mask_t.sum().item())

            temporal_input = summary_grid[valid_mask_t, :t + 1, :]          # (n_valid, t+1, D)
            temporal_lens  = torch.full((n_valid_t,), t + 1,
                                        dtype=torch.long, device=device)
            temporal_ctx   = self.temporal(temporal_input, temporal_lens)   # (n_valid, D)

            actor_out  = spatial_grid[valid_mask_t, t, 0, :]
            critic_out = spatial_grid[valid_mask_t, t, 1, :]
            act_ctx    = actctx_grid[valid_mask_t, t]
            legal_t    = legal_grid[valid_mask_t, t]

            logits = self.action_head(actor_out, temporal_ctx, act_ctx, legal_mask=legal_t)
            v_logits, val = self.value_head(critic_out, temporal_ctx)

            out_logits [valid_mask_t, t] = logits .to(out_logits.dtype)
            out_vlogits[valid_mask_t, t] = v_logits.to(out_vlogits.dtype)
            out_value  [valid_mask_t, t] = val    .to(out_value.dtype)

        return {
            "action_logits": out_logits,
            "value":         out_value,
            "v_logits":      out_vlogits,
        }


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
