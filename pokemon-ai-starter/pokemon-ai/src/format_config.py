# format_config.py — Single source of truth for format-dependent constants.
#
# Every magic number that changes between singles/doubles/triples lives here.
# Import FORMAT_SINGLES (or build a custom FormatConfig) instead of hardcoding
# 9, 5, 4, 6, etc. throughout the codebase.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FormatConfig:
    """Format-dependent constants for Pokemon battle AI.

    Changing format (singles → doubles → triples) changes the action space,
    team layout, and token structure. This dataclass captures all of those
    in one place so the rest of the code can be format-agnostic.
    """

    # --- Battle format identifier (passed to poke-env / battle server) ---
    battle_format: str = "gen9ou"

    # --- Team layout ---
    team_size: int = 6          # total pokemon per side
    n_active: int = 1           # active pokemon per side (2 for doubles, 3 for triples)
    n_bench: int = 5            # = team_size - n_active

    # --- Action space ---
    n_moves: int = 4            # move slots per active pokemon
    n_switches: int = 5         # = n_bench (switch targets)
    n_actions: int = 9          # = n_moves + n_switches (per active pokemon)

    # --- Type system (constant across all formats/gens) ---
    n_types: int = 19           # 18 types + typeless/unknown
    n_stats: int = 6            # HP/Atk/Def/SpA/SpD/Spe

    # --- Generation (affects available species/moves/abilities/items) ---
    gen: int = 9

    def __post_init__(self):
        assert self.n_bench == self.team_size - self.n_active, \
            f"n_bench ({self.n_bench}) must equal team_size - n_active ({self.team_size - self.n_active})"
        assert self.n_switches == self.n_bench, \
            f"n_switches ({self.n_switches}) must equal n_bench ({self.n_bench})"
        assert self.n_actions == self.n_moves + self.n_switches, \
            f"n_actions ({self.n_actions}) must equal n_moves + n_switches ({self.n_moves + self.n_switches})"


# --- Pre-built configs for common formats ---

FORMAT_SINGLES = FormatConfig()  # default: gen9ou singles

# Future (not yet implemented — action heads need redesign for multi-active):
# FORMAT_DOUBLES = FormatConfig(
#     battle_format="gen9vgc2024regulationh",
#     n_active=2, n_bench=4, n_switches=4,
#     n_moves=4, n_actions=8,  # per active pokemon; actual decision space is combinatorial
# )
# FORMAT_TRIPLES = FormatConfig(
#     battle_format="gen5triples",
#     n_active=3, n_bench=3, n_switches=3,
#     n_moves=4, n_actions=7,
# )


def format_from_str(fmt: str) -> FormatConfig:
    """Parse a battle format string like 'gen9ou' into a FormatConfig.

    For now, all formats are singles. When doubles/triples support is added,
    this function will detect the format type and return the right config.
    """
    gen = 9  # default
    for g in range(1, 10):
        if fmt.startswith(f"gen{g}"):
            gen = g
            break

    # All currently supported formats are singles
    return FormatConfig(battle_format=fmt, gen=gen)
