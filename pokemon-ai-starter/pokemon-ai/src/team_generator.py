# team_generator.py — Procedural team generator from Smogon usage stats
#
# Parses gen9 usage data (OU/UU/RU/NU/PU/ZU) and generates random teams
# weighted by competitive usage. For training diversity — eval stays on
# the 70 handcrafted teams in teams_ou.py.
#
# Usage:
#   from team_generator import ProceduralTeambuilder
#   tb = ProceduralTeambuilder("raw_data/pokemon_usage/2024-04")
#   # tb.yield_team() returns a new random team each call

from __future__ import annotations
import json
import mmap
import os
import random
import re
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PokemonData:
    name: str
    raw_count: int = 0
    abilities: List[Tuple[str, float]] = field(default_factory=list)  # (name, weight)
    items: List[Tuple[str, float]] = field(default_factory=list)
    moves: List[Tuple[str, float]] = field(default_factory=list)
    spreads: List[Tuple[str, str, List[int], float]] = field(default_factory=list)  # (nature, evs_str, evs_list, weight)


def _normalize_name(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "")


# ---------------------------------------------------------------------------
# OU ban lists per generation (Ubers / AG — cannot be used in OU)
# ---------------------------------------------------------------------------

_UBERS_BY_GEN = {
    9: {
        # Ubers
        "Arceus", "Calyrex-Ice", "Calyrex-Shadow", "Dialga", "Dialga-Origin",
        "Eternatus", "Giratina", "Giratina-Origin", "Groudon", "Ho-Oh",
        "Koraidon", "Kyogre", "Kyurem-White", "Lugia", "Lunala",
        "Mewtwo", "Miraidon", "Necrozma-Dawn-Wings", "Necrozma-Dusk-Mane",
        "Palkia", "Palkia-Origin", "Rayquaza", "Reshiram", "Solgaleo",
        "Terapagos", "Zacian", "Zacian-Crowned", "Zekrom",
        # AG only
        "Mega Rayquaza",
        # Common OU bans (clauses)
        "Flutter Mane", "Palafin", "Palafin-Hero", "Annihilape",
        "Espathra", "Iron Bundle", "Chi-Yu", "Roaring Moon",
        "Gouging Fire", "Volcarona",
    },
    8: {
        # Gen 8 OU bans (Sword/Shield era)
        "Arceus", "Calyrex-Ice", "Calyrex-Shadow", "Dialga", "Eternatus",
        "Giratina", "Giratina-Origin", "Groudon", "Ho-Oh", "Kyogre",
        "Kyurem-White", "Lugia", "Lunala", "Mewtwo", "Necrozma-Dawn-Wings",
        "Necrozma-Dusk-Mane", "Palkia", "Rayquaza", "Reshiram", "Solgaleo",
        "Zacian", "Zacian-Crowned", "Zekrom",
        # Gen 8 OU specific bans
        "Cinderace", "Darmanitan-Galar", "Dracovish", "Genesect",
        "Landorus", "Magearna", "Spectrier", "Urshifu",
    },
    7: {
        # Gen 7 OU bans (Sun/Moon era)
        "Arceus", "Blaziken", "Darkrai", "Deoxys", "Deoxys-Attack",
        "Dialga", "Giratina", "Giratina-Origin", "Groudon", "Ho-Oh",
        "Kyogre", "Lugia", "Lunala", "Marshadow", "Mewtwo",
        "Necrozma-Dawn-Wings", "Necrozma-Dusk-Mane", "Palkia",
        "Pheromosa", "Rayquaza", "Reshiram", "Solgaleo", "Xerneas",
        "Yveltal", "Zekrom", "Zygarde",
        "Mega Gengar", "Mega Lucario", "Mega Salamence", "Mega Kangaskhan",
    },
    6: {
        # Gen 6 OU bans (X/Y era)
        "Arceus", "Blaziken", "Darkrai", "Deoxys", "Deoxys-Attack",
        "Dialga", "Genesect", "Giratina", "Giratina-Origin", "Groudon",
        "Ho-Oh", "Kyogre", "Lugia", "Mewtwo", "Palkia", "Rayquaza",
        "Reshiram", "Xerneas", "Yveltal", "Zekrom",
        "Mega Gengar", "Mega Lucario", "Mega Salamence", "Mega Kangaskhan",
        "Mega Mawile",
    },
}


def get_ban_list(gen: int = 9) -> set:
    """Return the normalized OU ban list for a given generation."""
    raw = _UBERS_BY_GEN.get(gen, _UBERS_BY_GEN[9])  # fallback to gen9
    return {_normalize_name(n) for n in raw}


# Default for backward compat
UBERS_AG = _UBERS_BY_GEN[9]
UBERS_AG_LOWER = get_ban_list(9)


def _base_species(name: str) -> str:
    """Extract base species for species clause (e.g. 'Ogerpon-Wellspring' → 'ogerpon').
    Handles hyphenated formes while preserving base names like 'Porygon-Z'."""
    # Known single-species bases that have a hyphen in the base name
    HYPHEN_BASES = {
        "porygon-z", "porygon-2", "ho-oh", "jangmo-o", "hakamo-o", "kommo-o",
        "tapu koko", "tapu lele", "tapu bulu", "tapu fini",
        "mr. mime", "mr. rime", "mime jr.", "type: null",
    }
    low = name.lower().strip()
    if low in HYPHEN_BASES:
        return _normalize_name(low)
    # Split on hyphen, take first part as base
    parts = name.split("-")
    return _normalize_name(parts[0])


def _is_banned(name: str, ban_list: set = None) -> bool:
    if ban_list is None:
        ban_list = UBERS_AG_LOWER
    return _normalize_name(name) in ban_list


# ---------------------------------------------------------------------------
# Parser for Smogon usage stat files
# ---------------------------------------------------------------------------

def parse_usage_file(path: str, ban_list: set = None) -> List[PokemonData]:
    """Parse a Smogon moveset statistics file into PokemonData objects."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    pokemon_list = []
    # Split into per-Pokemon blocks. Each block starts with a name line between +---+
    # Pattern: two consecutive +---+ lines with a name between them at the top
    blocks = re.split(r'\n\s*\+[-]+\+\s*\n\s*\+[-]+\+\s*\n', text)

    # First block might be empty or partial — find the first name
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.split('\n')
        # Find the pokemon name — it's in a | Name | line
        name = None
        section = None
        abilities = []
        items = []
        moves = []
        spreads = []
        raw_count = 0

        for line in lines:
            line = line.strip()

            # Section header
            if line.startswith('|') and line.endswith('|'):
                content = line.strip('| ').strip()

                # Detect section headers
                if content == "Abilities":
                    section = "abilities"
                    continue
                elif content == "Items":
                    section = "items"
                    continue
                elif content == "Moves":
                    section = "moves"
                    continue
                elif content == "Spreads":
                    section = "spreads"
                    continue
                elif content == "Teammates":
                    section = "teammates"
                    continue
                elif content == "Checks and Counters":
                    section = "counters"
                    continue

                # Name detection (first content line that isn't a known section/stat)
                if name is None and not content.startswith("Raw count") and \
                   not content.startswith("Avg.") and not content.startswith("Viability") and \
                   section is None and content and not content.startswith("+"):
                    name = content
                    continue

                # Raw count
                m = re.match(r'Raw count:\s*(\d+)', content)
                if m:
                    raw_count = int(m.group(1))
                    continue

                # Parse entries in current section
                if section in ("abilities", "items", "moves"):
                    # Format: "Name  XX.XXX%" or "Other  XX.XXX%"
                    m = re.match(r'(.+?)\s+([\d.]+)%', content)
                    if m:
                        entry_name = m.group(1).strip()
                        pct = float(m.group(2))
                        if entry_name == "Other":
                            continue  # skip "Other" bucket
                        if section == "abilities":
                            abilities.append((entry_name, pct))
                        elif section == "items":
                            items.append((entry_name, pct))
                        elif section == "moves":
                            moves.append((entry_name, pct))

                elif section == "spreads":
                    # Format: "Nature:HP/Atk/Def/SpA/SpD/Spe  XX.XXX%"
                    m = re.match(r'(\w+):([\d/]+)\s+([\d.]+)%', content)
                    if m:
                        nature = m.group(1)
                        evs_str = m.group(2)
                        pct = float(m.group(3))
                        evs = [int(x) for x in evs_str.split('/')]
                        if len(evs) == 6:
                            spreads.append((nature, evs_str, evs, pct))
                    elif content.startswith("Other"):
                        continue

        if name and not _is_banned(name, ban_list):
            pd = PokemonData(
                name=name,
                raw_count=raw_count,
                abilities=abilities,
                items=items,
                moves=moves,
                spreads=spreads,
            )
            # Only include mons with enough data to build a set
            if pd.abilities and pd.items and len(pd.moves) >= 4 and pd.spreads:
                pokemon_list.append(pd)

    return pokemon_list


def _default_tiers(gen: int = 9) -> list:
    """Default tier list for a given generation.

    OU rating threshold differs by gen (Smogon's historical cutoff):
    - gen9: 1695 (current)
    - gen4-8: 1760 (legacy)
    Lower tiers use 1500 for all gens.
    """
    prefix = f"gen{gen}"
    ou_rating = "1695" if gen == 9 else "1760"
    return [
        (f"{prefix}ou", ou_rating),
        (f"{prefix}uu", "1500"),
        (f"{prefix}ru", "1500"),
        (f"{prefix}nu", "1500"),
        (f"{prefix}pu", "1500"),
        (f"{prefix}zu", "1500"),
    ]


def load_pokemon_pool(
    stats_dir: str,
    tiers: Optional[List[str]] = None,
    ban_list: set = None,
    gen: int = 9,
) -> Dict[str, PokemonData]:
    """Load and merge Pokemon data across tiers.

    For mons appearing in multiple tiers, we keep the entry with
    the highest raw_count (most data = most reliable distributions).
    """
    if tiers is None:
        tiers = _default_tiers(gen)

    pool: Dict[str, PokemonData] = {}

    for tier, rating in tiers:
        path = os.path.join(stats_dir, f"{tier}-{rating}.txt")
        if not os.path.exists(path):
            print(f"  [WARN] Missing usage file: {path}")
            continue

        mons = parse_usage_file(path, ban_list=ban_list)
        for mon in mons:
            key = _normalize_name(mon.name)
            if key not in pool or mon.raw_count > pool[key].raw_count:
                pool[key] = mon

    return pool


# ---------------------------------------------------------------------------
# Team generation
# ---------------------------------------------------------------------------

def _weighted_sample(choices: List[Tuple[str, float]]) -> str:
    """Sample one item from (name, weight) pairs."""
    names = [c[0] for c in choices]
    weights = [c[1] for c in choices]
    total = sum(weights)
    if total <= 0:
        return random.choice(names)
    return random.choices(names, weights=weights, k=1)[0]


def _weighted_sample_spread(spreads: List[Tuple[str, str, List[int], float]]) -> Tuple[str, List[int]]:
    """Sample a spread, returns (nature, [hp,atk,def,spa,spd,spe])."""
    weights = [s[3] for s in spreads]
    total = sum(weights)
    if total <= 0:
        idx = random.randrange(len(spreads))
    else:
        idx = random.choices(range(len(spreads)), weights=weights, k=1)[0]
    return spreads[idx][0], spreads[idx][2]


def _sample_moves(move_pool: List[Tuple[str, float]], n: int = 4) -> List[str]:
    """Sample n unique moves weighted by usage."""
    if len(move_pool) <= n:
        return [m[0] for m in move_pool]

    selected = []
    remaining = list(move_pool)
    for _ in range(n):
        if not remaining:
            break
        names = [m[0] for m in remaining]
        weights = [m[1] for m in remaining]
        total = sum(weights)
        if total <= 0:
            pick = random.choice(names)
        else:
            pick = random.choices(names, weights=weights, k=1)[0]
        selected.append(pick)
        remaining = [(n, w) for n, w in remaining if n != pick]
    return selected


def generate_team(
    pool: Dict[str, PokemonData],
    pool_weights: Optional[Dict[str, float]] = None,
) -> str:
    """Generate one random team in Showdown text format.

    Args:
        pool: Pokemon pool from load_pokemon_pool()
        pool_weights: optional {normalized_name: weight} for mon selection.
                      If None, uses raw_count.
    """
    all_mons = list(pool.values())
    if not all_mons:
        raise ValueError("Empty Pokemon pool")

    if pool_weights:
        mon_weights = [pool_weights.get(_normalize_name(m.name), 1.0) for m in all_mons]
    else:
        mon_weights = [float(m.raw_count) for m in all_mons]

    team_parts = []
    used_species = set()  # species clause
    used_items = set()    # item clause
    attempts = 0
    max_attempts = 100

    while len(team_parts) < 6 and attempts < max_attempts:
        attempts += 1

        # Pick a mon
        mon = random.choices(all_mons, weights=mon_weights, k=1)[0]
        species_key = _base_species(mon.name)

        # Species clause (uses base species — Ogerpon-Wellspring and Ogerpon are same)
        if species_key in used_species:
            continue

        # Filter invalid items/moves
        valid_items = [(n, w) for n, w in mon.items if n.lower() not in ("nothing", "other", "")]
        if not valid_items:
            continue

        # Sample item
        item = _weighted_sample(valid_items)

        # Item clause
        item_key = item.lower()
        if item_key in used_items:
            # Try a few more items before giving up on this mon
            found_alt = False
            for _ in range(5):
                alt_item = _weighted_sample(valid_items)
                if alt_item.lower() not in used_items:
                    item = alt_item
                    item_key = item.lower()
                    found_alt = True
                    break
            if not found_alt:
                continue

        # Sample ability, moves, spread
        ability = _weighted_sample(mon.abilities)
        valid_moves = [(n, w) for n, w in mon.moves if n.lower() not in ("nothing", "other", "")]
        moves = _sample_moves(valid_moves, 4)
        nature, evs = _weighted_sample_spread(mon.spreads)

        if len(moves) < 4 or len(valid_moves) < 4:
            continue  # need 4 valid moves

        # Showdown rejects all-zero EVs — add 1 to HP if needed
        if sum(evs) == 0:
            evs[0] = 4

        used_species.add(species_key)
        used_items.add(item_key)

        # Build Showdown format block
        ev_str = f"{evs[0]} HP / {evs[1]} Atk / {evs[2]} Def / {evs[3]} SpA / {evs[4]} SpD / {evs[5]} Spe"
        block = f"""{mon.name} @ {item}
Ability: {ability}
EVs: {ev_str}
{nature} Nature
- {moves[0]}
- {moves[1]}
- {moves[2]}
- {moves[3]}"""
        team_parts.append(block)

    if len(team_parts) < 6:
        # Couldn't build a full team — fill remaining slots with random picks
        for mon in random.sample(all_mons, min(len(all_mons), 20)):
            if len(team_parts) >= 6:
                break
            sk = _normalize_name(mon.name)
            if sk in used_species:
                continue
            if len(mon.moves) < 4 or not mon.abilities or not mon.items or not mon.spreads:
                continue
            ability = _weighted_sample(mon.abilities)
            item = _weighted_sample(mon.items)
            moves = _sample_moves(mon.moves, 4)
            nature, evs = _weighted_sample_spread(mon.spreads)
            used_species.add(sk)
            ev_str = f"{evs[0]} HP / {evs[1]} Atk / {evs[2]} Def / {evs[3]} SpA / {evs[4]} SpD / {evs[5]} Spe"
            block = f"""{mon.name} @ {item}
Ability: {ability}
EVs: {ev_str}
{nature} Nature
- {moves[0]}
- {moves[1]}
- {moves[2]}
- {moves[3]}"""
            team_parts.append(block)

    return "\n\n".join(team_parts)


# ---------------------------------------------------------------------------
# poke-env Teambuilder integration
# ---------------------------------------------------------------------------

try:
    from poke_env.teambuilder import Teambuilder as _Teambuilder
except ImportError:
    _Teambuilder = object


class ProceduralTeambuilder(_Teambuilder):
    """Teambuilder that generates a new procedural team for each battle.

    Args:
        stats_dir: path to usage stats directory (e.g. "raw_data/pokemon_usage/2024-04")
        random_pct: fraction of teams that are fully random from the pool
                    (uniform weights, ignoring usage). Default 0.05 (5%).
    """

    def __init__(self, stats_dir: str, random_pct: float = 0.05, gen: int = 9):
        super().__init__()
        self.gen = gen
        self.ban_list = get_ban_list(gen)
        self.pool = load_pokemon_pool(stats_dir, ban_list=self.ban_list, gen=gen)
        self.random_pct = random_pct
        # Precompute uniform weights for random teams
        self._uniform_weights = {
            _normalize_name(m.name): 1.0 for m in self.pool.values()
        }
        print(f"ProceduralTeambuilder: loaded {len(self.pool)} Pokemon from {stats_dir}")

    def yield_team(self) -> str:
        if random.random() < self.random_pct:
            team_str = generate_team(self.pool, pool_weights=self._uniform_weights)
        else:
            team_str = generate_team(self.pool)
        mons = self.parse_showdown_team(team_str)
        return self.join_team(mons)


def procedural_teambuilder(stats_dir: str, random_pct: float = 0.05, gen: int = 9) -> ProceduralTeambuilder:
    """Convenience constructor."""
    return ProceduralTeambuilder(stats_dir, random_pct=random_pct, gen=gen)


class StaticTeamPool(_Teambuilder):
    """Yields random teams from a directory of Showdown team text files.

    Used to wrap external team libraries (e.g. Foul Play's curated teams,
    or pre-generated pools from Metamon's TeamPredictor). Each call picks
    a uniformly-random team from the directory, parses it, and returns
    the packed format poke-env expects.

    Args:
        team_dir: directory containing one or more Showdown team text files
                  (recursively scanned). Each file should be a single team
                  in standard Showdown export format.
    """

    def __init__(self, team_dir):
        super().__init__()
        from pathlib import Path as _Path
        self.team_dir = _Path(team_dir)
        self._teams = []
        if self.team_dir.exists():
            for fp in self.team_dir.rglob('*'):
                if not fp.is_file() or fp.name.startswith('.'):
                    continue
                try:
                    with open(fp, encoding='utf-8') as f:
                        text = f.read().strip()
                except (OSError, UnicodeDecodeError):
                    continue
                if text:
                    self._teams.append(text)
        if not self._teams:
            raise ValueError(f"No team files found in {team_dir}")
        print(f"StaticTeamPool: loaded {len(self._teams)} teams from {team_dir}")

    def yield_team(self) -> str:
        text = random.choice(self._teams)
        mons = self.parse_showdown_team(text)
        return self.join_team(mons)


# ---------------------------------------------------------------------------
# Bundled / mmap-backed team pool
#
# StaticTeamPool loads every team file into a Python list of strings — at
# 90 workers × ~1 GB per source pair (hl + gl), that's ~85 GB of duplicated
# heap memory across the box. Each Python string is per-process, so kernel
# page cache for the team files doesn't help us.
#
# StaticTeamPoolMmap replaces the heap-resident list with a single mmap'd
# bundle file. The bundle is concatenated team text + an index of
# (offset, length) pairs. Workers mmap the same file → kernel shares pages
# via the page cache → per-worker resident memory drops from ~1 GB to a few
# MB. yield_team semantics are identical: random index into a uniform-
# probability pool, decode, parse, return packed format.
#
# Bundle format (little-endian):
#   [0:4]      magic "PTM1"
#   [4:8]      n_teams (uint32)
#   [8:8+8n]   index: for each team i, (offset_in_body uint32, length uint32)
#   [8+8n:]    body: concatenated UTF-8 team text
# ---------------------------------------------------------------------------

_BUNDLE_MAGIC = b'PTM1'
_BUNDLE_HEADER_SIZE = 8  # magic + n_teams
_BUNDLE_INDEX_ENTRY_SIZE = 8  # offset + length, each uint32


def bundle_path_for(team_dir: str) -> str:
    """Return the conventional bundle file path for a team directory.

    Sibling file: /path/to/hl_05_26/gen9ou/ → /path/to/hl_05_26/gen9ou.teampack
    """
    p = Path(team_dir).resolve()
    return str(p.parent / (p.name + ".teampack"))


def build_team_bundle(team_dir: str, output_path: str = None) -> Tuple[str, int]:
    """Build a .teampack bundle from a directory of team text files.

    Walks team_dir recursively (matching StaticTeamPool semantics), reads each
    non-hidden regular file, sorts by path for determinism, and writes a
    single mmap-friendly bundle file. Atomic via tmp-rename.

    Args:
        team_dir: directory of team text files (e.g. .gen9ou_team)
        output_path: bundle path (default: bundle_path_for(team_dir))

    Returns:
        (bundle_path, n_teams)
    """
    if output_path is None:
        output_path = bundle_path_for(team_dir)
    src = Path(team_dir)
    if not src.exists():
        raise FileNotFoundError(f"Team dir does not exist: {team_dir}")

    # Collect all team files, sorted for deterministic bundle order
    files = sorted(
        fp for fp in src.rglob('*')
        if fp.is_file() and not fp.name.startswith('.')
    )

    teams: List[bytes] = []
    for fp in files:
        try:
            with open(fp, encoding='utf-8') as f:
                text = f.read().strip()
        except (OSError, UnicodeDecodeError):
            continue
        if text:
            teams.append(text.encode('utf-8'))

    if not teams:
        raise ValueError(f"No team files found in {team_dir}")

    n = len(teams)
    tmp_path = output_path + ".tmp"

    # Compute offsets into body
    offsets: List[Tuple[int, int]] = []
    cur = 0
    for t in teams:
        offsets.append((cur, len(t)))
        cur += len(t)

    with open(tmp_path, 'wb') as f:
        f.write(_BUNDLE_MAGIC)
        f.write(struct.pack('<I', n))
        for off, length in offsets:
            f.write(struct.pack('<II', off, length))
        for t in teams:
            f.write(t)

    os.replace(tmp_path, output_path)
    return output_path, n


class StaticTeamPoolMmap(_Teambuilder):
    """Mmap-backed read-only team pool. Drop-in replacement for StaticTeamPool
    when a .teampack bundle exists.

    Per-worker resident memory: ~few KB (instance state) + transient
    decoded-team string per yield_team call. The bundle file's pages live in
    the kernel page cache, shared across all workers that mmap the same file.

    Args:
        bundle_path: path to .teampack file (produced by build_team_bundle).
    """

    def __init__(self, bundle_path: str):
        super().__init__()
        self.bundle_path = bundle_path
        # Keep an fd open for the lifetime of the mmap; closing the fd is OK
        # on Linux (mmap stays valid), but Windows requires the fd to stay open.
        self._fd = open(bundle_path, 'rb')
        try:
            self._mm = mmap.mmap(self._fd.fileno(), 0, prot=mmap.PROT_READ)
        except AttributeError:
            # Windows uses ACCESS_READ instead of PROT_READ
            self._mm = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)

        magic = bytes(self._mm[:4])
        if magic != _BUNDLE_MAGIC:
            raise ValueError(
                f"Bad bundle magic at {bundle_path}: got {magic!r}, "
                f"expected {_BUNDLE_MAGIC!r}. Rebuild with build_team_bundle()."
            )
        self.n_teams = struct.unpack_from('<I', self._mm, 4)[0]
        self._index_offset = _BUNDLE_HEADER_SIZE
        self._body_offset = _BUNDLE_HEADER_SIZE + self.n_teams * _BUNDLE_INDEX_ENTRY_SIZE
        print(f"StaticTeamPoolMmap: {self.n_teams} teams from {bundle_path}")

    def __len__(self):
        return self.n_teams

    def _read_team_bytes(self, i: int) -> bytes:
        """Read the raw UTF-8 bytes for team index i."""
        idx_pos = self._index_offset + i * _BUNDLE_INDEX_ENTRY_SIZE
        off, length = struct.unpack_from('<II', self._mm, idx_pos)
        start = self._body_offset + off
        return bytes(self._mm[start:start + length])

    def yield_team(self) -> str:
        i = random.randrange(self.n_teams)
        text = self._read_team_bytes(i).decode('utf-8')
        mons = self.parse_showdown_team(text)
        return self.join_team(mons)


class MultiSourceTeambuilder(_Teambuilder):
    """Teambuilder that delegates each yield_team() call to a randomly chosen source.

    Designed for training where we want each game to draw teams from one of
    several team-generation philosophies (e.g. our procedural Smogon-weighted
    builder, Metamon's TeamPredictor, a Foul-Play-curated pool). Within a
    single game both sides should call the SAME MultiSourceTeambuilder
    instance so they get matched-source teams; PFSP collection plumbing is
    responsible for that.

    Args:
        sources: dict {name: teambuilder} where each teambuilder has
                 .yield_team() -> str (packed format)
        weights: optional dict {name: float}; non-normalized values OK,
                 normalized internally. Defaults to uniform across sources.

    Diagnostics:
        last_source: name of the source picked on the most recent call,
                     useful for logging team-distribution stats per iter.
    """

    def __init__(self, sources, weights=None):
        super().__init__()
        if not sources:
            raise ValueError("sources must be non-empty")
        self.sources = dict(sources)
        names = list(self.sources.keys())
        if weights:
            raw = [float(weights.get(k, 1.0)) for k in names]
        else:
            raw = [1.0] * len(names)
        total = sum(raw)
        if total <= 0:
            raise ValueError(f"sum of weights must be positive, got {total}")
        self._names = names
        self._weights = [w / total for w in raw]
        self.last_source = None  # set after each yield_team for diagnostics
        # Track per-source selection counts for diagnostics
        self._selection_counts = {n: 0 for n in names}
        weight_str = ", ".join(f"{n}={w:.2f}" for n, w in zip(self._names, self._weights))
        print(f"MultiSourceTeambuilder: {len(self._names)} sources [{weight_str}]")

    def yield_team(self) -> str:
        name = random.choices(self._names, weights=self._weights, k=1)[0]
        self.last_source = name
        self._selection_counts[name] += 1
        return self.sources[name].yield_team()

    def selection_stats(self):
        """Return {source_name: count} for telemetry/logging."""
        return dict(self._selection_counts)


def multi_source_teambuilder(sources, weights=None) -> MultiSourceTeambuilder:
    """Convenience constructor."""
    return MultiSourceTeambuilder(sources, weights=weights)


# ---------------------------------------------------------------------------
# Queue-based teambuilder for cross-process team handoff
# ---------------------------------------------------------------------------

class QueueTeambuilder(_Teambuilder):
    """yield_team() pops the next team from a shared on-disk queue directory.

    Used by subprocess opponents (Metamon's metamon_accept_serve.py, real
    Foul Play in foul_play_venv) so we can hand them OUR procedural Smogon
    team per battle. The coordinator in our main process calls
    `enqueue_team(queue_dir, packed_team)` before sending each challenge;
    the subprocess's accept-challenges loop calls yield_team(), which pops
    the next file. Both sides matched per game without sharing process
    memory or Python venvs.

    Atomic semantics: enqueue_team writes `<stamp>.tmp` and renames to
    `<stamp>.team` (atomic on POSIX and modern Windows NTFS). yield_team
    selects the oldest `.team` file, reads it, and unlinks. Concurrent
    parallel-actor subprocesses race for the same files but each succeeds
    or fails atomically.

    Args:
        queue_dir: directory the coordinator writes packed teams into.
        wait_timeout_s: yield_team blocks up to this long for a file to
            appear; raises if none arrives. Default 30s — generous for
            slow PFSP waves; subprocess shouldn't hit this if the
            coordinator is running.
        poll_interval_s: how often to recheck the queue while waiting.
        clean_on_init: if True, delete any stale `.team`/`.tmp` files on
            init (defensive against previous-run leftovers when subprocess
            is restarted).
    """

    def __init__(self, queue_dir, wait_timeout_s: float = 30.0,
                 poll_interval_s: float = 0.05, clean_on_init: bool = True):
        super().__init__()
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.wait_timeout_s = float(wait_timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        if clean_on_init:
            for p in list(self.queue_dir.glob("*.team")) + list(self.queue_dir.glob("*.tmp")):
                try:
                    p.unlink()
                except OSError:
                    pass

    def _next_file(self) -> Optional[Path]:
        try:
            files = sorted(self.queue_dir.glob("*.team"),
                           key=lambda p: p.stat().st_mtime_ns)
        except OSError:
            return None
        return files[0] if files else None

    def yield_team(self) -> str:
        import time
        deadline = time.time() + self.wait_timeout_s
        while True:
            f = self._next_file()
            if f is not None:
                try:
                    text = f.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    # Lost the race or file corrupt — try to remove and continue.
                    try:
                        f.unlink()
                    except OSError:
                        pass
                    continue
                try:
                    f.unlink()
                except OSError:
                    # Another actor consumed the file first; loop to find another.
                    continue
                return text.strip()
            if time.time() >= deadline:
                raise RuntimeError(
                    f"QueueTeambuilder({self.queue_dir}) timed out after "
                    f"{self.wait_timeout_s:.1f}s waiting for a team file. "
                    f"Coordinator may not be writing teams — check that "
                    f"enqueue_team() is called before each challenge."
                )
            time.sleep(self.poll_interval_s)


def enqueue_team(queue_dir, packed_team: str) -> Path:
    """Write a packed-format team to the queue directory atomically.

    Returns the final `.team` path. Filename includes a nanosecond timestamp
    so QueueTeambuilder.yield_team() consumes in FIFO order.
    """
    import time
    import uuid as _uuid
    queue = Path(queue_dir)
    queue.mkdir(parents=True, exist_ok=True)
    stamp = f"{time.time_ns():020d}_{_uuid.uuid4().hex[:8]}"
    tmp = queue / f"{stamp}.tmp"
    final = queue / f"{stamp}.team"
    tmp.write_text(packed_team.strip() + "\n", encoding="utf-8")
    # Path.replace is atomic on the same filesystem, including Windows NTFS.
    tmp.replace(final)
    return final


# ---------------------------------------------------------------------------
# Hierarchical mixers — synergistic (hl/gl) + top-level (procedural/synergistic).
# Per-pair asymmetric_rate controls how often P1 and P2 use different sources.
# ---------------------------------------------------------------------------

class SynergisticMixer(_Teambuilder):
    """Mixer over synergistic team sources (e.g. hl_05_26 + gl_05_26).

    Composes N source teambuilders with weights. Exposes both:
      - yield_team(): independent weighted sample (one team).
      - yield_pair(): paired (P1, P2) sample with intra_asymmetric_rate
        controlling how often the two teams come from different sources.

    Args:
        sources: dict {name: teambuilder}; each teambuilder must have
                 .yield_team() -> str (packed format).
        weights: optional {name: float}; non-normalized OK. Default uniform.
        intra_asymmetric_rate: probability that yield_pair() returns teams
                 from two DIFFERENT sources (e.g. one hl, one gl). 0.0 = always
                 matched, 1.0 = always cross-source.
    """

    def __init__(self, sources, weights=None, intra_asymmetric_rate: float = 0.30):
        super().__init__()
        if not sources:
            raise ValueError("sources must be non-empty")
        if not (0.0 <= intra_asymmetric_rate <= 1.0):
            raise ValueError(f"intra_asymmetric_rate must be in [0,1], got {intra_asymmetric_rate}")
        self.sources = dict(sources)
        names = list(self.sources.keys())
        if weights:
            raw = [float(weights.get(k, 1.0)) for k in names]
        else:
            raw = [1.0] * len(names)
        total = sum(raw)
        if total <= 0:
            raise ValueError(f"sum of weights must be positive, got {total}")
        self._names = names
        self._weights = [w / total for w in raw]
        self.intra_asymmetric_rate = intra_asymmetric_rate
        self.last_source = None
        self._selection_counts = {n: 0 for n in names}
        self._pair_counts = {"matched": 0, "asymmetric": 0}
        weight_str = ", ".join(f"{n}={w:.2f}" for n, w in zip(self._names, self._weights))
        print(f"SynergisticMixer: {len(self._names)} sources [{weight_str}], "
              f"intra_asymmetric_rate={self.intra_asymmetric_rate:.2f}")

    def _pick_source(self) -> str:
        return random.choices(self._names, weights=self._weights, k=1)[0]

    def yield_team(self) -> str:
        """Independent weighted sample. Used when caller has no need for matched pairs."""
        name = self._pick_source()
        self.last_source = name
        self._selection_counts[name] += 1
        return self.sources[name].yield_team()

    def yield_pair(self) -> tuple[str, str, str, str]:
        """Paired (P1, P2) sample respecting intra_asymmetric_rate.

        Returns: (team_p1, team_p2, source_p1, source_p2).
        With probability intra_asymmetric_rate, sources differ; otherwise same source.
        Even when sources match, the two teams are independent draws from that source.
        """
        if len(self._names) >= 2 and random.random() < self.intra_asymmetric_rate:
            # Asymmetric: pick two distinct sources weighted by their probs
            src_p1 = self._pick_source()
            src_p2 = self._pick_source()
            attempts = 0
            while src_p2 == src_p1 and attempts < 8:
                src_p2 = self._pick_source()
                attempts += 1
            if src_p2 == src_p1:
                # Fallback: deterministically pick the other source
                other = [n for n in self._names if n != src_p1]
                if other:
                    src_p2 = random.choice(other)
            self._pair_counts["asymmetric"] += 1
        else:
            src_p1 = src_p2 = self._pick_source()
            self._pair_counts["matched"] += 1
        self._selection_counts[src_p1] += 1
        self._selection_counts[src_p2] += 1
        team_p1 = self.sources[src_p1].yield_team()
        team_p2 = self.sources[src_p2].yield_team()
        return (team_p1, team_p2, src_p1, src_p2)

    def selection_stats(self) -> dict:
        return {
            "sources": dict(self._selection_counts),
            "pairs": dict(self._pair_counts),
        }


class TopMixer(_Teambuilder):
    """Top-level mixer over procedural + synergistic teambuilders.

    Args:
        procedural: teambuilder for procedural (random Smogon) teams.
        synergistic: SynergisticMixer (or any teambuilder with yield_pair()).
        syn_pct: fraction of teams that should be synergistic. Default 0.3.
        top_asymmetric_rate: probability that yield_pair() returns one
                             procedural + one synergistic team (cross-quality).
                             Default 0.2.

    yield_team(): independent weighted sample.
    yield_pair(): paired sample respecting top_asymmetric_rate (and propagating
                  to SynergisticMixer's intra_asymmetric_rate when both sides syn).
    """

    def __init__(self, procedural, synergistic, syn_pct: float = 0.3,
                 top_asymmetric_rate: float = 0.2):
        super().__init__()
        if not (0.0 <= syn_pct <= 1.0):
            raise ValueError(f"syn_pct must be in [0,1], got {syn_pct}")
        if not (0.0 <= top_asymmetric_rate <= 1.0):
            raise ValueError(f"top_asymmetric_rate must be in [0,1], got {top_asymmetric_rate}")
        self.procedural = procedural
        self.synergistic = synergistic
        self.syn_pct = syn_pct
        self.top_asymmetric_rate = top_asymmetric_rate
        self._selection_counts = {"procedural": 0, "synergistic": 0}
        self._pair_counts = {"both_proc": 0, "both_syn": 0, "asymmetric": 0}
        print(f"TopMixer: syn_pct={syn_pct:.2f}, top_asymmetric_rate={top_asymmetric_rate:.2f}")

    def _pick_top(self) -> str:
        return "synergistic" if random.random() < self.syn_pct else "procedural"

    def _yield_from(self, top_choice: str) -> tuple[str, str]:
        """Returns (team, source_label) for a single side."""
        if top_choice == "synergistic":
            # If synergistic has yield_team(), use it. Otherwise fall back.
            team = self.synergistic.yield_team()
            label = getattr(self.synergistic, "last_source", "syn")
            return (team, f"syn:{label}")
        else:
            team = self.procedural.yield_team()
            return (team, "procedural")

    def yield_team(self) -> str:
        """Independent weighted sample."""
        choice = self._pick_top()
        self._selection_counts[choice] += 1
        team, _ = self._yield_from(choice)
        return team

    def yield_pair(self) -> tuple[str, str, str, str]:
        """Paired (P1, P2) sample respecting top_asymmetric_rate.

        Returns: (team_p1, team_p2, source_p1, source_p2).

        Branching:
          - With probability top_asymmetric_rate: one side procedural, other synergistic
          - Otherwise: both procedural OR both synergistic, picked by syn_pct
            - If both synergistic, delegate to SynergisticMixer.yield_pair()
              which respects ITS intra_asymmetric_rate.
        """
        if random.random() < self.top_asymmetric_rate:
            # Asymmetric: one proc, one syn (50/50 which side is which)
            if random.random() < 0.5:
                p1_top, p2_top = "procedural", "synergistic"
            else:
                p1_top, p2_top = "synergistic", "procedural"
            team_p1, src_p1 = self._yield_from(p1_top)
            team_p2, src_p2 = self._yield_from(p2_top)
            self._selection_counts[p1_top] += 1
            self._selection_counts[p2_top] += 1
            self._pair_counts["asymmetric"] += 1
            return (team_p1, team_p2, src_p1, src_p2)

        # Top-level matched: both proc or both syn
        choice = self._pick_top()
        self._selection_counts[choice] += 2  # both sides count toward this bucket
        if choice == "synergistic" and hasattr(self.synergistic, "yield_pair"):
            # Delegate to syn's yield_pair for intra-syn asymmetric handling.
            # Prefix syn's raw source labels with "syn:" for unified labeling.
            self._pair_counts["both_syn"] += 1
            team_p1, team_p2, src_p1, src_p2 = self.synergistic.yield_pair()
            return (team_p1, team_p2, f"syn:{src_p1}", f"syn:{src_p2}")
        # Both procedural (or syn without yield_pair — fall back to two independent draws)
        team_p1, src_p1 = self._yield_from(choice)
        team_p2, src_p2 = self._yield_from(choice)
        if choice == "procedural":
            self._pair_counts["both_proc"] += 1
        else:
            self._pair_counts["both_syn"] += 1
        return (team_p1, team_p2, src_p1, src_p2)

    def selection_stats(self) -> dict:
        out = {
            "tops": dict(self._selection_counts),
            "pairs": dict(self._pair_counts),
        }
        if hasattr(self.synergistic, "selection_stats"):
            out["synergistic_internal"] = self.synergistic.selection_stats()
        return out


# ---------------------------------------------------------------------------
# PairedQueueProducer — pre-fills two queue dirs with matched team pairs.
# Used by SP and external dispatch paths to deliver paired teams to both
# player and opponent via the existing QueueTeambuilder pop-FIFO mechanism.
# ---------------------------------------------------------------------------

class PairedQueueProducer:
    """Generate N pairs from a mixer with yield_pair() and enqueue to two dirs.

    Usage pattern (synchronous, before battle_against):
        producer = PairedQueueProducer(top_mixer, queue_p1_dir, queue_p2_dir)
        stats = producer.produce_all(n_battles)  # blocking pre-fill
        # ... then run battle_against; QueueTeambuilders pop FIFO

    Both queue dirs are populated in lockstep — pair K's team1 → queue_p1,
    team2 → queue_p2. FIFO-per-queue guarantees that each battle pops a
    matched pair (the pair-index identity is preserved across queue pop order
    because pair K's two teams sit at the same FIFO position in both queues).

    Args:
        mixer: object with .yield_pair() -> (team_p1, team_p2, src_p1, src_p2)
        queue_p1_dir: directory for P1 teams (created if missing)
        queue_p2_dir: directory for P2 teams (created if missing; can equal
                      an existing external opp queue dir)

    Returns from produce_all():
        dict with keys: 'n_pairs', 'matched', 'asymmetric' (counts), and
        'sources' (per-source selection counts if mixer tracks them).
    """

    def __init__(self, mixer, queue_p1_dir, queue_p2_dir):
        self.mixer = mixer
        self.queue_p1 = Path(queue_p1_dir)
        self.queue_p2 = Path(queue_p2_dir)
        self.queue_p1.mkdir(parents=True, exist_ok=True)
        self.queue_p2.mkdir(parents=True, exist_ok=True)

    def produce_all(self, n_pairs: int) -> dict:
        """Generate n_pairs and enqueue to both queues. Returns stat summary."""
        if n_pairs < 0:
            raise ValueError(f"n_pairs must be >= 0, got {n_pairs}")
        matched = 0
        asymmetric = 0
        # Snapshot stats before, diff after — handles either SynergisticMixer
        # or TopMixer without tightly coupling to mixer internals.
        stats_before = self.mixer.selection_stats() if hasattr(self.mixer, 'selection_stats') else None
        for _ in range(n_pairs):
            team_p1, team_p2, src_p1, src_p2 = self.mixer.yield_pair()
            enqueue_team(self.queue_p1, team_p1)
            enqueue_team(self.queue_p2, team_p2)
            if src_p1 == src_p2:
                matched += 1
            else:
                asymmetric += 1
        stats_after = self.mixer.selection_stats() if hasattr(self.mixer, 'selection_stats') else None
        result = {
            "n_pairs": n_pairs,
            "matched_this_batch": matched,
            "asymmetric_this_batch": asymmetric,
        }
        if stats_before is not None and stats_after is not None:
            # Only include the most useful summary; full stats remain on mixer.
            result["mixer_total_pairs"] = stats_after.get("pairs", {})
        return result


# ---------------------------------------------------------------------------
# Factory: build train_tb from config dict.
# Used by mp_centralized_collect.py worker setup to construct either
# ProceduralTeambuilder (legacy) or TopMixer (hierarchical) based on whether
# synergistic teams are configured.
# ---------------------------------------------------------------------------

# Per-process caches. Each worker is its own process → its own caches.
# Caches the lightweight per-process state (open mmap fd, parsed bundle
# header, ProceduralTeambuilder pool) so subsequent iters within a worker
# reuse them. Mixer wrappings (Synergistic, TopMixer) are rebuilt fresh
# per call so selection_stats() starts at zero each iter ([TEAM-DIST] log
# is per-iter, not cumulative).
#
# StaticTeamPoolMmap: caches the open mmap + parsed bundle header (~few KB
#   per worker). The actual team bytes live in kernel page cache as a
#   single shared copy across all workers — mmap'ing the same file from
#   multiple processes shares the same physical pages.
# ProceduralTeambuilder: caches 545 PokemonData usage stats (~2 MB per
#   worker). yield_team RE-GENERATES a fresh team via RNG each call — the
#   cache only avoids re-parsing usage .txt files, every battle still gets
#   a unique procedural team.
_MMAP_POOL_CACHE: Dict[str, 'StaticTeamPoolMmap'] = {}
_PROCEDURAL_CACHE: Dict[str, 'ProceduralTeambuilder'] = {}


def build_train_teambuilder(procedural_teams_path: str = None,
                            syn_config: dict = None):
    """Construct a training teambuilder from configuration.

    Args:
        procedural_teams_path: path to usage stats directory for ProceduralTeambuilder.
                               If None and syn_config is None, returns None.
        syn_config: optional dict with keys:
            - "team_dirs": list of (path, weight) tuples for synergistic sources
                          (e.g. [("hl_05_26/gen9ou/", 0.6), ("gl_05_26/gen9ou/", 0.4)])
                          Each dir loaded via StaticTeamPool.
            - "team_pct": float in [0,1], fraction of teams that should be synergistic (TopMixer.syn_pct)
            - "intra_asymmetric_rate": float in [0,1] for SynergisticMixer (default 0.30)
            - "top_asymmetric_rate": float in [0,1] for TopMixer (default 0.20)

    Returns:
        - ProceduralTeambuilder if only procedural_teams_path set
        - TopMixer (procedural + synergistic) if both set
        - None if neither set
    """
    proc_tb = None
    if procedural_teams_path:
        if procedural_teams_path not in _PROCEDURAL_CACHE:
            _PROCEDURAL_CACHE[procedural_teams_path] = procedural_teambuilder(procedural_teams_path)
        proc_tb = _PROCEDURAL_CACHE[procedural_teams_path]

    if not syn_config or not syn_config.get("team_dirs"):
        return proc_tb

    # Build synergistic sources, caching the slow StaticTeamPool loads.
    team_dirs = syn_config["team_dirs"]
    if not team_dirs:
        return proc_tb
    sources = {}
    weights = {}
    for path, w in team_dirs:
        # Use directory basename as source name (e.g. "hl_05_26", "gl_05_26")
        name = Path(path).parent.name if Path(path).name == "gen9ou" else Path(path).name
        if name in sources:
            # Duplicate names — disambiguate with full path component
            name = f"{Path(path).parents[1].name}_{name}"
        # Require a pre-built .teampack bundle. train_rl.py main() builds
        # these once before workers spawn, so by the time we get here the
        # file exists. Workers mmap the same bundle file → kernel page
        # cache serves the bytes once, shared across all 90 workers.
        bundle = bundle_path_for(path)
        if not os.path.exists(bundle):
            raise FileNotFoundError(
                f"Team bundle not found: {bundle}\n"
                f"  Source dir: {path}\n"
                f"  Run build_team_bundle() in train_rl.py main() before "
                f"spawning workers, or invoke it manually."
            )
        if bundle not in _MMAP_POOL_CACHE:
            _MMAP_POOL_CACHE[bundle] = StaticTeamPoolMmap(bundle)
        sources[name] = _MMAP_POOL_CACHE[bundle]
        weights[name] = float(w)

    syn_mixer = SynergisticMixer(
        sources=sources,
        weights=weights,
        intra_asymmetric_rate=float(syn_config.get("intra_asymmetric_rate", 0.30)),
    )

    if proc_tb is None:
        # No procedural — return syn mixer directly. yield_pair() works;
        # yield_team() falls back to weighted independent sampling within syn.
        return syn_mixer

    return TopMixer(
        procedural=proc_tb,
        synergistic=syn_mixer,
        syn_pct=float(syn_config.get("team_pct", 0.30)),
        top_asymmetric_rate=float(syn_config.get("top_asymmetric_rate", 0.20)),
    )


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    stats_dir = sys.argv[1] if len(sys.argv) > 1 else "raw_data/pokemon_usage/2024-04"

    pool = load_pokemon_pool(stats_dir)
    print(f"Loaded {len(pool)} Pokemon from pool\n")

    # Show top 20 by raw count
    by_count = sorted(pool.values(), key=lambda x: x.raw_count, reverse=True)
    print("Top 20 by usage:")
    for i, mon in enumerate(by_count[:20]):
        print(f"  {i+1:2d}. {mon.name:25s} count={mon.raw_count:>8d}  "
              f"moves={len(mon.moves)}  items={len(mon.items)}")

    # Generate 3 sample teams
    for i in range(3):
        print(f"\n{'='*50}")
        print(f"Team {i+1}:")
        print(f"{'='*50}")
        print(generate_team(pool))
