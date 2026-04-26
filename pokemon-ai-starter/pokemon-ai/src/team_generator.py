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
import os
import random
import re
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
