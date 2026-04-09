# src/vocab.py
# Build and load integer-ID vocabularies for species, moves, items, abilities.
# Each entity maps to a unique int. ID 0 is reserved for unknown/empty/pad.
#
# Usage:
#   from vocab import Vocab
#   v = Vocab.load()            # loads from src/data/vocab/*.json
#   v = Vocab.build_and_save()  # builds from poke-env + showdown data, saves to disk
#   species_id = v.species("garchomp")   # -> int
#   move_id    = v.move("earthquake")    # -> int
#   item_id    = v.item("choiceband")    # -> int
#   ability_id = v.ability("intimidate") # -> int
#
# All lookups return 0 for unknown/missing/None values (safe default).

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, Optional

_VOCAB_DIR = Path(__file__).parent / "data" / "vocab"


def _to_id_str(name: Optional[str]) -> str:
    """Normalize to lowercase alphanumeric (matches poke-env's to_id_str)."""
    if not name:
        return ""
    return "".join(c for c in name if c.isalnum()).lower()


class Vocab:
    """Integer-ID vocabularies for Pokemon entities."""

    def __init__(
        self,
        species_map: Dict[str, int],
        move_map: Dict[str, int],
        item_map: Dict[str, int],
        ability_map: Dict[str, int],
    ):
        self._species = species_map
        self._move = move_map
        self._item = item_map
        self._ability = ability_map

    # ---- lookups (return 0 for unknown) ----

    def species(self, name: Optional[str]) -> int:
        return self._species.get(_to_id_str(name), 0)

    def move(self, name: Optional[str]) -> int:
        return self._move.get(_to_id_str(name), 0)

    def item(self, name: Optional[str]) -> int:
        return self._item.get(_to_id_str(name), 0)

    def ability(self, name: Optional[str]) -> int:
        return self._ability.get(_to_id_str(name), 0)

    # ---- sizes (including pad=0) ----

    @property
    def n_species(self) -> int:
        return max(self._species.values()) + 1 if self._species else 1

    @property
    def n_moves(self) -> int:
        return max(self._move.values()) + 1 if self._move else 1

    @property
    def n_items(self) -> int:
        return max(self._item.values()) + 1 if self._item else 1

    @property
    def n_abilities(self) -> int:
        return max(self._ability.values()) + 1 if self._ability else 1

    # ---- persistence ----

    def save(self, vocab_dir: Optional[Path] = None):
        d = vocab_dir or _VOCAB_DIR
        d.mkdir(parents=True, exist_ok=True)
        for name, mapping in [
            ("species", self._species),
            ("moves", self._move),
            ("items", self._item),
            ("abilities", self._ability),
        ]:
            path = d / f"{name}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(mapping, f, indent=1, sort_keys=True)
        # Save summary
        summary = {
            "n_species": self.n_species,
            "n_moves": self.n_moves,
            "n_items": self.n_items,
            "n_abilities": self.n_abilities,
        }
        with open(d / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved vocabs to {d}/")
        print(f"  species: {self.n_species} (incl. pad)")
        print(f"  moves:   {self.n_moves} (incl. pad)")
        print(f"  items:   {self.n_items} (incl. pad)")
        print(f"  abilities: {self.n_abilities} (incl. pad)")

    @classmethod
    def load(cls, vocab_dir: Optional[Path] = None) -> Vocab:
        d = vocab_dir or _VOCAB_DIR
        maps = {}
        for name in ["species", "moves", "items", "abilities"]:
            path = d / f"{name}.json"
            with open(path, "r", encoding="utf-8") as f:
                maps[name] = json.load(f)
        return cls(maps["species"], maps["moves"], maps["items"], maps["abilities"])

    # ---- builder ----

    @classmethod
    def build_and_save(cls, vocab_dir: Optional[Path] = None) -> Vocab:
        """Build vocabularies from poke-env GenData + showdown items.ts."""
        from poke_env.data import GenData

        # --- Species: all gens 1-9 ---
        species_set: set[str] = set()
        for gen in range(1, 10):
            gd = GenData.from_gen(gen)
            species_set.update(gd.pokedex.keys())
        # Normalize (pokedex keys are already lowercase alphanumeric)
        species_set = {_to_id_str(s) for s in species_set if s}
        species_set.discard("")

        # --- Moves: all gens 1-9 ---
        move_set: set[str] = set()
        for gen in range(1, 10):
            gd = GenData.from_gen(gen)
            move_set.update(gd.moves.keys())
        move_set = {_to_id_str(m) for m in move_set if m}
        move_set.discard("")

        # --- Abilities: extract from all pokedex entries across gens ---
        ability_set: set[str] = set()
        for gen in range(1, 10):
            gd = GenData.from_gen(gen)
            for data in gd.pokedex.values():
                for ab in data.get("abilities", {}).values():
                    ability_set.add(_to_id_str(ab))
        ability_set.discard("")

        # --- Items: from showdown items.ts (most comprehensive source) ---
        item_set: set[str] = set()

        # Try showdown-reference data first
        showdown_items = (
            Path(__file__).parent.parent.parent
            / "showdown-reference"
            / "data"
            / "items.ts"
        )
        if showdown_items.exists():
            text = showdown_items.read_text(encoding="utf-8")
            # Each item entry starts with: '\titemname: {'
            entries = re.findall(r"^\t(\w+):\s*\{", text, re.MULTILINE)
            item_set.update(_to_id_str(e) for e in entries)
            print(f"Loaded {len(entries)} items from showdown items.ts")

        # Also pull from raw_data/items/items.csv as fallback/supplement
        items_csv = Path(__file__).parent.parent.parent.parent / "raw_data" / "items" / "items.csv"
        if items_csv.exists():
            import csv
            with open(items_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    item_name = row.get("item_name") or row.get("item")
                    if item_name and item_name.lower() != "none":
                        item_set.add(_to_id_str(item_name))
            print(f"Supplemented with items from items.csv (total: {len(item_set)})")

        item_set.discard("")

        # --- Build sorted maps (ID 0 = pad/unknown) ---
        species_map = {s: i + 1 for i, s in enumerate(sorted(species_set))}
        move_map = {m: i + 1 for i, m in enumerate(sorted(move_set))}
        item_map = {it: i + 1 for i, it in enumerate(sorted(item_set))}
        ability_map = {a: i + 1 for i, a in enumerate(sorted(ability_set))}

        vocab = cls(species_map, move_map, item_map, ability_map)
        vocab.save(vocab_dir)
        return vocab


# ---- CLI: python vocab.py ----
if __name__ == "__main__":
    v = Vocab.build_and_save()

    # Sanity checks
    print("\n--- Sanity checks ---")
    test_species = ["garchomp", "pikachu", "landorustherian", "urshifurapidstrike"]
    for s in test_species:
        print(f"  species({s!r}) = {v.species(s)}")

    test_moves = ["earthquake", "swordsdance", "uturn", "stealthrock"]
    for m in test_moves:
        print(f"  move({m!r}) = {v.move(m)}")

    test_items = ["choiceband", "leftovers", "lifeorb", "focussash"]
    for it in test_items:
        print(f"  item({it!r}) = {v.item(it)}")

    test_abilities = ["intimidate", "levitate", "sandstream", "multiscale"]
    for ab in test_abilities:
        print(f"  ability({ab!r}) = {v.ability(ab)}")

    # Check for unknowns
    print(f"\n  unknown species: {v.species('notapokemon')}")
    print(f"  unknown move:    {v.move('notamove')}")
    print(f"  None item:       {v.item(None)}")
    print(f"  empty ability:   {v.ability('')}")
