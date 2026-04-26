"""external_adapters.py — load in-process external opponent adapters into PoolEntry list.

Used by train_rl.py when --external-adapters is set. Reads a YAML config that
lists adapter specs (one per opponent) and returns a list of `PoolEntry`
objects ready to extend snapshot_pool with.

This is the "Phase 2 adapter" path (see docs/EXTERNAL_OPPONENTS_PHASE2.md):
each adapter is a normal poke-env Player subclass instantiated in our own
training process, so it slots into the PFSP collection without subprocess
spawning, custom Showdown clients, or message protocol bridging.

Currently supported adapter types:
- ``pokeengine``: Foul Play-style MCTS via the poke-engine Rust library
  (`PokeEnginePlayer`). Lightweight (no torch), no separate venv needed.
- ``metamon`` (NOT YET IMPLEMENTED): wrap a Metamon pretrained agent.
  Requires amago + transformers in the active Python environment.

Example YAML:

```yaml
opponents:
  - name: foulplay-fast
    type: pokeengine
    search_time_ms: 100
    weight: 1.0
  - name: foulplay-strong
    type: pokeengine
    search_time_ms: 400
    weight: 1.5
```
"""
from __future__ import annotations

import logging
from typing import List

import yaml

from rl_collection import PoolEntry

logger = logging.getLogger(__name__)


def _factory_pokeengine(spec: dict):
    """Return a callable that builds a PokeEnginePlayer matching `spec`."""
    search_time_ms = int(spec.get("search_time_ms", 200))

    def _build(server_configuration, account_configuration, team,
               battle_format, max_concurrent_battles, **kw):
        # Imported lazily so the rest of training works even when poke-engine
        # isn't installed and external adapters aren't requested.
        from pokeengine_player import PokeEnginePlayer
        return PokeEnginePlayer(
            search_time_ms=search_time_ms,
            battle_format=battle_format,
            team=team,
            max_concurrent_battles=max_concurrent_battles,
            account_configuration=account_configuration,
            server_configuration=server_configuration,
        )

    return _build


def _factory_metamon(spec: dict):
    raise NotImplementedError(
        "Metamon adapter not yet implemented — see docs/EXTERNAL_OPPONENTS_PHASE2.md "
        "for the design. Install metamon + amago in the active venv first."
    )


_FACTORY_REGISTRY = {
    "pokeengine": _factory_pokeengine,
    "metamon": _factory_metamon,
}


def load_pool_entries(config_path: str) -> List[PoolEntry]:
    """Read a YAML config and return a list of PoolEntry objects."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    entries: List[PoolEntry] = []
    for raw in cfg.get("opponents", []) or []:
        spec = dict(raw)  # don't mutate caller's dict
        name = spec.pop("name", None)
        kind = spec.pop("type", None)
        if not name or not kind:
            logger.warning("Skipping malformed external adapter entry: %s", raw)
            continue
        try:
            factory_builder = _FACTORY_REGISTRY[kind]
        except KeyError:
            logger.warning("Unknown adapter type %r for %s; skipping", kind, name)
            continue

        weight = float(spec.pop("weight", 1.0) or 1.0)
        try:
            factory = factory_builder(spec)
        except NotImplementedError as e:
            logger.warning("Adapter %s (%s) skipped: %s", name, kind, e)
            continue
        except Exception:
            logger.exception("Adapter %s (%s) failed to build factory; skipping", name, kind)
            continue

        entries.append(PoolEntry(
            kind="external",
            key=name,
            factory=factory,
            factory_kwargs={},
            weight=weight,
        ))
        logger.info("Loaded external adapter: %s (%s, weight=%.2f)", name, kind, weight)

    return entries
