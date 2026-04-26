"""external_adapters.py — load external opponent adapters into PoolEntry list.

Used by train_rl.py when --external-adapters is set. Reads a YAML config that
lists adapter specs (one per opponent) and returns a list of `PoolEntry`
objects ready to extend snapshot_pool with.

Two flavors of adapter, depending on whether the opponent's deps are
compatible with our main venv:

- **In-process adapter** (factory-based): a poke-env Player subclass we
  instantiate directly in our process. Used for `mcts` (poke-engine MCTS,
  lightweight, same venv).

- **Subprocess adapter** (showdown_username-based): an external bot running
  in its own venv as a separate process, connected to the same Showdown
  server. We send_challenges to its username. Used for `metamon` and real
  `foulplay` because their pinned deps conflict with ours. The subprocess
  is spawned + restarted by ExternalOpponentManager.

Both flavors plug into the same `PoolEntry` and the same PFSP win-rate
tracking — opponent identity is the entry's `key`.

Example YAML:

```yaml
opponents:
  - name: mcts-fast
    type: mcts
    search_time_ms: 100
    weight: 1.0
  - name: metamon-minikazam
    type: metamon
    model: Minikazam
    temperature: 1.0
    server_port: 9000
    weight: 1.0
```
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from external_opponent_manager import ExternalOpponent, ExternalOpponentManager
from rl_collection import PoolEntry

logger = logging.getLogger(__name__)

# Project root, used to resolve relative paths (metamon_venv, metamon_accept_serve.py, cache dir)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _factory_pokeengine(spec: dict, _ctx: dict) -> Tuple[PoolEntry, Optional[ExternalOpponent]]:
    """In-process adapter — returns a PoolEntry whose factory builds PokeEnginePlayer."""
    search_time_ms = int(spec.get("search_time_ms", 200))
    name = spec["__name__"]
    weight = spec["__weight__"]

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

    entry = PoolEntry(kind="external", key=name, factory=_build, weight=weight)
    return entry, None  # no subprocess to manage


def _factory_metamon(spec: dict, ctx: dict) -> Tuple[PoolEntry, ExternalOpponent]:
    """Subprocess adapter — spawns metamon_accept_serve.py in metamon_venv.

    Returns (PoolEntry pointing at the subprocess's Showdown username,
    ExternalOpponent describing how to spawn/supervise it). The caller wires
    the ExternalOpponent into a manager that handles spawn + auto-restart.
    """
    name = spec["__name__"]
    weight = spec["__weight__"]
    model = spec.get("model", "Minikazam")
    showdown_username = spec.get("showdown_username") or f"MM-{model}"
    team_set = spec.get("team_set", "competitive")
    temperature = float(spec.get("temperature", 1.0))
    checkpoint = spec.get("checkpoint")  # int or None
    battle_format = spec.get("format", "gen9ou")
    server_port = int(spec.get("server_port", ctx.get("default_server_port", 9000)))
    num_battles = int(spec.get("num_battles", 100000))

    venv_python = (
        _PROJECT_ROOT / "metamon_venv" / "Scripts" / "python.exe"
        if os.name == "nt"
        else _PROJECT_ROOT / "metamon_venv" / "bin" / "python"
    )
    if not venv_python.exists():
        raise FileNotFoundError(
            f"metamon_venv not found at {venv_python}. Run: python -m venv metamon_venv && "
            f"metamon_venv/Scripts/pip install -e metamon_ref/"
        )

    serve_script = Path(__file__).parent / "metamon_accept_serve.py"
    if not serve_script.exists():
        raise FileNotFoundError(f"metamon_accept_serve.py missing at {serve_script}")

    log_dir = _PROJECT_ROOT / "logs" / "external"
    log_path = log_dir / f"{name}.log"

    # Coordinator-managed team queue. We hand Metamon our procedural Smogon
    # teams per battle so both sides match per-game without sharing process
    # state. Set to a unique-per-name dir so multiple Metamon variants don't
    # collide. `use_our_teams: false` in the YAML reverts to metamon's static
    # team set (legacy behavior, useful for verifying the ladder rating side
    # without our team distribution).
    use_our_teams = bool(spec.get("use_our_teams", True))
    if use_our_teams:
        team_queue_dir = _PROJECT_ROOT / "data" / "external_team_queue" / name
        team_queue_dir.mkdir(parents=True, exist_ok=True)
    else:
        team_queue_dir = None

    cmd = [
        str(venv_python),
        str(serve_script),
        "--model", str(model),
        "--username", str(showdown_username),
        "--server-port", str(server_port),
        "--format", str(battle_format),
        "--num-battles", str(num_battles),
        "--temperature", str(temperature),
    ]
    if team_queue_dir is not None:
        cmd += ["--team-queue", str(team_queue_dir)]
    else:
        cmd += ["--team-set", str(team_set)]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(int(checkpoint))]

    # Pass env vars the subprocess needs (cache dir + version-check bypass)
    metamon_cache = os.environ.get("METAMON_CACHE_DIR") or str(_PROJECT_ROOT / "metamon_cache")
    os.environ["METAMON_CACHE_DIR"] = metamon_cache  # for completeness; subprocess inherits
    Path(metamon_cache).mkdir(parents=True, exist_ok=True)

    spawn_spec = ExternalOpponent(
        name=name,
        showdown_username=showdown_username,
        command=cmd,
        cwd=None,  # absolute paths in cmd, no cwd needed
        venv=None,
        auto_restart=True,
        log_file=str(log_path),
        description=f"Metamon {model} (in metamon_venv subprocess)",
    )

    pool_entry = PoolEntry(
        kind="external",
        key=name,
        showdown_username=showdown_username,
        team_queue_dir=str(team_queue_dir) if team_queue_dir else None,
        weight=weight,
    )
    return pool_entry, spawn_spec


def _factory_foulplay(spec: dict, ctx: dict) -> Tuple[PoolEntry, ExternalOpponent]:
    """Subprocess adapter — real Foul Play in foul_play_venv. Pops our
    procedural Smogon teams from a coordinator-controlled queue."""
    name = spec["__name__"]
    weight = spec["__weight__"]
    showdown_username = spec.get("showdown_username") or "FoulPlayBot"
    battle_format = spec.get("format", "gen9ou")
    server_port = int(spec.get("server_port", ctx.get("default_server_port", 9000)))
    num_battles = int(spec.get("num_battles", 100000))
    search_time_ms = int(spec.get("search_time_ms", 200))
    search_parallelism = int(spec.get("search_parallelism", 1))
    log_level = spec.get("log_level", "WARNING")

    venv_python = (
        _PROJECT_ROOT / "foul_play_venv" / "Scripts" / "python.exe"
        if os.name == "nt"
        else _PROJECT_ROOT / "foul_play_venv" / "bin" / "python"
    )
    if not venv_python.exists():
        raise FileNotFoundError(
            f"foul_play_venv not found at {venv_python}. Run: "
            f"python -m venv foul_play_venv && "
            f"foul_play_venv/Scripts/pip install -r foul_play_ref/requirements.txt"
        )

    serve_script = Path(__file__).parent / "foul_play_accept_serve.py"
    if not serve_script.exists():
        raise FileNotFoundError(f"foul_play_accept_serve.py missing at {serve_script}")

    log_dir = _PROJECT_ROOT / "logs" / "external"
    log_path = log_dir / f"{name}.log"

    # Real Foul Play always uses our procedural teams (matched source per game).
    team_queue_dir = _PROJECT_ROOT / "data" / "external_team_queue" / name
    team_queue_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(venv_python),
        str(serve_script),
        "--username", str(showdown_username),
        "--server-port", str(server_port),
        "--format", str(battle_format),
        "--num-battles", str(num_battles),
        "--search-time-ms", str(search_time_ms),
        "--search-parallelism", str(search_parallelism),
        "--team-queue", str(team_queue_dir),
        "--log-level", str(log_level),
    ]

    spawn_spec = ExternalOpponent(
        name=name,
        showdown_username=showdown_username,
        command=cmd,
        cwd=str(_PROJECT_ROOT / "foul_play_ref"),  # Foul Play imports relative to its own dir
        venv=None,
        auto_restart=True,
        log_file=str(log_path),
        description=f"Foul Play {search_time_ms}ms (in foul_play_venv subprocess)",
    )

    pool_entry = PoolEntry(
        kind="external",
        key=name,
        showdown_username=showdown_username,
        team_queue_dir=str(team_queue_dir),
        weight=weight,
    )
    return pool_entry, spawn_spec


_FACTORY_REGISTRY = {
    # `mcts` is the canonical name (it's MCTS via the poke-engine Rust library
    # — NOT the same as Foul Play, which adds Smogon-set-guessing on top of
    # poke-engine). `pokeengine` kept as alias for older YAML configs.
    "mcts": _factory_pokeengine,
    "pokeengine": _factory_pokeengine,
    # Real Foul Play (full strategy: prepare_battles + multi-MCTS averaging).
    "foulplay": _factory_foulplay,
    "metamon": _factory_metamon,
}


def load_pool_entries(
    config_path: str,
    default_server_port: int = 9000,
) -> Tuple[List[PoolEntry], Optional[ExternalOpponentManager]]:
    """Read a YAML config and return (PoolEntry list, manager-or-None).

    The manager is set when at least one adapter type needs subprocess
    supervision (currently: metamon). Caller is responsible for calling
    manager.start_all() before training begins and manager.stop_all() after.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    entries: List[PoolEntry] = []
    spawn_specs: List[ExternalOpponent] = []
    ctx = {"default_server_port": default_server_port}

    for raw in cfg.get("opponents", []) or []:
        spec = dict(raw)
        name = spec.pop("name", None)
        kind = spec.pop("type", None)
        if not name or not kind:
            logger.warning("Skipping malformed external adapter entry: %s", raw)
            continue
        if kind not in _FACTORY_REGISTRY:
            logger.warning("Unknown adapter type %r for %s; skipping", kind, name)
            continue

        weight = float(spec.pop("weight", 1.0) or 1.0)
        spec["__name__"] = name
        spec["__weight__"] = weight

        try:
            entry, spawn_spec = _FACTORY_REGISTRY[kind](spec, ctx)
        except NotImplementedError as e:
            logger.warning("Adapter %s (%s) skipped: %s", name, kind, e)
            continue
        except Exception:
            logger.exception("Adapter %s (%s) failed to build; skipping", name, kind)
            continue

        entries.append(entry)
        if spawn_spec is not None:
            spawn_specs.append(spawn_spec)
        logger.info("Loaded external adapter: %s (%s, weight=%.2f, %s)",
                    name, kind, weight,
                    "subprocess" if spawn_spec else "in-process")

    manager: Optional[ExternalOpponentManager] = None
    if spawn_specs:
        manager = _BareManager(spawn_specs)

    return entries, manager


class _BareManager(ExternalOpponentManager):
    """ExternalOpponentManager populated from already-built ExternalOpponent objects.

    The base class loads from a YAML on disk; here we already have the spec list
    in memory (assembled from individual factory calls), so we skip the file read.
    """

    def __init__(self, opponents: List[ExternalOpponent]):
        # bypass _load_config — we already have the opponents list
        self.config_path = Path("/in-memory")
        self.base_dir = _PROJECT_ROOT
        self.opponents = list(opponents)
        self.servers = []
        self.default_team_folder = None
        import threading
        self._stop_event = threading.Event()
        self._monitor_thread = None
