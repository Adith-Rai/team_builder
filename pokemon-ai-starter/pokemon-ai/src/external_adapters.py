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

    # S67-EXT: store factory_kwargs explicitly on PoolEntry so CIS mode
    # can reconstruct the player in worker subprocesses (the factory closure
    # itself isn't pickleable across multiprocessing boundary, but the kwargs
    # dict is). Workers import PokeEnginePlayer directly + construct from
    # these kwargs. See mp_centralized_collect._play_vs_opp 'external_inprocess'
    # branch + train_rl.py CIS-pool conversion.
    entry = PoolEntry(
        kind="external", key=name, factory=_build, weight=weight,
        factory_kwargs={"factory_type": "pokeengine",
                        "search_time_ms": search_time_ms},
    )
    return entry, None  # no subprocess to manage


def _factory_heuristic(spec: dict, ctx: dict) -> Tuple[PoolEntry, List]:
    """In-process heuristic-bot adapter (S68 2026-06-10).

    YAML spec:
      - name: heur-greedysev2
        type: heuristic
        bot_class: GreedySEv2
        weight: 1.0

    Supported bot_class values (mapped to imports lazily in worker):
      From policy_trainbots.py (v2 set, SH-base, ~950-1043 Elo):
        GreedySEv2, SetupThenSweepv2, SwitchAwareEscapev3,
        HazardSensev2, AntiSetupBot, StrategicV2, SwitchAwareEscapeV2
      From policy_rulebots.py (raw Player base, ~730-830 Elo):
        GreedySEPlayer, HazardSensePlayer, SwitchAwareEscapePlayer, SetupThenSweepPlayer
      From poke_env.player.baselines:
        RandomPlayer, MaxBasePowerPlayer
        (SimpleHeuristicsPlayer is excluded — it's the eval SH bot.)

    Returns (PoolEntry, []) — no subprocess (in-process Player instance).
    """
    name = spec["__name__"]
    weight = spec["__weight__"]
    bot_class_name = spec.get("bot_class")
    if not bot_class_name:
        raise ValueError(f"heuristic adapter {name!r} missing required 'bot_class' field")

    def _build(server_configuration, account_configuration, team,
               battle_format, max_concurrent_battles, **kw):
        # Lazy import so training works when the bot modules aren't on path
        bot_cls = _resolve_heuristic_class(bot_class_name)
        return bot_cls(
            battle_format=battle_format,
            team=team,
            max_concurrent_battles=max_concurrent_battles,
            account_configuration=account_configuration,
            server_configuration=server_configuration,
        )

    entry = PoolEntry(
        kind="external", key=name, factory=_build, weight=weight,
        factory_kwargs={"factory_type": "heuristic", "bot_class": bot_class_name},
    )
    return entry, []  # no subprocess to manage


def _resolve_heuristic_class(name: str):
    """Map bot_class name string to the actual Python class (lazy import).

    Returned class is wrapped with `_wrap_with_sh_fallback` (see below) —
    a try/except around choose_move that catches any exception from the
    bot's custom logic and falls through to SimpleHeuristicsPlayer's
    choose_move. Required because poke-env's `_handle_battle_request`
    (player.py:416) calls `choose_move` with no try/except: if a custom
    bot's helper raises on an edge-case battle state, the exception
    propagates up and crashes the asyncio task driving the bot's
    WebSocket listener. All in-flight battles on that account become
    zombies, our worker waits in poll() for next-move requests Showdown
    will never send, and the 30-min asyncio batch timeout eventually
    fires as "timeout vs in-process <opp>". See S68 Run #9 diagnosis
    (2026-06-12): 16/16 worker timeouts on bots that own custom
    choose_move + helpers; 0 on bots that fall through to SH.
    """
    cls = None
    # Try policy_trainbots first (v2 set is the primary target)
    try:
        import policy_trainbots as _train
        if hasattr(_train, name):
            cls = getattr(_train, name)
    except ImportError:
        pass
    # Fall back to policy_rulebots (raw originals)
    if cls is None:
        try:
            import policy_rulebots as _rule
            if hasattr(_rule, name):
                cls = getattr(_rule, name)
        except ImportError:
            pass
    # Final fallback: poke-env baselines (Random, MaxBP)
    if cls is None:
        try:
            from poke_env.player import baselines as _baseline
            if hasattr(_baseline, name):
                cls = getattr(_baseline, name)
        except ImportError:
            pass
    if cls is None:
        raise ValueError(
            f"heuristic bot_class {name!r} not found in policy_trainbots, "
            f"policy_rulebots, or poke_env.player.baselines"
        )
    return _wrap_with_sh_fallback(cls)


def _wrap_with_sh_fallback(cls):
    """Return a subclass of `cls` whose choose_move catches any exception
    and falls through to SimpleHeuristicsPlayer.choose_move.

    Identity-preserving: when the bot's own choose_move runs without
    raising (the 99% path), behaviour is unchanged. The wrapper only
    fires on the rare exception that would otherwise crash poke-env's
    WebSocket listener task.

    Logs each caught exception at WARNING (one line — no traceback, to
    avoid the Rust-style formatting cost we hit in pokeengine_player.py
    panic recovery). Gives concrete confirmation of which bots raise on
    which battle states, so the underlying helper bugs can be fixed
    upstream later — but the hang is contained at this boundary now,
    so it isn't urgent.

    If the resolved class already inherits SimpleHeuristicsPlayer with
    a `super().choose_move(battle)` fallback (the v2-strong set), the
    wrapper still applies — they never raise in production, so the
    wrapper is a no-op for them. We don't try to detect that and skip
    wrapping: cheaper to wrap uniformly than to maintain a class-set
    allowlist.
    """
    from poke_env.player import SimpleHeuristicsPlayer
    from poke_env import AccountConfiguration

    class _SafeHeuristic(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Lazy-init SH fallback. We need a REAL SimpleHeuristicsPlayer
            # instance (not just SH.choose_move called with self) because
            # SH.choose_move uses `self._stat_estimation`, `self._estimate_matchup`,
            # `self._should_dynamax`, etc. — instance methods that only exist
            # on SH-inheriting classes. Calling SH.choose_move with a Player-base
            # `self` raises AttributeError immediately. Cache one per wrapper
            # instance; built with start_listening=False so no Showdown connection.
            self._sh_fallback = None

        def _get_sh_fallback(self):
            if self._sh_fallback is False:
                return None  # prior init failed; don't retry every turn
            if self._sh_fallback is None:
                try:
                    self._sh_fallback = SimpleHeuristicsPlayer(
                        account_configuration=AccountConfiguration(
                            f"shfb-{id(self):x}"[:18], None
                        ),
                        start_listening=False,
                    )
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as e:
                    logger.warning(
                        "[heur-safe] %s SH-fallback init failed (%s: %s) — "
                        "will use choose_random_move for this instance",
                        cls.__name__, type(e).__name__, e,
                    )
                    self._sh_fallback = False
                    return None
            return self._sh_fallback

        def choose_move(self, battle):
            try:
                return super().choose_move(battle)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:
                logger.warning(
                    "[heur-safe] %s raised %s on battle %s: %s — falling to SH",
                    cls.__name__, type(e).__name__,
                    getattr(battle, "battle_tag", "?"), e,
                )
                sh = self._get_sh_fallback()
                if sh is not None:
                    try:
                        return sh.choose_move(battle)
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except BaseException as e2:
                        logger.warning(
                            "[heur-safe] %s SH-fallback also raised %s: %s — using random",
                            cls.__name__, type(e2).__name__, e2,
                        )
                return self.choose_random_move(battle)

    # Preserve the original class name for any reflection/logging downstream
    _SafeHeuristic.__name__ = cls.__name__
    _SafeHeuristic.__qualname__ = cls.__qualname__
    _SafeHeuristic.__module__ = cls.__module__
    return _SafeHeuristic


def _factory_metamon(spec: dict, ctx: dict) -> Tuple[PoolEntry, List[ExternalOpponent]]:
    """Subprocess adapter — spawns metamon_accept_serve.py in metamon_venv.

    Returns (PoolEntry with logical opp metadata + per-instance usernames,
    List[ExternalOpponent] — one spawn spec per instance). The caller wires
    each ExternalOpponent into a manager that handles spawn + auto-restart.

    S67-ext-multi-instance (2026-05-27): YAML `instances: N` (default 1)
    spawns N independent subprocesses, each its own username and team queue
    dir (e.g., MM-Minikazam-0, MM-Minikazam-1, ...). This solves the
    Phase 2-ext production scale fan-in: with 3 workers per logical MM,
    1 subprocess (parallel_actors=1) becomes a serial bottleneck. N=3
    instances absorbs all 3 workers with no queue depth. cis-orch picks
    one instance per worker at iter-start, round-robin within the same
    logical opp. PFSP win-rate stays under the logical key (battles
    aggregate). For instances=1 (default), behavior is identical to
    the pre-multi-instance path.
    """
    name = spec["__name__"]
    weight = spec["__weight__"]
    model = spec.get("model", "Minikazam")
    base_username = spec.get("showdown_username") or f"MM-{model}"
    team_set = spec.get("team_set", "competitive")
    temperature = float(spec.get("temperature", 1.0))
    checkpoint = spec.get("checkpoint")  # int or None
    battle_format = spec.get("format", "gen9ou")
    server_port = int(spec.get("server_port", ctx.get("default_server_port", 9000)))
    num_battles = int(spec.get("num_battles", 100000))
    n_instances = max(1, int(spec.get("instances", 1)))

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

    # Coordinator-managed team queue. We hand Metamon our procedural Smogon
    # teams per battle so both sides match per-game without sharing process
    # state. Set to a unique-per-name dir so multiple Metamon variants don't
    # collide. `use_our_teams: false` in the YAML reverts to metamon's static
    # team set.
    use_our_teams = bool(spec.get("use_our_teams", True))

    # Pass env vars the subprocess needs (cache dir + version-check bypass)
    metamon_cache = os.environ.get("METAMON_CACHE_DIR") or str(_PROJECT_ROOT / "metamon_cache")
    os.environ["METAMON_CACHE_DIR"] = metamon_cache
    Path(metamon_cache).mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    # S67-ext F4 (2026-05-27 eve): distribute instances across the
    # battle_server port pool so battle_server.js (single-threaded Node)
    # doesn't bottleneck. ctx["available_ports"] is the full list (default
    # [9000..9007]); ctx["_port_cursor"] is a shared counter advanced across
    # factory calls so consecutive MMs don't collide. First instance per MM
    # uses spec's server_port if present in available; subsequent cycle the
    # cursor. Loose target: 15 instances / 8 ports ≈ 2 per port → battle_server
    # WS load drops from 50+ to ~6-8 per server.
    available_ports = ctx.get("available_ports") or [9000 + i for i in range(16)]
    if "_port_cursor" not in ctx:
        ctx["_port_cursor"] = 0

    # Build N instances. For instances=1, suffix is empty (legacy username
    # preserved for backward compat). For instances>1, suffix is "-{i}".
    spawn_specs: List[ExternalOpponent] = []
    instance_usernames: List[str] = []
    instance_team_queue_dirs: List[Optional[str]] = []
    instance_ports: List[int] = []
    for i in range(n_instances):
        suffix = "" if n_instances == 1 else f"-{i}"
        instance_username = f"{base_username}{suffix}"
        instance_name = f"{name}{suffix}"
        # F4 port assignment:
        # - Legacy single-instance (n_instances=1): honor spec's server_port.
        #   Most existing YAMLs set server_port=9000; staying on that keeps
        #   backward compatibility for old single-MM setups.
        # - Multi-instance: pure round-robin across available_ports via the
        #   shared cursor. Spec's server_port is ignored to ensure balanced
        #   distribution (otherwise all MMs cluster on the first 3 ports
        #   because they all default to server_port=9000).
        if n_instances == 1:
            instance_port = server_port
        else:
            instance_port = available_ports[ctx["_port_cursor"] % len(available_ports)]
            ctx["_port_cursor"] = (ctx["_port_cursor"] + 1) % len(available_ports)

        if use_our_teams:
            instance_queue_dir = _PROJECT_ROOT / "data" / "external_team_queue" / instance_name
            instance_queue_dir.mkdir(parents=True, exist_ok=True)
            instance_queue_str: Optional[str] = str(instance_queue_dir)
        else:
            instance_queue_str = None

        cmd = [
            str(venv_python),
            str(serve_script),
            "--model", str(model),
            "--username", str(instance_username),
            "--server-port", str(instance_port),
            "--format", str(battle_format),
            "--num-battles", str(num_battles),
            "--temperature", str(temperature),
        ]
        if instance_queue_str is not None:
            cmd += ["--team-queue", instance_queue_str]
        else:
            cmd += ["--team-set", str(team_set)]
        if checkpoint is not None:
            cmd += ["--checkpoint", str(int(checkpoint))]

        log_path = log_dir / f"{instance_name}.log"
        spawn_specs.append(ExternalOpponent(
            name=instance_name,
            showdown_username=instance_username,
            command=cmd,
            cwd=None,
            venv=None,
            auto_restart=True,
            log_file=str(log_path),
            description=f"Metamon {model} instance {i+1}/{n_instances} (subprocess)",
            # S67-ext per-iter spawn: logical_name is the YAML opp name
            # (e.g., "mm-minikazam") — all instances share it. Manager uses
            # it to find all instances of a logical opp for spawn_active().
            logical_name=name,
        ))
        instance_usernames.append(instance_username)
        instance_team_queue_dirs.append(instance_queue_str)
        instance_ports.append(instance_port)

    # Logical PoolEntry: legacy fields point at instance 0 for backward
    # compat in any code paths that don't yet know about instances.
    # Multi-instance routing uses instance_usernames + instance_team_queue_dirs.
    pool_entry = PoolEntry(
        kind="external",
        key=name,
        showdown_username=instance_usernames[0],
        team_queue_dir=instance_team_queue_dirs[0],
        weight=weight,
        instance_usernames=instance_usernames if n_instances > 1 else None,
        instance_team_queue_dirs=instance_team_queue_dirs if n_instances > 1 else None,
        instance_ports=instance_ports if n_instances > 1 else None,
    )
    return pool_entry, spawn_specs


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
    # S68 (2026-06-10) heuristic-bot training adapter — in-process Player
    # instances from policy_smartbots / policy_rulebots / policy_trainbots.
    # Designed for Run #9 heuristic-opp diversity: provides categorically-
    # different decision processes (not BC-derived neural opps) as training
    # guardrails. See memory/project_s68_bot_elo_findings_2026_06_10.md +
    # the user's "guardrails" framing.
    "heuristic": _factory_heuristic,
}


def load_pool_entries(
    config_path: str,
    default_server_port: int = 9000,
    available_ports: Optional[List[int]] = None,
) -> Tuple[List[PoolEntry], Optional[ExternalOpponentManager]]:
    """Read a YAML config and return (PoolEntry list, manager-or-None).

    The manager is set when at least one adapter type needs subprocess
    supervision (currently: metamon). Caller is responsible for calling
    manager.start_all() before training begins and manager.stop_all() after.

    `available_ports` is the full battle_server port pool used for MM
    instance round-robin distribution. If None, defaults to 16 ports
    (9000-9015). Caller (train_rl.py) should pass the parsed --servers
    list so MM instances align with the actual battle_server pool — too
    few ports causes cursor wrap-around and instance clustering, which
    was the S67 phase2_ext smoke v9 bottleneck.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    entries: List[PoolEntry] = []
    spawn_specs: List[ExternalOpponent] = []
    ctx: dict = {"default_server_port": default_server_port}
    if available_ports:
        ctx["available_ports"] = list(available_ports)

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
        # S67-ext-multi-instance: factory may return a single ExternalOpponent
        # (legacy single-subprocess), a list of N (multi-instance), or None
        # (in-process adapter, no subprocess needed). Normalize to a flat list.
        if spawn_spec is None:
            n_subprocs = 0
        elif isinstance(spawn_spec, list):
            spawn_specs.extend(spawn_spec)
            n_subprocs = len(spawn_spec)
        else:
            spawn_specs.append(spawn_spec)
            n_subprocs = 1
        logger.info("Loaded external adapter: %s (%s, weight=%.2f, %s)",
                    name, kind, weight,
                    f"{n_subprocs} subprocess(es)" if n_subprocs else "in-process")

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
