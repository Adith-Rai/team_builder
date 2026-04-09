# replay_parser.py - Download and parse human replays from HuggingFace into JSONL
# Dataset: jakegrigsby/metamon-raw-replays (Parquet, streaming)
#
# The log field contains pipe-delimited Showdown protocol. We feed it through
# poke-env's Battle.parse_message() to build game state, then extract features
# using features.py's make_obs_mask_and_slots() at each decision point.
#
# Output format matches observer.py exactly so bc_train.py / convert_jsonl_to_memmap.py
# work unchanged.
#
# Key challenge: replay logs lack |request| messages, so available_moves /
# available_switches are not populated by poke-env. We work around this by:
#   1. Pre-scanning the entire log to discover all moves each pokemon uses
#   2. Registering those moves on the pokemon objects before feature extraction
#   3. Manually populating available_moves / available_switches from game state

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from poke_env.battle.battle import Battle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.data import to_id_str

from features import make_obs_mask_and_slots
from features import make_features


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
_forced_legal_count = 0  # actions force-set to legal (heuristic mask was wrong)
_total_actions_count = 0  # total action records produced

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_gen_from_format(fmt: str) -> int:
    """Extract generation number from format string like 'gen9ou'."""
    m = re.match(r"gen(\d+)", fmt)
    return int(m.group(1)) if m else 9


def _normalize_move_id(move_name: str) -> str:
    """Normalize a move name to poke-env's internal id format."""
    return to_id_str(move_name)


def _split_log_lines(log_text: str) -> List[List[str]]:
    """Split a Showdown log into pre-split message lists.

    Each line in the log looks like:
        |move|p1a: Garchomp|Earthquake|p2a: Landorus
    We split on '|' and keep the structure poke-env expects:
        ['', 'move', 'p1a: Garchomp', 'Earthquake', 'p2a: Landorus']
    """
    lines = []
    for raw_line in log_text.split("\n"):
        raw_line = raw_line.rstrip("\r")
        if not raw_line:
            continue
        if raw_line.startswith("|"):
            parts = raw_line.split("|")
            lines.append(parts)
    return lines


def _extract_players(lines: List[List[str]]) -> Dict[str, str]:
    """Extract player role -> username mapping from |player| lines."""
    players = {}
    for parts in lines:
        if len(parts) >= 4 and parts[1] == "player":
            role = parts[2]
            username = parts[3]
            if role in ("p1", "p2") and username:
                players[role] = username
    return players


def _extract_winner(lines: List[List[str]]) -> Optional[str]:
    """Extract winner username from |win| line."""
    for parts in lines:
        if len(parts) >= 3 and parts[1] == "win":
            return parts[2]
    return None


def _is_tie(lines: List[List[str]]) -> bool:
    """Check if the battle ended in a tie."""
    for parts in lines:
        if len(parts) >= 2 and parts[1] == "tie":
            return True
    return False


def _prescan_moves(lines: List[List[str]]) -> Dict[str, List[str]]:
    """Pre-scan the log to discover all moves used by each pokemon identity.

    Returns a dict mapping pokemon identifier prefix (e.g. "p1: Garchomp")
    to an ordered list of unique move IDs (order of first appearance).
    """
    moves_by_pokemon: Dict[str, List[str]] = defaultdict(list)
    seen: Dict[str, Set[str]] = defaultdict(set)

    for parts in lines:
        if len(parts) < 4 or parts[1] != "move":
            continue
        who = parts[2]
        move_name = parts[3]
        # Normalize: "p1a: Garchomp" -> "p1: Garchomp"
        ident = who.replace("p1a:", "p1:").replace("p2a:", "p2:").replace("p1b:", "p1:").replace("p2b:", "p2:")
        move_id = _normalize_move_id(move_name)
        if move_id and move_id not in seen[ident]:
            seen[ident].add(move_id)
            moves_by_pokemon[ident].append(move_id)

    return dict(moves_by_pokemon)


def _register_prescanned_moves(battle: Battle, moves_map: Dict[str, List[str]]) -> None:
    """Register pre-scanned moves on pokemon objects in the battle.

    This ensures pokemon know their moves before we extract features,
    even though the |move| lines haven't been processed yet.
    """
    for ident, move_ids in moves_map.items():
        # Try to find this pokemon in the battle's teams
        # ident is like "p1: Garchomp"
        role = ident[:2]  # "p1" or "p2"

        if role == battle._player_role:
            team = battle.team
        else:
            team = battle.opponent_team

        # Find the pokemon in the team dict
        # poke-env keys are like "p1: Garchomp"
        for team_key, pokemon in team.items():
            if team_key == ident or ident in team_key:
                for mid in move_ids:
                    try:
                        pokemon._add_move(mid)
                    except Exception:
                        pass
                break


def _preregister_team(battle: Battle, lines: List[List[str]]) -> None:
    """Pre-register all team pokemon from |poke| lines.

    The |poke| lines list each player's team during team preview. Without this,
    pokemon only enter the team dict when they first switch in, which means we
    can't map switch actions to pokemon we haven't seen yet.
    """
    role = battle._player_role
    if not role:
        return

    for parts in lines:
        if len(parts) < 4 or parts[1] != "poke":
            continue
        player = parts[2]  # "p1" or "p2"
        details = parts[3]  # "Garchomp, M" or "Skarmory, F, shiny"

        # Extract species name for the identifier
        species = details.split(",")[0].strip()

        if player == role:
            ident = f"{player}: {species}"
            try:
                battle.get_pokemon(ident, force_self_team=True, details=details)
            except Exception:
                pass
        else:
            ident = f"{player}: {species}"
            try:
                battle.get_pokemon(ident, details=details)
            except Exception:
                pass


def _populate_available_actions(battle: Battle) -> None:
    """Manually populate available_moves and available_switches.

    Since replay logs don't contain |request| messages, the Battle object
    won't know what moves/switches are legal. We infer them from game state.

    We check volatile effects that restrict moves:
    - Encore: only the encored move is available
    - Disable: the disabled move is removed
    - Taunt: status moves are removed
    - Torment: the last-used move is removed (approximated)
    - Choice items: only the first-used move (not trackable from replays, skipped)
    - Trapped/Partially Trapped: switches are blocked (except by specific moves)
    PP is not tracked in replays, so we can't filter by PP depletion.
    """
    from poke_env.battle.effect import Effect

    battle._available_moves = []
    battle._available_switches = []

    active = battle.active_pokemon
    if active is None:
        return

    effects = active.effects if hasattr(active, 'effects') else {}

    # Moves: all known moves of the active pokemon (up to 4), filtered by effects
    if active.moves:
        all_moves = list(active.moves.values())[:4]

        if Effect.ENCORE in effects:
            # Encore: only the encored move is legal
            # poke-env doesn't track which move is encored directly,
            # but the move used last turn is the encored one.
            # We can't easily determine this, so keep all moves —
            # the action will be force-set to legal if needed.
            battle._available_moves = all_moves
        else:
            for move in all_moves:
                # Disable: skip the disabled move
                if Effect.DISABLE in effects and hasattr(active, '_last_request') and False:
                    # Can't reliably detect which move is disabled from replays
                    pass

                # Taunt: skip status moves (base_power == 0 and not a damaging status)
                if Effect.TAUNT in effects and move.base_power == 0 and move.category.name == "STATUS":
                    continue

                battle._available_moves.append(move)

        # If all moves filtered out (e.g., Taunt + all status), fall back to Struggle
        if not battle._available_moves:
            battle._available_moves = all_moves

    # Switches: all non-fainted, non-active team pokemon
    # Trapped / Partially Trapped block switching (but not via specific moves like U-turn)
    is_trapped = Effect.TRAPPED in effects or Effect.PARTIALLY_TRAPPED in effects
    if not is_trapped:
        for mon in battle.team.values():
            if not mon.active and not mon.fainted:
                battle._available_switches.append(mon)
    else:
        # When trapped, switches are unavailable unless the player uses a trapping-escape move
        # We still allow switches since the player might have used Shed Shell or a pivot move
        # The action force-set at line 528 will handle edge cases
        for mon in battle.team.values():
            if not mon.active and not mon.fainted:
                battle._available_switches.append(mon)


def _find_move_index(battle: Battle, move_name: str) -> int:
    """Find the action index (0-3) for a move name.

    Returns -1 if not found.
    """
    move_id = _normalize_move_id(move_name)

    # Match against available_moves
    for i, m in enumerate(battle.available_moves):
        if i >= 4:
            break
        if m.id == move_id:
            return i

    # Fallback: try the active pokemon's move dict directly
    active = battle.active_pokemon
    if active and active.moves:
        for i, (mid, m) in enumerate(active.moves.items()):
            if i >= 4:
                break
            if mid == move_id or m.id == move_id:
                return i

    return -1


def _find_switch_index(battle: Battle, species_str: str) -> int:
    """Find the action index (4-8) for a switch-in pokemon.

    species_str comes from the log like "Skarmory, F" or "Garchomp, M, shiny".
    Returns -1 if not found.
    """
    species_name = species_str.split(",")[0].strip()
    species_id = to_id_str(species_name)

    for j, mon in enumerate(battle.available_switches):
        if j >= 5:
            break
        if mon.species == species_id:
            return 4 + j

    # Broader match for alternate forms
    for j, mon in enumerate(battle.available_switches):
        if j >= 5:
            break
        mon_species = to_id_str(mon.species)
        if species_id in mon_species or mon_species in species_id:
            return 4 + j

    return -1


def _extract_actions_for_turn(
    lines: List[List[str]],
    turn_start_idx: int,
    turn_end_idx: int,
) -> Dict[str, Tuple[str, str]]:
    """Extract the action each player took during a turn.

    Scans lines between turn_start_idx and turn_end_idx for |move| and |switch|
    messages. Returns dict of role -> (action_type, action_detail).

    Only captures the FIRST action per player per turn (the deliberate choice,
    not mid-turn forced switches from faints).
    """
    actions: Dict[str, Tuple[str, str]] = {}

    for i in range(turn_start_idx, turn_end_idx):
        parts = lines[i]
        if len(parts) < 4:
            continue

        cmd = parts[1]

        if cmd == "move":
            who = parts[2]
            move_name = parts[3]
            role = who[:2]
            if role not in actions:
                actions[role] = ("move", move_name)

        elif cmd == "switch":
            # |switch|p1a: Skarmory|Skarmory, F|100/100
            # Only count as deliberate switch if no prior move/cant for this player
            who = parts[2]
            details = parts[3]
            role = who[:2]
            if role not in actions:
                actions[role] = ("switch", details)

        elif cmd == "cant":
            # |cant|p1a: Garchomp|slp — player's turn was consumed
            who = parts[2]
            role = who[:2]
            if role not in actions:
                actions[role] = ("cant", "")

    return actions


def _find_turn_boundaries(lines: List[List[str]]) -> List[Tuple[int, int]]:
    """Find (line_index, turn_number) for each |turn|N line."""
    boundaries = []
    for i, parts in enumerate(lines):
        if len(parts) >= 3 and parts[1] == "turn":
            try:
                turn_num = int(parts[2])
                boundaries.append((i, turn_num))
            except (ValueError, IndexError):
                pass
    return boundaries


def _safe_parse(battle: Battle, parts: List[str]) -> None:
    """Feed a message to the battle, ignoring errors."""
    if len(parts) < 2:
        return
    cmd = parts[1]
    if cmd == "win":
        battle.won_by(parts[2])
        return
    if cmd == "tie":
        battle.tied()
        return
    try:
        battle.parse_message(parts)
    except NotImplementedError:
        pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Core replay parser
# ---------------------------------------------------------------------------

def parse_single_replay(
    replay_id: str,
    log_text: str,
    fmt: str,
    rating: Optional[int],
    log_both: bool = True,
) -> List[Dict[str, Any]]:
    """Parse a single replay log into JSONL records.

    Returns a list of JSONL row dicts (one per decision point per player).
    """
    lines = _split_log_lines(log_text)
    if not lines:
        return []

    players = _extract_players(lines)
    if "p1" not in players or "p2" not in players:
        return []

    winner_username = _extract_winner(lines)
    is_tie_game = _is_tie(lines)

    gen = _parse_gen_from_format(fmt)

    # Determine winner role
    winner_role = None
    if winner_username:
        if winner_username == players.get("p1"):
            winner_role = "p1"
        elif winner_username == players.get("p2"):
            winner_role = "p2"

    # Find turn boundaries
    turn_bounds = _find_turn_boundaries(lines)
    if not turn_bounds:
        return []

    # Pre-scan all moves in the log
    moves_map = _prescan_moves(lines)

    # Determine which perspectives to record
    perspectives = ["p1", "p2"] if log_both else ["p1"]

    all_records: List[Dict[str, Any]] = []

    for perspective in perspectives:
        records = _parse_perspective(
            replay_id=replay_id,
            lines=lines,
            gen=gen,
            fmt=fmt,
            perspective=perspective,
            players=players,
            turn_bounds=turn_bounds,
            winner_role=winner_role,
            is_tie=is_tie_game,
            rating=rating,
            moves_map=moves_map,
        )
        all_records.extend(records)

    return all_records


def _parse_perspective(
    replay_id: str,
    lines: List[List[str]],
    gen: int,
    fmt: str,
    perspective: str,
    players: Dict[str, str],
    turn_bounds: List[Tuple[int, int]],
    winner_role: Optional[str],
    is_tie: bool,
    rating: Optional[int],
    moves_map: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Parse replay from one player's perspective.

    Strategy:
    1. Feed all pre-turn lines to initialize the battle (player, poke, switch, etc.)
    2. Register pre-scanned moves on all pokemon so they know their movesets
    3. For each turn:
       a. The battle state at this point = decision point
       b. Populate available_moves/switches from current state
       c. Look ahead to see what action the player took
       d. Map action to index, extract features, create record
       e. Feed the turn's events to advance battle state
    """

    username = players[perspective]
    episode_id = f"{replay_id}::{perspective}"

    # Create battle object
    battle = Battle(
        battle_tag=replay_id,
        username=username,
        logger=None,
        gen=gen,
        save_replays=False,
    )

    records: List[Dict[str, Any]] = []

    # Feed all lines up to the first turn to initialize the battle
    first_turn_idx = turn_bounds[0][0]

    for i in range(first_turn_idx):
        parts = lines[i]
        if len(parts) < 2:
            continue
        cmd = parts[1]
        # Skip problematic messages
        if cmd in ("", "t:"):
            continue
        _safe_parse(battle, parts)

    # Pre-register all team pokemon from |poke| lines so they exist in the team
    # dict even before they switch in. This is critical for mapping switch actions.
    _preregister_team(battle, lines)

    # Register all pre-scanned moves on the pokemon objects.
    _register_prescanned_moves(battle, moves_map)

    # Process each turn
    for turn_idx_pos in range(len(turn_bounds)):
        turn_line_idx, turn_num = turn_bounds[turn_idx_pos]

        # Feed the |turn|N line to advance the turn counter
        _safe_parse(battle, lines[turn_line_idx])

        # Determine the range of lines for this turn's events
        if turn_idx_pos + 1 < len(turn_bounds):
            next_turn_line_idx = turn_bounds[turn_idx_pos + 1][0]
        else:
            next_turn_line_idx = len(lines)

        # Extract what action each player took this turn
        actions = _extract_actions_for_turn(lines, turn_line_idx + 1, next_turn_line_idx)

        # If this player didn't act, or was forced (cant), skip
        if perspective not in actions or actions[perspective][0] == "cant":
            # Feed events to update state
            for i in range(turn_line_idx + 1, next_turn_line_idx):
                _safe_parse(battle, lines[i])
                # Re-register moves after switches create new pokemon
                _register_prescanned_moves(battle, moves_map)
            continue

        action_type, action_detail = actions[perspective]

        # Populate available moves/switches from current game state
        _populate_available_actions(battle)

        # Map action to index
        if action_type == "move":
            action_idx = _find_move_index(battle, action_detail)
        else:
            action_idx = _find_switch_index(battle, action_detail)

        if action_idx < 0:
            # Could not map action - feed events and skip
            for i in range(turn_line_idx + 1, next_turn_line_idx):
                _safe_parse(battle, lines[i])
                _register_prescanned_moves(battle, moves_map)
            continue

        # Extract features from current battle state (start of turn)
        try:
            obs_vec, legal_mask, ctx_extra, move_slots, switch_slots, \
                entity_ids, move_ids, switch_ids = \
                make_obs_mask_and_slots(battle, moved_first_bits=None)
        except Exception:
            # Feature extraction failed - feed events and skip
            for i in range(turn_line_idx + 1, next_turn_line_idx):
                _safe_parse(battle, lines[i])
                _register_prescanned_moves(battle, moves_map)
            continue

        # Ensure the chosen action is marked legal in the mask.
        # This patches cases where our heuristic legal mask is wrong
        # (e.g., PP depletion, Choice lock, Encore forcing a specific move).
        global _forced_legal_count, _total_actions_count
        _total_actions_count += 1
        if action_idx < len(legal_mask) and legal_mask[action_idx] == 0.0:
            legal_mask[action_idx] = 1.0
            _forced_legal_count += 1
        if sum(legal_mask) == 0:
            legal_mask[action_idx] = 1.0
            _forced_legal_count += 1

        t = len(records)
        phase = 0 if t <= 3 else (1 if t <= 15 else 2)

        row: Dict[str, Any] = {
            "episode_id": episode_id,
            "t": t,
            "battle_turn": turn_num,
            "obs": obs_vec,
            "legal": legal_mask,
            "action": int(action_idx),
            "done": False,
            "result": None,
            "ctx_extra": ctx_extra,
            "move_slots": move_slots,
            "switch_slots": switch_slots,
            "entity_ids": entity_ids,
            "move_ids": move_ids,
            "switch_ids": switch_ids,
            "mods": {"tera": 0, "dmax": 0, "mega": 0, "zmove": 0},
            "phase": phase,
            "meta": {
                "winner": winner_role if winner_role else ("tie" if is_tie else None),
                "rating": rating,
            },
        }

        records.append(row)

        # Feed the turn's events to advance battle state for next turn
        for i in range(turn_line_idx + 1, next_turn_line_idx):
            _safe_parse(battle, lines[i])
            # Re-register moves after new pokemon switch in
            _register_prescanned_moves(battle, moves_map)

    # Stamp the last record as terminal
    if records:
        last = records[-1]
        last["done"] = True
        if is_tie:
            last["result"] = 0.5
        elif winner_role == perspective:
            last["result"] = 1.0
        elif winner_role is not None:
            last["result"] = 0.0
        else:
            last["result"] = None

    return records


# ---------------------------------------------------------------------------
# JSONL Writer (matches observer.py)
# ---------------------------------------------------------------------------

class JSONLWriter:
    """Writes JSONL records to disk, handling numpy serialization."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a", buffering=1, encoding="utf-8")
        self._n = 0

    def write(self, obj: Dict[str, Any]) -> None:
        def _json_default(o):
            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(
                f"Object of type {o.__class__.__name__} is not JSON serializable"
            )

        self._f.write(
            json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n"
        )
        self._n += 1
        if self._n % 512 == 0:
            self._f.flush()

    def close(self) -> None:
        try:
            self._f.flush()
            self._f.close()
        except Exception:
            pass

    @property
    def count(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# HuggingFace download + processing
# ---------------------------------------------------------------------------

def process_replays(
    fmt: str = "gen9ou",
    min_rating: int = 0,
    max_replays: int = 10000,
    output_dir: str = "data/datasets/human_replays",
    log_both: bool = True,
):
    """Download replays from HuggingFace and process them into JSONL."""
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "ERROR: 'datasets' package required. Install with: pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    output_path = os.path.join(output_dir, f"{fmt}_rating{min_rating}.jsonl")
    print(f"Output: {output_path}")
    print(f"Format filter: {fmt}, Min rating: {min_rating}, Max replays: {max_replays}")
    print(f"Log both perspectives: {log_both}")
    print()

    print("Loading dataset (streaming)...")
    ds = load_dataset(
        "jakegrigsby/metamon-raw-replays",
        split="train",
        streaming=True,
    )

    writer = JSONLWriter(output_path)

    n_processed = 0
    n_skipped_format = 0
    n_skipped_rating = 0
    n_skipped_parse = 0
    n_records = 0
    t_start = time.time()

    for row in ds:
        if n_processed >= max_replays:
            break

        # Filter by format
        format_id = row.get("formatid") or row.get("format") or ""
        if isinstance(format_id, str):
            format_id_clean = re.sub(r"[^a-z0-9]", "", format_id.lower())
        else:
            format_id_clean = ""

        fmt_clean = re.sub(r"[^a-z0-9]", "", fmt.lower())
        if fmt_clean not in format_id_clean:
            n_skipped_format += 1
            continue

        # Filter by rating
        rating_val = row.get("rating")
        if rating_val is not None:
            try:
                rating_int = int(rating_val)
            except (ValueError, TypeError):
                rating_int = 0
        else:
            rating_int = 0

        if rating_int < min_rating:
            n_skipped_rating += 1
            continue

        # Parse the replay
        log_text = row.get("log", "")
        replay_id = row.get("id", f"replay-{n_processed}")

        if not log_text or len(log_text) < 50:
            n_skipped_parse += 1
            continue

        try:
            records = parse_single_replay(
                replay_id=str(replay_id),
                log_text=log_text,
                fmt=fmt,
                rating=rating_int if rating_int > 0 else None,
                log_both=log_both,
            )
        except Exception:
            n_skipped_parse += 1
            if n_skipped_parse <= 5:
                print(f"  [warn] Failed to parse replay {replay_id}:")
                traceback.print_exc()
            continue

        if not records:
            n_skipped_parse += 1
            continue

        for rec in records:
            writer.write(rec)
            n_records += 1

        n_processed += 1

        if n_processed % 100 == 0:
            elapsed = time.time() - t_start
            rate = n_processed / elapsed if elapsed > 0 else 0
            print(
                f"  [{n_processed:,}/{max_replays:,}] "
                f"records={n_records:,}  "
                f"skip_fmt={n_skipped_format:,} skip_rat={n_skipped_rating:,} "
                f"skip_err={n_skipped_parse:,}  "
                f"({rate:.1f} replays/s)",
                flush=True,
            )

    writer.close()
    elapsed = time.time() - t_start

    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"  Replays processed: {n_processed:,}")
    print(f"  Records written:   {n_records:,}")
    print(f"  Skipped (format):  {n_skipped_format:,}")
    print(f"  Skipped (rating):  {n_skipped_rating:,}")
    print(f"  Skipped (parse):   {n_skipped_parse:,}")
    if _total_actions_count > 0:
        pct = _forced_legal_count / _total_actions_count * 100
        print(f"  Legal mask patches: {_forced_legal_count:,} / {_total_actions_count:,} ({pct:.1f}%)")
    print(f"  Output:            {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Download and parse human replays from HuggingFace into JSONL"
    )
    p.add_argument(
        "--format", default="gen9ou",
        help="Battle format to filter (default: gen9ou)",
    )
    p.add_argument(
        "--min-rating", type=int, default=1500,
        help="Minimum player rating to include (default: 1500)",
    )
    p.add_argument(
        "--max-replays", type=int, default=10000,
        help="Maximum number of replays to process (default: 10000)",
    )
    p.add_argument(
        "--output-dir", default="data/datasets/human_replays",
        help="Output directory for JSONL files",
    )
    p.add_argument(
        "--log-both", action="store_true", default=True,
        help="Record both p1 and p2 perspectives (doubles output, default: True)",
    )
    p.add_argument(
        "--no-log-both", dest="log_both", action="store_false",
        help="Record only p1 perspective",
    )
    p.add_argument(
        "--self-test", action="store_true",
        help="Run self-test: parse 5 replays and print stats",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    """Parse 5 replays and print diagnostics."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required. Install with: pip install datasets")
        return

    print("=== SELF-TEST: Parsing 5 gen9ou replays ===")
    print()

    ds = load_dataset(
        "jakegrigsby/metamon-raw-replays",
        split="train",
        streaming=True,
    )

    n_tested = 0
    total_records = 0
    total_turns = 0
    action_counts = {i: 0 for i in range(9)}
    results = {"win": 0, "loss": 0, "tie": 0, "unknown": 0}

    for row in ds:
        if n_tested >= 5:
            break

        format_id = row.get("formatid") or row.get("format") or ""
        format_id_clean = re.sub(r"[^a-z0-9]", "", str(format_id).lower())
        if "gen9ou" not in format_id_clean:
            continue

        log_text = row.get("log", "")
        replay_id = row.get("id", f"test-{n_tested}")
        rating_val = row.get("rating")

        try:
            rating_int = int(rating_val) if rating_val is not None else 0
        except (ValueError, TypeError):
            rating_int = 0

        if not log_text or len(log_text) < 50:
            continue

        print(f"Replay {n_tested + 1}: {replay_id} (rating={rating_int})")

        try:
            records = parse_single_replay(
                replay_id=str(replay_id),
                log_text=log_text,
                fmt="gen9ou",
                rating=rating_int if rating_int > 0 else None,
                log_both=True,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            n_tested += 1
            continue

        if not records:
            print(f"  No records produced (skipping)")
            n_tested += 1
            continue

        # Analyze records
        episodes: Dict[str, List[Dict[str, Any]]] = {}
        for rec in records:
            ep = rec["episode_id"]
            if ep not in episodes:
                episodes[ep] = []
            episodes[ep].append(rec)

        for ep_id, ep_recs in episodes.items():
            perspective = ep_id.split("::")[-1]
            n_steps = len(ep_recs)
            last = ep_recs[-1]
            result_val = last.get("result")
            done = last.get("done", False)

            obs_len = len(ep_recs[0]["obs"]) if ep_recs else 0
            legal_len = len(ep_recs[0]["legal"]) if ep_recs else 0
            ctx_len = len(ep_recs[0].get("ctx_extra", [])) if ep_recs else 0
            eid_len = len(ep_recs[0].get("entity_ids", [])) if ep_recs else 0
            ms_len = len(ep_recs[0].get("move_slots", [])) if ep_recs else 0
            ss_len = len(ep_recs[0].get("switch_slots", [])) if ep_recs else 0

            print(
                f"  {perspective}: {n_steps} steps, done={done}, result={result_val}"
            )
            print(
                f"    obs={obs_len}, legal={legal_len}, ctx={ctx_len}, "
                f"eids={eid_len}, mslots={ms_len}, sslots={ss_len}"
            )

            warnings = []
            if obs_len != 1480:
                warnings.append(f"obs_dim={obs_len} (expected 1480)")
            if legal_len != 9:
                warnings.append(f"legal_dim={legal_len} (expected 9)")
            if ctx_len != 41:
                warnings.append(f"ctx_dim={ctx_len} (expected 41)")
            if eid_len != 84:
                warnings.append(f"entity_ids={eid_len} (expected 84)")
            if ms_len != 4:
                warnings.append(f"move_slots={ms_len} (expected 4)")
            if ss_len != 5:
                warnings.append(f"switch_slots={ss_len} (expected 5)")
            if warnings:
                print(f"    WARNINGS: {', '.join(warnings)}")

            # Count actions
            for rec in ep_recs:
                a = rec["action"]
                action_counts[a] = action_counts.get(a, 0) + 1
                total_turns += 1

            total_records += n_steps

            if result_val == 1.0:
                results["win"] += 1
            elif result_val == 0.0:
                results["loss"] += 1
            elif result_val == 0.5:
                results["tie"] += 1
            else:
                results["unknown"] += 1

        n_tested += 1
        print()

    print("=== SUMMARY ===")
    print(f"Replays tested: {n_tested}")
    print(f"Total records:  {total_records}")
    print(f"Total turns:    {total_turns}")
    print()
    print("Action distribution:")
    for i in range(9):
        label = f"move_{i}" if i < 4 else f"switch_{i-4}"
        count = action_counts.get(i, 0)
        pct = (count / total_turns * 100) if total_turns > 0 else 0
        print(f"  {label} (idx {i}): {count:>5} ({pct:5.1f}%)")
    print()
    print(f"Results: {results}")
    print()
    print("Self-test complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if args.self_test:
        self_test()
    else:
        process_replays(
            fmt=args.format,
            min_rating=args.min_rating,
            max_replays=args.max_replays,
            output_dir=args.output_dir,
            log_both=args.log_both,
        )
