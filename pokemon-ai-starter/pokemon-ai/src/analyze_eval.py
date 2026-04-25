#!/usr/bin/env python3
"""Analyze eval replays to understand model behavior patterns and compare playstyles."""

import re, sys, os, glob, json, math
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

# ── Move categories for playstyle classification ──
SETUP_MOVES = {
    'Swords Dance', 'Dragon Dance', 'Nasty Plot', 'Calm Mind', 'Quiver Dance',
    'Bulk Up', 'Iron Defense', 'Agility', 'Shell Smash', 'Shift Gear',
    'Coil', 'Belly Drum', 'Work Up', 'Geomancy', 'Tail Glow', 'Growth',
    'Hone Claws', 'Autotomize', 'Cotton Guard', 'Cosmic Power', 'Curse',
    'Acid Armor', 'Amnesia', 'Barrier', 'Double Team', 'Minimize',
    'Charge Beam', 'Flame Charge', 'Victory Dance', 'Torch Song',
}
PIVOT_MOVES = {
    'U-turn', 'Volt Switch', 'Flip Turn', 'Parting Shot', 'Teleport',
    'Baton Pass', 'Chilly Reception',
}
HAZARD_MOVES = {
    'Stealth Rock', 'Spikes', 'Toxic Spikes', 'Sticky Web', 'Ceaseless Edge',
    'Stone Axe',
}
HAZARD_REMOVAL = {
    'Rapid Spin', 'Defog', 'Mortal Spin', 'Tidy Up', 'Court Change',
}
STATUS_MOVES = {
    'Will-O-Wisp', 'Thunder Wave', 'Toxic', 'Glare', 'Nuzzle', 'Yawn',
    'Stun Spore', 'Sleep Powder', 'Spore', 'Hypnosis', 'Lovely Kiss',
}
RECOVERY_MOVES = {
    'Roost', 'Recover', 'Soft-Boiled', 'Slack Off', 'Wish', 'Synthesis',
    'Morning Sun', 'Moonlight', 'Shore Up', 'Rest', 'Pain Split',
    'Strength Sap', 'Leech Life', 'Drain Punch', 'Giga Drain',
    'Oblivion Wing', 'Horn Leech',
}
PROTECT_MOVES = {
    'Protect', 'Detect', 'King\'s Shield', 'Baneful Bunker', 'Spiky Shield',
    'Silk Trap', 'Burning Bulwark', 'Max Guard',
}

# ── Type effectiveness chart for switch quality analysis ──
# Maps (attacking_type, defending_type) -> multiplier.  Only non-1.0 entries stored.
_SE = 2.0
_NVE = 0.5
_IMM = 0.0
_TYPE_CHART_RAW = {
    'Normal':   {'Rock': _NVE, 'Ghost': _IMM, 'Steel': _NVE},
    'Fire':     {'Fire': _NVE, 'Water': _NVE, 'Grass': _SE, 'Ice': _SE, 'Bug': _SE,
                 'Rock': _NVE, 'Dragon': _NVE, 'Steel': _SE},
    'Water':    {'Fire': _SE, 'Water': _NVE, 'Grass': _NVE, 'Ground': _SE, 'Rock': _SE, 'Dragon': _NVE},
    'Electric': {'Water': _SE, 'Electric': _NVE, 'Grass': _NVE, 'Ground': _IMM, 'Flying': _SE, 'Dragon': _NVE},
    'Grass':    {'Fire': _NVE, 'Water': _SE, 'Grass': _NVE, 'Poison': _NVE, 'Ground': _SE,
                 'Flying': _NVE, 'Bug': _NVE, 'Rock': _SE, 'Dragon': _NVE, 'Steel': _NVE},
    'Ice':      {'Fire': _NVE, 'Water': _NVE, 'Grass': _SE, 'Ice': _NVE, 'Ground': _SE,
                 'Flying': _SE, 'Dragon': _SE, 'Steel': _NVE},
    'Fighting': {'Normal': _SE, 'Ice': _SE, 'Poison': _NVE, 'Flying': _NVE, 'Psychic': _NVE,
                 'Bug': _NVE, 'Rock': _SE, 'Ghost': _IMM, 'Dark': _SE, 'Steel': _SE, 'Fairy': _NVE},
    'Poison':   {'Grass': _SE, 'Poison': _NVE, 'Ground': _NVE, 'Rock': _NVE, 'Ghost': _NVE,
                 'Steel': _IMM, 'Fairy': _SE},
    'Ground':   {'Fire': _SE, 'Electric': _SE, 'Grass': _NVE, 'Poison': _SE, 'Flying': _IMM,
                 'Bug': _NVE, 'Rock': _SE, 'Steel': _SE},
    'Flying':   {'Electric': _NVE, 'Grass': _SE, 'Fighting': _SE, 'Bug': _SE, 'Rock': _NVE, 'Steel': _NVE},
    'Psychic':  {'Fighting': _SE, 'Poison': _SE, 'Psychic': _NVE, 'Dark': _IMM, 'Steel': _NVE},
    'Bug':      {'Fire': _NVE, 'Grass': _SE, 'Fighting': _NVE, 'Poison': _NVE, 'Flying': _NVE,
                 'Psychic': _SE, 'Ghost': _NVE, 'Dark': _SE, 'Steel': _NVE, 'Fairy': _NVE},
    'Rock':     {'Fire': _SE, 'Ice': _SE, 'Fighting': _NVE, 'Ground': _NVE, 'Flying': _SE,
                 'Bug': _SE, 'Steel': _NVE},
    'Ghost':    {'Normal': _IMM, 'Psychic': _SE, 'Ghost': _SE, 'Dark': _NVE},
    'Dragon':   {'Dragon': _SE, 'Steel': _NVE, 'Fairy': _IMM},
    'Dark':     {'Fighting': _NVE, 'Psychic': _SE, 'Ghost': _SE, 'Dark': _NVE, 'Fairy': _NVE},
    'Steel':    {'Fire': _NVE, 'Water': _NVE, 'Electric': _NVE, 'Ice': _SE, 'Rock': _SE,
                 'Steel': _NVE, 'Fairy': _SE},
    'Fairy':    {'Fire': _NVE, 'Poison': _NVE, 'Fighting': _SE, 'Dragon': _SE, 'Dark': _SE, 'Steel': _NVE},
}

# Species -> types lookup via poke-env (covers all 1547 species)
try:
    from poke_env.data.gen_data import GenData as _GenData
    _POKEDEX = _GenData.from_gen(9).pokedex
except Exception:
    _POKEDEX = {}

def _get_species_types(species):
    """Look up types for a species name via poke-env pokedex."""
    if not _POKEDEX:
        return None
    # Convert display name to pokedex key: "Great Tusk" -> "greattusk"
    key = re.sub(r'[^a-z0-9]', '', species.lower())
    entry = _POKEDEX.get(key)
    if entry:
        return entry.get('types')
    # Try base form: "Rotom-Wash" -> "rotom" (fallback)
    base = re.sub(r'[^a-z0-9]', '', species.split('-')[0].lower())
    entry = _POKEDEX.get(base)
    if entry:
        return entry.get('types')
    return None


def _type_effectiveness(atk_types, def_types):
    """Compute aggregate type effectiveness multiplier.
    atk_types: list of attacker's types (STAB coverage)
    def_types: list of defender's types
    Returns best single-type multiplier (max over attacker types, product over defender types).
    """
    if not atk_types or not def_types:
        return 1.0
    best = 0.0
    for at in atk_types:
        mult = 1.0
        chart = _TYPE_CHART_RAW.get(at, {})
        for dt in def_types:
            mult *= chart.get(dt, 1.0)
        best = max(best, mult)
    return best


def parse_replay(filepath):
    """Parse a Showdown replay HTML file into structured battle data."""
    with open(filepath, encoding='utf-8') as f:
        html = f.read()

    m = re.search(r'<script type="text/plain" class="battle-log-data">(.*?)</script>', html, re.DOTALL)
    if not m:
        return None

    log = m.group(1).strip()
    lines = log.split('\n')

    # Extract battle_id from filename (e.g. "battle-gen9ou-220725")
    fname = os.path.basename(filepath)
    bid_match = re.search(r'(battle-\S+?)\.html', fname)
    battle_id = bid_match.group(1) if bid_match else fname

    battle = {
        'file': fname,
        'battle_id': battle_id,
        'p1': None, 'p2': None,
        'p1_team': [], 'p2_team': [],
        'turns': [],
        'winner': None,
        'total_turns': 0,
        'p1_faints': 0, 'p2_faints': 0,
        # Deep analysis fields
        'p1_active': None, 'p2_active': None,  # current active mon per side
        'hp_tracker': {},  # "p1: Mon" -> current HP pct (0-100)
    }

    current_turn = {'number': 0, 'p1_actions': [], 'p2_actions': [], 'events': []}
    last_move_player = None  # track which player used the last |move| within this turn

    def _parse_hp(hp_str):
        """Parse HP from strings like '78/100' or '0 fnt'. Returns pct 0-100 or None."""
        hp_str = hp_str.strip()
        if 'fnt' in hp_str:
            return 0.0
        m = re.match(r'(\d+)/(\d+)', hp_str)
        if m:
            cur, mx = int(m.group(1)), int(m.group(2))
            return (cur / mx * 100.0) if mx > 0 else 0.0
        return None

    for line in lines:
        if line.startswith('|player|p1|'):
            battle['p1'] = line.split('|')[3]
        elif line.startswith('|player|p2|'):
            battle['p2'] = line.split('|')[3]
        elif line.startswith('|poke|p1|'):
            species = line.split('|')[3].split(',')[0]
            battle['p1_team'].append(species)
        elif line.startswith('|poke|p2|'):
            species = line.split('|')[3].split(',')[0]
            battle['p2_team'].append(species)
        elif line.startswith('|turn|'):
            if current_turn['number'] > 0:
                battle['turns'].append(current_turn)
            turn_num = int(line.split('|')[2])
            current_turn = {'number': turn_num, 'p1_actions': [], 'p2_actions': [], 'events': []}
            last_move_player = None  # reset per-turn to prevent cross-turn misattribution
            battle['total_turns'] = turn_num
        elif line.startswith('|move|'):
            parts = line.split('|')
            player_mon = parts[2]
            move_name = parts[3]
            target = parts[4] if len(parts) > 4 else ''
            player = 'p1' if player_mon.startswith('p1') else 'p2'
            last_move_player = player
            mon = player_mon.split(': ')[1] if ': ' in player_mon else player_mon
            action = {'type': 'move', 'mon': mon, 'move': move_name, 'target': target}
            current_turn[f'{player}_actions'].append(action)
        elif line.startswith('|switch|'):
            parts = line.split('|')
            player_mon = parts[2]
            species = parts[3].split(',')[0]
            player = 'p1' if player_mon.startswith('p1') else 'p2'
            # Parse HP from switch line: "|switch|p1a: Species|Species, L100, M|HP/MAXHP"
            hp_pct = None
            if len(parts) > 4:
                hp_pct = _parse_hp(parts[4])
            action = {'type': 'switch', 'mon': species, 'hp': hp_pct}
            current_turn[f'{player}_actions'].append(action)
            # Track active mon and HP
            battle[f'{player}_active'] = species
            if hp_pct is not None:
                battle['hp_tracker'][f'{player}: {species}'] = hp_pct
        elif line.startswith('|drag|'):
            # |drag| is a forced switch (e.g. Whirlwind, Dragon Tail) — not a player decision.
            # But still update active tracking.
            parts = line.split('|')
            player_mon = parts[2]
            species = parts[3].split(',')[0]
            player = 'p1' if player_mon.startswith('p1') else 'p2'
            battle[f'{player}_active'] = species
            if len(parts) > 4:
                hp_pct = _parse_hp(parts[4])
                if hp_pct is not None:
                    battle['hp_tracker'][f'{player}: {species}'] = hp_pct
            current_turn['events'].append(('drag', player, species))
        elif line.startswith('|-damage|'):
            parts = line.split('|')
            target_ref = parts[2]
            hp_str = parts[3] if len(parts) > 3 else ''
            hp_pct = _parse_hp(hp_str)
            current_turn['events'].append(('damage', target_ref, hp_str))
            # Update HP tracker
            if hp_pct is not None:
                player = 'p1' if target_ref.startswith('p1') else 'p2'
                mon_name = target_ref.split(': ')[1] if ': ' in target_ref else target_ref
                battle['hp_tracker'][f'{player}: {mon_name}'] = hp_pct
        elif line.startswith('|-heal|'):
            parts = line.split('|')
            target_ref = parts[2]
            hp_str = parts[3] if len(parts) > 3 else ''
            hp_pct = _parse_hp(hp_str)
            source = parts[4] if len(parts) > 4 else ''
            current_turn['events'].append(('heal', target_ref, hp_str, source))
            if hp_pct is not None:
                player = 'p1' if target_ref.startswith('p1') else 'p2'
                mon_name = target_ref.split(': ')[1] if ': ' in target_ref else target_ref
                battle['hp_tracker'][f'{player}: {mon_name}'] = hp_pct
        elif line.startswith('|faint|'):
            parts = line.split('|')
            current_turn['events'].append(('faint', parts[2]))
            if parts[2].startswith('p1'):
                battle['p1_faints'] += 1
            elif parts[2].startswith('p2'):
                battle['p2_faints'] += 1
            # Set HP to 0
            player = 'p1' if parts[2].startswith('p1') else 'p2'
            mon_name = parts[2].split(': ')[1] if ': ' in parts[2] else parts[2]
            battle['hp_tracker'][f'{player}: {mon_name}'] = 0.0
        elif line.startswith('|-supereffective|'):
            current_turn['events'].append(('supereffective', last_move_player))
        elif line.startswith('|-resisted|'):
            current_turn['events'].append(('resisted', last_move_player))
        elif line.startswith('|-immune|'):
            current_turn['events'].append(('immune', last_move_player))
        elif line.startswith('|-boost|') or line.startswith('|-unboost|'):
            current_turn['events'].append(('boost', line))
        elif line.startswith('|-status|'):
            parts = line.split('|')
            current_turn['events'].append(('status', parts[2], parts[3] if len(parts) > 3 else ''))
        elif line.startswith('|-weather|'):
            parts = line.split('|')
            current_turn['events'].append(('weather', parts[2]))
        elif line.startswith('|-fieldstart|'):
            parts = line.split('|')
            current_turn['events'].append(('fieldstart', parts[2]))
        elif line.startswith('|-ability|'):
            parts = line.split('|')
            current_turn['events'].append(('ability', parts[2], parts[3] if len(parts) > 3 else ''))
        elif line.startswith('|-crit|'):
            parts = line.split('|')
            current_turn['events'].append(('crit', parts[2]))
        elif line.startswith('|win|'):
            battle['winner'] = line.split('|')[2]

    if current_turn['number'] > 0:
        battle['turns'].append(current_turn)

    # Infer winner from faint counts if |win| line is missing
    if battle['winner'] is None:
        if battle['p1_faints'] >= 6 and battle['p2_faints'] < 6:
            battle['winner'] = battle['p2']
        elif battle['p2_faints'] >= 6 and battle['p1_faints'] < 6:
            battle['winner'] = battle['p1']

    return battle


def analyze_battles(battles, our_player_prefix='BCPolicyPlayer'):
    """Analyze a list of parsed battles for behavioral patterns."""
    stats = {
        'total_battles': len(battles),
        'wins': 0, 'losses': 0, 'unknown': 0,
        'total_turns': 0,
        'our_moves': Counter(),
        'our_switches': 0,
        'our_total_actions': 0,
        'opp_moves': Counter(),
        'our_pokemon_usage': Counter(),
        'our_lead': Counter(),
        'our_faints': 0,
        'opp_faints': 0,
        'our_se_moves': 0,
        'our_resisted_moves': 0,
        'our_immune_moves': 0,
        'turn_lengths_win': [],
        'turn_lengths_loss': [],
        'moves_per_mon': defaultdict(Counter),
        'switch_after_faint': 0,
        'voluntary_switches': 0,
        'same_move_streaks': [],
        'move_diversity_per_battle': [],
        # Playstyle categories
        'setup_moves': 0,
        'pivot_moves': 0,
        'hazard_moves': 0,
        'hazard_removal': 0,
        'status_moves': 0,
        'recovery_moves': 0,
        'protect_moves': 0,
        'attacking_moves': 0,
        # Turn-by-turn patterns
        'turn1_actions': Counter(),  # what we do on turn 1
        'early_switches': 0,  # switches in turns 1-3
        'early_turns': 0,
        'unique_mons_per_battle': [],
    }

    for b in battles:
        if not b:
            continue

        if our_player_prefix == 'p1':
            is_p1 = True
        elif our_player_prefix == 'p2':
            is_p1 = False
        else:
            is_p1 = our_player_prefix in (b['p1'] or '')
        our_key = 'p1' if is_p1 else 'p2'
        opp_key = 'p2' if is_p1 else 'p1'

        won = b['winner'] and our_player_prefix in b['winner']
        if b['winner'] is None:
            stats['unknown'] += 1
        elif won:
            stats['wins'] += 1
            stats['turn_lengths_win'].append(b['total_turns'])
        else:
            stats['losses'] += 1
            stats['turn_lengths_loss'].append(b['total_turns'])

        stats['total_turns'] += b['total_turns']

        # Lead pokemon
        if b['turns']:
            first_turn = b['turns'][0]
            for action in first_turn[f'{our_key}_actions']:
                if action['type'] == 'switch':
                    stats['our_lead'][action['mon']] += 1
                    break

        # Analyze each turn
        battle_moves = set()
        battle_mons = set()
        prev_move = None
        streak = 1

        for turn in b['turns']:
            our_actions = turn[f'{our_key}_actions']
            # Reset faint tracking per turn — a faint only forces the
            # immediately following switch, not switches in later turns
            faint_happened = False

            # Check events first to set faint_happened before processing actions
            for event in turn['events']:
                if event[0] == 'faint':
                    mon_ref = event[1]
                    if mon_ref.startswith(our_key[0:2]):
                        faint_happened = True

            for action in our_actions:
                stats['our_total_actions'] += 1
                if action['type'] == 'move':
                    move = action['move']
                    stats['our_moves'][move] += 1
                    stats['moves_per_mon'][action['mon']][move] += 1
                    stats['our_pokemon_usage'][action['mon']] += 1
                    battle_moves.add(move)
                    battle_mons.add(action['mon'])

                    # Categorize move
                    if move in SETUP_MOVES:
                        stats['setup_moves'] += 1
                    elif move in PIVOT_MOVES:
                        stats['pivot_moves'] += 1
                    elif move in HAZARD_MOVES:
                        stats['hazard_moves'] += 1
                    elif move in HAZARD_REMOVAL:
                        stats['hazard_removal'] += 1
                    elif move in STATUS_MOVES:
                        stats['status_moves'] += 1
                    elif move in RECOVERY_MOVES:
                        stats['recovery_moves'] += 1
                    elif move in PROTECT_MOVES:
                        stats['protect_moves'] += 1
                    else:
                        stats['attacking_moves'] += 1

                    # Turn 1 action tracking
                    if turn['number'] == 1:
                        stats['turn1_actions'][f'move:{move}'] += 1

                    # Move spam tracking
                    if move == prev_move:
                        streak += 1
                    else:
                        if streak >= 3:
                            stats['same_move_streaks'].append((prev_move, streak))
                        streak = 1
                    prev_move = move

                elif action['type'] == 'switch':
                    stats['our_switches'] += 1
                    battle_mons.add(action['mon'])
                    if faint_happened:
                        stats['switch_after_faint'] += 1
                        faint_happened = False
                    else:
                        stats['voluntary_switches'] += 1
                    prev_move = None
                    streak = 1

                    # Early game switching
                    if turn['number'] <= 3:
                        stats['early_switches'] += 1

                    if turn['number'] == 1:
                        stats['turn1_actions'][f'switch:{action["mon"]}'] += 1

            if turn['number'] <= 3:
                stats['early_turns'] += 1

            # Check events (faint_happened already set by pre-scan above)
            for event in turn['events']:
                if event[0] == 'faint':
                    mon_ref = event[1]
                    if mon_ref.startswith(our_key[0:2]):
                        stats['our_faints'] += 1
                    else:
                        stats['opp_faints'] += 1
                elif event[0] == 'supereffective':
                    attacker = event[1] if len(event) > 1 else None
                    if attacker == our_key:
                        stats['our_se_moves'] += 1
                elif event[0] == 'resisted':
                    attacker = event[1] if len(event) > 1 else None
                    if attacker == our_key:
                        stats['our_resisted_moves'] += 1
                elif event[0] == 'immune':
                    attacker = event[1] if len(event) > 1 else None
                    if attacker == our_key:
                        stats['our_immune_moves'] += 1

        if streak >= 3:
            stats['same_move_streaks'].append((prev_move, streak))

        stats['move_diversity_per_battle'].append(len(battle_moves))
        stats['unique_mons_per_battle'].append(len(battle_mons))

    return stats


def compute_playstyle_profile(stats):
    """Compute a playstyle profile from battle stats."""
    total_moves = stats['our_total_actions'] - stats['our_switches']
    total_actions = stats['our_total_actions']
    total = stats['total_battles']
    if total_moves == 0 or total == 0:
        return {}

    profile = {}

    # Action ratios
    profile['switch_rate'] = stats['our_switches'] / total_actions
    profile['voluntary_switch_rate'] = stats['voluntary_switches'] / total_actions
    profile['forced_switch_rate'] = stats['switch_after_faint'] / total_actions

    # Move category ratios (% of all moves, not actions)
    profile['attack_pct'] = stats['attacking_moves'] / total_moves
    profile['setup_pct'] = stats['setup_moves'] / total_moves
    profile['pivot_pct'] = stats['pivot_moves'] / total_moves
    profile['hazard_pct'] = stats['hazard_moves'] / total_moves
    profile['status_pct'] = stats['status_moves'] / total_moves
    profile['recovery_pct'] = stats['recovery_moves'] / total_moves
    profile['protect_pct'] = stats['protect_moves'] / total_moves

    # Effectiveness (of flagged hits only — excludes neutral)
    se_total = stats['our_se_moves'] + stats['our_resisted_moves'] + stats['our_immune_moves']
    profile['se_ratio'] = stats['our_se_moves'] / max(1, se_total)
    profile['resisted_ratio'] = stats['our_resisted_moves'] / max(1, se_total)
    profile['immune_ratio'] = stats['our_immune_moves'] / max(1, se_total)

    # Effectiveness vs all moves (SE/resisted/immune/neutral+non-damaging as remainder)
    # Using total_moves as denominator — includes some non-damaging moves but
    # the ratio is consistent across evals so trends are valid
    profile['se_of_all'] = stats['our_se_moves'] / max(1, total_moves)
    profile['resisted_of_all'] = stats['our_resisted_moves'] / max(1, total_moves)
    profile['immune_of_all'] = stats['our_immune_moves'] / max(1, total_moves)
    profile['unflagged_of_all'] = max(0, 1.0 - profile['se_of_all'] - profile['resisted_of_all'] - profile['immune_of_all'])

    # KO exchange
    profile['ko_ratio'] = stats['opp_faints'] / max(1, stats['our_faints'])
    profile['our_faints_per_game'] = stats['our_faints'] / total
    profile['opp_faints_per_game'] = stats['opp_faints'] / total

    # Tempo
    profile['avg_turns'] = stats['total_turns'] / total
    if stats['move_diversity_per_battle']:
        profile['avg_move_diversity'] = sum(stats['move_diversity_per_battle']) / len(stats['move_diversity_per_battle'])
    if stats['unique_mons_per_battle']:
        profile['avg_mons_used'] = sum(stats['unique_mons_per_battle']) / len(stats['unique_mons_per_battle'])

    # Spam score: what fraction of move streaks are 3+
    total_spam_instances = len(stats['same_move_streaks'])
    profile['spam_streaks_per_game'] = total_spam_instances / total

    # Early game aggression
    if stats['early_turns'] > 0:
        profile['early_switch_rate'] = stats['early_switches'] / stats['early_turns']

    # Win rate
    decided = stats['wins'] + stats['losses']
    profile['win_rate'] = stats['wins'] / max(1, decided)

    return profile


def _parse_hp_str(hp_str):
    """Parse HP pct from event string like '78/100' or '0 fnt'. Returns float or None."""
    hp_str = hp_str.strip()
    if 'fnt' in hp_str:
        return 0.0
    m = re.match(r'(\d+)/(\d+)', hp_str)
    if m:
        cur, mx = int(m.group(1)), int(m.group(2))
        return (cur / mx * 100.0) if mx > 0 else 0.0
    return None


def deep_analyze_battles(battles, our_player_prefix='BCPolicyPlayer'):
    """Run deep qualitative analysis on parsed battles.
    Returns a dict of advanced metrics.
    """
    deep = {
        # 1. Switch quality
        'good_switches': 0, 'bad_switches': 0, 'neutral_switches': 0,
        'switch_quality_unknown': 0,
        # 2. Momentum
        'momentum_swings': [],  # list of (player, length) for runs >= 2
        'our_momentum_runs': 0, 'opp_momentum_runs': 0,
        'our_momentum_total_len': 0, 'opp_momentum_total_len': 0,
        # 3. Move waste
        'wasted_moves': 0, 'total_attacking_moves': 0,
        # mon -> set of move effectiveness seen {move: 'se'|'resisted'|'immune'|'neutral'}
        # 4. HP management
        'switchout_hps': [],  # HP% when voluntarily switching out
        'faint_hps': [],  # HP% on the turn before fainting (last known HP)
        # 5. Turn efficiency
        'kos_in_wins': 0, 'turns_in_wins': 0,
        'kos_in_losses': 0, 'turns_in_losses': 0,
        # 6. Lead performance
        'lead_stay_wins': 0, 'lead_stay_total': 0,
        'lead_switch_wins': 0, 'lead_switch_total': 0,
        'lead_mon_wins': Counter(), 'lead_mon_total': Counter(),
        # 7. Endgame
        'endgame_wins': 0, 'endgame_total': 0,  # both sides <= 2 mons
        # 8. Move prediction (switch prediction)
        'predicted_switches': 0, 'opp_switches_total': 0,
        # 9. Recovery efficiency
        'recovery_useful': 0, 'recovery_wasted': 0,
        # 10. Consistency
        'first_half_wins': 0, 'first_half_total': 0,
        'second_half_wins': 0, 'second_half_total': 0,
        'even_wins': 0, 'even_total': 0,
        'odd_wins': 0, 'odd_total': 0,
        # Crits
        'our_crits': 0, 'opp_crits': 0,
        # Status inflicted
        'our_statuses': Counter(), 'opp_statuses': Counter(),
    }

    half = len(battles) // 2

    for b_idx, b in enumerate(battles):
        if not b:
            continue

        if our_player_prefix == 'p1':
            is_p1 = True
        elif our_player_prefix == 'p2':
            is_p1 = False
        else:
            is_p1 = our_player_prefix in (b['p1'] or '')
        our_key = 'p1' if is_p1 else 'p2'
        opp_key = 'p2' if is_p1 else 'p1'
        our_team_types = {mon: _get_species_types(mon) for mon in b[f'{our_key}_team']}
        opp_team_types = {mon: _get_species_types(mon) for mon in b[f'{opp_key}_team']}

        won = b['winner'] and our_player_prefix in b['winner']
        is_decided = b['winner'] is not None

        # 10. Consistency subsets
        if is_decided:
            if b_idx < half:
                deep['first_half_total'] += 1
                if won: deep['first_half_wins'] += 1
            else:
                deep['second_half_total'] += 1
                if won: deep['second_half_wins'] += 1
            if b_idx % 2 == 0:
                deep['even_total'] += 1
                if won: deep['even_wins'] += 1
            else:
                deep['odd_total'] += 1
                if won: deep['odd_wins'] += 1

        # Track active mons per side, HP, and per-mon move effectiveness history
        our_active = None
        opp_active = None
        mon_se_moves = defaultdict(set)  # mon -> set of moves that were SE
        mon_resisted_moves = defaultdict(set)  # mon -> set of moves that were resisted/immune

        # Momentum tracking: per-turn who "won" (got a KO or forced a switch)
        turn_winners = []  # list of 'our' | 'opp' | None per turn

        # For recovery tracking: mon -> [(turn_used, mon_name)]
        recovery_events = []

        # For faint tracking: mon -> last known HP before faint
        last_known_hp = {}  # "p1: Mon" -> hp_pct

        # Track our faints and opp faints per turn for endgame detection
        our_faint_count = 0
        opp_faint_count = 0

        # 6. Lead detection
        lead_mon = None
        switched_turn1 = False

        for t_idx, turn in enumerate(b['turns']):
            our_actions = turn[f'{our_key}_actions']
            opp_actions = turn[f'{opp_key}_actions']
            turn_num = turn['number']

            # Determine who "won" this turn for momentum
            our_ko_this_turn = 0
            opp_ko_this_turn = 0
            opp_switched_this_turn = False
            our_switched_voluntarily = False
            faint_happened_our = False

            # Pre-scan events for this turn
            for event in turn['events']:
                if event[0] == 'faint':
                    mon_ref = event[1]
                    if mon_ref.startswith(our_key[:2]):
                        opp_ko_this_turn += 1
                        our_faint_count += 1
                    else:
                        our_ko_this_turn += 1
                        opp_faint_count += 1
                elif event[0] == 'crit':
                    mon_ref = event[1]
                    if mon_ref.startswith(our_key[:2]):
                        deep['opp_crits'] += 1
                    else:
                        deep['our_crits'] += 1
                elif event[0] == 'status':
                    mon_ref = event[1]
                    status = event[2] if len(event) > 2 else ''
                    if mon_ref.startswith(our_key[:2]):
                        deep['opp_statuses'][status] += 1
                    else:
                        deep['our_statuses'][status] += 1

            # Check events for last_move_player effectiveness (for move waste)
            last_move_mon = None
            last_move_name = None
            for action in our_actions:
                if action['type'] == 'move':
                    last_move_mon = action['mon']
                    last_move_name = action['move']
                    # Count attacking moves (non-status for waste tracking)
                    if action['move'] not in SETUP_MOVES | HAZARD_MOVES | HAZARD_REMOVAL | STATUS_MOVES | RECOVERY_MOVES | PROTECT_MOVES | PIVOT_MOVES:
                        deep['total_attacking_moves'] += 1

            # Track effectiveness per mon
            for event in turn['events']:
                if event[0] == 'supereffective' and len(event) > 1 and event[1] == our_key:
                    if last_move_mon and last_move_name:
                        mon_se_moves[last_move_mon].add(last_move_name)
                elif event[0] in ('resisted', 'immune') and len(event) > 1 and event[1] == our_key:
                    if last_move_mon and last_move_name:
                        mon_resisted_moves[last_move_mon].add(last_move_name)
                        # 3. Move waste: did this mon use a SE move before?
                        if mon_se_moves[last_move_mon]:
                            deep['wasted_moves'] += 1

            # Process actions for switch quality, lead, etc.
            for action in our_actions:
                if action['type'] == 'switch':
                    species = action['mon']

                    # Lead tracking (turn 1)
                    if turn_num == 1 and lead_mon is None:
                        lead_mon = species
                    elif turn_num == 1 and lead_mon is not None:
                        # Second switch on turn 1 means we switched our lead
                        switched_turn1 = True
                        lead_mon = species

                    # Detect voluntary switch (not after faint)
                    faint_this_turn = any(e[0] == 'faint' and e[1].startswith(our_key[:2])
                                          for e in turn['events'])

                    if not faint_this_turn:
                        our_switched_voluntarily = True
                        # 1. Switch quality: is the switch-in type-advantaged vs opp active?
                        our_types = _get_species_types(species)
                        opp_types = _get_species_types(opp_active) if opp_active else None
                        if our_types and opp_types:
                            # Check if we resist their STAB and/or are SE against them
                            our_eff = _type_effectiveness(our_types, opp_types)  # our attack vs them
                            their_eff = _type_effectiveness(opp_types, our_types)  # their attack vs us
                            if our_eff >= 2.0 or their_eff <= 0.5:
                                deep['good_switches'] += 1
                            elif our_eff <= 0.5 or their_eff >= 2.0:
                                deep['bad_switches'] += 1
                            else:
                                deep['neutral_switches'] += 1
                        else:
                            deep['switch_quality_unknown'] += 1

                        # 4. HP when switching out (the mon leaving)
                        if our_active:
                            hp_key = f'{our_key}: {our_active}'
                            hp = b['hp_tracker'].get(hp_key)
                            if hp is not None and hp > 0:
                                deep['switchout_hps'].append(hp)

                    our_active = species

                elif action['type'] == 'move':
                    if turn_num == 1 and lead_mon is None:
                        # First turn, used a move -> stayed in with lead
                        lead_mon = action['mon']

                    # 9. Recovery tracking
                    if action['move'] in RECOVERY_MOVES:
                        recovery_events.append((turn_num, action['mon']))

            # Track opp actions for active tracking and switch prediction
            for action in opp_actions:
                if action['type'] == 'switch':
                    opp_switched_this_turn = True
                    deep['opp_switches_total'] += 1
                    prev_opp = opp_active
                    opp_active = action['mon']
                    # 8. Move prediction: did we use a move SE vs their PREVIOUS mon?
                    if prev_opp and last_move_name:
                        # Check if our move would have been SE vs prev_opp
                        # (suggesting coverage/prediction rather than just hitting what's in front)
                        prev_types = _get_species_types(prev_opp)
                        our_mon_types = _get_species_types(last_move_mon) if last_move_mon else None
                        if prev_types and our_mon_types:
                            eff = _type_effectiveness(our_mon_types, prev_types)
                            if eff >= 2.0:
                                deep['predicted_switches'] += 1
                elif action['type'] == 'move':
                    if opp_active is None:
                        opp_active = action['mon']

            # Update HP tracker from events
            for event in turn['events']:
                if event[0] == 'damage' and len(event) > 2:
                    target_ref = event[1]
                    hp = _parse_hp_str(event[2])
                    if hp is not None:
                        player = 'p1' if target_ref.startswith('p1') else 'p2'
                        mon_name = target_ref.split(': ')[1] if ': ' in target_ref else target_ref
                        last_known_hp[f'{player}: {mon_name}'] = hp
                elif event[0] == 'heal' and len(event) > 2:
                    target_ref = event[1]
                    hp = _parse_hp_str(event[2])
                    if hp is not None:
                        player = 'p1' if target_ref.startswith('p1') else 'p2'
                        mon_name = target_ref.split(': ')[1] if ': ' in target_ref else target_ref
                        last_known_hp[f'{player}: {mon_name}'] = hp

            # Momentum: who won this turn?
            if our_ko_this_turn > opp_ko_this_turn:
                turn_winners.append('our')
            elif opp_ko_this_turn > our_ko_this_turn:
                turn_winners.append('opp')
            elif opp_switched_this_turn and not our_switched_voluntarily:
                turn_winners.append('our')  # forced opp to switch
            elif our_switched_voluntarily and not opp_switched_this_turn:
                turn_winners.append('opp')  # we had to switch
            else:
                turn_winners.append(None)

            # 7. Endgame detection: both sides <= 2 remaining
            our_remaining = len(b[f'{our_key}_team']) - our_faint_count
            opp_remaining = len(b[f'{opp_key}_team']) - opp_faint_count
            if our_remaining <= 2 and opp_remaining <= 2 and our_remaining > 0 and opp_remaining > 0:
                # Mark this battle as having reached endgame (only count once)
                if not b.get('_endgame_counted'):
                    b['_endgame_counted'] = True
                    deep['endgame_total'] += 1
                    if won:
                        deep['endgame_wins'] += 1

        # 2. Compute momentum runs
        current_run_player = None
        current_run_len = 0
        for tw in turn_winners:
            if tw == current_run_player and tw is not None:
                current_run_len += 1
            else:
                if current_run_len >= 2 and current_run_player is not None:
                    deep['momentum_swings'].append((current_run_player, current_run_len))
                    if current_run_player == 'our':
                        deep['our_momentum_runs'] += 1
                        deep['our_momentum_total_len'] += current_run_len
                    else:
                        deep['opp_momentum_runs'] += 1
                        deep['opp_momentum_total_len'] += current_run_len
                current_run_player = tw
                current_run_len = 1
        if current_run_len >= 2 and current_run_player is not None:
            deep['momentum_swings'].append((current_run_player, current_run_len))
            if current_run_player == 'our':
                deep['our_momentum_runs'] += 1
                deep['our_momentum_total_len'] += current_run_len
            else:
                deep['opp_momentum_runs'] += 1
                deep['opp_momentum_total_len'] += current_run_len

        # 5. Turn efficiency
        if is_decided:
            if won:
                deep['kos_in_wins'] += opp_faint_count
                deep['turns_in_wins'] += b['total_turns']
            else:
                deep['kos_in_losses'] += our_faint_count  # we got KOd
                deep['turns_in_losses'] += b['total_turns']

        # 6. Lead performance
        if lead_mon and is_decided:
            deep['lead_mon_total'][lead_mon] += 1
            if won:
                deep['lead_mon_wins'][lead_mon] += 1
            if switched_turn1:
                deep['lead_switch_total'] += 1
                if won: deep['lead_switch_wins'] += 1
            else:
                deep['lead_stay_total'] += 1
                if won: deep['lead_stay_wins'] += 1

        # 9. Recovery: check if mon survived 3+ turns after recovery
        for rec_turn, rec_mon in recovery_events:
            survived_turns = 0
            for turn in b['turns']:
                if turn['number'] <= rec_turn:
                    continue
                # Check if this mon is still active / alive
                mon_alive = True
                for event in turn['events']:
                    if event[0] == 'faint':
                        fainted_ref = event[1]
                        fainted_mon = fainted_ref.split(': ')[1] if ': ' in fainted_ref else fainted_ref
                        if fainted_mon == rec_mon and fainted_ref.startswith(our_key[:2]):
                            mon_alive = False
                            break
                if not mon_alive:
                    break
                # Check if mon is still in play (used a move or is active)
                for action in turn[f'{our_key}_actions']:
                    if action.get('mon') == rec_mon:
                        survived_turns += 1
                        break
            if survived_turns >= 3:
                deep['recovery_useful'] += 1
            else:
                deep['recovery_wasted'] += 1

    return deep



def print_deep_report(deep, total_battles):
    """Print the deep analysis report section."""
    if total_battles == 0:
        return

    print(f"\n  {'─'*61}")
    print(f"  DEEP ANALYSIS")
    print(f"  {'─'*61}")

    # 1. Switch Quality
    total_vol = deep['good_switches'] + deep['bad_switches'] + deep['neutral_switches']
    print(f"\n  SWITCH QUALITY ({total_vol} voluntary switches analyzed, "
          f"{deep['switch_quality_unknown']} unknown species)")
    if total_vol > 0:
        print(f"    Good (favorable matchup):   {deep['good_switches']:4d} ({deep['good_switches']/total_vol:.0%})")
        print(f"    Neutral:                    {deep['neutral_switches']:4d} ({deep['neutral_switches']/total_vol:.0%})")
        print(f"    Bad (unfavorable matchup):  {deep['bad_switches']:4d} ({deep['bad_switches']/total_vol:.0%})")
    else:
        print(f"    No voluntary switches with known types")

    # 2. Momentum
    our_runs = deep['our_momentum_runs']
    opp_runs = deep['opp_momentum_runs']
    total_swings = our_runs + opp_runs
    print(f"\n  MOMENTUM ({total_swings} momentum swings, runs of 2+ favorable turns)")
    print(f"    Our momentum runs:    {our_runs:4d}"
          f" (avg {deep['our_momentum_total_len']/max(1,our_runs):.1f} turns)")
    print(f"    Opp momentum runs:    {opp_runs:4d}"
          f" (avg {deep['opp_momentum_total_len']/max(1,opp_runs):.1f} turns)")
    if total_swings > 0:
        print(f"    Momentum dominance:   {our_runs/total_swings:.0%} ours")

    # 3. Move Waste
    print(f"\n  MOVE WASTE")
    print(f"    Wasted moves (resisted/immune when SE option known): {deep['wasted_moves']}")
    if deep['total_attacking_moves'] > 0:
        print(f"    Waste rate: {deep['wasted_moves']/deep['total_attacking_moves']:.1%}"
              f" of {deep['total_attacking_moves']} attacking moves")

    # 4. HP Management
    print(f"\n  HP MANAGEMENT")
    if deep['switchout_hps']:
        avg_hp = sum(deep['switchout_hps']) / len(deep['switchout_hps'])
        print(f"    Avg HP when switching out:  {avg_hp:.1f}%"
              f" ({len(deep['switchout_hps'])} switches)")
        # Distribution buckets
        low = sum(1 for h in deep['switchout_hps'] if h < 33)
        mid = sum(1 for h in deep['switchout_hps'] if 33 <= h < 66)
        high = sum(1 for h in deep['switchout_hps'] if h >= 66)
        n = len(deep['switchout_hps'])
        print(f"    HP distribution:  <33%: {low/n:.0%} | 33-66%: {mid/n:.0%} | >66%: {high/n:.0%}")
    else:
        print(f"    No HP data for switch-outs")

    # 5. Turn Efficiency
    print(f"\n  TURN EFFICIENCY")
    if deep['turns_in_wins'] > 0:
        print(f"    KOs/turn in wins:   {deep['kos_in_wins']/deep['turns_in_wins']:.3f}"
              f" ({deep['kos_in_wins']} KOs in {deep['turns_in_wins']} turns)")
    if deep['turns_in_losses'] > 0:
        print(f"    KOs/turn in losses: {deep['kos_in_losses']/deep['turns_in_losses']:.3f}"
              f" ({deep['kos_in_losses']} KOs in {deep['turns_in_losses']} turns)")

    # 6. Lead Performance
    print(f"\n  LEAD PERFORMANCE")
    if deep['lead_stay_total'] > 0:
        print(f"    Lead stays in:  {deep['lead_stay_wins']}/{deep['lead_stay_total']}"
              f" ({deep['lead_stay_wins']/deep['lead_stay_total']:.0%} WR)")
    if deep['lead_switch_total'] > 0:
        print(f"    Lead switches:  {deep['lead_switch_wins']}/{deep['lead_switch_total']}"
              f" ({deep['lead_switch_wins']/deep['lead_switch_total']:.0%} WR)")
    # Top leads
    if deep['lead_mon_total']:
        print(f"    Top leads:")
        for mon, total in deep['lead_mon_total'].most_common(5):
            wins = deep['lead_mon_wins'].get(mon, 0)
            wr = wins / total if total > 0 else 0
            print(f"      {mon:20s} {wins}W/{total-wins}L ({wr:.0%})")

    # 7. Endgame
    print(f"\n  ENDGAME PERFORMANCE (both sides <= 2 mons)")
    if deep['endgame_total'] > 0:
        print(f"    Endgame WR: {deep['endgame_wins']}/{deep['endgame_total']}"
              f" ({deep['endgame_wins']/deep['endgame_total']:.0%})")
    else:
        print(f"    No endgame situations detected")

    # 8. Move Prediction
    print(f"\n  SWITCH PREDICTION")
    if deep['opp_switches_total'] > 0:
        print(f"    Opp switches: {deep['opp_switches_total']}")
        print(f"    We used coverage move (SE vs prev opp): {deep['predicted_switches']}"
              f" ({deep['predicted_switches']/deep['opp_switches_total']:.0%})")
    else:
        print(f"    No opponent switches detected")

    # 9. Recovery Efficiency
    total_rec = deep['recovery_useful'] + deep['recovery_wasted']
    print(f"\n  RECOVERY EFFICIENCY ({total_rec} recovery moves)")
    if total_rec > 0:
        print(f"    Useful (survived 3+ turns): {deep['recovery_useful']}"
              f" ({deep['recovery_useful']/total_rec:.0%})")
        print(f"    Wasted (died within 2 turns): {deep['recovery_wasted']}"
              f" ({deep['recovery_wasted']/total_rec:.0%})")

    # 10. Consistency
    print(f"\n  CONSISTENCY")
    subsets = []
    if deep['first_half_total'] > 0:
        wr1 = deep['first_half_wins'] / deep['first_half_total']
        subsets.append(wr1)
        print(f"    First half WR:  {deep['first_half_wins']}/{deep['first_half_total']}"
              f" ({wr1:.0%})")
    if deep['second_half_total'] > 0:
        wr2 = deep['second_half_wins'] / deep['second_half_total']
        subsets.append(wr2)
        print(f"    Second half WR: {deep['second_half_wins']}/{deep['second_half_total']}"
              f" ({wr2:.0%})")
    if deep['even_total'] > 0:
        wr_e = deep['even_wins'] / deep['even_total']
        subsets.append(wr_e)
        print(f"    Even games WR:  {deep['even_wins']}/{deep['even_total']}"
              f" ({wr_e:.0%})")
    if deep['odd_total'] > 0:
        wr_o = deep['odd_wins'] / deep['odd_total']
        subsets.append(wr_o)
        print(f"    Odd games WR:   {deep['odd_wins']}/{deep['odd_total']}"
              f" ({wr_o:.0%})")
    if len(subsets) >= 2:
        mean_wr = sum(subsets) / len(subsets)
        std_wr = math.sqrt(sum((x - mean_wr) ** 2 for x in subsets) / len(subsets))
        print(f"    Consistency score: {std_wr:.3f} stdev (lower = more consistent)")

    # Crits
    if deep['our_crits'] or deep['opp_crits']:
        print(f"\n  CRITICAL HITS")
        print(f"    Our crits:  {deep['our_crits']}")
        print(f"    Opp crits:  {deep['opp_crits']}")

    # Status
    if deep['our_statuses'] or deep['opp_statuses']:
        print(f"\n  STATUS CONDITIONS INFLICTED")
        if deep['our_statuses']:
            parts = ', '.join(f"{s}: {c}" for s, c in deep['our_statuses'].most_common())
            print(f"    By us:   {parts}")
        if deep['opp_statuses']:
            parts = ', '.join(f"{s}: {c}" for s, c in deep['opp_statuses'].most_common())
            print(f"    By opp:  {parts}")


def print_playstyle_report(stats, profile, label):
    """Print a focused playstyle report."""
    total = stats['total_battles']
    if total == 0:
        print(f"  No battles found")
        return

    decided = stats['wins'] + stats['losses']
    wr = stats['wins'] / decided if decided else 0
    total_moves = stats['our_total_actions'] - stats['our_switches']

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  {stats['wins']}W/{stats['losses']}L/{stats['unknown']}? ({wr:.0%}) | "
          f"{total} games | Avg {profile.get('avg_turns', 0):.1f} turns")
    print(f"{'='*65}")

    # ── Playstyle breakdown ──
    print(f"\n  PLAYSTYLE BREAKDOWN ({total_moves} moves, {stats['our_switches']} switches)")
    bar_width = 40
    categories = [
        ('Attacking', profile.get('attack_pct', 0)),
        ('Setup', profile.get('setup_pct', 0)),
        ('Pivot', profile.get('pivot_pct', 0)),
        ('Hazards', profile.get('hazard_pct', 0)),
        ('Status', profile.get('status_pct', 0)),
        ('Recovery', profile.get('recovery_pct', 0)),
        ('Protect', profile.get('protect_pct', 0)),
    ]
    for name, pct in categories:
        bar = '#' * int(pct * bar_width)
        if pct >= 0.005:
            print(f"    {name:12s} {bar:40s} {pct:5.1%}")

    # ── Switching behavior ──
    print(f"\n  SWITCHING BEHAVIOR")
    print(f"    Total switch rate:     {profile.get('switch_rate', 0):5.1%} of all actions")
    print(f"    Voluntary switches:    {profile.get('voluntary_switch_rate', 0):5.1%} (proactive)")
    print(f"    Forced (after faint):  {profile.get('forced_switch_rate', 0):5.1%}")
    print(f"    Early game switches:   {profile.get('early_switch_rate', 0):5.1%} (turns 1-3)")

    # ── Combat effectiveness ──
    print(f"\n  COMBAT EFFECTIVENESS")
    print(f"    KO ratio:       {profile.get('ko_ratio', 0):.2f} (>1 = good trade)")
    print(f"    Our faints/game:  {profile.get('our_faints_per_game', 0):.1f}")
    print(f"    Opp faints/game:  {profile.get('opp_faints_per_game', 0):.1f}")
    print(f"    SE hit ratio:     {profile.get('se_ratio', 0):.0%} super / "
          f"{profile.get('resisted_ratio', 0):.0%} resisted / "
          f"{profile.get('immune_ratio', 0):.0%} immune"
          f"  (of flagged hits only)")
    print(f"    vs all moves:     {profile.get('se_of_all', 0):.0%} SE / "
          f"{profile.get('unflagged_of_all', 0):.0%} neutral+other / "
          f"{profile.get('resisted_of_all', 0):.0%} resisted / "
          f"{profile.get('immune_of_all', 0):.0%} immune"
          f"  ({total_moves} moves)")

    # ── Move diversity ──
    print(f"\n  DIVERSITY & SPAM")
    print(f"    Unique moves/game: {profile.get('avg_move_diversity', 0):.1f}")
    print(f"    Mons used/game:    {profile.get('avg_mons_used', 0):.1f}")
    print(f"    Spam streaks/game: {profile.get('spam_streaks_per_game', 0):.1f} (3+ same move)")

    # ── Top moves ──
    print(f"\n  TOP 10 MOVES")
    for move, count in stats['our_moves'].most_common(10):
        pct = count / total_moves * 100
        cat = ''
        if move in SETUP_MOVES: cat = '[setup]'
        elif move in PIVOT_MOVES: cat = '[pivot]'
        elif move in HAZARD_MOVES: cat = '[hazard]'
        elif move in STATUS_MOVES: cat = '[status]'
        elif move in RECOVERY_MOVES: cat = '[recov]'
        print(f"    {move:28s} {count:4d} ({pct:4.1f}%) {cat}")

    # ── Per-pokemon (top 5) ──
    print(f"\n  TOP 5 POKEMON (by turns in play)")
    sorted_mons = sorted(stats['moves_per_mon'].items(),
                         key=lambda x: sum(x[1].values()), reverse=True)[:5]
    for mon, moves in sorted_mons:
        total_mon = sum(moves.values())
        top_moves = moves.most_common(4)
        move_str = ', '.join(f"{m} {c/total_mon:.0%}" for m, c in top_moves)
        print(f"    {mon:20s} ({total_mon:3d} actions): {move_str}")

    # ── Spam detail ──
    if stats['same_move_streaks']:
        print(f"\n  MOVE SPAM DETAIL (3+ consecutive)")
        spam_counter = Counter()
        for move, streak in stats['same_move_streaks']:
            spam_counter[move] += 1
        for move, count in spam_counter.most_common(5):
            print(f"    {move:28s} {count:3d} streaks")


def print_comparison(profiles, labels):
    """Print a side-by-side comparison table."""
    if len(profiles) < 2:
        return

    print(f"\n\n{'#'*75}")
    print(f"  PLAYSTYLE COMPARISON")
    print(f"{'#'*75}")

    # Header
    col_w = 12
    header = f"  {'Metric':<28s}"
    for label in labels:
        # Truncate label
        short = label[:col_w]
        header += f" {short:>{col_w}s}"
    print(f"\n{header}")
    print(f"  {'-'*28}" + f" {'-'*col_w}" * len(labels))

    rows = [
        ('Win Rate', 'win_rate', '%'),
        ('Avg Turns/Game', 'avg_turns', 'f'),
        ('', None, None),  # separator
        ('Attack %', 'attack_pct', '%'),
        ('Setup %', 'setup_pct', '%'),
        ('Pivot %', 'pivot_pct', '%'),
        ('Hazard %', 'hazard_pct', '%'),
        ('Status %', 'status_pct', '%'),
        ('Recovery %', 'recovery_pct', '%'),
        ('', None, None),
        ('Total Switch Rate', 'switch_rate', '%'),
        ('Voluntary Switch', 'voluntary_switch_rate', '%'),
        ('Early Switch (t1-3)', 'early_switch_rate', '%'),
        ('', None, None),
        ('KO Ratio', 'ko_ratio', '2f'),
        ('SE Hit Ratio', 'se_ratio', '%'),
        ('Immune Hit Ratio', 'immune_ratio', '%'),
        ('', None, None),
        ('Move Diversity/Game', 'avg_move_diversity', '1f'),
        ('Mons Used/Game', 'avg_mons_used', '1f'),
        ('Spam Streaks/Game', 'spam_streaks_per_game', '1f'),
    ]

    for row_name, key, fmt in rows:
        if key is None:
            print()
            continue
        line = f"  {row_name:<28s}"
        for p in profiles:
            val = p.get(key, 0)
            if fmt == '%':
                line += f" {val:>{col_w}.1%}"
            elif fmt == '2f':
                line += f" {val:>{col_w}.2f}"
            elif fmt == '1f':
                line += f" {val:>{col_w}.1f}"
            elif fmt == 'f':
                line += f" {val:>{col_w}.1f}"
            else:
                line += f" {val:>{col_w}}"
        print(line)


def load_replays(base_dir):
    """Load and deduplicate replays from a directory."""
    base = Path(base_dir)
    if not base.exists():
        return []

    replay_files = list(base.rglob('*.html'))

    parsed = []
    for f in replay_files:
        battle = parse_replay(str(f))
        if battle:
            battle['_n_lines'] = len(battle['turns'])
            parsed.append(battle)

    # Deduplicate: use battle_id (unique per game) as primary key.
    # Falls back to (p1, p2, turns, faints) only if battle_id is missing.
    def battle_key(b):
        bid = b.get('battle_id')
        if bid:
            return bid
        p1 = re.sub(r'\s*\d+$', '', b.get('p1') or '')
        p2 = re.sub(r'\s*\d+$', '', b.get('p2') or '')
        return (p1, p2, b['total_turns'], b.get('p1_faints', 0), b.get('p2_faints', 0))

    parsed.sort(key=lambda b: (battle_key(b), -b['_n_lines']))
    deduped = []
    seen = set()
    for b in parsed:
        k = battle_key(b)
        if k not in seen:
            seen.add(k)
            deduped.append(b)

    return deduped


def filter_by_opponent(battles, our_prefix='BCPolicyPlayer'):
    """Group battles by opponent name."""
    grouped = defaultdict(list)
    for b in battles:
        p1, p2 = b.get('p1', ''), b.get('p2', '')
        if our_prefix in (p1 or ''):
            opp = re.sub(r'\s*\d+$', '', p2 or 'Unknown')
        elif our_prefix in (p2 or ''):
            opp = re.sub(r'\s*\d+$', '', p1 or 'Unknown')
        else:
            continue
        grouped[opp].append(b)
    return grouped


# ── Iter-trajectory mode ──
# Walks a PPO run dir's replays_iter*/ subdirs to produce a per-iter
# evolution table. Useful for spotting style shifts across training.

def _scan_iter_dirs(run_dir):
    """Find replays_iter*/ subdirs under run_dir, sorted by iter number."""
    run = Path(run_dir)
    if not run.exists():
        return []
    out = []
    for p in run.glob('replays_iter*'):
        m = re.search(r'iter(\d+)', p.name)
        if m and p.is_dir():
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


# Columns shown in trajectory and matchup tables. Tuple: (profile-key, header, format-spec).
# Format-spec: "%" → multiplied by 100 with trailing %, "f" → float as-is.
_TRAJ_COLS = [
    ('win_rate',              'WR',       '%'),
    ('switch_rate',           'Sw',       '%'),
    ('voluntary_switch_rate', 'vSw',      '%'),
    ('attack_pct',            'Atk',      '%'),
    ('setup_pct',             'Set',      '%'),
    ('pivot_pct',             'Piv',      '%'),
    ('hazard_pct',            'Haz',      '%'),
    ('status_pct',            'Sts',      '%'),
    ('se_ratio',              'SE',       '%'),
    ('immune_of_all',         'Imm',      '%'),
    ('ko_ratio',              'KO',       'f'),
    ('avg_turns',             'Turn',     'f'),
    ('spam_streaks_per_game', 'Spam',     'f'),
]


def _format_traj_value(key, val, fmt):
    if val is None:
        return '   -  '
    if fmt == '%':
        return f"{val * 100:5.1f}%"
    return f"{val:6.2f}"


def trajectory_mode(run_dir, our_prefix='BCPolicyPlayer', by_opponent=False):
    """Print a per-iteration playstyle trajectory across a PPO run's eval dirs.

    With by_opponent=True, prints a separate row per (iter, opponent) pair.
    """
    iter_dirs = _scan_iter_dirs(run_dir)
    if not iter_dirs:
        print(f"No replays_iter*/ subdirs found under {run_dir}")
        return

    print(f"\n{'='*65}")
    print(f"  TRAJECTORY: {run_dir}")
    print(f"  {len(iter_dirs)} eval iters: {iter_dirs[0][0]}..{iter_dirs[-1][0]}")
    print(f"{'='*65}\n")

    headers = ['iter', 'games'] + [c[1] for c in _TRAJ_COLS]
    if by_opponent:
        headers = ['iter', 'opp', 'games'] + [c[1] for c in _TRAJ_COLS]
    head_line = " ".join(f"{h:>6}" for h in headers)
    print(head_line)
    print('-' * len(head_line))

    for it, dpath in iter_dirs:
        battles = load_replays(str(dpath))
        if not battles:
            continue
        if by_opponent:
            grouped = filter_by_opponent(battles, our_prefix=our_prefix)
            for opp_name in sorted(grouped.keys()):
                stats = analyze_battles(grouped[opp_name], our_player_prefix=our_prefix)
                profile = compute_playstyle_profile(stats)
                if not profile:
                    continue
                vals = [_format_traj_value(k, profile.get(k), f) for k, _, f in _TRAJ_COLS]
                print(f"{it:>6} {opp_name[:6]:>6} {len(grouped[opp_name]):>6} {' '.join(vals)}")
        else:
            stats = analyze_battles(battles, our_player_prefix=our_prefix)
            profile = compute_playstyle_profile(stats)
            if not profile:
                continue
            vals = [_format_traj_value(k, profile.get(k), f) for k, _, f in _TRAJ_COLS]
            print(f"{it:>6} {len(battles):>6} {' '.join(vals)}")


# ── Team usage stats ──
# Tracks which mons we lead with, which mons get sent out most often,
# which faint first/last, and which we win most with.

def compute_team_usage(battles, our_player_prefix='BCPolicyPlayer'):
    """Aggregate team-level stats: lead mon, send-out frequency, faint order, win rate by lead."""
    lead = Counter()
    send_outs = Counter()        # any time we switch in
    first_faint = Counter()      # which of our mons faints first per battle
    last_faint = Counter()       # which of our mons is the LAST to faint (the one we lost on)
    faint_frequency = Counter()  # raw faint count per mon
    wins_by_lead = defaultdict(lambda: [0, 0])  # mon -> [wins, losses]
    total_wins = 0
    total_losses = 0

    for b in battles:
        if not b:
            continue
        if our_player_prefix == 'p1':
            our_key = 'p1'
        elif our_player_prefix == 'p2':
            our_key = 'p2'
        else:
            is_p1 = our_player_prefix in (b['p1'] or '')
            our_key = 'p1' if is_p1 else 'p2'
        our_short = our_key[:2]

        # Lead = first switch action of turn 1 by our side, or first non-switch mon mentioned.
        our_lead = None
        if b['turns']:
            for action in b['turns'][0][f'{our_key}_actions']:
                if action['type'] == 'switch':
                    our_lead = action['mon']
                    break
            if our_lead is None:
                # No explicit lead-switch — derive from |poke| order (team[0]).
                team = b.get(f'{our_key}_team', [])
                if team:
                    our_lead = team[0]
        if our_lead:
            lead[our_lead] += 1
            send_outs[our_lead] += 1

        # Walk events to capture faint order on our side; count send-outs from switches.
        our_faint_order = []
        for turn in b['turns']:
            for action in turn[f'{our_key}_actions']:
                if action['type'] == 'switch' and action['mon'] != our_lead:
                    send_outs[action['mon']] += 1
            for event in turn['events']:
                if event[0] == 'faint':
                    mon_ref = event[1] or ''
                    if mon_ref.startswith(our_short):
                        species = mon_ref.split(': ')[1] if ': ' in mon_ref else mon_ref
                        our_faint_order.append(species)
                        faint_frequency[species] += 1

        if our_faint_order:
            first_faint[our_faint_order[0]] += 1
            last_faint[our_faint_order[-1]] += 1

        # Win/loss attribution
        if b['winner'] is not None:
            won = our_player_prefix in (b['winner'] or '')
            if won:
                total_wins += 1
                if our_lead:
                    wins_by_lead[our_lead][0] += 1
            else:
                total_losses += 1
                if our_lead:
                    wins_by_lead[our_lead][1] += 1

    return {
        'lead': lead,
        'send_outs': send_outs,
        'first_faint': first_faint,
        'last_faint': last_faint,
        'faint_frequency': faint_frequency,
        'wins_by_lead': dict(wins_by_lead),
        'total_wins': total_wins,
        'total_losses': total_losses,
    }


def print_team_usage(usage, top_n=10):
    """Pretty-print team usage stats."""
    print(f"\n{'─'*65}")
    print(f"  TEAM USAGE")
    print(f"{'─'*65}")

    total_leads = sum(usage['lead'].values())
    print(f"\n  Lead pokemon (top {top_n}, of {total_leads} games with detected lead):")
    for mon, n in usage['lead'].most_common(top_n):
        wins, losses = usage['wins_by_lead'].get(mon, [0, 0])
        decided = wins + losses
        wr = wins / decided if decided else 0
        bar = '█' * int(n / max(1, total_leads) * 30)
        print(f"    {mon:20s}  {n:4d} ({n/total_leads*100:5.1f}%)  WR {wr*100:5.1f}%  {bar}")

    total_so = sum(usage['send_outs'].values())
    print(f"\n  Send-out frequency (top {top_n}, of {total_so} total send-outs):")
    for mon, n in usage['send_outs'].most_common(top_n):
        print(f"    {mon:20s}  {n:4d} ({n/max(1, total_so)*100:5.1f}%)")

    total_ff = sum(usage['first_faint'].values())
    if total_ff > 0:
        print(f"\n  First-to-faint (top {top_n}, of {total_ff} games where we lost a mon):")
        for mon, n in usage['first_faint'].most_common(top_n):
            print(f"    {mon:20s}  {n:4d} ({n/total_ff*100:5.1f}%)")

    total_lf = sum(usage['last_faint'].values())
    if total_lf > 0:
        print(f"\n  Last-stand (top {top_n}, of {total_lf} games — mons we lost the game on):")
        for mon, n in usage['last_faint'].most_common(top_n):
            print(f"    {mon:20s}  {n:4d} ({n/total_lf*100:5.1f}%)")


def main():
    import argparse
    p = argparse.ArgumentParser(description='Analyze Pokemon AI replay playstyles')
    p.add_argument('--replay-dir', nargs='+', default=None,
                   help='One or more replay directories to analyze and compare')
    p.add_argument('--labels', nargs='+', default=None,
                   help='Labels for each replay directory (default: dir names)')
    p.add_argument('--bot', default=None,
                   help='Filter to a specific opponent bot name')
    p.add_argument('--player-prefix', default='BCPolicyPlayer',
                   help='Player name prefix to identify "our" side (default: BCPolicyPlayer). Use "p1" or "p2" for H2H.')
    p.add_argument('--deep', action='store_true', default=False,
                   help='Enable deep qualitative analysis (switch quality, momentum, HP management, etc.)')
    p.add_argument('--iter-trajectory', default=None,
                   help='Path to a PPO run dir; auto-discovers replays_iter*/ subdirs and prints a per-iter trajectory table')
    p.add_argument('--by-opponent', action='store_true', default=False,
                   help='With --iter-trajectory, split each iter row by opponent bot')
    p.add_argument('--team-usage', action='store_true', default=False,
                   help='Show team usage stats (lead mon, send-out freq, faint order) for each --replay-dir')
    args = p.parse_args()

    # Iter-trajectory mode runs standalone and exits.
    if args.iter_trajectory:
        trajectory_mode(args.iter_trajectory, our_prefix=args.player_prefix,
                        by_opponent=args.by_opponent)
        return

    if not args.replay_dir:
        p.error('--replay-dir is required (or use --iter-trajectory)')

    labels = args.labels or [Path(d).name for d in args.replay_dir]

    all_profiles = []
    all_stats = []

    for replay_dir, label in zip(args.replay_dir, labels):
        battles = load_replays(replay_dir)
        print(f"\nLoaded {len(battles)} unique battles from {replay_dir}")

        if args.bot:
            grouped = filter_by_opponent(battles)
            # Find matching bot name
            matched = None
            for opp_name in grouped:
                if args.bot.lower() in opp_name.lower():
                    matched = opp_name
                    break
            if matched:
                battles = grouped[matched]
                print(f"  Filtered to {len(battles)} games vs {matched}")
            else:
                print(f"  WARNING: No battles found vs '{args.bot}'")
                continue

        stats = analyze_battles(battles, our_player_prefix=args.player_prefix)
        profile = compute_playstyle_profile(stats)
        all_profiles.append(profile)
        all_stats.append(stats)
        print_playstyle_report(stats, profile, label)

        if args.deep:
            deep = deep_analyze_battles(battles, our_player_prefix=args.player_prefix)
            print_deep_report(deep, stats['total_battles'])

        if args.team_usage:
            usage = compute_team_usage(battles, our_player_prefix=args.player_prefix)
            print_team_usage(usage)

    # Comparison table if multiple directories
    if len(all_profiles) >= 2:
        print_comparison(all_profiles, labels)

    # Per-opponent breakdown if not filtered
    if not args.bot and len(args.replay_dir) == 1:
        grouped = filter_by_opponent(load_replays(args.replay_dir[0]))
        print(f"\n\n{'#'*65}")
        print(f"  PER-OPPONENT BREAKDOWN")
        print(f"{'#'*65}")
        opp_profiles = []
        opp_labels = []
        for opp_name in sorted(grouped.keys()):
            opp_stats = analyze_battles(grouped[opp_name])
            opp_profile = compute_playstyle_profile(opp_stats)
            opp_profiles.append(opp_profile)
            opp_labels.append(opp_name[:12])
            decided = opp_stats['wins'] + opp_stats['losses']
            wr = opp_stats['wins'] / decided if decided else 0
            total_moves = opp_stats['our_total_actions'] - opp_stats['our_switches']
            print(f"\n  vs {opp_name}: {opp_stats['wins']}W/{opp_stats['losses']}L ({wr:.0%}) | "
                  f"Attack {opp_profile.get('attack_pct', 0):.0%} Setup {opp_profile.get('setup_pct', 0):.0%} "
                  f"Pivot {opp_profile.get('pivot_pct', 0):.0%} Switch {opp_profile.get('switch_rate', 0):.0%} "
                  f"KO {opp_profile.get('ko_ratio', 0):.2f} SE {opp_profile.get('se_ratio', 0):.0%}")
        if len(opp_profiles) >= 2:
            print_comparison(opp_profiles, opp_labels)


if __name__ == '__main__':
    main()
