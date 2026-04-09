"""Analyze whether the AI targets status moves intelligently.
Burns on physical attackers, paralysis on fast mons, toxic on bulky ones, etc."""

import os, re, sys, glob
from collections import defaultdict, Counter

# Pokemon stat categories (base stats from gen9)
# We'll classify mons by their primary role based on base stats
POKEMON_DATA = {
    # Physical attackers (high Atk)
    "Garchomp": {"atk": 130, "spa": 80, "spe": 102, "hp": 108, "def": 95, "spd": 85, "cat": "phys"},
    "Great Tusk": {"atk": 131, "spa": 53, "spe": 87, "hp": 115, "def": 131, "spd": 53, "cat": "phys"},
    "Dragonite": {"atk": 134, "spa": 100, "spe": 80, "hp": 91, "def": 95, "spd": 100, "cat": "phys"},
    "Kingambit": {"atk": 135, "spa": 60, "spe": 50, "hp": 100, "def": 120, "spd": 85, "cat": "phys"},
    "Roaring Moon": {"atk": 139, "spa": 55, "spe": 119, "hp": 105, "def": 101, "spd": 83, "cat": "phys"},
    "Palafin": {"atk": 160, "spa": 106, "spe": 100, "hp": 100, "def": 97, "spd": 87, "cat": "phys"},
    "Baxcalibur": {"atk": 145, "spa": 75, "spe": 87, "hp": 115, "def": 92, "spd": 86, "cat": "phys"},
    "Iron Valiant": {"atk": 130, "spa": 120, "spe": 116, "hp": 74, "def": 90, "spd": 60, "cat": "phys"},
    "Sneasler": {"atk": 130, "spa": 40, "spe": 120, "hp": 80, "def": 60, "spd": 80, "cat": "phys"},
    "Barraskewda": {"atk": 123, "spa": 60, "spe": 136, "hp": 61, "def": 60, "spd": 50, "cat": "phys"},
    "Weavile": {"atk": 120, "spa": 45, "spe": 125, "hp": 70, "def": 65, "spd": 85, "cat": "phys"},
    "Urshifu": {"atk": 130, "spa": 63, "spe": 97, "hp": 100, "def": 100, "spd": 63, "cat": "phys"},
    "Cinderace": {"atk": 116, "spa": 65, "spe": 119, "hp": 80, "def": 75, "spd": 75, "cat": "phys"},
    "Excadrill": {"atk": 135, "spa": 50, "spe": 88, "hp": 110, "def": 60, "spd": 65, "cat": "phys"},
    "Scizor": {"atk": 130, "spa": 55, "spe": 65, "hp": 70, "def": 100, "spd": 80, "cat": "phys"},
    "Hawlucha": {"atk": 92, "spa": 74, "spe": 118, "hp": 78, "def": 75, "spd": 63, "cat": "phys"},
    "Rhyperior": {"atk": 140, "spa": 55, "spe": 40, "hp": 115, "def": 130, "spd": 55, "cat": "phys"},
    "Conkeldurr": {"atk": 140, "spa": 55, "spe": 45, "hp": 105, "def": 95, "spd": 65, "cat": "phys"},
    "Metagross": {"atk": 135, "spa": 95, "spe": 70, "hp": 80, "def": 130, "spd": 90, "cat": "phys"},
    "Tyranitar": {"atk": 134, "spa": 95, "spe": 61, "hp": 100, "def": 110, "spd": 100, "cat": "phys"},
    "Corviknight": {"atk": 87, "spa": 53, "spe": 67, "hp": 98, "def": 105, "spd": 85, "cat": "phys"},
    "Gliscor": {"atk": 95, "spa": 45, "spe": 95, "hp": 75, "def": 125, "spd": 75, "cat": "phys"},
    "Landorus": {"atk": 125, "spa": 115, "spe": 101, "hp": 89, "def": 90, "spd": 80, "cat": "phys"},
    "Crawdaunt": {"atk": 120, "spa": 90, "spe": 55, "hp": 63, "def": 85, "spd": 55, "cat": "phys"},
    "Bisharp": {"atk": 125, "spa": 40, "spe": 70, "hp": 65, "def": 100, "spd": 70, "cat": "phys"},
    "Ting-Lu": {"atk": 110, "spa": 55, "spe": 45, "hp": 155, "def": 125, "spd": 80, "cat": "bulky_phys"},

    # Special attackers (high SpA)
    "Dragapult": {"atk": 120, "spa": 100, "spe": 142, "hp": 88, "def": 75, "spd": 75, "cat": "fast_special"},
    "Gholdengo": {"atk": 60, "spa": 133, "spe": 84, "hp": 87, "def": 91, "spd": 97, "cat": "special"},
    "Heatran": {"atk": 90, "spa": 130, "spe": 77, "hp": 91, "def": 106, "spd": 106, "cat": "special"},
    "Volcarona": {"atk": 60, "spa": 135, "spe": 100, "hp": 85, "def": 65, "spd": 105, "cat": "special"},
    "Tapu Lele": {"atk": 85, "spa": 130, "spe": 95, "hp": 70, "def": 75, "spd": 115, "cat": "special"},
    "Primarina": {"atk": 74, "spa": 126, "spe": 60, "hp": 80, "def": 74, "spd": 116, "cat": "special"},
    "Magnezone": {"atk": 70, "spa": 130, "spe": 60, "hp": 70, "def": 115, "spd": 90, "cat": "special"},
    "Hydreigon": {"atk": 105, "spa": 125, "spe": 98, "hp": 92, "def": 90, "spd": 90, "cat": "special"},
    "Gengar": {"atk": 65, "spa": 130, "spe": 110, "hp": 60, "def": 60, "spd": 75, "cat": "fast_special"},
    "Rotom-Wash": {"atk": 65, "spa": 105, "spe": 86, "hp": 50, "def": 107, "spd": 107, "cat": "special"},
    "Rotom": {"atk": 65, "spa": 105, "spe": 86, "hp": 50, "def": 107, "spd": 107, "cat": "special"},
    "Clefable": {"atk": 70, "spa": 95, "spe": 60, "hp": 95, "def": 73, "spd": 90, "cat": "bulky_special"},
    "Ninetales": {"atk": 76, "spa": 81, "spe": 100, "hp": 73, "def": 75, "spd": 100, "cat": "special"},
    "Alakazam": {"atk": 50, "spa": 135, "spe": 120, "hp": 55, "def": 45, "spd": 95, "cat": "fast_special"},
    "Specs Keldeo": {"atk": 72, "spa": 129, "spe": 108, "hp": 91, "def": 90, "spd": 90, "cat": "fast_special"},

    # Fast Pokemon (high Spe, various offense)
    "Iron Bundle": {"atk": 56, "spa": 124, "spe": 136, "hp": 56, "def": 64, "spd": 114, "cat": "fast_special"},
    "Flutter Mane": {"atk": 55, "spa": 135, "spe": 135, "hp": 55, "def": 55, "spd": 135, "cat": "fast_special"},
    "Cinderace": {"atk": 116, "spa": 65, "spe": 119, "hp": 80, "def": 75, "spd": 75, "cat": "fast_phys"},
    "Zeraora": {"atk": 112, "spa": 102, "spe": 143, "hp": 88, "def": 75, "spd": 80, "cat": "fast_phys"},
    "Jolteon": {"atk": 65, "spa": 110, "spe": 130, "hp": 65, "def": 60, "spd": 95, "cat": "fast_special"},
    "Tornadus": {"atk": 100, "spa": 125, "spe": 111, "hp": 79, "def": 70, "spd": 80, "cat": "fast_special"},
    "Rillaboom": {"atk": 125, "spa": 60, "spe": 85, "hp": 100, "def": 90, "spd": 70, "cat": "phys"},

    # Bulky/defensive (high HP/Def/SpD)
    "Toxapex": {"atk": 63, "spa": 53, "spe": 35, "hp": 50, "def": 152, "spd": 142, "cat": "wall"},
    "Blissey": {"atk": 10, "spa": 75, "spe": 55, "hp": 255, "def": 10, "spd": 135, "cat": "wall"},
    "Chansey": {"atk": 5, "spa": 35, "spe": 50, "hp": 250, "def": 5, "spd": 105, "cat": "wall"},
    "Dondozo": {"atk": 100, "spa": 65, "spe": 35, "hp": 150, "def": 115, "spd": 65, "cat": "wall"},
    "Garganacl": {"atk": 100, "spa": 45, "spe": 35, "hp": 100, "def": 130, "spd": 90, "cat": "wall"},
    "Slowking": {"atk": 75, "spa": 100, "spe": 30, "hp": 95, "def": 80, "spd": 110, "cat": "bulky_special"},
    "Hippowdon": {"atk": 112, "spa": 68, "spe": 47, "hp": 108, "def": 118, "spd": 72, "cat": "wall"},
    "Skarmory": {"atk": 80, "spa": 40, "spe": 70, "hp": 65, "def": 140, "spd": 70, "cat": "wall"},
    "Clodsire": {"atk": 75, "spa": 45, "spe": 20, "hp": 130, "def": 60, "spd": 100, "cat": "wall"},
    "Alomomola": {"atk": 75, "spa": 40, "spe": 65, "hp": 165, "def": 80, "spd": 45, "cat": "wall"},
    "Dusclops": {"atk": 70, "spa": 60, "spe": 25, "hp": 40, "def": 130, "spd": 130, "cat": "wall"},
    "Snorlax": {"atk": 110, "spa": 65, "spe": 30, "hp": 160, "def": 65, "spd": 110, "cat": "wall"},
    "Hatterene": {"atk": 90, "spa": 136, "spe": 29, "hp": 57, "def": 95, "spd": 103, "cat": "bulky_special"},
    "Gastrodon": {"atk": 83, "spa": 92, "spe": 39, "hp": 111, "def": 68, "spd": 82, "cat": "bulky_special"},
    "Milotic": {"atk": 60, "spa": 100, "spe": 81, "hp": 95, "def": 79, "spd": 125, "cat": "bulky_special"},
    "Mandibuzz": {"atk": 65, "spa": 55, "spe": 80, "hp": 110, "def": 105, "spd": 95, "cat": "wall"},
    "Ferrothorn": {"atk": 94, "spa": 54, "spe": 20, "hp": 74, "def": 131, "spd": 116, "cat": "wall"},
    "Umbreon": {"atk": 65, "spa": 60, "spe": 65, "hp": 95, "def": 110, "spd": 130, "cat": "wall"},
    "Vaporeon": {"atk": 65, "spa": 110, "spe": 65, "hp": 130, "def": 60, "spd": 95, "cat": "bulky_special"},
    "Sylveon": {"atk": 65, "spa": 110, "spe": 60, "hp": 95, "def": 65, "spd": 130, "cat": "bulky_special"},
    "Espeon": {"atk": 65, "spa": 130, "spe": 110, "hp": 65, "def": 60, "spd": 95, "cat": "fast_special"},
}

def classify(name):
    """Classify a Pokemon by what status would be smart to use on it."""
    name = name.strip()
    # Normalize some names
    for alias, canonical in [("Rotom-Wash", "Rotom"), ("Landorus-Therian", "Landorus"),
                             ("Urshifu-Rapid-Strike", "Urshifu"), ("Tornadus-Therian", "Tornadus")]:
        if name == alias:
            name = canonical

    if name in POKEMON_DATA:
        d = POKEMON_DATA[name]
        results = {}
        # Is it primarily physical? (burn is smart)
        results["physical"] = d["atk"] > d["spa"] and d["atk"] >= 90
        # Is it fast? (paralysis is smart)
        results["fast"] = d["spe"] >= 95
        # Is it bulky? (toxic is smart)
        results["bulky"] = (d["hp"] + d["def"] + d["spd"]) >= 280 or d["hp"] >= 100
        # Is it special? (burn doesn't help much)
        results["special"] = d["spa"] > d["atk"] and d["spa"] >= 90
        results["stats"] = d
        return results
    return None

def analyze_replays(replay_dir, our_prefix="p2"):
    """Parse replays to find status application and what was targeted."""
    opp = "p1" if our_prefix == "p2" else "p2"

    status_events = []  # (status_type, target_pokemon, source_move, who_applied)

    for bot_dir in ["SH", "SmartDmg", "Tactical", "Strategic"]:
        dirpath = os.path.join(replay_dir, bot_dir)
        if not os.path.isdir(dirpath):
            continue
        for f in os.listdir(dirpath):
            if not f.endswith(".html"):
                continue
            filepath = os.path.join(dirpath, f)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                lines = fh.readlines()

            last_move = {}  # player -> (pokemon, move)
            active = {}  # player -> pokemon name

            for line in lines:
                line = line.strip()

                # Track active Pokemon
                m = re.match(r'\|switch\|(p[12])a: ([^|]+)\|', line)
                if m:
                    active[m.group(1)] = m.group(2).split(',')[0].strip()
                    continue
                m = re.match(r'\|drag\|(p[12])a: ([^|]+)\|', line)
                if m:
                    active[m.group(1)] = m.group(2).split(',')[0].strip()
                    continue

                # Track moves
                m = re.match(r'\|move\|(p[12])a: ([^|]+)\|([^|]+)', line)
                if m:
                    player = m.group(1)
                    mon = m.group(2).strip()
                    move = m.group(3).strip()
                    active[player] = mon
                    last_move[player] = (mon, move)
                    continue

                # Track status application
                m = re.match(r'\|-status\|(p[12])a: ([^|]+)\|(\w+)', line)
                if m:
                    target_player = m.group(1)
                    target_mon = m.group(2).strip()
                    status = m.group(3)

                    # Skip self-inflicted (Toxic Orb, Flame Orb, etc.)
                    if "[from] item:" in line:
                        continue
                    if "[from] ability:" in line:
                        continue

                    # Who applied it?
                    applier = opp if target_player == opp else our_prefix
                    # If target is opponent's mon, we applied it (or it was from our hazards)
                    if target_player == opp:
                        applier = our_prefix
                    else:
                        applier = opp

                    # Check if from Toxic Spikes (no direct move)
                    if "[from] Toxic Spikes" in line or "from: Toxic Spikes" in line.lower():
                        source = "Toxic Spikes (hazard)"
                    elif applier in last_move:
                        source = last_move[applier][1]
                    else:
                        source = "unknown"

                    status_events.append({
                        "status": status,
                        "target": target_mon,
                        "source": source,
                        "by_us": applier == our_prefix,
                        "bot": bot_dir,
                    })

    return status_events

def main():
    replay_dir = sys.argv[1] if len(sys.argv) > 1 else "data/models/rl_v9/selfplay_v9_20260404_192922/replays_iter1059"
    label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(replay_dir)

    events = analyze_replays(replay_dir)
    our_events = [e for e in events if e["by_us"]]

    print(f"\n{'='*70}")
    print(f"  STATUS TARGETING ANALYSIS: {label}")
    print(f"  {len(our_events)} status applications by us (of {len(events)} total)")
    print(f"{'='*70}")

    # Group by status type
    for status in ["brn", "par", "tox", "psn", "slp", "frz"]:
        status_evts = [e for e in our_events if e["status"] == status]
        if not status_evts:
            continue

        targets = Counter(e["target"] for e in status_evts)
        sources = Counter(e["source"] for e in status_evts)

        # Classify targeting quality
        smart = 0
        neutral = 0
        bad = 0
        unknown = 0
        smart_targets = []
        bad_targets = []

        for e in status_evts:
            info = classify(e["target"])
            if info is None:
                unknown += 1
                continue

            if status == "brn":
                if info["physical"]:
                    smart += 1
                    smart_targets.append(e["target"])
                elif info["special"]:
                    bad += 1
                    bad_targets.append(e["target"])
                else:
                    neutral += 1
            elif status == "par":
                if info["fast"]:
                    smart += 1
                    smart_targets.append(e["target"])
                elif info["bulky"] and not info["fast"]:
                    bad += 1  # slow bulky mon doesn't care about par speed drop
                    bad_targets.append(e["target"])
                else:
                    neutral += 1
            elif status in ("tox", "psn"):
                if info["bulky"]:
                    smart += 1
                    smart_targets.append(e["target"])
                elif info["fast"] and not info["bulky"]:
                    bad += 1  # fast frail mon will die before tox matters
                    bad_targets.append(e["target"])
                else:
                    neutral += 1

        classified = smart + neutral + bad
        total = classified + unknown

        status_names = {"brn": "BURN", "par": "PARALYSIS", "tox": "TOXIC", "psn": "POISON", "slp": "SLEEP", "frz": "FREEZE"}

        ideal = {"brn": "physical attackers (halves Atk)",
                 "par": "fast threats (halves Speed)",
                 "tox": "bulky walls (%-based damage)",
                 "psn": "bulky walls (chip damage)"}

        print(f"\n  --- {status_names.get(status, status.upper())} ({total} applications) ---")
        print(f"  Ideal target: {ideal.get(status, 'any')}")
        if classified > 0:
            print(f"  Smart:   {smart:3d}/{classified} ({100*smart/classified:.0f}%) — correct target type")
            print(f"  Neutral: {neutral:3d}/{classified} ({100*neutral/classified:.0f}%) — not ideal but not wasteful")
            print(f"  Bad:     {bad:3d}/{classified} ({100*bad/classified:.0f}%) — wrong target type")
        if unknown:
            print(f"  Unknown: {unknown} (Pokemon not in database)")

        print(f"  Sources: {', '.join(f'{m}({c})' for m,c in sources.most_common(5))}")
        print(f"  Top targets: {', '.join(f'{m}({c})' for m,c in targets.most_common(8))}")
        if smart_targets:
            print(f"  Smart examples: {', '.join(Counter(smart_targets).most_common(5).__iter__().__next__()[0] for _ in range(min(5, len(set(smart_targets)))))}")
            sc = Counter(smart_targets).most_common(5)
            print(f"  Smart targets:  {', '.join(f'{m}({c})' for m,c in sc)}")
        if bad_targets:
            bc = Counter(bad_targets).most_common(5)
            print(f"  Bad targets:    {', '.join(f'{m}({c})' for m,c in bc)}")

    # Per-bot comparison
    print(f"\n  {'='*60}")
    print(f"  PER-BOT STATUS USAGE")
    print(f"  {'='*60}")
    for bot in ["SH", "SmartDmg", "Tactical", "Strategic"]:
        bot_evts = [e for e in our_events if e["bot"] == bot]
        if not bot_evts:
            continue
        by_status = Counter(e["status"] for e in bot_evts)
        print(f"  vs {bot:12s}: {', '.join(f'{s}={c}' for s,c in by_status.most_common())}")

    # Smart targeting summary
    print(f"\n  {'='*60}")
    print(f"  OVERALL TARGETING IQ")
    print(f"  {'='*60}")
    all_smart = 0
    all_bad = 0
    all_classified = 0
    for status in ["brn", "par", "tox", "psn"]:
        for e in [ev for ev in our_events if ev["status"] == status]:
            info = classify(e["target"])
            if info is None:
                continue
            all_classified += 1
            if status == "brn" and info["physical"]:
                all_smart += 1
            elif status == "brn" and info["special"]:
                all_bad += 1
            elif status == "par" and info["fast"]:
                all_smart += 1
            elif status == "par" and info["bulky"] and not info["fast"]:
                all_bad += 1
            elif status in ("tox", "psn") and info["bulky"]:
                all_smart += 1
            elif status in ("tox", "psn") and info["fast"] and not info.get("bulky"):
                all_bad += 1

    if all_classified:
        print(f"  Smart targeting rate: {all_smart}/{all_classified} ({100*all_smart/all_classified:.0f}%)")
        print(f"  Bad targeting rate:   {all_bad}/{all_classified} ({100*all_bad/all_classified:.0f}%)")
        print(f"  (Random targeting would be ~33% smart, ~33% bad)")

if __name__ == "__main__":
    main()
