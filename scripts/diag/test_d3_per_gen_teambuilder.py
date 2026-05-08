#!/usr/bin/env python
"""D3 per-gen procedural teambuilder validation.

team_generator.py was already gen-aware via dispatch (`get_ban_list(gen)`,
`_default_tiers(gen)`, `load_pokemon_pool(stats_dir, gen)`,
`ProceduralTeambuilder(stats_dir, ..., gen)`), but never tested for
gens 6/7/8 - all production runs were gen 9 OU. This test:

  1. Builds ProceduralTeambuilder for each of gens 6, 7, 8, 9
  2. Generates 5 teams per gen via yield_team()
  3. Parses each via parse_showdown_team
  4. Asserts each team has 6 Pokemon
  5. Cross-checks Pokemon names against the gen's UBERS ban list
     (none should appear)
  6. (Defensive) checks held items against raw_data/items/items.csv
     `gen_added` - any item with gen_added > target_gen is a bug

If a gen's stats data is missing locally, that gen is SKIPPED with a
warning (not a failure - missing data is a separate issue from the code
not working).

Usage (local Windows or cloud Linux):
  python scripts/diag/test_d3_per_gen_teambuilder.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Dict


def load_item_gen_added() -> Dict[str, int]:
    """Map normalized item name -> gen_added (int). Tries both local repo
    layout and pod /workspace layout."""
    candidates = [
        Path(__file__).resolve().parents[1] / "raw_data" / "items" / "items.csv",
        Path(__file__).resolve().parents[2] / "raw_data" / "items" / "items.csv",
        Path("/workspace/team_builder/raw_data/items/items.csv"),
    ]
    items_csv = next((p for p in candidates if p.exists()), None)
    if items_csv is None:
        print(f"WARN: items.csv not found in any of {candidates} - skipping item gen check")
        return {}
    print(f"item-gen lookup: {items_csv}")
    out: Dict[str, int] = {}
    with open(items_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("item_name") or "").strip().lower()
            gen_str = (row.get("gen_added") or "").strip()
            if not name:
                continue
            try:
                gen = int(float(gen_str))
            except (ValueError, TypeError):
                continue
            out[name] = gen
    return out


def main():
    # Find src/ - works both for local repo layout and pod /workspace layout.
    candidate_src = [
        Path(__file__).resolve().parents[1] / "pokemon-ai-starter" / "pokemon-ai" / "src",
        Path(__file__).resolve().parents[2] / "pokemon-ai-starter" / "pokemon-ai" / "src",
        Path("/workspace/team_builder/pokemon-ai-starter/pokemon-ai/src"),
    ]
    src_dir = next((p for p in candidate_src if p.exists()), None)
    if src_dir is None:
        print(f"FAIL: src not found in any of: {candidate_src}")
        return 1
    repo_root = src_dir.parents[2]
    sys.path.insert(0, str(src_dir))
    os.chdir(src_dir)

    from team_generator import (
        ProceduralTeambuilder, get_ban_list, _normalize_name,
    )

    stats_dir = repo_root / "raw_data" / "pokemon_usage" / "2024-04"
    if not stats_dir.exists():
        # Try cloud path
        stats_dir_alt = Path("/workspace/raw_data/pokemon_usage/2024-04")
        if stats_dir_alt.exists():
            stats_dir = stats_dir_alt
        else:
            print(f"FAIL: stats dir not found at {stats_dir} or {stats_dir_alt}")
            return 1

    print(f"=== D3 per-gen teambuilder validation ===")
    print(f"stats_dir: {stats_dir}")
    print()

    item_gens = load_item_gen_added()
    print(f"loaded {len(item_gens)} item gen_added entries")
    print()

    gens_to_test = [6, 7, 8, 9]
    overall_ok = True
    summary = {}

    for gen in gens_to_test:
        print(f"--- gen {gen} ---")
        # Check if any tier file exists for this gen
        any_tier = False
        for tier in (f"gen{gen}ou", f"gen{gen}uu"):
            for rating in ("0", "1500", "1630", "1695", "1760"):
                if (stats_dir / f"{tier}-{rating}.txt").exists():
                    any_tier = True
                    break
            if any_tier:
                break
        if not any_tier:
            print(f"  SKIP: no usage files for gen {gen} in {stats_dir}")
            summary[gen] = "skipped"
            continue

        try:
            tb = ProceduralTeambuilder(str(stats_dir), random_pct=0.0, gen=gen)
        except Exception as e:
            print(f"  FAIL: ProceduralTeambuilder(gen={gen}) raised: {e}")
            summary[gen] = "construct_fail"
            overall_ok = False
            continue

        if len(tb.pool) == 0:
            print(f"  FAIL: gen {gen} pool is empty (no usage data parsed)")
            summary[gen] = "empty_pool"
            overall_ok = False
            continue

        ban_list = get_ban_list(gen)
        print(f"  pool size: {len(tb.pool)}; ban list size: {len(ban_list)}")

        teams_ok = 0
        teams_fail = 0
        item_violations = 0
        ban_violations = 0
        # Parse packed format directly: in poke-env packed format, each mon
        # is "name|species|item|ability|moves|nature|evs|gender|ivs|shiny|level|tera"
        # separated by `]`. Empty trailing field after last `]` is OK.
        for i in range(5):
            try:
                packed = tb.yield_team()
                mon_blocks = [b for b in packed.split("]") if b.strip()]
                if len(mon_blocks) != 6:
                    print(f"  team {i}: expected 6 mons, got {len(mon_blocks)}; packed[:200]={packed[:200]!r}")
                    teams_fail += 1
                    continue

                for block in mon_blocks:
                    fields = block.split("|")
                    # fields[1] = species, fields[2] = item
                    species = fields[1] if len(fields) > 1 else ""
                    item = fields[2] if len(fields) > 2 else ""

                    name_norm = _normalize_name(species)
                    if name_norm in ban_list:
                        print(f"  team {i}: BANNED MON: {species}")
                        ban_violations += 1

                    if item_gens and item:
                        item_name = item.strip().lower()
                        if item_name and item_name != "none":
                            item_gen = item_gens.get(item_name)
                            if item_gen is not None and item_gen > gen:
                                print(f"  team {i}: gen-{item_gen} item {item!r} on gen-{gen} team ({species})")
                                item_violations += 1

                teams_ok += 1
            except Exception as e:
                print(f"  team {i}: parse failed: {e}")
                teams_fail += 1

        ok = teams_ok > 0 and ban_violations == 0 and item_violations == 0 and teams_fail == 0
        status = "PASS" if ok else "FAIL"
        print(f"  result: {status} ({teams_ok}/5 teams OK, {ban_violations} ban viol, "
              f"{item_violations} item viol, {teams_fail} parse fails)")
        summary[gen] = status
        if not ok:
            overall_ok = False
        print()

    print("=== Summary ===")
    for gen, s in summary.items():
        print(f"  gen {gen}: {s}")
    print()
    print("VERDICT: D3 PASS" if overall_ok else "VERDICT: D3 FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
