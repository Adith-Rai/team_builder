# fishbowl_v2 era4_chain_v2 results — REFUTED (CIS-Elo team-set bug)

**Status as of 2026-06-03**: Numbers below are INVALID due to a team-distribution mismatch
between the base ladder and the add-to run. Kept here as a historical record + so the
growth-trajectory direction can be cited (still valid in *relative* terms within the
fishbowl_v2 snapshots).

## The bug

CIS-Elo v1+v2 hardcoded `random_pool_teambuilder()` (procedural Smogon teams) while
classic `eval_elo_ladder.py` defaults to `metamon-competitive` teams. The original 591
matches in `era4_chain_FINAL.json` were under metamon-competitive; the 146 new matches
we added via CIS-Elo v2 were under procedural. BT refit treats them as same-distribution
skill data → Elos confounded.

**Cleanup performed**: `era4_chain_v2.json`, `era4_v2_workers.jsonl`,
`era4_v2_classic_shard*.json/jsonl` deleted from dev pod. `era4_chain_FINAL.json`
(pre-pollution source of truth) preserved.

**Fix shipped (commit `c02971a6`)**: `--team-set {metamon-competitive, pool}` flag on
both CIS-Elo v1 and v2, default `metamon-competitive`.

## Original placements (REFUTED — do not cite as absolute Elos)

39 players, 737 matches, BT-fit, SH=1000 anchor:

| Rank | Snapshot | Elo |
|---|---|---|
| #1 | POST_INIT_iter139 | 1169.6 |
| **#2** | **fishbowl_v2_iter149** | **1166.3** |
| #3 | POST_INIT_iter89 | 1166.1 |
| #4 | POST_INIT_iter119 | 1165.0 |
| #5 | phase2_vf05_v1_iter189 | 1163.7 |
| ... | ... | ... |
| #20 | fishbowl_v2_iter109 | 1153.6 |
| #24 | fishbowl_v2_iter69 | 1150.4 |
| #26 | fishbowl_v2_iter29 | 1145.9 |

## Growth curve (still directionally valid within fishbowl_v2 lineage)

All 4 snapshots share the same training-team distribution, so internal growth comparison
is internally consistent even though placements vs other lineages are confounded:

- iter29 = 1145.9 (rank #26, just below PRE_INIT_iter29 = 1146.7)
- iter69 = 1150.4 (#24, +4 Elo)
- iter109 = 1153.6 (#20, +8 cumulative)
- iter149 = 1166.3 (#2, **+20 cumulative, +12 in last 40 iters**)

**Back-loaded climb** — first 109 iters gained ~8 Elo (slow); last 40 iters gained ~12
Elo (the 18-rank #20→#2 leap). Late iters did most of the work. Suggested further growth
potential is plausible → motivated `fishbowl_v2_resume` (task #121).

## Direct head-to-heads vs POST_INIT_iter139 ("1178.4 record") at 500g/pair

- fishbowl_v2_iter29: 46% (loses 8%)
- fishbowl_v2_iter69: 50% (tied)
- fishbowl_v2_iter109: ~48% (close)
- fishbowl_v2_iter149: 55% (beats record by 10%)

Direct h2h shows iter149 actually beats POST_INIT_iter139 head-to-head. BT refit placed
it at #2 with 3.3-Elo gap because BT incorporates the full ladder of cross-matchups —
but this BT result is the one confounded by the team-distribution mismatch.

## Lessons captured

- LR=1e-4 + dense + BC v10 init + 62-entry pool **does work** at this scale. Don't drop
  the config. (This lesson survives the refutation — it's based on internally-consistent
  growth, not absolute Elo.)
- Don't trust bot WRs to discriminate within top cluster — smart_avg 70-74% is the
  universal ceiling for every model in this cluster. Use snap-snap matchups for actual
  placement.
- BT refit can mix team distributions silently. Always check `--team-set` matches the
  base ladder's team source. The fix flag now warns on mismatch.

## Re-eval pending

When `fishbowl_v2_resume` (task #121) finishes, run CIS-Elo v2 add-to from
`era4_chain_FINAL.json` with default `--team-set metamon-competitive` to get clean
placements for ALL 4 fishbowl_v2 snaps + the new resume snapshots.
