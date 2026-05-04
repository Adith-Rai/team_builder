# MODEL_REGISTRY.md — Canonical model names + roster

The set of checkpoints we treat as "significant" — anything else is
training-trajectory color, not a benchmark reference.

**Naming convention**: `<arch>_<era>_<identifier>`.
- `arch`: `bc` (behavioral cloning) or `ppo`
- `era`: `v8` / `v10` / `s35` / `s39` / `curated` / etc. — immutable era tag
- `identifier`: `iter<N>`, `epoch<N>`, `legacy`, `cloud_e<N>`, etc. — fact-based

Names are **role-neutral**. "Was once the best" is metadata, not part of the name —
that way the name doesn't go stale when a new contender supersedes the role.

---

## Pre-V1 Elo ladder roster (Session 48)

The ladder we build first; future Elo measurements need only play one new model
against this fixed slate (using `eval_elo_ladder.py --add-to`).

### Models (5)

| Canonical name | Path | Arch | Params | Canonical Elo (95% CI) | History role |
|----------------|------|------|--------|------------------------|--------------|
| `bc_v8_legacy` | `bc/v8_bc_20260423_195603/best.pt` | MLP (legacy) | 14.4M | **994** [969, 1021] | Pre-rewrite BC baseline. Recorded smart_avg 45.1% on 70-team pool (Session 39). Re-eval'd 42.8% on Metamon competitive (Session 48). |
| `bc_v10_cloud_e1` | `bc/v10_cloud_gen9/epoch_001.pt` | Transformer (new arch) | 20.0M | **1101** [1076, 1128] | Cloud BC, end of epoch 1. smart_avg 63.8%. +107 Elo over `bc_v8_legacy` validates the architectural rewrite. Behind the legacy PPO peaks by ~16-25 Elo. |
| `ppo_s35_iter2979` | `rl_v9/selfplay_v9_20260413_061236/snapshot_2979.pt` | MLP (legacy) | 13.4M | **1067** [1037, 1098] | Session 35 era. Was Elo 1058 in the canonical 33-player ladder (`elo_session35_exp1.json`) — that ladder used the 70-team pool; the 1067 here is on the new 16-team Metamon set so the numbers aren't directly comparable, but ranking holds. |
| `ppo_s39_iter229` | `rl_v9/_init_sp_0229/snapshot_0229.pt` | MLP (legacy) | 14.3M | **1117** [1090, 1146] | Session 39 PPO from new BC retrain. Recorded smart_avg 67.8% on Metamon competitive (peak of a 200-iter run). +16 Elo over `bc_v10_cloud_e1` — clearly stronger by Elo, marginally so by H2H over 200 games. |
| `ppo_curated_iter119` | `rl_v9_curated_pool/selfplay_v9_20260501_011537/iter_0119.pt` | MLP (legacy) | 14.3M | **1126** [1105, 1157] | Session 44 curated pool peak. Sustained 66.15% smart_avg (3-eval window). **Highest-ranked model in the pre-V1 ladder.** Phase 1 PPO target: beat this. |

### Bots (10) — all from `eval_elo_ladder.py:ALL_BOTS`

| Canonical name | Class | Source | Canonical Elo (95% CI) |
|----------------|-------|--------|------------------------|
| `Random` | `RandomPlayer` | poke-env built-in | **382** [335, 420] |
| `MaxBasePower` | `MaxBasePowerPlayer` | poke-env built-in | **711** [686, 739] |
| `HazardSense` | `HazardSensePlayer` | `policy_rulebots.py` | **766** [736, 791] |
| `SwitchAwareEscape` | `SwitchAwareEscapePlayer` | `policy_rulebots.py` | **769** [738, 798] |
| `GreedySE` | `GreedySEPlayer` | `policy_rulebots.py` | **783** [755, 811] |
| `SetupThenSweep` | `SetupThenSweepPlayer` | `policy_rulebots.py` | **836** [811, 860] |
| **`SH`** (anchor at Elo 1000) | `SimpleHeuristicsPlayer` | poke-env built-in | **1000** (anchor) |
| `SmartDmg` | `SmartDamagePlayer` | `policy_smartbots.py` | **1010** [983, 1040] |
| `Strategic` | `StrategicPlayer` | `policy_smartbots.py` | **1022** [1002, 1048] |
| `Tactical` | `TacticalPlayer` | `policy_smartbots.py` | **1023** [1001, 1048] |

Bots span ~75% win-rate spread per the Session 23 round-robin → ~400 Elo of dynamic range
for snapshot interpolation.

### Total roster: 15 entrants → 105 all-vs-all matchups → 10,500 games at 100 g/matchup.

---

## Pre-V1 Elo ladder — full ranking (Session 48, 2026-05-03)

```
Rank  Player                  Elo    95% CI       Type   Class
─────────────────────────────────────────────────────────────────
 1    ppo_curated_iter119    1126   [1105, 1157]  model  PPO ceiling
 2    ppo_s39_iter229        1117   [1090, 1146]  model  PPO #2
 3    bc_v10_cloud_e1        1101   [1076, 1128]  model  new-arch BC
 4    ppo_s35_iter2979       1067   [1037, 1098]  model  legacy PPO peak
 5    Tactical               1023   [1001, 1048]  bot    smart
 6    Strategic              1022   [1002, 1048]  bot    smart
 7    SmartDmg               1010   [ 983, 1040]  bot    smart
 8    SH (anchor)            1000   [1000, 1000]  bot    smart
 9    bc_v8_legacy            994   [ 969, 1021]  model  legacy BC
10    SetupThenSweep          836   [ 811,  860]  bot    rule
11    GreedySE                783   [ 755,  811]  bot    rule
12    SwitchAwareEscape       769   [ 738,  798]  bot    rule
13    HazardSense             766   [ 736,  791]  bot    rule
14    MaxBasePower            711   [ 686,  739]  bot    floor
15    Random                  382   [ 335,  420]  bot    floor
```

Source: `data/eval/registry/elo_v10_baseline.json` — 100 games/matchup, 105 matchups, 10,500 games total, ~53 min wall-clock on local RTX 3060.

### Key observations

- **Architectural rewrite is worth ~107 Elo at BC level** (`bc_v8_legacy` 994 → `bc_v10_cloud_e1` 1101). Cleanest measurement of the rewrite's gain.
- **Our BC is below the legacy PPO ceiling by ~25 Elo** (1101 vs 1126). Phase 1 PPO target: cross 1130-1150.
- **`bc_v10_cloud_e1` ≥ legacy s35 PPO peak** (1101 vs 1067). The new arch's BC alone matches or exceeds older PPO eras.
- **Smart bots cluster within ±25 Elo** (1000-1023). Their differentiation is real but narrow.
- **`bc_v8_legacy` is essentially equal to SH** (994 vs 1000). Legacy BC was barely better than a heuristic.
- **Random vs SH = ~620 Elo gap** — wide dynamic range for placing future models.

### When this ladder will be updated

Add a new model via `--add-to`:

```bash
python eval_elo_ladder.py \
  --add-to data/eval/registry/elo_v10_baseline.json \
  --snapshots <path> --names <canonical_name> \
  --bots all --n-games 100 --concurrency 100 --device cuda \
  --team-set metamon-competitive \
  --out-json data/eval/registry/elo_v10_plus_<name>.json
```

Existing players' Elos get small adjustments (10-game samples added to the BT-MLE fit), but rankings should stay stable. If a new model causes large Elo shifts on the existing players, that signals the new model exposed unmodeled exploit patterns — worth investigating.

---

## Annotation: history-of-roles

(Free-text, can be updated freely. Used to recall context, not for matching.)

- `ppo_s35_iter2979`: was the **canonical pool anchor** for Session 35's curated work.
  Also referenced in older docs as the "Elo 1058 peak" or "all-time-best pre-rewrite."
- `ppo_s39_iter229`: was the **smart_avg champion** through Session 47. Also referenced
  as `sp_0229` or "post-BC PPO peak."
- `ppo_curated_iter119`: was the **curated pool peak** (Session 44). Also referenced
  as `iter_0119` or `sp_0119` or "the Session 44 ceiling."
- `bc_v8_legacy`: was the **BC baseline** for all post-S39 PPO runs. Sometimes called
  "v8 BC" or just "legacy BC."
- `bc_v10_cloud_e1`: cloud BC contender. As of Session 48 it's the **strongest model
  we have by H2H** but not yet by recorded smart_avg.

These annotations decay; canonical names don't.

---

## Future entries (placeholders)

When phases 1-4 of `PPO_PHASED_TRAINING.md` complete, add:

- `ppo_v10_phase1_best` — Phase-1-best (self-play-only PPO from BC)
- `ppo_v10_phase2_best` — Phase-2-best (light external pool)
- `ppo_v10_phase3_best` — Phase-3-best (Metamon SmallRL/Medium/Abra tier)
- `ppo_v10_phase4_best` — Phase-4-best / V1 final

Each gets one `eval_elo_ladder.py --add-to` invocation when its phase ends.
Each gets a row in the table above plus a history-of-roles annotation.

---

## How the canonical name flows through the codebase

1. **Filenames stay arbitrary.** No rename-on-disk (would break old training resume paths).
2. **eval scripts take `label=path`** specs. Pass canonical name as the label.
   ```bash
   python eval_elo_ladder.py \
     --snapshots data/models/rl_v9/_init_sp_0229/snapshot_0229.pt \
     --names ppo_s39_iter229 \
     ...
   ```
3. **All result JSONs / evals.jsonl entries use the canonical name** going forward.
4. **Docs reference by canonical name only.** If you write "sp_0229" in a new doc, that's a bug to fix.

---

## Sanity-check the registry against reality

Quick test: can each model in the roster actually load?

```bash
cd pokemon-ai-starter/pokemon-ai/src
for ckpt in \
  data/models/bc/v8_bc_20260423_195603/best.pt \
  data/models/bc/v10_cloud_gen9/epoch_001.pt \
  data/models/rl_v9/selfplay_v9_20260413_061236/snapshot_2979.pt \
  data/models/rl_v9/_init_sp_0229/snapshot_0229.pt \
  data/models/rl_v9_curated_pool/selfplay_v9_20260501_011537/iter_0119.pt; do
  python -c "
from ppo import load_checkpoint
import torch
m, cfg, _ = load_checkpoint('$ckpt', torch.device('cuda'))
print(f'OK: $ckpt ({m.count_parameters():,} params)')
" 2>&1 | tail -1
done
```

(All 5 should print `OK: ... params`.)

---

## Where Elo ladder results live

`data/eval/registry/elo_<name>.json` — JSON output of `eval_elo_ladder.py`.

Sibling `data/eval/registry/elo_<name>.jsonl` — incremental per-matchup save
(crashed runs are partially recoverable from this).

For new ladder additions: `data/eval/registry/elo_<name>_plus_<new_player>.json`
preserves the original baseline file untouched.
