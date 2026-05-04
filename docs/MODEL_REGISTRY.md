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

| Canonical name | Path | Arch | Params | History role |
|----------------|------|------|--------|--------------|
| `bc_v8_legacy` | `bc/v8_bc_20260423_195603/best.pt` | MLP (legacy) | 14.4M | Pre-rewrite BC baseline. Recorded smart_avg 45.1% on 70-team pool (Session 39). Re-eval'd 42.8% on Metamon competitive (Session 48). |
| `bc_v10_cloud_e1` | `bc/v10_cloud_gen9/epoch_001.pt` | Transformer (new arch) | 20.0M | Cloud BC, end of epoch 1. smart_avg 63.8%. **Already beats every legacy PPO peak in head-to-head play (Session 48 gauntlet, +6 to +50 pt gaps).** |
| `ppo_s35_iter2979` | `rl_v9/selfplay_v9_20260413_061236/snapshot_2979.pt` | MLP (legacy) | 13.4M | Session 35 era. Achieved Elo 1058 in the canonical 33-player ladder (`elo_session35_exp1.json`) — the all-time pre-rewrite Elo champion in our records. |
| `ppo_s39_iter229` | `rl_v9/_init_sp_0229/snapshot_0229.pt` | MLP (legacy) | 14.3M | Session 39 PPO from new BC retrain. Recorded smart_avg 67.8% on Metamon competitive (peak of a 200-iter run). The strongest legacy-arch model by smart_avg metric. |
| `ppo_curated_iter119` | `rl_v9_curated_pool/selfplay_v9_20260501_011537/iter_0119.pt` | MLP (legacy) | 14.3M | Session 44 curated pool peak. Sustained 66.15% smart_avg (3-eval window). Most recent legacy-arch ceiling before the architectural rewrite. |

### Bots (10) — all from `eval_elo_ladder.py:ALL_BOTS`

| Canonical name | Class | Source |
|----------------|-------|--------|
| `Random` | `RandomPlayer` | poke-env built-in |
| `MaxBasePower` | `MaxBasePowerPlayer` | poke-env built-in |
| `GreedySE` | `GreedySEPlayer` | `policy_rulebots.py` |
| `HazardSense` | `HazardSensePlayer` | `policy_rulebots.py` |
| `SwitchAwareEscape` | `SwitchAwareEscapePlayer` | `policy_rulebots.py` |
| `SetupThenSweep` | `SetupThenSweepPlayer` | `policy_rulebots.py` |
| **`SH`** (anchor at Elo 1000) | `SimpleHeuristicsPlayer` | poke-env built-in |
| `SmartDmg` | `SmartDamagePlayer` | `policy_smartbots.py` |
| `Tactical` | `TacticalPlayer` | `policy_smartbots.py` |
| `Strategic` | `StrategicPlayer` | `policy_smartbots.py` |

Bots span ~75% win-rate spread per the Session 23 round-robin → ~400 Elo of dynamic range
for snapshot interpolation.

### Total roster: 15 entrants → 105 all-vs-all matchups → 10,500 games at 100 g/matchup.

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
