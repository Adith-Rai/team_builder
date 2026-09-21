# fishbowl_v2 resume recipe + the `--init-from` vs `--resume` lesson

S68 (2026-06-03). Full detail; the memory entry `project_fishbowl_v2_resume_launch.md`
keeps a 1-paragraph pointer.

## Current state

- **Launched** 2026-06-03 04:14 UTC. PID 1348650. Log `/tmp/fishbowl_v2_resume.log`.
- Out: `data/models/rl_v10/fishbowl_lr1e-4_v2_resume/selfplay_v9_20260603_041452/`.
- Tracked as task #121.
- Launch script (preserved on pod): `/tmp/launch_fishbowl_v2_resume.sh`.

## ⚠️ Mistake in this launch — used `--init-from` not `--resume` → optimizer momentum LOST

Snapshot files DO contain `optimizer_state_dict` (Adam m/v), but `--init-from` only
loads `model_state_dict`. Only `--resume` calls
`optimizer.load_state_dict(ckpt["optimizer_state_dict"])` (`train_rl.py:512`). So this
resume started with **fresh Adam momentum**, NOT "like it never stopped."

**Why I made the mistake**: `--resume snap.pt` sets `run_dir = Path(args.resume).parent`,
which would put new snapshots in the SAME dir as the source snapshot (snapshot_0159
would collide with originals snap_0149 etc). I took the easier `--init-from` path with
`--out-dir <new>` to avoid the collision without thinking through momentum.

**Impact**: PPO recovers Adam momentum in ~10-50 optimizer steps; by iter 5-10 of resume
it's fine. WR / bc_kl / smart_avg all looked normal at iter 89. Mostly OK, just not the
"exactly like never stopped" the user asked for.

## ✅ CORRECT workflow for a true "never stopped" resume

```bash
# 1. Make a new resume dir
mkdir -p data/models/rl_v10/<RUN>_resume

# 2. Copy the snapshot to the new dir (so train_rl saves new snapshots there, not in the source)
cp data/models/rl_v10/<RUN>/<TIMESTAMP>/snapshot_NNNN.pt \
   data/models/rl_v10/<RUN>_resume/<NEW_TIMESTAMP>/snapshot_NNNN.pt

# 3. Launch with --resume (NOT --init-from)
python train_rl.py \
  --resume data/models/rl_v10/<RUN>_resume/<NEW_TIMESTAMP>/snapshot_NNNN.pt \
  --pool-anchors <full pool incl in-run snapshots> \
  --warmup-iters 0 \
  --n-iters <continuation count> \
  ... (rest of original args identical) ...
```

`--resume` will:
- Load `model_state_dict` AND `optimizer_state_dict` (Adam m/v preserved)
- Read `iteration` from snapshot (correct iter counter, e.g., resumes at iter 150 not 0)
- Read `run_dir` as the new dir's parent (snapshots land in new dir, no collision)
- All scheduler / RNG / state info from ckpt

**Do NOT need `--init-from` when `--resume` is given** (per `train_rl.py:1158`: "if
init_from is None, use resume path as init source").

## "Exact same setup, like never stopped" pool reconstruction

The pool grows during training. `train_rl.py` adds saved snapshots to its own internal
pool each interval. A naive resume that only passes `--pool-anchors` with the ORIGINAL
list would start with a smaller pool than the run had — opponents would be less diverse,
distribution shift would re-warm the model in a misleading direction.

**Pool reconstruction**:
- Original 62 entries from `config.json["pool_anchors"]` in the source out-dir (split by `,`)
- + 14 in-run snapshots: `snapshot_{0019,0029,...,0149}.pt` from
  `selfplay_v9_20260601_122046/` (every snapshot_interval=10 from iter 19 → 149)
- = 76 total entries before validation
- = **75 effective** after pool-validate auto-excludes `bc_v10_cloud_e3` (pre-pad
  transformer, shape mismatch on `tokenizer.type_id_embed.weight: (28,256) vs (29,256)`)

**Why snap_0009 is NOT in the resume pool**: count is `original 62 + 14 = 76`, matching
the iter 149 log `pool=76`. snap_0009 was apparently excluded by some early-iter rule in
train_rl.py (possibly because saved during warmup window). Original training only added
snap_0019 onward.

## This launch's cmdline diffs from original (for reference)

Identical EXCEPT:
- `--init-from snap_0149.pt` (was `bc/v10_padded_for_cis_dev.pt`) ← should have been `--resume`, see above
- `--warmup-iters 0` (was `5`) — per resume convention
- `--n-iters 100` (was `150`)
- `--out-dir data/models/rl_v10/fishbowl_lr1e-4_v2_resume` — separate
- `--pool-anchors` expanded with the 14 in-run snapshots

ALL other args identical: `--lr 1e-4 --bc-anchor-coef 0.10 --vf-coef 0.5 --cis --bf16
--tier3 --tier3-minibatch-size 64 --packed --no-per-chunk-gc --cis-min-batch 16
--cis-timeout-ms 50 --mp-workers 90 --games-per-iter 1600 --max-concurrent 200
--max-opponents-per-iter 10 --force-anchors v10_padded --reward-style dense
--adaptive-entropy --adaptive-entropy-low 0.65 --adaptive-entropy-high 0.95
--snapshot-interval 10 --eval-interval 10 --eval-games 200 --eval-team-set
metamon-competitive --win-rate-mode ema --win-rate-ema-alpha 0.3 --win-rate-ema-window
50 --procedural-teams /workspace/raw_data/pokemon_usage/2024-04`

## Watch list

- **bc_kl trajectory**: prod fishbowl_prod_lr1e-4_v1 touched 0.2010 at its iter 266
  (single spike, recovered). If dev resume bc_kl climbs past 0.20 and stays there,
  that's reward-hacking signal → stop and investigate.
- **smart_avg**: universal ceiling at 70-74% confirmed across all configs. Don't expect
  to break it via more iters. The bet is on Elo (snap-snap WRs vs the cluster), not
  bot WRs.
- **Per-iter wall**: expect ~12 min/iter (matches original v2 timing). 100 iters →
  ~12-15h wall.

## Next action after this run completes

Do a PROPER `--resume` from the final snapshot of this run (`snapshot_0099.pt` in the
resume namespace, ≈ original-namespace iter 248). That snapshot will contain valid
`optimizer_state_dict`, so a `--resume`-based continuation preserves momentum from there
forward. Re-eval all snapshots with the fixed `--team-set metamon-competitive` CIS-Elo
v2 (see `FISHBOWL_V2_LADDER_RESULTS_REFUTED.md`).
