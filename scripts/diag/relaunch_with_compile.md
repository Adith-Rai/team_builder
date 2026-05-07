# Phase 1 v3 relaunch with --compile (Session 51 procedure)

How to kill the running Phase 1 v3 production run and resume it from the
latest snapshot with the Tier 1 optimizations (`--compile` per-submodule +
fused AdamW). Goal: ~10-20% per-iter speedup over the remaining iters.

**Prerequisites**: production has saved at least one snapshot in
`data/models/rl_v10/ppo_phase1_v3_resumed/selfplay_v9_*/snapshot_*.pt`.
At `--snapshot-interval 5`, that's iter 15 (or later).

**Commits required on pod**: `d5f500b8` (Tier 1) + `50d4e80a` (CIS Phases 1-3).

---

## Step 1: SSH to pod + verify state

```bash
ssh -i ~/.ssh/id_ed25519 -p 47913 root@195.26.233.30

# Confirm production is running:
ps -ef | grep train_rl | grep -v grep | head -1

# Confirm latest snapshot exists:
ls -lh /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src/data/models/rl_v10/ppo_phase1_v3_resumed/selfplay_v9_*/snapshot_*.pt | tail -1
```

If snapshot doesn't exist yet, **wait** — production hasn't hit
snapshot-interval yet.

## Step 2: Pull latest code on pod

```bash
cd /workspace/team_builder
git pull origin master
git log --oneline -2
# Expected:
#   50d4e80a CIS Phases 1-3: ...
#   d5f500b8 Tier 1 Session 51: ...
```

If you see modified `.jsonl` files (production output) blocking the pull:
```bash
git stash push --include-untracked -m "production output" -- \
  pokemon-ai-starter/pokemon-ai/src/data/eval/registry/evals.jsonl \
  pokemon-ai-starter/pokemon-ai/src/data/eval/registry/runs.jsonl
git pull origin master
git stash drop
```

## Step 3: Verify triton is matched on pod (one-time)

`--compile` requires `triton 2.2.x` to match `torch 2.2.x`. Pod was
patched in Session 51 — verify it stuck:

```bash
python -c "import triton; print('triton=', triton.__version__)"
# Expected: triton= 2.2.0
```

If anything else, run `pip install triton==2.2.0` first.

## Step 4: Identify resume snapshot + remaining iters

```bash
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
LATEST_SNAP=$(ls -t data/models/rl_v10/ppo_phase1_v3_resumed/selfplay_v9_*/snapshot_*.pt | head -1)
ITER_N=$(basename "$LATEST_SNAP" .pt | sed 's/snapshot_0*//')
N_REMAINING=$((200 - ITER_N))
echo "Resume from: $LATEST_SNAP (iter $ITER_N), $N_REMAINING iters remaining"
```

## Step 5: Stop the running production run cleanly

```bash
# Find the python PID
PYPID=$(pgrep -f "train_rl.py.*ppo_phase1_v3_resumed" | head -1)
echo "killing python PID $PYPID"

# Send SIGTERM and wait for graceful shutdown (workers cleanup)
kill -TERM $PYPID
sleep 30  # let workers cancel listeners + close ws

# Confirm dead:
ps -p $PYPID 2>/dev/null && echo "STILL ALIVE - escalate to SIGKILL" || echo "exited cleanly"
```

If `STILL ALIVE`:
```bash
kill -KILL $PYPID
sleep 5
# Clean up zombie worker processes:
pkill -KILL -f train_rl.py 2>/dev/null
```

## Step 6: Wipe stale /tmp state (optional but recommended)

```bash
# Stale weights/traj files from old run won't conflict with new iter
# numbers (resume starts at ITER_N), but removing them is hygienic:
rm -f /tmp/weights_iter*.pt /tmp/traj_w*_iter*.pkl.gz
rm -f /dev/shm/sem.mp-* /dev/shm/torch_*  # SemLock cleanup
```

Battle servers (9000-9007) keep running — they're shared infrastructure.
Verify:
```bash
ss -ltn | grep -cE ':900[0-7]'
# Expected: 8
```

## Step 7: Launch the upgraded run

```bash
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src

# In a screen so it survives ssh disconnect
screen -dmS train_compiled bash -c "
python -u train_rl.py \\
  --resume $LATEST_SNAP \\
  --pool-anchors data/models/bc/v10_cloud_gen9/epoch_003.pt \\
  --device cuda --servers 9000,9001,9002,9003,9004,9005,9006,9007 \\
  --fp16 --mp --mp-workers 8 --compile \\
  --games-per-iter 1600 --max-concurrent 200 --opponent-device cuda \\
  --n-iters $N_REMAINING --warmup-iters 0 --lr 1e-5 \\
  --lam 0.95 --ent-coef 0.02 --reward-style terminal --grad-accum 1 \\
  --adaptive-entropy --adaptive-entropy-low 0.65 --adaptive-entropy-high 0.95 \\
  --win-rate-mode ema --win-rate-ema-alpha 0.3 --win-rate-ema-window 50 \\
  --eval-interval 10 --eval-team-set metamon-competitive --eval-games 200 \\
  --snapshot-interval 5 --early-stop --early-stop-patience 3 --turn-cap 300 \\
  --procedural-teams /workspace/raw_data/pokemon_usage/2024-04 \\
  --out-dir data/models/rl_v10/ppo_phase1_v3_compiled \\
  2>&1 | tee /workspace/logs/ppo_phase1_v3_compiled.log
"
```

**Differences from production canonical**:
1. `--resume <LATEST_SNAP>` (not the smoke output)
2. `--n-iters $N_REMAINING` (200 minus iter resumed at)
3. `--out-dir` is `ppo_phase1_v3_compiled` (NOT `_resumed` to avoid collision)
4. Adds `--compile`
5. Keeps `--fp16` (NOT `--bf16`) — bf16 is a separate trial; don't change
   precision mid-run alongside compile (would confound the comparison
   against the iter 13 baseline)

## Step 8: Verify launch

```bash
sleep 10
# Should see: "torch.compile: 5/5 submodules compiled (mode=default, dynamic=True)"
# Should see: "optimizer: AdamW fused kernel enabled"
tail -F /workspace/logs/ppo_phase1_v3_compiled.log | head -30
```

Then watch for the first compiled iter completion (will take ~20 min for
collect + ~5-10 min compile trace + ~normal update):

```bash
tail -F /workspace/logs/ppo_phase1_v3_compiled.log | \
  grep --line-buffered -E "Iter [0-9]+: W/L|EVAL|smart_avg|FATAL|Snapshot|compiled|fused"
```

## Step 9: Compare iter time vs uncompiled baseline

Last 3 uncompiled iters (Phase 1 v3 original):
- Iter 11: collect=940s, update=2427s = 56 min
- Iter 12: collect=950s, update=3551s = 75 min
- Iter 13: collect=972s, update=2604s = 60 min

Compiled iter (first iter after relaunch will have +30-60s compile trace):
- Expected steady-state: collect ~800s, update ~1900-2200s = ~45-50 min
- 10-20% per-iter saving over uncompiled baseline

## Step 10: Update R2 sync if needed

The `--out-dir` changed, so the existing R2 sync loop (set up Session 50)
points at the old path. Update it:

```bash
# Find the screen with the sync loop:
screen -ls | grep r2_sync
# Reattach + update the script to point at ppo_phase1_v3_compiled, OR
# kill it and start a new one:
screen -dmS r2_sync_compiled bash -c '
source /workspace/team_builder/pokemon-ai-starter/pokemon-ai/scripts/r2_env.local.sh
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
while true; do
  aws s3 sync data/models/rl_v10/ppo_phase1_v3_compiled/ \
    s3://team-builder-data/models/rl_v10/ppo_phase1_v3_compiled/ \
    --endpoint-url $S3_ENDPOINT_URL \
    --exclude "*" \
    --include "snapshot_*.pt" --include "*.json" --include "config.json" \
    --include "win_rates.json" --include "evals.json" --include "final.pt" \
    --quiet 2>&1
  sleep 300
done
'
```

## Rollback procedure (if compiled run misbehaves)

If iter 0 of the compiled run shows NaN, divergent loss, or other
pathologies vs uncompiled baseline:

```bash
# Kill the compiled run
pkill -f "ppo_phase1_v3_compiled"
sleep 10

# Resume the un-compiled production from the same snapshot
# (drops --compile, restores --out-dir to original)
screen -dmS train_resumed_again bash -c "
python -u train_rl.py \\
  --resume $LATEST_SNAP \\
  --pool-anchors data/models/bc/v10_cloud_gen9/epoch_003.pt \\
  --device cuda --servers 9000,9001,9002,9003,9004,9005,9006,9007 \\
  --fp16 --mp --mp-workers 8 \\
  --games-per-iter 1600 --max-concurrent 200 --opponent-device cuda \\
  --n-iters $N_REMAINING --warmup-iters 0 --lr 1e-5 \\
  --lam 0.95 --ent-coef 0.02 --reward-style terminal --grad-accum 1 \\
  --adaptive-entropy --adaptive-entropy-low 0.65 --adaptive-entropy-high 0.95 \\
  --win-rate-mode ema --win-rate-ema-alpha 0.3 --win-rate-ema-window 50 \\
  --eval-interval 10 --eval-team-set metamon-competitive --eval-games 200 \\
  --snapshot-interval 5 --early-stop --early-stop-patience 3 --turn-cap 300 \\
  --procedural-teams /workspace/raw_data/pokemon_usage/2024-04 \\
  --out-dir data/models/rl_v10/ppo_phase1_v3_resumed_again \\
  2>&1 | tee /workspace/logs/ppo_phase1_v3_resumed_again.log
"
```

The snapshot is unchanged either way — rollback loses only the wall time
spent on the failed compiled iter (~50 min worst case).
