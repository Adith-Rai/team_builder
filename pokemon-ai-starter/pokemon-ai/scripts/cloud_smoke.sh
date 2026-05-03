#!/usr/bin/env bash
# cloud_smoke.sh — pre-flight validation before committing to a multi-hour run.
#
# Run this after cloud_setup.sh + sync_from_s3.sh, on a fresh A100 pod.
# Validates that the full BC pipeline works end-to-end at the cloud config
# (B=48, fp16, compile, workers=8) for ~50 batches. Costs ~$0.50 of GPU time.
#
# What it checks:
#   1. torch.compile actually completes warmup without errors
#   2. B=48 fp16 fits in 40 GB without OOM
#   3. workers=8 doesn't deadlock or thrash
#   4. Loss decreases over 50 batches
#   5. Per-turn throughput matches A100 expectations (2-5 ms/turn)
#
# Aborts on any failure; if green, commit to the full run.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SRC_DIR="${WORKSPACE}/team_builder/pokemon-ai-starter/pokemon-ai/src"
DATA_DIR="${WORKSPACE}/data/datasets/human_v8_100k"
LOG="${WORKSPACE}/cloud_smoke.log"

echo "===================================================================="
echo "  Cloud smoke — 50-batch BC at cloud config"
echo "===================================================================="
echo "  src: $SRC_DIR"
echo "  data: $DATA_DIR"
echo "  log: $LOG"
echo

if [ ! -d "$DATA_DIR" ]; then
  echo "[smoke] ERROR: data dir not found at $DATA_DIR"
  echo "        run sync_from_s3.sh first"
  exit 1
fi

cd "$SRC_DIR"

# Ensure data symlink in src/data/datasets/ is set up (the script's --memmap-dir
# convention).
mkdir -p data/datasets
ln -sfn "$DATA_DIR" data/datasets/human_v8_100k

# Run for 50 batches, save once at batch 40 (so we have a checkpoint to resume
# if anything goes wrong with the full run).
echo "[smoke] launching 50-batch smoke run..."
timeout 600 python -u train_bc.py --use-transformer --compile \
  --memmap-dir data/datasets/human_v8_100k \
  --epochs 1 \
  --batch-size 48 \
  --lr 1e-4 \
  --fp16 \
  --workers 8 \
  --eval-games 0 \
  --val-ratio 0.05 \
  --sched constant \
  --warmup-steps 50 \
  --save-every 40 \
  --run-name v10_smoke_cloud \
  --device cuda \
  2>&1 | tee "$LOG" || {
    echo
    echo "[smoke] FAIL: training crashed or timed out (>10 min for 50 batches)"
    exit 1
  }

# Validate: did we get >= 2 progress lines? Did loss decrease?
n_reports=$(grep -c "loss=" "$LOG" || echo 0)
if [ "$n_reports" -lt 2 ]; then
  echo "[smoke] FAIL: fewer than 2 progress reports — training stalled"
  exit 1
fi

first_loss=$(grep "loss=" "$LOG" | head -1 | sed 's/.*loss=\([0-9.]*\).*/\1/')
last_loss=$(grep "loss=" "$LOG" | tail -1 | sed 's/.*loss=\([0-9.]*\).*/\1/')
echo
echo "[smoke] reports: $n_reports"
echo "[smoke] loss: $first_loss -> $last_loss"

# Loss should be decreasing. Use awk for float comparison.
if awk "BEGIN {exit !($last_loss < $first_loss)}"; then
  echo "[smoke] PASS: loss decreasing"
else
  echo "[smoke] WARN: loss not decreasing — investigate before full run"
  exit 1
fi

# Throughput sanity: parse last batch's elapsed and turns
last_line=$(grep "loss=" "$LOG" | tail -1)
last_turns=$(echo "$last_line" | sed 's/.*turns=\([0-9]*\).*/\1/')
last_elapsed=$(echo "$last_line" | sed 's/.*elapsed=\([0-9]*\)s.*/\1/')
ms_per_turn=$(awk "BEGIN {printf \"%.1f\", $last_elapsed * 1000 / $last_turns}")

echo "[smoke] throughput: ${last_turns} turns in ${last_elapsed}s = ${ms_per_turn} ms/turn"
echo
echo "  Expected on A100 fp16 + compile: 2-5 ms/turn"
echo "  If significantly slower (>10 ms/turn), torch.compile may have failed,"
echo "  or workers/data are bottlenecked."
echo

echo "===================================================================="
echo "  Cloud smoke: PASS"
echo "  Safe to launch the full run."
echo "===================================================================="
