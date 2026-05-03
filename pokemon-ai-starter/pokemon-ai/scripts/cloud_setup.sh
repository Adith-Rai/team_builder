#!/usr/bin/env bash
# cloud_setup.sh — RunPod / Lambda / generic Linux + A100 startup script.
#
# Run this once on a fresh A100 instance. Prereqs:
#   - PyTorch 2.x + CUDA 12.1 base image (RunPod template "PyTorch 2.x" works)
#   - Network volume / persistent dir mounted at $WORKSPACE (typically /workspace)
#   - The 104 GB human_v8_100k memmap already at $WORKSPACE/data/datasets/human_v8_100k
#     (or downloadable from S3 — see "DATA SYNC" below)
#
# What it does:
#   1. Clones (or updates) the repo
#   2. Installs Python deps
#   3. Verifies tokenizer + policy tests still pass
#   4. Quick throughput bench at B=32 to confirm A100 is healthy
#   5. (Optional) Smoke train_bc for 50 batches to validate the full pipeline

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/Adith-Rai/team_builder.git}"  # update if repo moves
REPO_DIR="${REPO_DIR:-$WORKSPACE/team_builder}"
SRC_DIR="$REPO_DIR/pokemon-ai-starter/pokemon-ai/src"
BRANCH="${BRANCH:-master}"

echo "===================================================================="
echo "  Cloud setup — RunPod / Linux + A100"
echo "===================================================================="
echo "Workspace: $WORKSPACE"
echo "Repo dir:  $REPO_DIR"
echo "Branch:    $BRANCH"
echo

# --- 1. Repo ---
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "[setup] cloning repo..."
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
  echo "[setup] repo present; pulling latest..."
  cd "$REPO_DIR"
  git fetch origin "$BRANCH"
  git reset --hard "origin/$BRANCH"
fi

# --- 2. Deps ---
echo
echo "[setup] installing Python deps (assuming PyTorch is pre-baked)..."
cd "$REPO_DIR/pokemon-ai-starter/pokemon-ai"
pip install --no-cache-dir -r requirements.txt --no-deps
pip install --no-cache-dir einops gin-config orjson pyarrow PyYAML awscli

# --- 2b. R2 / S3 credentials ---
# If r2_env.local.sh is present, source it. Otherwise expect AWS_ACCESS_KEY_ID
# / AWS_SECRET_ACCESS_KEY / S3_ENDPOINT_URL / S3_BUCKET to be set externally.
SCRIPT_DIR="$REPO_DIR/pokemon-ai-starter/pokemon-ai/scripts"
if [ -f "$SCRIPT_DIR/r2_env.local.sh" ]; then
  echo "[setup] sourcing R2 env from r2_env.local.sh"
  source "$SCRIPT_DIR/r2_env.local.sh"
else
  echo "[setup] WARN: no r2_env.local.sh — set AWS_ACCESS_KEY_ID/SECRET/S3_ENDPOINT_URL manually"
fi

# --- 3. Sanity: GPU + CUDA ---
echo
echo "[setup] GPU + CUDA check..."
python -c "
import torch
print(f'  torch: {torch.__version__}')
print(f'  cuda available: {torch.cuda.is_available()}')
print(f'  device: {torch.cuda.get_device_name(0)}')
print(f'  vram: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

# --- 4. Verify tests still pass ---
cd "$SRC_DIR"
echo
echo "[setup] running test suites..."
python verify_move_lookup.py | tail -5
python test_tokenizer.py | tail -5
python test_policy.py | tail -5

# --- 5. Throughput bench (skip if no memmap yet) ---
DATA_DIR="${WORKSPACE}/data/datasets/human_v8_100k"
if [ -d "$DATA_DIR" ]; then
  echo
  echo "[setup] running throughput bench at B=32 fp16 (5 batches)..."
  cd "$SRC_DIR"
  ln -sfn "$DATA_DIR" data/datasets/human_v8_100k 2>/dev/null || true
  python bench_bc_step.py --batch-size 32 --n-batches 5 --device cuda --fp16 | tee /tmp/bench.log
  echo
  echo "  Expected on A100 fp16: 2-5 ms/turn, peak_mem 4-8 GB"
else
  echo
  echo "[setup] data dir not found at $DATA_DIR — skipping bench"
  echo "        sync the memmap first (see DATA SYNC section in CLOUD_RUNBOOK.md)"
fi

echo
echo "===================================================================="
echo "  Setup complete."
echo
echo "  To start BC training (after data sync):"
echo
echo "    cd $SRC_DIR"
echo "    python -u train_bc.py --use-transformer --compile \\"
echo "      --memmap-dir data/datasets/human_v8_100k \\"
echo "      --epochs 50 --batch-size 48 --lr 1e-4 --fp16 \\"
echo "      --workers 8 --eval-games 0 --val-ratio 0.1 \\"
echo "      --sched constant --warmup-steps 200 \\"
echo "      --early-stop-patience 2 \\"
echo "      --save-every 1000 --run-name v10_cloud_gen9 \\"
echo "      --device cuda 2>&1 | tee $WORKSPACE/v10_cloud_gen9.log"
echo "===================================================================="
