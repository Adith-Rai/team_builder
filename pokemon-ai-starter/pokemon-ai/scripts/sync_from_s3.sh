#!/usr/bin/env bash
# sync_from_s3.sh — pull the memmap from S3 to a cloud instance's local disk.
#
# Run this on the cloud A100 instance after spinning it up.
# S3 -> local NVMe in same region: ~5-10 min for 104 GB.
#
# Usage:
#   bash sync_from_s3.sh                       # uses defaults
#   DEST_DIR=/workspace/data bash sync_from_s3.sh

set -euo pipefail

S3_BUCKET="${S3_BUCKET:-team-builder-data}"
S3_PREFIX="${S3_PREFIX:-datasets/human_v8_100k}"
DEST_DIR="${DEST_DIR:-/workspace/data/datasets/human_v8_100k}"

echo "===================================================================="
echo "  S3 sync (s3://$S3_BUCKET/$S3_PREFIX -> $DEST_DIR)"
echo "===================================================================="

mkdir -p "$DEST_DIR"

# AWS CLI is usually pre-installed on RunPod templates; install if missing.
if ! command -v aws &>/dev/null; then
  echo "[sync] installing aws-cli..."
  pip install --no-cache-dir awscli
fi

# Same-region S3 to instance: 100+ MB/s typical. 50 MB chunks, 16 threads.
aws configure set default.s3.multipart_threshold 50MB
aws configure set default.s3.multipart_chunksize 50MB
aws configure set default.s3.max_concurrent_requests 16

aws s3 sync "s3://$S3_BUCKET/$S3_PREFIX/" "$DEST_DIR/" --no-progress

echo
echo "[sync] done. Local size:"
du -sh "$DEST_DIR"
