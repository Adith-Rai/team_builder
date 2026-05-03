#!/usr/bin/env bash
# sync_to_s3.sh — upload the local memmap dataset to S3 so cloud A100 instances
# can sync it back in minutes instead of paying GPU time during a 24-hr upload.
#
# Run this on the LOCAL laptop (or a CPU box near the data).
# Prereqs:
#   - aws CLI installed and configured (`aws configure`)
#   - S3 bucket created (e.g. s3://team-builder-data/)
#   - AWS_REGION set (use the same region as your A100 instance to avoid egress fees)
#
# Usage:
#   bash sync_to_s3.sh                       # uses defaults
#   S3_BUCKET=my-bucket bash sync_to_s3.sh   # override bucket
#
# Cost: ~$0.023/GB/mo storage. 104 GB ≈ $2.40/mo.
# Same-region egress to A100 is free. Different region: $0.02/GB ($2 per sync).

set -euo pipefail

S3_BUCKET="${S3_BUCKET:-team-builder-data}"
S3_PREFIX="${S3_PREFIX:-datasets/human_v8_100k}"
LOCAL_DIR="${LOCAL_DIR:-pokemon-ai-starter/pokemon-ai/src/data/datasets/human_v8_100k}"

echo "===================================================================="
echo "  S3 sync (local -> $S3_BUCKET/$S3_PREFIX)"
echo "===================================================================="
echo "  local dir: $LOCAL_DIR"
echo "  total size:"
du -sh "$LOCAL_DIR" 2>/dev/null || true
echo

# Multipart, parallel upload. The 50 MB chunk size + 16-thread setup is
# tuned for residential 10-50 Mbps upload — small chunks mean minimal
# rework on dropped connections. On gigabit, bump multipart_chunksize to 200MB.
aws configure set default.s3.multipart_threshold 50MB
aws configure set default.s3.multipart_chunksize 50MB
aws configure set default.s3.max_concurrent_requests 16

aws s3 sync "$LOCAL_DIR" "s3://$S3_BUCKET/$S3_PREFIX/" \
  --storage-class STANDARD \
  --no-progress

echo
echo "[sync] done."
echo "[sync] verify on cloud:"
echo "  aws s3 ls s3://$S3_BUCKET/$S3_PREFIX/"
