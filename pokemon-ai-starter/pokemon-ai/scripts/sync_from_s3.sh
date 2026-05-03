#!/usr/bin/env bash
# sync_from_s3.sh — pull the memmap from object storage to cloud instance NVMe.
#
# Works with AWS S3 OR Cloudflare R2. For R2:
#   export S3_ENDPOINT_URL="https://<account-id>.r2.cloudflarestorage.com"
#   export S3_BUCKET="team-builder-data"
#   bash sync_from_s3.sh
#
# Run this on the cloud A100 instance after spinning it up.
# Object storage -> local NVMe: ~5-10 min for 104 GB on a 1-10 Gbps cloud link.

set -euo pipefail

S3_BUCKET="${S3_BUCKET:-team-builder-data}"
S3_PREFIX="${S3_PREFIX:-datasets/human_v8_100k}"
DEST_DIR="${DEST_DIR:-/workspace/data/datasets/human_v8_100k}"

# Optional: S3-compatible endpoint (R2 / B2 / etc.)
ENDPOINT_FLAG=""
if [ -n "${S3_ENDPOINT_URL:-}" ]; then
  ENDPOINT_FLAG="--endpoint-url $S3_ENDPOINT_URL"
  echo "[sync] using custom S3 endpoint: $S3_ENDPOINT_URL"
fi

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

aws s3 sync "s3://$S3_BUCKET/$S3_PREFIX/" "$DEST_DIR/" $ENDPOINT_FLAG --no-progress

echo
echo "[sync] done. Local size:"
du -sh "$DEST_DIR"
