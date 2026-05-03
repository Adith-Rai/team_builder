#!/usr/bin/env bash
# sync_to_s3.sh — upload the local memmap dataset to object storage so cloud
# A100 instances can sync it back in minutes instead of paying GPU time during
# a 24-hr upload.
#
# Works with AWS S3 OR Cloudflare R2 (S3-compatible). For R2:
#   export S3_ENDPOINT_URL="https://<account-id>.r2.cloudflarestorage.com"
#   export S3_BUCKET="team-builder-data"
#   bash sync_to_s3.sh
#
# Run this on the LOCAL laptop (or a CPU box near the data).
# Prereqs:
#   - aws CLI installed and configured (`aws configure`)
#     For R2: configure with the R2 access key + secret (in `aws configure`)
#   - Bucket created
#
# Cost (Cloudflare R2):
#   $0.015/GB/mo storage; 104 GB ≈ $1.56/mo. ZERO egress fees ever.
# Cost (AWS S3):
#   $0.023/GB/mo storage; same-region egress free, cross-region $0.02/GB.

set -euo pipefail

S3_BUCKET="${S3_BUCKET:-team-builder-data}"
S3_PREFIX="${S3_PREFIX:-datasets/human_v8_100k}"
LOCAL_DIR="${LOCAL_DIR:-pokemon-ai-starter/pokemon-ai/src/data/datasets/human_v8_100k}"

# Optional: S3-compatible endpoint (set this for Cloudflare R2 / Backblaze B2 / etc.)
ENDPOINT_FLAG=""
if [ -n "${S3_ENDPOINT_URL:-}" ]; then
  ENDPOINT_FLAG="--endpoint-url $S3_ENDPOINT_URL"
  echo "[sync] using custom S3 endpoint: $S3_ENDPOINT_URL"
fi

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
  $ENDPOINT_FLAG \
  --no-progress

echo
echo "[sync] done."
echo "[sync] verify on cloud:"
echo "  aws s3 ls s3://$S3_BUCKET/$S3_PREFIX/"
