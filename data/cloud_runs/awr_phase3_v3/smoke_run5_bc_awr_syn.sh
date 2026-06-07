#!/bin/bash
# SMOKE — BC + AWR + 30% synergistic (paired-pool teambuilder)
# 5-iter smoke to validate the new hierarchical teambuilders integration
# before committing to the 200-iter Run #5.
#
# Branch: feat/hierarchical-teambuilders (must be checked out on prod)
#
# What to verify in the log:
#   [ ] "[cis-w*] paired-pool teambuilder active (syn_config=...)" appears
#       in iter 0 worker setup (confirms TopMixer constructed)
#   [ ] "PAIRED-POOL mode" banner in main process startup
#   [ ] No FileNotFoundError on syn dirs
#   [ ] No "can't start new thread" errors (thread budget under 9728)
#   [ ] AWR engages: "[AWR ] loss=... scaled=..." per iter
#   [ ] bc_kl reasonable (< 0.20 by iter 5)
#   [ ] KL well under cap (target 0.03)
#   [ ] No crashes between iters
#
# After this smoke passes, fire launch_run5_bc_awr_syn_v1.sh for the real run.

set -u
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src

POOL_ANCHORS=$(cat /tmp/phase3_pool_anchors.txt)
RUN_TAG=smoke_bc_awr_syn_v1
OUT_DIR=data/models/rl_v10/${RUN_TAG}
mkdir -p ${OUT_DIR}
INIT_CKPT=data/models/bc/v10_padded_for_cis_dev.pt
SERVERS="9000,9001,9002,9003,9004,9005,9006,9007,9008,9009,9010,9011,9012,9013,9014,9015"

SYN_HL=/workspace/metamon_cache/teams/hl_05_26/gen9ou
SYN_GL=/workspace/metamon_cache/teams/gl_05_26/gen9ou

# Sanity check syn dirs exist before launching
if [ ! -d "${SYN_HL}" ]; then
  echo "ERROR: hl team dir missing: ${SYN_HL}"
  echo "Run: METAMON_CACHE_DIR=/workspace/metamon_cache metamon_venv/bin/python -c \\"
  echo "  \"from metamon.data.download import download_teams; download_teams('gen9ou', 'hl_05_26')\""
  exit 1
fi
if [ ! -d "${SYN_GL}" ]; then
  echo "ERROR: gl team dir missing: ${SYN_GL}"
  echo "Run: METAMON_CACHE_DIR=/workspace/metamon_cache metamon_venv/bin/python -c \\"
  echo "  \"from metamon.data.download import download_teams; download_teams('gen9ou', 'gl_05_26')\""
  exit 1
fi

# Verify branch (smoke MUST be on feat branch with new flags)
BRANCH=$(cd /workspace/team_builder && git rev-parse --abbrev-ref HEAD)
if [ "${BRANCH}" != "feat/hierarchical-teambuilders" ]; then
  echo "WARNING: current branch is ${BRANCH}, not feat/hierarchical-teambuilders."
  echo "  The --syn-* flags won't exist on master and this will fail."
  echo "  Run: cd /workspace/team_builder && git checkout feat/hierarchical-teambuilders"
  exit 1
fi

setsid nohup python -u train_rl.py \
  --init-from ${INIT_CKPT} \
  --out-dir ${OUT_DIR} \
  --n-iters 5 --warmup-iters 0 \
  --games-per-iter 1600 --turn-cap 300 --lr 8e-5 \
  --lam 0.95 \
  --reward-style terminal \
  --procedural-teams /workspace/raw_data/pokemon_usage/2024-04 \
  --syn-team-dirs ${SYN_HL}:0.6,${SYN_GL}:0.4 \
  --syn-team-pct 0.30 \
  --syn-intra-asymmetric-rate 0.30 \
  --top-asymmetric-rate 0.20 \
  --bc-anchor-ckpt ${INIT_CKPT} --bc-anchor-coef 0.1 \
  --awr-replay-memmap data/datasets/human_v8_5k \
  --awr-mix-weight 0.15 --awr-batch-size 16 --awr-binary \
  --pool-anchors "${POOL_ANCHORS}" --force-anchors ${INIT_CKPT} \
  --max-opponents-per-iter 10 \
  --external-adapters external_adapters_phase3_full_v1.yaml --n-ext-per-iter 5 \
  --cis --tier3 --tier3-minibatch-size 64 --bf16 \
  --mp-workers 90 \
  --cis-min-batch 32 --cis-timeout-ms 50 \
  --servers ${SERVERS} \
  --snapshot-interval 99 --eval-interval 99 \
  --target-kl 0.03 --vf-coef 0.5 --max-grad-norm 0.5 --grad-accum 1 \
  --ent-coef 0.02 --adaptive-entropy \
  --adaptive-entropy-low 0.65 --adaptive-entropy-high 0.95 \
  </dev/null >/tmp/smoke_bc_awr_syn_v1.log 2>&1 &
PID=$!
disown
echo "[$(date -u +%H:%M:%S)] === SMOKE BC + AWR + 30% SYN LAUNCHED ==="
echo "PID: ${PID}"
echo "Log: /tmp/smoke_bc_awr_syn_v1.log"
echo "Branch: ${BRANCH}"
echo ""
echo "5 iters @ ~9 min/iter = ~45 min wall"
echo ""
echo "Watch for:"
echo "  grep 'paired-pool' /tmp/smoke_bc_awr_syn_v1.log    # paired teambuilder confirms"
echo "  grep 'PAIRED-POOL mode' /tmp/smoke_bc_awr_syn_v1.log  # main banner"
echo "  grep 'can.t start new thread' /tmp/smoke_bc_awr_syn_v1.log  # should be 0"
echo "  grep '\\[AWR ' /tmp/smoke_bc_awr_syn_v1.log         # AWR engaging"
