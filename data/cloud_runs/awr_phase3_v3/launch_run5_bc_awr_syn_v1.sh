#!/bin/bash
# RUN #5 — BC + AWR + 30% synergistic teams (hierarchical teambuilders)
# Branch: feat/hierarchical-teambuilders (must be checked out + pulled on prod)
#
# Tests the "AWR + synergistic team context" combination — the next iteration
# of the Phase 2 hypothesis from PLATEAU_HYPOTHESIS_AND_EXPERIMENTS.md.
#
# Run #4 (BC + AWR alone) showed AWR provides ~0 measurable lift on smart_avg
# vs Run #3 (BC + no AWR). User's tug-of-war hypothesis: AWR pulls policy
# toward elite plays, but PPO un-reinforces because procedural teams don't
# support those plays → tug-of-war, net ~0 lift.
#
# This run adds the missing piece: 30% of training games use synergistic teams
# from hl_05_26 (high ladder, 1400+/1600+ Elo) and gl_05_26 (general ladder)
# — matched-pool per battle via paired QueueTeambuilders. PPO now has team
# contexts where elite plays actually work → can reward AWR's pull → loop
# closes.
#
# Distribution under defaults:
#   - 56% proc-vs-proc (both sides procedural)
#   - 14.4% syn-vs-syn matched type (8.6% hl-hl + 5.8% gl-gl)
#   - 9.6% hl-vs-gl intra-syn cross
#   - 20% proc-vs-syn cross-quality
# → Per-side type freq: 66% procedural, 19% hl, 15% gl
#
# Pre-launch checklist:
#   [ ] hl_05_26.teampack + gl_05_26.teampack downloaded from R2 (see Step 1)
#   [ ] git checkout feat/hierarchical-teambuilders + verify on prod
#   [ ] battle servers healthy (16 BS on prod)
#   [ ] GPU clean (no zombie from prior runs)
#
# Step 1 (PREFERRED): Download bundles from R2 (~3 min, 256 MB total):
#   source /workspace/team_builder/pokemon-ai-starter/pokemon-ai/scripts/r2_env.local.sh
#   mkdir -p /workspace/metamon_cache/teams/hl_05_26 /workspace/metamon_cache/teams/gl_05_26
#   aws s3 cp s3://team-builder-data/team_bundles/hl_05_26.teampack \
#       /workspace/metamon_cache/teams/hl_05_26/gen9ou.teampack \
#       --endpoint-url $S3_ENDPOINT_URL
#   aws s3 cp s3://team-builder-data/team_bundles/gl_05_26.teampack \
#       /workspace/metamon_cache/teams/gl_05_26/gen9ou.teampack \
#       --endpoint-url $S3_ENDPOINT_URL
#
# Step 1 (FALLBACK): Rebuild bundles from raw HuggingFace download
# (~30 min, ~750 MB extracted). Only needed if R2 unavailable:
#   cd /workspace
#   METAMON_CACHE_DIR=/workspace/metamon_cache \
#     metamon_venv/bin/python -c "
# from metamon.data.download import download_teams
# download_teams('gen9ou', 'hl_05_26')
# download_teams('gen9ou', 'gl_05_26')
# "
#   # train_rl.py main() auto-runs build_team_bundle on missing .teampack
#   # if the raw dir exists alongside.

set -u
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src

POOL_ANCHORS=$(cat /tmp/phase3_pool_anchors.txt)
RUN_TAG=phase3_bc_awr_syn_v1
OUT_DIR=data/models/rl_v10/${RUN_TAG}
mkdir -p ${OUT_DIR}
INIT_CKPT=data/models/bc/v10_padded_for_cis_dev.pt
SERVERS="9000,9001,9002,9003,9004,9005,9006,9007,9008,9009,9010,9011,9012,9013,9014,9015"

# Synergistic team sources
SYN_HL=/workspace/metamon_cache/teams/hl_05_26/gen9ou
SYN_GL=/workspace/metamon_cache/teams/gl_05_26/gen9ou

# Sanity check syn dirs exist before launching
if [ ! -d "${SYN_HL}" ] || [ ! -d "${SYN_GL}" ]; then
  echo "ERROR: syn team dirs missing. Run Step 1 from header before launching."
  echo "  expected: ${SYN_HL}/ and ${SYN_GL}/"
  exit 1
fi

setsid nohup python -u train_rl.py \
  --init-from ${INIT_CKPT} \
  --out-dir ${OUT_DIR} \
  --n-iters 200 --warmup-iters 5 \
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
  --snapshot-interval 10 \
  --target-kl 0.03 --vf-coef 0.5 --max-grad-norm 0.5 --grad-accum 1 \
  --ent-coef 0.02 --adaptive-entropy \
  --adaptive-entropy-low 0.65 --adaptive-entropy-high 0.95 \
  --win-rate-mode ema --win-rate-ema-alpha 0.3 --win-rate-ema-window 50 \
  --early-stop --early-stop-patience 5 \
  </dev/null >/tmp/run5_bc_awr_syn_v1.log 2>&1 &
PID=$!
disown
echo "[$(date -u +%H:%M:%S)] === RUN #5 BC + AWR + 30% SYN LAUNCHED ==="
echo "PID: ${PID}"
echo "Log: /tmp/run5_bc_awr_syn_v1.log"
echo "Out: ${OUT_DIR}"
echo "Branch: feat/hierarchical-teambuilders (verify with: cd /workspace/team_builder && git log -1 --oneline)"
echo ""
echo "Syn config: 30% syn (60% hl_05_26 / 40% gl_05_26)"
echo "  intra_async=0.30, top_async=0.20"
echo "  Per-side type freq target: 66% proc, 19% hl, 15% gl"
echo ""
echo "200 iters @ projected ~10 min/iter = ~33 hr wall"
echo "Eval (canonical default): every 10 iters vs metamon-competitive (16 teams)"
echo ""
echo "Watch for: '[cis-w*] paired-pool teambuilder active' in iter 0 setup logs"
