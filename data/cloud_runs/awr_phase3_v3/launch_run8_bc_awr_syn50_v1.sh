#!/bin/bash
# RUN #8 — BC + AWR + 50% synergistic teams (dev pod)
# Branch: master (verify HEAD at b6799a09+ which has Run #5/#6 + H2H results)
#
# One-variable test of the syn-team lever, isolating syn% from anchor question.
#
# Differs from Run #5 (BC + AWR + 30% syn) ONLY in:
#   --syn-team-pct 0.50  (was 0.30)
#   --mp-workers 70      (dev pod, was 90 on prod)
#
# Everything else identical to Run #5:
#   BC v10 init, BC anchor coef 0.1, AWR binary (mix 0.15),
#   lr=8e-5, full pool (151 anchors), MMs in PFSP (3 instances × 4 MMs),
#   200 iters, 1600 g/iter, --bf16 --tier3 --cis --pipeline (canonical Phase 2 stack).
#
# Ablation matrix after Run #8 finishes (anchored on Run #5 baseline):
#
#                                     | syn 30%  | syn 50%
#   Run #5 (BC + AWR + anchor)        | done     | RUN #8
#   Run #7 (BC + AWR + NO anchor)     | running  | (not planned — confounded)
#
#   Run #5 → Run #8 = pure syn% lever effect
#   Run #5 → Run #7 = pure anchor effect
#   Both anchored to Run #5 → clean two-axis ablation
#
# Hypothesis (PROVISIONAL):
#   - If Run #8 ≈ Run #5 plateau (smart_avg 70-75, H2H avg ~43%) → syn% NOT
#     the lever; rules out "more elite contexts breaks the ceiling" framing.
#   - If Run #8 > Run #5 by >2pp aggregate H2H → syn IS a lever; consider
#     pushing to 70% in follow-up.
#   - If Run #8 < Run #5 → syn over-rotates at 50%; learn the elbow.
#
# Pre-launch (dev pod):
#   [ ] Run #6 (phase3_bc_noawr_syn_v1) completed + freed compute
#   [ ] git fetch + checkout master, verify HEAD ≥ b6799a09
#   [ ] hl_05_26.teampack + gl_05_26.teampack present in /workspace/metamon_cache
#       (download from R2 — see Step 1 below)
#   [ ] /tmp/phase3_pool_anchors.txt present (scp from prod if needed)
#   [ ] battle servers healthy (16 BS on dev)
#   [ ] GPU clean (no zombie processes from Run #6 finalization)
#
# Step 1 (PREFERRED): Download bundles from R2 — fast, deterministic.
#
#   source /workspace/team_builder/pokemon-ai-starter/pokemon-ai/scripts/r2_env.local.sh
#   mkdir -p /workspace/metamon_cache/teams/hl_05_26 /workspace/metamon_cache/teams/gl_05_26
#   aws s3 cp s3://team-builder-data/team_bundles/hl_05_26.teampack \
#       /workspace/metamon_cache/teams/hl_05_26/gen9ou.teampack \
#       --endpoint-url $S3_ENDPOINT_URL
#   aws s3 cp s3://team-builder-data/team_bundles/gl_05_26.teampack \
#       /workspace/metamon_cache/teams/gl_05_26/gen9ou.teampack \
#       --endpoint-url $S3_ENDPOINT_URL
#
# Step 1 (FALLBACK): rebuild from raw dirs (HuggingFace download) — slower.
# train_rl.py main() auto-runs build_team_bundle if .teampack missing AND
# raw dir exists alongside.

set -u
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src

POOL_ANCHORS=$(cat /tmp/phase3_pool_anchors.txt)
RUN_TAG=phase3_bc_awr_syn50_v1
OUT_DIR=data/models/rl_v10/${RUN_TAG}
mkdir -p ${OUT_DIR}
INIT_CKPT=data/models/bc/v10_padded_for_cis_dev.pt
SERVERS="9000,9001,9002,9003,9004,9005,9006,9007,9008,9009,9010,9011,9012,9013,9014,9015"

# Synergistic team sources (same as Run #5/#6)
SYN_HL=/workspace/metamon_cache/teams/hl_05_26/gen9ou
SYN_GL=/workspace/metamon_cache/teams/gl_05_26/gen9ou

# Sanity check bundles exist
if [ ! -f "${SYN_HL}.teampack" ] || [ ! -f "${SYN_GL}.teampack" ]; then
  echo "ERROR: syn team bundles missing. Run Step 1 from header before launching."
  echo "  expected: ${SYN_HL}.teampack and ${SYN_GL}.teampack"
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
  --syn-team-pct 0.50 \
  --syn-intra-asymmetric-rate 0.30 \
  --top-asymmetric-rate 0.20 \
  --bc-anchor-ckpt ${INIT_CKPT} --bc-anchor-coef 0.1 \
  --awr-replay-memmap data/datasets/human_v8_5k \
  --awr-mix-weight 0.15 --awr-batch-size 16 --awr-binary \
  --pool-anchors "${POOL_ANCHORS}" --force-anchors ${INIT_CKPT} \
  --max-opponents-per-iter 10 \
  --external-adapters external_adapters_phase3_full_v1.yaml --n-ext-per-iter 5 \
  --cis --tier3 --tier3-minibatch-size 64 --bf16 \
  --mp-workers 70 \
  --cis-min-batch 32 --cis-timeout-ms 50 \
  --servers ${SERVERS} \
  --snapshot-interval 10 \
  --target-kl 0.03 --vf-coef 0.5 --max-grad-norm 0.5 --grad-accum 1 \
  --ent-coef 0.02 --adaptive-entropy \
  --adaptive-entropy-low 0.65 --adaptive-entropy-high 0.95 \
  --win-rate-mode ema --win-rate-ema-alpha 0.3 --win-rate-ema-window 50 \
  --early-stop --early-stop-patience 5 \
  </dev/null >/tmp/run8_bc_awr_syn50_v1.log 2>&1 &
PID=$!
disown
echo "[$(date -u +%H:%M:%S)] === RUN #8 BC + AWR + 50% SYN LAUNCHED (dev) ==="
echo "PID: ${PID}"
echo "Log: /tmp/run8_bc_awr_syn50_v1.log"
echo "Out: ${OUT_DIR}"
echo ""
echo "Differs from Run #5 only by: --syn-team-pct 0.50 (was 0.30)"
echo "AWR + BC anchor + MMs + strong SP pool all unchanged"
echo ""
echo "Syn config: 50% syn (60% hl_05_26 / 40% gl_05_26)"
echo "  intra_async=0.30, top_async=0.20"
echo "  Per-side type freq target: ~50% proc, ~30% hl, ~20% gl (vs 66/19/15 at 30%)"
echo ""
echo "200 iters @ projected ~10-12 min/iter on dev = ~36-40 hr wall"
echo "Eval (canonical default): every 10 iters vs metamon-competitive (16 teams)"
echo ""
echo "Watch for:"
echo "  - '[cis-w*] paired-pool teambuilder active' in iter 0 setup logs"
echo "  - 'TopMixer: syn_pct=0.50' (not 0.30) — verifies the lever is on"
echo "  - smart_avg trajectory vs Run #5 (compare iter 19, 49, 99, 199)"
echo "  - bc_kl trajectory (should look like Run #5 — anchor on)"
