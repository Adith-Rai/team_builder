#!/bin/bash
# RUN #6 — BC + NO AWR + 30% synergistic teams (dev pod)
# Branch: feat/hierarchical-teambuilders (must be checked out + pulled on dev)
#
# Companion to Run #5: closes the 2×2 ablation
#
#                | Procedural-only |   +30% syn        |
#   No AWR       | Run #3 (done)   |   Run #6 (THIS)   |
#   AWR          | Run #4 (done)   |   Run #5 (prod)   |
#
# Run #5 vs Run #6 isolates AWR's contribution UNDER the syn-teams condition
# (held constant). Combined with Run #4 vs Run #3 (same contrast but
# procedural-only), this tells us whether AWR's value depends on having
# elite-team contexts to support its pull.
#
# Pre-launch (dev pod):
#   [ ] Run #3 (phase3_bc_noawr_v1) completed
#   [ ] git checkout feat/hierarchical-teambuilders + pull
#   [ ] hl_05_26.teampack + gl_05_26.teampack present in /workspace/metamon_cache
#       (transferred from prod via scp — see Step 1 below)
#   [ ] battle servers healthy (16 BS on dev)
#   [ ] GPU clean
#
# Step 1: Transfer .teampack bundles from prod (run from local machine):
#   scp -P 47913 -i ~/.ssh/id_ed25519 \
#     root@195.26.233.30:/workspace/metamon_cache/teams/hl_05_26/gen9ou.teampack \
#     /tmp/hl_05_26.teampack
#   scp -P 47913 -i ~/.ssh/id_ed25519 \
#     root@195.26.233.30:/workspace/metamon_cache/teams/gl_05_26/gen9ou.teampack \
#     /tmp/gl_05_26.teampack
#   scp -P 34576 -i ~/.ssh/id_ed25519 /tmp/hl_05_26.teampack \
#     root@213.173.105.9:/workspace/metamon_cache/teams/hl_05_26/gen9ou.teampack
#   scp -P 34576 -i ~/.ssh/id_ed25519 /tmp/gl_05_26.teampack \
#     root@213.173.105.9:/workspace/metamon_cache/teams/gl_05_26/gen9ou.teampack
#   # OR: just copy as gen9ou.teampack sibling — bundle_path_for computes
#   # /workspace/metamon_cache/teams/hl_05_26/gen9ou.teampack from the dir
#   # /workspace/metamon_cache/teams/hl_05_26/gen9ou — so the .teampack lives
#   # in the parent dir.

set -u
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src

POOL_ANCHORS=$(cat /tmp/phase3_pool_anchors.txt)
RUN_TAG=phase3_bc_noawr_syn_v1
OUT_DIR=data/models/rl_v10/${RUN_TAG}
mkdir -p ${OUT_DIR}
INIT_CKPT=data/models/bc/v10_padded_for_cis_dev.pt
SERVERS="9000,9001,9002,9003,9004,9005,9006,9007,9008,9009,9010,9011,9012,9013,9014,9015"

# Synergistic team sources (same as Run #5)
SYN_HL=/workspace/metamon_cache/teams/hl_05_26/gen9ou
SYN_GL=/workspace/metamon_cache/teams/gl_05_26/gen9ou

# Sanity check bundles exist (raw dir not required — only the .teampack)
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
  --syn-team-pct 0.30 \
  --syn-intra-asymmetric-rate 0.30 \
  --top-asymmetric-rate 0.20 \
  --bc-anchor-ckpt ${INIT_CKPT} --bc-anchor-coef 0.1 \
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
  </dev/null >/tmp/run6_bc_noawr_syn_v1.log 2>&1 &
PID=$!
disown
echo "[$(date -u +%H:%M:%S)] === RUN #6 BC + NO AWR + 30% SYN LAUNCHED (dev) ==="
echo "PID: ${PID}"
echo "Log: /tmp/run6_bc_noawr_syn_v1.log"
echo "Out: ${OUT_DIR}"
echo ""
echo "Companion to Run #5 — closes 2x2 ablation"
echo "  Differs from Run #5 only by removing --awr-* flags"
echo "  Dev: 70 workers (vs prod 90) — same ratio as Run #3 vs Run #4"
echo ""
echo "Syn config: 30% syn (60% hl_05_26 / 40% gl_05_26)"
echo "  intra_async=0.30, top_async=0.20"
echo "  Per-side type freq target: 66% proc, 19% hl, 15% gl"
echo ""
echo "200 iters @ projected ~10-12 min/iter on dev = ~36-40 hr wall"
echo "Eval (canonical default): every 10 iters vs metamon-competitive (16 teams)"
