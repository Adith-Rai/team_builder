#!/bin/bash
# RUN #7 RESUME from snapshot_0059.pt (post early-stop continuation)
#
# Context: Run #7 (BC + no anchor + AWR + 30% syn) early-stopped at iter 59
# due to --early-stop catching 5 consecutive smart_avg regressions. Decision-
# pattern analysis showed model was MID-TRANSITION (setup-spam halved 14% -> 7%,
# decision style genuinely shifting to direct damage + sparse switching).
# Resuming to let the transition complete and see if a new equilibrium emerges.
#
# Resume mechanics (verified in train_rl.py lines 574-645):
#   --resume <snapshot>: loads model_state_dict + optimizer_state_dict
#     (full Adam momentum preserved; no --lr-restart used)
#   start_iter = ckpt['iteration'] + 1 = 60
#   --n-iters 200 means 200 ITERS FROM start_iter (line 1796)
#     → iters 60-259, 200 more iters total
#   run_dir = parent of resume path → same out dir, snapshots append
#   Pool: ckpt's saved pool + disk-scanned snapshots from run_dir → dedupe
#     ensures all 6 Run #7 saved snaps (snapshot_0009..0059) are in pool
#
# Critical differences from original Run #7 launch:
#   1. --resume snapshot_0059.pt  (vs --init-from ... cold start)
#   2. NO --early-stop / --early-stop-patience  (user explicit: let it run)
#   3. --n-iters 200  (= 200 MORE, total run becomes ~260 iters)
#
# Everything else IDENTICAL to launch_run7_no_anchor_awr_syn_v1.sh:
#   - Same INIT_CKPT (used as init source per code requirement)
#   - Same POOL_ANCHORS
#   - Same syn config (30% / 60-40 split / 0.30 / 0.20 asym rates)
#   - Same AWR config (mix 0.15, binary, batch 16)
#   - Same compute (--cis --tier3 --bf16 --tier3-minibatch-size 64)
#   - Same mp-workers 90, cis-min-batch 32, cis-timeout-ms 50
#   - Same KL / entropy / adaptive / win-rate config

set -u
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src

POOL_ANCHORS=$(cat /tmp/phase3_pool_anchors.txt)
RUN_TAG=phase3_no_anchor_awr_syn_v1
OUT_DIR=data/models/rl_v10/${RUN_TAG}
INIT_CKPT=data/models/bc/v10_padded_for_cis_dev.pt
RESUME_CKPT=${OUT_DIR}/selfplay_v9_20260609_161129/snapshot_0059.pt
SERVERS="9000,9001,9002,9003,9004,9005,9006,9007,9008,9009,9010,9011,9012,9013,9014,9015"

# Synergistic team sources (same as Run #5/6/7)
SYN_HL=/workspace/metamon_cache/teams/hl_05_26/gen9ou
SYN_GL=/workspace/metamon_cache/teams/gl_05_26/gen9ou

# Sanity check resume checkpoint exists
if [ ! -f "${RESUME_CKPT}" ]; then
  echo "ERROR: resume checkpoint missing: ${RESUME_CKPT}"
  exit 1
fi

# Sanity check bundles still present
if [ ! -f "${SYN_HL}.teampack" ] || [ ! -f "${SYN_GL}.teampack" ]; then
  echo "ERROR: syn team bundles missing."
  echo "  expected: ${SYN_HL}.teampack and ${SYN_GL}.teampack"
  exit 1
fi

setsid nohup python -u train_rl.py \
  --init-from ${INIT_CKPT} \
  --resume ${RESUME_CKPT} \
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
  </dev/null >/tmp/run7_resume_from_iter59.log 2>&1 &
PID=$!
disown
echo "[$(date -u +%H:%M:%S)] === RUN #7 RESUMED from snapshot_0059 ==="
echo "PID: ${PID}"
echo "Log: /tmp/run7_resume_from_iter59.log"
echo "Out: ${OUT_DIR}/selfplay_v9_20260609_161129 (appends to existing run dir)"
echo ""
echo "Resume: ${RESUME_CKPT}"
echo "Loads model_state_dict + optimizer_state_dict (Adam momentum preserved)"
echo "start_iter = 60, runs 200 more iters -> ends at iter 259"
echo ""
echo "Differences from original Run #7:"
echo "  - Resumes from snapshot_0059 (vs cold start)"
echo "  - NO --early-stop (user explicit: let transition complete)"
echo "  - NO --early-stop-patience"
echo ""
echo "200 more iters @ ~10-11 min/iter = ~33-37 hr wall"
echo ""
echo "Watch for:"
echo "  - 'Resumed from .../snapshot_0059.pt, starting at iter 60, pool: ~150 checkpoints'"
echo "  - smart_avg trajectory iter 69, 79, 89, 99 — does it bottom + recover?"
echo "  - snapshot_0209 slope (currently -7.79/10it) — does it flatten/reverse?"
echo "  - setup % staying around 7% (vs return to 14% setup-spam)"
