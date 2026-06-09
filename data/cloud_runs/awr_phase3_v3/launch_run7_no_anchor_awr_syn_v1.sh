#!/bin/bash
# RUN #7 — NO BC ANCHOR + AWR + 30% synergistic teams (prod pod)
# Branch: master (Run #5/#6 merged; verify HEAD at f421b7c7+)
#
# Purpose: directly tests the user's hypothesis that BC anchor is the
# unrefuted common variable across the 5 RL configurations that all
# plateaued at smart_avg ~70-75. Remove the anchor, see if policy can
# diverge further productively.
#
# Differs from Run #5 (BC + AWR + syn) ONLY in:
#   - bc_anchor_coef removed (--bc-anchor-coef 0.0)
#   - --bc-anchor-ckpt also removed (no anchor model loaded at all)
#
# Everything else identical: AWR binary (mix=0.15), syn 30% (hl@0.6/gl@0.4),
# lr=8e-5, full pool (151 anchors), externals (3 MMs + minikazam + 2 MCTS),
# 200 iters.
#
# Safety mechanisms still in place (vs Phase 1 v3 collapse era):
#   - AWR rehearsal: pulls toward BC's winning plays (sparse but present)
#   - Syn teams (30%): elite contexts give positive PPO reward when elite
#     plays land
#   - MMs in PFSP pool (3 instances each of LargeRL/MediumRL_Aug/SynthRLV2/
#     Minikazam): WR vs them would crash if model degrades
#   - Strong prior SP-pool anchors (lr8e5_v1_flash snap_0139 = 1178 Elo etc):
#     WR vs them would crash if model degrades
#   - Watchdog (mm_watchdog.py): catches MM hangs at 30s
#
# Monitor manually (NO hard kill rules per user — eyeball at iter boundaries):
#   - bc_kl rising smoothly past ~0.3 → eye-watching territory
#   - bc_kl spiking to >0.5 → consider manual kill
#   - smart_avg dropping below ~65 → check carefully
#   - WR vs snap_0139 or lr8e5 family dropping below ~30% → degrading
#   - MM WR (mm-largerl) dropping below ~5% (vs Run #5's ~15-18%) → bad

set -u
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src

POOL_ANCHORS=$(cat /tmp/phase3_pool_anchors.txt)
RUN_TAG=phase3_no_anchor_awr_syn_v1
OUT_DIR=data/models/rl_v10/${RUN_TAG}
mkdir -p ${OUT_DIR}
INIT_CKPT=data/models/bc/v10_padded_for_cis_dev.pt
SERVERS="9000,9001,9002,9003,9004,9005,9006,9007,9008,9009,9010,9011,9012,9013,9014,9015"

# Synergistic team sources (same as Run #5)
SYN_HL=/workspace/metamon_cache/teams/hl_05_26/gen9ou
SYN_GL=/workspace/metamon_cache/teams/gl_05_26/gen9ou

# Sanity check bundles + raw dirs (mmap path needs .teampack files alongside dirs)
if [ ! -f "${SYN_HL}.teampack" ] || [ ! -f "${SYN_GL}.teampack" ]; then
  echo "ERROR: syn team bundles missing. Run R2 download from CLOUD_RUNBOOK.md."
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
  </dev/null >/tmp/run7_no_anchor_awr_syn_v1.log 2>&1 &
PID=$!
disown
echo "[$(date -u +%H:%M:%S)] === RUN #7 NO ANCHOR + AWR + 30% SYN LAUNCHED ==="
echo "PID: ${PID}"
echo "Log: /tmp/run7_no_anchor_awr_syn_v1.log"
echo "Out: ${OUT_DIR}"
echo ""
echo "Differs from Run #5 only by: NO --bc-anchor-ckpt and NO --bc-anchor-coef"
echo "AWR + syn-teams + MMs + strong SP pool are the grounding mechanisms"
echo ""
echo "200 iters @ projected ~10 min/iter = ~33 hr wall"
echo "Eval (canonical default): every 10 iters vs metamon-competitive (16 teams)"
echo ""
echo "Watch bc_kl in train log. If it climbs past ~0.5, consider manual kill."
echo "Watch SP-pool WR vs strong snaps (snap_0139, lr8e5_v1_flash) — drops below 30% = degrading."
