#!/bin/bash
# RUN #8 — BC + NO AWR + 50% synergistic teams (dev pod)
# Branch: master (verify HEAD at b2fbc88c+)
#
# REVISED design (2026-06-09 evening, after user pushback):
# Original Run #8 v1 was "Run #5 + syn 50%" (with AWR). Killed at iter 0
# because user correctly observed Run #6 (no AWR) was our better-performing
# baseline — iterating from the worse arm wasn't smart. v2 builds on Run #6.
#
# Differs from Run #6 (BC + NO AWR + 30% syn) ONLY in:
#   --syn-team-pct 0.50  (was 0.30)
#
# Everything else identical to Run #6:
#   BC v10 init, BC anchor coef 0.1, NO AWR, lr=8e-5, full pool (151 anchors
#   filed; dev resolves to ~30-46 valid due to missing snapshot files),
#   200 iters, 1600 g/iter, --bf16 --tier3 --cis --pipeline (canonical Phase
#   2 stack), 4 MMs × 3 instances, 70 workers.
#
# One-variable change from Run #6: syn% 30 → 50.
#
# Tests:
#   - If Run #8 > Run #6 (smart_avg peak, per-opp slopes, H2H) → syn% IS a
#     lever when not fighting AWR's BC-pull. Especially: SP-pool gains beyond
#     Run #6's +5-20pp dominance over BC-style snaps.
#   - If Run #8 ≈ Run #6 → syn% NOT a lever even from cleaner baseline.
#     Strongest evidence yet for "BC v10 / arch is the ceiling, not the syn
#     diversity knob."
#   - If Run #8 < Run #6 → syn over-rotates at 50% without AWR's anchor pull
#     to BC-style winning plays.
#
# Pre-launch (dev pod):
#   [ ] Run #6 (phase3_bc_noawr_syn_v1) completed + freed compute
#   [ ] git fetch + checkout master, verify HEAD ≥ b2fbc88c
#   [ ] hl_05_26.teampack + gl_05_26.teampack present in /workspace/metamon_cache
#   [ ] /tmp/phase3_pool_anchors.txt present
#   [ ] battle servers healthy (16 BS on dev)
#   [ ] GPU clean (no zombie processes from killed Run #8 v1)
#   [ ] Run #8 v1 outdir removed (phase3_bc_awr_syn50_v1)
#
# Note: Run #6 already had NO AWR so the AWR memmap is NOT required for this
# variant. (Run #8 v1 needed it because v1 had AWR on; v2 doesn't.)

set -u
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src

POOL_ANCHORS=$(cat /tmp/phase3_pool_anchors.txt)
RUN_TAG=phase3_bc_noawr_syn50_v1
OUT_DIR=data/models/rl_v10/${RUN_TAG}
mkdir -p ${OUT_DIR}
INIT_CKPT=data/models/bc/v10_padded_for_cis_dev.pt
SERVERS="9000,9001,9002,9003,9004,9005,9006,9007,9008,9009,9010,9011,9012,9013,9014,9015"

# Synergistic team sources (same as Run #5/#6)
SYN_HL=/workspace/metamon_cache/teams/hl_05_26/gen9ou
SYN_GL=/workspace/metamon_cache/teams/gl_05_26/gen9ou

# Sanity check bundles exist
if [ ! -f "${SYN_HL}.teampack" ] || [ ! -f "${SYN_GL}.teampack" ]; then
  echo "ERROR: syn team bundles missing. Download from R2 first."
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
  </dev/null >/tmp/run8_bc_noawr_syn50_v1.log 2>&1 &
PID=$!
disown
echo "[$(date -u +%H:%M:%S)] === RUN #8 BC + NO AWR + 50% SYN LAUNCHED (dev) ==="
echo "PID: ${PID}"
echo "Log: /tmp/run8_bc_noawr_syn50_v1.log"
echo "Out: ${OUT_DIR}"
echo ""
echo "Differs from Run #6 only by: --syn-team-pct 0.50 (was 0.30)"
echo "NO AWR (intentional — Run #6 was better baseline than Run #5)"
echo "BC anchor still on (coef=0.1)"
echo ""
echo "Syn config: 50% syn (60% hl_05_26 / 40% gl_05_26)"
echo "  intra_async=0.30, top_async=0.20"
echo "  Per-side type freq target: ~50% proc, ~30% hl, ~20% gl (vs 66/19/15 at 30%)"
echo ""
echo "200 iters @ projected ~10-12 min/iter on dev = ~36-40 hr wall"
echo "Eval (canonical default): every 10 iters vs metamon-competitive (16 teams)"
echo ""
echo "Watch for:"
echo "  - 'TopMixer: syn_pct=0.50' — verifies the syn lever is on"
echo "  - 'BCAnchor: ON (coef=0.1)' — anchor still on"
echo "  - No 'AWR' log lines — AWR off"
echo "  - smart_avg trajectory vs Run #6 (compare iter 19, 49, 99, 199)"
