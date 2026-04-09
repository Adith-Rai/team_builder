#!/bin/bash
# Full evaluation pipeline: bot evals + h2h tournament + playstyle analysis
# Run from src/ directory

set -e

IQL_DIR="data/models/iql/v5_iql-2026-03-17_05-36-21"
BC_CKPT="data/models/bc/v5_awbc-2026-03-15_18-38-23/best.pt"
BOTS="Random,MaxBasePower,SimpleHeuristics,SmartDamage,Tactical,Strategic"
N_BOT_GAMES=30
N_H2H_GAMES=30
REPLAY_ROOT="data/replays/full_eval"
DEVICE="cpu"

# Checkpoint list: name=path
declare -a NAMES=("BC" "ep07" "ep16" "ep18" "ep22" "ep28" "ep30" "ep35" "ep40" "ep45" "ep50" "ep55" "ep60")
declare -a CKPTS=(
    "$BC_CKPT"
    "$IQL_DIR/best_policy.pt"
    "$IQL_DIR/epoch_016_policy.pt"
    "$IQL_DIR/epoch_018_policy.pt"
    "$IQL_DIR/epoch_022_policy.pt"
    "$IQL_DIR/epoch_028_policy.pt"
    "$IQL_DIR/epoch_030_policy.pt"
    "$IQL_DIR/epoch_035_policy.pt"
    "$IQL_DIR/epoch_040_policy.pt"
    "$IQL_DIR/epoch_045_policy.pt"
    "$IQL_DIR/epoch_050_policy.pt"
    "$IQL_DIR/epoch_055_policy.pt"
    "$IQL_DIR/epoch_060_policy.pt"
)

NUM=${#NAMES[@]}
echo "=============================================="
echo "  FULL EVALUATION PIPELINE"
echo "  $NUM models, $N_BOT_GAMES bot games each, $N_H2H_GAMES h2h games each"
echo "=============================================="
echo ""

# ── Phase 1: Bot Evals ──
echo "=== PHASE 1: BOT EVALS ($NUM models x 6 bots x $N_BOT_GAMES games) ==="
for i in $(seq 0 $((NUM-1))); do
    name=${NAMES[$i]}
    ckpt=${CKPTS[$i]}
    echo ""
    echo "--- [$((i+1))/$NUM] $name ---"
    python eval_bc_vs_bots.py \
        --checkpoint "$ckpt" \
        --n-battles $N_BOT_GAMES \
        --device $DEVICE \
        --save-replays \
        --replays-root "$REPLAY_ROOT/${name}" \
        --bots "$BOTS" \
        2>&1 | grep -E "\[EVAL\]|ERROR|Traceback"
    echo "  $name done."
done

echo ""
echo "=== PHASE 1 COMPLETE ==="
echo ""

# ── Phase 2: Head-to-Head Tournament ──
echo "=== PHASE 2: HEAD-TO-HEAD TOURNAMENT ==="
echo "  ${NUM} models, $(( NUM * (NUM-1) / 2 )) matchups, $N_H2H_GAMES games each"

# Build checkpoint and name args
CKPT_ARGS=""
NAME_ARGS=""
for i in $(seq 0 $((NUM-1))); do
    CKPT_ARGS="$CKPT_ARGS ${CKPTS[$i]}"
    NAME_ARGS="$NAME_ARGS ${NAMES[$i]}"
done

python eval_head_to_head.py \
    --checkpoints $CKPT_ARGS \
    --names $NAME_ARGS \
    --n-battles $N_H2H_GAMES \
    --device $DEVICE \
    --save-replays \
    --replay-root "$REPLAY_ROOT/h2h" \
    --out-json "$REPLAY_ROOT/h2h_results.json" \
    2>&1

echo ""
echo "=== PHASE 2 COMPLETE ==="
echo ""

# ── Phase 3: Playstyle Analysis ──
echo "=== PHASE 3: PLAYSTYLE ANALYSIS ==="
for i in $(seq 0 $((NUM-1))); do
    name=${NAMES[$i]}
    replay_dir="$REPLAY_ROOT/${name}"
    if [ -d "$replay_dir" ]; then
        echo ""
        echo "--- PLAYSTYLE: $name ---"
        python analyze_eval.py --replay-dir "$replay_dir" 2>&1
    fi
done

echo ""
echo "=== ALL PHASES COMPLETE ==="
echo "Results in $REPLAY_ROOT/"
