#!/usr/bin/env bash
# launch_phase2_with_oom_monitor.sh
#
# Production launcher for Phase 2 with OOM detect-and-notify (NO auto-fallback).
#
# Why no auto-fallback (S67 decision, 2026-05-21):
#   30w/active=10/mb=64 has ~4.7 GB margin (well above the 0.2 GB margin that
#   killed Exp 1m and the 0.4 GB margin that killed 40w/pool=10). If 30w hits
#   OOM, that signals an architectural change (model grew, packing changed,
#   data distribution shift) — NOT a memory-pressure surprise. We want the
#   signal, not a silent hedge to 24w.
#
#   Auto-fallback options previously considered:
#     (A) ship 30w, no fallback — SELECTED
#     (B) 30w → 24w on OOM — loses ratio, hides architectural change
#     (C) 30w → 30w + mb=48 — untested at production scale
#
# Behavior:
#   - Launches training with --mp-workers 30
#   - Tails log every 60s for OOM / FATAL / "Training complete" / process death
#   - On OOM or unexpected death: exits with non-zero code, logs the tail,
#     and leaves all checkpoints + battle servers intact for manual investigation
#   - On "Training complete": exits 0
#
# Usage:
#   bash launch_phase2_with_oom_monitor.sh <run_name>
#
# Stage selector:
#   STAGE=stage1 (default — pure self-play warmup, init from BC v10)
#   STAGE=stage2 INIT_CKPT=<best Stage 1 snapshot> (with externals)
#
# Requires: working directory = pokemon-ai-starter/pokemon-ai/src
#           launch_rl.sh already bakes in PYTORCH_CUDA_ALLOC_CONF + LD_LIBRARY_PATH

set -uo pipefail

# =============================================================================
# Configuration
# =============================================================================
RUN_NAME="${1:-phase2_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="data/models/rl_v10/${RUN_NAME}"
LOG_FILE="/tmp/${RUN_NAME}.log"

# WORKERS — locked to 30 by S67 architectural ceiling (mb=64 update + worker CUDA contexts)
WORKERS=30

# Battle server ports (8 servers expected)
BS_PORTS=(9000 9001 9002 9003 9004 9005 9006 9007)

# Stage selector
STAGE="${STAGE:-stage1}"

case "$STAGE" in
  stage1)
    INIT_CKPT="data/models/bc/v10_padded_for_cis_dev.pt"
    EXT_ARGS=""
    ;;
  stage2)
    INIT_CKPT="${INIT_CKPT:?STAGE=stage2 requires INIT_CKPT env var (path to best Stage 1 snapshot)}"
    EXT_ARGS="--external-adapters external_adapters_phase2_day1.yaml"
    ;;
  *)
    echo "ERROR: STAGE must be stage1 or stage2 (got: $STAGE)" >&2
    exit 1
    ;;
esac

COMMON_ARGS="\
  --bc-anchor-ckpt data/models/bc/v10_padded_for_cis_dev.pt \
  --bc-anchor-coef 0.1 \
  --cis --bf16 --tier3 --tier3-minibatch-size 64 \
  --packed --no-per-chunk-gc \
  --cis-min-batch 32 --cis-timeout-ms 50 \
  --mp-workers ${WORKERS} \
  --games-per-iter 1600 --max-concurrent 200 \
  --n-iters 200 --warmup-iters 10 \
  --lr 1e-5 --lam 0.95 --ent-coef 0.02 --target-kl 0.03 \
  --grad-accum 1 --turn-cap 300 \
  --reward-style terminal \
  --procedural-teams /workspace/raw_data/pokemon_usage/2024-04 \
  --adaptive-entropy --adaptive-entropy-low 0.65 --adaptive-entropy-high 0.95 \
  --eval-interval 10 --eval-games 200 --eval-team-set metamon-competitive \
  --snapshot-interval 10 \
  --early-stop --early-stop-patience 3 \
  --win-rate-mode ema --win-rate-ema-alpha 0.3 --win-rate-ema-window 50 \
  ${EXT_ARGS}"

# =============================================================================
# Helpers
# =============================================================================
log() { echo "[wrapper $(date +%H:%M:%S)] $*"; }

restart_battle_servers() {
  log "Restarting battle servers (kill + relaunch on ports ${BS_PORTS[*]})..."
  pkill -9 -f battle_server.js 2>/dev/null || true
  sleep 5
  for p in "${BS_PORTS[@]}"; do
    nohup node battle_server.js --port "$p" > "/tmp/bs_${p}.log" 2>&1 &
  done
  sleep 10
  local n
  n=$(pgrep -cf battle_server.js || echo 0)
  log "Battle servers running: $n / ${#BS_PORTS[@]}"
  if [[ "$n" -lt "${#BS_PORTS[@]}" ]]; then
    log "WARN: expected ${#BS_PORTS[@]} BS but only $n started — investigate before launch"
  fi
}

# =============================================================================
# Main
# =============================================================================
log "=== Phase 2 launch wrapper (S67 final) ==="
log "Run name:    $RUN_NAME"
log "Stage:       $STAGE"
log "Init ckpt:   $INIT_CKPT"
log "Workers:     $WORKERS (locked by architectural ceiling)"
log "Out dir:     $OUT_DIR"
log "Log file:    $LOG_FILE"

mkdir -p "$OUT_DIR"

# Battle server clean restart (mandatory per feedback_battle_server_restart_after_kill)
restart_battle_servers

# Launch training
log "Launching training..."
nohup ./launch_rl.sh \
  $COMMON_ARGS \
  --init-from "$INIT_CKPT" \
  --out-dir "$OUT_DIR" \
  > "$LOG_FILE" 2>&1 &
PID=$!
log "Training PID: $PID"

# Monitor loop — poll every 60s
while kill -0 "$PID" 2>/dev/null; do
  if grep -qiE 'CUDA out of memory|torch\.OutOfMemoryError' "$LOG_FILE" 2>/dev/null; then
    log ""
    log "*** !!! OOM DETECTED on $WORKERS-worker config !!! ***"
    log ""
    log "This means an architectural change has occurred (model size grew,"
    log "sequence packing changed, batch composition shifted, etc)."
    log ""
    log "Latest checkpoint(s) preserved in: $OUT_DIR/selfplay_v9_*/iter_*.pt"
    log "Battle servers still running for your investigation."
    log ""
    log "DO NOT just relaunch with fewer workers — diagnose the architectural"
    log "change first. The 30w ship had 4.7 GB margin, so OOM = real shift."
    log ""
    log "=== Last 40 lines of training log ==="
    tail -40 "$LOG_FILE"
    log "=== End of training log ==="
    exit 1
  fi

  if grep -q 'Training complete' "$LOG_FILE" 2>/dev/null; then
    log "*** Training complete! ***"
    log "Final iter line:"
    grep -E '^\[..:..:..\] Iter [0-9]+:' "$LOG_FILE" | tail -3
    exit 0
  fi

  sleep 60
done

# Process exited without OOM and without "Training complete" — unexpected
if grep -q 'Training complete' "$LOG_FILE" 2>/dev/null; then
  log "Process exited (Training complete seen — late detection)."
  exit 0
fi

log "*** !!! UNEXPECTED EXIT (no OOM, no 'Training complete') !!! ***"
log "Last 40 lines of training log:"
tail -40 "$LOG_FILE"
exit 2
