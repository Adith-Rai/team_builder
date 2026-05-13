#!/bin/bash
# Launches profiling training run on pod. Designed for SSH-stable execution.
set -e
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src

# 1. Clean any prior state
pkill -9 -f "node battle_server" 2>/dev/null || true
pkill -9 -f "python.*train_rl" 2>/dev/null || true
pkill -9 -f "while true.*nvidia" 2>/dev/null || true
sleep 3

# 2. Start 8 battle_servers
for p in 9000 9001 9002 9003 9004 9005 9006 9007; do
  nohup node battle_server.js --port $p > /tmp/bs_$p.log 2>&1 &
done
sleep 5

# 3. Verify ports listening
LISTENING=$(ss -tlnp 2>/dev/null | grep -cE ':(900[0-7])')
if [ "$LISTENING" -ne 8 ]; then
  echo "ERROR: only $LISTENING/8 battle_servers listening"
  exit 1
fi
echo "battle_servers: 8/8 ready"

# 4. Clean profile artifacts
rm -f /tmp/profile_*.json /tmp/profile_run_diag.log
mkdir -p /tmp/profile_artifacts

# 5. Start GPU sampler in background
nohup bash -c 'while true; do echo $(date +%s.%N),$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits); sleep 1; done' > /tmp/profile_artifacts/gpu_util.csv 2>&1 &
GPU_PID=$!
echo "gpu_sampler_pid=$GPU_PID"

# 6. Launch profiling training run
export PROFILE_MODE=1
nohup python -u train_rl.py \
  --resume data/models/rl_v10/bc_anchor_1600g_v3/selfplay_v9_20260512_180539/snapshot_0049.pt \
  --bc-anchor-ckpt data/models/bc/v10_padded_for_cis_dev.pt \
  --bc-anchor-coef 0.1 \
  --device cuda \
  --servers 9000,9001,9002,9003,9004,9005,9006,9007 \
  --bf16 --cis --pipeline --mp-workers 8 \
  --tier3 --tier3-minibatch-size 16 \
  --games-per-iter 800 \
  --max-concurrent 200 \
  --n-iters 1 \
  --warmup-iters 0 \
  --lr 1e-5 --lam 0.95 --ent-coef 0.02 --target-kl 0.03 \
  --grad-accum 1 --turn-cap 300 \
  --reward-style terminal \
  --eval-interval 999 --eval-games 200 --eval-team-set metamon-competitive \
  --snapshot-interval 999 \
  --procedural-teams /workspace/raw_data/pokemon_usage/2024-04 \
  --adaptive-entropy --adaptive-entropy-low 0.65 --adaptive-entropy-high 0.95 \
  --win-rate-mode ema --win-rate-ema-alpha 0.3 --win-rate-ema-window 50 \
  --out-dir data/models/rl_v10/profile_pass_b \
  > /tmp/profile_run_diag.log 2>&1 &
TRAIN_PID=$!
disown
echo "train_pid=$TRAIN_PID"

# 7. Wait a moment for launch to settle
sleep 5

# 8. Verify it's actually running
if kill -0 $TRAIN_PID 2>/dev/null; then
  echo "train_rl alive after 5s"
else
  echo "ERROR: train_rl exited within 5s"
  tail -20 /tmp/profile_run_diag.log
  exit 1
fi

echo "LAUNCH OK"
