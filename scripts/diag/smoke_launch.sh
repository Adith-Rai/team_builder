#!/bin/bash
# 1-iter smoke validation of mp_disk_collect.py memory hygiene fixes.
# Resumes from snapshot_0009.pt with --warmup-iters 0 (warmup converged).
set -e
ulimit -n 65536
export PYTHONUNBUFFERED=1
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src

# Kill any leftover trainer/workers (defensive)
pkill -f "python train_rl.py" 2>/dev/null || true
pkill -f multiprocessing-fork 2>/dev/null || true
sleep 2

nohup python -u train_rl.py \
  --resume data/models/rl_v10/ppo_phase1_v3_cloud/selfplay_v9_20260506_232704/snapshot_0009.pt \
  --pool-anchors data/models/bc/v10_cloud_gen9/epoch_003.pt \
  --device cuda --servers 9000,9001,9002,9003,9004,9005,9006,9007 \
  --fp16 --mp --mp-workers 8 \
  --games-per-iter 1600 --max-concurrent 200 \
  --opponent-device cuda \
  --n-iters 1 --warmup-iters 0 \
  --lr 1e-5 --lam 0.95 --ent-coef 0.02 --reward-style terminal --grad-accum 1 \
  --adaptive-entropy --adaptive-entropy-low 0.65 --adaptive-entropy-high 0.95 \
  --win-rate-mode ema --win-rate-ema-alpha 0.3 --win-rate-ema-window 50 \
  --eval-interval 999 --eval-team-set metamon-competitive --eval-games 200 \
  --snapshot-interval 1 --early-stop --early-stop-patience 3 \
  --turn-cap 300 \
  --procedural-teams /workspace/raw_data/pokemon_usage/2024-04 \
  --out-dir data/models/rl_v10/ppo_phase1_v3_smoke \
  > /workspace/logs/ppo_phase1_v3_smoke.log 2>&1 &

echo "smoke launched pid=$!"
sleep 5
ps -ef | grep "python train_rl.py" | grep -v grep | head -3
