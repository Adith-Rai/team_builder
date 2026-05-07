#!/bin/bash
# Phase 1 v3 production resumed launch (iters 11-200, 190 iters total).
# Resumes from smoke-iter-10's snapshot_0010.pt with the validated
# memory hygiene fixes (commit bedcbc3) live. --warmup-iters 0 since
# value head is fully converged.
set -e
ulimit -n 65536
export PYTHONUNBUFFERED=1
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src

# Defensive: kill anything lingering
pkill -f "python -u train_rl.py" 2>/dev/null || true
pkill -f multiprocessing-fork 2>/dev/null || true
sleep 2

nohup python -u train_rl.py \
  --resume data/models/rl_v10/ppo_phase1_v3_smoke/selfplay_v9_20260507_083223/snapshot_0010.pt \
  --pool-anchors data/models/bc/v10_cloud_gen9/epoch_003.pt \
  --device cuda --servers 9000,9001,9002,9003,9004,9005,9006,9007 \
  --fp16 --mp --mp-workers 8 \
  --games-per-iter 1600 --max-concurrent 200 \
  --opponent-device cuda \
  --n-iters 190 --warmup-iters 0 \
  --lr 1e-5 --lam 0.95 --ent-coef 0.02 --reward-style terminal --grad-accum 1 \
  --adaptive-entropy --adaptive-entropy-low 0.65 --adaptive-entropy-high 0.95 \
  --win-rate-mode ema --win-rate-ema-alpha 0.3 --win-rate-ema-window 50 \
  --eval-interval 10 --eval-team-set metamon-competitive --eval-games 200 \
  --snapshot-interval 5 --early-stop --early-stop-patience 3 \
  --turn-cap 300 \
  --procedural-teams /workspace/raw_data/pokemon_usage/2024-04 \
  --out-dir data/models/rl_v10/ppo_phase1_v3_resumed \
  > /workspace/logs/ppo_phase1_v3_resumed.log 2>&1 &

echo "production launched pid=$!"
sleep 5
ps -ef | grep "python -u train_rl" | grep -v grep | head -3
