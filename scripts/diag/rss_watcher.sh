#!/bin/bash
# Tails the trainer log and records main+worker RSS at end of each iter.
# MAIN_PID resolved dynamically each iter so the watcher survives
# trainer kill+restart cycles (e.g., post-iter-9 leak-fix restart in
# Session 50 cont.) without producing stale data.
LOG=/workspace/logs/ppo_phase1_v3.log
OUT=/workspace/logs/rss_watcher.log

echo "=== watcher started at $(date) ===" >> $OUT
tail -F $LOG | while read line; do
  if [[ "$line" == *"Iter "*": W/L/T="* ]]; then
    iter=$(echo "$line" | grep -oP 'Iter \K[0-9]+')
    ts=$(date +"%H:%M:%S")
    # Resolve trainer PID dynamically: pick the python process running
    # train_rl.py that is NOT a multiprocessing-fork worker.
    MAIN_PID=$(ps -ef | grep "python train_rl.py" | grep -v grep | grep -v multiprocessing-fork | awk '{print $2}' | head -1)
    main_rss=$(ps -p $MAIN_PID -o rss= 2>/dev/null | tr -d ' ')
    workers_rss=$(ps --no-headers -eo rss,cmd | grep multiprocessing-fork | grep -v grep | awk '{print $1}' | paste -sd, -)
    workers_total=$(ps --no-headers -eo rss,cmd | grep multiprocessing-fork | grep -v grep | awk '{sum+=$1} END {print sum}')
    gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "$ts iter=$iter main_rss_mb=$((main_rss/1024)) workers_total_mb=$((workers_total/1024)) gpu_used_mb=$gpu_used workers_each_kb=[$workers_rss]" >> $OUT
  fi
done
