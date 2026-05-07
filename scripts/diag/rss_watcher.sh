#!/bin/bash
LOG=/workspace/logs/ppo_phase1_v3.log
OUT=/workspace/logs/rss_watcher.log
MAIN_PID=475509

echo "=== watcher started at $(date) ===" >> $OUT
tail -F $LOG | while read line; do
  if [[ "$line" == *"Iter "*": W/L/T="* ]]; then
    iter=$(echo "$line" | grep -oP 'Iter \K[0-9]+')
    ts=$(date +"%H:%M:%S")
    main_rss=$(ps -p $MAIN_PID -o rss= 2>/dev/null | tr -d ' ')
    workers_rss=$(ps --no-headers -eo rss,cmd | grep multiprocessing-fork | grep -v grep | awk '{print $1}' | paste -sd, -)
    workers_total=$(ps --no-headers -eo rss,cmd | grep multiprocessing-fork | grep -v grep | awk '{sum+=$1} END {print sum}')
    gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "$ts iter=$iter main_rss_mb=$((main_rss/1024)) workers_total_mb=$((workers_total/1024)) gpu_used_mb=$gpu_used workers_each_kb=[$workers_rss]" >> $OUT
  fi
done
