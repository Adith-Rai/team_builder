#!/bin/bash
# Triggers SIGUSR1 to all profile processes for synchronous viztracer save,
# waits for completion, then SIGKILLs leftovers + reports final state.
set +e

echo "=== Pre-SIGUSR1 process state ==="
ps -eo pid,etime,command | grep -E "python.*(train_rl|spawn_main)" | grep -v grep | head -15

echo "=== Sending SIGUSR1 to all train_rl/spawn_main processes ==="
pkill -USR1 -f "python.*train_rl" 2>/dev/null
pkill -USR1 -f "multiprocessing.spawn" 2>/dev/null

echo "=== Waiting up to 90 sec for saves to complete ==="
for i in $(seq 1 18); do
    sleep 5
    SAVED=$(ls /tmp/profile_*.json 2>/dev/null | wc -l)
    ALIVE=$(pgrep -cf "python.*(train_rl|spawn_main)" 2>/dev/null || echo 0)
    echo "  +${i}s: $SAVED JSONs saved, $ALIVE processes still alive"
    if [ "$ALIVE" -eq 0 ]; then
        echo "  all processes exited cleanly"
        break
    fi
done

echo "=== Cleanup any remaining ==="
pkill -KILL -f "python.*(train_rl|spawn_main)" 2>/dev/null
pkill -9 -f "while true.*nvidia" 2>/dev/null
sleep 2

echo "=== Final state ==="
pgrep -cf "python.*(train_rl|spawn_main)" || echo "  python: 0"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

echo "=== All profile JSON files saved ==="
ls -lh /tmp/profile_*.json 2>/dev/null

echo "=== GPU util sample summary (last 20 sec) ==="
tail -20 /tmp/profile_artifacts/gpu_util.csv 2>/dev/null
