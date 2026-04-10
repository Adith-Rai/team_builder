#!/bin/bash
# Run 3 Elo ladder shards in parallel, then combine
# Usage: bash run_elo_shards.sh

SNAPS=(
    data/models/rl_v8/BEST_PPO_iter80_h2h_52.8pct.pt
    data/models/rl_v9/selfplay_v9_20260329_070901/snapshot_0199.pt
    data/models/rl_v9/selfplay_v9_20260330_002727/snapshot_0284.pt
    data/models/rl_v9/selfplay_v9_20260330_002727/snapshot_0339.pt
    data/models/rl_v9/selfplay_v9_20260331_145929/snapshot_0439.pt
    data/models/rl_v9/selfplay_v9_20260401_141524/snapshot_0699.pt
    data/models/rl_v9/selfplay_v9_20260401_213855/snapshot_0739.pt
    data/models/rl_v9/selfplay_v9_20260402_081349/snapshot_0839.pt
    data/models/rl_v9/selfplay_v9_20260404_084945/snapshot_0999.pt
    data/models/rl_v9/selfplay_v9_20260404_192922/snapshot_1079.pt
    data/models/rl_v9/selfplay_v9_20260405_164115/snapshot_1199.pt
    data/models/rl_v9/selfplay_v9_20260405_164115/snapshot_1279.pt
    data/models/rl_v9/selfplay_v9_20260407_020353/snapshot_1439.pt
    data/models/rl_v9/selfplay_v9_20260407_124041/snapshot_1579.pt
    data/models/rl_v9/selfplay_v9_20260408_042048/snapshot_1784.pt
    data/models/rl_v9/selfplay_v9_20260409_080620/snapshot_1789.pt
    data/models/rl_v9/selfplay_v9_20260409_080620/snapshot_1809.pt
    data/models/rl_v9/selfplay_v9_20260409_080620/snapshot_1839.pt
    data/models/rl_v9/selfplay_v9_20260409_080620/snapshot_1879.pt
    data/models/rl_v9/selfplay_v9_20260409_080620/snapshot_1919.pt
    data/models/rl_v9/selfplay_v9_20260409_080620/snapshot_1959.pt
    data/models/rl_v9/selfplay_v9_20260409_080620/snapshot_1984.pt
    data/models/rl_v9/selfplay_v9_20260410_001804/snapshot_1999.pt
)

NAMES="BC_base sp0199 sp0284 sp0339 sp0439 sp0699 sp0739 sp0839 sp0999 sp1079 sp1199 sp1279 sp1439 sp1579 sp1784 sp1789 sp1809 sp1839 sp1879 sp1919 sp1959 sp1984 sp1999"

COMMON="--names $NAMES --bots all --n-games 100 --concurrency 100 --device cuda"

echo "Starting 3 Elo shards..."

python -u eval_elo_ladder.py --snapshots "${SNAPS[@]}" $COMMON \
    --server ws://127.0.0.1:9000/showdown/websocket \
    --shard 0/3 --out-json data/eval/elo_s35_shard0.json \
    2>&1 | tee elo_s35_shard0.log &
PID0=$!

python -u eval_elo_ladder.py --snapshots "${SNAPS[@]}" $COMMON \
    --server ws://127.0.0.1:9001/showdown/websocket \
    --shard 1/3 --out-json data/eval/elo_s35_shard1.json \
    2>&1 | tee elo_s35_shard1.log &
PID1=$!

python -u eval_elo_ladder.py --snapshots "${SNAPS[@]}" $COMMON \
    --server ws://127.0.0.1:9002/showdown/websocket \
    --shard 2/3 --out-json data/eval/elo_s35_shard2.json \
    2>&1 | tee elo_s35_shard2.log &
PID2=$!

echo "Shard PIDs: $PID0, $PID1, $PID2"
echo "Waiting for all shards..."
wait $PID0 $PID1 $PID2
echo "All shards done. Combining..."

python -u eval_elo_ladder.py --combine \
    data/eval/elo_s35_shard0.json data/eval/elo_s35_shard1.json data/eval/elo_s35_shard2.json \
    --out-json data/eval/elo_session35_exp1.json

echo "DONE. Results in data/eval/elo_session35_exp1.json"
