#!/bin/bash
# Wait for the all-pairings data gen to finish, then run smart bots extra data
set -e
cd "$(dirname "$0")"

echo "[smartbot_datagen] Waiting for main datagen to finish..."
while tasklist 2>/dev/null | grep -q python; do
    sleep 30
done
echo "[smartbot_datagen] Main datagen done. Starting smart bot extra generation..."

# Smart bots: SimpleHeuristics, SmartDamage, Tactical, Strategic
# 4x4 = 16 pairings, 100 games each, log both sides
BOTS=("SimpleHeuristics" "SmartDamage" "Tactical" "Strategic")

for a in "${BOTS[@]}"; do
    for b in "${BOTS[@]}"; do
        echo "[smartbot_datagen] $a vs $b (100 games)"
        PYTHONUNBUFFERED=1 python observer.py \
            --bots "$a,$b" \
            --games 100 \
            --max-concurrent 8 \
            --batch-per-worker 10 \
            --format gen9ou \
            --log-both
    done
done

echo "[smartbot_datagen] All done!"
