#!/bin/bash
# v4 full data generation: ~5M records weighted toward smart bots
# Then auto-converts to memmap when done
set -e
cd "$(dirname "$0")"

SMART=("SimpleHeuristics" "SmartDamage" "Tactical" "Strategic")
MEDIUM=("GreedySE" "HazardSense" "SwitchAwareEscape" "SetupThenSweep")
WEAK=("MaxBasePower" "Random")

run_pairs() {
    local -n arr1=$1
    local -n arr2=$2
    local games=$3
    local label=$4
    for a in "${arr1[@]}"; do
        for b in "${arr2[@]}"; do
            echo "[datagen] $label: $a vs $b ($games games)"
            PYTHONUNBUFFERED=1 python observer.py \
                --bots "$a,$b" --games "$games" --max-concurrent 8 \
                --batch-per-worker 10 --format gen9ou --log-both
        done
    done
}

echo "=== V4 DATA GENERATION START: $(date) ==="

echo "--- Smart x Smart (16 pairs, 2500 games each) ---"
run_pairs SMART SMART 2500 "S×S"

echo "--- Smart x Medium (32 pairs, 500 games each) ---"
run_pairs SMART MEDIUM 500 "S×M"

echo "--- Medium x Medium (16 pairs, 400 games each) ---"
run_pairs MEDIUM MEDIUM 400 "M×M"

echo "--- Smart x Weak (16 pairs, 300 games each) ---"
run_pairs SMART WEAK 300 "S×W"

echo "--- Medium x Weak (16 pairs, 200 games each) ---"
run_pairs MEDIUM WEAK 200 "M×W"

echo "--- Weak x Weak (4 pairs, 100 games each) ---"
run_pairs WEAK WEAK 100 "W×W"

echo "=== DATA GENERATION COMPLETE: $(date) ==="

# Count records
python -c "
import glob
files = sorted(glob.glob('data/datasets/obs/*.jsonl'))
total = sum(1 for f in files for _ in open(f))
print(f'Total: {len(files)} files, {total} records')
"

# Convert to memmap
echo "=== CONVERTING TO MEMMAP ==="
python convert_jsonl_to_memmap.py \
    --input-dir data/datasets/obs \
    --output-dir data/datasets/memmap

echo "=== ALL DONE: $(date) ==="
