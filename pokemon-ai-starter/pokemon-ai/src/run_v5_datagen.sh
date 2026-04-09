#!/bin/bash
# v5 full data generation: ~5M records weighted toward smart bots
# Runs one python process at a time per pairing (no background jobs = no zombie processes)
# Then auto-converts to memmap when done
set -e
cd "$(dirname "$0")"

# --- Tunables ---
MAX_CONCURRENT=16       # Async workers inside observer.py (all in one process, no name collisions)
BATCH_PER_WORKER=50     # Games per player before reconnect
FORMAT="gen9ou"

run_pair() {
    local a="$1" b="$2" games="$3" label="$4"
    echo ""
    echo "[datagen] $label: $a vs $b ($games games) — $(date +%H:%M:%S)"
    PYTHONUNBUFFERED=1 python observer.py \
        --bots "$a,$b" --games "$games" \
        --max-concurrent "$MAX_CONCURRENT" \
        --batch-per-worker "$BATCH_PER_WORKER" \
        --format "$FORMAT" --log-both
}

echo "=== V5 DATA GENERATION START: $(date) ==="

SMART=("SimpleHeuristics" "SmartDamage" "Tactical" "Strategic")
MEDIUM=("GreedySE" "HazardSense" "SwitchAwareEscape" "SetupThenSweep")
WEAK=("MaxBasePower" "Random")

# --- Smart x Smart: 16 pairs, 2500 games each (highest quality) ---
echo "=== Smart x Smart (16 pairs × 2500 games) ==="
for a in "${SMART[@]}"; do
    for b in "${SMART[@]}"; do
        run_pair "$a" "$b" 2500 "S×S"
    done
done

# --- Smart x Medium: 32 pairs, 500 games each ---
echo "=== Smart x Medium (32 pairs × 500 games) ==="
for a in "${SMART[@]}"; do
    for b in "${MEDIUM[@]}"; do
        run_pair "$a" "$b" 500 "S×M"
    done
done

# --- Medium x Medium: 16 pairs, 400 games each ---
echo "=== Medium x Medium (16 pairs × 400 games) ==="
for a in "${MEDIUM[@]}"; do
    for b in "${MEDIUM[@]}"; do
        run_pair "$a" "$b" 400 "M×M"
    done
done

# --- Smart x Weak: 8 pairs, 300 games each ---
echo "=== Smart x Weak (8 pairs × 300 games) ==="
for a in "${SMART[@]}"; do
    for b in "${WEAK[@]}"; do
        run_pair "$a" "$b" 300 "S×W"
    done
done

# --- Medium x Weak: 8 pairs, 200 games each ---
echo "=== Medium x Weak (8 pairs × 200 games) ==="
for a in "${MEDIUM[@]}"; do
    for b in "${WEAK[@]}"; do
        run_pair "$a" "$b" 200 "M×W"
    done
done

# --- Weak x Weak: 4 pairs, 100 games each ---
echo "=== Weak x Weak (4 pairs × 100 games) ==="
for a in "${WEAK[@]}"; do
    for b in "${WEAK[@]}"; do
        run_pair "$a" "$b" 100 "W×W"
    done
done

echo "=== DATA GENERATION COMPLETE: $(date) ==="

# Count records
echo "=== Counting records ==="
python -c "
import glob
files = sorted(glob.glob('data/datasets/obs/*.jsonl'))
total = sum(1 for f in files for _ in open(f, encoding='utf-8'))
print(f'Total: {len(files)} files, {total:,} records')
"

# Convert to memmap
echo "=== CONVERTING TO MEMMAP ==="
python convert_jsonl_to_memmap.py \
    --data "data/datasets/obs/*.jsonl" \
    --out-dir data/datasets/memmap

echo "=== ALL DONE: $(date) ==="
