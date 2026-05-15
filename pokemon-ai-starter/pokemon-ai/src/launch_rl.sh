#!/usr/bin/env bash
# launch_rl.sh — canonical entry point for train_rl.py launches on prod pod
# (S64 Phase B onward, torch 2.5.1+cu121).
#
# Why this wrapper exists:
#   torch 2.5.1 brings in nvidia-cudnn-cu12==9.1 as a transitive dep. The
#   wheel installs libcudnn.so.9 at a non-standard path that is NOT on the
#   dynamic linker search list. Any torch op that touches cuDNN (conv,
#   attention via SDPA/flex_attention) fails with `ImportError: libcudnn.so.9`
#   unless LD_LIBRARY_PATH is set. `import torch` alone works fine without it,
#   so the failure mode is "training launches, then crashes mid-iter."
#
# Usage:
#   ./launch_rl.sh [train_rl.py args...]
# Background:
#   nohup ./launch_rl.sh [args] > /tmp/run.log 2>&1 &
#
# Pod-side ~/.bashrc also exports this for any direct python invocations
# (defense in depth); this wrapper is the canonical training entry.

set -e

# cuDNN 9 path baked in by S64 Phase B (project_s64_phase_a_results.md §4-R.1).
# If you move python versions or torch installs, update this path.
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}

# Exec preserves the PID + signal semantics nohup expects.
exec python -u train_rl.py "$@"
