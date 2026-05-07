# Phase 1 v3 — RSS watcher log + fetch helper

This file is the cross-session record of the RSS watcher running on the
RunPod A100 pod during Phase 1 v3 (started 2026-05-06).

## Purpose

Diagnose iter-time growth (41.8 → 47.3 → 49.3 → 50.5 min over iters 0-6
of Phase 1 v3 warmup). Suspected cause: memory leak in mp_disk_collect
workers (opp_cache uses `weights_only=False` which loads full ckpt
including optimizer state, ~720 MB per cached opp).

## How to fetch the latest watcher log (any session)

```bash
ssh -i ~/.ssh/id_ed25519 -p 47913 -o StrictHostKeyChecking=no root@195.26.233.30 \
  "cat /workspace/logs/rss_watcher.log" \
  > docs/diag/rss_watcher_log_phase1_v3.txt
```

Then read `docs/diag/rss_watcher_log_phase1_v3.txt`.

## Watcher details

- Pod: 195.26.233.30:47913 (RunPod A100 80GB, pod ID `t56a2jndi8iyz9` or similar)
- Pod log path: `/workspace/logs/rss_watcher.log`
- Watcher script: `/workspace/scripts/rss_watcher.sh`
- Trigger: tails `/workspace/logs/ppo_phase1_v3.log` and records on each "Iter N: W/L/T=" line
- Records: timestamp, iter num, main process RSS (MB), workers total RSS (MB), GPU memory used (MB), per-worker RSS array (KB)

## Watcher process

Launched via `setsid nohup /workspace/scripts/rss_watcher.sh &`. Survives ssh disconnect. To kill: `pkill -f rss_watcher.sh` on pod.

## Snapshot at iter 6 (single-point baseline before watcher started)

| Metric | Value |
|---|---|
| Main RSS | 10.6 GB |
| Worker RSS (each) | 2.9 GB |
| Workers total RSS | 23.2 GB |
| GPU used | 20.7 GB |
| Disk | 6.2 GB / 150 GB |

## Acceptance criteria for "leak confirmed"

If between any two consecutive iter records:
- Main RSS grows by >300 MB → confirmed main-side leak
- Any worker RSS grows by >150 MB → confirmed worker-side leak
- Stable RSS (±50 MB) → not a memory leak; iter time growth is from disk/cudnn/fragmentation

## Suspected root cause (mp_disk_collect.py:535-536)

```python
cached_ckpt = torch.load(opp_path, map_location=opponent_device,
                         weights_only=False)
opp_cache[opp_path] = {"ckpt": cached_ckpt, "last_used": time.time()}
```

`weights_only=False` loads the full ckpt dict including AdamW optimizer
state (2× momenta = 480 MB) and scheduler. Local code (e.g.
`eval_elo_ladder.py`) uses `weights_only=True` for opp loads. Should fix
mp_disk_collect to extract only the state_dict needed for inference.

Per-worker overhead estimate:
- 240 MB main model
- 3 × 720 MB opp_cache (full ckpts) = 2.16 GB ← bug
- 200 MB asyncio + battle env + numpy buffers
- 500 MB Python interpreter + libs
- **Expected total: ~3.1 GB per worker** (close to observed 2.9 GB)

So the 2.9 GB **may not be a leak** — could be just `weights_only=False`
bloat that was always there. The iter-time growth could be a SEPARATE
issue (cuDNN benchmark autotuning, disk IO, allocator fragmentation).
Watcher data will tell us which.
