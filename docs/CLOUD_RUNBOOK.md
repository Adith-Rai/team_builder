# CLOUD_RUNBOOK.md — paint-by-numbers cloud BC

Get a gen-9 BC checkpoint trained on RunPod A100 in ~3-6 hours for ~$10-30.
Same machinery applies to multi-gen later (just bigger volume + longer run).

Last updated: Session 48 (2026-05-02). Tested on: not yet — first run is the
validation.

## TL;DR

```
1. Set up RunPod account + S3 bucket (~30 min, one-time)
2. Run sync_to_s3.sh from your laptop (overnight, ~24 hr at residential upload)
3. Spin up A100 RunPod pod, run cloud_setup.sh (~10 min)
4. Run sync_from_s3.sh on the pod (~5-10 min)
5. Run cloud_smoke.sh — 50-batch validation (~$0.50, ~5 min)
6. Launch train_bc.py with --compile --batch-size 48 --workers 8 (~3-6 hr to converge)
7. scp best.pt back, kill pod
```

**No RunPod network volume needed.** Each run: spin up pod with ephemeral
NVMe, sync from S3 (5-10 min), train, scp results, terminate. Cheaper
than $20/mo network volume and simpler.

## What lives where

- **Repo**: github (push from local). Cloud just `git clone`s it.
- **Memmap (104 GB)**: S3 bucket. Synced to RunPod network volume on first
  run, attached to A100 pod. Pay storage rates ($2.40/mo), not GPU rates.
- **Lookup tables, vocab, config**: in the repo (small files). Travel with
  the code.
- **Output checkpoints**: written to RunPod network volume during training,
  scp'd back to local for archival.

## One-time setup (~30 min, do it once)

### A. RunPod account

1. Sign up at https://runpod.io.
2. Add $50 to start (you'll spend $10-30 per gen-9 BC run).
3. (Optional) Verify identity for "Secure Cloud" — slightly more reliable
   pods at ~$1.69/hr A100 40GB. "Community Cloud" is cheaper ($0.80-1.20/hr)
   but pods can be reclaimed by the host.
4. Create an SSH key pair if you don't have one; add the public key to
   RunPod settings → SSH keys.

### B. S3 bucket

1. AWS account with billing set up.
2. Create a bucket: `aws s3 mb s3://team-builder-data --region us-east-1`
   (use the same region as your A100 pod — RunPod's `us-east-1` is closest
   if you have the option).
3. Pick the region: same region as RunPod = free egress. Different region =
   $0.02/GB egress (so $2/full-sync; not catastrophic but adds up).

### C. Local AWS credentials

```
aws configure
# enter your access key, secret, region (us-east-1), default output format (json)
```

## Recurring: per-run setup

### 1. Sync data to S3 (one-time per dataset)

On your laptop (this runs in the background while you work; takes ~24 hr
at 10 Mbps residential upload):

```
cd /c/Users/raiad/OneDrive/Desktop/team_builder
bash pokemon-ai-starter/pokemon-ai/scripts/sync_to_s3.sh
```

Verify when done: `aws s3 ls s3://team-builder-data/datasets/human_v8_100k/`

### 2. Spin up A100 pod

RunPod dashboard → Deploy:
- Template: **PyTorch 2.x** (pre-baked CUDA 12.1, Python 3.11)
- GPU: **A100 40GB** (Secure Cloud preferred for stability)
- Container disk: **150 GB** ephemeral (room for repo + 104 GB memmap + ckpts).
  No network volume needed — we sync S3 → ephemeral disk per run.
- Start.

You'll get an SSH command like:
```
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_rsa
```

### 3. Run setup on the pod

```
ssh root@<pod-ip> -p <port>
cd /workspace
git clone https://github.com/Adith-Rai/team_builder.git
cd team_builder
bash pokemon-ai-starter/pokemon-ai/scripts/cloud_setup.sh
```

This installs deps, runs the test suite, and benches throughput.
Expected output:
- 17/17 tokenizer tests pass
- 9/9 policy tests pass
- bench_bc_step.py at B=32 fp16: 2-5 ms/turn, peak_mem 4-8 GB

### 4. Sync data from S3 to pod's local NVMe

```
cd /workspace/team_builder
bash pokemon-ai-starter/pokemon-ai/scripts/sync_from_s3.sh
```

Takes 5-10 min for 104 GB at A100-pod's network speed.

### 5. Run smoke test (~$0.50, 5 min)

```
cd /workspace/team_builder
bash pokemon-ai-starter/pokemon-ai/scripts/cloud_smoke.sh
```

50 batches at the cloud config (B=48, fp16, compile, workers=8). Validates:
- torch.compile completes without errors
- B=48 fp16 fits without OOM
- workers=8 doesn't deadlock
- Loss decreases
- Throughput in 2-5 ms/turn range

If green, proceed. If anything fails, debug before committing GPU time
to a multi-hour run.

### 6. Launch BC training

```
cd /workspace/team_builder/pokemon-ai-starter/pokemon-ai/src
ln -s /workspace/data/datasets/human_v8_100k data/datasets/human_v8_100k

nohup python -u train_bc.py --use-transformer --compile \
  --memmap-dir data/datasets/human_v8_100k \
  --epochs 50 --batch-size 48 --lr 1e-4 --fp16 \
  --workers 8 --eval-games 0 --val-ratio 0.1 \
  --sched constant --warmup-steps 200 \
  --early-stop-patience 2 \
  --save-every 1000 --run-name v10_cloud_gen9 \
  --device cuda \
  > /workspace/v10_cloud_gen9.log 2>&1 &

echo $! > /workspace/v10.pid   # save PID for later kill
```

**Key flag changes vs the local config:**
| Local                | Cloud                | Why |
|----------------------|----------------------|-----|
| `--batch-size 4`     | `--batch-size 48`    | A100 has 40 GB; matches Metamon's published default |
| `--workers 2`        | `--workers 8`        | Linux fork makes more workers cheap |
| `--gradient-checkpoint` | (omitted)         | 40 GB has plenty of headroom |
| `--sched cosine`     | `--sched constant`   | matches Metamon's effective LR behavior |
| (no compile)         | `--compile`          | 10-25% speedup on Linux/A100 |
| `--epochs 5`         | `--epochs 50` + early-stop patience 2 | Metamon-style: train long, stop on plateau |

### 6. Monitor

```
tail -F /workspace/v10_cloud_gen9.log
```

Watch for:
- Batch reports every 20 batches: confirm loss decreasing, acc climbing
- Epoch lines: `Epoch N: train_loss=... val_loss=...`
- Mid-step checkpoints: `[checkpoint saved at batch ...]` every 1000 batches
- `New best smart_avg=...` or `New best val_loss=...`
- Early-stop messages if patience exhausts: `[early-stop] patience exhausted; stopping at epoch N.`

### 7. Pull results back

When training finishes (or hits early stop):

```
# from local laptop:
scp -P <port> root@<pod-ip>:/workspace/team_builder/pokemon-ai-starter/pokemon-ai/src/data/models/bc/v10_cloud_gen9/best.pt \
  ./pokemon-ai-starter/pokemon-ai/src/data/models/bc/v10_cloud_gen9_best.pt
scp -P <port> root@<pod-ip>:/workspace/v10_cloud_gen9.log ./logs/v10_cloud_gen9.log
```

Then run eval locally:
```
cd pokemon-ai-starter/pokemon-ai/src
python eval_metamon_competitive.py \
  --checkpoints v10_cloud=data/models/bc/v10_cloud_gen9_best.pt \
  --servers 9000 --n-games 200 --concurrency 100 --device cuda \
  --out-json data/eval/registry/v10_cloud_gen9.json
```

### 8. Kill the pod

RunPod dashboard → pod → Terminate. Network volume persists; container goes away.

## Troubleshooting

**"OOM on cloud"**: very unlikely at B=48 / 40GB / fp16, but if it happens,
drop to B=32 first. If still OOM, something is wrong with the install — re-check
torch version + cuda matched.

**"torch.compile error"**: drop `--compile` flag. Sometimes Inductor doesn't
play well with specific kernels; the fallback to eager is fine, just lose 10-25%
throughput.

**"DataLoader workers hang"**: drop `--workers 8` to `--workers 4` or `--workers 0`.
On Linux this should be rare but Persistent_workers + memmap can occasionally
deadlock on large filesystems.

**"S3 sync slow"**: same-region must be set. Check `aws configure get region`.
Cross-region sync runs at ~30-50 MB/s; same-region runs at 100+ MB/s.

**"Pod reclaimed mid-training"**: Community Cloud has this risk. You'll have
the latest mid-step ckpt on the network volume; spin up a new pod and resume
with `--resume <latest>.pt`.

## Cost ceiling for the gen-9 BC milestone

| Item | Cost |
|------|------|
| RunPod A100 Secure (4 hr at $1.69) | $7 |
| RunPod A100 Secure (8 hr conservative) | $14 |
| Smoke test (~5 min A100) | $0.50 |
| S3 storage (1 month, 104 GB) | $2.40 |
| S3 sync (same region) | $0 egress |
| **Total per run, conservative** | **~$10-15 + $2.40/mo storage** |

If you delete S3 data after multi-gen is done, you only pay storage for
the months you keep it. ~$5-10 total storage cost for a typical
research-phase project.

Multi-gen later: ~4× compute ($40-60) plus storage of additional gens
(~$10/mo). Ballpark $50-100 total for a multi-gen converging run.
