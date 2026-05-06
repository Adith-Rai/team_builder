# Phase 1 v2 — Cloud Launch Checklist

**Goal:** Run Phase 1 v2 on RunPod A100 80GB with the lr=1e-5 fix + bigger batches.
**Expected duration:** ~25-35 hours wall-clock for 200 iters.
**Expected cost:** $40-60 at RunPod's $1.50/hr A100 SXM rate.
**Prereq:** local lr=1e-5 confirmation eval lands at smart_avg ≥65% (= confirms diagnosis).

---

## Phase 0 — Before you provision (do locally first)

- [ ] Wait for local 20-iter lr=1e-5 confirmation to finish (currently iter 9/19, ~1.5 hr remaining)
- [ ] Eval iter-19 snapshot:
      ```
      python eval_diag.py --ckpt data/models/rl_v10/lr1e5_confirm/<run>/snapshot_0019.pt \
        --n-games 100 --max-conc 20 --label lr1e5_iter19
      ```
- [ ] Verify smart_avg ≥ 65% (= no catastrophic regression). If ≥ 67% (BC parity), proceed. If < 65%, regroup before launching cloud.

## Phase 1 — Provision pod (5-10 min)

- [ ] RunPod console → Deploy → A100 SXM 80GB ($1.50/hr)
- [ ] **Network Volume optional this run** (Container Disk is fine IF you don't `podStop+Resume`):
      - The Session 49 incident specifically required a stop+resume cycle to break Container Disk persistence.
      - Plan for THIS run: don't stop/resume, just terminate when done. R2-sync at every snapshot (Phase 8) gives a recovery point if the pod dies unexpectedly. Worst-case loss = 5 iters (~30 min, ~$1).
      - If you prefer the safety net, attach a Network Volume; not required.
- [ ] Region: same as your R2 bucket (US-East per `r2_env.local.sh`) for fast transfer
- [ ] Container template: PyTorch 2.1+ with CUDA 12.x base
- [ ] Note the pod ID — for SSH and termination later

## Phase 2 — Bootstrap pod (~15-20 min)

- [ ] SSH into pod (or web terminal for manual mode)
- [ ] Run cloud_setup.sh:
      ```
      cd /workspace
      git clone <your-repo-url> team_builder
      cd team_builder/pokemon-ai-starter/pokemon-ai
      bash scripts/cloud_setup.sh
      ```
      This handles: venv creation, dependency install, R2 credentials sourcing.
- [ ] Verify `python -c "import torch; print(torch.cuda.is_available())"` → True
- [ ] Verify `node --version` → Node 20+ (battle_server.js needs it)

## Phase 3 — Sync checkpoints (5-10 min)

For Phase 1 v2 PPO, we need TWO ckpts (NOT the human_v8 memmap — we're not retraining BC).

```bash
source pokemon-ai-starter/pokemon-ai/scripts/r2_env.local.sh
# Upload from local FIRST (do this on your local machine, before cloud)
aws s3 cp data/models/bc/v10_cloud_gen9/epoch_003.pt \
  s3://team-builder-data/models/bc/v10_cloud_gen9/epoch_003.pt \
  --endpoint-url $S3_ENDPOINT_URL

aws s3 cp data/models/rl_v10/ppo_phase1/selfplay_v9_20260504_223016/snapshot_0019.pt \
  s3://team-builder-data/models/rl_v10/snapshot_0019_warmup_end.pt \
  --endpoint-url $S3_ENDPOINT_URL

# Download from R2 (ON the pod)
aws s3 cp s3://team-builder-data/models/bc/v10_cloud_gen9/epoch_003.pt \
  data/models/bc/v10_cloud_gen9/epoch_003.pt --endpoint-url $S3_ENDPOINT_URL
aws s3 cp s3://team-builder-data/models/rl_v10/snapshot_0019_warmup_end.pt \
  data/models/rl_v10/ppo_phase1/selfplay_v9_20260504_223016/snapshot_0019.pt \
  --endpoint-url $S3_ENDPOINT_URL
```

Total transfer: ~160 MB (~1-2 min on a typical RunPod link).

## Phase 4 — Sync procedural team data (5 min)

The procedural teambuilder needs Smogon usage stats:
```bash
# On local machine: upload one month of usage stats
aws s3 sync raw_data/pokemon_usage/2024-04 \
  s3://team-builder-data/raw_data/pokemon_usage/2024-04 \
  --endpoint-url $S3_ENDPOINT_URL

# On pod: download
aws s3 sync s3://team-builder-data/raw_data/pokemon_usage/2024-04 \
  /workspace/raw_data/pokemon_usage/2024-04 --endpoint-url $S3_ENDPOINT_URL
```

## Phase 5 — Start battle_server.js × N (1 min)

Single battle_server is enough at games_per_iter=1000 (poke-env distributes via the 4-wave hack). For more headroom, start 4:

```bash
cd pokemon-ai-starter/pokemon-ai/src
mkdir -p ../../../logs/external

# Single battle_server (sufficient for Phase 1 v2):
nohup node battle_server.js --port 9000 \
  > ../../../logs/external/battle_server_9000.log 2>&1 &

# Or 4 servers (if you want true horizontal backend distribution):
for p in 9000 9001 9002 9003; do
  nohup node battle_server.js --port $p \
    > ../../../logs/external/battle_server_$p.log 2>&1 &
done
```

Verify with `netstat -ano | grep -E ':(9000|9001)\s'` or `ss -tlnp | grep 9000`.

## Phase 6 — Launch Phase 1 v2 (~25-35 hr)

**Recommended command** (single A100, no `--mp`, with warmup re-anchor at lr=1e-5):

```bash
cd pokemon-ai-starter/pokemon-ai/src

nohup python train_rl.py \
  --init-from data/models/rl_v10/ppo_phase1/selfplay_v9_20260504_223016/snapshot_0019.pt \
  --pool-anchors data/models/bc/v10_cloud_gen9/epoch_003.pt \
  --device cuda --servers 9000,9000,9000,9000 --fp16 \
  --games-per-iter 1000 --max-concurrent 500 \
  --n-iters 200 --warmup-iters 20 \
  --lr 1e-5 --lam 0.95 --ent-coef 0.02 --reward-style terminal \
  --grad-accum 1 \
  --adaptive-entropy --adaptive-entropy-low 0.65 --adaptive-entropy-high 0.95 \
  --win-rate-mode ema --win-rate-ema-alpha 0.3 --win-rate-ema-window 50 \
  --eval-interval 20 --eval-team-set metamon-competitive --eval-games 200 \
  --snapshot-interval 5 \
  --early-stop --early-stop-patience 3 \
  --turn-cap 300 \
  --pipeline \
  --procedural-teams /workspace/raw_data/pokemon_usage/2024-04 \
  --out-dir data/models/rl_v10/ppo_phase1_v2_cloud \
  > /workspace/logs/ppo_phase1_v2.log 2>&1 &
```

### Why each flag matters here

| Flag | Why |
|---|---|
| `--lr 1e-5` | Validated stable on new arch via 4-point lr ablation (Session 50 diagnosis) |
| `--games-per-iter 1000` | 5x more samples than local Phase 1 → cleaner gradient → small lr can actually move policy |
| `--max-concurrent 500` | A100 80GB has VRAM headroom; pushes throughput. Could go higher with `--mp` later. |
| `--turn-cap 300` | Historical default. T-quadratic memory means T=1000 would be ~40 GB (slows everything; no real battle goes that long). T=300 catches the safety-net case. |
| `--warmup-iters 20` | Re-anchor value head at lr=1e-5 (sp_0019 was warmup-trained at lr=3e-5). With eval-interval=20, this puts the first eval (iter 19) cleanly at end-of-warmup before any policy training muddies the signal. |
| `--pipeline` | A100 has VRAM for bg model deepcopy without contention; cuts iter time ~30% |
| `--early-stop --early-stop-patience 3` | Fail fast if smart_avg regresses past noise threshold |
| `--servers 9000,9000,9000,9000` | 4-wave parallelism on one battle_server (validated pattern) |

### Optional: with `--mp` for max throughput

If you set up 8 battle_servers, you can use the just-refactored `--mp` for ~3x more throughput:

```
--mp --servers 9000,9001,9002,9003,9004,9005,9006,9007 \
--max-concurrent 200 --games-per-iter 4000
```

(Note: --pipeline is incompatible with --mp; pick one.) Numerical equivalence proven; runtime untested at scale. Worth doing if you want a much faster iter cadence (~3-5 min/iter) at the cost of slightly more setup and zero local validation data.

## Phase 7 — Monitor (passive, ~25-35 hr)

Tail the log:
```bash
tail -f /workspace/logs/ppo_phase1_v2.log | grep -E "Iter [0-9]+: W/L|EVAL|Snapshot|FATAL|emergency"
```

What to watch:
- **Iters 0-15** (warmup): only value_head trains, rapid v_loss drop expected. wr noisy.
- **Iter 20** (first eval): smart_avg should be ≥67%. If not, abort and regroup.
- **Iters 20-79**: trajectory should hold or slowly climb. If smart_avg trends down past noise (target_threshold=2.0), --early-stop will fire.
- **Iter 79+**: hopefully smart_avg climbs to 70-75% range. That's the "BC + PPO actually working" territory.

## Phase 8 — R2 sync at every snapshot save (recovery point)

Replaces the Network Volume approach. Bash bg loop runs alongside training and syncs the run dir to R2 frequently. Worst-case progress loss if pod dies = 1 sync interval.

```bash
# Recommended: sync every 5 minutes. Catches every snapshot (every 5 iters
# ≈ ~30-60 min) plus interim PFSP win_rates updates and config changes.
nohup bash -c '
  while true; do
    aws s3 sync data/models/rl_v10/ppo_phase1_v2_cloud/ \
      s3://team-builder-data/models/rl_v10/ppo_phase1_v2_cloud/ \
      --endpoint-url $S3_ENDPOINT_URL \
      --exclude "*" \
      --include "snapshot_*.pt" --include "*.json" --include "config.json" \
      --include "win_rates.json" --include "evals.json" 2>&1
    sleep 300
  done
' > /workspace/logs/r2_sync.log 2>&1 &
```

This syncs:
- Every saved snapshot (`.pt` files)
- PFSP win-rate state (`win_rates.json`)
- Eval results (`evals.json`)
- Run config (`config.json`)

Skipped: tensorboard event files, training logs (rebuildable / not critical for resume).

**Recovery:** if pod dies, provision a new pod, sync FROM R2, resume with `--resume <latest_snapshot>.pt`. Loss bound: ≤5 min of progress (last sync interval).

## Phase 9 — Wrap (after run completes or early-stops)

- [ ] Final scp/aws sync ALL snapshots + JSONs from pod → R2 + local
- [ ] Run final ladder eval on the best snapshot vs BC v10 e3 anchor (`eval_elo_ladder.py`)
- [ ] **Pod terminate** (NOT pause — we don't have unique pod state worth keeping)
- [ ] Update `MODEL_REGISTRY.md` with new ckpt entries
- [ ] Update `NEXT_SESSION.md` with Phase 1 v2 outcome

## Failure modes & responses

| Symptom | Likely cause | Action |
|---|---|---|
| OOM on iter 0 | conc=500 too high for some long-T episode | Lower `--max-concurrent` to 300; rerun |
| smart_avg drops in first eval | lr might still be too high for new arch | Try lr=5e-6 (half); rerun from sp_0019 |
| smart_avg flat at BC parity over 5 evals | lr too low even at games=1000 | Try lr=2e-5 OR add lr-schedule (warmup→ramp); rerun |
| Iter time > 30 min | Pipeline contention or slow battles | Drop `--pipeline`, possibly drop `--max-concurrent` |
| Pod loses connection | Network volume not mounted (catastrophic) | Re-provision with verified Network Volume; restart from R2-stored snapshot |

## Quick reference

| Item | Value |
|---|---|
| BC base ckpt | `data/models/bc/v10_cloud_gen9/epoch_003.pt` (BC v10 e3, smart_avg=67%) |
| Init-from ckpt | `data/models/rl_v10/ppo_phase1/selfplay_v9_20260504_223016/snapshot_0019.pt` (= BC + warmup) |
| Pod config | A100 SXM 80GB + Network Volume + Node.js + PyTorch 2.1+ |
| Run dir | `data/models/rl_v10/ppo_phase1_v2_cloud/<timestamp>/` |
| Expected total cost | ~$40-60 |
| Success criterion | smart_avg ≥ 70% sustained over 50+ iters; ideally Elo ≥ +20 over BC anchor in ladder eval |
