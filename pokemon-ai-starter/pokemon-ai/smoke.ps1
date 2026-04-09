# smoke.ps1 — PS 5.x-safe end-to-end sanity for BC policy (joint logits)
# Flow: (1) observer (one shard) → (2) scan → (3) tiny train (LSTM + hierarchical + EMA) → (4) eval → (5) checks
param(
  [switch]$CPU = $true,
  [int]$Games = 96,
  [int]$TurnCap = 200
)

$ErrorActionPreference = "Stop"

function Compose {
  param([Parameter(ValueFromRemainingArguments = $true)]$Args)
  try { docker compose @Args } catch { docker-compose @Args }
}

function InTrainer([string]$Cmd) {
  Compose exec trainer bash -lc $Cmd
}

Write-Host "Detecting python inside trainer..." -NoNewline
$out = Compose exec trainer bash -lc 'command -v python >/dev/null 2>&1 && echo python || echo python3' 2>$null
$Py  = ($out | Out-String).Trim()
if (-not $Py) { $Py = "python" }
Write-Host " using $Py"

# Device
$device = "cuda"
if (-not $CPU) { $device = "cuda" }

# Ensure output dirs exist
InTrainer "mkdir -p data/datasets/obs data/datasets/replays_smoke data/models/bc"

# ----------------------------------------------------------------------------
# 1) Observer — generate exactly ONE fresh JSONL shard
# ----------------------------------------------------------------------------
Write-Host "=== 1) Observer (generate one shard) ==="
$ts = (Get-Date).ToString("yyyyMMdd_HHmmss")
$outShard = "data/datasets/obs/obs_smoke_${ts}.jsonl"

$obsCmd = "$Py -u src/observer.py " +
          "--format gen9ou " +
          ("--games {0} " -f $Games) +
          ("--turn-cap {0} " -f $TurnCap) +
          ("--out '{0}' " -f $outShard) +
          "--bots SimpleHeuristics,SimpleHeuristics"
InTrainer $obsCmd

# ----------------------------------------------------------------------------
# 2) Dataset scan (sanity)
# ----------------------------------------------------------------------------
Write-Host "=== 2) Scan ==="
$lsout = Compose exec trainer bash -lc "ls -1t data/datasets/obs/*.jsonl 2>/dev/null | head -n 1" 2>$null
$latestJsonl = ($lsout | Out-String).Trim()
if (-not $latestJsonl) {
  Write-Host "[FAIL] No JSONL found under data/datasets/obs/. Observer might have failed." -ForegroundColor Red
  exit 2
}
InTrainer "$Py -u src/scan_jsonl.py --glob `"$latestJsonl`" --max 25000 || true"
Write-Host "Newest shard: $latestJsonl"

# ----------------------------------------------------------------------------
# 3) Train (tiny) — uses only the newest shard
# ----------------------------------------------------------------------------
Write-Host "=== 3) Train (tiny) ==="
$runName = "bc_smoke_jointlogits_$ts"

$trainCmd = "$Py -u src/bc_train.py " +
            ("--data '{0}' " -f $latestJsonl) +
            "--run-name $runName " +
            "--device $device --use-lstm --hierarchical " +
            "--lstm-hidden 256 --mlp-hidden 256 " +
            "--batch-size 256 --workers 0 --amp " +
            "--epochs 1 --steps-per-epoch 100 --lr 1e-3 " +
            "--label-smoothing 0.0 --weight-decay 1e-4 " +
            "--ema 0.999 --use-ema-for-eval"
InTrainer $trainCmd

# ----------------------------------------------------------------------------
# 4) Eval vs bots (use last checkpoint)
# ----------------------------------------------------------------------------
Write-Host "=== 4) Eval vs bots ==="
$evalCmd = "$Py -u src/eval_bc_vs_bots.py " +
           ("--ckpt-glob 'data/models/bc/{0}/epoch_*.pt' " -f $runName) +
           "--epochs last " +
           "--bots Random,MaxDamage,SimpleHeuristics,HazardSense,SetupThenSweep,GreedySE,SwitchAwareEscape " +
           ("--games {0} " -f $Games) +
           "--format gen9ou --device $device " +
           "--max-concurrent 2 " +
           "--save-replays --replays-root 'data/replays/replays_smoke' " +
           "--out-csv 'data/replays/replays_smoke/results.csv' " +
           "--out-jsonl-battles 'data/replays/replays_smoke/battles.jsonl' " +
           "--elo-baseline 1500"
InTrainer $evalCmd

# ----------------------------------------------------------------------------
# 5) Artifact checks
# ----------------------------------------------------------------------------
Write-Host "=== 5) Artifacts ==="
InTrainer ("test -f data/models/bc/{0}/epoch_*.pt && echo '[OK] checkpoint epoch_000.pt' || {{ echo '[FAIL] checkpoint missing'; exit 1; }}" -f $runName)
InTrainer ("test -f data/models/bc/{0}/latest.pt && echo '[OK] latest.pt symlink' || echo '[WARN] latest.pt missing'" -f $runName)
InTrainer ("test -s '{0}' && echo '[OK] dataset exists: {0}' || echo '[WARN] dataset empty: {0}'" -f $latestJsonl)
InTrainer ("ls -lah data/models/bc/{0} | tail -n +1" -f $runName)
InTrainer "if [ -f data/replays/replays_smoke/results.csv ]; then echo '[OK] eval results.csv'; else echo '[WARN] eval results.csv not found (ok for tiny run)'; fi"

Write-Host "=== ✅ Smoke completed ==="
