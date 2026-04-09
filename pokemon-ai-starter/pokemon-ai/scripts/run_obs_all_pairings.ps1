# scripts/run_obs_all_pairings.ps1
# Run observer.py for ALL ordered bot-vs-bot pairings, in parallel, from the HOST.

param(
  # Edit if your repo lives somewhere else:
  [string] $ProjectDir = "C:\Users\raiad\OneDrive\Desktop\team_builder\pokemon-ai-starter\pokemon-ai",

  # Poke format and generation settings
  [string] $Format = "gen9ou",
  [int]    $GamesPerPair = 500,
  [int]    $MaxConcurrentBattlesPerRun = 2,
  [int]    $TurnCap = 300,
  [int]    $BattleTimeout = 900,
  [int]    $StepIdle = 120,

  # Parallelism across pairings (PowerShell jobs)
  [int]    $MaxParallelRuns = 6  # tune based on CPU/RAM
)

# All bots we support via observer.py (display names):
$Bots = @(
  "MaxDamage",
  "SimpleHeuristics",
  "GreedySE",
  "HazardSense",
  "SwitchAwareEscape",
  "SetupThenSweep",
  "Random"
)

# Compose command to run observer.py inside the *trainer* container
function Start-ObsJob {
  param(
    [string] $A,
    [string] $B,
    [string] $Ts
  )
  $outPath = "data/datasets/obs/obs_${Format}_${A}-vs-${B}_${Ts}.jsonl"

  # Build the inner bash command (no $(date ...) — we precompute $Ts on host)
  $bash = "python3 -u src/observer.py --format $Format --games $GamesPerPair --bots `"$A,$B`" --max-concurrent $MaxConcurrentBattlesPerRun --turn-cap $TurnCap --battle-timeout $BattleTimeout --step-idle-timeout $StepIdle --out $outPath"

  $dockerArgs = @(
    "--project-directory", $ProjectDir,
    "exec", "trainer", "bash", "-lc", $bash
  )

  Write-Host "[spawn] $A vs $B"
  Start-Job -Name "$A-vs-$B" -ScriptBlock {
    param($argsList)
    & docker compose @argsList
  } -ArgumentList (,$dockerArgs)
}

# ---- main ----
$ts = Get-Date -UFormat %Y%m%d_%H%M%S
Write-Host "Launching all ordered pairings for $Format at $ts ..."

# Create all ordered pairs (including mirrors)
$Pairs = foreach ($A in $Bots) { foreach ($B in $Bots) { [PSCustomObject]@{ A=$A; B=$B } } }

# Launch jobs with a throttle on parallelism
$Jobs = @()
foreach ($p in $Pairs) {
  # Backpressure: wait until < MaxParallelRuns jobs are running
  while ( ($Jobs | Where-Object { $_.State -eq 'Running' }).Count -ge $MaxParallelRuns ) {
    # surface any output from finished jobs while we wait
    $done = Wait-Job -Job $Jobs -Any -Timeout 1
    if ($done) {
      Receive-Job -Job $done -Keep | Write-Host
    }
  }
  $Jobs += Start-ObsJob -A $p.A -B $p.B -Ts $ts
}

# Drain: stream logs as jobs finish
while ( ($Jobs | Where-Object { $_.State -in @('NotStarted','Running') }).Count -gt 0 ) {
  $done = Wait-Job -Job $Jobs -Any -Timeout 2
  if ($done) {
    Receive-Job -Job $done -Keep | Write-Host
  }
}

# Final receive for anything left buffered
$Jobs | ForEach-Object {
  Receive-Job -Job $_ -Keep | Write-Host
}

Write-Host "All pairings finished. Summary:"
$Jobs | Select-Object Name, Id, State | Format-Table -AutoSize
