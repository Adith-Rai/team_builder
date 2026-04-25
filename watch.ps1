$logPath = Join-Path $PSScriptRoot 'system_watch.log'
$q = 'utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw'
Write-Host "Logging to $logPath (Ctrl+C to stop)" -ForegroundColor Cyan
while ($true) {
  $g = (nvidia-smi --query-gpu=$q --format=csv,noheader,nounits) -split ','
  $cpu = (Get-CimInstance Win32_Processor |
          Measure-Object LoadPercentage -Average).Average
  $os = Get-CimInstance Win32_OperatingSystem
  $rT = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
  $rF = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
  $rU = [math]::Round($rT - $rF, 1)
  $d  = Get-PSDrive C
  $dU = [math]::Round($d.Used / 1GB, 1)
  $dF = [math]::Round($d.Free / 1GB, 1)
  $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
  $line = ("{0}  GPU {1,3}% VRAM {2,5}/{3,5}MiB {4,2}C {5,5}W   CPU {6,3}%   RAM {7,4}/{8,4}GB (free {9})   Disk used {10} GB (free {11})" -f `
          $ts, $g[0].Trim(), $g[1].Trim(), $g[2].Trim(), $g[3].Trim(), $g[4].Trim(),
          $cpu, $rU, $rT, $rF, $dU, $dF)
  Add-Content -Path $logPath -Value $line -Encoding utf8
  Write-Host $line
  Start-Sleep -Seconds 5
}
