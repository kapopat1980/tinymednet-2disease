<#
run_parallel.ps1 -- Runs the evaluation repeats across several processes.

Each repeat writes its own files in artifacts\parts and shares no state with the
others, so they can run concurrently. Each process is pinned to a single torch
thread, which is what makes this worthwhile: the model is small enough that
multi-threading a single fit is far slower than running several fits at once.

    .\run_parallel.ps1                 # 4 workers
    .\run_parallel.ps1 -Workers 8      # 8 workers on a 16-core machine

Rerunning is safe and cheap: completed repeats are skipped. When it finishes,
continue with verify_all.ps1, which will find the work already done.
#>
param([int]$Workers = 4)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "1"

$PY = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" }
      else { "python" }

# Every (cohort, repeat) pair that the paper reports.
$jobs = @()
foreach ($t in @("TASK-DIA", "TASK-CKD", "TASK-SYN")) {
    foreach ($r in 0..9) { $jobs += ,@($t, $r) }
}

Write-Host "Dispatching $($jobs.Count) repeats across $Workers workers." -ForegroundColor Cyan
Write-Host "Completed repeats are skipped, so this is safe to re-run.`n"

$running = @()
foreach ($j in $jobs) {
    while ($running.Count -ge $Workers) {
        $running = @($running | Where-Object { -not $_.HasExited })
        Start-Sleep -Milliseconds 400
    }
    $t, $r = $j
    $p = Start-Process -FilePath $PY `
        -ArgumentList @("src\run_repeat.py", $t, "$r") `
        -NoNewWindow -PassThru
    Write-Host ("  started {0} repeat {1}  (pid {2})" -f $t, $r, $p.Id)
    $running += $p
}

Write-Host "`nWaiting for the last workers to finish..."
foreach ($p in $running) { $p.WaitForExit() }

Write-Host "`nAll repeats dispatched. Verifying completeness:" -ForegroundColor Cyan
& $PY -c @"
from pathlib import Path
part = Path('artifacts/parts')
ok = True
for t in ['TASK-DIA', 'TASK-CKD', 'TASK-SYN']:
    done = sorted(r for r in range(10)
                  if (part / f'{t}_r{r}.npz').exists()
                  and (part / f'{t}_r{r}.json').exists())
    missing = [r for r in range(10) if r not in done]
    print(f'  {t}: {len(done)}/10 complete' + (f'  MISSING {missing}' if missing else ''))
    ok = ok and not missing
print('\nAll repeats present.' if ok else
      '\nSome repeats are missing. Re-run this script; it will only redo those.')
"@

Write-Host "`nNext: powershell -ExecutionPolicy Bypass -File .\verify_all.ps1" -ForegroundColor Green
