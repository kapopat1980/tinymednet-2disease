<#
verify_all.ps1 -- Windows (PowerShell) runner for the full pipeline.

    .\verify_all.ps1              full run, roughly 2 hours on one core
    .\verify_all.ps1 -Quick       skips the 20-seed quantization study (~25 min)

Everything is logged to results\verification_run.log. The run is resumable:
run_repeat.py and quantization.py skip work that is already done, so if you
interrupt it, just start it again.

If PowerShell refuses to run this script, it is the execution policy, not an
error in the script. Either:
    powershell -ExecutionPolicy Bypass -File .\verify_all.ps1
or, once per user:
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#>
param([switch]$Quick)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# The pipeline writes ±, γ, ≥ and similar characters. Without this, Windows
# consoles default to cp1252 and printing them raises UnicodeEncodeError.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
# PyTorch and XGBoost each ship their own OpenMP runtime on Windows. Loading both
# in one process otherwise aborts with "OMP: Error #15".
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
# See the note in src/experiment.py: threading hurts badly at this model size.
$env:OMP_NUM_THREADS = "1"

# 'python' on Windows, 'python3' elsewhere; prefer whichever resolves.
$PY = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" }
      elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" }
      elseif (Get-Command py -ErrorAction SilentlyContinue) { "py" }
      else { throw "No Python interpreter found on PATH." }

# Preflight: a PowerShell session keeps an activated virtual environment on PATH
# even after the environment itself is deleted, which happens if the repository
# folder is replaced. The interpreter then fails with "failed to locate
# pyvenv.cfg", which does not obviously point at the cause.
# 2>$null does not suppress a native program's stderr in PowerShell; it must be
# merged into the success stream first and then discarded.
& $PY -c "import sys" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host @"

The Python interpreter on PATH is not usable.

This usually means a virtual environment is still activated in this window but
its folder no longer exists, typically after re-extracting the repository.

To fix it:
  1. Close this PowerShell window and open a new one.
  2. cd to this folder.
  3. Recreate the environment:
         python -m venv .venv
         .\.venv\Scripts\Activate.ps1
         pip install -r requirements-core.txt
  4. Run this script again.

Nothing is lost: completed repeats in artifacts\parts are reused.
"@ -ForegroundColor Yellow
    exit 1
}

New-Item -ItemType Directory -Force -Path results, artifacts | Out-Null
$LOG = "results\verification_run.log"
"" | Set-Content -Path $LOG -Encoding utf8

function Say($msg) {
    $line = "`n=== $msg ==="
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $LOG -Value $line -Encoding utf8
}

function Run {
    param([string[]]$CmdArgs)
    $shown = "`n> $PY $($CmdArgs -join ' ')"
    Write-Host $shown -ForegroundColor DarkGray
    Add-Content -Path $LOG -Value $shown -Encoding utf8
    # Native stderr arrives as ErrorRecord objects; PowerShell renders only the
    # first line of those and wraps it in NativeCommandError, which hides Python
    # tracebacks. Casting each to string preserves the whole traceback.
    & $PY @CmdArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $LOG -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nStep failed: $PY $($CmdArgs -join ' ')" -ForegroundColor Red
        Write-Host "Full output is in $LOG" -ForegroundColor Red
        throw "Step failed: $($CmdArgs -join ' ')"
    }
}

Say "environment"
& $PY -c @"
import sys, platform
print('python', sys.version.split()[0], platform.system(), platform.machine())
import torch, sklearn, numpy, pandas, scipy, xgboost
for n, m in [('torch',torch),('sklearn',sklearn),('numpy',numpy),
             ('pandas',pandas),('scipy',scipy),('xgboost',xgboost)]:
    print(n, m.__version__)
try:
    import tensorflow as tf; print('tensorflow', tf.__version__)
except Exception as e:
    print('tensorflow MISSING - step 5 will be skipped')
import os; print('cores:', os.cpu_count())
print('quantized backends:', torch.backends.quantized.supported_engines)
"@ 2>&1 | Tee-Object -FilePath $LOG -Append

Say "1. provenance audit"
Run @("src\data.py")

Say "2. withdrawn pooled task: leakage controls"
Run @("verify_findings.py")

Say "3. main evaluation (10 repeats per cohort)"
foreach ($t in @("TASK-DIA", "TASK-CKD", "TASK-SYN")) {
    Run @("src\run_repeat.py", $t, "0","1","2","3","4","5","6","7","8","9")
}
Run @("src\merge_parts.py")

if (-not $Quick) {
    Say "4. quantization study (20 seeds per cohort)"
    $chunks = @(@("0","1","2","3","4"), @("5","6","7","8","9"),
                @("10","11","12","13","14"), @("15","16","17","18","19"))
    foreach ($t in @("TASK-DIA", "TASK-CKD")) {
        foreach ($c in $chunks) { Run (@("src\quantization.py", $t) + $c) }
    }
} else {
    Say "4. quantization study SKIPPED (-Quick)"
}

Say "5. INT8 export"
# As with the preflight: 2>$null does not suppress a native program's stderr, so
# the ImportError traceback printed and, under ErrorActionPreference=Stop, ended
# the run. TensorFlow being absent is an expected, non-fatal condition.
& $PY -c "import tensorflow" 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Run @("src\tflite_export.py")
    Get-ChildItem artifacts\*.tflite | Format-Table Name, Length |
        Out-String | Tee-Object -FilePath $LOG -Append
} else {
    "tensorflow not installed; skipping" | Tee-Object -FilePath $LOG -Append
}

Say "6. analysis, controls, figures"
Run @("src\analysis.py")
Run @("src\controls_and_figures.py")

Say "7. regenerate tables and manuscript"
Run @("src\make_tables.py")
Run @("src\build_manuscript.py")
Run @("src\to_springer.py")

Say "8. compare against reported values"
Run @("src\check_expected.py")

Say "done"
Write-Host "Full log: $LOG" -ForegroundColor Green
