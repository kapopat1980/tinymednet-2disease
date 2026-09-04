# Running this on Windows

Everything in this repository runs on Windows. Three things differ from the
Linux instructions, and one of them matters.

## 1. Setup

Open **PowerShell** in the extracted folder.

```powershell
python --version          # check this first
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-core.txt
```

**Check your Python version before anything else.** TensorFlow lags new Python
releases by roughly a year. On Python 3.13 or newer, `tensorflow-cpu` has no
wheels and `pip install -r requirements.txt` fails outright — and because pip
resolves the whole file before installing anything, a single unavailable package
means nothing gets installed at all.

That is why the requirements are split:

| File | Contents | Needed for |
|---|---|---|
| `requirements-core.txt` | numpy, pandas, scipy, scikit-learn, torch, xgboost, matplotlib | everything except the INT8 export |
| `requirements-export.txt` | tensorflow-cpu | `src/tflite_export.py` only |
| `requirements.txt` | both, pinned | the exact environment used for the paper |

Install the core file. The runner detects TensorFlow's absence and skips step 5
automatically; every other result reproduces.

If `Activate.ps1` is blocked, run PowerShell once as:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Use `python`, not `python3` — on Windows `python3` is often a stub that opens
the Microsoft Store.

## 2. Run it

```powershell
.\verify_all.ps1              # full run, about 2 hours on one core
.\verify_all.ps1 -Quick       # skips the 20-seed quantization study, ~25 min
```

If the script is blocked:

```powershell
powershell -ExecutionPolicy Bypass -File .\verify_all.ps1
```

The run is resumable. `run_repeat.py` and `quantization.py` skip work already
done, so if you close the window or the machine sleeps, start it again and it
picks up where it stopped.

At any point you can check results on their own:

```powershell
python src\check_expected.py
```

## 3. Character encoding — the one that matters

The pipeline writes `±`, `γ`, `Δ`, `≥` and `→` into the generated tables and
manuscript. Windows defaults to cp1252 rather than UTF-8, which would otherwise
cause `UnicodeDecodeError` when the scripts read those files back, and
`UnicodeEncodeError` when printing to the console.

Every file operation in this repository now specifies `encoding="utf-8"`
explicitly, and `verify_all.ps1` sets `PYTHONUTF8=1` and
`PYTHONIOENCODING=utf-8`. If you run the Python scripts directly rather than
through the PowerShell runner, set these first:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

## Expected differences on Windows

**Quantization backends differ.** Linux x86 builds of PyTorch ship `fbgemm` and
`qnnpack`; Windows builds commonly ship `onednn` only. The study now detects
whichever engines your build provides and uses those, and `check_expected.py`
matches any validated backend rather than a named one. Verified on `onednn`:
PTQ 0.8236 against FP32 0.8205 on PIMA, the same conclusion the paper reports.
The environment block at the top of the log prints your available engines.

**TensorFlow and new Python versions.** TensorFlow publishes no wheels for
Python 3.13 or 3.14 at the time of writing. If `pip install -r
requirements-export.txt` fails with *"Could not find a version that satisfies
the requirement"*, that is why — it is not a broken install.

The cleanest fix is a second virtual environment on an older Python, used only
for the export step:

```powershell
# install Python 3.12 from python.org, then
py -3.12 -m venv .venv-export
.\.venv-export\Scripts\Activate.ps1
pip install -r requirements-export.txt
pip install numpy pandas scipy scikit-learn torch
python src\tflite_export.py
deactivate
```

The export writes `results\tflite_export.json` and the `.tflite` files, which
the rest of the pipeline reads from disk, so the two environments do not need to
interact.

Alternatives, if you prefer: run that one step under WSL2 (`wsl --install`, then
follow the Linux instructions inside it), or skip it. If you skip it you cannot
reproduce the 19.59 KiB deployment figure yourself, so **at least one author
should verify that step** on an older Python, WSL2, or Linux before submission.

**Rebuilding the Word files** (step 7 onward) additionally needs `pandoc` and,
for the PDF preview, LibreOffice. Neither is required to verify the numbers —
they only rebuild the manuscript from the generated tables. Install pandoc with
`winget install --id JohnMacFarlane.Pandoc` if you want it.

## Re-extracting the repository over an existing folder

If you replace the repository folder while a virtual environment is activated in
an open PowerShell window, the next command fails with:

```
failed to locate pyvenv.cfg: The system cannot find the file specified.
```

The shell still has the old `.venv\Scripts` on PATH, but the folder is gone.
Nothing is corrupted and no results are lost. Close the window, open a new one,
and recreate the environment:

```powershell
cd C:\path\to\repo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-core.txt
```

To avoid this entirely, extract updates into a **new** folder and copy
`artifacts\parts` across, or keep `.venv` outside the repository folder.

Completed repeats in `artifacts\parts` are always reused, so re-running the
script never repeats finished work.

## Making it faster on a multi-core machine

Each fit is pinned to one thread deliberately: TinyMed-Net has 3,942 parameters,
and at that size synchronising threads costs far more than the arithmetic
(measured: 0.78 s on one thread against 31 s on four for the same work). That
leaves the other cores idle.

Repeats are independent and each writes its own files, so run several at once:

```powershell
.\run_parallel.ps1 -Workers 8      # on a 16-core machine
```

This dispatches all 30 repeats across 8 processes, each single-threaded, then
reports which are complete. It is safe to interrupt and re-run: finished repeats
are skipped. Afterwards run `verify_all.ps1` as normal — it will find the work
already done and move straight on to the quantization study.

Expect roughly an eightfold reduction in wall time for step 3.

## Long runs

The synthetic cohort takes roughly four minutes per repeat, so step 3 dominates.
Two practical notes:

- Stop Windows sleeping mid-run: `powercfg /change standby-timeout-ac 0`
- Your machine almost certainly has more cores than the single core these
  timings assume, so expect the run to be substantially faster.
