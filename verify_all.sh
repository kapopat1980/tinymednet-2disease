#!/usr/bin/env bash
#
# verify_all.sh -- Runs the full pipeline end to end and logs everything.
#
# This exists so that each author can reproduce the reported results with one
# command rather than eight, and so that the run produces an execution log that
# can be shown to reviewers.
#
#   bash verify_all.sh              full run, roughly 2 hours on one core
#   bash verify_all.sh --quick      skips the 20-seed quantization study (~25 min)
#
# Everything is written to results/verification_run.log. The script never
# overwrites artifacts that already exist: run_repeat.py and quantization.py
# skip work that is already done, so an interrupted run can simply be restarted.

set -euo pipefail
cd "$(dirname "$0")"

QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

LOG=results/verification_run.log
mkdir -p results artifacts
: > "$LOG"

say() { printf '\n=== %s ===\n' "$1" | tee -a "$LOG"; }
run() { printf '\n$ %s\n' "$*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

export OMP_NUM_THREADS=1   # see the note in src/experiment.py

say "environment"
{
  date -u
  python3 -c "import sys; print('python', sys.version.split()[0])"
  python3 -c "import torch, sklearn, numpy, pandas, scipy, xgboost; \
print('torch', torch.__version__); print('sklearn', sklearn.__version__); \
print('numpy', numpy.__version__); print('pandas', pandas.__version__); \
print('scipy', scipy.__version__); print('xgboost', xgboost.__version__)"
  python3 -c "import tensorflow as tf; print('tensorflow', tf.__version__)" 2>/dev/null \
    || echo "tensorflow MISSING (step 5 will be skipped)"
  echo "cores: $(python3 -c 'import os; print(os.cpu_count())')"
} 2>&1 | tee -a "$LOG"

say "1. provenance audit"
run python3 src/data.py

say "2. withdrawn pooled task: leakage controls"
run python3 verify_findings.py

say "3. main evaluation (10 repeats per cohort)"
for t in TASK-DIA TASK-CKD TASK-SYN; do
  run python3 src/run_repeat.py "$t" 0 1 2 3 4 5 6 7 8 9
done
run python3 src/merge_parts.py

if [[ $QUICK -eq 0 ]]; then
  say "4. quantization study (20 seeds per cohort)"
  for t in TASK-DIA TASK-CKD; do
    for s in "0 1 2 3 4" "5 6 7 8 9" "10 11 12 13 14" "15 16 17 18 19"; do
      # shellcheck disable=SC2086
      run python3 src/quantization.py "$t" $s
    done
  done
else
  say "4. quantization study SKIPPED (--quick)"
fi

say "5. INT8 export"
if python3 -c "import tensorflow" 2>/dev/null; then
  run python3 src/tflite_export.py
  ls -l artifacts/*.tflite | tee -a "$LOG"
else
  echo "tensorflow not installed; skipping" | tee -a "$LOG"
fi

say "6. analysis, controls, figures"
run python3 src/analysis.py
run python3 src/controls_and_figures.py

say "7. regenerate tables and manuscript"
run python3 src/make_tables.py
run python3 src/build_manuscript.py
run python3 src/to_springer.py

say "8. compare against reported values"
run python3 src/check_expected.py

say "done"
echo "Full log: $LOG" | tee -a "$LOG"
