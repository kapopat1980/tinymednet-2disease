"""run_all.py -- Full evaluation. Writes frozen per-fold predictions to artifacts/."""
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import audit_pooled_leakage, audit_synthetic_signatures, get_task
from experiment import ART, run_cv, run_holdout

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "results").mkdir(exist_ok=True)

if __name__ == "__main__":
    t0 = time.time()

    syn = audit_synthetic_signatures()
    leak = audit_pooled_leakage()
    leak["feature_availability_by_source"] = leak["feature_availability_by_source"].to_dict("records")
    with open(ROOT / "results" / "data_audit.json", "w", encoding="utf-8") as f:
        json.dump({"synthetic_signatures": syn, "pooled_leakage": leak}, f, indent=2)
    print("[audit] written")

    for name, kw in [
        ("TASK-DIA", dict(n_splits=5, n_repeats=10)),
        ("TASK-CKD", dict(n_splits=5, n_repeats=10)),
    ]:
        print(f"[{name}] repeated stratified CV {kw}", flush=True)
        run_cv(get_task(name), **kw)
        print(f"[{name}] done ({time.time() - t0:.0f}s)", flush=True)

    # Secondary, clearly-labelled synthetic cohort. Subsampled: this cohort's labels
    # are a near-deterministic function of HbA1c (see results/data_audit.json), so
    # additional rows buy no additional information, and the container has one core.
    print("[TASK-SYN] holdout, 3 seeds, 20k subsample", flush=True)
    run_holdout(get_task("TASK-SYN"), n_seeds=3, epochs=20, subsample=20000)
    print(f"[TASK-SYN] done ({time.time() - t0:.0f}s)", flush=True)
    print(f"ALL DONE in {time.time() - t0:.0f}s")
