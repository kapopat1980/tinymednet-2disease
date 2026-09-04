"""
run_repeat.py -- Runs one CV repeat (or one holdout seed) and saves it.

Idempotent: a repeat whose artifact already exists is skipped. This makes the
full evaluation resumable, which matters because each repeat is a few minutes on
one core. merge_parts.py then assembles the per-repeat files into the frozen
prediction matrices that every table is computed from.

Usage:
    python run_repeat.py TASK-DIA 0 1 2
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import get_task
from experiment import ART, build_models, fit_torch, metrics, predict_torch

PART = ART / "parts"
PART.mkdir(parents=True, exist_ok=True)
# The synthetic cohort's label is a near-deterministic function of HbA1c outside a
# narrow band (results/data_audit.json), so additional rows carry almost no extra
# information. It is subsampled to keep it comparable in scale to the real cohorts
# and to fit the single-core compute budget. It is a secondary cohort throughout.
SUBSAMPLE = {"TASK-SYN": 5000}
EPOCHS = {"TASK-SYN": 25}


def subsampled(task, seed=0):
    X, y, sex = task["X"], task["y"], task.get("sex")
    n = SUBSAMPLE.get(task["name"])
    if n and n < len(y):
        i, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=0)
        X, y = X[i], y[i]
        sex = sex[i] if sex is not None else None
    return X, y, sex


def run_one(task_name, rep):
    out = PART / f"{task_name}_r{rep}.npz"
    meta_out = PART / f"{task_name}_r{rep}.json"
    # A repeat counts as done only when BOTH files exist. The predictions are
    # written before the metadata, so a run interrupted between the two leaves a
    # valid .npz with no .json; treating that as cached would satisfy this script
    # and then fail later in merge_parts.py.
    if out.exists() and meta_out.exists():
        print(f"  [{task_name} r{rep}] cached", flush=True)
        return
    if out.exists():
        print(f"  [{task_name} r{rep}] incomplete from an earlier run, redoing",
              flush=True)
        out.unlink()
    t0 = time.time()
    task = get_task(task_name)
    X, y, sex = subsampled(task)
    d = X.shape[1]
    cfgs, meta = build_models(d, seed=rep)
    epochs = EPOCHS.get(task_name, 120)

    preds = {k: np.full(len(y), np.nan) for k in cfgs}
    skf = StratifiedKFold(5, shuffle=True, random_state=1000 + rep)
    for fi, (tr, te) in enumerate(skf.split(X, y)):
        fs = rep * 1000 + fi
        imp = SimpleImputer(strategy="median").fit(X[tr])
        sc = StandardScaler().fit(imp.transform(X[tr]))
        Xtr = sc.transform(imp.transform(X[tr])).astype(np.float32)
        Xte = sc.transform(imp.transform(X[te])).astype(np.float32)

        # The distillation teacher is the same architecture as the conventional
        # wide-MLP baseline, so it is trained once and serves both roles.
        torch.manual_seed(fs)
        teacher = fit_torch(cfgs["MLP_wide"]["make"](), Xtr, y[tr], epochs=epochs, seed=fs)
        preds["MLP_wide"][te] = predict_torch(teacher, Xte)[:, 1]

        for name, cfg in cfgs.items():
            if name == "MLP_wide":
                continue
            if cfg["kind"] == "torch":
                torch.manual_seed(fs)
                m = fit_torch(cfg["make"](), Xtr, y[tr], epochs=epochs, seed=fs,
                              teacher=teacher if cfg.get("distil") else None)
                p = predict_torch(m, Xte)[:, 1]
            else:
                p = cfg["make"]().fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
            preds[name][te] = p

    extra = {} if sex is None else {"sex": np.asarray(sex, dtype="U16")}
    np.savez_compressed(out, y=y, **extra, **preds)
    meta_out.write_text(json.dumps(
        dict(task=task_name, repeat=rep, n=int(len(y)), d=int(d), epochs=epochs,
             meta=meta, seconds=round(time.time() - t0, 1),
             per_model={k: metrics(y, preds[k]) for k in preds}), indent=2), encoding="utf-8")
    print(f"  [{task_name} r{rep}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    import gc

    tname = sys.argv[1]
    for r in [int(a) for a in sys.argv[2:]]:
        run_one(tname, r)
        # Each repeat builds ~35 models and keeps cloned state dicts for early
        # stopping. Python frees them eventually, but the allocator holds the
        # arenas, and repeats later in a long run measurably slow down. Collecting
        # between repeats keeps the run flat rather than degrading.
        gc.collect()
