"""
external_validation.py -- Replacement for the original cross-population experiment.

Three faults in the original are corrected here.

  1. Contamination. The original trained on every non-PIMA row of the pooled
     corpus and then evaluated on a test file of which 95.0% of rows were in
     that training set. Here the two cohorts are separate datasets, so overlap
     is structurally impossible.

  2. Mismatched label spaces. The original scored a three-class model on PIMA,
     which contains two classes, so macro-F1 averaged in a structural zero for
     the absent class. That is why it reported 0.042, below chance. Both cohorts
     here carry the same binary endpoint.

  3. Mismatched features. Only variables recorded in BOTH cohorts are used
     (age, BMI, glucose), so transfer is not confounded with feature
     availability.

Both directions are reported, each against a within-cohort reference trained
and tested on the target cohort, since a transfer number is uninterpretable
without knowing what is achievable in-cohort.

A caveat that belongs in the manuscript: the glucose variable is not the same
measurement in the two cohorts. PIMA records a 2-hour oral glucose tolerance
test; the Kaggle cohort records an unspecified blood glucose level. Any transfer
gap therefore combines population shift with measurement shift and cannot
separate them.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_pima, load_synthetic
from experiment import fit_torch, metrics, predict_torch
from models import TinyMedNet

ROOT = Path(__file__).resolve().parents[1]
SHARED = ["age", "bmi", "glucose"]
N_SEEDS = 3
N_TRAIN = 5000      # comparator training size; its labels are near-deterministic in HbA1c
EPOCHS_BIG = 40     # epoch cap when training on the comparator cohort


def cohorts(match_eligibility=False):
    pima, _ = load_pima()
    syn, _ = load_synthetic()
    if match_eligibility:
        # PIMA enrolled women aged 21+. Restricting the comparator to the same
        # eligibility criteria separates population shift from cohort composition.
        syn = syn[(syn.sex == "Female") & (syn.age >= 21)].copy()
    P = dict(name="PIMA", X=pima[SHARED].to_numpy(float), y=pima["y"].to_numpy(int))
    S = dict(name="Kaggle-synthetic", X=syn[SHARED].to_numpy(float), y=syn["y"].to_numpy(int))
    return P, S


def _prep(Xtr, Xte):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    return (sc.transform(imp.transform(Xtr)).astype(np.float32),
            sc.transform(imp.transform(Xte)).astype(np.float32))


def _fit_predict(kind, Xtr, ytr, Xte, seed):
    if kind == "TinyMedNet":
        torch.manual_seed(seed)
        ep = EPOCHS_BIG if len(ytr) > 2000 else 120
        m = fit_torch(TinyMedNet(Xtr.shape[1]), Xtr, ytr, seed=seed, epochs=ep)
        return predict_torch(m, Xte)[:, 1]
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def transfer(src, dst, kind, n_train=N_TRAIN, seed=0):
    Xs, ys = src["X"], src["y"]
    if len(ys) > n_train:
        i, _ = train_test_split(np.arange(len(ys)), train_size=n_train,
                                stratify=ys, random_state=seed)
        Xs, ys = Xs[i], ys[i]
    Xtr, Xte = _prep(Xs, dst["X"])
    p = _fit_predict(kind, Xtr, ys, Xte, seed)
    return metrics(dst["y"], p)


def within(coh, kind, n_max=N_TRAIN, seed=0, n_splits=5):
    X, y = coh["X"], coh["y"]
    if len(y) > n_max:
        i, _ = train_test_split(np.arange(len(y)), train_size=n_max,
                                stratify=y, random_state=seed)
        X, y = X[i], y[i]
    out = []
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        Xtr, Xte = _prep(X[tr], X[te])
        p = _fit_predict(kind, Xtr, y[tr], Xte, seed)
        out.append(metrics(y[te], p))
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def run(which=(False, True), outfile="external_validation.json"):
    res = {}
    prev = ROOT / "results" / outfile
    if prev.exists():
        res = json.loads(prev.read_text(encoding="utf-8"))
    for matched in which:
        P, S = cohorts(match_eligibility=matched)
        tag = "eligibility_matched" if matched else "unrestricted"
        block = {"comparator_n": int(len(S["y"])),
                 "comparator_prevalence": float(S["y"].mean()),
                 "pima_n": int(len(P["y"])),
                 "pima_prevalence": float(P["y"].mean())}
        for kind in ["TinyMedNet", "LogisticRegression"]:
            runs = {"synthetic_to_pima": [], "pima_to_synthetic": []}
            for s in range(N_SEEDS):
                runs["synthetic_to_pima"].append(transfer(S, P, kind, seed=s))
                runs["pima_to_synthetic"].append(transfer(P, S, kind, seed=s))
            block[kind] = {
                d: {k: dict(mean=float(np.mean([r[k] for r in v])),
                            sd=float(np.std([r[k] for r in v], ddof=1)))
                    for k in v[0]}
                for d, v in runs.items()
            }
            block[kind]["within_pima"] = within(P, kind)
            block[kind]["within_synthetic"] = within(S, kind)
            print(f"  [{tag}/{kind}] done", flush=True)
        res[tag] = block

    res["_note"] = ("Shared features: age, BMI, glucose. PIMA glucose is a 2-hour OGTT "
                    "value; the comparator records an unspecified blood glucose level, "
                    "so transfer gaps confound population and measurement shift.")
    with open(ROOT / "results" / outfile, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"written results/{outfile}")


if __name__ == "__main__":
    sel = sys.argv[1] if len(sys.argv) > 1 else "both"
    run(which={"unrestricted": (False,), "matched": (True,), "both": (False, True)}[sel])
