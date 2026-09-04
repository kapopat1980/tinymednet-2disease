"""
experiment.py -- Training and evaluation engine.

Protocol decisions that address the reviewer comments directly:

  * Repeated stratified k-fold CV rather than one split and one seed, so every
    reported number carries an interval rather than a point estimate.
  * Imputation and scaling are fitted on the training fold only, inside the
    loop. No statistic crosses the fold boundary.
  * Class imbalance is handled with a weighted loss rather than SMOTE.
    SMOTE synthesises points from neighbours and is easy to apply before the
    split by accident; a weighted loss cannot leak.
  * Every fold writes its out-of-fold probabilities to disk. All tables and all
    calibration numbers are computed later from those frozen files, so the
    calibration analysis necessarily describes the same fitted models as the
    discrimination analysis.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, matthews_corrcoef,
                             roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from models import MLP, TinyMedNet, count_params, matched_mlp, wide_mlp

# TinyMed-Net has 3,942 parameters and trains on batches of 64. At that size the
# cost of synchronising OpenMP threads dwarfs the arithmetic: measured here,
# 150 forward+backward passes take 0.78 s on one thread and 31 s on four. Torch
# otherwise defaults to one thread per core, so on a 16-core machine the run is
# an order of magnitude slower than on a single core. scikit-learn and XGBoost
# are left alone; their n_jobs settings still use all cores.
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)


# ------------------------------------------------------------------ metrics
def ece(probs, y, n_bins=10):
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def metrics(y, p1, probs=None):
    """y: labels. p1: P(positive). probs: full (n,2) matrix for ECE."""
    pred = (p1 >= 0.5).astype(int)
    if probs is None:
        probs = np.column_stack([1 - p1, p1])
    return dict(
        f1_pos=float(f1_score(y, pred, zero_division=0)),
        f1_macro=float(f1_score(y, pred, average="macro", zero_division=0)),
        mcc=float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        auroc=float(roc_auc_score(y, p1)) if len(np.unique(y)) > 1 else float("nan"),
        auprc=float(average_precision_score(y, p1)),
        bal_acc=float(balanced_accuracy_score(y, pred)),
        brier=float(brier_score_loss(y, p1)),
        ece=ece(probs, y),
    )


# ------------------------------------------------------------------ torch fit
def fit_torch(model, Xtr, ytr, epochs=120, lr=1e-3, bs=64, teacher=None,
              T=3.0, alpha=0.5, seed=0, class_weight=True,
              val_frac=0.15, patience=25):
    """
    Trains with early stopping on a stratified inner validation split carved out
    of the training data. The number of epochs is therefore selected per fit
    without ever consulting the test fold. An unregularised fit on these cohorts
    reaches training AUROC 1.00 within ~80 epochs while test AUROC falls, so
    fixing an epoch budget in advance (as the original code did) systematically
    overfits the smaller cohorts.
    """
    g = torch.Generator().manual_seed(seed)
    use_val = val_frac and len(ytr) > 60 and np.bincount(ytr, minlength=2).min() >= 8
    if use_val:
        idx_tr, idx_va = train_test_split(
            np.arange(len(ytr)), test_size=val_frac, stratify=ytr, random_state=seed)
    else:
        idx_tr = np.arange(len(ytr))
        idx_va = None

    Xt = torch.tensor(Xtr[idx_tr], dtype=torch.float32)
    yt = torch.tensor(ytr[idx_tr], dtype=torch.long)
    if idx_va is not None:
        Xv = torch.tensor(Xtr[idx_va], dtype=torch.float32)
        yv = ytr[idx_va]

    if class_weight:
        cnt = np.bincount(ytr[idx_tr], minlength=2).astype(float)
        w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), dtype=torch.float32)
    else:
        w = None
    ce = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = len(yt)
    best = (-np.inf, 0, None)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = ce(out, yt[idx])
            if teacher is not None:
                with torch.no_grad():
                    tl = teacher(Xt[idx])
                kd = F.kl_div(F.log_softmax(out / T, 1),
                              F.softmax(tl / T, 1), reduction="batchmean") * T * T
                loss = alpha * loss + (1 - alpha) * kd
            loss.backward()
            opt.step()
        sched.step()
        if idx_va is not None:
            model.eval()
            with torch.no_grad():
                pv = torch.softmax(model(Xv), 1).numpy()[:, 1]
            score = roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else 0.0
            if score > best[0]:
                best = (score, ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
            elif ep - best[1] >= patience:
                break
    if best[2] is not None:
        model.load_state_dict(best[2])
    model.eval()
    return model


@torch.no_grad()
def predict_torch(model, X):
    model.eval()
    return torch.softmax(model(torch.tensor(X, dtype=torch.float32)), 1).numpy()


# ------------------------------------------------------------------ configs
def build_models(d, seed):
    tm_params = count_params(TinyMedNet(d))
    mm, hid, mm_params = matched_mlp(d, tm_params)
    return {
        "TinyMedNet":            dict(kind="torch", make=lambda: TinyMedNet(d), distil=True),
        "TinyMedNet_noSE":       dict(kind="torch", make=lambda: TinyMedNet(d, use_se=False), distil=True),
        "TinyMedNet_noResidual": dict(kind="torch", make=lambda: TinyMedNet(d, use_residual=False), distil=True),
        "TinyMedNet_noDistil":   dict(kind="torch", make=lambda: TinyMedNet(d), distil=False),
        "MLP_matched":           dict(kind="torch", make=lambda: matched_mlp(d, tm_params)[0], distil=False),
        "MLP_wide":              dict(kind="torch", make=lambda: wide_mlp(d), distil=False),
        "LogisticRegression":    dict(kind="sk", make=lambda: LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest":          dict(kind="sk", make=lambda: RandomForestClassifier(
                                     n_estimators=200, max_depth=12, class_weight="balanced",
                                     random_state=seed, n_jobs=-1)),
        "XGBoost":               dict(kind="sk", make=lambda: XGBClassifier(
                                     n_estimators=200, max_depth=6, learning_rate=0.1,
                                     eval_metric="logloss", random_state=seed, n_jobs=-1,
                                     verbosity=0)),
    }, dict(tinymednet_params=tm_params, matched_mlp_params=mm_params, matched_mlp_hidden=hid)


# ------------------------------------------------------------------ CV driver
def run_cv(task, n_splits=5, n_repeats=5, epochs=80, seed=0, models=None):
    X, y, d = task["X"], task["y"], task["X"].shape[1]
    cfgs, meta = build_models(d, seed)
    if models:
        cfgs = {k: v for k, v in cfgs.items() if k in models}

    rows, oof = [], {k: np.full((n_repeats, len(y)), np.nan) for k in cfgs}
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    t0 = time.time()
    for fi, (tr, te) in enumerate(rskf.split(X, y)):
        rep = fi // n_splits
        fold_seed = seed * 1000 + fi

        imp = SimpleImputer(strategy="median").fit(X[tr])
        sc = StandardScaler().fit(imp.transform(X[tr]))
        Xtr = sc.transform(imp.transform(X[tr])).astype(np.float32)
        Xte = sc.transform(imp.transform(X[te])).astype(np.float32)
        ytr, yte = y[tr], y[te]

        teacher = None
        if any(c.get("distil") for c in cfgs.values()):
            torch.manual_seed(fold_seed)
            teacher = fit_torch(wide_mlp(d), Xtr, ytr, epochs=epochs, seed=fold_seed)

        for name, cfg in cfgs.items():
            if cfg["kind"] == "torch":
                torch.manual_seed(fold_seed)
                m = fit_torch(cfg["make"](), Xtr, ytr, epochs=epochs, seed=fold_seed,
                              teacher=teacher if cfg.get("distil") else None)
                p = predict_torch(m, Xte)
            else:
                m = cfg["make"]().fit(Xtr, ytr)
                p = m.predict_proba(Xte)
            oof[name][rep, te] = p[:, 1]
            r = metrics(yte, p[:, 1], p)
            r.update(model=name, repeat=rep, fold=fi % n_splits)
            rows.append(r)
        if fi % n_splits == n_splits - 1:
            print(f"    repeat {rep + 1}/{n_repeats} done ({time.time() - t0:.0f}s)", flush=True)

    np.savez_compressed(ART / f"oof_{task['name']}.npz", y=y, **oof)
    with open(ART / f"cv_{task['name']}.json", "w", encoding="utf-8") as f:
        json.dump(dict(task=task["name"], real=task["real"], n=len(y), d=d,
                       n_splits=n_splits, n_repeats=n_repeats, epochs=epochs, seed=seed,
                       meta=meta, folds=rows), f, indent=2)
    return rows, meta


def run_holdout(task, n_seeds=10, epochs=25, seed=0, test_size=0.2, val_size=0.1,
                models=None, subsample=None):
    """Single stratified split, repeated over seeds. Used for the large synthetic cohort."""
    X, y, d = task["X"], task["y"], task["X"].shape[1]
    sex = task.get("sex")
    if subsample and subsample < len(y):
        idx, _ = train_test_split(np.arange(len(y)), train_size=subsample,
                                  stratify=y, random_state=seed)
        X, y = X[idx], y[idx]
        sex = sex[idx] if sex is not None else None
    cfgs, meta = build_models(d, seed)
    if models:
        cfgs = {k: v for k, v in cfgs.items() if k in models}

    idx_all = np.arange(len(y))
    itr, ite = train_test_split(idx_all, test_size=test_size, stratify=y, random_state=seed)
    Xtr, Xte, ytr, yte = X[itr], X[ite], y[itr], y[ite]
    sex_te = sex[ite] if sex is not None else None
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    Xtr_s = sc.transform(imp.transform(Xtr)).astype(np.float32)
    Xte_s = sc.transform(imp.transform(Xte)).astype(np.float32)

    rows, preds = [], {}
    for s in range(n_seeds):
        fs = seed * 100 + s
        teacher = None
        if any(c.get("distil") for c in cfgs.values()):
            torch.manual_seed(fs)
            teacher = fit_torch(wide_mlp(d), Xtr_s, ytr, epochs=epochs, seed=fs)
        for name, cfg in cfgs.items():
            if cfg["kind"] == "torch":
                torch.manual_seed(fs)
                m = fit_torch(cfg["make"](), Xtr_s, ytr, epochs=epochs, seed=fs,
                              teacher=teacher if cfg.get("distil") else None)
                p = predict_torch(m, Xte_s)
            else:
                m = cfg["make"]().fit(Xtr_s, ytr)
                p = m.predict_proba(Xte_s)
            preds.setdefault(name, []).append(p[:, 1])
            r = metrics(yte, p[:, 1], p)
            r.update(model=name, repeat=s, fold=0)
            rows.append(r)
        print(f"    seed {s + 1}/{n_seeds} done", flush=True)

    extra = {} if sex_te is None else {"sex": np.asarray(sex_te, dtype="U16")}
    np.savez_compressed(ART / f"holdout_{task['name']}.npz", y=yte,
                        test_idx=ite, **extra,
                        **{k: np.array(v) for k, v in preds.items()}, allow_pickle=True)
    with open(ART / f"cv_{task['name']}.json", "w", encoding="utf-8") as f:
        json.dump(dict(task=task["name"], real=task["real"], n=len(y), d=d,
                       protocol="holdout", n_seeds=n_seeds, epochs=epochs, seed=seed,
                       meta=meta, folds=rows), f, indent=2)
    return rows, meta
