"""
analysis.py -- Turns frozen predictions into every number the manuscript reports.

Nothing here retrains anything. It reads artifacts/*.npz, which were written
once during run_all.py, so the calibration figures necessarily describe the same
fitted models as the discrimination figures. In the original repository
compute_ece.py retrained the models, which is why the ECE column of Table 4
described different models from the rest of that table.

Fairness is computed as an actual equalized-odds quantity: the larger of the
TPR gap and the FPR gap between groups, following the standard definition. The
original computed only the TPR gap and called it EOD, on data that included the
model's own training rows.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment import ART, ece, metrics

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
METRICS = ["f1_pos", "f1_macro", "mcc", "auroc", "auprc", "bal_acc", "brier", "ece"]
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- aggregation
def ci_mean(v, alpha=0.05):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float(v.mean()) if len(v) else float("nan"), float("nan"), float("nan")
    m = v.mean()
    h = stats.t.ppf(1 - alpha / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(m), float(m - h), float(m + h)


def bootstrap_ci(y, p, metric, n_boot=600, alpha=0.05):
    """Percentile bootstrap over patients. Resamples are reduced for large cohorts,
    where the interval is narrow and stable long before 600 draws."""
    vals = []
    n = len(y)
    if n > 2000:
        n_boot = 150
    for _ in range(n_boot):
        i = RNG.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(metrics(y[i], p[i])[metric])
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def per_repeat_metrics(npz, model):
    """One metric vector per repeat, computed on that repeat's out-of-fold predictions."""
    y = npz["y"]
    P = npz[model]
    return [metrics(y, P[r]) for r in range(P.shape[0])]


def summarize(task):
    f = ART / f"oof_{task}.npz"
    if not f.exists():
        f = ART / f"holdout_{task}.npz"
    npz = np.load(f, allow_pickle=True)
    y = npz["y"]
    models = [k for k in npz.files if k not in ("y", "test_idx", "sex", "allow_pickle")]
    out = {}
    for m in models:
        per = per_repeat_metrics(npz, m)
        row = {}
        for k in METRICS:
            mu, lo, hi = ci_mean([p[k] for p in per])
            row[k] = dict(mean=mu, ci_lo=lo, ci_hi=hi)
        pooled = npz[m].mean(0)
        for k in ["f1_pos", "auroc", "mcc"]:
            blo, bhi = bootstrap_ci(y, pooled, k)
            row[k]["boot_lo"], row[k]["boot_hi"] = blo, bhi
        row["n_repeats"] = int(npz[m].shape[0])
        out[m] = row
    return out, npz, y, models


def paired(npz, models, ref="TinyMedNet", metric="f1_pos"):
    """Paired comparison on identical folds and identical test predictions."""
    if ref not in models:
        return {}
    R = np.array([m_[metric] for m_ in per_repeat_metrics(npz, ref)])
    out = {}
    for m in models:
        if m == ref:
            continue
        V = np.array([x[metric] for x in per_repeat_metrics(npz, m)])
        d = R - V
        mu, lo, hi = ci_mean(d)
        try:
            _, p = stats.wilcoxon(d)
        except Exception:
            p = float("nan")
        out[m] = dict(mean_diff=mu, ci_lo=lo, ci_hi=hi, wilcoxon_p=float(p), n=len(d))
    return out


# ---------------------------------------------------------------- calibration
def reliability(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum():
            rows.append(dict(bin_lo=float(lo), bin_hi=float(hi), n=int(m.sum()),
                             mean_pred=float(p[m].mean()), frac_pos=float(y[m].mean())))
    return rows


def temperature_scale(y, p, n_folds=5, seed=0):
    """
    Post-hoc temperature scaling, evaluated by cross-fitting.

    A single temperature T is fitted by minimising NLL on K-1 folds of the
    out-of-fold probabilities and applied to the held-out fold, so no sample
    contributes to fitting the temperature used to score it. This estimates the
    calibration achievable with one extra scalar, without leaking.
    """
    from sklearn.model_selection import StratifiedKFold
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    out = np.zeros_like(p)
    Ts = []
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(p.reshape(-1, 1), y):
        def nll(logT):
            q = 1 / (1 + np.exp(-logit[tr] / np.exp(logT)))
            q = np.clip(q, eps, 1 - eps)
            return -(y[tr] * np.log(q) + (1 - y[tr]) * np.log(1 - q)).mean()
        from scipy.optimize import minimize_scalar
        m = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
        T = float(np.exp(m.x))
        Ts.append(T)
        out[te] = 1 / (1 + np.exp(-logit[te] / T))
    probs = np.column_stack([1 - out, out])
    return dict(temperature_mean=float(np.mean(Ts)),
                temperature_sd=float(np.std(Ts, ddof=1)),
                ece_before=ece(np.column_stack([1 - p, p]), y),
                ece_after=ece(probs, y),
                brier_before=float(np.mean((p - y) ** 2)),
                brier_after=float(np.mean((out - y) ** 2)),
                auroc_unchanged=True,
                reliability_after=reliability(y, out))


# ---------------------------------------------------------------- fairness
def group_rates(y, pred, mask):
    ys, ps = y[mask], pred[mask]
    pos, neg = ys == 1, ys == 0
    tpr = float(ps[pos].mean()) if pos.sum() else float("nan")
    fpr = float(ps[neg].mean()) if neg.sum() else float("nan")
    return dict(n=int(mask.sum()), n_pos=int(pos.sum()), n_neg=int(neg.sum()),
                tpr=tpr, fpr=fpr, selection_rate=float(ps.mean()))


def fairness(y, p, groups, n_boot=600):
    """
    Demographic parity difference and a correct equalized-odds difference.
    EOD = max(|TPR_a - TPR_b|, |FPR_a - FPR_b|), not the TPR gap alone.
    """
    pred = (p >= 0.5).astype(int)
    levels = [g for g in np.unique(groups) if str(g) not in ("nan", "None")]
    if len(levels) != 2:
        return dict(error=f"needs exactly two groups, found {list(levels)}")
    a, b = levels
    ga, gb = group_rates(y, pred, groups == a), group_rates(y, pred, groups == b)
    dpd = abs(ga["selection_rate"] - gb["selection_rate"])
    tpr_gap = abs(ga["tpr"] - gb["tpr"])
    fpr_gap = abs(ga["fpr"] - gb["fpr"])
    eod = float(np.nanmax([tpr_gap, fpr_gap]))

    boot = {"dpd": [], "tpr_gap": [], "fpr_gap": [], "eod": []}
    n = len(y)
    for _ in range(n_boot):
        i = RNG.integers(0, n, n)
        try:
            A = group_rates(y[i], pred[i], groups[i] == a)
            B = group_rates(y[i], pred[i], groups[i] == b)
            t, fp = abs(A["tpr"] - B["tpr"]), abs(A["fpr"] - B["fpr"])
            boot["dpd"].append(abs(A["selection_rate"] - B["selection_rate"]))
            boot["tpr_gap"].append(t)
            boot["fpr_gap"].append(fp)
            boot["eod"].append(np.nanmax([t, fp]))
        except Exception:
            continue

    def pc(v):
        v = np.array([x for x in v if np.isfinite(x)])
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)

    return dict(groups={str(a): ga, str(b): gb},
                dpd=dpd, dpd_ci=pc(boot["dpd"]),
                tpr_gap=tpr_gap, tpr_gap_ci=pc(boot["tpr_gap"]),
                fpr_gap=fpr_gap, fpr_gap_ci=pc(boot["fpr_gap"]),
                eod=eod, eod_ci=pc(boot["eod"]))


def age_group_fairness(task, model="TinyMedNet", cut=None):
    """Age-stratified fairness on the real cohorts, which do not record usable sex."""
    from data import get_task
    t = get_task(task)
    f = ART / f"oof_{task}.npz"
    if not f.exists():
        return {}
    npz = np.load(f, allow_pickle=True)
    if model not in npz.files:
        return {}
    age = t["X"][:, t["features"].index("age")]
    cut = float(np.nanmedian(age)) if cut is None else cut
    g = np.where(age >= cut, f"age_ge_{cut:.0f}", f"age_lt_{cut:.0f}").astype(object)
    r = fairness(npz["y"], npz[model].mean(0), g)
    r["split_value"] = cut
    r["attribute"] = "age"
    return r


# ---------------------------------------------------------------- driver
def run():
    out = {}
    for task in ["TASK-DIA", "TASK-CKD", "TASK-SYN"]:
        if not (ART / f"oof_{task}.npz").exists() and not (ART / f"holdout_{task}.npz").exists():
            continue
        summ, npz, y, models = summarize(task)
        block = dict(models=summ,
                     paired_vs_TinyMedNet=paired(npz, models),
                     n=int(len(y)), positives=int(y.sum()))
        if "TinyMedNet" in models:
            praw = npz["TinyMedNet"].mean(0)
            block["reliability_TinyMedNet"] = reliability(y, praw)
            block["calibration_TinyMedNet"] = temperature_scale(y, praw)
            block["temperature_scaling"] = {
                m: temperature_scale(y, npz[m].mean(0))
                for m in ["TinyMedNet", "MLP_wide"] if m in models}
        if "sex" in npz.files:
            block["fairness_sex"] = fairness(y, npz["TinyMedNet"].mean(0),
                                             np.asarray(npz["sex"], dtype=object))
        if task in ("TASK-DIA", "TASK-CKD"):
            block["fairness_age"] = age_group_fairness(task)
        out[task] = block
        print(f"  [{task}] summarized ({len(models)} models)", flush=True)

    with open(RES / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("written results/analysis.json")


if __name__ == "__main__":
    run()
