"""
controls_and_figures.py -- Closes three remaining reviewer requests.

  1. Control baselines on the withdrawn pooled task (reviewer comment 2).
     The reviewers asked for missingness-only and source-only baselines. The
     pooled formulation is withdrawn, but the controls are what justify
     withdrawing it, so they are reported rather than merely asserted.

  2. Per-source, per-class record counts including duplicates (comment 8).
     The earlier Table 2 reported the training subset under a column labelled
     "N used". This reports raw counts, exact duplicates, and analysed counts
     for every source and class.

  3. Reliability diagrams (comment 12). Reliability was reported as binned
     tables; the reviewers asked for diagrams.

Writes results/controls.json and artifacts/figure2_reliability.png/.svg.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from data import load_ckd, load_pima, load_synthetic
from experiment import ART

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CLASSES = ["Healthy", "Diabetes", "CKD"]


# ------------------------------------------------------------------ 1. controls
def build_pooled():
    """
    Reconstructs the withdrawn three-class corpus exactly as it was formed:
    three single-disease sources pooled, each measuring a different panel.
    """
    pima, _ = load_pima()
    syn, _ = load_synthetic()
    ckd, _ = load_ckd()

    def frame(df, source, age, bp, glucose, bmi, hemo, creat, y):
        n = len(df)
        return pd.DataFrame({
            "source": source,
            "age": df[age] if age else np.nan,
            "bp": df[bp] if bp else np.full(n, np.nan),
            "glucose": df[glucose] if glucose else np.full(n, np.nan),
            "bmi": df[bmi] if bmi else np.full(n, np.nan),
            "hemoglobin": df[hemo] if hemo else np.full(n, np.nan),
            "creatinine": df[creat] if creat else np.full(n, np.nan),
            "y": y,
        })

    P = frame(pima, "PIMA", "age", "bp", "glucose", "bmi", None, None,
              np.where(pima["y"] == 1, 1, 0))
    S = frame(syn, "Kaggle-DPD", "age", None, "glucose", "bmi", None, None,
              np.where(syn["y"] == 1, 1, 0))
    C = frame(ckd, "UCI-CKD", "age", "bp", "bgr", None, "hemo", "sc",
              np.where(ckd["y"] == 1, 2, 0))
    pooled = pd.concat([P, S, C], ignore_index=True)
    for c in ["bp", "hemoglobin", "creatinine", "bmi"]:
        pooled[f"{c}_missing"] = pooled[c].isna().astype(int)
    return pooled


def controls():
    d = build_pooled()
    y = d["y"].to_numpy()
    MISS = [c for c in d.columns if c.endswith("_missing")]
    CLIN = ["age", "bp", "glucose", "bmi", "hemoglobin", "creatinine"]

    variants = {
        "Majority class": (None, "dummy"),
        "Missingness indicators only": (MISS, "tree"),
        "Source identity only": (["__source__"], "tree"),
        "Clinical values only (imputed)": (CLIN, "tree"),
        "Clinical values + missingness": (CLIN + MISS, "tree"),
    }
    d["__source__"] = pd.Categorical(d["source"]).codes

    out = []
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for name, (cols, kind) in variants.items():
        f1s, rec = [], []
        for tr, te in skf.split(d, y):
            if kind == "dummy":
                m = DummyClassifier(strategy="most_frequent").fit(
                    np.zeros((len(tr), 1)), y[tr])
                p = m.predict(np.zeros((len(te), 1)))
            else:
                X = d[cols].to_numpy(float)
                imp = SimpleImputer(strategy="median").fit(X[tr])
                m = DecisionTreeClassifier(max_depth=4, random_state=0).fit(
                    imp.transform(X[tr]), y[tr])
                p = m.predict(imp.transform(X[te]))
            f1s.append(f1_score(y[te], p, average="macro", zero_division=0))
            rec.append(recall_score(y[te], p, labels=[2], average="macro",
                                    zero_division=0))
        out.append(dict(baseline=name,
                        macro_f1=float(np.mean(f1s)),
                        ckd_recall=float(np.mean(rec)),
                        n_features=0 if cols is None else len(cols)))

    # The single-bit rule the reviewers identified.
    ind = ((d.hemoglobin_missing == 0) | (d.creatinine_missing == 0)).to_numpy()
    ckd = y == 2
    sens = float((ind & ckd).sum() / ckd.sum())
    spec = float((~ind & ~ckd).sum() / (~ckd).sum())
    return dict(baselines=out, n=int(len(d)),
                indicator_rule=dict(rule="haemoglobin or creatinine recorded",
                                    sensitivity=sens, specificity=spec))


# ------------------------------------------------------------------ 2. counts
def counts():
    pima, _ = load_pima()
    syn, _ = load_synthetic()
    ckd, _ = load_ckd()
    rows = []
    for source, df, pos, neg in [
        ("PIMA Indians Diabetes", pima, "Diabetes", "No diabetes"),
        ("Kaggle diabetes-prediction", syn, "Diabetes", "No diabetes"),
        ("UCI Chronic Kidney Disease", ckd, "CKD", "No CKD"),
    ]:
        dup = int(df.duplicated().sum())
        for label, mask in [(pos, df["y"] == 1), (neg, df["y"] == 0)]:
            rows.append(dict(source=source, cls=label, raw=int(mask.sum()),
                             duplicates=int(df[mask].duplicated().sum()),
                             analysed=int(mask.sum())))
        rows.append(dict(source=source, cls="All", raw=int(len(df)),
                         duplicates=dup, analysed=int(len(df))))
    return rows


# ------------------------------------------------------------------ 3. figure
def reliability_figure():
    A = json.loads((RES / "analysis.json").read_text(encoding="utf-8"))
    tasks = [("TASK-DIA", "PIMA Indians Diabetes (n=768)"),
             ("TASK-CKD", "UCI Chronic Kidney Disease (n=400)")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3))
    for ax, (t, title) in zip(axes, tasks):
        blk = A[t]
        rel = blk["reliability_TinyMedNet"]
        cal = blk["calibration_TinyMedNet"]
        ax.plot([0, 1], [0, 1], color="#8A8F9A", lw=1, ls="--", zorder=1,
                label="perfect calibration")
        ax.plot([r["mean_pred"] for r in rel], [r["frac_pos"] for r in rel],
                "o-", color="#B4553E", lw=1.6, ms=5, zorder=3,
                label=f"uncalibrated (ECE {cal['ece_before']:.3f})")
        after = cal.get("reliability_after")
        if after:
            ax.plot([r["mean_pred"] for r in after], [r["frac_pos"] for r in after],
                    "s-", color="#37578F", lw=1.6, ms=4.5, zorder=4,
                    label=f"temperature scaled (ECE {cal['ece_after']:.3f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7.5, loc="upper left", frameon=False)
        ax.grid(alpha=0.25, lw=0.6)
    plt.tight_layout()
    for ext in ("png", "svg"):
        plt.savefig(ART / f"figure2_reliability.{ext}", dpi=300, bbox_inches="tight")
    return str(ART / "figure2_reliability.png")


if __name__ == "__main__":
    out = dict(controls=controls(), counts=counts())
    (RES / "controls.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    fig = reliability_figure()
    print("written results/controls.json")
    print("written", fig)
    for b in out["controls"]["baselines"]:
        print(f"  {b['baseline']:<32} macro-F1={b['macro_f1']:.3f} "
              f"CKD recall={b['ckd_recall']:.3f}")
    r = out["controls"]["indicator_rule"]
    print(f"  single-bit rule: sens={r['sensitivity']:.3f} spec={r['specificity']:.3f}")
