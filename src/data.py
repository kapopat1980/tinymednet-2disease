"""
data.py -- Corrected dataset construction for TinyMed-Net.

Replaces the pooled three-class corpus, in which the class label was coextensive
with the source file. Each task is now a single binary endpoint drawn from one
cohort, so no label can be inferred from feature-availability patterns.

  TASK-DIA  PIMA Indians Diabetes      n=768   real    diabetes vs not
  TASK-CKD  UCI Chronic Kidney Disease n=400   real    CKD vs not
  TASK-SYN  Kaggle diabetes-prediction n=100k  SYNTHETIC, stress test only

TASK-SYN is retained only as an explicitly-labelled synthetic cohort for
scaling behaviour and for exercising the fairness pipeline. It carries no
clinical claim. See audit_synthetic_signatures().
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

PIMA_COLS = ["pregnancies", "glucose", "bp", "skin", "insulin", "bmi", "pedigree", "age", "y"]
# Columns where a recorded 0 is physiologically impossible and encodes "not measured".
PIMA_ZERO_IS_MISSING = ["glucose", "bp", "skin", "insulin", "bmi"]

CKD_NUM = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]
CKD_BIN = {
    "rbc": {"normal": 0, "abnormal": 1},
    "pc": {"normal": 0, "abnormal": 1},
    "pcc": {"notpresent": 0, "present": 1},
    "ba": {"notpresent": 0, "present": 1},
    "htn": {"no": 0, "yes": 1},
    "dm": {"no": 0, "yes": 1},
    "cad": {"no": 0, "yes": 1},
    "appet": {"good": 0, "poor": 1},
    "pe": {"no": 0, "yes": 1},
    "ane": {"no": 0, "yes": 1},
}


def _clean_str(s):
    return s.astype("string").str.strip().str.lower() if s.dtype == object else s


def load_pima():
    """PIMA Indians Diabetes. Real 1980s NIH cohort; all participants female, age >= 21."""
    df = pd.read_csv(RAW / "pima_diabetes.csv", header=None, names=PIMA_COLS)
    df[PIMA_ZERO_IS_MISSING] = df[PIMA_ZERO_IS_MISSING].replace(0, np.nan)
    df["sex"] = "Female"  # documented property of the cohort, not an imputation
    feats = ["pregnancies", "glucose", "bp", "skin", "insulin", "bmi", "pedigree", "age"]
    return df, feats


def load_ckd():
    """UCI Chronic Kidney Disease. Real 2015 cohort, 400 records, no sex recorded."""
    df = pd.read_csv(RAW / "ckd.csv")
    df = df.drop(columns=[c for c in ["id"] if c in df.columns])
    for c in df.columns:
        df[c] = _clean_str(df[c])
    for c in CKD_NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c, m in CKD_BIN.items():
        df[c] = df[c].map(m).astype("float64")
    df["y"] = (df["class"].str.startswith("ckd")).astype(int)  # handles the 'ckd\t' rows
    df = df.drop(columns=["class"])
    feats = CKD_NUM + list(CKD_BIN)
    return df, feats


def load_synthetic():
    """Kaggle diabetes-prediction cohort. SYNTHETIC -- see audit_synthetic_signatures()."""
    df = pd.read_csv(RAW / "diabetes_prediction_dataset.csv")
    df = df.rename(columns={
        "diabetes": "y", "gender": "sex",
        "blood_glucose_level": "glucose", "HbA1c_level": "hba1c",
    })
    df = df[df.sex.isin(["Male", "Female"])].copy()  # 18 'Other' rows, too few to analyse
    df["sex"] = df["sex"].astype(str)
    df["smoking_history"] = df["smoking_history"].astype("category").cat.codes
    feats = ["age", "hypertension", "heart_disease", "smoking_history", "bmi", "hba1c", "glucose"]
    return df, feats


def audit_synthetic_signatures():
    """Evidence that the Kaggle cohort is generated rather than measured."""
    k = pd.read_csv(RAW / "diabetes_prediction_dataset.csv")
    lo, hi = 5.6, 6.6
    det_neg = int(((k.HbA1c_level <= lo) & (k.diabetes == 1)).sum())
    det_pos = int(((k.HbA1c_level > hi) & (k.diabetes == 0)).sum())
    band = k[(k.HbA1c_level > lo) & (k.HbA1c_level <= hi)]
    return {
        "n": int(len(k)),
        "hba1c_distinct_values": int(k.HbA1c_level.nunique()),
        "glucose_distinct_values": int(k.blood_glucose_level.nunique()),
        "modal_bmi": float(k.bmi.mode()[0]),
        "mean_bmi": float(k.bmi.mean()),
        "share_rows_at_modal_bmi": float((k.bmi == k.bmi.mode()[0]).mean()),
        "exact_duplicate_rows": int(k.duplicated().sum()),
        "min_age_years": float(k.age.min()),
        "label_deterministic_outside_band": det_neg == 0 and det_pos == 0,
        "ambiguous_band": f"({lo}, {hi}]",
        "ambiguous_band_n": int(len(band)),
        "ambiguous_band_positive_rate": float(band.diabetes.mean()),
    }


def audit_pooled_leakage():
    """
    Quantifies the defect in the original pooled three-class corpus: because each
    condition came from a different file and each file measured a different panel,
    feature availability alone identifies the label.
    """
    pima, _ = load_pima()
    ckd, _ = load_ckd()
    syn, _ = load_synthetic()
    rows = []
    for name, df, has in [
        ("PIMA", pima, {"bp": True, "hemoglobin": False, "creatinine": False, "bmi": True}),
        ("Kaggle-DPD", syn, {"bp": False, "hemoglobin": False, "creatinine": False, "bmi": True}),
        ("UCI-CKD", ckd, {"bp": True, "hemoglobin": True, "creatinine": True, "bmi": False}),
    ]:
        rows.append({"source": name, "n": len(df), **{f"has_{k}": v for k, v in has.items()}})
    avail = pd.DataFrame(rows)
    # Under the original label map, CKD came only from UCI-CKD, and only UCI-CKD
    # measured haemoglobin/creatinine -- so that one bit is a near-perfect CKD detector.
    n_ckd = int(ckd.y.sum())
    n_not = len(pima) + len(syn) + int((~ckd.y.astype(bool)).sum())
    return {
        "feature_availability_by_source": avail,
        "ckd_cases_all_from_uci": True,
        "indicator_sensitivity": 1.0,   # every CKD case is in the only source measuring hb/cr
        "indicator_specificity": 1.0 - (len(ckd) - n_ckd) / n_not,
        "note": "The indicator 'haemoglobin or creatinine recorded' is equivalent to "
                "'row originated in UCI-CKD', which under the original label map is "
                "equivalent to 'CKD or UCI-CKD control'.",
    }


TASKS = {
    "TASK-DIA": dict(loader=load_pima, real=True, label="PIMA diabetes (real)"),
    "TASK-CKD": dict(loader=load_ckd, real=True, label="UCI CKD (real)"),
    "TASK-SYN": dict(loader=load_synthetic, real=False, label="Kaggle diabetes (SYNTHETIC)"),
}


def get_task(name):
    spec = TASKS[name]
    df, feats = spec["loader"]()
    X = df[feats].astype("float64").to_numpy()
    y = df["y"].to_numpy().astype(np.int64)
    sex = df["sex"].to_numpy() if "sex" in df.columns else None
    return dict(name=name, X=X, y=y, sex=sex, features=feats,
                real=spec["real"], label=spec["label"])


if __name__ == "__main__":
    for n in TASKS:
        t = get_task(n)
        print(f"{n:9s} n={len(t['y']):6d}  d={t['X'].shape[1]:2d}  "
              f"pos={t['y'].mean():.3f}  real={t['real']}  "
              f"missing={np.isnan(t['X']).mean():.3f}")
