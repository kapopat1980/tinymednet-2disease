"""
check_expected.py -- Compares freshly computed results against the values
reported in the manuscript, and prints a PASS/FAIL line for each.

Tolerances reflect what can legitimately differ between machines. Torch and
scikit-learn are deterministic given a seed on identical versions, but library
version and CPU differences move metrics in the third decimal. A difference
inside tolerance is version drift; a difference outside it means something is
genuinely different and must be understood before submission.
"""
import json
import pathlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

TOL_METRIC = 0.010     # discrimination metrics
TOL_TIGHT = 0.020      # quantization AUROC, noisier across seeds

EXPECTED = {
    "TASK-DIA": {"TinyMedNet": (0.6745, 0.8253), "LogisticRegression": (0.6689, 0.8345)},
    "TASK-CKD": {"TinyMedNet": (0.8957, 0.9785), "LogisticRegression": (0.9929, 1.0000)},
}
# Backend-agnostic: "PTQ_validated" matches whichever engine this machine has
# (fbgemm on Linux x86, onednn on most Windows builds). Validated PTQ tracks FP32
# on every engine, which is the claim being checked.
EXPECTED_QUANT = {
    "TASK-DIA": {"FP32": 0.8031, "PTQ_legacy": 0.6531,
                 "PTQ_validated": 0.8038, "QAT_validated": 0.8006},
    "TASK-CKD": {"FP32": 0.9685, "PTQ_legacy": 0.5668,
                 "PTQ_validated": 0.9637, "QAT_validated": 0.9622},
}
EXPECTED_MISC = {
    "tflite_bytes": (20056, 400),
    "tflite_ops": (10, 0),
    "param_bytes": (4294, 0),
    "eod_pima": (0.271, 0.03),
    "ece_ckd_before": (0.297, 0.03),
    "ece_ckd_after": (0.009, 0.02),
}

fails = []


def check(label, got, want, tol):
    ok = got is not None and abs(got - want) <= tol
    if not ok:
        fails.append(label)
    got_s = "missing" if got is None else f"{got:.4f}"
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<52} got {got_s:>10}  "
          f"expected {want:.4f} ± {tol}")


def main():
    A = json.loads((RES / "analysis.json").read_text(encoding="utf-8"))

    print("\nDiscrimination (mean across repeats)")
    for task, models in EXPECTED.items():
        for model, (f1, auroc) in models.items():
            m = A.get(task, {}).get("models", {}).get(model)
            check(f"{task} {model} F1", m and m["f1_pos"]["mean"], f1, TOL_METRIC)
            check(f"{task} {model} AUROC", m and m["auroc"]["mean"], auroc, TOL_METRIC)

    print("\nPaired comparison against logistic regression")
    for task, (d, p) in [("TASK-DIA", (0.0056, 0.625)), ("TASK-CKD", (-0.0972, 0.002))]:
        pc = A.get(task, {}).get("paired_vs_TinyMedNet", {}).get("LogisticRegression")
        check(f"{task} delta F1 vs LR", pc and pc["mean_diff"], d, TOL_METRIC)
        sig = pc and pc["wilcoxon_p"]
        expect_sig = p < 0.05
        ok = sig is not None and ((sig < 0.05) == expect_sig)
        if not ok:
            fails.append(f"{task} significance")
        print(f"  [{'PASS' if ok else 'FAIL'}] {task + ' significance direction':<52} "
              f"got p={sig:.4f}" if sig is not None else "missing")

    qf = RES / "quantization.json"
    if qf.exists():
        Q = json.loads(qf.read_text(encoding="utf-8"))
        print("\nQuantization (mean AUROC over seeds)")
        for task, conds in EXPECTED_QUANT.items():
            rows = Q.get(task, {}).get("rows", [])
            for cond, want in conds.items():
                # exact match for FP32/legacy; prefix match for backend-suffixed rows
                vals = [r["auroc"] for r in rows
                        if "auroc" in r and (r.get("cond") == cond
                                             or str(r.get("cond", "")).startswith(cond + "_"))]
                check(f"{task} {cond}", float(np.mean(vals)) if vals else None,
                      want, TOL_TIGHT)
        demo = Q.get("TASK-DIA", {}).get("batchnorm_demo")
        if demo:
            ok = demo["after_distinct_values"] == 1
            if not ok:
                fails.append("BatchNorm collapse")
            print(f"  [{'PASS' if ok else 'FAIL'}] "
                  f"{'BatchNorm gamma collapses to 1 distinct value':<52} "
                  f"got {demo['after_distinct_values']}")
    else:
        print("\nQuantization: results/quantization.json not found (--quick run?)")

    tf_ = RES / "tflite_export.json"
    # A results file that shipped with the repository is not evidence about this
    # machine. If TensorFlow is unavailable the export step cannot have run here,
    # so the deployment numbers are reported as unverified rather than passed.
    try:
        import tensorflow  # noqa: F401
        have_tf = True
    except Exception:
        have_tf = False
    if tf_.exists() and not have_tf:
        print("\nDeployment artifact")
        # The exported flatbuffers are committed, so the headline size claim can
        # be checked against the real files with nothing but the filesystem.
        # Operator coverage and INT8 accuracy need an interpreter and cannot.
        blobs = sorted(pathlib.Path("artifacts").glob("*.tflite"))
        if blobs:
            for b in blobs:
                check(f"{b.name} size on disk", float(b.stat().st_size),
                      *EXPECTED_MISC["tflite_bytes"])
            ratio = blobs[0].stat().st_size / EXPECTED_MISC["param_bytes"][0]
            ok = abs(ratio - 4.67) < 0.3
            if not ok:
                fails.append("footprint ratio")
            print(f"  [{'PASS' if ok else 'FAIL'}] "
                  f"{'exported file / parameter bytes ratio':<52} "
                  f"got {ratio:.2f}x  expected 4.67x")
        else:
            print("  [SKIPPED] no .tflite files found in artifacts/")
        print("  [SKIPPED] operator count, float-tensor count and INT8 accuracy:")
        print("            TensorFlow is absent, so src/tflite_export.py cannot "
              "have run here.")
        print("            results/tflite_export.json ships with the repository "
              "and is NOT")
        print("            independent confirmation. See WINDOWS_SETUP.md to "
              "verify on Python")
        print("            3.11/3.12 or WSL2.")
        tf_ = None
    if tf_ is not None and tf_.exists():
        T = json.loads(tf_.read_text(encoding="utf-8"))
        d = T["TASK-DIA"]
        print("\nDeployment artifact")
        check("exported .tflite bytes", float(d["deployment"]["file_bytes"]),
              *EXPECTED_MISC["tflite_bytes"])
        check("operator count", float(d["deployment"]["n_operators"]),
              *EXPECTED_MISC["tflite_ops"])
        check("INT8 parameter bytes", float(d["param_bytes_int8"]),
              *EXPECTED_MISC["param_bytes"])
        ok = d["deployment"]["float_tensors_remaining"] == 0
        if not ok:
            fails.append("float tensors")
        print(f"  [{'PASS' if ok else 'FAIL'}] {'zero float tensors remaining':<52} "
              f"got {d['deployment']['float_tensors_remaining']}")

    print("\nFairness and calibration")
    check("PIMA age EOD", A["TASK-DIA"]["fairness_age"]["eod"], *EXPECTED_MISC["eod_pima"])
    c = A["TASK-CKD"]["calibration_TinyMedNet"]
    check("CKD ECE before scaling", c["ece_before"], *EXPECTED_MISC["ece_ckd_before"])
    check("CKD ECE after scaling", c["ece_after"], *EXPECTED_MISC["ece_ckd_after"])

    print()
    if fails:
        print(f"{len(fails)} check(s) FAILED:")
        for f in fails:
            print(f"  - {f}")
        print("\nInvestigate before submitting. Differences larger than the stated "
              "tolerance are not version drift.")
        sys.exit(1)
    print("All checks passed. The reported results reproduce on this machine.")


if __name__ == "__main__":
    main()
