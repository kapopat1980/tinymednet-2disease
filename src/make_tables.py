"""
make_tables.py -- Regenerates every table in the manuscript from saved artifacts.

The reviewers asked for an automated script that produces the results tables
directly from the stored experimental outputs, so that no number in the paper is
hand-transcribed. That is what this is. It reads only:

    artifacts/oof_*.npz          frozen out-of-fold predictions
    artifacts/holdout_*.npz      frozen held-out predictions
    results/analysis.json        aggregates computed from the above
    results/quantization.json    PTQ / QAT study
    results/tflite_export.json   deployment artifact measurements
    results/external_validation.json
    results/data_audit.json

and writes results/tables.md plus results/tables.json. Any number appearing in
the manuscript that is not in results/tables.json is a transcription error.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(d, k, dec=3):
    """
    'mean ± half-width'. The intervals are t-based across repeats and therefore
    symmetric about the mean, so this carries exactly the same information as
    [lo, hi] in far less width; seven columns of "0.669 [0.665, 0.672]" wrap to
    three lines on a journal page.
    """
    if k not in d:
        return "--"
    b = d[k]
    if not np.isfinite(b.get("ci_lo", np.nan)):
        return f"{b['mean']:.{dec}f}"
    half = (b["ci_hi"] - b["ci_lo"]) / 2
    return f"{b['mean']:.{dec}f} ± {half:.{dec}f}"


def _group_label(g):
    """age_ge_55 -> '>=55'; age_lt_55 -> '<55'; other labels pass through."""
    if g.startswith("age_ge_"):
        return "\u2265" + g.split("_")[-1]
    if g.startswith("age_lt_"):
        return "<" + g.split("_")[-1]
    return g


def md_table(header, rows, cap=16, pad=6):
    """
    Emits a pipe table whose separator dashes are proportional to the width each
    column needs. Pandoc derives relative column widths from those dash runs.

    Two rules matter. The weight is capped so one long free-text column cannot
    starve the rest, and it is floored at the longest single word in the column,
    since a word cannot be broken: without that floor a narrow column renders as
    "Endpo int" or "100,0 00".
    """
    cols = len(header)
    weights = []
    for i in range(cols):
        cells = [str(header[i])] + [str(r[i]) for r in rows]
        longest_word = max((len(w) for c in cells for w in c.split()), default=1)
        longest_cell = max(len(c) for c in cells)
        weights.append(max(longest_word, min(longest_cell, cap)) + pad)
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join("-" * w for w in weights) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


METRICS_ALL = ["f1_pos", "f1_macro", "mcc", "auroc", "auprc", "bal_acc",
               "ece", "brier"]
# Short headers for the same reason.
METRICS_ALL_HDR = ["F1", "mF1", "MCC", "AUROC", "AUPRC", "BAcc", "ECE", "Brier"]


def fmt_mean(d, k, dec=3):
    """
    Mean only. Nine columns carrying "0.708±0.003" cannot fit a portrait page,
    and the appendix exists to give the complete metric set rather than to repeat
    the intervals; those are in the main tables for the headline metrics.
    """
    return "--" if k not in d else f"{d[k]['mean']:.{dec}f}"
ORDER = ["LogisticRegression", "RandomForest", "XGBoost", "MLP_wide", "MLP_matched",
         "TinyMedNet", "TinyMedNet_noSE", "TinyMedNet_noResidual", "TinyMedNet_noDistil"]
# Kept short: long labels wrap badly in a journal-width table column.
PRETTY = {
    "LogisticRegression": "Logistic reg.", "RandomForest": "Random forest",
    "XGBoost": "XGBoost", "MLP_wide": "MLP wide",
    "MLP_matched": "MLP matched", "TinyMedNet": "TinyMed-Net",
    "TinyMedNet_noSE": "\u2003\u2212 SE",
    "TinyMedNet_noResidual": "\u2003\u2212 residual",
    "TinyMedNet_noDistil": "\u2003\u2212 distillation",
}
SHORT = {
    "TASK-DIA": "PIMA (n=768)",
    "TASK-CKD": "UCI-CKD (n=400)",
    "TASK-SYN": "Synthetic (n=5,000)",
}
TASK_LABEL = {
    "TASK-DIA": "PIMA Indians Diabetes (real, n=768)",
    "TASK-CKD": "UCI Chronic Kidney Disease (real, n=400)",
    "TASK-SYN": "Kaggle diabetes-prediction (SYNTHETIC, 5k subsample)",
}


def main():
    A = load("analysis.json") or {}
    Q = load("quantization.json")
    T = load("tflite_export.json")
    E = load("external_validation.json")
    D = load("data_audit.json")
    C = load("controls.json")
    md, js = [], {}

    md.append("# Generated results tables\n")
    md.append("_Every value below is computed from frozen artifacts by "
              "`src/make_tables.py`. Do not edit by hand._\n")

    # ---- Table 1: cohorts
    if D:
        s = D["synthetic_signatures"]
        rows = [
            ["PIMA Indians Diabetes", "768", "diabetes", "real", "8", "all female, age \u2265 21"],
            ["UCI Chronic Kidney Disease", "400", "CKD", "real", "24", "sex not recorded"],
            ["Kaggle diabetes-prediction", f"{s['n']:,}", "diabetes", "**synthetic**", "7",
             f"{s['hba1c_distinct_values']} distinct HbA1c values; "
             f"{100 * s['share_rows_at_modal_bmi']:.1f}% share the modal BMI"],
        ]
        md.append("\n## Table 1. Cohorts\n")
        md.append(md_table(["Cohort", "N", "Endpoint", "Provenance", "Features", "Notes"], rows))
        js["table1_cohorts"] = rows

    # ---- Table 2: synthetic evidence
    if D:
        s = D["synthetic_signatures"]
        rows = [
            ["Distinct HbA1c values", s["hba1c_distinct_values"]],
            ["Distinct blood-glucose values", s["glucose_distinct_values"]],
            ["Rows at the modal BMI", f"{100 * s['share_rows_at_modal_bmi']:.1f}% "
                                      f"(BMI={s['modal_bmi']:.2f}, column mean={s['mean_bmi']:.2f})"],
            ["Exact duplicate rows", f"{s['exact_duplicate_rows']:,}"],
            ["Minimum recorded age", f"{s['min_age_years']:.2f} years"],
            [f"Label deterministic outside HbA1c in {s['ambiguous_band']}",
             "yes" if s["label_deterministic_outside_band"] else "no"],
            ["Positive rate inside that band",
             f"{s['ambiguous_band_positive_rate']:.4f} (n={s['ambiguous_band_n']:,})"],
        ]
        md.append("\n## Table 2. Evidence that the Kaggle cohort is generated, not measured\n")
        md.append(md_table(["Signature", "Value"], rows))
        js["table2_synthetic_evidence"] = rows

    # ---- Table 2b: per-source, per-class record counts (reviewer comment 8)
    if C:
        rows = [[r["source"], r["cls"], f"{r['raw']:,}", f"{r['duplicates']:,}",
                 f"{r['analysed']:,}"] for r in C["counts"]]
        md.append("\n## Table 2b. Record counts by source and class\n")
        md.append("_Every cohort is analysed in full under repeated stratified "
                  "five-fold cross-validation, so train, validation and test counts "
                  "are determined by the fold assignment rather than by a fixed "
                  "split; fold assignments are recorded in `artifacts/cv_*.json`. "
                  "Duplicates are exact repeated rows, reported but not removed, "
                  "since removing them from a generated cohort would not make it "
                  "measured._\n")
        md.append(md_table(["Source", "Class", "Raw", "Exact duplicates",
                            "Analysed"], rows))
        js["table2b_counts"] = rows

    # ---- Table 2c: control baselines on the withdrawn pooled task (comment 2)
    if C:
        rows = [[b["baseline"], str(b["n_features"]),
                 f"{b['macro_f1']:.3f}", f"{b['ckd_recall']:.3f}"]
                for b in C["controls"]["baselines"]]
        md.append("\n## Table 2c. Control baselines on the withdrawn pooled "
                  "three-class task\n")
        r = C["controls"]["indicator_rule"]
        md.append(f"_Depth-4 decision trees, stratified five-fold cross-validation, "
                  f"n={C['controls']['n']:,}. Source identity alone recovers CKD "
                  f"almost perfectly, while clinical values alone do not, which is "
                  f"why the pooled formulation is withdrawn. The single rule "
                  f"\u201c{r['rule']}\u201d separates CKD at "
                  f"{100 * r['sensitivity']:.1f}% sensitivity and "
                  f"{100 * r['specificity']:.1f}% specificity._\n")
        md.append(md_table(["Baseline", "Features", "Macro-F1", "CKD recall"], rows))
        js["table2c_controls"] = rows

    # ---- Table 3: main results per task
    for task in ["TASK-DIA", "TASK-CKD", "TASK-SYN"]:
        if task not in A:
            continue
        blk = A[task]
        rows = []
        for m in ORDER:
            if m not in blk["models"]:
                continue
            d = blk["models"][m]
            rows.append([PRETTY[m], fmt(d, "f1_pos"), fmt(d, "auroc"),
                         fmt(d, "mcc"), fmt(d, "ece")])
        md.append(f"\n## Table 3{'abc'[['TASK-DIA','TASK-CKD','TASK-SYN'].index(task)]}. "
                  f"{TASK_LABEL[task]}\n")
        md.append(f"_n={blk['n']}, positives={blk['positives']}, "
                  f"{blk['models'].get('TinyMedNet', {}).get('n_repeats', '?')} repeats. "
                  "Mean ± half-width of the 95% confidence interval across repeats. "
                  "AUPRC, Brier score and balanced accuracy are reported in "
                  "Appendix A._\n")
        md.append(md_table(["Model", "F1", "AUROC", "MCC", "ECE"], rows))
        js[f"table3_{task}"] = rows
        js[f"table3_{task}_all_metrics"] = [
            [PRETTY[m]] + [fmt(blk["models"][m], k) for k in METRICS_ALL]
            for m in ORDER if m in blk["models"]]

    # ---- Table 4: paired comparisons
    for task in ["TASK-DIA", "TASK-CKD"]:
        if task not in A or not A[task].get("paired_vs_TinyMedNet"):
            continue
        rows = []
        for m, d in A[task]["paired_vs_TinyMedNet"].items():
            rows.append([PRETTY.get(m, m),
                         f"{d['mean_diff']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]",
                         f"{d['wilcoxon_p']:.4f}", d["n"]])
        md.append(f"\n## Table 4{'ab'[['TASK-DIA','TASK-CKD'].index(task)]}. "
                  f"Paired comparison against TinyMed-Net, {TASK_LABEL[task]}\n")
        md.append("_Positive difference favours TinyMed-Net. Identical folds and "
                  "identical test predictions; Wilcoxon signed-rank across repeats._\n")
        md.append(md_table(["Comparator", "Delta F1 [95% CI]", "p", "repeats"], rows))
        js[f"table4_{task}"] = rows

    # ---- Table 5: quantization
    if Q:
        for task, blk in Q.items():
            agg = {}
            for r in blk["rows"]:
                if "error" in r:
                    continue
                agg.setdefault(r["cond"], []).append(r)
            rows = []
            for cond, rs in agg.items():
                f1 = [r["f1_pos"] for r in rs]
                au = [r["auroc"] for r in rs]
                rows.append([cond, f"{np.mean(f1):.4f} ± {np.std(f1, ddof=1):.4f}",
                             f"{np.mean(au):.4f} ± {np.std(au, ddof=1):.4f}",
                             f"{rs[0]['param_bytes'] / 1024:.2f}", len(rs)])
            md.append(f"\n## Table 5{'ab'[list(Q).index(task)]}. "
                      f"Quantization, {TASK_LABEL.get(task, task)}\n")
            md.append("_`PTQ_legacy` reproduces the quantizer used in the original "
                      "implementation. All other rows are real INT8 models evaluated "
                      "through integer kernels. Mean ± SD over seeds._\n")
            md.append(md_table(["Condition", "F1", "AUROC", "Param KiB", "seeds"], rows))
            js[f"table5_{task}"] = rows
            if blk.get("batchnorm_demo"):
                b = blk["batchnorm_demo"]
                md.append(f"\n_Failure mode of the legacy quantizer: a BatchNorm scale "
                          f"vector with mean {b['before_mean']:.4f} is mapped to mean "
                          f"{b['after_mean']:.4f}, collapsing to "
                          f"{b['after_distinct_values']} distinct "
                          f"value{'s' if b['after_distinct_values'] != 1 else ''} "
                          f"({100 * b['retained_fraction']:.1f}% of true scale)._\n")

    # ---- Table 6: deployment (transposed: metrics as rows, cohorts as columns,
    # because eleven columns cannot fit a single-column journal page)
    if T:
        tasks = list(T)
        spec = [
            ("Parameters", lambda r: f"{r['keras_params']:,}"),
            ("Parameter bytes, INT8 (KiB)", lambda r: f"{r['param_bytes_int8'] / 1024:.2f}"),
            ("Exported TFLite file (KiB)", lambda r: f"{r['deployment']['file_bytes'] / 1024:.2f}"),
            ("TFLite before name stripping (KiB)", lambda r: f"{r['tflite_bytes_with_names'] / 1024:.2f}"),
            ("Tensor-arena lower bound (KiB)", lambda r: f"{r['deployment']['arena_lower_bound_bytes'] / 1024:.2f}"),
            ("Operators", lambda r: str(r["deployment"]["n_operators"])),
            ("Float tensors remaining", lambda r: str(r["deployment"]["float_tensors_remaining"])),
            ("F1, FP32", lambda r: f"{r['fp32']['f1_pos']:.3f}"),
            ("F1, INT8", lambda r: f"{r['int8_tflite']['f1_pos']:.3f}"),
            ("FP32/INT8 argmax agreement", lambda r: f"{r['argmax_agreement']:.3f}"),
        ]
        rows = [[label] + [fn(T[t]) for t in tasks] for label, fn in spec]
        md.append("\n## Table 6. Exported INT8 deployment artifact\n")
        md.append("_Measured from the exported TFLite flatbuffer, not estimated from "
                  "parameter counts. The arena figure is a lower bound from the tensor "
                  "inventory; the true requirement depends on the TFLite Micro memory "
                  "planner. No physical microcontroller was available._\n")
        md.append(md_table(["Quantity"] + [SHORT.get(t, t) for t in tasks], rows))
        js["table6_deployment"] = rows
        ops = sorted({o for r in T.values() for o in r["deployment"]["operators"]
                      if o != "DELEGATE"})
        md.append(f"\n_Operator set required by the runtime: {', '.join(ops)}._\n")
        js["table6_operators"] = ops

    # ---- Table 7: external validation (long format; the wide layout needed ten
    # columns and wrapped mid-word)
    if E:
        NAME = {"TinyMedNet": "TinyMed-Net", "LogisticRegression": "Logistic reg."}
        EVAL = [("within_synthetic", "Synthetic (within)"),
                ("synthetic_to_pima", "Synthetic \u2192 PIMA"),
                ("within_pima", "PIMA (within)"),
                ("pima_to_synthetic", "PIMA \u2192 synthetic")]
        rows = []
        for tag in ["unrestricted", "eligibility_matched"]:
            if tag not in E:
                continue
            b = E[tag]
            for kind in ["TinyMedNet", "LogisticRegression"]:
                if kind not in b:
                    continue
                k = b[kind]
                for key, label in EVAL:
                    d = k[key]
                    if isinstance(d.get("f1_pos"), dict):      # transfer: mean +/- sd
                        f1 = f"{d['f1_pos']['mean']:.3f} \u00b1 {d['f1_pos']['sd']:.3f}"
                        au = f"{d['auroc']['mean']:.3f}"
                    else:                                       # within-cohort reference
                        f1 = f"{d['f1_pos']:.3f}"
                        au = f"{d['auroc']:.3f}"
                    rows.append([tag.replace("_", " "), NAME.get(kind, kind), label, f1, au])
        md.append("\n## Table 7. External validation across cohorts\n")
        md.append("_Shared features only (age, BMI, glucose); identical binary endpoint; "
                  "the two cohorts are separate datasets so overlap is impossible. "
                  "Within-cohort rows are the reference achievable in each cohort. "
                  "F1 at a fixed 0.5 threshold is not comparable across cohorts of "
                  "different prevalence (8.5% vs 34.9%), so AUROC is given alongside._\n")
        md.append(md_table(["Setting", "Model", "Train \u2192 test", "F1", "AUROC"], rows))
        js["table7_external"] = rows
        if E.get("_note"):
            md.append(f"\n_{E['_note']}_\n")

    # ---- Table 8: fairness (long format; the wide layout needed eight columns
    # and four asymmetric bootstrap intervals, which wrapped to three lines each.
    # These CIs are percentile-based and NOT symmetric, so they cannot be
    # abbreviated to mean +/- half-width the way the t-based CIs above can.)
    rows = []
    for task in ["TASK-DIA", "TASK-CKD", "TASK-SYN"]:
        for key, attr in [("fairness_age", "age"), ("fairness_sex", "sex")]:
            f = A.get(task, {}).get(key)
            if not f or "error" in f:
                continue
            gs = list(f["groups"].items())
            comparison = f"{attr}: " + " vs ".join(_group_label(g[0]) for g in gs)
            group_n = " / ".join(str(g[1]["n"]) for g in gs)
            for label, val_key, ci_key in [("TPR gap", "tpr_gap", "tpr_gap_ci"),
                                           ("FPR gap", "fpr_gap", "fpr_gap_ci"),
                                           ("EOD", "eod", "eod_ci"),
                                           ("DPD", "dpd", "dpd_ci")]:
                lo, hi = f[ci_key]
                rows.append([SHORT.get(task, task), comparison, group_n, label,
                             f"{f[val_key]:.3f}", f"{lo:.3f}\u2013{hi:.3f}"])
    if rows:
        md.append("\n## Table 8. Fairness on held-out predictions\n")
        md.append("_Equalized-odds difference (EOD) is max(|TPR gap|, |FPR gap|), not "
                  "the TPR gap alone. DPD is the demographic parity difference. "
                  "Computed on out-of-fold predictions only, with percentile "
                  "bootstrap 95% confidence intervals._\n")
        md.append(md_table(["Cohort", "Comparison", "Group n", "Metric",
                            "Estimate", "95% CI"], rows))
        js["table8_fairness"] = rows

    # ---- Table 10: temperature scaling
    rows = []
    for task in ["TASK-DIA", "TASK-CKD", "TASK-SYN"]:
        c = A.get(task, {}).get("calibration_TinyMedNet")
        if not c:
            continue
        rows.append([SHORT.get(task, task),
                     f"{c['temperature_mean']:.3f} ± {c['temperature_sd']:.3f}",
                     f"{c['ece_before']:.3f}", f"{c['ece_after']:.3f}",
                     f"{c['brier_before']:.3f}", f"{c['brier_after']:.3f}"])
    if rows:
        md.append("\n## Table 10. Cross-fitted temperature scaling, TinyMed-Net\n")
        md.append("_A single scalar fitted by NLL on K-1 folds of the out-of-fold "
                  "probabilities and applied to the held-out fold, so no patient "
                  "contributes to the temperature used to score them. Ranking is "
                  "unchanged, so AUROC is unaffected._\n")
        md.append(md_table(["Cohort", "Temperature", "ECE before", "ECE after",
                            "Brier before", "Brier after"], rows))
        js["table10_calibration"] = rows

    # ---------------------------------------------------------------- supplement
    sm = ["# Supplementary material\n",
          "_Generated by `src/make_tables.py` from the same frozen prediction "
          "artifacts as the main tables. Table numbering is independent of the "
          "main text._\n"]
    sj = {}
    s_no = 1

    # Full metric set for every model and cohort. The main tables report F1,
    # AUROC, MCC and ECE only, for width.
    for task in ["TASK-DIA", "TASK-CKD", "TASK-SYN"]:
        if task not in A:
            continue
        blk = A[task]
        rows = [[PRETTY[m].strip()] + [fmt_mean(blk["models"][m], k)
                                       for k in METRICS_ALL]
                for m in ORDER if m in blk["models"]]
        sm.append(f"\n## Table S{s_no}. Complete metric set, {TASK_LABEL[task]}\n")
        sm.append(f"_n={blk['n']}, positives={blk['positives']}, "
                  f"{blk['models'].get('TinyMedNet', {}).get('n_repeats', '?')} repeats. "
                  "Means across repeats; 95% confidence intervals for the "
                  "headline metrics are given in the main tables. mF1 is "
                  "macro-F1; BAcc is balanced accuracy._\n")
        sm.append(md_table(["Model"] + METRICS_ALL_HDR, rows))
        sj[f"S{s_no}"] = rows
        s_no += 1

    # Reliability diagrams in tabular form.
    for task in ["TASK-DIA", "TASK-CKD"]:
        rel = A.get(task, {}).get("reliability_TinyMedNet")
        if not rel:
            continue
        rows = [[f"({r['bin_lo']:.1f}, {r['bin_hi']:.1f}]", r["n"],
                 f"{r['mean_pred']:.3f}", f"{r['frac_pos']:.3f}",
                 f"{r['frac_pos'] - r['mean_pred']:+.3f}"] for r in rel]
        sm.append(f"\n## Table S{s_no}. Reliability, TinyMed-Net, {TASK_LABEL[task]}\n")
        sm.append("_From the same frozen predictions as the discrimination "
                  "results; no model was retrained. Positive gap means the "
                  "observed event rate exceeds the predicted probability._\n")
        sm.append(md_table(["Predicted bin", "n", "Mean predicted", "Observed",
                            "Gap"], rows))
        sj[f"S{s_no}"] = rows
        s_no += 1

    (RES / "supplementary.md").write_text("\n".join(sm) + "\n", encoding="utf-8")
    (RES / "supplementary.json").write_text(json.dumps(sj, indent=2), encoding="utf-8")
    print(f"written results/supplementary.md ({s_no - 1} tables)")

    (RES / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (RES / "tables.json").write_text(json.dumps(js, indent=2), encoding="utf-8")
    print(f"written results/tables.md ({len(md)} blocks) and results/tables.json")


if __name__ == "__main__":
    main()
