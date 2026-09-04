# TinyMed-Net

Code and frozen results for *"TinyMed-Net: What a Sub-4K-Parameter Network Can
and Cannot Do for Tabular Clinical Screening."*

Every number in the paper is generated from the frozen prediction artifacts in
`artifacts/` by `src/make_tables.py`, and spliced into the manuscript by
`src/build_manuscript.py`. No value in the paper is transcribed by hand.

> **This repository supersedes an earlier version of this project.** Several
> claims in that version were wrong, and this one contradicts them. The
> corrections are listed under [What changed](#what-changed-and-why) below,
> and in the submitted cover letter. The earlier code and results remain in
> git history under the tag `v1-superseded`.

## Headline results

Ten repeats of stratified 5-fold cross-validation; mean [95% CI across repeats].

| Model | PIMA F1 | PIMA AUROC | CKD F1 | CKD AUROC |
|---|---|---|---|---|
| Logistic regression | 0.669 | 0.835 | **0.993** | **1.000** |
| Random forest | 0.646 | 0.830 | 0.994 | 1.000 |
| XGBoost | 0.631 | 0.809 | 0.988 | 0.999 |
| MLP (parameter-matched) | 0.677 | 0.830 | 0.865 | 0.950 |
| TinyMed-Net | 0.675 | 0.825 | 0.896 | 0.978 |

**TinyMed-Net offers no accuracy advantage over logistic regression.** On PIMA
the difference is not significant (ΔF1 = +0.006, p = 0.63, paired Wilcoxon on
identical folds); on CKD, logistic regression is significantly better
(ΔF1 = −0.097, p = 0.002). The case for the architecture is deployability, not
accuracy.

UCI-CKD is near-separable — every method exceeds AUROC 0.95 — and cannot
discriminate between modelling approaches.

## Three findings that may be useful beyond this paper

**1. Post-training quantization is sufficient at this scale; QAT is not needed.**
Across 20 seeds with a validated toolchain (BatchNorm folding, per-channel
symmetric weights):

| Condition | PIMA AUROC | CKD AUROC |
|---|---|---|
| FP32 | 0.803 | 0.969 |
| PTQ (validated, per-channel) | 0.804 | 0.964 |
| PTQ (validated, per-tensor) | 0.804 | 0.970 |
| QAT (validated) | 0.799–0.801 | 0.962–0.968 |
| PTQ (the earlier version's quantizer) | 0.653 | 0.567 |

The last row is a reproduction of the earlier implementation, retained as a
control. It computed scale as `(max − min)/255` and rounded into `[−128, 127]`
with **no zero point**, applied to every parameter tensor including BatchNorm,
without folding. BatchNorm γ is strictly positive and clusters near 1.0, so this
maps that range onto the quantizer's positive half only: a γ vector of mean
1.009 becomes mean 0.275, collapsed to **a single distinct value**. The
collapse is the bug, not a property of PTQ.

**2. Parameter-count footprints understate real Flash by ~4.7×.** The exported
INT8 TFLite model holds 4.19 KiB of parameters but the flatbuffer is 19.59 KiB
(31.27 KiB before tensor-name stripping). At this scale, graph structure and
per-channel quantization metadata dominate.

**3. Three common Keras constructs break TFLite Micro export.**
`GlobalAveragePooling1D(keepdims=True)`, descriptor broadcasting via
`UpSampling1D`, and a dynamic batch dimension each emit shape-arithmetic
operators (SHAPE, PACK, STRIDED_SLICE, TILE). Replacing them with fixed-size
average pooling, native broadcast multiplication, and a static batch of one
reduced the graph from **86 operators to 10**, all standard TFLite Micro
builtins, with zero float tensors remaining.

## Data

| Source | Condition | N | Provenance |
|---|---|---|---|
| PIMA Indians Diabetes | Diabetes | 768 | Real. Smith et al. 1988 (UCI). All female, age ≥ 21 |
| UCI Chronic Kidney Disease | CKD | 400 | Real. Rubini et al. 2015 (UCI), DOI 10.24432/C5G020. Sex not recorded |
| Kaggle diabetes-prediction | Diabetes | 100,000 | **Synthetic — see below** |

### The Kaggle cohort is synthetic

Run `python src/data.py` and see `results/data_audit.json`. HbA1c and blood
glucose each take only **18 distinct values** across 100,000 records; **25.5% of
rows share an identical BMI of 27.32**, which is the column mean; there are
**3,854 exact duplicate rows**; the minimum age is **0.08 years**; and the label
is a **deterministic function of HbA1c** outside the band (5.6, 6.6].

It supports no clinical inference. We retain it only as an explicitly labelled
distribution-shift comparator (5,000-row stratified subsample).

### The pooled three-class task was solvable from missingness alone

The earlier version pooled all three cohorts into a Healthy/Diabetes/CKD task.
Because each condition came from a different source and each source measured a
different assay panel, feature availability alone identified the label. The
indicator *"haemoglobin or creatinine recorded"* separates CKD at **96.0%
sensitivity and 98.9% specificity**, and a classifier given only three
missingness flags and no clinical values reproduces the reported CKD recall
exactly. Diabetes and CKD are also comorbid, so a source-specific negative
cannot be a global "healthy". The formulation is withdrawn.

If you are pooling single-disease public datasets into a multi-class corpus,
check it against a missingness-only baseline first.

## Reproducing

```bash
pip install -r requirements.txt

# Audit + per-repeat evaluation (resumable; each repeat is a few minutes on one core)
python src/data.py                              # cohort audit -> results/data_audit.json
python src/run_repeat.py TASK-DIA 0 1 2 3 4 5 6 7 8 9
python src/run_repeat.py TASK-CKD 0 1 2 3 4 5 6 7 8 9
python src/run_repeat.py TASK-SYN 0 1 2
python src/merge_parts.py                       # -> artifacts/oof_*.npz

# Studies
python src/quantization.py TASK-DIA 0 1 2 3 4   # append seeds in chunks
python src/quantization.py TASK-CKD 0 1 2 3 4
python src/tflite_export.py                     # real INT8 flatbuffer
python src/external_validation.py both

# Analysis and paper
python src/analysis.py                          # CIs, paired tests, calibration, fairness
python src/make_tables.py                       # -> results/tables.md, tables.json
python src/build_manuscript.py                  # -> MANUSCRIPT_v2_final.md
```

`run_repeat.py` and `quantization.py` are idempotent and append, so long runs can
be built up across sessions.

## Protocol notes

- Imputation and scaling are fitted **inside** each training fold. No statistic
  crosses a fold boundary.
- Class imbalance uses a **weighted loss, not SMOTE** — resampling is easy to
  apply before the split by accident; a weighted loss cannot leak.
- Epochs are chosen by early stopping on an inner validation split. With a fixed
  epoch budget, training AUROC reaches 1.000 on PIMA while test AUROC falls to
  0.77.
- Fairness uses a **correct equalized-odds difference**, max(|ΔTPR|, |ΔFPR|).
  A TPR gap alone is not equalized odds.
- Calibration is computed from the **same frozen predictions** as the
  discrimination results; nothing is retrained.

## Known limitations

- **No microcontroller was used.** No latency, no energy, no on-device
  validation. The deployment analysis is a static characterisation of an
  exported artifact. The tensor-arena figure is a lower bound.
- Both real cohorts are small and neither is contemporary.
- No ethnicity data in any cohort; the fairness audit is restricted to age, and
  the disparity found on PIMA is substantial (EOD 0.271 [0.201, 0.358]).
- Not a validated diagnostic tool. No prospective clinical validation.

## What changed, and why

| Earlier claim | Status |
|---|---|
| Three-class Healthy/Diabetes/CKD task | **Withdrawn** — solvable from missingness patterns; clinically unsound |
| Kaggle cohort as real clinical data | **Withdrawn** — synthetic |
| "QAT is necessary; PTQ collapses to F1=0.198" | **Reversed** — the collapse was a quantizer defect; PTQ matches FP32 |
| QAT results at 3.7 KB | **Withdrawn** — fake-quant hooks were removed before evaluation, so those were FP32 measurements; footprint was `params × 1 byte`, never exported |
| Generalisation gap F1 0.616 → 0.042 | **Withdrawn** — 95.0% of the "held-out" rows were in that experiment's training set, and macro-F1 was averaged over 3 classes on a 2-class cohort |
| Fairness within thresholds (EOD 0.014) | **Reversed** — that was a TPR gap computed on training data; corrected EOD on PIMA is 0.271 |
| Single seed, single split | **Replaced** — 10 CV repeats, 20 quantization seeds, bootstrap CIs, paired tests |

`verify_findings.py` reproduces each defect against the superseded code.

## Windows

See `WINDOWS_SETUP.md`. Use `.\verify_all.ps1` instead of `verify_all.sh`.

## Before submitting

`REPRODUCTION_CHECKLIST.md` lists every headline number with its expected value
so each author can reproduce the results independently. `DEFENCE_BRIEFING.md`
explains the mechanism behind the three claims most likely to be challenged in
review.

## Repository layout

```
src/          pipeline: data, models, experiments, quantization, export, analysis
results/      generated tables (tables.md, supplementary.md) and raw result JSON
artifacts/    frozen out-of-fold predictions, exported .tflite models, figure
data/raw/     the three source datasets
paper/        submitted manuscript, supplementary tables, cover letter
```

Two further documents at the repository root:
`REPRODUCTION_CHECKLIST.md` (expected values for independent verification) and
`DEFENCE_BRIEFING.md` (the mechanism behind each contested claim).

Rebuild every table and the manuscript from the frozen artifacts:

```bash
python src/make_tables.py        # -> results/tables.md, results/supplementary.md
python src/build_manuscript.py   # -> paper/MANUSCRIPT_v2_final.md
python src/to_springer.py        # -> Springer-format markdown
```

## Authors

Vivek Gondalia, Kalpesh Popat, Ashwin Dobariya
Faculty of Computer Applications, Marwadi University, Rajkot, Gujarat, India

## Citation

Citation details will be added on acceptance.

## License

See `LICENSE`.
