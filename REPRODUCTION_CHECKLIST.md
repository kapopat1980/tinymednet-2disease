# Reproduction and verification checklist

**Purpose.** Every number in the manuscript was produced by code in this
repository. Before submission each author should reproduce the headline results
independently and be able to explain them without reference to this document.
Work through this in one sitting; it takes roughly half a day, most of which is
the machine running unattended.

Tick each box only after you have seen the number on your own screen.

---

## Quick route

If you would rather not run the steps individually:

Linux or macOS:

```bash
bash verify_all.sh            # full run, about 2 hours on one core
bash verify_all.sh --quick    # skips the 20-seed quantization study, ~25 min
```

On a multi-core machine, run the repeats in parallel first (each fit is
single-threaded by design, so this is close to a linear speedup):

```powershell
.\run_parallel.ps1 -Workers 8
```

Windows (PowerShell) — see `WINDOWS_SETUP.md` first:

```powershell
.\verify_all.ps1
.\verify_all.ps1 -Quick
```

It logs everything to `results/verification_run.log` and finishes by running
`src/check_expected.py`, which prints a PASS/FAIL line for every headline number
and exits non-zero if anything falls outside tolerance. You can also run that
comparison on its own at any time:

```bash
python src/check_expected.py
```

The sections below explain what each step is doing and what to look at, which
matters if you intend to defend the numbers rather than only reproduce them.

---

## 0. Environment

Linux or macOS:

```bash
git clone <your-repo-url> && cd tinymednet-2disease
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch, tensorflow, xgboost; print('ok')"
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Windows, read `WINDOWS_SETUP.md` before starting. Two things differ: the
`qnnpack` quantization backend is usually unavailable on Windows x86 (harmless,
and explained there), and TensorFlow may need an older release or WSL2.

- [ ] Installs cleanly on your machine
- [ ] `data/raw/` contains `pima_diabetes.csv`, `ckd.csv`,
      `diabetes_prediction_dataset.csv`

**Note on exact reproduction.** Torch and scikit-learn are deterministic given a
seed on the same versions, but results can shift in the third decimal across
library versions or CPU architectures. Treat a discrepancy under ±0.005 as
version drift; anything larger means something is genuinely different and must
be understood before you submit.

---

## 1. The provenance audit (2 minutes)

```bash
python src/data.py
cat results/data_audit.json
```

Expect exactly:

| Quantity | Value |
|---|---|
| Distinct HbA1c values | 18 |
| Distinct blood-glucose values | 18 |
| Modal BMI / column mean | 27.32 / 27.3208 |
| Rows at modal BMI | 25.495% |
| Exact duplicate rows | 3,854 |
| Minimum age | 0.08 years |
| Label deterministic outside HbA1c (5.6, 6.6] | true |
| Ambiguous-band n / positive rate | 58,248 / 0.0791 |

- [ ] All eight reproduce
- [ ] I can state in one sentence why this cohort cannot support a clinical claim

---

## 2. The withdrawn three-class task (2 minutes)

```bash
python verify_findings.py
```

- [ ] "haemoglobin or creatinine recorded" separates CKD at **96.0% sensitivity,
      98.9% specificity**
- [ ] A tree using only missingness flags, with no clinical values, reaches CKD
      recall **0.960** — the same recall the earlier manuscript reported for
      TinyMed-Net
- [ ] I understand why this makes the pooled formulation indefensible

---

## 3. Main evaluation (about 70 minutes on one core; the synthetic cohort dominates)

```bash
python src/run_repeat.py TASK-DIA 0 1 2 3 4 5 6 7 8 9
python src/run_repeat.py TASK-CKD 0 1 2 3 4 5 6 7 8 9
python src/run_repeat.py TASK-SYN 0 1 2 3 4 5 6 7 8 9   # ~4 min per repeat
python src/merge_parts.py
python src/analysis.py
```

Expect (mean across 10 repeats):

| | TinyMed-Net F1 | TinyMed-Net AUROC | Logistic reg. F1 | Logistic reg. AUROC |
|---|---|---|---|---|
| PIMA | 0.6745 | 0.8253 | 0.6689 | 0.8345 |
| UCI-CKD | 0.8957 | 0.9785 | 0.9929 | 1.0000 |

Paired comparison against logistic regression:

- PIMA: ΔF1 = **+0.0056**, p = **0.625** — not significant
- CKD: ΔF1 = **−0.0972**, p = **0.00195** — logistic regression significantly better

- [ ] Values reproduce within ±0.005
- [ ] **I accept that the paper's own result is that our architecture does not
      beat logistic regression.** This is the single most likely question at
      review; do not submit until all three authors are comfortable saying it out
      loud.

---

## 4. Quantization (about 40 minutes)

```bash
python src/quantization.py TASK-DIA 0 1 2 3 4
python src/quantization.py TASK-DIA 5 6 7 8 9
python src/quantization.py TASK-DIA 10 11 12 13 14
python src/quantization.py TASK-DIA 15 16 17 18 19
# repeat the four lines for TASK-CKD
```

Expect (mean AUROC over 20 seeds):

| Condition | PIMA | CKD |
|---|---|---|
| FP32 | 0.8031 | 0.9685 |
| PTQ, legacy quantizer | **0.6531** | **0.5668** |
| PTQ validated, fbgemm | 0.8038 | 0.9637 |
| PTQ validated, qnnpack | 0.8039 | 0.9703 |
| QAT validated, fbgemm | 0.8006 | 0.9622 |
| QAT validated, qnnpack | 0.7994 | 0.9682 |

BatchNorm demonstration: γ of mean **1.0093** → mean **0.2752**, collapsed to
**1 distinct value** (27.3% of true scale).

- [ ] Values reproduce
- [ ] **Read `src/quantization.py` lines defining `legacy_fake_quantize`.**
      Satisfy yourself that `scale = (max − min)/255` with rounding into
      `[−128, 127]` and no zero point does crush a strictly-positive tensor.
      This is the paper's central technical claim and a reviewer will probe it.

---

## 5. Deployment artifact (about 10 minutes)

```bash
python src/tflite_export.py
ls -l artifacts/*.tflite
```

- [ ] `.tflite` file is **20,056 bytes**, parameters are **4,294 bytes** → the
      **4.7×** gap that the paper reports
- [ ] Exported model uses **10 operators**, **zero float tensors**
- [ ] I can explain why parameter-count footprints understate real Flash

---

## 6. Fairness and calibration

```bash
python src/analysis.py
```

- [ ] PIMA age EOD = **0.271**, CI [0.201, 0.358] — well above the 0.05 threshold
- [ ] CKD ECE **0.297 → 0.009** after cross-fitted temperature scaling
      (temperature 0.139)
- [ ] I understand that EOD here is max(|ΔTPR|, |ΔFPR|), and why a TPR gap alone
      is not equalized odds

---

## 7. Regenerate the paper

```bash
python src/make_tables.py
python src/build_manuscript.py
python src/to_springer.py
```

- [ ] `results/tables.md` matches the tables in the submitted manuscript
- [ ] Spot-check three numbers in the Word file against `results/tables.json`

---

## 8. Before you submit

- [ ] Corresponding author email and three ORCIDs added
- [ ] **Author contributions rewritten by the authors.** The current statement is
      an inference, not a record. It is a declaration of fact that all three of
      you sign.
- [ ] AI-use disclosure added to the Methods section (see below)
- [ ] Reference [22], ADA *Standards of Care* — pin the edition year
- [ ] Cover letter: editor name, journal name, and whether the earlier version
      was formally withdrawn from its previous venue
- [ ] Target journal chosen and its table/abstract/reference limits checked
- [ ] Existing repo tagged `v1-superseded`, then this version pushed

---

## The AI-use disclosure

Springer Nature requires disclosure of AI use beyond basic copy editing, in the
Methods section, naming the tool and describing what it did. Copy editing is
exempt; what happened here is not copy editing.

Suggested wording, to be adjusted so that it is **true of your final workflow**:

> Claude (Anthropic) was used to implement the analysis pipeline, execute the
> reported experiments, and prepare an initial draft of this manuscript. All
> authors independently reproduced the reported results from the public
> repository, verified the code, and take full responsibility for the content of
> this article.

Do not write the second sentence until section 3 and section 4 above are ticked.
Authorship remains human and accountability cannot be delegated; the disclosure
is what makes that division honest and visible, and it costs the paper nothing.
