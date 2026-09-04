# Generated results tables

_Every value below is computed from frozen artifacts by `src/make_tables.py`. Do not edit by hand._


## Table 1. Cohorts

| Cohort | N | Endpoint | Provenance | Features | Notes |
|-------------------------|-------------|--------------|-------------------|--------------|----------------------|
| PIMA Indians Diabetes | 768 | diabetes | real | 8 | all female, age ≥ 21 |
| UCI Chronic Kidney Disease | 400 | CKD | real | 24 | sex not recorded |
| Kaggle diabetes-prediction | 100,000 | diabetes | **synthetic** | 7 | 18 distinct HbA1c values; 25.5% share the modal BMI |

## Table 2. Evidence that the Kaggle cohort is generated, not measured

| Signature | Value |
|----------------------|----------------------|
| Distinct HbA1c values | 18 |
| Distinct blood-glucose values | 18 |
| Rows at the modal BMI | 25.5% (BMI=27.32, column mean=27.32) |
| Exact duplicate rows | 3,854 |
| Minimum recorded age | 0.08 years |
| Label deterministic outside HbA1c in (5.6, 6.6] | yes |
| Positive rate inside that band | 0.0791 (n=58,248) |

## Table 2b. Record counts by source and class

_Every cohort is analysed in full under repeated stratified five-fold cross-validation, so train, validation and test counts are determined by the fold assignment rather than by a fixed split; fold assignments are recorded in `artifacts/cv_*.json`. Duplicates are exact repeated rows, reported but not removed, since removing them from a generated cohort would not make it measured._

| Source | Class | Raw | Exact duplicates | Analysed |
|-------------------------|-----------------|------------|----------------------|--------------|
| PIMA Indians Diabetes | Diabetes | 268 | 0 | 268 |
| PIMA Indians Diabetes | No diabetes | 500 | 0 | 500 |
| PIMA Indians Diabetes | All | 768 | 0 | 768 |
| Kaggle diabetes-prediction | Diabetes | 8,500 | 18 | 8,500 |
| Kaggle diabetes-prediction | No diabetes | 91,482 | 3,836 | 91,482 |
| Kaggle diabetes-prediction | All | 99,982 | 3,854 | 99,982 |
| UCI Chronic Kidney Disease | CKD | 250 | 0 | 250 |
| UCI Chronic Kidney Disease | No CKD | 150 | 0 | 150 |
| UCI Chronic Kidney Disease | All | 400 | 0 | 400 |

## Table 2c. Control baselines on the withdrawn pooled three-class task

_Depth-4 decision trees, stratified five-fold cross-validation, n=101,150. Source identity alone recovers CKD almost perfectly, while clinical values alone do not, which is why the pooled formulation is withdrawn. The single rule “haemoglobin or creatinine recorded” separates CKD at 97.6% sensitivity and 99.9% specificity._

| Baseline | Features | Macro-F1 | CKD recall |
|----------------------|--------------|--------------|----------------|
| Majority class | 0 | 0.318 | 0.000 |
| Missingness indicators only | 4 | 0.570 | 0.972 |
| Source identity only | 1 | 0.575 | 1.000 |
| Clinical values only (imputed) | 6 | 0.719 | 0.476 |
| Clinical values + missingness | 10 | 0.717 | 0.472 |

## Table 3a. PIMA Indians Diabetes (real, n=768)

_n=768, positives=268, 10 repeats. Mean ± half-width of the 95% confidence interval across repeats. AUPRC, Brier score and balanced accuracy are reported in Appendix A._

| Model | F1 | AUROC | MCC | ECE |
|---------------------|-------------------|-------------------|-------------------|-------------------|
| Logistic reg. | 0.669 ± 0.004 | 0.835 ± 0.002 | 0.476 ± 0.006 | 0.027 ± 0.005 |
| Random forest | 0.646 ± 0.007 | 0.830 ± 0.004 | 0.468 ± 0.010 | 0.019 ± 0.005 |
| XGBoost | 0.631 ± 0.010 | 0.809 ± 0.006 | 0.440 ± 0.014 | 0.129 ± 0.006 |
| MLP wide | 0.671 ± 0.009 | 0.827 ± 0.008 | 0.479 ± 0.015 | 0.043 ± 0.010 |
| MLP matched | 0.677 ± 0.008 | 0.830 ± 0.006 | 0.485 ± 0.012 | 0.029 ± 0.006 |
| TinyMed-Net | 0.675 ± 0.007 | 0.825 ± 0.005 | 0.482 ± 0.012 | 0.033 ± 0.007 |
|  − SE | 0.676 ± 0.008 | 0.823 ± 0.008 | 0.486 ± 0.011 | 0.035 ± 0.012 |
|  − residual | 0.670 ± 0.009 | 0.824 ± 0.005 | 0.472 ± 0.015 | 0.031 ± 0.007 |
|  − distillation | 0.652 ± 0.020 | 0.795 ± 0.011 | 0.440 ± 0.034 | 0.047 ± 0.016 |

## Table 3b. UCI Chronic Kidney Disease (real, n=400)

_n=400, positives=250, 10 repeats. Mean ± half-width of the 95% confidence interval across repeats. AUPRC, Brier score and balanced accuracy are reported in Appendix A._

| Model | F1 | AUROC | MCC | ECE |
|---------------------|-------------------|-------------------|-------------------|-------------------|
| Logistic reg. | 0.993 ± 0.002 | 1.000 ± 0.000 | 0.982 ± 0.004 | 0.020 ± 0.001 |
| Random forest | 0.994 ± 0.001 | 1.000 ± 0.000 | 0.984 ± 0.002 | 0.032 ± 0.002 |
| XGBoost | 0.988 ± 0.002 | 0.999 ± 0.000 | 0.969 ± 0.005 | 0.008 ± 0.001 |
| MLP wide | 0.955 ± 0.025 | 0.990 ± 0.008 | 0.897 ± 0.049 | 0.347 ± 0.022 |
| MLP matched | 0.865 ± 0.036 | 0.950 ± 0.028 | 0.691 ± 0.097 | 0.243 ± 0.035 |
| TinyMed-Net | 0.896 ± 0.028 | 0.978 ± 0.015 | 0.787 ± 0.048 | 0.227 ± 0.028 |
|  − SE | 0.905 ± 0.035 | 0.982 ± 0.010 | 0.804 ± 0.053 | 0.217 ± 0.029 |
|  − residual | 0.862 ± 0.032 | 0.959 ± 0.023 | 0.699 ± 0.092 | 0.205 ± 0.028 |
|  − distillation | 0.886 ± 0.036 | 0.978 ± 0.012 | 0.762 ± 0.075 | 0.161 ± 0.030 |

## Table 3c. Kaggle diabetes-prediction (SYNTHETIC, 5k subsample)

_n=5000, positives=425, 10 repeats. Mean ± half-width of the 95% confidence interval across repeats. AUPRC, Brier score and balanced accuracy are reported in Appendix A._

| Model | F1 | AUROC | MCC | ECE |
|---------------------|-------------------|-------------------|-------------------|-------------------|
| Logistic reg. | 0.553 ± 0.001 | 0.957 ± 0.000 | 0.545 ± 0.001 | 0.013 ± 0.001 |
| Random forest | 0.708 ± 0.003 | 0.964 ± 0.001 | 0.680 ± 0.003 | 0.032 ± 0.001 |
| XGBoost | 0.750 ± 0.004 | 0.966 ± 0.001 | 0.739 ± 0.005 | 0.013 ± 0.001 |
| MLP wide | 0.539 ± 0.006 | 0.959 ± 0.001 | 0.531 ± 0.005 | 0.040 ± 0.004 |
| MLP matched | 0.539 ± 0.010 | 0.955 ± 0.002 | 0.531 ± 0.008 | 0.024 ± 0.002 |
| TinyMed-Net | 0.554 ± 0.007 | 0.960 ± 0.001 | 0.545 ± 0.006 | 0.025 ± 0.004 |
|  − SE | 0.561 ± 0.003 | 0.960 ± 0.001 | 0.550 ± 0.003 | 0.024 ± 0.003 |
|  − residual | 0.561 ± 0.008 | 0.960 ± 0.001 | 0.550 ± 0.007 | 0.023 ± 0.005 |
|  − distillation | 0.563 ± 0.013 | 0.959 ± 0.002 | 0.551 ± 0.012 | 0.021 ± 0.004 |

## Table 4a. Paired comparison against TinyMed-Net, PIMA Indians Diabetes (real, n=768)

_Positive difference favours TinyMed-Net. Identical folds and identical test predictions; Wilcoxon signed-rank across repeats._

| Comparator | Delta F1 [95% CI] | p | repeats |
|---------------------|----------------------|------------|-------------|
|  − SE | -0.0019 [-0.0118, +0.0081] | 0.7695 | 10 |
|  − residual | +0.0041 [-0.0079, +0.0161] | 0.6953 | 10 |
|  − distillation | +0.0225 [+0.0011, +0.0439] | 0.0371 | 10 |
| MLP matched | -0.0025 [-0.0129, +0.0079] | 0.5566 | 10 |
| MLP wide | +0.0039 [-0.0081, +0.0158] | 0.4316 | 10 |
| Logistic reg. | +0.0056 [-0.0031, +0.0144] | 0.6250 | 10 |
| Random forest | +0.0282 [+0.0176, +0.0389] | 0.0020 | 10 |
| XGBoost | +0.0431 [+0.0318, +0.0544] | 0.0020 | 10 |

## Table 4b. Paired comparison against TinyMed-Net, UCI Chronic Kidney Disease (real, n=400)

_Positive difference favours TinyMed-Net. Identical folds and identical test predictions; Wilcoxon signed-rank across repeats._

| Comparator | Delta F1 [95% CI] | p | repeats |
|---------------------|----------------------|------------|-------------|
|  − SE | -0.0093 [-0.0544, +0.0358] | 0.6953 | 10 |
|  − residual | +0.0335 [+0.0146, +0.0524] | 0.0039 | 10 |
|  − distillation | +0.0098 [-0.0064, +0.0259] | 0.2324 | 10 |
| MLP matched | +0.0304 [-0.0244, +0.0851] | 0.2754 | 10 |
| MLP wide | -0.0596 [-0.1042, -0.0150] | 0.0195 | 10 |
| Logistic reg. | -0.0972 [-0.1259, -0.0686] | 0.0020 | 10 |
| Random forest | -0.0983 [-0.1267, -0.0699] | 0.0020 | 10 |
| XGBoost | -0.0925 [-0.1204, -0.0645] | 0.0020 | 10 |

## Table 5a. Quantization, PIMA Indians Diabetes (real, n=768)

_`PTQ_legacy` reproduces the quantizer used in the original implementation. All other rows are real INT8 models evaluated through integer kernels. Mean ± SD over seeds._

| Condition | F1 | AUROC | Param KiB | seeds |
|---------------------------|---------------------|---------------------|---------------|-----------|
| FP32 | 0.6429 ± 0.0713 | 0.8031 ± 0.0131 | 15.40 | 20 |
| PTQ_legacy | 0.2587 ± 0.2654 | 0.6531 ± 0.1909 | 3.85 | 20 |
| PTQ_validated_fbgemm | 0.6456 ± 0.0662 | 0.8038 ± 0.0131 | 3.85 | 20 |
| QAT_validated_fbgemm | 0.5872 ± 0.1978 | 0.8006 ± 0.0140 | 3.85 | 20 |
| PTQ_validated_qnnpack | 0.6477 ± 0.0722 | 0.8039 ± 0.0127 | 3.85 | 20 |
| QAT_validated_qnnpack | 0.6474 ± 0.0220 | 0.7994 ± 0.0105 | 3.85 | 20 |

_Failure mode of the legacy quantizer: a BatchNorm scale vector with mean 1.0093 is mapped to mean 0.2752, collapsing to 1 distinct value (27.3% of true scale)._


## Table 5b. Quantization, UCI Chronic Kidney Disease (real, n=400)

_`PTQ_legacy` reproduces the quantizer used in the original implementation. All other rows are real INT8 models evaluated through integer kernels. Mean ± SD over seeds._

| Condition | F1 | AUROC | Param KiB | seeds |
|---------------------------|---------------------|---------------------|---------------|-----------|
| FP32 | 0.7945 ± 0.2382 | 0.9685 ± 0.0339 | 15.40 | 20 |
| PTQ_legacy | 0.3444 ± 0.3907 | 0.5668 ± 0.1867 | 3.85 | 20 |
| PTQ_validated_fbgemm | 0.8087 ± 0.2249 | 0.9637 ± 0.0302 | 3.85 | 20 |
| QAT_validated_fbgemm | 0.8774 ± 0.1164 | 0.9622 ± 0.0282 | 3.85 | 20 |
| PTQ_validated_qnnpack | 0.8007 ± 0.2340 | 0.9703 ± 0.0317 | 3.85 | 20 |
| QAT_validated_qnnpack | 0.8694 ± 0.1208 | 0.9682 ± 0.0252 | 3.85 | 20 |

_Failure mode of the legacy quantizer: a BatchNorm scale vector with mean 1.0093 is mapped to mean 0.2752, collapsing to 1 distinct value (27.3% of true scale)._


## Table 6. Exported INT8 deployment artifact

_Measured from the exported TFLite flatbuffer, not estimated from parameter counts. The arena figure is a lower bound from the tensor inventory; the true requirement depends on the TFLite Micro memory planner. No physical microcontroller was available._

| Quantity | PIMA (n=768) | UCI-CKD (n=400) |
|----------------------|------------------|---------------------|
| Parameters | 4,294 | 4,294 |
| Parameter bytes, INT8 (KiB) | 4.19 | 4.19 |
| Exported TFLite file (KiB) | 19.59 | 19.59 |
| TFLite before name stripping (KiB) | 31.27 | 31.34 |
| Tensor-arena lower bound (KiB) | 8.45 | 15.48 |
| Operators | 10 | 10 |
| Float tensors remaining | 0 | 0 |
| F1, FP32 | 0.611 | 0.775 |
| F1, INT8 | 0.615 | 0.770 |
| FP32/INT8 argmax agreement | 0.969 | 0.980 |

_Operator set required by the runtime: ADD, AVERAGE_POOL_2D, CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, LOGISTIC, MUL, RESHAPE, SOFTMAX._


## Table 7. External validation across cohorts

_Shared features only (age, BMI, glucose); identical binary endpoint; the two cohorts are separate datasets so overlap is impossible. Within-cohort rows are the reference achievable in each cohort. F1 at a fixed 0.5 threshold is not comparable across cohorts of different prevalence (8.5% vs 34.9%), so AUROC is given alongside._

| Setting | Model | Train → test | F1 | AUROC |
|----------------------|-------------------|----------------------|-------------------|-----------|
| unrestricted | TinyMed-Net | Synthetic (within) | 0.392 | 0.907 |
| unrestricted | TinyMed-Net | Synthetic → PIMA | 0.324 ± 0.078 | 0.770 |
| unrestricted | TinyMed-Net | PIMA (within) | 0.667 | 0.825 |
| unrestricted | TinyMed-Net | PIMA → synthetic | 0.278 ± 0.020 | 0.834 |
| unrestricted | Logistic reg. | Synthetic (within) | 0.406 | 0.896 |
| unrestricted | Logistic reg. | Synthetic → PIMA | 0.426 ± 0.012 | 0.826 |
| unrestricted | Logistic reg. | PIMA (within) | 0.657 | 0.827 |
| unrestricted | Logistic reg. | PIMA → synthetic | 0.269 ± 0.000 | 0.888 |
| eligibility matched | TinyMed-Net | Synthetic (within) | 0.453 | 0.901 |
| eligibility matched | TinyMed-Net | Synthetic → PIMA | 0.294 ± 0.031 | 0.785 |
| eligibility matched | TinyMed-Net | PIMA (within) | 0.667 | 0.825 |
| eligibility matched | TinyMed-Net | PIMA → synthetic | 0.259 ± 0.021 | 0.804 |
| eligibility matched | Logistic reg. | Synthetic (within) | 0.413 | 0.884 |
| eligibility matched | Logistic reg. | Synthetic → PIMA | 0.368 ± 0.024 | 0.826 |
| eligibility matched | Logistic reg. | PIMA (within) | 0.657 | 0.827 |
| eligibility matched | Logistic reg. | PIMA → synthetic | 0.252 ± 0.000 | 0.870 |

_Shared features: age, BMI, glucose. PIMA glucose is a 2-hour OGTT value; the comparator records an unspecified blood glucose level, so transfer gaps confound population and measurement shift._


## Table 8. Fairness on held-out predictions

_Equalized-odds difference (EOD) is max(|TPR gap|, |FPR gap|), not the TPR gap alone. DPD is the demographic parity difference. Computed on out-of-fold predictions only, with percentile bootstrap 95% confidence intervals._

| Cohort | Comparison | Group n | Metric | Estimate | 95% CI |
|----------------------|----------------------|-----------------|-------------|--------------|-----------------|
| PIMA (n=768) | age: ≥29 vs <29 | 401 / 367 | TPR gap | 0.198 | 0.075–0.315 |
| PIMA (n=768) | age: ≥29 vs <29 | 401 / 367 | FPR gap | 0.271 | 0.189–0.357 |
| PIMA (n=768) | age: ≥29 vs <29 | 401 / 367 | EOD | 0.271 | 0.201–0.358 |
| PIMA (n=768) | age: ≥29 vs <29 | 401 / 367 | DPD | 0.386 | 0.327–0.448 |
| UCI-CKD (n=400) | age: ≥55 vs <55 | 196 / 204 | TPR gap | 0.097 | 0.024–0.167 |
| UCI-CKD (n=400) | age: ≥55 vs <55 | 196 / 204 | FPR gap | 0.000 | 0.000–0.000 |
| UCI-CKD (n=400) | age: ≥55 vs <55 | 196 / 204 | EOD | 0.097 | 0.024–0.167 |
| UCI-CKD (n=400) | age: ≥55 vs <55 | 196 / 204 | DPD | 0.258 | 0.166–0.357 |
| Synthetic (n=5,000) | sex: Female vs Male | 2949 / 2051 | TPR gap | 0.011 | 0.001–0.076 |
| Synthetic (n=5,000) | sex: Female vs Male | 2949 / 2051 | FPR gap | 0.016 | 0.001–0.032 |
| Synthetic (n=5,000) | sex: Female vs Male | 2949 / 2051 | EOD | 0.016 | 0.008–0.076 |
| Synthetic (n=5,000) | sex: Female vs Male | 2949 / 2051 | DPD | 0.000 | 0.000–0.024 |

## Table 10. Cross-fitted temperature scaling, TinyMed-Net

_A single scalar fitted by NLL on K-1 folds of the out-of-fold probabilities and applied to the held-out fold, so no patient contributes to the temperature used to score them. Ranking is unchanged, so AUROC is unaffected._

| Cohort | Temperature | ECE before | ECE after | Brier before | Brier after |
|----------------------|-------------------|----------------|---------------|------------------|-----------------|
| PIMA (n=768) | 0.909 ± 0.038 | 0.044 | 0.029 | 0.158 | 0.159 |
| UCI-CKD (n=400) | 0.139 ± 0.007 | 0.297 | 0.009 | 0.128 | 0.038 |
| Synthetic (n=5,000) | 1.065 ± 0.019 | 0.022 | 0.023 | 0.073 | 0.073 |
