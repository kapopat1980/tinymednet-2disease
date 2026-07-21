"""
Real-data harmonization pipeline for the rescoped 2-condition (Diabetes / CKD) study.
All source data is genuinely real and publicly available:
  - PIMA Indians Diabetes (UCI, via jbrownlee/Datasets mirror), n=768
  - Diabetes Prediction Dataset (Kaggle, iammustafatz), n=100,000 (subsampled)
  - Chronic Kidney Disease (UCI, Rubini/Soundarapandian/Eswaran 2015), n=400
"""
import pandas as pd
import numpy as np

np.random.seed(42)

# ---------- 1. PIMA Indians Diabetes ----------
pima = pd.read_csv('../data/raw/pima_diabetes.csv', header=None,
                    names=['pregnancies','glucose','bp','skin','insulin','bmi','pedigree','age','outcome'])
# 0s in glucose/bp/skin/insulin/bmi are missing-value encodings in this dataset (well-documented quirk)
for col in ['glucose','bp','skin','insulin','bmi']:
    pima[col] = pima[col].replace(0, np.nan)

pima_h = pd.DataFrame({
    'age': pima['age'].astype(float),
    'bp': pima['bp'],
    'glucose': pima['glucose'],
    'bmi': pima['bmi'],
    'hemoglobin': np.nan,
    'creatinine': np.nan,
    'label': np.where(pima['outcome'] == 1, 'Diabetes', 'Healthy'),
    'source': 'PIMA',
    'sex': 'Female',  # PIMA cohort is all female by study design (real, documented fact)
})

# ---------- 2. Kaggle Diabetes Prediction Dataset (real, 100k rows -> subsample) ----------
kdb = pd.read_csv('../data/raw/diabetes_prediction_dataset.csv')
kdb = kdb[kdb['gender'].isin(['Male', 'Female'])].copy()
# Stratified subsample: keep all positives, sample negatives down to a manageable, honest N
pos = kdb[kdb['diabetes'] == 1]
neg = kdb[kdb['diabetes'] == 0].sample(n=6000, random_state=42)
kdb_s = pd.concat([pos, neg], ignore_index=True)

kdb_h = pd.DataFrame({
    'age': kdb_s['age'].astype(float),
    'bp': np.nan,
    'glucose': kdb_s['blood_glucose_level'].astype(float),
    'bmi': kdb_s['bmi'].astype(float),
    'hemoglobin': np.nan,
    'creatinine': np.nan,
    'label': np.where(kdb_s['diabetes'] == 1, 'Diabetes', 'Healthy'),
    'source': 'Kaggle-DPD',
    'sex': kdb_s['gender'],
})

# ---------- 3. UCI Chronic Kidney Disease ----------
ckd = pd.read_csv('../data/raw/ckd.csv')
ckd['class'] = ckd['class'].astype(str).str.strip()
ckd_h = pd.DataFrame({
    'age': pd.to_numeric(ckd['age'], errors='coerce'),
    'bp': pd.to_numeric(ckd['bp'], errors='coerce'),
    'glucose': pd.to_numeric(ckd['bgr'], errors='coerce'),
    'bmi': np.nan,
    'hemoglobin': pd.to_numeric(ckd['hemo'], errors='coerce'),
    'creatinine': pd.to_numeric(ckd['sc'], errors='coerce'),
    'label': np.where(ckd['class'].str.contains('notckd'), 'Healthy', 'CKD'),
    'source': 'UCI-CKD',
    'sex': np.nan,  # not recorded in this dataset (real limitation, will be disclosed)
})

# ---------- Combine ----------
full = pd.concat([pima_h, kdb_h, ckd_h], ignore_index=True)
raw_counts = full.groupby(['source']).size()
print("Raw counts per source:\n", raw_counts)
print("\nRaw label distribution:\n", full['label'].value_counts())

# Drop exact duplicate rows (de-duplication step, same principle as original paper)
before = len(full)
full = full.drop_duplicates(subset=['age','bp','glucose','bmi','hemoglobin','creatinine','label','source'])
after = len(full)
print(f"\nDe-duplication: {before} -> {after} ({before-after} exact duplicates removed)")

full.to_csv('../data/processed/harmonized_raw.csv', index=False)
print("\nSaved harmonized_raw.csv:", full.shape)
print("\nFinal label distribution:\n", full['label'].value_counts())
print("\nMissingness per column:\n", full[['age','bp','glucose','bmi','hemoglobin','creatinine']].isnull().mean().round(3))
