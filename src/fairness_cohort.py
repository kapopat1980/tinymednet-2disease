import torch, torch.nn as nn
import numpy as np, pandas as pd, json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from tinymednet import TinyMedNet

torch.manual_seed(42); np.random.seed(42)
X_COLS = ['age', 'bp', 'glucose', 'bmi', 'hemoglobin', 'creatinine',
          'bp_missing', 'hemoglobin_missing', 'creatinine_missing']

df = pd.read_csv('../data/processed/harmonized_raw.csv')
label_map = {'Healthy': 0, 'Diabetes': 1, 'CKD': 2}
df['y'] = df['label'].map(label_map)
for col in ['bp', 'hemoglobin', 'creatinine']:
    df[f'{col}_missing'] = df[col].isnull().astype(int)

# ============ Held-out cohort test: train on Kaggle-DPD + UCI-CKD, test on PIMA ============
# This is a REAL distribution-shift test: PIMA is a different population (all-female, Arizona
# Pima Indian heritage, 1980s data collection) than the Kaggle-DPD source used for training.
train_src = df[df['source'] != 'PIMA'].copy()
cohort_pima = df[df['source'] == 'PIMA'].copy()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
train_src[X_COLS[:6]] = imputer.fit_transform(train_src[X_COLS[:6]])
cohort_pima[X_COLS[:6]] = imputer.transform(cohort_pima[X_COLS[:6]])

scaler = StandardScaler()
Xtr = scaler.fit_transform(train_src[X_COLS])
Xcohort = scaler.transform(cohort_pima[X_COLS])
ytr = train_src['y'].values
ycohort = cohort_pima['y'].values

sm = SMOTE(random_state=42, k_neighbors=5)
Xtr_b, ytr_b = sm.fit_resample(Xtr, ytr)
Xtr_t = torch.tensor(Xtr_b, dtype=torch.float32); ytr_t = torch.tensor(ytr_b, dtype=torch.long)

model = TinyMedNet(len(X_COLS), 3, use_se=True, use_residual=True)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
ce = nn.CrossEntropyLoss()
n = Xtr_t.shape[0]; bs=128
for ep in range(15):
    perm = torch.randperm(n)
    for i in range(0, n, bs):
        idx = perm[i:i+bs]
        opt.zero_grad()
        out = model(Xtr_t[idx])
        loss = ce(out, ytr_t[idx])
        loss.backward(); opt.step()

model.eval()
with torch.no_grad():
    preds_cohort = model(torch.tensor(Xcohort, dtype=torch.float32)).argmax(dim=1).numpy()
f1_cohort = f1_score(ycohort, preds_cohort, average='macro')
print(f"Held-out PIMA cohort (n={len(ycohort)}): macro-F1 = {f1_cohort:.4f}")
print("PIMA cohort label distribution:", pd.Series(ycohort).value_counts().to_dict())

# For comparison: in-distribution test performance (same model, standard test split)
test_df = pd.read_csv('../data/processed/test.csv')
Xte_std = scaler.transform(test_df[X_COLS])
yte_std = test_df['y'].values
with torch.no_grad():
    preds_std = model(torch.tensor(Xte_std, dtype=torch.float32)).argmax(dim=1).numpy()
f1_std = f1_score(yte_std, preds_std, average='macro')
print(f"In-distribution standard test set: macro-F1 = {f1_std:.4f}")

cohort_results = dict(
    in_distribution_f1=f1_std,
    pima_heldout_cohort_f1=f1_cohort,
    pima_cohort_n=int(len(ycohort))
)

# ============ Fairness audit (sex: available for PIMA + Kaggle-DPD rows, not UCI-CKD) ============
sex_df = df[df['sex'].notna()].copy()
sex_df[X_COLS[:6]] = imputer.transform(sex_df[X_COLS[:6]])
Xsex = scaler.transform(sex_df[X_COLS])
model.eval()
with torch.no_grad():
    preds_sex = model(torch.tensor(Xsex, dtype=torch.float32)).argmax(dim=1).numpy()
sex_df['pred'] = preds_sex
sex_df['correct'] = (sex_df['pred'] == sex_df['y']).astype(int)

# Demographic Parity Difference: difference in positive-prediction rate (any disease) across sex
sex_df['pred_positive'] = (sex_df['pred'] != 0).astype(int)
dpd_by_sex = sex_df.groupby('sex')['pred_positive'].mean()
dpd = abs(dpd_by_sex.get('Male', np.nan) - dpd_by_sex.get('Female', np.nan))

# Equalized Odds Difference: difference in true positive rate (recall) across sex, among actual positives
pos = sex_df[sex_df['y'] != 0]
tpr_by_sex = pos.groupby('sex')['correct'].mean()
eod = abs(tpr_by_sex.get('Male', np.nan) - tpr_by_sex.get('Female', np.nan))

print(f"\nDemographic Parity by sex: {dpd_by_sex.to_dict()}  -> DPD = {dpd:.4f}")
print(f"TPR by sex (actual positives): {tpr_by_sex.to_dict()}  -> EOD = {eod:.4f}")

fairness_results = dict(
    dpd_by_sex=dpd_by_sex.to_dict(),
    dpd=dpd,
    tpr_by_sex=tpr_by_sex.to_dict(),
    eod=eod,
    n_evaluated=int(len(sex_df)),
    note="Sex not recorded in UCI-CKD source; audit limited to PIMA+Kaggle-DPD rows (diabetes/healthy only)."
)

with open('../results/cohort_fairness_results.json', 'w') as f:
    json.dump(dict(cohort=cohort_results, fairness=fairness_results), f, indent=2, default=str)

print(json.dumps(dict(cohort=cohort_results, fairness=fairness_results), indent=2, default=str))
