import pandas as pd, numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

np.random.seed(42)
X_COLS = ['age', 'bp', 'glucose', 'bmi', 'hemoglobin', 'creatinine',
          'bp_missing', 'hemoglobin_missing', 'creatinine_missing']

train_df = pd.read_csv('../data/processed/train.csv')
test_df = pd.read_csv('../data/processed/test.csv')

Xtr_raw = train_df[X_COLS].values.astype(np.float32)
ytr = train_df['y'].values.astype(np.int64)
Xte_raw = test_df[X_COLS].values.astype(np.float32)
yte = test_df['y'].values.astype(np.int64)

scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr_raw)
Xte = scaler.transform(Xte_raw)

sm = SMOTE(random_state=42, k_neighbors=5)
Xtr_b, ytr_b = sm.fit_resample(Xtr, ytr)

def evaluate(model, Xte, yte, has_proba=True):
    preds = model.predict(Xte)
    f1 = f1_score(yte, preds, average='macro')
    mcc = matthews_corrcoef(yte, preds)
    try:
        proba = model.predict_proba(Xte)
        auc = roc_auc_score(yte, proba, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')
    return dict(macro_f1=f1, mcc=mcc, auc=auc)

results = {}

print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(Xtr_b, ytr_b)
results['LogisticRegression'] = evaluate(lr, Xte, yte)
results['LogisticRegression']['n_params'] = int(lr.coef_.size + lr.intercept_.size)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(Xtr_b, ytr_b)
results['RandomForest'] = evaluate(rf, Xte, yte)
# estimate size: total nodes across all trees
n_nodes = sum(t.tree_.node_count for t in rf.estimators_)
results['RandomForest']['n_nodes'] = int(n_nodes)

print("Training XGBoost...")
xgbm = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          objective='multi:softprob', num_class=3, random_state=42,
                          eval_metric='mlogloss')
xgbm.fit(Xtr_b, ytr_b)
results['XGBoost'] = evaluate(xgbm, Xte, yte)

with open('../results/baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save model sizes to disk to get REAL byte counts
import pickle, os
os.makedirs('../results/model_dumps', exist_ok=True)
with open('../results/model_dumps/lr.pkl', 'wb') as f: pickle.dump(lr, f)
with open('../results/model_dumps/rf.pkl', 'wb') as f: pickle.dump(rf, f)
with open('../results/model_dumps/xgb.pkl', 'wb') as f: pickle.dump(xgbm, f)

for name, path in [('LogisticRegression','lr.pkl'), ('RandomForest','rf.pkl'), ('XGBoost','xgb.pkl')]:
    size_kb = os.path.getsize(f'../results/model_dumps/{path}') / 1024
    results[name]['serialized_size_KB'] = round(size_kb, 1)

with open('../results/baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
