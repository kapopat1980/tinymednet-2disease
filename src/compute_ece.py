import torch, numpy as np, pandas as pd, json
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tinymednet import TinyMedNet, SimpleMLP
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

torch.manual_seed(42); np.random.seed(42)
X_COLS = ['age', 'bp', 'glucose', 'bmi', 'hemoglobin', 'creatinine',
          'bp_missing', 'hemoglobin_missing', 'creatinine_missing']
train_df = pd.read_csv('../data/processed/train.csv'); test_df = pd.read_csv('../data/processed/test.csv')
Xtr_raw = train_df[X_COLS].values.astype(np.float32); ytr = train_df['y'].values.astype(np.int64)
Xte_raw = test_df[X_COLS].values.astype(np.float32); yte = test_df['y'].values.astype(np.int64)
scaler = StandardScaler(); Xtr = scaler.fit_transform(Xtr_raw); Xte = scaler.transform(Xte_raw)
sm = SMOTE(random_state=42, k_neighbors=5); Xtr_b, ytr_b = sm.fit_resample(Xtr, ytr)

def ece_score(probs, labels, n_bins=10):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            acc = accuracies[mask].mean()
            conf = confidences[mask].mean()
            ece += (mask.sum()/len(labels)) * abs(acc - conf)
    return ece

results = {}

# Retrain TinyMedNet (same procedure as before) to get probs
model = TinyMedNet(len(X_COLS), 3, use_se=True, use_residual=True)
teacher = SimpleMLP(len(X_COLS), 3)
Xtr_t = torch.tensor(Xtr_b, dtype=torch.float32); ytr_t = torch.tensor(ytr_b, dtype=torch.long)
opt_t = torch.optim.Adam(teacher.parameters(), lr=1e-3)
ce = torch.nn.CrossEntropyLoss()
n = Xtr_t.shape[0]; bs=128
for ep in range(10):
    perm = torch.randperm(n)
    for i in range(0,n,bs):
        idx=perm[i:i+bs]; opt_t.zero_grad()
        loss = ce(teacher(Xtr_t[idx]), ytr_t[idx]); loss.backward(); opt_t.step()
teacher.eval()

opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
for ep in range(12):
    perm = torch.randperm(n)
    for i in range(0,n,bs):
        idx=perm[i:i+bs]; opt.zero_grad()
        out = model(Xtr_t[idx])
        with torch.no_grad(): t_out = teacher(Xtr_t[idx])
        soft = F.kl_div(F.log_softmax(out/3.0,dim=1), F.softmax(t_out/3.0,dim=1), reduction='batchmean')*9.0
        loss = 0.5*soft + 0.5*ce(out, ytr_t[idx]); loss.backward(); opt.step()
model.eval()
with torch.no_grad():
    probs_tmn = F.softmax(model(torch.tensor(Xte,dtype=torch.float32)), dim=1).numpy()
results['TinyMedNet'] = ece_score(probs_tmn, yte)

mlp = SimpleMLP(len(X_COLS), 3)
opt_m = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
for ep in range(12):
    perm = torch.randperm(n)
    for i in range(0,n,bs):
        idx=perm[i:i+bs]; opt_m.zero_grad()
        loss = ce(mlp(Xtr_t[idx]), ytr_t[idx]); loss.backward(); opt_m.step()
mlp.eval()
with torch.no_grad():
    probs_mlp = F.softmax(mlp(torch.tensor(Xte,dtype=torch.float32)), dim=1).numpy()
results['MLP'] = ece_score(probs_mlp, yte)

lr = LogisticRegression(max_iter=1000); lr.fit(Xtr_b, ytr_b)
results['LogisticRegression'] = ece_score(lr.predict_proba(Xte), yte)

rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(Xtr_b, ytr_b)
results['RandomForest'] = ece_score(rf.predict_proba(Xte), yte)

xgbm = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, objective='multi:softprob',
                          num_class=3, random_state=42, eval_metric='mlogloss')
xgbm.fit(Xtr_b, ytr_b)
results['XGBoost'] = ece_score(xgbm.predict_proba(Xte), yte)

print(json.dumps(results, indent=2))
with open('../results/ece_results.json','w') as f: json.dump(results, f, indent=2)
