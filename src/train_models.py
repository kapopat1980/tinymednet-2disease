import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from imblearn.over_sampling import SMOTE
import json, time

from tinymednet import TinyMedNet, SimpleMLP

torch.manual_seed(42)
np.random.seed(42)

X_COLS = ['age', 'bp', 'glucose', 'bmi', 'hemoglobin', 'creatinine',
          'bp_missing', 'hemoglobin_missing', 'creatinine_missing']

train_df = pd.read_csv('../data/processed/train.csv')
test_df = pd.read_csv('../data/processed/test.csv')

X_train_raw = train_df[X_COLS].values.astype(np.float32)
y_train = train_df['y'].values.astype(np.int64)
X_test_raw = test_df[X_COLS].values.astype(np.float32)
y_test = test_df['y'].values.astype(np.int64)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_raw)
X_test_s = scaler.transform(X_test_raw)

# SMOTE on training data only (matches original class-imbalance correction strategy)
sm = SMOTE(random_state=42, k_neighbors=5)
X_train_bal, y_train_bal = sm.fit_resample(X_train_s, y_train)
print("After SMOTE:", np.bincount(y_train_bal))

Xtr = torch.tensor(X_train_bal, dtype=torch.float32)
ytr = torch.tensor(y_train_bal, dtype=torch.long)
Xte = torch.tensor(X_test_s, dtype=torch.float32)
yte = torch.tensor(y_test, dtype=torch.long)

def train_model(model, epochs=12, lr=1e-3, distill_teacher=None, alpha=0.5, T=3.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    ce = nn.CrossEntropyLoss()
    n = Xtr.shape[0]
    batch_size = 128
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad()
            out = model(xb)
            loss = ce(out, yb)
            if distill_teacher is not None:
                with torch.no_grad():
                    t_out = distill_teacher(xb)
                soft_loss = F.kl_div(F.log_softmax(out/T, dim=1), F.softmax(t_out/T, dim=1),
                                      reduction='batchmean') * (T*T)
                loss = alpha*soft_loss + (1-alpha)*loss
            loss.backward()
            opt.step()
    return model

def evaluate(model, quantized_fn=None):
    model.eval()
    with torch.no_grad():
        logits = model(Xte)
        probs = F.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)
    f1 = f1_score(yte.numpy(), preds, average='macro')
    mcc = matthews_corrcoef(yte.numpy(), preds)
    try:
        auc = roc_auc_score(yte.numpy(), probs, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')
    return dict(macro_f1=f1, mcc=mcc, auc=auc)

results = {}

# ---- Full TinyMed-Net (teacher-distilled, SE + residual) ----
print("Training teacher (wider MLP) for distillation...")
teacher = SimpleMLP(len(X_COLS), 3)
train_model(teacher, epochs=10)
teacher.eval()

print("Training full TinyMed-Net...")
full_net = TinyMedNet(len(X_COLS), 3, use_se=True, use_residual=True)
train_model(full_net, epochs=12, distill_teacher=teacher)
results['TinyMedNet_full'] = evaluate(full_net)
results['TinyMedNet_full']['params'] = full_net.count_params()
print("Full TinyMed-Net:", results['TinyMedNet_full'])

# ---- Ablation: no SE ----
print("Training ablation: no SE...")
noSE = TinyMedNet(len(X_COLS), 3, use_se=False, use_residual=True)
train_model(noSE, epochs=12, distill_teacher=teacher)
results['ablation_no_SE'] = evaluate(noSE)
results['ablation_no_SE']['params'] = noSE.count_params()

# ---- Ablation: no residual ----
print("Training ablation: no residual...")
noRes = TinyMedNet(len(X_COLS), 3, use_se=True, use_residual=False)
train_model(noRes, epochs=12, distill_teacher=teacher)
results['ablation_no_residual'] = evaluate(noRes)
results['ablation_no_residual']['params'] = noRes.count_params()

# ---- Ablation: no distillation ----
print("Training ablation: no distillation...")
noKD = TinyMedNet(len(X_COLS), 3, use_se=True, use_residual=True)
train_model(noKD, epochs=12, distill_teacher=None)
results['ablation_no_distill'] = evaluate(noKD)
results['ablation_no_distill']['params'] = noKD.count_params()

# ---- Baseline MLP (3-layer 256/128/64) ----
print("Training baseline MLP...")
mlp = SimpleMLP(len(X_COLS), 3)
train_model(mlp, epochs=12)
results['MLP_baseline'] = evaluate(mlp)
results['MLP_baseline']['params'] = mlp.count_params()

with open('../results/nn_results.json', 'w') as f:
    json.dump(results, f, indent=2)

torch.save(full_net.state_dict(), '../results/full_net.pt')
torch.save(mlp.state_dict(), '../results/mlp.pt')

print(json.dumps(results, indent=2))
