import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from tinymednet import TinyMedNet, SimpleMLP

torch.manual_seed(42); np.random.seed(42)

X_COLS = ['age', 'bp', 'glucose', 'bmi', 'hemoglobin', 'creatinine',
          'bp_missing', 'hemoglobin_missing', 'creatinine_missing']
train_df = pd.read_csv('../data/processed/train.csv')
test_df = pd.read_csv('../data/processed/test.csv')
Xtr_raw = train_df[X_COLS].values.astype(np.float32); ytr = train_df['y'].values.astype(np.int64)
Xte_raw = test_df[X_COLS].values.astype(np.float32); yte = test_df['y'].values.astype(np.int64)
scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr_raw); Xte = scaler.transform(Xte_raw)
sm = SMOTE(random_state=42, k_neighbors=5)
Xtr_b, ytr_b = sm.fit_resample(Xtr, ytr)
Xtr_t = torch.tensor(Xtr_b, dtype=torch.float32); ytr_t = torch.tensor(ytr_b, dtype=torch.long)
Xte_t = torch.tensor(Xte, dtype=torch.float32); yte_t = torch.tensor(yte, dtype=torch.long)

def real_byte_size(model, bytes_per_param):
    return sum(p.numel() for p in model.parameters()) * bytes_per_param

def fake_quantize(x, n_bits=8):
    qmin, qmax = -(2**(n_bits-1)), 2**(n_bits-1)-1
    scale = (x.max() - x.min()) / (qmax - qmin + 1e-8)
    scale = torch.clamp(scale, min=1e-8)
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    return q * scale

class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return fake_quantize(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # straight-through estimator

def qat_forward_hook(module, inp, out):
    return FakeQuantSTE.apply(out)

def train_qat(model, epochs=12, lr=1e-3):
    """Genuine QAT: insert fake-quant STE on every Conv1d/Linear output during training."""
    hooks = []
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            hooks.append(m.register_forward_hook(qat_forward_hook))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    ce = nn.CrossEntropyLoss()
    n = Xtr_t.shape[0]; bs = 128
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            out = model(Xtr_t[idx])
            loss = ce(out, ytr_t[idx])
            loss.backward()
            opt.step()
    for h in hooks: h.remove()
    return model

def ptq_quantize_weights(model):
    """Simple uniform post-training quantization of stored weights (simulated INT8 rounding)."""
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(fake_quantize(p.data))
    return model

def evaluate(model):
    model.eval()
    with torch.no_grad():
        logits = model(Xte_t)
        preds = logits.argmax(dim=1).numpy()
    return f1_score(yte_t.numpy(), preds, average='macro')

results = {}

# Train a fresh FP32 TinyMed-Net baseline for this comparison
fp32_model = TinyMedNet(len(X_COLS), 3, use_se=True, use_residual=True)
opt = torch.optim.Adam(fp32_model.parameters(), lr=1e-3, weight_decay=1e-5)
ce = nn.CrossEntropyLoss()
n = Xtr_t.shape[0]; bs = 128
for ep in range(12):
    perm = torch.randperm(n)
    for i in range(0, n, bs):
        idx = perm[i:i+bs]
        opt.zero_grad()
        out = fp32_model(Xtr_t[idx])
        loss = ce(out, ytr_t[idx])
        loss.backward()
        opt.step()
f1_fp32 = evaluate(fp32_model)
n_params = fp32_model.count_params()
results['FP32'] = dict(macro_f1=f1_fp32, size_KB=round(real_byte_size(fp32_model, 4)/1024, 2))
print("FP32:", results['FP32'])

# ---- Uniform PTQ: quantize the already-trained FP32 weights, no retraining ----
import copy
ptq_model = copy.deepcopy(fp32_model)
ptq_quantize_weights(ptq_model)
f1_ptq = evaluate(ptq_model)
results['Uniform_INT8_PTQ'] = dict(macro_f1=f1_ptq, size_KB=round(real_byte_size(ptq_model, 1)/1024, 2))
print("Uniform INT8 PTQ:", results['Uniform_INT8_PTQ'])

# ---- AQAT: quantization-aware training from scratch with fake-quant in the loop ----
qat_model = TinyMedNet(len(X_COLS), 3, use_se=True, use_residual=True)
train_qat(qat_model, epochs=12)
f1_qat = evaluate(qat_model)
results['AQAT_INT8'] = dict(macro_f1=f1_qat, size_KB=round(real_byte_size(qat_model, 1)/1024, 2))
print("AQAT INT8 (QAT):", results['AQAT_INT8'])

with open('../results/quant_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
