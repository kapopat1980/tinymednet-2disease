"""
quantization.py -- PTQ vs QAT under a validated toolchain, plus a controlled
reproduction of the defect in the original implementation.

Four conditions per seed, all evaluated on the SAME frozen test split:

  FP32            trained baseline
  PTQ_legacy      the original repo's quantizer, reproduced exactly:
                  scale = (max-min)/255, round into [-128,127], NO zero point,
                  applied to every parameter tensor including BatchNorm, with
                  no folding and no activation quantization.
  PTQ_validated   torch.ao static PTQ: Conv+BN folded, per-channel symmetric
                  weights, calibrated affine activations. Real INT8 inference.
  QAT_validated   torch.ao QAT, same observers, converted and evaluated as a
                  real INT8 model.

The legacy condition is included because the original manuscript's central
technical claim -- that QAT is necessary because PTQ collapses -- rests on it.
Separating "PTQ collapses" from "this quantizer collapses" requires running both.

Note on why the legacy quantizer destroys the network: BatchNorm gamma is
strictly positive and clusters near 1.0. A symmetric quantizer with no zero
point maps that range onto its positive half only, so scale is roughly twice
what it should be and every gamma saturates at the same clamped value.
"""
import copy
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.ao.quantization as tq
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import get_task
from experiment import fit_torch, metrics, predict_torch
from models import TinyMedNet, count_params

ROOT = Path(__file__).resolve().parents[1]
N_SEEDS = 5


# ------------------------------------------------------- legacy (buggy) quantizer
def legacy_fake_quantize(x, n_bits=8):
    qmin, qmax = -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1
    scale = (x.max() - x.min()) / (qmax - qmin + 1e-8)
    scale = torch.clamp(scale, min=1e-8)
    return torch.clamp(torch.round(x / scale), qmin, qmax) * scale


def apply_legacy_ptq(model):
    m = copy.deepcopy(model)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(legacy_fake_quantize(p.data))
    return m


# ------------------------------------------------------- validated quantization
def serialized_bytes(m):
    b = io.BytesIO()
    torch.save(m.state_dict(), b)
    return b.getbuffer().nbytes


def quantize_ptq(fp32, Xcal, backend="fbgemm"):
    m = copy.deepcopy(fp32).eval()
    m.fuse_model()
    m.qconfig = tq.get_default_qconfig(backend)
    tq.prepare(m, inplace=True)
    with torch.no_grad():
        X = torch.tensor(Xcal, dtype=torch.float32)
        for i in range(0, len(X), 64):
            m(X[i:i + 64])
    tq.convert(m, inplace=True)
    return m


def quantize_qat(d, Xtr, ytr, Xval_seed, backend="fbgemm", epochs=60, seed=0, teacher=None):
    m = TinyMedNet(d).train()
    m.fuse_model()
    m.qconfig = tq.get_default_qat_qconfig(backend)
    tq.prepare_qat(m, inplace=True)
    fit_torch(m, Xtr, ytr, epochs=epochs, seed=seed, teacher=teacher)
    m.eval()
    tq.convert(m, inplace=True)
    return m


@torch.no_grad()
def predict_quantized(m, X):
    return torch.softmax(m(torch.tensor(X, dtype=torch.float32)), 1).numpy()


def op_inventory(m):
    """Which modules ended up as integer kernels, and which stayed float."""
    quant, flt = [], []
    for name, mod in m.named_modules():
        t = type(mod).__name__
        if not list(mod.children()):
            (quant if "quantized" in type(mod).__module__ else flt).append(t)
    return {"int8_modules": sorted(set(quant)), "float_modules": sorted(set(flt))}


QPART = ROOT / "artifacts" / "quant_parts"
QPART.mkdir(parents=True, exist_ok=True)


def available_backends():
    """
    Quantization engines vary by platform: Linux x86 ships fbgemm (per-channel)
    and often qnnpack (per-tensor, ARM-representative); Windows builds commonly
    ship onednn only. Hard-coding fbgemm silently produces no validated rows on
    Windows, so the engines actually present are used instead.
    """
    have = list(torch.backends.quantized.supported_engines)
    order = [b for b in ("fbgemm", "qnnpack", "onednn") if b in have]
    return tuple(order) if order else ("none",)


def run_task(task_name, backends=None, seeds=None):
    if backends is None:
        backends = available_backends()
        print(f"  quantization backends on this machine: {', '.join(backends)}",
              flush=True)
    """Per-seed results are cached so the study is resumable across sessions."""
    t = get_task(task_name)
    X, y, d = t["X"], t["y"], t["X"].shape[1]
    Xtr_r, Xte_r, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    imp = SimpleImputer(strategy="median").fit(Xtr_r)
    sc = StandardScaler().fit(imp.transform(Xtr_r))
    Xtr = sc.transform(imp.transform(Xtr_r)).astype(np.float32)
    Xte = sc.transform(imp.transform(Xte_r)).astype(np.float32)

    rows, inv = [], None
    for s in (range(N_SEEDS) if seeds is None else seeds):
        cache = QPART / f"{task_name}_s{s}.json"
        if cache.exists():
            rows.extend(json.loads(cache.read_text(encoding="utf-8")))
            continue
        srows = []
        torch.manual_seed(s)
        fp32 = fit_torch(TinyMedNet(d), Xtr, ytr, seed=s)
        r = metrics(yte, predict_torch(fp32, Xte)[:, 1])
        r.update(cond="FP32", seed=s, bytes=serialized_bytes(fp32),
                 param_bytes=count_params(fp32) * 4)
        srows.append(r)

        leg = apply_legacy_ptq(fp32)
        r = metrics(yte, predict_torch(leg, Xte)[:, 1])
        r.update(cond="PTQ_legacy", seed=s, bytes=count_params(leg) * 1,
                 param_bytes=count_params(leg) * 1)
        srows.append(r)

        for bk in backends:
            torch.backends.quantized.engine = bk
            try:
                q = quantize_ptq(fp32, Xtr, backend=bk)
                r = metrics(yte, predict_quantized(q, Xte)[:, 1])
                r.update(cond=f"PTQ_validated_{bk}", seed=s, bytes=serialized_bytes(q),
                         param_bytes=count_params(fp32) * 1)
                srows.append(r)
                if inv is None:
                    inv = op_inventory(q)
            except Exception as e:
                srows.append(dict(cond=f"PTQ_validated_{bk}", seed=s, error=str(e)[:200]))

            try:
                torch.manual_seed(s)
                qa = quantize_qat(d, Xtr, ytr, None, backend=bk, seed=s)
                r = metrics(yte, predict_quantized(qa, Xte)[:, 1])
                r.update(cond=f"QAT_validated_{bk}", seed=s, bytes=serialized_bytes(qa),
                         param_bytes=count_params(fp32) * 1)
                srows.append(r)
            except Exception as e:
                srows.append(dict(cond=f"QAT_validated_{bk}", seed=s, error=str(e)[:200]))
        cache.write_text(json.dumps(srows, indent=2), encoding="utf-8")
        rows.extend(srows)
        print(f"  seed {s} cached", flush=True)

    # Direct demonstration of the legacy quantizer's failure mode on BatchNorm scale.
    g = torch.empty(64).normal_(1.0, 0.1, generator=torch.Generator().manual_seed(0))
    gq = legacy_fake_quantize(g)
    demo = dict(before_mean=float(g.mean()), after_mean=float(gq.mean()),
                after_distinct_values=int(torch.unique(gq).numel()),
                retained_fraction=float(gq.mean() / g.mean()))
    return rows, inv, demo


if __name__ == "__main__":
    tn = sys.argv[1]
    seeds = [int(a) for a in sys.argv[2:]] or None
    print(f"[{tn}] quantization study", flush=True)
    rows, inv, demo = run_task(tn, seeds=seeds)
    outfile = ROOT / "results" / "quantization.json"
    out = json.loads(outfile.read_text(encoding="utf-8")) if outfile.exists() else {}
    blk = out.get(tn) or dict(rows=[], op_inventory=None, batchnorm_demo=None)
    # Append, keyed on (condition, seed), so the study can be built up in chunks
    # without a later run discarding an earlier one's seeds.
    have = {(r["cond"], r["seed"]) for r in blk["rows"]}
    blk["rows"] = blk["rows"] + [r for r in rows if (r["cond"], r["seed"]) not in have]
    blk["op_inventory"] = blk["op_inventory"] or inv
    blk["batchnorm_demo"] = blk["batchnorm_demo"] or demo
    out[tn] = blk
    outfile.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{tn}: {len({r['seed'] for r in blk['rows']})} seeds, {len(blk['rows'])} rows")
