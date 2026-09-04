#!/usr/bin/env python3
"""
verify_findings.py -- Reproduces the defects in the superseded version of this
work, using only the raw data in this repository.

The earlier version of this project is preserved in git history under the tag
`v1-superseded`. This script does not need that code: it rebuilds the withdrawn
pooled task from data/raw and re-implements the quantizer that produced the
earlier result, so each defect can be demonstrated from a clean checkout.

    python verify_findings.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from controls_and_figures import build_pooled


def rule(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def verdict(ok, msg):
    print(f"  [{'REPRODUCED' if ok else 'NOT REPRODUCED'}] {msg}")


# ------------------------------------------------------------------ finding 1
rule("FINDING 1 -- The withdrawn pooled task is solvable from data provenance")

d = build_pooled()
y = d["y"].to_numpy()
MISS = [c for c in d.columns if c.endswith("_missing")]
CLIN = ["age", "bp", "glucose", "bmi", "hemoglobin", "creatinine"]

print("  Class by source under the withdrawn three-class label map:")
ct = d.assign(cls=np.array(["Healthy", "Diabetes", "CKD"])[y]).groupby(
    ["source", "cls"], observed=True).size().unstack(fill_value=0)
print("  " + ct.to_string().replace("\n", "\n  "))

ind = ((d.hemoglobin_missing == 0) | (d.creatinine_missing == 0)).to_numpy()
ckd = y == 2
sens = (ind & ckd).sum() / ckd.sum()
spec = (~ind & ~ckd).sum() / (~ckd).sum()
print(f"\n  Rule: predict CKD iff haemoglobin or creatinine was recorded")
print(f"    sensitivity = {sens:.3f}   specificity = {spec:.3f}")

scores = {}
for name, cols in [("missingness flags only", MISS),
                   ("clinical values only", CLIN)]:
    recs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(d, y):
        X = d[cols].to_numpy(float)
        imp = SimpleImputer(strategy="median").fit(X[tr])
        m = DecisionTreeClassifier(max_depth=4, random_state=0).fit(
            imp.transform(X[tr]), y[tr])
        p = m.predict(imp.transform(X[te]))
        recs.append(recall_score(y[te], p, labels=[2], average="macro",
                                 zero_division=0))
    scores[name] = float(np.mean(recs))
    print(f"    CKD recall using {name:<24}: {scores[name]:.3f}")

verdict(sens > 0.95 and spec > 0.98 and
        scores["missingness flags only"] > scores["clinical values only"],
        "A single missingness bit separates CKD better than all six clinical "
        "measurements combined. The pooled task rewarded recognising the source "
        "file, not the disease.")

# ------------------------------------------------------------------ finding 2
rule("FINDING 2 -- The quantizer behind the earlier PTQ collapse")

def legacy_fake_quantize(x, n_bits=8):
    """scale = (max - min)/255, rounded into [-128, 127], with NO zero point."""
    qmin, qmax = -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1
    scale = (x.max() - x.min()) / (qmax - qmin + 1e-8)
    scale = torch.clamp(scale, min=1e-8)
    return torch.clamp(torch.round(x / scale), qmin, qmax) * scale

g = torch.empty(64).normal_(1.0, 0.1, generator=torch.Generator().manual_seed(0))
gq = legacy_fake_quantize(g)
print("  A BatchNorm scale vector is strictly positive and clusters near 1.0.")
print("  A symmetric quantizer with no zero point maps that range onto its")
print("  positive half only, so every element saturates at the same value:\n")
print(f"    before: mean {g.mean():.4f}  range [{g.min():.3f}, {g.max():.3f}]  "
      f"{g.unique().numel()} distinct values")
print(f"    after : mean {gq.mean():.4f}  range [{gq.min():.3f}, {gq.max():.3f}]  "
      f"{gq.unique().numel()} distinct value(s)")
print(f"    retained {100 * gq.mean() / g.mean():.1f}% of true scale")

sym = g / (g.abs().max() / 127)
sym = torch.clamp(torch.round(sym), -128, 127) * (g.abs().max() / 127)
print(f"\n  For comparison, a correct symmetric quantizer on the same tensor:")
print(f"    after : mean {sym.mean():.4f}  {sym.unique().numel()} distinct values")

verdict(gq.unique().numel() == 1 and sym.unique().numel() > 10,
        "The collapse is a property of this quantizer, not of post-training "
        "quantization. Folding BatchNorm into the preceding convolution, as "
        "standard practice prescribes, removes the tensor entirely.")

rule("SUMMARY")
print("""  Both defects reproduce from raw data alone.

  Claims withdrawn in the current version as a result:
    - the pooled three-class Healthy/Diabetes/CKD formulation
    - "QAT is necessary because PTQ collapses"

  See results/controls.json for the full control baselines and
  results/quantization.json for the 20-seed PTQ/QAT comparison.
""")
