"""
models.py -- TinyMed-Net and baselines, built for real INT8 conversion.

Differences from the original implementation that matter for the reviewer comments:

  * Every op that quantized inference needs a handler for is expressed as a
    quantizable module: residual addition and squeeze-excitation gating use
    torch.ao.nn.quantized.FloatFunctional rather than bare '+' and '*', and the
    network is wrapped in QuantStub/DeQuantStub. The original could not be
    converted at all, which is why it was never evaluated in INT8.
  * Conv+BN pairs are declared so they can be folded before quantization.
    Folding is what makes PTQ work; the original quantized raw BatchNorm
    parameters, which is what destroyed it.
  * matched_mlp() sizes its hidden layers to match TinyMed-Net's parameter
    count, so the storage comparison is not just restating a parameter gap.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import DeQuantStub, QuantStub


class SEBlock(nn.Module):
    """Squeeze-and-excitation over channels. Gating uses FloatFunctional.mul."""

    def __init__(self, ch, r=4):
        super().__init__()
        hidden = max(1, ch // r)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(ch, hidden, 1)
        self.act = nn.ReLU()
        self.fc2 = nn.Conv1d(hidden, ch, 1)
        self.gate = nn.Sigmoid()
        self.mul = FloatFunctional()

    def forward(self, x):
        s = self.gate(self.fc2(self.act(self.fc1(self.pool(x)))))
        return self.mul.mul(x, s)


class DSBlock(nn.Module):
    """Depthwise-separable block: DWConv -> PWConv -> BN -> ReLU, optional SE and residual."""

    def __init__(self, cin, cout, use_se=True, residual=False, r=4):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, 3, padding=1, groups=cin, bias=False)
        self.dw_bn = nn.BatchNorm1d(cin)
        self.dw_act = nn.ReLU()
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)
        self.pw_bn = nn.BatchNorm1d(cout)
        self.pw_act = nn.ReLU()
        self.se = SEBlock(cout, r) if use_se else None
        self.residual = residual and (cin == cout)
        self.add = FloatFunctional() if self.residual else None

    def forward(self, x):
        idt = x
        h = self.dw_act(self.dw_bn(self.dw(x)))
        h = self.pw_act(self.pw_bn(self.pw(h)))
        if self.se is not None:
            h = self.se(h)
        if self.residual:
            h = self.add.add(h, idt)
        return h

    def fuse_list(self, prefix):
        return [[f"{prefix}.dw", f"{prefix}.dw_bn", f"{prefix}.dw_act"],
                [f"{prefix}.pw", f"{prefix}.pw_bn", f"{prefix}.pw_act"]]


class TinyMedNet(nn.Module):
    """
    Input is a length-d feature vector treated as a 1-channel sequence, so the
    parameter count does not depend on the number of input features.
    """

    def __init__(self, n_features, n_classes=2, use_se=True, use_residual=True, width=16):
        super().__init__()
        self.n_features = n_features
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.stem = nn.Conv1d(1, width, 1, bias=False)
        self.stem_bn = nn.BatchNorm1d(width)
        self.stem_act = nn.ReLU()
        self.block1 = DSBlock(width, width * 2, use_se, residual=False)
        self.block2 = DSBlock(width * 2, width * 2, use_se, residual=use_residual)
        self.block3 = DSBlock(width * 2, width, use_se, residual=False)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)                       # (B, 1, d)
        x = self.quant(x)
        h = self.stem_act(self.stem_bn(self.stem(x)))
        h = self.block3(self.block2(self.block1(h)))
        h = self.pool(h).flatten(1)
        h = self.head(h)
        return self.dequant(h)

    def fuse_model(self):
        """Fold Conv+BN(+ReLU). Must run before PTQ or QAT conversion."""
        from torch.ao.quantization import fuse_modules, fuse_modules_qat
        fuse = fuse_modules_qat if self.training else fuse_modules
        groups = [["stem", "stem_bn", "stem_act"]]
        for name in ["block1", "block2", "block3"]:
            groups += getattr(self, name).fuse_list(name)
        fuse(self, groups, inplace=True)
        return self


class MLP(nn.Module):
    def __init__(self, n_features, hidden, n_classes=2):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def matched_mlp(n_features, target, n_classes=2, ratio=0.55):
    """Two-hidden-layer MLP whose parameter count is as close as possible to `target`."""
    best = None
    for h1 in range(4, 512):
        h2 = max(2, int(round(h1 * ratio)))
        n = (n_features * h1 + h1) + (h1 * h2 + h2) + (h2 * n_classes + n_classes)
        d = abs(n - target)
        if best is None or d < best[0]:
            best = (d, h1, h2, n)
    _, h1, h2, n = best
    return MLP(n_features, [h1, h2], n_classes), (h1, h2), n


def wide_mlp(n_features, n_classes=2):
    """The 256/128/64 MLP from the original manuscript, retained as a conventional baseline."""
    return MLP(n_features, [256, 128, 64], n_classes)


if __name__ == "__main__":
    for d in [8, 24, 7]:
        m = TinyMedNet(d)
        p = count_params(m)
        mm, hid, pm = matched_mlp(d, p)
        wm = wide_mlp(d)
        x = torch.randn(4, d)
        assert m(x).shape == (4, 2) and mm(x).shape == (4, 2)
        print(f"d={d:2d}  TinyMedNet={p:5d}  matchedMLP={pm:5d} {hid}  wideMLP={count_params(wm):6d}")
