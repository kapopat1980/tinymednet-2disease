import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):  # x: (B, C, L)
        s = x.mean(dim=2)               # squeeze -> (B, C)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))  # excite -> (B, C)
        return x * s.unsqueeze(2)


class DWSepBlock(nn.Module):
    """Depthwise-separable 1D conv block with optional SE and residual."""
    def __init__(self, in_ch, out_ch, use_se=True, use_residual=True):
        super().__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.use_se = use_se
        self.se = SEBlock(out_ch) if use_se else None
        self.use_residual = use_residual and (in_ch == out_ch)

    def forward(self, x):
        identity = x
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = F.relu6(out)
        if self.se is not None:
            out = self.se(out)
        if self.use_residual:
            out = out + identity
        return out


class TinyMedNet(nn.Module):
    def __init__(self, n_features, n_classes=3, use_se=True, use_residual=True, width=(16, 32, 16)):
        super().__init__()
        self.n_features = n_features
        c0, c1, c2 = width
        self.stem = nn.Conv1d(1, c0, kernel_size=1)
        self.block1 = DWSepBlock(c0, c1, use_se=use_se, use_residual=False)
        self.block2 = DWSepBlock(c1, c1, use_se=use_se, use_residual=use_residual)
        self.block3 = DWSepBlock(c1, c2, use_se=use_se, use_residual=False)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c2, n_classes)

    def forward(self, x):  # x: (B, n_features)
        x = x.unsqueeze(1)          # (B, 1, F)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class SimpleMLP(nn.Module):
    """Baseline MLP: 3 layers, 256/128/64 units (matches manuscript's baseline spec)."""
    def __init__(self, n_features, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.net(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
