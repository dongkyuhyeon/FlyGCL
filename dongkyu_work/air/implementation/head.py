"""AIR GeneralizedARM adapted from ZHUANGHP/Analytic-continual-learning (MIT).

See ../original/SOURCE_MANIFEST.json and ../original/air/LICENSE.
Keep full per-class covariance statistics; no diagonal approximation.
"""
import math
import torch
from torch import nn


class AIRHead(nn.Module):
    def __init__(self, in_features, num_classes, gamma=0.1, device='cpu', max_stats_gib=8.0):
        super().__init__()
        if min(in_features, num_classes) <= 0 or not math.isfinite(gamma) or gamma <= 0:
            raise ValueError('AIR dimensions and gamma must be positive and finite')
        if not math.isfinite(max_stats_gib) or max_stats_gib <= 0:
            raise ValueError('air_max_stats_gib must be positive and finite')
        gib = num_classes * in_features * in_features * 8 / 2**30
        if gib > max_stats_gib:
            raise ValueError(f'AIR class statistics require {gib:.2f} GiB, above air_max_stats_gib={max_stats_gib}; reduce air_expansion_size or explicitly raise the limit')
        self.gamma = gamma
        kw = dict(dtype=torch.float64, device=device)
        self.register_buffer('A', torch.zeros(num_classes, in_features, in_features, **kw))
        self.register_buffer('C', torch.zeros(in_features, num_classes, **kw))
        self.register_buffer('weight', torch.zeros(in_features, num_classes, **kw))
        self.register_buffer('cnt', torch.zeros(num_classes, dtype=torch.long, device=device))
        self.register_buffer('num_seen', torch.zeros((), dtype=torch.long, device=device))
        self.register_buffer('num_updates', torch.zeros((), dtype=torch.long, device=device))
        self.register_buffer('dirty', torch.ones((), dtype=torch.bool, device=device))

    @torch.no_grad()
    def fit(self, x, y):
        x, y = x.to(self.weight), y.to(device=self.weight.device, dtype=torch.long)
        self.num_seen.copy_(torch.maximum(self.num_seen, y.max() + 1))
        for c in y.unique():
            features = x[y == c]
            self.A[c].add_(features.T @ features)
            self.C[:, c].add_(features.sum(0))
            self.cnt[c].add_(len(features))
        self.num_updates.add_(len(x))
        self.dirty.fill_(True)

    @torch.no_grad()
    def update(self):
        n = int(self.num_seen)
        if not self.dirty or n == 0:
            return
        counts = self.cnt[:n].to(self.weight)
        factors = torch.where(counts > 0, counts.reciprocal(), 0)
        factors.mul_(n / factors.sum())
        a = torch.eye(self.weight.shape[0], dtype=self.weight.dtype, device=self.weight.device) * self.gamma
        # Avoid a second C x D x D tensor while preserving the upstream formula.
        for c in range(n):
            a.add_(self.A[c] * factors[c])
        self.weight[:, :n].copy_(torch.linalg.solve(a, self.C[:, :n] * factors))
        if not torch.isfinite(self.weight).all():
            raise FloatingPointError('Nonfinite AIR weights; check gamma and input features')
        self.dirty.fill_(False)

    def forward(self, x):
        self.update()
        return x.to(self.weight) @ self.weight
