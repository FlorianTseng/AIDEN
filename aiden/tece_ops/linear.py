from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ScaledLinear(nn.Module):
    """Linear layer using the variance scaling used by TACE's ``torchLinear``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.scale = 1.0 / math.sqrt(max(in_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.scale, self.bias)


class ScaledSiLU(nn.Module):
    scale_factor = 1.6791767923989418

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x) * self.scale_factor


class ScaledSigmoid(nn.Module):
    scale_factor = 1.8467055342154763

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * self.scale_factor


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = [int(in_dim), *map(int, hidden_dims), int(out_dim)]
        layers: list[nn.Module] = []
        for idx, (c_in, c_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(ScaledLinear(c_in, c_out))
            if idx < len(dims) - 2:
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
