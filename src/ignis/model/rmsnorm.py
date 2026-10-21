"""RMSNorm (Root Mean Square Layer Normalization)."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Normalização RMS, sem termo de centralização (média) nem bias."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype_original = x.dtype
        x_fp32 = x.float()
        variancia = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normalizado = x_fp32 * torch.rsqrt(variancia + self.eps)
        return x_normalizado.to(dtype_original) * self.weight
