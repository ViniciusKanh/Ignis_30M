"""MLP com ativação SwiGLU."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """gate_proj/up_proj/down_proj, sem bias por padrão (conforme config)."""

    def __init__(self, d_model: int, intermediate_size: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
