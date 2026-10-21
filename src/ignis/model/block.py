"""Bloco Transformer: pré-norma + atenção causal GQA + pré-norma + SwiGLU."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import IgnisModelConfig
from .attention import CausalSelfAttentionGQA
from .mlp import SwiGLU
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: IgnisModelConfig) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = CausalSelfAttentionGQA(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            bias=config.bias,
            dropout=config.dropout,
        )
        self.norm_mlp = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = SwiGLU(config.d_model, config.intermediate_size, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x
