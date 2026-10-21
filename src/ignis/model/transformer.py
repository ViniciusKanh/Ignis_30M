"""Modelo Ignis completo: decoder-only Transformer com embeddings compartilhados."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import IgnisModelConfig
from .block import TransformerBlock
from .rmsnorm import RMSNorm


class IgnisModel(nn.Module):
    def __init__(self, config: IgnisModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm_final = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for camada in self.layers:
            x = camada(x)
        x = self.norm_final(x)
        return self.lm_head(x)

    def contar_parametros(self, apenas_treinaveis: bool = True) -> int:
        if apenas_treinaveis:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
