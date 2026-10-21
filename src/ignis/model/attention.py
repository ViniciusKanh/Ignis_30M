"""Atenção causal com Grouped Query Attention (GQA) e RoPE.

Limitação conhecida (Fase 1): esta implementação sempre recalcula o forward
completo da sequência (sem cache de K/V), portanto `position_offset` é
mantido para uso futuro mas hoje é sempre 0 — suficiente para smoke tests e
geração autoregressiva simples. Cache incremental fica para otimização
posterior (fora do escopo da Fase 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rotary_pos_emb, build_rope_cache


class CausalSelfAttentionGQA(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rope_theta: float = 10000.0,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model deve ser divisível por n_heads.")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads deve ser divisível por n_kv_heads (GQA).")
        if head_dim * n_heads != d_model:
            raise ValueError("head_dim * n_heads deve ser igual a d_model.")

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.dropout_p = dropout
        self.grupo_gqa = n_heads // n_kv_heads

        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=bias)
        self.wk = nn.Linear(d_model, n_kv_heads * head_dim, bias=bias)
        self.wv = nn.Linear(d_model, n_kv_heads * head_dim, bias=bias)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = build_rope_cache(
            seq_len,
            self.head_dim,
            theta=self.rope_theta,
            device=x.device,
            dtype=x.dtype,
            offset=position_offset,
        )
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        if self.grupo_gqa > 1:
            k = k.repeat_interleave(self.grupo_gqa, dim=1)
            v = v.repeat_interleave(self.grupo_gqa, dim=1)

        dropout_p = self.dropout_p if self.training else 0.0
        # scaled_dot_product_attention com is_causal=True cuida do
        # mascaramento causal; fallback matemático em CPU é feito
        # internamente pelo PyTorch quando o backend otimizado não está
        # disponível para a combinação de dtypes/dispositivo em uso.
        saida = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=dropout_p
        )
        saida = saida.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.wo(saida)
