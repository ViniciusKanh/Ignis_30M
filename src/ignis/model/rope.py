"""Rotary Positional Embeddings (RoPE).

As posições usadas no treinamento (forward completo da sequência) e na
geração incremental devem ser consistentes: `offset` permite deslocar a
origem das posições quando tokens anteriores já foram processados.
"""

from __future__ import annotations

import torch


def _frequencias_inversas(head_dim: int, theta: float, device: torch.device) -> torch.Tensor:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim deve ser par para RoPE, recebido {head_dim}.")
    expoente = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    return 1.0 / (theta ** expoente)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retorna (cos, sin), cada um com shape (seq_len, head_dim)."""
    inv_freq = _frequencias_inversas(head_dim, theta, device)
    posicoes = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(posicoes, inv_freq)  # (seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotacionar_metade(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Aplica RoPE a um tensor (batch, n_heads, seq_len, head_dim)."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotacionar_metade(x) * sin
