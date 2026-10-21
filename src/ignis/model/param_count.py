"""Derivação analítica da contagem de parâmetros treináveis do Ignis.

Serve tanto para validar `IgnisModel.contar_parametros()` em testes quanto
para estimar o tamanho de configurações alternativas sem instanciar o
modelo. Ver docs/architecture.md para a derivação por extenso.
"""

from __future__ import annotations

from ..config import IgnisModelConfig


def contagem_parametros_esperada(config: IgnisModelConfig) -> int:
    d = config.d_model
    h = config.head_dim
    nh = config.n_heads
    nkv = config.n_kv_heads
    inter = config.intermediate_size
    com_bias = config.bias

    wq = d * (nh * h) + (nh * h if com_bias else 0)
    wk = d * (nkv * h) + (nkv * h if com_bias else 0)
    wv = d * (nkv * h) + (nkv * h if com_bias else 0)
    wo = (nh * h) * d + (d if com_bias else 0)
    attn_por_camada = wq + wk + wv + wo

    gate = d * inter + (inter if com_bias else 0)
    up = d * inter + (inter if com_bias else 0)
    down = inter * d + (d if com_bias else 0)
    mlp_por_camada = gate + up + down

    norms_por_camada = 2 * d  # RMSNorm pré-atenção + pré-MLP (apenas peso, sem bias)

    por_camada = attn_por_camada + mlp_por_camada + norms_por_camada
    total = config.n_layers * por_camada
    total += d  # RMSNorm final

    embedding = config.vocab_size * d
    total += embedding if config.tie_word_embeddings else 2 * embedding

    return total
