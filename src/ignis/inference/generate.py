"""Geração autoregressiva básica (sem cache de K/V — recálculo completo a cada passo).

Suporta decodificação greedy (temperature<=0), temperatura, top-k e top-p.
Cache incremental é uma otimização futura, fora do escopo da Fase 1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)

    gerado = input_ids
    greedy = temperature <= 0.0

    for _ in range(max_new_tokens):
        logits = model(gerado)
        proximos_logits = logits[:, -1, :]

        if greedy:
            proximo_token = torch.argmax(proximos_logits, dim=-1, keepdim=True)
        else:
            proximos_logits = proximos_logits / temperature

            if top_k is not None:
                valores, _ = torch.topk(proximos_logits, min(top_k, proximos_logits.size(-1)))
                limiar = valores[:, [-1]]
                proximos_logits = proximos_logits.masked_fill(
                    proximos_logits < limiar, float("-inf")
                )

            if top_p is not None:
                ordenados, indices = torch.sort(proximos_logits, descending=True)
                probs_ordenadas = F.softmax(ordenados, dim=-1)
                cumulativo = torch.cumsum(probs_ordenadas, dim=-1)
                remover = (cumulativo - probs_ordenadas) > top_p
                ordenados = ordenados.masked_fill(remover, float("-inf"))
                proximos_logits = torch.full_like(proximos_logits, float("-inf")).scatter(
                    1, indices, ordenados
                )

            probs = F.softmax(proximos_logits, dim=-1)
            proximo_token = torch.multinomial(probs, num_samples=1)

        gerado = torch.cat([gerado, proximo_token], dim=1)

        if eos_token_id is not None and bool((proximo_token == eos_token_id).all()):
            break

    return gerado
