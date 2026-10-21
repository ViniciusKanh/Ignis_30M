"""Métricas de eficiência da tokenização."""

from __future__ import annotations

from .bpe import IgnisTokenizer


def metricas_eficiencia(tokenizador: IgnisTokenizer, textos: list[str]) -> dict:
    total_caracteres = 0
    total_palavras = 0
    total_tokens = 0
    for texto in textos:
        ids = tokenizador.encode(texto, adicionar_bos_eos=False)
        total_tokens += len(ids)
        total_caracteres += len(texto)
        total_palavras += len(texto.split())

    return {
        "n_documentos": len(textos),
        "total_tokens": total_tokens,
        "total_caracteres": total_caracteres,
        "total_palavras": total_palavras,
        "caracteres_por_token": (total_caracteres / total_tokens) if total_tokens else 0.0,
        "tokens_por_palavra": (total_tokens / total_palavras) if total_palavras else 0.0,
    }
