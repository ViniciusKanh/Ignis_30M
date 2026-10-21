"""Detecção de duplicatas exatas e quase-duplicatas (heurística conservadora).

A detecção de quase-duplicatas usa shingles de palavras + similaridade de
Jaccard, O(n²) — adequada a corpora pequenos/piloto (até alguns milhares de
documentos). Para escala maior, substituir por MinHash/LSH é um trabalho
futuro, registrado em docs/limitations.md.
"""

from __future__ import annotations

import hashlib
import re

_RE_TOKEN = re.compile(r"\w+", re.UNICODE)


def sha256_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _shingles(texto: str, tamanho: int = 5) -> set[str]:
    tokens = _RE_TOKEN.findall(texto.lower())
    if len(tokens) < tamanho:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + tamanho]) for i in range(len(tokens) - tamanho + 1)}


def similaridade_jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def agrupar_quase_duplicatas(
    textos: dict[str, str], limiar: float = 0.8, tamanho_shingle: int = 5
) -> dict[str, str]:
    """Retorna um mapa documento_id -> id_do_grupo (union-find). Documentos
    sem nenhuma quase-duplicata formam grupo próprio (grupo = seu próprio id)."""
    ids = list(textos.keys())
    conjuntos = {doc_id: _shingles(texto, tamanho_shingle) for doc_id, texto in textos.items()}
    grupo_de: dict[str, str] = {doc_id: doc_id for doc_id in ids}

    def raiz(doc_id: str) -> str:
        while grupo_de[doc_id] != doc_id:
            doc_id = grupo_de[doc_id]
        return doc_id

    def unir(a: str, b: str) -> None:
        ra, rb = raiz(a), raiz(b)
        if ra != rb:
            grupo_de[ra] = rb

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if similaridade_jaccard(conjuntos[ids[i]], conjuntos[ids[j]]) >= limiar:
                unir(ids[i], ids[j])

    return {doc_id: raiz(doc_id) for doc_id in ids}
