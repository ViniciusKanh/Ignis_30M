"""Conversão de documentos tokenizados em sequências para predição do
próximo token, respeitando o contexto máximo do Ignis-30M (1.024 tokens por
padrão) e as fronteiras entre documentos (marcadas por <bos>/<eos>).

Deve ser chamada uma vez por split (treino, validação, teste) — nunca
misturando documentos de splits diferentes na mesma chamada, o que evita
vazamento entre splits por construção.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..tokenizer.bpe import IgnisTokenizer


@dataclass
class EstatisticasPreparo:
    n_documentos: int
    n_tokens_total: int
    n_sequencias: int
    n_sequencias_com_padding: int
    seq_len: int


def construir_sequencias(
    textos: dict[str, str],
    tokenizador: IgnisTokenizer,
    seq_len: int = 1024,
    id_padding: int = 0,
) -> tuple[np.ndarray, np.ndarray, EstatisticasPreparo]:
    """Concatena os tokens (com <bos>/<eos> entre documentos) e corta em
    blocos de `seq_len`. Retorna (sequencias, mascaras, estatisticas); a
    máscara vale 1 para token real e 0 para padding. O último bloco de uma
    chamada pode receber padding."""
    todos_ids: list[int] = []
    n_tokens_total = 0
    for doc_id in sorted(textos.keys()):
        ids = tokenizador.encode(textos[doc_id], adicionar_bos_eos=True)
        todos_ids.extend(ids)
        n_tokens_total += len(ids)

    sequencias: list[list[int]] = []
    mascaras: list[list[int]] = []
    n_com_padding = 0

    for inicio in range(0, max(len(todos_ids), 1), seq_len):
        bloco = todos_ids[inicio : inicio + seq_len]
        if not bloco:
            break
        mascara = [1] * len(bloco)
        if len(bloco) < seq_len:
            n_com_padding += 1
            faltam = seq_len - len(bloco)
            bloco = bloco + [id_padding] * faltam
            mascara = mascara + [0] * faltam
        sequencias.append(bloco)
        mascaras.append(mascara)

    array_sequencias = np.array(sequencias, dtype=np.int32) if sequencias else np.zeros((0, seq_len), dtype=np.int32)
    array_mascaras = np.array(mascaras, dtype=np.int8) if mascaras else np.zeros((0, seq_len), dtype=np.int8)

    estatisticas = EstatisticasPreparo(
        n_documentos=len(textos),
        n_tokens_total=n_tokens_total,
        n_sequencias=len(sequencias),
        n_sequencias_com_padding=n_com_padding,
        seq_len=seq_len,
    )
    return array_sequencias, array_mascaras, estatisticas


def salvar_dataset_tokenizado(
    caminho_base: str | Path,
    sequencias: np.ndarray,
    mascaras: np.ndarray,
    estatisticas: EstatisticasPreparo,
) -> None:
    caminho_base = Path(caminho_base)
    caminho_base.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{caminho_base}.sequencias.npy", sequencias)
    np.save(f"{caminho_base}.mascaras.npy", mascaras)
    with open(f"{caminho_base}.estatisticas.json", "w", encoding="utf-8") as f:
        json.dump(asdict(estatisticas), f, ensure_ascii=False, indent=2)


def carregar_dataset_tokenizado(caminho_base: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    caminho_base = Path(caminho_base)
    sequencias = np.load(f"{caminho_base}.sequencias.npy")
    mascaras = np.load(f"{caminho_base}.mascaras.npy")
    with open(f"{caminho_base}.estatisticas.json", encoding="utf-8") as f:
        estatisticas = json.load(f)
    return sequencias, mascaras, estatisticas
