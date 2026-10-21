"""Testes do DataLoader de sequências tokenizadas (Fase 3).

Requer torch — não executável na VM Linux de desenvolvimento (ver
docs/experiments/fase1-fundacao.md). Rode no Windows nativo com torch
instalado: `pytest tests/test_training_dataloader.py -v`.
"""

import numpy as np
import pytest
import torch

from ignis.data.pretrain_prep import EstatisticasPreparo, salvar_dataset_tokenizado
from ignis.training.dataloader import (
    IGNORE_INDEX,
    SequenciasTokenizadasDataset,
    criar_dataloader,
)


def _salvar_dataset_sintetico(tmp_path, seq_len=8, n_sequencias=6, n_com_padding=1, id_padding=0):
    rng = np.random.default_rng(0)
    sequencias = rng.integers(low=1, high=50, size=(n_sequencias, seq_len)).astype(np.int32)
    mascaras = np.ones((n_sequencias, seq_len), dtype=np.int8)
    # última sequência tem padding no final, como produzido por construir_sequencias.
    if n_com_padding > 0:
        mascaras[-1, -3:] = 0
        sequencias[-1, -3:] = id_padding

    estatisticas = EstatisticasPreparo(
        n_documentos=n_sequencias,
        n_tokens_total=int(mascaras.sum()),
        n_sequencias=n_sequencias,
        n_sequencias_com_padding=n_com_padding,
        seq_len=seq_len,
    )
    base = tmp_path / "treino"
    salvar_dataset_tokenizado(base, sequencias, mascaras, estatisticas)
    return base, sequencias, mascaras


def test_deslocamento_correto_de_labels(tmp_path):
    base, sequencias, _ = _salvar_dataset_sintetico(tmp_path, seq_len=8, n_sequencias=4, n_com_padding=0)
    dataset = SequenciasTokenizadasDataset(base)

    for i in range(len(dataset)):
        input_ids, target_ids = dataset[i]
        assert input_ids.shape[0] == sequencias.shape[1] - 1
        assert target_ids.shape[0] == sequencias.shape[1] - 1
        # alvo[j] deve ser o token seguinte a entrada[j] no bloco original.
        assert torch.equal(input_ids, torch.from_numpy(sequencias[i, :-1]).long())
        assert torch.equal(target_ids, torch.from_numpy(sequencias[i, 1:]).long())


def test_mascaras_e_padding_viram_ignore_index(tmp_path):
    base, sequencias, mascaras = _salvar_dataset_sintetico(
        tmp_path, seq_len=8, n_sequencias=4, n_com_padding=1
    )
    dataset = SequenciasTokenizadasDataset(base)
    _, target_ids = dataset[len(dataset) - 1]  # última sequência tem padding

    mascara_alvo = mascaras[-1, 1:]
    posicoes_padding = (mascara_alvo == 0)
    assert posicoes_padding.any(), "fixture de teste deveria ter padding"
    assert (target_ids[posicoes_padding] == IGNORE_INDEX).all()
    assert (target_ids[~posicoes_padding] != IGNORE_INDEX).all()


def test_dataset_vazio_levanta_erro_claro(tmp_path):
    estatisticas = EstatisticasPreparo(
        n_documentos=0, n_tokens_total=0, n_sequencias=0, n_sequencias_com_padding=0, seq_len=8
    )
    base = tmp_path / "vazio"
    salvar_dataset_tokenizado(
        base, np.zeros((0, 8), dtype=np.int32), np.zeros((0, 8), dtype=np.int8), estatisticas
    )
    with pytest.raises(ValueError, match="vazio"):
        SequenciasTokenizadasDataset(base)


def test_max_seq_len_incompativel_levanta_erro(tmp_path):
    base, _, _ = _salvar_dataset_sintetico(tmp_path, seq_len=1200, n_sequencias=2, n_com_padding=0)
    with pytest.raises(ValueError, match="max_seq_len"):
        SequenciasTokenizadasDataset(base, max_seq_len=1024)


def test_criar_dataloader_embaralha_apenas_quando_pedido(tmp_path):
    base, _, _ = _salvar_dataset_sintetico(tmp_path, seq_len=8, n_sequencias=20, n_com_padding=0)

    loader_ordenado = criar_dataloader(base, batch_size=4, embaralhar=False)
    primeira_entrada_ordenada = next(iter(loader_ordenado))[0]

    loader_embaralhado_a = criar_dataloader(base, batch_size=4, embaralhar=True, seed=1, epoch=0)
    loader_embaralhado_b = criar_dataloader(base, batch_size=4, embaralhar=True, seed=1, epoch=0)
    primeira_entrada_a = next(iter(loader_embaralhado_a))[0]
    primeira_entrada_b = next(iter(loader_embaralhado_b))[0]

    # mesma seed + mesma epoca -> mesma permutação (reprodutibilidade para retomada).
    assert torch.equal(primeira_entrada_a, primeira_entrada_b)

    loader_embaralhado_outra_epoca = criar_dataloader(base, batch_size=4, embaralhar=True, seed=1, epoch=1)
    primeira_entrada_outra_epoca = next(iter(loader_embaralhado_outra_epoca))[0]
    # épocas diferentes -> permutações (quase certamente) diferentes.
    assert not torch.equal(primeira_entrada_a, primeira_entrada_outra_epoca)
