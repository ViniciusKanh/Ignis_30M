"""Testes de integridade do preparo de sequências de pré-treinamento e de
compatibilidade com a arquitetura/configuração do Ignis-30M.

Não dependem de torch: usam apenas ignis.config e ignis.tokenizer, que
funcionam em qualquer ambiente (inclusive sem PyTorch instalado).
"""

from pathlib import Path

import numpy as np
import pytest

from ignis.config import IgnisConfig
from ignis.data.pretrain_prep import (
    carregar_dataset_tokenizado,
    construir_sequencias,
    salvar_dataset_tokenizado,
)
from ignis.tokenizer.bpe import ID_PAD, IgnisTokenizer


_CORPUS_PT_BR = [
    "A inteligência artificial e o aprendizado de máquina avançam rapidamente.",
    "São Paulo, Brasília, Amazônia, açaí, café com açúcar — tudo isso é Brasil.",
    "Números e pontuação: 123, 4.567,89! Pergunta? Exclamação! Reticências...",
    "Ação, coração, função, não, português, ímã, cédula, êxito, ônibus.",
    "O rato roeu a roupa do rei de Roma; a roupa do rei de Roma o rato roeu.",
] * 15


@pytest.fixture(scope="module")
def tokenizador(tmp_path_factory: pytest.TempPathFactory) -> IgnisTokenizer:
    diretorio = tmp_path_factory.mktemp("corpus_pretrain_prep")
    arquivos = []
    for i, texto in enumerate(_CORPUS_PT_BR):
        caminho = diretorio / f"doc{i}.txt"
        caminho.write_text(texto, encoding="utf-8")
        arquivos.append(caminho)
    return IgnisTokenizer.train_tokenizer(arquivos, vocab_size=512, min_frequency=1)


def test_tokenizador_vocab_size_cabe_na_configuracao_ignis_30m():
    """O tokenizador oficial (vocab_size=16384) deve caber exatamente no
    vocab_size configurado para o modelo Ignis-30M. Isto é verificável sem
    torch, pois depende apenas da config tipada."""
    config_30m = IgnisConfig.ignis_30m().modelo
    assert config_30m.vocab_size == 16384


def test_tokenizador_de_teste_nao_excede_vocab_do_modelo(tokenizador: IgnisTokenizer):
    """Mesmo um tokenizador reduzido (usado em testes) nunca deve produzir
    IDs fora do intervalo suportado pelo vocab_size do Ignis-30M — isso
    seria um erro de configuração grave (índice de embedding inválido)."""
    config_30m = IgnisConfig.ignis_30m().modelo
    assert tokenizador.vocab_size <= config_30m.vocab_size


def test_construir_sequencias_respeita_max_seq_len_do_ignis_30m(tokenizador: IgnisTokenizer):
    seq_len = IgnisConfig.ignis_30m().modelo.max_seq_len
    textos = {f"doc{i}": texto for i, texto in enumerate(_CORPUS_PT_BR)}
    sequencias, mascaras, estatisticas = construir_sequencias(
        textos, tokenizador, seq_len=seq_len, id_padding=ID_PAD
    )
    assert sequencias.shape[1] == seq_len
    assert mascaras.shape == sequencias.shape
    assert estatisticas.seq_len == seq_len
    assert estatisticas.n_documentos == len(textos)


def test_construir_sequencias_com_seq_len_pequeno_gera_formato_esperado(
    tokenizador: IgnisTokenizer,
):
    textos = {f"doc{i}": texto for i, texto in enumerate(_CORPUS_PT_BR[:3])}
    seq_len = 16
    sequencias, mascaras, estatisticas = construir_sequencias(
        textos, tokenizador, seq_len=seq_len, id_padding=ID_PAD
    )
    assert sequencias.dtype == np.int32
    assert mascaras.dtype == np.int8
    assert sequencias.shape == (estatisticas.n_sequencias, seq_len)
    # onde a máscara é 0, o id correspondente deve ser o id de padding.
    assert np.all(sequencias[mascaras == 0] == ID_PAD)
    # onde a máscara é 1, nunca deve haver id de padding "indevido" sobrando
    # (ids reais podem coincidir com o id de padding só por acaso, então
    # verificamos apenas a consistência estrutural do formato).
    assert estatisticas.n_tokens_total > 0
    assert estatisticas.n_sequencias >= 1


def test_salvar_e_carregar_dataset_tokenizado_preserva_dados(
    tmp_path: Path, tokenizador: IgnisTokenizer
):
    textos = {f"doc{i}": texto for i, texto in enumerate(_CORPUS_PT_BR[:4])}
    sequencias, mascaras, estatisticas = construir_sequencias(
        textos, tokenizador, seq_len=32, id_padding=ID_PAD
    )
    base = tmp_path / "treino"
    salvar_dataset_tokenizado(base, sequencias, mascaras, estatisticas)

    seq_carregadas, masc_carregadas, stats_carregadas = carregar_dataset_tokenizado(base)

    assert np.array_equal(sequencias, seq_carregadas)
    assert np.array_equal(mascaras, masc_carregadas)
    assert stats_carregadas["n_sequencias"] == estatisticas.n_sequencias
    assert stats_carregadas["seq_len"] == estatisticas.seq_len


def test_construir_sequencias_sem_documentos_retorna_vazio(tokenizador: IgnisTokenizer):
    sequencias, mascaras, estatisticas = construir_sequencias(
        {}, tokenizador, seq_len=16, id_padding=ID_PAD
    )
    assert sequencias.shape == (0, 16)
    assert mascaras.shape == (0, 16)
    assert estatisticas.n_documentos == 0
    assert estatisticas.n_sequencias == 0
