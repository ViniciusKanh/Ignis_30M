from pathlib import Path

import pytest

from ignis.tokenizer.bpe import ID_BOS, ID_EOS, ID_PAD, ID_UNK, IgnisTokenizer, TOKENS_ESPECIAIS


_CORPUS_PT_BR = [
    "A inteligência artificial e o aprendizado de máquina avançam rapidamente.",
    "São Paulo, Brasília, Amazônia, açaí, café com açúcar — tudo isso é Brasil.",
    "Números e pontuação: 123, 4.567,89! Pergunta? Exclamação! Reticências...",
    "Ação, coração, função, não, português, ímã, cédula, êxito, ônibus.",
    "O rato roeu a roupa do rei de Roma; a roupa do rei de Roma o rato roeu.",
] * 15


@pytest.fixture(scope="module")
def tokenizador_treinado(tmp_path_factory: pytest.TempPathFactory) -> IgnisTokenizer:
    diretorio = tmp_path_factory.mktemp("corpus_tokenizer")
    arquivos = []
    for i, texto in enumerate(_CORPUS_PT_BR):
        caminho = diretorio / f"doc{i}.txt"
        caminho.write_text(texto, encoding="utf-8")
        arquivos.append(caminho)
    # vocab_size reduzido de propósito para os testes — nunca é "completado
    # artificialmente"; o tokenizador oficial usa vocab_size=16384 com corpus real.
    return IgnisTokenizer.train_tokenizer(arquivos, vocab_size=512, min_frequency=1)


def test_tokens_especiais_tem_ids_fixos(tokenizador_treinado: IgnisTokenizer):
    assert (ID_PAD, ID_BOS, ID_EOS, ID_UNK) == (0, 1, 2, 3)
    assert TOKENS_ESPECIAIS == ["<pad>", "<bos>", "<eos>", "<unk>"]


def test_encode_adiciona_bos_e_eos(tokenizador_treinado: IgnisTokenizer):
    ids = tokenizador_treinado.encode("um texto qualquer")
    assert ids[0] == ID_BOS
    assert ids[-1] == ID_EOS


@pytest.mark.parametrize(
    "texto",
    [
        "Ação, coração, função, não, português, ímã.",
        "São Paulo, Brasília, Amazônia, açaí, café.",
        "Pontuação: vírgula, ponto; dois-pontos: interrogação? exclamação!",
        "Números: 123, 4.567,89, -42, 3.14.",
        "Texto simples sem acentos nem numeros.",
        "Emoji e símbolos fora do treino: 🚀 ✨ café™.",
    ],
)
def test_round_trip_preserva_texto(tokenizador_treinado: IgnisTokenizer, texto: str):
    ids = tokenizador_treinado.encode(texto)
    decodificado = tokenizador_treinado.decode(ids)
    assert decodificado.strip() == texto.strip()


def test_save_e_load_preservam_vocabulario_e_encoding(tokenizador_treinado: IgnisTokenizer, tmp_path: Path):
    caminho = tmp_path / "tokenizer.json"
    tokenizador_treinado.save(caminho)
    recarregado = IgnisTokenizer.load(caminho)

    assert recarregado.vocab_size == tokenizador_treinado.vocab_size
    texto = "Teste de persistência com acentuação: função, ação, português."
    assert recarregado.encode(texto) == tokenizador_treinado.encode(texto)


def test_vocab_size_nao_e_completado_artificialmente_com_corpus_pequeno(tmp_path: Path):
    arquivo = tmp_path / "pequeno.txt"
    arquivo.write_text("um texto bem pequeno e repetitivo pequeno pequeno", encoding="utf-8")
    tok = IgnisTokenizer.train_tokenizer([arquivo], vocab_size=16384, min_frequency=1)
    # com corpus minúsculo, o vocabulário real deve ficar MUITO abaixo do alvo
    assert tok.vocab_size < 16384


def test_treinar_sem_arquivos_levanta_erro():
    with pytest.raises(ValueError):
        IgnisTokenizer.train_tokenizer([], vocab_size=100)
