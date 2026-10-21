"""Testes de FiltrosQualidade/avaliar_qualidade, incluindo a regressao real
da Sprint 6.2: um artigo de algebra linear real foi rejeitado por "baixa
razao de caracteres alfabeticos" calculada sobre o tamanho TOTAL do texto,
quando na verdade a maior parte dos caracteres nao-alfabeticos eram espacos
de formatacao da renderizacao de formulas (nao sinal de baixa qualidade de
prosa). A razao agora e calculada sobre caracteres NAO-ESPACO (ver
docs/experiments/fase3-sprint6.2-qualidade-corpus.md)."""

from ignis.data.filtros import FiltrosQualidade, avaliar_qualidade


def _texto_base(sufixo: str = "") -> str:
    return ("Este e um texto de prosa normal em portugues, com bastante conteudo alfabetico real. " * 5) + sufixo


def test_texto_normal_e_aceito():
    ok, motivo = avaliar_qualidade(_texto_base(), FiltrosQualidade())
    assert ok, motivo


def test_texto_muito_curto_e_rejeitado():
    ok, motivo = avaliar_qualidade("curto", FiltrosQualidade())
    assert not ok
    assert "curto" in motivo


def test_texto_muito_longo_e_rejeitado():
    ok, motivo = avaliar_qualidade("a" * 10, FiltrosQualidade(min_caracteres=0, max_caracteres=5))
    assert not ok
    assert "longo" in motivo


def test_sopa_de_numeros_com_espacamento_normal_continua_rejeitada():
    # o filtro precisa continuar pegando texto genuinamente degenerado --
    # a mudanca para "nao-espaco" nao deve abrir uma brecha aqui, porque o
    # espacamento entre os numeros e normal (nao anormalmente denso como no
    # artefato de formula matematica).
    texto = " ".join(str(n) for n in range(2000))
    ok, motivo = avaliar_qualidade(texto, FiltrosQualidade())
    assert not ok
    assert "alfab" in motivo


def test_formula_matematica_espacada_e_aceita_apos_correcao():
    """Regressao Sprint 6.2: um bloco de formula renderizada como texto
    simples (um simbolo por linha, muito espaco em branco) nao deve, por
    si so, derrubar um artigo de prosa legitima abaixo do limiar. Exemplo
    sintetico, no mesmo estilo do artefato real de extracao da API da
    MediaWiki (nao copiado de nenhum artigo)."""
    prosa = (
        "Em matematica, mais especificamente em algebra linear e analise funcional, "
        "o nucleo ou espaco nulo de uma transformacao linear L entre dois espacos "
        "vetoriais e o conjunto de todos os elementos cuja imagem e o vetor nulo. "
    ) * 3
    bloco_formula_espacado = "\n".join(
        [" ", " ", " ", " ker", " ", " (", " L", " )", " =", " ", " ", " ", " "] * 8
    )
    texto = prosa + bloco_formula_espacado + prosa
    ok, motivo = avaliar_qualidade(texto, FiltrosQualidade())
    assert ok, motivo


def test_razao_e_calculada_sobre_nao_espaco_nao_sobre_total():
    # caso minimo e direto: um texto com muito espaco em branco mas cujo
    # conteudo nao-espaco e inteiramente alfabetico deve passar.
    texto = ("palavra " + " " * 50) * 10  # bastante espaco extra, mas so letras como conteudo
    ok, motivo = avaliar_qualidade(texto, FiltrosQualidade(min_caracteres=10))
    assert ok, motivo
