from ignis.data.normalize import normalizar_texto


def test_preserva_acentuacao_e_cedilha():
    texto = "Ação, coração, função, não, português, São Paulo, Brasília."
    assert normalizar_texto(texto) == texto


def test_normaliza_quebras_de_linha_crlf():
    texto = "linha um\r\nlinha dois\r\nlinha tres"
    resultado = normalizar_texto(texto)
    assert "\r" not in resultado
    assert resultado == "linha um\nlinha dois\nlinha tres"


def test_remove_caracteres_de_controle():
    texto = "texto com controle\x00invisivel\x1f aqui"
    resultado = normalizar_texto(texto)
    assert "\x00" not in resultado and "\x1f" not in resultado


def test_colapsa_espacos_e_linhas_excessivas():
    texto = "muitos    espacos   aqui\n\n\n\n\nmuitas linhas"
    resultado = normalizar_texto(texto)
    assert "    " not in resultado
    assert "\n\n\n" not in resultado


def test_forma_nfc_normaliza_combinacao_de_acentos():
    # "á" como 'a' + combining acute accent (NFD) deve virar a forma composta (NFC)
    nfd = "á"  # a + ́combining acute
    resultado = normalizar_texto(nfd)
    assert resultado == "á"


def test_preserva_formulas_latex_e_numeros():
    textos = [
        r"Resultado: {\displaystyle x=1} e {\displaystyle y=2}.",
        r"Nucleo: {\displaystyle \ker(L)=\left\{\mathbf{v}\right\}} fim.",
        r"Formula incompleta: {\displaystyle x=1",
        "Constantes: 3,14159; 6.022e23; x² + y² = z².",
    ]
    for texto in textos:
        assert normalizar_texto(texto) == texto
