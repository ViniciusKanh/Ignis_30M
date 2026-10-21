from ignis.data.pii import detectar_pii


def test_detecta_email():
    achados = detectar_pii("Contato: alguem@example.com para mais informacoes.")
    assert "email" in achados


def test_detecta_cpf():
    achados = detectar_pii("Meu CPF e 123.456.789-00, guarde em sigilo.")
    assert "cpf" in achados


def test_texto_sem_pii_nao_sinaliza_nada():
    achados = detectar_pii("Este e um texto comum sobre tecnologia e inovacao no Brasil.")
    assert achados == {}


# -- Sprint 6.2: regressao para os falsos positivos reais encontrados na
# auditoria do corpus da Wikipedia (ver docs/experiments/fase3-sprint6.2-qualidade-corpus.md).
# Todos os numeros abaixo sao inventados para o teste, nao sao PII real de
# ninguem nem dados extraidos do corpus real.


def test_isbn13_em_referencia_bibliografica_nao_e_falso_positivo_de_cartao():
    texto = "Alguma Editora. 2008. ISBN 978-3-85474-187-9 Outro autor, Outro titulo."
    achados = detectar_pii(texto)
    assert "cartao_credito" not in achados


def test_isbn10_em_referencia_bibliografica_nao_e_falso_positivo_de_telefone():
    texto = "Paris: Editions du Seuil, ISBN 2020091380, 1986."
    achados = detectar_pii(texto)
    assert "telefone_br" not in achados


def test_doi_em_referencia_cientifica_nao_e_falso_positivo_de_telefone():
    texto = "PMID 16129826. doi:10.1073/pnas.0505955102 Fulano, Ciclano."
    achados = detectar_pii(texto)
    assert "telefone_br" not in achados
    assert "cartao_credito" not in achados


def test_numero_de_13_digitos_sem_contexto_permanece_conservador():
    assert detectar_pii("Numero: 1234567890123 registrado.").get("cartao_credito") == 1


def test_numero_de_16_digitos_continua_detectado_como_cartao_credito():
    texto = "Numero do cartao: 4532015112830366 enviado por engano no email."
    achados = detectar_pii(texto)
    assert achados.get("cartao_credito") == 1


def test_telefone_br_formatado_continua_detectado_fora_de_contexto_bibliografico():
    texto = "Pode me ligar no (11) 98765-4321 amanha de manha."
    achados = detectar_pii(texto)
    assert achados.get("telefone_br") == 1


def test_isbn_fora_da_janela_de_contexto_ainda_e_sinalizado():
    # garante que a excecao e mesmo por PROXIMIDADE, nao uma whitelist cega
    # de qualquer numero que apareca em um texto que mencione ISBN em algum
    # lugar distante.
    texto = (
        "ISBN. " + ("palavra " * 20) + "ligue para (11) 98765-4321 o quanto antes."
    )
    achados = detectar_pii(texto)
    assert achados.get("telefone_br") == 1


def test_cnpj_continua_sendo_sinalizado_mesmo_sendo_pessoa_juridica():
    # decisao deliberada (ver docstring do modulo): CNPJ nao e PII de pessoa
    # fisica, mas a deteccao foi mantida por precaucao -- este teste
    # documenta e protege essa decisao contra remocao acidental futura.
    texto = "A empresa foi registrada com CNPJ de numero 30.464.598/0001-30, dissolvida em seguida."
    achados = detectar_pii(texto)
    assert achados.get("cnpj") == 1


def test_referencia_proxima_nao_oculta_telefone():
    for prefixo in ("ISBN: ligue ", "doi: ligue ", "ISBN 2020091380; ", "doi:10.1234/obra; "):
        assert detectar_pii(prefixo + "(11) 98765-4321").get("telefone_br") == 1
