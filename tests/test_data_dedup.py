from ignis.data.dedup import agrupar_quase_duplicatas, sha256_texto, similaridade_jaccard


def test_sha256_estavel_e_sensivel_ao_conteudo():
    a = sha256_texto("mesmo texto")
    b = sha256_texto("mesmo texto")
    c = sha256_texto("texto diferente")
    assert a == b
    assert a != c


def test_similaridade_jaccard_identico_e_disjunto():
    assert similaridade_jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert similaridade_jaccard({"a"}, {"b"}) == 0.0
    assert similaridade_jaccard(set(), set()) == 1.0


def test_agrupar_quase_duplicatas_agrupa_textos_quase_identicos():
    # Textos longos e lexicamente variados: um pequeno acréscimo no final
    # altera poucos shingles de 5 palavras em relação ao total, produzindo
    # similaridade de Jaccard real acima do limiar. Frases curtas e
    # repetitivas (poucos shingles únicos) não servem para este teste,
    # pois qualquer texto anexado desloca a similaridade para baixo do
    # limiar mesmo quando os textos são, na prática, quase-duplicatas.
    base = (
        "A pesquisa em processamento de linguagem natural avancou muito nos ultimos anos "
        "graças ao desenvolvimento de arquiteturas baseadas em atencao como o Transformer. "
        "Esses modelos conseguem capturar dependencias de longo alcance em sequencias de texto "
        "e sao treinados em grandes volumes de dados textuais coletados de fontes diversas. "
        "O treinamento do zero exige cuidado com a qualidade dos dados, a tokenizacao e a "
        "arquitetura escolhida para garantir resultados reprodutiveis e confiaveis."
    )
    quase_igual = base + " Este paragrafo final foi adicionado apenas para teste."
    diferente = (
        "O clima na regiao sul do Brasil costuma ser mais frio durante o inverno, com "
        "ocorrencia frequente de geadas e, em algumas areas serranas, ate mesmo neve. "
        "Essas condicoes atraem turistas interessados em paisagens diferentes das demais "
        "regioes do pais, especialmente durante os meses de junho e julho de cada ano."
    )

    textos = {"a": base, "b": quase_igual, "c": diferente}
    grupos = agrupar_quase_duplicatas(textos, limiar=0.8)

    assert grupos["a"] == grupos["b"], "textos quase identicos deveriam cair no mesmo grupo"
    assert grupos["c"] != grupos["a"], "texto diferente nao deveria ser agrupado"
