import json
from pathlib import Path

from ignis.data.manifest import EntradaManifesto, StatusDocumento
from ignis.data.splits import (
    ProporcoesSplit,
    carregar_splits,
    gerar_splits,
    salvar_splits,
    verificar_contaminacao,
)


def _manifesto_sintetico(n: int, fonte: str | None = None) -> list[EntradaManifesto]:
    """Gera `n` documentos aceitos sintéticos, sem relação de quase-duplicata
    entre si. Quando `fonte` é None (padrão), cada documento recebe uma
    fonte PRÓPRIA e distinta — ou seja, os `n` documentos são `n` unidades
    independentes mesmo com `agrupar_por_fonte=True` (o padrão de
    `gerar_splits`). Quando `fonte` é uma string explícita, todos os `n`
    documentos compartilham essa mesma fonte — útil para testes que querem
    forçar o agrupamento por fonte deliberadamente."""
    return [
        EntradaManifesto(
            documento_id=f"doc{i}",
            caminho_origem=f"/fake/doc{i}.txt",
            fonte=fonte if fonte is not None else f"fonte_{i}",
            licenca="CC-BY-4.0",
            permissao_treinamento=True,
            sha256=f"hash{i}",
            tamanho_caracteres=1000,
            status=StatusDocumento.ACEITO,
        )
        for i in range(n)
    ]


def _manifesto_multi_fonte(n_fontes: int, docs_por_fonte: int) -> list[EntradaManifesto]:
    """Como _manifesto_sintetico, mas com IDs globais únicos entre fontes —
    evita colisão de documento_id quando se quer simular várias fontes
    independentes (necessário para exercitar unidades > 1 por fonte)."""
    manifesto: list[EntradaManifesto] = []
    for fonte_idx in range(n_fontes):
        for doc_idx in range(docs_por_fonte):
            manifesto.append(
                EntradaManifesto(
                    documento_id=f"fonte{fonte_idx}-doc{doc_idx}",
                    caminho_origem=f"/fake/fonte{fonte_idx}/doc{doc_idx}.txt",
                    fonte=f"fonte_{fonte_idx}",
                    licenca="CC-BY-4.0",
                    permissao_treinamento=True,
                    sha256=f"hash-{fonte_idx}-{doc_idx}",
                    tamanho_caracteres=1000,
                    status=StatusDocumento.ACEITO,
                )
            )
    return manifesto


def test_splits_sao_reproduziveis_com_mesma_seed():
    manifesto = _manifesto_sintetico(40)
    a1 = gerar_splits(manifesto, seed=7)
    a2 = gerar_splits(manifesto, seed=7)
    assert a1.treino == a2.treino
    assert a1.validacao == a2.validacao
    assert a1.teste == a2.teste


def test_splits_mudam_com_seed_diferente():
    manifesto = _manifesto_sintetico(40)
    a1 = gerar_splits(manifesto, seed=1)
    a2 = gerar_splits(manifesto, seed=2)
    assert (a1.treino, a1.validacao, a1.teste) != (a2.treino, a2.validacao, a2.teste)


def test_nenhum_documento_repetido_entre_splits():
    manifesto = _manifesto_sintetico(100)
    atribuicao = gerar_splits(manifesto, seed=42)
    assert verificar_contaminacao(atribuicao) == []

    todos = atribuicao.treino + atribuicao.validacao + atribuicao.teste
    assert len(todos) == len(set(todos))


def test_documentos_nao_aceitos_ficam_fora_dos_splits():
    manifesto = _manifesto_sintetico(5)
    manifesto.append(
        EntradaManifesto(
            documento_id="rejeitado",
            caminho_origem="/fake/rejeitado.txt",
            fonte="fonte_a",
            sha256="hashx",
            tamanho_caracteres=10,
            status=StatusDocumento.REJEITADO,
            motivo="muito curto",
        )
    )
    atribuicao = gerar_splits(manifesto, seed=42)
    todos = atribuicao.treino + atribuicao.validacao + atribuicao.teste
    assert "rejeitado" not in todos


def test_proporcoes_invalidas_levantam_erro():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ProporcoesSplit(treino=0.5, validacao=0.3, teste=0.3)  # soma 1.1


def test_persistencia_de_splits_e_fiel(tmp_path: Path):
    manifesto = _manifesto_sintetico(20)
    atribuicao = gerar_splits(manifesto, seed=99)
    caminho = tmp_path / "splits.json"
    salvar_splits(atribuicao, caminho)

    recarregado = carregar_splits(caminho)
    assert recarregado.treino == atribuicao.treino
    assert recarregado.validacao == atribuicao.validacao
    assert recarregado.teste == atribuicao.teste
    assert recarregado.seed == 99

    # arquivo deve ser JSON legível, incluindo seed e critérios de separação
    dados_brutos = json.loads(caminho.read_text(encoding="utf-8"))
    assert dados_brutos["seed"] == 99
    assert "proporcoes" in dados_brutos


def test_quase_duplicatas_ficam_no_mesmo_split():
    manifesto = _manifesto_sintetico(10)
    # forca dois documentos a pertencerem ao mesmo grupo de quase-duplicata
    manifesto[0].grupo_quase_duplicata = "grupo_x"
    manifesto[1].grupo_quase_duplicata = "grupo_x"

    atribuicao = gerar_splits(manifesto, seed=42, agrupar_por_fonte=False)
    conjuntos = {
        "treino": set(atribuicao.treino),
        "validacao": set(atribuicao.validacao),
        "teste": set(atribuicao.teste),
    }
    for nome, conjunto in conjuntos.items():
        if "doc0" in conjunto:
            assert "doc1" in conjunto, f"doc0 e doc1 (mesmo grupo de quase-duplicata) separados: doc1 nao esta em {nome}"


def test_splits_garantem_treino_validacao_e_teste_nao_vazios_com_unidades_suficientes():
    """Regressão: com unidades independentes suficientes (fontes distintas),
    treino, validação e teste devem sair todos não-vazios, mesmo com
    proporções desiguais (90/5/5) — antes desta correção, um corpus pequeno
    com poucas unidades podia produzir validação/teste vazios silenciosamente."""
    manifesto = _manifesto_multi_fonte(n_fontes=6, docs_por_fonte=3)

    atribuicao = gerar_splits(manifesto, seed=42)
    # 6 fontes, 3 documentos cada: com agrupar_por_fonte=True (padrão), todos
    # os documentos de uma mesma fonte caem na mesma unidade -> 6 unidades.
    assert atribuicao.n_unidades == 6
    assert len(atribuicao.treino) > 0
    assert len(atribuicao.validacao) > 0, "validação ficou vazia mesmo com unidades suficientes"
    assert len(atribuicao.teste) > 0, "teste ficou vazio mesmo com unidades suficientes"
    assert verificar_contaminacao(atribuicao) == []


def test_grupos_insuficientes_levanta_erro_em_vez_de_split_vazio():
    """Regressão: quando NÃO há unidades suficientes para garantir pelo
    menos uma por split ativo, a função deve levantar um erro explícito —
    nunca aprovar silenciosamente um split vazio."""
    import pytest

    from ignis.data.splits import GruposInsuficientesError

    # Apenas 2 unidades independentes (mesma fonte, cada doc é sua própria
    # unidade), mas as proporções padrão (treino/validação/teste, todas > 0)
    # exigem 3 unidades no mínimo.
    manifesto = _manifesto_sintetico(2)
    with pytest.raises(GruposInsuficientesError):
        gerar_splits(manifesto, seed=42)


def test_splits_vazios_sem_documentos_aceitos_nao_e_erro():
    """Zero documentos aceitos não é um caso de 'grupos insuficientes' (não
    há ambiguidade sobre o que fazer) — deve retornar três splits vazios
    sem levantar exceção."""
    manifesto = [
        EntradaManifesto(
            documento_id="doc0",
            caminho_origem="/fake/doc0.txt",
            fonte="fonte_a",
            sha256="hash0",
            tamanho_caracteres=10,
            status=StatusDocumento.REJEITADO,
            motivo="muito curto",
        )
    ]
    atribuicao = gerar_splits(manifesto, seed=42)
    assert atribuicao.n_unidades == 0
    assert atribuicao.treino == []
    assert atribuicao.validacao == []
    assert atribuicao.teste == []


def test_proporcao_zero_nao_exige_unidade_propria():
    """Um split com proporção 0 (ex.: sem validação) nunca precisa de
    unidade própria e permanece vazio por design, não por erro."""
    manifesto = _manifesto_sintetico(2)
    proporcoes = ProporcoesSplit(treino=0.8, validacao=0.0, teste=0.2)
    atribuicao = gerar_splits(manifesto, seed=42, proporcoes=proporcoes)
    assert atribuicao.validacao == []
    assert len(atribuicao.treino) + len(atribuicao.teste) == 2


def test_proporcoes_realizadas_sao_documentadas():
    manifesto = _manifesto_multi_fonte(n_fontes=6, docs_por_fonte=5)

    atribuicao = gerar_splits(manifesto, seed=42)
    relatorio = atribuicao.relatorio()
    assert "proporcoes_solicitadas" in relatorio
    assert "proporcoes_realizadas" in relatorio
    soma_realizada = sum(relatorio["proporcoes_realizadas"].values())
    assert abs(soma_realizada - 1.0) < 1e-9


def test_unidades_relacionadas_nunca_sao_separadas_para_preencher_proporcao():
    """Mesmo quando a reserva do passo 1 poderia 'preferir' separar
    documentos relacionados para preencher validação/teste, a unidade
    inteira (grupo de quase-duplicata) deve ir para um único split.

    Usa 6 documentos (3 no grupo_y + 3 avulsos) para garantir unidades
    independentes suficientes (4 unidades) para os 3 splits ativos — com
    menos unidades, gerar_splits levantaria GruposInsuficientesError
    corretamente, o que não é o que este teste quer exercitar."""
    manifesto = _manifesto_sintetico(6, fonte="fonte_unica")
    manifesto[0].grupo_quase_duplicata = "grupo_y"
    manifesto[1].grupo_quase_duplicata = "grupo_y"
    manifesto[2].grupo_quase_duplicata = "grupo_y"

    atribuicao = gerar_splits(manifesto, seed=1, agrupar_por_fonte=False)
    conjuntos = {
        "treino": set(atribuicao.treino),
        "validacao": set(atribuicao.validacao),
        "teste": set(atribuicao.teste),
    }
    membros_grupo_y = {"doc0", "doc1", "doc2"}
    for nome, conjunto in conjuntos.items():
        intersecao = conjunto & membros_grupo_y
        if intersecao:
            assert intersecao == membros_grupo_y, (
                f"grupo_y foi dividido entre splits: apenas {intersecao} caiu em {nome}"
            )


def test_mesma_fonte_sem_quase_duplicata_fica_no_mesmo_split():
    """Regressão: uma implementação anterior usava a chave `fonte::grupo`,
    que só agrupava por fonte documentos que TAMBÉM fossem quase-duplicatas
    entre si — documentos distintos (não quase-duplicados) da mesma fonte
    viravam unidades independentes, na prática ignorando `agrupar_por_fonte`.
    Isso é uma forma real de vazamento (duas partes de um mesmo livro/site
    podiam cair uma no treino e outra no teste). Com `agrupar_por_fonte=True`
    (padrão), TODOS os documentos de uma fonte devem cair no mesmo split,
    mesmo sem nenhuma relação de quase-duplicata entre eles."""
    manifesto = _manifesto_multi_fonte(n_fontes=4, docs_por_fonte=6)
    atribuicao = gerar_splits(manifesto, seed=7)

    conjuntos = {
        "treino": set(atribuicao.treino),
        "validacao": set(atribuicao.validacao),
        "teste": set(atribuicao.teste),
    }
    for fonte_idx in range(4):
        ids_da_fonte = {f"fonte{fonte_idx}-doc{doc_idx}" for doc_idx in range(6)}
        splits_atingidos = [nome for nome, conjunto in conjuntos.items() if conjunto & ids_da_fonte]
        assert len(splits_atingidos) == 1, (
            f"documentos da fonte_{fonte_idx} apareceram em mais de um split: {splits_atingidos}"
        )
        assert (conjuntos[splits_atingidos[0]] & ids_da_fonte) == ids_da_fonte, (
            f"nem todos os documentos da fonte_{fonte_idx} foram para o mesmo split"
        )
