"""Testes de `ignis.cli._distribuicao_por_fonte` — em particular a auditoria
de granularidade adicionada na Sprint 5 (`n_unidades_agrupamento` por fonte
e o aviso de "fonte colapsada"), que usa `ignis.data.splits.
contar_unidades_por_fonte` para detectar quando uma fonte com muitos
documentos aceitos forma, na prática, uma única unidade indivisível de
split. Torch-free: `ignis.cli` só importa torch dentro de funções que não
são exercitadas aqui (`cmd_train`/`_executar_treinamento_real`)."""

from ignis.cli import _distribuicao_por_fonte
from ignis.data.manifest import EntradaManifesto, StatusDocumento


def _entrada(doc_id: str, fonte: str, unidade: str | None, status: StatusDocumento = StatusDocumento.ACEITO) -> EntradaManifesto:
    return EntradaManifesto(
        documento_id=doc_id,
        caminho_origem=f"/fake/{doc_id}.txt",
        fonte=fonte,
        licenca="CC-BY-4.0" if status == StatusDocumento.ACEITO else None,
        permissao_treinamento=True if status == StatusDocumento.ACEITO else None,
        sha256=f"hash-{doc_id}",
        tamanho_caracteres=1000,
        status=status,
        motivo="" if status == StatusDocumento.ACEITO else "motivo de teste",
        unidade_agrupamento=unidade,
    )


def test_distribuicao_inclui_n_unidades_agrupamento_por_fonte():
    manifesto = [
        _entrada("a1", fonte="fonte_a", unidade="u1"),
        _entrada("a2", fonte="fonte_a", unidade="u2"),
        _entrada("b1", fonte="fonte_b", unidade="u3"),
        _entrada("b2", fonte="fonte_b", unidade="u3"),
    ]
    distribuicao = _distribuicao_por_fonte(manifesto)
    assert distribuicao["por_fonte"]["fonte_a"]["n_unidades_agrupamento"] == 2
    assert distribuicao["por_fonte"]["fonte_b"]["n_unidades_agrupamento"] == 1


def test_fonte_colapsada_gera_aviso_especifico():
    """Uma fonte com vários documentos aceitos mas apenas 1 unidade de
    agrupamento (ex.: um provedor grande ingerido sem distinguir shards)
    deve gerar um aviso específico de granularidade — distinto do aviso de
    concentração percentual já existente desde a Sprint 4."""
    manifesto = [
        _entrada(f"wiki-{i}", fonte="wikipedia-pt", unidade=None) for i in range(5)
    ] + [
        _entrada("penso-1", fonte="penso-textos-proprios", unidade="shard_unico"),
    ]
    distribuicao = _distribuicao_por_fonte(manifesto)
    avisos = " ".join(distribuicao["avisos_concentracao"])
    assert "wikipedia-pt" in avisos
    assert "UNICA" in avisos or "unica" in avisos.lower()


def test_fonte_diversa_nao_gera_aviso_de_granularidade():
    manifesto = [
        _entrada(f"doc-{i}", fonte="fonte_diversa", unidade=f"shard_{i}") for i in range(5)
    ]
    distribuicao = _distribuicao_por_fonte(manifesto)
    avisos = " ".join(distribuicao["avisos_concentracao"])
    assert "unica unidade de agrupamento independente" not in avisos.lower()


def test_documentos_nao_aceitos_nao_contam_para_n_unidades_agrupamento():
    manifesto = [
        _entrada("rej1", fonte="fonte_a", unidade="u1", status=StatusDocumento.REJEITADO),
    ]
    distribuicao = _distribuicao_por_fonte(manifesto)
    assert distribuicao["por_fonte"]["fonte_a"]["n_unidades_agrupamento"] == 0
