"""Testes da distinção PROVEDOR (`fonte`) vs. UNIDADE DE AGRUPAMENTO
(`unidade_agrupamento`), introduzida na Sprint 5 para corrigir um problema
real de granularidade de splits: agrupar estritamente por `fonte` fazia um
provedor grande inteiro (ex.: toda a Wikipédia em português) colapsar em
uma única unidade indivisível, mesmo contendo milhares de documentos
independentes entre si — o que impede splits de validação/teste realistas
para corpora dominados por um provedor grande.

Três camadas são exercitadas aqui:
1. `ignis.data.ingest` — resolução do padrão de `unidade_agrupamento`
   (nome da pasta imediata) e sua independência de `fonte`.
2. `ignis.data.splits` — `_componentes_conexas` usando `unidade_agrupamento`
   (com fallback para `fonte`), e o novo auditor `contar_unidades_por_fonte`.
3. Integração ponta a ponta com arquivos reais em disco, simulando um
   provedor grande ingerido em múltiplos shards (subpastas).
"""

import json
from pathlib import Path

from ignis.data.ingest import ingerir_arquivo, ingerir_diretorio
from ignis.data.manifest import EntradaManifesto, StatusDocumento
from ignis.data.pipeline import processar_diretorio
from ignis.data.splits import contar_unidades_por_fonte, gerar_splits, verificar_contaminacao


# ---------------------------------------------------------------------------
# Camada 1: ignis.data.ingest
# ---------------------------------------------------------------------------


def test_ingest_unidade_agrupamento_padrao_e_pasta_imediata(tmp_path: Path):
    """Sem `unidade_agrupamento` explícita, o padrão é o nome da pasta
    imediata do arquivo — DIFERENTE de `fonte` quando `fonte` é passada
    explicitamente (caso de uso: um provedor grande organizado em shards)."""
    shard_a = tmp_path / "shard_0001"
    shard_a.mkdir()
    (shard_a / "artigo1.txt").write_text("Texto do artigo um.", encoding="utf-8")

    shard_b = tmp_path / "shard_0002"
    shard_b.mkdir()
    (shard_b / "artigo2.txt").write_text("Texto do artigo dois.", encoding="utf-8")

    docs = ingerir_diretorio(tmp_path, fonte="wikipedia-pt")

    por_id = {d.documento_id: d for d in docs}
    assert por_id["artigo1"].fonte == "wikipedia-pt"
    assert por_id["artigo2"].fonte == "wikipedia-pt"
    # mesma fonte (provedor), mas unidades de agrupamento DIFERENTES (shards).
    assert por_id["artigo1"].unidade_agrupamento == "shard_0001"
    assert por_id["artigo2"].unidade_agrupamento == "shard_0002"
    assert por_id["artigo1"].unidade_agrupamento != por_id["artigo2"].unidade_agrupamento


def test_ingest_sem_fonte_explicita_unidade_coincide_com_fonte(tmp_path: Path):
    """Quando `fonte` não é passada (comportamento anterior à Sprint 5),
    `fonte` e `unidade_agrupamento` coincidem (ambas = nome da pasta) —
    preserva exatamente o comportamento anterior."""
    pasta = tmp_path / "meu_corpus"
    pasta.mkdir()
    arquivo = pasta / "doc.txt"
    arquivo.write_text("Um texto qualquer.", encoding="utf-8")

    [doc] = ingerir_arquivo(arquivo)
    assert doc.fonte == "meu_corpus"
    assert doc.unidade_agrupamento == "meu_corpus"
    assert doc.unidade_agrupamento == doc.fonte


def test_ingest_jsonl_unidade_agrupamento_explicita_tem_prioridade(tmp_path: Path):
    """Uma chave `unidade_agrupamento` explícita no registro JSONL tem
    prioridade sobre o padrão derivado da pasta — permite, por exemplo,
    marcar que duas linhas de pastas diferentes são a mesma unidade (ex.:
    duas revisões do mesmo artigo)."""
    pasta = tmp_path / "revisoes"
    pasta.mkdir()
    arquivo = pasta / "artigo.jsonl"
    registros = [
        {"id": "rev1", "texto": "Primeira revisão do artigo.", "unidade_agrupamento": "artigo-x"},
        {"id": "rev2", "texto": "Segunda revisão do artigo.", "unidade_agrupamento": "artigo-x"},
    ]
    arquivo.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in registros) + "\n", encoding="utf-8")

    docs = ingerir_arquivo(arquivo, fonte="wikipedia-pt")
    assert {d.unidade_agrupamento for d in docs} == {"artigo-x"}
    assert {d.fonte for d in docs} == {"wikipedia-pt"}


# ---------------------------------------------------------------------------
# Camada 2: ignis.data.splits
# ---------------------------------------------------------------------------


def _entrada(doc_id: str, fonte: str, unidade: str | None, sha: str | None = None) -> EntradaManifesto:
    return EntradaManifesto(
        documento_id=doc_id,
        caminho_origem=f"/fake/{doc_id}.txt",
        fonte=fonte,
        licenca="CC-BY-4.0",
        permissao_treinamento=True,
        sha256=sha or f"hash-{doc_id}",
        tamanho_caracteres=1000,
        status=StatusDocumento.ACEITO,
        unidade_agrupamento=unidade,
    )


def test_fonte_grande_com_unidade_agrupamento_gera_multiplas_unidades_independentes():
    """Caso motivador da Sprint 5: um único provedor grande (`fonte`
    constante) organizado em vários shards (`unidade_agrupamento` distinta
    por shard) deve gerar MÚLTIPLAS unidades independentes — não uma só —
    permitindo splits de validação/teste realistas mesmo quando o corpus é
    dominado por esse provedor."""
    manifesto = [
        _entrada(f"doc{i}", fonte="wikipedia-pt", unidade=f"shard_{i:04d}")
        for i in range(12)
    ]
    atribuicao = gerar_splits(manifesto, seed=42)
    assert atribuicao.n_unidades == 12
    assert len(atribuicao.treino) > 0
    assert len(atribuicao.validacao) > 0, "validação vazia mesmo com 12 shards independentes"
    assert len(atribuicao.teste) > 0, "teste vazio mesmo com 12 shards independentes"
    assert verificar_contaminacao(atribuicao) == []


def test_sem_unidade_agrupamento_mesma_fonte_colapsa_em_uma_unidade():
    """Contraste documentando o comportamento anterior (preservado como
    fallback): SEM `unidade_agrupamento` explícita, todos os documentos de
    uma mesma fonte caem em uma única unidade indivisível — exatamente o
    problema que motivou a Sprint 5 quando a fonte é um provedor grande.
    Por isso `gerar_splits` levanta `GruposInsuficientesError` aqui (1
    unidade não basta para 3 splits ativos)."""
    import pytest

    from ignis.data.splits import GruposInsuficientesError

    manifesto = [_entrada(f"doc{i}", fonte="wikipedia-pt", unidade=None) for i in range(12)]
    with pytest.raises(GruposInsuficientesError):
        gerar_splits(manifesto, seed=42)


def test_contar_unidades_por_fonte_detecta_fonte_colapsada():
    """`contar_unidades_por_fonte` deve apontar, por fonte, quantas unidades
    independentes ela realmente contribui — a auditoria que permite
    detectar, no relatório do pipeline, uma fonte com muitos documentos mas
    uma única unidade (sinal de granularidade inadequada)."""
    manifesto = [
        # fonte_colapsada: 5 documentos, mas NENHUMA unidade_agrupamento
        # própria -> cai inteira em 1 única unidade (fallback para fonte).
        *[_entrada(f"colapsada-doc{i}", fonte="fonte_colapsada", unidade=None) for i in range(5)],
        # fonte_diversa: 5 documentos, cada um com sua própria unidade
        # (shards distintos) -> 5 unidades independentes.
        *[_entrada(f"diversa-doc{i}", fonte="fonte_diversa", unidade=f"shard_{i}") for i in range(5)],
    ]
    contagem = contar_unidades_por_fonte(manifesto)
    assert contagem["fonte_colapsada"] == 1
    assert contagem["fonte_diversa"] == 5


def test_contar_unidades_por_fonte_ignora_documentos_nao_aceitos():
    manifesto = [
        _entrada("aceito1", fonte="fonte_a", unidade="u1"),
        EntradaManifesto(
            documento_id="rejeitado1",
            caminho_origem="/fake/rejeitado1.txt",
            fonte="fonte_a",
            sha256="hash-rejeitado1",
            tamanho_caracteres=10,
            status=StatusDocumento.REJEITADO,
            motivo="muito curto",
            unidade_agrupamento="u2",
        ),
    ]
    contagem = contar_unidades_por_fonte(manifesto)
    assert contagem == {"fonte_a": 1}


# ---------------------------------------------------------------------------
# Camada 3: integração ponta a ponta com arquivos reais em disco
# ---------------------------------------------------------------------------


def test_integracao_provedor_grande_em_shards_produz_splits_realistas(tmp_path: Path):
    """Simula um provedor grande (ex.: um dump da Wikipédia em português)
    ingerido em várias subpastas (shards) sob uma única `fonte` constante,
    do arquivo em disco até os splits finais. Com a correção da Sprint 5,
    isso deve produzir muitas unidades independentes e splits de
    treino/validação/teste todos não-vazios e sem contaminação — o cenário
    realista de corpus com múltiplas unidades independentes exigido pela
    diretriz da Sprint 5."""
    corpus_dir = tmp_path / "corpus_wikipedia"
    n_shards = 10
    for i in range(n_shards):
        shard = corpus_dir / f"shard_{i:04d}"
        shard.mkdir(parents=True)
        registro = {
            "id": f"artigo-{i}",
            "texto": (
                f"Este e o conteudo distinto do artigo numero {i} da Wikipedia em "
                "portugues, com frases proprias e suficientes para passar pelos "
                "filtros de qualidade minima do pipeline de ingestao do Ignis. "
                f"Cada artigo aborda um tema diferente, aqui o tema {i}, para "
                "evitar deteccao de quase-duplicata entre shards distintos."
            ),
            "licenca": "CC-BY-SA-4.0",
            "permissao_treinamento": True,
        }
        (shard / "artigo.jsonl").write_text(json.dumps(registro, ensure_ascii=False) + "\n", encoding="utf-8")

    resultado = processar_diretorio(corpus_dir, fonte="wikipedia-pt")
    aceitos = [e for e in resultado.manifesto if e.status == StatusDocumento.ACEITO]
    assert len(aceitos) == n_shards
    assert all(e.fonte == "wikipedia-pt" for e in aceitos)
    # unidade_agrupamento deve variar por shard (não ser constante como fonte).
    assert len({e.unidade_agrupamento for e in aceitos}) == n_shards

    atribuicao = gerar_splits(resultado.manifesto, seed=42)
    assert atribuicao.n_unidades == n_shards
    assert len(atribuicao.treino) > 0
    assert len(atribuicao.validacao) > 0
    assert len(atribuicao.teste) > 0
    assert verificar_contaminacao(atribuicao) == []

    contagem_unidades = contar_unidades_por_fonte(resultado.manifesto)
    assert contagem_unidades["wikipedia-pt"] == n_shards
