"""Testes de ingestao incremental e retomavel de `processar_diretorio`
(Sprint 5) -- `manifesto_existente` (pula documentos ja processados e
inalterados, por hash SHA-256) e `limite_novos_documentos` (processa no
maximo N documentos novos/atualizados por chamada, deixando o resto para
uma chamada futura)."""

from pathlib import Path

from ignis.data.manifest import StatusDocumento
from ignis.data.pipeline import processar_diretorio


def _escrever_doc(pasta: Path, nome: str, texto: str, licenca: str = "CC-BY-4.0") -> None:
    import json

    pasta.mkdir(parents=True, exist_ok=True)
    registro = {"id": nome, "texto": texto, "licenca": licenca, "permissao_treinamento": True}
    (pasta / f"{nome}.jsonl").write_text(json.dumps(registro, ensure_ascii=False) + "\n", encoding="utf-8")


def test_execucao_incremental_pula_documentos_inalterados(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    _escrever_doc(corpus_dir, "doc1", "Primeiro documento do corpus de teste incremental, com texto suficiente para passar pelo filtro minimo de qualidade de duzentos caracteres exigido pelo pipeline de ingestao do Ignis antes de ser aceito no manifesto.")

    primeira = processar_diretorio(corpus_dir)
    assert primeira.n_novos_processados == 1
    assert len(primeira.manifesto) == 1

    # Segunda chamada, mesmo diretorio, SEM nenhuma mudanca: nada deve ser
    # reprocessado (n_novos_processados == 0), mas o documento continua no
    # manifesto resultante (preservado do manifesto existente).
    segunda = processar_diretorio(corpus_dir, manifesto_existente=primeira.manifesto)
    assert segunda.n_novos_processados == 0
    assert len(segunda.manifesto) == 1
    assert segunda.manifesto[0].sha256 == primeira.manifesto[0].sha256


def test_execucao_incremental_detecta_arquivo_novo(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    _escrever_doc(corpus_dir, "doc1", "Primeiro documento, ja existente na primeira execucao do pipeline de ingestao, com texto longo o suficiente para passar pelo filtro minimo de qualidade de duzentos caracteres do Ignis.")

    primeira = processar_diretorio(corpus_dir)

    _escrever_doc(corpus_dir, "doc2", "Segundo documento, adicionado somente depois da primeira execucao do pipeline, tambem com texto longo o suficiente para passar pelo filtro minimo de qualidade do Ignis.")
    segunda = processar_diretorio(corpus_dir, manifesto_existente=primeira.manifesto)

    assert segunda.n_novos_processados == 1  # so o doc2 e novo
    ids = {e.documento_id for e in segunda.manifesto}
    assert ids == {"doc1", "doc2"}


def test_execucao_incremental_detecta_arquivo_alterado(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    _escrever_doc(corpus_dir, "doc1", "Conteudo original do documento antes de qualquer alteracao de teste, com texto longo o suficiente para passar pelo filtro minimo de qualidade de duzentos caracteres do pipeline.")

    primeira = processar_diretorio(corpus_dir)
    sha_original = primeira.manifesto[0].sha256

    _escrever_doc(corpus_dir, "doc1", "Conteudo COMPLETAMENTE DIFERENTE apos a alteracao simulada no teste, tambem com texto longo o suficiente para passar pelo filtro minimo de qualidade do pipeline do Ignis.")
    segunda = processar_diretorio(corpus_dir, manifesto_existente=primeira.manifesto)

    assert segunda.n_novos_processados == 1  # tratado como atualizacao, nao pulado
    assert len(segunda.manifesto) == 1
    assert segunda.manifesto[0].sha256 != sha_original


def test_hashes_vistos_detecta_duplicata_exata_entre_execucoes(tmp_path: Path):
    """Duplicata exata (mesmo texto normalizado) introduzida em uma pasta
    DIFERENTE numa execucao POSTERIOR deve ser rejeitada -- a deteccao de
    hash exato persiste entre chamadas via `manifesto_existente`, mesmo que
    a quase-duplicata (Jaccard) nao seja recalculada retroativamente."""
    corpus_dir = tmp_path / "corpus"
    texto_comum = (
        "Este texto sera duplicado exatamente em uma segunda pasta na segunda execucao, "
        "com conteudo longo o suficiente para passar pelo filtro minimo de qualidade de "
        "duzentos caracteres exigido pelo pipeline de ingestao do Ignis antes da triagem."
    )
    _escrever_doc(corpus_dir / "pastaA", "original", texto_comum)

    primeira = processar_diretorio(corpus_dir)
    assert primeira.manifesto[0].status == StatusDocumento.ACEITO

    _escrever_doc(corpus_dir / "pastaB", "copia", texto_comum)
    segunda = processar_diretorio(corpus_dir, manifesto_existente=primeira.manifesto)

    por_id = {e.documento_id: e for e in segunda.manifesto}
    assert por_id["copia"].status == StatusDocumento.REJEITADO
    assert "duplicata exata" in por_id["copia"].motivo


def test_limite_novos_documentos_respeita_o_teto_e_permite_retomar(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    for i in range(5):
        _escrever_doc(corpus_dir, f"doc{i}", f"Documento numero {i} do lote de teste de limite incremental de ingestao, com texto proprio e "
            f"longo o suficiente para passar pelo filtro minimo de qualidade de duzentos caracteres do Ignis.")

    primeiro_lote = processar_diretorio(corpus_dir, limite_novos_documentos=2)
    assert primeiro_lote.n_novos_processados == 2
    assert len(primeiro_lote.manifesto) == 2

    segundo_lote = processar_diretorio(
        corpus_dir, manifesto_existente=primeiro_lote.manifesto, limite_novos_documentos=2
    )
    assert segundo_lote.n_novos_processados == 2
    assert len(segundo_lote.manifesto) == 4  # 2 do primeiro lote + 2 novos

    terceiro_lote = processar_diretorio(
        corpus_dir, manifesto_existente=segundo_lote.manifesto, limite_novos_documentos=2
    )
    assert terceiro_lote.n_novos_processados == 1  # so sobrou 1 documento
    assert len(terceiro_lote.manifesto) == 5  # todos os 5 presentes ao final

    ids_finais = {e.documento_id for e in terceiro_lote.manifesto}
    assert ids_finais == {f"doc{i}" for i in range(5)}


def test_sem_manifesto_existente_comportamento_identico_ao_anterior_a_sprint5(tmp_path: Path):
    """Regressao: sem `manifesto_existente` (uso padrao, como em todas as
    chamadas pre-Sprint-5), `n_novos_processados` deve ser igual ao total de
    documentos brutos ingeridos -- nenhuma mudanca de comportamento."""
    corpus_dir = tmp_path / "corpus"
    for i in range(3):
        _escrever_doc(corpus_dir, f"doc{i}", f"Documento {i} para o teste de regressao de comportamento padrao do pipeline, com texto longo o "
            f"suficiente para passar pelo filtro minimo de qualidade de duzentos caracteres do Ignis.")

    resultado = processar_diretorio(corpus_dir)
    assert resultado.n_novos_processados == 3
    assert len(resultado.manifesto) == 3


def test_quase_duplicata_entre_lotes_fica_no_mesmo_grupo(tmp_path):
    import json
    from ignis.data.pipeline import processar_diretorio
    base = " ".join(f"palavra{i}" for i in range(150))
    caminho = tmp_path / "shard.jsonl"
    def registro(id_, texto):
        return json.dumps(dict(id=id_, texto=texto, licenca="CC-BY-4.0", permissao_treinamento=True, unidade_agrupamento=id_))
    caminho.write_text(registro("a", base) + "\n", encoding="utf-8")
    primeiro = processar_diretorio(tmp_path)
    with caminho.open("a", encoding="utf-8") as arquivo:
        arquivo.write(registro("b", base + " Acrescentado um detalhe final.") + "\n")
    segundo = processar_diretorio(tmp_path, manifesto_existente=primeiro.manifesto)
    assert segundo.n_novos_processados == 1
    assert len({e.grupo_quase_duplicata for e in segundo.manifesto}) == 1
