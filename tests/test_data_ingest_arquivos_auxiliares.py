"""Testes de regressao para o bug real da Sprint 6.1: um arquivo auxiliar
(`_metadados.jsonl`, escrito por `scripts/coletar_wikipedia_pt.py` ao lado
dos shards de conteudo) estava sendo lido por `ingerir_diretorio` como se
cada linha fosse um documento, e falhava com "nao possui a chave
obrigatoria 'texto'" (linhas de metadados tem `titulo`/`url`/`revisao`, nao
`texto`). Corrigido em `ignis.data.ingest`: qualquer arquivo cujo NOME
comece com `_` e tratado como auxiliar e nunca lido como conteudo."""

import json
from pathlib import Path

import pytest

from ignis.data.ingest import eh_arquivo_auxiliar, ingerir_arquivo, ingerir_diretorio
from ignis.data.manifest import StatusDocumento
from ignis.data.pipeline import processar_diretorio


def _texto_longo(sufixo: str = "") -> str:
    return ("Este e um texto suficientemente longo para passar nos filtros de qualidade. " * 10) + sufixo


def test_eh_arquivo_auxiliar_reconhece_prefixo_underscore():
    assert eh_arquivo_auxiliar(Path("/qualquer/pasta/_metadados.jsonl"))
    assert eh_arquivo_auxiliar(Path("_estado_coleta.json"))
    assert not eh_arquivo_auxiliar(Path("/qualquer/pasta/shard_0000.jsonl"))
    assert not eh_arquivo_auxiliar(Path("documento.txt"))


def test_ingerir_diretorio_ignora_metadados_jsonl_mesmo_malformado(tmp_path: Path):
    """O caso exato do bug real: _metadados.jsonl tem linhas sem a chave
    'texto' -- antes da correcao, isso levantava ValueError. Agora o
    arquivo e simplesmente ignorado por ingerir_diretorio."""
    fonte = tmp_path / "wikipedia_pt"
    fonte.mkdir()
    (fonte / "shard_0000.jsonl").write_text(
        json.dumps({"id": "wikipedia-pt-1", "texto": _texto_longo(), "licenca": "CC BY-SA 4.0", "permissao_treinamento": True}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    (fonte / "_metadados.jsonl").write_text(
        json.dumps({"id": "wikipedia-pt-1", "pageid": 1, "titulo": "Exemplo", "url": "https://pt.wikipedia.org/wiki/Exemplo"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    documentos = ingerir_diretorio(fonte)
    assert len(documentos) == 1
    assert documentos[0].documento_id == "wikipedia-pt-1"


def test_ingerir_arquivo_em_arquivo_auxiliar_explicito_tambem_e_bloqueado(tmp_path: Path):
    """Chamar ingerir_arquivo diretamente em um arquivo auxiliar deve ser
    evitado pelo chamador -- mas documentamos aqui que ingerir_diretorio
    (o caminho real usado pela CLI) nunca chega a chamar ingerir_arquivo
    para esses arquivos, que e a protecao que importa na pratica."""
    fonte = tmp_path / "fonte"
    fonte.mkdir()
    auxiliar = fonte / "_metadados.jsonl"
    auxiliar.write_text(json.dumps({"id": "x", "titulo": "sem texto"}) + "\n", encoding="utf-8")

    # ingerir_arquivo sozinho nao sabe que e auxiliar (a checagem fica em
    # ingerir_diretorio); confirma que SEM a protecao de ingerir_diretorio
    # o arquivo realmente falharia, o que mostra por que a correcao tinha
    # que ficar no nivel do diretorio.
    with pytest.raises(ValueError, match="texto"):
        ingerir_arquivo(auxiliar)


def test_processar_diretorio_end_to_end_com_metadados_ao_lado(tmp_path: Path):
    """Regressao de ponta a ponta no nivel do pipeline (o caminho real
    usado por 'ignis.cli validate-corpus'): um diretorio no formato
    exato produzido por coletar_wikipedia_pt.py (shards + _metadados.jsonl
    + _estado_coleta.json) deve ser ingerido sem erro, processando so os
    shards."""
    corpus_dir = tmp_path / "wikipedia_pt"
    corpus_dir.mkdir()
    for i in range(3):
        linha = {
            "id": f"wikipedia-pt-{i}",
            "texto": _texto_longo(f" artigo {i}"),
            "licenca": "CC BY-SA 4.0 + GFDL (Wikimedia Foundation Terms of Use)",
            "permissao_treinamento": True,
            "unidade_agrupamento": f"wikipedia-pt-{i}",
        }
        (corpus_dir / "shard_0000.jsonl").open("a", encoding="utf-8").write(json.dumps(linha, ensure_ascii=False) + "\n")
    metadados = corpus_dir / "_metadados.jsonl"
    for i in range(3):
        metadados.open("a", encoding="utf-8").write(
            json.dumps({"id": f"wikipedia-pt-{i}", "pageid": i, "titulo": f"Artigo {i}"}, ensure_ascii=False) + "\n"
        )
    (corpus_dir / "_estado_coleta.json").write_text(
        json.dumps({"pageids_coletados": [0, 1, 2], "bytes_baixados": 123, "tokens_aproximados": 456}), encoding="utf-8"
    )

    resultado = processar_diretorio(corpus_dir, fonte="wikipedia-pt")
    assert len(resultado.manifesto) == 3
    assert all(e.status == StatusDocumento.ACEITO for e in resultado.manifesto)
    # cada artigo e sua propria unidade de agrupamento (ver Sprint 5) --
    # nunca colapsado em uma unica unidade para um provedor grande.
    unidades = {e.unidade_agrupamento for e in resultado.manifesto}
    assert len(unidades) == 3
