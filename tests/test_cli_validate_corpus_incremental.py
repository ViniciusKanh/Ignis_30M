"""Testes do subcomando `validate-corpus --incremental`/`--limite-novos-documentos`
do CLI -- a camada fina sobre `processar_diretorio` (testada em
tests/test_data_pipeline_incremental.py) que persiste o manifesto em disco
entre chamadas."""

import argparse
import json
from pathlib import Path

from ignis.cli import cmd_validate_corpus
from ignis.data.pipeline import carregar_manifesto


def _args(corpus_dir: Path, manifesto_saida: Path, incremental: bool = False, limite: int | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        corpus_dir=str(corpus_dir),
        fonte=None,
        manifesto_saida=str(manifesto_saida),
        incremental=incremental,
        limite_novos_documentos=limite,
    )


def _escrever_doc(pasta: Path, nome: str, extra: str = "") -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    texto = (
        f"Documento {nome} do teste de validate-corpus incremental, com texto longo o suficiente "
        f"para passar pelo filtro minimo de qualidade de duzentos caracteres do Ignis. {extra}"
    )
    (pasta / f"{nome}.txt").write_text(texto, encoding="utf-8")


def test_validate_corpus_incremental_persiste_entre_chamadas(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    manifesto_path = tmp_path / "manifesto.jsonl"
    _escrever_doc(corpus_dir, "doc1")

    cmd_validate_corpus(_args(corpus_dir, manifesto_path, incremental=True))
    assert len(carregar_manifesto(manifesto_path)) == 1

    _escrever_doc(corpus_dir, "doc2")
    cmd_validate_corpus(_args(corpus_dir, manifesto_path, incremental=True))
    manifesto_final = carregar_manifesto(manifesto_path)
    assert {e.documento_id for e in manifesto_final} == {"doc1", "doc2"}


def test_validate_corpus_sem_incremental_reprocessa_do_zero(tmp_path: Path, capsys):
    corpus_dir = tmp_path / "corpus"
    manifesto_path = tmp_path / "manifesto.jsonl"
    _escrever_doc(corpus_dir, "doc1")
    cmd_validate_corpus(_args(corpus_dir, manifesto_path, incremental=False))

    saida = capsys.readouterr().out
    relatorio = json.loads(saida.split("\n\n")[0])
    assert relatorio["novos_processados_nesta_execucao"] == 1


def test_validate_corpus_limite_novos_documentos_processa_em_lotes(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    manifesto_path = tmp_path / "manifesto.jsonl"
    for i in range(3):
        _escrever_doc(corpus_dir, f"doc{i}")

    cmd_validate_corpus(_args(corpus_dir, manifesto_path, incremental=True, limite=1))
    assert len(carregar_manifesto(manifesto_path)) == 1

    cmd_validate_corpus(_args(corpus_dir, manifesto_path, incremental=True, limite=1))
    assert len(carregar_manifesto(manifesto_path)) == 2

    cmd_validate_corpus(_args(corpus_dir, manifesto_path, incremental=True, limite=1))
    assert len(carregar_manifesto(manifesto_path)) == 3
