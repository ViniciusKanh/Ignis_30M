"""Testes de `ignis.cli audit-tokenizer` -- o portao mecanico que bloqueia a
promocao de um corpus/tokenizador a "oficial do piloto Ignis-30M" enquanto
os 4 criterios objetivos da secao 8 de docs/plano-corpus-piloto.md nao forem
TODOS atendidos simultaneamente (licencas documentadas, splits sem
contaminacao, round-trip sem falhas em treino/validacao/externo, e volume de
tokens de treino na faixa alvo). Torch-free: `audit-tokenizer` so usa
`ignis.data` e `ignis.tokenizer`, nunca `ignis.model`/torch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ignis.cli import (
    cmd_audit_tokenizer,
    cmd_prepare_dataset,
    cmd_prepare_splits,
    cmd_train_tokenizer,
    cmd_validate_corpus,
)
from ignis.data.manifest import StatusDocumento
from ignis.data.pipeline import carregar_manifesto, salvar_manifesto
from ignis.data.splits import carregar_splits, salvar_splits

_TEMAS = [
    (
        "Transformers",
        "A arquitetura Transformer substitui recorrencia por atencao, permitindo capturar "
        "dependencias de longo alcance em paralelo durante o treinamento de modelos de linguagem.",
    ),
    (
        "Diversidade do Brasil",
        "O Brasil possui grande diversidade linguistica regional, com variacoes de vocabulario "
        "e sotaque entre o sul, o nordeste, o sudeste, o norte e o centro-oeste do pais.",
    ),
    (
        "Reprodutibilidade",
        "Reprodutibilidade em experimentos de aprendizado de maquina exige registrar sementes "
        "aleatorias, versoes de bibliotecas, hardware utilizado e hiperparametros do experimento.",
    ),
    (
        "Tokenizacao",
        "Um tokenizador BPE constroi seu vocabulario a partir de pares de bytes frequentes no "
        "corpus de treino, permitindo representar qualquer texto sem depender de um dicionario fixo.",
    ),
    (
        "Avaliacao de modelos",
        "A avaliacao de modelos de linguagem combina metricas quantitativas como perplexidade "
        "com inspecao qualitativa de amostras geradas, evitando conclusoes precipitadas.",
    ),
]


def _montar_pipeline_completo(tmp_path: Path, vocab_size: int = 96) -> dict:
    """Gera um corpus sintetico diverso (5 fontes/temas distintos, 2
    documentos cada), roda validate-corpus -> prepare-splits ->
    train-tokenizer -> prepare-dataset, e devolve os caminhos relevantes
    prontos para `audit-tokenizer`."""
    corpus_dir = tmp_path / "corpus"
    for i_fonte, (titulo, base) in enumerate(_TEMAS):
        pasta = corpus_dir / f"fonte_{i_fonte}"
        pasta.mkdir(parents=True)
        for i_doc in range(2):
            texto = (
                base
                + f" Este e o paragrafo numero {i_doc} sobre '{titulo}', com uma frase final "
                "distinta para nao ser confundido com quase-duplicata de outro documento."
            )
            (pasta / f"doc{i_fonte}_{i_doc}.txt").write_text(texto, encoding="utf-8")

    manifesto_path = tmp_path / "manifesto.jsonl"
    cmd_validate_corpus(
        argparse.Namespace(corpus_dir=str(corpus_dir), fonte=None, manifesto_saida=str(manifesto_path))
    )
    # validate-corpus so aceita documentos com licenca/permissao declaradas;
    # arquivos .txt nao carregam essas chaves, entao marcamos manualmente
    # aqui como se tivessem sido ingeridos via JSONL com essas chaves.
    manifesto = carregar_manifesto(manifesto_path)
    for entrada in manifesto:
        if entrada.status == StatusDocumento.LICENCA_DESCONHECIDA:
            entrada.licenca = "CC-BY-4.0"
            entrada.permissao_treinamento = True
            entrada.status = StatusDocumento.ACEITO
            entrada.motivo = "licenca e permissao atribuidas no teste"
    salvar_manifesto(type("R", (), {"manifesto": manifesto})(), manifesto_path)

    splits_path = tmp_path / "splits.json"
    cmd_prepare_splits(
        argparse.Namespace(
            manifesto=str(manifesto_path),
            splits_saida=str(splits_path),
            seed=42,
            treino=0.6,
            validacao=0.2,
            teste=0.2,
            ignorar_fonte=False,
        )
    )

    tokenizer_path = tmp_path / "tokenizer.json"
    cmd_train_tokenizer(
        argparse.Namespace(
            corpus_dir=str(corpus_dir),
            manifesto=str(manifesto_path),
            splits=str(splits_path),
            vocab_size=vocab_size,
            min_frequency=1,
            saida=str(tokenizer_path),
            sobrescrever=True,
        )
    )

    dataset_base = tmp_path / "tokenizado"
    cmd_prepare_dataset(
        argparse.Namespace(
            tokenizer=str(tokenizer_path),
            corpus_dir=str(corpus_dir),
            manifesto=str(manifesto_path),
            splits=str(splits_path),
            saida_base=str(dataset_base),
            seq_len=16,
        )
    )

    textos_externos_path = tmp_path / "textos_externos.txt"
    textos_externos_path.write_text(
        "Este e um texto externo nunca usado em nenhum split do corpus de teste.\n"
        "Outra frase externa, tambem fora de qualquer split, para avaliar o vocabulario.\n",
        encoding="utf-8",
    )

    return {
        "corpus_dir": corpus_dir,
        "manifesto_path": manifesto_path,
        "splits_path": splits_path,
        "tokenizer_path": tokenizer_path,
        "dataset_base": dataset_base,
        "textos_externos_path": textos_externos_path,
    }


def _args_audit(pipeline: dict, saida: Path, **overrides) -> argparse.Namespace:
    base = dict(
        tokenizer=str(pipeline["tokenizer_path"]),
        corpus_dir=str(pipeline["corpus_dir"]),
        manifesto=str(pipeline["manifesto_path"]),
        splits=str(pipeline["splits_path"]),
        textos_externos=str(pipeline["textos_externos_path"]),
        dataset_base=str(pipeline["dataset_base"]),
        volume_min_tokens=1_000_000,
        volume_max_tokens=5_000_000,
        saida=str(saida),
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_corpus_pequeno_falha_em_volume_e_vocabulario(tmp_path: Path):
    """Com um corpus de teste minusculo, todos os criterios objetivos devem
    passar EXCETO o volume de tokens (muito abaixo da faixa alvo de
    1-5M) -- o que deve bloquear `oficial_do_piloto`, sem mascarar os
    demais criterios como tambem reprovados."""
    pipeline = _montar_pipeline_completo(tmp_path)
    saida = tmp_path / "auditoria.json"

    codigo = cmd_audit_tokenizer(_args_audit(pipeline, saida))

    relatorio = json.loads(saida.read_text(encoding="utf-8"))
    criterios = {c["nome"]: c["ok"] for c in relatorio["criterios"]}

    assert criterios["vocabulario_compativel"] is False
    assert criterios["cobertura_dos_splits"] is True
    assert criterios["licencas_documentadas"] is True
    assert criterios["splits_sem_contaminacao"] is True
    assert criterios["round_trip_sem_falhas:treino"] is True
    assert criterios["round_trip_sem_falhas:validacao"] is True
    assert criterios["round_trip_sem_falhas:externo"] is True
    assert criterios["volume_de_tokens_na_faixa_alvo"] is False
    assert relatorio["oficial_do_piloto"] is False
    assert codigo == 1


def test_sem_textos_externos_bloqueia_mesmo_que_o_resto_passe(tmp_path: Path):
    pipeline = _montar_pipeline_completo(tmp_path)
    saida = tmp_path / "auditoria_sem_externo.json"

    cmd_audit_tokenizer(_args_audit(pipeline, saida, textos_externos=None))

    relatorio = json.loads(saida.read_text(encoding="utf-8"))
    criterios = {c["nome"]: c["ok"] for c in relatorio["criterios"]}
    assert criterios["round_trip_sem_falhas:externo"] is False
    assert relatorio["oficial_do_piloto"] is False


def test_sem_estatisticas_de_prepare_dataset_bloqueia_volume(tmp_path: Path):
    pipeline = _montar_pipeline_completo(tmp_path)
    saida = tmp_path / "auditoria_sem_stats.json"

    cmd_audit_tokenizer(_args_audit(pipeline, saida, dataset_base=str(tmp_path / "nao-existe" / "tokenizado")))

    relatorio = json.loads(saida.read_text(encoding="utf-8"))
    criterios = {c["nome"]: c["ok"] for c in relatorio["criterios"]}
    assert criterios["volume_de_tokens_na_faixa_alvo"] is False


def test_licenca_desconhecida_pendente_bloqueia_mesmo_com_resto_ok(tmp_path: Path):
    pipeline = _montar_pipeline_completo(tmp_path)

    manifesto = carregar_manifesto(pipeline["manifesto_path"])
    # Introduz um documento adicional com licenca desconhecida (nao afeta
    # splits/tokenizador ja gerados, apenas o criterio de licencas).
    from ignis.data.manifest import EntradaManifesto

    manifesto.append(
        EntradaManifesto(
            documento_id="pendente-1",
            caminho_origem="/fake/pendente.txt",
            fonte="fonte_pendente",
            sha256="hash-pendente",
            tamanho_caracteres=100,
            status=StatusDocumento.LICENCA_DESCONHECIDA,
            motivo="licenca nao declarada",
        )
    )
    salvar_manifesto(type("R", (), {"manifesto": manifesto})(), pipeline["manifesto_path"])

    saida = tmp_path / "auditoria_licenca.json"
    cmd_audit_tokenizer(_args_audit(pipeline, saida))

    relatorio = json.loads(saida.read_text(encoding="utf-8"))
    criterios = {c["nome"]: c["ok"] for c in relatorio["criterios"]}
    assert criterios["licencas_documentadas"] is False
    assert relatorio["oficial_do_piloto"] is False


def test_contaminacao_entre_splits_bloqueia_mesmo_com_resto_ok(tmp_path: Path):
    pipeline = _montar_pipeline_completo(tmp_path)

    splits = carregar_splits(pipeline["splits_path"])
    if splits.treino and splits.validacao:
        # Forca contaminacao: duplica o primeiro id de treino tambem em validacao.
        doc_contaminado = splits.treino[0]
        splits_contaminados = splits.model_copy(update={"validacao": [*splits.validacao, doc_contaminado]})
        salvar_splits(splits_contaminados, pipeline["splits_path"])

        saida = tmp_path / "auditoria_contaminacao.json"
        cmd_audit_tokenizer(_args_audit(pipeline, saida))

        relatorio = json.loads(saida.read_text(encoding="utf-8"))
        criterios = {c["nome"]: c["ok"] for c in relatorio["criterios"]}
        assert criterios["splits_sem_contaminacao"] is False
        assert relatorio["oficial_do_piloto"] is False


def test_id_inexistente_no_split_bloqueia_auditoria(tmp_path):
    pipeline = _montar_pipeline_completo(tmp_path)
    splits = carregar_splits(pipeline["splits_path"])
    splits.validacao = ["documento-inexistente"]
    salvar_splits(splits, pipeline["splits_path"])
    saida = tmp_path / "auditoria.json"
    assert cmd_audit_tokenizer(_args_audit(pipeline, saida)) == 1
    criterios = {c["nome"]: c["ok"] for c in json.loads(saida.read_text(encoding="utf-8"))["criterios"]}
    assert not criterios["cobertura_dos_splits"]
    assert not criterios["round_trip_sem_falhas:validacao"]
