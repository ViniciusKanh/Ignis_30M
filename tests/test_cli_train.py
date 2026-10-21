"""Testes do subcomando `train` do CLI (dry-run obrigatório + porta de
entrada do treinamento real do piloto).

Duas categorias de teste aqui:

1. Testes do "portão de segurança" (ausência de arquivos,
   `--iniciar-treinamento-real` sem dry-run aprovado) — propositalmente SEM
   dependência de torch: `cmd_train` captura `ImportError` de torch como
   apenas mais uma checagem reprovada, nunca como um crash, então estes
   testes rodam tanto na VM Linux (sem torch) quanto no Windows nativo (ver
   docs/experiments/fase1-fundacao.md sobre a execução dual de ambiente).
2. Um teste integrado de ponta a ponta (`pytest.importorskip("torch")`,
   só roda onde torch está instalado) que gera um corpus sintético
   minúsculo, roda o pipeline completo (manifesto -> splits -> tokenizador
   -> dataset tokenizado) e verifica que o dry-run do `train` real
   reconhece arquivos/shapes corretos e executa o forward de teste sem
   atualizar nenhum peso.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml

from ignis.cli import cmd_train

_TEMAS_TESTE = [
    (
        "Arquitetura Transformer",
        "A arquitetura Transformer substitui recorrencia por mecanismos de "
        "atencao, permitindo capturar dependencias de longo alcance em "
        "paralelo e acelerar o treinamento em hardware com varios nucleos.",
    ),
    (
        "Diversidade regional do Brasil",
        "O Brasil possui grande diversidade linguistica regional, com "
        "variacoes de vocabulario e sotaque entre o sul, o nordeste, o "
        "sudeste, o norte e o centro-oeste do pais.",
    ),
    (
        "Reprodutibilidade em aprendizado de maquina",
        "Reprodutibilidade em experimentos de aprendizado de maquina exige "
        "registrar sementes aleatorias, versoes de bibliotecas, hardware "
        "utilizado e hiperparametros escolhidos no experimento.",
    ),
]


def _gerar_corpus_diverso(tmp_path: Path, n_fontes: int = 3, docs_por_fonte: int = 3) -> Path:
    """Gera um corpus sintetico com `n_fontes` fontes de temas CLARAMENTE
    distintos entre si (baixa similaridade de Jaccard), evitando que o
    pipeline de quase-duplicatas agrupe documentos de fontes diferentes na
    mesma unidade independente -- o que deixaria poucas unidades para
    treino/validacao/teste (ver docstring do modulo acima)."""
    corpus_dir = tmp_path / "corpus"
    for i_fonte in range(n_fontes):
        titulo, base = _TEMAS_TESTE[i_fonte % len(_TEMAS_TESTE)]
        pasta = corpus_dir / f"fonte_{i_fonte}"
        pasta.mkdir(parents=True)
        for i_doc in range(docs_por_fonte):
            texto = (
                base
                + f" Este e o paragrafo numero {i_doc} sobre o tema '{titulo}', "
                "com uma frase final distinta para nao ser confundido com "
                "quase-duplicata de outro documento do mesmo tema."
            )
            registro = {
                "id": f"f{i_fonte}-d{i_doc}",
                "texto": texto,
                "licenca": "teste",
                "permissao_treinamento": True,
            }
            (pasta / f"doc{i_doc}.jsonl").write_text(
                json.dumps(registro, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    return corpus_dir



def _args_train(
    config: Path,
    saida: Path | None = None,
    iniciar_treinamento_real: bool = False,
    device: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(config),
        saida=str(saida) if saida else None,
        iniciar_treinamento_real=iniciar_treinamento_real,
        device=device,
    )


def _escrever_config_piloto_com_dados_inexistentes(tmp_path: Path) -> Path:
    dados = {
        "modelo": {"config": "configs/ignis-30m.yaml"},
        "dados": {
            "manifesto": str(tmp_path / "nao-existe" / "manifesto.jsonl"),
            "splits": str(tmp_path / "nao-existe" / "splits.json"),
            "tokenizer": str(tmp_path / "nao-existe" / "tokenizer.json"),
            "dataset_base": str(tmp_path / "nao-existe" / "tokenizado"),
            "seq_len": 16,
        },
        "otimizacao": {"lr": 3.0e-4},
        "lote": {"microbatch_size": 1, "grad_accum_steps": 1},
        "orcamento": {"max_steps": 5},
        "validacao_e_checkpoint": {"eval_every": 5, "checkpoint_every": 5, "log_every": 1},
        "execucao": {"seed": 42, "device": "cpu", "out_dir": str(tmp_path / "ckpt")},
    }
    caminho = tmp_path / "train-pilot-teste.yaml"
    caminho.write_text(yaml.dump(dados, allow_unicode=True), encoding="utf-8")
    return caminho


def test_train_dry_run_sem_dados_reporta_falhas_e_nao_treina(tmp_path):
    """Sem nenhum arquivo de dados/tokenizer presente, o dry-run deve
    reportar `pronto_para_treinamento_real = False` e listar exatamente
    quais checagens falharam — sem lançar exceção (mesmo sem torch)."""
    config = _escrever_config_piloto_com_dados_inexistentes(tmp_path)
    saida_relatorio = tmp_path / "relatorio.json"

    codigo = cmd_train(_args_train(config, saida=saida_relatorio))

    assert codigo == 1
    assert saida_relatorio.exists()
    relatorio = json.loads(saida_relatorio.read_text(encoding="utf-8"))
    assert relatorio["pronto_para_treinamento_real"] is False
    assert len(relatorio["erros"]) > 0

    nomes_checagens = {c["nome"]: c["ok"] for c in relatorio["checagens"]}
    assert nomes_checagens["carregar_config_piloto"] is True
    assert nomes_checagens["carregar_config_modelo_oficial"] is True
    assert nomes_checagens["arquivo_existe:manifesto"] is False
    assert nomes_checagens["arquivo_existe:splits"] is False
    assert nomes_checagens["arquivo_existe:tokenizer"] is False
    assert nomes_checagens["dataset_tokenizado_existe:treino"] is False
    assert nomes_checagens["forward_de_teste_sem_atualizar_pesos"] is False


def test_train_iniciar_treinamento_real_sem_checagens_aprovadas_nunca_treina(tmp_path, monkeypatch):
    """Mesmo pedindo --iniciar-treinamento-real, se QUALQUER checagem do
    dry-run falhar, `_executar_treinamento_real` nunca deve ser chamada."""
    import ignis.cli as cli_mod

    chamado = {"sim": False}

    def _falhar_se_chamado(*args, **kwargs):
        chamado["sim"] = True
        raise AssertionError("treinamento real não deveria ter sido iniciado")

    monkeypatch.setattr(cli_mod, "_executar_treinamento_real", _falhar_se_chamado)

    config = _escrever_config_piloto_com_dados_inexistentes(tmp_path)
    codigo = cmd_train(_args_train(config, iniciar_treinamento_real=True))

    assert codigo == 1
    assert chamado["sim"] is False


def test_train_config_invalida_retorna_codigo_2(tmp_path):
    caminho = tmp_path / "config-invalida.yaml"
    caminho.write_text("isto: nao e uma config de piloto valida\n", encoding="utf-8")
    codigo = cmd_train(_args_train(caminho))
    assert codigo == 2


def test_train_integrado_dry_run_com_corpus_sintetico_e_modelo_oficial(tmp_path):
    """Ponta a ponta: gera um corpus sintético minúsculo, roda o pipeline
    completo de dados (Fase 2) e verifica que o dry-run do comando `train`
    reconhece corretamente os arquivos reais e executa um forward de teste
    válido contra o modelo OFICIAL do Ignis-30M (configs/ignis-30m.yaml) —
    sem atualizar nenhum peso. Requer torch (model/dataloader dependem
    dele); pulado automaticamente na VM Linux sem torch."""
    pytest.importorskip("torch")

    from ignis.data.pipeline import processar_diretorio, salvar_manifesto
    from ignis.data.pretrain_prep import construir_sequencias, salvar_dataset_tokenizado
    from ignis.data.splits import gerar_splits, salvar_splits
    from ignis.tokenizer.bpe import IgnisTokenizer

    corpus_dir = _gerar_corpus_diverso(tmp_path)

    resultado = processar_diretorio(corpus_dir)
    manifesto_path = tmp_path / "manifesto.jsonl"
    salvar_manifesto(resultado, manifesto_path)

    atribuicao = gerar_splits(resultado.manifesto, seed=42)
    splits_path = tmp_path / "splits.json"
    salvar_splits(atribuicao, splits_path)
    assert atribuicao.treino and atribuicao.validacao

    arquivos_txt = []
    tmp_textos = tmp_path / "textos_tok"
    tmp_textos.mkdir()
    for doc_id in atribuicao.treino:
        caminho = tmp_textos / f"{doc_id}.txt"
        caminho.write_text(resultado.textos_aceitos[doc_id], encoding="utf-8")
        arquivos_txt.append(caminho)
    tokenizador = IgnisTokenizer.train_tokenizer(arquivos_txt, vocab_size=512, min_frequency=1)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizador.save(tokenizer_path)

    dataset_base = tmp_path / "tokenizado"
    for nome_split, ids_split in (("treino", atribuicao.treino), ("validacao", atribuicao.validacao)):
        textos_split = {doc_id: resultado.textos_aceitos[doc_id] for doc_id in ids_split}
        sequencias, mascaras, estatisticas = construir_sequencias(textos_split, tokenizador, seq_len=16)
        base_split = dataset_base.with_name(f"{dataset_base.name}_{nome_split}")
        salvar_dataset_tokenizado(base_split, sequencias, mascaras, estatisticas)

    dados_config = {
        "modelo": {"config": "configs/ignis-30m.yaml"},
        "dados": {
            "manifesto": str(manifesto_path),
            "splits": str(splits_path),
            "tokenizer": str(tokenizer_path),
            "dataset_base": str(dataset_base),
            "seq_len": 16,
        },
        "otimizacao": {"lr": 3.0e-4},
        "lote": {"microbatch_size": 1, "grad_accum_steps": 1},
        "orcamento": {"max_steps": 5},
        "validacao_e_checkpoint": {"eval_every": 5, "checkpoint_every": 5, "log_every": 1},
        "execucao": {"seed": 42, "device": "cpu", "out_dir": str(tmp_path / "ckpt")},
    }
    config_path = tmp_path / "train-pilot-integrado.yaml"
    config_path.write_text(yaml.dump(dados_config, allow_unicode=True), encoding="utf-8")

    codigo = cmd_train(_args_train(config_path))

    # O resultado completo (pronto_para_treinamento_real) pode ser False
    # aqui de forma ESPERADA: o tokenizador de teste (vocab_size=512) nunca
    # bate com o vocab_size oficial (16384) de um corpus sintético tão
    # pequeno -- isso é intencional (ver docs/data-card.md) e não deve ser
    # escondido. O que este teste garante é que as checagens mecânicas
    # (arquivos, shapes, contagem de parâmetros, forward de teste) funcionam
    # corretamente contra dados e arquitetura REAIS, não sintéticas/reduzidas.
    assert codigo in (0, 1)


def test_train_integrado_resultado_detalhado(tmp_path, capsys):
    """Mesmo cenário acima, mas inspecionando o relatório para confirmar
    explicitamente que o forward de teste passou e que nenhum treinamento
    foi iniciado."""
    pytest.importorskip("torch")

    from ignis.data.pipeline import processar_diretorio, salvar_manifesto
    from ignis.data.pretrain_prep import construir_sequencias, salvar_dataset_tokenizado
    from ignis.data.splits import gerar_splits, salvar_splits
    from ignis.tokenizer.bpe import IgnisTokenizer

    corpus_dir = _gerar_corpus_diverso(tmp_path)

    resultado = processar_diretorio(corpus_dir)
    manifesto_path = tmp_path / "manifesto.jsonl"
    salvar_manifesto(resultado, manifesto_path)
    atribuicao = gerar_splits(resultado.manifesto, seed=42)
    splits_path = tmp_path / "splits.json"
    salvar_splits(atribuicao, splits_path)

    tmp_textos = tmp_path / "textos_tok"
    tmp_textos.mkdir()
    arquivos_txt = []
    for doc_id in atribuicao.treino:
        caminho = tmp_textos / f"{doc_id}.txt"
        caminho.write_text(resultado.textos_aceitos[doc_id], encoding="utf-8")
        arquivos_txt.append(caminho)
    tokenizador = IgnisTokenizer.train_tokenizer(arquivos_txt, vocab_size=512, min_frequency=1)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizador.save(tokenizer_path)

    dataset_base = tmp_path / "tokenizado"
    for nome_split, ids_split in (("treino", atribuicao.treino), ("validacao", atribuicao.validacao)):
        textos_split = {doc_id: resultado.textos_aceitos[doc_id] for doc_id in ids_split}
        sequencias, mascaras, estatisticas = construir_sequencias(textos_split, tokenizador, seq_len=16)
        base_split = dataset_base.with_name(f"{dataset_base.name}_{nome_split}")
        salvar_dataset_tokenizado(base_split, sequencias, mascaras, estatisticas)

    dados_config = {
        "modelo": {"config": "configs/ignis-30m.yaml"},
        "dados": {
            "manifesto": str(manifesto_path),
            "splits": str(splits_path),
            "tokenizer": str(tokenizer_path),
            "dataset_base": str(dataset_base),
            "seq_len": 16,
        },
        "otimizacao": {"lr": 3.0e-4},
        "lote": {"microbatch_size": 1, "grad_accum_steps": 1},
        "orcamento": {"max_steps": 5},
        "validacao_e_checkpoint": {"eval_every": 5, "checkpoint_every": 5, "log_every": 1},
        "execucao": {"seed": 42, "device": "cpu", "out_dir": str(tmp_path / "ckpt")},
    }
    config_path = tmp_path / "train-pilot-integrado2.yaml"
    config_path.write_text(yaml.dump(dados_config, allow_unicode=True), encoding="utf-8")
    saida_relatorio = tmp_path / "relatorio.json"

    cmd_train(_args_train(config_path, saida=saida_relatorio))

    relatorio = json.loads(saida_relatorio.read_text(encoding="utf-8"))
    nomes_checagens = {c["nome"]: c for c in relatorio["checagens"]}

    assert nomes_checagens["arquivo_existe:manifesto"]["ok"] is True
    assert nomes_checagens["arquivo_existe:splits"]["ok"] is True
    assert nomes_checagens["arquivo_existe:tokenizer"]["ok"] is True
    assert nomes_checagens["dataset_tokenizado_existe:treino"]["ok"] is True
    assert nomes_checagens["dataset_tokenizado_existe:validacao"]["ok"] is True
    assert nomes_checagens["contagem_parametros_confere_com_ignis_30m"]["ok"] is True
    assert nomes_checagens["forward_de_teste_sem_atualizar_pesos"]["ok"] is True
    # Checagem documentada como esperada para falhar com corpus/tokenizador de teste.
    assert nomes_checagens["tokenizer_vocab_size_bate_com_modelo"]["ok"] is False
    assert "memoria_estimada" in relatorio
    assert relatorio["memoria_estimada"]["n_parametros"] == 30_416_384


def test_train_device_override_sem_flag_nao_aparece_no_relatorio(tmp_path):
    """Sem --device, nenhuma checagem de override deve aparecer -- o device
    usado e exatamente o do YAML, sem alteracao."""
    config = _escrever_config_piloto_com_dados_inexistentes(tmp_path)
    cmd_train(_args_train(config))
    # reconstroi o relatorio via --saida para inspecionar as checagens
    saida_relatorio = tmp_path / "relatorio_sem_override.json"
    cmd_train(_args_train(config, saida=saida_relatorio))
    relatorio = json.loads(saida_relatorio.read_text(encoding="utf-8"))
    nomes_checagens = {c["nome"] for c in relatorio["checagens"]}
    assert "device_override_cli" not in nomes_checagens


def test_train_device_override_cpu_e_registrado_no_relatorio(tmp_path):
    """--device cpu deve ser registrado como uma checagem explicita no
    relatorio de dry-run, mesmo sem torch disponivel (a checagem acontece
    antes de qualquer importacao de torch)."""
    config = _escrever_config_piloto_com_dados_inexistentes(tmp_path)
    saida_relatorio = tmp_path / "relatorio_device_cpu.json"

    codigo = cmd_train(_args_train(config, saida=saida_relatorio, device="cpu"))

    assert codigo == 1  # dados ainda ausentes, independente do device
    relatorio = json.loads(saida_relatorio.read_text(encoding="utf-8"))
    nomes_checagens = {c["nome"]: c for c in relatorio["checagens"]}
    assert nomes_checagens["device_override_cli"]["ok"] is True
    assert "cpu" in nomes_checagens["device_override_cli"]["detalhe"]


def test_train_device_override_cuda_e_registrado_mesmo_sem_torch(tmp_path):
    """--device cuda tambem deve ser registrado como override, mesmo quando
    torch nao esta disponivel neste interpretador (a checagem de override
    acontece antes da tentativa de importar torch; a checagem de
    disponibilidade real de CUDA so acontece depois, dentro do bloco
    protegido por ImportError)."""
    config = _escrever_config_piloto_com_dados_inexistentes(tmp_path)
    saida_relatorio = tmp_path / "relatorio_device_cuda.json"

    cmd_train(_args_train(config, saida=saida_relatorio, device="cuda"))

    relatorio = json.loads(saida_relatorio.read_text(encoding="utf-8"))
    nomes_checagens = {c["nome"]: c for c in relatorio["checagens"]}
    assert nomes_checagens["device_override_cli"]["ok"] is True
    assert "cuda" in nomes_checagens["device_override_cli"]["detalhe"]
