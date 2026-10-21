"""Testes de carregamento/validação de configs/train-pilot.yaml (Fase 3).

Propositalmente SEM dependência de torch: `ignis.config.PilotoConfig` só usa
pydantic + PyYAML, para que a validação do arquivo de configuração do piloto
possa ser testada mesmo em um ambiente sem PyTorch instalado. A conversão
para `TreinadorConfig` (torch-dependente) é testada separadamente, onde
torch estiver disponível (ver tests/test_training_trainer.py).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ignis.config import IgnisConfig, PilotoConfig


def _dados_minimos(**overrides) -> dict:
    base = {
        "modelo": {"config": "configs/ignis-30m.yaml"},
        "dados": {
            "manifesto": "data/manifesto.jsonl",
            "splits": "data/splits.json",
            "tokenizer": "data/tokenizer.json",
            "dataset_base": "data/tokenizado",
            "seq_len": 128,
        },
        "otimizacao": {"lr": 3e-4},
        "lote": {"microbatch_size": 1, "grad_accum_steps": 32},
        "orcamento": {"max_steps": 200},
        "validacao_e_checkpoint": {
            "eval_every": 20,
            "checkpoint_every": 20,
            "log_every": 5,
        },
    }
    base.update(overrides)
    return base


def test_carrega_train_pilot_yaml_real_do_repositorio():
    """O arquivo configs/train-pilot.yaml versionado precisa continuar
    válido — este teste falha se alguém editá-lo e quebrar a validação."""
    piloto = PilotoConfig.from_yaml("configs/train-pilot.yaml")
    assert piloto.modelo.config == "configs/ignis-30m.yaml"
    assert piloto.dados.seq_len == 128
    assert piloto.lote.microbatch_size == 1
    assert piloto.orcamento.max_steps > 0


def test_lote_efetivo_e_o_produto_microbatch_por_grad_accum():
    piloto = PilotoConfig.model_validate(_dados_minimos())
    assert piloto.lote.lote_efetivo == piloto.lote.microbatch_size * piloto.lote.grad_accum_steps
    assert piloto.lote.lote_efetivo == 32


def test_max_steps_e_obrigatorio():
    dados = _dados_minimos()
    dados["orcamento"] = {}
    with pytest.raises(ValidationError):
        PilotoConfig.model_validate(dados)


def test_max_steps_deve_ser_positivo():
    dados = _dados_minimos()
    dados["orcamento"] = {"max_steps": 0}
    with pytest.raises(ValidationError):
        PilotoConfig.model_validate(dados)


def test_seq_len_menor_que_dois_e_invalido():
    dados = _dados_minimos()
    dados["dados"]["seq_len"] = 1
    with pytest.raises(ValidationError):
        PilotoConfig.model_validate(dados)


def test_lr_deve_ser_positivo():
    dados = _dados_minimos()
    dados["otimizacao"] = {"lr": 0.0}
    with pytest.raises(ValidationError):
        PilotoConfig.model_validate(dados)


def test_validar_contexto_contra_modelo_aceita_contexto_compativel():
    piloto = PilotoConfig.model_validate(_dados_minimos())
    modelo = IgnisConfig.ignis_30m().modelo
    piloto.validar_contexto_contra_modelo(modelo)  # não deve levantar


def test_validar_contexto_contra_modelo_rejeita_contexto_maior_que_max_seq_len():
    dados = _dados_minimos()
    dados["dados"]["seq_len"] = 2000  # contexto efetivo 1999 > max_seq_len (1024)
    piloto = PilotoConfig.model_validate(dados)
    modelo = IgnisConfig.ignis_30m().modelo
    with pytest.raises(ValueError, match="excede max_seq_len"):
        piloto.validar_contexto_contra_modelo(modelo)


def test_campos_opcionais_tem_defaults_documentados():
    piloto = PilotoConfig.model_validate(_dados_minimos())
    assert piloto.execucao.device == "cpu"
    assert piloto.execucao.parar_em_falha_numerica is True
    assert piloto.execucao.parar_em_falha_de_memoria is True
    assert piloto.execucao.mixed_precision == "no"
    assert piloto.smoke_test.habilitado is True
    assert piloto.smoke_test.max_steps == 10


def test_mixed_precision_e_parar_em_falha_de_memoria_sao_configuraveis():
    """Sprint 5: ambos os campos devem ser configuraveis via YAML (secao
    execucao), nao apenas ter um default fixo."""
    piloto = PilotoConfig.model_validate(
        _dados_minimos(
            execucao={
                "device": "cpu",
                "mixed_precision": "bf16",
                "parar_em_falha_de_memoria": False,
            }
        )
    )
    assert piloto.execucao.mixed_precision == "bf16"
    assert piloto.execucao.parar_em_falha_de_memoria is False


def test_mixed_precision_invalido_no_yaml_e_rejeitado():
    with pytest.raises(ValidationError):
        PilotoConfig.model_validate(_dados_minimos(execucao={"mixed_precision": "int8"}))


def test_tokens_alvo_hipotese_e_opcional_e_positivo_quando_presente():
    dados = _dados_minimos()
    dados["orcamento"]["tokens_alvo_hipotese"] = 0
    with pytest.raises(ValidationError):
        PilotoConfig.model_validate(dados)

    dados["orcamento"]["tokens_alvo_hipotese"] = 100
    piloto = PilotoConfig.model_validate(dados)
    assert piloto.orcamento.tokens_alvo_hipotese == 100


def test_modelo_config_nunca_duplica_hiperparametros_arquiteturais():
    """Teste de documentação executável: `PilotoModeloRefConfig` só tem o
    campo `config` (uma referência de caminho) — reforça por código que o
    piloto nunca pode declarar vocab_size/n_layers/etc. próprios que
    divirjam silenciosamente da arquitetura oficial."""
    from ignis.config import PilotoModeloRefConfig

    campos = set(PilotoModeloRefConfig.model_fields.keys())
    assert campos == {"config"}
