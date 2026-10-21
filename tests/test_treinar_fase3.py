"""Protege a separação entre preparação e atualização real de pesos."""
import importlib.util
import json
from pathlib import Path

import pytest

from ignis.config import PilotoConfig

RAIZ = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("treinar_fase3", RAIZ / "scripts/treinar_fase3.py")
modulo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulo)


def test_config_proposta_e_limites_inalterados():
    piloto = PilotoConfig.from_yaml(RAIZ / "configs/train-pilot-corpus-v1-fase3a.yaml")
    config = modulo.validar_config(piloto)
    assert config.salvar_melhor
    assert config.eval_max_batches is None
    assert config.max_steps == 1000
    for campo, valor in (("max_steps", 2000),):
        alterada = piloto.model_copy(deep=True)
        setattr(alterada.orcamento, campo, valor)
        with pytest.raises(RuntimeError, match="fora da proposta"):
            modulo.validar_config(alterada)
    alterada = piloto.model_copy(deep=True)
    alterada.dados.dataset_base = "data/outro_corpus"
    with pytest.raises(RuntimeError, match="Dados fora da proposta"):
        modulo.validar_config(alterada)


def test_sem_flag_nao_constroi_treinador_nem_cria_pasta_de_pesos(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    saida = tmp_path / "checkpoints" / "fase3"
    monkeypatch.setattr(modulo, "SAIDA", saida)
    monkeypatch.setattr(modulo, "preparar", lambda: (None, None, None, None, None, {"fase": "teste"}))

    def proibido(*args, **kwargs):
        raise AssertionError("Dry-run não pode construir treinador.")

    monkeypatch.setattr(modulo, "Treinador", proibido)
    modulo.executar([])
    assert not saida.exists()
    relatorios = list((tmp_path / "docs/experiments").glob("fase3a_dryrun_*/relatorio.json"))
    assert len(relatorios) == 1
    resultado = json.loads(relatorios[0].read_text(encoding="utf-8"))
    assert resultado["dry_run_aprovado"] is True
    assert resultado["pesos_atualizados"] is False
