"""Regressões de proveniência da avaliação de checkpoints legados."""
import importlib.util
from pathlib import Path

import pytest

from ignis.training.checkpoint import sha256_arquivo

spec = importlib.util.spec_from_file_location(
    "avaliar_checkpoint", Path(__file__).parents[1] / "scripts/avaliar_checkpoint.py"
)
modulo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulo)


def test_hash_ausente_exige_opcao_e_nao_afirma_compatibilidade(tmp_path):
    caminho = tmp_path / "splits.json"
    caminho.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="legado sem hash"):
        modulo.verificar_splits({"dataset_sha256": None}, caminho)
    resultado = modulo.verificar_splits({}, caminho, permitir_legado=True)
    assert resultado["status"] == "origem_nao_verificada"
    assert resultado["sha256_checkpoint"] is None
    assert resultado["sha256_atual"] == sha256_arquivo(caminho)


def test_opcao_legado_nao_libera_hash_divergente(tmp_path):
    caminho = tmp_path / "splits.json"
    caminho.write_text("{}", encoding="utf-8")
    for permitir in (False, True):
        with pytest.raises(RuntimeError, match="hashes diferentes"):
            modulo.verificar_splits({"dataset_sha256": "outro"}, caminho, permitir)


def test_hash_compativel_verificado(tmp_path):
    caminho = tmp_path / "splits.json"
    caminho.write_text("{}", encoding="utf-8")
    resultado = modulo.verificar_splits(
        {"dataset_sha256": sha256_arquivo(caminho)}, caminho
    )
    assert resultado["status"] == "hash_compativel"
