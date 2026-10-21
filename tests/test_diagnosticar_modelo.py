"""Verificações metodológicas dos diagnósticos sem treinamento."""
import importlib.util
import math
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("diagnosticar_modelo",
    Path(__file__).parents[1] / "scripts/diagnosticar_modelo.py")
modulo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulo)


def test_controle_preserva_ultimo_token_e_conteudo_do_prefixo():
    entrada = torch.arange(32).reshape(2, 16)
    copia = entrada.clone()
    resultado = modulo.embaralhar_prefixos(entrada, torch.Generator().manual_seed(42))
    assert torch.equal(entrada, copia)
    assert torch.equal(resultado[:, -1], entrada[:, -1])
    assert torch.equal(resultado[:, :-1].sort().values, entrada[:, :-1].sort().values)
    assert not torch.equal(resultado[:, :-1], entrada[:, :-1])
    repetido = modulo.embaralhar_prefixos(entrada, torch.Generator().manual_seed(42))
    assert torch.equal(resultado, repetido)


def test_controle_rejeita_contexto_curto():
    with pytest.raises(ValueError):
        modulo.embaralhar_prefixos(torch.ones(1, 2), torch.Generator())


def test_avaliacao_pondera_alvos_e_ignora_padding_sem_gradientes():
    class ModeloFixo(torch.nn.Module):
        def forward(self, entrada):
            assert not torch.is_grad_enabled()
            # Probabilidades conhecidas: p(0)=1/2, p(1)=1/4, p(2)=p(3)=1/8.
            return torch.tensor([0.5, 0.25, 0.125, 0.125]).log().expand(*entrada.shape, 4)

    loader = [(torch.ones(1, 2, dtype=torch.long), torch.tensor([[0, -100]])),
              (torch.ones(1, 2, dtype=torch.long), torch.tensor([[1, 1]]))]
    resultado = modulo.avaliar_completo(ModeloFixo(), loader, "cpu", torch.full((4,), -math.log(4)))
    assert resultado["tokens_alvo"] == 3
    assert resultado["loss"] == pytest.approx((math.log(2) + 2 * math.log(4)) / 3)
    assert resultado["loss_unigrama_treino"] == pytest.approx(math.log(4))


def test_sinais_lexicais_exigem_dois_termos_e_nao_sao_exclusivos():
    assert modulo.indicadores_texto("Um algoritmo.") == {}
    grupos = modulo.indicadores_texto("Um algoritmo de software na cidade com população.")
    assert "computacao" in grupos
    assert "geografia_demografia" in grupos
