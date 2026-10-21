import torch

from ignis.inference.generate import generate
from ignis.model import IgnisModel


def test_geracao_greedy_shape_e_prefixo(config_reduzida):
    torch.manual_seed(0)
    model = IgnisModel(config_reduzida)
    entrada = torch.randint(0, config_reduzida.vocab_size, (1, 4))

    saida = generate(model, entrada, max_new_tokens=6, temperature=0.0)

    assert saida.shape == (1, 10)
    assert torch.equal(saida[:, :4], entrada)
    assert (saida >= 0).all() and (saida < config_reduzida.vocab_size).all()


def test_geracao_amostrada_com_eos(config_reduzida):
    torch.manual_seed(0)
    model = IgnisModel(config_reduzida)
    entrada = torch.randint(0, config_reduzida.vocab_size, (1, 3))

    saida = generate(
        model, entrada, max_new_tokens=50, temperature=0.8,
        top_k=5, eos_token_id=0, seed=123,
    )
    assert saida.shape[1] <= 3 + 50
    assert (saida >= 0).all() and (saida < config_reduzida.vocab_size).all()
