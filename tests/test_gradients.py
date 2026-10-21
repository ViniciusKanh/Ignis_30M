import torch

from ignis.model import IgnisModel


def test_gradientes_finitos(config_reduzida):
    torch.manual_seed(0)
    model = IgnisModel(config_reduzida)
    model.train()

    x = torch.randint(0, config_reduzida.vocab_size, (2, 16))
    y = torch.randint(0, config_reduzida.vocab_size, (2, 16))

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1)
    )
    loss.backward()

    for nome, p in model.named_parameters():
        assert p.grad is not None, f"sem gradiente para {nome}"
        assert torch.isfinite(p.grad).all(), f"gradiente não finito em {nome}"
