import torch

from ignis.model import IgnisModel


def test_forward_shape(config_reduzida):
    torch.manual_seed(0)
    model = IgnisModel(config_reduzida)
    x = torch.randint(0, config_reduzida.vocab_size, (3, 12))
    logits = model(x)
    assert logits.shape == (3, 12, config_reduzida.vocab_size)
    assert torch.isfinite(logits).all()


def test_eval_mode_deterministico(config_reduzida):
    torch.manual_seed(0)
    model = IgnisModel(config_reduzida)
    model.eval()
    x = torch.randint(0, config_reduzida.vocab_size, (2, 8))
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)
    assert torch.equal(y1, y2)
