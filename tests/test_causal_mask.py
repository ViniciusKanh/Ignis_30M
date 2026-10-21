import torch

from ignis.model import IgnisModel


def test_isolamento_causal(config_reduzida):
    """Alterar um token na posição p não pode mudar a saída em posições < p."""
    torch.manual_seed(0)
    model = IgnisModel(config_reduzida)
    model.eval()

    seq_len = 10
    pos_alterada = 5
    entrada_a = torch.randint(0, config_reduzida.vocab_size, (1, seq_len))
    entrada_b = entrada_a.clone()
    entrada_b[0, pos_alterada] = (entrada_b[0, pos_alterada] + 1) % config_reduzida.vocab_size

    with torch.no_grad():
        saida_a = model(entrada_a)
        saida_b = model(entrada_b)

    assert torch.allclose(saida_a[0, :pos_alterada], saida_b[0, :pos_alterada], atol=1e-5), (
        "vazamento de token futuro detectado: saída em posição anterior "
        "mudou ao alterar um token posterior"
    )
    assert not torch.allclose(saida_a[0, pos_alterada:], saida_b[0, pos_alterada:], atol=1e-5), (
        "saída não mudou nas posições >= à alterada; teste pode estar sem efeito"
    )
