import torch

from ignis.config import IgnisModelConfig
from ignis.model import IgnisModel
from ignis.training.loop import train_steps


def test_overfit_controlado_modelo_reduzido():
    """Smoke test de aprendizagem: um modelo pequeno, usando exatamente os
    mesmos componentes arquiteturais do Ignis-30M, deve conseguir decorar
    um microcorpus sintético fixo. Isso NÃO é evidência de generalização
    nem de competência em português — apenas de que o forward/backward e
    a otimização estão matematicamente corretos.
    """
    torch.manual_seed(0)
    config = IgnisModelConfig(
        vocab_size=32, max_seq_len=16, n_layers=2, d_model=32,
        n_heads=4, n_kv_heads=2, head_dim=8, intermediate_size=64,
    )
    model = IgnisModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    torch.manual_seed(1)
    sequencias = torch.randint(0, config.vocab_size, (4, 12))
    entradas, alvos = sequencias[:, :-1], sequencias[:, 1:]

    perdas: list[float] = []
    for _ in range(300):
        perdas.extend(train_steps(model, optimizer, [(entradas, alvos)]))

    assert perdas[0] > perdas[-1], "a perda não diminuiu durante o overfit"
    assert perdas[-1] < 1.0, (
        f"overfit não demonstrado de forma convincente: perda final={perdas[-1]:.4f}"
    )
