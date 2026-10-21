import torch

from ignis.model import IgnisModel
from ignis.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path, config_reduzida):
    torch.manual_seed(0)
    model = IgnisModel(config_reduzida)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    caminho = tmp_path / "ckpt.pt"
    save_checkpoint(caminho, model, optimizer, step=7)

    modelo2 = IgnisModel(config_reduzida)
    optimizer2 = torch.optim.AdamW(modelo2.parameters(), lr=1e-3)
    estado = load_checkpoint(caminho, modelo2, optimizer2)

    assert estado["step"] == 7
    for (nome1, p1), (nome2, p2) in zip(
        model.state_dict().items(), modelo2.state_dict().items()
    ):
        assert nome1 == nome2
        assert torch.equal(p1, p2), f"pesos divergem em {nome1} após load_checkpoint"
