import pytest
from pydantic import ValidationError

from ignis.config import IgnisConfig, IgnisModelConfig


def test_config_canonica_ignis_30m():
    cfg = IgnisConfig.ignis_30m()
    assert cfg.modelo.vocab_size == 16384
    assert cfg.modelo.n_layers == 7
    assert cfg.modelo.d_model == 512
    assert cfg.modelo.n_heads == 8
    assert cfg.modelo.n_kv_heads == 4
    assert cfg.modelo.head_dim == 64
    assert cfg.modelo.intermediate_size == 1536
    assert cfg.modelo.tie_word_embeddings is True
    assert cfg.modelo.bias is False


def test_config_yaml_bate_com_canonica():
    cfg_yaml = IgnisConfig.from_yaml("configs/ignis-30m.yaml")
    assert cfg_yaml.modelo == IgnisConfig.ignis_30m().modelo


def test_config_invalida_d_model_nao_divisivel_por_n_heads():
    with pytest.raises(ValidationError):
        IgnisModelConfig(
            vocab_size=100, max_seq_len=10, n_layers=1, d_model=100,
            n_heads=7, n_kv_heads=1, head_dim=8, intermediate_size=100,
        )


def test_config_invalida_n_heads_nao_divisivel_por_n_kv_heads():
    with pytest.raises(ValidationError):
        IgnisModelConfig(
            vocab_size=100, max_seq_len=10, n_layers=1, d_model=64,
            n_heads=8, n_kv_heads=3, head_dim=8, intermediate_size=100,
        )


def test_config_invalida_head_dim_vezes_n_heads_diferente_de_d_model():
    with pytest.raises(ValidationError):
        IgnisModelConfig(
            vocab_size=100, max_seq_len=10, n_layers=1, d_model=64,
            n_heads=8, n_kv_heads=4, head_dim=4, intermediate_size=100,
        )
