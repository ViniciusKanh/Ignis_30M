import pytest

from ignis.config import IgnisConfig, IgnisModelConfig


@pytest.fixture
def config_reduzida() -> IgnisModelConfig:
    """Config minúscula, reaproveitando exatamente os mesmos componentes
    arquiteturais do Ignis-30M, usada para testes rápidos em CPU."""
    return IgnisModelConfig(
        vocab_size=64,
        max_seq_len=32,
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        intermediate_size=64,
    )


@pytest.fixture
def config_ignis_30m() -> IgnisModelConfig:
    return IgnisConfig.ignis_30m().modelo
