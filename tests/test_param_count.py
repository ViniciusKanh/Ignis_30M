from ignis.config import IgnisConfig
from ignis.model import IgnisModel, contagem_parametros_esperada


def test_contagem_parametros_ignis_30m():
    config = IgnisConfig.ignis_30m().modelo
    esperado = contagem_parametros_esperada(config)
    assert esperado == 30_416_384

    model = IgnisModel(config)
    real = model.contar_parametros()
    assert real == esperado


def test_contagem_parametros_config_reduzida(config_reduzida):
    model = IgnisModel(config_reduzida)
    assert model.contar_parametros() == contagem_parametros_esperada(config_reduzida)
