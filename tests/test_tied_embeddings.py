from ignis.model import IgnisModel, contagem_parametros_esperada


def test_embeddings_compartilhados_mesma_instancia(config_reduzida):
    model = IgnisModel(config_reduzida)
    assert model.lm_head.weight is model.embed_tokens.weight


def test_contagem_nao_duplica_embedding_compartilhado(config_reduzida):
    model = IgnisModel(config_reduzida)
    assert model.contar_parametros() == contagem_parametros_esperada(config_reduzida)
