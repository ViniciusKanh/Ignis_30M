"""Testes do loop de treinamento completo (Fase 3 — `ignis.training.trainer`).

Requer torch — não executável na VM Linux de desenvolvimento (ver
docs/experiments/fase1-fundacao.md). Rode no Windows nativo com torch
instalado: `pytest tests/test_training_trainer.py -v`.

Todos os testes usam `config_reduzida` (fixture de tests/conftest.py) — um
modelo minúsculo que reaproveita exatamente os mesmos componentes
arquiteturais do Ignis-30M, nunca o modelo completo (30,4M parâmetros),
para rodar em segundos em CPU.
"""

import copy
import json
import random

import numpy as np
import pytest
import torch

from ignis.config import IgnisModelConfig
from ignis.data.pretrain_prep import EstatisticasPreparo, salvar_dataset_tokenizado
from ignis.model import IgnisModel
from ignis.training.checkpoint import CheckpointMetadata
from ignis.training.dataloader import criar_dataloader
from ignis.training.metrics import ler_metricas
from ignis.training.trainer import FalhaNumericaError, Treinador, TreinadorConfig, lr_no_passo


def _salvar_dataset_sintetico(tmp_path, nome, vocab_size, seq_len=8, n_sequencias=16, seed=0):
    rng = np.random.default_rng(seed)
    sequencias = rng.integers(low=1, high=vocab_size, size=(n_sequencias, seq_len)).astype(np.int32)
    mascaras = np.ones((n_sequencias, seq_len), dtype=np.int8)
    estatisticas = EstatisticasPreparo(
        n_documentos=n_sequencias,
        n_tokens_total=int(mascaras.sum()),
        n_sequencias=n_sequencias,
        n_sequencias_com_padding=0,
        seq_len=seq_len,
    )
    base = tmp_path / nome
    salvar_dataset_tokenizado(base, sequencias, mascaras, estatisticas)
    return base


def _config_treino_rapido(out_dir, max_steps=5, **overrides) -> TreinadorConfig:
    base = dict(
        lr=1e-2,
        warmup_steps=2,
        max_steps=max_steps,
        eval_every=10_000,  # desativa avaliação por padrão nos testes que não a exercitam
        checkpoint_every=10_000,  # idem para checkpoint
        log_every=1,
        seed=42,
        device="cpu",
        out_dir=str(out_dir),
    )
    base.update(overrides)
    return TreinadorConfig(**base)


def test_final_preserva_passos_apos_checkpoint_periodico(config_reduzida, tmp_path):
    modelo = IgnisModel(config_reduzida)
    base = _salvar_dataset_sintetico(tmp_path, "dados_final", config_reduzida.vocab_size)
    treinador = Treinador(modelo, _config_treino_rapido(
        tmp_path / "final", max_steps=5, checkpoint_every=2))
    resultado = treinador.treinar(lambda e: criar_dataloader(
        base, batch_size=2, embaralhar=True, seed=42, epoch=e), None)
    estado = torch.load(resultado.ultimo_checkpoint, weights_only=False)
    assert estado["step"] == 5
    assert (tmp_path / "final" / "step4.pt").exists()
    for nome, peso in modelo.state_dict().items():
        assert torch.equal(estado["model_state_dict"][nome], peso)


def test_retomada_meio_epoca_preserva_pesos_adam_rng_e_paciencia(config_reduzida, tmp_path):
    config_reduzida.dropout = 0.1
    base = _salvar_dataset_sintetico(tmp_path, "dados_cursor", config_reduzida.vocab_size,
                                    n_sequencias=20)

    def loader(epoca):
        return criar_dataloader(base, batch_size=2, embaralhar=True, seed=42, epoch=epoca)

    def novo(nome):
        torch.manual_seed(123)
        return Treinador(IgnisModel(config_reduzida), _config_treino_rapido(
            tmp_path / nome, max_steps=8, grad_accum_steps=2, checkpoint_every=2))

    continuo = novo("cont_cursor")
    continuo.treinar(loader, None)
    rng_continuo = torch.get_rng_state().clone()
    parcial = novo("par_cursor")
    parcial.avaliacoes_sem_melhora = 3
    resultado = parcial.treinar(loader, None, parar_no_passo=3)
    retomado = novo("ret_cursor")
    retomado.retomar_de_checkpoint(resultado.ultimo_checkpoint)
    assert retomado.microbatches_consumidos == 6
    assert retomado.epoca_atual == 0
    assert retomado.avaliacoes_sem_melhora == 3
    retomado.treinar(loader, None)
    assert torch.equal(torch.get_rng_state(), rng_continuo)
    for nome, peso in continuo.model.state_dict().items():
        assert torch.equal(peso, retomado.model.state_dict()[nome]), nome
    for indice, estado in continuo.optimizer.state_dict()["state"].items():
        for chave, valor in estado.items():
            assert torch.equal(valor, retomado.optimizer.state_dict()["state"][indice][chave])


def test_paciencia_retomada_dispara_no_mesmo_passo(config_reduzida, tmp_path, monkeypatch):
    base = _salvar_dataset_sintetico(tmp_path, "dados_paciencia", config_reduzida.vocab_size)

    def novo(nome):
        treinador = Treinador(IgnisModel(config_reduzida), _config_treino_rapido(
            tmp_path / nome, max_steps=10, eval_every=1, paciencia_avaliacoes=3))
        monkeypatch.setattr(treinador, "avaliar", lambda loader: {"loss_validacao": 2.0})
        return treinador

    loader = lambda e: criar_dataloader(base, batch_size=1, embaralhar=False)
    parcial = novo("par_paciencia")
    parcial.melhor_loss_validacao = 1.0
    resultado = parcial.treinar(loader, [], parar_no_passo=2)
    retomado = novo("ret_paciencia")
    retomado.retomar_de_checkpoint(resultado.ultimo_checkpoint)
    final = retomado.treinar(loader, [])
    assert final.passos_executados == 3
    assert final.motivo_parada == "early_stopping_sem_melhora_na_validacao"


def test_best_preserva_melhor_avaliacao_apos_piora(config_reduzida, tmp_path, monkeypatch):
    base = _salvar_dataset_sintetico(tmp_path, "dados_best", config_reduzida.vocab_size)
    treinador = Treinador(IgnisModel(config_reduzida), _config_treino_rapido(
        tmp_path / "best", max_steps=3, eval_every=1, salvar_melhor=True))
    perdas = iter([0.8, 0.9, 1.1])
    monkeypatch.setattr(treinador, "avaliar", lambda loader: {"loss_validacao": next(perdas)})
    treinador.melhor_loss_validacao = 1.0
    treinador.salvar_checkpoint("best.pt")
    treinador.treinar(lambda e: criar_dataloader(base, batch_size=1, embaralhar=False), [])
    melhor = torch.load(tmp_path / "best/best.pt", weights_only=False)
    assert melhor["step"] == 1
    assert melhor["extra"]["melhor_loss_validacao"] == 0.8


def test_lr_no_passo_tem_warmup_linear_e_decai_apos():
    config = TreinadorConfig(lr=0.1, lr_min=0.0, warmup_steps=10, max_steps=100)
    lr_inicio = lr_no_passo(0, config)
    lr_meio_warmup = lr_no_passo(5, config)
    lr_fim_warmup = lr_no_passo(10, config)
    lr_final = lr_no_passo(99, config)

    assert lr_inicio < lr_meio_warmup < lr_fim_warmup
    assert lr_fim_warmup <= config.lr * 1.0001
    assert lr_final < lr_fim_warmup  # decaimento em cosseno após o warmup
    assert lr_final >= config.lr_min - 1e-9


def test_atualizacao_real_de_parametros_apos_optimizer_step(config_reduzida: IgnisModelConfig, tmp_path):
    model = IgnisModel(config_reduzida)
    pesos_antes = {nome: p.detach().clone() for nome, p in model.named_parameters()}

    dataset_base = _salvar_dataset_sintetico(tmp_path, "treino", config_reduzida.vocab_size, seq_len=8)
    config_treino = _config_treino_rapido(tmp_path / "ckpt", max_steps=3, grad_accum_steps=1)
    treinador = Treinador(model, config_treino)

    treinador.treinar(
        criar_loader_treino=lambda epoca: criar_dataloader(
            dataset_base, batch_size=4, embaralhar=True, seed=config_treino.seed, epoch=epoca
        ),
        val_loader=None,
    )

    mudou_algum_peso = any(
        not torch.equal(pesos_antes[nome], p.detach()) for nome, p in model.named_parameters()
    )
    assert mudou_algum_peso, "nenhum parâmetro mudou após optimizer.step() — otimização não está ocorrendo"


def test_reducao_de_loss_em_experimento_controlado(config_reduzida: IgnisModelConfig, tmp_path):
    """Overfit controlado: poucos exemplos fixos, muitos passos — a loss de
    treino deve cair de forma clara (não precisa chegar perto de zero)."""
    model = IgnisModel(config_reduzida)
    dataset_base = _salvar_dataset_sintetico(
        tmp_path, "treino", config_reduzida.vocab_size, seq_len=8, n_sequencias=4
    )
    config_treino = _config_treino_rapido(
        tmp_path / "ckpt", max_steps=150, lr=5e-3, warmup_steps=5, log_every=1
    )
    treinador = Treinador(model, config_treino)
    treinador.treinar(
        criar_loader_treino=lambda epoca: criar_dataloader(
            dataset_base, batch_size=4, embaralhar=True, seed=config_treino.seed, epoch=epoca
        ),
        val_loader=None,
    )

    from ignis.training.metrics import ler_metricas

    linhas = [l for l in ler_metricas(config_treino.out_dir + "/metricas.jsonl") if "training_loss" in l]
    assert len(linhas) >= 2
    primeira_loss = linhas[0]["training_loss"]
    ultima_loss = linhas[-1]["training_loss"]
    assert ultima_loss < primeira_loss, f"loss não caiu: {primeira_loss} -> {ultima_loss}"


def test_avaliacao_nao_calcula_gradientes_nem_altera_pesos(config_reduzida: IgnisModelConfig, tmp_path):
    model = IgnisModel(config_reduzida)
    dataset_base = _salvar_dataset_sintetico(tmp_path, "val", config_reduzida.vocab_size, seq_len=8)
    config_treino = _config_treino_rapido(tmp_path / "ckpt")
    treinador = Treinador(model, config_treino)

    pesos_antes = {nome: p.detach().clone() for nome, p in model.named_parameters()}
    val_loader = criar_dataloader(dataset_base, batch_size=4, embaralhar=False)
    metricas = treinador.avaliar(val_loader)

    assert metricas["loss_validacao"] is not None
    assert metricas["perplexidade"] is not None
    assert all(p.grad is None for p in model.parameters()), "avaliar() não deveria deixar gradientes computados"
    assert all(
        torch.equal(pesos_antes[nome], p.detach()) for nome, p in model.named_parameters()
    ), "avaliar() não deveria alterar nenhum peso do modelo"
    assert model.training, "o modelo deveria voltar para model.train() após avaliar()"


def test_limite_de_steps_e_respeitado(config_reduzida: IgnisModelConfig, tmp_path):
    model = IgnisModel(config_reduzida)
    dataset_base = _salvar_dataset_sintetico(
        tmp_path, "treino", config_reduzida.vocab_size, seq_len=8, n_sequencias=100
    )
    config_treino = _config_treino_rapido(tmp_path / "ckpt", max_steps=5)
    treinador = Treinador(model, config_treino)
    resultado = treinador.treinar(
        criar_loader_treino=lambda epoca: criar_dataloader(
            dataset_base, batch_size=2, embaralhar=True, seed=config_treino.seed, epoch=epoca
        ),
        val_loader=None,
    )
    assert resultado.passos_executados == 5
    assert resultado.motivo_parada == "max_steps_atingido"


def test_falha_numerica_interrompe_treinamento_e_registra(config_reduzida: IgnisModelConfig, tmp_path):
    model = IgnisModel(config_reduzida)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(float("nan"))  # força loss não-finita já no primeiro passo

    dataset_base = _salvar_dataset_sintetico(tmp_path, "treino", config_reduzida.vocab_size, seq_len=8)
    config_treino = _config_treino_rapido(tmp_path / "ckpt", max_steps=10)
    treinador = Treinador(model, config_treino)

    resultado = treinador.treinar(
        criar_loader_treino=lambda epoca: criar_dataloader(
            dataset_base, batch_size=4, embaralhar=True, seed=config_treino.seed, epoch=epoca
        ),
        val_loader=None,
    )

    assert resultado.motivo_parada == "falha_numerica"
    assert resultado.passos_executados == 0  # nenhum optimizer.step() chegou a ocorrer
    assert (tmp_path / "ckpt" / "falha_numerica.pt").exists()

    from ignis.training.metrics import ler_metricas

    eventos = [l for l in ler_metricas(config_treino.out_dir + "/metricas.jsonl") if l.get("evento") == "falha_numerica"]
    assert len(eventos) == 1


def test_falha_numerica_propaga_quando_configurado(config_reduzida: IgnisModelConfig, tmp_path):
    model = IgnisModel(config_reduzida)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(float("nan"))

    dataset_base = _salvar_dataset_sintetico(tmp_path, "treino", config_reduzida.vocab_size, seq_len=8)
    config_treino = _config_treino_rapido(tmp_path / "ckpt", max_steps=10, parar_em_falha_numerica=False)
    treinador = Treinador(model, config_treino)

    with pytest.raises(FalhaNumericaError):
        treinador.treinar(
            criar_loader_treino=lambda epoca: criar_dataloader(
                dataset_base, batch_size=4, embaralhar=True, seed=config_treino.seed, epoch=epoca
            ),
            val_loader=None,
        )


def test_determinismo_no_backend_cpu_com_mesma_seed(config_reduzida: IgnisModelConfig, tmp_path):
    """Em CPU (sem GPU envolvida), com a mesma seed, dois treinamentos
    independentes sobre os mesmos dados devem produzir exatamente os mesmos
    pesos finais. Isto é o limite de determinismo que a skill pede para
    verificar — não se aplica a um eventual backend de GPU, onde operações
    atômicas não-determinísticas podem impedir igualdade bit-a-bit."""
    dataset_base = _salvar_dataset_sintetico(tmp_path, "treino", config_reduzida.vocab_size, seq_len=8)

    def _rodar(out_dir):
        # IMPORTANTE: o modelo precisa ser (re)inicializado com uma seed
        # EXPLICITA antes de ser criado -- Treinador.__init__ so chama
        # set_seed() DEPOIS que o modelo ja foi instanciado (ele recebe o
        # modelo pronto), entao sem este torch.manual_seed aqui os dois
        # modelos desta funcao comecariam com pesos iniciais DIFERENTES
        # (o gerador global do PyTorch continua avancando entre chamadas),
        # o que já derrubaria o teste antes mesmo de qualquer passo de
        # treino -- bug de teste encontrado ao rodar de verdade no Windows.
        torch.manual_seed(123)
        model = IgnisModel(config_reduzida)
        config_treino = _config_treino_rapido(out_dir, max_steps=8, seed=123)
        treinador = Treinador(model, config_treino)
        treinador.treinar(
            criar_loader_treino=lambda epoca: criar_dataloader(
                dataset_base, batch_size=4, embaralhar=True, seed=config_treino.seed, epoch=epoca
            ),
            val_loader=None,
        )
        return model

    modelo_a = _rodar(tmp_path / "ckpt_a")
    modelo_b = _rodar(tmp_path / "ckpt_b")

    for (nome_a, pa), (nome_b, pb) in zip(modelo_a.named_parameters(), modelo_b.named_parameters()):
        assert nome_a == nome_b
        assert torch.equal(pa, pb), f"parâmetro '{nome_a}' divergiu entre duas execuções com a mesma seed"


def test_treinar_aceita_apenas_val_loader_nunca_teste_por_assinatura(config_reduzida: IgnisModelConfig, tmp_path):
    """Teste de documentação executável: a assinatura de `Treinador.treinar`
    só tem parâmetro para o carregador de validação, nunca de teste — reforça
    por código que o split de teste não deve ser passado ao treinador."""
    import inspect

    assinatura = inspect.signature(Treinador.treinar)
    nomes_parametros = list(assinatura.parameters.keys())
    assert "criar_loader_treino" in nomes_parametros
    assert "val_loader" in nomes_parametros
    assert not any("teste" in nome or "test" in nome for nome in nomes_parametros)


def test_equivalencia_treinamento_continuo_versus_interrompido_e_retomado(
    config_reduzida: IgnisModelConfig, tmp_path
):
    """Compara um treinamento contínuo de 8 passos com um treinamento
    interrompido em uma FRONTEIRA DE ÉPOCA (após 4 passos, com exatamente 4
    lotes por época) e depois retomado por mais 4 passos via checkpoint.

    CAUSA RAIZ DA DIVERGÊNCIA ORIGINALMENTE OBSERVADA (diagnosticada e
    corrigida nesta suite): a versão anterior deste teste usava
    `TreinadorConfig(max_steps=4)` para a fase "interrompida" e
    `TreinadorConfig(max_steps=8)` para as fases contínua e retomada —
    `max_steps` é o HORIZONTE do scheduler de LR (`lr_no_passo`), então os
    dois caminhos usavam curvas de learning rate DIFERENTES mesmo nos
    passos 1-4, que deveriam ser idênticos. Isso não era um problema de
    não-determinismo numérico, e sim um erro de modelagem do teste: ele
    confundia "ponto de parada operacional desta execução" com "horizonte
    total planejado do experimento". A correção foi adicionar o parâmetro
    `parar_no_passo` a `Treinador.treinar`, que afeta apenas a condição de
    parada do laço — o `self.config.max_steps` usado pelo scheduler
    permanece constante (8) nas três fases abaixo, como o critério de
    aceite exige.

    Limite de determinismo documentado (ver `Treinador.treinar`): a
    equivalência exata só é esperada quando a interrupção acontece em uma
    fronteira de época — uma interrupção no meio de uma época reprocessaria
    parte dela, divergindo de um treinamento contínuo hipotético."""
    dataset_base = _salvar_dataset_sintetico(
        tmp_path, "treino", config_reduzida.vocab_size, seq_len=8, n_sequencias=16
    )
    batch_size = 4  # 16 sequencias / batch_size=4 -> exatamente 4 passos por epoca
    seed_compartilhada = 999
    horizonte_do_experimento = 8  # constante nas tres fases -- ponto central da correcao

    def _novo_modelo():
        torch.manual_seed(0)
        return IgnisModel(config_reduzida)

    def _loader_factory(epoca):
        return criar_dataloader(dataset_base, batch_size=batch_size, embaralhar=True, seed=seed_compartilhada, epoch=epoca)

    # --- treinamento contínuo: 8 passos de uma vez ---
    modelo_continuo = _novo_modelo()
    config_continuo = _config_treino_rapido(
        tmp_path / "continuo", max_steps=horizonte_do_experimento, seed=seed_compartilhada
    )
    treinador_continuo = Treinador(modelo_continuo, config_continuo)
    resultado_continuo = treinador_continuo.treinar(criar_loader_treino=_loader_factory, val_loader=None)
    assert resultado_continuo.motivo_parada == "max_steps_atingido"

    # --- treinamento interrompido em 4 passos (fronteira de época) + retomado por mais 4 ---
    # MESMO horizonte (max_steps=8) da fase contínua -- apenas a PARADA
    # OPERACIONAL acontece em 4, via `parar_no_passo`, nunca via um
    # `max_steps` diferente.
    modelo_interrompido = _novo_modelo()
    config_interrompido = _config_treino_rapido(
        tmp_path / "interrompido", max_steps=horizonte_do_experimento, seed=seed_compartilhada
    )
    treinador_interrompido = Treinador(modelo_interrompido, config_interrompido)
    resultado_parcial = treinador_interrompido.treinar(
        criar_loader_treino=_loader_factory, val_loader=None, parar_no_passo=4
    )
    assert resultado_parcial.passos_executados == 4
    assert resultado_parcial.motivo_parada == "parada_operacional_solicitada"

    # out_dir PROPRIO (nao "interrompido") -- o Treinador salva seu
    # checkpoint final como "final.pt" dentro de out_dir; se reusasse o
    # mesmo out_dir da fase interrompida, ao terminar sobrescreveria o
    # proprio arquivo "final.pt" que e o `resultado_parcial.ultimo_checkpoint`
    # (mesmo bug real encontrado em scripts/smoke_training.py, ver
    # docs/limitations.md).
    config_retomado = _config_treino_rapido(
        tmp_path / "interrompido_retomado", max_steps=horizonte_do_experimento, seed=seed_compartilhada
    )
    treinador_retomado = Treinador(modelo_interrompido, config_retomado)
    treinador_retomado.retomar_de_checkpoint(resultado_parcial.ultimo_checkpoint)
    assert treinador_retomado.step == 4
    assert treinador_retomado.epoca_atual == 1
    resultado_final = treinador_retomado.treinar(criar_loader_treino=_loader_factory, val_loader=None)
    assert resultado_final.motivo_parada == "max_steps_atingido"

    for (nome_a, pa), (nome_b, pb) in zip(
        modelo_continuo.named_parameters(), modelo_interrompido.named_parameters()
    ):
        assert nome_a == nome_b
        assert torch.equal(pa, pb), (
            f"parâmetro '{nome_a}' divergiu entre treinamento contínuo e "
            "interrompido/retomado na fronteira de época"
        )


def test_treinamento_continuo_executa_todos_os_passos_planejados(
    config_reduzida: IgnisModelConfig, tmp_path
):
    """Caso-base isolado (sem interrupção): 8 passos contínuos devem
    executar exatamente 8 passos e parar por `max_steps_atingido` — serve
    de referência mínima para os testes de `parar_no_passo` abaixo."""
    dataset_base = _salvar_dataset_sintetico(
        tmp_path, "treino", config_reduzida.vocab_size, seq_len=8, n_sequencias=16
    )
    torch.manual_seed(0)
    model = IgnisModel(config_reduzida)
    config_treino = _config_treino_rapido(tmp_path / "ckpt", max_steps=8)
    treinador = Treinador(model, config_treino)
    resultado = treinador.treinar(
        criar_loader_treino=lambda epoca: criar_dataloader(
            dataset_base, batch_size=4, embaralhar=True, seed=config_treino.seed, epoch=epoca
        ),
        val_loader=None,
    )
    assert resultado.passos_executados == 8
    assert treinador.step == 8
    assert resultado.motivo_parada == "max_steps_atingido"


def test_parar_no_passo_interrompe_antes_do_horizonte_planejado(
    config_reduzida: IgnisModelConfig, tmp_path
):
    """`parar_no_passo` deve parar o laço de treino no ponto pedido SEM
    alterar `self.config.max_steps` — o horizonte do scheduler continua 8 —
    e o motivo de parada deve distinguir isso de atingir o horizonte real,
    para que nenhum chamador confunda uma parada operacional com o fim
    verdadeiro do experimento."""
    dataset_base = _salvar_dataset_sintetico(
        tmp_path, "treino", config_reduzida.vocab_size, seq_len=8, n_sequencias=16
    )
    torch.manual_seed(0)
    model = IgnisModel(config_reduzida)
    config_treino = _config_treino_rapido(tmp_path / "ckpt", max_steps=8)
    treinador = Treinador(model, config_treino)
    resultado = treinador.treinar(
        criar_loader_treino=lambda epoca: criar_dataloader(
            dataset_base, batch_size=4, embaralhar=True, seed=config_treino.seed, epoch=epoca
        ),
        val_loader=None,
        parar_no_passo=4,
    )
    assert resultado.passos_executados == 4
    assert treinador.step == 4
    assert resultado.motivo_parada == "parada_operacional_solicitada"
    assert treinador.config.max_steps == 8  # horizonte do scheduler permanece intacto


def test_scheduler_horizonte_diferente_entre_execucoes_nao_e_equivalencia(
    config_reduzida: IgnisModelConfig, tmp_path
):
    """Documenta, de forma explícita e executável, a distinção entre
    'mudar o horizonte do scheduler' (NÃO é equivalência, é um erro de uso)
    e uma retomada de verdade (equivalente — coberta por
    `test_equivalencia_treinamento_continuo_versus_interrompido_e_retomado`).

    Reproduz DELIBERADAMENTE o padrão que causava a divergência
    originalmente observada (duas `TreinadorConfig` com `max_steps`
    diferentes — 4 e 8 — para os mesmos 4 passos iniciais) e comprova que
    os pesos finais DIVERGEM, como esperado — isto não é um bug, é a prova
    de que o scheduler de fato depende de `max_steps`, e de que o teste de
    equivalência acima não estaria passando "por acaso"."""
    dataset_base = _salvar_dataset_sintetico(
        tmp_path, "treino", config_reduzida.vocab_size, seq_len=8, n_sequencias=16
    )
    batch_size = 4
    seed_compartilhada = 999

    def _novo_modelo():
        torch.manual_seed(0)
        return IgnisModel(config_reduzida)

    def _loader_factory(epoca):
        return criar_dataloader(dataset_base, batch_size=batch_size, embaralhar=True, seed=seed_compartilhada, epoch=epoca)

    modelo_continuo = _novo_modelo()
    config_continuo = _config_treino_rapido(tmp_path / "continuo_horiz", max_steps=8, seed=seed_compartilhada)
    treinador_continuo = Treinador(modelo_continuo, config_continuo)
    treinador_continuo.treinar(criar_loader_treino=_loader_factory, val_loader=None)

    modelo_horizonte_diferente = _novo_modelo()
    config_horizonte_diferente = _config_treino_rapido(
        tmp_path / "horizonte_diferente", max_steps=4, seed=seed_compartilhada
    )
    treinador_horizonte_diferente = Treinador(modelo_horizonte_diferente, config_horizonte_diferente)
    treinador_horizonte_diferente.treinar(criar_loader_treino=_loader_factory, val_loader=None)

    algum_parametro_divergiu = any(
        not torch.equal(pa, pb)
        for (_, pa), (_, pb) in zip(
            modelo_continuo.named_parameters(), modelo_horizonte_diferente.named_parameters()
        )
    )
    assert algum_parametro_divergiu, (
        "mudar max_steps (horizonte do scheduler) entre duas execuções NÃO "
        "deveria produzir pesos idênticos -- se produziu, o teste de "
        "equivalência real acima pode estar passando por coincidência, não "
        "por retomada corretamente implementada"
    )


def test_sequencia_de_learning_rate_e_continua_apos_retomada(
    config_reduzida: IgnisModelConfig, tmp_path
):
    """Prova mais direta da correção: a sequência de learning rates
    logada (metricas.jsonl) ao longo de um treinamento contínuo de 8 passos
    deve ser EXATAMENTE igual à sequência obtida concatenando uma parada
    operacional em 4 passos com a retomada até 8 passos."""
    dataset_base = _salvar_dataset_sintetico(
        tmp_path, "treino", config_reduzida.vocab_size, seq_len=8, n_sequencias=16
    )
    batch_size = 4
    seed_compartilhada = 555

    def _novo_modelo():
        torch.manual_seed(0)
        return IgnisModel(config_reduzida)

    def _loader_factory(epoca):
        return criar_dataloader(dataset_base, batch_size=batch_size, embaralhar=True, seed=seed_compartilhada, epoch=epoca)

    def _learning_rates(caminho_metricas):
        return [r["learning_rate"] for r in ler_metricas(caminho_metricas) if "learning_rate" in r]

    out_continuo = tmp_path / "continuo_lr"
    config_continuo = _config_treino_rapido(out_continuo, max_steps=8, seed=seed_compartilhada)
    treinador_continuo = Treinador(_novo_modelo(), config_continuo)
    treinador_continuo.treinar(criar_loader_treino=_loader_factory, val_loader=None)
    lrs_continuo = _learning_rates(out_continuo / "metricas.jsonl")

    out_interrompido = tmp_path / "interrompido_lr"
    config_interrompido = _config_treino_rapido(out_interrompido, max_steps=8, seed=seed_compartilhada)
    modelo_interrompido = _novo_modelo()
    treinador_interrompido = Treinador(modelo_interrompido, config_interrompido)
    resultado_parcial = treinador_interrompido.treinar(
        criar_loader_treino=_loader_factory, val_loader=None, parar_no_passo=4
    )
    config_retomado = _config_treino_rapido(out_interrompido, max_steps=8, seed=seed_compartilhada)
    treinador_retomado = Treinador(modelo_interrompido, config_retomado)
    treinador_retomado.retomar_de_checkpoint(resultado_parcial.ultimo_checkpoint)
    treinador_retomado.treinar(criar_loader_treino=_loader_factory, val_loader=None)
    lrs_interrompido = _learning_rates(out_interrompido / "metricas.jsonl")

    assert len(lrs_continuo) == 8
    assert len(lrs_interrompido) == 8
    assert lrs_continuo == lrs_interrompido


def test_estado_do_otimizador_adamw_e_identico_apos_parada_operacional_e_retomada(
    config_reduzida: IgnisModelConfig, tmp_path
):
    """Não basta os PESOS finais coincidirem — os momentos do AdamW
    (`exp_avg`, `exp_avg_sq`) e o contador `step` por parâmetro também
    precisam ser exatamente os mesmos entre o treinamento contínuo e o
    interrompido/retomado, senão os dois modelos divergiriam no primeiro
    passo de treino adicional, mesmo com pesos iguais no momento da
    comparação."""
    dataset_base = _salvar_dataset_sintetico(
        tmp_path, "treino", config_reduzida.vocab_size, seq_len=8, n_sequencias=16
    )
    batch_size = 4
    seed_compartilhada = 777

    def _novo_modelo():
        torch.manual_seed(0)
        return IgnisModel(config_reduzida)

    def _loader_factory(epoca):
        return criar_dataloader(dataset_base, batch_size=batch_size, embaralhar=True, seed=seed_compartilhada, epoch=epoca)

    modelo_continuo = _novo_modelo()
    config_continuo = _config_treino_rapido(tmp_path / "continuo_adam", max_steps=8, seed=seed_compartilhada)
    treinador_continuo = Treinador(modelo_continuo, config_continuo)
    treinador_continuo.treinar(criar_loader_treino=_loader_factory, val_loader=None)

    modelo_interrompido = _novo_modelo()
    config_interrompido = _config_treino_rapido(tmp_path / "interrompido_adam", max_steps=8, seed=seed_compartilhada)
    treinador_interrompido = Treinador(modelo_interrompido, config_interrompido)
    resultado_parcial = treinador_interrompido.treinar(
        criar_loader_treino=_loader_factory, val_loader=None, parar_no_passo=4
    )

    # out_dir PROPRIO (nao "interrompido_adam") -- mesma razao do teste de
    # equivalencia acima: evitar sobrescrever o "final.pt" da fase
    # interrompida com o "final.pt" da fase retomada.
    config_retomado = _config_treino_rapido(
        tmp_path / "interrompido_adam_retomado", max_steps=8, seed=seed_compartilhada
    )
    treinador_retomado = Treinador(modelo_interrompido, config_retomado)
    treinador_retomado.retomar_de_checkpoint(resultado_parcial.ultimo_checkpoint)
    treinador_retomado.treinar(criar_loader_treino=_loader_factory, val_loader=None)

    estado_continuo = treinador_continuo.optimizer.state_dict()["state"]
    estado_retomado = treinador_retomado.optimizer.state_dict()["state"]
    assert set(estado_continuo.keys()) == set(estado_retomado.keys())
    for chave in estado_continuo:
        original = estado_continuo[chave]
        retomado = estado_retomado[chave]
        assert original["step"] == retomado["step"]
        assert torch.equal(original["exp_avg"], retomado["exp_avg"])
        assert torch.equal(original["exp_avg_sq"], retomado["exp_avg_sq"])


def test_rng_state_e_persistido_e_restaurado_pelo_checkpoint(
    config_reduzida: IgnisModelConfig, tmp_path
):
    """O checkpoint salva o estado dos geradores de números aleatórios
    (python/numpy/torch) no momento em que o treinamento parou, e
    `retomar_de_checkpoint` deve restaurar EXATAMENTE esse estado — nunca
    apenas re-semear com `config.seed`, que não reproduziria o estado real
    (derivado de todo o histórico de números consumidos durante os passos
    de treino já executados)."""
    dataset_base = _salvar_dataset_sintetico(tmp_path, "treino", config_reduzida.vocab_size, seq_len=8)

    def _loader_factory(epoca):
        return criar_dataloader(dataset_base, batch_size=4, embaralhar=True, seed=42, epoch=epoca)

    torch.manual_seed(0)
    model = IgnisModel(config_reduzida)
    config_treino = _config_treino_rapido(tmp_path / "ckpt_rng", max_steps=4, seed=42)
    treinador = Treinador(model, config_treino)
    resultado = treinador.treinar(criar_loader_treino=_loader_factory, val_loader=None)

    # estado dos geradores globais logo apos o treinamento parar -- e
    # exatamente isto que `salvar_checkpoint` capturou no checkpoint final,
    # pois nada entre o ultimo optimizer.step() e o salvamento consome
    # aleatoriedade.
    estado_torch_esperado = torch.get_rng_state().clone()
    estado_numpy_esperado = np.random.get_state()

    modelo_novo = IgnisModel(config_reduzida)
    config_retomado = _config_treino_rapido(tmp_path / "ckpt_rng", max_steps=8, seed=42)
    treinador_retomado = Treinador(modelo_novo, config_retomado)

    # perturba deliberadamente os geradores globais ANTES de retomar, para
    # provar que a retomada de fato RESTAURA o estado salvo, em vez de so
    # "coincidir" com ele por nao ter sido alterado.
    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)

    treinador_retomado.retomar_de_checkpoint(resultado.ultimo_checkpoint)

    assert torch.equal(torch.get_rng_state(), estado_torch_esperado)
    estado_numpy_restaurado = np.random.get_state()
    assert estado_numpy_restaurado[0] == estado_numpy_esperado[0]
    assert np.array_equal(estado_numpy_restaurado[1], estado_numpy_esperado[1])
    assert estado_numpy_restaurado[2] == estado_numpy_esperado[2]


def test_retomar_de_checkpoint_rejeita_metadata_incompativel(
    config_reduzida: IgnisModelConfig, tmp_path
):
    """Verifica, no nível de `Treinador` (não apenas de `load_checkpoint`
    isoladamente), que retomar com metadata de tokenizador incompatível
    levanta erro explícito em vez de corromper o treinamento em silêncio —
    requisito explícito do critério de aceite sobre "rejeição de
    checkpoints incompatíveis"."""
    dataset_base = _salvar_dataset_sintetico(tmp_path, "treino", config_reduzida.vocab_size, seq_len=8)

    arquivo_tok_a = tmp_path / "tok_a.json"
    arquivo_tok_a.write_text(json.dumps({"versao": "a"}), encoding="utf-8")
    arquivo_tok_b = tmp_path / "tok_b.json"
    arquivo_tok_b.write_text(json.dumps({"versao": "b"}), encoding="utf-8")

    torch.manual_seed(0)
    model = IgnisModel(config_reduzida)
    config_treino = _config_treino_rapido(tmp_path / "ckpt_meta", max_steps=2)
    metadata_original = CheckpointMetadata.construir(tokenizer_caminho=arquivo_tok_a)
    treinador = Treinador(model, config_treino, metadata_checkpoint=metadata_original)
    resultado = treinador.treinar(
        criar_loader_treino=lambda epoca: criar_dataloader(
            dataset_base, batch_size=4, embaralhar=True, seed=config_treino.seed, epoch=epoca
        ),
        val_loader=None,
    )

    model_novo = IgnisModel(config_reduzida)
    config_retomado = _config_treino_rapido(tmp_path / "ckpt_meta2", max_steps=4)
    treinador_retomado = Treinador(model_novo, config_retomado)
    metadata_incompativel = CheckpointMetadata.construir(tokenizer_caminho=arquivo_tok_b)
    with pytest.raises(ValueError, match="tokenizador"):
        treinador_retomado.retomar_de_checkpoint(
            resultado.ultimo_checkpoint, verificar_metadata=metadata_incompativel
        )


# ---------------------------------------------------------------------------
# Precisão mista (Sprint 5) -- opt-in via TreinadorConfig.mixed_precision.
# ---------------------------------------------------------------------------


def test_mixed_precision_invalido_levanta_erro_na_construcao(config_reduzida: IgnisModelConfig, tmp_path):
    model = IgnisModel(config_reduzida)
    config_treino = _config_treino_rapido(tmp_path / "ckpt", mixed_precision="fp8_invalido")
    with pytest.raises(ValueError):
        Treinador(model, config_treino)


def test_mixed_precision_fp16_em_cpu_levanta_erro_na_construcao(config_reduzida: IgnisModelConfig, tmp_path):
    """fp16 exige CUDA -- o autocast de CPU do PyTorch não suporta float16.
    Deve falhar eagerly na construção do Treinador, nunca silenciosamente
    no meio do treinamento."""
    model = IgnisModel(config_reduzida)
    config_treino = _config_treino_rapido(tmp_path / "ckpt", mixed_precision="fp16", device="cpu")
    with pytest.raises(ValueError):
        Treinador(model, config_treino)


def test_mixed_precision_bf16_em_cpu_treina_sem_erro(config_reduzida: IgnisModelConfig, tmp_path):
    """bf16 é suportado em CPU -- o treinamento deve rodar normalmente
    (mesmo comportamento funcional de 'no', só muda a precisão interna)."""
    model = IgnisModel(config_reduzida)
    dataset_base = _salvar_dataset_sintetico(tmp_path, "treino", config_reduzida.vocab_size, seq_len=8)
    config_treino = _config_treino_rapido(tmp_path / "ckpt", max_steps=3, mixed_precision="bf16")
    treinador = Treinador(model, config_treino)

    resultado = treinador.treinar(
        criar_loader_treino=lambda epoca: criar_dataloader(
            dataset_base, batch_size=4, embaralhar=True, seed=config_treino.seed, epoch=epoca
        ),
        val_loader=None,
    )

    assert resultado.motivo_parada == "max_steps_atingido"
    assert resultado.passos_executados == 3


def test_mixed_precision_padrao_e_no(config_reduzida: IgnisModelConfig, tmp_path):
    """Regressão: sem especificar mixed_precision, o padrão é 'no' -- o
    scaler fica desabilitado e o comportamento é idêntico ao de antes da
    Sprint 5."""
    model = IgnisModel(config_reduzida)
    config_treino = _config_treino_rapido(tmp_path / "ckpt")
    assert config_treino.mixed_precision == "no"
    treinador = Treinador(model, config_treino)
    assert treinador.scaler.is_enabled() is False


# ---------------------------------------------------------------------------
# Falha de memória (OOM) -- Sprint 5. Simulada via monkeypatch do forward do
# modelo (não é possível provocar um OOM real de forma confiável em CI).
# ---------------------------------------------------------------------------


def test_falha_de_memoria_no_forward_interrompe_treinamento_e_registra(
    config_reduzida: IgnisModelConfig, tmp_path, monkeypatch
):
    model = IgnisModel(config_reduzida)
    dataset_base = _salvar_dataset_sintetico(tmp_path, "treino", config_reduzida.vocab_size, seq_len=8)
    config_treino = _config_treino_rapido(tmp_path / "ckpt", max_steps=10)
    treinador = Treinador(model, config_treino)

    def _forward_com_oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB (GPU 0; 6.00 GiB total capacity)")

    monkeypatch.setattr(treinador.model, "forward", _forward_com_oom)

    from ignis.training.trainer import FalhaDeMemoriaError

    resultado = treinador.treinar(
        criar_loader_treino=lambda epoca: criar_dataloader(
            dataset_base, batch_size=4, embaralhar=True, seed=config_treino.seed, epoch=epoca
        ),
        val_loader=None,
    )

    assert resultado.motivo_parada == "falha_de_memoria"
    assert resultado.passos_executados == 0
    assert (tmp_path / "ckpt" / "falha_de_memoria.pt").exists()

    eventos = [l for l in ler_metricas(config_treino.out_dir + "/metricas.jsonl") if l.get("evento") == "falha_de_memoria"]
    assert len(eventos) == 1


def test_falha_de_memoria_propaga_quando_configurado(config_reduzida: IgnisModelConfig, tmp_path, monkeypatch):
    model = IgnisModel(config_reduzida)
    dataset_base = _salvar_dataset_sintetico(tmp_path, "treino", config_reduzida.vocab_size, seq_len=8)
    config_treino = _config_treino_rapido(tmp_path / "ckpt", max_steps=10, parar_em_falha_de_memoria=False)
    treinador = Treinador(model, config_treino)

    def _forward_com_oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory.")

    monkeypatch.setattr(treinador.model, "forward", _forward_com_oom)

    from ignis.training.trainer import FalhaDeMemoriaError

    with pytest.raises(FalhaDeMemoriaError):
        treinador.treinar(
            criar_loader_treino=lambda epoca: criar_dataloader(
                dataset_base, batch_size=4, embaralhar=True, seed=config_treino.seed, epoch=epoca
            ),
            val_loader=None,
        )


def test_falha_de_memoria_nao_e_confundida_com_falha_numerica(
    config_reduzida: IgnisModelConfig, tmp_path, monkeypatch
):
    """Regressão crítica: uma FalhaNumericaError genuína (loss NaN) nunca
    deve ser reclassificada como falha_de_memoria -- o bloco try/except de
    detecção de OOM em `_passo_de_forward_backward` não deve capturar nem
    mascarar `FalhaNumericaError`."""
    model = IgnisModel(config_reduzida)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(float("nan"))

    dataset_base = _salvar_dataset_sintetico(tmp_path, "treino", config_reduzida.vocab_size, seq_len=8)
    config_treino = _config_treino_rapido(tmp_path / "ckpt", max_steps=10)
    treinador = Treinador(model, config_treino)

    resultado = treinador.treinar(
        criar_loader_treino=lambda epoca: criar_dataloader(
            dataset_base, batch_size=4, embaralhar=True, seed=config_treino.seed, epoch=epoca
        ),
        val_loader=None,
    )

    assert resultado.motivo_parada == "falha_numerica"  # NAO "falha_de_memoria"
    assert (tmp_path / "ckpt" / "falha_numerica.pt").exists()


def test_checkpoint_de_emergencia_falhando_nao_lanca_excecao_nao_tratada(
    config_reduzida: IgnisModelConfig, tmp_path, monkeypatch
):
    """Se o próprio checkpoint de emergência falhar (ex.: disco também sem
    espaço), `treinar()` deve retornar um ResultadoTreinamento normal com
    `ultimo_checkpoint=None` e o evento registrado -- nunca propagar uma
    segunda exceção não tratada mascarando a causa raiz."""
    model = IgnisModel(config_reduzida)
    dataset_base = _salvar_dataset_sintetico(tmp_path, "treino", config_reduzida.vocab_size, seq_len=8)
    config_treino = _config_treino_rapido(tmp_path / "ckpt", max_steps=10)
    treinador = Treinador(model, config_treino)

    def _forward_com_oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory.")

    def _checkpoint_que_falha(*args, **kwargs):
        raise OSError("disco sem espaço (simulado no teste)")

    monkeypatch.setattr(treinador.model, "forward", _forward_com_oom)
    monkeypatch.setattr(treinador, "salvar_checkpoint", _checkpoint_que_falha)

    resultado = treinador.treinar(
        criar_loader_treino=lambda epoca: criar_dataloader(
            dataset_base, batch_size=4, embaralhar=True, seed=config_treino.seed, epoch=epoca
        ),
        val_loader=None,
    )

    assert resultado.motivo_parada == "falha_de_memoria"
    assert resultado.ultimo_checkpoint is None

    eventos = [
        l for l in ler_metricas(config_treino.out_dir + "/metricas.jsonl")
        if l.get("evento") == "checkpoint_de_emergencia_falhou"
    ]
    assert len(eventos) == 1


def test_para_treinador_config_propaga_mixed_precision_e_falha_de_memoria(tmp_path):
    """`PilotoConfig.para_treinador_config()` deve propagar os novos campos
    da Sprint 5 (mixed_precision, parar_em_falha_de_memoria) para o
    TreinadorConfig resultante, nao apenas os campos pre-existentes."""
    from ignis.config import PilotoConfig

    dados = {
        "modelo": {"config": "configs/ignis-30m.yaml"},
        "dados": {
            "manifesto": "data/manifesto.jsonl",
            "splits": "data/splits.json",
            "tokenizer": "data/tokenizer.json",
            "dataset_base": "data/tokenizado",
            "seq_len": 128,
        },
        "otimizacao": {"lr": 3e-4},
        "lote": {"microbatch_size": 1, "grad_accum_steps": 32},
        "orcamento": {"max_steps": 200},
        "validacao_e_checkpoint": {"eval_every": 20, "checkpoint_every": 20, "log_every": 5},
        "execucao": {
            "device": "cpu",
            "mixed_precision": "bf16",
            "parar_em_falha_de_memoria": False,
        },
    }
    piloto = PilotoConfig.model_validate(dados)
    treinador_config = piloto.para_treinador_config()

    assert treinador_config.mixed_precision == "bf16"
    assert treinador_config.parar_em_falha_de_memoria is False
