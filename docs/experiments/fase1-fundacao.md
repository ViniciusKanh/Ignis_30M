# Fase 1 — Evidências de execução

Todos os comandos abaixo foram executados pelo usuário (Vinicius Santos) no
Python nativo do Windows, em `E:\Models\Ignis\.venv` — não na VM Linux da
ponte (ver `docs/limitations.md` para por que a execução acontece lá e não
aqui). As saídas foram coladas integralmente, sem edição.

## Ambiente

- Python 3.13.13 (Windows nativo)
- PyTorch 2.14.1+cpu, instalado via `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- `torch.cuda.is_available() == False` (confirmado)
- Pacote `ignis` instalado em modo editável (`pip install -e ".[dev]"`)

## Contagem de parâmetros (scripts/count_params.py)

```
Configuração: ignis-30m
Parâmetros esperados (derivação analítica): 30.416.384
Parâmetros reais (sum(p.numel()) do modelo): 30.416.384
Coincidem: True
```

A contagem real do modelo instanciado bate exatamente com a derivação
analítica documentada em `docs/architecture.md` — **30.416.384 parâmetros
treináveis**, confirmados por execução real, não apenas por cálculo.

## Suíte de testes automatizados (pytest)

```
platform win32 -- Python 3.13.13, pytest-9.1.1, pluggy-1.7.0
collected 17 items

tests/test_causal_mask.py::test_isolamento_causal PASSED
tests/test_checkpoint.py::test_checkpoint_round_trip PASSED
tests/test_config.py::test_config_canonica_ignis_30m PASSED
tests/test_config.py::test_config_yaml_bate_com_canonica PASSED
tests/test_config.py::test_config_invalida_d_model_nao_divisivel_por_n_heads PASSED
tests/test_config.py::test_config_invalida_n_heads_nao_divisivel_por_n_kv_heads PASSED
tests/test_config.py::test_config_invalida_head_dim_vezes_n_heads_diferente_de_d_model PASSED
tests/test_generation.py::test_geracao_greedy_shape_e_prefixo PASSED
tests/test_generation.py::test_geracao_amostrada_com_eos PASSED
tests/test_gradients.py::test_gradientes_finitos PASSED
tests/test_model_shapes.py::test_forward_shape PASSED
tests/test_model_shapes.py::test_eval_mode_deterministico PASSED
tests/test_overfit_smoke.py::test_overfit_controlado_modelo_reduzido PASSED
tests/test_param_count.py::test_contagem_parametros_ignis_30m PASSED
tests/test_param_count.py::test_contagem_parametros_config_reduzida PASSED
tests/test_tied_embeddings.py::test_embeddings_compartilhados_mesma_instancia PASSED
tests/test_tied_embeddings.py::test_contagem_nao_duplica_embedding_compartilhado PASSED

17 passed in 10.64s
```

Nenhum teste falhou. Nenhum teste foi pulado ou marcado como esperado-falhar.
Os testes `test_overfit_smoke.py` e os de `config_reduzida` usam uma
configuração reduzida (vocab_size=64, d_model=32, 2 camadas) que reutiliza
exatamente os mesmos componentes arquiteturais do Ignis-30M — isso agiliza a
suíte, mas não substitui a verificação do modelo na configuração real de
30,4M parâmetros, feita separadamente pelo smoke test abaixo.

## Smoke test do Ignis-30M completo (30,4M parâmetros)

Executado pelo usuário no mesmo ambiente Windows nativo (Python 3.13.13,
torch 2.14.1+cpu), via:

```
python scripts/smoke_test_full.py --batch-size 1 --seq-len 32 --repeticoes 3
```

**Importante: isto NÃO é treinamento.** O script instancia o Ignis-30M
completo (30.416.384 parâmetros), executa forward → cross_entropy → backward
em dados sintéticos, e verifica que logits, loss e gradientes são finitos.
Em nenhum momento `optimizer.step()` é chamado — nenhum peso é atualizado.

### Resultados observados (3 repetições)

| Repetição | Loss (cross-entropy) | Tempo forward (s) | Tempo backward (s) |
|-----------|----------------------|--------------------|---------------------|
| 1         | 9.9624                | 0.024              | 0.043               |
| 2         | 9.8943                | 0.016              | 0.037               |
| 3         | 9.8281                | 0.014              | 0.028               |

- **Tempo médio de forward:** 0.018 s (batch=1, seq_len=32, CPU)
- **Tempo médio de backward:** 0.036 s (batch=1, seq_len=32, CPU)
- **RAM do processo antes:** 353,2 MB
- **RAM do processo depois:** 561,5 MB
- **Delta de RAM:** 208,3 MB (carregamento do modelo + grafo de autograd de
  uma única iteração)
- **Repetições com sucesso:** 3/3
- **Loss finita em todas as repetições:** sim
- **Gradientes finitos em todas as repetições:** sim (verificado para todos
  os parâmetros treináveis)
- **`optimizer.step()` executado:** não — nenhuma atualização de peso
  ocorreu; isto confirma apenas que o grafo computacional do modelo completo
  é numericamente estável em uma iteração, não que o modelo aprende algo.

### Limitações explícitas desta medição

- Medição com `batch_size=1` e `seq_len=32`, muito abaixo do `max_seq_len=1024`
  da configuração canônica — os tempos e o uso de RAM aqui **não** devem ser
  extrapolados linearmente para lotes maiores ou para a janela de contexto
  completa; uma medição de performance real do pipeline de treinamento
  (Fase 3) precisará repetir isso com os tamanhos de lote e sequência
  efetivamente planejados.
- Execução em CPU única, sem otimizações de kernel (sem `torch.compile`,
  sem fusão de operadores); não representa o desempenho esperado de uma
  GPU, caso uma venha a ser usada futuramente.
- Uma única execução de 3 repetições em uma máquina; não há replicação em
  múltiplas máquinas ou sementes diferentes.


## Encerramento formal da Fase 1

Com base nas evidências acima, a Fase 1 (Fundação e smoke tests) é
considerada **tecnicamente concluída**, conforme os critérios de aceite da
skill `ignis-llm-engineer`:

| Critério de aceite (skill) | Status | Evidência |
|---|---|---|
| Testes automatizados aprovados em CPU | ✅ Atendido | 17/17 testes `pytest`, Python 3.13.13 + torch 2.14.1+cpu, Windows nativo, CPU |
| Quantidade de parâmetros verificada e documentada | ✅ Atendido | 30.416.384 (analítico) == 30.416.384 (`sum(p.numel())`), `scripts/count_params.py` |
| Máscara causal aprovada por teste de alteração de tokens futuros | ✅ Atendido | `tests/test_causal_mask.py::test_isolamento_causal` |
| Forward/backward com gradientes finitos | ✅ Atendido | 17 testes unitários (`config_reduzida`) + smoke test do modelo completo (30,4M) com 3/3 repetições |
| Um pequeno overfit controlado demonstrado ou uma falha explicada | ✅ Atendido | `tests/test_overfit_smoke.py` (300 passos, modelo reduzido, loss final < 1.0) |
| Geração de tokens executável sem internet | ✅ Atendido | `tests/test_generation.py` (greedy e amostrada), execução local, sem chamadas de rede |
| Logs e experimentos reproduzíveis, com limitações honestas | ✅ Atendido | Este documento + `docs/limitations.md`, seed fixa (42), ambiente documentado |
| Modelo, documentação e versionamento coerentes | ✅ Atendido | `docs/architecture.md`, `configs/ignis-30m.yaml`, estrutura `src/ignis/` |

### O que a Fase 1 **não** estabelece (limites explícitos)

- O modelo **não foi treinado**. Os pesos permanecem na inicialização
  aleatória (`_init_weights`, normal std=0.02). Nenhuma capacidade de
  geração útil, compreensão de linguagem ou conhecimento foi demonstrada
  ou deve ser presumida.
- O smoke test de overfit (`test_overfit_smoke.py`) usa uma **configuração
  reduzida** (vocab_size=64, d_model=32, 2 camadas) e um microcorpus
  sintético fixo — demonstra que o pipeline de treinamento é numericamente
  funcional, não que o Ignis-30M completo memoriza ou generaliza.
- O smoke test do modelo completo (30,4M parâmetros) cobre apenas
  forward/backward em `batch=1, seq_len=32`, sem otimização — não mede
  throughput de treinamento real nem estabilidade ao longo de muitas
  iterações.
- Nenhum dado real de treinamento foi usado nesta fase; todos os testes
  usam dados sintéticos ou microcorpora de exemplo.

### Decisão operacional registrada: execução dual de ambiente

Por restrição de rede específica da VM Linux da ponte (bloqueio ao domínio
`download.pytorch.org` nesse ambiente, ver `docs/limitations.md`), e pela
política do projeto de nunca instalar a distribuição padrão do PyPI para
`torch` (que traz a stack completa de dependências CUDA, violando a regra
de "sem CUDA"), ficou estabelecido que:

- A **VM Linux** (ponte `device_bash`) é usada para edição de código,
  testes que não dependem de `torch` (ex.: `tokenizers`, `pydantic`,
  pipeline de dados) e inspeção geral do repositório.
- A **execução real de qualquer código que dependa de `torch`** (modelo,
  testes de arquitetura, smoke tests, e futuramente treinamento/inferência)
  acontece no **Python nativo do Windows** (`E:\Models\Ignis\.venv`,
  Python 3.13.13, torch 2.14.1+cpu), operado diretamente pelo usuário.

Essa decisão é registrada aqui como parte da documentação de ambiente do
projeto e deve ser mantida até que uma alternativa (ex.: liberação de rede
na VM, ou uso de outra máquina Linux com acesso a `download.pytorch.org`)
seja validada.
