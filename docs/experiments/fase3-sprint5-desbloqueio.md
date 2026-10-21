# Fase 3 (preparação) — Sprint 5: Desbloqueio do treinamento real

Este documento registra o que foi construído e verificado no Sprint 5, cujo
objetivo foi remover os bloqueios técnicos restantes antes do primeiro
pré-treinamento piloto real (aceleração de GPU, corpus real, granularidade
de splits, gate de promoção do tokenizador, execução experimental do
treinamento). **Nenhum pré-treinamento real foi iniciado. Nenhum corpus
real foi baixado. Nenhuma arquitetura foi alterada. Nenhuma instalação de
CUDA/GPU foi realizada. Nenhum commit foi feito.**

## O que foi implementado nesta rodada

1. **Correção de granularidade de splits (fonte vs. unidade de
   agrupamento):** `EntradaManifesto` ganhou o campo
   `unidade_agrupamento`, resolvido em `ingest.py` como o nome da pasta
   imediata do arquivo, independente de `fonte` (que continua sendo um
   rótulo de proveniência/atribuição, podendo ser constante para um
   provedor inteiro). `splits.py` agora agrupa por `unidade_agrupamento`
   (com fallback para `fonte`), evitando que todos os documentos de um
   provedor grande colapsem numa única unidade indivisível de split. Nova
   função `contar_unidades_por_fonte()` e novo aviso "fonte colapsada" no
   relatório quando uma fonte tem muitos documentos mas só 1 unidade.
2. **Ingestão incremental e retomável:** `processar_diretorio()` aceita
   `manifesto_existente` e `limite_novos_documentos`, permitindo reprocessar
   apenas documentos novos/alterados (detecção por hash SHA-256) e ingerir
   diretórios grandes em lotes controlados — novas flags `--incremental` e
   `--limite-novos-documentos` em `validate-corpus`. Limitação documentada:
   a detecção de quase-duplicatas (Jaccard) não é retroativa entre
   chamadas, apenas dentro do lote de cada chamada.
3. **Gate mecânico de promoção do tokenizador (`audit-tokenizer`):** novo
   subcomando que verifica simultaneamente os 4 critérios objetivos da
   seção 8 de `docs/plano-corpus-piloto.md` (licenças documentadas, zero
   contaminação de splits, round-trip sem falhas em treino+validação+
   **externo obrigatório**, volume de tokens medido na faixa-alvo) e só
   reporta `oficial_do_piloto: true` quando todos passam — nunca promove
   parcialmente.
4. **`train --device {cpu,cuda}`:** permite sobrescrever
   `execucao.device` do YAML sem editá-lo; com `cuda`, o dry-run verifica
   `torch.cuda.is_available()` e reporta falha explícita (sem iniciar
   treinamento) se a GPU não estiver disponível.
5. **Mixed precision no `Treinador`** (`mixed_precision: "no"|"bf16"|
   "fp16"`, padrão `"no"`): implementado via `torch.autocast` +
   `torch.cuda.amp.GradScaler` (sempre construído, mas como no-op quando
   desativado); validação eager rejeita `fp16` fora de `cuda` (autocast
   fp16 não é suportado em CPU).
6. **Tratamento de falta de memória (OOM) como parada controlada:** nova
   exceção `FalhaDeMemoriaError`, detectada em cada fronteira
   forward/backward/unscale/clip/optimizer-step, convertida em parada
   controlada com checkpoint de emergência best-effort — paralelo ao
   tratamento já existente de `FalhaNumericaError`. Durante a implementação,
   uma revisão manual cuidadosa (não poderia ser pega por execução de
   teste, já que torch não está disponível neste ambiente de
   desenvolvimento) encontrou e corrigiu um bug real: se o checkpoint de
   emergência também falhasse, o código cairia num segundo salvamento de
   checkpoint sem proteção (herdado do tratamento de `FalhaNumericaError`
   pré-existente), que provavelmente falharia de novo, desta vez sem
   captura — corrigido com uma flag `falha_tratada`.
7. **Scripts de preparação para CUDA** (nenhum executa instalação sem
   confirmação explícita):
   - `scripts\verificar_cuda_disponivel.ps1` — descoberta empírica e
     somente leitura de quais tags de build CUDA do torch estão
     instaláveis agora (`pip index versions`), usando o driver/compute
     capability reais desta máquina via `nvidia-smi`.
   - `scripts\preparar_venv_cuda.ps1` — instalação gated (`-Confirmar` +
     `-IndexUrl` obrigatórios) de um `.venv-cuda` separado, nunca
     substituindo o `.venv` CPU atual.
   - `scripts\verificar_gpu_cuda.py` — verificação funcional real
     (`torch.cuda.is_available()`, alocação de tensor, forward/backward
     real, benchmark CPU-vs-GPU).
   Ver `docs/plano-migracao-cuda.md` para o plano completo e o fluxo de
   autorização em dois passos.
8. **Documentação atualizada:** `docs/plano-corpus-piloto.md` (seção sobre
   fonte vs. unidade de agrupamento, ingestão incremental, referência ao
   `audit-tokenizer`, nota de limitação de pesquisa — seção 8.1) e
   `docs/plano-execucao-piloto.md` (passos 1, 1.1, 2, 3, 4 e 6 atualizados
   para refletir as novas ferramentas e pontos de autorização).

## Evidência de execução real (Sprint 5)

Executado nesta sessão, na VM Linux de desenvolvimento (sem torch — ver
`docs/experiments/fase1-fundacao.md` para a justificativa do ambiente
dual):

```
pytest -q --ignore=tests/test_causal_mask.py --ignore=tests/test_checkpoint.py \
         --ignore=tests/test_generation.py --ignore=tests/test_gradients.py \
         --ignore=tests/test_model_shapes.py --ignore=tests/test_overfit_smoke.py \
         --ignore=tests/test_param_count.py --ignore=tests/test_tied_embeddings.py \
         --ignore=tests/test_training_checkpoint.py --ignore=tests/test_training_dataloader.py \
         --ignore=tests/test_training_trainer.py
...
98 passed, 2 skipped in 1.17s
```

Esses 98 testes cobrem toda a superfície testável sem torch: pipeline de
dados (ingestão incremental, unidade de agrupamento, PII, dedup, qualidade),
splits (granularidade, `contar_unidades_por_fonte`), tokenizador (treino,
avaliação, round-trip, `audit-tokenizer`), config (`PilotoConfig`,
`mixed_precision`/`parar_em_falha_de_memoria` configuráveis e validados),
e o CLI `train` (dry-run, override `--device`, sem nenhum teste que exija
um forward real). Os 2 testes pulados já eram pulados antes deste sprint
(`pytest.importorskip("torch")`). Partindo da baseline de 67 passados/2
pulados ao final do Sprint 4, o Sprint 5 adicionou 31 testes novos
(`test_unidade_agrupamento.py`: 8; `test_cli_distribuicao_por_fonte.py`:
4; `test_cli_audit_tokenizer.py`: 5; `test_data_pipeline_incremental.py`:
6; `test_cli_validate_corpus_incremental.py`: 3; `test_cli_train.py`: +3;
`test_config_piloto.py`: +2) — zero regressões em qualquer ponto.

Também confirmado por execução real nesta sessão: `diagnostico_hardware_resultado.json`
já existia no repositório, gerado por uma execução real anterior de
`scripts\diagnostico_hardware.ps1` no Windows nativo do usuário em
2026-10-18, confirmando hardware real (RTX 4050 Laptop GPU, ~6 GB VRAM,
driver 610.91, compute capability 8.9, torch 2.14.1+cpu) — usado como base
factual de `docs/plano-migracao-cuda.md`.

## Atualização (2026-10-18, mesmo dia): CUDA instalado e validado com sucesso

Após o fechamento inicial deste relatório, o usuário executou os dois
passos pendentes de CUDA diretamente em sua máquina:

1. `scripts\verificar_cuda_disponivel.ps1` — confirmou `cu118`, `cu124`,
   `cu126` e `cu128` como instaláveis; `cu126` identificada como a melhor
   escolha por publicar exatamente a versão de torch já em uso (2.14.1).
2. Com autorização explícita, `scripts\preparar_venv_cuda.ps1 -Confirmar
   -IndexUrl https://download.pytorch.org/whl/cu126` — criou `.venv-cuda`
   (sem alterar o `.venv` CPU) e rodou `verificar_gpu_cuda.py`
   automaticamente, com resultado real: `torch.cuda.is_available()=True`,
   alocação de tensor na GPU OK, forward/backward real do `IgnisModel` OK
   (loss finito), comparação CPU-vs-GPU em matmul 2048x2048 com aceleração
   de 11,3x (CPU 0,6415s / GPU 0,0567s) — código de saída 0.

Isto conclui a AUTORIZAÇÃO 1. Detalhe completo em
`docs/plano-migracao-cuda.md`, seção 6, e `docs/checklist-autorizacao-sprint5.md`.
**Ainda não há, porém, um benchmark real do Ignis-30M completo em GPU**
(o teste acima usa uma config mínima de verificação, não o modelo oficial
de 30,4M parâmetros, nem mede VRAM, mixed precision ou um passo de
otimizador completo) — ver Sprint 6.

## O que ainda NÃO foi verificado por execução real

- As ~10 novas asserções torch-dependentes de `tests/test_training_trainer.py`
  (mixed precision construindo/rejeitando corretamente, `FalhaDeMemoriaError`
  detectada e não confundida com `FalhaNumericaError`, checkpoint de
  emergência e a correção do bug de double-fallback) **nunca rodaram com
  torch instalado** — escritas e verificadas apenas por `ast.parse`
  (sintaxe válida) e por revisão manual cuidadosa de corretude. Pendente
  de execução em `.venv` (CPU) ou `.venv-cuda` no Windows nativo.
- Nenhum benchmark real do Ignis-30M completo (30.416.384 parâmetros) foi
  executado em GPU — apenas a verificação funcional mínima da seção
  anterior. `scripts/benchmark_training.py --device cuda` ainda não foi
  rodado.
- Nenhuma fonte de corpus real foi baixada, validada ou declarada aceita;
  a tabela de candidatas em `docs/plano-corpus-piloto.md` não foi
  reverificada por pesquisa ao vivo nesta sessão (ferramenta de busca
  indisponível para esta organização — ver seção 8.1 daquele documento).
- `python -m ignis.cli train` contra `configs/train-pilot.yaml` continua
  reportando `pronto_para_treinamento_real = False`, pois
  `data/manifesto.jsonl`/`data/splits.json`/`data/tokenizer.json`/
  `data/tokenizado_*` continuam não existindo neste repositório — exatamente
  como esperado antes de qualquer corpus real.

## Conclusão desta etapa

Todos os bloqueios técnicos da Sprint 5 foram resolvidos: a parte
torch-free foi verificada por execução real (98 passed/2 skipped), e a
parte de CUDA foi não só preparada como efetivamente instalada e validada
com sucesso pelo usuário. O único item que falta antes do primeiro piloto
real é a AUTORIZAÇÃO 2 — aquisição limitada de corpus real — descrita em
`docs/checklist-autorizacao-sprint5.md`. A suíte completa de testes
(incluindo os torch-dependentes) ainda precisa ser rodada no Windows
nativo, e um benchmark real do Ignis-30M em GPU ainda precisa ser feito —
ambos sem depender de nenhuma autorização adicional (ver Sprint 6 em
`docs/experiments/` quando aberto).
