# Plano de execução do primeiro piloto real — Ignis-30M

Este documento sequencia os passos concretos entre o estado atual do
projeto (infraestrutura pronta, nenhum dado real, nenhum treinamento
iniciado) e o primeiro pré-treinamento piloto real. Cada passo indica qual
comando executar, o que verificar antes de prosseguir, e onde uma
autorização explícita separada é exigida antes de continuar — ver
`docs/plano-corpus-piloto.md` e a skill `ignis-llm-engineer` para as regras
que já valem independentemente deste documento.

Nenhum passo abaixo foi executado. Este é um plano, não um registro de
execução.

## Passo 0 — Pré-requisitos já concluídos (Sprints 1-4)

- Arquitetura do Ignis-30M implementada e validada (30.416.384 parâmetros
  confirmados por execução real).
- Pipeline de dados (ingestão, triagem, dedup, PII, splits, relatório com
  distribuição por fonte) implementado e testado com corpus sintético.
- Tokenizador BPE (código) e CLI de treino/avaliação implementados e
  testados com corpus sintético.
- Infraestrutura de treinamento (`Treinador`, checkpoints, retomada,
  benchmark com distinção microbatch/passo de otimizador) implementada e
  testada com modelo reduzido e dados sintéticos.
- CLI `ignis.cli train` com dry-run obrigatório implementado.

## Passo 1 — Diagnóstico de hardware e descoberta de CUDA (sem autorização adicional necessária)

```powershell
.\scripts\diagnostico_hardware.ps1
.\scripts\verificar_cuda_disponivel.ps1
```

Resultado esperado: confirmação de CPU/RAM disponíveis, GPU detectada,
driver/compute capability, e quais tags de build CUDA do torch estão
empiricamente instaláveis agora (consulta de rede somente leitura, nunca
instala nada). Nenhum dos dois comandos requer autorização adicional.

### Passo 1.1 — AUTORIZAÇÃO CUDA (opcional, só se a GPU for utilizada)

**CONCLUÍDO em 2026-10-18.** O usuário autorizou e executou este passo:
`.venv-cuda` criado com torch 2.14.1+cu126, GPU (RTX 4050) confirmada
funcional (`torch.cuda.is_available()=True`, forward/backward real OK,
aceleração de 11,3x em matmul). Ver `docs/plano-migracao-cuda.md`, seção 6,
para o resultado completo. `.venv` CPU original inalterado. `--device cuda`
já pode ser usado a partir de agora em `ignis.cli train` e em
`scripts/benchmark_training.py`.

Comando usado (referência, não precisa ser repetido):
```powershell
.\scripts\preparar_venv_cuda.ps1 -Confirmar -IndexUrl https://download.pytorch.org/whl/cu126
```

## Passo 2 — AUTORIZAÇÃO 1: seleção e ingestão do corpus real

**Ponto de parada: requer autorização explícita e específica do usuário,
citando a(s) fonte(s) aprovada(s) de `docs/plano-corpus-piloto.md`.**

Após autorização, para cada fonte aprovada:

1. Obter o texto bruto (download manual ou ferramenta própria da fonte).
2. Converter para `.jsonl` com `id`, `texto`, `licenca`,
   `permissao_treinamento` preenchidos corretamente para aquela fonte.
3. `python -m ignis.cli validate-corpus --corpus-dir <dir> --fonte <nome-estavel> --manifesto-saida data/manifesto.jsonl --incremental --limite-novos-documentos <N>`
   — para um provedor GRANDE, organizar `<dir>` em várias subpastas
   (shards) sob o mesmo `--fonte`, nunca tudo em uma pasta só (ver
   `docs/plano-corpus-piloto.md`, seção "Provedor vs. unidade de
   agrupamento"); `--incremental`/`--limite-novos-documentos` permitem
   ingerir em lotes controlados, retomando de onde parou.
4. Inspecionar manualmente uma amostra de documentos `aceito`, `rejeitado`
   e `quarentena` (nunca confiar cegamente nos filtros automáticos).

## Passo 3 — Splits e auditoria de distribuição

```bash
python -m ignis.cli prepare-splits --manifesto data/manifesto.jsonl --splits-saida data/splits.json
python -m ignis.cli report --manifesto data/manifesto.jsonl --splits data/splits.json --saida data/relatorio_pre_tokenizador.json
```

Verificar no relatório: zero contaminação entre splits, proporções
realizadas razoáveis, **nenhum aviso em `distribuicao_por_fonte.avisos_concentracao`**
(ou, se houver aviso, decidir e documentar explicitamente por que o corpus
ainda assim é aceitável, antes de prosseguir), e que nenhuma fonte com
muitos documentos tenha `n_unidades_agrupamento == 1` (sinal de que ela
está colapsada em uma única unidade indivisível — ver
`docs/plano-corpus-piloto.md`, seção "Provedor vs. unidade de
agrupamento").

## Passo 4 — Tokenizador BPE oficial

```bash
python -m ignis.cli train-tokenizer --corpus-dir <dir> --manifesto data/manifesto.jsonl --splits data/splits.json --vocab-size 16384 --saida data/tokenizer.json
python -m ignis.cli eval-tokenizer --tokenizer data/tokenizer.json --corpus-dir <dir> --manifesto data/manifesto.jsonl --splits data/splits.json --textos-externos <arquivo_textos_fora_do_corpus.txt> --saida data/avaliacao_tokenizer.json
```

Após `prepare-dataset` (Passo 5), rodar a auditoria mecânica antes de
declarar o tokenizador "oficial do piloto":

```bash
python -m ignis.cli audit-tokenizer --tokenizer data/tokenizer.json --corpus-dir <dir> --manifesto data/manifesto.jsonl --splits data/splits.json --textos-externos <arquivo_textos_fora_do_corpus.txt> --dataset-base data/tokenizado --saida data/auditoria_tokenizer.json
```

Só prosseguir se `oficial_do_piloto` for `true` no relatório (ver
`docs/plano-corpus-piloto.md`, seção 8, para os 4 critérios exatos
verificados) — o comando bloqueia a promoção, nunca aprova parcialmente.

## Passo 5 — Dataset tokenizado e benchmark real

```bash
python -m ignis.cli prepare-dataset --tokenizer data/tokenizer.json --corpus-dir <dir> --manifesto data/manifesto.jsonl --splits data/splits.json --saida-base data/tokenizado --seq-len 128
```

```powershell
.\scripts\benchmark_training.py --seq-lens 32 128 256 --grad-accum-steps 32 --n-steps-piloto 200 --seq-len-piloto 128 --saida benchmark_piloto.json
# com GPU (apos a AUTORIZACAO CUDA do Passo 1.1) e, opcionalmente, mixed precision (Sprint 6):
.venv-cuda\Scripts\python.exe scripts\benchmark_training.py --device cuda --mixed-precision fp16 --seq-lens 32 128 256 --grad-accum-steps 32 --n-steps-piloto 200 --seq-len-piloto 128 --saida benchmark_piloto_gpu.json
```

Em `--device cuda`, o relatório também inclui VRAM alocada/reservada e o
pico de cada uma por seq_len; se alguma combinação estourar a memória da
GPU, ela é marcada como `"oom": true` no relatório e o script continua com
as demais combinações, em vez de travar.

Revisar a estimativa de tempo do piloto (`estimativa_piloto`, baseada em
`throughput_passo_otimizador`, nunca em `throughput_microbatch`) antes de
prosseguir — se o tempo estimado for inaceitável, ajustar `max_steps`/
`grad_accum_steps`/`seq_len` em `configs/train-pilot.yaml` e medir de novo.

## Passo 6 — Dry-run obrigatório do piloto real

```bash
python -m ignis.cli train --config configs/train-pilot.yaml --saida dry_run_piloto.json
# ou, para sobrescrever o device sem editar o YAML (ex.: testar em cpu
# antes de usar cuda, ou vice-versa, apos a AUTORIZACAO CUDA do Passo 1.1):
python -m ignis.cli train --config configs/train-pilot.yaml --device cuda --saida dry_run_piloto.json
```

Revisar `dry_run_piloto.json`: `pronto_para_treinamento_real` deve ser
`true`, com `forward_de_teste_sem_atualizar_pesos` e
`contagem_parametros_confere_com_ignis_30m` ambos `true`. Qualquer checagem
falha deve ser corrigida antes de prosseguir — este passo nunca atualiza
nenhum peso, independentemente do resultado.

## Passo 7 — AUTORIZAÇÃO 2: início do treinamento real

**Ponto de parada: requer autorização explícita e separada do usuário para
iniciar o treinamento real, mesmo com o dry-run aprovado no Passo 6.** Esta
autorização deve ocorrer DEPOIS de revisar o relatório de dry-run e a
estimativa de tempo do Passo 5 — nunca antes.

```bash
python -m ignis.cli train --config configs/train-pilot.yaml --iniciar-treinamento-real
```

Durante e após a execução: registrar curva de loss treino/validação,
perplexidade, tokens/s, RAM, tempo total, seed, commit do git, hash do
corpus/tokenizador (já gravados automaticamente no checkpoint via
`CheckpointMetadata`) em um novo documento de experimento
(`docs/experiments/fase3-piloto-real.md` ou nome equivalente), com os
resultados efetivamente observados — nunca estimados e apresentados como
medidos.

## Passo 8 — Avaliação (sem usar o split de teste ainda)

Avaliar apenas sobre o split de validação até que todas as decisões de
hiperparâmetros estejam finalizadas. O split de teste só é usado uma única
vez, no final, para o relatório definitivo do piloto — nunca durante
iteração de desenvolvimento (regra já vigente, repetida aqui por ênfase).
