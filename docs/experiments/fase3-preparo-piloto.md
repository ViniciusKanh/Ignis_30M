# Fase 3 (preparação) — Sprint 4: Corpus real, tokenizador oficial e preparação do piloto

Este documento registra o que foi construído e verificado no Sprint 4, que
teve como objetivo preparar (não executar) o primeiro pré-treinamento
científico real do Ignis-30M. **Nenhum pré-treinamento real foi iniciado.
O modelo permanece com pesos de inicialização aleatória.** Nenhum corpus
real foi baixado. Nenhuma arquitetura foi alterada.

## O que foi implementado nesta rodada

1. **Diagnóstico de hardware (`scripts/diagnostico_hardware.ps1`):** script
   PowerShell somente leitura que coleta CPU, RAM, GPUs detectadas pelo
   Windows (`Win32_VideoController`), saída de `nvidia-smi` quando
   disponível, e o que o `torch` já instalado no projeto enxerga via
   `torch.cuda`. Produz uma recomendação textual (nunca uma instalação
   automática) e salva `diagnostico_hardware_resultado.json`. Não foi
   executado neste ambiente (requer PowerShell no Windows nativo do
   usuário) — **pendente de execução real pelo usuário**.
2. **Lista de fontes candidatas de corpus real** (`docs/plano-corpus-piloto.md`,
   seção 1): tabela com nome, URL, licença declarada, evidência de
   permissão, categoria, volume estimado e limitações para Wikipédia em
   português, Domínio Público (FBN/MinC), Wikisource em português, portais
   governamentais brasileiros, Corpus Carolina (IME-USP/C4AI) e textos
   produzidos pela própria organização. Nenhuma fonte foi aprovada para
   ingestão; nenhum download foi realizado.
3. **Auditoria científica do pipeline de dados** (`ignis.cli report`):
   adicionado `distribuicao_por_fonte` ao relatório consolidado — contagem
   de documentos por fonte e por status de triagem, proporção de cada
   fonte no total de documentos aceitos, e avisos automáticos quando uma
   única fonte concentra mais de 50% do corpus aceito (ou quando só existe
   uma fonte), para que uma validação cientificamente frágil (corpus pouco
   diverso) seja visível no relatório, não descoberta só depois do
   treinamento. Verificado com o corpus sintético de 7 fontes
   (`scripts/gerar_corpus_sintetico.py`): nenhum aviso de concentração
   disparado, como esperado (fontes balanceadas).
4. **Avaliação de tokenizador em textos externos** (`ignis.cli eval-tokenizer
   --textos-externos`): nova opção que avalia round-trip e eficiência sobre
   um conjunto de textos explicitamente fora de qualquer split, reportado
   separadamente (`avaliacao_externa`) do resultado interno — suporte para
   detectar overfitting do vocabulário do BPE, ainda sem nenhum texto
   externo "oficial" definido.
5. **Benchmark com distinção microbatch vs. passo de otimizador**
   (`scripts/benchmark_training.py --grad-accum-steps N`): o benchmark agora
   mede e reporta separadamente `throughput_microbatch` (tokens/s de um
   único forward+backward, sem o custo do `optimizer.step()`) e
   `throughput_passo_otimizador` (tokens/s de um passo de otimizador
   completo — `grad_accum_steps` microbatches + um `optimizer.step()`, lote
   efetivo = `batch_size * grad_accum_steps`), explicitamente documentando
   que são métricas diferentes e que a estimativa de tempo de piloto deve
   usar sempre a segunda. Campos antigos (`tempo_medio_forward_s` etc.)
   foram preservados com a mesma semântica de antes para não quebrar
   relatórios já existentes.
6. **CLI de treinamento com dry-run obrigatório** (`ignis.cli train`): novo
   subcomando que SEMPRE roda primeiro uma verificação completa (arquivos
   de manifesto/splits/tokenizer/dataset existem; shapes e contagem de
   sequências dos splits; vocabulário do tokenizador contra o
   `vocab_size` oficial; contagem de parâmetros do modelo oficial
   (30.416.384, confirmada) contra a esperada; estimativa heurística de
   memória; um forward de teste real contra o modelo oficial, SEM
   `backward()` e SEM `optimizer.step()`) e só inicia o treinamento real
   se **todas** as checagens passarem **e** a flag
   `--iniciar-treinamento-real` for explicitamente passada. Sem essa flag,
   o comando sempre para após o relatório JSON de dry-run — este
   comportamento é a implementação em código da regra de autorização
   explícita da skill `ignis-llm-engineer`.

## Evidência de execução real (Sprint 4)

Executado nesta sessão, na VM Linux de desenvolvimento (sem torch — ver
`docs/experiments/fase1-fundacao.md`):

```
pytest tests/test_data_*.py tests/test_tokenizer_*.py tests/test_config_piloto.py \
       tests/test_config.py tests/test_cli_train.py -v
...
67 passed, 2 skipped in 1.94s
```

Os 2 testes pulados (`test_train_integrado_*` em `tests/test_cli_train.py`)
dependem de `torch` (instanciam o modelo e rodam um forward real) e são
pulados automaticamente via `pytest.importorskip("torch")` — **ainda não
executados em nenhum ambiente com torch instalado**. O subcomando `train`
também foi exercitado manualmente nesta VM (sem torch) em cenários de
arquivos ausentes, confirmando que ele relata `pronto_para_treinamento_real
= False` corretamente sem lançar exceção, e que `--iniciar-treinamento-real`
nunca inicia o treinamento quando alguma checagem falha (verificado com
monkeypatch de `_executar_treinamento_real`, que não é chamada).

Também executado nesta VM: `ignis.cli report` sobre o corpus sintético
padrão (7 fontes, 30 documentos, 25 aceitos) — `distribuicao_por_fonte`
calculada corretamente, sem falsos avisos de concentração.

## O que ainda NÃO foi verificado por execução real

- `scripts/diagnostico_hardware.ps1` nunca foi executado (precisa do
  PowerShell do Windows nativo do usuário).
- Os 2 testes `test_train_integrado_*` (forward real contra o modelo
  oficial de 30,4M parâmetros) nunca rodaram com torch instalado.
- `scripts/benchmark_training.py --grad-accum-steps` nunca foi executado
  de verdade (requer torch no Windows nativo).
- Nenhuma fonte de corpus real foi baixada, validada ou declarada aceita.
- Nenhum tokenizador BPE "oficial" (16.384 tokens, corpus real) foi
  treinado — o único tokenizador treinado até agora usa corpus sintético.
- O piloto real (`configs/train-pilot.yaml`) continua apontando para
  `data/manifesto.jsonl`/`data/splits.json`/`data/tokenizer.json`/
  `data/tokenizado_*`, nenhum dos quais existe neste repositório — rodar
  `python -m ignis.cli train` hoje produz `pronto_para_treinamento_real =
  False` por ausência desses arquivos, exatamente como esperado.

## Conclusão desta etapa

A INFRAESTRUTURA para o primeiro piloto real está pronta e parcialmente
verificada; os DADOS para o piloto (corpus real, tokenizador oficial,
dataset tokenizado real) ainda não existem. Ver
`docs/plano-execucao-piloto.md` para a sequência completa de passos e
pontos de autorização necessários antes do primeiro treinamento real.
