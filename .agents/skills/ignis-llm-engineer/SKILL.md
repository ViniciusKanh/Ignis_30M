---
name: ignis-llm-engineer
description: Implementar, corrigir, testar e avaliar o Ignis-30M em PyTorch, incluindo corpus, tokenização BPE, CUDA, pré-treinamento, inferência e rigor científico. Usar em tarefas de engenharia do repositório Ignis.
---

# Ignis LLM Engineer — Codex

## Quando usar

Ative esta skill para desenvolvimento, revisão, depuração ou análise científica do Ignis-30M: arquitetura Transformer, pipelines de dados e licenças, tokenizador, splits, treino, CUDA, checkpoints, geração, RAG futuro e comparação de hardware.

Não usar para tarefas sem relação com o Ignis. Não confundir assistência de engenharia com autorização para downloads, instalações ou treinamento.

## Identidade e invariantes

- Ignis é uma família experimental de **SLMs próprios, treinados do zero**, com prioridade de linguagem geral em português brasileiro; programação é prioridade técnica secundária. RAG não faz parte da arquitetura base.
- Inspiração no Qwen3 é **conceitual/arquitetural**: não há importação de pesos, reprodução exata de arquitetura, nem equivalência de desempenho.
- Ignis-30M aprovado: decoder-only; 7 blocos; `d_model=512`; `n_heads=8`, `n_kv_heads=4`, `head_dim=64`; SwiGLU `intermediate_size=1536`; RMSNorm, RoPE, causal GQA; vocabulário `16384`; `max_seq_len=1024`; embeddings empatados; sem vieses lineares no baseline; **30.416.384 parâmetros**.
- Não alterar a arquitetura sem pedido explícito, cálculo de impacto, registro de decisão e novos testes estruturais.
- Estado conhecido em 18–20/10/2026: corpus piloto real da Wikipédia em português sendo preparado; **modelo de 30M ainda não pré-treinado**. Sempre validar contra arquivos/resultados mais recentes.

## Protocolo de execução — rápido e verificável

1. **Contexto mínimo.** Confirmar a pasta/repositório (`E:\Models\Ignis` quando local), `git status --short`, sistema operacional, e arquivos relevantes. Ler `AGENTS.md`, relatório mais recente e somente os módulos/testes necessários. Não reler todos os PDFs em cada tarefa.
2. **Escopo.** Distinguir claramente decisão aprovada, hipótese, estimativa e medição. Se o pedido exigir autorização adicional, executar apenas a parte segura e explicar o bloqueio específico.
3. **Implementação focada.** Corrigir causas raiz, preservar APIs/esquemas existentes e criar pequenos testes de regressão. Não reimplementar componentes que passaram em testes reais.
4. **Verificação econômica.** Executar testes específicos uma vez após o lote; corrigir; executar suíte inteira uma vez antes de entregar. Benchmark só quando relevante; GPU sincronizada antes/depois das medições.
5. **Relato curto.** Arquivos tocados, testes/ambiente, números observados, pendências, comando seguinte. Nunca chamar ausência de erro de evidência de qualidade do modelo.

## Autorizações e preservação

- **Sem aprovação específica:** não instalar pacotes, trocar PyTorch, baixar fontes novas, ampliar cotas de download, realizar treinamento real de pesos, exportar/publicar dados ou pesos, criar commits ou pushes.
- A coleta **historicamente autorizada** é Wikipédia em português, meta aproximada de 1 milhão de tokens e limite de 500 MB usado no coletor; **verificar estado e demais limites reais antes de retomar**, inclusive possíveis divergências entre armazenamento e tráfego. Nenhuma fonte extra está aprovada.
- Não fazer requisições HTTP por padrão durante depuração: preferir fixtures e corpus local. Não contornar HTTP 429, bloqueios, `Retry-After` ou regras de API.
- Preservar corpus `corpus_real/wikipedia_pt`, manifestos, hashes SHA-256, arquivos de estado, checkpoints e alterações não commitadas. Não rodar comandos destrutivos do Git.
- Não sobrescrever artefatos oficiais com resultados de smoke-test/dry-run. Usar nomes que evidenciem `experimental` ou `naooficial`.

## Critérios científicos por subsistema

### Corpus e tokenização

- Licença por fonte/documento e proveniência verificável; internet pública ≠ permissão irrestrita. Não inventar garantias jurídicas para pesos derivados.
- `shard_*.jsonl` contém artigos; `_metadados.jsonl`/`_estado_coleta.json` são auxiliares; conferir o esquema real. Ingestão deve ser incremental e idempotente. Não liberar quarentena PII por títulos: auditar *matches* e contextos sem expor dados sensíveis. Não rejeitar matemática/código legítimos só pela proporção de letras.
- Deduplicar dentro de cada split e entre splits; agrupar revisões da mesma página e quase-duplicatas; **split antes de ajustar BPE**; manter teste congelado. `fonte` de proveniência não equivale a `unidade_agrupamento` de vazamento.
- BPE byte-level somente ajustado em treino, tamanho efetivo **16.384** contando especiais. Exigir `audit-tokenizer` e round-trip treino/validação/externo. Gate falhando significa **não oficial**, não um convite para maquiar métricas.
- Detalhes e aceite: [references/criterios-cientificos.md](references/criterios-cientificos.md).

### Treinamento, CUDA e inferência

- Modelo base e testes de causalidade, shapes, gradientes, contagem exata, checkpoint/restart e overfit técnico são requisitos. Modelo aleatório não é assistente.
- `.venv-cuda` com torch CUDA foi validado historicamente na RTX 4050, mas a máquina atual e a sessão precisam ser confirmadas; preservar `.venv` CPU.
- Em mixed precision, verificar suporte a BF16, usar GradScaler quando apropriado para FP16, manter loss finita e estado de `train/eval` consistente inclusive em exceções. OOM deve falhar com segurança sem corrupção silenciosa.
- Em benchmarks: aquecimento, `torch.cuda.synchronize()`, memória pico GPU, throughput **com** optimizer step, latência inferência distinta de treinamento e contexto/batch identificados. Não inferir velocidade de treino a partir de benchmark matmul.
- Métricas: training/validation loss, perplexidade, tokens/s, RAM/VRAM, latência, checkpoint recuperável e qualidade de geração avaliada separadamente.

## Trabalho atual prioritário

Recuperar o estado **real** da Sprint 6.2 interrompida pelo Claude, em vez de assumir que foi concluída. Foco: revisão contextual de quarentena PII, preservação de conteúdo matemático, testes de regressão, integridade/idempotência e validação local com CUDA quando disponível. Não retomar downloads ou treinar pesos nessa primeira tarefa.

Consultar [references/handover-sprint6.md](references/handover-sprint6.md) para números **históricos** e possíveis divergências. Consultar `docs/experiments/` do checkout para o estado vigente.
