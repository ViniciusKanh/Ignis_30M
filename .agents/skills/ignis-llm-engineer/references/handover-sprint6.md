# Handover histórico — Claude Code → Codex

**Registro de contexto, não substitui inspeção do repositório.** Dados de 18–20 de outubro de 2026 informados pelo operador do projeto.

## Progresso estrutural

- Fase 0: diagnóstico e matemática concluídos.
- Fase 1: Transformer PyTorch implementado, 30.416.384 parâmetros confirmados, testes estruturais e geração não treinada.
- Fase 2: pipeline ingestão/normalização/dedup/PII/splits + ByteLevel BPE e CLI implementados; corpus real piloto **ainda em preparação**.
- Fase 3: infraestrutura de treinamento, AdamW, warmup/cosine, checkpoint/resume e dry-run implementada; não equivale a pré-treino real.
- Sprint 5: infraestrutura CUDA isolada na RTX 4050 testada: `.venv-cuda` torch `2.14.1+cu126`, dispositivo operacional, forward/backward GPU mínimo. Benchmark de matmul apontou ~11,3x, **não é benchmark de treinamento**.
- Sprint 6.1: falha de ingestão `_metadados.jsonl` foi corrigida; validação incremental passou duas vezes sem duplicação. Coleção no ponto medido: 234 artigos preservados; 221 aceitos, 1 rejeitado (matemática), 12 em quarentena (suspeita PII). Tokenizador experimental contou 232.538 tokens; `audit-tokenizer` retornou `oficial_do_piloto=False` por volume insuficiente. Testes VM Linux sem torch: 115 passed / 2 skipped. **Nenhuma evidência de suíte inteira Windows/CUDA pós-Sprint 6.1 foi fornecida.**
- Sprint 6.2 iniciada em Claude Code e interrompida. O que já foi alterado durante esta sprint é **desconhecido**; Codex deve fazer `git status` e comparar com o histórico antes de editar.

## Arquivos a localizar (não assumir existência)

- `.claude/skills/ignis-llm-engineer/SKILL.md`
- `docs/experiments/fase3-sprint6.1-correcao-corpus.md`
- `src/ignis/data/ingest.py`, módulos de qualidade/PII e pipeline
- `scripts/coletar_wikipedia_pt.py`
- `corpus_real/wikipedia_pt/` (pode existir apenas no notebook RTX)
- `data/manifesto.jsonl`, splits e artefatos do tokenizer
- `tests/` para ingestão, PII, qualidade, tokenização e CUDA
- `.venv-cuda/Scripts/python.exe` e `.venv/Scripts/python.exe`

## Notas importantes

- Houve menções conflitantes a limites de `50 MiB` de tráfego e `500 MB` de armazenamento versus execução com `--teto-mb 500`. Não reinterpretar silenciosamente autorização: reconhecer o teto ativo informado pelo operador, verificar como o script mede tráfego, registrar as diferenças, solicitar decisão antes de **aumentar** o permitido. Na primeira tarefa não fazer chamadas HTTP.
- 12 itens considerados falsos positivos de PII por títulos é hipótese, **não prova**. Reavaliar trechos anonimizados.
- Aprendizagem real dos 30M ainda não foi realizada; não alterar pesos nesta sprint.
