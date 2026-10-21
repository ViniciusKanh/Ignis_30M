# Critérios científicos e de aceite do Ignis

## Ordem de evidência

- Execução atual reproduzível e seus artefatos > relatório experimental recente > configuração/código atual > documentação original > intenção de prompt.
- Separar **aprovado**, **hipótese**, **estimado**, **medido** e **pendente**. Nunca preencher lacunas experimentais por plausibilidade.

## Pipeline de corpus

- Registrar origem/URL, página e revisão, licença/atribuição, timestamp de coleta, SHA-256, idioma e grupo de vazamento.
- Amostragem por temas não substitui medição de qualidade: estatísticas de distribuição, densidade de texto, repetição, ruído, idioma e rejeições.
- PII: executar verificações com amostras contextualizadas, mascarar identificadores em logs; diferenciar CPF/telefone real de coordenada, ID, data, fórmula, código e referência. Casos inconclusivos ficam em quarentena.
- Matemática, algoritmos e código têm padrões não alfabéticos legítimos; filtros de texto geral não devem desqualificá-los automaticamente.
- Separar arquivos auxiliares `_*.jsonl`, estado e artefatos temporários do contrato dos shards. Idempotência em dois passes exige manifesto sem duplicatas e bytes/hashes estáveis.
- Respeitar bloqueio de API: timeout, Retry-After, maxlag, suspensão e limites cumulativos. Não elevar volume, taxa ou fonte sem autorização.

## Splits e tokenizer

- Agrupar revisões do mesmo artigo e near-duplicates; detectar sobreposição dos splits por IDs, hashes e similaridade; semente e proporções reais registradas.
- Não ajustar tokenizador em validação/teste. Teste final não usado para tuning.
- Byte-level BPE de 16.384 entradas, especiais coerentes, `encode/decode` Unicode e round-trip para pt-BR, números, matemática e programação. Medir fertilidade, tamanho real de vocabulário e contagem de tokens, não apenas estimativas.
- Corpus piloto de 1–5M tokens é **verificação de processo e sinais de aprendizagem**, não argumento de capacidade geral; 30M parâmetros exigirão escala maior para desempenho competitivo.
- `audit-tokenizer` deve falhar fechado: não promover se licença, vazamento, roundtrip externo ou volume falharem.

## Treinamento

- Arquitetura exata + tied embeddings e causalidade demonstradas em testes, não presumidas.
- Primeiro dry-run; **nenhum treino de pesos** sem aprovação expressa de orçamento de passos, dataset, hardware e checkpoint.
- Salvar seed, versão das dependências, hashes de configuração e dados, tamanho dos splits, optimizer/scheduler e RNG no checkpoint retomável.
- Comparar throughput do modelo completo com mesmo batch/contexto e sincronização correta, incluir optimizer e pico de memória.
- Qualidade conversacional, factualidade, programação e RAG dependem de protocolos e treinamento próprios; redução de loss não implica capacidades já existentes.

## Aceite de uma sprint

1. Mudança menor possível, causas raiz registradas.
2. Testes direcionados e pelo menos uma suíte relevante completa no ambiente realmente acessível.
3. Nenhuma regressão conhecida, nenhum artefato corrompido e nenhum dado indevidamente exposto.
4. Evidências com comandos e resultados, pendências objetivas, sem promessas de treinamento já concluído.
