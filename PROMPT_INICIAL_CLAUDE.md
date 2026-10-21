# PROMPT_INICIAL_CLAUDE.md — Ignis

> **Nota de reconstrução:** este arquivo era referenciado como ponto de partida do projeto, mas não foi encontrado em `E:\Models\Ignis` durante o diagnóstico da Fase 0 (18/10/2026) — a pasta do projeto estava vazia. Por autorização explícita de Vinicius Santos (vinicius.souza@penso.com.br, Penso Tecnologia) na autorização da Fase 1, este documento foi **reconstruído** a partir: (a) da descrição oficial do projeto "Ignis" fornecida à sessão, (b) da skill `ignis-llm-engineer` (`.claude/skills/ignis-llm-engineer/SKILL.md`) e (c) das decisões confirmadas explicitamente pelo usuário nas mensagens de autorização da Fase 0 e da Fase 1. Nenhum conteúdo do arquivo original foi presumido ou inventado; qualquer informação não coberta por essas três fontes foi deixada como pendência explícita.

## 1. Visão geral do projeto

O **Ignis** é uma família independente de Small Language Models (SLMs), com arquitetura Transformer, pesos próprios (inicializados aleatoriamente, nunca herdados de Qwen/Llama/Gemma ou qualquer outro LLM) e capacidade de executar inferência local e offline em computadores com recursos limitados.

O primeiro modelo da família é o **Ignis-30M**, decoder-only, atenção causal, com RoPE, RMSNorm, Grouped Query Attention (GQA) e SwiGLU. O projeto é desenvolvido em Python e PyTorch, cobrindo tokenização própria, preparação de datasets, pré-treinamento do zero, geração autoregressiva, avaliação estatística, otimização computacional e documentação científica.

### Objetivos principais
- Criar e treinar o Ignis-30M do zero, sem utilizar pesos de LLMs existentes.
- Priorizar o português brasileiro e a eficiência computacional.
- Implementar um pipeline reproduzível de treinamento e avaliação.
- Permitir geração de texto completamente offline.
- Projetar a evolução para Ignis-100M, Ignis-350M e futuros modelos especializados em finanças, saúde e educação.
- Manter o projeto modular, testável e preparado para publicação open source.

## 2. Especificação técnica do Ignis-30M (configuração aprovada)

| Parâmetro | Valor |
|---|---|
| Arquitetura | Transformer decoder-only |
| n_layers | 7 |
| d_model | 512 |
| n_heads | 8 |
| n_kv_heads | 4 |
| head_dim | 64 |
| intermediate_size | 1536 |
| vocab_size | 16384 |
| max_seq_len | 1024 |
| Ativação | SwiGLU |
| Normalização | RMSNorm |
| Codificação posicional | RoPE |
| Atenção | Grouped Query Attention (GQA) |
| tie_word_embeddings | true |
| bias | false |
| dropout | 0.0 |

Contagem de parâmetros validada analiticamente na Fase 0 e confirmada na Fase 1 por `sum(p.numel() for p in model.parameters())`: **30.416.384 parâmetros treináveis**. Essa configuração foi formalmente aprovada por Vinicius Santos como ponto de partida ("Decisão confirmada #2" da autorização da Fase 1).

## 3. Decisões confirmadas pelo usuário (autorização da Fase 1, 18/10/2026)

1. O nome do projeto é Ignis; o primeiro modelo é o Ignis-30M.
2. A arquitetura acima, com 30.416.384 parâmetros, está aprovada como configuração inicial.
3. Este arquivo (`PROMPT_INICIAL_CLAUDE.md`), ausente do disco, deveria ser reconstruído como documentação de escopo com base nas decisões já confirmadas e na skill `ignis-llm-engineer` — sem presumir acesso ao conteúdo original.
4. O desenvolvimento ocorre em `E:\Models\Ignis`.
5. A VM Linux de desenvolvimento (acessada via ponte com o computador do usuário) é utilizada **exclusivamente** para desenvolvimento, testes unitários e smoke tests de baixo custo.
6. O treinamento completo (pré-treinamento real do Ignis-30M) será realizado posteriormente, em ambiente com recursos computacionais mais adequados — a VM atual tem apenas ~1 núcleo físico / 2 threads e ~3,8 GB de RAM, sem GPU, o que é insuficiente para pré-treinamento em escala.
7. O corpus piloto tem como objetivo inicial de 1 a 5 milhões de tokens em português brasileiro. A implementação de coleta e preparação dos dados fica para a Fase 2.

## 4. Regras obrigatórias (vigentes em todas as fases, reforçadas na Fase 1)

- Não declarar que o modelo está treinado, ou que compreende português, antes de medir.
- Não iniciar treinamento extenso, downloads volumosos de corpora ou instalações globais sem autorização explícita.
- Não instalar CUDA ou dependências de GPU desnecessárias (ambiente de desenvolvimento é CPU-only).
- Não alterar componentes fora do escopo da fase em andamento.
- Não realizar commits Git automaticamente sem autorização explícita do usuário.
- Toda implementação, documentação, mensagens de CLI e comentários em português brasileiro; identificadores de código podem seguir convenções técnicas em inglês.
- Preservar a estrutura de skills do Claude (`.claude/skills/ignis-llm-engineer/SKILL.md`) dentro do repositório do projeto.

## 5. Fases do projeto

| Fase | Escopo | Status |
|---|---|---|
| 0 | Diagnóstico do ambiente computacional, validação da arquitetura proposta, estimativa de parâmetros, plano de implementação | Concluída em 18/10/2026 |
| 1 | Fundação da arquitetura (estrutura do projeto, modelo, testes, smoke tests) | Em andamento |
| 2 | Corpus e tokenizador (coleta, licenciamento, BPE, splits) | Pendente |
| 3 | Pré-treinamento experimental (piloto com orçamento explícito de tokens/tempo) | Pendente — depende de ambiente com mais recursos computacionais |
| 4 | Inferência e avaliação (CLI local, benchmarks, comparação com baseline) | Pendente |

## 6. Pendências explícitas (não cobertas pelas fontes usadas na reconstrução)

- Fontes exatas, licenças e critérios de curadoria do corpus piloto em português — a definir na Fase 2.
- Especificação do ambiente de treinamento completo mencionado na decisão #6 (hardware específico) — ainda não definido pelo usuário.
- Roadmap detalhado de Ignis-100M, Ignis-350M e modelos especializados (finanças, saúde, educação) além da menção nos objetivos principais.
