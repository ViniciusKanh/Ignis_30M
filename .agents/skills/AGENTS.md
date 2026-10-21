# Ignis — Instruções para agentes de engenharia

## Objetivo
Desenvolver o Ignis-30M, SLM generativo de propósito geral, treinado do zero e priorizando português brasileiro. Programação é domínio secundário prioritário. RAG será integração externa futura. A arquitetura foi inspirada em conceitos do Qwen3, sem reutilizar seus pesos e sem pressupor equivalência funcional.

## Contrato arquitetural aprovado
Transformer decoder-only; 7 camadas; d_model=512; 8 query heads; 4 KV heads (GQA); head_dim=64; SwiGLU=1536; vocabulário=16.384; contexto máximo=1.024; RoPE; RMSNorm; atenção causal; embeddings compartilhados; 30.416.384 parâmetros. Não modificar sem justificativa matemática e autorização explícita.

## Fontes de verdade
- Código e estado atual do Git prevalecem sobre relatos antigos.
- Consultar `.claude/skills/ignis-llm-engineer/SKILL.md` quando alterar engenharia de modelos, dados ou treinamento; preservar a skill original.
- Consultar apenas os documentos relevantes em `docs/` e `configs/`, sem reler todos a cada tarefa.
- Distinguir medição real, teste simulado, hipótese e estimativa.

## Ambiente
- Windows, repositório local `E:\Models\Ignis`.
- `.venv\Scripts\python.exe`: ambiente CPU preservado.
- `.venv-cuda\Scripts\python.exe`: PyTorch CUDA para testes e benchmarks de GPU (verificar disponibilidade em tempo de execução).
- GPU identificada em relatório anterior: NVIDIA RTX 4050 Laptop GPU (~6 GiB segundo nvidia-smi).
- VM Linux auxiliar pode não ter torch/GPU: nunca tratar seus testes como validação CUDA.

## Princípios de execução eficiente
1. Antes de editar: `git status --short`, identificar apenas módulos envolvidos e conferir interfaces atuais.
2. Preservar alterações preexistentes, datasets, shards, estados de retomada, manifestos e checkpoints.
3. Executar tarefas independentes em conjunto, mas evitar edições concorrentes nos mesmos arquivos.
4. Corrigir causas raiz e adicionar testes de regressão focados; não reformular componentes que já funcionam.
5. Rodar testes direcionados durante a implementação; rodar a suíte completa uma vez ao final do lote, não após cada arquivo.
6. Para CUDA, medir forward + backward + optimizer step, sincronizar GPU, registrar VRAM, throughput e versões. Não extrapolar benchmark de matmul.
7. Saída final concisa: mudanças, comandos e testes realmente executados, métricas, riscos e próximo bloqueio.
8. Não criar relatórios, documentos ou abstrações adicionais sem utilidade concreta.

## Dados e rigor científico
- Corpus Wikipédia em português: aquisição piloto previamente autorizada, com meta aproximada de 1 milhão de tokens; não ampliar fontes ou limites sem nova autorização.
- Respeitar Retry-After, maxlag e suspensão em HTTP 429 persistentes; jamais contornar rate limits.
- Preservar proveniência, licenças, hashes, PII/quarentena e deduplicação.
- Separar treino/validação/teste por unidades independentes; tokenizador BPE é treinado apenas no treino.
- Não promover tokenizador ou corpus sem aprovação do `audit-tokenizer`.
- Não confundir redução de loss com habilidade conversacional ou de programação.

## Ações que exigem autorização expressa
- Treinamento real de duração significativa ou atualização dos pesos oficiais.
- Download de fontes adicionais, aumento de limites de coleta ou instalação de dependências relevantes.
- Alterações arquiteturais, publicação, push, commits, exclusões destrutivas ou sobrescrita de checkpoints.

## Critério de conclusão
Entregar código funcional e validado com evidências reproduzíveis. Se o ambiente impedir um teste, relatar como não executado — nunca como aprovado.
