# Ignis — instruções do repositório para Codex

Este repositório implementa o **Ignis-30M**, um Transformer decoder-only próprio, treinado do zero, para geração de texto geral com prioridade para português brasileiro. Programação é domínio secundário; RAG é uma camada externa futura. Inspirar-se no Qwen3 **não** significa usar seus pesos nem reivindicar equivalência funcional.

## Fonte de verdade

1. Código e testes efetivamente presentes no checkout local.
2. Relatórios experimentais mais recentes em `docs/experiments/`, com evidência de execução.
3. Arquivos de configuração e especificações arquiteturais aprovadas.
4. Documentação de planejamento mais antiga; não tratar planos como resultados.

Use a skill `$ignis-llm-engineer` ao trabalhar com arquitetura, dados, tokenização, treinamento, CUDA, inferência ou avaliação. A skill está em `.agents/skills/ignis-llm-engineer/SKILL.md`.

## Regras que não podem ser alteradas implicitamente

- Arquitetura aprovada: 7 camadas; dimensão 512; 8 Q / 4 KV; cabeça 64; SwiGLU 1536; vocabulário 16.384; contexto 1.024; GQA causal; RoPE; RMSNorm; pesos de embeddings compartilhados; **30.416.384 parâmetros**. Alterações exigem autorização e justificativa matemática.
- Não iniciar treinamento/pretreinamento real ou prolongado, ampliar downloads de corpus, instalar ou atualizar dependências, alterar ambientes CUDA, enviar dados para serviços externos, publicar pesos/dados ou fazer `git commit`/`push` sem aprovação específica.
- A autorização histórica para coleta controlada da Wikipédia em português **não** autoriza outras fontes, aumento de cotas ou treinamento de pesos. Verificar os limites no comando e no estado local antes de qualquer retomada.
- Preservar alterações não commitadas, corpus, manifestos, tokenizadores, checkpoints e estado de retomada. Proibidos `git reset --hard`, `git clean -fd`, remoção ou substituição destrutiva sem autorização.
- Manter rastreabilidade de origem/licença, hashes, seed, split sem contaminação, configurações, versões e métricas. Não afirmar capacidade conversacional, desempenho, aprovação de teste ou status de treinamento sem evidência.

## Política de trabalho eficiente

- Trabalhar **na pasta local existente**, não criar outro repositório. Antes de editar, executar `git status --short` e inspecionar só arquivos relevantes.
- Implementar o menor conjunto coerente de mudanças; reutilizar módulos, testes e dados existentes.
- Rodar testes direcionados após cada lote coerente, e a suíte completa **uma vez ao término**, salvo falha ou mudança estrutural. Não suprimir testes para obter resultado verde.
- Preferir `.venv-cuda\Scripts\python.exe` apenas se comprovado acessível no Windows da RTX 4050; usar o ambiente disponível e declarar limitações. Não presumir GPU em nuvem/VM.
- Descrever brevemente: arquivos modificados, evidências medidas, testes realizados, pendências e próximo comando. Evitar narrativas de terminal repetitivas.

Leia os arquivos de referência da skill apenas conforme a tarefa exigir.
