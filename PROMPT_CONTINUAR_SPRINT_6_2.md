# PROMPT PARA O CODEX — Ignis-30M / retomada no notebook RTX 4050

Use obrigatoriamente a skill `$ignis-llm-engineer` e as instruções `AGENTS.md` do repositório.

**Objetivo:** concluir a Sprint 6.2 interrompida no Claude Code, trabalhando sobre o checkout local existente `E:\Models\Ignis` no notebook com RTX 4050. **Não recrie o projeto** e **não faça outro clone**.

## Contexto comprovado historicamente — valide antes de afirmar como atual

- Modelo próprio decoder-only, 7 camadas, 512 oculto, GQA 8Q/4KV, SwiGLU 1536, 16.384 tokens, contexto 1.024, 30.416.384 parâmetros.
- Corpus real da Wikipédia em português (último estado informado): 234 artigos; 221 aceitos, 1 rejeitado e 12 quarentena; ~232.538 tokens pelo BPE experimental; tokenizador **não oficial**.
- Corrigida na Sprint 6.1 a leitura indevida de `_metadados.jsonl`; ingestão incremental idempotente. Última suíte sem torch na VM Linux: 115 testes aprovados/2 ignorados.
- Ambiente Windows local já teve `.venv-cuda` + PyTorch `2.14.1+cu126` operacional na RTX 4050. Confirmar novamente antes de usá-lo.
- Não houve pré-treinamento real. A Sprint 6.2 pode conter alterações parciais não commitadas.

## Trabalho solicitado — faça, não apenas planeje

1. **Recuperação segura (rápida):** confirme pasta, branch, `git status --short` e arquivos modificados. Inspecione `docs/experiments/fase3-sprint6.1-correcao-corpus.md`, a skill antiga `.claude/skills/ignis-llm-engineer/SKILL.md`, testes e módulos da Sprint 6.2. Identifique o que Claude já concluiu, o que falta e eventuais conflitos com `AGENTS.md`. Preserve 100% do trabalho preexistente.
2. **Garantia de integridade local:** execute somente `--verificar` do coletor se o comando/sintaxe existir; valide shards, metadados, IDs/revisões/hashes e estado de retomada sem fazer requisições HTTP. Não rebaixar nem apagar nada.
3. **Corrigir qualidade do corpus:** examine as correspondências reais de PII dos 12 documentos em quarentena, com exposição mínima de identificadores; diferencie falsos positivos verificáveis de casos incertos. Não libere documentos apenas pelos títulos. Revise a rejeição do texto matemático: ajuste critérios com exemplos técnicos, equações e código preservados, sem abrir a porta para lixo textual. Registre razões de aceite/quarentena/rejeição.
4. **Compatibilidade e idempotência:** mantenha seleção correta de `shard_*.jsonl` e exclusão de auxiliares; testes com fixtures e simulações de API para HTTP 429, `Retry-After`, `maxlag`, suspensão e retomada. Não faça chamadas de rede nesta tarefa.
5. **Testes eficientes:** rode uma vez os testes específicos afetados; corrija falhas; em seguida rode a suíte completa **uma vez** com `.venv-cuda\Scripts\python.exe -m pytest -q`, caso esse Python exista. Se testes CUDA forem relevantes, execute smoke curto sem atualizar pesos nem iniciar treino. Se o ambiente não for local Windows/GPU, registre bloqueio em vez de simular sucesso.
6. **Reprocessamento seletivo:** atualize apenas os resultados derivados impactados por correções, de maneira idempotente e sem sobrescrever checkpoints ou artefatos oficiais. Verifique manifesto e hashes; se os dados restantes não atingirem o gate `audit-tokenizer`, mantenha `oficial_do_piloto=False` e reporte razões.
7. **Entrega:** indique causa raiz de cada problema, arquivos alterados, comandos e resultados reais, número final de documentos por status, situação do tokenizador, eventuais dados em risco e próximo passo objetivo. Relatório breve, sem repetir narrativas de progresso.

## Política de eficiência

- Uma única inspeção localizada e um lote coerente de edições.
- Testes direcionados, depois uma única suíte final; repetir só se o código for alterado depois.
- Reusar código existente e preferir mudanças pequenas.
- Não percorrer todos os PDFs nem reescrever todos os docs; consultar fontes originais somente se existir decisão técnica conflitante.
- Não afirmar resultados não medidos.

## Restrições absolutas

**NÃO** baixar ou retomar coleta, ampliar limites, instalar/atualizar pacotes, treinar pesos do Ignis-30M, alterar arquitetura, exportar/publicar corpus ou checkpoints, fazer `git commit`/`push`, apagar artefatos ou sobrescrever trabalho anterior sem autorização expressa.

No encerramento, entregue um comando PowerShell seguro para o próximo passo, distinguindo claramente ação que exige aprovação da ação que já está autorizada.
