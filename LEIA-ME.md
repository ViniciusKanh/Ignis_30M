# Instalar a skill do Ignis no aplicativo Codex para Windows

O pacote inclui:

- `AGENTS.md`: regras gerais do repositório para Codex.
- `.agents/skills/ignis-llm-engineer/SKILL.md`: skill invocável como `$ignis-llm-engineer`.
- `.agents/skills/ignis-llm-engineer/references/*.md`: critérios científicos e handover.
- `PROMPT_CONTINUAR_SPRINT_6_2.md`: primeira tarefa.
- `INSTALAR-CODEX.ps1`: copia somente os arquivos da skill/instruções e protege destinos preexistentes.

## Instalação recomendada (notebook com RTX 4050)

1. Extraia o ZIP numa pasta temporária, sem colocar por engano um novo projeto dentro do repositório.
2. Abra PowerShell **na pasta extraída**.
3. Execute:

   `powershell -NoProfile -ExecutionPolicy Bypass -File .\INSTALAR-CODEX.ps1 -Repositorio E:\Models\Ignis`

   O instalador não modifica dependências, corpus, checkpoints nem código do modelo. Se já existir algum destino divergente, **não sobrescreve**: escreve um `.novo` ao lado para comparação e informa a necessidade de mescla manual.
4. No aplicativo Codex para Windows, abra **a pasta local existente** `E:\Models\Ignis`, não Codex Cloud. Inicie nova conversa depois de instalar a skill para permitir redescoberta.
5. Cole o conteúdo de `PROMPT_CONTINUAR_SPRINT_6_2.md` e inclua `$ignis-llm-engineer` no início.
6. Opcional: peça ao `$skill-creator` para revisar e validar a skill, **sem sobrescrever** os arquivos aprovados e sem mexer no código do Ignis.

O pacote é baseado nos documentos do projeto e nos logs fornecidos, **não** em uma inspeção direta do drive `E:`. O Codex deverá confirmar o estado real do checkout e do hardware.
