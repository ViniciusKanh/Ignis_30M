# Visão de produto e roadmap — Ignis

Este documento registra decisões de escopo e direção de longo prazo do
projeto Ignis, complementando (não substituindo) `docs/architecture.md`
(decisões técnicas de arquitetura) e os planos de execução por fase em
`docs/plano-*.md`. Atualizado em 2026-10-18 por decisão do responsável
pelo projeto, após a conclusão técnica da Sprint 5 (CUDA validado).

## O que o Ignis-30M é

Um modelo de linguagem generativo de propósito geral, decoder-only,
treinado inteiramente do zero (pesos inicializados aleatoriamente, nunca
copiados, destilados ou ajustados a partir de pesos de terceiros). As
escolhas de arquitetura (RoPE, RMSNorm, Grouped Query Attention, SwiGLU)
seguem convenções de design popularizadas por famílias de modelos recentes
como o Qwen3 — isto é uma referência de **estilo arquitetural**, não de
proveniência: nenhum peso, checkpoint ou saída de qualquer modelo Qwen
(ou de qualquer outro modelo de terceiros) é usado em nenhuma etapa do
Ignis. Este princípio (já vigente desde o início do projeto, via skill
`ignis-llm-engineer`, item 1) não muda com esta atualização.

## Prioridades (em ordem)

1. **Aprendizado de linguagem geral em português brasileiro** — compreensão
   e geração de texto corrido, coerente e gramaticalmente correto em
   pt-BR. Esta continua sendo a prioridade central do projeto.
2. **Geração e compreensão de texto em geral** (não apenas pt-BR, na
   medida em que o corpus incluir outras línguas de forma incidental).
3. **Programação como domínio técnico secundário.** Documentação técnica e
   código-fonte licenciado podem ser incluídos no corpus de pré-treinamento
   em uma fase futura (nunca nesta fase inicial, e nunca sem a mesma
   disciplina de licença/proveniência já exigida para qualquer fonte —
   ver `docs/plano-corpus-piloto.md`). O Ignis-30M Base **não** deve virar
   um modelo exclusivamente voltado a código — isso é reservado a uma
   eventual especialização futura (`Ignis-30M-Code` ou equivalente, ver
   roadmap de especializações abaixo).
4. **Inferência local e offline** — já um requisito desde o início do
   projeto; não muda.

## Camadas futuras, fora do escopo do pré-treinamento do modelo base

- **RAG (Retrieval-Augmented Generation):** tratado como uma camada
  externa, posterior ao modelo base — nunca misturado com o pré-treinamento
  em si. A avaliação de "o modelo sabe X" e "o sistema com RAG consegue
  responder X usando uma fonte externa" deve continuar sendo medida e
  reportada separadamente, para não inflar artificialmente a percepção de
  capacidade do modelo base.
- **Ajuste por instruções (instruction tuning):** já havia a regra de
  separar claramente pré-treinamento de pós-treinamento orientado a
  instruções (skill `ignis-llm-engineer`, princípio 6); esta atualização
  reafirma essa separação também em relação ao RAG.

## Roadmap de especializações futuras

O planejamento original do projeto (descrição do Project Ignis) já previa
evolução para `Ignis-100M`, `Ignis-350M` e modelos especializados em
finanças, saúde e educação. Esta atualização **adiciona** a essa lista,
sem substituí-la, uma possível especialização futura em programação
(`Ignis-30M-Code` ou equivalente em uma geração maior), e a integração de
RAG como uma capacidade de sistema (não do modelo em si) aplicável a
qualquer uma dessas especializações. Nenhuma dessas especializações futuras
está em desenvolvimento agora — são direções de roadmap, não compromissos
de escopo da fase atual.

## Avaliação

Quando o corpus de pré-treinamento vier a incluir código/documentação
técnica (fase futura, não autorizada ainda), a avaliação deve medir
linguagem geral e programação **separadamente** — nunca um único número
agregado que esconda um desequilíbrio entre os dois domínios.

## O que esta atualização explicitamente NÃO autoriza nem muda

- Não autoriza, por si só, a inclusão de código-fonte ou documentação
  técnica no corpus de pré-treinamento desta fase — isso permanece sujeito
  às mesmas regras de licença/proveniência de qualquer fonte, com
  autorização específica própria quando chegar a hora.
- Não altera a arquitetura já implementada e validada (30.416.384
  parâmetros, GQA, RoPE, RMSNorm, SwiGLU) — é uma atualização de visão de
  produto e priorização, não uma mudança de código.
- Não implica que a aquisição de corpus real (Wikipédia em português ou
  qualquer outra fonte) esteja autorizada — essa autorização continua
  sendo específica e separada (ver `docs/checklist-autorizacao-sprint5.md`).
