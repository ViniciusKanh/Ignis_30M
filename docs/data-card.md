# Data Card — Ignis-30M (Fase 2, atualizado na Sprint 6.1)

Este documento descreve a política de dados, o esquema do manifesto de
proveniência e o tratamento de licenciamento/PII implementados na Fase 2.
O pipeline foi validado inicialmente com corpora sintéticos de teste (ver
`tests/test_data_*.py`) e, a partir da Sprint 6, também com uma primeira
fonte real (seção dedicada abaixo). O corpus real ainda está muito abaixo
da meta de volume e **não é considerado "oficial do piloto"** (ver
`docs/plano-corpus-piloto.md`, seção 8, e `docs/checklist-autorizacao-sprint5.md`).

## Primeira fonte real: Wikipedia em português (Sprint 6/6.1)

- **Fonte:** Wikipedia em português (`pt.wikipedia.org`), via API pública
  da MediaWiki (`generator=random`, extratos em texto simples, nunca o
  dump XML completo). Coletor: `scripts/coletar_wikipedia_pt.py`.
- **Licença:** CC BY-SA 4.0 + GFDL (dual-licenciado), conforme os Termos
  de Uso da Wikimedia Foundation. `permissao_treinamento=true` registrado
  em cada documento coletado — a ressalva sobre compatibilidade de
  CC BY-SA com "treinar e distribuir pesos" (não unânime) está detalhada
  em `docs/plano-corpus-piloto.md`, seção 1, e não foi resolvida
  juridicamente nesta sessão.
- **Autorização de volume vigente:** meta de ~1.000.000 tokens
  aproximados, teto RÍGIDO de 500 MB de download — autorizada diretamente
  pelo responsável pelo projeto em 2026-10-18 (ver
  `docs/checklist-autorizacao-sprint5.md`, seção D, incluindo o registro
  de uma alegação de limite diferente que não foi adotada por não ser
  verificável).
- **Progresso real:** 234 artigos coletados até agora (muito abaixo da
  meta), 1,27 MB de texto, ~200.842 tokens aproximados na coleta (269.119
  tokens reais, medidos pelo tokenizador BPE, nos 221 documentos aceitos
  após triagem). Metadados de proveniência completos (pageid, título, URL,
  revisão, data de coleta UTC, hash SHA-256 do texto) ficam em
  `corpus_real/wikipedia_pt/_metadados.jsonl`, ao lado dos arquivos de
  conteúdo `shard_*.jsonl`.
- **Resultado da triagem de qualidade (234 artigos):** 221 aceitos, 1
  rejeitado (baixa razão de caracteres alfabéticos — artigo de matemática
  com muita notação), 12 em quarentena por "possível PII" — inspeção
  manual dos títulos confirma que são **falsos positivos** do detector
  heurístico de PII (números de página, coordenadas, códigos militares
  etc. combinando com padrões de cartão de crédito/telefone/CNPJ), não PII
  real. Ver `docs/experiments/fase3-sprint6.1-correcao-corpus.md`, seção
  6, para a lista completa.
- **Diversidade temática:** amostra real mostra ampla variedade de temas
  (lugares, astronomia, biologia, matemática, religião, história militar,
  aviação, música, empresas, biografias) — sem sinal de concentração
  excessiva em uma única categoria no corpus coletado até agora.
- **Contrato de arquivo auxiliar (Sprint 6.1):** qualquer arquivo cujo
  NOME comece com `_` (como `_metadados.jsonl` e `_estado_coleta.json`,
  escritos pelo coletor ao lado do conteúdo) é tratado como auxiliar e
  NUNCA lido como documento por `ignis.data.ingest` — ver
  `eh_arquivo_auxiliar()` em `src/ignis/data/ingest.py`.

## Política de proveniência e licenciamento

Um documento só é marcado como `aceito` quando possui, simultaneamente:

1. `licenca` declarada (string não vazia, ex.: nome da licença ou URL).
2. `permissao_treinamento` declarada como `true` (booleano explícito, não
   inferido).

Essas informações vêm do próprio arquivo de origem (campos opcionais
`licenca` e `permissao_treinamento` em entradas `.jsonl`) ou são atribuídas
manualmente no momento da ingestão. **Nenhum documento é aceito por
omissão** — a ausência de informação de licença resulta em
`status = licenca_desconhecida`, nunca em `aceito`.

## Ordem de triagem (pipeline.py)

Cada documento passa pelas seguintes checagens, nesta ordem, e a primeira
que reprovar decide o status final:

1. **Duplicata exata** (hash SHA-256 do texto normalizado) → `rejeitado`.
2. **Qualidade/tamanho** (`FiltrosQualidade`: mínimo 200 caracteres, máximo
   2.000.000 caracteres, mínimo 50% de caracteres alfabéticos) → `rejeitado`.
3. **Possível PII** (regex conservadora: e-mail, CPF, CNPJ, telefone
   brasileiro, número de cartão de crédito) → `quarentena` — **o documento
   nunca é excluído automaticamente**, apenas sinalizado para revisão
   humana, com o tipo e a contagem de padrões encontrados registrados no
   campo `motivo`.
4. **Licença/permissão de treinamento** → `aceito` (se ambas presentes) ou
   `licenca_desconhecida` (caso contrário).

## Esquema do manifesto (`EntradaManifesto`)

| Campo | Tipo | Descrição |
|---|---|---|
| `documento_id` | str | Identificador único do documento |
| `caminho_origem` | str | Caminho do arquivo de origem |
| `fonte` | str | Nome da fonte/coleção (usado para agrupar nos splits) |
| `licenca` | str \| null | Licença declarada, se houver |
| `permissao_treinamento` | bool \| null | Permissão explícita de uso para treinamento |
| `sha256` | str | Hash do texto normalizado (dedup exata) |
| `tamanho_caracteres` | int | Tamanho do texto normalizado |
| `idioma_assumido` | str | Padrão `"pt-BR"` |
| `status` | enum | `aceito` \| `rejeitado` \| `quarentena` \| `licenca_desconhecida` |
| `motivo` | str | Explicação textual do status atribuído |
| `grupo_quase_duplicata` | str \| null | Id do grupo de quase-duplicatas (union-find), preenchido na etapa de splits |

O manifesto é persistido em formato JSONL (`pipeline.salvar_manifesto` /
`carregar_manifesto`), uma entrada por linha, para ser auditável e
versionável.

## Deduplicação

- **Exata:** hash SHA-256 do texto normalizado (NFC, sem caracteres de
  controle, quebras de linha normalizadas). Documentos idênticos após
  normalização são rejeitados, mantendo apenas a primeira ocorrência.
- **Quase-duplicatas:** shingles de 5 palavras + similaridade de Jaccard
  (limiar padrão 0,8), agrupados via union-find. **Limitação conhecida:**
  implementação O(n²), adequada a corpora pequenos/piloto (até alguns
  milhares de documentos); para escala maior, substituir por MinHash/LSH é
  um trabalho futuro (ver `docs/limitations.md`).

## Triagem de PII

Detecção por regex para: e-mail, CPF, CNPJ, telefone brasileiro e número de
cartão de crédito (`src/ignis/data/pii.py`). É uma heurística
**conservadora e não exaustiva** — pode gerar falsos positivos (ex.:
sequências numéricas longas não relacionadas a cartão de crédito) e falsos
negativos (padrões de PII fora do conjunto coberto, PII em formatos não
previstos). Qualquer detecção resulta em quarentena, nunca em exclusão
automática ou aceitação silenciosa.

## Splits (treino/validação/teste)

- Atribuição por hash SHA-256 estável (`_hash_estavel`, não usa `hash()`
  nativo do Python, que varia com `PYTHONHASHSEED`) — garante
  reprodutibilidade entre execuções com a mesma seed.
- Agrupamento por `grupo_quase_duplicata` (documentos quase-idênticos caem
  sempre no mesmo split) e, quando possível, por `fonte` — evita
  contaminação entre splits.
- Apenas documentos com `status = aceito` entram nos splits.
- `verificar_contaminacao()` confirma ausência de sobreposição de IDs entre
  splits antes de persistir.
- Proporções padrão: treino 90% / validação 5% / teste 5% (configurável via
  CLI), validadas para somar exatamente 1.0.
- Splits persistidos em JSON, incluindo IDs, hashes, critérios e seed
  usados — auditável e reproduzível.

## Tokenizador

Tokenizador BPE byte-level (Hugging Face `tokenizers`), treinado
**exclusivamente sobre o split de treino** — nunca sobre validação ou
teste, para não contaminar a avaliação. Vocabulário alvo: 16.384 tokens
(configuração oficial do Ignis-30M); tokens especiais `<pad>`, `<bos>`,
`<eos>`, `<unk>` fixados nos IDs 0–3.

## Auditoria quantitativa de distribuição (Sprint 4)

`ignis.cli report` inclui um bloco `distribuicao_por_fonte`: para cada
fonte presente no manifesto, quantos documentos caíram em cada status de
triagem e qual fração do total de documentos ACEITOS aquela fonte
representa. Quando uma única fonte concentra mais de 50% do corpus aceito
(ou quando só existe uma fonte contribuindo documentos aceitos), o
relatório inclui um aviso explícito em `avisos_concentracao` — isto existe
para que um corpus cientificamente frágil (pouco diverso) seja visível
antes do treinamento, nunca descoberto depois. Nenhum limiar de
concentração foi calibrado estatisticamente; 50% é uma heurística inicial
de alerta, documentada como tal.

## Pendências científicas conhecidas (Fase 2)

- Nenhum corpus real/licenciado foi processado — todo o pipeline foi
  validado apenas com dados sintéticos de teste.
- O limiar de Jaccard para quase-duplicatas (0,8) é uma heurística inicial,
  não calibrada estatisticamente sobre um corpus real.
- As regras de PII podem ter taxa de falso-positivo/falso-negativo não
  medida formalmente; recomenda-se amostragem manual da quarentena antes de
  qualquer liberação de corpus para treinamento.
- A razão mínima de caracteres alfabéticos (50%) como proxy de qualidade é
  uma heurística simples; pode rejeitar texto legítimo rico em números/
  símbolos (ex.: tabelas) ou aceitar texto de baixa qualidade que ainda
  assim seja majoritariamente alfabético.
