# Plano do Corpus Piloto Real — Ignis-30M (Fase 3)

**Status: documento de planejamento. Nenhuma ação deste plano foi executada.**
Nenhum corpus real foi baixado, selecionado ou declarado "oficial" até agora
— todo o pipeline de dados (Fase 2) e de treinamento (Fase 3) foi validado
apenas com o corpus sintético de teste (`scripts/gerar_corpus_sintetico.py`).
Este documento descreve o PROCEDIMENTO a seguir quando a seleção de um
corpus piloto real (1 a 5 milhões de tokens, português brasileiro) for
autorizada — ele não autoriza, por si só, nenhum download ou execução.

Regras que seguem valendo integralmente para tudo neste documento (skill
`ignis-llm-engineer`):
- Nenhum download de corpus, por menor que seja, acontece sem autorização
  explícita e específica (não basta a aprovação geral deste plano).
- Nenhum dado sem licença clara e permissão de uso para treinamento
  documentada é aceito — a ausência de informação é tratada como
  `licenca_desconhecida`, nunca como `aceito` (ver `docs/data-card.md`).
- Nenhum dado pessoal (PII) é processado deliberadamente; o pipeline de
  triagem (`ignis.data.pii`) já existe e deve ser aplicado a qualquer corpus
  real antes de qualquer uso.
- Nenhum corpus ou tokenizador é declarado "oficial" antes de validação
  real medida (não apenas planejada).

## 1. Fontes candidatas (a avaliar, não pré-aprovadas)

**Atualização (2026-10-18): Wikipédia em português AUTORIZADA para a
primeira coleta real**, diretamente pelo responsável pelo projeto nesta
sessão — meta de ~1.000.000 tokens, teto rígido de 500 MB de download.
Coleta via `scripts/coletar_wikipedia_pt.py` (API pública da MediaWiki,
nunca o dump XML completo), gated por `--confirmar`, ainda não executada
(requer rede, rode no Windows nativo). As demais fontes da tabela abaixo
permanecem apenas candidatas, não autorizadas.

Esta seção lista candidatas PLAUSÍVEIS para texto em português brasileiro,
com os campos exigidos pela skill `ignis-llm-engineer` (nome, URL, licença,
evidência de permissão, proveniência, categoria, volume estimado,
limitações). **"Candidata" significa apenas "razoável para avaliar
individualmente antes de ingerir"; nenhuma linha desta tabela está
aprovada para uso, e nenhum download foi realizado a partir dela.** Antes
de qualquer ingestão real, a licença e a permissão de treinamento de cada
fonte específica devem ser reverificadas na data da ingestão (licenças e
termos de uso podem mudar) e registradas no manifesto (`docs/data-card.md`).

| Nome | URL | Licença declarada | Evidência de permissão | Categoria | Volume estimado | Limitações |
|---|---|---|---|---|---|---|
| Wikipédia em português | https://pt.wikipedia.org (dumps em https://dumps.wikimedia.org/ptwiki/) | CC BY-SA 4.0 + GFDL (dual-licenciado) | Termos de uso da Wikimedia Foundation permitem reuso e redistribuição, inclusive para fins comerciais e de pesquisa, desde que atribuição e compartilhamento pela mesma licença sejam preservados (ver https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use) — **ainda assim, a compatibilidade do CC BY-SA com "treinar um modelo e distribuir os pesos" não é unânime na comunidade jurídica e deve ser avaliada antes de uso em produção**; aceitável para um piloto experimental com atribuição registrada. | Enciclopédico, multi-temático | Múltiplos GB em texto bruto (muito acima dos 1-5M tokens do piloto); uma AMOSTRA pequena e diversificada de artigos cobre a faixa-alvo facilmente | Qualidade heterogênea entre verbetes curtos/longos; exige extração de texto limpo a partir do XML/wikitexto (marcação wiki, infoboxes, referências) antes da ingestão; viés de cobertura enciclopédica, não representa português coloquial/conversacional |
| Domínio Público (Fundação Biblioteca Nacional / MinC) | http://www.dominiopublico.gov.br/ | Domínio público (obras com direitos autorais extintos, Lei 9.610/98) | Cada obra deve ser verificada individualmente quanto à data de falecimento do autor (70 anos post-mortem no Brasil) antes de ser tratada como domínio público — **não presumir domínio público apenas por estar listada no portal** | Literário (livros, textos clássicos em português) | Variável por obra; dezenas de milhares de tokens por livro típico | Portal com disponibilidade/estrutura técnica que pode mudar; português de obras antigas pode ter ortografia/vocabulário defasados frente ao português contemporâneo; checagem de domínio público obra-a-obra é trabalho manual não trivial em escala |
| Wikisource em português | https://pt.wikisource.org | CC BY-SA 4.0 (texto da Wikisource) + domínio público (para as obras-fonte, quando aplicável) | Mesmos termos de uso da Wikimedia Foundation citados acima | Literário/documental (textos transcritos) | Até alguns milhões de tokens dependendo do recorte | Mistura textos em domínio público com a licença CC BY-SA da própria transcrição/formatação; qualidade de OCR/transcrição variável em obras mais antigas |
| Portais governamentais brasileiros (ex.: `gov.br`, Diário Oficial da União via https://www.in.gov.br/) | https://www.gov.br/ , https://www.in.gov.br/ | Presumivelmente domínio público por serem atos oficiais da administração pública (Lei 9.610/98, art. 8º, IV), mas **cada portal pode ter termos de uso próprios que precisam ser lidos e citados individualmente** | A ser verificada por portal específico antes de qualquer ingestão — esta linha NÃO constitui verificação já feita | Governamental/institucional | Potencialmente grande (publicação diária), mas um piloto só precisa de uma amostra pequena e datada | Linguagem formal/jurídica pouco representativa do português geral; estrutura de documento (tabelas, portarias, numerações) exige limpeza cuidadosa para não poluir o corpus com ruído não-textual |
| Corpus Carolina (IME-USP / C4AI) | https://sites.usp.br/corpuscarolina/ | Divulgado publicamente para pesquisa em PLN; licença específica de cada subcorpus deve ser lida na documentação oficial antes de uso | Publicação acadêmica não equivale automaticamente a permissão de uso para treinar modelos de terceiros — **verificar explicitamente os termos de uso/licença do corpus, não apenas a citação acadêmica**, antes de qualquer ingestão | Corpus acadêmico de PLN (multi-gênero, pt-BR) | Projetado para bilhões de tokens; uma fração pequena bastaria para o piloto | Requer avaliação jurídica própria dos termos de redistribuição/uso de cada versão publicada; composição exata das fontes internas do corpus precisa ser entendida para não herdar problemas de proveniência de terceiros |
| Textos produzidos pelo próprio usuário/organização (Penso) | N/A (interno) | Definida pelo titular dos direitos (o próprio usuário/organização) | Permissão explícita do titular, documentada no momento da ingestão (`permissao_treinamento=true` com a origem anotada) | Variável (institucional, técnico, etc., conforme o que for disponibilizado) | Limitado ao volume real disponível | Forma mais segura de licenciamento, mas tipicamente insuficiente em volume isolado para 1-5M tokens; útil como complemento de alta confiança, não como fonte única |

Explicitamente **fora de escopo sem avaliação jurídica própria**: qualquer
scraping de sites com termos de uso que proíbam coleta automatizada ou reuso
para treinamento de modelos, redes sociais, conteúdo com copyright comercial
sem licença explícita, qualquer fonte que exija login/autenticação para
acesso, e corpora web crawl de proveniência mista sem licença consolidada
por documento (ex.: coleções genéricas de "common crawl" em português sem
filtragem de licença por página) — são compostos por milhões de páginas de
origem e licença desconhecidas/mistas, e portanto incompatíveis com a regra
deste projeto de nunca aceitar um documento sem licença e permissão de
treinamento declaradas (ver `docs/data-card.md`).

### Conectores de ingestão preparados (sem download executado)

O pipeline de ingestão (`ignis.data.ingest.ingerir_diretorio`) já lê
`.txt`/`.md`/`.jsonl` de um diretório local — é o conector "genérico" usado
por todas as fontes acima: o procedimento operacional, quando uma fonte for
autorizada, é (1) baixar/exportar manualmente ou com uma ferramenta própria
da fonte (ex.: `wikiextractor` para dumps da Wikipédia) os textos brutos,
(2) converter cada documento em uma entrada `.jsonl` com os campos `id`,
`texto`, `licenca` e `permissao_treinamento` preenchidos (ver
`src/ignis/data/ingest.py::_ler_jsonl`), (3) rodar `ignis.cli validate-corpus`
normalmente. **Nenhum destes passos foi executado para nenhuma fonte real
até o momento** — isto é apenas o procedimento já suportado pelo código
existente, documentado aqui para quando uma fonte for efetivamente
autorizada. Nenhum script de scraping/crawling foi escrito ou é proposto
neste documento.

## 2. Documentação de licença e permissão de treinamento

Para cada fonte candidata efetivamente selecionada, antes de qualquer
ingestão:

1. Registrar a licença exata (nome e, quando existir, URL/texto da licença)
   no próprio manifesto (`EntradaManifesto.licenca`, ver `docs/data-card.md`).
2. Registrar explicitamente `permissao_treinamento = true` apenas quando a
   licença permitir de forma razoavelmente clara o uso para treinamento de
   modelos de linguagem — em caso de dúvida real, tratar como
   `licenca_desconhecida` (comportamento já implementado em
   `ignis.data.pipeline`) e excluir da Fase 3 até esclarecimento.
3. Manter um registro humano-legível (ex.: uma tabela em uma futura revisão
   deste documento) listando fonte → licença → data de verificação → link
   de origem, para auditoria futura.

## 3. Preservação de proveniência

- Cada documento ingerido mantém `fonte` e `caminho_origem` no manifesto
  (já implementado na Fase 2) — nenhuma transformação de texto pode apagar
  a rastreabilidade até a fonte original.
- Ao ingerir uma fonte real, usar `--fonte` explícito e estável por coleção
  (ex.: `wikipedia-pt`, `dominio-publico-literatura`) em vez do nome de
  subpasta padrão, para que a ATRIBUIÇÃO no relatório (`distribuicao_por_fonte`)
  reflita a origem real dos dados, não um acidente de organização de
  diretórios.

### Provedor (`fonte`) vs. unidade de agrupamento (correção da Sprint 5)

**Atenção, correção importante em relação a uma versão anterior deste
documento:** usar um único `--fonte` constante para um provedor GRANDE
(ex.: todo um dump da Wikipédia em português) é correto e recomendado para
fins de atribuição/auditoria, mas **não** deve, por si só, controlar o
agrupamento de splits — ver `src/ignis/data/ingest.py` e
`src/ignis/data/splits.py`. A versão anterior deste documento recomendava
usar `--fonte` estável "para que o agrupamento por fonte... reflita a
origem real dos dados", o que tinha um efeito colateral perigoso não
percebido na época: com `agrupar_por_fonte=True` (o padrão), TODOS os
documentos de uma mesma `fonte` caem obrigatoriamente no mesmo split —
então ingerir um provedor inteiro (milhares de artigos independentes) sob
um único `--fonte` o transforma em uma ÚNICA unidade indivisível, e esse
provedor inteiro vai parar inteiro em treino OU validação OU teste, nunca
distribuído entre os três. Isso torna splits de validação/teste pouco
realistas (ou impossíveis, se o provedor for a maioria do corpus) quando
esse provedor domina o corpus.

A correção (Sprint 5) separa dois conceitos antes colapsados em um só:

- **`fonte`**: rótulo de proveniência/atribuição. Pode e deve permanecer
  constante para um provedor inteiro, por maior que seja.
- **`unidade_agrupamento`**: a chave que de fato controla o agrupamento de
  splits em `ignis.data.splits.gerar_splits`. Por padrão, é o nome da
  PASTA IMEDIATA que contém o arquivo — diferente de `fonte` sempre que
  `--fonte` é passado explicitamente via `validate-corpus`.

**Prática recomendada para ingerir um provedor grande:** organizar os
arquivos em VÁRIAS subpastas (shards) — por exemplo, um lote por mês de
extração, por categoria de artigo, ou por intervalo de IDs — e ingerir
todas elas com o MESMO `--fonte wikipedia-pt`. Isso preserva a atribuição
correta no relatório (`distribuicao_por_fonte` mostrará "wikipedia-pt" como
uma única fonte) enquanto garante, por padrão, que cada subpasta vire uma
unidade de agrupamento independente — permitindo que o provedor seja
corretamente distribuído entre treino, validação e teste.

**Auditoria mecânica desta armadilha:** `ignis.cli report` agora inclui,
para cada fonte em `distribuicao_por_fonte.por_fonte`, o campo
`n_unidades_agrupamento` — o número de unidades independentes que aquela
fonte realmente contribui. Uma fonte com muitos documentos aceitos mas
`n_unidades_agrupamento == 1` está colapsada (todos os seus documentos
sempre cairão no mesmo split) e dispara um aviso explícito em
`avisos_concentracao`. **Revisar este campo é obrigatório antes de
prosseguir para o Passo 3 de `docs/plano-execucao-piloto.md`** quando
qualquer fonte representar uma fração substancial do corpus aceito.

### Ingestão incremental e retomável (Sprint 5)

`ignis.cli validate-corpus` aceita `--incremental` (carrega o manifesto
existente, se houver, e só reprocessa documentos novos ou com conteúdo
alterado — detectado por hash SHA-256 por `caminho_origem`, nunca
reprocessando do zero um diretório que só cresceu) e
`--limite-novos-documentos N` (processa no máximo N documentos
novos/atualizados nesta chamada, deixando o restante para uma execução
futura). Isso permite ingerir um corpus grande em lotes controlados,
chamando o mesmo comando repetidamente, em vez de processar/baixar tudo de
uma vez só. Limitação conhecida e documentada (ver docstring de
`ignis.data.pipeline.processar_diretorio`): a detecção de quase-duplicata
(Jaccard) só é recalculada dentro do lote de cada chamada, não
retroativamente contra documentos já aceitos em chamadas anteriores
(duplicatas EXATAS, por hash, continuam detectadas corretamente entre
chamadas).

## 4. Avaliação de qualidade e diversidade

Antes de aceitar uma fonte candidata em volume:

- Rodar `validate-corpus` em uma AMOSTRA pequena primeiro e inspecionar
  manualmente uma fração dos documentos `aceito`, `rejeitado` e
  `quarentena` — não confiar cegamente nos filtros automáticos (ver
  limitações já documentadas em `docs/data-card.md` sobre os limiares de
  qualidade e PII serem heurísticas não calibradas).
- Avaliar diversidade temática e estilística (evitar que 1–5M tokens venham
  de uma única fonte homogênea, ex.: só textos jurídicos ou só um único
  autor) — um corpus piloto pequeno e homogêneo tende a produzir um modelo
  que não generaliza nem para avaliação básica.
- Registrar a distribuição de fontes/tamanhos no relatório
  (`ignis.cli report`) e revisar antes de prosseguir.

## 5. Deduplicação e separação de conjuntos

- Usar o pipeline de deduplicação já implementado (`ignis.data.dedup`):
  exata por hash SHA-256 e quase-duplicatas por Jaccard sobre shingles,
  **sem alterações** de algoritmo até que uma necessidade real e medida
  justifique revisão do limiar (0,8).
- Gerar os splits com `prepare-splits`, preservando as garantias já
  implementadas e testadas (Fase 3, `src/ignis/data/splits.py`):
  documentos quase-duplicados e documentos da mesma fonte nunca são
  separados entre treino/validação/teste; se as unidades independentes
  disponíveis forem insuficientes para as proporções pedidas, o comando
  falha explicitamente (`GruposInsuficientesError`) em vez de gerar um
  split vazio silenciosamente — comportamento já coberto por testes
  automatizados (`tests/test_data_splits.py`).
- **O tokenizador BPE nunca é treinado sobre validação ou teste** — apenas
  sobre o split de treino (`train-tokenizer --splits ...`, já assim
  implementado). Isso deve ser conferido manualmente (não só assumido) ao
  rodar com dados reais, inspecionando o comando efetivamente executado.

## 6. Planejamento do BPE oficial de 16.384 entradas

- O vocabulário alvo (16.384 tokens) é parte da arquitetura já aprovada do
  Ignis-30M (`configs/ignis-30m.yaml`) — não deve ser alterado nesta etapa.
- Treinar o BPE real apenas depois que o corpus piloto de treino estiver
  selecionado, licenciado e dedupicado — nunca treinar um BPE "definitivo"
  sobre o corpus sintético de desenvolvimento.
- Após o treinamento, rodar `eval-tokenizer` sobre os três splits
  (principalmente treino e validação) e registrar: taxa de round-trip,
  tokens por caractere/palavra (eficiência), e qualquer falha de
  round-trip encontrada — critérios objetivos antes de declarar o
  tokenizador "pronto para o piloto".
- Um vocabulário de 16.384 entradas é pequeno para português (rico em
  flexão); é esperado — e deve ser medido, não assumido — uma eficiência de
  tokenização (tokens/palavra) pior do que a de tokenizadores maiores
  (32k–50k+) usados por LLMs comerciais. Isso deve constar no relatório do
  piloto como uma limitação conhecida, não ser escondido.

## 7. Avaliação do tokenizador em textos externos

Além do round-trip sobre os próprios splits, avaliar o tokenizador treinado
sobre um pequeno conjunto de textos em português **não usados em nenhum
split** (ex.: parágrafos digitados manualmente ou de uma fonte adicional,
claramente marcados como "fora do treino do tokenizador"), para detectar
overfitting do vocabulário aos dados de treino. Registrar taxa de
round-trip e eficiência também nesse conjunto externo, lado a lado com os
números internos.

**Suporte operacional já implementado (Sprint 4):** `ignis.cli eval-tokenizer`
aceita `--textos-externos <arquivo.txt>` (um texto por linha, nunca usado em
nenhum split) e inclui, na saída, uma seção `avaliacao_externa` separada com
round-trip e métricas de eficiência sobre esse conjunto — sem misturar os
números com os do split interno avaliado. Nenhum arquivo de textos externos
"oficial" foi criado ainda; esta é apenas a capacidade do comando.

## 8. Critérios para declarar o corpus/tokenizador "do piloto"

Nenhum dos dois pode ser chamado de "oficial do piloto Ignis-30M" até que,
simultaneamente:

1. Todas as fontes tenham licença e permissão de treinamento documentadas
   (seção 2), sem nenhuma pendência em `licenca_desconhecida` relevante.
2. O relatório de splits (`ignis.cli report`) mostre zero contaminação
   entre treino/validação/teste e proporções realizadas documentadas.
3. O tokenizador tenha sido avaliado (round-trip + eficiência) sobre
   treino, validação e um conjunto externo, com os números registrados
   neste repositório (não apenas observados uma vez e descartados).
4. O volume de tokens de treino esteja na faixa alvo (1–5M tokens) medido
   de fato pelas estatísticas de `prepare-dataset`, não estimado.

Só depois disso o corpus/tokenizador podem alimentar `configs/train-pilot.yaml`
(seção `dados`) para um piloto real — e mesmo assim, o início do
treinamento continua exigindo autorização explícita separada, conforme a
skill `ignis-llm-engineer`.

**Suporte operacional já implementado (Sprint 5):** `ignis.cli audit-tokenizer`
verifica mecanicamente os 4 critérios acima (licenças documentadas, zero
contaminação de splits, round-trip sem falhas em treino/validação/externo,
volume de tokens na faixa alvo medido por `prepare-dataset`) e só reporta
`oficial_do_piloto: true` quando TODOS passam simultaneamente — nunca
promove parcialmente. Ver `--textos-externos` (obrigatório aqui, diferente
de `eval-tokenizer`) e `--dataset-base`. Testado com corpus sintético
(5 fontes/temas distintos): todos os critérios passam exceto o volume de
tokens (esperado, dado o tamanho minúsculo do corpus de teste) —
comportamento confirmado em `tests/test_cli_audit_tokenizer.py`.

## 8.1. Nota sobre pesquisa de fontes nesta sessão (Sprint 5)

A ferramenta de busca na web esteve indisponível para esta organização
durante a Sprint 5 (erro `PROXY_REJECTED`, HTTP 403 — recurso não habilitado
para esta organização). Por isso, nenhuma informação de licenciamento,
disponibilidade de fonte ou termos de uso foi verificada ou atualizada por
pesquisa ao vivo nesta rodada — a tabela da seção 1 reflete apenas o que já
constava deste documento antes da Sprint 5. Qualquer decisão de ingestão
real deve, antes de prosseguir, verificar manualmente (ou com busca na web
habilitada) a licença/termos de uso atuais da fonte escolhida, já que
termos de licenciamento podem mudar com o tempo e não devem ser assumidos
como estáticos a partir deste documento.

## 9. O que este documento explicitamente NÃO faz

- Não baixa, não lista URLs de download diretas nem executa nenhuma coleta
  de dados.
- Não declara nenhuma fonte acima como já aprovada para uso — "candidata"
  significa apenas "razoável para avaliar", nunca "aprovada".
- Não substitui uma eventual revisão jurídica/de compliance da organização
  sobre uso de dados de terceiros para treinamento de modelos.
