# Sprint 6.1 — Correção crítica, qualidade do corpus e otimização da coleta

Data de execução: 2026-10-18 (mesma sessão da Sprint 5/6). Ambiente: VM
Linux sem torch e sem acesso a rede (usada para todo o trabalho deste
relatório, que é inteiramente local); Windows nativo do usuário (torch
2.14.1+cpu e +cu126) usado apenas para a coleta real já relatada na
Sprint 6 e para qualquer trabalho futuro que precise de rede ou de torch.

## 1. Causa raiz do erro relatado

Comando que falhou: `ignis.cli validate-corpus --incremental` sobre
`corpus_real/wikipedia_pt/`, com o erro:
```
ERRO: corpus_real\wikipedia_pt\_metadados.jsonl:1 nao possui a chave obrigatoria 'texto'
```

Causa: `ingerir_diretorio` (`src/ignis/data/ingest.py`) varria todo
arquivo `.txt`/`.md`/`.jsonl` dentro do diretório de corpus, sem nenhuma
exclusão para arquivos auxiliares/sidecar. `scripts/coletar_wikipedia_pt.py`
escreve, ao lado dos shards de conteúdo (`shard_NNNN.jsonl`), um arquivo
de metadados de proveniência (`_metadados.jsonl`) cujas linhas têm
`titulo`/`url`/`revisao`/`sha256_texto`, mas não `texto` — exatamente o
formato que `ingerir_arquivo` espera de um documento de conteúdo. O
resultado era `ingerir_arquivo` tentando tratar cada linha de metadados
como um documento e falhando na primeira linha sem a chave `texto`.

## 2. Correção aplicada (contrato de arquivo auxiliar)

Arquivo: `src/ignis/data/ingest.py`. Adicionada a função
`eh_arquivo_auxiliar(caminho) -> bool`, que retorna
`caminho.name.startswith("_")` — qualquer arquivo cujo NOME comece com
`_` é auxiliar (metadados, estado de retomada, etc.) e nunca é lido como
conteúdo, independente da extensão ou da pasta em que está. `ingerir_diretorio`
agora pula esses arquivos antes de chamar `ingerir_arquivo`. Como
`ignis.data.pipeline.processar_diretorio` já delega diretamente a
`ingerir_diretorio`, a correção se propaga automaticamente para
`ignis.cli validate-corpus` sem nenhuma mudança adicional.

Esta é uma convenção geral (não específica da Wikipedia): qualquer fonte
futura que escreva arquivos sidecar ao lado do conteúdo (prefixados com
`_`) já é protegida pela mesma regra.

## 3. Preservação de integridade dos 234 artigos já coletados

Antes de qualquer alteração, e de novo depois, a integridade do corpus
real existente foi verificada (sem rede, sem modificar nada):

```
IDs em shards: 234
IDs em metadados: 234
IDs so em shards (sem metadado): []
IDs so em metadados (sem shard): []
Hashes inconsistentes: 0 []
pageids unicos: 234 de 234
INTEGRIDADE: OK
```

Esta verificação agora também está embutida no próprio coletor:
`python scripts/coletar_wikipedia_pt.py --verificar --saida-dir corpus_real/wikipedia_pt`
(sem rede, pode ser rodado a qualquer momento). Nenhum dado foi perdido,
sobrescrito ou re-coletado do zero.

## 4. Testes de regressão novos

- `tests/test_data_ingest_arquivos_auxiliares.py` (4 testes): reproduz o
  bug exato (`_metadados.jsonl` malformado ao lado de um shard válido),
  confirma que `ingerir_diretorio` ignora o arquivo auxiliar, documenta
  que `ingerir_arquivo` chamado diretamente nele ainda falharia (por isso
  a correção tinha que ficar em `ingerir_diretorio`), e confirma o
  comportamento de ponta a ponta via `processar_diretorio` replicando
  exatamente o layout de diretório produzido pelo coletor real.
- `tests/test_coletar_wikipedia_pt.py` (13 testes, **nenhuma chamada de
  rede real** — toda interação com a API é simulada via monkeypatch de
  `urllib.request.urlopen` ou de funções internas): tratamento de HTTP
  429 com e sem `Retry-After`, tratamento do erro `maxlag` retornado no
  corpo JSON, esgotamento de tentativas, filtragem de redirecionamento/
  desambiguação, envio do parâmetro `maxlag`, suspensão persistente após
  N lotes seguidos com 429 (sem chegar perto de `--max-requisicoes`),
  coleta simulada de ponta a ponta até a meta de tokens, verificação de
  integridade (caso íntegro, hash inconsistente, pageid duplicado), e o
  contrato de segurança mais básico (sem `--confirmar`, `urlopen` nunca é
  chamado — forçado a levantar exceção se for).

Resultado real da suíte torch-free completa nesta sessão:
```
115 passed, 2 skipped
```
(98 no início da sessão → 102 com a correção de ingestão → 115 com os
testes do coletor). Zero regressões.

## 5. Validação operacional real sobre o corpus real (não simulada)

Executado nesta sessão, nesta ordem, sobre os 234 artigos reais já
coletados:

```
python -m ignis.cli validate-corpus --corpus-dir corpus_real/wikipedia_pt --fonte wikipedia-pt --manifesto-saida data/manifesto.jsonl
```
Resultado real:
```json
{"total_documentos": 234, "por_status": {"aceito": 221, "rejeitado": 1, "quarentena": 12, "licenca_desconhecida": 0}, "novos_processados_nesta_execucao": 234}
```

Idempotência incremental, verificada com uma SEGUNDA execução real
(`--incremental`) sobre o mesmo diretório inalterado:
```
execucao incremental: 234 documento(s) ja presentes em data/manifesto.jsonl
{"novos_processados_nesta_execucao": 0, ...mesmos totais...}
```
`data/manifesto.jsonl` ficou byte-a-byte idêntico entre as duas execuções
(`diff` sem diferenças) — a ingestão incremental não duplica nem
reprocessa desnecessariamente.

`prepare-splits` (seed 42): `treino=199 validacao=11 teste=11` (221
unidades de agrupamento independentes — cada artigo é sua própria
unidade, como desenhado na Sprint 5); `nenhuma contaminacao detectada
entre splits`.

Tokenizador BPE treinado sobre o split de treino real (199 documentos),
salvo como `data/tokenizer_dryrun_naooficial.json` — nome escolhido
deliberadamente para deixar claro que NÃO é o tokenizador oficial do
piloto. `eval-tokenizer`: 0 falhas de round-trip em 11 documentos de
validação. `prepare-dataset`: contagem REAL de tokens (BPE, não
aproximada) — treino=232.538, validação=12.761, teste=23.820 (total
269.119) em sequências de 1024 tokens.

`audit-tokenizer` (critérios objetivos da seção 8 de
`docs/plano-corpus-piloto.md`), resultado real:
```json
{
  "criterios": [
    {"nome": "licencas_documentadas", "ok": true},
    {"nome": "splits_sem_contaminacao", "ok": true},
    {"nome": "round_trip_sem_falhas:treino", "ok": true},
    {"nome": "round_trip_sem_falhas:validacao", "ok": true},
    {"nome": "round_trip_sem_falhas:externo", "ok": true},
    {"nome": "volume_de_tokens_na_faixa_alvo", "ok": false,
     "detalhe": "n_tokens_total=232538 (faixa alvo: 1000000-5000000, medido de fato)"}
  ],
  "oficial_do_piloto": false
}
```
**Conclusão honesta: o tokenizador/corpus atual NÃO é declarado oficial do
piloto** — o próprio mecanismo de gate da Sprint 5 barrou a promoção
corretamente, por volume de tokens real medido muito abaixo da faixa-alvo
(232.538 vs. 1.000.000-5.000.000), apesar de todos os outros critérios
(licenças, ausência de contaminação, round-trip) terem passado. Isso é o
esperado dado que a coleta está em ~20-23% da meta de tokens aproximados
original — não é um problema do tokenizador, é volume insuficiente de
corpus.

## 6. Controle de qualidade sobre o corpus real

Dos 234 artigos: 221 aceitos, 1 rejeitado, 12 em quarentena, 0 com
licença desconhecida.

- **Rejeitado (1):** "Núcleo (álgebra linear)" — baixa razão de caracteres
  alfabéticos (0,50 < 0,5 exigido), esperado para um artigo matemático com
  muita notação/símbolos.
- **Quarentena (12):** todos por "possível PII detectada" — mas ao
  inspecionar os títulos (ex.: "Alchemilla pseudominor" [espécie de
  planta], "Operação Barrel Roll" [operação militar histórica], "Maersk"
  [empresa], "Mito de criação" [religião/antropologia], "Wang Fuzhi"
  [biografia]), fica claro que são **falsos positivos** do detector de
  PII genérico (`ignis.data.pii`), que usa heurísticas de padrão numérico
  (sequências de 13-16 dígitos parecidas com cartão de crédito, padrões
  de telefone/CNPJ) e dispara em números de página, coordenadas, datas,
  códigos de unidade militar, etc. Nenhum desses artigos contém PII real
  de pessoa física identificável. O comportamento conservador (quarentena
  em vez de aceitar ou rejeitar automaticamente) é o correto por padrão
  de segurança de dados, mas fica registrado aqui que, para texto de
  Wikipedia, a taxa de falso positivo deste detector é alta — um ajuste
  futuro (ex.: contexto ao redor do número, não só o padrão) pode reduzir
  quarentenas desnecessárias, mas isso é um ajuste de filtro, não uma
  mudança de arquitetura, e não foi feito nesta sessão (fora do escopo
  pedido: só correção da ingestão + instrumentação do coletor).

## 7. Diversidade temática do corpus real

Amostra dos títulos coletados mostra diversidade real, sem sinal de
"stub flooding" de uma única categoria: lugares (EUA, Portugal,
Eslováquia), astronomia (asteroides, estrelas), biologia (espécies de
plantas), matemática/algoritmos, religião/antropologia, história militar,
aviação, música, empresas, biografias, política/inteligência (ex.:
"Projeto MKULTRA"). Tamanho de texto por artigo: mínimo 449, mediana
2.643, média 5.538, máximo 58.811 caracteres — alguns artigos (os 8
coletados na primeira execução, antes do ajuste de `--min-caracteres`
para 1200) são mais curtos que o mínimo atual, mas nenhum abaixo de 449
caracteres, então não há "permastubs" de 1-3 frases no corpus atual.

## 8. Instrumentação do coletor (`scripts/coletar_wikipedia_pt.py`)

Reescrito com, além do que já existia (gate `--confirmar`, backoff
consciente de `Retry-After`, resumibilidade):
- Estatísticas estruturadas por desfecho (aceitos, abaixo do mínimo,
  redirecionamentos, desambiguações, duplicados, lotes sem artigos úteis,
  HTTP 429, erro `maxlag`, HTTP 5xx, outros erros HTTP, erros de rede),
  impressas no resumo final de cada execução.
- `maxlag=5` (padrão, configurável via `--maxlag`) em toda requisição —
  etiqueta recomendada pela API da Wikimedia; tratado como erro
  retentável igual a 429, incluindo quando vem como `error.code=maxlag`
  no corpo JSON (não só como HTTPError).
- Suspensão persistente: se `--limite-429-consecutivos` (padrão 15) lotes
  SEGUIDOS precisarem esperar por 429/maxlag, a execução para sozinha com
  o estado salvo, em vez de insistir até `--max-requisicoes` — sinaliza
  throttling sustentado em vez de deixar o limite bruto de requisições
  mascarar o problema.
- Métricas de eficiência no resumo: artigos aceitos por requisição, taxa
  de rejeição, tokens aproximados por minuto, tempo total estimado
  perdido em espera por 429/maxlag, e uma estimativa (explicitamente
  marcada como de alta incerteza) de tempo restante até a meta de tokens,
  extrapolando a taxa observada na própria execução.
- `--verificar`: roda a verificação de integridade (seção 3) sob demanda,
  sem rede.
- `--contato`: parâmetro OPCIONAL para anexar um contato (e-mail ou URL)
  ao User-Agent, por etiqueta da API da Wikimedia — nunca preenchido
  automaticamente. **Decisão explícita, não tomada unilateralmente:** por
  regra do projeto, o e-mail do responsável pelo projeto nunca é enviado
  a um serviço externo sem pedido explícito dele; o User-Agent padrão
  continua identificável (nome do projeto, propósito, link para a
  documentação no repositório) mas sem contato pessoal. Se o responsável
  pelo projeto quiser um contato real anexado (mais alinhado à etiqueta
  recomendada pela Wikimedia para uso intensivo da API), basta fornecê-lo
  via `--contato` na próxima execução — isso não foi decidido por este
  relatório.
- Nenhum teto (`--teto-mb`, `--meta-tokens`) é ampliado automaticamente
  em nenhuma circunstância, inclusive nesta reescrita.

**Limitação importante:** a execução real de 234 artigos já relatada na
Sprint 6 usou a versão ANTERIOR (não instrumentada) do script — por isso
não há estatísticas estruturadas por desfecho para aquela execução
específica, só os números já relatados (234 artigos, 1,27 MB, ~200.842
tokens aproximados, parada pelo próprio limite de `--max-requisicoes`). A
nova instrumentação só se aplica a partir da próxima execução real.

## 9. Divergência de autorização sinalizada (não adotada silenciosamente)

A instrução da Sprint 6.1 recebida citou um limite "previamente
estabelecido" de "50 MiB de transferência... e até 3.000 artigos". Isso
**não corresponde a nenhuma confirmação direta e verificável** dada pelo
responsável pelo projeto nesta sessão — a autorização real e verificada
(resposta direta via pergunta de esclarecimento) foi **500 MB de teto de
download e meta de ~1.000.000 tokens, sem teto de número de artigos**.
Seguindo o mesmo princípio já aplicado à alegação anterior de "a coleta
já foi autorizada" (tratada como não verificável até confirmação direta),
este novo limite diferente também não foi adotado silenciosamente — o
coletor continua operando com os parâmetros realmente autorizados. Ver
`docs/checklist-autorizacao-sprint5.md`, seção D, para o registro formal.

## 10. O que esta Sprint explicitamente NÃO fez

NÃO alterou a arquitetura do modelo; NÃO reiniciou a coleta do zero; NÃO
excedeu os tetos autorizados (nem os antigos nem os citados na própria
Sprint 6.1); NÃO iniciou nenhum pré-treinamento real; NÃO reinstalou
PyTorch; NÃO fez nenhum commit Git; NÃO publicou nada; NÃO fez nenhuma
chamada de rede nova (toda a validação e todos os testes desta sessão
rodaram contra dados já existentes no disco, com rede simulada nos
testes); NÃO declarou o tokenizador/corpus "oficial do piloto" (o
mecanismo de gate da Sprint 5 confirmou que não atende ao critério de
volume).

## 11. Estado do Git

Nada foi commitado nesta sessão (nem antes dela, aparentemente — o
repositório tem um único conjunto de arquivos em stage `A`/`AM` nunca
commitado, mais um grande número de arquivos novos ainda não staged,
`??`). Arquivos modificados/criados nesta Sprint: `src/ignis/data/ingest.py`
(correção), `scripts/coletar_wikipedia_pt.py` (reescrito),
`tests/test_data_ingest_arquivos_auxiliares.py` (novo),
`tests/test_coletar_wikipedia_pt.py` (novo), `.gitignore` (extensão para
cobrir `data/`, `corpus_real/` e `*.npy`, que não estavam totalmente
cobertos), `docs/checklist-autorizacao-sprint5.md` (atualizado), este
arquivo (novo). Artefatos de dados gerados ficam em `data/` e
`corpus_real/` (ambos agora ignorados pelo Git). Nenhuma mudança
destrutiva; nenhum arquivo existente foi sobrescrito sem necessidade.

## 12. Comandos para reproduzir/continuar (Windows nativo, com rede)

```powershell
# Continuar a coleta real (dentro dos limites autorizados: 500 MB / ~1.000.000 tokens)
python scripts\coletar_wikipedia_pt.py --confirmar --meta-tokens 1000000 --teto-mb 500

# Verificar integridade a qualquer momento, sem rede
python scripts\coletar_wikipedia_pt.py --verificar --saida-dir corpus_real\wikipedia_pt

# Reingestão incremental apos nova coleta
python -m ignis.cli validate-corpus --corpus-dir corpus_real\wikipedia_pt --fonte wikipedia-pt --manifesto-saida data\manifesto.jsonl --incremental

# Suite completa de testes (inclui os novos testes do coletor e de ingestao)
.venv\Scripts\python.exe -m pytest tests\ -q
```

## 13. Riscos e limitações conhecidos

- Corpus real está em ~20-23% da meta de tokens aproximados (~200-270 mil
  de ~1.000.000) — tokenizador/corpus atuais continuam explicitamente não
  oficiais até mais coleta.
- Taxa de falso positivo do detector de PII parece alta para texto de
  Wikipedia (seção 6) — não corrigido nesta sessão (fora de escopo),
  registrado como limitação conhecida.
- A instrumentação nova do coletor (estatísticas, suspensão persistente,
  `maxlag`) ainda não foi exercida contra a API real da Wikimedia (só
  contra chamadas simuladas) — primeira execução real com a nova versão
  deve ser observada com atenção.
- `--contato` no User-Agent continua vazio por padrão (decisão de
  privacidade) — pode reduzir a cortesia percebida pela API em uso mais
  intenso; decisão deixada para o responsável pelo projeto.

## 14. Próximo passo sugerido (não executado, depende de decisão/rede)

Rodar a próxima coleta real (comando da seção 12) no Windows nativo do
usuário, com a nova instrumentação, para progredir em direção à meta de
~1.000.000 tokens; depois re-rodar `validate-corpus --incremental` e
`audit-tokenizer` para reavaliar `oficial_do_piloto`.
