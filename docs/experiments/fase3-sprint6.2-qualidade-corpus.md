# Sprint 6.2 — Consolidação do corpus e preparação científica

Data: 2026-10-18 (mesma sessão das Sprints 5/6/6.1). Ambiente: VM Linux sem
torch e sem rede (todo o trabalho deste relatório é local, sobre dados já
existentes); Windows nativo do usuário é a única máquina com torch,
`.venv-cuda` e acesso à rede — necessário para os itens 1 e 3 abaixo, que
ainda NÃO foram executados nesta sessão (ver seção "Pendências").

## Estado confirmado no início desta Sprint

234 artigos da Wikipédia em português preservados e íntegros (verificado
de novo via `--verificar`: 234/234 IDs batendo, 0 hashes inconsistentes, 0
pageids duplicados). Antes desta Sprint: 221 aceitos, 1 rejeitado, 12 em
quarentena, 232.538 tokens reais medidos no split de treino, tokenizador
explicitamente não oficial, 115 testes torch-free aprovados.

## 1. Consolidação da validação (Windows, `.venv-cuda`) — NÃO EXECUTADO NESTA SESSÃO

Esta VM de desenvolvimento não tem torch nem GPU — rodar a suíte completa
com `.venv-cuda` requer o Windows nativo do usuário. Comando exato a rodar
lá (nenhuma mudança de arquitetura, pesos ou dados é necessária para isso):
```powershell
.venv-cuda\Scripts\python.exe -m pytest tests\ -q
```
Resultado REAL ainda não obtido — **não declaro aprovação de testes que
não foram executados**. O que já está confirmado NESTA sessão (VM Linux,
sem torch) é a suíte torch-free completa: **134 passed, 2 skipped** (era
115 no início desta Sprint; +19 testes novos das correções de qualidade
abaixo), zero regressões. Os testes que dependem de torch (`test_causal_mask`,
`test_checkpoint`, `test_generation`, `test_gradients`, `test_model_shapes`,
`test_overfit_smoke`, `test_param_count`, `test_tied_embeddings`,
`test_training_checkpoint`, `test_training_dataloader`, `test_training_trainer`)
seguem pendentes de execução real no Windows.

## 2. Controle de qualidade — correções aplicadas e validadas com dados reais

### 2.1 Causa raiz do único documento rejeitado (artigo de matemática)

O artigo "Núcleo (álgebra linear)" foi rejeitado por "baixa razão de
caracteres alfabéticos (0,50 < 0,5)". Investigação: a razão era calculada
sobre o tamanho TOTAL do texto normalizado, mas a extração de fórmulas
matemáticas pela API da MediaWiki produz um artefato de formatação — cada
símbolo da fórmula vira uma "linha" curta cercada de espaços (renderização
em texto simples de MathML), inflando artificialmente a contagem de
caracteres não-alfabéticos sem refletir baixa qualidade de prosa real.

**Correção 1** (`src/ignis/data/filtros.py`): a razão agora é calculada
sobre caracteres NÃO-ESPAÇO, não sobre o total. Para texto normal (onde
espaços são ~15-20% do conteúdo) o efeito é desprezível; para o artefato
de fórmula espaçada, a diferença é decisiva. Verificado com o artigo real:
razão subiu de 0,50 (texto bruto) para 0,88 (após a correção 2 abaixo) —
folgadamente acima do limiar.

**Correção 2** (`src/ignis/data/normalize.py`): a API também embute o
código-fonte LaTeX bruto de cada fórmula logo após a renderização em texto
(bloco `{\displaystyle ...}`) — puro ruído de marcação, redundante com o
texto que já veio antes, sem nenhum valor para treinar um modelo de
português. Removido via um scanner de chaves balanceadas (não regex
simples, pois o LaTeX interno tem chaves aninhadas como `\left\{...\right\}`);
qualquer bloco mal formado (chave nunca fechada) é preservado sem cortar
texto, por segurança. Só este artigo teve ~4.875 caracteres de LaTeX bruto
removidos.

**Resultado real**: o artigo agora passa no filtro de qualidade. Com a
correção 1 sozinha já seria suficiente (razão sobre não-espaço = 0,76 sem
nem remover o LaTeX); as duas juntas dão 0,88 — ampla margem.

### 2.2 Causa raiz dos 12 documentos em quarentena por "possível PII"

Auditoria manual de cada um dos 12 trechos sinalizados (ver Sprint 6.1)
confirmou: **11 dos 12 eram falsos positivos causados por números de ISBN**
em referências bibliográficas (`cartao_credito` casando com ISBN-13 de 13
dígitos; `telefone_br` casando com ISBN-10 de 10 dígitos ou com o sufixo de
ISBN-13 após o hífen) e **2 ocorrências adicionais** (dentro de um dos 12
documentos) eram sufixos numéricos de **DOI** (`doi:10.1073/pnas.0505955102`),
comuns em referências de artigos científicos. Apenas **1 dos 12** (um
artigo sobre uma associação esportiva, citando o CNPJ histórico da
entidade) tinha um padrão de CNPJ genuíno — mas CNPJ identifica uma PESSOA
JURÍDICA (empresa), não uma pessoa física, e é informação pública (Receita
Federal); tecnicamente não é "PII" no sentido usual.

**Correções aplicadas** (`src/ignis/data/pii.py`):
- `cartao_credito`: mínimo elevado de 13 para 15 dígitos (cartões reais têm
  15 ou 16 dígitos; 13 dígitos é, neste corpus, exclusivo de ISBN-13).
- `cartao_credito` e `telefone_br`: qualquer ocorrência precedida pela
  palavra "ISBN" ou "doi" numa janela de 20 caracteres é descartada —
  contexto bibliográfico inequívoco (essas palavras sempre vêm
  imediatamente antes do número, por convenção editorial), verificado por
  PROXIMIDADE (não é uma whitelist cega de qualquer texto que mencione
  ISBN em algum lugar).
- `cnpj`: **mantido sem alteração, deliberadamente**. Mesmo não sendo PII
  de pessoa física, o princípio de "preservar em quarentena quando há
  incerteza" foi aplicado — um CNPJ pequeno pode, em outros contextos,
  vir acompanhado do nome do proprietário. Decisão documentada, não um
  descuido.

**Resultado real, medido com `validate-corpus` rodado do zero sobre os 234
artigos**:
```
antes:  221 aceitos, 1 rejeitado,  12 quarentena
depois: 233 aceitos, 0 rejeitados,  1 quarentena (só o CNPJ, mantido por decisão)
```
Idempotência incremental reconfirmada com a nova lógica: segunda execução
com `--incremental` processou 0 documentos novos, manifesto idêntico.

### 2.3 Testes de regressão novos (19, todos com exemplos sintéticos ou dados reais desidentificados)

- `tests/test_data_pii.py` (8 testes): ISBN-13/ISBN-10/DOI não geram mais
  falso positivo; número de 13 dígitos solto não é mais "cartão"; número
  de 16 dígitos continua detectado; telefone BR genuíno fora de contexto
  bibliográfico continua detectado; ISBN fora da janela de proximidade
  ainda é sinalizado (prova que a exceção é por proximidade, não global);
  CNPJ continua sinalizado (protege a decisão deliberada contra remoção
  futura acidental).
- `tests/test_data_normalize.py` (+5 testes): remoção de bloco
  `{\displaystyle...}` simples, com chaves aninhadas, múltiplos blocos no
  mesmo texto, bloco malformado (não corta texto), e texto sem fórmula
  (inalterado).
- `tests/test_data_filtros.py` (6 testes, arquivo novo): texto normal
  aceito, texto curto/longo rejeitado, sopa de números com espaçamento
  NORMAL continua rejeitada (a mudança não abre brecha para texto
  genuinamente degenerado), fórmula matemática espaçada sintética agora
  aceita, razão calculada sobre não-espaço confirmada diretamente.

Nenhum exemplo usa PII real de qualquer pessoa — todos os números de
cartão/telefone/ISBN/DOI nos testes são inventados ou são identificadores
bibliográficos públicos (ISBN/DOI de obras publicadas, não dados pessoais).

## 3. Retomada segura da coleta — inspeção feita, execução ainda NÃO realizada

Estado salvo (`corpus_real/wikipedia_pt/_estado_coleta.json`) inspecionado:
234 pageids únicos, 1.330.006 bytes baixados (1,27 MB de 500 MB autorizados
— 0,27% do teto), 200.842 tokens aproximados de ~1.000.000 (20%).
`--verificar` confirma integridade total (ver acima). Nenhum bloqueio
conhecido do lado do servidor (a última execução real parou pelo próprio
limite de segurança `--max-requisicoes`, não por erro).

Política de requisições já implementada (Sprint 6.1, reconfirmada nesta
Sprint por leitura de código — nada foi adicionado que contornasse limites):
`maxlag=5` em toda requisição, backoff que respeita `Retry-After` quando
presente, suspensão persistente após N lotes seguidos com 429/maxlag, sem
paralelismo (um único loop sequencial, uma requisição HTTP por vez), sem
rotação de identidade ou qualquer mecanismo de evasão de limite. Nenhuma
mudança foi feita para contornar os limites do servidor — essa política já
era a correta.

**Única mudança feita nesta Sprint na coleta**: o log por artigo
(`print` a cada artigo aceito) foi substituído por um resumo por LOTE (uma
linha a cada requisição, com contagem acumulada), reduzindo drasticamente o
volume de saída em execuções longas sem perder nenhum detalhe — cada artigo
continua tendo seu registro completo em `_metadados.jsonl`.

A meta continua ~1.000.000 tokens, exclusivamente Wikipédia em português,
dentro dos limites já autorizados (500 MB / sem teto de artigos) — nenhum
teto foi ampliado. **Comando seguro de retomada** (rodar no Windows, não
nesta VM, que não tem rede):
```powershell
python scripts\coletar_wikipedia_pt.py --confirmar --meta-tokens 1000000 --teto-mb 500
```

## 4. Estatísticas de coleta (estado atual, antes de qualquer nova coleta)

| Métrica | Valor |
|---|---|
| Artigos coletados | 234 |
| Artigos aceitos (pós Sprint 6.2) | 233 |
| Artigos em quarentena | 1 (CNPJ, decisão deliberada) |
| Artigos rejeitados | 0 |
| Transferência acumulada | 1,27 MB de 500 MB (0,27%) |
| Tokens aproximados (coleta) | 200.842 de ~1.000.000 (20%) |
| Tokens reais (BPE, split de treino) | 272.056 de faixa-alvo 1.000.000–5.000.000 |
| Requisições HTTP por status | não disponível para a coleta já feita (script não instrumentado na época — ver Sprint 6.1, seção 8); será capturado a partir da próxima execução real |

## 5. Situação do corpus e do tokenizador

`audit-tokenizer`, rodado de novo sobre o corpus corrigido:
```json
{
  "criterios": [
    {"nome": "licencas_documentadas", "ok": true},
    {"nome": "splits_sem_contaminacao", "ok": true},
    {"nome": "round_trip_sem_falhas:treino", "ok": true},
    {"nome": "round_trip_sem_falhas:validacao", "ok": true},
    {"nome": "round_trip_sem_falhas:externo", "ok": true},
    {"nome": "volume_de_tokens_na_faixa_alvo", "ok": false,
     "detalhe": "n_tokens_total=272056 (faixa alvo: 1000000-5000000, medido de fato)"}
  ],
  "oficial_do_piloto": false
}
```
Ainda **não oficial** — único critério que falha é volume, como esperado
(272.056 de 1.000.000-5.000.000). Tokenizador salvo como
`data/tokenizer_dryrun_naooficial.json` (nome deliberado). Splits
reproduzíveis: treino=210, validação=12, teste=11 (233 unidades
independentes), seed 42, sem contaminação.

## 6. Pendências para o primeiro treinamento piloto

1. Rodar a suíte completa no Windows com `.venv-cuda` (seção 1) — ainda
   não executado nesta sessão.
2. Retomar a coleta real (comando da seção 3) até aproximar-se de
   ~1.000.000 tokens — ainda não executado (requer rede, só no Windows).
3. Após nova coleta: `validate-corpus --incremental` → `prepare-splits` →
   `train-tokenizer` → `audit-tokenizer` de novo, até `oficial_do_piloto=true`.
4. Só então: dry-run de `ignis.cli train --device cuda` (sem
   `--iniciar-treinamento-real`) para validar memória/forward na GPU real,
   sem alterar pesos.
5. Pré-treinamento real continua não autorizado — qualquer início depende
   de nova autorização explícita e direta, como as anteriores.

## 7. O que esta Sprint explicitamente NÃO fez

Não alterou a arquitetura do Ignis-30M; não fez commits; não instalou outra
versão do PyTorch; não adquiriu fontes adicionais (só reavaliou os mesmos
234 artigos já coletados); não iniciou pré-treinamento; não ampliou nenhum
teto de coleta; não introduziu paralelismo ou rotação de identidade no
coletor; não declarou testes do Windows como aprovados sem tê-los
executado; não declarou o tokenizador/corpus oficial.
