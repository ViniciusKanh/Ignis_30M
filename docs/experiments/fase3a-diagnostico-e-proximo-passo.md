# Diagnóstico após a fase 3A e próximo experimento de dados

## Execução medida

Script novo: scripts/diagnosticar_modelo.py. Executado em CUDA na RTX 4050,
torch 2.14.1+cu126, sem atualização de pesos, alteração de dados ou uso do
teste final. Checkpoint best.pt da fase 3A, passo local 1.000, SHA-256
6a74b44292698d77683f705a85b491826bb76cd3a2005adbf950ec693f1c0e86.
Todos os arquivos de entrada foram verificados por hash antes e depois.
Relatório bruto: diagnostico_modelo_20261020_210956_712451/relatorio.json.

| Split completo | Sequências | Posições-alvo | Loss | Perplexidade |
|---|---:|---:|---:|---:|
| Treino | 11.867 | 1.507.040 | 5,47170162 | 237,8646 |
| Validação | 664 | 84.265 | 5,95206217 | 384,5455 |

O gap de loss é 0,48036055. O modelo tem pior desempenho fora do treino;
isso não estabelece, isoladamente, a gravidade do sobreajuste. A avaliação
em lotes de oito difere da avaliação histórica em lote um em cerca de 3e-8
na loss de validação. Os valores são numericamente consistentes.

Referência unigrama ajustada somente nos alvos do treino, com suavização
aditiva alpha=1 por token: loss 7,54148816 no treino e 7,63549839 na
validação. O Transformer supera essa referência simples. A referência não
constitui uma arquitetura concorrente forte nem mede conhecimento factual.

## Controle de contexto

Foram selecionados 512 blocos completos da validação, seed 42, com um alvo
por bloco: o próximo token após o final da entrada. Nos pares original e
prefixo embaralhado, o alvo, o último token da entrada, o comprimento e o
multiconjunto do prefixo são iguais. Somente a ordem anterior ao último
token muda. Não foram usados prefixos ou alvos do teste.

| Condição | Loss média dos 512 alvos |
|---|---:|
| Prefixo original | 5,91369886 |
| Prefixo anterior ao último token embaralhado | 6,56193468 |
| Último token apenas | 6,68581022 |

A perturbação da ordem aumenta a loss em 0,64823582. A ordem original
beneficia 68,55% dos alvos. Isso indica sensibilidade útil ao contexto nesta
amostra; não demonstra compreensão ou ausência de bugs em todo o sistema.
O controle último-token-apenas muda também o comprimento. Há somente uma
seed, e blocos adjacentes não são documentos estatisticamente independentes.
Essas médias não são as losses da validação completa e não devem substituí-las.

## Composição local

Os 1.063 documentos de treino são da mesma fonte: wikipedia-pt. Os hashes
dos textos normalizados foram conferidos contra o manifesto. Um detector
exploratório exigindo dois termos distintos por grupo encontrou sinais em
203 documentos para história/política, 146 para artes/entretenimento, 123
para geografia/demografia, 92 para esportes, 17 para energia, 14 para
computação e 8 para matemática. 588 não atenderam a nenhum indicador.

Os grupos se sobrepõem e o vocabulário é limitado. Esses números NÃO são
uma classificação temática validada nem proporções verdadeiras dos domínios.
Os léxicos, critérios e exemplos por ID estão no JSON, para inspeção.

O treino tem 1.518.907 tokens BPE incluindo especiais. Os aproximadamente
8,94 milhões de posições-alvo processadas ao longo das três fases incluem
repetição do mesmo corpus e não representam esse volume de texto novo.

## Decisão proposta

Não modificar a arquitetura ou o tokenizador agora. Não iniciar outro ciclo
idêntico de treinamento antes de ampliar e revisar os dados. Os diagnósticos
sustentam a aprendizagem de padrões e contexto; a insuficiência de dados e
de cobertura é uma hipótese prioritária, não uma causa única comprovada.

Próximo experimento de coleta para aprovação:

- Fonte mantida: Wikipédia em português, pela API do coletor existente.
- Meta acumulada: 3.000.000 tokens aproximados por separação em espaços;
  o estado atual registra 1.004.991. Não confundir com tokens BPE.
- Teto acumulado de 500 MB segundo a métrica atual do coletor: bytes de
  texto UTF-8 de extratos, não o tráfego HTTP completo nem uma cota de GPU.
- Máximo de 2.500 lotes de consulta nesta execução; manter pausa, maxlag,
  tratamento de Retry-After e suspensão por throttling existentes.
- Preservar artigos e estado existentes, com retomada incremental. O estado
  atual registra 1.146 páginas e 6.689.475 bytes de texto. Não executar
  repetidamente para contornar o limite sem avaliar o resultado.
- A coleta não garante que a meta seja atingida nem resolve a diversidade
  temática por si só: a amostragem atual é aleatória. Revisar a cobertura
  antes de autorizar fontes ou estratégias adicionais.

Plano sem chamadas de rede, já executado com sucesso:

```powershell
.\.venv-cuda\Scripts\python.exe scripts\coletar_wikipedia_pt.py --meta-tokens 3000000 --teto-mb 500 --max-requisicoes 2500 --saida-dir corpus_real\wikipedia_pt
```

O aumento da meta requer autorização específica conforme AGENTS.md e a
skill ignis-llm-engineer. Nenhuma execução com --confirmar foi feita.

Após eventual coleta aprovada, preparar nova versão de manifesto e treino/
validação, preservando as atribuições históricas e o teste final congelado.
Auditar licença/PII, duplicatas e quase-duplicatas entre splits. Usar o mesmo
tokenizador dos pesos atuais, sem reajustar BPE, e novos caminhos para os
datasets. Auditar as seções editoriais antes de aplicar qualquer remoção.
Novos textos podem aparecer somente nos novos arquivos versionados; manter
uma validação histórica comum para comparação com os checkpoints anteriores.
Só então definir novo orçamento e dry-run de treinamento.

## Reprodutibilidade

```powershell
cd E:\Models\Ignis
.\.venv-cuda\Scripts\python.exe -u scripts\diagnosticar_modelo.py --device cuda
```

O diagnóstico real já foi concluído; repetir é opcional. Quatro testes
direcionados passaram em 6,05 s, verificando o controle de contexto,
ponderação por alvos válidos, ausência de gradientes e limites dos sinais
lexicais. Nenhuma dependência foi instalada e nenhum commit/push foi feito.
Suíte completa após a implementação: **202 passed, 25 warnings in 48.06s**,
com TEMP/TMP e basetemp novos dentro do checkout e uma thread OMP/MKL.
Os avisos existentes são da API depreciada de GradScaler. Nenhum teste excluído.
