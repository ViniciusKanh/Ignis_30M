# Sprint 6.2 — conclusão da validação local no Windows

Data: 20/10/2026. Repositório existente: `E:\Models\Ignis`.
Esta execução complementa e corrige o relatório histórico
`fase3-sprint6.2-qualidade-corpus.md`; as afirmações históricas sobre remoção
segura de LaTeX e exclusividade de ISBN para 13 dígitos não são adotadas.

## Estado recuperado

O checkout já continha as correções de ingestão de auxiliares da Sprint 6.1,
os filtros sobre caracteres não brancos, as exceções de ISBN/DOI, a remoção
balanceada de displaystyle, os 19 testes relatados anteriormente e a
instrumentação do coletor. O Git tinha muitos arquivos A/AM e não rastreados;
não há uma revisão commitada da Sprint 6.2 que permita atribuir cada mudança
a um autor. Todo esse trabalho foi preservado. A suíte Windows ainda não tinha
evidência no relatório anterior. Coleta e treinamento continuavam pendentes.

## Correções desta execução

- `src/ignis/data/normalize.py`: preserva integralmente fórmulas LaTeX,
  números e notação. Não é seguro presumir que toda fórmula displaystyle é
  redundante com o texto extraído; removê-la causava perda de informação.
- `src/ignis/data/pii.py`: exceções restritas aos intervalos sintáticos de
  ISBN/DOI; palavras próximas já não ocultam um telefone separado. Restaura
  triagem conservadora de sequências de 13–19 dígitos: cartões de 13 dígitos
  não podem ser descartados como impossíveis. A heurística sinaliza candidatos,
  não comprova identidade pessoal. CNPJ permanece em quarentena.
- `src/ignis/data/pipeline.py`: quase-duplicatas são comparadas também com
  textos aceitos inalterados do mesmo diretório em execuções incrementais.
- `src/ignis/cli.py`: audit-tokenizer exige vocabulário efetivo de 16.384,
  cobertura exata dos aceitos nos splits, ausência de vazamento pelos hashes,
  unidades e grupos, e leitura de todos os documentos solicitados no round-trip.
- Regressões em `tests/test_data_normalize.py`, `tests/test_data_pii.py`,
  `tests/test_data_pipeline_incremental.py`, `tests/test_cli_audit_tokenizer.py`.
- Novo `scripts/validar_sprint62.py`: validação local reproduzível, sem rede,
  sem ajuste do BPE e sem treinamento de pesos. Preserva os splits históricos.

## Medições atuais

| Medida | Resultado |
|---|---:|
| Artigos em shards e metadados | 234 / 234 |
| IDs únicos de conteúdo | 234 |
| Hashes inconsistentes / pageids duplicados | 0 / 0 |
| Aceitos / rejeitados / quarentena | 233 / 0 / 1 |
| Licença desconhecida | 0 |
| Novos no segundo passe incremental | 0 |
| Manifesto byte a byte idêntico no segundo passe | sim |
| Splits treino / validação / teste | 210 / 12 / 11 |
| Vocabulário BPE efetivo | 16.384 |
| Round-trip de documentos locais | 234 / 234, sem falhas |
| Round-trip externo | 5 / 5, sem falhas |
| Tokens treino / validação / teste | 275.803 / 14.162 / 18.200 |
| Tokens totais | 308.165 |

A diferença frente aos 272.056 tokens históricos de treino decorre da
preservação do conteúdo matemático com o mesmo BPE existente. Não se ajustou
BPE em validação ou teste nesta sessão. O arquivo histórico do BPE foi mantido;
a proveniência de seu ajuste anterior depende do pipeline e dos registros
históricos, não de um novo treinamento nesta execução.

Auditoria contextual: 106 ocorrências candidatas em 14 documentos; 103 em
referências ISBN/DOI, três CNPJs no mesmo documento. Contextos reais revisados,
com números mascarados nos registros, sem liberação por títulos. A contagem
inclui sobreposição entre detectores; não são 106 identificadores independentes.
A ampliação conservadora para 13–19 dígitos explica a diferença frente às
contagens do relatório anterior. O documento em quarentena não foi liberado.

Sem duplicatas exatas aceitas e sem vazamento pelos IDs, hashes normalizados,
unidades ou grupos de quase-duplicata (Jaccard de shingles, limiar 0,8).
Os IDs do teste histórico foram congelados e reaproveitados, sem novo sorteio.
Esta é evidência sob essa heurística, não garantia de ausência de toda
similaridade semântica ou de ausência universal de PII.

`audit-tokenizer` terminou com código **1 esperado**, `oficial_do_piloto=false`.
Todos os nove critérios passaram exceto volume de treino: 275.803 fora da faixa
1.000.000–5.000.000. O tokenizer continua `tokenizer_dryrun_naooficial.json`.
As estatísticas de tokenização foram regeneradas para a auditoria atual.

Evidências em `data/sprint62_validacao_20261020/`: `evidencias.json`,
`auditoria_pii_mascarada.json`, `audit-tokenizer.json`, `execucao.log`, manifesto,
splits e datasets experimentais. Os arquivos anteriores em `data/` foram
preservados e não devem ser confundidos com os artefatos atuais.
Os SHA-256 dos dois shards, metadados e estado de coleta foram iguais antes e
depois da execução e constam de `evidencias.json`.

## Ambiente e testes executados

Windows; `.venv-cuda\Scripts\python.exe`; PyTorch `2.14.1+cu126`;
CUDA disponível; NVIDIA GeForce RTX 4050 Laptop GPU.
Disponibilidade de CUDA não significa que todos os testes rodaram na GPU:
os testes usam os dispositivos definidos por suas próprias implementações.

- Primeira rodada direcionada: 24 aprovados, seis erros de criação de
  temporários fora da área permitida. Resolvido usando TEMP/TMP locais.
- Após a primeira correção: 30 aprovados. A restauração do detector de
  13 dígitos revelou uma falha de limite de intervalo (espaço final do match),
  corrigida antes da rodada final.
- Rodada direcionada final: **37 passed in 8.35s**.
- Uma única suíte completa: **186 passed, 21 warnings in 52.45s**.
  Avisos: API depreciada `torch.cuda.amp.GradScaler` em trainer.py:186.
  Nenhum teste omitido para obter aprovação.

Comando completo efetivamente usado:
```powershell
$env:TEMP = 'E:\Models\Ignis\data\temp_sprint62'
$env:TMP = $env:TEMP
$env:PYTEST_ADDOPTS = '--basetemp=data/temp_sprint62/completa'
.venv-cuda\Scripts\python.exe -m pytest -q
```
Os testes de aprendizagem são testes sintéticos reduzidos da suíte existente;
não constituem treinamento do corpus real ou atualização de checkpoints existentes.

## Pendências antes do piloto

1. Volume real de treino abaixo do gate. Novos dados dependem de tarefa e
   autorização próprias; nenhuma coleta foi realizada nesta execução.
2. Após a futura expansão, manter o teste congelado, atribuir novos grupos
   sem vazamento, ajustar BPE exclusivamente em treino e repetir preparo/auditoria.
3. Antes de promover artefatos oficiais, registrar fingerprints de corpus,
   configuração, splits e tokenizer no pacote definitivo do piloto; o gate
   atual lê estatísticas fornecidas e não prova sozinho a proveniência histórica
   de um BPE nem vincula criptograficamente estatísticas antigas ao corpus.
4. Executar o dry-run CUDA do piloto com os artefatos aprovados e obter
   autorização específica para treinamento real, orçamento e checkpoints.
5. Revisão humana continua necessária para casos inconclusivos e para qualidade
   linguística/semântica que os filtros automáticos não comprovam.

Arquitetura de 30.416.384 parâmetros preservada. Nenhuma instalação, rede,
coleta, publicação, commit, push ou treinamento real de pesos nesta execução.

Próximo comando local recomendado, apenas para reproduzir a validação:
```powershell
.\.venv-cuda\Scripts\python.exe scripts\validar_sprint62.py
```
