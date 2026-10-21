# Ampliação do corpus — análise e tentativa de retomada

Data: 20/10/2026. Skill ignis-llm-engineer e AGENTS.md aplicados.

O responsável confirmou nesta conversa a intenção de ampliar o corpus após
a validação da Sprint 6.2. Escopo adotado: continuar a Wikipédia em português,
meta operacional acumulada de 1.000.000 tokens aproximados e teto existente
de 500 MB de texto; nenhuma nova fonte ou treinamento de pesos.

## Confronto da visão com o checkout

- Modelo próprio decoder-only, inicialização aleatória e embeddings empatados
  confirmados em src/ignis/model/transformer.py. Arquitetura aprovada mantida
  em configs/ignis-30m.yaml: 7 camadas, 512 dimensões, 8 Q / 4 KV, SwiGLU
  1536, RoPE, RMSNorm, vocabulário 16.384 e contexto máximo 1.024.
- A visão geral fornecida está alinhada ao projeto. A prioridade implementada
  é linguagem geral em português; programação é domínio secundário no roadmap,
  e RAG é camada futura externa. Não há capacidade conversacional demonstrada.
- configs/train-pilot.yaml contém hipóteses de um piloto curto, não um treino
  aprovado: 200 passos, microbatch 1, acumulação 32, blocos de 128 tokens
  (contexto preditivo de 127), device CPU por padrão e caminhos ainda genéricos.
  Isso não equivale a treinamento já realizado nem a contexto treinado de 1.024.
- O corpus atual é enciclopédico em português e não foi medido como
  exclusivamente brasileiro. Diversidade temática não comprova cobertura de
  conversação, instruções ou programação.

## Estado antes da tentativa

234 artigos; 1.330.006 bytes de texto; 200.842 palavras/tokens aproximados
no estado de coleta. A validação anterior mediu 233 aceitos, um em quarentena,
275.803 tokens BPE de treino; tokenizer experimental não oficial.

## Tentativa desta sessão

Comando executado com lote inicial restrito:
```powershell
.\.venv-cuda\Scripts\python.exe -u scripts\coletar_wikipedia_pt.py --confirmar --meta-tokens 1000000 --teto-mb 500 --max-requisicoes 1 --tamanho-lote 10 --pausa-s 5 --limite-429-consecutivos 3
```
A resolução de nome falhou: `urlopen error [Errno 11001] getaddrinfo failed`.
Não houve recebimento de novos artigos. Não se tentou contornar a restrição.
A coleta real precisa ocorrer em uma sessão com acesso de rede permitido.

## Pontos a resolver na expansão

1. A métrica de tokens do coletor é len(texto.split()), não BPE. O critério
   final é 1–5 milhões de tokens BPE de TREINO após triagem e splits; atingir
   a meta aproximada não garante passar no gate.
2. O teto atual contabiliza extratos aceitos, não todo o corpo JSON,
   respostas descartadas ou tentativas HTTP. Não deve ser descrito como
   limite rigoroso de tráfego total. A medição histórica desse tráfego é
   desconhecida; instrumentação futura não pode reconstruí-la.
3. Preservar teste histórico; não repetir prepare-splits indiscriminadamente,
   pois sua redistribuição global pode mover documentos antigos. Novos grupos
   exigem atribuição compatível com o teste congelado e deduplicação global.
4. scripts/validar_sprint62.py é reprodução da fotografia da Sprint 6.2:
   exige a cobertura dos splits históricos. Depois da coleta, poderá falhar
   legitimamente nessa asserção até preparar splits expandidos.
5. Preservar BPE e datasets antigos. Ajuste futuro do BPE somente em treino,
   em novo caminho experimental, com hashes e configuração registrados.
6. Atualizar a verificação das licenças e termos da fonte antes da nova
   ingestão. O registro local de permissao_treinamento não substitui isso.
7. Triagem, revisão contextual da quarentena, idempotência, auditoria e
   dry-run CUDA antecedem qualquer autorização específica para treino real.

Nenhum módulo, arquitetura, dependência ou peso foi alterado nesta tentativa.


## Retorno do usuario e verificacao posterior (mesma data)

A execucao externa enviada pelo usuario concluiu 100 consultas em 590,2 s:
39 artigos novos, 273 acumulados; 231.935 tokens aproximados acumulados,
1.528.069 bytes de texto no estado local; oito respostas HTTP 429,
sem erros de rede. Parada por limite de consultas, nao por meta de volume.
O estado local foi conferido: 273 artigos/273 metadados, hashes consistentes,
pageids unicos. Nova triagem em caminho separado, sem substituir os artefatos
anteriores: data/manifesto_expansao_20261020.jsonl, 272 aceitos, zero rejeitados,
um em quarentena e zero licencas desconhecidas. Nao foram refeitos os splits
nem ajustado o tokenizer. Os numeros BPE da Sprint 6.2 nao representam o
corpus expandido.

Diagnostico confirmado pela documentacao oficial de TextExtracts:
https://www.mediawiki.org/wiki/Extension:TextExtracts#API
Multiplos extratos exigem exintro=true, que restringe o conteudo a introducao.
O coletor solicitava 50 paginas, mas obtinha um extrato completo por consulta.
Ajustado para grnlimit=1/exlimit=1 explicitamente; argumentos antigos maiores
continuam aceitos, com aviso e normalizacao para 1. Nao se introduziu
paralelismo, nova fonte, ampliacao de cotas ou troca para introducoes curtas.
Essa correcao torna o comportamento explicito; nao promete aceleracao.

Corrigida tambem a taxa de tokens: usa o incremento da sessao, nao o total
historico. O log enviado mostrava 23.580 tokens/min e ~33 minutos restantes,
mas foram 31.093 tokens novos em 590,2 s: aproximadamente 3.161/min;
extrapolacao simples de ~243 minutos restantes, altamente incerta.
Nenhum novo download foi executado durante estas correcoes.

Arquivos alterados: scripts/coletar_wikipedia_pt.py,
tests/test_coletar_wikipedia_pt.py e este relatorio. Regressao adicionada para
a taxa incremental, e teste de parametros da API reforcado. A primeira
regressao de tempo tinha uma expectativa incorreta para o relogio simulado;
foi corrigida com instantes explicitos. Rodada direcionada final: 14 aprovados
em 0,99 s, sem chamadas reais de rede.

Suite completa unica apos as correcoes: 187 passed, 21 warnings in 38.15s,
com .venv-cuda/Scripts/python.exe -m pytest -q. TEMP/TMP e basetemp locais;
OMP_NUM_THREADS=1 e MKL_NUM_THREADS=1, sem exclusao de testes. Avisos existentes
de depreciacao de GradScaler. Passe incremental real: zero novos e SHA-256
do manifesto identico. Os artefatos anteriores e os splits foram preservados.
