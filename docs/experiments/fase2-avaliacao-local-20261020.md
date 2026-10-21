# Avaliação local do checkpoint da fase 2

Avaliação executada em 20/10/2026 no checkout E:\Models\Ignis, com PyTorch
2.14.1+cu126 e NVIDIA GeForce RTX 4050 Laptop GPU. Não houve treinamento,
downloads, alteração de arquitetura, instalação ou atualização de dependências.

## Evidência medida

- Checkpoint: checkpoints/pilot-corpus-v1-fase2/step1000.pt.
- SHA-256: 60dbb12aa8727cb1cafbd48cb5c4afdc44886b8f7b7438325acb6ffbccbe566f.
- Carregamento estrito dos pesos e compatibilidade de arquitetura, tokenizador
  e splits conferidos; contagem de 30.416.384 parâmetros.
- Validação completa: 664 lotes de uma sequência, 84.265 posições-alvo,
  loss 6,132591157108642, perplexidade 460,6281756357188.
- O teste final não foi utilizado. Dez amostras: cinco prompts, greedy e
  amostragem com temperatura 0,8, top-k 40, top-p 0,9, seed 42 e até 64 tokens.
- Relatório bruto: avaliacao_checkpoint_20261020_171319_795909/relatorio.json.

As amostras possuem trechos de frases, porém apresentam repetição, deriva de
assunto e referências desconexas. Exemplo: o prompt sobre energia solar tem
continuação greedy "a cidade de Janeiro"; o prompt sobre algoritmo entra em
repetição de "o que". Há recorrência de "Referências" e "Ligações externas".
Isso sugere aprendizagem de padrões editoriais do corpus; não demonstra
qualidade conversacional, conhecimento factual confiável ou domínio técnico.
Cinco prompts e uma seed de amostragem não constituem um benchmark abrangente.

## Revisão focada do código e dos artefatos

- O audit-tokenizer salvo registra todos os critérios aprovados e
  oficial_do_piloto=true, embora o arquivo mantenha o nome naooficial.
  Trata-se da auditoria histórica salva, não de uma nova auditoria nesta revisão.
- Treino contém 1.518.907 tokens tokenizados. Repetir o corpus não amplia
  a diversidade de documentos. Uma nova fase precisa de hipótese e orçamento.
- O contexto arquitetural é 1.024, mas este piloto usa blocos de 128 e
  entradas efetivas de 127 tokens. Não foi avaliada qualidade em contexto longo.
- A geração recalcula o prefixo inteiro; não há cache KV. Isso afeta custo de
  inferência e não explica, por si só, a baixa qualidade semântica.
- O treinador descarta o acúmulo incompleto no final da época: com 11.867
  sequências, microbatch 1 e acúmulo 32, restam 27 microbatches sem optimizer
  step por época completa. Não se trata de perda silenciosa de checkpoint.
- A retomada não salva a posição do iterador dentro da época nem a paciência
  de early stopping; não deve ser apresentada como retomada exata geral.
- Se o treinamento termina fora de um múltiplo de checkpoint_every e já
  existe checkpoint periódico, o código não salva automaticamente os últimos
  passos (condição ultimo_checkpoint is None). Isso não afetou esta fase,
  encerrada no passo 1.000, múltiplo de 100.
- A validação inicial da fase 2 é salva, mas não inicializa a melhor loss do
  treinador para early stopping. A primeira avaliação periódica vira a
  referência. Não alterou a conclusão deste caso, cuja validação melhorou.

Esses pontos foram inspecionados; não foram corrigidos nesta revisão. Os
resultados anteriores e as alterações preexistentes foram preservados.

## Verificação

Execução real do novo script aprovada, com a validação e as dez amostras
registradas acima. Suíte completa: **187 passed, 21 warnings in 40.23s**.
Os avisos são da API depreciada de GradScaler. A primeira tentativa teve
88 testes aprovados e 99 erros de preparação das fixtures de diretório
temporário; a repetição passou com TEMP/TMP e --basetemp em uma pasta nova
do checkout, sem excluir testes nem alterar dependências. OMP_NUM_THREADS=1
e MKL_NUM_THREADS=1 nas duas tentativas. git diff --check sem erros de
whitespace nos arquivos rastreados, com avisos preexistentes de LF/CRLF.

## Continuidade

Comando reproduzível adicionado:

```powershell
.\.venv-cuda\Scripts\python.exe -u scripts\avaliar_checkpoint.py --device cuda
```

Para comparar com o piloto anterior, preservando parâmetros e validação:

```powershell
.\.venv-cuda\Scripts\python.exe -u scripts\avaliar_checkpoint.py --checkpoint checkpoints\pilot-corpus-v1\step200.pt --device cuda --permitir-splits-sem-hash
```

Cada execução cria outro diretório de relatório. Não atualiza pesos e não
consulta o teste final. Antes de nova fase, priorizar correções de preservação
do estado do treinador e auditoria da incidência de seções editoriais; qualquer
novo corpus/tokenização requer outra versão e rastreabilidade. O tokenizador
compatível com os pesos atuais deve ser preservado.

## Correção da avaliação do checkpoint legado

O primeiro comando de comparação fornecido omitia a opção de legado: o
avaliador comparava None com o hash atual e informava incorretamente uma
incompatibilidade. Inspeção direta de step200.pt confirmou dataset_sha256=null.
A CLI do piloto constrói os metadados sem dataset_caminho; a fase 2 já o
preenche. Os checkpoints não foram regravados e a CLI não foi alterada.

O avaliador agora distingue hash ausente de hash divergente. A opção explícita
--permitir-splits-sem-hash permite apenas avaliação de legado sem hash,
registrando origem_nao_verificada no relatório. Ela nunca libera um hash
divergente. Tokenizador e arquitetura continuam obrigatoriamente compatíveis.
Essa opção não comprova retrospectivamente os splits do piloto.

Execução real na RTX 4050 do checkpoint 200 concluída: loss
6,992820456263814, perplexidade 1088,7880286298496, 84.265 posições-alvo,
664 lotes e dez amostras. Relatório:
avaliacao_checkpoint_20261020_171936_009255/relatorio.json.
O teste final não foi usado e nenhum peso foi atualizado. Os três testes
direcionados da nova política de proveniência passaram em 4,73 s.
Suíte completa após a correção: 190 passed, 21 warnings in 43.03s, com
TEMP/TMP e basetemp novos dentro do checkout e uma thread OMP/MKL.
