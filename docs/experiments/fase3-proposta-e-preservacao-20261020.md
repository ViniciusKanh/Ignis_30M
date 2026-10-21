# Preservação do treinamento e proposta da fase 3A

## Estado verificado em 20/10/2026

Correções aplicadas no checkout local existente. Suíte completa executada
pelo responsável no Windows: **195 passed, 24 warnings in 37.02s**, conforme
registro de terminal enviado em Texto colado(20261020-203628).txt. Os avisos
são da API depreciada torch.cuda.amp.GradScaler; não houve falhas de teste.
A suíte inclui os testes novos do detector editorial e da preservação.
Os 45 testes direcionados anteriores também passaram (24,50 s).

O treinador agora salva os passos finais posteriores a um checkpoint
periódico, em stepN_final.pt. Novos checkpoints registram cursor de
microbatches, paciência e estado do GradScaler. Também é registrado RNG CUDA
quando já inicializado. A CLI registra o hash dos splits em futuras execuções.
As alterações não reescrevem nem completam metadados de checkpoints antigos.

A regressão de retomada no meio da época, em CPU com dropout e acúmulo de
gradientes, verifica igualdade dos pesos, estado do AdamW e RNG. A retomada
exige loader determinístico reconstruído por seed/época, num_workers=0 e
ausência de transformações aleatórias. Não se garante equivalência em CUDA,
em checkpoints legados ou de emergência. Gradientes parciais não são salvos.
A cauda incompleta do acúmulo continua descartada por compatibilidade, agora
registrada explicitamente nas métricas.

## Auditoria editorial medida

Execução local de scripts/auditar_editorial.py concluída; hashes dos textos
normalizados conferidos contra o manifesto e contagens de tokens conferidas
contra as estatísticas do piloto. Somente treino e validação foram analisados.
Relatório bruto: auditoria_editorial_20261020_172554_434118/relatorio.json.

| Medida | Treino | Validação |
|---|---:|---:|
| Documentos | 1.063 | 71 |
| Tokens com especiais | 1.518.907 | 84.929 |
| Documentos com título editorial isolado | 1.026 (96,52%) | 66 (92,96%) |
| Ocorrências de Referências | 937 | 57 |
| Ocorrências de Ligações externas | 542 | 35 |
| Documentos com repetição do mesmo título | 10 | 0 |
| Tokens candidatos desde o primeiro título | 84.364 (5,55%) | 4.589 (5,40%) |

O detector procura Referências, Ligações externas, Bibliografia, Notas e Ver
também como títulos em linhas isoladas. O trecho após o primeiro título é
candidato a revisão, não ruído comprovado: pode conter informação científica
legítima. Sua tokenização separada também pode mudar a fronteira BPE.
Nenhum conteúdo foi removido. Não se demonstra que títulos editoriais sejam
a única causa da qualidade semântica insuficiente do modelo.

## Proposta para revisão: fase 3A de controle

Hipótese: mais otimização pode melhorar predição e geração no corpus atual.
O experimento mantém os dados para isolar essa variável antes de testar uma
variante editorial. Não garante coerência ou factualidade.

| Item | Proposta |
|---|---|
| Inicialização | Pesos de checkpoints/pilot-corpus-v1-fase2/step1000.pt |
| SHA-256 inicial | 60dbb12aa8727cb1cafbd48cb5c4afdc44886b8f7b7438325acb6ffbccbe566f |
| Arquitetura e tokenizador | Preservados |
| Dados | Mesmo manifesto, splits e dataset tokenizado do piloto |
| Contexto | Blocos 128; entrada efetiva 127 |
| Estado de otimização | Novo AdamW e novo scheduler; contador local zero |
| Orçamento máximo | 1.000 passos adicionais |
| Microbatch e acúmulo | 1 e 32 |
| Teto teórico | 4.064.000 posições-alvo, antes de padding e caudas |
| LR | 5e-5; mínimo 5e-6; warmup 50; cosseno até o passo 1.000 |
| AdamW | beta1=0,9; beta2=0,95; eps=1e-8; weight_decay=0,1 |
| Clipping | 1,0 |
| Seed e precisão | 42; mixed_precision=no |
| Device | CUDA, na RTX 4050 local |
| Saída nova | checkpoints/pilot-corpus-v1-fase3a-controle |
| Validação | Completa antes do treino e a cada 100 passos |
| Checkpoint e logs | A cada 100 e 10 passos, respectivamente |
| Paciência | Três avaliações consecutivas sem melhora |

A avaliação inicial deverá inicializar melhor_loss_validacao; referência
atual 6,132591157108642. Preservar o melhor checkpoint, inclusive os pesos
iniciais se nenhuma avaliação posterior melhorar. Repetir os cinco prompts
com os mesmos parâmetros e seed 42; seeds 7 e 123 podem complementar a
avaliação de amostragem, separadamente da comparação histórica.

Critérios de parada: teto de passos, paciência, NaN/Inf ou falta de memória.
Critérios de avaliação: loss na validação completa, relevância temática,
coerência e repetição nas amostras. Loss menor isoladamente não aprova o
modelo para entrega. O teste final permanece reservado.

Esta proposta não é uma autorização de treinamento. O executor está em
scripts/treinar_fase3.py, com configuração separada em
configs/train-pilot-corpus-v1-fase3a.yaml. Não continuar pela CLI train
diretamente: ela inicializa outro modelo.

Uma variante editorial deve ser separada e versionada após revisão contextual
dos trechos candidatos, com avaliação comum e tokenizador compatível. Não
reajustar BPE nem cortar indiscriminadamente tudo após Referências.

## Comandos de leitura e avaliação disponíveis

```powershell
cd E:\Models\Ignis
.\.venv-cuda\Scripts\python.exe -u scripts\auditar_editorial.py
.\.venv-cuda\Scripts\python.exe -u scripts\avaliar_checkpoint.py --device cuda
```

As execuções reais desses comandos já foram registradas. Não há necessidade
de repeti-las para confirmar os resultados desta etapa. Nenhum treinamento
do modelo completo, download, instalação, commit ou push foi feito nesta etapa.

## Executor e preparação medidos

O executor recusa hashes diferentes dos artefatos avaliados, mudanças no
orçamento/hiperparâmetros da proposta e pasta de saída preexistente. O modo
padrão só faz leitura dos pesos e validação completa; não constrói Treinador
nem cria a pasta de checkpoints. O teste final não é usado.

Dry-run real aprovado na RTX 4050, torch 2.14.1+cu126: checkpoint da fase 2
carregado, 30.416.384 parâmetros, loss inicial 6,132591157108642,
perplexidade 460,6281756357188, 664 lotes e 84.265 posições-alvo.
Relatório: fase3a_dryrun_20261020_174222_007031/relatorio.json.
A pasta de saída de treinamento continuou ausente após o dry-run.

Na execução real, o executor constrói novo AdamW/scheduler, inicializa a
melhor loss pela validação inicial e salva best.pt no passo zero. O treinador
ganhou a opção salvar_melhor, desativada por padrão; nesta fase ela fica
ativa e substitui best.pt somente quando a validação melhora. Essa opção
não altera a política dos experimentos antigos.

Testes direcionados do executor e treinador: 32 passed, 25 warnings in 22.77s.
Incluem separação dry-run/treinador, limites da proposta e preservação do
melhor checkpoint quando a validação piora depois.
Suíte completa após esta implementação: **198 passed, 25 warnings in 94.04s**.
TEMP/TMP e basetemp em pasta nova do checkout, OMP_NUM_THREADS=1 e
MKL_NUM_THREADS=1; nenhum teste excluído. Avisos de GradScaler depreciado.

Preparação sem atualizar pesos:

```powershell
.\.venv-cuda\Scripts\python.exe -u scripts\treinar_fase3.py
```

Início real, exclusivamente após aprovação do orçamento pelo responsável:

```powershell
.\.venv-cuda\Scripts\python.exe -u scripts\treinar_fase3.py --iniciar-treinamento-real
```

Avaliação do melhor checkpoint após o treinamento:

```powershell
.\.venv-cuda\Scripts\python.exe -u scripts\avaliar_checkpoint.py --checkpoint checkpoints\pilot-corpus-v1-fase3a-controle\best.pt --device cuda
```

As linhas de início real e avaliação futura acima não foram executadas nesta
preparação. O dry-run aprovado não garante melhoria na geração de texto.
