# Limitações — Fase 1

## Ambientes de execução do projeto (atualizado)

Duas Pythons distintas coexistem neste projeto, por uma restrição descoberta durante a Fase 1:

1. **VM Linux da ponte** (usada pelo Claude para editar arquivos, rodar scripts leves e explorar o projeto): Ubuntu 22.04, Python 3.10.12, 1 núcleo físico / 2 threads, ~3,8 GB de RAM, sem GPU. **Não consegue instalar PyTorch**: `download.pytorch.org` (único índice com wheels CPU-only, sem CUDA) é bloqueado nessa VM a nível de proxy (`403 Forbidden` no CONNECT, bloqueio de domínio inteiro). O índice padrão do PyPI foi testado em `--dry-run` e traria obrigatoriamente toda a pilha CUDA 12.1 (vários GB de bibliotecas de GPU inúteis), o que violaria a regra de não instalar dependências de GPU — por isso a instalação foi interrompida nessa VM.
2. **Python nativo do Windows do usuário** (`E:\Models\Ignis\.venv`, Python 3.13.13): aqui `download.pytorch.org` é acessível normalmente. O usuário instalou com sucesso `torch==2.14.1+cpu` via `pip install torch --index-url https://download.pytorch.org/whl/cpu`, confirmado funcionando (`torch.cuda.is_available() == False`, operações de tensor executando).

**Consequência prática:** a VM Linux da ponte continua sendo usada para escrever e revisar código (conforme decisão #5 do usuário), mas toda execução que depende de PyTorch — suíte de testes (`pytest`), contagem real de parâmetros, overfit controlado, smoke test do modelo completo — precisa ser rodada pelo próprio usuário no ambiente virtual nativo do Windows, já que o Claude não tem acesso de execução a esse shell Windows diretamente (apenas à VM Linux da ponte). O usuário roda os comandos e reporta a saída para validação.

Nota: a pasta `.venv` em `E:\Models\Ignis` agora contém arquivos de duas origens diferentes (layout Linux: `bin/`, `lib/python3.10/`, criado pela ponte; layout Windows: `Scripts/`, `Lib/`, criado pelo usuário) coexistindo no mesmo diretório. Isso não quebra nenhum dos dois ambientes (cada Python usa apenas sua própria subestrutura), mas é um pouco confuso de inspecionar manualmente — não é necessário corrigir agora.

## Capacidade computacional da VM Linux de desenvolvimento

- CPU: 1 núcleo físico / 2 threads (Intel Core Ultra 7 155H exposto via hypervisor Microsoft) — provavelmente uma alocação restrita da ponte, não a capacidade total do notebook do usuário.
- RAM: ~3,8 GB total, ~3,5 GB livres.
- GPU: nenhuma.
- Disco da VM (raiz): ~3,9 GB livres — por isso o ambiente virtual Linux foi criado dentro de `E:\Models\Ignis` (disco do projeto, ~1,6 TB livres) e não na raiz da VM.

O Python nativo do Windows do usuário tem capacidade de hardware real do notebook (não a alocação restrita da VM da ponte), então pode ser mais adequado inclusive para os smoke tests da Fase 1, não apenas para tê-los pré-requisito de PyTorch.

## Geração autoregressiva sem cache de K/V

A implementação atual de geração (`src/ignis/inference/generate.py`) recalcula o forward completo a cada novo token, por simplicidade e correção. Isso é aceitável para smoke tests com poucos tokens, mas não deve ser usado como indicação de desempenho real de inferência — isso será endereçado como otimização em fase futura.

## Não-determinismo de ponto flutuante em CPU multi-thread (Fase 3)

Descoberto rodando de verdade a suíte de testes da Fase 3 no Windows
nativo (não foi algo assumido de antemão): dois treinamentos com a MESMA
seed, nos MESMOS dados, podiam terminar com pesos ligeiramente diferentes
em `embed_tokens.weight` (tensor compartilhado com `lm_head.weight` via
`tie_word_embeddings`). Causa provável: o backward de `nn.Embedding` em
CPU acumula o gradiente de linhas de embedding repetidas no mesmo lote
(tokens repetidos) via soma paralela (`index_add_`); soma em ponto
flutuante não é associativa, então a ordem em que as threads terminam
pode variar entre execuções, mesmo com RNG idêntico — isso não depende de
nenhuma GPU, acontece só com o backend CPU multi-thread do PyTorch.

**Correção aplicada:** `ignis.seeding.set_seed()` agora também chama
`torch.use_deterministic_algorithms(True, warn_only=True)`. Isso força o
PyTorch a usar a implementação determinística do backward de embedding
(entre outras) quando disponível; `warn_only=True` evita quebrar o
treinamento caso algum operador usado no futuro não tenha implementação
determinística em CPU (nesse caso o PyTorch só avisa, não levanta
exceção).

**Status da verificação:** CONFIRMADA rodando de verdade no Windows
nativo (torch 2.14.1+cpu) — corrigiu
`test_determinismo_no_backend_cpu_com_mesma_seed`. Esta correção sozinha,
porém, NÃO foi suficiente para corrigir
`test_equivalencia_treinamento_continuo_versus_interrompido_e_retomado`,
que continuou falhando mesmo depois dela — ver a seção seguinte para a
causa raiz real desse segundo teste, que era um problema diferente e
independente deste.

**Trade-off explícito:** forçar algoritmos determinísticos pode deixar o
treinamento mais lento (a soma determinística tende a ser sequencial, não
paralela) — o custo real deve ser medido com
`scripts/benchmark_training.py` depois da correção, nunca assumido.

Dois bugs de TESTE (não do pipeline de treinamento) também foram
encontrados e corrigidos junto desta investigação:
- `test_training_dataloader.py::test_dataset_vazio_levanta_erro_claro`
  esperava a palavra "vazia" na mensagem de erro, mas a mensagem real diz
  "split vazio" (masculino) — erro de digitação no teste, corrigido.
- `test_training_trainer.py::test_determinismo_no_backend_cpu_com_mesma_seed`
  criava dois modelos (`IgnisModel(config_reduzida)`) sem fixar a seed do
  PyTorch imediatamente antes de cada criação — como `Treinador.__init__`
  só chama `set_seed()` depois de receber o modelo já instanciado, os dois
  modelos comparados podiam começar com pesos iniciais diferentes antes
  mesmo de qualquer passo de treino. Corrigido adicionando
  `torch.manual_seed(123)` explícito antes de cada instanciação.


## Causa raiz real da divergência em resume (`test_equivalencia_...`) — Fase 3

A correção de não-determinismo de embedding (seção acima), embora real e
confirmada, NÃO explicava a divergência observada em
`test_equivalencia_treinamento_continuo_versus_interrompido_e_retomado`.
Diagnóstico com `scripts/diagnostico_resume2.py` (rodado de verdade no
Windows) mostrou que a primeira loss e a primeira norma de gradiente eram
IDÊNTICAS entre o treinamento contínuo e o interrompido, e que a
divergência só aparecia DEPOIS do primeiro `optimizer.step()` — um padrão
incompatível com não-determinismo de ponto flutuante (que normalmente já
afetaria a própria loss ou o gradiente) e compatível com uma learning rate
efetivamente diferente sendo aplicada nos dois caminhos.

**Causa raiz confirmada:** o próprio teste (não o pipeline de
treinamento) criava `TreinadorConfig(max_steps=4)` para a fase
"interrompida" e `TreinadorConfig(max_steps=8)` para as fases contínua e
retomada. `max_steps` é o HORIZONTE usado por `lr_no_passo` (warmup linear
+ decaimento em cosseno até `max_steps`) — logo os dois caminhos usavam
curvas de learning rate diferentes já nos passos 1-4, que deveriam ser
idênticos. Verificação analítica: no passo 3 (após `warmup_steps=2`), com
`max_steps=8` o progresso do cosseno é `1/6 ≈ 0.167` (fator ≈ 0.933),
enquanto com `max_steps=4` o progresso é `1/2 = 0.5` (fator = 0.5) — uma
diferença grande o suficiente para explicar sozinha a magnitude da
divergência observada (~7e-3 nos pesos), sem precisar de nenhuma hipótese
de não-determinismo numérico.

Isto NÃO é um bug no pipeline de treinamento (`lr_no_passo` sempre
calculou corretamente a partir de `config.max_steps`) — é um erro de
modelagem do teste (e, em potencial, de qualquer script real que tentasse
"continuar" um treinamento simplesmente aumentando `max_steps` de uma
`TreinadorConfig` já usada): confundir "ponto de parada operacional desta
execução" com "horizonte total planejado do experimento".

**Correção aplicada:** `Treinador.treinar` ganhou o parâmetro opcional
`parar_no_passo: int | None`, que controla APENAS a condição de parada do
laço de treino (`limite_este_treino`). O cálculo de learning rate
(`_lr_atual` / `lr_no_passo`) continua usando sempre `self.config.max_steps`,
nunca `parar_no_passo` — isso é o que garante que uma execução interrompida
em `parar_no_passo` e depois retomada até `max_steps` produza a MESMA
sequência de learning rates que uma execução contínua de `max_steps`
passos. `ResultadoTreinamento.motivo_parada` agora distingue
`"max_steps_atingido"` (horizonte real atingido) de
`"parada_operacional_solicitada"` (parada via `parar_no_passo`, antes do
horizonte).

O teste `test_equivalencia_...` foi reescrito para usar um único
`max_steps=8` nas três fases (contínua, interrompida, retomada), com a
fase interrompida chamando `treinar(..., parar_no_passo=4)` em vez de uma
`TreinadorConfig` separada com `max_steps=4`. Um novo teste,
`test_scheduler_horizonte_diferente_entre_execucoes_nao_e_equivalencia`,
reproduz deliberadamente o padrão ANTIGO (dois `max_steps` diferentes) e
confirma que os pesos DEVEM divergir nesse caso — documentando essa
distinção como intencional, não como uma regressão.

**Medida defensiva adicional (não comprovadamente necessária):**
`foreach=False` foi fixado explicitamente na criação do `AdamW` em
`Treinador.__init__`, para eliminar qualquer possível ambiguidade de
despacho automático do PyTorch entre a implementação vetorizada
("foreach") e a implementação em laço do otimizador, entre uma instância
recém-criada e uma cujo estado foi restaurado via `load_state_dict`. Isto
NÃO foi identificado como causa de nenhuma falha observada — é uma
precaução extra, registrada aqui para que não seja confundida com uma
correção comprovada.

**Status da verificação:** a causa raiz foi confirmada por análise
estática e cálculo manual de `lr_no_passo` (não foi possível executar
PyTorch na VM Linux da ponte). As mudanças em `trainer.py`,
`checkpoint.py` e os testes precisam ser confirmadas rodando de verdade
`pytest tests/test_training_trainer.py tests/test_training_checkpoint.py -v`
no Windows nativo antes de serem consideradas definitivamente resolvidas.

**Atualização (confirmada rodando de verdade no Windows):** a correção de
`max_steps`/`parar_no_passo` acima está correta, mas a primeira rodada real
no Windows revelou um SEGUNDO bug, independente, que bloqueava toda a
suíte: com `safetensors` de fato instalado (ver seção seguinte), TODO
`save_checkpoint` passou a falhar com
`RuntimeError: Some tensors share memory... {'embed_tokens.weight', 'lm_head.weight'}`.
Isso já existia antes — só não aparecia porque `safetensors` nunca estava
instalado nas rodadas anteriores (os testes eram pulados via
`pytest.importorskip`), mascarando o problema. Corrigido na mesma rodada:
ver seção "Exportação Safetensors e pesos compartilhados (tie_word_embeddings)"
abaixo.


## Exportação Safetensors e pesos compartilhados (tie_word_embeddings)

**Causa:** o Ignis usa `tie_word_embeddings=True` por padrão —
`embed_tokens.weight` e `lm_head.weight` compartilham o mesmo tensor na
memória. `safetensors.torch.save_file` recusa salvar um `state_dict` com
tensores que compartilham memória (seria preciso duplicar o tensor no
arquivo, o que o formato não permite por padrão, para evitar
dessincronização ao recarregar).

**Correção:** `salvar_pesos_safetensors` e `carregar_pesos_safetensors`
(`src/ignis/training/checkpoint.py`) passaram a usar
`safetensors.torch.save_model`/`load_model` em vez de `save_file`/`load_file`
com um dicionário de tensores montado manualmente. Essas duas funções
existem na própria biblioteca Safetensors especificamente para este caso:
`save_model` detecta e remove tensores duplicados (por compartilharem
armazenamento) antes de salvar, e `load_model` reconstrói corretamente o
tensor compartilhado ao carregar, mesmo em modo `strict=True` — verificado
lendo o código-fonte da biblioteca (`safetensors==0.8.0`), não apenas
supondo o comportamento pela documentação.

**Status:** corrigido e com fundamentação verificada no código-fonte da
biblioteca; ainda precisa de confirmação executando de verdade
`pytest tests/test_training_checkpoint.py -v` no Windows (os 18 testes que
falharam por este motivo chamavam, direta ou indiretamente,
`save_checkpoint`/`salvar_pesos_safetensors`).
