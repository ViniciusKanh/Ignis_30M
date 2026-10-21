# Plano de migração para CUDA — Ignis-30M (Sprint 5)

**Status: CUDA instalado e verificado por execução real em 2026-10-18,
em `.venv-cuda`, com autorização explícita do responsável pelo projeto.
O `.venv` CPU original permanece intacto (torch 2.14.1+cpu, inalterado)
— a migração foi aditiva, nunca uma substituição.** Ver seção 7 para o
resultado completo da instalação e verificação de GPU.

Este documento descreve o procedimento seguido para habilitar aceleração
de GPU (NVIDIA CUDA) para o treinamento do Ignis-30M, sem substituir ou
arriscar o ambiente CPU já funcional.

## 1. Hardware confirmado (execução real, Sprint 4/5)

Dados abaixo vêm de `diagnostico_hardware_resultado.json` (gerado em
2026-10-18 pela execução real de `scripts\diagnostico_hardware.ps1` no
Windows nativo do usuário — não estimados):

- CPU: Intel(R) Core(TM) Ultra 7 155H (16 núcleos físicos, 22 processadores
  lógicos).
- RAM: 31,51 GB total.
- GPU NVIDIA: GeForce RTX 4050 Laptop GPU — detectada pelo Windows e por
  `nvidia-smi`. VRAM relatada: 4 GB pelo driver do Windows
  (`Win32_VideoController`) e 6141 MiB (~6 GB) por `nvidia-smi` — as duas
  consultas usam métodos diferentes e divergem; tratar 6 GB como teto
  otimista e validar o valor real via `verificar_gpu_cuda.py`
  (`torch.cuda.get_device_properties`) antes de dimensionar batch/seq_len
  para um treinamento real em GPU.
- Driver NVIDIA: versão `610.91` (saída de `nvidia-smi`).
- Compute capability: `8.9` (arquitetura Ada Lovelace).
- GPU secundária: Intel(R) Arc(TM) Graphics (2 GB) — integrada, não
  relevante para CUDA (NVIDIA-only).
- torch atual no `.venv`: `2.14.1+cpu` — `torch.cuda.is_available()` é
  `False` porque esta é uma build CPU-only, não por ausência de GPU.

## 2. Resultado real da descoberta de tags CUDA (executado em 2026-10-18)

`scripts\verificar_cuda_disponivel.ps1` foi executado na máquina do
usuário em 2026-10-18 (`verificar_cuda_disponivel_resultado.json`,
consulta de rede real via `pip index versions`, nenhuma instalação
realizada). Resultado:

| Tag | Consulta | Versões de torch disponíveis |
|---|---|---|
| `cu118` | OK | 2.7.1, 2.7.0, 2.6.0 |
| `cu121` | falhou (`No matching distribution found for torch`) — tag não existe mais neste índice | — |
| `cu124` | OK | 2.6.0 |
| `cu126` | OK | 2.14.1, 2.14.0, 2.13.0, 2.12.1, 2.12.0, 2.11.0, 2.10.0, 2.9.1, 2.9.0, 2.8.0, 2.7.1, 2.7.0, 2.6.0 |
| `cu128` | OK | 2.11.0, 2.10.0, 2.9.1, 2.9.0, 2.8.0, 2.7.1, 2.7.0 |

**Tag recomendada: `cu126`.** É a única tag confirmada que publica
exatamente a versão já instalada no `.venv` CPU atual (`torch==2.14.1`),
então migrar para `cu126` troca apenas o *build* (CPU → CUDA) sem mudar a
versão do torch em nenhum outro ponto da stack do projeto — reduz o risco
de qualquer diferença de comportamento não relacionada à GPU. `cu128`
ficaria preso a uma versão mais antiga (2.11.0 é a mais recente publicada
ali) e `cu118`/`cu124` ainda mais antigas; nenhuma delas tem motivo técnico
para ser preferida a `cu126` neste caso. Compute capability 8.9 (Ada
Lovelace) é compatível com qualquer uma das quatro tags confirmadas (todas
>= CUDA 11.8).

Antes desta execução real, esta seção listava apenas o procedimento de
descoberta a ser seguido:

- Consulta ao vivo (`pip index versions torch --index-url
  https://download.pytorch.org/whl/<tag>`) quais tags de build CUDA do
  torch (`cu118`, `cu121`, `cu124`, `cu126`, `cu128` — lista candidata, que
  pode estar desatualizada) estão **de fato instaláveis agora**, em vez de
  assumir uma versão "atual" a partir de conhecimento potencialmente
  desatualizado deste assistente.
- **Nota de limitação desta sessão:** a ferramenta de busca na web esteve
  indisponível para esta organização durante a Sprint 5 (`PROXY_REJECTED`,
  HTTP 403 — ver `docs/plano-corpus-piloto.md`, seção 8.1, para o mesmo
  problema aplicado ao corpus). Por isso, nenhuma versão mínima "atual" de
  CUDA Toolkit foi verificada por pesquisa ao vivo nesta rodada. O único
  fato tratado aqui como estável (arquitetural, não dependente de
  pesquisa ao vivo) é que compute capability 8.9 (Ada Lovelace) requer
  CUDA Toolkit >= 11.8 — qualquer tag `cu118` ou mais recente confirmada
  como instalável por `verificar_cuda_disponivel.ps1` é, portanto,
  arquiteturalmente compatível com esta GPU; a escolha entre as tags
  confirmadas (ex.: preferir uma mais recente) deve considerar também a
  versão do driver 610.91 no momento da execução real, idealmente
  conferindo a tabela de compatibilidade driver/CUDA Toolkit oficial da
  NVIDIA àquele momento.

## 3. Os três scripts e como se encaixam

1. **`scripts\verificar_cuda_disponivel.ps1`** (descoberta, somente
   leitura, sem autorização necessária): coleta GPU/driver/compute
   capability via `nvidia-smi`, testa empiricamente cada tag candidata via
   `pip index versions` (sem instalar nada) e recomenda, em texto, qual(is)
   tag(s) estão confirmadas como instaláveis agora. Salva
   `verificar_cuda_disponivel_resultado.json`. **Ainda não executado nesta
   máquina — é o próximo passo, antes de qualquer autorização.**
2. **`scripts\preparar_venv_cuda.ps1`** (instalação, GATED — requer
   `-Confirmar` E `-IndexUrl` explícitos): sem `-Confirmar`, apenas
   imprime o plano e sai sem alterar nada. Com ambos, cria um venv NOVO e
   SEPARADO (`.venv-cuda`, nome configurável via `-NomeVenv`), nunca toca
   em `.venv`; instala torch a partir da `-IndexUrl` fornecida (nunca uma
   tag adivinhada pelo script) e as demais dependências do projeto a
   partir do PyPI padrão; ao final, roda automaticamente
   `verificar_gpu_cuda.py` com o Python do novo venv. **Não deve ser
   executado sem autorização explícita e sem antes rodar o script de
   descoberta.**
3. **`scripts\verificar_gpu_cuda.py`** (verificação funcional, somente
   leitura): confirma mecanicamente `torch.cuda.is_available()`, aloca um
   tensor na GPU, roda um forward/backward real (usando `IgnisModel` com
   uma config mínima, ou um fallback `nn.Linear` se o pacote `ignis` não
   estiver instalado no venv) e mede uma comparação simples de
   desempenho CPU-vs-GPU (multiplicação de matrizes — não é um benchmark
   de treinamento real, apenas um sinal rápido de que a GPU está sendo
   usada de fato). Testado nesta sessão na VM Linux (sem torch): saída de
   código 3 ("torch não está instalado"), como esperado — **nunca
   executado com torch+CUDA real**.

## 4. Fluxo de autorização em dois passos (nunca um só)

1. ~~Rodar `scripts\verificar_cuda_disponivel.ps1`~~ — **feito** em
   2026-10-18 (ver seção 2): `cu126` é a tag recomendada.
2. Pedir e obter autorização explícita do responsável pelo projeto,
   citando a tag exata escolhida (`cu126`, recomendada na seção 2) — a
   skill `ignis-llm-engineer` proíbe qualquer instalação/alteração de
   ambiente sem essa autorização específica, e este projeto não assume
   consentimento geral do Sprint 5 como suficiente.
3. Só então rodar:
   ```powershell
   .\scripts\preparar_venv_cuda.ps1 -Confirmar -IndexUrl https://download.pytorch.org/whl/cu126
   ```
   **Atenção:** `-IndexUrl` precisa de uma URL real, nunca o placeholder
   literal `<tag>` — PowerShell interpreta `<` como operador de
   redirecionamento reservado e recusa o comando com erro de parse antes
   mesmo de tentar executá-lo.
4. Revisar a saída de `verificar_gpu_cuda.py` (rodada automaticamente pelo
   passo anterior) antes de usar `--device cuda` em qualquer treinamento.

## 5. Uso após a migração (se autorizada e bem-sucedida)

```powershell
.venv-cuda\Scripts\python.exe -m ignis.cli train --config configs\train-pilot.yaml --device cuda
```

O `.venv` CPU original permanece intacto e utilizável a qualquer momento
(`--device cpu`, o padrão caso `--device` seja omitido) — a migração é
aditiva, nunca uma substituição.

## 6. Resultado real da instalação e verificação de GPU (executado em 2026-10-18)

Após autorização explícita do responsável pelo projeto, `scripts\preparar_venv_cuda.ps1
-Confirmar -IndexUrl https://download.pytorch.org/whl/cu126` foi executado
na máquina do usuário. Resultado real (não estimado):

- `.venv-cuda` criado com sucesso, separado do `.venv` CPU (que permanece
  inalterado).
- torch instalado: `2.14.1+cu126` (build compilada com CUDA 12.6) — mesma
  versão de torch já usada no `.venv` CPU, como planejado na seção 2.
- `torch.cuda.is_available()` = `True`.
- GPU reconhecida pelo torch: NVIDIA GeForce RTX 4050 Laptop GPU, compute
  capability 8.9.
- Alocação de tensor na GPU: OK.
- Forward/backward real do `IgnisModel` (config mínima) na GPU: OK, loss
  finito.
- Comparação de desempenho CPU vs. GPU (multiplicação de matrizes
  2048x2048, 20 repetições): CPU 0,6415 s, GPU 0,0567 s — aceleração de
  11,3x. **Isto não é um benchmark de treinamento real** (nenhuma camada
  de atenção, AdamW, acumulação de gradiente ou medição de VRAM envolvida)
  — apenas um sinal de que a GPU está sendo usada de fato. Um benchmark
  real do Ignis-30M completo (`scripts/benchmark_training.py --device
  cuda`) ainda não foi executado.

Nenhuma arquitetura foi alterada. Nenhum corpus real foi baixado. Nenhum
pré-treinamento real foi iniciado. O `.venv` CPU segue disponível e
funcional a qualquer momento.

## 7. O que este documento (e os scripts que descreve) explicitamente NÃO fizeram

- Não alterou o `.venv` CPU original em nenhum momento — a instalação em
  `.venv-cuda` (seção 6) foi inteiramente aditiva.
- Não afirmou uma versão "atual" de CUDA Toolkit/driver a partir de
  pesquisa ao vivo (indisponível nesta sessão — ver seção 2); a tag usada
  (`cu126`) veio de consulta empírica real, não de suposição.
- Não instalou nada sem autorização explícita — a instalação da seção 6
  só ocorreu depois de autorização específica do responsável pelo
  projeto, citando a tag exata.
- Não iniciou nenhum benchmark de treinamento real nem pré-treinamento —
  o teste de GPU (seção 6) é apenas verificação funcional (alocação,
  forward/backward mínimo, matmul), não um benchmark do Ignis-30M
  completo.
