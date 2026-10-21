# validar_fase5.ps1
#
# Script de validacao do Sprint 5 (desbloqueio do treinamento real) do
# projeto Ignis: correcao de granularidade de splits (fonte vs. unidade de
# agrupamento), ingestao incremental/retomavel, gate mecanico de promocao
# do tokenizador (audit-tokenizer), override de device no treino, mixed
# precision e tratamento de falta de memoria no Treinador, e descoberta
# (somente leitura) de disponibilidade de CUDA. Complementar a
# validar_fase2.ps1, validar_fase3.ps1 e validar_fase4.ps1. Deve ser
# executado no PowerShell (Windows PowerShell 5.1 ou PowerShell 7+), a
# partir de qualquer diretorio, no ambiente Windows nativo com torch
# instalado.
#
# Uso:
#   .\scripts\validar_fase5.ps1
#
# Este script NAO inicia nenhum pre-treinamento real, NAO baixa nenhum
# corpus real, NAO altera a arquitetura/contagem de parametros do
# Ignis-30M, NAO instala/reinstala CUDA ou GPU drivers (mesmo que uma GPU
# seja detectada) e NAO faz commits. Ele apenas:
#   1. Verifica o Python do ambiente virtual do projeto e instala
#      dependencias (incluindo o extra opcional "training").
#   2. Roda a suite COMPLETA de testes automatizados (pytest -v), incluindo
#      os testes novos de mixed precision e falta de memoria em
#      tests/test_training_trainer.py (nunca executados antes com torch
#      real).
#   3. Roda o diagnostico de hardware e a descoberta de CUDA, ambos
#      somente leitura, nunca instalam nada.
#   4. Gera um corpus sintetico e exercita validate-corpus (incluindo o
#      modo --incremental), prepare-splits, report (confirmando
#      n_unidades_agrupamento no relatorio por fonte).
#   5. Roda train-tokenizer, prepare-dataset e audit-tokenizer sobre esse
#      corpus sintetico, confirmando que o comando roda e produz um
#      veredito objetivo (espera-se oficial_do_piloto = false, pois o
#      corpus sintetico e minusculo demais para a faixa de volume alvo --
#      isso e o comportamento correto, nao uma falha deste script).
#   6. Roda "ignis.cli train" (dry-run, SEM --iniciar-treinamento-real)
#      com --device cpu contra configs/train-pilot.yaml - espera-se codigo
#      de saida 1 (dados reais ainda nao existem), NUNCA codigo 2 (erro de
#      config) e NUNCA inicio de treinamento de verdade. Se uma GPU CUDA
#      estiver disponivel neste ambiente (torch.cuda.is_available() via o
#      proprio venv), tambem testa --device cuda da mesma forma --
#      puramente informativo, nunca requer ou aciona instalacao de nada.
#   7. Imprime um relatorio final dizendo claramente o que passou, falhou
#      ou nao foi executado.
#
# NOTA DE CODIFICACAO: assim como validar_fase2.ps1/validar_fase3.ps1/
# validar_fase4.ps1, este arquivo e deliberadamente escrito apenas com
# caracteres ASCII.

$ErrorActionPreference = "Stop"

$falhas = New-Object System.Collections.Generic.List[string]
$sucessos = New-Object System.Collections.Generic.List[string]

function Escrever-Secao {
    param([string]$titulo)
    Write-Host ""
    Write-Host "==== $titulo ====" -ForegroundColor Cyan
}

function Verificar-CodigoSaida {
    param([string]$descricao)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FALHA: $descricao (codigo de saida $LASTEXITCODE)" -ForegroundColor Red
        $script:falhas.Add($descricao)
        throw "Etapa falhou: $descricao"
    }
    $script:sucessos.Add($descricao)
}

function Verificar-CodigoSaidaEntre {
    param([string]$descricao, [int[]]$codigosAceitos)
    if ($codigosAceitos -notcontains $LASTEXITCODE) {
        Write-Host "FALHA: $descricao (codigo de saida $LASTEXITCODE, esperado um de: $codigosAceitos)" -ForegroundColor Red
        $script:falhas.Add($descricao)
        throw "Etapa falhou: $descricao"
    }
    $script:sucessos.Add("$descricao (codigo de saida $LASTEXITCODE, dentro do esperado)")
}

$raizProjeto = Split-Path -Parent $PSScriptRoot
Set-Location $raizProjeto

Escrever-Secao "1. Verificando ambiente virtual e dependencias"
$pythonExe = Join-Path $raizProjeto ".venv\Scripts\python.exe"
if (-Not (Test-Path $pythonExe)) {
    Write-Host "Ambiente virtual nao encontrado em .venv - criando..." -ForegroundColor Yellow
    py -m venv .venv
    Verificar-CodigoSaida "criacao do ambiente virtual (.venv)"
}
& $pythonExe --version
Verificar-CodigoSaida "verificacao da versao do Python do venv"

& $pythonExe -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "torch nao encontrado - instalando build CPU-only..." -ForegroundColor Yellow
    & $pythonExe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    Verificar-CodigoSaida "instalacao do torch CPU-only"
} else {
    $sucessos.Add("torch ja estava instalado (nao foi reinstalado)")
}
& $pythonExe -m pip install -e ".[dev,training]" --quiet
Verificar-CodigoSaida "instalacao do pacote ignis em modo editavel ([dev,training])"

Escrever-Secao "2. Suite completa de testes automatizados (inclui mixed precision e falta de memoria)"
& $pythonExe -m pytest -v
Verificar-CodigoSaida "suite completa de testes (pytest)"

Escrever-Secao "3. Diagnostico de hardware e descoberta de CUDA (ambos somente leitura)"
powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $raizProjeto "scripts\diagnostico_hardware.ps1")
Verificar-CodigoSaida "diagnostico de hardware (scripts\diagnostico_hardware.ps1)"
powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $raizProjeto "scripts\verificar_cuda_disponivel.ps1")
Verificar-CodigoSaida "descoberta de CUDA (scripts\verificar_cuda_disponivel.ps1)"
Write-Host "Lembrete: nenhuma instalacao de CUDA foi realizada pelos comandos acima." -ForegroundColor Yellow

Escrever-Secao "4. Corpus sintetico -> validate-corpus (normal e --incremental) -> prepare-splits -> report"
$dirCorpus = Join-Path $env:TEMP ("ignis_corpus5_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $dirCorpus | Out-Null
& $pythonExe scripts\gerar_corpus_sintetico.py --saida (Join-Path $dirCorpus "corpus")
Verificar-CodigoSaida "geracao do corpus sintetico"
$manifestoPath = Join-Path $dirCorpus "manifesto.jsonl"
& $pythonExe -m ignis.cli validate-corpus --corpus-dir (Join-Path $dirCorpus "corpus") --manifesto-saida $manifestoPath
Verificar-CodigoSaida "validate-corpus (execucao normal)"
& $pythonExe -m ignis.cli validate-corpus --corpus-dir (Join-Path $dirCorpus "corpus") --manifesto-saida $manifestoPath --incremental --limite-novos-documentos 1000
Verificar-CodigoSaida "validate-corpus --incremental (segunda execucao, deve detectar que nao ha documentos novos)"
$splitsPath = Join-Path $dirCorpus "splits.json"
& $pythonExe -m ignis.cli prepare-splits --manifesto $manifestoPath --splits-saida $splitsPath
Verificar-CodigoSaida "prepare-splits (corpus sintetico)"
$relatorioJson = Join-Path $dirCorpus "relatorio.json"
& $pythonExe -m ignis.cli report --manifesto $manifestoPath --splits $splitsPath --saida $relatorioJson
Verificar-CodigoSaida "report (corpus sintetico)"
$conteudoRelatorio = Get-Content $relatorioJson -Raw
if ($conteudoRelatorio -notmatch "n_unidades_agrupamento") {
    Write-Host "FALHA: relatorio nao contem o campo n_unidades_agrupamento esperado (Sprint 5)" -ForegroundColor Red
    $falhas.Add("relatorio com n_unidades_agrupamento")
} else {
    $sucessos.Add("relatorio inclui n_unidades_agrupamento por fonte")
}

Escrever-Secao "5. Tokenizador, dataset e auditoria mecanica (audit-tokenizer)"
$tokenizerPath = Join-Path $dirCorpus "tokenizer.json"
& $pythonExe -m ignis.cli train-tokenizer --corpus-dir (Join-Path $dirCorpus "corpus") --manifesto $manifestoPath --splits $splitsPath --saida $tokenizerPath
Verificar-CodigoSaida "train-tokenizer (corpus sintetico)"
$datasetBase = Join-Path $dirCorpus "tokenizado"
& $pythonExe -m ignis.cli prepare-dataset --tokenizer $tokenizerPath --corpus-dir (Join-Path $dirCorpus "corpus") --manifesto $manifestoPath --splits $splitsPath --saida-base $datasetBase --seq-len 32
Verificar-CodigoSaida "prepare-dataset (corpus sintetico)"
$externosPath = Join-Path $dirCorpus "externos.txt"
"Texto de exemplo nao incluido em nenhum split, usado apenas para avaliar o tokenizador." | Set-Content -Path $externosPath -Encoding UTF8
$auditoriaPath = Join-Path $dirCorpus "auditoria_tokenizer.json"
& $pythonExe -m ignis.cli audit-tokenizer --tokenizer $tokenizerPath --corpus-dir (Join-Path $dirCorpus "corpus") --manifesto $manifestoPath --splits $splitsPath --textos-externos $externosPath --dataset-base $datasetBase --saida $auditoriaPath
# codigo 0 = oficial_do_piloto true, codigo 1 = algum criterio falhou (esperado aqui,
# pois o corpus sintetico e muito pequeno para a faixa-alvo de volume de tokens) --
# qualquer outro codigo indica um erro real do comando, nao uma reprovacao esperada.
Verificar-CodigoSaidaEntre "audit-tokenizer (esperado reprovar por volume de tokens, corpus sintetico e minusculo)" @(0, 1)
Write-Host "Relatorio de auditoria do tokenizador salvo em: $auditoriaPath"

Escrever-Secao "6. Dry-run do comando 'train' contra configs/train-pilot.yaml (com --device)"
$relatorioTrainCpu = Join-Path $dirCorpus "dry_run_train_cpu.json"
& $pythonExe -m ignis.cli train --config configs\train-pilot.yaml --device cpu --saida $relatorioTrainCpu
Verificar-CodigoSaidaEntre "dry-run de 'ignis.cli train --device cpu' (esperado NAO pronto, pois data/ ainda nao existe)" @(0, 1)
Write-Host "Relatorio de dry-run (cpu) salvo em: $relatorioTrainCpu"

$cudaDisponivel = & $pythonExe -c "import torch; print(torch.cuda.is_available())" 2>$null
if ($cudaDisponivel -eq "True") {
    Write-Host "GPU CUDA disponivel neste venv - testando tambem --device cuda (apenas dry-run, informativo)." -ForegroundColor Cyan
    $relatorioTrainCuda = Join-Path $dirCorpus "dry_run_train_cuda.json"
    & $pythonExe -m ignis.cli train --config configs\train-pilot.yaml --device cuda --saida $relatorioTrainCuda
    Verificar-CodigoSaidaEntre "dry-run de 'ignis.cli train --device cuda'" @(0, 1)
    Write-Host "Relatorio de dry-run (cuda) salvo em: $relatorioTrainCuda"
} else {
    Write-Host "GPU CUDA nao disponivel neste venv (torch.cuda.is_available() = $cudaDisponivel) - --device cuda nao testado aqui, isso e esperado no .venv CPU atual." -ForegroundColor Yellow
    $sucessos.Add("deteccao correta de ausencia de CUDA neste venv (--device cuda nao testado, como esperado)")
}
Write-Host "Lembrete: isto e apenas o dry-run. Nenhum treinamento real foi iniciado por este script." -ForegroundColor Yellow

Escrever-Secao "Relatorio final da validacao (Sprint 5)"
Write-Host ""
Write-Host "Etapas aprovadas (" $sucessos.Count "):" -ForegroundColor Green
foreach ($item in $sucessos) {
    Write-Host "  [OK] $item" -ForegroundColor Green
}
if ($falhas.Count -gt 0) {
    Write-Host ""
    Write-Host "Etapas com falha (" $falhas.Count "):" -ForegroundColor Red
    foreach ($item in $falhas) {
        Write-Host "  [FALHOU] $item" -ForegroundColor Red
    }
    exit 1
}

Write-Host ""
Write-Host "Validacao do Sprint 5 concluida com sucesso." -ForegroundColor Green
Write-Host "Nenhum pre-treinamento real foi executado. Nenhum corpus real foi baixado." -ForegroundColor Green
Write-Host "Nenhuma arquitetura foi alterada. Nenhuma instalacao de GPU/CUDA foi realizada." -ForegroundColor Green
Write-Host "O inicio de um treinamento real e a instalacao de um ambiente CUDA continuam exigindo" -ForegroundColor Yellow
Write-Host "autorizacao explicita separada (ver docs\checklist-autorizacao-sprint5.md)." -ForegroundColor Yellow
exit 0
