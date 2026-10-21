# validar_fase4.ps1
#
# Script de validacao do Sprint 4 (corpus real - preparacao, tokenizador
# oficial - procedimento, benchmark com distincao microbatch/passo de
# otimizador, CLI de treino com dry-run obrigatorio) do projeto Ignis,
# complementar a validar_fase2.ps1 e validar_fase3.ps1. Deve ser executado
# no PowerShell (Windows PowerShell 5.1 ou PowerShell 7+), a partir de
# qualquer diretorio, no ambiente Windows nativo com torch instalado.
#
# Uso:
#   .\scripts\validar_fase4.ps1
#
# Este script NAO inicia nenhum pre-treinamento real, NAO baixa nenhum
# corpus real, NAO altera a arquitetura/contagem de parametros do
# Ignis-30M, NAO instala/reinstala CUDA ou GPU drivers e NAO faz commits.
# Ele apenas:
#   1. Verifica o Python do ambiente virtual do projeto e instala
#      dependencias (incluindo o extra opcional "training").
#   2. Roda a suite COMPLETA de testes automatizados (pytest -v), que agora
#      inclui tests/test_cli_train.py (novo no Sprint 4).
#   3. Roda o diagnostico de hardware (somente leitura, nunca instala nada).
#   4. Roda o benchmark de treinamento com --grad-accum-steps > 1, em modo
#      SMOKE (poucas repeticoes), para confirmar que a distincao
#      microbatch/passo de otimizador funciona mecanicamente.
#   5. Gera um corpus sintetico, roda validate-corpus/prepare-splits/report
#      e confirma que o relatorio inclui distribuicao_por_fonte.
#   6. Roda "ignis.cli train" (dry-run, SEM --iniciar-treinamento-real)
#      contra configs/train-pilot.yaml - espera-se codigo de saida 1
#      (dados reais ainda nao existem), NUNCA codigo 2 (erro de config) e
#      NUNCA inicio de treinamento de verdade.
#   7. Imprime um relatorio final dizendo claramente o que passou, falhou ou
#      nao foi executado.
#
# NOTA DE CODIFICACAO: assim como validar_fase2.ps1/validar_fase3.ps1, este
# arquivo e deliberadamente escrito apenas com caracteres ASCII.

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

Escrever-Secao "2. Suite completa de testes automatizados (inclui tests/test_cli_train.py)"
& $pythonExe -m pytest -v
Verificar-CodigoSaida "suite completa de testes (pytest)"

Escrever-Secao "3. Diagnostico de hardware (somente leitura)"
powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $raizProjeto "scripts\diagnostico_hardware.ps1")
Verificar-CodigoSaida "diagnostico de hardware (scripts\diagnostico_hardware.ps1)"

Escrever-Secao "4. Benchmark com distincao microbatch / passo de otimizador (execucao SMOKE)"
$dirBenchmark = Join-Path $env:TEMP ("ignis_benchmark4_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $dirBenchmark | Out-Null
$relatorioBenchmark = Join-Path $dirBenchmark "benchmark_smoke_grad_accum.json"
& $pythonExe scripts\benchmark_training.py --seq-lens 32 --batch-size 1 --grad-accum-steps 4 --repeticoes 2 --aquecimento 1 --saida $relatorioBenchmark
Verificar-CodigoSaida "benchmark com grad-accum-steps (execucao smoke)"
Write-Host "Relatorio de benchmark (smoke, grad-accum) salvo em: $relatorioBenchmark"

Escrever-Secao "5. Corpus sintetico -> validate-corpus -> prepare-splits -> report (distribuicao por fonte)"
$dirCorpus = Join-Path $env:TEMP ("ignis_corpus4_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $dirCorpus | Out-Null
& $pythonExe scripts\gerar_corpus_sintetico.py --saida (Join-Path $dirCorpus "corpus")
Verificar-CodigoSaida "geracao do corpus sintetico"
& $pythonExe -m ignis.cli validate-corpus --corpus-dir (Join-Path $dirCorpus "corpus") --manifesto-saida (Join-Path $dirCorpus "manifesto.jsonl")
Verificar-CodigoSaida "validate-corpus (corpus sintetico)"
& $pythonExe -m ignis.cli prepare-splits --manifesto (Join-Path $dirCorpus "manifesto.jsonl") --splits-saida (Join-Path $dirCorpus "splits.json")
Verificar-CodigoSaida "prepare-splits (corpus sintetico)"
$relatorioJson = Join-Path $dirCorpus "relatorio.json"
& $pythonExe -m ignis.cli report --manifesto (Join-Path $dirCorpus "manifesto.jsonl") --splits (Join-Path $dirCorpus "splits.json") --saida $relatorioJson
Verificar-CodigoSaida "report (corpus sintetico)"
$conteudoRelatorio = Get-Content $relatorioJson -Raw
if ($conteudoRelatorio -notmatch "distribuicao_por_fonte") {
    Write-Host "FALHA: relatorio nao contem o campo distribuicao_por_fonte esperado" -ForegroundColor Red
    $falhas.Add("relatorio com distribuicao_por_fonte")
} else {
    $sucessos.Add("relatorio inclui distribuicao_por_fonte")
}

Escrever-Secao "6. Dry-run do comando 'train' contra configs/train-pilot.yaml (SEM dados reais ainda)"
$relatorioTrain = Join-Path $dirCorpus "dry_run_train.json"
& $pythonExe -m ignis.cli train --config configs\train-pilot.yaml --saida $relatorioTrain
Verificar-CodigoSaidaEntre "dry-run de 'ignis.cli train' (esperado NAO pronto, pois data/ ainda nao existe)" @(0, 1)
Write-Host "Relatorio de dry-run salvo em: $relatorioTrain"
Write-Host "Lembrete: isto e apenas o dry-run. Nenhum treinamento real foi iniciado por este script." -ForegroundColor Yellow

Escrever-Secao "Relatorio final da validacao (Sprint 4)"
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
Write-Host "Validacao do Sprint 4 concluida com sucesso." -ForegroundColor Green
Write-Host "Nenhum pre-treinamento real foi executado. Nenhum corpus real foi baixado." -ForegroundColor Green
Write-Host "Nenhuma arquitetura foi alterada. Nenhuma instalacao de GPU/CUDA foi realizada." -ForegroundColor Green
Write-Host "O inicio de um treinamento real continua exigindo autorizacao explicita separada (ver docs\plano-execucao-piloto.md)." -ForegroundColor Yellow
exit 0
