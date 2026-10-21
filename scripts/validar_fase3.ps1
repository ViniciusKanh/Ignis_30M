# validar_fase3.ps1
#
# Script de validacao da Fase 3 (infraestrutura de treinamento) do projeto
# Ignis, complementar a validar_fase2.ps1 (que continua valendo para a
# Fase 1/2). Deve ser executado no PowerShell (Windows PowerShell 5.1 ou
# PowerShell 7+), a partir de qualquer diretorio, no ambiente Windows nativo
# com torch instalado.
#
# Uso:
#   .\scripts\validar_fase3.ps1
#
# Este script NAO inicia nenhum pre-treinamento prolongado, NAO baixa
# nenhum corpus real, NAO altera a arquitetura/contagem de parametros do
# Ignis-30M e NAO faz commits. Ele apenas:
#   1. Verifica o Python do ambiente virtual do projeto.
#   2. Verifica/instala dependencias do projeto, incluindo o extra opcional
#      "training" (safetensors) - nunca obrigatorio.
#   3. Roda a suite completa de testes automatizados (pytest -v), que
#      inclui Fase 1, Fase 2 e os novos testes da Fase 3 (dataloader,
#      trainer, checkpoint, configs/train-pilot.yaml).
#   4. Valida que configs/train-pilot.yaml carrega e e compativel com
#      configs/ignis-30m.yaml (sem rodar nenhum treinamento).
#   5. Roda scripts/benchmark_training.py com parametros PEQUENOS (smoke),
#      so para confirmar que o benchmark funciona e gerar um relatorio
#      JSON de exemplo - NAO e o benchmark completo usado para planejar o
#      piloto real.
#   6. Roda scripts/smoke_training.py (modelo reduzido + corpus sintetico,
#      poucos passos) para verificar mecanicamente atualizacao de
#      parametros e retomada de checkpoint do pipeline real da Fase 3.
#   7. Imprime um relatorio final dizendo claramente o que passou, falhou ou
#      nao foi executado.
#
# NOTA DE CODIFICACAO: assim como validar_fase2.ps1, este arquivo e
# deliberadamente escrito apenas com caracteres ASCII (sem acentos, sem
# travessao "-", apenas hifen simples "-"), para evitar o problema de
# decodificacao ja documentado em docs/limitations.md (Windows PowerShell
# 5.1 le .ps1 sem BOM pela code page ANSI do sistema, corrompendo bytes
# UTF-8 nao-ASCII).

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

# Raiz do projeto = diretorio pai da pasta scripts/, onde este arquivo esta.
$raizProjeto = Split-Path -Parent $PSScriptRoot
Set-Location $raizProjeto

Escrever-Secao "1. Verificando ambiente virtual e Python"
$pythonExe = Join-Path $raizProjeto ".venv\Scripts\python.exe"
if (-Not (Test-Path $pythonExe)) {
    Write-Host "Ambiente virtual nao encontrado em .venv - criando..." -ForegroundColor Yellow
    py -m venv .venv
    Verificar-CodigoSaida "criacao do ambiente virtual (.venv)"
}
if (-Not (Test-Path $pythonExe)) {
    Write-Host "FALHA CRITICA: nao foi possivel localizar $pythonExe apos a criacao do venv." -ForegroundColor Red
    exit 1
}
Write-Host "Usando interpretador: $pythonExe"
& $pythonExe --version
Verificar-CodigoSaida "verificacao da versao do Python do venv"

Escrever-Secao "2. Verificando/instalando dependencias do projeto (incluindo extra opcional training)"
& $pythonExe -m pip install --upgrade pip --quiet
Verificar-CodigoSaida "atualizacao do pip"

& $pythonExe -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "torch nao encontrado - instalando build CPU-only (download.pytorch.org)..." -ForegroundColor Yellow
    & $pythonExe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    Verificar-CodigoSaida "instalacao do torch CPU-only"
} else {
    $sucessos.Add("torch ja estava instalado (nao foi reinstalado)")
}

& $pythonExe -m pip install -e ".[dev,training]" --quiet
Verificar-CodigoSaida "instalacao do pacote ignis em modo editavel ([dev,training])"

& $pythonExe -c "import safetensors" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "safetensors nao instalado - exportacao extra de pesos em checkpoint sera pulada (opcional, nao e um erro)." -ForegroundColor Yellow
} else {
    $sucessos.Add("safetensors disponivel (exportacao opcional de pesos de checkpoint ativa)")
}

Escrever-Secao "3. Suite completa de testes automatizados (Fase 1 + Fase 2 + Fase 3)"
& $pythonExe -m pytest -v
Verificar-CodigoSaida "suite completa de testes (pytest)"

Escrever-Secao "4. Validando configs/train-pilot.yaml contra configs/ignis-30m.yaml"
& $pythonExe -c "from ignis.config import PilotoConfig, IgnisConfig; p = PilotoConfig.from_yaml('configs/train-pilot.yaml'); m = IgnisConfig.from_yaml(p.modelo.config).modelo; p.validar_contexto_contra_modelo(m); print('OK: train-pilot.yaml valido e compativel com', p.modelo.config); print('lote efetivo (hipotese):', p.lote.lote_efetivo); print('max_steps (piloto):', p.orcamento.max_steps)"
Verificar-CodigoSaida "validacao de configs/train-pilot.yaml"

Escrever-Secao "5. Benchmark de hardware/treinamento (execucao SMOKE, nao o benchmark completo)"
$dirBenchmark = Join-Path $env:TEMP ("ignis_benchmark_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $dirBenchmark | Out-Null
try {
    $relatorioBenchmark = Join-Path $dirBenchmark "benchmark_smoke.json"
    & $pythonExe scripts\benchmark_training.py --seq-lens 32 --batch-size 1 --repeticoes 2 --aquecimento 1 --saida $relatorioBenchmark
    Verificar-CodigoSaida "benchmark de treinamento (execucao smoke)"
    Write-Host "Relatorio de benchmark (smoke) salvo em: $relatorioBenchmark"
}
finally {
    # Mantido ate o fim do script (nao apagado aqui) para permitir inspecao
    # manual caso algo falhe acima; sera limpo junto do diretorio temporario
    # abaixo se tudo estiver OK. Em caso de erro, o PowerShell ja encerrou
    # via $ErrorActionPreference antes de chegar aqui.
}

Escrever-Secao "6. Smoke training (modelo reduzido + corpus sintetico) - NAO e o piloto real"
& $pythonExe scripts\smoke_training.py --max-steps 8
Verificar-CodigoSaida "smoke training (scripts/smoke_training.py)"

Escrever-Secao "Relatorio final da validacao"
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
Write-Host "Validacao da Fase 3 concluida com sucesso." -ForegroundColor Green
Write-Host "Nenhum pre-treinamento prolongado foi executado." -ForegroundColor Green
Write-Host "Nenhum corpus real foi baixado. Nenhuma arquitetura foi alterada." -ForegroundColor Green
Write-Host "O inicio de um piloto real (configs/train-pilot.yaml) continua exigindo autorizacao explicita separada." -ForegroundColor Yellow
exit 0
