# preparar_venv_cuda.ps1
#
# Script de INSTALACAO de um ambiente virtual CUDA SEPARADO para o
# Ignis-30M (Sprint 5, secao 1). NUNCA execute este script sem ter lido
# docs/plano-migracao-cuda.md e sem autorizacao explicita separada do
# responsavel pelo projeto -- a politica da skill ignis-llm-engineer proibe
# instalacoes/alteracoes de ambiente sem essa autorizacao.
#
# O QUE ESTE SCRIPT FAZ, SOMENTE QUANDO CHAMADO COM -Confirmar E -IndexUrl:
#   1. Cria um venv NOVO em .venv-cuda (NUNCA toca em .venv, que continua
#      sendo o ambiente CPU em uso).
#   2. Instala torch nesse venv novo, a partir da -IndexUrl fornecida
#      (NUNCA uma tag assumida por este script -- rode primeiro
#      scripts\verificar_cuda_disponivel.ps1 para descobrir empiricamente
#      quais tags sao instalaveis agora, e escolha uma delas).
#   3. Instala o restante das dependencias do projeto (pip install -e .[dev,training])
#      a partir do indice padrao do PyPI (nao da IndexUrl do CUDA).
#   4. Roda scripts\verificar_gpu_cuda.py com o Python do .venv-cuda recem-criado,
#      para confirmar mecanicamente que torch.cuda.is_available() e um
#      forward/backward real na GPU funcionam -- e reporta o resultado.
#
# SEM -Confirmar (uso padrao / seguro para explorar): o script apenas
# imprime o que FARIA e termina sem alterar nada no disco.
#
# Uso (descoberta, nao instala nada):
#   .\scripts\preparar_venv_cuda.ps1
#
# Uso (instala de fato, REQUER autorizacao explicita previamente obtida):
#   .\scripts\preparar_venv_cuda.ps1 -Confirmar -IndexUrl https://download.pytorch.org/whl/cu124
#
# NOTA DE CODIFICACAO: apenas caracteres ASCII (sem acentos, sem travessao),
# mesmo motivo documentado em validar_fase2.ps1/diagnostico_hardware.ps1.

param(
    [switch]$Confirmar,
    [string]$IndexUrl = $null,
    [string]$NomeVenv = ".venv-cuda"
)

$ErrorActionPreference = "Stop"

function Escrever-Secao {
    param([string]$titulo)
    Write-Host ""
    Write-Host "==== $titulo ====" -ForegroundColor Cyan
}

$raizProjeto = Split-Path -Parent $PSScriptRoot
Set-Location $raizProjeto

$caminhoVenvCuda = Join-Path $raizProjeto $NomeVenv
$caminhoVenvCpuAtual = Join-Path $raizProjeto ".venv"

Escrever-Secao "Plano desta execucao"
Write-Host ("  Projeto: " + $raizProjeto)
Write-Host ("  Venv CPU atual (NUNCA alterado por este script): " + $caminhoVenvCpuAtual)
Write-Host ("  Novo venv CUDA a ser criado: " + $caminhoVenvCuda)
Write-Host ("  Index URL de CUDA a usar: " + $(if ($IndexUrl) { $IndexUrl } else { "(nao fornecida)" }))
Write-Host ("  Modo: " + $(if ($Confirmar) { "EXECUCAO REAL (vai instalar)" } else { "SOMENTE PLANO (nada sera instalado)" }))

if (-not $Confirmar) {
    Write-Host ""
    Write-Host "Nenhuma instalacao foi realizada (faltou -Confirmar)." -ForegroundColor Yellow
    Write-Host "Para executar de verdade, apos autorizacao explicita e apos rodar scripts\verificar_cuda_disponivel.ps1 para escolher uma -IndexUrl real:" -ForegroundColor Yellow
    Write-Host "  .\scripts\preparar_venv_cuda.ps1 -Confirmar -IndexUrl https://download.pytorch.org/whl/<tag escolhida>" -ForegroundColor Yellow
    exit 0
}

if (-not $IndexUrl) {
    Write-Host ""
    Write-Host "ERRO: -Confirmar foi passado mas -IndexUrl nao foi fornecida." -ForegroundColor Red
    Write-Host "Este script nunca adivinha uma tag de CUDA -- rode scripts\verificar_cuda_disponivel.ps1 primeiro" -ForegroundColor Red
    Write-Host "para descobrir empiricamente quais tags estao disponiveis, e passe uma delas explicitamente." -ForegroundColor Red
    exit 2
}

if (Test-Path $caminhoVenvCuda) {
    Write-Host ""
    Write-Host "ERRO: $caminhoVenvCuda ja existe. Este script nao sobrescreve um venv existente -- remova-o manualmente primeiro se quiser recriar do zero, ou escolha outro -NomeVenv." -ForegroundColor Red
    exit 2
}

Escrever-Secao "1. Criando venv CUDA separado (NAO mexe em .venv)"
$pythonBase = "python"
try {
    & $pythonBase -m venv $caminhoVenvCuda
    if ($LASTEXITCODE -ne 0) { throw "python -m venv retornou codigo $LASTEXITCODE" }
    Write-Host ("  Criado: " + $caminhoVenvCuda) -ForegroundColor Green
} catch {
    Write-Host ("  ERRO ao criar o venv: " + $_.Exception.Message) -ForegroundColor Red
    exit 1
}

$pythonVenvCuda = Join-Path $caminhoVenvCuda "Scripts\python.exe"
if (-not (Test-Path $pythonVenvCuda)) {
    Write-Host ("  ERRO: python.exe nao encontrado em " + $pythonVenvCuda + " apos a criacao do venv.") -ForegroundColor Red
    exit 1
}

Escrever-Secao "2. Instalando torch a partir de $IndexUrl (isolado em $NomeVenv)"
try {
    & $pythonVenvCuda -m pip install --upgrade pip
    & $pythonVenvCuda -m pip install torch --index-url $IndexUrl
    if ($LASTEXITCODE -ne 0) { throw "pip install torch retornou codigo $LASTEXITCODE" }
} catch {
    Write-Host ("  ERRO ao instalar torch: " + $_.Exception.Message) -ForegroundColor Red
    Write-Host ("  O venv " + $caminhoVenvCuda + " foi criado mas a instalacao de torch falhou; .venv (CPU) nao foi afetado.") -ForegroundColor Yellow
    exit 1
}

Escrever-Secao "3. Instalando as demais dependencias do projeto (indice padrao do PyPI)"
try {
    & $pythonVenvCuda -m pip install -e ".[dev,training]"
    if ($LASTEXITCODE -ne 0) { throw "pip install -e .[dev,training] retornou codigo $LASTEXITCODE" }
} catch {
    Write-Host ("  ERRO ao instalar dependencias do projeto: " + $_.Exception.Message) -ForegroundColor Red
    exit 1
}

Escrever-Secao "4. Verificando a GPU via torch (scripts\verificar_gpu_cuda.py)"
$caminhoVerificacao = Join-Path $raizProjeto "scripts\verificar_gpu_cuda.py"
if (Test-Path $caminhoVerificacao) {
    & $pythonVenvCuda $caminhoVerificacao
    $codigoVerificacao = $LASTEXITCODE
} else {
    Write-Host "  AVISO: scripts\verificar_gpu_cuda.py nao encontrado; pulando verificacao automatica." -ForegroundColor Yellow
    $codigoVerificacao = -1
}

Escrever-Secao "Resumo"
Write-Host ("  Venv CPU original (.venv): inalterado.") -ForegroundColor Green
Write-Host ("  Novo venv CUDA: " + $caminhoVenvCuda)
Write-Host ("  Codigo de saida da verificacao de GPU: " + $codigoVerificacao + " (0 = GPU utilizavel confirmada; ver saida acima para detalhes)")
Write-Host ""
Write-Host "Para usar este ambiente explicitamente: $caminhoVenvCuda\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "Para rodar o treinamento/benchmark com ele: $pythonVenvCuda -m ignis.cli train --config configs\train-pilot.yaml --device cuda" -ForegroundColor Cyan
exit 0
