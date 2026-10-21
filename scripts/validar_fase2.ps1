# validar_fase2.ps1
#
# Script unico de validacao da Fase 2 (e revalidacao da Fase 1) do projeto Ignis.
# Deve ser executado no PowerShell (Windows PowerShell 5.1 ou PowerShell 7+),
# a partir de qualquer diretorio, no ambiente Windows nativo com torch instalado.
#
# Uso:
#   .\scripts\validar_fase2.ps1
#
# Este script NAO inicia nenhum treinamento, NAO reinstala PyTorch (so instala
# se estiver ausente), NAO baixa datasets e NAO faz commits. Ele apenas:
#   1. Verifica o Python do ambiente virtual do projeto.
#   2. Verifica/instala dependencias do projeto (torch so se ausente).
#   3. Roda a contagem de parametros do Ignis-30M.
#   4. Roda a suite completa de testes automatizados (pytest).
#   5. Roda o smoke test do modelo completo (NAO e treinamento).
#   6. Gera um corpus sintetico descartavel e roda a CLI de dados/tokenizador
#      de ponta a ponta (ingestao, splits, tokenizador, preparo de dataset).
#   7. Imprime um relatorio final dizendo claramente o que passou, falhou ou
#      nao foi executado.
#
# NOTA DE CODIFICACAO: este arquivo e deliberadamente escrito apenas com
# caracteres ASCII (sem acentos, sem travessao "-", apenas hifen simples "-").
# Uma versao anterior deste script usava travessoes e foi salva como UTF-8
# sem BOM; o Windows PowerShell 5.1 le arquivos .ps1 sem BOM usando a code
# page ANSI padrao do sistema, e os bytes UTF-8 do travessao foram
# reinterpretados como uma aspa curva, quebrando o parser em cascata a partir
# dali (erros "Argumento ausente" e "cadeia de caracteres sem terminador").
# Manter o script 100% ASCII elimina essa classe inteira de problema,
# independentemente da code page do Windows de quem for executa-lo.

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

Escrever-Secao "2. Verificando/instalando dependencias do projeto"
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

& $pythonExe -m pip install -e ".[dev]" --quiet
Verificar-CodigoSaida "instalacao do pacote ignis em modo editavel ([dev])"

Escrever-Secao "3. Contagem de parametros do Ignis-30M"
& $pythonExe scripts\count_params.py
Verificar-CodigoSaida "contagem de parametros (scripts/count_params.py)"

Escrever-Secao "4. Suite completa de testes automatizados (Fase 1 + Fase 2)"
& $pythonExe -m pytest -v
Verificar-CodigoSaida "suite completa de testes (pytest)"

Escrever-Secao "5. Smoke test do modelo completo (30,4M parametros) - NAO e treinamento"
& $pythonExe scripts\smoke_test_full.py --batch-size 1 --seq-len 32 --repeticoes 3
Verificar-CodigoSaida "smoke test do modelo completo (scripts/smoke_test_full.py)"

Escrever-Secao "6. Smoke test do pipeline de dados e do tokenizador (corpus sintetico)"
$dirTemp = Join-Path $env:TEMP ("ignis_validacao_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $dirTemp | Out-Null

try {
    $corpusDir = Join-Path $dirTemp "corpus"
    New-Item -ItemType Directory -Path $corpusDir | Out-Null
    $corpusArquivo = Join-Path $corpusDir "corpus.jsonl"
    $manifesto = Join-Path $dirTemp "manifesto.jsonl"
    $splits = Join-Path $dirTemp "splits.json"
    $tokenizer = Join-Path $dirTemp "tokenizer.json"
    $datasetBase = Join-Path $dirTemp "dataset"

    Write-Host "6.1 Gerando corpus sintetico (Python, UTF-8 sem BOM)..."
    & $pythonExe scripts\gerar_corpus_sintetico.py --saida $corpusArquivo
    Verificar-CodigoSaida "geracao do corpus sintetico"

    Write-Host "6.2 validate-corpus (ingestao, limpeza, triagem, manifesto)..."
    & $pythonExe -m ignis.cli validate-corpus --corpus-dir $corpusDir --manifesto-saida $manifesto
    Verificar-CodigoSaida "comando validate-corpus"

    Write-Host "6.3 prepare-splits (splits reproduziveis + verificacao de contaminacao)..."
    & $pythonExe -m ignis.cli prepare-splits --manifesto $manifesto --splits-saida $splits --seed 42
    Verificar-CodigoSaida "comando prepare-splits"

    Write-Host "6.4 train-tokenizer (BPE reduzido sobre o split de treino)..."
    & $pythonExe -m ignis.cli train-tokenizer --corpus-dir $corpusDir --manifesto $manifesto --splits $splits --vocab-size 512 --min-frequency 1 --saida $tokenizer
    Verificar-CodigoSaida "comando train-tokenizer"

    Write-Host "6.5 eval-tokenizer (round-trip e eficiencia)..."
    & $pythonExe -m ignis.cli eval-tokenizer --tokenizer $tokenizer --corpus-dir $corpusDir --manifesto $manifesto --splits $splits
    Verificar-CodigoSaida "comando eval-tokenizer"

    Write-Host "6.6 prepare-dataset (sequencias para predicao do proximo token)..."
    & $pythonExe -m ignis.cli prepare-dataset --tokenizer $tokenizer --corpus-dir $corpusDir --manifesto $manifesto --splits $splits --saida-base $datasetBase --seq-len 64
    Verificar-CodigoSaida "comando prepare-dataset"

    Write-Host "6.7 report (relatorio consolidado)..."
    & $pythonExe -m ignis.cli report --manifesto $manifesto --splits $splits --tokenizer $tokenizer
    Verificar-CodigoSaida "comando report"
}
finally {
    Remove-Item -Recurse -Force $dirTemp -ErrorAction SilentlyContinue
}

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
Write-Host "Validacao da Fase 2 concluida com sucesso." -ForegroundColor Green
Write-Host "Nenhum treinamento foi executado. Nenhum peso do modelo foi atualizado." -ForegroundColor Green
exit 0
