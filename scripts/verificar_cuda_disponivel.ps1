# verificar_cuda_disponivel.ps1
#
# Descoberta EMPIRICA (somente leitura) de disponibilidade de CUDA para o
# Ignis-30M (Sprint 5, secao 1). Este script:
#   - NAO instala, NAO desinstala e NAO reinstala nada (torch incluido).
#   - NAO assume uma versao especifica de PyTorch/CUDA como "a atual": em
#     vez disso, consulta o indice de pacotes do PyTorch (pip index
#     versions, que so LISTA versoes disponiveis, nunca baixa/instala nada)
#     para descobrir, no momento em que o script roda, quais tags de build
#     CUDA do torch existem e sao instalaveis no Python deste projeto.
#   - Usa o driver/compute capability REAIS desta maquina (via nvidia-smi)
#     para contextualizar a escolha, em vez de uma tabela fixa que pode
#     ficar desatualizada.
#
# Em que isso difere de diagnostico_hardware.ps1: aquele script e um
# diagnostico geral (CPU/RAM/GPU/torch atual); este script e focado
# especificamente em responder "quais builds CUDA do torch sao instalaveis
# agora, para este driver/GPU, e isso requer acesso a rede" -- o resultado
# alimenta os parametros de scripts/preparar_venv_cuda.ps1 (que exige que o
# operador passe explicitamente a tag escolhida, nunca adivinha uma).
#
# Uso:
#   .\scripts\verificar_cuda_disponivel.ps1
#
# NOTA DE CODIFICACAO: apenas caracteres ASCII (sem acentos, sem travessao),
# pelo mesmo motivo documentado em validar_fase2.ps1/diagnostico_hardware.ps1
# (Windows PowerShell 5.1 le .ps1 sem BOM pela code page ANSI do sistema).

$ErrorActionPreference = "Continue"

function Escrever-Secao {
    param([string]$titulo)
    Write-Host ""
    Write-Host "==== $titulo ====" -ForegroundColor Cyan
}

$raizProjeto = Split-Path -Parent $PSScriptRoot
Set-Location $raizProjeto

$relatorio = [ordered]@{
    gerado_em_utc = (Get-Date).ToUniversalTime().ToString("o")
    aviso = "Este script e SOMENTE LEITURA: nenhuma instalacao, desinstalacao ou alteracao de ambiente foi realizada."
    gpu_nvidia = $null
    nvidia_smi = $null
    python_do_projeto = $null
    tags_cuda_testadas = @()
    recomendacao = @()
}

Escrever-Secao "1. GPU NVIDIA e driver (via nvidia-smi, somente leitura)"
$nvidiaSmiDisponivel = $false
$computeCapDetectada = $null
$driverVersao = $null
try {
    $cmdNvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
    if ($cmdNvidiaSmi) {
        $nvidiaSmiDisponivel = $true
        $saidaBasica = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1
        Write-Host ("  " + $saidaBasica)
        $relatorio.nvidia_smi = [ordered]@{ disponivel = $true; saida_basica = ($saidaBasica -join " | ") }

        # compute_cap pode nao ser suportado por drivers/nvidia-smi muito
        # antigos; tratado como best-effort, nunca assumido.
        try {
            $saidaCap = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>&1
            if ($LASTEXITCODE -eq 0 -and $saidaCap -and ($saidaCap -notmatch "not a valid field")) {
                $computeCapDetectada = ($saidaCap | Select-Object -First 1).ToString().Trim()
                Write-Host ("  Compute capability (via nvidia-smi): " + $computeCapDetectada)
            } else {
                Write-Host "  nvidia-smi nesta maquina nao suporta --query-gpu=compute_cap; compute capability nao determinada automaticamente." -ForegroundColor Yellow
            }
        } catch {
            Write-Host "  AVISO: falha ao consultar compute_cap via nvidia-smi." -ForegroundColor Yellow
        }

        try {
            $driverVersao = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | Select-Object -First 1).ToString().Trim()
        } catch {
            $driverVersao = $null
        }
        $relatorio.gpu_nvidia = [ordered]@{
            detectada = $true
            compute_capability = $computeCapDetectada
            driver_version = $driverVersao
        }
    } else {
        Write-Host "  nvidia-smi nao encontrado no PATH -- sem GPU NVIDIA utilizavel, ou driver nao instalado."
        $relatorio.nvidia_smi = [ordered]@{ disponivel = $false }
        $relatorio.gpu_nvidia = [ordered]@{ detectada = $false }
    }
} catch {
    Write-Host "  AVISO: erro ao tentar executar nvidia-smi." -ForegroundColor Yellow
    $relatorio.nvidia_smi = [ordered]@{ disponivel = $false }
}

Escrever-Secao "2. Python/pip do projeto (.venv atual, NUNCA modificado por este script)"
$pythonExe = Join-Path $raizProjeto ".venv\Scripts\python.exe"
$pythonVersao = $null
if (Test-Path $pythonExe) {
    try {
        $pythonVersao = (& $pythonExe --version 2>&1).ToString().Trim()
        Write-Host ("  Interpretador: " + $pythonExe)
        Write-Host ("  Versao: " + $pythonVersao)
    } catch {
        Write-Host "  AVISO: nao foi possivel executar o Python do .venv atual." -ForegroundColor Yellow
    }
} else {
    Write-Host "  .venv do projeto nao encontrado em: $pythonExe" -ForegroundColor Yellow
}
$relatorio.python_do_projeto = [ordered]@{
    caminho = $pythonExe
    encontrado = (Test-Path $pythonExe)
    versao = $pythonVersao
}

Escrever-Secao "3. Descoberta EMPIRICA de tags de build CUDA do torch (consulta de rede, SEM instalar nada)"
Write-Host "  'pip index versions' apenas LISTA versoes disponiveis em um indice; nao baixa nem instala pacotes."
Write-Host "  Se esta maquina nao tiver acesso a download.pytorch.org, as consultas abaixo falharao - isso e"
Write-Host "  reportado como tal, nunca escondido ou substituido por um palpite."
Write-Host ""

# Tags de indice de build CUDA historicamente publicadas pelo PyTorch
# (https://download.pytorch.org/whl/<tag>). Esta lista pode ficar
# desatualizada com o tempo -- por isso o script TESTA cada uma de verdade
# em vez de assumir que a lista esta correta; tags que nao existirem mais
# simplesmente retornarao "nao encontrada" abaixo, sem quebrar o script.
$tagsCandidatas = @("cu118", "cu121", "cu124", "cu126", "cu128")

if (-not (Test-Path $pythonExe)) {
    Write-Host "  PULADO: .venv do projeto nao encontrado; nao e possivel consultar pip deste ambiente." -ForegroundColor Yellow
} else {
    foreach ($tag in $tagsCandidatas) {
        $indexUrl = "https://download.pytorch.org/whl/$tag"
        Write-Host ("  Testando tag '$tag' ($indexUrl) ...")
        $resultadoTag = [ordered]@{
            tag = $tag
            index_url = $indexUrl
            consulta_bem_sucedida = $false
            versoes_encontradas = @()
            detalhe = $null
        }
        try {
            $saidaPip = & $pythonExe -m pip index versions torch --index-url $indexUrl 2>&1
            $textoSaida = ($saidaPip -join "`n")
            if ($LASTEXITCODE -eq 0 -and $textoSaida -match "Available versions:") {
                $resultadoTag.consulta_bem_sucedida = $true
                $linhaVersoes = ($textoSaida -split "`n" | Where-Object { $_ -match "Available versions:" } | Select-Object -First 1)
                $resultadoTag.versoes_encontradas = $linhaVersoes.Trim()
                Write-Host ("    OK: " + $linhaVersoes.Trim())
            } else {
                $resultadoTag.detalhe = ($textoSaida -split "`n" | Select-Object -First 3) -join " / "
                Write-Host ("    Nao encontrada ou consulta falhou (tag pode nao existir mais, ou sem acesso de rede).") -ForegroundColor Yellow
            }
        } catch {
            $resultadoTag.detalhe = $_.Exception.Message
            Write-Host ("    ERRO ao consultar: " + $_.Exception.Message) -ForegroundColor Yellow
        }
        $relatorio.tags_cuda_testadas += $resultadoTag
    }
}

Escrever-Secao "4. Recomendacao (NAO EXECUTADA AUTOMATICAMENTE)"
$recomendacao = @()
$tagsOk = @($relatorio.tags_cuda_testadas | Where-Object { $_.consulta_bem_sucedida })

if (-not $relatorio.nvidia_smi.disponivel) {
    $recomendacao += "Nenhuma GPU NVIDIA detectada (nvidia-smi ausente) -- instalacao de CUDA nao se aplica; continue em CPU."
} elseif ($tagsOk.Count -eq 0) {
    $recomendacao += "Nao foi possivel confirmar NENHUMA tag de build CUDA do torch via rede nesta execucao (ver secao 3 para detalhes de cada tentativa). Isso pode significar falta de acesso a download.pytorch.org, ou que as tags desta lista mudaram. NENHUMA instalacao deve ser proposta sem essa confirmacao; rode este script novamente com acesso de rede, ou verifique manualmente em pytorch.org/get-started/locally qual tag se aplica a este driver (compute capability detectada: $computeCapDetectada, driver: $driverVersao)."
} else {
    $tagsNomes = ($tagsOk | ForEach-Object { $_.tag }) -join ", "
    $recomendacao += "Tag(s) de build CUDA confirmada(s) como instalaveis neste momento: $tagsNomes."
    $recomendacao += "Compute capability detectada nesta GPU: $(if ($computeCapDetectada) { $computeCapDetectada } else { 'nao determinada automaticamente - confirme manualmente antes de prosseguir' })."
    $recomendacao += "PROXIMO PASSO (requer autorizacao explicita separada): rode scripts\preparar_venv_cuda.ps1 -Confirmar -IndexUrl https://download.pytorch.org/whl/<tag escolhida> -- esse script cria um .venv-cuda SEPARADO (nunca mexe no .venv atual) e so instala algo apos essa confirmacao explicita."
    $recomendacao += "NENHUMA instalacao foi executada por este script de descoberta."
}
$relatorio.recomendacao = $recomendacao
foreach ($linha in $recomendacao) {
    Write-Host ("  - " + $linha)
}

Escrever-Secao "Relatorio salvo"
$caminhoRelatorio = Join-Path $raizProjeto "verificar_cuda_disponivel_resultado.json"
$relatorio | ConvertTo-Json -Depth 8 | Set-Content -Path $caminhoRelatorio -Encoding UTF8
Write-Host ("Relatorio completo salvo em: " + $caminhoRelatorio)
Write-Host ""
Write-Host "Nenhuma instalacao, desinstalacao ou alteracao de ambiente foi realizada por este script." -ForegroundColor Green
exit 0
