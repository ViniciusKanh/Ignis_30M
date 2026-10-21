# diagnostico_hardware.ps1
#
# Diagnostico de hardware/aceleracao do Ignis-30M (Sprint 4, secao 1).
# Script SOMENTE LEITURA: nao instala nada, nao desinstala nada, nao
# reinstala o PyTorch e nao altera nenhuma configuracao do ambiente. Apenas
# coleta informacoes do computador (CPU, RAM, GPU, driver, CUDA exposto
# pelo PyTorch ja instalado) e imprime uma RECOMENDACAO em texto, que exige
# aprovacao explicita separada antes de qualquer instalacao.
#
# Uso:
#   .\scripts\diagnostico_hardware.ps1
#
# NOTA DE CODIFICACAO: este arquivo e deliberadamente escrito apenas com
# caracteres ASCII (sem acentos, sem travessao "-", apenas hifen simples),
# pelo mesmo motivo documentado em validar_fase2.ps1/validar_fase3.ps1
# (Windows PowerShell 5.1 le .ps1 sem BOM pela code page ANSI do sistema,
# corrompendo bytes UTF-8 nao-ASCII).

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
    cpu = $null
    ram = $null
    gpus_detectadas_pelo_windows = @()
    nvidia_smi = $null
    torch = $null
    recomendacao = $null
}

Escrever-Secao "1. CPU"
try {
    $cpu = Get-CimInstance -ClassName Win32_Processor -ErrorAction Stop | Select-Object -First 1
    $infoCpu = [ordered]@{
        nome = $cpu.Name
        nucleos_fisicos = $cpu.NumberOfCores
        processadores_logicos = $cpu.NumberOfLogicalProcessors
        velocidade_max_mhz = $cpu.MaxClockSpeed
    }
    $relatorio.cpu = $infoCpu
    Write-Host ("  Nome: " + $infoCpu.nome)
    Write-Host ("  Nucleos fisicos: " + $infoCpu.nucleos_fisicos + "   Processadores logicos: " + $infoCpu.processadores_logicos)
    Write-Host ("  Velocidade maxima: " + $infoCpu.velocidade_max_mhz + " MHz")
} catch {
    Write-Host "  AVISO: nao foi possivel consultar informacoes de CPU via Win32_Processor." -ForegroundColor Yellow
}

Escrever-Secao "2. Memoria RAM"
try {
    $so = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop
    $ramTotalGb = [math]::Round($so.TotalVisibleMemorySize / 1MB, 2)
    $ramLivreGb = [math]::Round($so.FreePhysicalMemory / 1MB, 2)
    $relatorio.ram = [ordered]@{
        total_gb = $ramTotalGb
        livre_gb_no_momento = $ramLivreGb
    }
    Write-Host ("  RAM total: " + $ramTotalGb + " GB")
    Write-Host ("  RAM livre no momento desta consulta: " + $ramLivreGb + " GB (varia a cada execucao; nao usar para planejar batch size sem medir de novo)")
} catch {
    Write-Host "  AVISO: nao foi possivel consultar RAM via Win32_OperatingSystem." -ForegroundColor Yellow
}

Escrever-Secao "3. GPUs detectadas pelo Windows (Win32_VideoController)"
try {
    $gpus = Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop
    $listaGpus = @()
    foreach ($gpu in $gpus) {
        $vramGb = $null
        if ($gpu.AdapterRAM -and $gpu.AdapterRAM -gt 0) {
            $vramGb = [math]::Round($gpu.AdapterRAM / 1GB, 2)
        }
        $infoGpu = [ordered]@{
            nome = $gpu.Name
            vram_gb_relatada_pelo_driver = $vramGb
            versao_driver = $gpu.DriverVersion
            status = $gpu.Status
        }
        $listaGpus += $infoGpu
        Write-Host ("  Nome: " + $infoGpu.nome)
        Write-Host ("    VRAM relatada pelo driver: " + $(if ($vramGb) { "$vramGb GB" } else { "nao informada (comum em drivers WDDM mais novos; nao e necessariamente 0 GB real)" }))
        Write-Host ("    Versao do driver: " + $infoGpu.versao_driver + "   Status: " + $infoGpu.status)
    }
    $relatorio.gpus_detectadas_pelo_windows = $listaGpus
    if ($listaGpus.Count -eq 0) {
        Write-Host "  Nenhuma GPU detectada pelo Windows." -ForegroundColor Yellow
    }
} catch {
    Write-Host "  AVISO: nao foi possivel consultar GPUs via Win32_VideoController." -ForegroundColor Yellow
}

Escrever-Secao "4. nvidia-smi (se disponivel)"
$nvidiaSmiDisponivel = $false
try {
    $cmdNvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
    if ($cmdNvidiaSmi) {
        $nvidiaSmiDisponivel = $true
        Write-Host "  nvidia-smi encontrado - executando consulta (somente leitura)..."
        $saidaNvidia = & nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader 2>&1
        Write-Host "  $saidaNvidia"
        $relatorio.nvidia_smi = [ordered]@{
            disponivel = $true
            saida_bruta = ($saidaNvidia -join " | ")
        }
    } else {
        Write-Host "  nvidia-smi nao encontrado no PATH (normal se nao houver GPU NVIDIA ou driver NVIDIA instalado)."
        $relatorio.nvidia_smi = [ordered]@{ disponivel = $false }
    }
} catch {
    Write-Host "  AVISO: erro ao tentar executar nvidia-smi (tratado como indisponivel)." -ForegroundColor Yellow
    $relatorio.nvidia_smi = [ordered]@{ disponivel = $false }
}

Escrever-Secao "5. PyTorch instalado neste projeto (torch.cuda, SEM reinstalar nada)"
$pythonExe = Join-Path $raizProjeto ".venv\Scripts\python.exe"
$infoTorch = [ordered]@{
    encontrado = $false
    versao = $null
    cuda_disponivel = $false
    cuda_versao_compilada = $null
    nome_gpu_torch = $null
    capacidade_computo = $null
}
if (Test-Path $pythonExe) {
    $scriptPython = @"
import json
resultado = {"encontrado": False, "versao": None, "cuda_disponivel": False, "cuda_versao_compilada": None, "nome_gpu_torch": None, "capacidade_computo": None}
try:
    import torch
    resultado["encontrado"] = True
    resultado["versao"] = torch.__version__
    resultado["cuda_disponivel"] = bool(torch.cuda.is_available())
    resultado["cuda_versao_compilada"] = torch.version.cuda
    if resultado["cuda_disponivel"]:
        resultado["nome_gpu_torch"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        resultado["capacidade_computo"] = f"{major}.{minor}"
except ImportError:
    pass
print(json.dumps(resultado))
"@
    $tmpScript = Join-Path $env:TEMP ("ignis_diag_torch_" + [System.Guid]::NewGuid().ToString("N") + ".py")
    try {
        Set-Content -Path $tmpScript -Value $scriptPython -Encoding ASCII
        $saidaJson = & $pythonExe $tmpScript 2>&1
        try {
            $parsed = $saidaJson | ConvertFrom-Json
            $infoTorch.encontrado = $parsed.encontrado
            $infoTorch.versao = $parsed.versao
            $infoTorch.cuda_disponivel = $parsed.cuda_disponivel
            $infoTorch.cuda_versao_compilada = $parsed.cuda_versao_compilada
            $infoTorch.nome_gpu_torch = $parsed.nome_gpu_torch
            $infoTorch.capacidade_computo = $parsed.capacidade_computo
        } catch {
            Write-Host "  AVISO: nao foi possivel interpretar a saida do Python (torch pode nao estar instalado ainda)." -ForegroundColor Yellow
        }
    } finally {
        Remove-Item -Path $tmpScript -ErrorAction SilentlyContinue
    }
    Write-Host ("  torch encontrado: " + $infoTorch.encontrado)
    if ($infoTorch.encontrado) {
        Write-Host ("  Versao do torch: " + $infoTorch.versao)
        Write-Host ("  torch.cuda.is_available(): " + $infoTorch.cuda_disponivel)
        Write-Host ("  Versao CUDA com que o torch foi compilado: " + $(if ($infoTorch.cuda_versao_compilada) { $infoTorch.cuda_versao_compilada } else { "N/A (build CPU-only)" }))
        if ($infoTorch.cuda_disponivel) {
            Write-Host ("  GPU vista pelo torch: " + $infoTorch.nome_gpu_torch + "   Capacidade de computo: " + $infoTorch.capacidade_computo)
        }
    }
} else {
    Write-Host "  Ambiente virtual .venv nao encontrado - rode este script depois de criar o venv do projeto." -ForegroundColor Yellow
}
$relatorio.torch = $infoTorch

Escrever-Secao "6. Recomendacao (NAO EXECUTADA AUTOMATICAMENTE)"
$temGpuNvidia = $false
foreach ($gpu in $relatorio.gpus_detectadas_pelo_windows) {
    if ($gpu.nome -match "NVIDIA") { $temGpuNvidia = $true }
}

$recomendacao = @()
if ($infoTorch.encontrado -and $infoTorch.cuda_disponivel) {
    $recomendacao += "torch ja enxerga uma GPU CUDA utilizavel (" + $infoTorch.nome_gpu_torch + "). Nenhuma mudanca de ambiente e necessaria para usar aceleracao de GPU; o treinamento/benchmark pode ser rodado com --device cuda quando autorizado."
} elseif ($temGpuNvidia -or $nvidiaSmiDisponivel) {
    $recomendacao += "Uma GPU NVIDIA foi detectada pelo Windows e/ou nvidia-smi, mas o torch instalado neste projeto e uma build CPU-only (ou torch.cuda.is_available() retornou False)."
    $recomendacao += "Para habilitar aceleracao de GPU seria necessario REINSTALAR o torch com uma build CUDA compativel com o driver detectado (ex.: pip install torch --index-url https://download.pytorch.org/whl/cuXXX, onde XXX depende da versao do driver/CUDA Toolkit)."
    $recomendacao += "ISTO NAO FOI EXECUTADO. A politica do projeto (skill ignis-llm-engineer) exige autorizacao explicita antes de qualquer instalacao global ou substituicao do ambiente atual. Qualquer reinstalacao do torch deve ser proposta e aprovada separadamente, citando a versao exata de CUDA/driver compativel encontrada acima."
} else {
    $recomendacao += "Nenhuma GPU NVIDIA foi detectada. CPU continua sendo o unico caminho de execucao recomendado para este projeto no momento; nenhuma instalacao adicional e necessaria ou recomendada."
    $recomendacao += "GPUs AMD/Intel detectadas (se houver, ver secao 3) nao tem suporte robusto e amplamente validado para treinamento PyTorch no Windows nativo (ROCm e Linux-only na pratica atual; DirectML e experimental e NAO deve ser adotado sem avaliacao propria) - tratar como fora de escopo por ora, nao como uma recomendacao de instalacao."
}
$relatorio.recomendacao = $recomendacao
foreach ($linha in $recomendacao) {
    Write-Host ("  - " + $linha)
}

Escrever-Secao "Relatorio salvo"
$caminhoRelatorio = Join-Path $raizProjeto "diagnostico_hardware_resultado.json"
$relatorio | ConvertTo-Json -Depth 6 | Set-Content -Path $caminhoRelatorio -Encoding UTF8
Write-Host ("Relatorio completo salvo em: " + $caminhoRelatorio)
Write-Host ""
Write-Host "Nenhuma instalacao, desinstalacao ou alteracao de ambiente foi realizada por este script." -ForegroundColor Green
exit 0
