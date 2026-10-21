# Copia as instrucoes Codex sem substituir conteudo existente divergente.
param(
    [Parameter(Mandatory = $true)]
    [string]$Repositorio
)

$ErrorActionPreference = 'Stop'
$raizPacote = Split-Path -Parent $MyInvocation.MyCommand.Path
$raizRepo = (Resolve-Path -LiteralPath $Repositorio).Path

if (-not (Test-Path -LiteralPath (Join-Path $raizRepo 'src\ignis') -PathType Container)) {
    throw "Destino nao parece ser o repositorio Ignis: $raizRepo (src\ignis nao encontrado)"
}

$arquivos = @(
    'AGENTS.md',
    '.agents\skills\ignis-llm-engineer\SKILL.md',
    '.agents\skills\ignis-llm-engineer\references\criterios-cientificos.md',
    '.agents\skills\ignis-llm-engineer\references\handover-sprint6.md'
)

foreach ($relativo in $arquivos) {
    $origem = Join-Path $raizPacote $relativo
    $destino = Join-Path $raizRepo $relativo
    $pasta = Split-Path -Parent $destino
    New-Item -ItemType Directory -Force -Path $pasta | Out-Null

    if (Test-Path -LiteralPath $destino) {
        $hashOrigem = (Get-FileHash -LiteralPath $origem -Algorithm SHA256).Hash
        $hashDestino = (Get-FileHash -LiteralPath $destino -Algorithm SHA256).Hash
        if ($hashOrigem -eq $hashDestino) {
            Write-Host "[OK] Ja instalado: $relativo"
            continue
        }
        $alternativo = "$destino.novo"
        if (Test-Path -LiteralPath $alternativo) {
            Write-Warning "CONFLITO: $relativo ja existe e $alternativo tambem. Nenhum arquivo alterado."
            continue
        }
        Copy-Item -LiteralPath $origem -Destination $alternativo
        Write-Warning "CONFLITO: $relativo mantido; nova versao em $alternativo. Comparar manualmente."
        continue
    }

    Copy-Item -LiteralPath $origem -Destination $destino
    Write-Host "[OK] Instalado: $relativo"
}

Write-Host 'Concluido. Nenhuma dependencia, corpus, checkpoint ou commit foi alterado.'
