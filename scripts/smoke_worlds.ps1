$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$worlds = & $python -c "from worlds.registry import world_ids; print('\n'.join(world_ids()))"
if ($LASTEXITCODE -ne 0) {
    throw "Could not load world registry."
}

foreach ($world in $worlds) {
    if ([string]::IsNullOrWhiteSpace($world)) {
        continue
    }
    Write-Host "Smoke testing world: $world"
    & $python main.py `
        --world $world `
        -wnd glfw `
        --width 320 `
        --height 180 `
        --tile-size 24 `
        --substeps 1 `
        --ray-steps 32 `
        --fx-intensity 0.3 `
        --cpu-workers 0 `
        --no-thermal-hold `
        --no-hud `
        --quit-after 1.0
    if ($LASTEXITCODE -ne 0) {
        throw "World smoke failed: $world"
    }
}

Write-Host "All world smoke tests passed."
