$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    py -3.9 -m venv .venv
}

& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt pyinstaller

$separator = if ($IsWindows -or $env:OS -eq "Windows_NT") { ";" } else { ":" }
$addData = @(
    "main.py${separator}.",
    "engine.py${separator}.",
    "garage_life_presets.py${separator}.",
    "worlds${separator}worlds",
    "assets${separator}assets",
    "shaders${separator}shaders"
)

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name GarageLifeLab `
    --add-data $($addData[0]) `
    --add-data $($addData[1]) `
    --add-data $($addData[2]) `
    --add-data $($addData[3]) `
    --add-data $($addData[4]) `
    --add-data $($addData[5]) `
    launcher.py

Write-Host ""
Write-Host "Built: $root\dist\GarageLifeLab\GarageLifeLab.exe"
