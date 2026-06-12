$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$exePath = Join-Path $root "dist\GarageLifeLab\GarageLifeLab.exe"
$registryPath = Join-Path $root "dist\GarageLifeLab\_internal\worlds\registry.py"
$candidatePath = Join-Path $root "dist\GarageLifeLab\_internal\worlds\minecraft_perfect_ecosystem_3d.py"
$sourcePaths = @(
    "launcher.py",
    "worlds\registry.py",
    "worlds\minecraft_perfect_ecosystem_3d.py"
)

if (-not (Test-Path $exePath) -or -not (Test-Path $registryPath) -or -not (Test-Path $candidatePath)) {
    exit 1
}

$exe = Get-Item -LiteralPath $exePath
$sourceFiles = @(
    $sourcePaths |
        ForEach-Object { Join-Path $root $_ } |
        Where-Object { Test-Path $_ } |
        ForEach-Object { Get-Item -LiteralPath $_ }
)

if ($sourceFiles.Count -ne $sourcePaths.Count) {
    exit 1
}

$latestSource = ($sourceFiles | Measure-Object -Property LastWriteTimeUtc -Maximum).Maximum
$hasCandidate = Select-String `
    -LiteralPath $registryPath `
    -SimpleMatch "minecraft_perfect_ecosystem_3d" `
    -Quiet

if ($hasCandidate -and $exe.LastWriteTimeUtc -ge $latestSource) {
    exit 0
}

exit 1
