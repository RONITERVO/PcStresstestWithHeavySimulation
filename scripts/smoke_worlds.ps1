param(
    [string[]]$World = @(),
    [switch]$MinecraftOnly,
    [string]$WorldPattern = "",
    [int]$Width = 320,
    [int]$Height = 180,
    [int]$TileSize = 24,
    [int]$Substeps = 1,
    [int]$RaySteps = 32,
    [double]$FxIntensity = 0.3,
    [int]$CpuWorkers = 0,
    [double]$QuitAfter = 1.0,
    [int]$TimeoutSeconds = 20,
    [switch]$DryRun,
    [switch]$KeepGoing,
    [switch]$DisableThermalHold,
    [double]$MaxCpuTemp = 75.0,
    [double]$MaxGpuTemp = 70.0,
    [double]$ThermalPollSeconds = 1.0,
    [string]$ReportPath = "",
    [string]$LogDir = "logs\validation"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Format-PowerShellCommandArg {
    param([AllowEmptyString()][string]$Value)
    if ($null -eq $Value) {
        return "''"
    }
    return "'" + ($Value -replace "'", "''") + "'"
}

function Format-PowerShellCommandLine {
    param([string[]]$Parts)
    if ($Parts.Count -eq 0) {
        return ""
    }
    $command = Format-PowerShellCommandArg $Parts[0]
    $arguments = @($Parts | Select-Object -Skip 1 | ForEach-Object { Format-PowerShellCommandArg $_ })
    return ("& {0} {1}" -f $command, ($arguments -join " ")).TrimEnd()
}

function Format-ProcessCommandArg {
    param([AllowEmptyString()][string]$Value)
    if ($null -eq $Value) {
        return '""'
    }
    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') {
        return $Value
    }

    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append('"')
    $backslashes = 0
    foreach ($char in $Value.ToCharArray()) {
        if ($char -eq '\') {
            $backslashes += 1
            continue
        }
        if ($char -eq '"') {
            [void]$builder.Append('\' * (($backslashes * 2) + 1))
            [void]$builder.Append('"')
            $backslashes = 0
            continue
        }
        if ($backslashes -gt 0) {
            [void]$builder.Append('\' * $backslashes)
            $backslashes = 0
        }
        [void]$builder.Append($char)
    }
    if ($backslashes -gt 0) {
        [void]$builder.Append('\' * ($backslashes * 2))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Format-ProcessCommandLine {
    param([string[]]$Parts)
    return (($Parts | ForEach-Object { Format-ProcessCommandArg $_ }) -join " ")
}

function Format-MarkdownCell {
    param([object]$Value)
    if ($null -eq $Value) {
        return ""
    }
    $text = ([string]$Value) -replace '\r?\n', ' '
    $text = $text.Trim()
    if ($text.Length -gt 700) {
        $text = $text.Substring(0, 697) + "..."
    }
    $text = $text.Replace('\', '\\')
    $text = $text.Replace('|', '\|')
    $text = $text.Replace('`', '\`')
    return $text
}

function Resolve-PythonCandidate {
    param(
        [string]$Exe,
        [string[]]$PrefixArgs,
        [string]$Label
    )
    try {
        $testArgs = @($PrefixArgs + @("-c", "import sys; print(sys.executable)"))
        $output = & $Exe @testArgs 2>$null
        if ($LASTEXITCODE -ne 0) {
            return $null
        }

        $resolvedExe = @($output | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Last 1)
        if ($resolvedExe.Count -eq 0) {
            return $null
        }

        $resolvedPath = [string]$resolvedExe[0]
        if (Test-Path $resolvedPath) {
            $resolvedPath = (Resolve-Path $resolvedPath).Path
            return [pscustomobject]@{
                Exe = $resolvedPath
                PrefixArgs = @()
                Label = $resolvedPath
            }
        }

        return [pscustomobject]@{
            Exe = $Exe
            PrefixArgs = @($PrefixArgs)
            Label = $Label
        }
    }
    catch {
        return $null
    }
}

function Resolve-PythonCommand {
    $candidates = New-Object System.Collections.Generic.List[object]
    $venvPython = Join-Path $root ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $candidates.Add([pscustomobject]@{
            Exe = $venvPython
            PrefixArgs = @()
            Label = $venvPython
        })
    }
    $candidates.Add([pscustomobject]@{ Exe = "py"; PrefixArgs = @("-3.9"); Label = "py -3.9" })
    $candidates.Add([pscustomobject]@{ Exe = "py"; PrefixArgs = @("-3"); Label = "py -3" })
    $candidates.Add([pscustomobject]@{ Exe = "python"; PrefixArgs = @(); Label = "python" })

    foreach ($candidate in $candidates) {
        $resolved = Resolve-PythonCandidate `
            -Exe $candidate.Exe `
            -PrefixArgs @($candidate.PrefixArgs) `
            -Label $candidate.Label
        if ($null -ne $resolved) {
            return $resolved
        }
    }

    throw "Could not find a usable Python interpreter. Tried .venv\Scripts\python.exe, py -3.9, py -3, and python. Create the venv from requirements.txt and retry."
}

function Invoke-PythonChecked {
    param(
        [object]$Python,
        [string[]]$Arguments,
        [string]$FailureMessage
    )
    $allArgs = @($Python.PrefixArgs + $Arguments)
    $output = & $Python.Exe @allArgs
    if ($LASTEXITCODE -ne 0) {
        $command = Format-PowerShellCommandLine @(@($Python.Exe) + @($Python.PrefixArgs) + $Arguments)
        throw "$FailureMessage Command: $command"
    }
    return $output
}

function Get-RegisteredWorldInfo {
    param([object]$Python)
    $json = Invoke-PythonChecked `
        -Python $Python `
        -Arguments @("-c", "import json; from worlds.registry import iter_worlds; print(json.dumps([{'id': w.id, 'display_name': w.display_name} for w in iter_worlds()]))") `
        -FailureMessage "Could not load world registry."
    return @(($json -join "`n") | ConvertFrom-Json)
}

function Resolve-ValidationPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return Join-Path $root $Path
}

function Get-SafeName {
    param([string]$Value)
    return ($Value -replace '[^A-Za-z0-9._-]', '_')
}

function Classify-SmokeFailure {
    param([string]$Notes)
    if ($Notes -match '(?i)(traceback|shader|compile|compiler|syntax error|link error|fragment|vertex|uniform|GLSL|program failed|RuntimeError|ValueError)') {
        return "FAIL"
    }
    if ($Notes -match '(?i)(glfw.*(failed|error)|could not create.*(window|context)|failed to create.*(window|context)|context creation|wgl|glx|display.*(missing|unavailable|failed)|driver.*(missing|unavailable|failed)|no available video device)') {
        return "BLOCKED"
    }
    return "FAIL"
}

function Write-SmokeReport {
    param(
        [string]$Path,
        [object[]]$Results,
        [string[]]$Commands,
        [string]$RunLogDir,
        [bool]$ThermalHoldEnabled,
        [bool]$DryRunMode = $false
    )
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return
    }

    $reportFile = Resolve-ValidationPath $Path
    $reportDir = Split-Path -Parent $reportFile
    if (-not [string]::IsNullOrWhiteSpace($reportDir)) {
        New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
    }

    $lines = @(
        "# Garage Life Lab World Smoke Report",
        "",
        "Generated: $(Get-Date -Format o)",
        "Root: $root",
        "Mode: $(if ($DryRunMode) { 'dry run; commands generated only' } else { 'executed smoke commands' })",
        "Log directory: $RunLogDir",
        "Timeout seconds: $TimeoutSeconds",
        "Thermal limits: $(if ($ThermalHoldEnabled) { "enabled; poll ${ThermalPollSeconds}s" } else { 'disabled for legacy tiny smoke' })",
        "",
        "## Results",
        "",
        "| World | Result | Exit code | Seconds | Stdout | Stderr | Notes |",
        "| --- | --- | ---: | ---: | --- | --- | --- |"
    )
    foreach ($result in $Results) {
        $lines += "| $(Format-MarkdownCell $result.World) | $(Format-MarkdownCell $result.Result) | $(Format-MarkdownCell $result.ExitCode) | $(Format-MarkdownCell $result.Seconds) | $(Format-MarkdownCell $result.StdoutPath) | $(Format-MarkdownCell $result.StderrPath) | $(Format-MarkdownCell $result.Notes) |"
    }

    $lines += @(
        "",
        "## Commands",
        "",
        '```powershell'
    )
    $lines += $Commands
    $lines += @(
        '```',
        "",
        "## Visual And Performance Notes",
        "",
        "| World | FPS / responsiveness | Visual notes | Artifacts or blockers | Evidence | Follow-up |",
        "| --- | --- | --- | --- | --- | --- |"
    )
    foreach ($result in $Results) {
        $lines += "| $(Format-MarkdownCell $result.World) |  |  |  | stdout/stderr logs |  |"
    }

    Set-Content -Path $reportFile -Value $lines -Encoding UTF8
    Write-Host "Smoke report: $reportFile"
}

function Invoke-SmokeCommand {
    param(
        [object]$Python,
        [string[]]$Arguments,
        [string]$StdoutFile,
        [string]$StderrFile
    )

    $processArgs = @($Python.PrefixArgs + $Arguments)
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = $Python.Exe
    $processInfo.Arguments = Format-ProcessCommandLine $processArgs
    $processInfo.WorkingDirectory = $root
    $processInfo.UseShellExecute = $false
    $processInfo.CreateNoWindow = $true
    $processInfo.RedirectStandardOutput = $true
    $processInfo.RedirectStandardError = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $processInfo
    $null = $process.Start()
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()

    $timedOut = $false
    if ($TimeoutSeconds -gt 0) {
        $timedOut = -not $process.WaitForExit([int]($TimeoutSeconds * 1000))
    }
    else {
        $process.WaitForExit()
    }

    if ($timedOut) {
        try {
            $process.Kill()
            $process.WaitForExit()
        }
        catch {
        }
    }
    else {
        # Flush async redirected output after the process exits.
        $process.WaitForExit()
    }

    $stdout = $stdoutTask.Result
    $stderr = $stderrTask.Result
    Set-Content -Path $StdoutFile -Value $stdout -Encoding UTF8
    Set-Content -Path $StderrFile -Value $stderr -Encoding UTF8

    if ($timedOut) {
        return [pscustomobject]@{
            ExitCode = ""
            TimedOut = $true
            Notes = "Timed out after $TimeoutSeconds seconds"
        }
    }

    $notes = ""
    if ($process.ExitCode -ne 0 -and (Test-Path $StderrFile)) {
        $stderrText = $stderr.Trim()
        if (-not [string]::IsNullOrWhiteSpace($stderrText)) {
            $notes = $stderrText
        }
    }

    return [pscustomobject]@{
        ExitCode = $process.ExitCode
        TimedOut = $false
        Notes = $notes
    }
}

Push-Location $root
try {
    $python = Resolve-PythonCommand
    Write-Host "Using Python: $($python.Label)"

    $registeredWorldInfo = Get-RegisteredWorldInfo -Python $python
    $registeredWorldIds = @($registeredWorldInfo | ForEach-Object { $_.id })
    $isSelectedComparison = $World.Count -gt 0 -or $MinecraftOnly -or -not [string]::IsNullOrWhiteSpace($WorldPattern)
    $thermalHoldEnabled = -not $DisableThermalHold -and $isSelectedComparison

    if ($World.Count -gt 0) {
        $worlds = @($World)
        $missing = @($worlds | Where-Object { $registeredWorldIds -notcontains $_ })
        if ($missing.Count -gt 0) {
            throw "Unknown world id(s): $($missing -join ', ')"
        }
    }
    else {
        $worlds = @($registeredWorldIds)
    }

    if ($MinecraftOnly) {
        $worlds = @(
            $registeredWorldInfo |
                Where-Object {
                    $worlds -contains $_.id -and
                    (("{0} {1}" -f $_.id, $_.display_name) -like "*minecraft*")
                } |
                ForEach-Object { $_.id }
        )
    }

    if (-not [string]::IsNullOrWhiteSpace($WorldPattern)) {
        $worlds = @($worlds | Where-Object { $_ -like $WorldPattern })
    }

    $worlds = @($worlds | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($worlds.Count -eq 0) {
        throw "No worlds selected for smoke testing."
    }

    if ($DryRun) {
        $runLogDir = "dry run; no stdout/stderr logs written"
    }
    else {
        $logRoot = Resolve-ValidationPath $LogDir
        $runLogDir = Join-Path $logRoot ("smoke-{0}-pid{1}" -f (Get-Date -Format "yyyyMMdd-HHmmss-fff"), $PID)
        New-Item -ItemType Directory -Force -Path $runLogDir | Out-Null
    }

    $results = New-Object System.Collections.Generic.List[object]
    $commands = New-Object System.Collections.Generic.List[string]

    foreach ($world in $worlds) {
        $arguments = @(
            "main.py",
            "--world", $world,
            "-wnd", "glfw",
            "--width", "$Width",
            "--height", "$Height",
            "--tile-size", "$TileSize",
            "--substeps", "$Substeps",
            "--ray-steps", "$RaySteps",
            "--fx-intensity", "$FxIntensity",
            "--cpu-workers", "$CpuWorkers",
            "--no-hud",
            "--quit-after", "$QuitAfter"
        )
        if ($thermalHoldEnabled) {
            $arguments += @(
                "--max-cpu-temp", "$MaxCpuTemp",
                "--max-gpu-temp", "$MaxGpuTemp",
                "--thermal-poll-seconds", "$ThermalPollSeconds"
            )
        }
        else {
            $arguments += "--no-thermal-hold"
        }

        $commandParts = @($python.Exe) + @($python.PrefixArgs) + $arguments
        $commandLine = Format-PowerShellCommandLine $commandParts
        $commands.Add($commandLine)

        if ($DryRun) {
            Write-Host "Dry run world: $world"
            $results.Add([pscustomobject]@{
                World = $world
                Result = "DRY RUN"
                ExitCode = ""
                Seconds = 0
                Command = $commandLine
                StdoutPath = ""
                StderrPath = ""
                Notes = "Command generated but not executed."
            })
            continue
        }

        $safeWorld = Get-SafeName $world
        $stdoutFile = Join-Path $runLogDir "$safeWorld.stdout.log"
        $stderrFile = Join-Path $runLogDir "$safeWorld.stderr.log"
        $thermalLogPath = Join-Path $root "logs\thermal_events.log"
        $thermalLogBefore = $null
        if (Test-Path $thermalLogPath) {
            $thermalLogBefore = (Get-Item $thermalLogPath).LastWriteTimeUtc
        }

        Write-Host "Smoke testing world: $world"
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $result = "PASS"
        $notes = ""
        $exitCode = 0
        $smokeResult = $null
        try {
            $smokeResult = Invoke-SmokeCommand `
                -Python $python `
                -Arguments $arguments `
                -StdoutFile $stdoutFile `
                -StderrFile $stderrFile
            $exitCode = $smokeResult.ExitCode
            $notes = $smokeResult.Notes
            if ($smokeResult.TimedOut) {
                if ([string]::IsNullOrWhiteSpace($notes)) {
                    $notes = $smokeResult.Notes
                }
                $result = "TIMEOUT"
            }
            elseif ($exitCode -ne 0) {
                if ([string]::IsNullOrWhiteSpace($notes)) {
                    $notes = "Exited with code $exitCode"
                }
                $result = Classify-SmokeFailure $notes
            }
            elseif ($thermalHoldEnabled -and (Test-Path $thermalLogPath)) {
                $thermalLogAfter = (Get-Item $thermalLogPath).LastWriteTimeUtc
                if ($null -eq $thermalLogBefore -or $thermalLogAfter -gt $thermalLogBefore) {
                    $result = "THERMAL HOLD"
                    $notes = "Thermal hold log updated during smoke: $thermalLogPath"
                }
            }
        }
        catch {
            $result = Classify-SmokeFailure $_.Exception.Message
            $notes = $_.Exception.Message
            $exitCode = ""
        }
        finally {
            $stopwatch.Stop()
        }

        $results.Add([pscustomobject]@{
            World = $world
            Result = $result
            ExitCode = $exitCode
            Seconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 2)
            Command = $commandLine
            StdoutPath = $stdoutFile
            StderrPath = $stderrFile
            Notes = $notes
        })

        if ($result -ne "PASS" -and -not $KeepGoing) {
            Write-SmokeReport `
                -Path $ReportPath `
                -Results $results.ToArray() `
                -Commands $commands.ToArray() `
                -RunLogDir $runLogDir `
                -ThermalHoldEnabled $thermalHoldEnabled `
                -DryRunMode $DryRun.IsPresent
            throw "World smoke $($result.ToLowerInvariant()): $world. Command: $commandLine. Stdout: $stdoutFile. Stderr: $stderrFile. $notes"
        }
    }

    Write-SmokeReport `
        -Path $ReportPath `
        -Results $results.ToArray() `
        -Commands $commands.ToArray() `
        -RunLogDir $runLogDir `
        -ThermalHoldEnabled $thermalHoldEnabled `
        -DryRunMode $DryRun.IsPresent
    if ($DryRun) {
        Write-Host "Dry run complete. Commands generated for $($results.Count) world(s)."
    }
    elseif (@($results | Where-Object { $_.Result -ne "PASS" }).Count -gt 0) {
        $failedWorlds = @($results | Where-Object { $_.Result -ne "PASS" } | ForEach-Object { "$($_.World)=$($_.Result)" })
        throw "One or more world smoke tests did not pass: $($failedWorlds -join ', '). Report: $ReportPath"
    }
    else {
        Write-Host "All selected world smoke tests passed. Logs: $runLogDir"
    }
}
finally {
    Pop-Location
}
