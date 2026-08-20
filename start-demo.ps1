[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,

    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendRoot = Join-Path $projectRoot "frontend"
$frontendDist = Join-Path $frontendRoot "dist\index.html"
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$url = "http://127.0.0.1:$Port"

function Test-TwoGatesServer {
    param([string]$BaseUrl)

    try {
        $meta = Invoke-RestMethod -Uri "$BaseUrl/api/meta" -TimeoutSec 1
        return $meta.appName -eq "Two Gates"
    }
    catch {
        return $false
    }
}

function Open-TwoGatesBrowser {
    param([string]$BaseUrl)

    if (-not $NoBrowser) {
        Start-Process $BaseUrl
    }
}

function Install-PythonEnvironment {
    Write-Host "First launch: preparing the Python environment..." -ForegroundColor Yellow

    $uvCommand = Get-Command uv.exe -ErrorAction SilentlyContinue
    if ($null -ne $uvCommand) {
        & $uvCommand.Source sync --extra dev
        if ($LASTEXITCODE -ne 0) {
            throw "uv could not create the Python environment."
        }
        return
    }

    $pyCommand = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($null -eq $pyCommand) {
        throw "Python 3.12 or 3.13 is required. Install Python, then double-click Start Two Gates.cmd again."
    }

    & $pyCommand.Source -3.12 -m venv (Join-Path $projectRoot ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.12 is required. Install it, then double-click Start Two Gates.cmd again."
    }

    & $python -m pip install --disable-pip-version-check -e $projectRoot
    if ($LASTEXITCODE -ne 0) {
        throw "The Python dependencies could not be installed. Check the internet connection and try again."
    }
}

function Test-PythonEnvironment {
    if (-not (Test-Path -LiteralPath $python)) {
        return $false
    }

    & $python -c "import fastapi, torch, uvicorn, two_gates" 2>$null
    return $LASTEXITCODE -eq 0
}

function Resolve-PnpmCommand {
    $pnpmCommand = Get-Command pnpm.cmd -ErrorAction SilentlyContinue
    if ($null -ne $pnpmCommand) {
        return [PSCustomObject]@{
            Executable = $pnpmCommand.Source
            Prefix = @()
        }
    }

    $corepackCommand = Get-Command corepack.cmd -ErrorAction SilentlyContinue
    if ($null -ne $corepackCommand) {
        return [PSCustomObject]@{
            Executable = $corepackCommand.Source
            Prefix = @("pnpm")
        }
    }

    throw "Node.js with Corepack or pnpm is required. Install Node.js, then double-click Start Two Gates.cmd again."
}

function Test-FrontendBuildRequired {
    if (-not (Test-Path -LiteralPath $frontendDist)) {
        return $true
    }

    $sourcePaths = @(
        (Join-Path $frontendRoot "src"),
        (Join-Path $frontendRoot "public"),
        (Join-Path $frontendRoot "index.html"),
        (Join-Path $frontendRoot "package.json"),
        (Join-Path $frontendRoot "pnpm-lock.yaml"),
        (Join-Path $frontendRoot "tsconfig.json"),
        (Join-Path $frontendRoot "tsconfig.app.json"),
        (Join-Path $frontendRoot "tsconfig.node.json"),
        (Join-Path $frontendRoot "vite.config.ts")
    )

    $latestSource = Get-ChildItem -LiteralPath $sourcePaths -File -Recurse |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1

    if ($null -eq $latestSource) {
        return $false
    }

    $builtFile = Get-Item -LiteralPath $frontendDist
    return $latestSource.LastWriteTimeUtc -gt $builtFile.LastWriteTimeUtc
}

function Build-Frontend {
    Write-Host "Preparing the web interface..." -ForegroundColor Yellow
    $pnpm = Resolve-PnpmCommand
    $pnpmExecutable = $pnpm.Executable
    $pnpmPrefix = @($pnpm.Prefix)

    Push-Location $frontendRoot
    try {
        if (-not (Test-Path -LiteralPath (Join-Path $frontendRoot "node_modules"))) {
            & $pnpmExecutable @pnpmPrefix install --frozen-lockfile
            if ($LASTEXITCODE -ne 0) {
                throw "The frontend dependencies could not be installed. Check the internet connection and try again."
            }
        }

        & $pnpmExecutable @pnpmPrefix run build
        if ($LASTEXITCODE -ne 0) {
            throw "The frontend build failed."
        }
    }
    finally {
        Pop-Location
    }
}

function Start-BrowserWhenReady {
    param([string]$BaseUrl)

    if ($NoBrowser) {
        return
    }

    $watcher = @"
`$ProgressPreference = "SilentlyContinue"
for (`$attempt = 0; `$attempt -lt 120; `$attempt++) {
    try {
        `$meta = Invoke-RestMethod -Uri "$BaseUrl/api/meta" -TimeoutSec 1
        if (`$meta.appName -eq "Two Gates") {
            Start-Process "$BaseUrl"
            exit 0
        }
    }
    catch {}
    Start-Sleep -Milliseconds 250
}
"@

    $encodedWatcher = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($watcher))
    $powershell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
    Start-Process -FilePath $powershell -WindowStyle Hidden -ArgumentList @(
        "-NoProfile",
        "-EncodedCommand",
        $encodedWatcher
    ) | Out-Null
}

Push-Location $projectRoot
try {
    if (Test-TwoGatesServer -BaseUrl $url) {
        Write-Host "Two Gates is already running at $url" -ForegroundColor Green
        Open-TwoGatesBrowser -BaseUrl $url
        return
    }

    $env:PYTHONPATH = Join-Path $projectRoot "backend"

    if (-not (Test-PythonEnvironment)) {
        Install-PythonEnvironment
    }

    if (-not (Test-PythonEnvironment)) {
        throw "The Python environment is incomplete. Remove .venv and try the launcher again."
    }

    if (Test-FrontendBuildRequired) {
        Build-Frontend
    }

    Write-Host ""
    Write-Host "Two Gates is starting at $url" -ForegroundColor Cyan
    Write-Host "The browser opens automatically when the demo is ready." -ForegroundColor DarkGray
    Write-Host "Close this window or press Ctrl+C to stop the local server." -ForegroundColor DarkGray
    Write-Host ""

    Start-BrowserWhenReady -BaseUrl $url
    & $python -m uvicorn two_gates.api:app --host 127.0.0.1 --port $Port

    if ($LASTEXITCODE -ne 0) {
        throw "The Two Gates server stopped unexpectedly."
    }
}
catch {
    Write-Host ""
    Write-Host "Two Gates could not start" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Yellow
    exit 1
}
finally {
    Pop-Location
}
