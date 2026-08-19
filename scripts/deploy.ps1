<#
.SYNOPSIS
    gogaghe Deployment & Lifecycle Automation Script

.DESCRIPTION
    Automates building, running, stopping, and health checking gogaghe services.

.PARAMETER Mode
    Deployment mode: "docker" (default) or "local".

.PARAMETER AiBundle
    Enable AI Embedding sidecar profile (--profile ai-bundle).

.PARAMETER Down
    Stop and tear down the running Docker Compose stack.

.PARAMETER Status
    Check service health and metrics endpoint.

.EXAMPLE
    .\scripts\deploy.ps1
    .\scripts\deploy.ps1 -AiBundle
    .\scripts\deploy.ps1 -Local
    .\scripts\deploy.ps1 -Down
    .\scripts\deploy.ps1 -Status
#>

param (
    [ValidateSet("docker", "local")]
    [string]$Mode = "docker",

    [switch]$AiBundle,
    [switch]$Down,
    [switch]$Status
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ComposeFile = Join-Path $ProjectRoot "deployments\docker-compose\docker-compose.yml"
$BinaryPath  = Join-Path $ProjectRoot "bin\gogaghe-server.exe"
$ConfigFile  = Join-Path $ProjectRoot "configs\config.yaml"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  gogaghe Deployment & Management Script  " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

if ($Down) {
    Write-Host "[*] Stopping Docker Compose stack..." -ForegroundColor Yellow
    docker compose -f $ComposeFile down
    Write-Host "[+] All services stopped." -ForegroundColor Green
    exit 0
}

if ($Status) {
    Write-Host "[*] Checking service status..." -ForegroundColor Yellow
    docker compose -f $ComposeFile ps
    Write-Host ""
    Write-Host "[*] Probing metrics endpoint (http://localhost:2112/metrics)..." -ForegroundColor Yellow
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:2112/metrics" -TimeoutSec 3
        Write-Host "[+] Metrics endpoint healthy! HTTP Status: $($resp.StatusCode)" -ForegroundColor Green
    } catch {
        Write-Host "[-] Metrics endpoint unreachable. Is the server running?" -ForegroundColor Red
    }
    exit 0
}

if ($Mode -eq "local") {
    Write-Host "[*] Building local binary (CGO_ENABLED=0)..." -ForegroundColor Yellow
    $env:CGO_ENABLED = "0"
    Push-Location $ProjectRoot
    try {
        go build -ldflags="-s -w" -o $BinaryPath ./cmd/gogaghe-server/...
        Write-Host "[+] Build complete: $BinaryPath" -ForegroundColor Green
        Write-Host "[*] Starting gogaghe-server locally..." -ForegroundColor Cyan
        Write-Host "    gRPC Server   : localhost:50051" -ForegroundColor Gray
        Write-Host "    Metrics Server: http://localhost:2112/metrics" -ForegroundColor Gray
        Write-Host "    Press Ctrl+C to stop." -ForegroundColor DarkGray
        & $BinaryPath --config $ConfigFile
    } finally {
        Pop-Location
    }
    exit 0
}

# Mode Docker
Write-Host "[*] Deploying via Docker Compose..." -ForegroundColor Yellow

$profileArgs = @()
if ($AiBundle) {
    Write-Host "[+] AI Embedding sidecar enabled (--profile ai-bundle)" -ForegroundColor Magenta
    $profileArgs += "--profile"
    $profileArgs += "ai-bundle"
}

docker compose -f $ComposeFile @profileArgs up --build -d

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  [+] gogaghe stack is now live!          " -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  - gogaghe gRPC     : localhost:50051" -ForegroundColor Cyan
Write-Host "  - Prometheus /metrics: http://localhost:2112/metrics" -ForegroundColor Cyan
Write-Host "  - Prometheus UI    : http://localhost:9090" -ForegroundColor Cyan
Write-Host "  - Grafana Dashboard: http://localhost:3000 (admin / admin)" -ForegroundColor Cyan
if ($AiBundle) {
    Write-Host "  - Embedder Sidecar : http://localhost:8000" -ForegroundColor Magenta
}
Write-Host ""
Write-Host "To run smoke test: go run scripts/smoke_test.go" -ForegroundColor Yellow
Write-Host "To stop stack    : .\scripts\deploy.ps1 -Down" -ForegroundColor Yellow
