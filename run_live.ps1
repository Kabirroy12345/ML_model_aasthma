# HridyaVayu Live Launcher
Write-Host '==========================================================' -ForegroundColor Cyan
Write-Host '   Starting HridyaVayu: Live Connected Cloud Platform     ' -ForegroundColor Yellow
Write-Host '==========================================================' -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $scriptDir) { $scriptDir = Get-Location }

$pythonExe = Join-Path $scriptDir 'venv\Scripts\python.exe'
$appPy = Join-Path $scriptDir 'app.py'
$cloudflared = Join-Path $scriptDir 'cloudflared_win.exe'

$portCheck = Get-NetTCPConnection -LocalPort 7860 -ErrorAction SilentlyContinue
if (-not $portCheck) {
    Write-Host '[1/2] Starting Flask Backend in background...' -ForegroundColor Green
    Start-Process -FilePath $pythonExe -ArgumentList $appPy -WorkingDirectory $scriptDir
    Start-Sleep -Seconds 4
} else {
    Write-Host '[1/2] Flask Backend is already active on port 7860.' -ForegroundColor Green
}

Write-Host '[2/2] Launching Cloudflare HTTPS Public Tunnel...' -ForegroundColor Green
& $cloudflared tunnel --url http://127.0.0.1:7860
