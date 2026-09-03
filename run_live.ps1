# HridyaVayu Live Launcher
Write-Host '==========================================================' -ForegroundColor Cyan
Write-Host '   Starting HridyaVayu: Live Connected Cloud Platform     ' -ForegroundColor Yellow
Write-Host '==========================================================' -ForegroundColor Cyan

# Check if app.py is already running on port 7860
$portCheck = Get-NetTCPConnection -LocalPort 7860 -ErrorAction SilentlyContinue
if (-not $portCheck) {
    Write-Host '[1/2] Starting Flask Backend (port 7860)...' -ForegroundColor Green
    Start-Process -FilePath '.\venv\Scripts\python.exe' -ArgumentList 'app.py' -NoNewWindow
    Start-Sleep -Seconds 3
} else {
    Write-Host '[1/2] Flask Backend is already active on port 7860.' -ForegroundColor Green
}

Write-Host '[2/2] Launching Cloudflare HTTPS Public Tunnel...' -ForegroundColor Green
.\cloudflared_win.exe tunnel --url http://127.0.0.1:7860
