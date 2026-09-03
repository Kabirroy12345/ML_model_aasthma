@echo off
title HridyaVayu Live Platform
echo ==========================================================
echo    Starting HridyaVayu: Live Connected Cloud Platform
echo ==========================================================
cd /d "%~dp0"

echo [1/2] Starting Flask Backend in background...
start "HridyaVayu Backend" /B ".\venv\Scripts\python.exe" app.py

echo Waiting for Flask to initialize on port 7860...
timeout /t 4 /nobreak >nul

echo [2/2] Launching Cloudflare Public HTTPS Tunnel...
".\cloudflared_win.exe" tunnel --url http://127.0.0.1:7860
pause
