@echo off
REM ============================================================
REM  Start the MiniMax-H3 LoRA training web UI
REM  URL:  http://127.0.0.1:8199
REM  Loopback only (127.0.0.1) - nothing is exposed to the LAN.
REM  KEEP THIS FILE PURE ASCII
REM ============================================================
setlocal
chcp 65001 >nul
title H3 LoRA Training WebUI

set "PY=E:\ai-train\env-aitk\Scripts\python.exe"
set "APP=E:\ai-train\webui\app.py"
set "LOG=E:\ai-train\webui\server.log"

echo ============================================================
echo   H3 LoRA Training WebUI
echo ------------------------------------------------------------
echo   address : http://127.0.0.1:8199
echo   log     : %LOG%
echo ============================================================
echo.

REM already running?
netstat -ano | findstr /c:"127.0.0.1:8199" | findstr LISTENING >nul
if %errorlevel% equ 0 (
  echo [info] server already running - opening browser
  start "" "http://127.0.0.1:8199/"
  timeout /t 2 /nobreak >nul
  exit /b 0
)

echo [1/2] starting server ...
start "" /min "%PY%" "%APP%"

echo [2/2] waiting for it to come up ...
timeout /t 6 /nobreak >nul
netstat -ano | findstr /c:"127.0.0.1:8199" | findstr LISTENING >nul
if %errorlevel% equ 0 (
  echo       OK - opening browser
  start "" "http://127.0.0.1:8199/"
) else (
  echo       [WARN] not listening yet - check %LOG%
)

echo.
echo Keep this window open while training. Close it to leave the server running.
echo.
pause
