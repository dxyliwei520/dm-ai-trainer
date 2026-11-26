@echo off
setlocal
title Qwen-Image-2.1 LoRA Trainer  (127.0.0.1:8198)
cd /d E:\ai-train

set "PY=C:\Users\liwei\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=E:\ai-train\env-aitk\Scripts\python.exe"

rem --- is something already listening on 8198? (python probe: netstat TIME_WAIT would fool findstr) ---
"%PY%" -c "import socket,sys; s=socket.socket(); s.settimeout(0.5); sys.exit(0 if s.connect_ex(('127.0.0.1',8198))==0 else 1)"
if %errorlevel%==0 goto OPENONLY

echo ==========================================================
echo   Qwen-Image-2.1  Image Edit LoRA Trainer
echo ----------------------------------------------------------
echo   Web UI : http://127.0.0.1:8198
echo   Backend: same process (FastAPI on 127.0.0.1:8198)
echo   Stop   : close this window, or press Ctrl+C
echo ==========================================================
echo.

rem --- open the browser 3 seconds later (server needs a moment) ---
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:8198'"

"%PY%" "E:\ai-train\webui_qwen\app.py"

echo.
echo [server stopped]
pause
exit /b

:OPENONLY
echo Server is already running on port 8198.
echo Opening http://127.0.0.1:8198 ...
start "" http://127.0.0.1:8198
timeout /t 2 >nul
exit /b
