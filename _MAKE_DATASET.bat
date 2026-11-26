@echo off
REM ============================================================
REM  Auto-cut raw videos into MiniMax-H3 training clips
REM  1) drop any videos into  E:\ai-train\datasets\<PROJECT>\raw\
REM  2) edit PROJECT below
REM  3) double click this file
REM  Output -> E:\ai-train\datasets\<PROJECT>\videos\   (NN.mp4 + NN.txt)
REM  KEEP THIS FILE PURE ASCII
REM ============================================================
setlocal
chcp 65001 >nul
title H3 dataset cutter

REM ============ EDIT HERE ============
set "PROJECT=h3_koi_neon"
set "FRAMES=56"
set "WIDTH=448"
set "HEIGHT=256"
set "MODE=blur"
REM ===================================

set "PY=E:\ai-train\env-musubi\Scripts\python.exe"
set "SCRIPT=E:\ai-train\scripts\make_h3_dataset.py"
set "SRC=E:\ai-train\datasets\%PROJECT%\raw"
set "DST=E:\ai-train\datasets\%PROJECT%\videos"

echo ============================================================
echo   Cut raw videos into H3 training clips
echo ------------------------------------------------------------
echo   project : %PROJECT%
echo   input   : %SRC%
echo   output  : %DST%
echo   spec    : %FRAMES% frames (17n+5) / %WIDTH%x%HEIGHT% / 24 fps
echo ============================================================
echo.

if not exist "%SRC%" (
  echo [ERROR] input folder not found:
  echo         %SRC%
  echo         create it and drop your .mp4 files there first.
  pause
  exit /b 1
)

"%PY%" "%SCRIPT%" --project "%PROJECT%" --frames %FRAMES% --width %WIDTH% --height %HEIGHT% --mode %MODE%
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
  echo [done] clips are in: %DST%
  echo.
  echo Next steps:
  echo   1^) open each NN.txt and write ONE sentence describing that clip
  echo   2^) run _ONE_CLICK_TRAIN.bat with PROJECT=%PROJECT%
) else (
  echo [FAILED] see messages above
)
echo.
pause
