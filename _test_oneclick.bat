@echo off
REM ============================================================
REM  MiniMax-H3 video LoRA - one click trainer
REM  1) edit the 4 variables below with Notepad
REM  2) save
REM  3) double click this file
REM  NOTE: keep this file PURE ASCII (cmd mis-parses CJK here)
REM ============================================================
setlocal
chcp 65001 >nul
title MiniMax-H3 LoRA Trainer

REM ============ EDIT ONLY THESE 4 LINES ============
set "PROJECT=h3_koi_neon"
set "STEPS=2"
set "RANK=16"
set "FRAMES=22"
REM ==================================================

set "PY=E:\ai-train\env-musubi\Scripts\python.exe"
set "SRC=E:\ai-train\src\musubi-tuner"
set "MODELS=F:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models"
set "DS=E:\ai-train\recipes\%PROJECT%-dataset.toml"
set "OUT=E:\ai-train\runs\%PROJECT%"
set "VIDDIR=E:/ai-train/datasets/%PROJECT%/videos"
set "CACHEDIR=E:/ai-train/runs/%PROJECT%/cache"

echo ============================================================
echo   MiniMax-H3 LoRA  Training
echo ------------------------------------------------------------
echo   project : %PROJECT%
echo   steps   : %STEPS%
echo   rank    : %RANK%
echo   frames  : %FRAMES%
echo ============================================================
echo.

if not exist "E:\ai-train\datasets\%PROJECT%\videos" (
  echo [ERROR] video folder not found:
  echo         E:\ai-train\datasets\%PROJECT%\videos
  echo.
  pause
  exit /b 1
)

set /a VIDCOUNT=0
for %%f in ("E:\ai-train\datasets\%PROJECT%\videos\*.mp4") do set /a VIDCOUNT+=1
echo [check] found %VIDCOUNT% mp4 files
if %VIDCOUNT%==0 (
  echo [ERROR] no .mp4 in the videos folder
  pause
  exit /b 1
)
if not exist "%OUT%" mkdir "%OUT%"

echo [1/4] writing dataset config ...
> "%DS%" echo [general]
>>"%DS%" echo resolution = [448, 256]
>>"%DS%" echo caption_extension = ".txt"
>>"%DS%" echo batch_size = 1
>>"%DS%" echo enable_bucket = false
>>"%DS%" echo.
>>"%DS%" echo [[datasets]]
>>"%DS%" echo video_directory = "%VIDDIR%"
>>"%DS%" echo cache_directory = "%CACHEDIR%"
>>"%DS%" echo caption_extension = ".txt"
>>"%DS%" echo target_frames = [%FRAMES%]
echo       wrote %DS%

echo [2/4] stopping ComfyUI 8189 to free VRAM ...
schtasks /End /TN ComfyUI-D-8189 >nul 2>nul
taskkill /F /IM python.exe /FI "COMMANDLINE eq *main.py*" >nul 2>nul
timeout /t 4 /nobreak >nul

echo [3/4] caching latents ...
cd /d "%SRC%"
"%PY%" minimax_h3_cache_latents.py ^
  --dataset_config "%DS%" --task t2va ^
  --video_vae "%MODELS%\vae\minimax_h3_video_vae_fp16.safetensors" ^
  --audio_vae "%MODELS%\vae\minimax_h3_audio_vae_fp32.safetensors" ^
  --cache_seed 42 --allow_experimental_duration ^
  > "%OUT%\1_cache_latents.log" 2>&1
if errorlevel 1 (
  echo [ERROR] latent caching failed - see %OUT%\1_cache_latents.log
  pause
  exit /b 1
)
echo       done

echo       caching text encoder outputs ...
"%PY%" minimax_h3_cache_text_encoder_outputs.py ^
  --dataset_config "%DS%" --task t2va ^
  --text_encoder "%MODELS%\text_encoders\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" ^
  --text_cache_dtype bf16 --text_encoder_blocks_to_swap 50 ^
  > "%OUT%\2_cache_text.log" 2>&1
if errorlevel 1 (
  echo [ERROR] text caching failed - see %OUT%\2_cache_text.log
  pause
  exit /b 1
)
echo       done

echo [4/4] training %STEPS% steps ...
"%PY%" minimax_h3_train_network.py ^
  --dataset_config "%DS%" --task t2va --video_only ^
  --dit "%MODELS%\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors" ^
  --vae "%MODELS%\vae\minimax_h3_video_vae_fp16.safetensors" ^
  --network_dim %RANK% --network_alpha %RANK% ^
  --sdpa --mixed_precision bf16 --gradient_checkpointing ^
  --blocks_to_swap 28 --block_swap_h2d_only ^
  --optimizer_type adamw8bit --learning_rate 1e-4 ^
  --max_train_steps %STEPS% --save_every_n_steps 100 ^
  --seed 42 ^
  --output_dir "%OUT%" --output_name %PROJECT%_r%RANK% ^
  > "%OUT%\3_train.log" 2>&1

if errorlevel 1 (
  echo [ERROR] training failed - see %OUT%\3_train.log
  schtasks /Run /TN ComfyUI-D-8189 >nul 2>nul
  pause
  exit /b 1
)

echo.
echo [done] restarting ComfyUI 8189 ...
schtasks /Run /TN ComfyUI-D-8189 >nul 2>nul

echo.
echo ============================================================
echo   LoRA files:
dir /b "%OUT%\*.safetensors" 2>nul
echo   folder: %OUT%
echo.
echo   copy the .safetensors into:
echo   %MODELS%\loras\
echo ============================================================
echo.
pause
