@echo off
REM ============================================================
REM  Stage 1/3: cache latents for MiniMax H3 LoRA (musubi-tuner)
REM  Only the VAE is loaded -> low memory footprint.
REM ============================================================
setlocal
chcp 65001 >nul

set "PY=E:\ai-train\env-musubi\Scripts\python.exe"
set "SRC=E:\ai-train\src\musubi-tuner"
set "MODELS=F:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models"
set "DS=E:\ai-train\recipes\h3_koi_neon-dataset.toml"
set "LOG=E:\ai-train\runs\h3_koi_musubi\cache_latents.log"

cd /d "%SRC%"

"%PY%" minimax_h3_cache_latents.py ^
  --dataset_config "%DS%" ^
  --task t2va ^
  --video_vae "%MODELS%\vae\minimax_h3_video_vae_fp16.safetensors" ^
  --audio_vae "%MODELS%\vae\minimax_h3_audio_vae_fp32.safetensors" ^
  --cache_seed 42 ^
  --allow_experimental_duration ^
  --skip_existing > "%LOG%" 2>&1

echo EXITCODE=%ERRORLEVEL% >> "%LOG%"
