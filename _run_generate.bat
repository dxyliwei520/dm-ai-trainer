@echo off
REM ============================================================
REM  Generate ONE comparison clip with MiniMax-H3 (musubi-tuner)
REM    %1 = output folder suffix   (baseline | lora)
REM    %2 = extra lora args        (empty | --lora_weight ... --lora_multiplier 0.8)
REM ============================================================
setlocal
chcp 65001 >nul

set "PY=E:\ai-train\env-musubi\Scripts\python.exe"
set "SRC=E:\ai-train\src\musubi-tuner"
set "MODELS=F:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models"
set "TAG=%~1"
set "LORA="
if not "%~2"=="" set "LORA=%~2"
set "OUT=E:\ai-train\runs\gen_%TAG%"
set "LOG=E:\ai-train\runs\gen_%TAG%.log"

cd /d "%SRC%"

"%PY%" minimax_h3_generate_video.py ^
  --task t2va ^
  --dit "%MODELS%\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors" ^
  --video_vae "%MODELS%\vae\minimax_h3_video_vae_fp16.safetensors" ^
  --audio_vae "%MODELS%\vae\minimax_h3_audio_vae_fp32.safetensors" ^
  --text_encoder "%MODELS%\text_encoders\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" ^
  --text_encoder_blocks_to_swap 50 ^
  --attn_mode sdpa ^
  --blocks_to_swap 28 ^
  --prompt "a glowing neon koi fish swimming through a rainy Japanese alley at night, neon signs reflecting on wet pavement, slow drifting camera, rain ambience" ^
  --video_size 256 448 ^
  --video_length 22 ^
  --infer_steps 12 ^
  --seed 42 ^
  --allow_experimental_duration ^
  --output_type video ^
  --save_path "%OUT%" ^
  %LORA% ^
  > "%LOG%" 2>&1

echo EXITCODE=%ERRORLEVEL% >> "%LOG%"
