@echo off
REM ============================================================
REM  Stage 2/3: cache text-encoder outputs (Qwen3-VL 32B nvfp4)
REM  --text_encoder_blocks_to_swap keeps RAM low.
REM ============================================================
setlocal
chcp 65001 >nul

set "PY=E:\ai-train\env-musubi\Scripts\python.exe"
set "SRC=E:\ai-train\src\musubi-tuner"
set "MODELS=F:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models"
set "DS=E:\ai-train\recipes\h3_koi_neon-dataset.toml"
set "LOG=E:\ai-train\runs\h3_koi_musubi\cache_text.log"

cd /d "%SRC%"

"%PY%" minimax_h3_cache_text_encoder_outputs.py ^
  --dataset_config "%DS%" ^
  --task t2va ^
  --text_encoder "%MODELS%\text_encoders\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" ^
  --text_cache_dtype bf16 ^
  --text_encoder_blocks_to_swap 50 ^
  > "%LOG%" 2>&1

echo EXITCODE=%ERRORLEVEL% >> "%LOG%"
