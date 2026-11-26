@echo off
REM ============================================================
REM  Stage 3/3: train MiniMax-H3 LoRA (5 steps smoke test)
REM  --blocks_to_swap streams N transformer blocks from RAM/disk
REM  so the 33B model fits in 16 GB VRAM / ~32 GB RAM.
REM ============================================================
setlocal
chcp 65001 >nul

set "PY=E:\ai-train\env-musubi\Scripts\python.exe"
set "SRC=E:\ai-train\src\musubi-tuner"
set "MODELS=F:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models"
set "DS=E:\ai-train\recipes\h3_koi_neon-dataset.toml"
set "OUT=E:\ai-train\runs\h3_koi_musubi"
set "LOG=%OUT%\train_100steps.log"

cd /d "%SRC%"

"%PY%" minimax_h3_train_network.py ^
  --dataset_config "%DS%" ^
  --task t2va ^
  --video_only ^
  --dit "%MODELS%\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors" ^
  --vae "%MODELS%\vae\minimax_h3_video_vae_fp16.safetensors" ^
  --network_dim 16 --network_alpha 16 ^
  --sdpa --mixed_precision bf16 ^
  --gradient_checkpointing ^
  --blocks_to_swap 28 --block_swap_h2d_only ^
  --optimizer_type adamw8bit ^
  --learning_rate 1e-4 ^
  --max_train_steps 100 ^
  --save_every_n_steps 50 ^
  --seed 42 ^
  --output_dir "%OUT%" ^
  --output_name h3_koi_neon_r16_v1 ^
  > "%LOG%" 2>&1

echo EXITCODE=%ERRORLEVEL% >> "%LOG%"
