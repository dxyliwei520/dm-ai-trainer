# DM AI Trainer (H3 video + Qwen image training)

Scripts, recipes and small web UIs used to prepare datasets and run local
LoRA/finetune training for video (H3 / LTX-style) and image (Qwen-Image) models.

## Layout
- `scripts/` — dataset prep, latent/text caching, training entrypoints
- `recipes/` — training configs
- `webui/`, `webui_qwen/` — small local Gradio UIs
- `docs/` — notes
- `*.bat` — one-click wrappers (Windows)
- `fetch-src.ps1` — fetches the upstream trainers (ai-toolkit, musubi-tuner, diffusers)

## Not included (intentionally)
- `datasets/`, `models/`, `runs/` (large binaries)
- vendored upstream sources (`src/ai-toolkit`, `src/musubi-tuner`, `src/diffusers-pinned`) — re-fetch with `fetch-src.ps1`
