# 教程 02 · 视频 LoRA（MiniMax H3 / FastH3）

> 写于 2026-10-29。目标：**让模型学会“你的角色 / 你的风格 / 你的动作 / 你的声音”**——
> 比如固定人物出场、固定的镜头语感、固定的配音音色（H3 是音视频联合模型，LoRA 能连着声音一起学）。
> 工具：**ai-toolkit**（主）＋ **musubi-tuner**（备），环境 `E:\ai-train\env-aitk` / `env-musubi`
> 你不需要敲命令，把素材放好、跟我说一句就行。

---

## 0. 一句话原理

MiniMax H3 是 **33B 的音视频联合 DiT**（视频 + 立体声同时生成）。它太大了，整模型微调要 8 张 A800；
**LoRA 只训 1000 万左右参数的小贴片**，产物通常 **几十 MB**，能挂到任何一份 H3 权重上，随时关掉。

⚠️ **LoRA 认底座**：给「FastH3 8 步版」训的 LoRA 只能用在 FastH3 上；给「原版 fl2va」训的只能用在原版上。选哪个底座，决定 LoRA 用在哪。

---

## 1. 现状（已装好，**零下载**）

ai-toolkit 的 H3 扩展按 ComfyUI 的目录结构找文件，而你 `F:\…\ComfyUI\models\` 的结构**完全对得上**：

```powershell
$env:MODELS_PATH = 'F:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models'
```

| 工具要的（相对 `MODELS_PATH`） | 你已有的 | 用途 |
|---|---|---|
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | ✅ 19.5 GB | 文生/首帧生（**推荐先用这个**，质量最稳） |
| `diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors` | ✅ 20.6 GB | **你三个站正在用的 8 步快版**，训它 = 站点直接受益 |
| `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | ✅ 19.5 GB | 多参考图（换脸/换装/多角色） |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | ✅ 14.6 GB | 文本编码器（保持量化，省显存） |
| `vae/minimax_h3_video_vae_fp16` + `audio_vae_fp32` | ✅ | 视频/音频 VAE |

→ **不用下 66 GB 的 bf16 原版**。这一条是这次安装最大的收获。

---

## 2. 素材怎么准备（硬约束，不合规直接报错）

```
E:\ai-train\datasets\h3_<你的名字>\
  videos\    01.mp4   02.mp4   03.mp4  ...
  captions\  01.txt   02.txt   03.txt  ...   ← 每条视频一句描述（画面+动作+声音）
```

| 约束 | 值 | 为什么 |
|---|---|---|
| 帧率 | **24 fps** | 模型原生 24fps |
| 宽高 | **32 的倍数**（如 448×768、768×1344） | 模型几何要求 |
| 帧数 | **17n+5**：`22 / 39 / 56 / 73 / 90 / 107 / 124 …`（**124 帧≈5.2 秒最省显存**） | 训练/推理都按这个格点 |
| 时长 | **先控制在 5~6 秒** | 序列长度上限；30 秒的素材单卡根本吃不下 |
| 音频 | 有原始音频就带上；没有也能量（音轨当静音占位） | H3 是音视频联合的，带声音能学音色 |
| 数量 | **10~30 条**跑通，效果好再加到 100+ | 每条都是一次完整前向，多了很慢 |
| 命名 | 纯英文数字，无空格无中文 | 脚本处理 |

**caption 怎么写**：一句话说清「谁 + 在做什么 + 镜头/风格 + 什么声音」，例如：

```
一个穿黑色风衣的男人在雨夜街头点烟，霓虹灯侧逆光，浅景深，近景缓慢推镜，回响的雨声和低沉男声旁白
```

---

## 3. 路线 A：ai-toolkit（推荐，配置统一，好调）

### 命令（我实际跑的）

```powershell
$env:MODELS_PATH = 'F:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models'
E:\ai-train\env-aitk\Scripts\python.exe `
  E:\ai-train\src\ai-toolkit\run.py `
  E:\ai-train\recipes\h3_<名字>.yaml *> E:\ai-train\runs\h3_<名字>\train.log
```

### 配方模板（我会按素材调）

```yaml
job: extension
config:
  name: "h3_<名字>"
  process:
    - type: 'sd_trainer'
      training_folder: "E:/ai-train/runs/h3_<名字>"
      device: cuda:0
      network:
        type: "lora"
        linear: 16            # 角色/风格 16 起；复杂动作 32
        linear_alpha: 16
      save:
        dtype: float16
        save_every: 100
        max_step_saves_to_keep: 5
      datasets:
        - folder_path: "E:/ai-train/datasets/h3_<名字>/videos"
          caption_ext: "txt"
          resolution: [ 448, 768 ]     # 先小分辨率跑通，再放大
      train:
        batch_size: 1                  # H3 必须 1，靠梯度累积提等效 batch
        steps: 600
        gradient_accumulation: 2
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 1.0e-4
        dtype: bf16
      model:
        arch: "minimax_h3"
        name_or_path: "MiniMaxAI/MiniMax-H3"   # 只是占位标识；真实文件由 MODELS_PATH + partition 决定
        quantize: true
        low_vram: true                          # 16 GB 必开
        model_kwargs:
          partition: "fl2va_pruned"             # 或 fasth3 / ref2va_pruned
      sample:
        sampler: "flowmatch"
        sample_every: 100
        width: 448
        height: 768
        num_frames: 22                          # 样张先 22 帧（17*1+5）
        samples:
          - prompt: "和素材同一类描述"
        neg: ""
        seed: 42
meta:
  name: "[name]"
```

> 这份是按扩展代码写的模板，**还没在真机上跑过**（缺素材）。第一次我会先拿 1 条视频、5 步做“空转测试”，确认能加载、能存盘，再放全量。

---

## 4. 路线 B：musubi-tuner（显存/精度控制更细，备选）

musubi 的三段式：**缓存 latent → 缓存文本 → 训练**。它的 `--blocks_to_swap` 能把 48/50 个 block 换到内存，
是 16 GB 卡跑 33B 模型最实在的办法。

### ① 数据集 TOML（`recipes\h3_<名字>-dataset.toml`）

```toml
[general]
resolution = [768, 1344]
caption_extension = ".txt"
batch_size = 1

[[datasets]]
video_directory = "E:/ai-train/datasets/h3_<名字>/videos"
cache_directory = "E:/ai-train/runs/h3_<名字>/cache"
caption_extension = ".txt"
target_frames = [124]
```

### ② 两步缓存（各跑一次，之后复用）

```powershell
$m = 'E:\ai-train\env-musubi\Scripts\python.exe'
$M = 'F:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models'

& $m E:\ai-train\src\musubi-tuner\minimax_h3_cache_latents.py `
  --dataset_config E:\ai-train\recipes\h3_<名字>-dataset.toml --task t2va `
  --video_vae "$M\vae\minimax_h3_video_vae_fp16.safetensors" `
  --audio_vae "$M\vae\minimax_h3_audio_vae_fp32.safetensors" --cache_seed 42 --skip_existing

& $m E:\ai-train\src\musubi-tuner\minimax_h3_cache_text_encoder_outputs.py `
  --dataset_config E:\ai-train\recipes\h3_<名字>-dataset.toml --task t2va `
  --text_encoder "$M\text_encoders\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" `
  --text_cache_dtype bf16 --skip_existing --text_encoder_blocks_to_swap 50
```

### ③ 训练

```powershell
& $m E:\ai-train\src\musubi-tuner\minimax_h3_train_network.py `
  --dataset_config E:\ai-train\recipes\h3_<名字>-dataset.toml --task t2va `
  --dit "$M\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors" `
  --network_dim 16 --network_alpha 16 --sdpa --mixed_precision bf16 `
  --gradient_checkpointing --blocks_to_swap 40 --block_swap_h2d_only `
  --optimizer_type adamw8bit --learning_rate 1e-4 `
  --max_train_steps 600 --save_every_n_epochs 1 `
  --output_dir E:\ai-train\runs\h3_<名字> --output_name h3_<名字>
```

> `--blocks_to_swap` 是显存/内存的旋钮：**给得越大越省显存、越慢、越吃内存**。
> 官方说超过 45 建议 64 GB 内存，你只有 **31.8 GB** → 我从 30~40 试起，边跑边看。

---

## 5. 时间与资源预期（诚实版，别信"几分钟出片"）

| 项 | 预期 |
|---|---|
| 首次加载 | 5~15 分钟（19.5 GB int8 权重 + 14.6 GB 文本编码器从磁盘读） |
| 缓存 latent（每条 5 秒视频） | 数十秒~2 分钟 |
| **每步训练** | **未实测**。乐观 1~3 分钟/步；悲观的块换出场景可能 5 分钟+ |
| 600 步 | **大概率是"睡前跑、第二天看"**，不是几小时能完的小活 |
| 显存 | 目标 ≤ 15 GB；**必须停 8189**（它常驻 4.6 GB） |
| 内存 | 31.8 GB 是硬顶，块换出开太大就会开始用虚拟内存（变极慢） |

**正确打开方式**：先跑 `--max_train_steps 5`（5 步）测出“每步多少秒”，再决定总步数。别一上来就 2000 步。

---

## 6. 产出与使用

- 产物：`E:\ai-train\runs\h3_<名字>\*.safetensors`（几十 MB）
- 用法：拷到 `F:\…\ComfyUI\models\loras\`，在 FastH3 工作流里用 **LoraLoader** 挂在 unet 线上，权重 0.5~0.8 起试
- ⚠️ **底座要对上**：给 FastH3 训的 LoRA 挂到 FastH3；给原版 fl2va 训的挂原版。混用会出怪图但不是损坏
- 关掉 LoRA 即恢复原样，**原权重零改动**

---

## 7. 常见坑

| 现象 | 原因 | 处理 |
|---|---|---|
| 报错说帧数/尺寸不对 | 不是 17n+5 或不是 32 的倍数 | 用 `124 帧 @ 448×768` 这种安全值 |
| `Audio stream is discontinuous` | 音频和视频起点不齐 | musubi 新版会自动对齐；别切开再拼 |
| 训练极慢、显存不足来回抖 | 块换出开得太大 | `--blocks_to_swap` 从 40 往 30 调 |
| 训完 LoRA 没效果 | 底座不匹配 / 权重给太低 / 步数太少 | 先确认底座、权重拉到 0.8，步数加 2 倍 |
| OOM 在加载阶段 | 8189 / Ollama 还在占显存 | 训练前我自己会检查并提醒你关 |
| 想训"多参考图换装/换脸" | 那是 ref2va 路线 | `partition: ref2va_pruned` + 素材要给参考图，规则另说 |
