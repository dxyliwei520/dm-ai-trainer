# 教程 01 · 图片 LoRA（Qwen-Image-2.1 编辑模型）

> 写于 2026-10-29。目标：**教会模型一种“改图风格/改图逻辑”**——比如“把任何商品图换成你家统一的白底+光影”、“把我画的草图统一变成某种画风”、“固定的一组修图动作”。
> 工具：**ai-toolkit**（`E:\ai-train\src\ai-toolkit`，环境 `E:\ai-train\env-aitk`）
> 你不需要敲命令，把素材放好、跟我说一句就行。本文档是给你看“它在干什么 / 为什么这么要求”。

---

## 0. 一句话原理

Qwen-Image-2.1 是一个「文生图 + 改图」二合一模型：**给参考图它就是编辑，不给就是生成**。
LoRA 微调 = 不动原模型（6.76 GB 的 int8 权重一个字节都不改），只额外训一个几十 MB 的“小贴片”，让模型学会你这一批素材里的规律。
→ 训完得到一个 `.safetensors`（通常 20~150 MB），放进 ComfyUI 的 `models\loras\` 就能用，**随时可以关掉/换掉**。

---

## 1. 现状（已装好，不用你再装）

| 项 | 值 |
|---|---|
| 训练环境 | `E:\ai-train\env-aitk`（Python 3.12.9 + torch 2.13.0+cu130，已认到 5070 Ti） |
| 工具 | ai-toolkit（内置 `qwen_image_2` 扩展，已导入成功） |
| 主干模型 | `F:\…\ComfyUI\models\diffusion_models\qwen_image_2.1_int8_convrot.safetensors`（6.76 GB，✅ 已有） |
| 显存 | 16 GB（**训练前必须停 8189**，否则装不下） |

---

## 2. ⚠️ 第一次训之前，可能要先补 30 GB（唯一未解决项）

ai-toolkit 的 Qwen 加载器除了主干，还要一个 **diffusers 结构的目录**（里面要有 `text_encoder/`、`vae/`、`tokenizer`、`processor` 这些配置），而你 F 盘只有 ComfyUI 的单文件。

两条路，我到时候按实测选：

| 方案 | 内容 | 代价 |
|---|---|---|
| **A（稳）** | 从 hf-mirror 下 `Qwen/Qwen-Image-2.1`（约 **30 GB**：主干 13 GB + 文本编码器 16 GB + VAE 0.6 GB） | 一次性 30 GB 磁盘 + 下载时间；之后一直能用 |
| **B（省）** | 只下配置类小文件（几 MB），把文本编码器/VAE 指到你已有的 `qwen3vl_8b_int8_convrot` / `qwen_image_2.1_vae_bf16` | 省 30 GB，但**能不能被它加载还没验证**（ComfyUI 的 int8 量化格式不一定被它认） |

**建议**：等你确定要训图片 LoRA 时，我先花 10 分钟试 B；B 不行就下 A。别现在下，可能白占 30 GB。

---

## 3. 素材怎么准备（这一步决定成败）

```
E:\ai-train\datasets\qwen_<你的名字>\
  control\    001.png   002.png   003.png  ...   ← 输入图（要改的图）
  target\     001.png   002.png   003.png  ...   ← 目标图（改完应该长什么样）
  captions\   001.txt   002.txt   003.txt  ...   ← 这一对是「怎么改」的指令
```

**硬要求**

| 项 | 要求 | 原因 |
|---|---|---|
| 配对 | `control/001.png` ↔ `target/001.png` **同名** | 工具按文件名配对；对不上会被跳过 |
| 数量 | **30 对起**，50~100 对最佳 | 少于 20 对基本学不出稳定效果 |
| 尺寸 | 边长 1024 附近（1024×1024 / 1024×1536 / 1536×1024） | 2.1 原生 2K，但 1024 训练省显存 |
| 格式 | PNG / JPG，**文件名纯英文数字，无空格无中文** | 脚本处理路径最稳 |
| 内容 | control→target 的变化要**一致**（同一类改动），不要一张换背景一张改人脸 | LoRA 学的是“共同规律” |

**caption 怎么写（很重要）**

每张 `.txt` 写一句“**对这张图做什么**”，用自然语言，中文也行：

```
把背景换成纯白，保留商品和它的投影，整体提亮，商业摄影风格
```

✅ 好：描述**动作 + 目标**，每对之间措辞尽量统一（同一套词），模型才抓得住规律
❌ 差：`好看一点`（太虚）、`换背景+改颜色+加文字`（一次太多件事）

---

## 4. 我实际会跑的命令（你看个大概就行）

```powershell
# 1) 让工具去 F 盘找模型，而不是自己下载
$env:MODELS_PATH = 'F:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models'

# 2) 起训练（配方由我按你的素材生成，放在 recipes\ 下）
E:\ai-train\env-aitk\Scripts\python.exe `
  E:\ai-train\src\ai-toolkit\run.py `
  E:\ai-train\recipes\qwen_<名字>.yaml *> E:\ai-train\runs\qwen_<名字>\train.log
```

### 配方模板（我会按素材调这几个数）

```yaml
job: extension
config:
  name: "qwen_<名字>"
  process:
    - type: 'sd_trainer'
      training_folder: "E:/ai-train/runs/qwen_<名字>"
      device: cuda:0
      network:
        type: "lora"
        linear: 16            # 秩：风格学 8~16，复杂改图 32
        linear_alpha: 16
      save:
        dtype: float16
        save_every: 250
        max_step_saves_to_keep: 4
      datasets:
        - folder_path: "E:/ai-train/datasets/qwen_<名字>/target"
          control_path: "E:/ai-train/datasets/qwen_<名字>/control"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          resolution: [ 1024 ]
      train:
        batch_size: 1
        steps: 1500           # 30 对 → 1000~2000 步
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true     # 16 GB 必须开
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 1.0e-4
        dtype: bf16
      model:
        name_or_path: "F:/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/ComfyUI/models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors"
        arch: "qwen_image_2"
        quantize: true
        low_vram: true                    # 16 GB 必开
        # 若走方案 A，再把 extras_name_or_path 指向下载好的 Qwen-Image-2.1 目录
      sample:
        sampler: "flowmatch"
        sample_every: 250
        width: 1024
        height: 1024
        samples:
          - prompt: "把背景换成纯白，保留商品和它的投影"
            ctrl_img: "E:/ai-train/datasets/qwen_<名字>/control/001.png"
        neg: ""
        seed: 42
        guidance_scale: 3
        sample_steps: 20
meta:
  name: "[name]"
```

> 上面这份是**按工具代码结构写的模板**，还没在你机器上跑过（缺素材）。第一次跑我会先拿 5 对图、50 步做“空转测试”，确认能加载、能出样张，再放全量。

---

## 5. 时间与显存预期（诚实版）

| 阶段 | 预计 |
|---|---|
| 加载模型（第一次） | 3~8 分钟（19 GB 权重从磁盘读） |
| 每 100 步 | **未实测**，乐观 10~20 分钟；低显存模式会更慢 |
| 完整 1500 步 | 数小时级（**建议睡前跑**） |
| 显存峰值 | 目标 ≤ 15 GB（`low_vram` + 梯度检查点 + uint4/fp8 量化） |
| 内存 | 32 GB 够（图片模型比视频小得多） |

**中途可以停**：每 250 步存一次，`save_every` 的中间档都能拿来用（挑效果最好的那档，不一定要最后一档）。

---

## 6. 产出与使用

- 产物：`E:\ai-train\runs\qwen_<名字>\*.safetensors`
- 用法：拷到 `F:\…\ComfyUI\models\loras\`（或 D 盘实验区），在 ComfyUI 里用 **LoraLoader** 挂到 Qwen-Image-2.1 的模型线上，权重 0.6~0.9
- 关掉 LoRA = 回到原模型，原权重没被改过，**零风险**

---

## 7. 常见坑（我先替你踩）

| 现象 | 原因 | 处理 |
|---|---|---|
| 样张全是噪点/糊 | 步数太少或 lr 太大 | 降 lr 到 5e-5，加步数 |
| 学成了“只认那 30 张图” | 过拟合 | 减少步数、把 `linear` 降到 8、加 caption dropout |
| 训练崩在加载 | 缺 diffusers 目录（就是上面第 2 节的坑） | 走方案 A 下 30 GB |
| OOM | 8189 没停 / 分辨率给太大 | 停 8189，分辨率降到 768 |
| 换了模型但效果没变 | LoRA 没挂上或权重太低 | 检查 LoraLoader 连线、权重调到 1.0 试 |
