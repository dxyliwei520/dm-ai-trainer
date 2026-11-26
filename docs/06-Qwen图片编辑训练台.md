# 教程 06 · Qwen-Image-2.1 图片编辑 LoRA 训练台（网页版，端口 8198）

> 写于 2026-11-09 深夜，**当天实测跑通**（冒烟 40 步 + 网页端 20 步，均出 LoRA 与样张）。
> 与 H3 视频台（8199）**完全隔离**：不同端口、不同数据集目录、不同配方、不同运行目录，互不影响。

---

## 0. 一句话

```
双击 E:\ai-train\_START_WEBUI_QWEN.bat → 浏览器开 http://127.0.0.1:8198
  ① 建项目（qwen_xxx）
  ② 上传 control（要改的图）+ target（改完的样子），同名配对
  ③ 每对填一条「改图指令」（caption）
  ④ 点「校验素材」→ 绿了就点「开始训练」
  ⑤ 看进度/样张 → 点「装进 ComfyUI」把 .safetensors 拷到 models\loras
```

---

## 1. 为什么只能用 ai-toolkit（不是 musubi）

| 工具 | 支持的架构 | 结论 |
|---|---|---|
| **ai-toolkit**（`src\ai-toolkit`，env `env-aitk`） | 自带 `qwen_image_2` 扩展 = **Qwen-Image-2.1**，且 `has_multiple_control_images=True` | ✅ **唯一可用** |
| musubi-tuner（H3 用的那套，env `env-musubi`） | 只到 Qwen-Image / Edit / Edit-2509 / Edit-2511 / Layered，**没有 2.1** | ❌ 不能用 |

`qwen_image_2` 扩展的关键点（源码实证）：
- `encode_control_in_text_embeddings = True`：参考图走文本编码器的 vision token
- `has_multiple_control_images = True`：数据集支持 `control_path` / `control_path_1..3`（**最多 4 张参考图**）
- 文本编码器写死了 ComfyUI 单文件映射 `text_encoders/qwen3vl_8b_int8_convrot.safetensors` 并自动改写 comfy 的 key 前缀

**所以「多图编辑」不用改代码**，只是数据集要多给几路 control。

---

## 2. 不用下 30 GB（实测推翻文档 01 §2 的担心）

文档 01 曾担心要下 `Qwen/Qwen-Image-2.1` 全量 diffusers（约 30 GB）。**实测不需要**：
F 盘 ComfyUI 现成的三件套直接被吃：

| 组件 | 文件 |
|---|---|
| DiT | `F:\…\ComfyUI\models\diffusion_models\qwen_image_2.1_int8_convrot.safetensors`（6.76 GB） |
| 文本编码器 | `F:\…\ComfyUI\models\text_encoders\qwen3vl_8b_int8_convrot.safetensors`（8.71 GB） |
| VAE | `F:\…\ComfyUI\models\vae\qwen_image_2.1_vae_bf16.safetensors`（0.63 GB） |

配方里这样写即可（`MODELS_PATH` 指向 ComfyUI 的 models 目录）：

```yaml
model:
  name_or_path: "F:/…/ComfyUI/models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors"
  extras_name_or_path: "Qwen/Qwen-Image-2.1"   # 只用来取「配置」和 processor（已全部预下载进缓存）
  arch: "qwen_image_2"
  quantize: true
  low_vram: true
```

⚠️ `extras_name_or_path` **必须写仓库名**（不能写本地目录），否则文本编码器的权重解析会绕过 ComfyUI 单文件、去找 diffusers 分片，然后失败。

---

## 3. 素材规范（这一步决定成败）

```
E:\ai-train\datasets\qwen_<名字>\
  control\  001.png 002.png ...   ← 要改的图（参考图）
  target\   001.png 002.png ...   ← 改完应该长什么样
  target\   001.txt 002.txt ...   ← caption，必须与 target 图片同名、同目录
```

| 项 | 要求 |
|---|---|
| 配对 | `control/001.png` ↔ `target/001.png` **同名** |
| caption 位置 | **`target/001.txt`**（⚠️ **不是** `captions/` 子目录——文档 01 那种写法是错的，ai-toolkit 只在图片旁边找同名 txt，或用同目录 `default.txt` 兜底） |
| 数量 | ≥30 对（网页会提示），50~100 对最佳 |
| 尺寸 | 1024 边长附近；网页接受任意尺寸，会自动裁整到 16 的倍数 |
| 文件名 | 纯 ASCII，无空格（网页上传会自动净化成 `[A-Za-z0-9._-]`） |
| 内容 | 每对之间「改动类型」要一致，模型学的是共同规律 |

caption 写法：描述**动作 + 目标**，同一套词反复用。例：
`把背景换成纯白，保留商品和它的投影，整体提亮，商业摄影风格`

---

## 4. 实测数据（RTX 5070 Ti 16 GB / 31.8 GB RAM）

| 项 | 结果 |
|---|---|
| 模型加载（DiT+TE+VAE，首次） | **约 2~4 分钟**（含 CPU 反量化；内存峰值很高，跑完回落） |
| 512×512 / batch 1 | **2.4 秒/步** |
| 40 步冒烟 | 约 3 分钟，产出 `qwen_smoke.safetensors`（**76 MB**）+ 样张 2 张 |
| 显存峰值 | **约 12 GB**（`low_vram` + 梯度检查点） |
| 内存 | 加载期一度只剩 0.1 GB 空闲，**但没崩**（提交峰值 41 GB/85 GB，靠 pagefile） |
| 文本编码器 | 缓存 5 条 caption 用时 ~6 秒，缓存完立刻卸载 |
| 续训 | 目录里有 `.safetensors` 会自动 resume（会读 metadata 里的 step） |

**产物位置**：`E:\ai-train\runs\<项目>\<项目>\*.safetensors`（注意是两层同名目录）；样张在 `…\samples\*.jpg`。
点网页上的「装进 ComfyUI」会复制到 `F:\…\ComfyUI\models\loras\`。

---

## 5. 本次踩的 4 个坑（都已绕过）

### 坑 1：huggingface_hub 走本机代理会卡死在 TLS 握手
**现象**：训练卡在 `Loading text encoder` 十几分钟，GPU 0%、CPU 不涨、进程有一条到 `127.0.0.1:51081`（系统代理）的 ESTABLISHED。
**根因**：huggingface_hub 的 httpx 客户端不走 urllib 的代理路径；直连 hf-mirror 的 TLS 又被卡；两者都挂 → 无限等。
**修法**（已固化在网页里）：
1. 先把要用的**小配置文件**用 curl 直接灌进 HF 缓存（curl 直连是通的）
2. 训练进程里设 **`HF_HUB_OFFLINE=1`**（缓存已全，不需要联网）
3. 顺手把 `HTTP_PROXY/HTTPS_PROXY` 从子进程环境里删掉

**判据**：日志能过 `Loading text encoder` 并且看到 `Caching text embeddings to disk: 5/5`。

### 坑 2：`Qwen/Qwen-Image-2.1` 仓库里**没有根 config.json**
**现象**：`OSError: Qwen/Qwen-Image-2.1 does not appear to have a file named config.json`（第 1 次跑就是这样挂的）。
**根因**：DiT 单文件加载时要读配置；diffusers 取的是仓库对应目录下的 `config.json`（`transformer/config.json` 有），但 tokenizer 侧还会找 `processor/config.json` —— 仓库里没有这个文件。
**修法**：把 `text_encoder/config.json` 复制成 `processor/config.json` 放进缓存快照目录。

### 坑 3：`tokenizer.json` 会下成残缺文件
**现象**：`json.decoder.JSONDecodeError: Expecting value: line 432048`（下到 7.28 MB 就断了，真身 11.4 MB）。
**根因**：镜像限速/超时，`curl` 返回 200 但内容是截断的；第二次直接 429 限流。
**修法**：带重试 + 校验大小（>7 MB）+ 解析 JSON 验证，直到通过为止。

### 坑 4：进度条解析会把「加载阶段」当成训练步数
**现象**：网页显示 `32/32 步`，实际还在加载。
**根因**：`Quantizing blocks: 100%|…| 32/32 [5.45it/s]` 与训练进度条格式相同。
**修法**：只认同时含 `loss` 或 `s/it` 的行（训练条一定带这两个字段）。

### 附带：断点续训会「瞬间跳过」
`runs\<项目>\<项目>\*.safetensors` 里有 metadata step=40；这时把步数设成 20 → 程序认为已经练到 40，直接存盘结束。**改小步数前先删检查点**。

---

## 6. 与 H3 台的关系

| | H3 视频台 | Qwen 图片台 |
|---|---|---|
| 端口 | 8199 | **8198** |
| 启动 | `_START_WEBUI.bat` | **`_START_WEBUI_QWEN.bat`** |
| 后端 | musubi-tuner（env-musubi） | **ai-toolkit（env-aitk）** |
| 数据 | `datasets\h3_*`（videos + captions） | **`datasets\qwen_*`（control + target + 同名 txt）** |
| 显存 | 训练前停 8189 | **同样停 8189（网页会自动停并跑完恢复）** |

两个台子可以同时开着（互不干扰），但**不要同时训练**——只有一块 16 GB 显卡。

---

## 7. 文件清单

| 用途 | 路径 |
|---|---|
| 网页主程序 | `E:\ai-train\webui_qwen\app.py`（FastAPI + 内嵌前端引用 `ui.html`） |
| 前端 | `E:\ai-train\webui_qwen\ui.html` |
| 启动 | `E:\ai-train\_START_WEBUI_QWEN.bat`（纯 ASCII） |
| 冒烟素材生成 | `E:\ai-train\scripts\_make_qwen_smoke.py` |
| 诊断（打印 Python 栈） | `E:\ai-train\scripts\_run_with_faulthandler.py` |
| 配方 | `E:\ai-train\recipes\qwen_<项目>.yaml`（网页自动生成） |
| 产出 | `E:\ai-train\runs\<项目>\<项目>\*.safetensors` |
| HF 缓存（配置类小文件） | `C:\Users\liwei\.cache\huggingface\hub\models--Qwen--Qwen-Image-2.1\snapshots\<sha>\` |

**回滚**：删 `webui_qwen\` 即可（训练台是独立目录，不影响 H3 台和 ComfyUI）；产物在 `runs\` 里，随时可删。
---

## 8. 多参考图（4 路）落地方式（2026-11-10 修订：单目录 + 后缀）

**用户的素材布局（唯一需要维护的）**：

``````
datasets/qwen_xxx/
  control/001_a.png    ← 图1
  control/001_b.png    ← 图2
  control/001_c.png    ← 图3（可选）
  control/001_d.png    ← 图4（可选）
  target/001.png 001.txt
``````

**训练器要求的是「每路一个目录 + 文件与 target 同名」**，所以 `build_recipe()` 在写配方前自动摊平：

``````
_slots/1/001.png   ← 硬链接自 control/001_a.png
_slots/2/001.png   ← 硬链接自 control/001_b.png
``````

生成的配方：

``````yaml
datasets:
  - folder_path: ".../target"
    control_path:                  # 多路 = 列表，单路 = 字符串
      - ".../datasets/qwen_xxx/_slots/1"
      - ".../datasets/qwen_xxx/_slots/2"
sample:
  samples:
    - prompt: "图1的人拿着图2的苹果在吃"
      ctrl_img:   ".../_slots/1/001.png"
      ctrl_img_1: ".../_slots/2/001.png"
``````

**实现要点（改代码时别破坏）**：

1. `_split_slot()`：只把 `_a/_b/_c/_d` 当槽位标记；**不要**把 `_1`/`_2` 当槽位（真实文件名里 `img_2.png` 很常见，会误拆）
2. 摊平时**保留参考图自己的扩展名**（target 是 jpg、参考图是 png 时不能张冠李戴），训练器按「stem + 各种图片扩展名」查找
3. `_slots/` 每次点训练都会重建（先删后建），不要把它当数据目录
4. 质检会检查：每对参考图张数是否一致、captions 是否带「图1/图2」指代、参考图之间是否重复、target 是否等于某张参考图