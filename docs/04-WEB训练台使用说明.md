# H3 LoRA 训练台 · 使用说明（网页版）

> 最后更新：2026-11-09
> 入口：`http://127.0.0.1:8199`（只监听本机 loopback，端口已确认不冲突）
> 启动：双击 `E:\ai-train\_START_WEBUI.bat`

---

## 一、整体结构

```
E:\ai-train\
├─ webui\app.py              ← 网页服务（FastAPI + 内嵌前端，单文件）
├─ scripts\
│   ├─ make_h3_dataset.py    ← 自动切段脚本
│   └─ _patch_*.py           ← 历史补丁（可留作参考）
├─ docs\                     ← 本目录
├─ datasets\<项目名>\         ← 素材与训练片段
│   ├─ raw\                  原始视频（拖进来就行）
│   ├─ videos\               训练片段 NN.mp4 + NN.txt（caption）
│   └─ captions\             原始 caption（可选）
├─ recipes\<项目名>-dataset.toml   ← 自动生成的数据集配置
├─ runs\<项目名>\             ← 训练产物 + 日志 + 缓存
├─ models\h3_configs\        ← ⭐ 本地 processor/文本编码器配置（防网络卡死）
├─ env-aitk\                 ai-toolkit 环境（跑网页服务）
├─ env-musubi\               训练环境（跑 musubi）
└─ src\                      ai-toolkit / musubi-tuner 源码
```

---

## 二、完整使用流程

```
① 双击 _START_WEBUI.bat  → 自动开浏览器
② 左栏「新项目名」→ 建
③ 中栏「上传素材」→ 拖视频进去（可多选）
④ 每个素材后面点「切这段」→ 自动切成训练片段
   （或点全局「切段」→ 只切还没切过的）
⑤ 每段卡片里填 caption（中文即可，同一项目统一语言）
⑥ 右栏选模型 + 设参数 → 点「开始训练」
⑦ 看进度条/日志 → 训练完点「装进 ComfyUI」
```

### 右栏参数说明

| 参数 | 默认 | 说明 |
|---|---|---|
| **主干模型** | `minimax_h3_fl2va_pruned_int8` | 三个可选，见 §四 |
| **文本编码器** | `qwen3vl_32b_minimax_h3_nvfp4_awq` | 14.61 GB，最省 |
| **视频/音频 VAE** | fp16 / fp32 | — |
| **步数** | **1000** | 15 段素材建议 800~1500 |
| **Rank** | 16 | 人物 16 / 动作 32 |
| **帧数** | **22** | 见 §五，别用 56 |
| **缩放方式** | 模糊填充 | 不裁切内容 |
| **宽 × 高** | 448×256 | 32 的倍数 |
| **块换出** | 28 | 16GB 显存用 28~32 |
| **学习率** | 1e-4 | 不收敛降到 5e-5 |
| **停 ComfyUI** | 勾上 | 否则抢显存 OOM |

---

## 三、按钮与接口

| 按钮 | 作用 |
|---|---|
| 建 | 新建项目（自动建 raw/videos/captions） |
| 上传区（拖/点） | 上传视频到 raw |
| raw 后的 `×` | 删除该素材（不删已生成的片段） |
| raw 后的「切这段」/「重切」 | **只切这个素材**（不会重切别的） |
| **清空全部标注** | 把该功能所有 caption 清成 `-`（片段保留） |
| **清空全部片段** | 删掉所有 mp4+txt+latent 缓存（**raw 保留**，可重切） |
| 片段卡片「删除」 | 删这一段（mp4 + txt） |
| 切段 | 只切**还没切过**的素材 |
| 开始训练 / 训练中… / 停止 | 训练控制（训练中切段与再次训练都禁用） |
| 装进 ComfyUI | 把 LoRA 复制到 `F:\…\ComfyUI\models\loras\` |

**后端接口**

```
GET    /api/projects            项目列表
POST   /api/project             新建项目
GET    /api/project/{name}      项目详情（片段/raw/LoRA/切段状态）
PUT    /api/upload              上传素材（裸 body）
DELETE /api/raw                 删除素材
POST   /api/cut                 切段（only / force / mode）
GET    /api/clip                预览片段（video/mp4）
DELETE /api/clip                删除片段
POST   /api/captions/clear      清空全部标注
POST   /api/clips/clear         清空全部片段
PUT    /api/caption             保存单条 caption
GET    /api/models              可选模型列表
POST   /api/train               开始训练
GET    /api/status              训练状态 + 进度
POST   /api/stop                停止训练
GET    /api/log                 日志尾部
POST   /api/install             安装 LoRA 到 ComfyUI
```

---

## 四、三个主干模型的区别（重要）

| 模型 | 是什么 | 步数 | 训练建议 |
|---|---|---|---|
| `minimax_h3_fl2va_pruned_int8` | **官方基础版**（首/尾帧→视频+音频） | 20~30 | ✅ **训练首选，最稳** |
| `fastvideo_fasth3_8step_v2_pruned_int8` | 社区**蒸馏加速版** | 8 | ⚠️ 可训但特征弱；**你站点工作流用的是它** |
| `minimax_h3_ref2va_pruned_int8` | 官方**多参考图版** | 20~30 | ⚠️ 数据集要加 `control/` 参考图，`--task ref2va` |

**LoRA 认底座**：给哪个训的只能挂哪个，混用会怪（不损坏）。

---

## 五、帧数与速度（实测，16GB RTX 5070 Ti）

| 帧数 | 每段时长 | 每步耗时 | 30 秒素材切成几段 |
|---|---|---|---|
| **22** | 0.92s | **4 秒** ✅ | 32 段 |
| 39 | 1.62s | 未测（约 8~10 秒） | 18 段 |
| **56** | 2.33s | **253 秒** ❌ | 12 段 |
| 73 | 3.04s | 未测 | 9 段 |

**结论：训练一律用 22 帧。** 56 帧时注意力显存爆掉，块换出疯狂搬运，慢 60 倍。

**时间预算**（22 帧 / 448×256 / 块换出 28）：

| 素材 | 步数 | 训练 | +缓存 | 合计 |
|---|---|---|---|---|
| 5 段 | 600 | 40 分 | 1 分 | 41 分 |
| **13~15 段** | **1000** | **67 分** | 3 分 | **70 分** |
| 20 段 | 1200 | 80 分 | 5 分 | 85 分 |

---

## 六、caption 写法

```
主体 + 在做什么 + 镜头/景别 + 光线/氛围 + 声音
```

- **同一项目统一一种语言**（中文完全可用，H3 的文本编码器是 Qwen3-VL，中文原生）
- 每段都必须有描述，别用 `-` 占位
- 别写"好看/高级"这种抽象词
- 训练用中文标注 → 生成时也用中文提示词，命中率更高

---

## 七、输出与使用

```
E:\ai-train\runs\<项目名>\
    <项目名>_r16.safetensors              ← 最终 LoRA（约 284 MB）
    <项目名>_r16-step00000500.safetensors ← 每 100 步一个中间存档
    web_train.log                          ← 完整训练日志
    cache\                                 ← latent + 文本缓存
```

**挑存档的经验**：总步数 40~50% 位置的那个通常最"听话"（特征学会、还没僵化）。

**权重建议**：
- 单人物：0.7~0.85
- 动作/风格：0.5~0.65
- 两个叠加：各 0.5~0.6

---

## 八、常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 点开始训练后 0% GPU、日志停在 `Loading Qwen3-VL processor` | 旧版从 HF 拉配置卡住 | **已修复**（本地配置，见踩坑文档 §3） |
| 再点一次训练被拒 `已有训练在跑` | 防重复保护 | 正常，点「停止」后再训 |
| 刷新网页后按钮锁死 | 正常（训练继续，防重复） | 要重训先点「停止」 |
| `No training items found in the dataset` | 改了帧数没重新缓存 | 重跑切段/训练（会自动重缓存） |
| `target duration 2.3s is outside 5-15s range` | 片段太短 | 缓存命令带 `--allow_experimental_duration`（网页已带） |
| 训练极慢（20 分钟/步） | 帧数或分辨率太高 | 降到 22 帧 / 448×256 |

---

## 九、命令行版本（不想用网页时）

```bat
E:\ai-train\_MAKE_DATASET.bat      切段（改 PROJECT/FRAMES/WIDTH/HEIGHT/MODE）
E:\ai-train\_ONE_CLICK_TRAIN.bat   一键训练（改 PROJECT/STEPS/RANK/FRAMES）
```

两者都**纯 ASCII**，可双击，日志写到 `runs\<项目名>\`。
