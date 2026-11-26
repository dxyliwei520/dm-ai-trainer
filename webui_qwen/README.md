# webui_qwen —— Qwen-Image-2.1 图片编辑 LoRA 训练台（网页版）

本地网页训练台：给一张图 + 一句指令 → 训练出「图生图编辑」LoRA。

```
双击  E:\ai-train\_START_WEBUI_QWEN.bat
浏览器 http://127.0.0.1:8198
```

- 📖 **完整使用手册**：[`../docs/07-Qwen训练台-使用说明.md`](../docs/07-Qwen训练台-使用说明.md)
- 🔧 技术细节与踩坑：[`../docs/06-Qwen图片编辑训练台.md`](../docs/06-Qwen图片编辑训练台.md)
- 📄 一页纸速查（纯文本）：[`../训练要求说明.txt`](../训练要求说明.txt)

## 组成

| 文件 | 作用 |
|---|---|
| `app.py` | FastAPI 后端 + 全部接口（单文件，无构建步骤） |
| `ui.html` | 单页前端（原生 JS，零依赖） |

## 它做什么

| 环节 | 说明 |
|---|---|
| 项目 | 每个项目 = `datasets\qwen_<名字>\{control,target}` |
| 素材 | 上传/配对（名字不同也能配）/批量写 caption/校验/删除（进回收站） |
| 训练 | 调 ai-toolkit `qwen_image_2` 扩展；可选底座、步数、rank、lr、分辨率、存档间隔 |
| 护栏 | 单实例锁、外部训练进程检测、训练前停 ComfyUI、训练后自动清理 |
| 产物 | `.safetensors` 一键装进 ComfyUI；样张预览；单个/批量清理（进回收站） |

## 依赖

- Python 3.12（系统 Python 或 `env-aitk` 均可，只需 fastapi + uvicorn）
- 训练用 `E:\ai-train\env-aitk\Scripts\python.exe`
- 底座 / TE / VAE 从 ComfyUI 的 `models\` 目录读取（不额外下载）
- 端口 **8198**，只绑 `127.0.0.1`

## 接口一览

```
GET  /                          页面
GET  /api/projects              项目列表
POST /api/project               新建项目（顺带建 runs 目录）
GET  /api/project/{name}        项目的配对明细
GET  /api/paths/{name}          素材/输出/配方三个路径
POST /api/open_folder           打开某目录（dataset/control/target/run/samples/recipe/dit/lora）
POST /api/upload                上传 control / target
POST /api/caption               写单条 caption
POST /api/caption_all           批量写（只补空）
POST /api/pair                  把两个孤立文件配成一对（改名对齐）
POST /api/check                 素材校验
GET  /api/models                扫描可用底座 / TE / VAE
POST /api/train                 开始训练
POST /api/stop                  停止训练
GET  /api/status                进度 / 释放报告 / 产物
GET  /api/log                   日志尾巴
POST /api/delete_image          删单张
POST /api/delete_pair           删整对
POST /api/clear_images          一键清空素材（可只清一边）
POST /api/clear_trash           清空素材回收站
POST /api/delete_artifact       删单个产物
POST /api/clear_artifacts       一键清理产物
POST /api/clear_runs_trash      清空产物回收站
POST /api/open_artifact_folder  打开产物所在文件夹（并选中该文件）
POST /api/install               复制 LoRA 到 ComfyUI\models\loras
POST /api/restore_comfy         训练后恢复 ComfyUI(8189)
GET  /api/file                  读素材/样张图片
```

## 许可

私有仓库，仅供本项目使用。
