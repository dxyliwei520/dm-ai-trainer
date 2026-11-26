"""Qwen-Image-2.1 图片编辑 LoRA 训练台（本地） http://127.0.0.1:8198

和 H3 视频台（8199）完全隔离：不同端口、不同数据集、不同配方、不同运行目录。
训练后端 = ai-toolkit（env-aitk），因为只有它的 qwen_image_2 扩展支持 2.1 的多参考图编辑。

数据约定（与 ai-toolkit 一致，文档 01 里写的 captions/ 子目录是错的）：
    datasets/<project>/control/001.png   参考图（要改的图）
    datasets/<project>/target/001.png    目标图
    datasets/<project>/target/001.txt    caption（必须与 target 同名同目录）
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

ROOT = Path(os.environ.get("AI_TRAIN_ROOT", r"E:\ai-train"))
DATASETS = ROOT / "datasets"
RECIPES = ROOT / "recipes"
RUNS = ROOT / "runs"
PY_TRAIN = ROOT / "env-aitk" / "Scripts" / "python.exe"
TOOLKIT = ROOT / "src" / "ai-toolkit"

COMFY = Path(r"F:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI")
MODELS = COMFY / "models"
DIT = MODELS / "diffusion_models" / "qwen_image_2.1_int8_convrot.safetensors"
EXTRA_REPO = "Qwen/Qwen-Image-2.1"
LORA_DIR = MODELS / "loras"
COMFY_TASK = "ComfyUI-D-8189"
PORT = int(os.environ.get("QWEN_TRAIN_PORT", "8198"))

# 底座模型扫描范围（想加自己的权重，丢进 E:\ai-train\models\dit 就会出现）
DIT_ROOTS = [
    MODELS / "diffusion_models",
    Path(r"D:\ComfyUI_experimental\ComfyUI_windows_portable\ComfyUI\models\diffusion_models"),
    ROOT / "models" / "dit",
    Path(r"F:\ai-models"),
]
TE_ROOTS = [MODELS / "text_encoders", ROOT / "models" / "text_encoder"]
VAE_ROOTS = [MODELS / "vae", ROOT / "models" / "vae"]

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
# 多参考图用「同目录 + 后缀」约定：control/001_a.png(图1) 001_b.png(图2) …
REF_SLOTS = ["a", "b", "c", "d"]
SLOT_LABEL = {"a": "图1", "b": "图2", "c": "图3", "d": "图4"}
# 训练器按"与 target 同名"在控制目录里找图，所以训练前把 a/b/c/d 摊成每路一个目录
STAGE_ROOTNAME = "_slots"
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
PROJECT_NAME = re.compile(r"^qwen_[A-Za-z0-9_]{1,40}$")

STATE = {
    "running": False,
    "proc": None,
    "project": None,
    "log": None,
    "started": 0.0,
    "comfy_stopped": False,
    "error": "",
    "opts": {},
}
LOCK = threading.Lock()

app = FastAPI(title="Qwen Image Edit LoRA Trainer", docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def no_store_api(request, call_next):
    """接口响应一律不缓存：否则浏览器可能缓存 /api/status，页面状态就"冻结"了。"""
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


# ----------------------------------------------------------------------------- helpers
def project_dir(name: str) -> Path:
    if not PROJECT_NAME.match(name or ""):
        raise HTTPException(400, "项目名必须是 qwen_ + 字母数字下划线（<=40）")
    return DATASETS / name


def _run_dir(name: str) -> Path:
    return RUNS / name


def project_paths(name: str) -> dict:
    """项目相关的所有目录，前端直接显示（省得用户猜输出在哪）。"""
    return {
        "dataset": str(DATASETS / name),
        "control": str(DATASETS / name / "control"),
        "target": str(DATASETS / name / "target"),
        "run": str(RUNS / name),
        "recipe": str(RECIPES / f"{name}.yaml"),
    }


def _scan_weights(roots, keyword: Optional[str] = None) -> List[dict]:
    """扫多个目录下的 *.safetensors，同名去重。"""
    found: dict = {}
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.safetensors")):
            if p.name in found:
                continue
            if keyword and keyword.lower() not in p.name.lower():
                continue
            found[p.name] = {
                "name": p.name,
                "path": str(p),
                "size_gb": round(p.stat().st_size / 1073741824, 2),
            }
    return sorted(found.values(), key=lambda x: x["name"])


def scan_models() -> dict:
    dit = _scan_weights(DIT_ROOTS)
    for m in dit:
        low = m["name"].lower()
        # 只认 image 2.1 系列（qwen_image_2.1_* / 被改名成 wen_image_2.1_* 的那种也算）
        m["compat"] = ("2.1" in low) and ("image" in low)
        m["is_default"] = m["path"] == str(DIT)
    dit.sort(key=lambda x: (not x["compat"], not x["is_default"], x["name"]))
    te = _scan_weights(TE_ROOTS, keyword="qwen")
    vae = _scan_weights(VAE_ROOTS, keyword="qwen")
    return {
        "dit": dit,
        "text_encoder": te,
        "vae": vae,
        "default_dit": str(DIT),
        "roots": [str(r) for r in DIT_ROOTS],
    }


def resolve_dit(name_or_path: str) -> Path:
    """把前端传来的模型名/路径解析成实际文件（只允许扫描范围内的文件）。"""
    if not name_or_path:
        return DIT
    for m in scan_models()["dit"]:
        if name_or_path in (m["path"], m["name"]):
            p = Path(m["path"])
            if not m["compat"]:
                raise HTTPException(400, "这个底座不是 Qwen-Image-2.1，训出来的 LoRA 用不了（只支持 2.1 架构）")
            if not p.exists():
                raise HTTPException(400, "模型文件不存在")
            return p
    raise HTTPException(400, "没找到这个模型（先用「刷新模型」重新扫描）")


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "img"
    return stem[:60]


def _images(folder: Path) -> dict:
    out = {}
    if folder.is_dir():
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                out[_safe_stem(p.name)] = p
    return out


def _caption_for(target_img: Path) -> str:
    cap = target_img.with_suffix(".txt")
    if cap.exists():
        try:
            return cap.read_text(encoding="utf-8").replace("\ufeff", "").strip()
        except Exception:
            return ""
    return ""


def _image_size(path: Path):
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def _md5(path: Path) -> str:
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except Exception:
        return ""


def _split_slot(stem: str):
    """把 001_a 拆成 ('001', 'a')；没有后缀就当第 1 张。"""
    m = re.match(r"^(?P<base>.+)_(?P<slot>[a-dA-D])$", stem)
    if m:
        return m.group("base"), m.group("slot").lower()
    return stem, "a"


def scan_project(name: str) -> dict:
    base = project_dir(name)
    control_dir = base / "control"
    target = _images(base / "target")

    # control 里按「去掉 _a/_b 后的名字」分组，组内按 a→b→c→d 排序
    groups: dict = {}
    if control_dir.is_dir():
        for p in sorted(control_dir.iterdir()):
            if not (p.is_file() and p.suffix.lower() in IMG_EXT):
                continue
            stem = _safe_stem(p.name)
            b, slot = _split_slot(stem)
            groups.setdefault(b, {})[slot] = p

    stems = sorted(set(target) | set(groups))
    pairs = []
    for stem in stems:
        slot_map = groups.get(stem, {})
        refs = [
            {"slot": slot, "label": SLOT_LABEL[slot], "stem": _safe_stem(slot_map[slot].name),
             "name": slot_map[slot].name, "size": _image_size(slot_map[slot])}
            for slot in REF_SLOTS if slot in slot_map
        ]
        tgt = target.get(stem)
        pairs.append(
            {
                "stem": stem,
                "has_control": bool(refs),
                "has_target": bool(tgt),
                "ref_count": len(refs),
                "refs": refs,
                "control": refs[0]["name"] if refs else None,
                "control_size": refs[0]["size"] if refs else None,
                "target": tgt.name if tgt else None,
                "target_size": _image_size(tgt) if tgt else None,
                "caption": (_caption_for(tgt) if tgt else ""),
            }
        )
    ready = [p for p in pairs if p["has_control"] and p["has_target"]]
    return {
        "project": name,
        "pairs": pairs,
        "count": len(pairs),
        "ready": len(ready),
        "captioned": sum(1 for p in ready if p["caption"]),
        "ref_counts": sorted({p["ref_count"] for p in ready}),
        # 孤立项：有的 target 没参考图 / 有的参考图没 target —— 给「配对」「多图合成」用
        "orphan_targets": [p["stem"] for p in pairs if p["has_target"] and not p["has_control"]],
        "orphan_refs": [
            {"stem": r["stem"], "name": r["name"], "group": p["stem"], "label": r["label"]}
            for p in pairs
            if p["has_control"] and not p["has_target"]
            for r in p["refs"]
        ],
    }


def validation(name: str) -> dict:
    """素材质检：配对 / 参考图张数 / caption 指代 / 重复图 / 尺寸。"""
    info = scan_project(name)
    base = project_dir(name)
    problems, warnings, notes = [], [], []
    ready = [p for p in info["pairs"] if p["has_control"] and p["has_target"]]

    if not ready:
        problems.append("没有可用的配对：control 里的图名（去掉 _a/_b 后）必须与 target 同名，例如 control/001_a.png ↔ target/001.png")
    if len(ready) < 30:
        warnings.append(f"只有 {len(ready)} 对可用素材，建议 >=30 对（少于 20 对基本学不出稳定效果）")
    if len(info["ref_counts"]) > 1:
        warnings.append(f"各对的参考图张数不一致（{info['ref_counts']} 张）；多图训练建议所有对统一张数")

    cap_hint = re.compile(r"图\s*[1-4]|image\s*[1-4]|第\s*[1-4]\s*张", re.I)
    dup = same_as_tgt = 0

    for p in info["pairs"]:
        stem = p["stem"]
        if not p["has_control"]:
            problems.append(f"{stem}: 只有 target，缺参考图")
            continue
        if not p["has_target"]:
            problems.append(f"{stem}: 只有参考图，缺 target（应放 {Path('target') / (stem + '.png')}）")
            continue
        if re.search(r"_[a-dA-D]$", p["stem"]):
            problems.append(f"{stem}: target 名字不该带 _a/_b 后缀，后缀只用于 control 的参考图")
        if not p["caption"]:
            problems.append(f"{stem}: 缺 caption（{Path('target') / (stem + '.txt')}）")
        elif p["ref_count"] >= 2 and not cap_hint.search(p["caption"]):
            warnings.append(f"{stem}: 有 {p['ref_count']} 张参考图，但 caption 里没有「图1/图2」这类指代，模型分不清哪张是哪张")
        if p["caption"]:
            mentioned = [int(x) for x in re.findall(r"图\s*([1-4])", p["caption"])]
            mentioned += [int(x) for x in re.findall(r"image\s*([1-4])", p["caption"], re.I)]
            if mentioned and max(mentioned) > p["ref_count"]:
                problems.append(
                    f"{stem}: caption 提到「图{max(mentioned)}」，但这对只配了 {p['ref_count']} 张参考图"
                    f"（缺 {'、'.join('_' + REF_SLOTS[i] for i in range(p['ref_count'], max(mentioned)))}）"
                )

        hashes = {}
        for r in p["refs"]:
            h = _md5(base / "control" / r["name"])
            if h:
                hashes.setdefault(h, []).append(r["label"])
        for h, labels in hashes.items():
            if len(labels) > 1:
                dup += 1
                problems.append(f"{stem}: 参考图重复（{'、'.join(labels)} 是同一张图）")
        if p["target"]:
            th = _md5(base / "target" / p["target"])
            if th and th in hashes:
                same_as_tgt += 1
                problems.append(f"{stem}: target 与某张参考图完全相同 → 等于教模型「别改」，学不出东西")

        for r in p["refs"]:
            if r["size"]:
                w, h2 = r["size"]
                if w % 16 or h2 % 16:
                    warnings.append(f"{stem}/{r['label']}: {w}x{h2} 不是 16 的倍数，会被裁整")
        if p["target_size"]:
            w, h2 = p["target_size"]
            if w % 16 or h2 % 16:
                warnings.append(f"{stem}/target: {w}x{h2} 不是 16 的倍数，会被裁整")

    if ready:
        notes.append(f"{len(ready)} 对可用，参考图分布 {info['ref_counts']} 张/对（后缀 a=图1, b=图2, c=图3, d=图4）")
        if not dup and not same_as_tgt:
            notes.append("没有重复参考图，也没有「target 等于参考图」的项")

    return {
        "ok": not problems,
        "problems": problems[:40],
        "warnings": warnings[:30],
        "notes": notes,
        "count": info["count"],
        "ready": info["ready"],
        "captioned": info["captioned"],
        "ref_counts": info["ref_counts"],
    }


def _quick_caption(pairs: List[dict]) -> str:
    for p in pairs:
        if p["caption"]:
            return p["caption"][:200]
    return "把这张图按训练集的规律改一下"


def build_recipe(name: str, opts: dict) -> Path:
    info = scan_project(name)
    pairs = [p for p in info["pairs"] if p["has_control"] and p["has_target"]]
    if not pairs:
        raise HTTPException(400, "没有可用配对，先上传 control/target")
    steps = int(opts.get("steps", 1500))
    rank = int(opts.get("rank", 16))
    lr = float(opts.get("lr", 1e-4))
    res = int(opts.get("resolution", 1024))
    batch = int(opts.get("batch_size", 1))
    save_every = int(opts.get("save_every", 250))
    dit_path = Path(opts["dit"]) if opts.get("dit") else DIT
    sample_every = int(opts.get("sample_every", 250))

    tgt_dir = (DATASETS / name / "target").as_posix()
    # 只用"这批配对里真的有文件"的参考图目录（第 1 路在前，顺序 = 图1/图2/…）
    # 把 control/001_a.png、001_b.png … 摊成 _slots/1/001.png、_slots/2/001.png …
    # （ai-toolkit 是在每个控制目录里按「与 target 同名」找图的）
    stage = DATASETS / name / STAGE_ROOTNAME
    if stage.is_dir():
        shutil.rmtree(stage, ignore_errors=True)
    slot_dirs = []
    max_slot = max((p["ref_count"] for p in pairs), default=1)
    for i in range(max_slot):
        slot_dir = stage / str(i + 1)
        slot_dir.mkdir(parents=True, exist_ok=True)
        slot_dirs.append(slot_dir.as_posix())
    staged = {}   # (base_stem, 第几路) -> 摊平后的文件名
    for p in pairs:
        for i, ref in enumerate(p["refs"][:max_slot]):
            src_img = DATASETS / name / "control" / ref["name"]
            if not src_img.exists():
                continue
            # 保留参考图自己的扩展名：训练器是按「stem + 各种图片扩展名」去找的
            fname = f"{p['stem']}{src_img.suffix.lower()}"
            dst = stage / str(i + 1) / fname
            try:
                os.link(src_img, dst)          # 同盘硬链接，不占空间
            except Exception:
                shutil.copy2(src_img, dst)     # 跨盘/不支持就复制
            staged[(p["stem"], i)] = fname
    control_path = slot_dirs[0] if len(slot_dirs) == 1 else slot_dirs
    # training_folder 用 RUNS 根（ai-toolkit 会自动再建一层 config.name），
    # 这样产物直接落在 runs\<项目>\，不会变成 runs\<项目>\<项目>\ 双层。
    run_root = RUNS.as_posix()

    samples = []
    for p in pairs[:4]:
        item = {"prompt": (p["caption"] or _quick_caption(pairs)).replace('"', "'")}
        for i in range(len(p["refs"])):
            fname = staged.get((p["stem"], i))
            if not fname:
                continue
            key = "ctrl_img" if i == 0 else f"ctrl_img_{i}"
            item[key] = f"{slot_dirs[i]}/{fname}"
        if "ctrl_img" in item:
            samples.append(item)

    cfg = {
        "job": "extension",
        "config": {
            "name": name,
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": run_root,
                    "device": "cuda:0",
                    "network": {"type": "lora", "linear": rank, "linear_alpha": rank},
                    "save": {"dtype": "float16", "save_every": save_every, "max_step_saves_to_keep": 4},
                    "datasets": [
                        {
                            "folder_path": tgt_dir,
                            "control_path": control_path,
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.05,
                            "resolution": [res],
                        }
                    ],
                    "train": {
                        "batch_size": batch,
                        "cache_text_embeddings": True,
                        "steps": steps,
                        "gradient_accumulation": 1,
                        "timestep_type": "weighted",
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw8bit",
                        "lr": lr,
                        "dtype": "bf16",
                        "skip_first_sample": bool(opts.get("skip_first_sample")),
                    },
                    "model": {
                        "name_or_path": dit_path.as_posix(),
                        "extras_name_or_path": EXTRA_REPO,
                        "arch": "qwen_image_2",
                        "quantize": True,
                        "low_vram": True,
                    },
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": sample_every,
                        "width": res,
                        "height": res,
                        "samples": samples,
                        "neg": "",
                        "seed": 42,
                        "walk_seed": True,
                        "guidance_scale": 3,
                        "sample_steps": 20,
                    },
                }
            ],
        },
        "meta": {"name": name, "version": "1.0"},
    }

    def _yaml(obj, indent=0):
        pad = "  " * indent
        if isinstance(obj, dict):
            out = []
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    out.append(f"{pad}{k}:")
                    out.append(_yaml(v, indent + 1))
                elif isinstance(v, bool):
                    out.append(f"{pad}{k}: {'true' if v else 'false'}")
                elif isinstance(v, str):
                    out.append(f'{pad}{k}: "{v}"')
                else:
                    out.append(f"{pad}{k}: {v}")
            return "\n".join(out)
        if isinstance(obj, list):
            out = []
            for item in obj:
                if isinstance(item, dict):
                    block = _yaml(item, indent + 1).split("\n")
                    block[0] = pad + "- " + block[0].strip()
                    out.extend(block)
                else:
                    out.append(f"{pad}- {item}")
            return "\n".join(out)
        return f"{pad}{obj}"

    RECIPES.mkdir(parents=True, exist_ok=True)
    path = RECIPES / f"{name}.yaml"
    path.write_text(_yaml(cfg) + "\n", encoding="utf-8")
    return path


def _kill_strays(name: str):
    """杀掉上次残留的同项目训练进程（网页被刷新/崩溃时常见）。"""
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        f"Where-Object {{ $_.CommandLine -like '*{name}.yaml*' }} | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)


def _comfy(action: str):
    subprocess.run(["schtasks", f"/{action}", "/TN", COMFY_TASK], capture_output=True, text=True)


def _external_trainers():
    """网页之外启动的训练进程（比如你从命令行直接跑 run.py）—— 训练前必须避开它们，否则必 OOM。"""
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*run.py*' -and $_.CommandLine -like '*recipes*' } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=20)
        return [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    except Exception:
        return []


def _train_worker(name: str, recipe: Path, log_path: Path, stop_comfy: bool):
    env = os.environ.copy()
    env["MODELS_PATH"] = MODELS.as_posix()
    # 配置文件已全部预下载进 HF 缓存；强制离线可避免 hub 走代理时长时间卡住（实测卡在 TLS 握手）。
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["HF_HUB_ETAG_TIMEOUT"] = "10"
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    try:
        if stop_comfy:
            _comfy("End")
            STATE["comfy_stopped"] = True
            time.sleep(3)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
            proc = subprocess.Popen(
                [str(PY_TRAIN), "run.py", str(recipe)],
                cwd=str(TOOLKIT),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            STATE["proc"] = proc
            proc.wait()
        if STATE.get("manual_stop"):
            STATE["error"] = ""
        else:
            STATE["error"] = ("" if STATE.get("proc") and STATE["proc"].returncode == 0
                              else f"训练进程退出码 {STATE['proc'].returncode if STATE.get('proc') else '?'}")
    except Exception as exc:  # noqa: BLE001
        STATE["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        before_mb = _gpu_used_mb()
        STATE["running"] = False
        STATE["proc"] = None
        # ① 杀掉本次训练残留的 trainer 进程（异常退出时也会留下）
        try:
            _kill_strays(project)
        except Exception:
            pass
        # ② 修剪本进程工作集，把自己占的内存还给系统
        trimmed = _trim_self()
        # ③ 视设置决定是否把 ComfyUI 拉回来（拉回来会重新占 ~2.4 GB 显存）
        keep_stopped = bool(STATE.get("keep_comfy_stopped"))
        if STATE.get("comfy_stopped"):
            if keep_stopped:
                STATE["comfy_left_stopped"] = True
            else:
                _comfy("Run")
            STATE["comfy_stopped"] = False
        # ④ 报告：显存回落到多少 / 是否修剪了内存
        after_mb = _gpu_used_mb()
        STATE["post"] = {
            "gpu_before_mb": before_mb,
            "gpu_after_mb": after_mb,
            "trimmed": trimmed,
            "comfy_left_stopped": bool(STATE.get("comfy_left_stopped")),
            "at": time.strftime("%H:%M:%S"),
        }


def _gpu_used_mb() -> Optional[int]:
    """查一次显存占用（训练结束时调用一次，开销可忽略）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=12, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip().splitlines()[0].strip())
    except Exception:
        pass
    return None


def _trim_self() -> bool:
    """把本进程的工作集交还给系统（Windows 下 EmptyWorkingSet 的等价做法）。"""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentProcess()
        # -1,-1 => 尽量回收工作集
        return bool(kernel32.SetProcessWorkingSetSize(handle, ctypes.c_size_t(-1), ctypes.c_size_t(-1)))
    except Exception:
        return False


def _log_progress(log_path: Optional[Path]) -> dict:
    if not log_path or not log_path.exists():
        return {"step": 0, "total": 0, "loss": None, "last": ""}
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 60000))
            tail = fh.read().decode("utf-8", "replace")
    except Exception:
        return {"step": 0, "total": 0, "loss": None, "last": ""}
    step = total = 0
    loss = None
    # 只认训练进度条（"<项目>: 72%|###| 29/40 [01:23<00:26, 2.41s/it, lr: .., loss: ..]"），
    # 否则会把 "Quantizing blocks: 32/32" 这类加载期进度条错当成训练步数。
    lines = [ln.strip() for ln in tail.replace("\r", "\n").split("\n") if ln.strip()]
    for ln in lines:
        if "loss" not in ln and "s/it" not in ln:
            continue
        m = re.search(r"(\d+)\s*/\s*(\d+)\s*\[", ln)
        if m:
            step, total = int(m.group(1)), int(m.group(2))
        m2 = re.search(r"loss[:= ]+([0-9.]+(?:[eE][-+]?\d+)?)", ln)
        if m2:
            try:
                loss = float(m2.group(1))
            except ValueError:
                pass
    return {"step": step, "total": total, "loss": loss, "last": lines[-1][:200] if lines else ""}


# ----------------------------------------------------------------------------- API
@app.get("/api/projects")
def api_projects():
    items = []
    if DATASETS.is_dir():
        for d in sorted(DATASETS.iterdir()):
            if d.is_dir() and PROJECT_NAME.match(d.name):
                info = scan_project(d.name)
                items.append({"name": d.name, "pairs": info["count"], "ready": info["ready"], "captioned": info["captioned"]})
    return {"projects": items, "running": STATE["running"], "project": STATE["project"]}


@app.post("/api/upload_group")
async def api_upload_group(
    project: str = Form(...),
    caption: str = Form(""),
    base_name: str = Form(""),
    ref1: UploadFile = File(None),
    ref2: UploadFile = File(None),
    ref3: UploadFile = File(None),
    ref4: UploadFile = File(None),
    target: UploadFile = File(None),
):
    """一次上传一组：参考图 1~4（图1..图4）+ 目标图 A + 指令。

    自动命名：control/<base>_a|_b|_c|_d.<ext>，target/<base>.<ext>，target/<base>.txt
    base 默认取「当前最大编号 + 1」（001、002…）。
    """
    base_dir = project_dir(project)
    info = scan_project(project)
    nums = [int(p["stem"]) for p in info["pairs"] if p["stem"].isdigit()]
    auto_base = f"{(max(nums) + 1) if nums else 1:03d}"
    base = (base_name or auto_base).strip()
    if not SAFE_NAME.match(base):
        raise HTTPException(400, "组编号非法（字母数字）")

    refs = [f for f in (ref1, ref2, ref3, ref4) if f is not None and getattr(f, "filename", None)]
    if not refs:
        raise HTTPException(400, "至少上传 1 张参考图（图1）")
    if target is None or not getattr(target, "filename", None):
        raise HTTPException(400, "必须上传目标图 A")

    (base_dir / "control").mkdir(parents=True, exist_ok=True)
    (base_dir / "target").mkdir(parents=True, exist_ok=True)

    saved = []
    for i, f in enumerate(refs):
        ext = Path(f.filename).suffix.lower()
        if ext not in IMG_EXT:
            raise HTTPException(400, f"参考图 {i + 1} 不是图片：{f.filename}")
        out = base_dir / "control" / f"{base}_{REF_SLOTS[i]}{ext}"
        with open(out, "wb") as fh:
            shutil.copyfileobj(f.file, fh)
        saved.append(f"control/{out.name}")

    ext = Path(target.filename).suffix.lower()
    if ext not in IMG_EXT:
        raise HTTPException(400, f"目标图不是图片：{target.filename}")
    tgt_out = base_dir / "target" / f"{base}{ext}"
    with open(tgt_out, "wb") as fh:
        shutil.copyfileobj(target.file, fh)
    saved.append(f"target/{tgt_out.name}")

    if caption.strip():
        tgt_out.with_suffix(".txt").write_text(caption.strip() + "\n", encoding="utf-8")
        saved.append(f"target/{base}.txt")

    next_base = f"{(max(nums) + 2) if nums else 2:03d}"
    return {"ok": True, "base": base, "refs": len(refs), "files": saved, "next_base": next_base}


@app.post("/api/project")
def api_create_project(name: str = Form(...)):
    base = project_dir(name)
    for sub in ("control", "target"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    # 输出目录一并建出来，这样建完项目就能在 runs\ 下看到同名文件夹（不用等训练）
    _run_dir(name).mkdir(parents=True, exist_ok=True)
    return {"ok": True, "project": name, "paths": project_paths(name)}


@app.get("/api/paths/{name}")
def api_paths(name: str):
    return project_paths(name)


@app.post("/api/open_folder")
def api_open_folder(project: str = Form(""), kind: str = Form("run")):
    """在资源管理器里打开某个目录（服务必须跑在你自己的桌面会话里才弹得出来）。"""
    if project:
        project_dir(project)  # 校验项目名
    if kind == "dataset":
        target = DATASETS / project
    elif kind == "control":
        target = DATASETS / project / "control"
    elif kind == "target":
        target = DATASETS / project / "target"
    elif kind == "run":
        target = RUNS / project
    elif kind == "samples":
        target = _first_samples_dir(project) if project else RUNS
    elif kind == "recipe":
        target = RECIPES
    elif kind == "dit":
        target = MODELS / "diffusion_models"
    elif kind == "lora":
        target = LORA_DIR
    else:
        target = None
    if target is None:
        raise HTTPException(400, "kind 必须是 dataset / control / target / run / samples / recipe / dit / lora")
    target.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.Popen(["explorer", str(target)])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"打不开资源管理器：{exc}")
    return {"ok": True, "path": str(target)}


def _first_samples_dir(project: str) -> Path:
    run_dir = _run_dir(project)
    if run_dir.is_dir():
        for p in run_dir.rglob("samples"):
            if p.is_dir():
                return p
    return run_dir / "samples"


@app.post("/api/open_artifact_folder")
def api_open_artifact_folder(rel_path: str = Form(...)):
    """打开某个产物（.safetensors / 样张）所在的文件夹，并尽量选中它。"""
    p = _safe_run_path(rel_path)
    folder = p.parent if p.suffix else p
    if not folder.is_dir():
        folder = _run_dir(p.relative_to(RUNS).parts[0]) if p.relative_to(RUNS).parts else RUNS
    if not folder.is_dir():
        raise HTTPException(404, "文件夹不存在")
    try:
        if p.exists() and p.is_file():
            subprocess.Popen(["explorer", "/select,", str(p)])
        else:
            subprocess.Popen(["explorer", str(folder)])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"打不开资源管理器：{exc}")
    return {"ok": True, "path": str(folder), "selected": str(p) if p.exists() else None}


@app.post("/api/upload")
async def api_upload(project: str = Form(...), kind: str = Form(...), files: List[UploadFile] = File(...),
                     slot: str = Form(""), auto: bool = Form(True)):
    if kind not in ("control", "target"):
        raise HTTPException(400, "kind 必须是 control 或 target")
    if slot and slot not in REF_SLOTS:
        raise HTTPException(400, "slot 必须是 a / b / c / d")
    base = project_dir(project)
    dest = base / kind
    dest.mkdir(parents=True, exist_ok=True)
    saved, skipped = [], []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in IMG_EXT:
            skipped.append(f"{f.filename}: 不是图片")
            continue
        stem = _safe_stem(f.filename or "img")
        if auto:
            # 自动编号：你完全不用管文件名
            _info = scan_project(project)
            open_bases = sorted(p["stem"] for p in _info["pairs"] if p["has_control"] and not p["has_target"])
            nums = [int(p["stem"]) for p in _info["pairs"] if p["stem"].isdigit()]
            next_idx = (max(nums) + 1) if nums else 1
            if kind == "control":
                want = slot or "a"
                if want == "a":
                    base = f"{next_idx:03d}"
                else:
                    cand = [
                        p["stem"] for p in _info["pairs"]
                        if p["has_control"] and not p["has_target"]
                        and want not in {r["slot"] for r in p["refs"]}
                    ]
                    base = sorted(cand)[0] if cand else f"{next_idx:03d}"
                stem = f"{base}_{want}"
            else:  # target
                base = open_bases[0] if open_bases else f"{next_idx:03d}"
                stem = base
        elif kind == "control" and slot:
            # 关闭自动编号时也方便：给文件名补上 _a/_b 后缀
            stem, cur_slot = _split_slot(stem)
            if cur_slot == "a" and not re.search(r"_[a-dA-D]$", _safe_stem(f.filename or "")):
                stem = stem + "_" + slot
        out = dest / f"{stem}{ext}"
        n = 2
        while out.exists():                      # 撞名就顺延
            out = dest / f"{stem}_{n}{ext}"
            n += 1
        with open(out, "wb") as fh:
            shutil.copyfileobj(f.file, fh)
        saved.append(out.name)
    return {"ok": True, "saved": saved, "skipped": skipped}


@app.get("/api/project/{name}")
def api_project(name: str):
    return scan_project(name)


@app.post("/api/caption")
def api_caption(project: str = Form(...), stem: str = Form(...), text: str = Form("")):
    """写 caption。target 图还没传也能先写 —— 直接写 target/<stem>.txt（免去手动建 txt）。"""
    base = project_dir(project)
    if not SAFE_NAME.match(stem or ""):
        raise HTTPException(400, "文件名非法")
    tgt = _images(base / "target").get(stem)
    if tgt:
        path = tgt.with_suffix(".txt")
    else:
        (base / "target").mkdir(parents=True, exist_ok=True)
        path = base / "target" / f"{stem}.txt"
    if not text.strip() and not path.exists():
        raise HTTPException(400, "内容是空的")
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return {"ok": True, "file": path.name, "has_target": bool(tgt)}


@app.post("/api/caption_all")
def api_caption_all(project: str = Form(...), text: str = Form(...)):
    """给所有缺 caption 的配对批量写同一条指令（覆盖式批量更方便时用 /api/caption_each）。"""
    base = project_dir(project)
    n = 0
    for stem, tgt in _images(base / "target").items():
        if not _caption_for(tgt):
            tgt.with_suffix(".txt").write_text(text.strip() + "\n", encoding="utf-8")
            n += 1
    return {"ok": True, "written": n}


@app.post("/api/check")
def api_check(project: str = Form(...)):
    return validation(project)


@app.post("/api/pair")
def api_pair(project: str = Form(...), control_stem: str = Form(...), target_stem: str = Form(...)):
    """把一张只有 control 的图和一个只有 target 的图配成一对（改 target 的文件名去对齐 control）。"""
    for s in (control_stem, target_stem):
        if not SAFE_NAME.match(s or ""):
            raise HTTPException(400, "文件名非法")
    base = project_dir(project)
    ctrl = _images(base / "control").get(control_stem)
    tgt = _images(base / "target").get(target_stem)
    if not ctrl:
        raise HTTPException(404, f"control 里没有 {control_stem}")
    if not tgt:
        raise HTTPException(404, f"target 里没有 {target_stem}")
    if control_stem == target_stem:
        return {"ok": True, "renamed": [], "note": "已经是同名，无需配对"}

    final_tgt = (base / "target") / f"{control_stem}{tgt.suffix}"
    if final_tgt.exists():
        raise HTTPException(409, f"target 里已经有 {final_tgt.name}，先删掉它或换个名字")

    renamed = []
    tgt.rename(final_tgt)
    renamed.append(final_tgt.name)
    old_txt = tgt.with_suffix(".txt")
    if old_txt.exists():
        new_txt = final_tgt.with_suffix(".txt")
        old_txt.rename(new_txt)
        renamed.append(new_txt.name)
    return {"ok": True, "renamed": renamed, "pair": control_stem}


# ----------------------------------------------------------------------------- 删除素材
TRASH_DIRNAME = "_deleted"


def _to_trash(project: str, path: Path) -> None:
    """删除 = 移到项目内 _deleted/<时间戳>/，可手动恢复（不可逆操作留一条后路）。"""
    if not path.exists() or not path.is_file():
        return
    dest = project_dir(project) / TRASH_DIRNAME / time.strftime("%Y%m%d_%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / f"{path.parent.name}_{path.name}"
    n = 1
    while target.exists():
        target = dest / f"{path.parent.name}_{path.stem}_{n}{path.suffix}"
        n += 1
    shutil.move(str(path), str(target))


def _delete_one(project: str, kind: str, stem: str) -> int:
    """删一个 stem 指定 kind 的图（target 顺带删同名 caption）。返回删除的文件数。"""
    if kind not in ("control", "target"):
        raise HTTPException(400, "kind 必须是 control 或 target")
    if not SAFE_NAME.match(stem or ""):
        raise HTTPException(400, "文件名非法")
    base = project_dir(project)
    removed = 0
    img = _images(base / kind).get(stem)
    if img:
        if kind == "target":
            cap = img.with_suffix(".txt")
            if cap.exists():
                _to_trash(project, cap)
                removed += 1
        _to_trash(project, img)
        removed += 1
    return removed


@app.post("/api/delete_image")
def api_delete_image(project: str = Form(...), kind: str = Form(...), stem: str = Form(...)):
    n = _delete_one(project, kind, stem)
    if not n:
        raise HTTPException(404, "没有这个文件")
    return {"ok": True, "removed": n}


def _pair_multi(project: str, target_stem: str, control_stems, common: str) -> dict:
    """把 1 张 target + 最多 4 张参考图统一命名：target/<common> + control/<common>_a|_b|…"""
    base = project_dir(project)
    tgt = _images(base / "target").get(target_stem)
    if not tgt:
        raise HTTPException(404, f"target 里没有 {target_stem}")
    stems = [s for s in control_stems if s]
    if not stems:
        raise HTTPException(400, "至少选 1 张参考图")
    if len(stems) > len(REF_SLOTS):
        raise HTTPException(400, f"最多 {len(REF_SLOTS)} 张参考图")

    common = re.sub(r"_[a-dA-D]$", "", (common or target_stem).strip())
    if not SAFE_NAME.match(common):
        raise HTTPException(400, "统一名非法（只能用字母数字 . _ -）")

    ctrl_map = _images(base / "control")
    ops = []
    new_tgt = base / "target" / f"{common}{tgt.suffix}"
    if new_tgt.resolve() != tgt.resolve():
        if new_tgt.exists():
            raise HTTPException(409, f"target 里已经有 {new_tgt.name}，换个统一名")
        ops.append((tgt, new_tgt))
        cap_old = tgt.with_suffix(".txt")
        if cap_old.exists():
            ops.append((cap_old, new_tgt.with_suffix(".txt")))

    refs_new = []
    for i, stem in enumerate(stems):
        img = ctrl_map.get(stem)
        if not img:
            raise HTTPException(404, f"control 里没有 {stem}")
        new_name = f"{common}_{REF_SLOTS[i]}{img.suffix.lower()}"
        dst = base / "control" / new_name
        if dst.resolve() != img.resolve():
            if dst.exists():
                raise HTTPException(409, f"control 里已经有 {new_name}，换个统一名")
            ops.append((img, dst))
        refs_new.append(new_name)

    for a, b in ops:
        a.rename(b)
    return {"ok": True, "base": common, "refs": refs_new, "target": new_tgt.name,
            "renamed": [b.name for _, b in ops]}


@app.post("/api/pair_multi")
def api_pair_multi(project: str = Form(...), target_stem: str = Form(...),
                   control_stems: str = Form(...), base_name: str = Form("")):
    stems = [s.strip() for s in control_stems.split(",") if s.strip()]
    return _pair_multi(project, target_stem, stems, base_name)


@app.post("/api/pair_auto_all")
def api_pair_auto_all(project: str = Form(...), start_index: int = Form(1)):
    """一键同名：把所有孤立的 target 与参考图按「数量平均」配对，统一命名 001/002/…

    规则：参考图数必须能被 target 数整除（每对张数相同），每对最多 4 张。
    """
    info = scan_project(project)
    targets = list(info.get("orphan_targets") or [])
    refs = [r["stem"] for r in (info.get("orphan_refs") or [])]
    if not targets:
        raise HTTPException(400, "没有待配对的 target（可能都已经配好了）")
    if not refs:
        raise HTTPException(400, "没有待配对的参考图")
    if len(refs) % len(targets) != 0:
        raise HTTPException(
            400,
            f"参考图 {len(refs)} 张、target {len(targets)} 个，没法平均分（每对必须相同张数）；"
            f"请先补齐/删掉多余的文件",
        )
    per = len(refs) // len(targets)
    if per > len(REF_SLOTS):
        raise HTTPException(400, f"每对 {per} 张参考图，超过上限 {len(REF_SLOTS)} 张")

    taken = {p["stem"] for p in info["pairs"]}
    idx = max(1, int(start_index))
    mapping = []
    for i, tgt in enumerate(sorted(targets)):
        while f"{idx:03d}" in taken:
            idx += 1
        common = f"{idx:03d}"
        taken.add(common)
        idx += 1
        group = refs[i * per:(i + 1) * per]
        res = _pair_multi(project, tgt, group, common)
        mapping.append({"target": tgt, "base": common, "refs": res["refs"], "target_new": res["target"]})
    return {"ok": True, "pairs": len(mapping), "per_pair": per, "mapping": mapping}


@app.post("/api/delete_pair")
def api_delete_pair(project: str = Form(...), stem: str = Form(...)):
    if not SAFE_NAME.match(stem or ""):
        raise HTTPException(400, "文件名非法")
    n = _delete_one(project, "control", stem) + _delete_one(project, "target", stem)
    if not n:
        raise HTTPException(404, "没有这一对")
    return {"ok": True, "removed": n}


@app.post("/api/clear_images")
def api_clear_images(project: str = Form(...), scope: str = Form("all")):
    """一键清空：scope = all | control | target（caption 跟着 target 一起走）。"""
    if scope not in ("all", "control", "target"):
        raise HTTPException(400, "scope 必须是 all / control / target")
    base = project_dir(project)
    kinds = ["control", "target"] if scope == "all" else [scope]
    removed = 0
    for kind in kinds:
        for img in list(_images(base / kind).values()):
            if kind == "target":
                cap = img.with_suffix(".txt")
                if cap.exists():
                    _to_trash(project, cap)
                    removed += 1
            _to_trash(project, img)
            removed += 1
    return {"ok": True, "removed": removed, "scope": scope}


@app.post("/api/clear_trash")
def api_clear_trash(project: str = Form(...)):
    """彻底清空 _deleted（释放空间，不可恢复）。"""
    d = project_dir(project) / TRASH_DIRNAME
    n = 0
    if d.is_dir():
        for f in d.rglob("*"):
            if f.is_file():
                n += 1
        shutil.rmtree(d, ignore_errors=True)
    return {"ok": True, "removed": n}


@app.post("/api/train")
def api_train(project: str = Form(...), steps: int = Form(1500), rank: int = Form(16), lr: float = Form(1e-4),
              resolution: int = Form(1024), save_every: int = Form(250), stop_comfy: bool = Form(True),
              dit: str = Form(""), skip_first_sample: bool = Form(False),
              keep_comfy_stopped: bool = Form(False)):
    with LOCK:
        if STATE["running"]:
            raise HTTPException(409, "已有训练在跑")
        v = validation(project)
        if not v["ok"]:
            raise HTTPException(400, "素材校验未通过：" + "；".join(v["problems"][:5]))
        ext = _external_trainers()
        if ext:
            raise HTTPException(409, "检测到已有训练进程在跑（PID " + ", ".join(map(str, ext[:4])) + "），可能是你从命令行启动的；等它结束或先停掉再点开始")
        dit_path = resolve_dit(dit)
        _kill_strays(project)
        recipe = build_recipe(
            project,
            {"steps": steps, "rank": rank, "lr": lr, "resolution": resolution,
             "save_every": save_every, "dit": str(dit_path),
             "skip_first_sample": skip_first_sample},
        )
        STATE["manual_stop"] = False
        STATE["keep_comfy_stopped"] = keep_comfy_stopped
        STATE["comfy_left_stopped"] = False
        STATE["post"] = None
        log_path = _run_dir(project) / "train.log"
        STATE.update(
            {
                "running": True,
                "project": project,
                "log": str(log_path),
                "started": time.time(),
                "error": "",
                "opts": {"steps": steps, "rank": rank, "lr": lr, "resolution": resolution,
                         "dit": dit_path.name},
            }
        )
        threading.Thread(target=_train_worker, args=(project, recipe, log_path, stop_comfy), daemon=True).start()
    return {"ok": True, "recipe": str(recipe), "log": str(log_path), "dit": str(dit_path)}


@app.get("/api/models")
def api_models():
    return scan_models()


@app.post("/api/restore_comfy")
def api_restore_comfy():
    """训练后如果 ComfyUI 被留成关闭状态，这个接口把它拉回来。"""
    _comfy("Run")
    STATE["comfy_left_stopped"] = False
    for _ in range(24):
        time.sleep(2)
        try:
            import socket

            s = socket.socket(); s.settimeout(0.5)
            ok = s.connect_ex(("127.0.0.1", 8189)) == 0
            s.close()
            if ok:
                break
        except Exception:
            pass
    return {"ok": True, "gpu_used_mb": _gpu_used_mb()}


@app.post("/api/stop")
def api_stop():
    proc = STATE.get("proc")
    if proc and proc.poll() is None:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, text=True)
    if STATE.get("project"):
        _kill_strays(STATE["project"])
    STATE["running"] = False
    STATE["proc"] = None
    STATE["manual_stop"] = True      # 手动停止 → 不算错误
    STATE["error"] = ""
    if STATE.get("comfy_stopped"):
        _comfy("Run")
        STATE["comfy_stopped"] = False
    return {"ok": True}


@app.get("/api/status")
def api_status(project: Optional[str] = None):
    # 训练中只认 STATE；空闲时允许用 ?project= 浏览历史产物（重启服务后也能看到上次的结果）
    proj = STATE["project"] if STATE["running"] else (project or STATE["project"])
    if proj and not PROJECT_NAME.match(proj):
        proj = STATE["project"]
    log_path = Path(STATE["log"]) if (STATE.get("log") and proj == STATE.get("project")) else (_run_dir(proj) / "train.log" if proj else None)
    prog = _log_progress(log_path)
    run_dir = _run_dir(proj) if proj else None
    samples = []
    loras = []
    if run_dir and run_dir.is_dir():
        for p in sorted(run_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXT and "sample" in str(p).lower() and ".thumbs" not in str(p).lower():
                samples.append(p.relative_to(RUNS).as_posix())
        samples = samples[:12]
        for p in sorted(run_dir.rglob("*.safetensors")):
            loras.append({"name": p.name, "path": p.relative_to(RUNS).as_posix(), "mb": round(p.stat().st_size / 1048576, 1)})
    return {
        "running": STATE["running"],
        "project": proj,
        "elapsed": int(time.time() - STATE["started"]) if STATE["started"] else 0,
        "error": STATE["error"],
        "opts": STATE["opts"],
        "samples": samples,
        "loras": loras,
        "post": STATE.get("post"),
        "comfy_left_stopped": bool(STATE.get("comfy_left_stopped")),
        **prog,
    }


def _stage_of(text: str) -> str:
    """从日志尾巴判断当前阶段（给人看的一句话）。"""
    t = text[-8000:]
    if "Generating Samples" in t:
        return "正在生成样张"
    if "Saved checkpoint" in t and "s/it" not in t:
        return "正在保存 LoRA"
    if "s/it" in t and "loss" in t:
        return "训练中"
    if "Caching text embeddings" in t:
        return "正在缓存文本向量"
    if "Loading text encoder" in t or "Dequantizing shipped convrot8" in t:
        return "正在加载文本编码器（CPU 反量化，1~4 分钟）"
    if "Loading VAE" in t:
        return "正在加载 VAE"
    if "Quantizing" in t or "Loading transformer" in t:
        return "正在加载 DiT 主模型"
    if "Model Loaded" in t:
        return "模型已加载，准备训练"
    if "Running 1 job" in t or "Running job" in t:
        return "任务已启动"
    return "准备中"


@app.get("/api/log")
def api_log(lines: int = 200, project: Optional[str] = None):
    """日志默认取当前训练；也可以按项目取（训练停了也能看历史日志）。"""
    proj = project or STATE.get("project")
    log_path = None
    if STATE.get("log") and (not project or project == STATE.get("project")):
        cand = Path(STATE["log"])
        if cand.exists():
            log_path = cand
    if log_path is None and proj:
        d = _run_dir(proj)
        if d.is_dir():
            cands = sorted(d.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            if cands:
                log_path = cands[0]
    if not log_path or not log_path.exists():
        return {"lines": [], "stage": "暂无日志", "path": ""}
    try:
        raw = log_path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    except Exception:
        return {"lines": [], "stage": "日志读取失败", "path": str(log_path)}
    data = [ln for ln in raw.split("\n") if ln.strip()]
    return {
        "lines": data[-lines:],
        "stage": _stage_of(raw),
        "path": str(log_path),
    }


@app.post("/api/install")
def api_install(rel_path: str = Form(...)):
    src = (RUNS / rel_path).resolve()
    if not str(src).startswith(str(RUNS.resolve())) or not src.exists():
        raise HTTPException(400, "路径非法")
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    dst = LORA_DIR / src.name
    shutil.copy2(src, dst)
    return {"ok": True, "dst": str(dst)}


# ----------------------------------------------------------------------------- 清理产物
RUNS_TRASH = RUNS / "_deleted"


def _safe_run_path(rel_path: str) -> Path:
    p = (RUNS / rel_path).resolve()
    if not str(p).startswith(str(RUNS.resolve())):
        raise HTTPException(400, "路径非法")
    if str(p).startswith(str(RUNS_TRASH.resolve())):
        raise HTTPException(400, "这已经是回收站里的文件")
    return p


def _artifact_to_trash(project: str, path: Path, stamp: str) -> int:
    """把产物移到 runs\\_deleted\\<项目>_<时间>\\，返回移动的文件数。"""
    if not path.exists():
        return 0
    dest = RUNS_TRASH / f"{project}_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    moved = 0
    items = [path] if path.is_file() else [f for f in path.rglob("*") if f.is_file()]
    for src in items:
        try:
            parent = src.parent.name
            name = src.name if parent in ("samples",) else f"{parent}_{src.name}"
            shutil.move(str(src), str(dest / name))
            moved += 1
        except Exception:
            continue
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    return moved


@app.post("/api/delete_artifact")
def api_delete_artifact(rel_path: str = Form(...)):
    p = _safe_run_path(rel_path)
    if not p.exists():
        raise HTTPException(404, "产物不存在")
    project = p.relative_to(RUNS).parts[0] if p.relative_to(RUNS).parts else "unknown"
    if project == "_deleted":
        raise HTTPException(400, "路径非法")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    n = _artifact_to_trash(project, p, stamp)
    # 样张的缩略图也一起清掉
    if p.suffix.lower() in IMG_EXT and ".thumbs" not in p.parts:
        thumb = p.parent / ".thumbs" / (p.name + ".jpg")
        if thumb.exists():
            _artifact_to_trash(project, thumb, stamp)
    return {"ok": True, "removed": n, "trash": str(RUNS_TRASH / f"{project}_{stamp}")}


@app.post("/api/clear_artifacts")
def api_clear_artifacts(project: str = Form(...)):
    """一键清理该项目的产物：*.safetensors、optimizer.pt、samples\\，日志和配方保留。"""
    project_dir(project)  # 校验项目名
    run_dir = _run_dir(project)
    if not run_dir.is_dir():
        return {"ok": True, "removed": 0, "mb": 0, "trash": None}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    targets = []
    for p in run_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix == ".safetensors" or p.name == "optimizer.pt" or ".thumbs" in p.parts or ".tmp" in p.parts:
            targets.append(p)
    samples_dir = None
    for p in run_dir.rglob("samples"):
        if p.is_dir():
            samples_dir = p
            break
    total_mb = round(sum(f.stat().st_size for f in targets) / 1048576, 1) if targets else 0
    moved = 0
    for f in targets:
        moved += _artifact_to_trash(project, f, stamp)
    if samples_dir and samples_dir.is_dir():
        moved += _artifact_to_trash(project, samples_dir, stamp)
    return {
        "ok": True,
        "removed": moved,
        "mb": total_mb,
        "trash": str(RUNS_TRASH / f"{project}_{stamp}") if moved else None,
    }


@app.post("/api/clear_runs_trash")
def api_clear_runs_trash(project: str = Form(...)):
    project_dir(project)
    n = 0
    for d in RUNS_TRASH.glob(f"{project}_*"):
        if d.is_dir():
            n += sum(1 for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d, ignore_errors=True)
    return {"ok": True, "removed": n}


@app.get("/api/file")
def api_file(path: str):
    # 素材路径是相对 ROOT（datasets/...），产物/样张路径是相对 RUNS（<项目>/<项目>/samples/...），
    # 两个根都要试，否则缩略图会 403。
    for base in (ROOT, RUNS):
        p = (base / path).resolve()
        allowed = str(p).startswith(str(DATASETS.resolve())) or str(p).startswith(str(RUNS.resolve()))
        if allowed and p.is_file():
            return FileResponse(p)
    raise HTTPException(404, "文件不存在或路径非法")


@app.get("/", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
def index():
    # 每次请求都重读 ui.html：改前端不用重启服务（避免打断正在跑的训练）
    try:
        page = (Path(__file__).with_name("ui.html")).read_text(encoding="utf-8")
    except Exception:
        page = HTML
    # 页面必须 no-store：否则浏览器会按 last-modified 猜缓存，改了 UI 还看到旧的
    resp = HTMLResponse(page)
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


HTML = (Path(__file__).with_name("ui.html")).read_text(encoding="utf-8")


if __name__ == "__main__":
    print(f"Qwen Image Edit LoRA Trainer -> http://127.0.0.1:{PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
