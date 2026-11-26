#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MiniMax-H3 LoRA 训练控制台（本地网页版）

启动：  E:\\ai-train\\_START_WEBUI.bat   （或 python app.py）
地址：  http://127.0.0.1:8199

功能
----
* 项目列表 / 新建项目
* 素材浏览 + caption 在线编辑
* 一键切段（调用 make_h3_dataset.py）
* 可视化设置训练参数并启动（自动写 toml、自动停 ComfyUI、跑完自动恢复）
* 实时日志 + 进度（解析 musubi 的 tqdm 输出）
* 训练产物列表 + 一键安装到 ComfyUI loras
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

# ---------------------------------------------------------------- paths
ROOT = Path(r"E:\ai-train")
DATA = ROOT / "datasets"
RECIPES = ROOT / "recipes"
RUNS = ROOT / "runs"
SCRIPTS = ROOT / "scripts"
MODELS = Path(r"F:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models")
LORAS = MODELS / "loras"

H3_CONFIGS = ROOT / "models" / "h3_configs"      # local processor/text-encoder configs (no network)
DEFAULT_DIT = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
DEFAULT_TE = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_VAE = "minimax_h3_video_vae_fp16.safetensors"
DEFAULT_AVAE = "minimax_h3_audio_vae_fp32.safetensors"

MUSUBI_PY = ROOT / "env-musubi" / "Scripts" / "python.exe"
MUSUBI_SRC = ROOT / "src" / "musubi-tuner"
CUT_SCRIPT = SCRIPTS / "make_h3_dataset.py"

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

for p in (DATA, RECIPES, RUNS):
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="H3 LoRA Trainer")

# ---------------------------------------------------------------- state
STATE = {
    "running": False,
    "stage": "",
    "project": "",
    "started": 0.0,
    "log_file": "",
    "proc": None,
    "error": "",
}
LOCK = threading.Lock()


def _safe(name: str) -> str:
    name = (name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,48}", name):
        raise HTTPException(400, "项目名只能用字母/数字/下划线/短横线")
    return name


def _proj_dir(name: str) -> Path:
    return DATA / _safe(name)


def _log_tail(path: str, n: int = 200) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception:
        return ""


def _parse_progress(text: str) -> dict:
    """从 musubi 的 tqdm 输出里解析 steps / loss / speed"""
    out = {"step": 0, "total": 0, "loss": None, "s_it": None, "last": ""}
    for line in text.splitlines():
        m = re.search(r"steps:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^\]]*)\]", line)
        if m:
            out["step"] = int(m.group(1))
            out["total"] = int(m.group(2))
            out["last"] = line.strip()[-120:]
        ml = re.search(r"avr_loss=([\d.]+)", line)
        if ml:
            out["loss"] = float(ml.group(1))
        ms = re.search(r"([\d.]+)s/it", line)
        if ms:
            out["s_it"] = float(ms.group(1))
    return out


# ---------------------------------------------------------------- projects
@app.get("/api/projects")
def list_projects():
    items = []
    if DATA.exists():
        for d in sorted(DATA.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            vids = d / "videos"
            clips = sorted([f for f in vids.glob("*.mp4")]) if vids.exists() else []
            capt = 0
            for c in clips:
                t = c.with_suffix(".txt")
                if t.exists() and t.read_text(encoding="utf-8", errors="replace").strip() not in ("", "-"):
                    capt += 1
            raw = d / "raw"
            raw_n = len([f for f in raw.glob("*") if f.suffix.lower() in VIDEO_EXT]) if raw.exists() else 0
            loras = list((RUNS / d.name).glob("*.safetensors")) if (RUNS / d.name).exists() else []
            items.append({
                "name": d.name,
                "raw": raw_n,
                "clips": len(clips),
                "captioned": capt,
                "loras": len(loras),
                "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime)),
            })
    return {"projects": items}


@app.post("/api/project")
def create_project(payload: dict):
    name = _safe(payload.get("name", ""))
    d = DATA / name
    if d.exists():
        raise HTTPException(400, "项目已存在")
    for sub in ("raw", "videos", "captions"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return {"ok": True, "name": name}


@app.get("/api/project/{name}")
def get_project(name: str):
    d = _proj_dir(name)
    if not d.exists():
        raise HTTPException(404, "项目不存在")
    vids = d / "videos"
    clips = []
    if vids.exists():
        for f in sorted(vids.glob("*.mp4")):
            t = f.with_suffix(".txt")
            cap = t.read_text(encoding="utf-8", errors="replace").strip() if t.exists() else ""
            clips.append({
                "name": f.name,
                "size_kb": round(f.stat().st_size / 1024),
                "caption": "" if cap == "-" else cap,
            })
    raw = d / "raw"
    raws = [f.name for f in sorted(raw.glob("*")) if f.suffix.lower() in VIDEO_EXT] if raw.exists() else []
    loras = []
    rd = RUNS / name
    if rd.exists():
        for f in sorted(rd.glob("*.safetensors")):
            loras.append({"name": f.name, "mb": round(f.stat().st_size / 1024 / 1024, 1)})
    installed = []
    if LORAS.exists():
        installed = [f.name for f in LORAS.glob("*.safetensors")]
    manifest = {}
    mf = d / "videos" / "_cut_manifest.json"
    if mf.exists():
        try:
            manifest = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    toml = RECIPES / f"{name}-dataset.toml"
    return {
        "name": name,
        "clips": clips,
        "raw": raws,
        "loras": loras,
        "installed": installed,
        "cut": manifest,
        "toml": toml.read_text(encoding="utf-8", errors="replace") if toml.exists() else "",
    }


@app.put("/api/caption")
def save_caption(payload: dict):
    name = _safe(payload.get("project", ""))
    clip = payload.get("clip", "")
    text = (payload.get("text") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-\.]{1,64}\.mp4", clip):
        raise HTTPException(400, "非法文件名")
    t = _proj_dir(name) / "videos" / (Path(clip).stem + ".txt")
    if not t.parent.exists():
        raise HTTPException(404, "项目不存在")
    t.write_text(text or "-", encoding="utf-8")
    return {"ok": True}


@app.post("/api/clips/clear")
def clear_clips(payload: dict):
    """Delete every training clip of a project (videos/*.mp4 + *.txt) and its latent cache.

    raw/ is NOT touched - the original footage stays so you can re-cut it.
    """
    name = _safe(payload.get("project", ""))
    d = _proj_dir(name)
    vids = d / "videos"
    if not vids.exists():
        raise HTTPException(404, "项目不存在")
    mp4s = sorted(vids.glob("*.mp4"))
    txs = sorted(vids.glob("*.txt"))
    if not mp4s and not txs:
        raise HTTPException(400, "videos 里没有片段")

    removed = 0
    for f in mp4s + txs:
        if f.resolve().parent != vids.resolve():
            continue
        try:
            f.unlink()
            removed += 1
        except Exception:
            pass

    mf = vids / "_cut_manifest.json"
    if mf.exists():
        try:
            mf.unlink()
        except Exception:
            pass

    # stale latents/text caches would reference clips that no longer exist
    cache = RUNS / name / "cache"
    cache_files = 0
    if cache.exists():
        for f in list(cache.glob("*")):
            try:
                f.unlink()
                cache_files += 1
            except Exception:
                pass
    return {"ok": True, "removed": removed, "cache_cleared": cache_files}


@app.post("/api/captions/clear")
def clear_captions(payload: dict):
    """Reset every clip caption of a project to the placeholder '-'.

    Keeps the .txt files (the trainer expects one next to each clip) but empties
    their content, so nothing from a previous dataset leaks into a new run.
    """
    name = _safe(payload.get("project", ""))
    d = _proj_dir(name)
    vids = d / "videos"
    if not vids.exists():
        raise HTTPException(404, "项目不存在")
    clips = sorted(vids.glob("*.mp4"))
    if not clips:
        raise HTTPException(400, "videos 里没有片段")
    n = 0
    for c in clips:
        t = c.with_suffix(".txt")
        t.write_text("-", encoding="utf-8")
        n += 1
    return {"ok": True, "cleared": n}


@app.get("/api/clip")
def get_clip(project: str, name: str):
    """Serve one training clip so the browser can preview it."""
    d = _proj_dir(project)
    if not re.fullmatch(r"[A-Za-z0-9_\-\.]{1,90}\.mp4", name, re.I):
        raise HTTPException(400, "非法文件名")
    f = d / "videos" / name
    if not f.exists():
        raise HTTPException(404, "片段不存在")
    from fastapi.responses import FileResponse
    return FileResponse(str(f), media_type="video/mp4")


@app.delete("/api/clip")
def delete_clip(project: str, name: str):
    """Delete one training clip + its caption."""
    d = _proj_dir(project)
    if not re.fullmatch(r"[A-Za-z0-9_\-\.]{1,90}\.mp4", name, re.I):
        raise HTTPException(400, "非法文件名")
    vids = d / "videos"
    f = vids / name
    if not f.exists():
        raise HTTPException(404, "片段不存在")
    if f.resolve().parent != vids.resolve():
        raise HTTPException(400, "非法路径")
    f.unlink()
    t = f.with_suffix(".txt")
    if t.exists():
        t.unlink()
    return {"ok": True, "deleted": name}


@app.put("/api/upload")
async def upload(project: str, name: str, request: Request):
    """Raw body upload: PUT /api/upload?project=X&name=file.mp4 with the file as body."""
    d = _proj_dir(project)
    raw = d / "raw"
    if not raw.exists():
        raise HTTPException(404, "项目不存在")
    if not re.fullmatch(r"[A-Za-z0-9 _\-\.]{1,90}\.(mp4|mov|mkv|webm|avi|m4v)", name, re.I):
        raise HTTPException(400, "只接受视频文件（mp4/mov/mkv/webm/avi/m4v）")
    data = await request.body()
    if not data:
        raise HTTPException(400, "空文件")
    (raw / name).write_bytes(data)
    return {"ok": True, "name": name, "mb": round(len(data) / 1024 / 1024, 2)}


@app.delete("/api/raw")
def delete_raw(project: str, name: str):
    d = _proj_dir(project)
    f = d / "raw" / name
    if not f.exists():
        raise HTTPException(404, "文件不存在")
    if f.parent.resolve() != (d / "raw").resolve():
        raise HTTPException(400, "非法路径")
    f.unlink()
    return {"ok": True}


@app.post("/api/cut")
def cut_dataset(payload: dict):
    name = _safe(payload.get("project", ""))
    frames = int(payload.get("frames", 56))
    w = int(payload.get("width", 448))
    h = int(payload.get("height", 256))
    mode = (payload.get("mode") or "blur").strip()
    if mode not in ("blur", "pad", "crop"):
        raise HTTPException(400, "mode 只能是 blur / pad / crop")
    if w % 32 or h % 32:
        raise HTTPException(400, "宽高必须是 32 的倍数")
    if frames not in (22, 39, 56, 73, 90, 107, 124, 141, 158):
        raise HTTPException(400, "帧数必须是 17n+5（22/39/56/73/90/107/124）")
    d = _proj_dir(name)
    if not (d / "raw").exists():
        raise HTTPException(404, "raw 目录不存在")
    py = str(MUSUBI_PY) if MUSUBI_PY.exists() else sys_executable()
    only = payload.get("only") or []
    force = bool(payload.get("force", False))
    cmd = [py, str(CUT_SCRIPT), "--project", name,
           "--frames", str(frames), "--width", str(w), "--height", str(h),
           "--mode", mode]
    if only:
        for f in only:
            cmd += ["--only", str(f)]
    if force:
        cmd += ["--force"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {"ok": p.returncode == 0, "out": (p.stdout or "")[-6000:], "err": (p.stderr or "")[-2000:]}


def sys_executable() -> str:
    import sys
    return sys.executable


# ---------------------------------------------------------------- training
STAGES = [("cache_latents", "1_cache_latents"),
          ("cache_text", "2_cache_text"),
          ("train", "3_train")]


def _write_toml(name: str, frames: int, w: int, h: int) -> Path:
    toml = RECIPES / f"{name}-dataset.toml"
    toml.write_text(
        "[general]\n"
        f"resolution = [{w}, {h}]\n"
        'caption_extension = ".txt"\n'
        "batch_size = 1\n"
        "enable_bucket = false\n\n"
        "[[datasets]]\n"
        f'video_directory = "E:/ai-train/datasets/{name}/videos"\n'
        f'cache_directory = "E:/ai-train/runs/{name}/cache"\n'
        'caption_extension = ".txt"\n'
        f"target_frames = [{frames}]\n",
        encoding="utf-8",
    )
    return toml


def _kill_strays() -> int:
    """Kill orphaned musubi processes left behind by a killed/stopped run.

    Without this, a stopped training leaves its real python child alive while
    STATE already says "not running" -> the UI re-enables and a second click
    launches a duplicate pipeline (observed 2026-11-09).
    """
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*minimax_h3_cache*' -or "
        "$_.CommandLine -like '*minimax_h3_train*' }"
    )
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                           capture_output=True, text=True, timeout=30)
        ids = [int(x) for x in re.findall(r"ProcessId\s*:\s*(\d+)", r.stdout)]
        for pid in ids:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        return len(ids)
    except Exception:
        return 0


def _run_pipeline(name: str, steps: int, rank: int, frames: int, w: int, h: int,
                  blocks: int, lr: float, stop_comfy: bool,
                  dit: str = DEFAULT_DIT, te: str = DEFAULT_TE,
                  vae: str = DEFAULT_VAE, avae: str = DEFAULT_AVAE):
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    log = out / "web_train.log"
    toml = _write_toml(name, frames, w, h)

    def wlog(msg: str):
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n=== {time.strftime('%H:%M:%S')} {msg} ===\n")

    log.write_text("", encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    # The Qwen3-VL processor configs are already in the HF cache; without these
    # flags transformers blocks on a network HEAD and the run looks "stuck"
    # (observed: 0% GPU / 0 CPU for minutes at "Loading ... processor").
    # hf-mirror is reachable but a stalled HEAD can hang the run for minutes
    # (observed: 0% GPU / 0 CPU at "Loading ... processor"). Fail fast instead.
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    # load the Qwen3-VL processor / encoder config from local copies so a slow
    # HF HEAD can never stall the run (it sat at 0% GPU for minutes before)
    if (H3_CONFIGS / "processor").is_dir():
        env.setdefault("H3_PROCESSOR_DIR", str(H3_CONFIGS / "processor"))
    if (H3_CONFIGS / "text_encoder").is_dir():
        env.setdefault("H3_TEXT_ENCODER_CONFIG_DIR", str(H3_CONFIGS / "text_encoder"))

    cmds = [
        ("cache_latents", [
            str(MUSUBI_PY), "minimax_h3_cache_latents.py",
            "--dataset_config", str(toml), "--task", "t2va",
            "--video_vae", str(MODELS / "vae" / vae),
            "--audio_vae", str(MODELS / "vae" / avae),
            "--cache_seed", "42", "--allow_experimental_duration",
        ]),
        ("cache_text", [
            str(MUSUBI_PY), "minimax_h3_cache_text_encoder_outputs.py",
            "--dataset_config", str(toml), "--task", "t2va",
            "--text_encoder", str(MODELS / "text_encoders" / te),
            "--text_cache_dtype", "bf16", "--text_encoder_blocks_to_swap", "50",
        ]),
        ("train", [
            str(MUSUBI_PY), "minimax_h3_train_network.py",
            "--dataset_config", str(toml), "--task", "t2va", "--video_only",
            "--dit", str(MODELS / "diffusion_models" / dit),
            "--vae", str(MODELS / "vae" / vae),
            "--network_dim", str(rank), "--network_alpha", str(rank),
            "--sdpa", "--mixed_precision", "bf16", "--gradient_checkpointing",
            "--blocks_to_swap", str(blocks), "--block_swap_h2d_only",
            "--optimizer_type", "adamw8bit", "--learning_rate", str(lr),
            "--max_train_steps", str(steps), "--save_every_n_steps", "100",
            "--seed", "42",
            "--output_dir", str(out), "--output_name", f"{name}_r{rank}",
        ]),
    ]

    def work():
        try:
            strays = _kill_strays()
            if strays:
                wlog(f"cleaned {strays} stray process(es) from a previous run")
            if stop_comfy:
                wlog("stopping ComfyUI 8189")
                STATE["stage"] = "停止 ComfyUI"
                subprocess.run(["schtasks", "/End", "/TN", "ComfyUI-D-8189"],
                               capture_output=True, text=True)
                time.sleep(4)
            for stage, cmd in cmds:
                with LOCK:
                    STATE["stage"] = stage
                wlog(f"START {stage}")
                with open(log, "a", encoding="utf-8") as lf:
                    proc = subprocess.Popen(cmd, cwd=str(MUSUBI_SRC), env=env,
                                            stdout=lf, stderr=subprocess.STDOUT)
                    with LOCK:
                        STATE["proc"] = proc
                    proc.wait()
                wlog(f"END {stage} rc={proc.returncode}")
                if proc.returncode != 0:
                    with LOCK:
                        STATE["error"] = f"{stage} failed (rc={proc.returncode})"
                    break
        except Exception as e:  # noqa: BLE001
            with LOCK:
                STATE["error"] = str(e)
            wlog(f"EXCEPTION {e}")
        finally:
            if stop_comfy:
                subprocess.run(["schtasks", "/Run", "/TN", "ComfyUI-D-8189"],
                               capture_output=True, text=True)
                wlog("restarted ComfyUI 8189")
            with LOCK:
                STATE.update({"running": False, "stage": "done", "proc": None})

    with LOCK:
        STATE.update({"running": True, "stage": "starting", "project": name,
                      "started": time.time(), "log_file": str(log), "error": ""})
    threading.Thread(target=work, daemon=True).start()
    return {"ok": True, "log": str(log)}


@app.get("/api/models")
def list_models():
    """List selectable DiT / text-encoder / VAE files."""
    def scan(sub: str, patterns: tuple[str, ...]) -> list[dict]:
        d = MODELS / sub
        out = []
        if d.exists():
            for f in sorted(d.rglob("*.safetensors")):
                low = f.name.lower()
                if any(p in low for p in patterns):
                    out.append({
                        "name": f.name,
                        "gb": round(f.stat().st_size / 1024 ** 3, 2),
                        "path": str(f),
                    })
        return out

    return {
        "dit": scan("diffusion_models", ("minimax_h3", "fasth3", "h3_")),
        "te": scan("text_encoders", ("qwen3vl_32b", "minimax_h3")),
        "vae": scan("vae", ("minimax_h3_video_vae",)),
        "avae": scan("vae", ("minimax_h3_audio_vae",)),
        "defaults": {
            "dit": DEFAULT_DIT,
            "te": DEFAULT_TE,
            "vae": DEFAULT_VAE,
            "avae": DEFAULT_AVAE,
        },
    }


@app.post("/api/train")
def start_train(payload: dict):
    with LOCK:
        if STATE["running"]:
            raise HTTPException(409, "已有训练在跑")
        # reserve the slot immediately so a double click cannot spawn two pipelines
        STATE.update({"running": True, "stage": "starting", "started": time.time(), "error": ""})
    name = _safe(payload.get("project", ""))
    d = _proj_dir(name)
    if not (d / "videos").exists():
        raise HTTPException(404, "项目不存在")
    clips = list((d / "videos").glob("*.mp4"))
    if not clips:
        raise HTTPException(400, "videos 里还没有片段")
    steps = max(1, int(payload.get("steps", 600)))
    rank = max(4, int(payload.get("rank", 16)))
    frames = int(payload.get("frames", 56))
    w = int(payload.get("width", 448))
    h = int(payload.get("height", 256))
    blocks = max(0, int(payload.get("blocks", 28)))
    lr = float(payload.get("lr", 1e-4))
    stop_comfy = bool(payload.get("stop_comfy", True))

    def pick(key: str, sub: str, default: str) -> str:
        val = (payload.get(key) or default).strip()
        f = MODELS / sub / val
        if not f.is_file():
            raise HTTPException(400, f"模型文件不存在: {sub}/{val}")
        return val

    dit = pick("dit", "diffusion_models", DEFAULT_DIT)
    te = pick("te", "text_encoders", DEFAULT_TE)
    vae = pick("vae", "vae", DEFAULT_VAE)
    avae = pick("avae", "vae", DEFAULT_AVAE)
    return _run_pipeline(name, steps, rank, frames, w, h, blocks, lr, stop_comfy,
                         dit=dit, te=te, vae=vae, avae=avae)


@app.get("/api/status")
def status():
    with LOCK:
        s = {k: v for k, v in STATE.items() if k != "proc"}
    if s.get("log_file"):
        txt = _log_tail(s["log_file"], 400)
        s.update(_parse_progress(txt))
        s["elapsed"] = int(time.time() - s["started"]) if s.get("started") else 0
        s["log_tail"] = txt[-4000:]
    return s


@app.post("/api/stop")
def stop_train():
    with LOCK:
        proc = STATE.get("proc")
    if not proc:
        return {"ok": False, "msg": "没有在跑的训练"}
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, text=True)
        with LOCK:
            STATE["error"] = "用户手动停止"
            STATE["stage"] = "stopped"
            STATE["running"] = False
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "msg": str(e)}


@app.post("/api/install")
def install_lora(payload: dict):
    name = _safe(payload.get("project", ""))
    fname = payload.get("file", "")
    if not re.fullmatch(r"[A-Za-z0-9_\-\.]{1,80}\.safetensors", fname):
        raise HTTPException(400, "非法文件名")
    src = RUNS / name / fname
    if not src.exists():
        raise HTTPException(404, "文件不存在")
    LORAS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, LORAS / fname)
    return {"ok": True, "dest": str(LORAS / fname)}


@app.get("/api/log")
def get_log(lines: int = 200):
    if not STATE.get("log_file"):
        return {"log": ""}
    return {"log": _log_tail(STATE["log_file"], max(20, min(2000, lines)))}


# ---------------------------------------------------------------- UI
@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>H3 LoRA 训练台</title>
<style>
:root{--bg:#0f1115;--card:#161a21;--line:#2a303c;--fg:#e5e7eb;--dim:#8b93a3;--acc:#3b82f6;--ok:#22c55e;--warn:#f59e0b;--bad:#ef4444}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--fg)}
.wrap{display:grid;grid-template-columns:230px 1fr 360px;gap:14px;padding:14px;height:100vh}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;overflow:auto}
h1{font-size:16px;margin:0 0 10px}h2{font-size:13px;margin:0 0 8px;color:var(--dim);font-weight:600;text-transform:uppercase;letter-spacing:.5px}
.proj{padding:8px 10px;border-radius:10px;cursor:pointer;margin-bottom:6px;border:1px solid transparent}
.proj:hover{background:#1b2029}
.proj.active{background:#1d2838;border-color:var(--acc)}
.proj .t{font-weight:600}.proj .s{font-size:11.5px;color:var(--dim)}
button{background:var(--acc);color:#fff;border:0;border-radius:9px;padding:8px 14px;font-size:13px;font-weight:600;cursor:pointer}
button.ghost{background:#242a35;color:var(--fg)}
button:disabled{opacity:.45;cursor:not-allowed}
input,select{background:#0f1115;border:1px solid #33384a;color:#fff;border-radius:8px;padding:7px 9px;font-size:13px;width:100%}
.row{display:flex;gap:8px;align-items:center;margin-bottom:8px}
.row label{flex:0 0 90px;color:var(--dim);font-size:12.5px}
.clip{border:1px solid var(--line);border-radius:10px;padding:8px;margin-bottom:8px}
.clip .hd{display:flex;justify-content:space-between;font-size:12px;color:var(--dim);margin-bottom:6px}
.clip textarea{width:100%;height:44px;background:#0f1115;border:1px solid #33384a;color:#fff;border-radius:8px;padding:6px;font-size:12.5px;resize:vertical}
#log{font:11.5px/1.45 ui-monospace,Consolas,monospace;white-space:pre-wrap;background:#0b0d11;border:1px solid var(--line);border-radius:10px;padding:10px;height:270px;overflow:auto;color:#c7d2fe}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700}
.pbar{flex:1;height:10px;background:#0b0d11;border:1px solid #2a303c;border-radius:99px;overflow:hidden;position:relative}
.pfill{height:100%;width:0%;background:linear-gradient(90deg,#1d4ed8,#3b82f6);transition:width .45s ease}
.pfill.ind{width:100%;background:repeating-linear-gradient(90deg,#1e3a8a 0 14px,#3b82f6 14px 28px);background-size:28px 100%;animation:slide 0.9s linear infinite;opacity:.85}
@keyframes slide{from{background-position:0 0}to{background-position:28px 0}}
.b-ok{background:#14361f;color:#86efac}.b-bad{background:#3a1414;color:#fca5a5}.b-run{background:#1e3a8a;color:#bfdbfe}
.hint{font-size:11.5px;color:var(--dim);margin-top:6px}
</style></head><body><div class="wrap">

<div class="card">
  <h1>项目</h1>
  <div class="row"><input id="newName" placeholder="新项目名"><button onclick="newProject()">建</button></div>
  <div id="projects"></div>
</div>

<div class="card">
  <h2 id="ptitle">选择一个项目</h2>
  <div id="upbox" style="margin-bottom:12px">
    <h2>上传素材 <span id="rawcount" class="hint" style="font-weight:400"></span></h2>
    <div id="drop" style="border:2px dashed #33384a;border-radius:12px;padding:18px;text-align:center;cursor:pointer;color:#8b93a3;font-size:13px">
      把视频拖到这里，或点击选择文件<br><span style="font-size:11.5px">支持 mp4 / mov / mkv / webm / avi，可多选</span>
    </div>
    <input id="fileInput" type="file" multiple accept="video/*" style="display:none">
    <div id="upmsg" class="hint"></div>
    <div id="rawlist" style="margin-top:8px;font-size:12px;color:#8b93a3"></div>
  </div>
  <h2 style="display:flex;justify-content:space-between;align-items:center">
    训练片段
    <span><button class="ghost" style="padding:4px 10px;font-size:11.5px" onclick="clearCaptions()">清空全部标注</button>
    <button class="ghost" style="padding:4px 10px;font-size:11.5px;color:#fca5a5" onclick="clearClips()">清空全部片段</button></span>
  </h2>
  <div id="clips"></div>
  <div class="hint" id="phint">素材放 raw\ 里，点「切段」自动生成训练片段。</div>
</div>

<div class="card">
  <h2>训练参数</h2>
  <div class="row"><label>主干模型</label><select id="dit"><option>加载中…</option></select></div>
  <div class="row"><label>文本编码器</label><select id="te"><option>加载中…</option></select></div>
  <div class="row"><label>视频 VAE</label><select id="vae"><option>加载中…</option></select></div>
  <div class="row"><label>音频 VAE</label><select id="avae"><option>加载中…</option></select></div>
  <div class="row"><label>步数</label><input id="steps" type="number" value="1000"></div>
  <div class="row"><label>Rank</label><input id="rank" type="number" value="16"></div>
  <div class="row"><label>帧数</label><select id="frames">
    <option value="22" selected>22 (0.92s · 最快)</option><option value="39">39 (1.63s)</option>
    <option value="56">56 (2.33s · 慢60倍)</option><option value="73">73 (3.04s)</option>
    <option value="124">124 (5.17s)</option></select></div>
  <div class="row"><label>缩放方式</label><select id="mode">
    <option value="blur" selected>模糊填充（不丢内容）</option>
    <option value="pad">黑边填充（不丢内容）</option>
    <option value="crop">居中裁切（会切边）</option></select></div>
  <div class="row"><label>宽 × 高</label><input id="width" type="number" value="448" style="flex:1"><input id="height" type="number" value="256" style="flex:1"></div>
  <div class="row"><label>块换出</label><input id="blocks" type="number" value="28"></div>
  <div class="row"><label>学习率</label><input id="lr" value="0.0001"></div>
  <div class="row"><label>停 ComfyUI</label><input id="stopComfy" type="checkbox" checked style="width:auto"></div>
  <div class="row"><button id="btnCut" onclick="doCut()">切段</button><button id="btnTrain" onclick="doTrain()">开始训练</button><button id="btnStop" class="ghost" onclick="doStop()" disabled style="opacity:.45">停止</button></div>
  <div class="hint" id="msg"></div><div class="hint">⚠️ 实测：22 帧 4 秒/步；56 帧 250 秒/步（60×，16GB 显存撑不住 56 帧的注意力）。</div>
  <h2 style="margin-top:14px">进度 <span id="badge" class="badge b-ok">空闲</span></h2>
  <div style="display:flex;align-items:center;gap:8px;margin:2px 0 8px">
    <div class="pbar"><div id="pfill" class="pfill"></div></div>
    <span id="ppct" class="hint" style="min-width:46px;text-align:right;font-variant-numeric:tabular-nums">0%</span>
  </div>
  <div id="prog" class="hint">—</div>
  <h2 style="margin-top:10px">日志</h2>
  <div id="log">（未开始）</div>
</div>
</div>
<script>
let cur=null, timer=null;
const $=id=>document.getElementById(id);
async function api(u,m='GET',b){
  const o={method:m,headers:{'Content-Type':'application/json'}};
  if(b)o.body=JSON.stringify(b);
  const r=await fetch(u,o); const j=await r.json().catch(()=>({}));
  if(!r.ok) throw new Error(j.detail||r.status);
  return j;
}

// ---------- upload ----------
const drop=$('drop'), fi=$('fileInput');
drop.onclick=()=>fi.click();
fi.onchange=e=>{ uploadFiles(e.target.files); fi.value=''; };
drop.ondragover=e=>{e.preventDefault(); drop.style.borderColor='#3b82f6'; drop.style.background='#14203a';};
drop.ondragleave=()=>{drop.style.borderColor='#33384a'; drop.style.background='';};
drop.ondrop=e=>{e.preventDefault(); drop.style.borderColor='#33384a'; drop.style.background=''; uploadFiles(e.dataTransfer.files);};
async function uploadFiles(files){
  if(!cur) return alert('先选项目');
  const arr=[...files].filter(f=>/\.(mp4|mov|mkv|webm|avi|m4v)$/i.test(f.name));
  if(!arr.length) return alert('没有可上传的视频文件');
  let okN=0;
  for(const f of arr){
    $('upmsg').textContent='上传中 '+f.name+'（'+Math.round(f.size/1024/1024*10)/10+' MB）…';
    try{
      const r=await fetch('/api/upload?project='+encodeURIComponent(cur)+'&name='+encodeURIComponent(f.name),
                          {method:'PUT',body:f});
      if(r.ok) okN++; else { const j=await r.json().catch(()=>({})); $('upmsg').textContent='失败 '+f.name+': '+(j.detail||r.status); }
    }catch(e){ $('upmsg').textContent='失败 '+f.name+': '+e.message; }
  }
  $('upmsg').textContent='上传完成 '+okN+'/'+arr.length+'，现在可以点「切段」';
  await openProject(cur);
}
async function clearClips(){
  if(!cur) return alert('先选项目');
  if(!confirm('删除 '+cur+' 的所有训练片段（mp4 + caption + latent 缓存）？\n\n· raw/ 里的原始素材会保留，可以重新切\n· 此操作不可撤销')) return;
  if(!confirm('再确认一次：真的要清空全部片段吗？')) return;
  try{
    const d=await api('/api/clips/clear','POST',{project:cur});
    $('msg').textContent='已删除 '+d.removed+' 个文件，清缓存 '+d.cache_cleared+' 个';
    await openProject(cur); await loadProjects();
  }catch(e){ alert('清空失败: '+e.message); }
}
async function clearCaptions(){
  if(!cur) return alert('先选项目');
  if(!confirm('把 '+cur+' 里所有片段的 caption 清空为 "-" ？\n（片段本身不会删，只清文字）')) return;
  try{
    const d=await api('/api/captions/clear','POST',{project:cur});
    $('msg').textContent='已清空 '+d.cleared+' 条标注';
    await openProject(cur); await loadProjects();
  }catch(e){ alert('清空失败: '+e.message); }
}
async function delClip(n){
  if(!confirm('删除片段 '+n+' ？（连同 caption 一起删）')) return;
  try{
    await api('/api/clip?project='+encodeURIComponent(cur)+'&name='+encodeURIComponent(n),'DELETE');
    $('msg').textContent='已删除 '+n;
    await openProject(cur); await loadProjects();
  }catch(e){ alert('删除失败: '+e.message); }
}
async function delRaw(n){
  if(!confirm('删除 '+n+' ?')) return;
  try{ await api('/api/raw?project='+encodeURIComponent(cur)+'&name='+encodeURIComponent(n),'DELETE'); await openProject(cur);}catch(e){alert(e.message)}
}

async function loadModels(){
  try{
    const d = await fetch('/api/models').then(r=>r.json());
    const fill = (id, items, def, showGb)=>{
      const el = $(id);
      if(!el) return;
      el.innerHTML = items.map(m=>{
        const label = showGb ? `${m.name}  (${m.gb} GB)` : m.name;
        const sel = (m.name === def) ? ' selected' : '';
        return `<option value="${m.name}"${sel}>${label}</option>`;
      }).join('') || '<option value="">（没找到）</option>';
      if(def && items.some(m=>m.name===def)) el.value = def;
    };
    fill('dit', d.dit,   d.defaults.dit,  true);
    fill('te',  d.te,    d.defaults.te,   true);
    fill('vae', d.vae,   d.defaults.vae,  true);
    fill('avae',d.avae,  d.defaults.avae, false);
  }catch(e){ console.warn('loadModels failed', e); }
}
async function loadProjects(){
  const d=await api('/api/projects');
  $('projects').innerHTML = d.projects.map(p=>`
    <div class="proj ${cur===p.name?'active':''}" onclick="openProject('${p.name}')">
      <div class="t">${p.name}</div>
      <div class="s">片段 ${p.clips} · 标注 ${p.captioned} · raw ${p.raw} · LoRA ${p.loras}</div>
    </div>`).join('') || '<div class="hint">还没有项目</div>';
}
async function newProject(){
  const n=$('newName').value.trim(); if(!n)return;
  try{ await api('/api/project','POST',{name:n}); $('newName').value=''; cur=n; await loadProjects(); openProject(n);}catch(e){alert(e.message)}
}
async function openProject(n){
  cur=n; await loadProjects();
  const d=await api('/api/project/'+encodeURIComponent(n));
  $('ptitle').textContent=n+`　（片段 ${d.clips.length} / raw ${d.raw.length}）`;
  $('clips').innerHTML = d.clips.map(c=>`
    <div class="clip">
      <div class="hd">
        <b>${c.name}</b>
        <span>${c.size_kb} KB
          <a style="color:#ef4444;cursor:pointer;margin-left:8px" onclick="delClip('${c.name}')">删除</a>
        </span>
      </div>
      <video src="/api/clip?project=${encodeURIComponent(n)}&name=${encodeURIComponent(c.name)}"
             muted loop playsinline preload="metadata"
             style="width:100%;max-height:130px;background:#000;border-radius:8px;margin-bottom:6px"
             onmouseover="this.play()" onmouseout="this.pause();this.currentTime=0"></video>
      <textarea onchange="saveCap('${c.name}',this.value)" placeholder="一句话描述：谁 + 在做什么 + 镜头 + 声音">${c.caption||''}</textarea>
    </div>`).join('') || '<div class="hint">videos 里没有片段，先点「切段」</div>';
  $('rawcount').textContent = `（raw ${d.raw.length} 个）`;
  const cutMap = d.cut || {};
  $('rawlist').innerHTML = d.raw.length
    ? d.raw.map(n=>{
        const info = cutMap[n];
        const badge = info
          ? `<span style="color:#22c55e;margin-left:6px">已切 ${info.clips} 段</span>`
          : `<span style="color:#f59e0b;margin-left:6px">未切</span>`;
        const btn = info
          ? `<a style="color:#3b82f6;cursor:pointer;margin-left:8px" onclick="cutOne('${n}',true)">重切</a>`
          : `<a style="color:#3b82f6;cursor:pointer;margin-left:8px" onclick="cutOne('${n}',false)">切这段</a>`;
        return `<span style="display:inline-block;background:#1b2029;border:1px solid #2a303c;border-radius:8px;padding:4px 10px;margin:0 6px 6px 0">${n}${badge}${btn}
          <a style="color:#ef4444;cursor:pointer;margin-left:6px" onclick="delRaw('${n}')">×</a></span>`;
      }).join('')
    : '';
  if(d.loras.length){
    $('phint').innerHTML='训练产物：'+d.loras.map(l=>`${l.name} (${l.mb} MB)
      <button class="ghost" style="padding:3px 8px;font-size:11px" onclick="install('${l.name}')">装进 ComfyUI</button>`).join('　');
  }
}
async function saveCap(clip,text){
  try{ await api('/api/caption','PUT',{project:cur,clip,text}); }catch(e){alert(e.message)}
}
async function cutOne(rawName, force){
  if(!cur) return alert('先选项目');
  if(force && !confirm('重新切 '+rawName+' ？会覆盖它之前生成的片段')) return;
  $('msg').textContent='切段中 '+rawName+' …';
  try{
    const d=await api('/api/cut','POST',{project:cur,frames:+$('frames').value,
      width:+$('width').value,height:+$('height').value,mode:$('mode').value,
      only:[rawName],force:force});
    $('msg').textContent = d.ok ? ('切段完成：'+rawName) : ('切段失败：'+((d.out||'')+(d.err||'')).slice(-200));
    $('log').textContent=(d.out||'')+(d.err||'');
    await openProject(cur); await loadProjects();
  }catch(e){ $('msg').textContent='切段失败: '+e.message; }
}
async function doCut(){
  if(!cur)return alert('先选项目');
  $('msg').textContent='切段中…';
  try{
    const d=await api('/api/cut','POST',{project:cur,frames:+$('frames').value,width:+$('width').value,height:+$('height').value,mode:$('mode').value});
    $('msg').textContent=d.ok?'切段完成':'切段失败';
    $('log').textContent=(d.out||'')+(d.err||'');
    await openProject(cur);
  }catch(e){$('msg').textContent='切段失败: '+e.message}
}
let _trainedOnce = false;
let _wasRunning = false;
function setProgress(pct, indeterminate){
  const f=$('pfill'), t=$('ppct');
  if(indeterminate){
    f.className='pfill ind';
    t.textContent='…';
    return;
  }
  const p=Math.max(0,Math.min(100,pct||0));
  f.className='pfill';
  f.style.width=p.toFixed(1)+'%';
  t.textContent=p.toFixed(0)+'%';
}
function setTrainingUI(running){
  const bt=$('btnTrain'), bc=$('btnCut'), bs=$('btnStop');
  if(running){
    bt.textContent='训练中…'; bt.disabled=true; bt.style.opacity=.6;
    bc.disabled=true; bc.style.opacity=.5;
    bs.disabled=false; bs.textContent='停止'; bs.style.opacity=1;
  }else{
    bt.textContent='开始训练'; bt.disabled=false; bt.style.opacity=1;
    bc.disabled=false; bc.style.opacity=1;
    bs.disabled=true; bs.textContent='停止'; bs.style.opacity=.45;
  }
}
async function doTrain(){
  if(!cur)return alert('先选项目');
  try{
    await api('/api/train','POST',{project:cur,steps:+$('steps').value,rank:+$('rank').value,
      frames:+$('frames').value,width:+$('width').value,height:+$('height').value,
      blocks:+$('blocks').value,lr:parseFloat($('lr').value),stop_comfy:$('stopComfy').checked,
      dit:$('dit')?$('dit').value:'', te:$('te')?$('te').value:'',
      vae:$('vae')?$('vae').value:'', avae:$('avae')?$('avae').value:''});
    $('msg').textContent='已启动训练';
    setTrainingUI(true);
    if(!timer) timer=setInterval(poll,2000);
  }catch(e){$('msg').textContent='启动失败: '+e.message}
}
async function doStop(){
  try{
    await api('/api/stop','POST',{});
    $('msg').textContent='已请求停止，正在收尾…';
    $('btnStop').disabled=true;
    $('btnStop').textContent='停止中…';
  }catch(e){ alert(e.message); }
}
async function install(f){
  try{ const d=await api('/api/install','POST',{project:cur,file:f}); $('msg').textContent='已安装到 '+d.dest; }catch(e){alert(e.message)}
}
async function poll(){
  try{
    const s=await api('/api/status');
    const b=$('badge');
    setTrainingUI(!!s.running);
    if(s.running){
      _trainedOnce=true;
      if(s.stage==='train' && s.total>0){
        setProgress(s.step/s.total*100, false);
      }else{
        setProgress(0, true);          // 缓存阶段没有总数，用流动条纹表示进行中
      }
    }else if(s.error && s.error!=='用户手动停止'){
      setProgress(0, false);
    }else if(_trainedOnce){
      setProgress(100, false);
    }else{
      setProgress(0, false);
    }
    if(s.running){ b.className='badge b-run'; b.textContent=s.stage||'运行中'; }
    else if(s.error){ b.className='badge b-bad'; b.textContent='失败'; }
    else { b.className='badge b-ok'; b.textContent='空闲'; }
    $('prog').textContent = s.running
      ? `${s.stage}　${s.step||0}/${s.total? s.total:''}　loss=${s.loss??'-'}　${s.s_it??'-'}s/it　已用 ${s.elapsed}s`
      : (s.error? '错误: '+s.error : '—');
    if(s.log_tail) $('log').textContent=s.log_tail;
    $('log').scrollTop=$('log').scrollHeight;
    if(!s.running && _wasRunning){
      _wasRunning=false;
      await loadProjects();
      if(cur) await openProject(cur);
    }
    if(s.running) _wasRunning=true;
  }catch(e){}
}
loadProjects();
loadModels();
poll();
setInterval(poll, 2000);   // keep polling forever so a page refresh still shows live state
</script></body></html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8199, log_level="warning")
