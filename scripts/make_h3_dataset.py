#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_h3_dataset.py -- turn arbitrary videos into MiniMax-H3 training clips.

What it does
------------
* reads every video in  <project>/raw/
* splits each one into fixed-length chunks (default 56 frames = 2.33 s @24fps,
  which is a legal H3 length: 17n+5 -> 22/39/56/73/90/107/124)
* rescales each chunk to a 32-multiple size (default 448x256, center-cropped,
  no letterbox bars) and re-encodes at 24 fps
* writes  <project>/videos/NN.mp4  +  NN.txt  (caption copied from raw if a
  same-named .txt exists, otherwise a placeholder is written)
* prints a verification table (fps / frames / size / 32-multiple check)

Usage
-----
    python make_h3_dataset.py --project h3_char_A
    python make_h3_dataset.py --project h3_char_A --frames 124 --width 768 --height 448
    python make_h3_dataset.py --src "D:/src" --dst "E:/out" --dry-run

Requirements: ffmpeg + ffprobe on PATH.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".wmv"}
BASE = r"E:\ai-train\datasets"
FPS = 24
LEGAL_FRAMES = (22, 39, 56, 73, 90, 107, 124, 141, 158)


def sh(cmd: list[str]) -> str:
    """Run a command, return stdout (text). Raises on failure."""
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "").strip()[:400])
    return p.stdout


def probe(path: str) -> dict:
    out = sh([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames:format=duration",
        "-of", "json", path,
    ])
    d = json.loads(out)
    st = (d.get("streams") or [{}])[0]
    dur = float((d.get("format") or {}).get("duration") or 0)
    num, _, den = (st.get("r_frame_rate") or "24/1").partition("/")
    fps = (float(num) / float(den or 1)) if num else 24.0
    return {
        "w": int(st.get("width") or 0),
        "h": int(st.get("height") or 0),
        "fps": fps,
        "duration": dur,
    }


def is_mult32(v: int) -> bool:
    return v > 0 and v % 32 == 0


def pick_size(w: int, h: int, want_w: int, want_h: int) -> tuple[int, int]:
    """Return a 32-multiple (w, h) close to the requested size and the source aspect."""
    if want_w and want_h:
        return want_w, want_h
    # keep aspect ratio, fit inside 448x256, snap down to 32
    scale = min(448 / max(w, 1), 256 / max(h, 1))
    tw = max(32, int(w * scale) // 32 * 32)
    th = max(32, int(h * scale) // 32 * 32)
    return tw, th


def make_clip(src: str, dst: str, start: float, frames: int, size: tuple[int, int],
              keep_audio: bool, mode: str = "blur") -> None:
    """mode:
         crop  - cover + center crop (cuts the edges, no bars)
         pad   - fit + black bars (nothing cut, but black borders)
         blur  - fit + blurred background fill (nothing cut, no black bars)  [default]
    """
    tw, th = size
    tail = (
        "setsar=1,"
        "tpad=stop_mode=clone:stop_duration=1.0,"      # pad short sources by cloning the last frame
        "format=yuv420p"
    )
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", src]

    if mode == "crop":
        vf = (f"fps={FPS},"
              f"scale={tw}:{th}:force_original_aspect_ratio=increase,"
              f"crop={tw}:{th}," + tail)
        cmd += ["-vf", vf]
    elif mode == "pad":
        vf = (f"fps={FPS},"
              f"scale={tw}:{th}:force_original_aspect_ratio=decrease,"
              f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black," + tail)
        cmd += ["-vf", vf]
    else:  # blur
        fc = (
            f"[0:v]fps={FPS},split=2[bg][fg];"
            f"[bg]scale={tw}:{th}:force_original_aspect_ratio=increase,crop={tw}:{th},"
            f"gblur=sigma=24,eq=brightness=-0.06[bgb];"
            f"[fg]scale={tw}:{th}:force_original_aspect_ratio=decrease[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2," + tail + "[out]"
        )
        cmd += ["-filter_complex", fc, "-map", "[out]"]

    cmd += ["-frames:v", str(frames), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]
    if keep_audio:
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-an"]
    cmd += [dst]
    sh(cmd)


def verify(path: str, frames: int, size: tuple[int, int]) -> tuple[bool, str]:
    info = probe(path)
    ok = (
        abs(info["fps"] - FPS) < 0.6
        and info["w"] == size[0]
        and info["h"] == size[1]
        and is_mult32(info["w"])
        and is_mult32(info["h"])
    )
    n = round(info["duration"] * FPS)
    ok = ok and abs(n - frames) <= 2
    return ok, f"{info['w']}x{info['h']} {info['fps']:.1f}fps {info['duration']:.2f}s ~{n}f"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", help="project folder name under E:\\ai-train\\datasets")
    ap.add_argument("--src", help="source folder (overrides --project)")
    ap.add_argument("--dst", help="output folder (overrides --project)")
    ap.add_argument("--frames", type=int, default=56, help="frames per clip (17n+5), default 56")
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--auto-size", action="store_true", help="derive size from source aspect")
    ap.add_argument("--keep-audio", action="store_true", help="keep audio track")
    ap.add_argument("--mode", choices=["blur", "pad", "crop"], default="blur",
                    help="blur=fit+blurred bg (default, nothing cut) / pad=fit+black bars / crop=center crop")
    ap.add_argument("--start-offset", type=float, default=0.0, help="skip first N seconds of each source")
    ap.add_argument("--max-clips-per-src", type=int, default=0, help="0 = no limit")
    ap.add_argument("--only", action="append", default=None,
                    help="cut ONLY this source file (repeatable). e.g. --only a.mp4")
    ap.add_argument("--naming", choices=["source", "seq"], default="source",
                    help="source=<srcstem>_NN.mp4 (default, never overwrites other sources) / seq=01.mp4...")
    ap.add_argument("--force", action="store_true",
                    help="re-cut sources that are already recorded in the cut manifest")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.frames not in LEGAL_FRAMES:
        print(f"[warn] {a.frames} frames is not a legal H3 length; legal: {LEGAL_FRAMES}")
        print("       continuing anyway (H3 will trim it down)")

    if not a.src:
        if not a.project:
            ap.error("need --project or --src")
        a.src = os.path.join(BASE, a.project, "raw")
    if not a.dst:
        if not a.project:
            ap.error("need --project or --dst")
        a.dst = os.path.join(BASE, a.project, "videos")

    if not os.path.isdir(a.src):
        print(f"[error] source folder not found: {a.src}")
        return 1
    os.makedirs(a.dst, exist_ok=True)

    all_srcs = sorted(
        f for f in os.listdir(a.src) if os.path.splitext(f)[1].lower() in VIDEO_EXT
    )
    if not all_srcs:
        print(f"[error] no video files in {a.src}")
        return 1

    # only cut the requested sources (default: every source not cut yet)
    manifest_path = os.path.join(a.dst, "_cut_manifest.json")
    manifest = {}
    try:
        if os.path.isfile(manifest_path):
            manifest = json.load(open(manifest_path, encoding="utf-8"))
    except Exception:
        manifest = {}

    if a.only:
        wanted = {os.path.basename(x) for x in a.only}
        srcs = [f for f in all_srcs if f in wanted]
        missing = wanted - set(srcs)
        if missing:
            print(f"[error] source not found in {a.src}: {sorted(missing)}")
            return 1
    else:
        srcs = [f for f in all_srcs if a.force or f not in manifest]
        skipped = [f for f in all_srcs if f in manifest and not a.force]
        if skipped:
            print(f"[skip] already cut ({len(skipped)}): {', '.join(skipped[:6])}"
                  + (" ..." if len(skipped) > 6 else ""))
        if not srcs:
            print("[info] nothing to do - every source is already cut (use --force to re-cut)")
            return 0

    clip_sec = a.frames / FPS
    print(f"source : {a.src}")
    print(f"output : {a.dst}")
    print(f"spec   : {a.frames} frames ({clip_sec:.2f}s) @ {FPS}fps, "
          f"size {'auto(32-mult)' if a.auto_size else f'{a.width}x{a.height}'}")
    print(f"files  : {len(srcs)}")
    print()

    idx = 0
    made = 0
    failed: list[str] = []
    for name in srcs:
        full = os.path.join(a.src, name)
        try:
            info = probe(full)
        except Exception as e:
            print(f"[skip] {name}: probe failed ({e})")
            failed.append(name)
            continue

        size = pick_size(info["w"], info["h"], 0, 0) if a.auto_size else (a.width, a.height)
        if not (is_mult32(size[0]) and is_mult32(size[1])):
            print(f"[skip] {name}: target size {size} is not a multiple of 32")
            failed.append(name)
            continue

        usable = max(0.0, info["duration"] - a.start_offset)
        n_chunks = int(usable // clip_sec) if clip_sec > 0 else 0
        if a.max_clips_per_src:
            n_chunks = min(n_chunks, a.max_clips_per_src)
        if n_chunks == 0:
            # a source slightly shorter than one clip is still usable: pad the tail
            if usable >= 1.0:
                n_chunks = 1
                print(f"[pad] {name}: {info['duration']:.2f}s < {clip_sec:.2f}s -> 1 padded clip")
            else:
                print(f"[skip] {name}: only {info['duration']:.2f}s, too short (need >= 1s)")
                failed.append(name)
                continue

        print(f"[{name}] {info['w']}x{info['h']} {info['fps']:.1f}fps "
              f"{info['duration']:.2f}s -> {n_chunks} clip(s)")

        # caption base: same-name .txt next to the source
        cap_src = os.path.splitext(full)[0] + ".txt"
        cap_text = ""
        if os.path.isfile(cap_src):
            cap_text = open(cap_src, encoding="utf-8", errors="replace").read().strip()

        stem = os.path.splitext(name)[0]
        for c in range(n_chunks):
            idx += 1
            if a.naming == "source":
                out_name = f"{stem}_{c+1:03d}"
            else:
                out_name = f"{idx:02d}"
            out_mp4 = os.path.join(a.dst, out_name + ".mp4")
            out_txt = os.path.join(a.dst, out_name + ".txt")
            start = a.start_offset + c * clip_sec
            if a.dry_run:
                print(f"   -> {os.path.basename(out_mp4)} start={start:.2f}s (dry run)")
                made += 1
                continue
            try:
                make_clip(full, out_mp4, start, a.frames, size, a.keep_audio, a.mode)
            except Exception as e:
                print(f"   !! clip {idx} failed: {e}")
                failed.append(f"{name}#{c+1}")
                continue
            ok, desc = verify(out_mp4, a.frames, size)
            open(out_txt, "w", encoding="utf-8").write(cap_text or "-")
            print(f"   -> {os.path.basename(out_mp4)}  {desc}  {'OK' if ok else 'CHECK'}")
            made += 1

        # remember that this source has been cut, so "cut all" will not redo it
        if not a.dry_run and made:
            manifest[name] = {
                "clips": n_chunks,
                "frames": a.frames,
                "size": list(size),
                "mode": a.mode,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            try:
                json.dump(manifest, open(manifest_path, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
            except Exception as e:
                print(f"[warn] could not write manifest: {e}")

    print()
    print(f"done: {made} clip(s) written to {a.dst}")
    if failed:
        print(f"problems: {len(failed)} -> {failed[:8]}")
        print("         (see messages above)")
    print()
    print("next:")
    print("  1) fill captions  <project>/videos/NN.txt   (one sentence per clip)")
    print("  2) run  _ONE_CLICK_TRAIN.bat  with PROJECT set to this project")
    return 0 if made else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
