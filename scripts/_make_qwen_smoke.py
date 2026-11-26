"""生成 qwen_smoke 冒烟数据集：5 对 control/target（512x512）+ caption。

来源：E:\\WBE下载目录\\Qwen21-局部替换-素材\\assets 里的现成图。
target = control 的「提亮 + 偏暖 + 加饱和」版本（只为验证管线能跑通，不代表训练质量）。
"""

from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

ASSETS = Path(r"E:\WBE下载目录\Qwen21-局部替换-素材\assets")
OUT = Path(r"E:\ai-train\datasets\qwen_smoke")
SIZE = 512
CAPTION = "把画面调亮、偏暖、提高饱和度，构图和内容保持不变，摄影后期风格"


def blur_pad(im: Image.Image, size: int) -> Image.Image:
    """等比缩放 + 高斯模糊背景填充，不裁切内容。"""
    im = im.convert("RGB")
    scale = min(size / im.width, size / im.height)
    new = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
    canvas = im.resize((size, size), Image.LANCZOS).filter(ImageFilter.GaussianBlur(24))
    canvas.paste(new, ((size - new.width) // 2, (size - new.height) // 2))
    return canvas


def grade(im: Image.Image) -> Image.Image:
    im = ImageEnhance.Brightness(im).enhance(1.18)
    im = ImageEnhance.Color(im).enhance(1.12)
    r, g, b = im.split()
    r = r.point(lambda v: min(255, int(v * 1.05 + 6)))
    b = b.point(lambda v: max(0, int(v * 0.95)))
    return Image.merge("RGB", (r, g, b))


def main() -> None:
    srcs = sorted([p for p in ASSETS.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])[:5]
    if len(srcs) < 5:
        raise SystemExit(f"素材不足 5 张：{len(srcs)}")
    for sub in ("control", "target"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(srcs, 1):
        with Image.open(src) as im:
            base = blur_pad(im, SIZE)
        stem = f"{i:03d}"
        base.save(OUT / "control" / f"{stem}.png")
        grade(base).save(OUT / "target" / f"{stem}.png")
        (OUT / "target" / f"{stem}.txt").write_text(CAPTION + "\n", encoding="utf-8")
        print(f"[ok] {stem} <- {src.name}")
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
