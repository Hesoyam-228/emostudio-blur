"""
Собирает иконку приложения из логотипа assets/logo.png (белый глаз на прозрачном фоне).

    venv\\Scripts\\python tools\\make_icon.py            -> assets/icon.ico и assets/icon-256.png
    venv\\Scripts\\python tools\\make_icon.py --preview  -> только картинка-сравнение вариантов
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
LOGO = Image.open(ROOT / "assets" / "logo.png").getchannel("A")
SIZES = [16, 24, 32, 48, 64, 128, 256]

VARIANTS = {
    # тёмная плитка, белый глаз, свечение бренда
    "glow": dict(bg=((58, 26, 98), (24, 13, 40), (10, 8, 16)), eye="white", glow=1.0),
    # тёмная плитка, глаз в градиенте бренда
    "gradient": dict(bg=((44, 22, 74), (20, 12, 34), (9, 7, 14)), eye="grad", glow=0.8),
    # насыщенная, но заметно темнее прежней
    "deep": dict(bg=((92, 38, 168), (70, 22, 104), (46, 12, 60)), eye="white", glow=0.55),
}


def lerp(a, b, t):
    return np.asarray(a, np.float32) * (1 - t) + np.asarray(b, np.float32) * t


def eye_alpha(width, dilate=0):
    h = round(LOGO.height * width / LOGO.width)
    a = LOGO.resize((width, h), Image.LANCZOS)
    if dilate:
        a = a.filter(ImageFilter.MaxFilter(dilate * 2 + 1))
    return a


def brand_gradient(w, h, alpha=255):
    t = np.linspace(0, 1, w)[None, :, None]
    rgb = lerp((176, 108, 255), (255, 102, 176), t) * np.ones((h, 1, 1))
    im = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")
    return im


def tile(size, v):
    ss = 4 if size < 128 else 2
    S = size * ss
    # фон: радиальный градиент от центра чуть выше середины к тёмным краям
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / (S - 1)
    d = np.clip(np.hypot(xx - 0.5, (yy - 0.44) * 1.05) / 0.72, 0, 1)[..., None]
    c0, c1, c2 = v["bg"]
    rgb = np.where(d < 0.5, lerp(c0, c1, d / 0.5), lerp(c1, c2, (d - 0.5) / 0.5))
    bg = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")

    eye_w = int(S * (0.92 if size <= 32 else 0.84))
    dil = (ss if size <= 24 else ss // 2) if size <= 48 else 0
    a = eye_alpha(eye_w, dil)
    ox, oy = (S - a.width) // 2, (S - a.height) // 2 + int(S * 0.01)

    # свечение за глазом
    if v["glow"]:
        g = Image.new("L", (S, S), 0)
        g.paste(a, (ox, oy))
        g = g.filter(ImageFilter.GaussianBlur(S * 0.07))
        glow = brand_gradient(S, S)
        glow.putalpha(g.point(lambda p: int(min(255, p * 2.2 * v["glow"]))))
        bg.alpha_composite(glow)

    # сам глаз
    if v["eye"] == "grad":
        eye = brand_gradient(a.width, a.height)
        # верх чуть светлее, чтобы глаз не терялся
        hi = Image.new("RGBA", a.size, (255, 255, 255, 90))
        eye.alpha_composite(hi)
    else:
        eye = Image.new("RGBA", a.size, (255, 255, 255, 255))
    eye.putalpha(a)
    bg.alpha_composite(eye, (ox, oy))

    # скругление и тонкая светлая кромка
    m = Image.new("L", (S, S), 0)
    r = int(S * 0.225)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, S - 1, S - 1), radius=r, fill=255)
    edge = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(edge).rounded_rectangle((0, 0, S - 1, S - 1), radius=r,
                                           outline=(255, 255, 255, 34), width=max(1, int(S * 0.012)))
    bg.alpha_composite(edge)
    bg.putalpha(m)
    return bg.resize((size, size), Image.LANCZOS)


def preview(path):
    sizes = [128, 64, 48, 32, 24, 16]
    row_h = 150
    W = 40 + sum(s + 24 for s in sizes) + 40
    out = Image.new("RGBA", (W * 2, row_h * len(VARIANTS) + 20), (0, 0, 0, 255))
    for half, bgc in enumerate([(32, 32, 36, 255), (214, 220, 228, 255)]):
        ImageDraw.Draw(out).rectangle((half * W, 0, half * W + W, out.height), fill=bgc)
        for row, (name, v) in enumerate(VARIANTS.items()):
            x = half * W + 40
            y0 = 10 + row * row_h
            for s in sizes:
                out.alpha_composite(tile(s, v), (x, y0 + (128 - s) // 2 + 6))
                x += s + 24
    out.save(path)


def main():
    if "--preview" in sys.argv:
        preview(sys.argv[sys.argv.index("--preview") + 1])
        return
    name = sys.argv[1] if len(sys.argv) > 1 else "glow"
    tiles = [tile(s, VARIANTS[name]) for s in SIZES]
    tiles[-1].save(ROOT / "assets" / "icon.ico", sizes=[(s, s) for s in SIZES], append_images=tiles[:-1])
    tiles[-1].save(ROOT / "assets" / "icon-256.png")
    print(f"icon.ico и icon-256.png обновлены (вариант {name})")


if __name__ == "__main__":
    main()
