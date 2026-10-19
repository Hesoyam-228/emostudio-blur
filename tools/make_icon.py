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
# все размеры, которые Windows берёт при разных масштабах экрана (100–300%)
SIZES = [16, 20, 24, 32, 40, 48, 64, 72, 80, 96, 128, 256]

VARIANTS = {
    # тёмная плитка, белый глаз, сильное свечение бренда
    "glow": dict(bg=((58, 26, 98), (24, 13, 40), (10, 8, 16)), eye="white", glow=1.0),
    # то же, но свечение еле заметное и глаз чуть приглушён
    "soft": dict(bg=((48, 24, 80), (22, 13, 36), (10, 8, 15)), eye=(226, 219, 240), glow=0.28),
    # глаз пыльно-лавандовый с розовым, свечения почти нет
    "muted": dict(bg=((40, 22, 64), (20, 12, 31), (10, 8, 14)), eye=((170, 146, 214), (212, 150, 188)), glow=0.12),
    # совсем спокойный: нейтральная тёмная плитка, светло-серый глаз, без свечения
    "mono": dict(bg=((38, 32, 50), (21, 18, 28), (11, 10, 15)), eye=(196, 188, 212), glow=0),
}


def lerp(a, b, t):
    return np.asarray(a, np.float32) * (1 - t) + np.asarray(b, np.float32) * t


def eye_alpha(width, dilate=0):
    h = round(LOGO.height * width / LOGO.width)
    a = LOGO.resize((width, h), Image.LANCZOS)
    if dilate:
        a = a.filter(ImageFilter.MaxFilter(dilate * 2 + 1))
    return a


def brand_gradient(w, h, c1=(176, 108, 255), c2=(255, 102, 176)):
    t = np.linspace(0, 1, w)[None, :, None]
    rgb = lerp(c1, c2, t) * np.ones((h, 1, 1))
    im = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")
    return im


def tile(size, v):
    # рисуем в 8 раз крупнее и честно усредняем пиксели — так края чёткие, без «мыла»
    ss = 8 if size <= 96 else 4
    S = size * ss
    # фон: радиальный градиент от центра чуть выше середины к тёмным краям
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / (S - 1)
    d = np.clip(np.hypot(xx - 0.5, (yy - 0.44) * 1.05) / 0.72, 0, 1)[..., None]
    c0, c1, c2 = v["bg"]
    rgb = np.where(d < 0.5, lerp(c0, c1, d / 0.5), lerp(c1, c2, (d - 0.5) / 0.5))
    bg = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")

    eye_w = int(S * (0.96 if size <= 32 else 0.9 if size <= 96 else 0.84))
    # тонкие линии глаза на маленьких размерах чуть утолщаем, чтобы не рассыпались
    dil = round(ss * (0.55 if size <= 24 else 0.4 if size <= 40 else 0.25 if size <= 72 else 0))
    a = eye_alpha(eye_w, dil)
    ox, oy = (S - a.width) // 2, (S - a.height) // 2 + int(S * 0.01)

    # свечение за глазом (на маленьких размерах только мешает чёткости)
    if v["glow"] and size >= 64:
        g = Image.new("L", (S, S), 0)
        g.paste(a, (ox, oy))
        g = g.filter(ImageFilter.GaussianBlur(S * 0.07))
        glow = brand_gradient(S, S)
        glow.putalpha(g.point(lambda p: int(min(255, p * 2.2 * v["glow"]))))
        bg.alpha_composite(glow)

    # сам глаз
    e = v["eye"]
    if e == "grad":
        eye = brand_gradient(a.width, a.height)
        # верх чуть светлее, чтобы глаз не терялся
        eye.alpha_composite(Image.new("RGBA", a.size, (255, 255, 255, 90)))
    elif isinstance(e, tuple) and isinstance(e[0], tuple):
        eye = brand_gradient(a.width, a.height, *e)
    else:
        eye = Image.new("RGBA", a.size, (255, 255, 255, 255) if e == "white" else e + (255,))
    eye.putalpha(a)
    bg.alpha_composite(eye, (ox, oy))

    # скругление и тонкая светлая кромка
    m = Image.new("L", (S, S), 0)
    r = int(S * 0.225)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, S - 1, S - 1), radius=r, fill=255)
    edge = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(edge).rounded_rectangle((0, 0, S - 1, S - 1), radius=r,
                                           outline=(255, 255, 255, 34), width=max(ss, int(S * 0.012)))
    bg.alpha_composite(edge)
    bg.putalpha(m)
    out = bg.reduce(ss)
    if size <= 48:
        out = out.filter(ImageFilter.UnsharpMask(radius=0.6, percent=70, threshold=0))
    return out


def preview(path):
    sizes = [96, 48, 32, 24]
    row_h = 120
    W = 70 + sum(sz + 26 for sz in sizes) + 20
    out = Image.new("RGBA", (W * 2, row_h * len(VARIANTS) + 20), (0, 0, 0, 255))
    d = ImageDraw.Draw(out)
    for half, (bgc, fg) in enumerate([((205, 214, 226, 255), (40, 40, 50, 255)), ((30, 30, 34, 255), (220, 220, 230, 255))]):
        d.rectangle((half * W, 0, half * W + W, out.height), fill=bgc)
        for row, name in enumerate(VARIANTS):
            y0 = 10 + row * row_h
            d.text((half * W + 22, y0 + 40), str(row + 1), fill=fg)
            x = half * W + 60
            for sz in sizes:
                out.alpha_composite(tile(sz, VARIANTS[name]), (x, y0 + (96 - sz) // 2 + 6))
                x += sz + 26
    out.save(path)


def main():
    if "--preview" in sys.argv:
        preview(sys.argv[sys.argv.index("--preview") + 1])
        return
    name = sys.argv[1] if len(sys.argv) > 1 else "soft"
    tiles = [tile(s, VARIANTS[name]) for s in SIZES]
    tiles[-1].save(ROOT / "assets" / "icon.ico", sizes=[(s, s) for s in SIZES], append_images=tiles[:-1])
    tiles[-1].save(ROOT / "assets" / "icon-256.png")
    print(f"icon.ico и icon-256.png обновлены (вариант {name})")


if __name__ == "__main__":
    main()
