"""Ядро: поиск интимных зон и блюр. Используется и приложением, и консольным скриптом."""
import sys
from collections import deque

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
LABELS = {"nipple_f": "nipples", "penis": "penis", "pussy": "pussy"}

DEFAULTS = {
    "parts": ["nipples", "penis", "pussy", "anus"],
    "penis_mode": "tip",   # "tip" = только головка, "full" = целиком
    "tip_size": 1.0,       # длина зоны головки (1.0 = головка и чуть ниже)
    "conf": 0.3,           # порог для сосков
    "conf_genitals": 0.2,  # порог для гениталий
    "pad": 0.15,           # расширение зоны
    "strength": 1.0,       # сила блюра
    "style": "blur",       # "blur" или "pixel"
}


def load_rgb(path):
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return np.array(img.convert("RGB"))


def save_rgb(rgb, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(rgb)
    if dst.suffix.lower() in (".jpg", ".jpeg", ".webp"):
        img.save(dst, quality=95)
    else:
        img.save(dst)


def _box_spot(box, kind, score, pad):
    x0, y0, x1, y1 = box
    return {
        "x": (x0 + x1) / 2, "y": (y0 + y1) / 2,
        "rx": (x1 - x0) / 2 * (1 + pad), "ry": (y1 - y0) / 2 * (1 + pad),
        "kind": kind, "score": round(float(score), 2),
    }


def _bfs(mask, sources):
    """Геодезическое расстояние внутри маски от набора стартовых пикселей."""
    dist = np.full(mask.shape, -1, np.int32)
    q = deque()
    for y, x in sources:
        dist[y, x] = 0
        q.append((y, x))
    h, w = mask.shape
    while q:
        y, x = q.popleft()
        dv = dist[y, x] + 1
        for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and dist[yy, xx] < 0:
                dist[yy, xx] = dv
                q.append((yy, xx))
    return dist


def find_tip(rgb, box, tip_size=1.0, body_hint=None):
    """
    Ищет головку внутри рамки члена: выделяет сам член (GrabCut),
    находит два его конца и берёт тот, что дальше от головы персонажа.
    Возвращает (x, y, r) или None.
    """
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    bw, bh = x1 - x0, y1 - y0
    if bw < 12 or bh < 12:
        return None
    m = int(max(bw, bh) * 0.18)
    ex0, ey0 = max(0, x0 - m), max(0, y0 - m)
    ex1, ey1 = min(W, x1 + m), min(H, y1 + m)

    # работаем в уменьшенном кропе — так быстрее
    crop = cv2.cvtColor(rgb[ey0:ey1, ex0:ex1], cv2.COLOR_RGB2BGR)
    scale = min(1.0, 260 / max(crop.shape[:2]))
    small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else crop
    sh, sw = small.shape[:2]
    bx0, by0 = int((x0 - ex0) * scale), int((y0 - ey0) * scale)
    bx1, by1 = int((x1 - ex0) * scale), int((y1 - ey0) * scale)

    gc = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    gc[by0:by1, bx0:bx1] = cv2.GC_PR_FGD
    gc[:2, :] = gc[-2:, :] = cv2.GC_BGD
    gc[:, :2] = gc[:, -2:] = cv2.GC_BGD
    try:
        cv2.grabCut(small, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    inside = np.zeros_like(fg)
    inside[by0:by1, bx0:bx1] = 1
    lab, n = ndimage.label(fg & inside)
    if n == 0:
        return None
    main = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)

    # два самых дальних друг от друга конца фигуры (по пути внутри неё)
    ys, xs = np.nonzero(main)
    d0 = _bfs(main, [(ys[0], xs[0])])
    ay, ax = np.unravel_index(d0.argmax(), d0.shape)
    da = _bfs(main, [(ay, ax)])
    by_, bx_ = np.unravel_index(da.argmax(), da.shape)
    far = da.max()
    if far < 5:
        return None

    # головка — тот конец, что дальше от головы персонажа
    if body_hint is not None:
        hx, hy = (body_hint[0] - ex0) * scale, (body_hint[1] - ey0) * scale
    else:
        hx, hy = sw / 2, -sh
    if (ax - hx) ** 2 + (ay - hy) ** 2 > (bx_ - hx) ** 2 + (by_ - hy) ** 2:
        ty, tx = ay, ax
    else:
        ty, tx = by_, bx_

    # толщина члена у кончика
    dt = cv2.distanceTransform(main.astype(np.uint8), cv2.DIST_L2, 3)
    tip_dist = _bfs(main, [(ty, tx)])
    near = (tip_dist >= 0) & (tip_dist < far * 0.3)
    thick = 2 * float(dt[near].max()) if near.any() else far * 0.3

    # зона головки: пиксели, геодезически близкие к кончику
    length = min(far * 0.6, thick * 1.25 * tip_size)
    zone = (tip_dist >= 0) & (tip_dist <= length)
    zy, zx = np.nonzero(zone)
    cx, cy = zx.mean(), zy.mean()
    r = np.sqrt(((zx - cx) ** 2 + (zy - cy) ** 2).max()) * 1.1
    r = max(r, thick * 0.6)
    return (cx / scale + ex0, cy / scale + ey0, r / scale)


RAW_CONF = 0.1

# вторая модель (аниме-арты, лицензия MIT) — нужна для зоны «анус», которой нет в первой
ANUS_REPO = "01miku/anime-nsfw-segm-yolo26"
ANUS_FILE = "nsfw-anime-nano-x640.onnx"
ANUS_SIZE = 640
ANUS_PAD = 0.35  # рамки у этой модели плотные — зону делаем чуть шире
_anus_session = None


def _anus_model():
    global _anus_session
    if _anus_session is None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(ANUS_REPO, ANUS_FILE)
        _anus_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _anus_session


def detect_anus(rgb, conf=RAW_CONF):
    """YOLO-сегментация: выход [4 рамка + 7 классов + 32 маски] на каждый якорь, класс 0 — anus."""
    sess = _anus_model()
    h, w = rgb.shape[:2]
    r = ANUS_SIZE / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    top, left = (ANUS_SIZE - nh) // 2, (ANUS_SIZE - nw) // 2
    canvas = np.full((ANUS_SIZE, ANUS_SIZE, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0].T
    score = out[:, 4]
    keep = (score >= conf) & (out[:, 4:11].argmax(1) == 0)
    if not keep.any():
        return []
    cx, cy, bw, bh = out[keep, :4].T
    sc = score[keep]
    xywh = np.stack([(cx - bw / 2 - left) / r, (cy - bh / 2 - top) / r, bw / r, bh / r], 1)
    idx = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), sc.tolist(), conf, 0.5)).flatten()
    return [((float(a), float(b), float(a + c), float(b + d)), "anus", float(sc[i]))
            for i, (a, b, c, d) in ((i, xywh[i]) for i in idx)]


def raw_detect(rgb):
    """Тяжёлая часть (нейросеть). Результат можно кэшировать и потом быстро пересобирать зоны."""
    from imgutils.detect import detect_censors, detect_heads

    pil = Image.fromarray(rgb)
    boxes = []
    for level in ("s", "n"):
        for box, label, score in detect_censors(pil, level=level, conf_threshold=RAW_CONF):
            boxes.append((tuple(float(v) for v in box), LABELS.get(label, label), float(score)))
    try:
        boxes += detect_anus(rgb)
    except Exception as e:  # без второй модели программа всё равно работает
        print(f"(зона «анус» недоступна: {e})", file=sys.stderr)
    head = None
    if any(k == "penis" for _, k, _ in boxes):
        heads = detect_heads(pil)
        if heads:
            (hx0, hy0, hx1, hy1), _, _ = max(heads, key=lambda d: d[2])
            head = ((hx0 + hx1) / 2, (hy0 + hy1) / 2)
    return {"boxes": boxes, "head": head}


def make_spots(rgb, raw, settings=None):
    """
    Быстрая часть: из сырых рамок делает зоны блюра по настройкам.
    Возвращает (spots, warnings).
    """
    s = {**DEFAULTS, **(settings or {})}
    parts = set(s["parts"])
    boxes = []
    for box, kind, score in raw["boxes"]:
        limit = s["conf"] if kind == "nipples" else s["conf_genitals"]
        if kind in parts and score >= limit:
            boxes.append((box, kind, score))
    boxes = _dedupe(boxes)

    hint = raw.get("head")
    nipples = [b for b in boxes if b[1] == "nipples"]
    if hint is None and nipples:
        hint = (np.mean([(b[0][0] + b[0][2]) / 2 for b in nipples]),
                np.mean([(b[0][1] + b[0][3]) / 2 for b in nipples]))

    spots, warnings = [], []
    for box, kind, score in boxes:
        if kind == "penis" and s["penis_mode"] == "tip":
            tip = find_tip(rgb, box, s["tip_size"], hint)
            if tip:
                x, y, r = tip
                r *= 1 + s["pad"]
                spots.append({"x": x, "y": y, "rx": r, "ry": r, "kind": kind, "score": round(score, 2)})
                continue
            warnings.append("Не нашёл головку — заблюрен весь член")
        spots.append(_box_spot(box, kind, score, s["pad"] + (ANUS_PAD if kind == "anus" else 0)))
    if not spots:
        warnings.append("Ничего не найдено")
    return spots, warnings


def detect(rgb, settings=None):
    return make_spots(rgb, raw_detect(rgb), settings)[0]


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0


def _dedupe(boxes):
    """Две модели часто находят одно и то же — оставляем более уверенное."""
    out = []
    for b in sorted(boxes, key=lambda b: -b[2]):
        if all(o[1] != b[1] or _iou(o[0], b[0]) < 0.4 for o in out):
            out.append(b)
    return out


def render(rgb, spots, strength=1.0, style="blur"):
    h, w = rgb.shape[:2]
    if not spots:
        return rgb
    mask = np.zeros((h, w), np.float32)
    for sp in spots:
        cv2.ellipse(mask, (int(sp["x"]), int(sp["y"])),
                    (max(1, int(sp["rx"])), max(1, int(sp["ry"]))), float(sp.get("angle", 0)), 0, 360, 1.0, -1)
    feather = max(3, int(min(h, w) * 0.012))
    mask = cv2.GaussianBlur(mask, (0, 0), feather)
    mask = np.clip(mask * 1.6, 0, 1)[..., None]

    if style == "pixel":
        block = max(4, int(min(h, w) * strength / 100 * 1.5))
        small = cv2.resize(rgb, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
        censored = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        sigma = max(2.0, min(h, w) * strength / 100)
        censored = cv2.GaussianBlur(rgb, (0, 0), sigma)

    out = rgb.astype(np.float32) * (1 - mask) + censored.astype(np.float32) * mask
    return out.astype(np.uint8)


def draw_debug(rgb, spots):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for sp in spots:
        cv2.ellipse(img, (int(sp["x"]), int(sp["y"])), (int(sp["rx"]), int(sp["ry"])),
                    float(sp.get("angle", 0)), 0, 360, (0, 0, 255), 2)
        cv2.putText(img, f'{sp["kind"]} {sp["score"]}', (int(sp["x"] - sp["rx"]), int(sp["y"] - sp["ry"]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def warmup():
    """Скачивает все нужные модели заранее, чтобы первый запуск не висел."""
    from imgutils.detect import detect_censors, detect_heads

    img = Image.new("RGB", (256, 256), (128, 128, 128))
    for level in ("s", "n"):
        detect_censors(img, level=level)
    detect_heads(img)
    _anus_model()


if __name__ == "__main__":
    if "--download" in sys.argv:
        print("Скачиваю модели распознавания (один раз, около 100 МБ)...")
        warmup()
        print("Модели готовы.")
