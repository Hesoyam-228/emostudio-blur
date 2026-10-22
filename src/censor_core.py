"""Ядро: поиск интимных зон и блюр. Используется и приложением, и консольным скриптом."""
import os
import sys
import threading

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from config import (ANUS_PAD, DEFAULTS, EXTRA_CLASSES, EXTRA_FILE, EXTRA_REPO, EXTRA_SIZE, EXTS,  # noqa: F401
                    HEAD_MODEL, MODELS_ID, RAW_CONF)

LABELS = {"nipple_f": "nipples", "penis": "penis", "pussy": "pussy"}
# зона без подтверждения второй моделью остаётся, только если первая в ней очень уверена
VERIFY_KEEP = 0.6
# что вторая модель нашла, а первая нет, добавляем при такой уверенности
EXTRA_ADD = 0.5
# размер и число итераций GrabCut при поиске кончика (меньше — быстрее, но грубее)
TIP_RES = 160
TIP_ITERS = 3


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


_threads = os.cpu_count() or 4
_use_gpu = True
_device = "cpu"  # что реально используется: "gpu" или "cpu"


_gpu_lock = threading.Lock()  # одна очередь на все модели: DirectML падает от одновременных запросов
_gpu_sessions = {}  # одна сессия на модель: imgutils открывает модель заново в каждом потоке


class _LockedSession:
    """
    Видеокарта обрабатывает запросы по одному — все потоки делят одну сессию и встают в общую очередь.
    Если видеокарта сбоит, запрос пересчитывается на процессоре, а после двух сбоев подряд модель уходит туда насовсем.
    """

    def __init__(self, sess, path):
        self._s, self._path = sess, path
        self._cpu = None
        self._fails = 0

    def run(self, *a, **k):
        if self._fails < 2:
            try:
                with _gpu_lock:
                    out = self._s.run(*a, **k)
                self._fails = 0
                return out
            except Exception as e:
                self._fails += 1
                print(f"(сбой видеокарты, пересчитываю на процессоре: {type(e).__name__} {e})", file=sys.stderr)
        with _gpu_lock:
            if self._cpu is None:
                self._cpu = _cpu_session(self._path)
        return self._cpu.run(*a, **k)

    def __getattr__(self, name):
        return getattr(self._s, name)


def _new_session(path):
    """Нейросеть на видеокарте (DirectML), а если не вышло — на процессоре."""
    global _device
    import onnxruntime as ort

    if _use_gpu and "DmlExecutionProvider" in ort.get_available_providers():
        with _gpu_lock:  # пока сессия создаётся, видеокарта не должна ничего считать
            if path in _gpu_sessions:
                return _gpu_sessions[path]
            try:
                o = ort.SessionOptions()
                o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                o.enable_mem_pattern = False  # требование DirectML
                o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                sess = ort.InferenceSession(path, o, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
                if sess.get_providers()[0] == "DmlExecutionProvider":
                    _device = "gpu"
                    _gpu_sessions[path] = _LockedSession(sess, path)
                    return _gpu_sessions[path]
            except Exception as e:
                print(f"(видеокарта недоступна, считаю на процессоре: {e})", file=sys.stderr)
    return _cpu_session(path)


def _cpu_session(path):
    import onnxruntime as ort

    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.intra_op_num_threads = _threads
    o.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


def configure(threads=None, use_gpu=True):
    """
    Вызывать до первого распознавания. threads — сколько ядер даём каждой нейросети на процессоре
    (программа обрабатывает несколько картинок сразу, и если каждая модель берёт все ядра, они мешают друг другу).
    """
    global _threads, _use_gpu
    if threads:
        _threads = max(1, int(threads))
    _use_gpu = bool(use_gpu)
    import imgutils.utils.onnxruntime as iort

    iort._open_onnx_model = lambda ckpt, provider=None, use_cpu=True: _new_session(ckpt)


def set_threads(n):
    configure(threads=n, use_gpu=_use_gpu)


def device():
    return _device


_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)


def _bfs(mask, sources):
    """
    Геодезическое расстояние внутри маски от набора стартовых пикселей (4-связность).
    Волна расширяется целым фронтом через cv2.dilate — в десятки раз быстрее обхода по пикселям.
    """
    m = mask.astype(np.uint8)
    dist = np.full(m.shape, -1, np.int32)
    front = np.zeros_like(m)
    for y, x in sources:
        front[y, x] = 1
    front &= m
    visited = front.copy()
    d = 0
    while front.any():
        dist[front.astype(bool)] = d
        d += 1
        front = cv2.dilate(front, _CROSS) & m & (1 - visited)
        visited |= front
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
    scale = min(1.0, TIP_RES / max(crop.shape[:2]))
    small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else crop
    sh, sw = small.shape[:2]
    bx0, by0 = int((x0 - ex0) * scale), int((y0 - ey0) * scale)
    bx1, by1 = int((x1 - ex0) * scale), int((y1 - ey0) * scale)

    gc = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    gc[by0:by1, bx0:bx1] = cv2.GC_PR_FGD
    gc[:2, :] = gc[-2:, :] = cv2.GC_BGD
    gc[:, :2] = gc[:, -2:] = cv2.GC_BGD
    try:
        cv2.grabCut(small, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), TIP_ITERS, cv2.GC_INIT_WITH_MASK)
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
        ty, tx, tip_dist = ay, ax, da  # расстояния от этого конца уже посчитаны
    else:
        ty, tx = by_, bx_
        tip_dist = _bfs(main, [(ty, tx)])

    # толщина члена у кончика
    dt = cv2.distanceTransform(main.astype(np.uint8), cv2.DIST_L2, 3)
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


_extra_session = None
_extra_lock = threading.Lock()


def _extra_model():
    global _extra_session
    with _extra_lock:
        if _extra_session is None:
            from huggingface_hub import hf_hub_download

            _extra_session = _new_session(hf_hub_download(EXTRA_REPO, EXTRA_FILE))
    return _extra_session


def detect_extra(rgb, conf=RAW_CONF):
    """
    Вторая модель (YOLO-сегментация): выход [4 рамка + 7 классов + 32 маски] на каждый якорь.
    Берём классы anus / nipple / penis / vagina: анус — как зону, остальные — для перепроверки.
    """
    sess = _extra_model()
    h, w = rgb.shape[:2]
    r = EXTRA_SIZE / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    top, left = (EXTRA_SIZE - nh) // 2, (EXTRA_SIZE - nw) // 2
    canvas = np.full((EXTRA_SIZE, EXTRA_SIZE, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0].T
    cls = out[:, 4:11].argmax(1)
    score = out[:, 4:11].max(1)
    res = []
    for c, kind in EXTRA_CLASSES.items():
        keep = (score >= conf) & (cls == c)
        if not keep.any():
            continue
        cx, cy, bw, bh = out[keep, :4].T
        sc = score[keep]
        xywh = np.stack([(cx - bw / 2 - left) / r, (cy - bh / 2 - top) / r, bw / r, bh / r], 1)
        for i in np.array(cv2.dnn.NMSBoxes(xywh.tolist(), sc.tolist(), conf, 0.5)).flatten():
            a, b, bw_, bh_ = xywh[i]
            res.append(((float(a), float(b), float(a + bw_), float(b + bh_)), kind, float(sc[i])))
    return res


def detect_anus(rgb, conf=RAW_CONF):
    return [b for b in detect_extra(rgb, conf) if b[1] == "anus"]


def warmup_sessions():
    """Загрузить все нейросети заранее (в фоне при старте программы), чтобы первая картинка не ждала."""
    from imgutils.detect import detect_censors, detect_heads

    img = Image.new("RGB", (64, 64), (128, 128, 128))
    for level in ("s", "n"):
        detect_censors(img, level=level)
    detect_heads(img, model_name=HEAD_MODEL)
    detect_extra(np.zeros((64, 64, 3), np.uint8))


def raw_detect(rgb):
    """Тяжёлая часть (нейросеть). Результат можно кэшировать и потом быстро пересобирать зоны."""
    from imgutils.detect import detect_censors, detect_heads

    pil = Image.fromarray(rgb)
    boxes = []
    for level in ("s", "n"):
        for box, label, score in detect_censors(pil, level=level, conf_threshold=RAW_CONF):
            boxes.append((tuple(float(v) for v in box), LABELS.get(label, label), float(score)))
    try:
        extra = detect_extra(rgb)
    except Exception as e:  # без второй модели программа всё равно работает
        print(f"(вторая модель недоступна: {e})", file=sys.stderr)
        extra = None
    head = None
    if any(k == "penis" for _, k, _ in boxes):
        # голова нужна только как ориентир «где тело» — хватает лёгкой модели
        heads = detect_heads(pil, model_name=HEAD_MODEL)
        if heads:
            (hx0, hy0, hx1, hy1), _, _ = max(heads, key=lambda d: d[2])
            head = ((hx0 + hx1) / 2, (hy0 + hy1) / 2)
    return {"boxes": boxes, "extra": extra, "head": head}


def make_spots(rgb, raw, settings=None):
    """
    Быстрая часть: из сырых рамок делает зоны блюра по настройкам.
    Возвращает (spots, warnings).
    """
    s = {**DEFAULTS, **(settings or {})}
    parts = set(s["parts"])
    extra = raw.get("extra")

    def limit(kind):
        return s["conf"] if kind == "nipples" else s["conf_genitals"]

    boxes = []
    for box, kind, score in raw["boxes"]:
        if kind not in parts or score < limit(kind):
            continue
        # перепроверка: слабую зону без подтверждения второй моделью отбрасываем (пальцы, складки и т.п.)
        if s.get("verify", True) and extra is not None and score < VERIFY_KEEP and not _confirmed(box, kind, extra):
            continue
        boxes.append((box, kind, score))
    for box, kind, score in extra or []:
        if kind not in parts:
            continue
        if kind == "anus":
            if score >= limit(kind):
                boxes.append((box, kind, score))
        elif score >= max(EXTRA_ADD, limit(kind)) and not any(
                k == kind and _overlaps(box, b) for b, k, _ in boxes):
            boxes.append((box, kind, score))  # первая модель это пропустила
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


def _overlaps(a, b, min_iou=0.05):
    """Рамки заметно пересекаются или центр одной лежит внутри другой."""
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    def inside(c, r):
        return r[0] <= c[0] <= r[2] and r[1] <= c[1] <= r[3]

    return _iou(a, b) > min_iou or inside(ca, b) or inside(cb, a)


def _confirmed(box, kind, extra):
    # для подтверждения нужно настоящее совпадение, а не касание краем (иначе пройдёт рамка «на всю ступню»)
    return any(k == kind and _overlaps(box, b, 0.15) for b, k, _ in extra)


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


def _merge_rects(rects):
    """Сливаем пересекающиеся прямоугольники, пока не останутся только раздельные."""
    rects = [list(r) for r in rects]
    merged = True
    while merged:
        merged = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    rects[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    rects.pop(j)
                    merged = True
                    break
            if merged:
                break
    return rects


def render(rgb, spots, strength=1.0, style="blur"):
    h, w = rgb.shape[:2]
    if not spots:
        return rgb
    feather = max(3, int(min(h, w) * 0.012))
    sigma = max(2.0, min(h, w) * strength / 100)

    if style == "pixel":
        # пикселизация дешёвая, считаем на всей картинке, чтобы сетка квадратов совпадала с предпросмотром
        block = max(4, int(min(h, w) * strength / 100 * 1.5))
        small = cv2.resize(rgb, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
        censored_full = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        margin = 3 * feather
    else:
        censored_full = None
        margin = 3 * feather + int(4 * sigma)  # запас, чтобы размытие у края кусочка не отличалось

    # блюрим не всю картинку, а только кусочки вокруг зон — результат тот же, но в разы быстрее
    rects = []
    for sp in spots:
        e = max(sp["rx"], sp["ry"]) + margin
        rects.append((max(0, int(sp["x"] - e)), max(0, int(sp["y"] - e)),
                      min(w, int(sp["x"] + e) + 1), min(h, int(sp["y"] + e) + 1)))
    out = rgb.copy()
    for x0, y0, x1, y1 in _merge_rects(rects):
        if x1 <= x0 or y1 <= y0:
            continue
        crop = rgb[y0:y1, x0:x1]
        mask = np.zeros(crop.shape[:2], np.float32)
        for sp in spots:
            cv2.ellipse(mask, (int(sp["x"]) - x0, int(sp["y"]) - y0),
                        (max(1, int(sp["rx"])), max(1, int(sp["ry"]))), float(sp.get("angle", 0)), 0, 360, 1.0, -1)
        mask = np.clip(cv2.GaussianBlur(mask, (0, 0), feather) * 1.6, 0, 1)[..., None]
        if censored_full is not None:
            censored = censored_full[y0:y1, x0:x1]
        else:
            censored = cv2.GaussianBlur(crop, (0, 0), sigma)
        out[y0:y1, x0:x1] = (crop.astype(np.float32) * (1 - mask) + censored.astype(np.float32) * mask).astype(np.uint8)
    return out


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
    configure(use_gpu=False)  # для скачивания видеокарта не нужна
    warmup_sessions()


if __name__ == "__main__":
    if "--download" in sys.argv:
        print("Скачиваю модели распознавания (один раз, около 100 МБ)...")
        warmup()
        print("Модели готовы.")
