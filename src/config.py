"""Лёгкие настройки и константы. Без тяжёлых импортов — окно программы может открыться сразу."""

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

DEFAULTS = {
    "parts": ["nipples", "penis", "pussy", "anus"],
    "penis_mode": "tip",   # "tip" = только головка, "full" = целиком
    "tip_size": 1.0,       # длина зоны головки (1.0 = головка и чуть ниже)
    "conf": 0.3,           # порог для сосков
    "conf_genitals": 0.2,  # порог для гениталий
    "pad": 0.15,           # расширение зоны
    "strength": 1.0,       # сила блюра
    "style": "blur",       # "blur" или "pixel"
    "verify": True,        # перепроверять зоны второй моделью (меньше ложных)
    "gpu": True,           # считать нейросети на видеокарте, если она есть
}

RAW_CONF = 0.1
HEAD_MODEL = "head_detect_v2.0_n"

# вторая модель (аниме-арты, лицензия MIT): даёт зону «анус» и перепроверяет остальные
EXTRA_REPO = "01miku/anime-nsfw-segm-yolo26"
EXTRA_FILE = "nsfw-anime-nano-x640.onnx"
EXTRA_SIZE = 640
EXTRA_CLASSES = {0: "anus", 1: "nipples", 2: "penis", 3: "pussy"}
ANUS_PAD = 0.35  # рамки у этой модели плотные — зону делаем чуть шире

# меняется вместе с моделями — тогда кэш распознавания программы сбрасывается сам
MODELS_ID = f"censor_v1.0_s+n|{HEAD_MODEL}|{EXTRA_FILE}:all|{RAW_CONF}"
