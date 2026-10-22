"""EmoStudio Blur — приложение с интерфейсом. Запуск: «EmoStudio Blur.bat» или ярлык на рабочем столе."""
import base64
import ctypes
import io
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

APP_NAME = "EmoStudio Blur"
APP_ID = "EmoStudio.Blur"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ICON = ROOT / "assets" / "icon.ico"
DATA_DIR = Path(os.environ.get("APPDATA", Path.home())) / APP_NAME
DATA_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = DATA_DIR / "settings.json"
REPO = "Hesoyam-228/emostudio-blur"
# адрес можно подменить для проверки обновления без публикации
VERSION_URL = os.environ.get("EMOSTUDIO_VERSION_URL", f"https://raw.githubusercontent.com/{REPO}/main/version.txt")

# без консоли (pythonw) stdout/stderr отсутствуют — пишем всё в лог, чтобы ничего не падало
if sys.stdout is None or sys.stderr is None:
    _log = open(DATA_DIR / "log.txt", "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or _log
    sys.stderr = sys.stderr or _log
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

import webview  # noqa: E402
from webview.dom import DOMEventHandler  # noqa: E402
from PIL import Image  # noqa: E402

import config  # noqa: E402


class _LazyCore:
    """
    censor_core тянет OpenCV, SciPy и нейросети — это секунды при первом запуске после включения компа.
    Поэтому импортируем его не до окна, а в фоне, когда окно уже открыто.
    """
    _mod = None
    _lock = threading.Lock()

    def __getattr__(self, name):
        if _LazyCore._mod is None:
            with _LazyCore._lock:
                if _LazyCore._mod is None:
                    import censor_core
                    _LazyCore._mod = censor_core
        return getattr(_LazyCore._mod, name)


core = _LazyCore()

# сколько картинок обрабатывать одновременно (нейросети и так используют несколько ядер)
WORKERS = int(os.environ.get("EMOSTUDIO_WORKERS", 0)) or max(2, min(4, (os.cpu_count() or 4) // 3))
CACHE_FILE = DATA_DIR / "detect_cache.jsonl"


class DetectCache:
    """
    Результаты нейросети на диске: повторно открытая папка распознаётся мгновенно.
    Ключ — путь, размер и время изменения файла плюс версия моделей; поменялся файл — считаем заново.
    """

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.data = {}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                try:
                    rec = json.loads(line)
                    self.data[rec["k"]] = rec["raw"]
                except Exception:
                    pass
            if len(lines) > 2 * len(self.data) + 100 or len(self.data) > 20000:
                keep = list(self.data.items())[-10000:]
                self.data = dict(keep)
                path.write_text("".join(json.dumps({"k": k, "raw": v}) + "\n" for k, v in keep), encoding="utf-8")
        except FileNotFoundError:
            pass
        except Exception:
            traceback.print_exc()

    @staticmethod
    def key(path):
        st = Path(path).stat()
        return f"{config.MODELS_ID}|{Path(path).resolve()}|{st.st_size}|{st.st_mtime_ns}"

    def get(self, path):
        try:
            return self.data.get(self.key(path))
        except OSError:
            return None

    def put(self, path, raw):
        try:
            k = self.key(path)
        except OSError:
            return
        with self.lock:
            self.data[k] = raw
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "raw": raw}) + "\n")
            except Exception:
                pass


# настройки, от которых зависят зоны блюра (меняются — пересобираем зоны)
SPOT_KEYS = {"parts", "penis_mode", "tip_size", "conf", "conf_genitals", "pad", "verify"}


def read_version():
    try:
        return (ROOT / "version.txt").read_text(encoding="utf-8").strip() or "0.0.0"
    except Exception:
        return "0.0.0"


def version_key(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3]) or (0,)


def load_settings():
    s = dict(config.DEFAULTS, out_dir="")
    try:
        s.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    except Exception:
        pass
    # в тестовой сборке 1.3.0 были стили «полоса», «сердце» и «стикер» — их больше нет
    if s.get("style") not in ("blur", "pixel"):
        s["style"] = "blur"
    s.pop("sticker_path", None)
    # версия 2 добавила зону «анус» — включаем её и тем, у кого настройки сохранены раньше
    if s.get("parts_version", 1) < 2:
        if "anus" not in s["parts"]:
            s["parts"] = list(s["parts"]) + ["anus"]
        s["parts_version"] = 2
    return s


class Item:
    def __init__(self, iid, path, root):
        self.id = iid
        self.path = Path(path)
        self.root = Path(root)
        self.status = "queued"   # queued / working / ready / saved / error
        self.raw = None
        self.spots = []
        self.warnings = []
        self.edited = False
        self.thumb = None
        self.error = ""
        self.out = ""
        self.autosave = False  # пришло из папки, за которой следим, — сохранить сразу после распознавания

    def public(self):
        return {
            "id": self.id, "name": self.path.name, "path": str(self.path), "status": self.status,
            "spots": self.spots, "warnings": self.warnings, "edited": self.edited,
            "thumb": self.thumb, "error": self.error, "out": self.out,
        }


class Api:
    def __init__(self):
        self._settings = load_settings()
        self._items = {}
        self._order = []
        self._next_id = 1
        self._lock = threading.Lock()
        self._tasks = queue.Queue()
        self._window = None
        self._version = read_version()
        self._latest = None
        self._cache = DetectCache(CACHE_FILE)
        self._ready = threading.Event()
        self._engine = {"state": "loading", "device": None}
        threading.Thread(target=self._prepare_engine, daemon=True).start()
        for _ in range(WORKERS):
            threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._check_update, daemon=True).start()
        threading.Thread(target=self._first_run_shortcut, daemon=True).start()
        threading.Thread(target=self._watch_loop, daemon=True).start()

    # ---------- нейросети ----------
    def _prepare_engine(self):
        """Грузим ядро и все нейросети в фоне сразу после старта — первая картинка уже не ждёт."""
        try:
            core.configure(threads=max(1, (os.cpu_count() or 4) // WORKERS),
                           use_gpu=self._settings.get("gpu", True))
            core.warmup_sessions()
            self._engine = {"state": "ready", "device": core.device()}
        except Exception as e:
            traceback.print_exc()
            self._engine = {"state": "error", "device": None, "error": str(e)}
        self._ready.set()
        self._push("onEngine", self._engine)

    # ---------- ярлык ----------
    def _first_run_shortcut(self):
        """Если установка не успела сделать ярлык — делаем его сами, но только один раз."""
        if self._settings.get("shortcut_checked"):
            return
        try:
            import shortcuts
            if not shortcuts.desktop_shortcut_exists():
                made, _ = shortcuts.create_all()
                if made:
                    self._push("onToast", "Создал ярлык EmoStudio Blur на рабочем столе")
        except Exception:
            traceback.print_exc()
        self.update_settings({"shortcut_checked": True})

    def make_shortcuts(self):
        try:
            import shortcuts
            made, errors = shortcuts.create_all()
            return {"ok": bool(made), "error": "; ".join(errors)}
        except Exception as e:
            traceback.print_exc()
            return {"ok": False, "error": str(e)}

    # ---------- обновления ----------
    def _fetch_latest(self):
        req = urllib.request.Request(VERSION_URL, headers={"Cache-Control": "no-cache"})
        return urllib.request.urlopen(req, timeout=8).read().decode("utf-8").strip()

    def _check_update(self):
        """Тихо узнаём номер последней версии на GitHub. Нет интернета — просто молчим."""
        try:
            latest = self._fetch_latest()
        except Exception:
            return
        if latest and version_key(latest) > version_key(self._version):
            self._latest = latest
            self._push("onUpdate", {"current": self._version, "latest": latest})

    def check_update_now(self):
        """Кнопка «Проверить обновления»."""
        try:
            latest = self._fetch_latest()
        except Exception as e:
            return {"current": self._version, "error": str(e)}
        available = bool(latest) and version_key(latest) > version_key(self._version)
        if available:
            self._latest = latest
        return {"current": self._version, "latest": latest, "available": available}

    def start_update(self):
        """Открываем окно обновления и закрываем программу — обновление перезапустит её само."""
        script = ROOT / "tools" / "update.ps1"
        if not script.exists():
            return False
        subprocess.Popen(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
             "-Relaunch", "-WaitPid", str(os.getpid())],
            cwd=str(ROOT), creationflags=subprocess.CREATE_NEW_CONSOLE)
        if self._window:
            threading.Timer(0.6, self._window.destroy).start()
        return True

    # ---------- общение с окном ----------
    def _push(self, fn, payload):
        if self._window:
            try:
                self._window.evaluate_js(f"window.{fn} && window.{fn}({json.dumps(payload, ensure_ascii=False)})")
            except Exception:
                pass

    def _update(self, item):
        self._push("onItem", item.public())

    def _progress(self):
        with self._lock:
            busy = sum(1 for i in self._items.values() if i.status in ("queued", "working"))
        self._push("onBusy", {"busy": busy, "total": len(self._items)})

    # ---------- фоновая работа ----------
    def _worker(self):
        self._ready.wait()  # пока нейросети грузятся, задачи просто копятся в очереди
        while True:
            kind, iid = self._tasks.get()
            item = self._items.get(iid)
            if item is None:
                continue
            try:
                if kind == "detect":
                    self._detect(item)
                elif kind == "respot":
                    if not item.edited and item.raw is not None:
                        rgb = core.load_rgb(item.path)
                        item.spots, item.warnings = core.make_spots(rgb, item.raw, self._settings)
                        if item.status == "saved":
                            item.status = "ready"
                elif kind == "save":
                    if item.raw is None:
                        self._detect(item)
                    self._save(item)
            except Exception as e:
                traceback.print_exc()
                item.status = "error"
                item.error = str(e) or type(e).__name__
            self._update(item)
            self._progress()

    def _detect(self, item):
        item.status = "working"
        self._update(item)
        rgb = core.load_rgb(item.path)
        if item.thumb is None:
            im = Image.fromarray(rgb)
            im.thumbnail((240, 240))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=80)
            item.thumb = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        raw = self._cache.get(item.path)
        if raw is None:
            raw = core.raw_detect(rgb)
            self._cache.put(item.path, raw)
        item.raw = raw
        if not item.edited:
            item.spots, item.warnings = core.make_spots(rgb, item.raw, self._settings)
        item.status = "ready"
        if item.autosave:
            item.autosave = False
            self._tasks.put(("save", item.id))

    def _out_path(self, item):
        out_dir = self._settings.get("out_dir")
        root = Path(out_dir) if out_dir else item.root / "censored"
        try:
            rel = item.path.relative_to(item.root)
        except ValueError:
            rel = Path(item.path.name)
        return root / rel

    def _save(self, item):
        rgb = core.load_rgb(item.path)
        out = core.render(rgb, item.spots, self._settings["strength"], self._settings["style"])
        dst = self._out_path(item)
        core.save_rgb(out, dst)
        item.out = str(dst)
        item.status = "saved"

    # ---------- методы для интерфейса ----------
    def get_state(self):
        return {"settings": self._settings, "items": [self._items[i].public() for i in self._order],
                "version": self._version, "update": self._latest, "engine": self._engine}

    # ---------- слежение за папкой ----------
    def _watch_files(self, root):
        out_dir = self._settings.get("out_dir")
        out = Path(out_dir).resolve() if out_dir else None
        for f in root.rglob("*"):
            if f.suffix.lower() not in config.EXTS or "censored" in f.relative_to(root).parts:
                continue
            if out is not None and (out == f.resolve().parent or out in f.resolve().parents):
                continue  # свои же результаты не подхватываем
            yield f

    def _watch_loop(self):
        """Раз в 2 секунды ищем новые арты. Файл берём, когда его размер перестал меняться — ComfyUI дописал."""
        root, seen, pending = None, set(), {}
        while True:
            time.sleep(2)
            d = self._settings.get("watch_dir")
            try:
                if not d or not Path(d).is_dir():
                    root = None
                    continue
                if root != Path(d):  # начали следить: всё, что уже лежит в папке, не трогаем
                    root, pending = Path(d), {}
                    seen = set(self._watch_files(root))
                    continue
                for f in self._watch_files(root):
                    if f in seen:
                        continue
                    size = f.stat().st_size
                    if size > 0 and pending.get(f) == size:
                        seen.add(f)
                        pending.pop(f, None)
                        self._add_files([(f, root)], autosave=True)
                    else:
                        pending[f] = size
            except Exception:
                traceback.print_exc()

    def start_watch(self):
        res = self._window.create_file_dialog(webview.FileDialog.FOLDER)
        if res:
            self.update_settings({"watch_dir": res[0]})
        return self._settings

    def stop_watch(self):
        return self.update_settings({"watch_dir": ""})

    def _add_files(self, files, autosave=False):
        added = []
        for f, root in files:
            with self._lock:
                if any(i.path == f for i in self._items.values()):
                    continue
                item = Item(self._next_id, f, root)
                item.autosave = autosave
                self._next_id += 1
                self._items[item.id] = item
                self._order.append(item.id)
            added.append(item.public())
            self._tasks.put(("detect", item.id))
        if added:
            self._push("onAdded", added if not autosave else [dict(i, watched=True) for i in added])
        self._progress()
        return added

    def add_paths(self, paths):
        added = []
        for p in map(Path, paths or []):
            if p.is_dir():
                files = [(f, p) for f in sorted(p.rglob("*"))
                         if f.suffix.lower() in config.EXTS and "censored" not in f.relative_to(p).parts]
            elif p.suffix.lower() in config.EXTS:
                files = [(p, p.parent)]
            else:
                files = []
            added += self._add_files(files)
        if not added:
            self._push("onAdded", [])
        return len(added)

    def pick_folder(self):
        res = self._window.create_file_dialog(webview.FileDialog.FOLDER)
        return self.add_paths(res if res else [])

    def pick_files(self):
        res = self._window.create_file_dialog(
            webview.FileDialog.OPEN, allow_multiple=True,
            file_types=("Картинки (*.png;*.jpg;*.jpeg;*.webp;*.bmp)", "Все файлы (*.*)"))
        return self.add_paths(res if res else [])

    def get_image(self, iid):
        item = self._items[iid]
        data = item.path.read_bytes()
        ext = item.path.suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg"}.get(ext, ext)
        return {"url": f"data:image/{mime};base64," + base64.b64encode(data).decode(), **item.public()}

    def set_spots(self, iid, spots):
        item = self._items[iid]
        item.spots = spots
        item.edited = True
        if item.status == "saved":
            item.status = "ready"
        self._update(item)

    def reset_spots(self, iid):
        item = self._items[iid]
        item.edited = False
        self._tasks.put(("respot", iid))

    def update_settings(self, patch):
        changed = {k for k, v in patch.items() if self._settings.get(k) != v}
        self._settings.update(patch)
        try:
            SETTINGS_FILE.write_text(json.dumps(self._settings, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        if changed & SPOT_KEYS:
            for iid in self._order:
                item = self._items[iid]
                if item.raw is not None and not item.edited:
                    self._tasks.put(("respot", iid))
        elif changed & {"strength", "style", "out_dir"}:
            for item in self._items.values():
                if item.status == "saved":
                    item.status = "ready"
                    self._update(item)
        return self._settings

    def reset_settings(self):
        keep = self._settings.get("out_dir", "")
        return self.update_settings(dict(config.DEFAULTS, out_dir=keep))

    def save(self, iid):
        self._items[iid].status = "queued"
        self._tasks.put(("save", iid))
        self._progress()

    def save_all(self):
        n = 0
        for iid in self._order:
            item = self._items[iid]
            if item.status in ("ready", "error"):
                item.status = "queued"
                self._update(item)
                self._tasks.put(("save", iid))
                n += 1
        self._progress()
        return n

    def choose_output(self):
        res = self._window.create_file_dialog(webview.FileDialog.FOLDER)
        if res:
            return self.update_settings({"out_dir": res[0]})
        return self._settings

    def clear_output(self):
        return self.update_settings({"out_dir": ""})

    def open_output(self, iid=None):
        if iid and self._items[iid].out:
            os.startfile(Path(self._items[iid].out).parent)
            return
        if self._settings.get("out_dir"):
            target = Path(self._settings["out_dir"])
        elif self._order:
            target = self._out_path(self._items[self._order[0]]).parent
        else:
            return
        target.mkdir(parents=True, exist_ok=True)
        os.startfile(target)

    def remove(self, iid):
        with self._lock:
            self._items.pop(iid, None)
            if iid in self._order:
                self._order.remove(iid)
        self._progress()

    def clear(self):
        with self._lock:
            self._items.clear()
            self._order.clear()
        self._progress()


def bind_drop(window, api):
    def on_drop(e):
        files = e.get("dataTransfer", {}).get("files", [])
        paths = [f.get("pywebviewFullPath") for f in files if f.get("pywebviewFullPath")]
        if paths:
            api.add_paths(paths)

    doc = window.dom.document
    doc.events.dragover += DOMEventHandler(lambda e: None, True, True)
    doc.events.drop += DOMEventHandler(on_drop, True, True)


def main():
    # своя «личность» на панели задач: своя иконка и группировка, а не Python
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        pass
    api = Api()
    window = webview.create_window(
        APP_NAME, str(HERE / "ui" / "index.html"), js_api=api,
        width=1400, height=880, min_size=(1000, 640), background_color="#0e0d14")
    api._window = window
    window.events.loaded += lambda: bind_drop(window, api)
    webview.start(icon=str(ICON) if ICON.exists() else None,
                  debug=os.environ.get("EMOSTUDIO_DEBUG") == "1")


if __name__ == "__main__":
    main()
