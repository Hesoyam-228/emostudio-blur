"""
Создаёт ярлыки «EmoStudio Blur» на рабочем столе и в меню «Пуск».

    python src/shortcuts.py           создать ярлыки
    python src/shortcuts.py --remove  удалить ярлыки

У ярлыка тот же AppUserModelID, что и у окна приложения, поэтому
закреплённый на панели задач значок и открытое окно — это одна кнопка.
Если pywin32 по какой-то причине недоступен, ярлык создаётся запасным
способом (WScript.Shell) — без AppUserModelID, но рабочий.

Иконка для ярлыка копируется в %APPDATA%\\EmoStudio Blur под именем с отпечатком
содержимого: Windows кэширует иконки по пути, а новый путь заставляет её
перечитать иконку после обновления.
"""
import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import traceback
import uuid
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

APP_NAME = "EmoStudio Blur"
APP_ID = "EmoStudio.Blur"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("APPDATA", Path.home())) / APP_NAME
LOG = DATA_DIR / "shortcuts.log"

# FOLDERID_Desktop и FOLDERID_Programs — понимают и OneDrive, и перенесённый рабочий стол
_FOLDERS = {"desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
            "programs": "{A77F5D77-2E2B-44C3-A6A2-ABA601054A51}"}


def _log(text):
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {text}\n")
    except Exception:
        pass


def _known_folder(guid):
    class GUID(ctypes.Structure):
        _fields_ = [("d1", wintypes.DWORD), ("d2", wintypes.WORD), ("d3", wintypes.WORD), ("d4", ctypes.c_ubyte * 8)]

    u = uuid.UUID(guid)
    g = GUID(u.fields[0], u.fields[1], u.fields[2], (ctypes.c_ubyte * 8)(*u.bytes[8:]))
    p = ctypes.c_wchar_p()
    if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(p)) != 0:
        raise OSError(f"не удалось узнать папку {guid}")
    try:
        return Path(p.value)
    finally:
        ctypes.windll.ole32.CoTaskMemFree(p)


def shortcut_paths():
    test_dir = os.environ.get("EMOSTUDIO_SHORTCUT_DIR")  # для проверки установки, не трогая рабочий стол
    if test_dir:
        return [Path(test_dir) / "desktop" / f"{APP_NAME}.lnk", Path(test_dir) / "programs" / f"{APP_NAME}.lnk"]
    out = []
    for key in ("desktop", "programs"):
        try:
            out.append(_known_folder(_FOLDERS[key]) / f"{APP_NAME}.lnk")
        except Exception as e:
            _log(f"папка {key}: {e}")
            if key == "desktop":
                out.append(Path.home() / "Desktop" / f"{APP_NAME}.lnk")
    return out


def shortcut_icon():
    src = ROOT / "assets" / "icon.ico"
    name = f"icon-{hashlib.sha1(src.read_bytes()).hexdigest()[:10]}.ico"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for old in DATA_DIR.glob("icon-*.ico"):
        if old.name != name:
            old.unlink(missing_ok=True)
    dst = DATA_DIR / name
    if not dst.exists():
        shutil.copyfile(src, dst)
    return dst


def refresh_shell():
    """Просим Проводник перерисовать иконки (SHCNE_ASSOCCHANGED)."""
    try:
        ctypes.windll.shell32.SHChangeNotify(0x08000000, 0, None, None)
    except Exception:
        pass


def _target():
    return (str(ROOT / "venv" / "Scripts" / "pythonw.exe"), f'"{ROOT / "src" / "app.py"}"', str(ROOT))


def _make_pywin32(lnk, icon):
    import pythoncom
    from win32com.propsys import propsys, pscon
    from win32com.shell import shell

    exe, args, cwd = _target()
    link = pythoncom.CoCreateInstance(
        shell.CLSID_ShellLink, None, pythoncom.CLSCTX_INPROC_SERVER, shell.IID_IShellLink)
    link.SetPath(exe)
    link.SetArguments(args)
    link.SetWorkingDirectory(cwd)
    link.SetIconLocation(str(icon), 0)
    link.SetDescription("EmoStudio Blur — автоматический блюр для артов")
    store = link.QueryInterface(propsys.IID_IPropertyStore)
    store.SetValue(pscon.PKEY_AppUserModel_ID, propsys.PROPVARIANTType(APP_ID, pythoncom.VT_LPWSTR))
    store.Commit()
    link.QueryInterface(pythoncom.IID_IPersistFile).Save(str(lnk), 0)


def _make_wscript(lnk, icon):
    """Запасной способ: стандартный WScript.Shell через PowerShell (пути передаём через переменные окружения)."""
    exe, args, cwd = _target()
    env = dict(os.environ, ES_LNK=str(lnk), ES_EXE=exe, ES_ARGS=args, ES_CWD=cwd, ES_ICON=f"{icon},0")
    script = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:ES_LNK); "
              "$s.TargetPath = $env:ES_EXE; $s.Arguments = $env:ES_ARGS; $s.WorkingDirectory = $env:ES_CWD; "
              "$s.IconLocation = $env:ES_ICON; $s.Description = 'EmoStudio Blur'; $s.Save()")
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                       env=env, capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0 or not Path(lnk).exists():
        raise OSError((r.stderr or r.stdout or "WScript.Shell не создал ярлык").strip())


def make_shortcut(lnk, icon):
    lnk = Path(lnk)
    lnk.parent.mkdir(parents=True, exist_ok=True)
    try:
        _make_pywin32(lnk, icon)
    except Exception:
        _log(f"pywin32 не смог создать {lnk}:\n{traceback.format_exc()}")
        _make_wscript(lnk, icon)


def create_all():
    """Создаёт все ярлыки. Возвращает (созданные, ошибки)."""
    made, errors = [], []
    try:
        icon = shortcut_icon()
    except Exception as e:
        _log(f"иконка: {traceback.format_exc()}")
        icon = ROOT / "assets" / "icon.ico"
        errors.append(f"иконка: {e}")
    for lnk in shortcut_paths():
        try:
            make_shortcut(lnk, icon)
            made.append(lnk)
        except Exception as e:
            _log(f"ярлык {lnk}: {traceback.format_exc()}")
            errors.append(f"{lnk}: {e}")
    refresh_shell()
    return made, errors


def desktop_shortcut_exists():
    try:
        return shortcut_paths()[0].exists()
    except Exception:
        return False


def main():
    if "--remove" in sys.argv:
        for lnk in shortcut_paths():
            if lnk.exists():
                lnk.unlink()
                print(f"Удалён ярлык: {lnk}")
        refresh_shell()
        return 0
    made, errors = create_all()
    for lnk in made:
        print(f"Создан ярлык: {lnk}")
    for e in errors:
        print(f"Не получилось: {e}")
    if errors:
        print(f"Подробности в {LOG}")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
