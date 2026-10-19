"""
Создаёт ярлыки «EmoStudio Blur» на рабочем столе и в меню «Пуск».

    python src/shortcuts.py           создать ярлыки
    python src/shortcuts.py --remove  удалить ярлыки

У ярлыка тот же AppUserModelID, что и у окна приложения, поэтому
закреплённый на панели задач значок и открытое окно — это одна кнопка.

Иконка для ярлыка копируется в %APPDATA%\EmoStudio Blur под именем с отпечатком
содержимого: Windows кэширует иконки по пути, а новый путь заставляет её
перечитать иконку после обновления.
"""
import ctypes
import hashlib
import os
import shutil
import sys
from pathlib import Path

import pythoncom
from win32com.propsys import propsys, pscon
from win32com.shell import shell, shellcon

APP_NAME = "EmoStudio Blur"
APP_ID = "EmoStudio.Blur"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("APPDATA", Path.home())) / APP_NAME


def shortcut_paths():
    desktop = Path(shell.SHGetFolderPath(0, shellcon.CSIDL_DESKTOPDIRECTORY, None, 0))
    programs = Path(shell.SHGetFolderPath(0, shellcon.CSIDL_PROGRAMS, None, 0))
    return [desktop / f"{APP_NAME}.lnk", programs / f"{APP_NAME}.lnk"]


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


def make_shortcut(lnk, icon):
    link = pythoncom.CoCreateInstance(
        shell.CLSID_ShellLink, None, pythoncom.CLSCTX_INPROC_SERVER, shell.IID_IShellLink)
    link.SetPath(str(ROOT / "venv" / "Scripts" / "pythonw.exe"))
    link.SetArguments(f'"{ROOT / "src" / "app.py"}"')
    link.SetWorkingDirectory(str(ROOT))
    link.SetIconLocation(str(icon), 0)
    link.SetDescription("EmoStudio Blur — автоматический блюр для артов")
    store = link.QueryInterface(propsys.IID_IPropertyStore)
    store.SetValue(pscon.PKEY_AppUserModel_ID, propsys.PROPVARIANTType(APP_ID, pythoncom.VT_LPWSTR))
    store.Commit()
    lnk.parent.mkdir(parents=True, exist_ok=True)
    link.QueryInterface(pythoncom.IID_IPersistFile).Save(str(lnk), 0)


def main():
    remove = "--remove" in sys.argv
    icon = None if remove else shortcut_icon()
    for lnk in shortcut_paths():
        if remove:
            if lnk.exists():
                lnk.unlink()
                print(f"Удалён ярлык: {lnk}")
        else:
            make_shortcut(lnk, icon)
            print(f"Создан ярлык: {lnk}")
    refresh_shell()


if __name__ == "__main__":
    main()
