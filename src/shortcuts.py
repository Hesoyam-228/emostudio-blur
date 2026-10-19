"""
Создаёт ярлыки «EmoStudio Blur» на рабочем столе и в меню «Пуск».

    python src/shortcuts.py           создать ярлыки
    python src/shortcuts.py --remove  удалить ярлыки

У ярлыка тот же AppUserModelID, что и у окна приложения, поэтому
закреплённый на панели задач значок и открытое окно — это одна кнопка.
"""
import sys
from pathlib import Path

import pythoncom
from win32com.propsys import propsys, pscon
from win32com.shell import shell, shellcon

APP_NAME = "EmoStudio Blur"
APP_ID = "EmoStudio.Blur"
ROOT = Path(__file__).resolve().parent.parent


def shortcut_paths():
    desktop = Path(shell.SHGetFolderPath(0, shellcon.CSIDL_DESKTOPDIRECTORY, None, 0))
    programs = Path(shell.SHGetFolderPath(0, shellcon.CSIDL_PROGRAMS, None, 0))
    return [desktop / f"{APP_NAME}.lnk", programs / f"{APP_NAME}.lnk"]


def make_shortcut(lnk):
    link = pythoncom.CoCreateInstance(
        shell.CLSID_ShellLink, None, pythoncom.CLSCTX_INPROC_SERVER, shell.IID_IShellLink)
    link.SetPath(str(ROOT / "venv" / "Scripts" / "pythonw.exe"))
    link.SetArguments(f'"{ROOT / "src" / "app.py"}"')
    link.SetWorkingDirectory(str(ROOT))
    link.SetIconLocation(str(ROOT / "assets" / "icon.ico"), 0)
    link.SetDescription("EmoStudio Blur — автоматический блюр для артов")
    store = link.QueryInterface(propsys.IID_IPropertyStore)
    store.SetValue(pscon.PKEY_AppUserModel_ID, propsys.PROPVARIANTType(APP_ID, pythoncom.VT_LPWSTR))
    store.Commit()
    lnk.parent.mkdir(parents=True, exist_ok=True)
    link.QueryInterface(pythoncom.IID_IPersistFile).Save(str(lnk), 0)


def main():
    remove = "--remove" in sys.argv
    for lnk in shortcut_paths():
        if remove:
            if lnk.exists():
                lnk.unlink()
                print(f"Удалён ярлык: {lnk}")
        else:
            make_shortcut(lnk)
            print(f"Создан ярлык: {lnk}")


if __name__ == "__main__":
    main()
