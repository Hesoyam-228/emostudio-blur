"""
EmoStudio Blur (без интерфейса): находит соски и гениталии на артах и заблюривает их.

Использование:
    python autoblur.py <файл или папка> [ещё файлы/папки...] [опции]

Результат кладётся в папку "censored" рядом с исходниками (оригиналы не трогаются).
"""
import argparse
import sys
from pathlib import Path

import censor_core as core


def collect(inputs):
    files = []
    for p in map(Path, inputs):
        if p.is_dir():
            files += [(f, p) for f in sorted(p.rglob("*"))
                      if f.suffix.lower() in core.EXTS and "censored" not in f.parts]
        elif p.suffix.lower() in core.EXTS:
            files.append((p, p.parent))
    return files


def main():
    d = core.DEFAULTS
    ap = argparse.ArgumentParser(description="Автоматический блюр сосков и гениталий")
    ap.add_argument("inputs", nargs="*", help="файлы или папки")
    ap.add_argument("-o", "--out", help="папка для результата (по умолчанию <папка>/censored)")
    ap.add_argument("--parts", default=",".join(d["parts"]),
                    help="что блюрить, через запятую: nipples,penis,pussy,anus")
    ap.add_argument("--full", action="store_true", help="блюрить член целиком, а не только головку")
    ap.add_argument("--tip-size", type=float, default=d["tip_size"], help="длина зоны головки (1.0 = головка и чуть ниже)")
    ap.add_argument("--conf", type=float, default=d["conf"], help="порог для сосков 0..1 (меньше = находит больше)")
    ap.add_argument("--conf-genitals", type=float, default=d["conf_genitals"], help="порог для гениталий 0..1")
    ap.add_argument("--pad", type=float, default=d["pad"], help="насколько расширять зону блюра (0.15 = +15%%)")
    ap.add_argument("--strength", type=float, default=d["strength"], help="сила блюра")
    ap.add_argument("--style", choices=["blur", "pixel"], default=d["style"])
    ap.add_argument("--debug", action="store_true", help="дополнительно сохранять картинки с рамками найденного")
    args = ap.parse_args()

    if not args.inputs:
        print("Перетащи картинки или папку на «Quick Blur - drop folder here.bat» (или запусти «EmoStudio Blur.bat» для интерфейса).")
        return 0

    settings = {
        "parts": [s.strip() for s in args.parts.split(",") if s.strip()],
        "penis_mode": "full" if args.full else "tip",
        "tip_size": args.tip_size, "conf": args.conf, "conf_genitals": args.conf_genitals, "pad": args.pad,
    }
    core.configure(use_gpu=True)  # видеокарта, если есть; иначе процессор

    files = collect(args.inputs)
    if not files:
        print("Не найдено картинок.")
        return 1

    check = []
    for i, (f, root) in enumerate(files, 1):
        out_root = Path(args.out) if args.out else root / "censored"
        dst = out_root / f.relative_to(root)
        try:
            rgb = core.load_rgb(f)
            spots, warnings = core.make_spots(rgb, core.raw_detect(rgb), settings)
            core.save_rgb(core.render(rgb, spots, args.strength, args.style), dst)
            if args.debug:
                core.save_rgb(core.draw_debug(rgb, spots), dst.with_name(dst.stem + "_debug.png"))
            names = ", ".join(sorted({s["kind"] for s in spots})) or "-"
            if warnings:
                check.append(f"{f}  ({'; '.join(warnings)})")
            print(f"[{i}/{len(files)}] {f.name}: {names}" + (f"  ⚠ {'; '.join(warnings)}" if warnings else ""))
        except Exception as e:
            check.append(f"{f}  (ошибка: {e})")
            print(f"[{i}/{len(files)}] {f.name}: ОШИБКА {e}")

    if check:
        print("\nПроверь вручную:")
        for m in check:
            print("  " + m)
    print("\nГотово.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
