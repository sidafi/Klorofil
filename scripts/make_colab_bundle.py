"""Bungkus data/processed + kode training jadi colab_bundle.zip untuk Google Colab.

Pakai:  python scripts/make_colab_bundle.py   -> colab_bundle.zip (upload ke Google Drive: MyDrive/)
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ["preprocess.py", "model.py", "train.py", "evaluate.py", "inference.py", "requirements-dev.txt"]
OUT = ROOT / "colab_bundle.zip"

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:  # JPEG sudah terkompres, STORED lebih cepat
    for f in CODE:
        z.write(ROOT / f, f)
    z.write(ROOT / "data/processed/split.csv", "data/processed/split.csv")
    n = 0
    for p in sorted((ROOT / "data/processed").glob("*/*/*.jpg")):
        z.write(p, p.relative_to(ROOT).as_posix())
        n += 1

with zipfile.ZipFile(OUT) as z:
    assert z.testzip() is None, "zip rusak"
    expected = sum(1 for _ in (ROOT / "data/processed/split.csv").open(encoding="utf-8")) - 1  # minus header
    assert sum(1 for i in z.namelist() if i.endswith(".jpg")) == n == expected, (n, expected)
print("OK {} citra -> {} ({:.0f} MB)".format(n, OUT.name, OUT.stat().st_size / 2**20))
