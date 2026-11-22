"""Bungkus data + kode training jadi zip untuk Google Colab.

Pakai:  python scripts/make_colab_bundle.py                 -> colab_bundle.zip   (tahap 1: data/processed)
        python scripts/make_colab_bundle.py --task penyakit|jenis -> <task>_bundle.zip (tahap 2: data/<task>, dari <task>.py)
Upload zip ke Google Drive: MyDrive/
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ["preprocess.py", "model.py", "train.py", "evaluate.py", "inference.py", "penyakit.py", "jenis.py", "requirements-dev.txt"]

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=("sehat", "penyakit", "jenis"), default="sehat")
task = ap.parse_args().task
DATA = "data/processed" if task == "sehat" else "data/" + task
OUT = ROOT / ("colab_bundle.zip" if task == "sehat" else task + "_bundle.zip")

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:  # JPEG sudah terkompres, STORED lebih cepat
    for f in CODE:
        z.write(ROOT / f, f)
    z.write(ROOT / DATA / "split.csv", DATA + "/split.csv")
    n = 0
    for p in sorted((ROOT / DATA).glob("*/*/*.jpg")):
        z.write(p, p.relative_to(ROOT).as_posix())
        n += 1

with zipfile.ZipFile(OUT) as z:
    assert z.testzip() is None, "zip rusak"
    expected = sum(1 for _ in (ROOT / DATA / "split.csv").open(encoding="utf-8")) - 1  # minus header
    assert sum(1 for i in z.namelist() if i.startswith(DATA) and i.endswith(".jpg")) == n == expected, (n, expected)
print("OK {} citra -> {} ({:.0f} MB)".format(n, OUT.name, OUT.stat().st_size / 2**20))
