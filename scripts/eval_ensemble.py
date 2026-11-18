"""Ensemble 2 model TFLite: p = w * p_A + (1 - w) * p_B. Tuning bobot w HANYA di val; test dipakai sekali setelahnya.

Pakai:  python scripts/eval_ensemble.py --a models/model.tflite --b models_v3/model.tflite [--split val] [--w 0.5]
Tanpa --w: sapu w = 0.0..1.0 dan tampilkan semua kriteria aturan stop (macro F1, sumber lama, PlantDoc, inat, foto lapangan).
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from inference import Predictor  # noqa: E402
from preprocess import CLASS_NAMES, PROCESSED  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def probs(pred, files):
    return np.stack([pred.probs(f) for f in files])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(ROOT / "models" / "model.tflite"))
    ap.add_argument("--b", default=str(ROOT / "models_v3" / "model.tflite"))
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--w", type=float, nargs="+", help="bobot model A (boleh beberapa); kosong = sapu 0..1")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(PROCESSED / "split.csv", encoding="utf-8")) if r["split"] == args.split]
    files = [PROCESSED / r["file"] for r in rows]
    yt = np.array([CLASS_NAMES.index(r["label"]) for r in rows])
    src = np.array([r["source"] for r in rows])
    field = [(f, CLASS_NAMES.index(c)) for c in CLASS_NAMES
             for f in sorted((ROOT / "data" / "field_test" / c).glob("*")) if f.suffix.lower() in EXT]
    ft = np.array([t for _, t in field])

    A, B = Predictor(args.a), Predictor(args.b)
    pa, pb = probs(A, files), probs(B, files)
    fa, fb = probs(A, [f for f, _ in field]), probs(B, [f for f, _ in field])
    print("split={} n={} | foto lapangan n={}".format(args.split, len(yt), len(ft)))
    print("  w(A)  macroF1  acc    lama   PlantDoc  inat   lapangan")
    for w in (args.w if args.w else np.round(np.arange(0, 1.01, 0.1), 1)):
        yp = (w * pa + (1 - w) * pb).argmax(1)
        fp = (w * fa + (1 - w) * fb).argmax(1)
        old = src != "inat"
        acc = lambda m: (yt[m] == yp[m]).mean() if m.any() else float("nan")  # noqa: E731
        print("  {:.1f}   {:.4f}   {:.4f} {:.4f} {:.4f}    {:.4f} {}/{}".format(
            w, f1_score(yt, yp, average="macro"), (yt == yp).mean(), acc(old), acc(src == "plantdoc"),
            acc(src == "inat"), int((fp == ft).sum()), len(ft)))


if __name__ == "__main__":
    main()
