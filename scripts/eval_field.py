"""Uji model pada foto lapangan milik tim (di luar dataset Kaggle). HANYA untuk test, jangan ikut training.

Susun foto:  data/field_test/sehat/*.jpg  dan  data/field_test/sakit/*.jpg  (jenis daun bebas)
Pakai:       python scripts/eval_field.py [--model models/model.tflite] [--out models/eval_field]
Hasil:       <out>/confusion_matrix.png, report.txt, predictions.csv
Pipeline identik dengan web app (inference.Predictor: crop persegi + resize 128 + /255, TFLite).
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from evaluate import plot_cm, report_block  # noqa: E402
from inference import Predictor  # noqa: E402
from preprocess import CLASS_NAMES  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data" / "field_test"))
    ap.add_argument("--model", default=str(ROOT / "models" / "model.tflite"))
    ap.add_argument("--out", default=str(ROOT / "models" / "eval_field"))
    args = ap.parse_args()

    files = [(f, CLASS_NAMES.index(c)) for c in CLASS_NAMES
             for f in sorted((Path(args.dir) / c).glob("*")) if f.suffix.lower() in EXT]
    assert files, "tidak ada foto di {}/{{sehat,sakit}}/".format(args.dir)
    pred = Predictor(args.model)
    rows, y_true, y_pred = [], [], []
    for f, t in files:
        r = pred.predict(f)
        y_true.append(t)
        y_pred.append(CLASS_NAMES.index(r["label"]))
        rows.append([f.name, CLASS_NAMES[t], r["label"], "{:.4f}".format(r["confidence"]), int(r["uncertain"])])

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cm, text = report_block("FOTO LAPANGAN ({})".format(Path(args.model).name), y_true, y_pred)
    sure = np.array([r[4] == 0 for r in rows])
    text += "\nditandai tidak yakin: {}/{} | akurasi yang yakin: {}".format(
        (~sure).sum(), len(rows), "{:.4f}".format((y_true[sure] == y_pred[sure]).mean()) if sure.any() else "-")
    plot_cm(cm, "Foto lapangan", out / "confusion_matrix.png")
    (out / "report.txt").write_text(text, encoding="utf-8")
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "true", "pred", "confidence", "uncertain"])
        w.writerows(rows)
    print(text)


if __name__ == "__main__":
    main()
