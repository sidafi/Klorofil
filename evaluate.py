"""Evaluasi model: confusion matrix, precision, recall, F1 per kelas (keseluruhan + per sumber dataset).

Pakai:  python evaluate.py [--model models/best.keras] [--split test] [--task penyakit|jenis]
Hasil:  models/eval_<split>/ -> confusion_matrix.png, report.txt, predictions.csv
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from preprocess import CLASS_NAMES, load_datasets

ROOT = Path(__file__).resolve().parent


def report_block(title, y_true, y_pred, names=CLASS_NAMES):
    labels = list(range(len(names)))
    w = max(6, max(map(len, names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rep = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                digits=4, zero_division=0)
    head = "== {} (n={}) ==".format(title, len(y_true))
    rows = "\n".join("  true {} {}".format(names[i].ljust(w), "  ".join("{:>6}".format(v) for v in r))
                     for i, r in enumerate(cm))
    cols = "  pred " + " " * w + "  ".join("{:>6}".format(c[:6]) for c in names)
    return cm, "{}\n{}\n{}\n\n{}".format(head, cols, rows, rep)


def plot_cm(cm, title, path, names=CLASS_NAMES):
    s = max(4, 0.6 * len(names))
    fig, ax = plt.subplots(figsize=(s + 0.5, s))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False,
                                                          xticks_rotation=90 if len(names) > 4 else "horizontal")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "best.keras"))
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--out", help="folder hasil (default models/eval_<split>)")
    ap.add_argument("--task", choices=("sehat", "penyakit", "jenis"), default="sehat")
    args = ap.parse_args()
    import tensorflow as tf  # di sini agar report_block/plot_cm bisa diimpor tanpa TF

    out = Path(args.out) if args.out else ROOT / "models" / "eval_{}".format(args.split)
    out.mkdir(parents=True, exist_ok=True)

    if args.task != "sehat":
        import importlib
        task = importlib.import_module(args.task)  # penyakit.py / jenis.py
        names = task.CLASSES
        _, val_ds, test_ds = load_datasets(32, task.DATA, names)
    else:
        names = CLASS_NAMES
        _, val_ds, test_ds = load_datasets()
    ds = test_ds if args.split == "test" else val_ds
    model = tf.keras.models.load_model(args.model)

    probs = model.predict(ds, verbose=0)
    y_true = np.concatenate([y.numpy() for _, y in ds])
    y_pred = probs.argmax(axis=1)
    paths = ds.file_paths
    assert len(paths) == len(y_true) == len(y_pred), "urutan dataset tidak sinkron"
    sources = np.array([Path(p).name.split("__")[0] for p in paths])  # nama file: <sumber>__<kelas>__...

    cm, text = report_block("KESELURUHAN ({})".format(args.split), y_true, y_pred, names)
    plot_cm(cm, "Confusion matrix ({})".format(args.split), out / "confusion_matrix.png", names)
    blocks = [text]
    for s in sorted(set(sources)):
        m = sources == s
        cm_s, t = report_block("sumber: {}".format(s), y_true[m], y_pred[m], names)
        plot_cm(cm_s, "{} ({})".format(s, args.split), out / "confusion_matrix_{}.png".format(s), names)
        blocks.append(t)

    report = "\n\n".join(blocks)
    (out / "report.txt").write_text(report, encoding="utf-8")
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "source", "true", "pred", "confidence"] + ["p_" + c for c in names])
        for p, s, t, pr, pb in zip(paths, sources, y_true, y_pred, probs):
            w.writerow([Path(p).name, s, names[t], names[pr], "{:.4f}".format(pb.max())]
                       + ["{:.4f}".format(v) for v in pb])

    print(report)
    print("Hasil -> {}".format(out.relative_to(ROOT).as_posix()))


if __name__ == "__main__":
    main()
