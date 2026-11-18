"""Evaluasi model: confusion matrix, precision, recall, F1 per kelas (keseluruhan + per sumber dataset).

Pakai:  python evaluate.py [--model models/best.keras] [--split test]
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
LABELS = list(range(len(CLASS_NAMES)))


def report_block(title, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    rep = classification_report(y_true, y_pred, labels=LABELS, target_names=CLASS_NAMES,
                                digits=4, zero_division=0)
    head = "== {} (n={}) ==".format(title, len(y_true))
    rows = "\n".join("  true {:<6} {}".format(CLASS_NAMES[i], "  ".join("{:>6}".format(v) for v in r))
                     for i, r in enumerate(cm))
    cols = "  pred      " + "  ".join("{:>6}".format(c) for c in CLASS_NAMES)
    return cm, "{}\n{}\n{}\n\n{}".format(head, cols, rows, rep)


def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "best.keras"))
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--out", help="folder hasil (default models/eval_<split>)")
    args = ap.parse_args()
    import tensorflow as tf  # di sini agar report_block/plot_cm bisa diimpor tanpa TF

    out = Path(args.out) if args.out else ROOT / "models" / "eval_{}".format(args.split)
    out.mkdir(parents=True, exist_ok=True)

    _, val_ds, test_ds = load_datasets()
    ds = test_ds if args.split == "test" else val_ds
    model = tf.keras.models.load_model(args.model)

    probs = model.predict(ds, verbose=0)
    y_true = np.concatenate([y.numpy() for _, y in ds])
    y_pred = probs.argmax(axis=1)
    paths = ds.file_paths
    assert len(paths) == len(y_true) == len(y_pred), "urutan dataset tidak sinkron"
    sources = np.array([Path(p).name.split("__")[0] for p in paths])  # nama file: <sumber>__<kelas>__...

    cm, text = report_block("KESELURUHAN ({})".format(args.split), y_true, y_pred)
    plot_cm(cm, "Confusion matrix ({})".format(args.split), out / "confusion_matrix.png")
    blocks = [text]
    for s in sorted(set(sources)):
        m = sources == s
        cm_s, t = report_block("sumber: {}".format(s), y_true[m], y_pred[m])
        plot_cm(cm_s, "{} ({})".format(s, args.split), out / "confusion_matrix_{}.png".format(s))
        blocks.append(t)

    report = "\n\n".join(blocks)
    (out / "report.txt").write_text(report, encoding="utf-8")
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "source", "true", "pred", "confidence"] + ["p_" + c for c in CLASS_NAMES])
        for p, s, t, pr, pb in zip(paths, sources, y_true, y_pred, probs):
            w.writerow([Path(p).name, s, CLASS_NAMES[t], CLASS_NAMES[pr], "{:.4f}".format(pb.max())]
                       + ["{:.4f}".format(v) for v in pb])

    print(report)
    print("Hasil -> {}".format(out.relative_to(ROOT).as_posix()))


if __name__ == "__main__":
    main()
