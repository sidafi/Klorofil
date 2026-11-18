"""Training CNN sehat/sakit.

Pakai:  python train.py [--epochs 50] [--batch 32] [--lr 1e-3]
Hasil:  models/best.keras (val_loss terbaik), models/history.csv, models/history.png
"""
import argparse
from pathlib import Path

import tensorflow as tf

from model import build_model
from preprocess import SEED, class_weights, load_datasets

MODELS = Path(__file__).resolve().parent / "models"


def plot_history(csv_path, out_path):
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ep = [int(r["epoch"]) + 1 for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, ("loss", "accuracy")):
        ax.plot(ep, [float(r[key]) for r in rows], label="train")
        ax.plot(ep, [float(r["val_" + key]) for r in rows], label="val")
        ax.set_xlabel("epoch")
        ax.set_title(key)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--bn-momentum", type=float, default=0.99)
    ap.add_argument("--no-class-weight", action="store_true", help="matikan class_weight (ablasi)")
    args = ap.parse_args()
    print("args:", vars(args))

    tf.keras.utils.set_random_seed(SEED)
    MODELS.mkdir(exist_ok=True)
    print("GPU:", tf.config.list_physical_devices("GPU") or "tidak ada (CPU)")

    train_ds, val_ds, _ = load_datasets(args.batch)
    cw = None if args.no_class_weight else class_weights()
    print("class_weight:", cw)

    model = build_model(bn_momentum=args.bn_momentum)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODELS / "best.keras", monitor="val_loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        tf.keras.callbacks.CSVLogger(MODELS / "history.csv"),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=cw, callbacks=callbacks)

    plot_history(MODELS / "history.csv", MODELS / "history.png")
    loss, acc = model.evaluate(val_ds, verbose=0)
    print("Model terbaik -> models/best.keras | val_loss={:.4f} val_acc={:.4f}".format(loss, acc))


if __name__ == "__main__":
    main()
