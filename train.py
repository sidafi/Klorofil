"""Training CNN sehat/sakit.

Pakai:  python train.py [--epochs 50] [--batch 32] [--lr 1e-3]
        python train.py --task penyakit|jenis --patience 12   (tahap 2, data dari penyakit.py / jenis.py)
Hasil:  models/best.keras (val_loss terbaik), models/history.csv, models/history.png
"""
import argparse
from pathlib import Path

import tensorflow as tf

from model import build_model, build_transfer
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
    ap.add_argument("--task", choices=("sehat", "penyakit", "jenis"), default="sehat", help="tahap 2: penyakit.py / jenis.py")
    ap.add_argument("--arch", choices=("leafnet", "mobilenetv2"), default="leafnet", help="mobilenetv2 = transfer learning (J2)")
    ap.add_argument("--finetune-epochs", type=int, default=20, help="mobilenetv2: epoch fase 2 (fine-tune)")
    ap.add_argument("--finetune-lr", type=float, default=1e-5)
    ap.add_argument("--unfreeze", type=int, default=40, help="mobilenetv2: jumlah lapisan teratas base yang dilatih di fase 2")
    args = ap.parse_args()
    print("args:", vars(args))

    tf.keras.utils.set_random_seed(SEED)
    MODELS.mkdir(exist_ok=True)
    print("GPU:", tf.config.list_physical_devices("GPU") or "tidak ada (CPU)")

    if args.task != "sehat":
        import importlib
        task = importlib.import_module(args.task)  # penyakit.py / jenis.py: DATA + CLASSES
        names = task.CLASSES
        train_ds, val_ds, _ = load_datasets(args.batch, task.DATA, names)
        cw = None  # tahap 2: tanpa class_weight (aturan dikunci di CLAUDE.md)
    else:
        names = None
        train_ds, val_ds, _ = load_datasets(args.batch)
        cw = None if args.no_class_weight else class_weights()
    print("class_weight:", cw)

    base = None
    if args.arch == "mobilenetv2":
        model, base = build_transfer(len(names) if names else 2)
    else:
        model = build_model(bn_momentum=args.bn_momentum) if names is None else build_model(len(names), bn_momentum=args.bn_momentum)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def callbacks(min_lr, append=False, best=None):
        return [
            tf.keras.callbacks.ModelCheckpoint(MODELS / "best.keras", monitor="val_loss", save_best_only=True, verbose=1,
                                               initial_value_threshold=best),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=min_lr, verbose=1),
            tf.keras.callbacks.CSVLogger(MODELS / "history.csv", append=append),
        ]
    h = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=cw, callbacks=callbacks(1e-5))

    if base is not None and args.finetune_epochs > 0:  # fase 2: fine-tune lapisan atas base, BN tetap beku
        base.trainable = True
        for layer in base.layers[:-args.unfreeze]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(args.finetune_lr),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        done = len(h.history["loss"])
        print("Fine-tune: {} lapisan base dilatih, mulai epoch {}".format(sum(l.trainable for l in base.layers), done + 1))
        model.fit(train_ds, validation_data=val_ds, initial_epoch=done, epochs=done + args.finetune_epochs, class_weight=cw,
                  callbacks=callbacks(1e-7, append=True, best=min(h.history["val_loss"])))

    plot_history(MODELS / "history.csv", MODELS / "history.png")
    loss, acc = model.evaluate(val_ds, verbose=0)
    print("Model terbaik -> models/best.keras | val_loss={:.4f} val_acc={:.4f}".format(loss, acc))


if __name__ == "__main__":
    main()
