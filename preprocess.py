"""Preprocessing dataset sehat/sakit.

1) Offline (sekali):  python preprocess.py
   - split 70/15/15 stratifikasi (label x sumber, SEED tetap) SEBELUM augmentasi
   - center-crop persegi + resize 128x128, simpan JPEG ke data/processed/{train,val,test}/{sakit,sehat}/
   - tulis data/processed/split.csv (file, split, label, source, orig_class)
2) Runtime (train.py/evaluate.py):  from preprocess import load_datasets
   - normalisasi 1/255 semua split, augmentasi hanya pada train

load_image() juga dipakai inference.py agar preprocessing identik saat prediksi.
"""
import csv
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
IMG_SIZE = 128
SEED = 42
CLASS_NAMES = ["sakit", "sehat"]  # indeks 0 = sakit, 1 = sehat
SPLITS = ("train", "val", "test")


def load_image(src, size=IMG_SIZE):
    """Buka citra (path / file-like) -> RGB, center-crop persegi, resize. Return PIL.Image."""
    with Image.open(src) as im:
        im.draft("RGB", (size * 2, size * 2))  # decode JPEG besar (6000x4000) jauh lebih cepat
        im = im.convert("RGB")
        w, h = im.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        # crop, bukan pad: padding hitam pada rice (3081x897) jadi jalan pintas "sumber = label"
        return im.crop((left, top, left + s, top + s)).resize((size, size), Image.Resampling.LANCZOS)


def _process(job):
    src, dst = job
    if not dst.exists():
        load_image(src).save(dst, "JPEG", quality=95)


def split_rows(rows):
    """Split lengket: file yang sudah ada di split.csv lama tetap di split-nya (test lama tetap utuh
    agar model lama vs baru bisa dibandingkan); hanya file baru yang di-split 70/15/15 stratifikasi."""
    from sklearn.model_selection import train_test_split

    old = {}
    if (PROCESSED / "split.csv").exists():
        with (PROCESSED / "split.csv").open(encoding="utf-8") as f:
            old = {Path(r["file"]).stem: r["split"] for r in csv.DictReader(f)}
    new = [r for r in rows if Path(r["file"]).stem not in old]
    for r in rows:
        r["split"] = old.get(Path(r["file"]).stem)
    if new:
        strat = ["{}|{}".format(r["label"], r["source"]) for r in new]
        train, rest, _, strat_rest = train_test_split(new, strat, test_size=0.30, stratify=strat, random_state=SEED)
        val, test = train_test_split(rest, test_size=0.50, stratify=strat_rest, random_state=SEED)
        for name, part in zip(SPLITS, (train, val, test)):
            for r in part:
                r["split"] = name
    print("split: {} file lama dipertahankan, {} file baru".format(len(rows) - len(new), len(new)))
    return rows


def main():
    with (RAW / "sources.csv").open(encoding="utf-8") as f:
        rows = split_rows(list(csv.DictReader(f)))

    jobs, out = [], []
    for r in rows:
        rel = "{}/{}/{}".format(r["split"], r["label"], Path(r["file"]).stem + ".jpg")
        jobs.append((RAW / r["file"], PROCESSED / rel))
        out.append({"file": rel, "split": r["split"], "label": r["label"],
                    "source": r["source"], "orig_class": r["orig_class"]})
    for s in SPLITS:
        for c in CLASS_NAMES:
            (PROCESSED / s / c).mkdir(parents=True, exist_ok=True)

    print("Memproses {} citra -> {}x{} ...".format(len(jobs), IMG_SIZE, IMG_SIZE))
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        list(ex.map(_process, jobs, chunksize=32))

    with (PROCESSED / "split.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    # self-check: tiap file tepat di satu split, jumlah di disk == csv
    assert len({Path(o["file"]).name for o in out}) == len(out), "file muncul di >1 split"
    cnt = Counter((o["split"], o["label"]) for o in out)
    for s in SPLITS:
        for c in CLASS_NAMES:
            on_disk = sum(1 for p in (PROCESSED / s / c).iterdir() if not p.name.startswith("bgaug__"))  # bgaug: make_bgaug.py
            assert on_disk == cnt[(s, c)], "{}/{}: disk={} csv={}".format(s, c, on_disk, cnt[(s, c)])
        n = cnt[(s, "sakit")] + cnt[(s, "sehat")]
        print("  {:<5} sakit={:>6} sehat={:>6} total={:>6} ({:.1%})".format(
            s, cnt[(s, "sakit")], cnt[(s, "sehat")], n, n / len(out)))
    by_src = Counter((o["split"], o["source"]) for o in out)
    print("  per sumber:", {s: [by_src[(sp, s)] for sp in SPLITS] for s in sorted({o["source"] for o in out})})
    print("Selesai: data/processed/ + split.csv")


def load_datasets(batch_size=32):
    """Return (train_ds, val_ds, test_ds) tf.data, label int (0=sakit, 1=sehat).

    val/test tidak diacak: urutan sama dengan ds.file_paths (untuk evaluasi per sumber).
    """
    import tensorflow as tf

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        tf.keras.layers.RandomRotation(0.1, seed=SEED),
        tf.keras.layers.RandomZoom(0.1, seed=SEED),
        tf.keras.layers.RandomContrast(0.1, seed=SEED),
    ], name="augment")
    rescale = tf.keras.layers.Rescaling(1.0 / 255)

    def load(split, shuffle):
        return tf.keras.utils.image_dataset_from_directory(
            PROCESSED / split, labels="inferred", label_mode="int", class_names=CLASS_NAMES,
            image_size=(IMG_SIZE, IMG_SIZE), batch_size=batch_size, shuffle=shuffle, seed=SEED)

    AUTO = tf.data.AUTOTUNE
    train = load("train", True).map(lambda x, y: (rescale(augment(x, training=True)), y), num_parallel_calls=AUTO)
    val = load("val", False)
    test = load("test", False)
    out = [train.prefetch(AUTO)]
    for ds in (val, test):
        mapped = ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTO).prefetch(AUTO)
        mapped.file_paths = ds.file_paths  # map() membuang atribut ini
        out.append(mapped)
    return tuple(out)


def class_weights():
    """Bobot kelas dari data train (atasi rasio sehat:sakit ±1:3,6). Untuk model.fit(class_weight=...)."""
    with (PROCESSED / "split.csv").open(encoding="utf-8") as f:
        c = Counter(r["label"] for r in csv.DictReader(f) if r["split"] == "train")
    total = sum(c.values())
    return {i: total / (len(CLASS_NAMES) * c[name]) for i, name in enumerate(CLASS_NAMES)}


if __name__ == "__main__":
    main()
