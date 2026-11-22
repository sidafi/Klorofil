"""Augmentasi ganti latar untuk data JENIS (J4), HANYA train. Lihat CLAUDE.md "J4".

Masalah J3: model memakai latar sebagai petunjuk jenis (latar studio PlantVillage = tomat/kentang/paprika, latar hitam leaves =
mangga/jambu, foto kebun = lainnya). Daun dari foto berlatar polos disegmentasi (leaf_mask, sama dgn make_bgaug.py) lalu ditempel ke
latar acak: foto kebun (blur), putih, hitam, abu, warna polos, gradien alami -> latar tidak lagi memprediksi jenis.
- Foreground: train data/jenis bersumber plantvillage/leaves (maks N_PER_KELAS per kelas) + daun leaves spesies di luar daftar
  (train di data/processed) untuk kelas "lainnya" (agar "lainnya" tidak identik dengan foto kebun).
- Hasil: data/jenis/train/<kelas>/bgaug__*.jpg + baris di data/jenis/split.csv (source=bgaug). Idempotent.
Urutan: python jenis.py -> python scripts/make_bgaug_jenis.py -> python scripts/make_colab_bundle.py --task jenis
"""
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import jenis  # noqa: E402
from make_bgaug import PREFIX, background, composite, leaf_mask  # noqa: E402
from preprocess import IMG_SIZE, PROCESSED, SEED  # noqa: E402
from scripts.make_plant_gallery import plant_of  # noqa: E402

N_PER_KELAS = 600
PLAIN = [(250, 250, 250), (235, 235, 230), (15, 15, 15), (128, 128, 128), (180, 180, 175), (60, 60, 60)]  # BGR


def background_jenis(rng, pool):
    """45% foto kebun / gradien (background() asli), 55% latar polos (putih, hitam, abu) + sedikit noise."""
    if rng.random() < 0.45:
        return background(rng, pool)
    c = np.array(rng.choice(PLAIN), np.float32) + rng.uniform(-12, 12)
    img = np.full((IMG_SIZE, IMG_SIZE, 3), c, np.float32)
    img += np.random.RandomState(rng.randint(0, 2**31)).normal(0, 4, img.shape)
    return np.clip(img, 0, 255)


def main():
    rng = random.Random(SEED)
    split_csv = jenis.DATA / "split.csv"
    with split_csv.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["source"] != "bgaug"]
    for p in jenis.DATA.glob("train/*/{}*".format(PREFIX)):
        p.unlink()

    train = [r for r in rows if r["split"] == "train"]
    pool = [jenis.DATA / r["file"] for r in train if r["source"] in ("inat", "inatj", "plantdoc")]
    fg = []  # (kelas, path)
    for c in jenis.CLASSES:
        cand = [jenis.DATA / r["file"] for r in train if r["label"] == c and r["source"] in ("plantvillage", "leaves")]
        fg += [(c, p) for p in rng.sample(cand, min(N_PER_KELAS, len(cand)))]
    with (PROCESSED / "split.csv").open(encoding="utf-8") as f:  # daun studio spesies di luar daftar -> lainnya
        other = [PROCESSED / r["file"] for r in csv.DictReader(f) if r["split"] == "train" and r["source"] == "leaves"]
    other = [p for p in other if (lambda g: jenis.GALLERY.get(g, g))(plant_of("leaves", "", "x__" + p.name)) not in jenis.NAMA]
    fg += [("lainnya", p) for p in rng.sample(other, min(N_PER_KELAS, len(other)))]

    new, skipped = [], 0
    for c, path in fg:
        img = cv2.imread(str(path))
        mask = leaf_mask(img) if img is not None else None
        if mask is None:
            skipped += 1
            continue
        rel = "train/{}/{}{}".format(c, PREFIX, path.name)
        cv2.imwrite(str(jenis.DATA / rel), composite(img, mask, background_jenis(rng, pool), rng), [cv2.IMWRITE_JPEG_QUALITY, 95])
        new.append({"file": rel, "split": "train", "label": c, "source": "bgaug"})

    with split_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows + new)

    on_disk = sum(1 for _ in jenis.DATA.glob("*/*/{}*".format(PREFIX)))
    assert on_disk == len(new) == len({n["file"] for n in new}) and all(n["split"] == "train" for n in new), (on_disk, len(new))
    print("bgaug jenis: {} citra | per kelas {} | dilewati: {}".format(
        len(new), {c: sum(n["label"] == c for n in new) for c in jenis.CLASSES}, skipped))


if __name__ == "__main__":
    main()
