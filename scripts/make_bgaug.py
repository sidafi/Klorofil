"""Augmentasi ganti latar (v3): daun berlatar polos ditempel ke latar kebun acak, HANYA data train.

Tujuan: kurangi bias "latar polos" (model gagal di foto kebun). Label daun tidak berubah.
- Foreground: citra train sumber leaves (semua) + PlantVillage (N_PV per label), latar abu-abu
  -> segmentasi saturasi HSV, komponen terbesar. Lubang daun TIDAK diisi (latar baru terlihat lewat lubang).
- Latar: 70% crop foto lapangan train (PlantDoc + iNaturalist) di-blur kuat (bokeh kebun), 30% gradien warna alami + noise.
- Hasil: data/processed/train/<label>/bgaug__*.jpg + baris di split.csv (split=train, source=bgaug).

Urutan: python preprocess.py  ->  python scripts/make_bgaug.py  (preprocess menulis ulang split.csv).
Idempotent: hasil bgaug lama dihapus dulu.
"""
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from preprocess import CLASS_NAMES, IMG_SIZE, PROCESSED, SEED  # noqa: E402

PREFIX = "bgaug__"
N_PV = 2000  # per label dari PlantVillage
K3 = np.ones((3, 3), np.uint8)
GREENS = [(34, 85, 34), (60, 110, 50), (30, 60, 25), (90, 140, 70), (70, 90, 60),
          (100, 80, 50), (60, 50, 40), (140, 150, 120), (40, 70, 70)]  # BGR: hijau daun, tanah, batang, langit pucat


def leaf_mask(bgr):
    """Mask daun 0/1 atau None jika segmentasi meragukan."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]
    not_blue = (h < 90) | (h > 150)  # latar dataset leaves abu-kebiruan; daun tidak pernah biru
    m = ((hsv[..., 1] > 45) & (hsv[..., 2] > 40) & not_blue).astype(np.uint8)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, K3), cv2.MORPH_CLOSE, K3)
    m = cv2.erode(m, K3)  # buang pinggiran sisa latar lama
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    m = (lab == 1 + stats[1:, cv2.CC_STAT_AREA].argmax()).astype(np.uint8)
    # isi lubang dalam: bagian pucat/mengilap (saturasi rendah) bukan lubang asli; lubang palsu di daun sehat
    # akan mengajari "berlubang = sehat" (bertentangan dengan definisi label)
    flood = np.pad(1 - m, 1, constant_values=1).astype(np.uint8)
    cv2.floodFill(flood, None, (0, 0), 0)
    m = m | flood[1:-1, 1:-1]
    return m if 0.08 < m.mean() < 0.92 else None


def background(rng, pool):
    s = IMG_SIZE
    if pool and rng.random() < 0.7:
        img = cv2.imread(str(rng.choice(pool)))
        c = rng.randint(s // 3, s)  # crop acak lalu perbesar
        x, y = rng.randint(0, s - c), rng.randint(0, s - c)
        img = cv2.resize(img[y:y + c, x:x + c], (s, s))
        k = rng.choice((11, 17, 25, 35))
        img = cv2.GaussianBlur(img, (k, k), 0)
    else:
        a, b = (np.array(rng.choice(GREENS), np.float32) for _ in range(2))
        t = np.linspace(0, 1, s, dtype=np.float32)
        g = t[None, :, None] if rng.random() < 0.5 else t[:, None, None]
        img = a * (1 - g) + b * g + np.random.RandomState(rng.randint(0, 2**31)).normal(0, 12, (s, s, 3))
        img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (9, 9), 0)
    gain = rng.uniform(0.7, 1.2)
    return np.clip(img.astype(np.float32) * gain, 0, 255)


def composite(fg, mask, bg, rng):
    s = IMG_SIZE
    scale = rng.uniform(0.6, 1.0)
    n = max(8, int(s * scale))
    fg = cv2.resize(fg, (n, n))
    a = cv2.resize(mask.astype(np.float32), (n, n))
    if rng.random() < 0.5:
        fg, a = fg[:, ::-1], a[:, ::-1]
    a = cv2.GaussianBlur(a, (3, 3), 0)[..., None]  # tepi halus
    x, y = rng.randint(0, s - n), rng.randint(0, s - n)
    out = bg.copy()
    out[y:y + n, x:x + n] = a * fg + (1 - a) * out[y:y + n, x:x + n]
    return out.astype(np.uint8)


def main():
    rng = random.Random(SEED)
    split_csv = PROCESSED / "split.csv"
    with split_csv.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["source"] != "bgaug"]
    for p in PROCESSED.glob("train/*/{}*".format(PREFIX)):
        p.unlink()

    train = [r for r in rows if r["split"] == "train"]
    pool = [PROCESSED / r["file"] for r in train if r["source"] in ("plantdoc", "inat")]  # latar: foto lapangan train
    fg_rows = [r for r in train if r["source"] == "leaves"]
    for label in CLASS_NAMES:
        pv = [r for r in train if r["source"] == "plantvillage" and r["label"] == label]
        fg_rows += rng.sample(pv, min(N_PV, len(pv)))

    new, skipped = [], 0
    for r in fg_rows:
        fg = cv2.imread(str(PROCESSED / r["file"]))
        mask = leaf_mask(fg)
        if mask is None:
            skipped += 1
            continue
        rel = "train/{}/{}{}".format(r["label"], PREFIX, Path(r["file"]).name)
        cv2.imwrite(str(PROCESSED / rel), composite(fg, mask, background(rng, pool), rng), [cv2.IMWRITE_JPEG_QUALITY, 95])
        new.append({"file": rel, "split": "train", "label": r["label"], "source": "bgaug",
                    "orig_class": "{}|{}".format(r["source"], r["orig_class"])})

    with split_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows + new)

    # self-check: file bgaug di disk == baris csv, semua di train
    on_disk = sum(1 for _ in PROCESSED.glob("*/*/{}*".format(PREFIX)))
    assert on_disk == len(new) and all(n["split"] == "train" for n in new), (on_disk, len(new))
    by = {c: sum(n["label"] == c for n in new) for c in CLASS_NAMES}
    print("bgaug: {} citra dibuat {} | dilewati (segmentasi meragukan): {}".format(len(new), by, skipped))


if __name__ == "__main__":
    main()
