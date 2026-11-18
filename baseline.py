"""Baseline pembanding CNN: fitur manual + SVM / Random Forest.

Fitur per citra (128x128, sama dengan input CNN), total 51:
  - histogram HSV: H 16 bin + S 8 + V 8 (dinormalisasi)        = 32
  - GLCM (grayscale 64 level, jarak 1 & 2, rata-rata 4 sudut):
    contrast, dissimilarity, homogeneity, energy, correlation, ASM = 12
  - Hu moments (grayscale, skala log)                           = 7

Pakai:  python baseline.py [--split test]
Hasil:  models/baseline_<split>/ -> report.txt, confusion_matrix_<model>.png
Cache fitur: data/processed/features.npz (hapus jika data/processed berubah)
"""
import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluate import plot_cm, report_block
from preprocess import CLASS_NAMES, PROCESSED, SEED

ROOT = Path(__file__).resolve().parent
CACHE = PROCESSED / "features.npz"
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


def extract(path):
    bgr = cv2.imread(str(path))
    assert bgr is not None, "gagal baca " + str(path)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)  # H 0-179, S/V 0-255
    hist = [cv2.calcHist([hsv], [c], None, [b], [0, r]).ravel()
            for c, b, r in ((0, 16, 180), (1, 8, 256), (2, 8, 256))]
    hist = np.concatenate([h / h.sum() for h in hist])

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray >> 2, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=64, symmetric=True, normed=True)  # kuantisasi 64 level: lebih cepat & stabil
    tex = np.concatenate([graycoprops(glcm, p).mean(axis=1) for p in GLCM_PROPS])  # (2 jarak) x 6

    hu = cv2.HuMoments(cv2.moments(gray)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return np.concatenate([hist, tex, hu]).astype(np.float32)


def load_features():
    with (PROCESSED / "split.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    files = np.array([r["file"] for r in rows])
    if CACHE.exists():
        c = np.load(CACHE)
        if np.array_equal(c["files"], files):
            return c["X"], c["y"], c["split"], c["source"]
    print("Ekstraksi fitur {} citra ...".format(len(rows)))
    # thread, bukan proses: spawn 16 proses (sklearn+skimage+cv2) crash kehabisan memori di Windows
    with ThreadPoolExecutor(8) as ex:
        X = np.stack(list(ex.map(extract, [PROCESSED / f for f in files], chunksize=64)))
    y = np.array([CLASS_NAMES.index(r["label"]) for r in rows])
    split = np.array([r["split"] for r in rows])
    source = np.array([r["source"] for r in rows])
    np.savez_compressed(CACHE, X=X, y=y, split=split, source=source, files=files)
    return X, y, split, source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("val", "test"), default="test")
    args = ap.parse_args()

    X, y, split, source = load_features()
    assert X.shape[1] == 51 and np.isfinite(X).all(), "fitur tidak valid"
    tr, ev = split == "train", split == args.split
    out = ROOT / "models" / "baseline_{}".format(args.split)
    out.mkdir(parents=True, exist_ok=True)

    models = {
        "svm": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale",
                                                   class_weight="balanced", random_state=SEED)),
        "rf": RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED),
    }
    blocks, summary = [], []
    for name, clf in models.items():
        t = time.time()
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[ev])
        cm, text = report_block("{} - KESELURUHAN ({})".format(name.upper(), args.split), y[ev], pred)
        plot_cm(cm, "{} ({})".format(name.upper(), args.split), out / "confusion_matrix_{}.png".format(name))
        blocks.append(text)
        for s in sorted(set(source[ev])):
            m = source[ev] == s
            blocks.append(report_block("{} - sumber: {}".format(name.upper(), s), y[ev][m], pred[m])[1])
        summary.append("{:<4} acc={:.4f} macroF1={:.4f} ({:.0f}s)".format(
            name, (pred == y[ev]).mean(), f1_score(y[ev], pred, average="macro"), time.time() - t))

    report = "\n".join(summary) + "\n\n" + "\n\n".join(blocks)
    (out / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print("Hasil -> {}".format(out.relative_to(ROOT).as_posix()))


if __name__ == "__main__":
    main()
