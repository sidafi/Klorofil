"""Eksperimen TTA (multi-potongan) untuk ensemble, TANPA training. Lihat aturan TTA di CLAUDE.md.

Pakai:  python scripts/eval_tta.py   (bisa dihentikan kapan saja; jalankan lagi -> lanjut dari checkpoint)
Checkpoint: data/tta_val_ckpt.npy (hapus untuk mulai ulang dari nol).
"""
import csv, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import f1_score
from inference import Ensemble
from preprocess import CLASS_NAMES, PROCESSED
CKPT = Path("data/tta_val_ckpt.npy")
E = Ensemble()
def crops(src, size=128, z=0.6):
    with Image.open(src) as im:
        im.draft("RGB", (640, 640)); im = im.convert("RGB")
    w, h = im.size; s = min(w, h); l, t = (w - s) // 2, (h - s) // 2
    c = int(s * z); o = s - c
    boxes = [(0, 0, s, s), ((o // 2), (o // 2), o // 2 + c, o // 2 + c), (0, 0, c, c), (o, 0, s, c), (0, o, c, s), (o, o, s, s)]
    return [np.asarray(im.crop((l + a, t + b, l + cc, t + d)).resize((size, size), Image.Resampling.LANCZOS), np.float32) / 255 for a, b, cc, d in boxes]
ps = lambda x: float(sum(w * m.probs_array(x) for m, w in E.members)[1])
def all_ps(src): return [ps(x) for x in crops(src)]
AGG = {"A tanpa TTA": lambda p: p[0], "B rata2 6": lambda p: np.mean(p), "C min (ada sakit=sakit)": lambda p: min(p), "D .5 penuh+.5 zoom": lambda p: .5 * p[0] + .5 * np.mean(p[1:])}
# val dari citra asli
raw = {Path(r["file"]).stem: Path("data/raw") / r["file"] for r in csv.DictReader(open("data/raw/sources.csv", encoding="utf-8"))}
rows = [r for r in csv.DictReader(open(PROCESSED / "split.csv", encoding="utf-8")) if r["split"] == "val"]
done = list(np.load(CKPT)) if CKPT.exists() else []  # lanjut dari checkpoint
for i in range(len(done), len(rows)):
    done.append(all_ps(raw[Path(rows[i]["file"]).stem]))
    if len(done) % 200 == 0 or len(done) == len(rows):
        np.save(CKPT, np.array(done)); print("val %d/%d tersimpan" % (len(done), len(rows)), flush=True)
P = np.array(done)
assert len(P) == len(rows)
yt = np.array([CLASS_NAMES.index(r["label"]) for r in rows]); old = np.array([r["source"] != "inat" for r in rows])
fl = [(p, CLASS_NAMES.index(c)) for c in CLASS_NAMES for p in sorted(Path("data/field_test", c).iterdir())]
F = [all_ps(p) for p, _ in fl]; ft = np.array([t for _, t in fl])
print("agregasi                 | val macroF1 | val lama | lapangan benar | salah-yakin | tidak yakin")
for name, g in AGG.items():
    pv = np.array([g(p) for p in P]); yp = (pv >= .5).astype(int)
    pf = np.array([g(p) for p in F]); yf = (pf >= .5).astype(int); conf = np.maximum(pf, 1 - pf)
    print("%-24s | %.4f      | %.4f   | %2d/24          | %d           | %d" % (name, f1_score(yt, yp, average="macro"), (yt[old] == yp[old]).mean(),
          (yf == ft).sum(), ((yf != ft) & (conf >= .7)).sum(), (conf < .7).sum()))
