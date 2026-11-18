"""Kalibrasi ambang keputusan t (sehat jika p_sehat >= t), TANPA training. Aturan: CLAUDE.md "Kalibrasi ambang keputusan".
Butuh data/tta_val_ckpt.npy (dari scripts/eval_tta.py; kolom 0 = p_sehat citra penuh, jalur app)."""
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from inference import Ensemble
from preprocess import CLASS_NAMES, PROCESSED
rows = [r for r in csv.DictReader(open(PROCESSED / "split.csv", encoding="utf-8")) if r["split"] == "val"]
P = np.load("data/tta_val_ckpt.npy")[:, 0]; assert len(P) == len(rows)
yt = np.array([CLASS_NAMES.index(r["label"]) for r in rows])
src = np.array([r["source"] for r in rows]); old, inat = src != "inat", src == "inat"
def f1_score(y, p, average="macro"):  # macro F1 2 kelas (tanpa sklearn: hemat RAM)
    f = [2 * ((p == k) & (y == k)).sum() / max((p == k).sum() + (y == k).sum(), 1) for k in (0, 1)]
    return float(np.mean(f))
best = None
print("t    | F1 val | lama   | F1 inat")
for t in np.round(np.arange(.30, .705, .01), 2):
    yp = (P >= t).astype(int)
    f, lama, fi = f1_score(yt, yp, average="macro"), (yp[old] == yt[old]).mean(), f1_score(yt[inat], yp[inat], average="macro")
    ok = f >= .8987 and lama >= .9545
    if t in (.3, .4, .45, .5, .55, .6, .65, .7) or ok: print("%.2f | %.4f | %.4f | %.4f %s" % (t, f, lama, fi, "" if ok else "x"))
    if ok and (best is None or fi > best[1]): best = (t, fi)
t = best[0]; print("t terpilih di val:", t)
E = Ensemble(); n = cw = 0; tot = 0
for c in CLASS_NAMES:
    for p in sorted(Path("data/field_test", c).iterdir()):
        ps = float(E.probs(str(p))[1]); pred = "sehat" if ps >= t else "sakit"; conf = max(ps, 1 - ps)
        tot += 1; n += pred == c; cw += pred != c and conf >= .7
        if pred != c: print("  miss", c, p.name, round(ps, 2))
print("lapangan benar %d/%d, salah-yakin %d -> %s" % (n, tot, cw, "LOLOS" if n > 21 and cw <= 4 else "GAGAL (tetap t=0,5)"))
