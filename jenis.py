"""Tahap 2b: data JENIS tanaman (22 kelas termasuk "lainnya"). Lihat CLAUDE.md "Tahap 2b".

Pakai:  python jenis.py   -> data/jenis/{train,val,test}/<kelas>/*.jpg + data/jenis/split.csv
Sumber: citra tahap 1 yang jenisnya diketahui (hardlink dari data/processed, split ikut tahap 1)
        + iNaturalist jenis (archive (5)/inat_jenis, dari `scripts/fetch_inat.py --set jenis`; split baru 70/15/15).
Per (split, kelas) dibatasi CAP citra; foto lapangan (non-PlantVillage) diprioritaskan agar model tidak didominasi foto studio.
"""
import csv
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
DATA = ROOT / "data" / "jenis"
NEW = ROOT / "archive (5)" / "inat_jenis"
SEED = 42
CAP = {"train": 1400, "val": 300, "test": 300}

# J1/J2 memakai 22 kelas (NAMA_22). J3 (CLAUDE.md): 8 jenis terkuat di val J2 + "lainnya"; jenis lain digabung ke "lainnya".
NAMA_22 = {
    "anggur": "Anggur", "apel": "Apel", "bayam": "Bayam", "jagung": "Jagung", "jambu_biji": "Jambu biji",
    "jarak_kepyar": "Jarak kepyar", "jarak_pagar": "Jarak pagar", "jati": "Jati", "kangkung": "Kangkung",
    "kentang": "Kentang", "lainnya": "Tanaman lain (di luar daftar)", "mangga": "Mangga", "mawar": "Mawar", "mint": "Mint",
    "monstera": "Monstera & kerabatnya", "padi": "Padi", "paprika_cabai": "Paprika / cabai", "pepaya": "Pepaya",
    "singkong": "Singkong", "sirih": "Sirih", "tomat": "Tomat", "ubi_jalar": "Ubi jalar",
}
KEEP = ("paprika_cabai", "padi", "jagung", "jambu_biji", "mangga", "monstera", "kentang", "tomat", "lainnya")  # J3
NAMA = {k: NAMA_22[k] for k in KEEP}  # untuk J1/J2: NAMA = NAMA_22
CLASSES = sorted(NAMA)  # indeks model = urutan ini
# id tanaman di scripts/make_plant_gallery.py -> kelas jenis (yang tidak ada di sini & bukan kelas = "lainnya")
GALLERY = {"paprika": "paprika_cabai", "jambu": "jambu_biji", "ubi": "ubi_jalar"}
# urutan isi kuota CAP: lapangan (iNat) > web/lapangan (PlantDoc, rice) > studio (leaves, PlantVillage)
PRIORITY = {"inatj": 0, "inat": 0, "plantdoc": 1, "rice": 1, "leaves": 2, "plantvillage": 3}


def main():
    sys.path.insert(0, str(ROOT))
    from sklearn.model_selection import train_test_split

    from preprocess import load_image
    from scripts.make_plant_gallery import plant_of

    pool = defaultdict(list)  # (split, kelas) -> [(sumber, path asal, perlu_resize)]
    with (PROCESSED / "split.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = plant_of(r["source"], r["orig_class"], r["file"])
            if p is None:  # kelompok penyakit/hama iNat: jenis tanaman tidak diketahui
                continue
            c = GALLERY.get(p, p) if GALLERY.get(p, p) in NAMA else "lainnya"
            pool[(r["split"], c)].append((r["source"], PROCESSED / r["file"], False))

    with (NEW / "attribution.csv").open(encoding="utf-8") as f:
        new = [r for r in csv.DictReader(f) if (ROOT / r["file"]).exists()]
    tr, rest = train_test_split(new, test_size=0.30, stratify=[r["label"] for r in new], random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=[r["label"] for r in rest], random_state=SEED)
    for split, part in (("train", tr), ("val", va), ("test", te)):
        for r in part:
            pool[(split, r["label"] if r["label"] in NAMA else "lainnya")].append(("inatj", ROOT / r["file"], True))

    rng = random.Random(SEED); out = []
    for (split, c), items in sorted(pool.items()):
        items = sorted(items, key=lambda x: str(x[1]))
        rng.shuffle(items)
        items.sort(key=lambda x: PRIORITY.get(x[0], 1))  # stabil: foto lapangan dulu, studio terakhir
        for src, path, resize in items[:CAP[split]]:
            name = path.name if not resize else "inatj__{}__{}.jpg".format(path.parent.name.replace(" ", "_"), path.stem)
            dst = DATA / split / c / (Path(name).stem + ".jpg")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                if resize:
                    load_image(path).save(dst, "JPEG", quality=95)
                else:
                    os.link(path, dst)
            out.append({"file": dst.relative_to(DATA).as_posix(), "split": split, "label": c, "source": src})
    with (DATA / "split.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    # self-check: disk == csv, semua kelas punya data di tiap split, tidak ada nama ganda lintas split
    cnt = Counter((o["split"], o["label"]) for o in out)
    for s in CAP:
        for c in CLASSES:
            d = DATA / s / c
            assert (sum(1 for _ in d.iterdir()) if d.exists() else 0) == cnt[(s, c)], (s, c)
            assert cnt[(s, c)] > 0, "kelas {} kosong di {}".format(c, s)
    assert len({Path(o["file"]).name for o in out}) == len(out), "citra muncul di >1 split/kelas"
    src = Counter((o["label"], o["source"]) for o in out if o["split"] == "train")
    print("{:<15} {:>5} {:>4} {:>4}   train per sumber".format("kelas", "train", "val", "test"))
    for c in CLASSES:
        print("{:<15} {:>5} {:>4} {:>4}   {}".format(c, cnt[("train", c)], cnt[("val", c)], cnt[("test", c)],
                                                 {s: n for (k, s), n in sorted(src.items()) if k == c}))
    print("total {} citra -> {}".format(len(out), DATA.relative_to(ROOT).as_posix()))


if __name__ == "__main__":
    main()
