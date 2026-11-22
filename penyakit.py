"""Tahap 2: data "tanaman_penyakit" (tomat, kentang, paprika) dari data/processed (tahap 1).

Pakai:  python penyakit.py   -> data/penyakit/{train,val,test}/<kelas>/*.jpg (hardlink) + data/penyakit/split.csv
Split mengikuti data/processed/split.csv (citra test tahap 1 tetap test). Citra sudah 128x128 (preprocess.py).
Kelas di luar cakupan (leaves, iNat, PlantDoc tanaman lain, padi sejak P2) tidak dipakai. Lihat CLAUDE.md "Tahap 2".
"""
import csv
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
DATA = ROOT / "data" / "penyakit"

# id kelas (nama folder) -> label tampilan
NAMA = {
    "kentang_hawar_awal": "Kentang · Hawar awal (early blight)",
    "kentang_hawar_daun": "Kentang · Hawar daun (late blight)",
    "kentang_sehat": "Kentang · Sehat",
    "paprika_bercak_bakteri": "Paprika · Bercak bakteri",
    "paprika_sehat": "Paprika · Sehat",
    "tomat_bercak_bakteri": "Tomat · Bercak bakteri",
    "tomat_bercak_septoria": "Tomat · Bercak Septoria",
    "tomat_bercak_target": "Tomat · Bercak target",
    "tomat_hawar_awal": "Tomat · Hawar awal (early blight)",
    "tomat_hawar_daun": "Tomat · Hawar daun (late blight)",
    "tomat_jamur_daun": "Tomat · Jamur daun (leaf mold)",
    "tomat_keriting_kuning": "Tomat · Virus keriting daun kuning",
    "tomat_mosaik": "Tomat · Virus mosaik",
    "tomat_sehat": "Tomat · Sehat",
    "tomat_tungau": "Tomat · Tungau laba-laba",
}
CLASSES = sorted(NAMA)  # indeks model = urutan ini

# orig_class dinormalisasi (huruf kecil, tanpa simbol) -> kelas. PV = PlantVillage, PD = PlantDoc.
MAP = {
    "pepperbellbacterialspot": "paprika_bercak_bakteri", "bellpepperleafspot": "paprika_bercak_bakteri",  # PD "leaf spot"
    "pepperbellhealthy": "paprika_sehat", "bellpepperleaf": "paprika_sehat",
    "potatoearlyblight": "kentang_hawar_awal", "potatoleafearlyblight": "kentang_hawar_awal",
    "potatolateblight": "kentang_hawar_daun", "potatoleaflateblight": "kentang_hawar_daun",
    "potatohealthy": "kentang_sehat",
    "tomatobacterialspot": "tomat_bercak_bakteri", "tomatoleafbacterialspot": "tomat_bercak_bakteri",
    "tomatoearlyblight": "tomat_hawar_awal", "tomatoearlyblightleaf": "tomat_hawar_awal",
    "tomatolateblight": "tomat_hawar_daun", "tomatoleaflateblight": "tomat_hawar_daun",
    "tomatoleafmold": "tomat_jamur_daun", "tomatomoldleaf": "tomat_jamur_daun",
    "tomatoseptorialeafspot": "tomat_bercak_septoria",
    "tomatospidermitestwospottedspidermite": "tomat_tungau", "tomatotwospottedspidermitesleaf": "tomat_tungau",
    "tomatotargetspot": "tomat_bercak_target",
    "tomatotomatoyellowleafcurlvirus": "tomat_keriting_kuning", "tomatoleafyellowvirus": "tomat_keriting_kuning",
    "tomatotomatomosaicvirus": "tomat_mosaik", "tomatoleafmosaicvirus": "tomat_mosaik",
    "tomatohealthy": "tomat_sehat", "tomatoleaf": "tomat_sehat",
}
SOURCES = ("plantvillage", "plantdoc")  # padi (rice) dikeluarkan di P2: ±28 citra train/kelas, tidak cukup


def class_of(source, orig_class):
    """Kelas tahap 2, atau None bila di luar cakupan."""
    if source not in SOURCES:
        return None
    return MAP.get(re.sub(r"[^a-z]", "", orig_class.lower()))


def main():
    with (PROCESSED / "split.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        c = class_of(r["source"], r["orig_class"])
        if c is None:
            # PlantVillage wajib terpetakan semua; yang lain boleh di luar cakupan
            assert r["source"] != "plantvillage", "kelas tak terpetakan: " + r["orig_class"]
            continue
        dst = DATA / r["split"] / c / Path(r["file"]).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.link(PROCESSED / r["file"], dst)
        out.append({"file": dst.relative_to(DATA).as_posix(), "split": r["split"], "label": c,
                    "source": r["source"], "orig_class": r["orig_class"]})
    with (DATA / "split.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    # self-check: jumlah di disk == csv per split x kelas, semua kelas ada di train
    cnt = Counter((o["split"], o["label"]) for o in out)
    for s in ("train", "val", "test"):
        for c in CLASSES:
            d = DATA / s / c
            on_disk = sum(1 for _ in d.iterdir()) if d.exists() else 0
            assert on_disk == cnt[(s, c)], "{}/{}: disk={} csv={}".format(s, c, on_disk, cnt[(s, c)])
    assert all(cnt[("train", c)] > 0 for c in CLASSES), "ada kelas tanpa data train"
    print("{:<24} {:>6} {:>5} {:>5}   (PlantDoc train)".format("kelas", "train", "val", "test"))
    pd = Counter(o["label"] for o in out if o["source"] == "plantdoc" and o["split"] == "train")
    for c in CLASSES:
        print("{:<24} {:>6} {:>5} {:>5}   {}".format(c, cnt[("train", c)], cnt[("val", c)], cnt[("test", c)], pd[c]))
    print("total {} citra -> {}".format(len(out), DATA.relative_to(ROOT).as_posix()))


if __name__ == "__main__":
    main()
