"""Inspeksi dataset mentah: jumlah per kelas/sumber, file rusak, duplikat (hash), resolusi.

Pakai:  python scripts/inspect_data.py
Hasil:  data/manifest.csv (satu baris per file) + ringkasan di terminal.
"""
import csv
import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "manifest.csv"

# (nama_sumber, folder, subfolder yang diabaikan)
SOURCES = [
    ("plantvillage", ROOT / "archive (1)" / "PlantVillage", {"PlantVillage"}),  # salinan ganda
    ("leaves", ROOT / "archive (2)" / "leaves_healthy_or_diseased", set()),
    ("rice", ROOT / "archive" / "rice_leaf_diseases", set()),
    # PlantDoc (foto lapangan, github.com/pratikkayal/PlantDoc-Dataset, CC-BY-4.0); split aslinya digabung
    ("plantdoc", ROOT / "archive (3)" / "PlantDoc" / "train", set()),
    ("plantdoc", ROOT / "archive (3)" / "PlantDoc" / "test", set()),
    # iNaturalist (foto lapangan CC0/CC-BY/CC-BY-NC, scripts/fetch_inat.py); folder induk = label, subfolder = takson
    # v4: kelompok sehat dengan mayoritas foto buah/bunga/serangga dibuang (jalan pintas "buah/serangga = sehat")
    ("inat", ROOT / "archive (4)" / "inaturalist" / "sehat", {"Artocarpus heterophyllus", "Theobroma cacao", "Annona muricata",
                                                               "Coffea arabica", "Musa", "Citrus"}),
    ("inat", ROOT / "archive (4)" / "inaturalist" / "sakit", {"scab (Venturia inaequalis)"}),  # mayoritas foto buah, bukan daun
]

# kelas PlantDoc tanpa kata "healthy": daun sehat = "<tanaman> leaf" tanpa nama penyakit
PLANTDOC_HEALTHY = {"apple leaf", "bell_pepper leaf", "blueberry leaf", "cherry leaf", "peach leaf",
                    "raspberry leaf", "soyabean leaf", "strawberry leaf", "tomato leaf", "grape leaf"}


def label_of(class_dir, base=None):
    if base is not None and base.name in ("sehat", "sakit"):  # sumber berlabel dari folder induk (inat)
        return base.name
    name = class_dir.lower()
    return "sehat" if "healthy" in name or name in PLANTDOC_HEALTHY else "sakit"


def inspect(item, retries=3):
    row = _inspect(item)
    while retries and row["error"].startswith("MemoryError"):  # RAM penuh sesaat != file rusak
        retries -= 1
        row = _inspect(item)
    return row


def _inspect(item):
    source, class_dir, path, label = item
    row = {"path": path.relative_to(ROOT).as_posix(), "source": source, "orig_class": class_dir,
           "label": label, "width": "", "height": "", "mode": "", "md5": "",
           "ok": 0, "error": "", "dup_of": ""}
    try:
        data = path.read_bytes()
        row["md5"] = hashlib.md5(data).hexdigest()
        with Image.open(path) as im:
            row["width"], row["height"], row["mode"] = im.width, im.height, im.mode
            im.verify()
        row["ok"] = 1
    except Exception as e:  # file rusak / bukan citra
        row["error"] = "{}: {}".format(type(e).__name__, e)[:200]
    return row


def collect():
    items = []
    for source, base, skip in SOURCES:
        if not base.is_dir():
            print("PERINGATAN: folder tidak ada:", base)
            continue
        for d in sorted(p for p in base.iterdir() if p.is_dir() and p.name not in skip):
            items += [(source, d.name, f, label_of(d.name, base)) for f in sorted(d.iterdir()) if f.is_file()]
    return items


def main():
    items = collect()
    print("Memeriksa {} file...".format(len(items)))
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(inspect, items, chunksize=64))

    first = {}
    for r in rows:
        if r["md5"] in first:
            r["dup_of"] = first[r["md5"]]
        elif r["md5"]:
            first[r["md5"]] = r["path"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    good = [r for r in rows if r["ok"] and not r["dup_of"]]
    bad = [r for r in rows if not r["ok"]]
    dups = [r for r in rows if r["dup_of"]]
    by_path = {r["path"]: r for r in rows}
    cross = [r for r in dups if by_path[r["dup_of"]]["source"] != r["source"]]
    conflict = [r for r in dups if by_path[r["dup_of"]]["label"] != r["label"]]

    print("\n== Jumlah per sumber x label (bersih: valid & unik) ==")
    c = Counter((r["source"], r["label"]) for r in good)
    for s in dict.fromkeys(s for s, _, _ in SOURCES):  # unik, urutan tetap
        print("  {:<13} sehat={:>6}  sakit={:>6}".format(s, c[(s, "sehat")], c[(s, "sakit")]))
    tot = Counter(r["label"] for r in good)
    print("  {:<13} sehat={:>6}  sakit={:>6}  total={}".format("TOTAL", tot["sehat"], tot["sakit"], len(good)))
    if tot["sehat"]:
        print("  rasio sehat:sakit = 1:{:.2f}".format(tot["sakit"] / tot["sehat"]))

    print("\n== Per kelas asal ==")
    for (s, k, lab), n in sorted(Counter((r["source"], r["orig_class"], r["label"]) for r in good).items()):
        print("  {:<13} {:<45} {:>5}  -> {}".format(s, k, n, lab))

    print("\n== Resolusi (WxH terbanyak per sumber) ==")
    res = defaultdict(Counter)
    for r in good:
        res[r["source"]]["{}x{}".format(r["width"], r["height"])] += 1
    for s, cnt in res.items():
        print("  {:<13} {}".format(s, ", ".join("{} ({})".format(k, v) for k, v in cnt.most_common(3))))
    modes = Counter(r["mode"] for r in good)
    print("  mode warna:", dict(modes))

    print("\n== Masalah ==")
    print("  file rusak/bukan citra : {}".format(len(bad)))
    for r in bad[:10]:
        print("    {}  [{}]".format(r["path"], r["error"]))
    print("  duplikat persis (md5)  : {}  (lintas sumber: {}, beda label: {})".format(len(dups), len(cross), len(conflict)))
    for r in conflict[:10]:
        print("    KONFLIK {} == {}".format(r["path"], r["dup_of"]))
    print("\nManifest:", OUT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
