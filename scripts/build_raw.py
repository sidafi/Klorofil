"""Susun data/raw/{sehat,sakit}/ dari data/manifest.csv (hasil inspect_data.py).

Hanya file valid & unik (ok=1, dup_of kosong). File di-hardlink (hemat disk, NTFS);
jika gagal, di-copy. Nama file: <sumber>__<kelas_asal>__<nama_asli> agar tidak bentrok.
Juga menulis data/raw/sources.csv (file, label, source, orig_class) untuk evaluasi per sumber.

Pakai:  python scripts/build_raw.py
"""
import csv
import os
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.csv"
RAW = ROOT / "data" / "raw"


def slug(s):
    return re.sub(r"[^A-Za-z0-9.-]+", "_", s).strip("_")


def main():
    with MANIFEST.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["ok"] == "1" and not r["dup_of"]]

    for label in ("sehat", "sakit"):
        (RAW / label).mkdir(parents=True, exist_ok=True)

    out, names, copied = [], set(), 0
    for r in rows:
        src = ROOT / r["path"]
        name = "{}__{}__{}".format(r["source"], slug(r["orig_class"]), slug(src.name))
        # nama sama, isi beda (mis. PlantDoc train vs test): bedakan dengan md5. lower(): NTFS tidak case-sensitive
        if name.lower() in names:
            p = Path(name)
            name = "{}_{}{}".format(p.stem, r["md5"][:8], p.suffix)
        assert name.lower() not in names, "nama bentrok: " + name
        names.add(name.lower())
        dst = RAW / r["label"] / name
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
                copied += 1
        out.append({"file": "{}/{}".format(r["label"], name), "label": r["label"],
                    "source": r["source"], "orig_class": r["orig_class"]})

    with (RAW / "sources.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    # self-check: isi folder harus sama persis dengan sources.csv
    for label, n in Counter(r["label"] for r in out).items():
        on_disk = sum(1 for p in (RAW / label).iterdir() if p.is_file())
        assert on_disk == n, "{}: disk={} csv={}".format(label, on_disk, n)
        print("  {:<6} {:>6} file".format(label, n))
    print("total {} | di-copy (hardlink gagal): {} | data/raw/sources.csv".format(len(out), copied))


if __name__ == "__main__":
    main()
