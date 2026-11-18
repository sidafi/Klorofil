"""Galeri "daun yang dikenali" untuk web app: static/plants/<id>.jpg + static/plants/plants.json.

Per tanaman: jumlah citra sehat/sakit (seluruh data), akurasi ensemble di VAL (citra asli, jalur app, t=0,5),
status, dan satu contoh gambar dari split TRAIN (bukan test).
Butuh data/raw/sources.csv, data/processed/split.csv, data/tta_val_ckpt.npy (dari scripts/eval_tta.py).
Pakai:  python scripts/make_plant_gallery.py
"""
import csv, io, json, random, re, urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "static" / "plants"
SEED = 42

# (id, nama, latin)
PLANTS = {
    "tomat": ("Tomat", "Solanum lycopersicum"), "kentang": ("Kentang", "Solanum tuberosum"),
    "paprika": ("Paprika", "Capsicum annuum"), "padi": ("Padi", "Oryza sativa"),
    "mangga": ("Mangga", "Mangifera indica"), "jambu": ("Jambu biji", "Psidium guajava"),
    "jamblang": ("Jamblang (duwet)", "Syzygium cumini"), "jarak_pagar": ("Jarak pagar", "Jatropha curcas"),
    "delima": ("Delima", "Punica granatum"), "lemon": ("Lemon", "Citrus limon"),
    "kemangi": ("Kemangi", "Ocimum basilicum"), "arjuna": ("Arjuna", "Terminalia arjuna"),
    "pulai": ("Pulai", "Alstonia scholaris"), "malapari": ("Malapari", "Pongamia pinnata"),
    "maja": ("Maja", "Aegle marmelos"), "chinar": ("Chinar", "Platanus orientalis"),
    "pepaya": ("Pepaya", "Carica papaya"), "singkong": ("Singkong", "Manihot esculenta"),
    "ubi": ("Ubi jalar", "Ipomoea batatas"), "jati": ("Jati", "Tectona grandis"),
    "monstera": ("Monstera & kerabatnya", "Monstera, Philodendron, Epipremnum, dll."),
    "apel": ("Apel", "Malus domestica"), "anggur": ("Anggur", "Vitis vinifera"),
    "jagung": ("Jagung", "Zea mays"), "kedelai": ("Kedelai", "Glycine max"),
    "stroberi": ("Stroberi", "Fragaria × ananassa"), "labu": ("Labu", "Cucurbita sp."),
    "persik": ("Persik", "Prunus persica"), "ceri": ("Ceri", "Prunus avium"),
    "blueberry": ("Blueberry", "Vaccinium sp."), "raspberry": ("Raspberry", "Rubus idaeus"),
}
# dataset "leaves" (Chouhan dkk., 2019): kode berkas -> tanaman (0001-0011 sehat, 0012-0022 sakit)
LEAVES = ["mangga", "arjuna", "pulai", "jambu", "jamblang", "jarak_pagar", "malapari", "kemangi", "delima", "lemon", "chinar",
          "mangga", "arjuna", "pulai", "jambu", "maja", "jamblang", "jarak_pagar", "malapari", "delima", "lemon", "chinar"]
PREFIX = {"pepper": "paprika", "bell_pepper": "paprika", "potato": "kentang", "tomato": "tomat", "apple": "apel",
          "blueberry": "blueberry", "cherry": "ceri", "corn": "jagung", "peach": "persik", "raspberry": "raspberry",
          "soyabean": "kedelai", "squash": "labu", "strawberry": "stroberi", "grape": "anggur"}
INAT = {"Carica_papaya": "pepaya", "Ipomoea_batatas": "ubi", "Mangifera_indica": "mangga", "Manihot_esculenta": "singkong",
        "Psidium_guajava": "jambu", "Tectona_grandis": "jati"}
AROID = ("Monstera", "Philodendron", "Epipremnum", "Rhaphidophora", "Amydrium", "Anthurium", "Alocasia", "Cercestis", "Dracontium")
SOURCE_NAME = {"plantvillage": "PlantVillage", "leaves": "Leaf Images DB (Chouhan dkk., 2019)", "rice": "Rice Leaf Diseases (UCI)",
               "plantdoc": "PlantDoc (Singh dkk., 2020)", "inat": "iNaturalist"}

# Foto contoh pengganti (iNaturalist CC0/CC-BY, tanpa bunga/buah; dipilih manual): dipakai bila contoh dari dataset hanya
# foto PlantDoc (hasil scraping web, hak cipta tak jelas) atau tidak menampilkan daun dengan jelas.
I = "https://inaturalist-open-data.s3.amazonaws.com/photos/"
PHOTO = {
    "apel": (I + "288214338/medium.jpeg", "(c) Fyodor Pudovikov, CC BY"),
    "anggur": (I + "127305260/medium.jpeg", "(c) Patrick Hacker, CC BY"),
    "jagung": (I + "574303418/medium.jpg", "CC0 (no rights reserved)"),
    "kedelai": (I + "146321069/medium.jpeg", "(c) chiuluan, CC BY"),
    "stroberi": (I + "657242112/medium.jpg", "(c) Sergey Basalaev, CC BY"),
    "labu": (I + "264954073/medium.jpeg", "(c) Pablo Cauê da Silva Toledo, CC BY"),
    "persik": (I + "420085678/medium.jpeg", "(c) Pto, CC BY"),
    "ceri": (I + "407020348/medium.jpeg", "(c) Daniel Cahen, CC BY"),
    "blueberry": (I + "723580808/medium.jpg", "CC0 (no rights reserved)"),
    "raspberry": (I + "539292389/medium.jpg", "(c) Jo Roberts, CC BY"),
    "ubi": (I + "339921708/medium.jpeg", "(c) Emily Abrego, CC BY"),
    "monstera": (I + "399401498/medium.jpeg", "(c) Lisa Wiencek Gerbec, CC BY"),
}

# contoh dari data lokal yang dipilih manual (id observasi iNaturalist; daun terlihat jelas, bukan bunga/pohon utuh)
LOCAL = {"jati": "394141343_", "singkong": "404933958_", "pepaya": "413112706_"}


def save_thumb(src, dst):
    with Image.open(src) as im:
        im.draft("RGB", (640, 640)); im = im.convert("RGB")
        w, h = im.size; s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((320, 320), Image.Resampling.LANCZOS)
        im.save(dst, quality=80, optimize=True)


def plant_of(source, orig, file):
    orig = orig.replace(" ", "_")
    if source == "rice":
        return "padi"
    if source == "leaves":
        return LEAVES[int(re.search(r"__(\d{4})_", file).group(1)) - 1]
    if source == "inat":
        return "monstera" if orig.startswith(AROID) else INAT.get(orig)  # None = kelompok penyakit lintas tanaman
    key = orig.lower().replace("__", "_")
    for p in sorted(PREFIX, key=len, reverse=True):
        if key.startswith(p):
            return PREFIX[p]
    return None


def status(acc, n, labels):
    if len(labels) < 2:
        return "terbatas", "Model hanya pernah melihat daun %s tanaman ini" % next(iter(labels))
    if n < 30:
        return "terbatas", "Data uji terlalu sedikit untuk dinilai"
    if acc >= .90:
        return "andal", "Akurasi tinggi pada data uji"
    if acc >= .75:
        return "cukup", "Cukup baik, kadang keliru"
    return "terbatas", "Masih sering keliru"


def main():
    raw = list(csv.DictReader(open(ROOT / "data/raw/sources.csv", encoding="utf-8")))
    split = list(csv.DictReader(open(ROOT / "data/processed/split.csv", encoding="utf-8")))
    which = {Path(r["file"]).stem: r["split"] for r in split}
    val = [r for r in split if r["split"] == "val"]
    P = np.load(ROOT / "data/tta_val_ckpt.npy")[:, 0]
    assert len(P) == len(val), "checkpoint val tidak cocok dengan split.csv"
    attrib = {}
    att_csv = ROOT / "archive (4)/inaturalist/attribution.csv"
    if att_csv.exists():
        for a in csv.DictReader(open(att_csv, encoding="utf-8")):
            attrib[Path(a["file"]).name] = a

    count = defaultdict(lambda: {"sehat": 0, "sakit": 0}); cand = defaultdict(list)
    for r in raw:
        p = plant_of(r["source"], r["orig_class"], r["file"])
        if p is None:
            continue
        count[p][r["label"]] += 1
        if which.get(Path(r["file"]).stem) == "train":
            cand[p].append(r)
    hit = defaultdict(list)
    for r, ps in zip(val, P):
        p = plant_of(r["source"], r["orig_class"], r["file"])
        if p:
            hit[p].append(("sehat" if ps >= .5 else "sakit") == r["label"])

    OUT.mkdir(parents=True, exist_ok=True); rng = random.Random(SEED); out = []
    order = {"leaves": 0, "plantvillage": 1, "inat": 2, "rice": 3, "plantdoc": 4}  # utamakan foto satu daun yang jelas
    for pid, (nama, latin) in PLANTS.items():
        c = count[pid]; labels = {k for k, v in c.items() if v}
        h = hit[pid]; acc = float(np.mean(h)) if h else 0.0
        st, ket = status(acc, len(h), labels)
        if pid in PHOTO:
            url, credit = PHOTO[pid]
            req = urllib.request.Request(url, headers={"User-Agent": "leaf-gallery-script/1.0"})
            save_thumb(io.BytesIO(urllib.request.urlopen(req, timeout=60).read()), OUT / f"{pid}.jpg")
            credit += " via iNaturalist"; contoh = "lapangan"
        else:
            pool = cand[pid]
            best_src = min(order[r["source"]] for r in pool)
            pool = [r for r in pool if order[r["source"]] == best_src]
            if pid != "padi":
                pool = [r for r in pool if r["label"] == "sehat"] or pool
            if best_src == order["inat"]:  # utamakan lisensi paling bebas (CC0/CC-BY)
                pool = [r for r in pool if attrib.get(Path(r["file"]).name.split("__")[-1], {}).get("license") in ("cc0", "cc-by")] or pool
            if pid in LOCAL:
                pool = [r for r in pool if "__" + LOCAL[pid] in r["file"]]
                assert len(pool) == 1, (pid, len(pool))
            pick = rng.choice(sorted(pool, key=lambda r: r["file"]))
            save_thumb(ROOT / "data/raw" / pick["file"], OUT / f"{pid}.jpg")
            a = attrib.get(Path(pick["file"]).name.split("__")[-1])
            credit = a["attribution"] if a else SOURCE_NAME[pick["source"]]
            contoh = pick["label"]
        out.append({"id": pid, "nama": nama, "latin": latin, "sehat": c["sehat"], "sakit": c["sakit"],
                    "n_uji": len(h), "akurasi": round(acc, 3), "status": st, "keterangan": ket,
                    "contoh": contoh, "kredit": credit})
    rank = {"andal": 0, "cukup": 1, "terbatas": 2}
    out.sort(key=lambda x: (rank[x["status"]], -(x["sehat"] + x["sakit"])))
    json.dump(out, open(OUT / "plants.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for x in out:
        print("%-12s %-9s acc %.3f n=%4d  sehat %5d sakit %5d" % (x["id"], x["status"], x["akurasi"], x["n_uji"], x["sehat"], x["sakit"]))
    # self-check: semua tanaman punya gambar & tidak ada citra yang tak terpetakan selain kelompok penyakit iNat
    assert all((OUT / f"{x['id']}.jpg").exists() for x in out)
    unmapped = {(r["source"], r["orig_class"]) for r in raw if plant_of(r["source"], r["orig_class"], r["file"]) is None}
    assert all(s == "inat" for s, _ in unmapped), unmapped


if __name__ == "__main__":
    main()
