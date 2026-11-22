"""Unduh foto lapangan berlisensi dari iNaturalist (API publik, tanpa login) untuk v3.

sehat: daun berlubang/terbelah ALAMI (Monstera dkk.) -> agar lubang alami tidak dianggap sakit.
sakit: daun terserang patogen/hama di lapangan (karat, embun tepung, bercak, penggorok daun)
       -> penyeimbang, agar model tidak belajar "latar kebun = sehat".
Hanya observasi research-grade, lisensi foto CC0 / CC-BY / CC-BY-NC, 1 foto per observasi (cegah bocor split).
Hasil: archive (4)/inaturalist/{sehat,sakit}/<takson>/<obs>_<photo>.jpg + attribution.csv (wajib dikutip).

Pakai:  python scripts/fetch_inat.py              (data sehat/sakit, archive (4))
        python scripts/fetch_inat.py --set jenis  (tahap 2 jenis tanaman, archive (5)/inat_jenis; label = nama jenis)
"""
import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "archive (4)" / "inaturalist"
UA = {"User-Agent": "leaf-health-student-project/1.0"}
API = "https://api.inaturalist.org/v1/observations?"
LICENSES = "cc0,cc-by,cc-by-nc"

# (label, nama folder, parameter query, target jumlah)
QUERIES = [
    ("sehat", "Monstera adansonii", {"taxon_name": "Monstera adansonii"}, 500),
    ("sehat", "Monstera obliqua", {"taxon_name": "Monstera obliqua"}, 150),
    ("sehat", "Monstera deliciosa", {"taxon_name": "Monstera deliciosa"}, 600),  # termasuk var. borsigiana
    ("sehat", "Monstera epipremnoides", {"taxon_name": "Monstera epipremnoides"}, 50),
    ("sehat", "Rhaphidophora foraminifera", {"taxon_name": "Rhaphidophora foraminifera"}, 100),
    ("sehat", "Rhaphidophora tetrasperma", {"taxon_name": "Rhaphidophora tetrasperma"}, 50),
    ("sehat", "Epipremnum pinnatum", {"taxon_name": "Epipremnum pinnatum"}, 500),
    ("sehat", "Aponogeton madagascariensis", {"taxon_id": 472470}, 50),
    ("sehat", "Amydrium medium", {"taxon_name": "Amydrium medium"}, 150),
    ("sehat", "Cercestis mirabilis", {"taxon_name": "Cercestis mirabilis"}, 50),
    ("sehat", "Dracontium pittieri", {"taxon_name": "Dracontium pittieri"}, 50),
    ("sehat", "Philodendron selloum", {"taxon_name": "Thaumatophyllum bipinnatifidum"}, 400),
    ("sehat", "Anthurium pedatoradiatum", {"taxon_name": "Anthurium pedatoradiatum"}, 50),
    ("sehat", "Alocasia portei", {"taxon_name": "Alocasia portei"}, 50),
    ("sakit", "karat (Pucciniales)", {"taxon_id": 69968}, 500),
    ("sakit", "embun tepung (Erysiphaceae)", {"taxon_id": 55525}, 500),
    ("sakit", "bercak Septoria", {"taxon_id": 327977}, 300),
    ("sakit", "bercak Cercospora", {"taxon_id": 327955}, 300),
    ("sakit", "tar spot (Rhytisma)", {"taxon_name": "Rhytisma"}, 300),
    ("sakit", "black spot (Diplocarpon)", {"taxon_name": "Diplocarpon"}, 300),
    ("sakit", "scab (Venturia inaequalis)", {"taxon_name": "Venturia inaequalis"}, 200),
    ("sakit", "penggorok daun (leafmine)", {"term_id": 22, "term_value_id": 32}, 400),  # Evidence of Presence = Leafmine
    # --- v4: kerusakan hama di lapangan (semua quality grade: penyebab kerusakan sering tak teridentifikasi)
    ("sakit", "kerusakan hama (Unidentifiable Plant Damage)", {"project_id": 91554, "quality_grade": "casual,needs_id,research"}, 3200),
    ("sakit", "kerusakan hama (Feeding Damage on Plants)", {"project_id": 251377, "quality_grade": "casual,needs_id,research"}, 450),
    # --- v4: daun biasa tanaman tropis umum di lapangan (diasumsikan sehat -> ada noise)
    ("sehat", "Tectona grandis", {"taxon_name": "Tectona grandis"}, 300),
    ("sehat", "Mangifera indica", {"taxon_name": "Mangifera indica"}, 300),
    ("sehat", "Psidium guajava", {"taxon_name": "Psidium guajava"}, 300),
    ("sehat", "Carica papaya", {"taxon_name": "Carica papaya"}, 300),
    ("sehat", "Musa", {"taxon_name": "Musa"}, 300),
    ("sehat", "Manihot esculenta", {"taxon_name": "Manihot esculenta"}, 300),
    ("sehat", "Ipomoea batatas", {"taxon_name": "Ipomoea batatas"}, 250),
    ("sehat", "Artocarpus heterophyllus", {"taxon_name": "Artocarpus heterophyllus"}, 250),
    ("sehat", "Coffea arabica", {"taxon_name": "Coffea arabica"}, 250),
    ("sehat", "Theobroma cacao", {"taxon_name": "Theobroma cacao"}, 250),
    ("sehat", "Annona muricata", {"taxon_name": "Annona muricata"}, 200),
    ("sehat", "Citrus", {"taxon_name": "Citrus"}, 300),
]

# Tahap 2 jenis: (jenis, nama folder, query, target). Buang observasi beranotasi berbunga/berbuah (13, 14) agar daun dominan.
OUT_JENIS = ROOT / "archive (5)" / "inat_jenis"
NO_FLOWER = {"without_term_value_id": "13,14"}
# genus yang punya kelas sendiri (atau sudah di data) -> dikecualikan dari "lainnya"
GENUS_JENIS = {"Solanum", "Capsicum", "Oryza", "Zea", "Mangifera", "Psidium", "Carica", "Manihot", "Ipomoea", "Tectona",
               "Monstera", "Thaumatophyllum", "Philodendron", "Epipremnum", "Rhaphidophora", "Amydrium", "Piper", "Ricinus",
               "Jatropha", "Mentha", "Amaranthus", "Malus", "Vitis", "Rosa"}
QUERIES_JENIS = [
    ("tomat", "Solanum lycopersicum", {"taxon_name": "Solanum lycopersicum", **NO_FLOWER}, 400),
    ("kentang", "Solanum tuberosum", {"taxon_name": "Solanum tuberosum", **NO_FLOWER}, 400),
    ("paprika_cabai", "Capsicum", {"taxon_name": "Capsicum", **NO_FLOWER}, 400),
    ("padi", "Oryza sativa", {"taxon_name": "Oryza sativa", "quality_grade": "research,needs_id"}, 400),
    ("jagung", "Zea mays", {"taxon_name": "Zea mays", **NO_FLOWER}, 400),
    ("sirih", "Piper betle", {"taxon_name": "Piper betle", "quality_grade": "research,needs_id"}, 400),
    ("jarak_kepyar", "Ricinus communis", {"taxon_name": "Ricinus communis", **NO_FLOWER}, 400),
    ("jarak_pagar", "Jatropha curcas", {"taxon_name": "Jatropha curcas", **NO_FLOWER}, 400),
    ("mint", "Mentha", {"taxon_name": "Mentha", **NO_FLOWER}, 400),
    ("bayam", "Amaranthus", {"taxon_name": "Amaranthus", **NO_FLOWER}, 400),
    ("kangkung", "Ipomoea aquatica", {"taxon_name": "Ipomoea aquatica", **NO_FLOWER}, 400),
    ("apel", "Malus domestica", {"taxon_name": "Malus domestica", **NO_FLOWER}, 400),
    ("anggur", "Vitis vinifera", {"taxon_name": "Vitis vinifera", **NO_FLOWER}, 400),
    ("mawar", "Rosa", {"taxon_name": "Rosa", **NO_FLOWER}, 400),
    # "lainnya": tumbuhan di Indonesia selain GENUS_JENIS (agar model bisa menjawab "bukan tanaman di daftar")
    ("lainnya", "Plantae Indonesia", {"taxon_id": 47126, "place_id": 6966, **NO_FLOWER}, 1600),
]


def api(params):
    q = dict(photos="true", quality_grade="research", photo_license=LICENSES, per_page=200, order_by="id", order="desc")
    q.update(params)
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(API + urllib.parse.urlencode(q), headers=UA), timeout=60) as r:
                return json.load(r)
        except Exception as e:  # rate limit / jaringan: tunggu lalu ulang
            print("  api retry", attempt + 1, e)
            time.sleep(min(60, 5 * (attempt + 1)))
    return None  # gagal terus: pemanggil simpan hasil sebagian kelompok ini


def download(job):
    url, dst = job
    if dst.exists():
        return True
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                data = r.read()
            dst.write_bytes(data)
            return True
        except Exception:
            time.sleep(2 * (attempt + 1))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("sehat_sakit", "jenis"), default="sehat_sakit")
    jenis = ap.parse_args().set == "jenis"
    out, queries = (OUT_JENIS, QUERIES_JENIS) if jenis else (OUT, QUERIES)
    out.mkdir(parents=True, exist_ok=True)
    old = []
    if (out / "attribution.csv").exists():
        with (out / "attribution.csv").open(encoding="utf-8") as f:
            old = list(csv.DictReader(f))
    done = {r["group"] for r in old}
    # observasi yang sudah ada di data lain tidak diunduh ulang (cegah duplikat lintas split)
    seen = {r["observation"] for r in old}
    if jenis and (OUT / "attribution.csv").exists():
        with (OUT / "attribution.csv").open(encoding="utf-8") as f:
            seen |= {r["observation"] for r in csv.DictReader(f)}
    rows, jobs = [], []
    for label, folder, params, target in queries:
        if folder in done:  # sudah diunduh sebelumnya: jangan ubah data lama
            continue
        got, id_below = 0, None
        while got < target:
            d = api(dict(params, **({"id_below": id_below} if id_below else {})))
            time.sleep(1.1)  # sopan ke API (<= 60 req/menit)
            if d is None:
                print("  PERINGATAN: API gagal, {} disimpan sebagian ({} foto)".format(folder, got))
                break
            if not d["results"]:
                break
            for o in d["results"]:
                obs = "https://www.inaturalist.org/observations/{}".format(o["id"])
                genus = ((o.get("taxon") or {}).get("name") or "").split(" ")[0]
                if obs in seen or (label == "lainnya" and (genus in GENUS_JENIS or (o.get("taxon") or {}).get("rank") not in
                                                            ("species", "subspecies", "variety", "genus"))):
                    continue
                seen.add(obs)
                photo = next((p for p in o["photos"] if p.get("license_code") in LICENSES.split(",")), None)
                if photo is None:
                    continue
                dst = out / label / re.sub(r"[^\w .()-]", "_", folder) / "{}_{}.jpg".format(o["id"], photo["id"])
                jobs.append((photo["url"].replace("/square.", "/medium."), dst))
                rows.append({"file": dst.relative_to(ROOT).as_posix(), "label": label, "group": folder,
                             "taxon": (o.get("taxon") or {}).get("name", ""), "observation": obs,
                             "license": photo["license_code"], "attribution": photo.get("attribution", "")})
                got += 1
                if got >= target:
                    break
            id_below = d["results"][-1]["id"]
        print("{:<5} {:<32} {:>4}".format(label, folder, got), flush=True)

    for _, dst in jobs:
        dst.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        ok = list(ex.map(download, jobs))
    rows = old + [r for r, k in zip(rows, ok) if k]
    with (out / "attribution.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print("unduh baru OK {}/{} | per label {} -> {}".format(sum(ok), len(ok), dict(Counter(r["label"] for r in rows)), out.relative_to(ROOT)))


if __name__ == "__main__":
    main()
