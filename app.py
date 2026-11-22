"""Web app Flask: landing page (/) + deteksi daun (/deteksi: upload/kamera -> sehat/sakit + confidence).

Lokal:  python app.py  -> http://127.0.0.1:5000
PythonAnywhere: WSGI file -> `from app import app as application`
Butuh models/model.tflite (R1) + model_v3.tflite + model_v4.tflite: ensemble, lihat inference.ENSEMBLE.
Opsional: models/model_jenis.tflite (jenis tanaman, J3) dan models/model_penyakit.tflite (nonaktif, SHOW_DISEASE).
"""
import base64
import io
import json
from pathlib import Path

from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

import jenis
import penyakit
from inference import JENIS_MODEL, JENIS_THRESHOLD, PENYAKIT_MODEL, PENYAKIT_THRESHOLD, Ensemble, Predictor

ALLOWED = {"jpg", "jpeg", "png", "webp", "bmp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB, cukup untuk foto HP
predictor = Ensemble()  # R1 + V3 + V4, dimuat sekali saat start
# tahap 2 (opsional): dugaan tanaman & penyakit untuk tomat/kentang/paprika
# DINONAKTIFKAN (aturan Tahap 2b, CLAUDE.md): model jenis J1 gagal -> dugaan penyakit tanpa penyaring jenis keliru
# pada daun di luar tomat/kentang/paprika (uji 51 foto: 3 tampil, 3 salah). Aktifkan lagi hanya dengan model jenis yang lolos.
SHOW_DISEASE = False
disease = Predictor(PENYAKIT_MODEL, PENYAKIT_THRESHOLD, penyakit.CLASSES) if SHOW_DISEASE and PENYAKIT_MODEL.exists() else None
# jenis tanaman (J3, lolos aturan val & test; CLAUDE.md). Fitur nama penyakit sengaja tidak ditampilkan (keputusan user).
species = Predictor(JENIS_MODEL, JENIS_THRESHOLD, jenis.CLASSES) if JENIS_MODEL.exists() else None
_plants = Path(__file__).with_name("static") / "plants" / "plants.json"  # dibuat scripts/make_plant_gallery.py
PLANTS = json.loads(_plants.read_text(encoding="utf-8")) if _plants.exists() else []


def preview_data_url(data):
    """Thumbnail base64 untuk ditampilkan (tidak menyimpan file upload ke disk)."""
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        im.thumbnail((320, 320))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/tanaman")
def tanaman():
    groups = [(st, [p for p in PLANTS if p["status"] == st]) for st in ("andal", "cukup", "terbatas")]
    return render_template("tanaman.html", groups=groups, total=len(PLANTS))


@app.route("/deteksi", methods=["GET", "POST"])
def deteksi():
    if request.method == "GET":
        return render_template("deteksi.html")
    f = request.files.get("image")
    if not f or not f.filename:
        return render_template("deteksi.html", error="Pilih file foto daun terlebih dahulu."), 400
    if f.filename.rsplit(".", 1)[-1].lower() not in ALLOWED:
        return render_template("deteksi.html", error="Format harus: " + ", ".join(sorted(ALLOWED))), 400
    data = f.read()
    try:
        result = predictor.predict(io.BytesIO(data))
        preview = preview_data_url(data)
        jenis_hasil = None
        if species:
            j = species.predict(io.BytesIO(data))
            top = sorted(j["probs"].items(), key=lambda kv: -kv[1])[:3]
            jenis_hasil = {"uncertain": j["uncertain"], "other": j["label"] == "lainnya",
                           "top": [(jenis.NAMA[c], p) for c, p in top]}
        dugaan = None
        if disease:
            d = disease.predict(io.BytesIO(data))
            top = sorted(d["probs"].items(), key=lambda kv: -kv[1])[:3]
            dugaan = {"shown": not d["uncertain"], "top": [(penyakit.NAMA[c], p) for c, p in top]}
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return render_template("deteksi.html", error="File bukan citra yang valid atau rusak."), 400
    return render_template("deteksi.html", result=result, preview=preview, threshold=predictor.threshold, dugaan=dugaan, jenis=jenis_hasil)


@app.errorhandler(413)
def too_large(_):
    return render_template("deteksi.html", error="Ukuran file maksimal 10 MB."), 413


if __name__ == "__main__":
    app.run(debug=True)
