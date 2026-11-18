"""Web app Flask: landing page (/) + deteksi daun (/deteksi: upload/kamera -> sehat/sakit + confidence).

Lokal:  python app.py  -> http://127.0.0.1:5000
PythonAnywhere: WSGI file -> `from app import app as application`
Butuh models/model.tflite (R1) + models/model_v3.tflite (V3): ensemble, lihat inference.ENSEMBLE.
"""
import base64
import io
import json
from pathlib import Path

from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

from inference import Ensemble

ALLOWED = {"jpg", "jpeg", "png", "webp", "bmp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB, cukup untuk foto HP
predictor = Ensemble()  # R1 + V3, dimuat sekali saat start
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
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return render_template("deteksi.html", error="File bukan citra yang valid atau rusak."), 400
    return render_template("deteksi.html", result=result, preview=preview, threshold=predictor.threshold)


@app.errorhandler(413)
def too_large(_):
    return render_template("deteksi.html", error="Ukuran file maksimal 10 MB."), 413


if __name__ == "__main__":
    app.run(debug=True)
