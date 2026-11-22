"""Inferensi sehat/sakit dari citra daun (dipakai app.py).

Ekspor sekali (butuh TensorFlow):  python inference.py --export   -> models/model.tflite
                                   python inference.py --export --task penyakit|jenis -> models/model_<task>.tflite (tahap 2)
Prediksi:                          python inference.py foto1.jpg foto2.jpg

Runtime cukup interpreter TFLite (ai_edge_litert / tflite_runtime / tensorflow), muat di akun gratis PythonAnywhere.
Preprocessing identik dengan training: load_image() (crop persegi + resize 128) lalu / 255.
"""
import argparse
from pathlib import Path

import numpy as np

from preprocess import CLASS_NAMES, load_image

ROOT = Path(__file__).resolve().parent
KERAS_MODEL = ROOT / "models" / "best.keras"
TFLITE_MODEL = ROOT / "models" / "model.tflite"  # R1: kuat di foto studio
# Model deploy = ensemble R1 + V3 + V4 (bobot 0,1/0,1/0,8 dipilih di val; lolos aturan stop v4).
# V3: + foto lapangan iNaturalist; V4: + kerusakan hama & daun tropis umum.
ENSEMBLE = [(TFLITE_MODEL, 0.1), (ROOT / "models" / "model_v3.tflite", 0.1), (ROOT / "models" / "model_v4.tflite", 0.8)]
# ensemble: confidence lebih rendah (model kadang tak sepakat). Tuning di val (R1+V3+V4): 13,9% ditandai, menangkap 61,5% error, akurasi yakin 0,968
ENSEMBLE_THRESHOLD = 0.70
# confidence < ini -> "tidak yakin". Tuning di val (model R1): 8,6% ditandai, menangkap 65% error, akurasi yang yakin 0,987
THRESHOLD = 0.85
PENYAKIT_MODEL = ROOT / "models" / "model_penyakit.tflite"  # tahap 2 (penyakit.py), run P2
# tahap 2: tampilkan dugaan penyakit hanya bila keyakinan >= ini. Ambang terkecil dgn akurasi tampil >= 0,95 di val
# (val: 81% tampil, akurasi 0,958; foto lapangan PlantDoc hanya 5% lolos -> sebagian besar "tidak dikenali")
PENYAKIT_THRESHOLD = 0.80
JENIS_MODEL = ROOT / "models" / "model_jenis.tflite"  # tahap 2b J3: MobileNetV2 transfer learning, 9 kelas (jenis.py)
# ambang terkecil dgn akurasi tampil >= 0,95 di val (jalur app): 59,6% tampil yakin, akurasi 0,954; di bawahnya = "kemungkinan"
JENIS_THRESHOLD = 0.85


def _interpreter(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf  # `from tensorflow.lite import Interpreter` gagal di TF lazy-load (Py3.13)
            Interpreter = tf.lite.Interpreter
    return Interpreter(model_path=str(path))


class Predictor:
    def __init__(self, model_path=TFLITE_MODEL, threshold=THRESHOLD, names=CLASS_NAMES):
        self.threshold = threshold
        self.names = names
        self.interp = _interpreter(model_path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.size = int(self.inp["shape"][1])

    def probs(self, image):
        """image: path / file-like (mis. werkzeug FileStorage.stream). Return array prob per kelas."""
        return self.probs_array(np.asarray(load_image(image, self.size), dtype=np.float32) / 255.0)

    def probs_array(self, x):
        """x: citra (size, size, 3) float 0-1 yang sudah dipreproses."""
        self.interp.set_tensor(self.inp["index"], x[None])
        self.interp.invoke()
        return self.interp.get_tensor(self.out["index"])[0]

    def predict(self, image):
        p = self.probs(image)
        i = int(p.argmax())
        return {"label": self.names[i], "confidence": float(p[i]),
                "uncertain": bool(p[i] < self.threshold),
                "probs": {c: float(v) for c, v in zip(self.names, p)}}


class Ensemble(Predictor):
    """Rata-rata berbobot beberapa model TFLite. Antarmuka sama dengan Predictor (dipakai app.py)."""

    def __init__(self, members=ENSEMBLE, threshold=ENSEMBLE_THRESHOLD):
        self.threshold = threshold
        self.names = CLASS_NAMES
        self.members = [(Predictor(p), w) for p, w in members]
        self.size = self.members[0][0].size
        assert all(m.size == self.size for m, _ in self.members) and abs(sum(w for _, w in self.members) - 1) < 1e-6

    def probs(self, image):
        x = np.asarray(load_image(image, self.size), dtype=np.float32) / 255.0  # baca sekali (stream hanya bisa dibaca sekali)
        return sum(w * m.probs_array(x) for m, w in self.members)


def export(n_check=64, task="sehat"):
    import tensorflow as tf

    if task != "sehat":
        import importlib
        data, out = importlib.import_module(task).DATA, ROOT / "models" / "model_{}.tflite".format(task)
    else:
        from preprocess import PROCESSED as data
        out = TFLITE_MODEL
    model = tf.keras.models.load_model(KERAS_MODEL)
    out.write_bytes(tf.lite.TFLiteConverter.from_keras_model(model).convert())

    # self-check: TFLite harus sama dengan Keras pada citra test
    files = sorted((data / "test").rglob("*.jpg"))
    assert files, "tidak ada citra test di " + str(data)
    files = files[::max(1, len(files) // n_check)][:n_check]  # sampel merata (semua kelas)
    pred = Predictor(out)
    x = np.stack([np.asarray(load_image(f), dtype=np.float32) / 255.0 for f in files])
    diff = np.abs(model.predict(x, verbose=0) - np.stack([pred.probs(f) for f in files])).max()
    assert diff < 1e-4, "TFLite beda dari Keras: {}".format(diff)
    print("Ekspor OK -> {} ({:.0f} KB), selisih maks vs Keras {:.2e} pada {} citra".format(
        out.relative_to(ROOT).as_posix(), out.stat().st_size / 1024, diff, len(files)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--export", action="store_true", help="konversi models/best.keras -> models/model.tflite")
    ap.add_argument("--task", choices=("sehat", "penyakit", "jenis"), default="sehat")
    args = ap.parse_args()
    if args.export:
        export(task=args.task)
    if args.images:
        pred = Ensemble()
        for f in args.images:
            r = pred.predict(f)
            print("{}: {} ({:.1%}){}".format(f, r["label"], r["confidence"], "  [TIDAK YAKIN]" if r["uncertain"] else ""))


if __name__ == "__main__":
    main()
