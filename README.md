# Deteksi Kondisi Daun (Sehat/Sakit) dengan CNN dari Nol

Tugas Besar AIT. Tahap 1: klasifikasi biner `sehat` / `sakit` dari citra daun.
Tahap 2 (pengembangan): jenis tanaman dan nama penyakit.

CNN disusun sendiri (Keras), bobot inisialisasi acak, tanpa arsitektur/bobot pretrained.

## Dataset
Gabungan 3 dataset Kaggle (±25.246 citra unik, sehat:sakit ±1:3,6):
- [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [Leaves healthy or diseased](https://www.kaggle.com/datasets/prasanshasatpathy/leaves-healthy-or-diseased)
- [Rice leaf diseases](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases)

Ekstrak ke root proyek sebagai `archive (1)/PlantVillage/`, `archive (2)/leaves_healthy_or_diseased/`, `archive/rice_leaf_diseases/`.

## Jalankan cepat (web app saja, tanpa dataset)
Model deploy (`models/*.tflite`) sudah ada di repo.
```
pip install -r requirements.txt
python app.py        # http://127.0.0.1:5000
```

## Setup lengkap (training/evaluasi)
```
pip install -r requirements-dev.txt
```
Dataset tidak ada di repo (±7 GB): unduh dari link di atas, ekstrak sesuai nama folder, lalu jalankan pipeline di bawah.
Data iNaturalist: `python scripts/fetch_inat.py`. Foto uji tim (`data/field_test/`) dibagikan terpisah.

## Pipeline
| Langkah | Perintah | Hasil |
|---|---|---|
| 1. Inspeksi data | `python scripts/inspect_data.py` | `data/manifest.csv` |
| 2. Susun kelas | `python scripts/build_raw.py` | `data/raw/{sehat,sakit}/` |
| 3. Split + resize | `python preprocess.py` | `data/processed/{train,val,test}/` (70/15/15) |
| 4. Cek model | `python model.py` | ringkasan arsitektur (±423 ribu parameter) |
| 5. Training | `python train.py` | `models/best.keras`, `models/history.png` |
| 6. Evaluasi | `python evaluate.py --split val` lalu (sekali, akhir) `python evaluate.py` | `models/eval_<split>/` (confusion matrix, report) |
| 7. Baseline | `python baseline.py` | `models/baseline_<split>/` (SVM & RF, fitur HSV+GLCM+Hu) |
| 8. Ekspor | `python inference.py --export` | `models/model.tflite` |
| 9. Galeri daun | `python scripts/make_plant_gallery.py` (butuh `data/tta_val_ckpt.npy` dari `scripts/eval_tta.py`) | `static/plants/` (halaman `/tanaman`) |
| 10. Web app | `python app.py` | http://127.0.0.1:5000 |

Jalankan langkah berat (train, baseline, preprocess) satu per satu, jangan paralel: RAM bisa habis dan proses crash.

## Arsitektur
Input 128x128x3 → 4 blok [Conv3x3 (32, 64, 128, 256) → BatchNorm → ReLU → MaxPool] → GlobalAvgPool → Dropout 0.4 → Dense 128 ReLU → Dense softmax (2).

## Hasil (split test v2, n=4.172, termasuk PlantDoc)
| Model | Akurasi | Macro F1 | PlantDoc | leaves | PlantVillage |
|---|---|---|---|---|---|
| **CNN R1 (final)** | **0,9643** | **0,9498** | **0,818** | **0,923** | 0,992 |
| SVM (HSV+GLCM+Hu) | 0,9528 | 0,9342 | 0,740 | 0,899 | 0,993 |
| Random Forest (HSV+GLCM+Hu) | 0,9501 | 0,9253 | 0,782 | 0,895 | – |
| CNN v1 (tanpa PlantDoc) | – | – | 0,730 | – | – |

Model final (R1): dilatih di Google Colab T4 via `notebooks/train_colab.ipynb` dengan
`python train.py --no-class-weight --patience 12` (Adam lr 1e-3, batch 32, epoch terbaik 15).
Ambang "tidak yakin" = 0,85 (dituning di val).

### Riwayat eksperimen (val v2, n=4.172)
| Run | Perubahan | Val acc | PlantDoc | leaves |
|---|---|---|---|---|
| v1 | data tanpa PlantDoc, class_weight | (test lama 0,9807) | (test 0,730) | – |
| v2 | + PlantDoc | 0,9348 | 0,795 | 0,803 |
| v2b | + BN momentum 0,9, patience 12 | 0,9132 | 0,670 | 0,788 |
| **R1** | tanpa class_weight, patience 12 | **0,9660** | **0,826** | **0,918** |
| R2 | R1 + batch 128 | 0,9355 | 0,823 | 0,917 |

### Model deploy: ensemble R1 + V3 (test v3, n=4.920, termasuk foto lapangan iNaturalist)
V3 = config R1 + data iNaturalist (foto lapangan berlisensi CC, atribusi di `archive (4)/inaturalist/attribution.csv`)
+ augmentasi ganti latar (`scripts/make_bgaug.py`). Ensemble = rata-rata probabilitas (bobot 0,5 dipilih di val).

| Model | Macro F1 | Akurasi | Sumber lama | PlantDoc | iNaturalist | 13 foto lapangan |
|---|---|---|---|---|---|---|
| R1 | 0,8592 | 0,8961 | 0,9636 | 0,823 | 0,520 | 9/13 |
| V3 | 0,8787 | 0,9067 | 0,9195 | 0,738 | 0,836 | 4/13 |
| Ensemble R1+V3 | 0,9030 | 0,9258 | 0,9609 | 0,797 | 0,730 | 9/13 |

### Model deploy v4: ensemble R1 + V3 + V4 (bobot 0,1/0,1/0,8; test v4, n=5.708)
V4 = + kerusakan hama & daun tropis umum dari iNaturalist. Angka lewat jalur web app (PIL + TFLite).

| Model | Macro F1 | Akurasi | Sumber lama | PlantDoc | iNaturalist | 13 foto lapangan |
|---|---|---|---|---|---|---|
| Ensemble R1+V3 | 0,8568 | 0,8888 | 0,9609 | 0,797 | 0,693 | 9/13 |
| V4 | 0,8933 | 0,9150 | 0,9528 | 0,740 | 0,813 | 10/13 |
| **Ensemble R1+V3+V4** | **0,8970** | **0,9180** | 0,9569 | 0,761 | 0,813 | 10/13 |

Ambang "tidak yakin" ensemble = 0,70 (dituning di val, berlaku juga untuk ensemble v4).

Temuan: `class_weight` membuat model condong ke `sehat` (kesalahan sakit→sehat 220 → 91 setelah dimatikan).
Keterbatasan: val_acc per epoch tetap fluktuatif; akurasi foto lapangan (PlantDoc) ±0,82, jauh di bawah foto studio (±0,99).

## Deploy (PythonAnywhere)
Runtime hanya butuh `requirements.txt` (Flask + interpreter TFLite), `models/model.tflite`, `model_v3.tflite`, `model_v4.tflite`, dan `static/plants/` (galeri, ±0,6 MB).
WSGI: `from app import app as application`.

## Tim
- Muhammad Daffi Izzuddin: web app, inferensi, deploy, repo
- Agam Gunata: data, preprocessing, tuning, baseline, pengujian aplikasi
- Azfa Hafshiam Dilaga: model CNN, training, evaluasi
