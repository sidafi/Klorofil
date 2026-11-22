# Tugas Besar AIT: Identifikasi Jenis dan Penyakit Tanaman dari Citra Daun (CNN from scratch)

Konteks proyek untuk Claude Code. Ringkasan dari diskusi perencanaan sebelumnya.

## Tim
- Muhammad Daffi Izzuddin: web app (Flask), inferensi, deploy PythonAnywhere, setup repo
- Agam Gunata: data, preprocessing, tuning, baseline fitur manual, pengujian aplikasi
- Azfa Hafshiam Dilaga: model CNN, training, evaluasi
- Setiap anggota wajib punya commit kode sendiri.

## Aturan dosen (wajib dipatuhi)
1. Model CNN, data berupa citra.
2. Teks dosen: "tidak boleh menggunakan model AI yang sudah ada, KECUALI untuk transfer learning". Keputusan tim: tetap CNN
   disusun sendiri + bobot acak (lebih ketat dari aturan; jadi nilai jual di jurnal). Dataset wajib citra RGB.
3. Aplikasi online di PythonAnywhere (akun gratis: model harus kecil).
4. Pengujian wajib menyajikan confusion matrix.
5. Keluaran lain: jurnal (submit OJS gratis), laporan, PPT, dokumentasi Trello (Scrum).

## Tahapan scope
- **Tahap 1 (sekarang):** klasifikasi biner kondisi daun: `sehat` / `sakit`. Satu keluaran.
- **Tahap 2 (pengembangan setelah tahap 1 jadi & deploy):** jenis tanaman dan/atau nama penyakit. Judul proyek mengacu ke tujuan akhir ini; laporan/jurnal tahap 1 menyebut cakupan biner secara eksplisit.

## Kelas (tahap 1)
| Kelas | Sumber folder |
|---|---|
| `sehat` | `healthy`, `*_healthy` (PlantVillage) |
| `sakit` | `diseased`, semua kelas penyakit PlantVillage, semua kelas rice |

Nama kelas `sehat`/`sakit` dipakai konsisten dari `data/raw/` sampai UI.

## Dataset (sekunder, Kaggle)
- https://www.kaggle.com/datasets/prasanshasatpathy/leaves-healthy-or-diseased
- https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases
- https://www.kaggle.com/datasets/emmarex/plantdisease?hl=id-ID&select=PlantVillage

Hasil cek folder hasil ekstrak (±25.261 citra unik):

| Folder lokal | Dataset | Sehat | Sakit | Resolusi |
|---|---|---|---|---|
| `archive (1)/PlantVillage/` | PlantVillage (15 kelas) | 3.221 | 17.418 | 256x256 |
| `archive (2)/leaves_healthy_or_diseased/` | Leaves healthy/diseased | 2.277 | 2.225 | 6000x4000 (±1,5 MB/file) |
| `archive/rice_leaf_diseases/` | Rice leaf (3 penyakit) | 0 | 120 | mayoritas 3081x897 |
| **Total** | | **5.498** | **19.763** | |

Catatan data:
- `archive (1)/PlantVillage/PlantVillage/` adalah salinan ganda; ABAIKAN (jangan ikut dihitung/di-split).
- Tidak seimbang (sehat:sakit ±1:3,6): pakai `class_weight` atau undersampling; metrik utama F1/recall per kelas, bukan akurasi saja.
- Risiko bias sumber (latar, resolusi beda; rice selalu sakit): resize semua ke ukuran sama, laporkan juga hasil evaluasi per sumber dataset.
- Ada beberapa file non-JPG (png/jpeg/tanpa ekstensi): validasi saat inspeksi.

## Keputusan
- Framework: **Keras** (TensorFlow). Model sehat/sakit: layer disusun sendiri, tanpa bobot pretrained. Model jenis (J2-J4):
  transfer learning MobileNetV2 ImageNet (diizinkan aturan dosen) -> WAJIB disebut jelas di laporan/jurnal.
- Deploy: PythonAnywhere gratis kuota disk 512 MB; `tensorflow` penuh berisiko tidak muat. Cek lebih awal: `tensorflow-cpu` atau konversi ke TFLite (`tflite-runtime`).
- Seed tetap (`SEED = 42`) untuk split dan training agar hasil bisa direproduksi.
- Resize: center-crop persegi lalu 128x128 (bukan padding: bar hitam pada rice jadi jalan pintas sumber=label).
- Deploy pakai `models/model.tflite` (ekspor: `python inference.py --export`); runtime hanya `requirements.txt`.
- Jangan jalankan proses berat (train, baseline, preprocess) bersamaan; training lokal CPU sempat segfault acak (lihat Status).

## Status (2026-11-15)
- Selesai & teruji: inspect_data, build_raw, preprocess, model, evaluate, baseline, inference (TFLite), app (Flask).
- Baseline test: SVM acc 0,9757 / macro F1 0,9648; RF acc 0,9686 / macro F1 0,9528.
- Training lokal crash (Windows access violation di `model.fit`, acak) -> training dipindah ke Google Colab T4 (`notebooks/train_colab.ipynb`, data via `scripts/make_colab_bundle.py`).
- CNN final (run 1, lr 1e-3, epoch terbaik 14): test acc 0,9807 / macro F1 0,9725 (leaves 0,9142, PlantVillage 0,9955) -> mengungguli SVM. Run lr 3e-4 lebih buruk (val acc 0,9464).
- Ambang "tidak yakin" 0,90 (tuning di val). `models/best.keras` & `models/model.tflite` = model final.
- Catatan: val_acc per epoch sangat fluktuatif & model condong ke `sehat` (kesalahan dominan sakit->sehat di leaves); kandidat perbaikan: tanpa/kurangi class_weight, momentum BatchNorm.
- Uji foto lapangan: v1 gagal (foto kebun terbalik). Ditambah PlantDoc (sumber `plantdoc`, split lengket). v1 di test PlantDoc 0,730.
- v2 (+PlantDoc): val 0,935 / PlantDoc 0,795 / leaves 0,803. v2b (BN 0.9): val 0,913 (lebih buruk). BN & augmentasi sudah diuji, bukan penyebab fluktuasi val.
- SESI TRAINING TERAKHIR tahap 1 = ablasi R1 (tanpa class_weight) & R2 (tanpa class_weight + batch 128). Aturan fiksasi (ditetapkan sebelum hasil):
  1. Kandidat v1, v2, R1, R2.  2. R1/R2 val >= 0,956 & PlantDoc >= 0,78 -> fiks.  3. Jika tidak: macro F1 val tertinggi (seri: PlantDoc) -> fiks + tulis keterbatasan.
  4. Lalu TEST sekali, ekspor, eval_field, deploy. TIDAK ada training ulang tahap 1.
- HASIL: R1 lolos aturan 2 (val 0,9660, PlantDoc 0,826) -> DIFIKSASI. R2 val 0,9355. class_weight terbukti penyebab bias ke `sehat`.
- R1 test: acc 0,9643 / macro F1 0,9498 (PlantDoc 0,818, leaves 0,923, PlantVillage 0,992) > SVM 0,9528 / RF 0,9501. Ambang "tidak yakin" 0,85.
- `models/` = R1 final; `models_v1/` = model lama; laporan ablasi di `ablation_reports/`. Python proyek: pakai path Python39 eksplisit (virtualenv lain bisa aktif di shell).
- TAHAP 1 DIFIKSASI (R1). Berikutnya: foto lapangan tim -> `data/field_test/{sehat,sakit}/` + `scripts/eval_field.py`, lalu deploy PythonAnywhere.
- Uji 13 foto web lapangan (`data/field_test/`, HANYA test): R1 benar 9/13 (0,692), 9 ditandai tidak yakin. Model belum andal di foto kebun (bias latar polos).

## Iterasi v3 (disepakati 2026-11-16)
- Urutan: deploy R1 dulu (keterbatasan ditulis jujur), lalu v3.
- Data v3: TANPA unduhan baru. Augmentasi ganti latar (`scripts/make_bgaug.py`): daun PlantVillage/leaves (train saja) disegmentasi lalu ditempel ke latar kebun acak. Data & proses lama tetap.
- Definisi label: lubang ALAMI (mis. Monstera) = `sehat`; lubang/kerusakan yang tidak seharusnya (termasuk hama) = `sakit`.
  Catatan: tidak ada data Monstera/daun berlubang alami di dataset -> kasus ini belum bisa diajarkan di v3.
- Data v3 ditambah iNaturalist (`scripts/fetch_inat.py`, sumber `inat`, CC0/CC-BY/CC-BY-NC, atribusi di
  `archive (4)/inaturalist/attribution.csv`): sehat 2.386 (Monstera dkk., lubang alami), sakit 2.600 (karat, embun tepung,
  Septoria, Cercospora, Rhytisma, Diplocarpon, penggorok daun). Venturia dibuang (mayoritas foto buah). Split lengket 70/15/15.
- Patokan R1 di val baru (n=4.920): acc 0,9018 / macro F1 0,8678; inat 0,544; sumber lama 0,966; PlantDoc 0,834.
- Aturan stop v3 (DIKUNCI ULANG sebelum training karena val berubah; SATU run, config R1 + data inat + bgaug):
  v3 dipakai jika macro F1 val > 0,8678 DAN acc sumber lama >= 0,956 DAN PlantDoc >= 0,826 DAN foto lapangan >= 9/13.
  Jika tidak: tetap R1. Selesai, tidak ada run lain.
- HASIL v3 sendiri: GAGAL aturan (val macro F1 0,8908 ok, tapi sumber lama 0,9255 & PlantDoc 0,813). inat 0,860.
- ENSEMBLE R1+V3 (tanpa training; rata-rata prob, bobot dipilih di val via `scripts/eval_ensemble.py`):
  w=0,5 satu-satunya bobot terbaik yang lolos SEMUA kriteria (val: macro F1 0,9140, sumber lama 0,9645, PlantDoc 0,831, lapangan 9/13).
  TEST sekali: ensemble macro F1 0,9030 / acc 0,9258 (R1 0,8592 / V3 0,8787); sumber lama 0,9609; PlantDoc 0,797; inat 0,730.
- MODEL DEPLOY = ensemble `models/model.tflite` (R1) + `models/model_v3.tflite` (V3), `inference.Ensemble`, ambang tidak yakin 0,70.
- Uji 13 foto lapangan (ensemble, dicek langsung dari model): benar 9/13; 8/13 "tidak yakin"; 1 salah-yakin (web_10: 1 lubang kecil
  latar putih -> sehat 0,93). Miss lain (ragu): web_4 jati di kebun, web_14 kacang dimakan hama, web_18 Monstera indoor.
- Keterbatasan / data yang dibutuhkan iterasi berikut: daun biasa SEHAT di lapangan (data sehat lapangan kini hampir semua
  Monstera dkk.), daun bekas HAMA (gigitan/lubang) di lapangan & studio, foto tim sendiri. R1 & V3 sering bertentangan ->
  confidence ensemble rendah (~0,5-0,6) di foto lapangan. Val per epoch fluktuatif.
- Web app: tombol kamera (getUserMedia, butuh HTTPS/localhost), kotak panduan = area crop model.

## Iterasi v4 (disepakati 2026-11-16)
- Link dataset usulan user ditolak (dicek): Mendeley hb74ynkjcn = sumber `leaves` (duplikat); Kaggle hmohamedhussain = DB sama,
  tanpa label sehat/sakit; datasetninja PlantDoc = sudah dipakai; Mendeley 5g238dv4ht = latar dihapus + tanpa label sehat/sakit.
- Data v4 (iNaturalist, mengisi celah uji lapangan): sakit = kerusakan hama di lapangan (proyek "Unidentifiable Plant Damage",
  "Feeding Damage on Plants"); sehat = daun biasa tanaman tropis umum di lapangan (jati, mangga, jambu, pepaya, pisang, singkong, dll.;
  label diasumsikan sehat -> ada noise). Foto tim tetap HANYA test.
- Aturan stop v4 (dikunci SEBELUM training; SATU run, config R1 = tanpa class_weight, patience 12):
  patokan = ensemble R1+V3 saat ini, diukur di val v4 (dicatat sebelum training).
  Kandidat: V4 sendiri, atau ensemble dengan model lama (bobot dipilih di val). Dipakai jika
  macro F1 val > patokan DAN acc sumber lama >= patokan - 0,01 DAN foto lapangan >= 10/13. Jika tidak: tetap ensemble R1+V3.
- Data v4 terpasang: + kerusakan hama 3.504 (sakit) + jati/mangga/jambu/pepaya/singkong/ubi 1.747 (sehat). Dibuang (mayoritas
  buah/bunga/serangga): nangka, kakao, sirsak, kopi, pisang, jeruk. Total 38.049 citra (val/test 5.708) + bgaug 6.690.
- PATOKAN (ensemble R1+V3 w=0,5 di val v4, dicatat SEBELUM training): macro F1 0,8647; sumber lama 0,9645; PlantDoc 0,8312;
  inat 0,7064; lapangan 9/13.  => V4 lolos jika macro F1 > 0,8647 DAN sumber lama >= 0,9545 DAN lapangan >= 10/13.
- HASIL v4 (Colab, epoch terbaik 10/22; dievaluasi lokal via TFLite+PIL = jalur web app): V4 sendiri val F1 0,9048 / lama 0,9593 / lapangan 10/13.
  Ensemble R1/V3/V4 = 0,1/0,1/0,8 = F1 val tertinggi yang lolos SEMUA (F1 0,9087, lama 0,9621, PlantDoc 0,813, inat 0,832, lapangan 10/13) -> DIPAKAI.
  TEST sekali: ensemble F1 0,8970 / acc 0,9180 / lama 0,9569 / PlantDoc 0,761 / inat 0,813 (patokan 0,8568 / 0,8888; V4 0,8933).
- MODEL DEPLOY = `models/model.tflite` (R1) + `model_v3.tflite` + `model_v4.tflite`, bobot 0,1/0,1/0,8; ambang tidak yakin 0,70.
- Catatan: TFLite V4 dikonversi lokal dari best.keras (identik: selisih 0 dgn decode TF). Decode JPEG PIL (app) vs TF (Colab) beda kecil
  (label beda ~0,4%) -> angka resmi dihitung lewat jalur app. Lapangan masih miss: web_4 jati (sakit 0,83), web_10 1 lubang, web_16.
  PlantDoc turun (test 0,797 -> 0,761). Notebook Colab: path report sel 5 sempat salah ketik (eval_val_report.txt).

## Audit & TTA (2026-11-17)
- Audit (occlusion map, uji lubang buatan, komposisi data): model menilai kesan global (hijau rimbun = sehat); lesi/lubang kecil
  tenggelam setelah resize 128 px + GlobalAvgPool. Hipotesis "lubang = sehat" GUGUR (lubang buatan menurunkan p_sehat).
  Celah data: padi train hanya 84 (semua sakit, studio). Watermark ikut berpengaruh kecil (jati).
- `data/field_test/` diperluas jadi 24 foto (13 sehat / 11 sakit; ilustrasi & duplikat pepaya dikecualikan). Patokan ensemble: 18/24, salah-yakin 3.
- Aturan TTA (dikunci SEBELUM diukur; TANPA training): agregasi dipilih di val (input dari citra asli `data/raw`, jalur app).
  TTA dipakai jika lapangan benar > 18/24 DAN salah-yakin <= 3 DAN macro F1 val >= 0,9087 - 0,01 DAN sumber lama val >= 0,9545.
  Jika tidak: tetap tanpa TTA.
- HASIL TTA (`scripts/eval_tta.py`, val dari citra asli): A tanpa TTA F1 0,9106 / lama 0,9676 / lapangan 18/24 / salah-yakin 3;
  B rata-rata 6 potongan 0,8586 / 0,9163 / 19/24 / 0; C min 0,7760 / 0,8785 / 17/24 / 2; D .5 penuh+.5 zoom 0,9091 / 0,9657 / 18/24 / 1.
  -> TIDAK ADA yang lolos aturan (B & C gagal di val; D gagal "lapangan > 18/24"). KEPUTUSAN: TANPA TTA (app tidak diubah).
  Catatan jurnal: D menurunkan salah-yakin 3->1 tanpa rugi val, tapi tidak menambah jawaban benar; zoom-crop saja tidak cukup
  untuk lesi kecil -> perbaikan nyata butuh resolusi input lebih besar / data (v5).

## Alur kerja
1. `scripts/inspect_data.py`: hitung citra per kelas, cek file rusak, duplikat (hash, termasuk lintas dataset), resolusi.
2. Susun `data/raw/{sehat,sakit}/` dari ketiga dataset (label = nama folder, tanpa bounding box); simpan sumber asal tiap file (mis. CSV) untuk evaluasi per sumber.
3. `preprocess.py`: dedup → split 70/15/15 stratifikasi (seed tetap) SEBELUM augmentasi → resize 128x128 sekali lalu simpan ke `data/processed/` → normalisasi 1/255 → augmentasi hanya pada train.
4. `model.py`: CNN dari nol.
5. `train.py`: Adam lr 1e-3, batch 32, `class_weight`, early stopping, simpan model terbaik.
6. `evaluate.py`: confusion matrix, precision, recall, F1 per kelas (keseluruhan + per sumber dataset).
7. `baseline.py`: fitur manual (histogram HSV, GLCM, Hu moments) + SVM/Random Forest sebagai pembanding di jurnal.
8. `inference.py` + `app.py` (Flask): upload foto, tampilkan kelas + confidence, ambang "tidak yakin".
9. Deploy PythonAnywhere, uji end-to-end.

## Draf arsitektur (v1)
Input 128x128x3 → 4 blok [Conv3x3 (32, 64, 128, 256) → BatchNorm → ReLU → MaxPool] → GlobalAvgPool → Dropout 0.4 → Dense 128 ReLU → Dense softmax (2 kelas).
Estimasi ±430 ribu parameter (target < 1 juta).

## Struktur folder
```
data/raw/  data/processed/  models/  notebooks/  scripts/
preprocess.py  model.py  train.py  evaluate.py  baseline.py  inference.py  app.py
templates/  static/  requirements.txt  README.md
```
Dataset mentah (`archive*/`, `*.zip`) tidak di-commit ke repo.

## Gaya kerja
- Jawaban langsung ke inti, Bahasa Indonesia, kode efisien.
- Trello: https://trello.com/b/kBK3bc2p (Scrum, 4 sprint).

## Kalibrasi ambang keputusan (2026-11-17, TANPA training)
- `data/field_test/` + 9 foto web3_* (8 sehat, 1 sakit: 68 daun dimakan hama) = 33 foto (21 sehat / 12 sakit), HANYA test.
  Patokan ensemble di 33 foto: benar 21/33, salah-yakin 4 (web3_* diuji sebelum aturan ini: 3/9).
- Aturan (dikunci SEBELUM diukur di lapangan): sehat jika p_sehat >= t. t dipilih HANYA di val (`data/tta_val_ckpt.npy`, kolom
  citra penuh), grid 0,30-0,70 langkah 0,01: maksimalkan macro F1 subset inat (foto lapangan) dengan syarat macro F1 val total
  >= 0,8987 DAN sumber lama >= 0,9545. Dipakai jika lapangan 33 > 21/33 DAN salah-yakin <= 4. Jika tidak: t = 0,5 (tetap).
- HASIL (`scripts/eval_calib.py`): t terpilih di val = 0,46 (F1 val 0,9138, lama 0,9676, F1 inat 0,8256 vs 0,8173 di t=0,5),
  tapi lapangan tetap 21/33, salah-yakin 4 -> GAGAL. KEPUTUSAN: t = 0,5 (app tidak diubah). Tanpa training, tidak ada tuas sah
  yang membetulkan web3_*: probabilitasnya ~0,5 (spesies tak dikenal) atau salah-yakin (68 lubang hama -> sehat 0,71). Butuh v5.

## Halaman "Daun dikenali" (2026-11-17)
- `/tanaman` (`templates/tanaman.html`, data `static/plants/plants.json` + 31 foto, dibuat `scripts/make_plant_gallery.py`).
  Per tanaman: jumlah data sehat/sakit, akurasi ensemble di VAL (t=0,5), status andal (>=0,90) / cukup (>=0,75) / terbatas
  (hanya 1 label, n<30, atau <0,75). Hasil: 10 andal, 4 cukup, 17 terbatas. Jagung hanya punya data SAKIT (penjelas foto jagung sehat -> sakit).
- Foto contoh: data train lokal; tanaman yang hanya dari PlantDoc (hak cipta scraping tak jelas) + ubi & monstera diganti foto
  iNaturalist CC0/CC-BY (dipilih manual, URL & atribusi di skrip). Pemetaan kode dataset `leaves` -> spesies: Chouhan dkk. 2019.

## Tahap 2: penyakit per tanaman (disepakati 2026-11-18, pilihan 1)
- Cakupan: tomat, kentang, paprika, padi (data nama penyakit kuat). 18 kelas "tanaman_penyakit" (termasuk *_sehat; padi tanpa sehat).
  Sumber: PlantVillage + rice + PlantDoc yang namanya cocok (`penyakit.py`, data/penyakit/ = hardlink dari data/processed).
  Di luar cakupan: leaves (tanpa nama penyakit), iNat (tanpa jenis tanaman), PlantDoc tanaman lain (data terlalu sedikit).
- Split IKUT split.csv tahap 1 (file test tahap 1 tetap test) -> tidak ada kebocoran antar tahap.
- Model: arsitektur leafnet sama (num_classes=18), dari nol, input 128. App: tahap 1 (sehat/sakit) tetap; model tahap 2 tampil
  sebagai "dugaan tanaman & penyakit" hanya bila keyakinan >= ambang (dituning di val).
- Aturan stop (dikunci SEBELUM training; SATU run di Colab, config R1 = `--no-class-weight --patience 12`):
  dipakai di app jika macro F1 val >= 0,85. Jika tidak: tidak ditampilkan di app, tulis keterbatasan. TEST sekali setelah lolos.
  Catatan risiko: padi hanya ±28 citra train per kelas (studio) -> F1 padi dilaporkan terpisah.
- HASIL P1 (Colab, epoch terbaik 16/28): val acc 0,8738 / macro F1 0,8311 -> GAGAL aturan (< 0,85). Model tahap 2 TIDAK dipasang
  di app; TEST tidak dijalankan. Kelemahan: padi (6 citra val/kelas; F1 gosong 0,36, bercak coklat 0,67), tungau->tomat_sehat
  (71/249) & bercak target->tomat_sehat (29/203) -> presisi tomat_sehat 0,64; val loss sangat fluktuatif (seperti tahap 1).
  Kuat: keriting kuning 0,96, kentang sehat 0,94, paprika 0,93. Hasil di Drive MyDrive/models_P1.
- P2 (disepakati 2026-11-18, dikunci SEBELUM training; run TERAKHIR tahap 2): padi DIKELUARKAN dari cakupan (data tidak cukup:
  ±28 train/kelas) -> 15 kelas tomat/kentang/paprika; lr 5e-4 (`--lr 0.0005 --patience 12`, tanpa class_weight). Dipakai di app
  jika macro F1 val >= 0,85. Jika gagal: tahap 2 berhenti (tulis di jurnal sebagai keterbatasan), proyek lanjut tanpa tahap 2.
- HASIL P2: val acc 0,8740 / macro F1 0,8637 -> LOLOS (>= 0,85). Kelemahan tetap: tungau->tomat_sehat 75/249, bercak target->
  tomat_sehat 36/203 (presisi tomat_sehat 0,64); mosaik F1 0,74, hawar awal tomat 0,79. Berikutnya: TEST sekali, ekspor, ambang di val, app.
- TEST P2 (sekali): acc 0,8828 / macro F1 0,8764; PlantVillage acc 0,917 (macro F1 0,914), PlantDoc (foto lapangan) acc 0,276
  (n=174) -> model penyakit praktis hanya andal di foto studio/latar polos. Jalur app (PIL+TFLite) di val: acc 0,8734 (= Colab).
- DEPLOY tahap 2: `models/model_penyakit.tflite` (hasil di `models_penyakit/`), `inference.PENYAKIT_THRESHOLD` = 0,80 (ambang terkecil
  dgn akurasi tampil >= 0,95 di val: 79,6% tampil, akurasi 0,966; PlantDoc hanya ±5% lolos). App: blok "Dugaan tanaman & penyakit"
  (top-3) di bawah hasil sehat/sakit; di bawah ambang -> "tidak dikenali". Tahap 2 SELESAI (tidak ada run lain).

## Tahap 2b: JENIS tanaman (disepakati 2026-11-18; uji 51 foto user: penyakit P2 tampil 3/51 & ketiganya salah -> fokus jenis)
- Target demo: daun yang wajar ditemui di Indonesia (pinggir jalan/kebun). 22 kelas: tomat, kentang, paprika_cabai, padi, jagung,
  mangga, jambu_biji, pepaya, singkong, ubi_jalar, jati, monstera (Monstera dkk.), sirih, jarak_kepyar, jarak_pagar, mint, bayam,
  kangkung, apel, anggur, mawar, lainnya (tumbuhan lain: iNat Indonesia + spesies leaves/PlantDoc di luar daftar).
- Data: data lama berlabel jenis (PlantVillage, PlantDoc, rice, leaves, iNat jenis) + unduhan baru `fetch_inat.py --set jenis`
  (archive (5)/inat_jenis, obs duplikat dgn archive (4) dilewati). iNat kelompok penyakit/hama (tanpa jenis) TIDAK dipakai.
  Split: citra lama ikut split tahap 1; citra baru 70/15/15 stratifikasi per kelas (SEED 42). Batas per kelas di train agar seimbang.
- Folder `dataset daun collective/`: 330 foto tim (±65 daun x 5 jepretan, meja putih), label sehat/sakit ada, jenis kosong ->
  HANYA uji/demo. Berisi `transfer_learning.py` (ResNet18 pretrained; diizinkan sbg transfer learning, tapi bukan
  bagian pipeline tim -> tidak dipakai/di-commit).
- App: model jenis tampil dulu; dugaan penyakit (P2) hanya ditampilkan bila jenis = tomat/kentang/paprika_cabai dgn yakin.
- Aturan stop (dikunci SEBELUM training; SATU run, config R1 `--no-class-weight --patience 12`): dipakai jika macro F1 val >= 0,80.
  Jika gagal: app tanpa model jenis (penyakit juga disembunyikan), tulis keterbatasan.
- HASIL J1 (Colab): val acc 0,5274 / macro F1 0,4029 -> GAGAL (< 0,80). Model jenis TIDAK dipakai; dugaan penyakit juga
  disembunyikan dari app (`app.SHOW_DISEASE = False`; kode & model_penyakit.tflite tetap ada untuk jurnal). TEST J1 tidak dijalankan.
  Analisis: model belajar latar/sumber, bukan bentuk daun -> "monstera" (satu-satunya kelas 1.400 foto kebun) jadi kelas buangan
  foto lapangan (sirih 31, singkong 25, pepaya 22, lainnya 61); tomat/paprika/kentang (dominan studio PlantVillage) saling tertukar;
  kelas ±280 foto iNat hampir gagal (jarak kepyar F1 0,00, mint 0,14, pepaya 0,14). Kuat: jambu 0,78, paprika 0,78, padi 0,70.
  Tahap 2 SELESAI sebagai eksperimen (P2 lolos di foto studio; J1 gagal) -> keterbatasan + saran (data lapangan seimbang per jenis,
  resolusi > 128, atau transfer learning yang diizinkan dosen) ditulis di jurnal. App = tahap 1 saja.
- J2 TRANSFER LEARNING (disetujui user 2026-11-18; dosen membolehkan transfer learning): HANYA model jenis. Model sehat/sakit
  tetap CNN dari nol (J1 vs J2 = perbandingan dari-nol vs transfer di jurnal). Backbone MobileNetV2 (alpha 1.0, bobot ImageNet,
  input 128 = data/jenis apa adanya; Rescaling 0-1 -> [-1,1] di dalam model, jadi inference.py tidak berubah).
  Training 2 fase: kepala (base beku, lr 1e-3) lalu fine-tune lapisan atas base (lr 1e-5, BN tetap beku). Data & split = J1.
  Aturan (dikunci SEBELUM training; SATU run): J2 dipakai jika macro F1 val >= 0,80. Jika lolos: app tampilkan jenis
  (ambang dituning di val) + dugaan penyakit HANYA bila jenis = tomat/kentang/paprika_cabai yakin. Jika gagal: app tetap tahap 1.
- HASIL J2 (Colab, 15 epoch kepala + 20 fine-tune): val acc 0,6907 / macro F1 0,6195 -> GAGAL (< 0,80). App tetap tahap 1.
  J1 -> J2: macro F1 0,403 -> 0,620 (+0,22): transfer learning jelas membantu (temuan jurnal). Kuat: paprika 0,88, padi 0,83,
  jagung/jambu/mangga 0,81, monstera 0,79, kentang 0,78. Lemah: kelas ±280 foto (mint/kangkung 0,43, ubi 0,44, pepaya 0,46).
  Val loss masih turun pelan di epoch 35 (belum konvergen) -> keterbatasan, bukan alasan run ulang. Test J2 tidak dijalankan.
- JUDUL (user 2026-11-19: tanpa fitur nama penyakit; fokus kesehatan DAUN): karena J2 gagal, judul disarankan
  "Klasifikasi Status Kesehatan Daun (Sehat/Sakit) Berbasis Convolutional Neural Network". Tahap 2 (P2, J1, J2) = eksperimen di jurnal.
- J3 (disepakati user 2026-11-19, dikunci SEBELUM training; SATU run): 9 kelas = 8 jenis terkuat di val J2 (paprika_cabai, padi,
  jagung, jambu_biji, mangga, monstera, kentang, tomat) + lainnya (13 jenis lain digabung ke "lainnya"). Data/split sama dgn J2
  (file test tetap test), CAP per kelas sama (1.400/300/300). Config = J2 (MobileNetV2, 15 epoch kepala + 20 fine-tune, patience 6).
  Karena kelas dipilih dari skor val J2 (bias seleksi), syarat dipakai: macro F1 val >= 0,80 DAN macro F1 TEST >= 0,80
  (test dijalankan sekali, hanya bila val lolos). Lolos -> app tampilkan jenis (ambang dituning di val). Gagal -> app tetap tahap 1.
- HASIL J3 VAL: acc 0,8019 / macro F1 0,8132 -> lolos syarat val (>= 0,80). Kuat: paprika 0,90, padi 0,87, jambu 0,85, monstera 0,85,
  kentang 0,81, mangga 0,80. Lemah: lainnya 0,66 (presisi 0,59: foto kebun kelas lain -> lainnya), tomat 0,76 (recall 0,68).
  Berikutnya: TEST sekali (syarat kedua: macro F1 test >= 0,80), ekspor models/model_jenis.tflite, ambang tampil di val.
- TEST J3 (sekali): acc 0,7963 / macro F1 0,8044 -> LOLOS (val 0,8132 & test >= 0,80). Per sumber: PlantVillage 0,982, leaves 0,985,
  rice 0,944, inat 0,763, PlantDoc 0,667, inatj 0,645 (foto lapangan jauh lebih sulit). Jalur app (PIL+TFLite) val acc 0,7882.
- DEPLOY J3: `models/model_jenis.tflite` (hasil di `models_jenis_j3/`), `inference.JENIS_THRESHOLD` = 0,85 (ambang terkecil dgn
  akurasi tampil >= 0,95 di val jalur app: 59,6% yakin, akurasi 0,954). App: blok "Jenis tanaman" (top-3) di hasil deteksi;
  di bawah ambang -> "Kemungkinan X"; label lainnya -> "bukan salah satu dari 8 jenis". Dugaan nama penyakit TETAP nonaktif
  (keputusan user: tanpa fitur nama penyakit). Judul -> "Identifikasi Jenis Tanaman dan Status Kesehatan Daun Berbasis CNN".
- J4 (disepakati user 2026-11-20, dikunci SEBELUM training; SATU run). Masalah J3 (uji 12 foto user: 6/12): model memakai LATAR
  (latar putih stok -> paprika/kentang; ganti latar abu mengubah cabai kentang 0,62 -> paprika 0,98). J4 = J3 + augmentasi ganti
  latar khusus TRAIN data jenis: daun dari foto studio (PlantVillage, leaves) disegmentasi lalu ditempel ke latar acak
  (putih, hitam, abu, warna polos, foto kebun). Kelas, split, val/test, config sama dgn J3. Segmentasi TIDAK dipakai di app.
  Dipakai (menggantikan J3) jika macro F1 val >= 0,8132 (J3) DAN macro F1 test >= 0,80 DAN 12 foto user (uji J3 6/12) > 6/12.
  Jika tidak: tetap J3. Foto user 12 = HANYA uji (data/field_jenis/), tidak dipakai melatih/memilih.
- HASIL J4 VAL: acc 0,8051 / macro F1 0,8155 -> lolos syarat 1 (>= 0,8132, selisih tipis +0,002). Paprika 0,92, jagung 0,83,
  kentang 0,83, tomat 0,77, lainnya 0,64 (vs J3 0,66). Berikutnya: TEST sekali + uji 12 foto user (data/field_jenis).
- TEST J4: acc 0,7957 / macro F1 0,7974 -> GAGAL syarat 2 (< 0,80). 12 foto user: 6/12 (= J3) -> GAGAL syarat 3. APP TETAP J3.
  Efek augmentasi latar: cabai latar putih kentang 0,65 -> paprika 0,99, padi 140 naik; tapi jambu 134 -> mangga, mangga latar
  putih tetap paprika, tomat tetap lainnya/kentang. Temuan jurnal: ganti latar memperbaiki kasus yg jelas bergantung latar, tapi
  tidak cukup mengatasi data lapangan yang sedikit per jenis. Hasil J4 di `models_jenis_j4/` (tidak dipakai). Tahap jenis SELESAI.
