# PSD Audio Detect — Klasifikasi Suara Berisik dan Non-berisik

**Kelompok 1 — PSD-A**

Aplikasi mendeteksi kondisi suara sekitar melalui mikrofon browser dan mengklasifikasikannya menjadi **Berisik** atau **Non-berisik** menggunakan ekstraksi fitur Librosa dan Random Forest.

## Anggota kelompok

1. 24-100 Verdi Setyawan Ardiansyah Putra
2. 24-125 Triswanti Jannatul Ma'wa
3. 24-137 Deo Candra Saputra
4. 24-123 Alghifari Amar Mukhasyafah

## Metode dan fitur

Model menggunakan 16 fitur ringkasan (mean):

- RMS mean
- Spectral Centroid mean
- Zero Crossing Rate mean
- MFCC 1 sampai 13 mean

Ekstraksi fitur menggunakan audio mono dengan sample rate **22.050 Hz** agar konsisten dengan dataset fitur yang diberikan kelas. Model Random Forest menggunakan 200 pohon keputusan.

## Struktur proyek

```text
PSD_Audio_detect/
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requerements.txt
├── train_model.py
├── prepare_dataset.py
├── LAPORAN_PSD.md
├── DOCKER_GUIDE.md
├── README_DEPLOYMENT.md
├── assets/
│   └── jangan_berisik.mp3
├── data/
│   ├── dataset_fitur_suara_original.csv
│   ├── dataset_manifest.csv
│   ├── duplicate_audit.csv
│   └── features_model.csv
├── models/
│   ├── random_forest.joblib
│   └── model_metadata.json
├── reports/
│   ├── evaluation_results.json
│   ├── feature_importance.csv
│   ├── grouped_cv_folds.csv
│   └── oof_predictions.csv
└── src/
    ├── audio_features.py
    └── wave_visualizer.py
```

## Dataset

Dataset awal berasal dari dua arsip MP3 kelas:

- Berisik: 100 file
- Non-berisik: 100 file
- Total: 200 file

Audit SHA-256 menemukan 18 kelompok konten identik. Salinan identik dalam kelas yang sama dikurangi menjadi satu per konten; satu konten yang memiliki konflik label di dua kelas dikeluarkan sementara. Dataset pemodelan menjadi **176 sampel unik (88 berisik dan 88 non-berisik)**.

CSV fitur asli 200 baris disimpan sebagai `data/dataset_fitur_suara_original.csv`. CSV bersih yang dipakai untuk training adalah `data/features_model.csv`. Arsip MP3 mentah tidak dimasukkan ke repository agar ukuran repository tetap kecil; file fitur, manifest dan audit duplikasi disertakan.

## Menjalankan secara lokal dengan Python (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Buka `http://localhost:8501`. Izinkan akses mikrofon, lalu tekan **START** pada komponen WebRTC.

## Menjalankan dengan Docker Compose (cara pengumpulan)

Pastikan Docker Desktop sudah aktif. Jalankan perintah dari direktori root proyek:

```powershell
docker compose config
docker compose up --build -d
docker compose ps
docker compose logs -f
```

Buka aplikasi pada:

**http://localhost:8001**

Hentikan aplikasi:

```powershell
docker compose down
```

Kelompok 1 menggunakan host port **8001** yang dipetakan ke port container **8080**. Hanya satu port aplikasi yang dipublikasikan oleh Compose.

## Melatih ulang model

```powershell
python train_model.py
```

Script menggunakan `data/features_model.csv`, melakukan 5-fold StratifiedGroupKFold berdasarkan `source_student_id`, membuat evaluasi out-of-fold, lalu melatih model final pada seluruh 176 sampel bersih.

## Catatan deployment mikrofon

- `localhost` merupakan konteks aman untuk izin mikrofon saat uji lokal.
- Jika aplikasi diakses dari domain/server remote, gunakan HTTPS.
- WebRTC memakai STUN; pada jaringan tertentu TURN atau konfigurasi firewall tambahan mungkin diperlukan.
- Aplikasi memproses audio live untuk prediksi dan tidak menyimpan rekaman permanen ke disk.

## Referensi teknis

- Dockerfile: https://docs.docker.com/reference/dockerfile/
- Docker Compose: https://docs.docker.com/compose/
- Streamlit-WebRTC deployment: https://whitphx.github.io/streamlit-webrtc/deployment/

## Catatan hasil

Evaluasi awal menggunakan 5-fold StratifiedGroupKFold dengan grup berdasarkan ID sumber/mahasiswa. Akurasi out-of-fold adalah sekitar **75,57%** dan macro F1 sekitar **75,57%**. Angka ini mengukur kesesuaian model dengan label dataset, bukan jaminan akurasi universal. Label dataset belum seluruhnya diverifikasi secara manual.
