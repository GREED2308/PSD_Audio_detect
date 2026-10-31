# LAPORAN PROYEK PSD
## Sistem Klasifikasi Suara Berisik dan Non-berisik Berbasis Random Forest

### Identitas Kelompok
**Kelompok 1 — PSD-A**

- 24-100 Verdi Setyawan Ardiansyah Putra
- 24-125 Triswanti Jannatul Ma'wa
- 24-137 Deo Candra Saputra
- 24-123 Alghifari Amar Mukhasyafah

---

## 1. Business Understanding

### Latar belakang
Kondisi suara di sekitar dapat diamati dari karakteristik sinyal audio seperti energi, distribusi frekuensi, dan pola spektral. Pemeriksaan secara manual terhadap banyak rekaman membutuhkan waktu. Proyek ini mengembangkan prototipe yang mengklasifikasikan audio menjadi dua kelas, yaitu berisik dan non-berisik, dengan memanfaatkan fitur akustik dan algoritma Random Forest.

### Rumusan masalah
Bagaimana membangun sistem yang dapat mengklasifikasikan suara sekitar sebagai berisik atau non-berisik berdasarkan fitur akustik dari sinyal audio?

### Tujuan
1. Memahami dan memeriksa dataset audio dua kelas.
2. Menggunakan fitur RMS, Spectral Centroid, ZCR, dan MFCC 1–13.
3. Melatih dan mengevaluasi Random Forest.
4. Mengintegrasikan model dengan mikrofon browser agar prediksi dapat dilakukan secara langsung.
5. Mengemas aplikasi menggunakan Dockerfile dan Docker Compose.

### Batasan
Label berisik/non-berisik mengikuti label dataset yang disediakan kelas. Sistem ini merupakan klasifikasi prototipe, bukan sound level meter terkalibrasi dalam satuan dB. Label sumber belum diverifikasi secara manual satu per satu.

## 2. Data Understanding

Dataset terdiri atas dua arsip audio MP3:

| Kelas | Jumlah |
|---|---:|
| Berisik | 100 |
| Non-berisik | 100 |
| **Total** | **200** |

CSV fitur `dataset_fitur_suara.csv` berisi 200 baris, 16 fitur numerik, nama file dan label. Distribusi label seimbang (100:100), tidak ditemukan nilai kosong, dan tidak ada baris CSV yang identik sepenuhnya.

Pencocokan nama file CSV dengan kedua arsip MP3 menunjukkan seluruh 200 entri mempunyai pasangan nama file di dalam ZIP.

### Audit duplikasi
Pemeriksaan SHA-256 pada berkas MP3 menemukan:
- 18 kelompok hash yang memiliki konten identik.
- 17 kelompok merupakan duplikasi dalam kelas yang sama, dengan 22 salinan berlebih.
- 1 kelompok memiliki konten identik tetapi berada pada dua kelas berbeda, yaitu `24-034 non-berisik 1.mp3` dan `24-034 berisik 1.mp3`.

Untuk mencegah duplikasi dan konflik label memengaruhi model, satu perwakilan dipertahankan dari setiap kelompok duplikasi dalam kelas yang sama, sedangkan kedua entri konflik lintas kelas dikeluarkan sementara. Dataset bersih menjadi 176 sampel: 88 berisik dan 88 non-berisik.

## 3. Data Preparation

CSV fitur bersih terdiri dari 16 fitur mean. Fitur yang digunakan adalah:

| Kelompok | Fitur |
|---|---|
| RMS | RMS mean |
| Spectral Centroid | Centroid mean |
| Zero Crossing Rate | ZCR mean |
| MFCC | MFCC 1 mean sampai MFCC 13 mean |
| **Total** | **16 fitur** |

Ekstraksi live disamakan dengan kolom CSV menggunakan Librosa, audio mono, dan sample rate 22.050 Hz. Urutan fitur di aplikasi mengikuti urutan yang digunakan ketika training.

Untuk menghindari kebocoran data, pembagian validasi dilakukan secara berkelompok menggunakan ID sumber/mahasiswa yang diambil dari nama file. Sampel dari satu ID sumber tidak diletakkan pada fold training dan validasi yang sama.

## 4. Modeling

Algoritma yang digunakan adalah Random Forest Classifier.

Parameter:
- `n_estimators = 200`
- `max_depth = None`
- `max_features = sqrt`
- `min_samples_leaf = 4`
- `class_weight = balanced`
- `random_state = 42`

Model final dilatih ulang menggunakan semua 176 baris dataset bersih. Artefak model disimpan sebagai `models/random_forest.joblib`.

## 5. Evaluation

Evaluasi menggunakan 5-fold StratifiedGroupKFold dengan prediksi out-of-fold. Total 176 sampel mendapatkan prediksi dari model yang tidak dilatih menggunakan grup sumber sampel tersebut.

### Hasil metrik

| Metrik | Hasil |
|---|---:|
| Accuracy | 75,57% |
| Balanced accuracy | 75,57% |
| Macro precision | 75,57% |
| Macro recall | 75,57% |
| Macro F1-score | 75,57% |

### Classification report

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Non-berisik | 75,28% | 76,14% | 75,71% | 88 |
| Berisik | 75,86% | 75,00% | 75,43% | 88 |

### Confusion matrix
Urutan label aktual dan prediksi: Non-berisik, Berisik.

| Aktual \ Prediksi | Non-berisik | Berisik |
|---|---:|---:|
| Non-berisik | 67 | 21 |
| Berisik | 22 | 66 |

Hasil menunjukkan model dapat mempelajari pola yang membedakan kedua kelas, tetapi masih terdapat salah klasifikasi. Performa setiap fold juga bervariasi sehingga hasil ini perlu ditafsirkan sebagai estimasi awal, bukan jaminan performa untuk semua lingkungan.

## 6. Deployment

Aplikasi Streamlit menerima audio langsung dari mikrofon browser menggunakan `streamlit-webrtc`. Audio diproses sebagai mono 22.050 Hz, diekstraksi menjadi 16 fitur mean, lalu diberikan kepada model Random Forest.

Fitur aplikasi:
1. Start/stop mikrofon.
2. Visualisasi waveform dan equalizer dari sinyal aktual.
3. Prediksi Berisik atau Non-berisik secara berkala.
4. Tampilan skor model.
5. Audio peringatan ketika prediksi berubah menjadi Berisik.
6. Audio live tidak disimpan permanen oleh aplikasi.

### Docker
- `Dockerfile` berada di root dan membangun image aplikasi.
- `docker-compose.yml` berada di root dan menjalankan image.
- `requirements.txt` berada di root sebagai daftar dependensi Python.
- `requerements.txt` juga disediakan sebagai salinan untuk mengikuti ejaan pada instruksi tugas.
- Kelompok 1 menggunakan port host `8001` ke port container `8080`.

Perintah:
```bash
docker compose up --build -d
```
Aplikasi dapat dibuka pada `http://localhost:8001`.

Untuk akses dari domain/server remote, mikrofon browser memerlukan HTTPS. Koneksi WebRTC juga bergantung pada STUN/TURN dan konfigurasi jaringan.

## 7. Kesimpulan

Proyek menghasilkan prototipe klasifikasi suara berisik dan non-berisik menggunakan 16 fitur akustik dan Random Forest dengan 200 pohon keputusan. Dataset 200 entri dibersihkan menjadi 176 sampel unik seimbang setelah audit duplikasi. Evaluasi berkelompok menghasilkan akurasi out-of-fold 75,57%. Aplikasi dikemas dengan Dockerfile dan Docker Compose menggunakan satu port aplikasi untuk kelompok 1.

### Keterbatasan dan pengembangan
1. Label audio belum seluruhnya diverifikasi manual.
2. Satu konflik label dikeluarkan sementara.
3. Evaluasi perlu diperluas dengan data rekaman baru dari lingkungan yang berbeda.
4. Performa mikrofon live perlu diuji karena panjang jendela live berbeda dari durasi rekaman training.
5. Deployment remote memerlukan HTTPS dan konfigurasi WebRTC yang sesuai.
