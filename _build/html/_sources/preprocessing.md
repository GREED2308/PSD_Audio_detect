# Preprocessing — Pembersihan dan Persiapan Data

Tahap *Data Preparation / Preprocessing* memastikan bahwa sinyal audio mentah yang beragam ditransformasikan ke dalam representasi digital yang konsisten, bersih dari distorsi duplikasi, serta terbebas dari risiko kebocoran data (*data leakage*).

## 4.1 Pemeriksaan Format dan Arsitektur Decoding

Seluruh berkas audio mentah disimpan utuh di direktori `data/raw/` tanpa mengubah metadata aslinya. Untuk mengatasi keragaman kontainer berkas (seperti MP3, AAC, dan kontainer MOV/MP4), arsitektur decoding dibangun dengan strategi berlapis (*fallback strategy*):
1. **Jalur Utama (Librosa)**: Menggunakan fungsi `librosa.load` {cite}`librosa2020` untuk membaca berkas standar dengan cepat.
2. **Jalur Cadangan (FFmpeg Fallback)**: Jika pustaka utama gagal membaca berkas karena struktur kontainer non-standar (misalnya berkas AAC biner yang diberi nama ekstensi `.mp3`), sistem secara otomatis menjalankan proses `subprocess` berbasis FFmpeg untuk mengekstraksi aliran audio mentah ke dalam format PCM WAV 16-bit melalui antarmuka *pipe* standar (`pipe:1`).

Dengan mekanisme ini, seluruh 207 rekaman dapat terbaca sempurna tanpa kehilangan sampel data akibat kegagalan decoding teknis.

## 4.2 Standardisasi Sinyal Audio

Setiap sinyal audio yang didekode ditransformasikan menjadi sinyal digital terstandar dengan spesifikasi berikut:
1. **Konversi Saluran Tunggal (Mono)**: Sinyal stereo (dua kanal) diubah menjadi mono dengan merata-ratakan amplitudo kedua saluran secara matematis:
   $$y_{\text{mono}}[n] = \frac{y_{\text{left}}[n] + y_{\text{right}}[n]}{2}$$
2. **Penyelarasan Frekuensi Sampling (Resampling 16.000 Hz)**: Seluruh audio di-*resample* menjadi 16 kHz. Menurut Teorema Sampling Nyquist-Shannon, laju sampling 16 kHz mampu merepresentasikan komponen frekuensi sinyal hingga batas 8 kHz (*Nyquist frequency*). Batas ini mencakup rentang spektral vokal manusia (300 Hz – 3,4 kHz) serta sebagian besar komponen derau lingkungan, sekaligus mereduksi kompleksitas komputasi secara signifikan.
3. **Tanpa Normalisasi Volume Otomatis**: Normalisasi amplitudo puncak (*peak normalization*) sengaja **tidak dilakukan**. Intensitas energi gelombang suara merupakan informasi biologis dan fisik yang fundamental untuk membedakan antara lingkungan berisik (energi tinggi) dan lingkungan hening (energi rendah).
4. **Retensi Frame Keheningan (*Silence Retention*)**: Bagian hening dalam rekaman sengaja **tidak dibuang**. Keberadaan jeda atau latar belakang senyap merupakan karakteristik intrinsik dari kelas non-berisik yang esensial bagi pembelajaran model.

## 4.3 Deduplikasi dan Penanganan Konflik Label

Berdasarkan audit hash SHA-256 pada Bab 2, dilakukan penyaringan dataset untuk menyusun matriks pemodelan yang objektif:
- Dari 207 entri awal, teridentifikasi 184 hash unik.
- **Eksklusi Konflik Antar-Kelas**: 1 kelompok duplikat dengan label bertentangan (`24-034 berisik 1.mp3` dan `24-034 non-berisik 1.mp3`) dikeluarkan sementara dari pemodelan karena inkonsistensi semantik yang belum dapat dikonfirmasi.
- **Deduplikasi Intra-Kelas**: Untuk 17 kelompok duplikat yang memiliki label identik, dipertahankan tepat satu berkas perwakilan secara deterministik.
- **Hasil Pembersihan Matriks Modeling**:
  
  | Kategori | Jumlah Entri Awal | Konflik Dikeluarkan | Salinan Dieliminasi | Sampel Bersih Modeling |
  |---|---:|---:|---:|---:|
  | Berisik | 104 | 1 | 11 | 92 |
  | Non-berisik | 103 | 1 | 11 | 91 |
  | **Total** | **207** | **2** | **22** | **183** |

Seluruh 183 sampel akhir telah diverifikasi tidak memiliki nilai *non-finite* (bebas dari nilai NaN atau tak-hingga).

## 4.4 Pemisahan Berbasis Grup (Pencegahan Data Leakage)

Pemisahan data menjadi data latih (*training set*) dan data uji (*test set*) merupakan tahap kritis. Dalam dataset audio yang dihimpun oleh mahasiswa, rekaman dari satu kontributor memiliki karakteristik instrumen mikrofon, ruangan, dan derau latar (*room acoustics*) yang serupa. Jika rekaman dari kontributor yang sama terbagi ke dalam data latih dan data uji, model dapat mengalami *data leakage*—di mana model menghafal karakteristik mikrofon/ruangan tertentu, bukan mengenali konsep kebisingan secara umum.

Untuk mencegah permasalahan tersebut, partisi data dilakukan menggunakan algoritma `StratifiedGroupKFold` 5-fold dari scikit-learn {cite}`scikitlearn`:
- Seluruh sampel dari satu kelompok sumber rekaman (*source group*) dialokasikan secara eksklusif ke satu partisi saja.
- Fold ke-1 dialokasikan sebagai *held-out test set* independen (34 sampel dari 6 kelompok sumber).
- Sisa 4 fold dialokasikan sebagai *training set* (149 sampel dari 21 kelompok sumber).

Rincian pembagian data pemodelan disajikan pada tabel berikut:

| Partisi Dataset | Total Sampel | Berisik | Non-berisik | Jumlah Kelompok Sumber |
|---|---:|---:|---:|---:|
| Data Latih (*Train*) | 149 | 74 | 75 | 21 |
| Data Uji (*Held-out Test*) | 34 | 18 | 16 | 6 |
| **Overlap Kelompok Sumber** | **0** | **-** | **-** | **0** |

Nilai *group overlap* bernilai tepat **0** membuktikan bahwa data uji benar-benar berasal dari sumber independen yang belum pernah dilihat oleh model selama proses pelatihan.
