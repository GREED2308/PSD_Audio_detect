# Data Understanding — Deskripsi Dataset dan Eksplorasi Awal

Tahap *Data Understanding* berfokus pada pengenalan dataset audio mentah, identifikasi format fisik, pemeriksaan distribusi karakteristik temporal, serta audit terhadap keaslian dan konsistensi label rekaman.

## 2.1 Asal dan Jumlah Data

Dataset proyek ini bersumber dari kontribusi mahasiswa dalam kelas perkuliahan Proyek Sains Data yang dihimpun dalam dua berkas arsip terkompresi melalui Google Drive kelas. Audit awal terhadap dataset mentah menghasilkan rincian kuantitatif sebagai berikut:

| Kelas Target | Jumlah Entri Awal | Persentase |
|---|---:|---:|
| Berisik | 104 | 50,24% |
| Non-berisik | 103 | 49,76% |
| **Total** | **207** | **100,00%** |

Seluruh 207 berkas audio berhasil didekode secara tuntas pada proses audit awal. Kendati mayoritas berkas menggunakan ekstensi nama `.mp3`, inspeksi struktur *container* dan *codec* riil menunjukkan keberagaman format berkas sebagai berikut:
- **204 berkas**: Terverifikasi sebagai format MPEG Audio (MP3).
- **2 berkas**: Berformat Advanced Audio Coding (AAC).
- **1 berkas**: Merupakan *container* QuickTime/MOV/MP4 dengan *stream* audio berkodek AAC.

Temuan ini membuktikan bahwa pipeline pemrosesan tidak boleh hanya bergantung pada ekstensi berkas visual, melainkan harus memanfaatkan pustaka decoding yang mampu membaca struktur internal berkas (*magic bytes*).

Selain variasi format, parameter akustik dasar rekaman juga menunjukkan keragaman yang signifikan sebelum standardisasi:
- **Sample Rate Sumber**: 129 berkas memiliki *sample rate* 48.000 Hz, 72 berkas sebesar 44.100 Hz, dan 6 berkas sebesar 16.000 Hz.
- **Konfigurasi Kanal**: 147 berkas memiliki dua saluran audio (stereo), sedangkan 60 berkas memiliki satu saluran (mono).

## 2.2 Distribusi Durasi Rekaman

Analisis durasi rekaman audio mentah dihitung untuk masing-masing kelas target. Hasil statistik deskriptif disajikan pada tabel berikut:

| Kelas | Durasi Minimum | Median | Rata-rata (*Mean*) | Durasi Maksimum |
|---|---:|---:|---:|---:|
| Berisik | 18,039 detik | 30,216 detik | 31,135 detik | 48,170 detik |
| Non-berisik | 29,510 detik | 30,250 detik | 30,673 detik | 38,496 detik |

Berdasarkan tabel di atas, sebagian besar rekaman audio terpusat pada durasi sekitar 30 detik (median 30,2 detik). Dalam eksperimen pemodelan ini, setiap berkas audio utuh diperlakukan sebagai satu sampel pengamatan (*instance*) independen. Berkas tidak dipecah menjadi segmen-segmen pendek (*chunking*) guna menghindari korelasi spasio-temporal semu antar-segmen yang berasal dari rekaman yang sama.

## 2.3 Audit Duplikasi dan Konflik Label

Integritas dataset diperiksa menggunakan teknik *hash fingerprinting* SHA-256 terhadap seluruh berkas audio mentah. Temuan audit menunjukkan:
- Dari 207 entri awal, teridentifikasi sebanyak **184 nilai hash SHA-256 unik**.
- Ditemukan **18 kelompok duplikasi byte-identik** yang mencakup total **41 entri berkas**.
- Sebanyak **17 kelompok duplikat** berada dalam kelas label yang konsisten (*intra-class duplicate*).
- Ditemukan **1 kelompok duplikat yang memiliki konflik label antar-kelas (*cross-label conflict*)**, di mana satu nilai hash byte yang sama persis muncul pada dua nama berkas dengan label berlawanan:
  1. `24-034 berisik 1.mp3` (diberi label folder `berisik`)
  2. `24-034 non-berisik 1.mp3` (diberi label folder `non_berisik`)

Karena kedua berkas tersebut identik hingga level biner (*byte-for-byte*), metadata rekaman saja tidak dapat memastikan kebenaran semantiknya tanpa verifikasi langsung dari pemilik rekaman. Oleh sebab itu, kedua berkas yang berkonflik tersebut dikeluarkan sementara dari matriks pemodelan. Sementara itu, untuk 17 kelompok duplikat dalam kelas yang sama, dipertahankan satu representasi berkas unik secara deterministik agar konten yang sama tidak mendistorsi bobot pembelajaran model.

## 2.4 Visualisasi Eksploratif

Berikut adalah representasi visual dari distribusi kelas dan karakteristik dataset:

```{figure} images/raw_class_distribution.png
---
name: fig-raw-class-dist
width: 75%
align: center
---
Distribusi kelas pada dataset awal (207 rekaman mentah: 104 berisik dan 103 non-berisik).
```

```{figure} images/cleaned_class_distribution.png
---
name: fig-cleaned-class-dist
width: 75%
align: center
---
Distribusi kelas setelah tahap deduplikasi dan eksklusi berkas konflik label (183 sampel unik: 92 berisik dan 91 non-berisik).
```

```{figure} images/duration_distribution.png
---
name: fig-duration-dist
width: 85%
align: center
---
Distribusi durasi rekaman audio untuk kategori berisik dan non-berisik, menunjukkan pemusatan data pada kisaran 30 detik.
```
