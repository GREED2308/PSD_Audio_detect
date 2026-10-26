# Evaluation — Evaluasi Performa Model

Tahap *Evaluation* mengukur sejauh mana model Random Forest yang dibangun mampu mengenali pola kebisingan akustik secara akurat pada data independen yang belum pernah dilihat sebelumnya (*held-out test set*), serta menganalisis kontribusi fitur dan keterbatasan model.

## 7.1 Hasil Evaluasi pada Held-out Test Set

Pengujian model dilakukan pada partisi uji independen yang terdiri dari **34 sampel** (18 berisik dan 16 non-berisik) dari 6 kelompok sumber rekaman. Sebagai pembanding dasar (*benchmark*), digunakan model `DummyClassifier(strategy='most_frequent')` yang selalu memprediksi kelas mayoritas secara konstan.

Tabel perbandingan performa komprehensif disajikan berikut:

| Metrik Evaluasi | Model Random Forest | Baseline Mayoritas (*Dummy*) |
|---|---:|---:|
| **Accuracy** | **88,24%** (0,8824) | 47,06% (0,4706) |
| **Balanced Accuracy** | **88,54%** (0,8854) | 50,00% (0,5000) |
| **Macro Precision** | **88,54%** (0,8854) | 23,53% (0,2353) |
| **Macro Recall** | **88,54%** (0,8854) | 50,00% (0,5000) |
| **Macro F1-score** | **88,24%** (0,8824) | 32,00% (0,3200) |
| *Best Inner Grouped-CV Macro F1* | 73,23% (0,7323) | - |

Model Random Forest membuktikan keunggulan yang sangat signifikan dibandingkan baseline mayoritas, dengan peningkatan akurasi dari 47,06% menjadi 88,24% dan Macro F1 dari 32,00% menjadi 88,24%.

## 7.2 Laporan Klasifikasi per Kelas (*Classification Report*)

Rincian metrik performa untuk masing-masing kelas target diuraikan pada tabel berikut:

| Kelas Target | Presisi (*Precision*) | *Recall* | *F1-score* | Jumlah Data Uji (*Support*) |
|---|---:|---:|---:|---:|
| **Non-berisik** | 83,33% (0,8333) | 93,75% (0,9375) | 88,24% (0,8824) | 16 |
| **Berisik** | 93,75% (0,9375) | 83,33% (0,8333) | 88,24% (0,8824) | 18 |
| **Macro Average** | **88,54%** | **88,54%** | **88,24%** | **34** |
| **Weighted Average** | **88,85%** | **88,24%** | **88,24%** | **34** |

- **Kelas Non-berisik**: Memiliki *recall* yang sangat tinggi (93,75%), menandakan bahwa model sangat handal dalam mengenali kondisi tenang dan jarang salah mendeteksi keheningan sebagai keramaian.
- **Kelas Berisik**: Memiliki tingkat presisi tinggi (93,75%), yang berarti ketika model memprediksi status "Berisik", tingkat keyakinannya sangat akurat.

## 7.3 Analisis Matriks Konfusi (*Confusion Matrix*)

Matriks konfusi menyajikan perbandingan silang antara label aktual di lapangan dengan label hasil prediksi model (urutan baris dan kolom: `non_berisik`, `berisik`):

| Label Aktual \ Prediksi Model | Prediksi Non-berisik | Prediksi Berisik | Total Aktual |
|---|---:|---:|---:|
| **Aktual Non-berisik** | **15** (*True Negative*) | **1** (*False Positive*) | 16 |
| **Aktual Berisik** | **3** (*False Negative*) | **15** (*True Positive*) | 18 |
| **Total Prediksi** | 18 | 16 | 34 |

```{figure} images/confusion_matrix.png
---
name: fig-confusion-matrix
width: 75%
align: center
---
Matriks konfusi Random Forest pada *held-out test set* (34 sampel: 15 TN, 1 FP, 3 FN, 15 TP).
```

Dari 34 data uji, model hanya melakukan 4 kesalahan klasifikasi:
- 1 sampel non-berisik salah diprediksi sebagai berisik (*false positive*).
- 3 sampel berisik salah diprediksi sebagai non-berisik (*false negative*).

## 7.4 Tingkat Kepentingan Fitur (*Feature Importance*)

Analisis kontribusi fitur dihitung berdasarkan penurunan ketidakmurnian Gini (*mean decrease in impurity*) pada seluruh pohon keputusan di dalam model Random Forest:

```{figure} images/feature_importance.png
---
name: fig-feature-importance
width: 85%
align: center
---
Tingkat kepentingan 32 fitur akustik pada model Random Forest. Statistik koefisien MFCC dan energi RMS mendominasi pemisahan percabangan pohon.
```

Berdasarkan berkas `reports/feature_importance.csv`, fitur-fitur yang menduduki peringkat teratas antara lain:
- **`rms_mean`** dan **`rms_std`**: Menjadi pembeda primer tingkat energi amplitudo.
- **`mfcc_01_mean`**, **`mfcc_02_mean`**, dan **`mfcc_03_mean`**: Menangkap kontur selubung spektral frekuensi rendah hingga menengah.
- **`spectral_centroid_mean`**: Memberikan informasi bobot frekuensi tinggi yang membedakan desis kebisingan dengan keheningan ruangan.

## 7.5 Keterbatasan Evaluasi (*Evaluation Limitations*)

Dalam menafsirkan capaian metrik di atas, terdapat beberapa keterbatasan metodologis yang harus diperhatikan secara jujur:
1. **Ukuran Data Uji**: Partisi data uji hanya mencakup 34 sampel dari 6 kelompok sumber. Jumlah ini relatif terbatas sehingga satu kesalahan klasifikasi memiliki bobot persentase sebesar ~2,94%.
2. **Kesenjangan Metrik (*Metric Gap*)**: Nilai Macro F1 pada validasi silang internal adalah 73,23%, sedangkan pada held-out test set mencapai 88,24%. Perbedaan ini menunjukkan bahwa performa model cukup sensitif terhadap kelompok sumber rekaman yang dialokasikan.
3. **Validasi Semantik Label**: Label target berasal langsung dari pengelompokan folder oleh mahasiswa pengunggah (`berisik` dan `non_berisik`) dan **belum diverifikasi secara manual satu per satu** oleh ahli akustik. Evaluasi ini mengukur tingkat keselarasan (*agreement*) model terhadap label sumber, bukan kebenaran absolut semantik audio.
4. **Generalisasi ke Kondisi Nyata**: Dataset berasal dari lingkungan mahasiswa perkuliahan tertentu dan belum tentu mewakili seluruh ragam kebisingan industri, lalu lintas ekstrem, atau akustik ruangan khusus lainnya.
