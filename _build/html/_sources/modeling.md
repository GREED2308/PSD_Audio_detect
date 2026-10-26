# Modeling — Pembangunan Model Klasifikasi

Tahap *Modeling* mencakup pemilihan algoritma pembelajaran mesin, perancangan skema validasi silang bebas kebocoran data (*leak-free cross-validation*), serta penalaan hyperparameter optimal.

## 6.1 Algoritma Random Forest

Model klasifikasi utama yang diimplementasikan adalah `RandomForestClassifier` dari pustaka scikit-learn {cite}`scikitlearn`. Random Forest merupakan algoritma *ensemble learning* berbasis pohon keputusan (*decision trees*) yang mengombinasikan teknik *bootstrap aggregating* (bagging) dan seleksi acak fitur (*random subspace method*).

Beberapa pertimbangan akademis dan praktis dalam pemilihan Random Forest untuk proyek ini meliputi:
1. **Ketahanan terhadap Data Berdimensi Sedang**: Algoritma ini terbukti sangat efektif dan stabil pada data tabular berdimensi kecil hingga menengah ($183 \times 32$).
2. **Invariansi terhadap Skala Fitur**: Berbeda dengan model linier, Support Vector Machine (SVM), atau Neural Network, Random Forest tidak memerlukan normalisasi skala fitur (*feature scaling*) karena pemisahan percabangan (*node split*) didasarkan pada perbandingan nilai urutan (*rank order*).
3. **Mereduksi Risiko Overfitting**: Penggabungan prediksi dari ratusan pohon keputusan individual secara signifikan memperkecil varians model dibandingkan penggunaan satu pohon keputusan tunggal (*Decision Tree*).
4. **Efisiensi Komputasi**: Proses inferensi sangat cepat, memungkinkan respons prediksi langsung (*near real-time*) pada antarmuka aplikasi web.

## 6.2 Skema Validasi Berkelompok (*StratifiedGroupKFold*)

Untuk menghindari estimasi performa yang overoptimis akibat kemiripan karakteristik rekaman dari sumber yang sama, partisi data dilakukan dengan skema `StratifiedGroupKFold` 5-fold:
- **Held-out Test Set**: Satu fold independen berisi 34 sampel (18 berisik dan 16 non-berisik) dari 6 kelompok sumber rekaman dialokasikan khusus untuk pengujian performa akhir.
- **Training Set**: Empat fold lainnya berisi 149 sampel (74 berisik dan 75 non-berisik) dari 21 kelompok sumber rekaman digunakan untuk penalaan parameter dan pelatihan model.
- **Pencegahan Data Leakage**: Overlap kelompok sumber (*source-group overlap*) antara himpunan latih dan uji dipastikan bernilai **0**.

| Karakteristik Partisi | Subset Pelatihan (*Train Set*) | Subset Pengujian (*Test Set*) |
|---|---:|---:|
| Total Sampel | 149 | 34 |
| Jumlah Kelas Berisik | 74 | 18 |
| Jumlah Kelas Non-berisik | 75 | 16 |
| Jumlah Kelompok Sumber (*Groups*) | 21 | 6 |
| **Tingkat Tumpang-Tindih Kelompok (*Overlap*)** | \multicolumn{2}{c|}{**0 kelompok**} |

## 6.3 Pencarian Hyperparameter Optimal (*Grid Search*)

Penalaan hyperparameter dilakukan pada subset data latih (149 sampel) menggunakan skema validasi silang 4-fold `StratifiedGroupKFold` internal (*inner CV*). Ruang pencarian (*parameter grid*) yang dievaluasi mencakup:

| Hyperparameter | Kandidat Nilai yang Dievaluasi | Parameter Terbaik Terpilih |
|---|---|---|
| `n_estimators` | [200, 400] | **200** |
| `max_depth` | [`None`, 12] | **`None`** (pohon tumbuh penuh) |
| `min_samples_leaf` | [1, 2, 4] | **4** |
| `max_features` | [`'sqrt'`, 0.7] | **`'sqrt'`** |

Kriteria seleksi model terbaik didasarkan pada nilai **Macro F1-score** tertinggi pada validasi silang internal:
- **Best Inner Grouped-CV Macro F1**: **0,7323** (73,23%).

Hyperparameter `min_samples_leaf=4` berperan penting sebagai mekanisme regularisasi untuk membatasi kompleksitas cabang pohon terkecil, sehingga mencegah pohon keputusan menghafal derau lokal (*noise*) dari rekaman pelatihan. Model akhir dilatih ulang pada seluruh 149 data latih dengan konfigurasi parameter terbaik tersebut dan disimpan ke dalam berkas biner `models/random_forest.joblib`.
