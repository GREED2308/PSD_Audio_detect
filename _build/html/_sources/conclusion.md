# Conclusion — Kesimpulan dan Saran Pengembangan Lanjutan

Bab penutup ini merangkum capaian esensial dari keseluruhan siklus metodologi CRISP-DM pada proyek klasifikasi audio, serta menguraikan rekomendasi strategis bagi pengembangan sistem di masa mendatang.

## 9.1 Kesimpulan Proyek

1. **Penerapan Siklus Lengkap CRISP-DM**: Proyek ini berhasil mengimplementasikan alur kerja sains data yang utuh, mulai dari pemahaman masalah (*Business Understanding*), pemahaman dan audit dataset (*Data Understanding*), akuisisi dan pra-pemrosesan (*Crawling & Preprocessing*), rekayasa fitur (*Feature Extraction*), pemodelan (*Modeling*), evaluasi empiris (*Evaluation*), hingga penerapan aplikasi inferensi interaktif (*Deployment*).
2. **Audit dan Integritas Dataset**: Dari 207 berkas audio awal, audit hash SHA-256 berhasil mengungkap 18 kelompok duplikasi (41 entri) dan 1 konflik label biner lintas-kelas (`24-034`). Melalui prosedur deduplikasi intra-kelas dan isolasi konflik label, diperoleh 183 sampel pemodelan yang bersih dan berimbang (92 sampel `berisik` dan 91 sampel `non_berisik`).
3. **Representasi Fitur Akustik Standar**: Berhasil diekstraksi 32 fitur akustik berdimensi tetap per sampel audio menggunakan Librosa {cite}`librosa2020` (2 fitur RMS, 2 fitur *spectral centroid*, 2 fitur *zero-crossing rate*, dan 26 fitur statistik MFCC 1–13) tanpa mengalami kegagalan ekstraksi numerik.
4. **Pencegahan Kebocoran Data (*Data Leakage*)**: Skema partisi `StratifiedGroupKFold` 5-fold berhasil memisahkan 149 data latih (21 kelompok sumber) dan 34 data uji (6 kelompok sumber) dengan tingkat tumpang-tindih kelompok tepat **0**.
5. **Kinerja Model Klasifikasi**: Model Random Forest {cite}`scikitlearn` dengan konfigurasi terbaik (`n_estimators=200`, `max_depth=None`, `min_samples_leaf=4`, `max_features='sqrt'`) menghasilkan:
   - **Akurasi**: 88,24%
   - **Balanced Accuracy**: 88,54%
   - **Macro F1-score**: 88,24%
   - Kinerja ini melampaui performa *majority baseline* (akurasi 47,06% dan Macro F1 32,00%) secara sangat signifikan pada *held-out test set* yang sama.
6. **Implementasi Aplikasi Nyata**: Model berhasil diintegrasikan ke dalam antarmuka web interaktif Streamlit dan WebRTC {cite}`streamlitwebrtc` yang mampu memproses audio mikrofon peramban secara *in-memory* dalam jendela bergulir (default 10 detik), menyajikan visualizer osiloskop/equalizer dinamis berbasis HTML Canvas, serta memberikan respons audio peringatan otomatis ketika lingkungan terdeteksi berisik.

## 9.2 Saran dan Rekomendasi Lanjutan (*Future Work*)

Berdasarkan temuan dan keterbatasan yang diidentifikasi selama penelitian, beberapa langkah pengembangan lanjutan yang disarankan adalah:

1. **Verifikasi Semantik Label Secara Manual**: Melakukan audit dengar (*listening test*) manual satu per satu terhadap seluruh berkas audio, khususnya untuk menyelesaikan anomali pada berkas berkonflik `24-034` serta sampel-sampel yang memiliki prediksi model bertolak belakang dengan label folder.
2. **Ekspansi dan Diversifikasi Dataset**: Menambah jumlah rekaman mandiri dengan variasi kondisi akustik yang lebih luas, seperti ruang pabrik, perkantoran terbuka, kafe luar ruangan, variasi jarak mikrofon terhadap sumber suara, serta penggunaan jenis perangkat perekam yang beragam.
3. **Validasi Silang Bertingkat (*Nested Cross-Validation*)**: Menerapkan skema *nested cross-validation* atau *repeated grouped cross-validation* guna memperoleh estimasi selang kepercayaan (*confidence interval*) performa model yang lebih tahan terhadap variasi pembagian kelompok sumber.
4. **Eksplorasi Representasi Spektrogram 2D dan *Deep Learning***: Mengembangkan pemodelan berbasis representasi spektrogram Mel dua dimensi menggunakan arsitektur *Convolutional Neural Networks* (CNN, misalnya arsitektur ResNet atau MobileNet audio) atau model transformer audio terkini (*Audio Spectrogram Transformer* / AST dan Wav2Vec) untuk menangkap pola temporal dan frekuensi secara simultan.
5. **Kalibrasi Probabilitas dan Ambang Adaptif**: Menerapkan metode kalibrasi probabilitas (*Platt Scaling* atau *Isotonic Regression*) pada skor prediksi model, serta merancang ambang deteksi dinamis yang dapat menyesuaikan diri secara otomatis terhadap derau dasar (*noise floor*) ruangan pengguna.

---

## 9.3 Daftar Pustaka

```{bibliography}
```

