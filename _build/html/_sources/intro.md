# Sistem Klasifikasi Audio Berisik dan Non-berisik

**Klasifikasi Audio Suara Sekitar Menggunakan Ekstraksi Fitur Librosa dan Random Forest**

- **Mata Kuliah:** Proyek Sains Data (PSD)
- **Anggota Kelompok:** 4 orang — [perlu diisi kelompok]
- **Dataset:** Rekaman audio lingkungan (berisik dan non-berisik) dari Google Drive kelas
- **Metodologi:** CRISP-DM (*Cross-Industry Standard Process for Data Mining*) {cite}`chapman2000crisp`

---

## Ringkasan Proyek

Proyek ini membangun pipeline klasifikasi audio dua kelas dari tahap Business Understanding sampai Deployment dengan metodologi CRISP-DM. Audio diproses menjadi mono dengan sample rate 16 kHz, lalu direpresentasikan oleh 32 fitur akustik yang mencakup RMS, *spectral centroid*, *zero-crossing rate*, dan 13 koefisien MFCC beserta statistik *mean* dan *standard deviation* {cite}`librosa2020`. Model Random Forest {cite}`scikitlearn` dilatih menggunakan data yang telah dideduplikasi. Pemisahan data dilakukan berdasarkan kelompok sumber rekaman agar rekaman dari sumber yang sama tidak tersebar ke data latih dan uji untuk mencegah kebocoran data (*data leakage*). Dataset awal memuat 207 entri (104 berisik dan 103 non-berisik). Audit menemukan 18 kelompok konten identik yang melibatkan 41 entri serta satu *hash* yang memiliki label bertentangan. Setelah salinan identik dalam label yang sama dipertahankan satu wakilnya dan konflik lintas kelas dikeluarkan sementara, diperoleh 183 sampel unik: 92 berisik dan 91 non-berisik. Dalam eksperimen ini, Random Forest dengan parameter hasil pencarian memberikan *accuracy* 88,24%, *balanced accuracy* 88,54%, dan *macro F1* 88,24% pada satu *hold-out test set* berisi 34 sampel. Nilai *macro F1* rata-rata terbaik pada *grouped cross-validation* internal adalah 73,23%, sedangkan *baseline* mayoritas menghasilkan *accuracy* 47,06% pada *test set* yang sama. Model ini kemudian diintegrasikan ke dalam aplikasi web Streamlit dengan koneksi WebRTC {cite}`streamlitwebrtc` untuk klasifikasi audio mikrofon secara langsung.

---

## Daftar Isi

1. [Business Understanding — Latar Belakang, Tujuan, dan Rumusan Masalah](business-understanding.md)
2. [Data Understanding — Deskripsi Dataset dan Eksplorasi Awal](data-understanding.md)
3. [Crawling Data — Proses Pengumpulan Data Audio](crawling.md)
4. [Preprocessing — Pembersihan dan Persiapan Data](preprocessing.md)
5. [Feature Extraction — Ekstraksi Fitur dari Sinyal Audio](feature-extraction.md)
6. [Modeling — Pembangunan Model Klasifikasi](modeling.md)
7. [Evaluation — Evaluasi Performa Model](evaluation.md)
8. [Deployment — Sistem Inferensi Aplikasi Web Streamlit](deployment.md)
9. [Conclusion — Kesimpulan dan Saran Pengembangan Lanjutan](conclusion.md)
