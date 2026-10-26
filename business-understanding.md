# Business Understanding — Latar Belakang, Tujuan, dan Rumusan Masalah

Tahap pertama dari metodologi CRISP-DM (*Cross-Industry Standard Process for Data Mining*) {cite}`chapman2000crisp` adalah *Business Understanding*, yang bertujuan untuk memahami tujuan dan kebutuhan proyek dari sudut pandang pemecahan masalah praktis, kemudian menerjemahkannya ke dalam definisi masalah sains data.

## 1.1 Latar Belakang

Audio lingkungan memiliki karakteristik yang sangat bervariasi ditinjau dari segi energi sinyal, distribusi frekuensi spektral, serta dinamika temporal. Dalam berbagai skenario, seperti pemantauan kebisingan di ruang publik, ruang kerja, ruang belajar, hingga perpustakaan, keberadaan suara berisik (*noise*) dapat mengganggu konsentrasi dan kenyamanan. Pemeriksaan dan pemantauan manual terhadap setiap sumber audio memakan waktu dan tenaga yang besar. Oleh karena itu, proyek ini mengeksplorasi pembangunan sistem klasifikasi otomatis yang mampu memprediksi apakah suatu sinyal audio lingkungan tergolong ke dalam kategori **berisik** atau **non-berisik** berdasarkan ekstraksi representasi fitur-fitur akustik.

## 1.2 Rumusan Masalah

Berdasarkan latar belakang tersebut, rumusan masalah utama dalam proyek ini adalah:
> Bagaimana merancang dan membangun sistem yang mampu mengklasifikasikan audio ke dalam kelas berisik dan non-berisik secara otomatis menggunakan ekstraksi fitur sinyal audio berbasis Librosa dan algoritma Random Forest?

## 1.3 Tujuan Proyek

Tujuan spesifik yang ingin dicapai melalui proyek ini mencakup:
1. **Memahami karakteristik dan kualitas dataset audio**: Menganalisis variasi format berkas, *sample rate*, jumlah kanal, serta distribusi durasi rekaman lingkungan.
2. **Menyiapkan pipeline data yang konsisten**: Menyusun prosedur pra-pemrosesan (*preprocessing*) yang mencakup decoding fleksibel, konversi mono, *resampling* seragam pada 16 kHz, deduplikasi data, dan isolasi konflik label.
3. **Mengekstrak fitur akustik berdimensi tetap**: Menghasilkan 32 fitur akustik standar (RMS, *spectral centroid*, *zero-crossing rate*, dan koefisien MFCC 1–13) untuk merepresentasikan setiap rekaman audio.
4. **Melatih dan mengevaluasi model pembelajaran mesin**: Mengembangkan model klasifikasi Random Forest dengan strategi pemisahan data berbasis kelompok sumber rekaman (*grouped split*) untuk mengeliminasi potensi kebocoran data (*data leakage*).
5. **Menyediakan prototipe aplikasi inferensi langsung**: Mengintegrasikan model terlatih ke dalam aplikasi berbasis web interaktif Streamlit yang mampu menerima masukan audio mikrofon peramban secara *real-time*.

## 1.4 Pengguna dan Ruang Lingkup

- **Sasaran Pengguna**: Mahasiswa, dosen penilai, serta penguji proyek yang ingin menguji dan mendemonstrasikan kapabilitas klasifikasi suara lingkungan secara praktis melalui antarmuka web.
- **Ruang Lingkup Input**: Aliran (*stream*) audio mikrofon peramban pengguna yang ditangkap secara lokal melalui protokol WebRTC dan dianalisis dalam jendela bergulir (*rolling window*) berdurasi default 10 detik.
- **Ruang Lingkup Output**: Prediksi label kelas (`berisik` atau `non_berisik`), estimasi skor probabilitas model per kelas, visualisasi osiloskop/equalizer dinamis, serta notifikasi audio peringatan ("Jangan berisik!").
- **Batasan Sistem**: Proyek ini dirancang sebagai instrumen prototipe akademik dan pembuktian konsep (*proof-of-concept*). Sistem ini bukan merupakan alat ukur tingkat intensitas kebisingan (desibel / dB) yang terkalibrasi secara akustik industri.

## 1.5 Definisi Target dan Kriteria Keberhasilan

- **Definisi Target**: Label kelas target didefinisikan secara biner, yaitu `berisik` dan `non_berisik`, yang mengacu pada label folder sumber dataset perkuliahan.
- **Kriteria Keberhasilan Teknis**:
  1. Keberhasilan pembangunan pipeline end-to-end tanpa galat *runtime* pada proses decoding, ekstraksi fitur, hingga inferensi.
  2. Penerapan pemisahan data berbasis kelompok (*GroupKFold*) dengan nilai overlap kelompok sumber rekaman bernilai 0 antara subset latih dan uji guna mencegah *data leakage*.
  3. Kinerja model yang dievaluasi melalui metrik komprehensif (*Accuracy*, *Balanced Accuracy*, dan *Macro F1-score*) yang melampaui performa *majority baseline classifier*.
  4. Aplikasi web mampu menerima aliran mikrofon pengguna, memproses data audio dalam memori (*in-memory*), serta memperbarui prediksi inferensi secara berkala.
