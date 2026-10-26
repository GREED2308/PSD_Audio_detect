# Crawling Data — Proses Pengumpulan Data Audio

Bab ini mendokumentasikan prosedur pengumpulan dan perolehan data (*data acquisition/crawling*) yang menjadi fondasi empiris dalam proyek klasifikasi audio ini.

## 3.1 Sumber dan Mekanisme Pengumpulan Data

Data audio dalam penelitian ini tidak diambil menggunakan teknik *web scraping* otomatis berbasis bot, melainkan dihimpun secara terkoordinasi melalui kontribusi mahasiswa peserta mata kuliah Proyek Sains Data (PSD):
- **Penyedia Dataset**: Mahasiswa kelas Proyek Sains Data — daftar nama kontributor, NIM, dan rincian pembagian kelompok: [perlu diisi kelompok].
- **Kanal Pengumpulan Data**: Dua berkas arsip terkompresi (`berisik.zip` dan `non-berisik.zip`) yang dibagikan melalui Google Drive kelas: tautan folder/URL Google Drive: [perlu diisi kelompok].
- **Instrumen Perekaman**: Perekaman dilakukan secara mandiri oleh masing-masing kontributor menggunakan perangkat mikrofon ponsel pintar (*smartphone*) atau laptop pribadi dalam variasi kondisi lingkungan sehari-hari.

## 3.2 Karakteristik Lingkungan Audio yang Dikumpulkan

Pengumpulan sampel audio ditujukan untuk mencakup dua kondisi akustik yang kontras:

1. **Kategori Berisik (`berisik`)**:
   - Sampel merekam kondisi ruang dengan intensitas energi suara tinggi atau polusi akustik yang nyata.
   - Sumber kebisingan mencakup: lalu lintas kendaraan bermotor di tepi jalan raya, keramaian interaksi manusia di area publik (kafetaria, kantin, stasiun, ruang keluarga), dengung mesin pendingin ruangan (AC) atau kipas angin dari jarak dekat, serta kebisingan peralatan mekanik dan musik latar dengan volume dominan.

2. **Kategori Non-berisik (`non_berisik`)**:
   - Sampel merekam kondisi lingkungan yang tenang dan minim interferensi suara berenergi tinggi.
   - Sumber lingkungan mencakup: kamar tidur tertutup pada malam hari, ruang belajar mandiri atau perpustakaan sunyi, suasana hening pedesaan tanpa deru mesin, serta rekaman latar belakang (*ambient background noise*) dengan tingkat amplitudo minimal.

## 3.3 Struktur Penyimpanan Data Mentah

Seluruh berkas audio yang diunduh dari arsip Google Drive diekstraksi ke direktori penyimpanan data mentah lokal (`data/raw/`) tanpa dilakukan manipulasi awal:
```text
data/raw/
├── berisik/
│   ├── [kode_sumber] berisik [indeks].mp3
│   └── ...
└── non-berisik/
    ├── [kode_sumber] non-berisik [indeks].mp3
    └── ...
```

Format penamaan berkas umumnya mencantumkan identitas pengunggah berupa kode mahasiswa (misalnya `24-034`) diikuti label kelas dan nomor urut rekaman. Informasi kode pengunggah ini sangat berharga karena nantinya dimanfaatkan untuk memetakan kelompok sumber rekaman (*source group*) demi mencegah kebocoran data (*data leakage*).

## 3.4 Audit Integritas Data Mentah

Sebelum data diproses ke tahap pembersihan dan ekstraksi fitur, dilakukan serangkaian pengujian integritas awal:
1. **Verifikasi Keterbacaan Berkas**: Pengujian pembacaan *header* dan decoding awal untuk memastikan seluruh berkas dapat didekode tanpa korupsi biner (*corrupted files*). Seluruh 207 berkas dinyatakan lolos uji decoding.
2. **Pencatatan Fingerprint SHA-256**: Setiap berkas dihitung nilai intisari SHA-256 miliknya guna membangun basis data katalog unik (*fingerprint manifest*), mempermudah deteksi berkas identik, serta mendokumentasikan asal-usul data secara akuntabel.
