# Pembaruan Visualizer Suara Langsung

## Perilaku

Aplikasi menampilkan visualisasi waveform/equalizer yang dibuat dengan HTML Canvas dan menerima sampel dari stream mikrofon WebRTC yang sama dengan pipeline klasifikasi. Tidak ada izin mikrofon kedua atau file audio tambahan.

- Garis menjadi lurus ketika RMS dari sekitar 300 ms audio terakhir berada di bawah ambang visualisasi.
- Ketika suara melewati ambang, garis waveform menggambarkan sampel audio terbaru dan batang simetris menggambarkan envelope RMS selama sekitar 1,8 detik.
- Visualizer diperbarui sekitar setiap 0,3 detik; prediksi Random Forest tetap berjalan sesuai interval prediksi yang dipilih.
- Ambang visualisasi diatur lewat sidebar dan tidak mengubah masukan model maupun label prediksi.
- Sampel mentah tidak disimpan permanen. Jika aplikasi di-host di server, stream audio diproses oleh server tersebut.

## Pengujian

- Sinyal diam: fungsi data visual mengembalikan amplitudo nol dan status hening.
- Sinyal sinus 440 Hz: fungsi mengembalikan waveform non-datar dan envelope non-nol.
- HTML memuat status hening/aktif yang benar.
- Canvas HTML dirender di Chromium headless untuk kondisi hening dan nada uji 440 Hz; keduanya menggambar visual sesuai status.
- Sintaks Python dan JavaScript diperiksa; seluruh suite `pytest` lulus (8 tes).

## Keterbatasan

Widget browser dengan mikrofon sungguhan belum diuji end-to-end pada lingkungan build. Jalankan aplikasi secara lokal, izinkan mikrofon browser, uji ruangan hening dan uji dengan bicara/tepuk tangan. Jika garis masih bergerak ketika hening, naikkan ambang visualisasi. Jika suara pelan tidak terlihat, turunkan ambang.
