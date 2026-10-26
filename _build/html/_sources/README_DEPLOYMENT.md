# Paket Deployment — PSD Ambient Audio Classifier

Paket ini menjalankan klasifikasi suara sekitar secara langsung melalui mikrofon browser. Pengguna tidak perlu mengunggah file audio. Paket memuat aplikasi Streamlit, model Random Forest terlatih, fungsi ekstraksi fitur, laporan evaluasi, dan notebook ringkasan. Dataset audio asli tidak disertakan.

## Jalankan lokal di CachyOS/Linux

1. Pastikan Python dan FFmpeg tersedia.
2. Buat environment dan install paket:

```bash
sudo pacman -S --needed ffmpeg python python-pip
python -m venv .venv
source .venv/bin/activate       # Bash/Zsh
# Fish: source .venv/bin/activate.fish
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Jalankan aplikasi:

```bash
streamlit run app.py
```

4. Buka URL localhost yang ditampilkan. Tekan **START** pada komponen mikrofon dan izinkan akses mikrofon. Grafik audio mulai merespons saat stream diterima; dalam keadaan hening yang berada di bawah ambang, grafik menjadi garis lurus. Setelah jendela audio pertama terkumpul (default 10 detik), hasil klasifikasi akan muncul dan diperbarui berkala. Tekan **STOP** untuk berhenti.

## Bagaimana deteksi live bekerja?

- Audio dari browser dikirim melalui WebRTC ke server yang menjalankan Streamlit.
- Audio distandarkan ke mono 16 kHz, kemudian dihitung 32 fitur: RMS (mean/std), spectral centroid (mean/std), ZCR (mean/std), serta MFCC 1–13 (mean/std).
- Random Forest memprediksi kelas `berisik` atau `non_berisik` dari jendela audio bergulir.
- Visualisasi Canvas khusus menampilkan garis waveform dan batang equalizer dari sinyal mikrofon yang sama. Grafik diperbarui sekitar setiap 0,3 detik; ketika RMS 300 ms terakhir berada di bawah ambang visualisasi, grafik menjadi garis lurus.
- Pengaturan **Ambang hening untuk visualisasi** ada di sidebar. Naikkan nilainya jika grafik masih bergerak saat ruangan terasa hening; turunkan jika suara pelan tidak terlihat. Pengaturan ini hanya memengaruhi visualisasi, bukan prediksi model.
- Grafik ini digambar secara dinamis dengan gaya yang terinspirasi oleh referensi equalizer yang diberikan, bukan menampilkan gambar statis.
- Aplikasi ini tidak menyimpan stream audio ke file secara permanen. Ketika di-host online, audio diproses oleh server hosting.
- Jendela mikrofon berbeda dari rekaman training yang rata-rata lebih panjang; kualitas prediksi live masih harus divalidasi pada kondisi nyata. Skor model bukan probabilitas terkalibrasi atau ukuran dB.

## Troubleshooting mikrofon

- Jika browser tidak meminta akses mikrofon, cek izin mikrofon di pengaturan situs/browser.
- Di komputer yang sama, buka `http://localhost:8501`. Untuk membuka dari perangkat lain atau hosting online, browser umumnya mensyaratkan HTTPS sebagai secure context.
- Jaringan ketat/firewall dapat menghalangi WebRTC. Hosting publik mungkin memerlukan konfigurasi STUN/TURN sesuai lingkungan deployment.
- Pastikan browser/perangkat memilih mikrofon yang benar dan tidak digunakan secara eksklusif oleh aplikasi lain.

## Hosting publik

- Aplikasi belum dipublikasikan ke URL publik; pemilik proyek perlu mengonfigurasi hosting.
- Hosting harus memakai HTTPS dan mendukung koneksi WebRTC. STUN saja tidak menjamin semua jaringan dapat terhubung; beberapa jaringan membutuhkan TURN.
- Minimal file: `app.py`, `src/audio_features.py`, `src/__init__.py`, `models/random_forest.joblib`, `requirements.txt`, dan `.streamlit/config.toml`.
- Karena audio pengguna dikirim ke server hosting, tampilkan pemberitahuan privasi yang sesuai sebelum demo/penggunaan.

### Uji visualizer

- Ruangan hening: grafik harus menjadi garis lurus di tengah.
- Tepuk tangan atau bicara dekat mikrofon: garis dan batang equalizer harus bergerak mengikuti amplitudo.
- Jika tidak sesuai, ubah **Ambang hening untuk visualisasi** pada sidebar.
- Visualizer diuji dengan sinyal sintetis (hening dan sinus 440 Hz); browser + mikrofon langsung tetap perlu dites di laptop sebelum presentasi.
