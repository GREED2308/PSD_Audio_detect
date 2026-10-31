# Deployment dengan Docker

1. Install dan jalankan Docker Desktop.
2. Buka PowerShell di direktori root proyek.
3. Validasi file Compose:
   ```powershell
   docker compose config
   ```
4. Build image dan jalankan:
   ```powershell
   docker compose up --build -d
   ```
5. Periksa status:
   ```powershell
   docker compose ps
   docker compose logs -f
   ```
6. Buka `http://localhost:8001`.
7. Untuk menghentikan:
   ```powershell
   docker compose down
   ```

Compose Kelompok 1 memetakan host `8001` ke port aplikasi `8080` di dalam container. Dockerfile membangun image dan memulai Streamlit pada port internal 8080.

## Push ke GitHub

```powershell
git status
git add .
git commit -m "Complete PSD audio classifier with Docker"
git branch -M main
git push -u origin main
```

Pastikan repository memakai `.gitignore`. Jangan push `.venv/`, file `.env`, atau ZIP MP3 mentah. CSV fitur dan model sudah disertakan di proyek.

## Pengujian microphone
Uji di localhost terlebih dahulu. Untuk domain/server remote, HTTPS dibutuhkan oleh browser untuk mengakses mikrofon. Koneksi WebRTC mungkin memerlukan TURN atau pengaturan firewall tambahan.
