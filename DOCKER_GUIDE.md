# Panduan Docker — Kelompok 1

## Persyaratan
- Docker Desktop terpasang dan berjalan.
- Terminal berada di folder root proyek (folder yang memuat `Dockerfile` dan `docker-compose.yml`).

## Validasi Compose
```powershell
docker compose config
```

## Build image dan jalankan container
```powershell
docker compose up --build -d
```

## Periksa container
```powershell
docker compose ps
docker compose logs -f
```

Buka browser ke `http://localhost:8001`.

## Hentikan aplikasi
```powershell
docker compose down
```

## Bangun ulang setelah mengubah kode/model
```powershell
docker compose down
docker compose up --build -d
```

## Port
Kelompok 1: `8001:8080`
- `8001` = port komputer/host.
- `8080` = port Streamlit di dalam container.

## Catatan mikrofon
Browser mengizinkan mikrofon pada `localhost`. Jika aplikasi diakses melalui IP/domain server, gunakan HTTPS. WebRTC dapat membutuhkan STUN/TURN atau penyesuaian firewall/jaringan.
