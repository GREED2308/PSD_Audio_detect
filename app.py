"""Live ambient-sound classifier for the PSD project.

The browser microphone sends audio frames through WebRTC to this Streamlit app.
Audio is analyzed in a rolling window and is not saved to disk by this app.
"""
from __future__ import annotations

import queue
import time
from pathlib import Path

import av
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src.audio_features import FEATURE_NAMES, SAMPLE_RATE, extract_feature_vector
from src.wave_visualizer import render_waveform

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
ALERT_SOUND_PATH = ROOT / "assets" / "jangan_berisik.mp3"
ALERT_SOUND_BYTES = ALERT_SOUND_PATH.read_bytes() if ALERT_SOUND_PATH.exists() else None
DISPLAY = {"berisik": "Berisik", "non_berisik": "Non-berisik"}

st.set_page_config(page_title="Deteksi Suara Sekitar | PSD", page_icon="🔊", layout="centered")
st.title("Deteksi Suara di Sekitar")
st.caption("Proyek PSD · 16 fitur audio Librosa · Random Forest · Mikrofon langsung")
st.write(
    "Aplikasi mendengarkan mikrofon setelah kamu menekan **START**, lalu menganalisis "
    "audio secara berkala untuk memperkirakan apakah lingkungan terdengar berisik atau non-berisik. "
    "Tidak perlu mengunggah file audio."
)

if not MODEL_PATH.exists():
    st.error("Model belum tersedia. Pastikan `models/random_forest.joblib` ada di folder proyek.")
    st.stop()

@st.cache_resource
def load_model_payload():
    return joblib.load(MODEL_PATH)

payload = load_model_payload()
model = payload["model"]
features_order = payload.get("feature_names", FEATURE_NAMES)

with st.expander("Cara kerja dan privasi", expanded=True):
    st.markdown(
        """
        1. Tekan **START** pada panel mikrofon di bawah dan izinkan akses mikrofon di browser.
        2. Tunggu sampai jendela audio pertama terkumpul.
        3. Hasil akan diperbarui berkala selama deteksi berjalan. Tekan **STOP** untuk berhenti.

        Audio dikirim melalui koneksi WebRTC ke proses Python yang menjalankan aplikasi agar dapat
        dianalisis. Aplikasi ini tidak menyimpan rekaman audio ke file secara permanen.
        Pada hosting online, audio diproses oleh server hosting tersebut.
        """
    )
    st.warning(
        "Model dilatih menggunakan ringkasan fitur mean dari rekaman sekitar 30 detik. Jendela "
        "mikrofon langsung dapat lebih pendek sehingga hasilnya merupakan estimasi. Ini bukan "
        "pengukuran tingkat kebisingan terkalibrasi. Label sumber belum seluruhnya diverifikasi manual."
    )

with st.sidebar:
    st.header("Pengaturan deteksi")
    window_seconds = st.slider(
        "Panjang jendela audio (detik)", min_value=5, max_value=15, value=10, step=1,
        help="Audio terbaru sepanjang ini digunakan untuk setiap prediksi. Jendela yang lebih panjang cenderung lebih stabil, tetapi respons lebih lambat.",
    )
    update_seconds = st.slider(
        "Interval pembaruan prediksi (detik)", min_value=1, max_value=5, value=2, step=1,
        help="Seberapa sering model menjalankan prediksi setelah jendela pertama penuh.",
    )
    play_voice_alert = st.toggle(
        "Putar suara peringatan saat terdeteksi Berisik", value=True,
        help="Memutar audio 'Jangan berisik!' ketika hasil berubah menjadi Berisik.",
    )
    test_alert_clicked = st.button("Tes suara peringatan")
    st.caption("Jika browser memblokir autoplay, gunakan tombol putar pada pemutar audio peringatan.")
    if ALERT_SOUND_BYTES is None:
        st.warning("Audio peringatan tidak ditemukan. Pastikan assets/jangan_berisik.mp3 tersedia.")
    visual_threshold = st.slider(
        "Ambang hening untuk visualisasi", min_value=0.001, max_value=0.030,
        value=0.008, step=0.001, format="%.3f",
        help="Jika RMS suara terbaru di bawah ambang ini, grafik diratakan. Naikkan jika garis masih bergerak saat ruangan terasa hening.",
    )
    st.caption("Pipeline klasifikasi: mono 22,05 kHz → 16 fitur mean → Random Forest")

if test_alert_clicked and ALERT_SOUND_BYTES is not None:
    st.success("Tes audio peringatan: 'Jangan berisik!'")
    st.audio(
        ALERT_SOUND_BYTES,
        format="audio/mpeg",
        autoplay=True,
        alt="Tes suara peringatan Jangan berisik",
    )

if not hasattr(av, "AudioResampler"):
    st.error("PyAV tidak mendukung AudioResampler pada instalasi ini. Perbarui paket `av` dan `streamlit-webrtc`.")
    st.stop()

ctx = webrtc_streamer(
    key="ambient-audio-classifier",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": False,
        "audio": {
            "echoCancellation": False,
            "noiseSuppression": False,
            "autoGainControl": False,
        },
    },
)

st.subheader("Visualisasi suara sekitar")
st.caption(
    "Grafik dibuat khusus dengan gaya equalizer seperti referensi yang kamu kirim. "
    "Saat suara berada di bawah ambang hening, grafik menjadi garis lurus; saat ada suara, "
    "gelombang dan batang amplitudo mengikuti sinyal mikrofon."
)
waveform_place = st.empty()
render_waveform(
    waveform_place, np.zeros(int(0.28 * SAMPLE_RATE), dtype=np.float32),
    SAMPLE_RATE, visual_threshold, "MENUNGGU MIKROFON",
)

status_place = st.empty()
result_place = st.empty()
detail_place = st.empty()
alert_audio_place = st.empty()
progress_place = st.empty()

if not ctx.state.playing:
    status_place.info(
        "Mikrofon belum aktif. Tekan START pada komponen di atas, lalu izinkan browser menggunakan mikrofon."
    )
else:
    status_place.info("Mikrofon aktif. Mengumpulkan audio sekitar…")
    window_samples = int(window_seconds * SAMPLE_RATE)
    update_samples = int(update_seconds * SAMPLE_RATE)
    rolling_audio = np.empty(0, dtype=np.float32)
    total_samples_received = 0
    samples_at_last_prediction = 0
    last_prediction: dict[str, object] | None = None
    alert_event_id = 0
    last_visual_update = 0.0
    last_progress_update = 0.0
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    receiver = ctx.audio_receiver

    if receiver is None:
        status_place.warning("Koneksi mikrofon sedang disiapkan. Jika tidak mulai, tekan STOP lalu START lagi.")
    else:
        try:
            while ctx.state.playing:
                try:
                    audio_frames = receiver.get_frames(timeout=1)
                except queue.Empty:
                    # Silence is valid audio; continue waiting for the next frame batch.
                    continue
                except Exception as exc:
                    status_place.warning(f"Koneksi audio terhenti: {exc}")
                    break

                if not audio_frames:
                    continue

                sample_parts: list[np.ndarray] = []
                for audio_frame in audio_frames:
                    try:
                        converted_frames = resampler.resample(audio_frame)
                    except Exception:
                        # Recover from rare timestamp/format discontinuities by retrying with a fresh resampler.
                        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
                        converted_frames = resampler.resample(audio_frame)
                    for converted in converted_frames:
                        values = converted.to_ndarray().reshape(-1).astype(np.float32)
                        sample_parts.append(values / 32768.0)

                if not sample_parts:
                    continue

                chunk = np.concatenate(sample_parts)
                if chunk.size == 0 or not np.isfinite(chunk).all():
                    continue

                total_samples_received += int(chunk.size)
                rolling_audio = np.concatenate((rolling_audio, chunk))
                if rolling_audio.size > window_samples:
                    rolling_audio = rolling_audio[-window_samples:]

                collected_seconds = min(rolling_audio.size / SAMPLE_RATE, float(window_seconds))
                now_monotonic = time.monotonic()
                # Refresh the actual-signal visual independently of the slower classifier cadence.
                if now_monotonic - last_visual_update >= 0.30:
                    recent_count = min(rolling_audio.size, int(0.30 * SAMPLE_RATE))
                    recent_rms = float(np.sqrt(np.mean(np.square(rolling_audio[-recent_count:].astype(np.float64))))) if recent_count else 0.0
                    state_text = "SUARA TERDETEKSI" if recent_rms >= visual_threshold else "HENING / DI BAWAH AMBANG"
                    render_waveform(
                        waveform_place, rolling_audio, SAMPLE_RATE, visual_threshold, state_text
                    )
                    last_visual_update = now_monotonic
                if now_monotonic - last_progress_update >= 0.30:
                    progress_place.progress(
                        min(1.0, collected_seconds / float(window_seconds)),
                        text=f"Mengumpulkan audio: {collected_seconds:.1f}/{window_seconds} detik",
                    )
                    last_progress_update = now_monotonic

                enough_audio = rolling_audio.size >= window_samples
                due_for_prediction = (
                    total_samples_received - samples_at_last_prediction >= update_samples
                )
                if not enough_audio or (last_prediction is not None and not due_for_prediction):
                    continue

                try:
                    previous_label = (
                        str(last_prediction["prediction"])
                        if last_prediction is not None else None
                    )
                    feature_vector = extract_feature_vector(rolling_audio, SAMPLE_RATE)
                    frame = pd.DataFrame([feature_vector], columns=features_order)
                    prediction = str(model.predict(frame)[0])
                    scores_arr = model.predict_proba(frame)[0]
                    scores = {
                        str(cls): float(score)
                        for cls, score in zip(model.classes_, scores_arr)
                    }
                    rms_now = float(np.sqrt(np.mean(np.square(rolling_audio, dtype=np.float64))))
                    last_prediction = {
                        "prediction": prediction,
                        "scores": scores,
                        "rms": rms_now,
                        "updated_at": time.strftime("%H:%M:%S"),
                    }
                    samples_at_last_prediction = total_samples_received
                except Exception as exc:
                    result_place.error(f"Audio live gagal dianalisis: {exc}")
                    continue

                pred_label = str(last_prediction["prediction"])
                if pred_label == "berisik":
                    result_place.error(f"Hasil sementara: **{DISPLAY.get(pred_label, pred_label)}**")
                else:
                    result_place.success(f"Hasil sementara: **{DISPLAY.get(pred_label, pred_label)}**")

                # Play the warning once per transition into the Berisik class,
                # not on every prediction update while the class remains Berisik.
                should_play_alert = (
                    play_voice_alert
                    and pred_label == "berisik"
                    and previous_label != "berisik"
                )
                if pred_label != "berisik":
                    alert_audio_place.empty()
                elif should_play_alert and ALERT_SOUND_BYTES is not None:
                    alert_event_id += 1
                    alert_audio_place.audio(
                        ALERT_SOUND_BYTES,
                        format="audio/mpeg",
                        autoplay=True,
                        alt=f"Peringatan suara Jangan berisik, peringatan ke-{alert_event_id}",
                    )

                score_map = last_prediction["scores"]
                metrics = detail_place.container()
                with metrics:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Durasi jendela", f"{window_seconds} dtk")
                    col2.metric("RMS sinyal", f"{float(last_prediction['rms']):.4f}")
                    col3.metric("Pembaruan", str(last_prediction["updated_at"]))
                    score_frame = pd.DataFrame(
                        {
                            "Kelas": [DISPLAY.get(k, k) for k in score_map],
                            "Skor model": [score_map[k] for k in score_map],
                        }
                    )
                    st.bar_chart(score_frame.set_index("Kelas"))
                    st.caption(
                        "Skor model bukan probabilitas terkalibrasi dan tidak sama dengan persentase tingkat kebisingan."
                    )

        except Exception as exc:
            status_place.error(f"Pengolahan mikrofon mengalami masalah: {exc}")
        finally:
            status_place.info("Sesi deteksi berakhir. Tekan START untuk memulai lagi.")
            render_waveform(
                waveform_place, np.zeros(int(0.28 * SAMPLE_RATE), dtype=np.float32),
                SAMPLE_RATE, visual_threshold, "MIKROFON BERHENTI",
            )