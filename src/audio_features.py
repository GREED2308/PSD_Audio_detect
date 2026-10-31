"""16-feature audio preprocessing for the PSD live classifier.

Feature order matches dataset/dataset_fitur_suara_original.csv:
RMS mean, Spectral Centroid mean, ZCR mean, then MFCC 1..13 means.
The dataset CSV values were checked against Librosa using its default 22,050 Hz
sample rate; the live microphone is resampled to the same rate.
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 22_050
N_MFCC = 13

FEATURE_NAMES = [
    "rms_mean",
    "centroid_mean",
    "zcr_mean",
] + [f"mfcc_{i}_mean" for i in range(1, N_MFCC + 1)]

assert len(FEATURE_NAMES) == 16


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 at sr Hz."""
    path = str(path)
    first_error: Exception | None = None
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        y = np.asarray(y, dtype=np.float32)
        if y.size == 0 or not np.isfinite(y).all():
            raise ValueError("Audio kosong atau mengandung nilai non-finite")
        return y, sr
    except Exception as exc:
        first_error = exc

    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", path,
        "-f", "wav", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sr), "pipe:1"
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, timeout=120)
        y, actual_sr = sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=False)
        if y.size == 0:
            raise ValueError("FFmpeg menghasilkan audio kosong")
        if y.ndim == 2:
            y = y.mean(axis=1)
        if actual_sr != sr:
            y = librosa.resample(y.astype(np.float32), orig_sr=actual_sr, target_sr=sr)
        y = np.asarray(y, dtype=np.float32)
        if not np.isfinite(y).all():
            raise ValueError("Audio mengandung nilai non-finite")
        return y, sr
    except Exception as fallback_error:
        raise RuntimeError(
            f"Gagal membaca audio {Path(path).name}. "
            f"Librosa: {first_error}; FFmpeg: {fallback_error}"
        ) from fallback_error


def extract_feature_vector(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return exactly 16 mean features in the same order as the training CSV."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        raise ValueError("Audio kosong")
    if not np.isfinite(y).all():
        raise ValueError("Audio mengandung nilai non-finite")
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    values = [
        float(np.mean(rms)),
        float(np.mean(centroid)),
        float(np.mean(zcr)),
    ]
    values.extend(np.mean(mfcc, axis=1).astype(float).tolist())

    vector = np.asarray(values, dtype=np.float64)
    if vector.size != len(FEATURE_NAMES):
        raise RuntimeError(f"Diharapkan 16 fitur, diperoleh {vector.size}")
    if not np.isfinite(vector).all():
        raise ValueError("Ekstraksi menghasilkan fitur non-finite")
    return vector


def extract_features(path: str | Path) -> np.ndarray:
    y, sr = load_audio(path, SAMPLE_RATE)
    return extract_feature_vector(y, sr)
