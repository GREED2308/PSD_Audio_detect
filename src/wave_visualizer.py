"""Live audio waveform/equalizer visualization for the Streamlit microphone app."""
from __future__ import annotations

import json

import numpy as np


def make_visual_data(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float,
    waveform_points: int = 640,
    envelope_bins: int = 96,
) -> tuple[list[float], list[float], float, bool]:
    """Summarize recent microphone audio for a live oscilloscope/equalizer visual.

    The graph is deliberately flat when the latest 300 ms RMS is under the
    configured threshold, suppressing low-level microphone noise during silence.
    This is only a visualization gate; classifier input is unchanged.
    """
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return [0.0] * waveform_points, [0.0] * envelope_bins, 0.0, True

    short_count = min(values.size, max(1, int(0.30 * sample_rate)))
    current = values[-short_count:].astype(np.float64, copy=False)
    current_rms = float(np.sqrt(np.mean(np.square(current)))) if current.size else 0.0
    is_silent = (not np.isfinite(current_rms)) or current_rms < threshold

    # Show the latest ~0.28 s as a waveform with a fixed display gain.
    wave_count = min(values.size, max(1, int(0.28 * sample_rate)))
    recent_wave = values[-wave_count:]
    if recent_wave.size > waveform_points:
        indices = np.linspace(0, recent_wave.size - 1, waveform_points).astype(np.int64)
        recent_wave = recent_wave[indices]
    else:
        recent_wave = np.pad(recent_wave, (waveform_points - recent_wave.size, 0))
    wave = np.clip(recent_wave / 0.25, -1.0, 1.0).astype(float)

    # RMS envelope over the latest ~1.8 s creates equalizer bars inspired by
    # the reference graphic, while remaining grounded in the real input signal.
    env_count = min(values.size, max(1, int(1.8 * sample_rate)))
    recent_env = values[-env_count:]
    segments = np.array_split(recent_env, envelope_bins)
    envelope = np.array(
        [
            float(np.sqrt(np.mean(np.square(seg.astype(np.float64))))) if seg.size else 0.0
            for seg in segments
        ],
        dtype=float,
    )
    envelope = np.clip((envelope - threshold) / 0.12, 0.0, 1.0)

    if is_silent:
        wave = np.zeros(waveform_points, dtype=float)
        envelope = np.zeros(envelope_bins, dtype=float)
    return wave.tolist(), envelope.tolist(), current_rms, is_silent


def waveform_html(
    wave: list[float],
    envelope: list[float],
    current_rms: float,
    threshold: float,
    is_silent: bool,
    state_text: str,
) -> str:
    """Build a self-contained canvas visualizer fed by actual microphone samples."""
    wave_json = json.dumps(wave, separators=(",", ":"))
    env_json = json.dumps(envelope, separators=(",", ":"))
    silent_js = "true" if is_silent else "false"
    state_json = json.dumps(state_text)
    dot_color = "#78979b" if is_silent else "#62e5ca"
    state_color = "#91abad" if is_silent else "#7bf0d7"
    glow_color = "transparent" if is_silent else "rgba(98,229,202,.55)"

    return f'''<!DOCTYPE html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; background: transparent; font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
.frame {{ width: 100%; border: 1px solid rgba(91, 151, 156, .34); border-radius: 16px; overflow: hidden; background: linear-gradient(135deg,#07191b 0%,#0a2529 55%,#0c2025 100%); box-shadow: 0 8px 26px rgba(5, 20, 24, .12); }}
.top {{ height: 42px; display:flex; align-items:center; justify-content:space-between; gap:12px; padding: 0 16px; color:#a8c5c9; font-size:11px; letter-spacing:.13em; text-transform:uppercase; }}
.brand {{ display:flex; align-items:center; gap:8px; font-weight:700; }}
.dot {{ width:7px; height:7px; border-radius:50%; background:{dot_color}; box-shadow: 0 0 10px {glow_color}; }}
.state {{ color:{state_color}; letter-spacing:.04em; font-size:10px; font-weight:700; text-align:right; }}
canvas {{ display:block; width:100%; height:148px; }}
.bottom {{ display:flex; justify-content:space-between; gap:8px; padding: 0 16px 13px; font-size:11px; color:#8baeb2; }}
.value {{ color:#c6f7ef; font-variant-numeric:tabular-nums; }}
</style></head><body>
<div class="frame">
  <div class="top"><div class="brand"><span class="dot"></span> AUDIO SIGNAL</div><div class="state" id="state"></div></div>
  <canvas id="signal" aria-label="Visualisasi gelombang suara langsung"></canvas>
  <div class="bottom"><span>LIVE MIC · 16 kHz</span><span>RMS <span class="value">{current_rms:.4f}</span> · Ambang {threshold:.3f}</span></div>
</div>
<script>
const wave = {wave_json};
const envelope = {env_json};
const silent = {silent_js};
document.getElementById('state').textContent = {state_json};
const canvas = document.getElementById('signal');
const ctx = canvas.getContext('2d');
function draw() {{
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const cssW = Math.max(320, canvas.clientWidth || 800);
  const cssH = 148;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);
  const w = cssW, h = cssH, mid = h * .55;
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle = 'rgba(115, 177, 181, .10)'; ctx.lineWidth = 1;
  for (let x=0; x<w; x+=Math.max(28,w/28)) {{ ctx.beginPath(); ctx.moveTo(x,8); ctx.lineTo(x,h-9); ctx.stroke(); }}
  for (let y=mid-48; y<=mid+48; y+=24) {{ ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }}
  // No sound above the threshold => a perfectly straight baseline.
  ctx.beginPath(); ctx.moveTo(0,mid); ctx.lineTo(w,mid);
  ctx.strokeStyle = silent ? 'rgba(120, 230, 216, .88)' : 'rgba(120, 230, 216, .25)'; ctx.lineWidth = silent ? 2 : 1;
  ctx.stroke();
  if (!silent) {{
    const gap = w / envelope.length;
    for (let i=0; i<envelope.length; i++) {{
      const strength = Math.max(0, Math.min(1,envelope[i]));
      const barH = 2 + strength * (h * .39);
      const x = i * gap + gap * .25;
      const barW = Math.max(1.5, gap * .52);
      const grad = ctx.createLinearGradient(0,mid-barH,0,mid+barH);
      grad.addColorStop(0,'#58c6f5'); grad.addColorStop(.5,'#8af3df'); grad.addColorStop(1,'#58c6f5');
      ctx.fillStyle = grad;
      ctx.globalAlpha = .38 + strength * .62;
      ctx.fillRect(x, mid-barH, barW, barH*2);
    }}
    ctx.globalAlpha = 1;
    ctx.beginPath();
    for (let i=0; i<wave.length; i++) {{
      const x = i / Math.max(1,wave.length-1) * w;
      const y = mid - Math.max(-1,Math.min(1,wave[i])) * (h * .36);
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }}
    ctx.lineWidth = 1.7; ctx.strokeStyle = '#b4fff0'; ctx.shadowColor = 'rgba(108,247,218,.72)'; ctx.shadowBlur = 5; ctx.stroke(); ctx.shadowBlur = 0;
  }}
}}
draw();
window.addEventListener('resize', draw);
</script></body></html>'''


def render_waveform(placeholder, audio: np.ndarray, sample_rate: int, threshold: float, state_text: str):
    # Lazy import keeps the signal-processing helper testable without starting Streamlit.
    import streamlit.components.v1 as components

    wave, envelope, rms, is_silent = make_visual_data(audio, sample_rate, threshold)
    with placeholder.container():
        components.html(
            waveform_html(wave, envelope, rms, threshold, is_silent, state_text),
            height=224,
            scrolling=False,
        )
    return rms, is_silent
