from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        value = float(x)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def _quality_label(value: float, low: float, high: float, low_name: str = "low", mid_name: str = "normal", high_name: str = "high") -> str:
    if value < low:
        return low_name
    if value > high:
        return high_name
    return mid_name


def prosody_diagnostics(audio_path: str, target_sr: int = 16000) -> Dict[str, Any]:
    """Compute lightweight prosody / tonality diagnostics for an audio file.

    This module is intentionally diagnostic. It does not replace AASIST and it
    does not decide whether an audio file is fake. It extracts voice-pattern
    clues that can support the explanation layer, such as pitch variation,
    energy variation, pauses, and tonal stability.
    """
    if librosa is None:
        return {
            "available": False,
            "summary": "Prosody diagnostics unavailable because librosa could not be imported.",
            "metrics": {},
            "interpretation": [],
            "warnings": ["Install librosa to enable prosody diagnostics."],
        }

    try:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    except Exception as exc:
        return {
            "available": False,
            "summary": f"Prosody diagnostics could not load the audio: {exc}",
            "metrics": {},
            "interpretation": [],
            "warnings": [str(exc)],
        }

    if y is None or len(y) == 0:
        return {
            "available": False,
            "summary": "Prosody diagnostics unavailable because the audio is empty.",
            "metrics": {},
            "interpretation": [],
            "warnings": ["Empty audio file."],
        }

    y = np.asarray(y, dtype=np.float32)
    duration_sec = len(y) / float(sr)

    # Normalize only for analysis stability. This does not change the model input.
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0:
        y_norm = y / peak
    else:
        y_norm = y

    hop_length = 512
    frame_length = 2048

    # Energy and pause pattern
    rms = librosa.feature.rms(y=y_norm, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-8), ref=np.max)
    active = rms_db > -35.0
    silence_ratio = float(1.0 - np.mean(active)) if len(active) else 1.0
    energy_mean = float(np.mean(rms)) if len(rms) else 0.0
    energy_std = float(np.std(rms)) if len(rms) else 0.0
    energy_variation = float(energy_std / (energy_mean + 1e-8))

    # Count longer inactive regions as pause-like gaps.
    frame_duration = hop_length / float(sr)
    pause_count = 0
    current_pause = 0
    min_pause_frames = max(1, int(0.25 / frame_duration))
    for is_active in active:
        if not bool(is_active):
            current_pause += 1
        else:
            if current_pause >= min_pause_frames:
                pause_count += 1
            current_pause = 0
    if current_pause >= min_pause_frames:
        pause_count += 1

    # Pitch / F0 pattern. pyin may fail on noisy audio, so keep it safe.
    pitch_mean_hz = 0.0
    pitch_std_hz = 0.0
    pitch_std_semitones = 0.0
    pitch_range_semitones = 0.0
    median_pitch_step_semitones = 0.0
    voiced_fraction = 0.0
    pitch_available = False

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y_norm,
            fmin=50,
            fmax=500,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        if f0 is not None:
            voiced = f0[np.isfinite(f0)]
            voiced_fraction = float(len(voiced) / max(1, len(f0)))
            if len(voiced) >= 5:
                pitch_available = True
                pitch_mean_hz = float(np.mean(voiced))
                pitch_std_hz = float(np.std(voiced))
                median_f0 = float(np.median(voiced))
                semitone_values = 12.0 * np.log2(np.maximum(voiced, 1e-6) / max(median_f0, 1e-6))
                pitch_std_semitones = float(np.std(semitone_values))
                pitch_range_semitones = float(np.percentile(semitone_values, 95) - np.percentile(semitone_values, 5))
                if len(semitone_values) >= 2:
                    median_pitch_step_semitones = float(np.median(np.abs(np.diff(semitone_values))))
    except Exception:
        pitch_available = False

    # Extra acoustic texture clues. These are not direct fake proof, only support diagnostics.
    try:
        spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y_norm)))
    except Exception:
        spectral_flatness = 0.0

    try:
        zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y_norm)))
    except Exception:
        zero_crossing_rate = 0.0

    interpretation: List[str] = []
    warnings: List[str] = []

    if duration_sec < 2.0:
        warnings.append("Audio is very short, so prosody/tonality diagnostics may be unreliable.")

    if not pitch_available:
        warnings.append("Pitch tracking was weak or unavailable; this can happen with noisy, quiet, or non-speech audio.")
        interpretation.append("Pitch movement could not be reliably measured.")
    else:
        if pitch_range_semitones < 3.0:
            interpretation.append("Pitch variation is low, which can indicate a flat or monotone voice pattern.")
        elif pitch_range_semitones > 14.0:
            interpretation.append("Pitch variation is high, which may reflect expressive speech or unstable tonal movement.")
        else:
            interpretation.append("Pitch variation looks within a normal speaking range for this short clip.")

        if median_pitch_step_semitones > 1.2:
            interpretation.append("Some frame-to-frame pitch movement is abrupt, which may be worth reviewing with the model score.")
        else:
            interpretation.append("Pitch movement is relatively smooth between nearby frames.")

    if energy_variation < 0.35:
        interpretation.append("Energy variation is low, so the voice loudness is very stable and smooth.")
    elif energy_variation > 1.20:
        interpretation.append("Energy variation is high, which may reflect strong loudness changes, noise, or recording variation.")
    else:
        interpretation.append("Energy variation looks reasonably natural for speech.")

    if silence_ratio > 0.55:
        interpretation.append("A large part of the clip is silence or very low energy, so the prediction should be interpreted carefully.")
    elif pause_count == 0 and duration_sec > 3.0:
        interpretation.append("Few clear pause-like gaps were detected; the speech may be continuous or tightly edited.")
    else:
        interpretation.append(f"Pause-like gaps detected: {pause_count}.")

    # Human-readable summary
    if pitch_available:
        pitch_label = _quality_label(pitch_range_semitones, 3.0, 14.0, "low", "normal", "high")
    else:
        pitch_label = "unavailable"
    energy_label = _quality_label(energy_variation, 0.35, 1.20, "low", "normal", "high")
    pause_label = _quality_label(silence_ratio, 0.10, 0.55, "few/low", "normal", "high")

    summary = (
        f"Prosody check: pitch variation is {pitch_label}, energy variation is {energy_label}, "
        f"and silence/pause ratio is {pause_label}. These are supporting voice-pattern clues, not final proof."
    )

    metrics = {
        "duration_sec": round(_safe_float(duration_sec), 3),
        "pitch_available": bool(pitch_available),
        "pitch_mean_hz": round(_safe_float(pitch_mean_hz), 3),
        "pitch_std_hz": round(_safe_float(pitch_std_hz), 3),
        "pitch_std_semitones": round(_safe_float(pitch_std_semitones), 3),
        "pitch_range_semitones": round(_safe_float(pitch_range_semitones), 3),
        "median_pitch_step_semitones": round(_safe_float(median_pitch_step_semitones), 3),
        "voiced_fraction": round(_safe_float(voiced_fraction), 3),
        "energy_variation": round(_safe_float(energy_variation), 3),
        "silence_ratio": round(_safe_float(silence_ratio), 3),
        "pause_like_gap_count": int(pause_count),
        "spectral_flatness": round(_safe_float(spectral_flatness), 5),
        "zero_crossing_rate": round(_safe_float(zero_crossing_rate), 5),
    }

    return {
        "available": True,
        "summary": summary,
        "metrics": metrics,
        "interpretation": interpretation,
        "warnings": warnings,
        "note": "Prosody diagnostics support interpretation only; the final fake/real score still comes from the AASIST-based model.",
    }
