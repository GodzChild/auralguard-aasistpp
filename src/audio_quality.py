
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torchaudio


def load_audio_mono(path: str, target_sr: int = 16000):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def audio_quality_report(path: str, target_sr: int = 16000) -> Dict:
    wav, sr = load_audio_mono(path, target_sr)
    x = wav.detach().cpu().numpy().astype(float)

    if len(x) == 0:
        return {"warnings": ["Audio appears empty."], "metrics": {"duration_sec": 0.0}}

    duration = len(x) / sr
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    peak = float(np.max(np.abs(x)) + 1e-12)
    silence_ratio = float(np.mean(np.abs(x) < 1e-4))
    clipping_ratio = float(np.mean(np.abs(x) > 0.98))

    warnings: List[str] = []
    if duration < 1.0:
        warnings.append("Audio is very short; prediction may be unreliable.")
    if rms < 0.005:
        warnings.append("Audio volume is very low.")
    if clipping_ratio > 0.01:
        warnings.append("Audio may be clipped or distorted.")
    if silence_ratio > 0.70:
        warnings.append("Audio contains a lot of silence.")
    if peak < 0.02:
        warnings.append("Audio peak level is very low.")

    return {
        "warnings": warnings,
        "metrics": {
            "sample_rate": sr,
            "duration_sec": duration,
            "rms": rms,
            "peak": peak,
            "silence_ratio": silence_ratio,
            "clipping_ratio": clipping_ratio,
        },
    }
