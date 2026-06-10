from __future__ import annotations

from typing import List, Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def sliding_window_localization(
    model,
    wav: torch.Tensor,
    sample_rate: int = 16000,
    window_sec: float = 2.0,
    hop_sec: float = 1.0,
    threshold: float = 0.75,
    device: str | torch.device = "cpu",
) -> List[Dict[str, float]]:
    """Localize suspicious regions with overlapping windows.

    Args:
        model: AuralGuardAASISTPP.
        wav: Tensor [samples] or [1, samples].
    """
    model.eval()
    device = torch.device(device)

    if wav.ndim == 2:
        wav = wav.squeeze(0)

    wav = wav.to(device).float()

    window = int(window_sec * sample_rate)
    hop = int(hop_sec * sample_rate)

    if wav.numel() < window:
        wav = F.pad(wav, (0, window - wav.numel()))

    suspicious = []
    for start in range(0, wav.numel() - window + 1, hop):
        end = start + window
        chunk = wav[start:end].unsqueeze(0)

        outputs = model(chunk)
        fake_score = torch.softmax(outputs["binary_logits"], dim=-1)[0, 1].item()

        if fake_score >= threshold:
            suspicious.append({
                "start": round(start / sample_rate, 3),
                "end": round(end / sample_rate, 3),
                "fake_score": round(fake_score, 4),
            })

    return merge_overlapping_segments(suspicious)


def merge_overlapping_segments(segments: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Merge overlapping suspicious windows for cleaner output."""
    if not segments:
        return []

    segments = sorted(segments, key=lambda x: x["start"])
    merged = [segments[0].copy()]

    for seg in segments[1:]:
        last = merged[-1]
        if seg["start"] <= last["end"]:
            last["end"] = max(last["end"], seg["end"])
            last["fake_score"] = round(max(last["fake_score"], seg["fake_score"]), 4)
        else:
            merged.append(seg.copy())

    return merged
