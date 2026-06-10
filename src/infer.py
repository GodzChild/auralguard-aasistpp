from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .aasist_loader import build_aasist_backbone
from .dataset import load_audio_fixed
from .explain import build_report
from .labels import ID_TO_ATTACK
from .model_aasistpp import AuralGuardAASISTPP
from .sliding_localization import sliding_window_localization


def load_model(args, device):
    backbone = build_aasist_backbone(args.aasist_root, args.aasist_config, device=device)
    model = AuralGuardAASISTPP(backbone, feature_dim=args.feature_dim).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt, strict=True)
    model.eval()
    return model


@torch.no_grad()
def run_auralguard(
    audio_path: str,
    model,
    sample_rate: int = 16000,
    duration_sec: float = 4.0,
    localization_window_sec: float = 2.0,
    localization_hop_sec: float = 1.0,
    suspicious_threshold: float = 0.75,
    device: str | torch.device = "cpu",
):
    device = torch.device(device)

    fixed_wav = load_audio_fixed(
        audio_path, sample_rate=sample_rate, num_samples=int(sample_rate * duration_sec), random_crop=False
    ).unsqueeze(0).to(device)

    outputs = model(fixed_wav)
    fake_probability = torch.softmax(outputs["binary_logits"], dim=-1)[0, 1].item()
    attack_id = torch.argmax(outputs["attack_logits"], dim=-1)[0].item()
    explanation_id = torch.argmax(outputs["explanation_logits"], dim=-1)[0].item()
    attack_type = ID_TO_ATTACK[attack_id]

    # For timestamps, use the full audio instead of the fixed 4s crop.
    full_wav = load_audio_fixed(audio_path, sample_rate=sample_rate, num_samples=None, random_crop=False)
    segments = sliding_window_localization(
        model=model,
        wav=full_wav,
        sample_rate=sample_rate,
        window_sec=localization_window_sec,
        hop_sec=localization_hop_sec,
        threshold=suspicious_threshold,
        device=device,
    )

    return build_report(
        fake_probability,
        attack_type,
        explanation_id,
        segments,
        suspicious_window_threshold=suspicious_threshold,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Run AuralGuard-AASIST++ inference on one audio file")
    p.add_argument("--audio", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--aasist-root", default="external/aasist")
    p.add_argument("--aasist-config", default="external/aasist/config/AASIST.conf")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--duration-sec", type=float, default=4.0)
    p.add_argument("--feature-dim", type=int, default=160)
    p.add_argument("--window-sec", type=float, default=2.0)
    p.add_argument("--hop-sec", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.75)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args, device)
    report = run_auralguard(
        args.audio,
        model,
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        localization_window_sec=args.window_sec,
        localization_hop_sec=args.hop_sec,
        suspicious_threshold=args.threshold,
        device=device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
