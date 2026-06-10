
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from src.infer import load_model, run_auralguard
from src.audio_quality import audio_quality_report
from src.decision_gate import make_safe_report
from src.explain_plus import beginner_friendly_explanation_plus


def parse_args():
    p = argparse.ArgumentParser(description="Compare multiple checkpoints on the same audio clip.")
    p.add_argument("--audio", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--aasist-root", default="external/aasist")
    p.add_argument("--aasist-config", default="external/aasist/config/AASIST.conf")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--duration-sec", type=float, default=4.0)
    p.add_argument("--feature-dim", type=int, default=160)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-csv", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    if len(args.names) != len(args.checkpoints):
        raise ValueError("--names and --checkpoints must have the same length")

    device = torch.device(args.device)
    quality = audio_quality_report(args.audio, args.sample_rate)

    rows = []
    for name, ckpt in zip(args.names, args.checkpoints):
        model_args = SimpleNamespace(**vars(args))
        model_args.checkpoint = ckpt
        model = load_model(model_args, device)

        report = run_auralguard(
            args.audio,
            model,
            sample_rate=args.sample_rate,
            duration_sec=args.duration_sec,
            device=device,
        )
        safe = make_safe_report(report, quality_warnings=quality["warnings"], ood_warning=False)
        gate = safe.get("decision_gate", {})
        beginner = beginner_friendly_explanation_plus(
            fake_prob=float(report.get("fake_probability", 0.0)),
            attack_type=report.get("attack_type", "unknown"),
            suspicious_segments=report.get("suspicious_segments", []),
            safe_decision=safe.get("decision", ""),
            warnings=gate.get("warnings", []),
        )

        rows.append({
            "model": name,
            "checkpoint": ckpt,
            "original_decision": report.get("original_decision", report.get("decision", "")),
            "safe_decision": safe.get("decision", ""),
            "fake_probability": float(report.get("fake_probability", 0.0)),
            "attack_type": report.get("attack_type", ""),
            "review_required": safe.get("review_required", False),
            "beginner_explanation": beginner,
            "quality_warnings": " | ".join(quality["warnings"]),
        })

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(out[["model", "safe_decision", "fake_probability", "attack_type", "review_required"]])
    print("\nSaved:", args.out_csv)


if __name__ == "__main__":
    main()
