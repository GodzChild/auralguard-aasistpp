
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from tqdm import tqdm

from src.infer import load_model, run_auralguard
from src.audio_quality import audio_quality_report
from src.decision_gate import make_safe_report


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate false fake alarms on real/bonafide datasets.")
    p.add_argument("--csv", required=True, help="Metadata CSV containing real/bonafide rows")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--aasist-root", default="external/aasist")
    p.add_argument("--aasist-config", default="external/aasist/config/AASIST.conf")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--duration-sec", type=float, default=4.0)
    p.add_argument("--feature-dim", type=int, default=160)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out-csv", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv, low_memory=False)
    df = df[df["binary_label"].astype(int) == 0].reset_index(drop=True)

    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    device = torch.device(args.device)
    model_args = SimpleNamespace(**vars(args))
    model = load_model(model_args, device)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="False-alarm eval"):
        path = row["file_path"]
        report = run_auralguard(
            path,
            model,
            sample_rate=args.sample_rate,
            duration_sec=args.duration_sec,
            device=device,
        )
        quality = audio_quality_report(path, target_sr=args.sample_rate)
        safe = make_safe_report(report, quality_warnings=quality["warnings"], ood_warning=False)

        fake_prob = float(report.get("fake_probability", 0.0))
        rows.append({
            "file_path": path,
            "dataset": row.get("dataset", ""),
            "original_decision": report.get("decision", ""),
            "safe_decision": safe.get("decision", ""),
            "fake_probability": fake_prob,
            "attack_type": report.get("attack_type", ""),
            "is_false_fake_original_065": int(fake_prob >= 0.65),
            "is_false_fake_strict_085": int(fake_prob >= 0.85),
            "review_required": int(safe.get("review_required", False)),
            "quality_warnings": " | ".join(quality["warnings"]),
        })

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print("\nSaved:", args.out_csv)
    print("Rows:", len(out))
    if len(out):
        print("False fake rate, threshold 0.65:", round(out["is_false_fake_original_065"].mean() * 100, 2), "%")
        print("False fake rate, threshold 0.85:", round(out["is_false_fake_strict_085"].mean() * 100, 2), "%")
        print("\nSafe decision counts:")
        print(out["safe_decision"].value_counts())


if __name__ == "__main__":
    main()
