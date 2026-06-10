from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .aasist_loader import build_aasist_backbone
from .dataset import load_audio_fixed
from .localization_metrics import best_center_error_seconds, best_segment_iou, localization_hit
from .model_aasistpp import AuralGuardAASISTPP
from .sliding_localization import sliding_window_localization


def _parse_fake_time(value, duration_hint: float | None = None) -> float:
    if pd.isna(value):
        return -1.0
    text = str(value).strip().lower()
    if text == "full":
        return float(duration_hint) if duration_hint is not None else -1.0
    try:
        return float(text)
    except ValueError:
        return -1.0


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate sliding-window partial-fake localization")
    p.add_argument("--csv", required=True, help="CSV with file_path,start_fake,end_fake columns")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--aasist-root", default="external/aasist")
    p.add_argument("--aasist-config", default="external/aasist/config/AASIST.conf")
    p.add_argument("--out-csv", default="results/localization_predictions.csv")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--feature-dim", type=int, default=160)
    p.add_argument("--window-sec", type=float, default=2.0)
    p.add_argument("--hop-sec", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.75)
    p.add_argument("--min-iou", type=float, default=0.10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    df = pd.read_csv(args.csv)

    required = {"file_path", "start_fake", "end_fake"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Localization evaluation needs columns: {sorted(missing)}")

    backbone = build_aasist_backbone(args.aasist_root, args.aasist_config, device=device)
    model = AuralGuardAASISTPP(backbone, feature_dim=args.feature_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt, strict=True)
    model.eval()

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="localization"):
        wav = load_audio_fixed(row["file_path"], sample_rate=args.sample_rate, num_samples=None, random_crop=False)
        duration = wav.numel() / args.sample_rate
        true_start = _parse_fake_time(row["start_fake"], duration_hint=duration)
        true_end = _parse_fake_time(row["end_fake"], duration_hint=duration)

        segments = sliding_window_localization(
            model,
            wav,
            sample_rate=args.sample_rate,
            window_sec=args.window_sec,
            hop_sec=args.hop_sec,
            threshold=args.threshold,
            device=device,
        )
        iou = best_segment_iou(segments, true_start, true_end)
        center_error = best_center_error_seconds(segments, true_start, true_end)
        hit = localization_hit(segments, true_start, true_end, min_iou=args.min_iou)

        rows.append({
            "file_path": row["file_path"],
            "true_start": true_start,
            "true_end": true_end,
            "predicted_segments_json": json.dumps(segments),
            "best_iou": iou,
            "center_error_sec": center_error,
            "hit_at_min_iou": hit,
        })

    out_df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    valid = out_df[out_df["best_iou"].notna()]
    summary = {
        "num_examples": int(len(out_df)),
        "num_valid_partial_examples": int(len(valid)),
        "mean_best_iou": float(valid["best_iou"].mean()) if len(valid) else float("nan"),
        "hit_rate_at_min_iou": float(valid["hit_at_min_iou"].mean()) if len(valid) else float("nan"),
        "median_center_error_sec": float(valid["center_error_sec"].replace(float("inf"), pd.NA).median()) if len(valid) else float("nan"),
    }
    metrics_path = out_csv.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved localization predictions to {out_csv}")
    print(f"Saved localization metrics to {metrics_path}")


if __name__ == "__main__":
    main()
