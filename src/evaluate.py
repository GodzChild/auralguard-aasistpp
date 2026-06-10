from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .aasist_loader import build_aasist_backbone
from .dataset import AuralGuardDataset, AudioConfig
from .labels import ID_TO_ATTACK, ID_TO_EXPLANATION
from .metrics import classification_report_dict, make_confusion_matrix
from .model_aasistpp import AuralGuardAASISTPP


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AuralGuard-AASIST++")
    p.add_argument("--csv", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--aasist-root", default="external/aasist")
    p.add_argument("--aasist-config", default="external/aasist/config/AASIST.conf")
    p.add_argument("--out-csv", default="results/predictions.csv")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--duration-sec", type=float, default=4.0)
    p.add_argument("--feature-dim", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ds = AuralGuardDataset(args.csv, audio_config=AudioConfig(args.sample_rate, args.duration_sec))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    backbone = build_aasist_backbone(args.aasist_root, args.aasist_config, device=device)
    model = AuralGuardAASISTPP(backbone, feature_dim=args.feature_dim).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate"):
            wav = batch["wav"].to(device)
            outputs = model(wav)

            bin_probs = torch.softmax(outputs["binary_logits"], dim=-1)
            fake_scores = bin_probs[:, 1]
            bin_pred = torch.argmax(outputs["binary_logits"], dim=-1)
            attack_pred = torch.argmax(outputs["attack_logits"], dim=-1)
            exp_pred = torch.argmax(outputs["explanation_logits"], dim=-1)

            for i in range(wav.shape[0]):
                rows.append({
                    "file_path": batch["file_path"][i],
                    "true_binary": int(batch["binary_label"][i]),
                    "pred_binary": int(bin_pred[i].cpu()),
                    "fake_probability": float(fake_scores[i].cpu()),
                    "true_attack": int(batch["attack_label"][i]),
                    "pred_attack": int(attack_pred[i].cpu()),
                    "pred_attack_name": ID_TO_ATTACK[int(attack_pred[i].cpu())],
                    "true_explanation": int(batch["explanation_label"][i]),
                    "pred_explanation": int(exp_pred[i].cpu()),
                    "pred_explanation_name": ID_TO_EXPLANATION[int(exp_pred[i].cpu())],
                })

    pred_df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_csv, index=False)

    metrics = classification_report_dict(
        pred_df["true_binary"], pred_df["pred_binary"], pred_df["fake_probability"],
        pred_df["true_attack"], pred_df["pred_attack"],
        pred_df["true_explanation"], pred_df["pred_explanation"],
    )
    metrics["attack_confusion_matrix"] = make_confusion_matrix(pred_df["true_attack"], pred_df["pred_attack"])

    metrics_path = out_csv.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions to {out_csv}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
