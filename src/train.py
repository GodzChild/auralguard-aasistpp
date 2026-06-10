from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .aasist_loader import build_aasist_backbone
from .dataset import AuralGuardDataset, AudioConfig
from .model_aasistpp import AuralGuardAASISTPP, compute_multitask_loss
from .metrics import classification_report_dict


def parse_args():
    p = argparse.ArgumentParser(description="Train AuralGuard-AASIST++")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--aasist-root", default="external/aasist")
    p.add_argument("--aasist-config", default="external/aasist/config/AASIST.conf")
    p.add_argument("--aasist-checkpoint", default=None, help="Optional pretrained AASIST checkpoint")
    p.add_argument("--out-dir", default="results/run1")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--duration-sec", type=float, default=4.0)
    p.add_argument("--feature-dim", type=int, default=160)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--freeze-backbone", action="store_true")
    return p.parse_args()


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)

    total_loss = 0.0
    y_true, y_pred, fake_scores = [], [], []
    attack_true, attack_pred = [], []
    exp_true, exp_pred = [], []

    loop = tqdm(loader, desc="train" if train else "val", leave=False)
    for batch in loop:
        batch = move_batch_to_device(batch, device)

        with torch.set_grad_enabled(train):
            outputs = model(batch["wav"], freq_aug=train)
            losses = compute_multitask_loss(
                outputs,
                batch["binary_label"],
                batch["attack_label"],
                batch["explanation_label"],
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                optimizer.step()

        probs = torch.softmax(outputs["binary_logits"], dim=-1)
        fake_prob = probs[:, 1]
        pred = torch.argmax(outputs["binary_logits"], dim=-1)
        attack_p = torch.argmax(outputs["attack_logits"], dim=-1)
        exp_p = torch.argmax(outputs["explanation_logits"], dim=-1)

        bs = batch["wav"].shape[0]
        total_loss += losses["loss"].item() * bs

        y_true.extend(batch["binary_label"].detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
        fake_scores.extend(fake_prob.detach().cpu().tolist())
        attack_true.extend(batch["attack_label"].detach().cpu().tolist())
        attack_pred.extend(attack_p.detach().cpu().tolist())
        exp_true.extend(batch["explanation_label"].detach().cpu().tolist())
        exp_pred.extend(exp_p.detach().cpu().tolist())

        loop.set_postfix(loss=losses["loss"].item())

    metrics = classification_report_dict(
        y_true, y_pred, fake_scores, attack_true, attack_pred, exp_true, exp_pred
    )
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    audio_train = AudioConfig(args.sample_rate, args.duration_sec, random_crop=True)
    audio_eval = AudioConfig(args.sample_rate, args.duration_sec, random_crop=False)

    train_ds = AuralGuardDataset(args.train_csv, audio_config=audio_train)
    val_ds = AuralGuardDataset(args.val_csv, audio_config=audio_eval)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    backbone = build_aasist_backbone(
        args.aasist_root, args.aasist_config, checkpoint=args.aasist_checkpoint, device=device
    )
    model = AuralGuardAASISTPP(
        backbone,
        feature_dim=args.feature_dim,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_eer = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)

        print("train:", json.dumps(train_metrics, indent=2))
        print("val:", json.dumps(val_metrics, indent=2))

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "args": vars(args),
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, out_dir / "last.pt")

        val_eer = val_metrics.get("eer", float("inf"))
        if val_eer < best_eer:
            best_eer = val_eer
            torch.save(ckpt, out_dir / "best.pt")
            print(f"Saved new best checkpoint with EER={best_eer:.4f}")

        with (out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"Done. Best checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
