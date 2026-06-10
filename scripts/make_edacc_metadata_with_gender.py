from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

from datasets import load_dataset, Audio


def export_audio_file(audio_info, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    audio_bytes = audio_info.get("bytes", None)
    audio_path = audio_info.get("path", None)

    if audio_bytes is not None:
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        return True

    if audio_path is not None and os.path.exists(audio_path):
        shutil.copy2(audio_path, out_path)
        return True

    return False


def get_value(item, keys, default="unknown"):
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return default


def save_split(ds, split_name, out_audio_dir, out_csv, max_items):
    out_audio_dir = Path(out_audio_dir)
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n = min(len(ds), max_items)

    print(f"Saving {split_name}: {n} samples")

    for i in range(n):
        item = ds[i]

        audio_info = item["audio"]

        speaker = get_value(item, ["speaker", "speaker_id", "client_id"], f"speaker_{i}")
        accent = get_value(item, ["accent", "accent_group", "dialect"], "unknown")
        gender = get_value(item, ["gender", "sex"], "unknown")
        text = get_value(item, ["text", "sentence", "transcript", "transcription"], "")

        out_path = out_audio_dir / f"EDACC_{split_name}_{i:06d}.wav"

        ok = export_audio_file(audio_info, out_path)
        if not ok:
            print(f"Skipping {split_name} sample {i}: could not save audio")
            continue

        rows.append({
            "file_path": out_path.resolve().as_posix(),
            "binary_label": 0,
            "attack_type": "bonafide",
            "start_fake": -1,
            "end_fake": -1,
            "dataset": "EdAcc",
            "split": split_name,
            "speaker": speaker,
            "accent": accent,
            "gender": gender,
            "text": text,
        })

        if (i + 1) % 500 == 0:
            print(f"Saved {i + 1}/{n}")

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file_path",
        "binary_label",
        "attack_type",
        "start_fake",
        "end_fake",
        "dataset",
        "split",
        "speaker",
        "accent",
        "gender",
        "text",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", out_csv)
    print("Rows:", len(rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-train", type=int, default=3000)
    p.add_argument("--max-val", type=int, default=500)
    p.add_argument("--out-dir", default="data\\metadata")
    p.add_argument("--out-audio-root", default="data\\edacc_audio_gender")
    args = p.parse_args()

    print("Loading EdAcc without audio decoding...")
    ds = load_dataset("edinburghcstr/edacc")

    print("Available splits:", list(ds.keys()))

    # Important: this prevents TorchCodec requirement
    for split in ds.keys():
        ds[split] = ds[split].cast_column("audio", Audio(decode=False))

    train_ds = ds["validation"]
    val_ds = ds["test"]

    save_split(
        train_ds,
        "train",
        Path(args.out_audio_root) / "train",
        Path(args.out_dir) / "train_edacc_gender.csv",
        args.max_train,
    )

    save_split(
        val_ds,
        "val",
        Path(args.out_audio_root) / "val",
        Path(args.out_dir) / "val_edacc_gender.csv",
        args.max_val,
    )

    print("Done.")


if __name__ == "__main__":
    main()