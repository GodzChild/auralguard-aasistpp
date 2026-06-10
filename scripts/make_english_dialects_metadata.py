from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

from datasets import load_dataset, Audio


CONFIGS = [
    "irish_male",
    "midlands_female",
    "midlands_male",
    "northern_female",
    "northern_male",
    "scottish_female",
    "scottish_male",
    "southern_female",
    "southern_male",
    "welsh_female",
    "welsh_male",
]


def safe_name(text):
    text = str(text)
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


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


def get_text(item):
    for key in ["text", "sentence", "transcript", "transcription"]:
        if key in item and item[key] is not None:
            return str(item[key])
    return ""


def save_rows(rows, out_csv):
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
        "accent",
        "gender",
        "config",
        "text",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {out_csv}")


def process_config(config, max_train_per_config, max_val_per_config, audio_root):
    print(f"\nLoading config: {config}")

    ds_dict = load_dataset("ylacombe/english_dialects", config)

    for split_name in ds_dict.keys():
        ds_dict[split_name] = ds_dict[split_name].cast_column("audio", Audio(decode=False))

    available_splits = list(ds_dict.keys())
    print(f"Available splits for {config}: {available_splits}")

    if "train" in ds_dict:
        train_ds = ds_dict["train"]
    else:
        first_split = available_splits[0]
        train_ds = ds_dict[first_split]

    if "validation" in ds_dict:
        val_ds = ds_dict["validation"]
    elif "test" in ds_dict:
        val_ds = ds_dict["test"]
    else:
        val_ds = train_ds

    accent = config.replace("_male", "").replace("_female", "")
    gender = "male" if config.endswith("_male") else "female"

    train_rows = []
    val_rows = []

    train_n = min(len(train_ds), max_train_per_config)
    val_n = min(len(val_ds), max_val_per_config)

    print(f"Using {train_n} train samples and {val_n} val samples from {config}")

    for i in range(train_n):
        item = train_ds[i]
        audio_info = item["audio"]

        file_name = f"{config}_train_{i:06d}.wav"
        out_audio_path = Path(audio_root) / "train" / config / file_name

        ok = export_audio_file(audio_info, out_audio_path)
        if not ok:
            print(f"Skipping train sample {i} from {config}")
            continue

        train_rows.append({
            "file_path": str(out_audio_path).replace("/", "\\"),
            "binary_label": 0,
            "attack_type": "bonafide",
            "start_fake": -1,
            "end_fake": -1,
            "dataset": "EnglishDialects",
            "split": "train",
            "accent": accent,
            "gender": gender,
            "config": config,
            "text": get_text(item),
        })

    for i in range(val_n):
        item = val_ds[i]
        audio_info = item["audio"]

        file_name = f"{config}_val_{i:06d}.wav"
        out_audio_path = Path(audio_root) / "val" / config / file_name

        ok = export_audio_file(audio_info, out_audio_path)
        if not ok:
            print(f"Skipping val sample {i} from {config}")
            continue

        val_rows.append({
            "file_path": str(out_audio_path).replace("/", "\\"),
            "binary_label": 0,
            "attack_type": "bonafide",
            "start_fake": -1,
            "end_fake": -1,
            "dataset": "EnglishDialects",
            "split": "val",
            "accent": accent,
            "gender": gender,
            "config": config,
            "text": get_text(item),
        })

    return train_rows, val_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=3000)
    parser.add_argument("--max-val", type=int, default=500)
    parser.add_argument("--out-train", default="data\\metadata\\train_english_dialects.csv")
    parser.add_argument("--out-val", default="data\\metadata\\val_english_dialects.csv")
    parser.add_argument("--audio-root", default="data\\english_dialects_audio")
    args = parser.parse_args()

    max_train_per_config = max(1, args.max_train // len(CONFIGS))
    max_val_per_config = max(1, args.max_val // len(CONFIGS))

    print("Loading all English Dialects configs...")
    print(f"Configs: {CONFIGS}")
    print(f"Train per config: {max_train_per_config}")
    print(f"Val per config: {max_val_per_config}")

    all_train_rows = []
    all_val_rows = []

    for config in CONFIGS:
        train_rows, val_rows = process_config(
            config=config,
            max_train_per_config=max_train_per_config,
            max_val_per_config=max_val_per_config,
            audio_root=args.audio_root,
        )

        all_train_rows.extend(train_rows)
        all_val_rows.extend(val_rows)

    save_rows(all_train_rows, args.out_train)
    save_rows(all_val_rows, args.out_val)

    print("\nDone.")
    print(f"Train CSV: {args.out_train}")
    print(f"Val CSV: {args.out_val}")


if __name__ == "__main__":
    main()