from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

from datasets import load_dataset, Audio
from tqdm import tqdm


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


def save_split(iterator, split_name, max_items, out_audio_dir):
    rows = []
    out_audio_dir = Path(out_audio_dir)
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving {split_name}: {max_items} samples")

    count = 0
    skipped = 0

    for item in tqdm(iterator, desc=f"Saving {split_name}"):
        if count >= max_items:
            break

        if "audio" not in item:
            skipped += 1
            continue

        audio_info = item["audio"]

        file_name = f"GLOBE_{split_name}_{count:06d}.wav"
        out_path = out_audio_dir / file_name

        ok = export_audio_file(audio_info, out_path)

        if not ok:
            skipped += 1
            continue

        text = get_value(item, ["text", "sentence", "transcript", "transcription"], "")
        speaker = get_value(item, ["speaker", "speaker_id", "client_id"], "unknown")
        accent = get_value(item, ["accent", "country", "region", "dialect"], "unknown")
        gender = get_value(item, ["gender", "sex"], "unknown")

        rows.append({
            "file_path": out_path.resolve().as_posix(),
            "binary_label": 0,
            "attack_type": "bonafide",
            "start_fake": -1,
            "end_fake": -1,
            "dataset": "GLOBE",
            "split": split_name,
            "speaker": speaker,
            "accent": accent,
            "gender": gender,
            "text": text,
        })

        count += 1

    print(f"{split_name} saved rows: {len(rows)}")
    print(f"{split_name} skipped rows: {skipped}")

    return rows


def write_csv(rows, out_csv):
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
    p.add_argument("--max-train", type=int, default=1000)
    p.add_argument("--max-val", type=int, default=200)
    p.add_argument("--max-test", type=int, default=200)
    p.add_argument("--out-audio-root", default="data\\globe_audio")
    p.add_argument("--out-dir", default="data\\metadata")
    args = p.parse_args()

    print("Loading GLOBE with streaming=True...")
    ds = load_dataset("MushanW/GLOBE", split="train", streaming=True)

    print("Disabling audio decoding to avoid TorchCodec...")
    ds = ds.cast_column("audio", Audio(decode=False))

    total_needed = args.max_train + args.max_val + args.max_test

    all_items = iter(ds.take(total_needed))

    train_rows = save_split(
        all_items,
        "train",
        args.max_train,
        Path(args.out_audio_root) / "train",
    )

    val_rows = save_split(
        all_items,
        "val",
        args.max_val,
        Path(args.out_audio_root) / "val",
    )

    test_rows = save_split(
        all_items,
        "test",
        args.max_test,
        Path(args.out_audio_root) / "test",
    )

    out_dir = Path(args.out_dir)

    write_csv(train_rows, out_dir / "globe_train.csv")
    write_csv(val_rows, out_dir / "globe_val.csv")
    write_csv(test_rows, out_dir / "globe_test.csv")

    print("Done.")


if __name__ == "__main__":
    main()