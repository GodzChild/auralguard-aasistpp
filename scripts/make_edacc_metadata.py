import argparse
import os
import csv
import shutil
from pathlib import Path

from datasets import load_dataset, Audio


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
    """
    Save/copy audio without decoding.

    Hugging Face Audio(decode=False) usually gives:
    {
        "bytes": ...,
        "path": ...
    }
    """

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


def save_split(ds, split_name, out_csv, audio_dir, max_items):
    rows = []
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    n = len(ds)
    if max_items is not None and max_items > 0:
        n = min(n, max_items)

    print(f"Saving {split_name}: {n} samples")

    skipped = 0

    for i in range(n):
        item = ds[i]

        audio_info = item["audio"]
        speaker = item.get("speaker", f"{split_name}_{i}")
        accent = item.get("accent", "unknown")
        text = item.get("text", "")

        original_path = audio_info.get("path", None)

        if original_path:
            base = Path(original_path).stem
        else:
            base = f"{split_name}_{i:06d}"

        file_name = safe_name(base) + ".wav"
        out_audio_path = audio_dir / file_name

        ok = export_audio_file(audio_info, out_audio_path)

        if not ok:
            skipped += 1
            print(f"Skipping sample {i}: could not save audio")
            continue

        rows.append({
            "file_path": str(out_audio_path).replace("/", "\\"),
            "binary_label": 0,
            "attack_type": "bonafide",
            "start_fake": -1,
            "end_fake": -1,
            "dataset": "EdAcc",
            "split": split_name,
            "speaker": speaker,
            "accent": accent,
            "text": text,
        })

        if (i + 1) % 500 == 0:
            print(f"Saved {i + 1}/{n}")

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file_path",
            "binary_label",
            "attack_type",
            "start_fake",
            "end_fake",
            "dataset",
            "split",
            "speaker",
            "accent",
            "text",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {out_csv}")
    print(f"Rows saved: {len(rows)}")
    print(f"Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=3000)
    parser.add_argument("--max-val", type=int, default=500)
    parser.add_argument("--out-train", default="data\\metadata\\train_edacc.csv")
    parser.add_argument("--out-val", default="data\\metadata\\val_edacc.csv")
    parser.add_argument("--audio-root", default="data\\edacc_audio")
    args = parser.parse_args()

    print("Downloading/loading EdAcc without audio decoding...")
    edacc = load_dataset("edinburghcstr/edacc")

    print("Disabling Hugging Face audio decoding...")
    for split in edacc.keys():
        edacc[split] = edacc[split].cast_column("audio", Audio(decode=False))

    print("Available splits:", list(edacc.keys()))

    # EdAcc normally has validation and test splits.
    # We use validation as our train subset and test as our val subset.
    train_source = "validation"
    val_source = "test"

    save_split(
        edacc[train_source],
        "train",
        args.out_train,
        os.path.join(args.audio_root, "train"),
        args.max_train,
    )

    save_split(
        edacc[val_source],
        "val",
        args.out_val,
        os.path.join(args.audio_root, "val"),
        args.max_val,
    )

    print("Done.")


if __name__ == "__main__":
    main()