from datasets import load_dataset, Audio
from pathlib import Path
import soundfile as sf
import pandas as pd
import argparse
import re


def safe_name(x):
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_-]+", "_", x)
    return x[:80]


def save_rows(ds, split_name, out_audio_dir, out_csv, max_items=None):
    out_audio_dir = Path(out_audio_dir)
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    n = len(ds) if max_items is None else min(len(ds), max_items)
    rows = []

    print(f"Saving {split_name}: {n} samples")

    for i in range(n):
        item = ds[i]
        audio = item["audio"]

        accent = item.get("accent", item.get("speaker_id", "unknown"))
        speaker = item.get("speaker_id", item.get("speaker", f"spk{i}"))

        wav_path = out_audio_dir / f"{split_name}_{i:06d}_{safe_name(speaker)}_{safe_name(accent)}.wav"

        sf.write(wav_path, audio["array"], audio["sampling_rate"])

        rows.append({
            "file_path": wav_path.resolve().as_posix(),
            "binary_label": 0,
            "attack_type": "bonafide",
            "start_fake": -1,
            "end_fake": -1,
            "dataset": "EnglishDialects",
            "split": split_name,
        })

        if (i + 1) % 500 == 0:
            print(f"Saved {i + 1}/{n}")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved CSV: {out_csv}")
    print(f"Rows: {len(rows)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-audio-root", default="data/english_dialects/audio")
    parser.add_argument("--out-dir", default="data/metadata")
    parser.add_argument("--max-train", type=int, default=3000)
    parser.add_argument("--max-val", type=int, default=500)
    args = parser.parse_args()

    print("Loading English Dialects dataset...")
    ds = load_dataset("ylacombe/english_dialects")

    print("Available splits:", list(ds.keys()))

    # Most HF audio datasets have train split only.
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    full = ds[split_name].cast_column("audio", Audio())

    # Make simple split from the same dataset.
    train_ds = full.select(range(0, min(args.max_train, len(full))))
    val_start = min(args.max_train, len(full))
    val_end = min(val_start + args.max_val, len(full))
    val_ds = full.select(range(val_start, val_end))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_rows(
        train_ds,
        "train",
        Path(args.out_audio_root) / "train",
        out_dir / "train_english_dialects.csv",
        max_items=len(train_ds),
    )

    save_rows(
        val_ds,
        "val",
        Path(args.out_audio_root) / "val",
        out_dir / "val_english_dialects.csv",
        max_items=len(val_ds),
    )


if __name__ == "__main__":
    main()