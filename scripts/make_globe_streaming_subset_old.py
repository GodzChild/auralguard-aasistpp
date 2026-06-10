from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio


def save_split(ds, split_name, out_audio_dir, out_csv, max_items):
    out_audio_dir = Path(out_audio_dir)
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    saved = 0

    print(f"Saving {split_name}: {max_items} samples")

    for item in ds:
        if saved >= max_items:
            break

        # Most HF audio datasets use an "audio" column.
        if "audio" not in item:
            print("Available columns:", item.keys())
            raise ValueError("Could not find an 'audio' column in this dataset.")

        audio = item["audio"]

        # audio should contain array + sampling_rate after cast_column(Audio())
        array = audio["array"]
        sr = audio["sampling_rate"]

        wav_path = out_audio_dir / f"globe_{split_name}_{saved:06d}.wav"
        sf.write(wav_path, array, sr)

        rows.append({
            "file_path": wav_path.resolve().as_posix(),
            "binary_label": 0,
            "attack_type": "bonafide",
            "start_fake": -1,
            "end_fake": -1,
            "dataset": "GLOBE",
            "split": split_name,
        })

        saved += 1

        if saved % 100 == 0:
            print(f"Saved {saved}/{max_items}")

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Saved CSV: {out_csv}")
    print(f"Rows: {len(df)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=1000)
    parser.add_argument("--max-val", type=int, default=200)
    parser.add_argument("--max-test", type=int, default=200)
    parser.add_argument("--out-audio-root", default="data/globe/audio")
    parser.add_argument("--out-dir", default="data/metadata")
    args = parser.parse_args()

    print("Loading GLOBE with streaming=True...")
    ds = load_dataset("MushanW/GLOBE", split="train", streaming=True)

    print("Enabling audio decoding...")
    ds = ds.cast_column("audio", Audio())

    # Make three small streamed subsets.
    train_ds = ds.take(args.max_train)
    remaining = ds.skip(args.max_train)

    val_ds = remaining.take(args.max_val)
    remaining = remaining.skip(args.max_val)

    test_ds = remaining.take(args.max_test)

    save_split(
        train_ds,
        "train",
        Path(args.out_audio_root) / "train",
        Path(args.out_dir) / "train_globe.csv",
        args.max_train,
    )

    save_split(
        val_ds,
        "val",
        Path(args.out_audio_root) / "val",
        Path(args.out_dir) / "val_globe.csv",
        args.max_val,
    )

    save_split(
        test_ds,
        "test",
        Path(args.out_audio_root) / "test",
        Path(args.out_dir) / "globe_test.csv",
        args.max_test,
    )


if __name__ == "__main__":
    main()