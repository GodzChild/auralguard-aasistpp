
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


AUDIO_EXTENSIONS = {
    ".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"
}


def find_audio_files(audio_root: Path):
    files = []
    for p in audio_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(p)
    return sorted(files)


def make_rows(files, split_name: str):
    rows = []
    for p in files:
        rows.append({
            "file_path": p.resolve().as_posix(),
            "binary_label": 0,
            "attack_type": "bonafide",
            "start_fake": -1,
            "end_fake": -1,
            "dataset": "GLOBE",
            "split": split_name,
            "source_file": p.name,
            "source_folder": p.parent.name,
        })
    return rows


def save_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(f"Rows: {len(df)}")
    if len(df):
        print(df[["dataset", "split", "binary_label", "attack_type"]].value_counts())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create small real/bonafide GLOBE metadata CSVs for AuralGuard. "
            "This does not train anything; it only creates train/val/test CSV files."
        )
    )
    parser.add_argument(
        "--audio-root",
        required=True,
        help="Folder containing GLOBE audio files. The script searches recursively."
    )
    parser.add_argument(
        "--out-dir",
        default="data/metadata",
        help="Where to save train_globe.csv, val_globe.csv, globe_test.csv"
    )
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-val", type=int, default=300)
    parser.add_argument("--max-test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    audio_root = Path(args.audio_root)
    if not audio_root.exists():
        raise FileNotFoundError(f"Audio root does not exist: {audio_root}")

    files = find_audio_files(audio_root)
    print(f"Found audio files: {len(files)}")

    if len(files) == 0:
        raise RuntimeError(
            "No audio files found. Check the folder path or supported extensions: "
            + ", ".join(sorted(AUDIO_EXTENSIONS))
        )

    rng = random.Random(args.seed)
    rng.shuffle(files)

    n_train = min(args.max_train, len(files))
    n_val = min(args.max_val, max(0, len(files) - n_train))
    n_test = min(args.max_test, max(0, len(files) - n_train - n_val))

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:n_train + n_val + n_test]

    out_dir = Path(args.out_dir)
    save_csv(make_rows(train_files, "train"), out_dir / "train_globe.csv")
    save_csv(make_rows(val_files, "val"), out_dir / "val_globe.csv")
    save_csv(make_rows(test_files, "test"), out_dir / "globe_test.csv")

    print("\nDone.")
    print("Recommended next step:")
    print('python scripts\\check_metadata.py --csv "data\\metadata\\train_globe.csv"')
    print('python scripts\\check_metadata.py --csv "data\\metadata\\val_globe.csv"')
    print('python scripts\\check_metadata.py --csv "data\\metadata\\globe_test.csv"')


if __name__ == "__main__":
    main()
