import argparse
from pathlib import Path
import pandas as pd
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--attack-type", required=True, choices=["tts_vc", "codec", "partial"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exts = {".wav", ".flac", ".mp3"}
    files = sorted([p for p in Path(args.audio_root).rglob("*") if p.suffix.lower() in exts])

    if not files:
        raise RuntimeError(f"No audio files found in {args.audio_root}")

    random.seed(args.seed)
    random.shuffle(files)

    n = len(files)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:]
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, split_files in splits.items():
        rows = []
        for f in split_files:
            rows.append({
                "file_path": f.resolve().as_posix(),
                "binary_label": 1,
                "attack_type": args.attack_type,
                "start_fake": 0.0,
                "end_fake": "full",
                "dataset": args.dataset_name,
                "split": split
            })

        df = pd.DataFrame(rows)
        out_path = out_dir / f"{args.dataset_name.lower()}_{split}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {out_path} rows={len(df)}")

if __name__ == "__main__":
    main()
    