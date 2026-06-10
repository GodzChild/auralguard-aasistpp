import argparse
import random
from pathlib import Path
import pandas as pd


def collect_audio_files(chunks_root):
    chunks_root = Path(chunks_root)
    audio_exts = {".wav", ".flac", ".mp3"}

    files = []
    for p in chunks_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in audio_exts:
            files.append(p)

    return sorted(files)


def group_by_original_audio(files):
    """
    Groups chunks by their parent folder.
    This avoids putting chunks from the same long interview
    into both train and validation/test.
    """
    groups = {}

    for f in files:
        group_name = str(f.parent)
        groups.setdefault(group_name, []).append(f)

    return list(groups.values())


def split_groups(groups, train_ratio, val_ratio, seed):
    random.seed(seed)
    random.shuffle(groups)

    n = len(groups)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_groups = groups[:n_train]
    val_groups = groups[n_train:n_train + n_val]
    test_groups = groups[n_train + n_val:]

    return train_groups, val_groups, test_groups


def flatten(groups):
    out = []
    for g in groups:
        out.extend(g)
    return out


def make_df(files, dataset_name, split):
    rows = []

    for f in files:
        rows.append({
            "file_path": f.resolve().as_posix(),
            "binary_label": 0,
            "attack_type": "bonafide",
            "start_fake": -1,
            "end_fake": -1,
            "dataset": dataset_name,
            "split": split
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-root", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    files = collect_audio_files(args.chunks_root)

    if not files:
        raise RuntimeError(f"No audio files found in {args.chunks_root}")

    groups = group_by_original_audio(files)

    train_groups, val_groups, test_groups = split_groups(
        groups,
        args.train_ratio,
        args.val_ratio,
        args.seed
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = make_df(flatten(train_groups), args.dataset_name, "train")
    val_df = make_df(flatten(val_groups), args.dataset_name, "val")
    test_df = make_df(flatten(test_groups), args.dataset_name, "test")

    train_df.to_csv(out_dir / f"{args.dataset_name.lower()}_train.csv", index=False)
    val_df.to_csv(out_dir / f"{args.dataset_name.lower()}_val.csv", index=False)
    test_df.to_csv(out_dir / f"{args.dataset_name.lower()}_test.csv", index=False)

    print(f"Dataset: {args.dataset_name}")
    print(f"Total audio chunks: {len(files)}")
    print(f"Original audio groups: {len(groups)}")
    print(f"Train chunks: {len(train_df)}")
    print(f"Val chunks: {len(val_df)}")
    print(f"Test chunks: {len(test_df)}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
    