
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"file_path", "binary_label", "attack_type", "start_fake", "end_fake", "dataset", "split"}


def print_counts(df, title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print("Rows:", len(df))
    print("\nBinary label counts:")
    print(df["binary_label"].value_counts().sort_index())
    print("\nDataset counts:")
    print(df["dataset"].value_counts())
    print("\nDataset x label counts:")
    print(pd.crosstab(df["dataset"], df["binary_label"]))


def main():
    p = argparse.ArgumentParser(description="Balance metadata by keeping all real rows and downsampling fake rows.")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fake-multiplier", type=float, default=1.0, help="1.0 means fake count roughly equals real count")
    p.add_argument("--cap-wavefake", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    df["binary_label"] = df["binary_label"].astype(int)
    print_counts(df, "Before balancing")

    if args.cap_wavefake > 0:
        mask = df["dataset"].astype(str).str.lower() == "wavefake"
        wave = df[mask]
        other = df[~mask]
        if len(wave) > args.cap_wavefake:
            wave = wave.sample(n=args.cap_wavefake, random_state=args.seed)
        df = pd.concat([other, wave], ignore_index=True)
        print_counts(df, f"After WaveFake cap: {args.cap_wavefake}")

    real = df[df["binary_label"] == 0]
    fake = df[df["binary_label"] == 1]
    max_fake = int(len(real) * args.fake_multiplier)
    fake = fake.sample(n=min(len(fake), max_fake), random_state=args.seed)

    out = pd.concat([real, fake], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print_counts(out, "After balancing")
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
