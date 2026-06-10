from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Split one metadata file into train/val/test CSV files using the split column.")
    p.add_argument("--metadata", required=True)
    p.add_argument("--out-dir", default="data/metadata")
    args = p.parse_args()

    df = pd.read_csv(args.metadata)
    if "split" not in df.columns:
        raise ValueError("Input metadata must contain a 'split' column.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "dev", "test", "eval"]:
        part = df[df["split"].astype(str).str.lower() == split]
        if len(part) == 0:
            continue
        name = "val" if split == "dev" else "test" if split == "eval" else split
        out = out_dir / f"{name}.csv"
        part.to_csv(out, index=False)
        print(f"Saved {len(part)} rows to {out}")


if __name__ == "__main__":
    main()
