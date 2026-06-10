from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Check that metadata CSV has required columns and existing audio paths.")
    p.add_argument("--csv", required=True)
    p.add_argument("--root-dir", default=None)
    p.add_argument("--max-missing", type=int, default=20)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    required = {"file_path", "binary_label", "attack_type"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    root = Path(args.root_dir) if args.root_dir else None
    missing_files = []
    for path_value in df["file_path"]:
        path = Path(path_value)
        if not path.is_absolute() and root is not None:
            path = root / path
        if not path.exists():
            missing_files.append(str(path))
            if len(missing_files) >= args.max_missing:
                break

    print(f"Rows: {len(df)}")
    print(f"Binary label counts:\n{df['binary_label'].value_counts(dropna=False)}")
    print(f"Attack type counts:\n{df['attack_type'].value_counts(dropna=False)}")

    if missing_files:
        print("\nMissing file examples:")
        for m in missing_files:
            print(" -", m)
        raise SystemExit(1)

    print("\nMetadata looks OK.")


if __name__ == "__main__":
    main()
