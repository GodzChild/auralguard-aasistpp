import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dfs = []
    for csv_path in args.inputs:
        df = pd.read_csv(csv_path)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    print(f"Saved: {args.out}")
    print(f"Rows: {len(merged)}")
    print(merged["dataset"].value_counts())
    print(merged["binary_label"].value_counts())


if __name__ == "__main__":
    main()
    