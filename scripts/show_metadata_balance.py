
from __future__ import annotations

import argparse
import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Show metadata dataset and label balance.")
    p.add_argument("--csv", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    df["binary_label"] = df["binary_label"].astype(int)
    print("CSV:", args.csv)
    print("Rows:", len(df))
    print("\nBinary counts:")
    print(df["binary_label"].value_counts().sort_index())
    print("\nBinary percentages:")
    print((df["binary_label"].value_counts(normalize=True).sort_index() * 100).round(2))
    print("\nDataset counts:")
    print(df["dataset"].value_counts())
    print("\nDataset x binary_label:")
    print(pd.crosstab(df["dataset"], df["binary_label"]))


if __name__ == "__main__":
    main()
