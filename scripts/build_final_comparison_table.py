
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def summarize_false_alarm(path, label):
    df = pd.read_csv(path)
    return {
        "model_or_eval": label,
        "rows": len(df),
        "false_fake_rate_065_percent": round(df["is_false_fake_original_065"].mean() * 100, 2) if len(df) else None,
        "false_fake_rate_085_percent": round(df["is_false_fake_strict_085"].mean() * 100, 2) if len(df) else None,
        "review_rate_percent": round(df["review_required"].mean() * 100, 2) if "review_required" in df.columns and len(df) else None,
    }


def main():
    p = argparse.ArgumentParser(description="Build final comparison table from false-alarm CSVs.")
    p.add_argument("--false-alarm-csvs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args()

    if len(args.false_alarm_csvs) != len(args.labels):
        raise ValueError("Number of CSVs must match number of labels.")

    rows = [summarize_false_alarm(path, label) for path, label in zip(args.false_alarm_csvs, args.labels)]
    out = pd.DataFrame(rows)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    md = out.to_markdown(index=False)
    Path(args.out_md).write_text(md, encoding="utf-8")

    print(md)
    print("Saved:", args.out_csv)
    print("Saved:", args.out_md)


if __name__ == "__main__":
    main()
