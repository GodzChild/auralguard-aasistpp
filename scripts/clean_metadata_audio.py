import argparse
import os
import pandas as pd
import torchaudio

def audio_ok(path):
    if not os.path.exists(path):
        return False

    try:
        torchaudio.load(path)
        return True
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    good_rows = []
    bad_rows = []

    for i, row in df.iterrows():
        path = str(row["file_path"])

        if audio_ok(path):
            good_rows.append(row)
        else:
            bad_rows.append((i, path))
            print(f"Skipping bad file row={i}: {path}")

    clean_df = pd.DataFrame(good_rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    clean_df.to_csv(args.out_csv, index=False)

    print()
    print(f"Original rows: {len(df)}")
    print(f"Good rows: {len(clean_df)}")
    print(f"Bad rows removed: {len(bad_rows)}")
    print(f"Saved clean CSV to: {args.out_csv}")

if __name__ == "__main__":
    main()