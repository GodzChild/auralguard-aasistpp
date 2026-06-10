import argparse
import os
import pandas as pd
import torchaudio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    print(f"Checking CSV: {args.csv}")
    print(f"Rows: {len(df)}")

    bad = []

    limit = len(df)
    if args.max_files > 0:
        limit = min(limit, args.max_files)

    for i in range(limit):
        path = str(df.iloc[i]["file_path"])

        if not os.path.exists(path):
            print(f"[MISSING] row={i} path={path}")
            bad.append((i, path, "missing"))
            continue

        try:
            wav, sr = torchaudio.load(path)
        except Exception as e:
            print(f"[BAD AUDIO] row={i}")
            print(f"  path={path}")
            print(f"  error={repr(e)}")
            bad.append((i, path, repr(e)))
            continue

        if i % 500 == 0:
            print(f"Checked {i}/{limit}: OK")

    print()
    print(f"Finished. Bad files found: {len(bad)}")

    if bad:
        print("First bad files:")
        for item in bad[:20]:
            print(item)

if __name__ == "__main__":
    main()