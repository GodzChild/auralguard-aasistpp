
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
from tqdm import tqdm


def load_audio(path, target_sr=16000):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.numpy(), target_sr


def augment(x, kind):
    x = x.astype(np.float32)
    if kind == "noise":
        noise = np.random.normal(0, 0.005, size=x.shape).astype(np.float32)
        return np.clip(x + noise, -1, 1)
    if kind == "volume_low":
        return np.clip(x * 0.65, -1, 1)
    if kind == "volume_high":
        return np.clip(x * 1.25, -1, 1)
    if kind == "clip":
        return np.clip(x * 1.8, -0.8, 0.8)
    if kind == "dropout":
        y = x.copy()
        if len(y) > 1000:
            start = np.random.randint(0, max(1, len(y) - len(y)//10))
            y[start:start + len(y)//20] = 0
        return y
    return x


def main():
    p = argparse.ArgumentParser(description="Create simple offline audio augmentations and a matching metadata CSV.")
    p.add_argument("--csv", required=True, help="Input metadata CSV")
    p.add_argument("--out-audio-dir", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--target-sr", type=int, default=16000)
    p.add_argument("--augmentations", nargs="+", default=["noise", "volume_low", "volume_high", "clip", "dropout"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    df = pd.read_csv(args.csv, low_memory=False)
    if args.limit > 0:
        df = df.sample(n=min(len(df), args.limit), random_state=args.seed).reset_index(drop=True)

    out_dir = Path(args.out_audio_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
        x, sr = load_audio(row["file_path"], args.target_sr)
        for kind in args.augmentations:
            y = augment(x, kind)
            out_path = out_dir / f"aug_{i:06d}_{kind}.wav"
            sf.write(out_path, y, sr)

            new_row = row.to_dict()
            new_row["file_path"] = out_path.resolve().as_posix()
            new_row["dataset"] = str(row.get("dataset", "")) + "_Aug"
            rows.append(new_row)

    out_df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)
    print("Rows:", len(out_df))


if __name__ == "__main__":
    main()
