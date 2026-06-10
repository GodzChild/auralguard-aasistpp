from datasets import load_dataset
from pathlib import Path
import soundfile as sf
import pandas as pd
import argparse


def save_split(ds, split_name, out_audio_dir, out_csv, max_items=None):
    out_audio_dir = Path(out_audio_dir)
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n = len(ds) if max_items is None else min(len(ds), max_items)

    for i in range(n):
        item = ds[i]
        audio = item["audio"]
        speaker = item.get("speaker", f"unknown_{i}")
        accent = item.get("accent", "unknown").replace(" ", "_").replace("/", "_")

        wav_path = out_audio_dir / f"{split_name}_{i:06d}_{speaker}_{accent}.wav"
        sf.write(wav_path, audio["array"], audio["sampling_rate"])

        rows.append({
            "file_path": wav_path.resolve().as_posix(),
            "binary_label": 0,
            "attack_type": "bonafide",
            "start_fake": -1,
            "end_fake": -1,
            "dataset": "EdAcc",
            "split": split_name
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {out_csv} rows={len(rows)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-audio-root", default="data/edacc/audio")
    parser.add_argument("--out-dir", default="data/metadata")
    parser.add_argument("--max-train", type=int, default=3000)
    parser.add_argument("--max-val", type=int, default=500)
    args = parser.parse_args()

    print("Downloading/loading EdAcc...")
    edacc = load_dataset("edinburghcstr/edacc")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_split(
        edacc["validation"],
        "train",
        Path(args.out_audio_root) / "train",
        out_dir / "edacc_train.csv",
        max_items=args.max_train
    )

    save_split(
        edacc["test"],
        "val",
        Path(args.out_audio_root) / "val",
        out_dir / "edacc_val.csv",
        max_items=args.max_val
    )


if __name__ == "__main__":
    main()
    