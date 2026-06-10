from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_asvspoof_protocol(protocol_file: Path, audio_root: Path, split: str) -> pd.DataFrame:
    """Create metadata from an ASVspoof 2019 LA protocol file.

    Protocol lines usually look like:
        speaker_id file_id system_id - bonafide/spoof

    This script assumes audio files are under audio_root and uses .flac paths.
    Adjust `suffix` if your files use a different extension.
    """
    rows = []
    for line in protocol_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        file_id = parts[1]
        label_text = parts[-1].lower()
        is_fake = 0 if label_text == "bonafide" else 1

        rows.append({
            "file_path": str(audio_root / f"{file_id}.flac"),
            "binary_label": is_fake,
            "attack_type": "bonafide" if is_fake == 0 else "tts_vc",
            "start_fake": -1 if is_fake == 0 else 0.0,
            "end_fake": -1 if is_fake == 0 else "full",
            "dataset": "ASVspoof2019_LA",
            "split": split,
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="Build AuralGuard metadata from ASVspoof 2019 LA protocol files.")
    p.add_argument("--protocol", required=True)
    p.add_argument("--audio-root", required=True)
    p.add_argument("--split", required=True, choices=["train", "val", "dev", "test", "eval"])
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()

    df = parse_asvspoof_protocol(Path(args.protocol), Path(args.audio_root), args.split)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved {len(df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
