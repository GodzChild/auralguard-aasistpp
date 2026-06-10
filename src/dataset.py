from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from .labels import ATTACK_TO_ID, normalize_attack_type, make_explanation_label


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    duration_sec: float = 4.0
    random_crop: bool = False

    @property
    def num_samples(self) -> int:
        return int(self.sample_rate * self.duration_sec)


def _to_float(value: Any, default: float = -1.0) -> float:
    if pd.isna(value):
        return default
    value = str(value).strip().lower()
    if value == "full":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_audio_fixed(
    file_path: str | Path,
    sample_rate: int = 16000,
    num_samples: Optional[int] = None,
    random_crop: bool = False,
) -> torch.Tensor:
    """Load audio as mono waveform, resample, then pad/crop to num_samples.

    Returns:
        wav: Tensor with shape [num_samples] if num_samples is provided,
             otherwise [num_audio_samples].
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    wav, sr = torchaudio.load(str(path))

    # Convert channels x samples -> mono samples
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    else:
        wav = wav.flatten()

    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    wav = wav.float()

    if num_samples is None:
        return wav

    if wav.numel() < num_samples:
        return F.pad(wav, (0, num_samples - wav.numel()))

    if wav.numel() == num_samples:
        return wav

    if random_crop:
        max_start = wav.numel() - num_samples
        start = torch.randint(0, max_start + 1, (1,)).item()
    else:
        start = 0

    return wav[start : start + num_samples]


class AuralGuardDataset(Dataset):
    """Dataset for AuralGuard-AASIST++.

    Required CSV columns:
        file_path,binary_label,attack_type

    Optional columns:
        start_fake,end_fake,dataset,split
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        audio_config: Optional[AudioConfig] = None,
        root_dir: str | Path | None = None,
    ) -> None:
        self.metadata_csv = Path(metadata_csv)
        self.df = pd.read_csv(self.metadata_csv)
        self.audio_config = audio_config or AudioConfig()
        self.root_dir = Path(root_dir) if root_dir is not None else None

        required = {"file_path", "binary_label", "attack_type"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Metadata CSV is missing required columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        if self.root_dir is not None:
            return self.root_dir / path
        return path

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        file_path = self._resolve_path(row["file_path"])

        wav = load_audio_fixed(
            file_path=file_path,
            sample_rate=self.audio_config.sample_rate,
            num_samples=self.audio_config.num_samples,
            random_crop=self.audio_config.random_crop,
        )

        binary_label = int(row["binary_label"])
        attack_type = normalize_attack_type(row["attack_type"])
        attack_label = ATTACK_TO_ID[attack_type]
        explanation_label = make_explanation_label(binary_label, attack_type)

        item: Dict[str, Any] = {
            "wav": wav,
            "binary_label": torch.tensor(binary_label, dtype=torch.long),
            "attack_label": torch.tensor(attack_label, dtype=torch.long),
            "explanation_label": torch.tensor(explanation_label, dtype=torch.long),
            "file_path": str(file_path),
            "attack_type": attack_type,
            "dataset": str(row["dataset"]) if "dataset" in row else "unknown",
            "start_fake": _to_float(row["start_fake"]) if "start_fake" in row else -1.0,
            "end_fake": _to_float(row["end_fake"]) if "end_fake" in row else -1.0,
        }
        return item
