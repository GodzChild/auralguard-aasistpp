from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import hf_hub_download

from src.demo_professional import create_demo


PROJECT_ROOT = Path(__file__).resolve().parent
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "aasist"

AASIST_COMMIT = os.getenv(
    "AURALGUARD_AASIST_COMMIT",
    "a04c9863f63d44471dde8a6abcb3b082b07cd1d1",
)
AASIST_BASE_URL = f"https://raw.githubusercontent.com/clovaai/aasist/{AASIST_COMMIT}"

MODEL_REPO = os.getenv("AURALGUARD_MODEL_REPO", "GodzChild/auralguard-aasistpp-model")
CHECKPOINT_FILE = os.getenv("AURALGUARD_CHECKPOINT_FILE", "best.pt")
MODEL_REVISION = os.getenv("AURALGUARD_MODEL_REVISION", "main")
HF_TOKEN = os.getenv("HF_TOKEN")


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    with urllib.request.urlopen(url) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def ensure_aasist() -> tuple[Path, Path]:
    model_path = EXTERNAL_ROOT / "models" / "AASIST.py"
    config_path = EXTERNAL_ROOT / "config" / "AASIST.conf"
    license_path = EXTERNAL_ROOT / "LICENSE"

    _download(f"{AASIST_BASE_URL}/models/AASIST.py", model_path)
    _download(f"{AASIST_BASE_URL}/config/AASIST.conf", config_path)
    _download(f"{AASIST_BASE_URL}/LICENSE", license_path)

    return EXTERNAL_ROOT, config_path


def get_checkpoint() -> Path:
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=CHECKPOINT_FILE,
        repo_type="model",
        revision=MODEL_REVISION,
        token=HF_TOKEN,
    )
    return Path(path)


def build_args(checkpoint: Path, aasist_root: Path, aasist_config: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SimpleNamespace(
        checkpoint=str(checkpoint),
        aasist_root=str(aasist_root),
        aasist_config=str(aasist_config),
        sample_rate=16000,
        duration_sec=4.0,
        feature_dim=160,
        device=device,
        share=False,
    )


aasist_root, aasist_config = ensure_aasist()
checkpoint = get_checkpoint()
demo = create_demo(build_args(checkpoint, aasist_root, aasist_config))


if __name__ == "__main__":
    demo.launch()
