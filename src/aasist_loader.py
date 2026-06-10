from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch


def load_aasist_config(config_path: str | Path) -> Dict[str, Any]:
    """Load AASIST JSON-style config and return the model_config section if present."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Official AASIST configs usually contain a top-level "model_config".
    return cfg.get("model_config", cfg)


def import_aasist_model_class(aasist_root: str | Path):
    """Dynamically import external/aasist/models/AASIST.py."""
    aasist_root = Path(aasist_root)
    model_file = aasist_root / "models" / "AASIST.py"

    if not model_file.exists():
        raise FileNotFoundError(
            f"Could not find AASIST.py at {model_file}. "
            "Clone the official repo with: git clone https://github.com/clovaai/aasist.git external/aasist"
        )

    # AASIST.py may import modules relative to the repo root.
    sys.path.insert(0, str(aasist_root))

    spec = importlib.util.spec_from_file_location("external_aasist_model", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {model_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "Model"):
        raise AttributeError(f"{model_file} does not define class Model")

    return module.Model


def build_aasist_backbone(
    aasist_root: str | Path,
    aasist_config: str | Path,
    checkpoint: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Build the official AASIST backbone and optionally load weights."""
    Model = import_aasist_model_class(aasist_root)
    model_config = load_aasist_config(aasist_config)
    backbone = Model(model_config).to(device)

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        backbone.load_state_dict(state, strict=False)

    return backbone
