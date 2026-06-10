from __future__ import annotations

from typing import Dict, Any

import torch
from torch import nn


class AuralGuardAASISTPP(nn.Module):
    """Multi-task wrapper around AASIST.

    Expected AASIST backbone behavior:
        features, original_binary_logits = backbone(wav)

    The official AASIST implementation returns `last_hidden, output`.
    This wrapper uses `last_hidden` and learns its own task heads.
    """

    def __init__(
        self,
        aasist_backbone: nn.Module,
        feature_dim: int = 160,
        num_attack_types: int = 4,
        num_explanation_types: int = 4,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = aasist_backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.binary_head = nn.Linear(feature_dim, 2)
        self.attack_head = nn.Linear(feature_dim, num_attack_types)
        self.explanation_head = nn.Linear(feature_dim, num_explanation_types)

    def extract_features(self, wav: torch.Tensor, freq_aug: bool = False) -> torch.Tensor:
        """Run AASIST and return the hidden feature vector."""
        result = self.backbone(wav, Freq_aug=freq_aug)

        if isinstance(result, tuple) and len(result) >= 1:
            features = result[0]
        elif isinstance(result, dict) and "features" in result:
            features = result["features"]
        else:
            raise RuntimeError(
                "AASIST backbone did not return hidden features. "
                "Expected `(last_hidden, output)` from the official AASIST Model.forward. "
                "Patch the backbone to return the feature vector before the final output layer."
            )

        if features.ndim != 2:
            raise RuntimeError(f"Expected features with shape [batch, dim], got {tuple(features.shape)}")

        return features

    def forward(self, wav: torch.Tensor, freq_aug: bool = False) -> Dict[str, torch.Tensor]:
        features = self.extract_features(wav, freq_aug=freq_aug)
        return {
            "features": features,
            "binary_logits": self.binary_head(features),
            "attack_logits": self.attack_head(features),
            "explanation_logits": self.explanation_head(features),
        }


def compute_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    binary_label: torch.Tensor,
    attack_label: torch.Tensor,
    explanation_label: torch.Tensor,
    binary_weight: float = 1.0,
    attack_weight: float = 0.5,
    explanation_weight: float = 0.2,
) -> Dict[str, torch.Tensor]:
    ce = nn.CrossEntropyLoss()

    binary_loss = ce(outputs["binary_logits"], binary_label)
    attack_loss = ce(outputs["attack_logits"], attack_label)
    explanation_loss = ce(outputs["explanation_logits"], explanation_label)

    total = (
        binary_weight * binary_loss
        + attack_weight * attack_loss
        + explanation_weight * explanation_loss
    )

    return {
        "loss": total,
        "binary_loss": binary_loss.detach(),
        "attack_loss": attack_loss.detach(),
        "explanation_loss": explanation_loss.detach(),
    }
