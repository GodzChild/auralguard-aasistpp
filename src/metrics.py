from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def compute_eer(y_true: np.ndarray, fake_scores: np.ndarray) -> float:
    """Equal Error Rate for binary labels where 1 means fake/spoof."""
    y_true = np.asarray(y_true).astype(int)
    fake_scores = np.asarray(fake_scores).astype(float)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(y_true, fake_scores, pos_label=1)
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except Exception:
        fnr = 1.0 - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)

    return float(eer)


def classification_report_dict(
    binary_true,
    binary_pred,
    fake_scores,
    attack_true=None,
    attack_pred=None,
    explanation_true=None,
    explanation_pred=None,
) -> Dict[str, float]:
    binary_true = np.asarray(binary_true)
    binary_pred = np.asarray(binary_pred)
    fake_scores = np.asarray(fake_scores)

    metrics = {
        "binary_accuracy": float(accuracy_score(binary_true, binary_pred)),
        "binary_f1": float(f1_score(binary_true, binary_pred, zero_division=0)),
        "eer": compute_eer(binary_true, fake_scores),
    }

    if len(np.unique(binary_true)) > 1:
        metrics["auc"] = float(roc_auc_score(binary_true, fake_scores))
    else:
        metrics["auc"] = float("nan")

    if attack_true is not None and attack_pred is not None:
        metrics["attack_accuracy"] = float(accuracy_score(attack_true, attack_pred))
        metrics["attack_macro_f1"] = float(f1_score(attack_true, attack_pred, average="macro", zero_division=0))

    if explanation_true is not None and explanation_pred is not None:
        metrics["explanation_accuracy"] = float(accuracy_score(explanation_true, explanation_pred))
        metrics["explanation_macro_f1"] = float(
            f1_score(explanation_true, explanation_pred, average="macro", zero_division=0)
        )

    return metrics


def make_confusion_matrix(y_true, y_pred) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred).astype(int).tolist()
