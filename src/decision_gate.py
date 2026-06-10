
from __future__ import annotations

from typing import Any, Dict, List, Optional


def safe_decision_gate(
    fake_probability: float,
    attack_type: str = "unknown",
    suspicious_segments: Optional[List[Dict[str, Any]]] = None,
    quality_warnings: Optional[List[str]] = None,
    ood_warning: bool = False,
    real_threshold: float = 0.35,
    fake_threshold: float = 0.85,
    window_threshold: float = 0.75,
) -> Dict[str, Any]:
    suspicious_segments = suspicious_segments or []
    quality_warnings = quality_warnings or []

    max_window_score = 0.0
    if suspicious_segments:
        max_window_score = max(float(x.get("fake_score", 0.0)) for x in suspicious_segments)

    evidence_consistent = (
        fake_probability >= fake_threshold
        and attack_type not in {"bonafide", "unknown", ""}
        and (not suspicious_segments or max_window_score >= window_threshold)
    )

    warnings = []
    if quality_warnings:
        warnings.append("Audio quality may affect reliability.")
    if ood_warning:
        warnings.append("Audio may be outside the model training domain.")

    if fake_probability < real_threshold and attack_type == "bonafide":
        decision = "likely real"
        review_required = False
    elif fake_probability < real_threshold:
        decision = "likely real"
        review_required = False
        warnings.append("Fake probability is low, but the attack-type head is not fully aligned.")
    elif fake_probability < fake_threshold:
        decision = "suspicious / human review"
        review_required = True
        warnings.append("Fake probability is in the uncertain middle range.")
    elif ood_warning or quality_warnings:
        decision = "suspicious / human review"
        review_required = True
        warnings.append("High fake score, but reliability warnings are present.")
    elif evidence_consistent:
        decision = "likely fake"
        review_required = False
    else:
        decision = "suspicious / human review"
        review_required = True
        warnings.append("High fake score, but evidence is not fully consistent.")

    return {
        "safe_decision": decision,
        "review_required": review_required,
        "warnings": warnings,
        "thresholds": {
            "real_threshold": real_threshold,
            "fake_threshold": fake_threshold,
            "window_threshold": window_threshold,
        },
        "evidence_consistent": evidence_consistent,
        "max_window_fake_score": max_window_score,
    }


def make_safe_report(report: Dict[str, Any], quality_warnings=None, ood_warning: bool = False) -> Dict[str, Any]:
    gate = safe_decision_gate(
        fake_probability=float(report.get("fake_probability", 0.0)),
        attack_type=str(report.get("attack_type", "unknown")),
        suspicious_segments=report.get("suspicious_segments", []),
        quality_warnings=quality_warnings or [],
        ood_warning=ood_warning,
    )
    out = dict(report)
    out["raw_decision"] = report.get("decision", "unknown")
    out["decision"] = gate["safe_decision"]
    out["review_required"] = gate["review_required"]
    out["decision_gate"] = gate
    return out
