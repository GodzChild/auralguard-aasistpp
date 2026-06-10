from __future__ import annotations

from typing import Any, Dict, List, Optional


def summarize_segment_scores(segments: List[Dict[str, float]]) -> Dict[str, Any]:
    """Create compact evidence from suspicious-window predictions.

    The project intentionally keeps explanations evidence-grounded: text reports are
    produced from model scores, attack prediction, and localization windows rather
    than from an unconstrained language model.
    """
    if not segments:
        return {
            "num_suspicious_segments": 0,
            "top_segment": None,
            "max_segment_fake_score": None,
        }

    top_segment = max(segments, key=lambda item: float(item.get("fake_score", 0.0)))
    return {
        "num_suspicious_segments": len(segments),
        "top_segment": top_segment,
        "max_segment_fake_score": float(top_segment.get("fake_score", 0.0)),
    }


def build_evidence_packet(
    *,
    fake_probability: float,
    attack_type: str,
    explanation_category: str,
    suspicious_segments: List[Dict[str, float]],
    thresholds: Optional[Dict[str, float]] = None,
    method_notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return the evidence object that supports the human-readable explanation."""
    thresholds = thresholds or {"real": 0.35, "fake": 0.65, "suspicious_window": 0.75}
    method_notes = method_notes or [
        "AASIST hidden representation is shared by all task heads.",
        "Attack-type prediction is used as one forensic clue, not as standalone source tracing.",
        "Suspicious timestamps are produced by overlapping-window inference.",
    ]

    return {
        "fake_probability": float(fake_probability),
        "attack_type": attack_type,
        "explanation_category": explanation_category,
        "thresholds": thresholds,
        "localization_summary": summarize_segment_scores(suspicious_segments),
        "method_notes": method_notes,
    }
