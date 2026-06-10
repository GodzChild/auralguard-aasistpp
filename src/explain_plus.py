
from __future__ import annotations

from typing import Any, Dict, List, Optional


def beginner_friendly_explanation_plus(
    fake_prob: float,
    attack_type: str,
    suspicious_segments: Optional[List[Dict[str, Any]]] = None,
    safe_decision: Optional[str] = None,
    warnings: Optional[List[str]] = None,
) -> str:
    suspicious_segments = suspicious_segments or []
    warnings = warnings or []

    decision_text = safe_decision or (
        "likely real" if fake_prob < 0.35 else "likely fake" if fake_prob > 0.85 else "suspicious / human review"
    )

    if decision_text == "likely real":
        msg = (
            "The model thinks this audio is likely real because the fake score is low. "
            "Its sound patterns are closer to real human speech examples than to generated or manipulated speech."
        )
    elif decision_text == "likely fake":
        if attack_type == "tts_vc":
            msg = (
                "The model thinks this audio may be fake because it contains patterns similar to text-to-speech "
                "or voice-conversion examples seen during training."
            )
        elif attack_type == "codec":
            msg = (
                "The model thinks this audio may be fake because it contains patterns often linked to codec-based generated speech."
            )
        elif attack_type == "partial":
            msg = (
                "The model thinks part of the audio may be manipulated because suspicious evidence appears in specific time regions."
            )
        else:
            msg = "The model found strong fake-like patterns in the audio."
    else:
        msg = (
            "The model is not fully sure. The score or evidence suggests the audio needs human review instead of being "
            "automatically labelled fake or real."
        )

    if suspicious_segments:
        top = max(suspicious_segments, key=lambda x: float(x.get("fake_score", 0.0)))
        msg += (
            f" The strongest suspicious region is around {float(top.get('start', 0.0)):.1f}s to "
            f"{float(top.get('end', 0.0)):.1f}s, with a fake score of {float(top.get('fake_score', 0.0)):.2f}."
        )

    if warnings:
        msg += " Reliability warning: " + " ".join(warnings)

    return msg
