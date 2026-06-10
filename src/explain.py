from __future__ import annotations

from typing import Any, Dict, List

from .evidence import build_evidence_packet
from .labels import ID_TO_EXPLANATION


def decision_from_probability(
    fake_probability: float,
    real_threshold: float = 0.35,
    fake_threshold: float = 0.85,
    suspicious_segments: List[Dict[str, float]] | None = None,
    suspicious_window_threshold: float = 0.75,
) -> str:
    """Safer decision gate for real-world speech.

    Old/simple logic: fake_prob >= 0.65 => likely fake.
    New logic: fake_prob must be very high, otherwise suspicious/human review.

    This reduces false accusations on real accented/interview/noisy speech.
    """
    suspicious_segments = suspicious_segments or []

    if fake_probability < real_threshold:
        return "likely real"

    # If probability is not very high, avoid a hard fake decision.
    if fake_probability < fake_threshold:
        return "suspicious / human review"

    # If very high probability, still check if there is some consistent evidence.
    strong_windows = [
        s for s in suspicious_segments
        if float(s.get("fake_score", 0.0)) >= suspicious_window_threshold
    ]

    # If no localization evidence is available, allow likely fake based on high utterance score.
    if not suspicious_segments:
        return "likely fake"

    # If localization evidence exists and at least one strong window agrees, likely fake.
    if strong_windows:
        return "likely fake"

    # High utterance score but weak/unclear window evidence: safer to review.
    return "suspicious / human review"


def beginner_friendly_explanation(
    fake_prob: float,
    attack_type: str,
    suspicious_segments: List[Dict[str, float]] | None = None,
) -> str:
    suspicious_segments = suspicious_segments or []

    if fake_prob < 0.35:
        return (
            "The model thinks this audio is likely real because the fake score is low. "
            "The speech patterns look closer to real human speech examples than to synthetic or manipulated speech."
        )

    if fake_prob < 0.85:
        return (
            "The model is not fully sure. The fake score is not low, but it is also not safe enough to call the audio fake. "
            "This should be treated as suspicious and reviewed by a human, especially if the clip contains an accent, interview speech, noise, or recording-quality differences."
        )

    if attack_type == "tts_vc":
        reason = (
            "The model thinks this audio may be fake because it contains patterns similar to text-to-speech "
            "or voice-conversion examples from training."
        )
    elif attack_type == "codec":
        reason = (
            "The model thinks this audio may be fake because it contains patterns often linked to codec-based generated speech."
        )
    elif attack_type == "partial":
        reason = (
            "The model thinks only part of the audio may be manipulated because suspicious evidence appears in specific time regions."
        )
    elif attack_type == "bonafide":
        reason = (
            "The fake score is high, but the attack-type head predicts bonafide speech. "
            "This conflict means the safest decision is human review rather than automatic accusation."
        )
    else:
        reason = "The model found strong fake-like patterns in the audio."

    if suspicious_segments:
        top = max(suspicious_segments, key=lambda x: float(x.get("fake_score", 0.0)))
        time_reason = (
            f" The strongest suspicious part is around {float(top['start']):.1f}s to {float(top['end']):.1f}s, "
            f"with a fake score of {float(top['fake_score']):.2f}."
        )
    else:
        time_reason = " The evidence appears to affect the whole clip rather than one clear small region."

    return reason + time_reason


def explanation_text(
    fake_probability: float,
    attack_type: str,
    explanation_id: int,
    suspicious_segments: List[Dict[str, float]] | None = None,
) -> List[str]:
    suspicious_segments = suspicious_segments or []
    explanation_name = ID_TO_EXPLANATION.get(int(explanation_id), "unknown")

    reasons: List[str] = []

    if fake_probability >= 0.85:
        reasons.append("The shared AASIST-based representation produced very strong utterance-level fake evidence.")
    elif fake_probability >= 0.65:
        reasons.append("The shared AASIST-based representation produced moderate-to-strong fake evidence, but this is kept in the human-review range for real-world speech.")
    elif fake_probability >= 0.35:
        reasons.append("The fake probability is in the uncertain range, so the audio should be treated as suspicious rather than automatically fake.")
    else:
        reasons.append("The full-audio fake probability is low, so no strong utterance-level fake evidence was detected.")

    if attack_type == "codec":
        reasons.append("The attack-type head predicts a codec-based fake category; this is treated as one clue, not standalone codec source tracing.")
    elif attack_type == "partial":
        reasons.append("The attack-type head predicts localized or partially manipulated speech.")
    elif attack_type == "tts_vc":
        reasons.append("The attack-type head predicts a text-to-speech or voice-conversion style spoof.")
    elif attack_type == "bonafide":
        reasons.append("The attack-type head predicts bonafide speech.")

    if suspicious_segments:
        best = max(suspicious_segments, key=lambda x: float(x.get("fake_score", 0.0)))
        reasons.append(
            f"Sliding-window inference found suspicious evidence, with the strongest region at "
            f"{best['start']}s–{best['end']}s and window fake score {float(best['fake_score']):.2f}."
        )
    else:
        reasons.append("Sliding-window inference did not find a window above the suspicious threshold.")

    if explanation_name == "codec_based_artifact":
        reasons.append("The explanation head groups the evidence as codec-related artifacts.")
    elif explanation_name == "localized_partial_manipulation":
        reasons.append("The explanation head groups the evidence as localized partial manipulation.")
    elif explanation_name == "global_synthetic_artifact":
        reasons.append("The explanation head groups the evidence as global synthetic speech artifacts.")
    elif explanation_name == "no_strong_fake_evidence":
        reasons.append("The explanation head predicts the no-strong-fake-evidence category.")

    return reasons


def build_report(
    fake_probability: float,
    attack_type: str,
    explanation_id: int,
    suspicious_segments: List[Dict[str, float]],
    real_threshold: float = 0.35,
    fake_threshold: float = 0.85,
    suspicious_window_threshold: float = 0.75,
) -> Dict[str, Any]:
    decision = decision_from_probability(
        fake_probability,
        real_threshold=real_threshold,
        fake_threshold=fake_threshold,
        suspicious_segments=suspicious_segments,
        suspicious_window_threshold=suspicious_window_threshold,
    )
    explanation_category = ID_TO_EXPLANATION.get(int(explanation_id), "unknown")

    evidence = build_evidence_packet(
        fake_probability=fake_probability,
        attack_type=attack_type,
        explanation_category=explanation_category,
        suspicious_segments=suspicious_segments,
        thresholds={
            "real": real_threshold,
            "fake": fake_threshold,
            "suspicious_window": suspicious_window_threshold,
        },
    )

    return {
        "decision": decision,
        "fake_probability": round(float(fake_probability), 4),
        "attack_type": attack_type,
        "suspicious_segments": suspicious_segments,
        "explanation_category": explanation_category,
        "explanation": explanation_text(fake_probability, attack_type, explanation_id, suspicious_segments),
        "beginner_explanation": beginner_friendly_explanation(fake_probability, attack_type, suspicious_segments),
        "evidence": evidence,
    }
