from __future__ import annotations

ATTACK_TO_ID = {
    "bonafide": 0,
    "tts_vc": 1,
    "codec": 2,
    "partial": 3,
}

ID_TO_ATTACK = {v: k for k, v in ATTACK_TO_ID.items()}

EXPLANATION_TO_ID = {
    "no_strong_fake_evidence": 0,
    "global_synthetic_artifact": 1,
    "codec_based_artifact": 2,
    "localized_partial_manipulation": 3,
}

ID_TO_EXPLANATION = {v: k for k, v in EXPLANATION_TO_ID.items()}


def normalize_attack_type(value: str) -> str:
    """Map dataset-specific attack names to the compact labels used by this project."""
    value = str(value).strip().lower()

    if value in {"bonafide", "real", "genuine", "human"}:
        return "bonafide"

    if value in {"codec", "codecfake", "codecfake+", "cosg", "codec_based"}:
        return "codec"

    if value in {"partial", "partialspoof", "partially_fake", "spliced", "localized"}:
        return "partial"

    if value in {"tts", "vc", "tts_vc", "voice_conversion", "synthetic", "spoof", "fake"}:
        return "tts_vc"

    raise ValueError(
        f"Unknown attack_type={value!r}. Expected one of: {sorted(ATTACK_TO_ID)} "
        "or a supported alias such as real, fake, codec, partial, tts, vc."
    )


def make_explanation_label(binary_label: int, attack_type: str) -> int:
    """Create a simple explanation-category label from existing metadata.

    This is useful when no human explanation labels are available.
    """
    attack_type = normalize_attack_type(attack_type)

    if int(binary_label) == 0:
        return EXPLANATION_TO_ID["no_strong_fake_evidence"]

    if attack_type == "codec":
        return EXPLANATION_TO_ID["codec_based_artifact"]

    if attack_type == "partial":
        return EXPLANATION_TO_ID["localized_partial_manipulation"]

    return EXPLANATION_TO_ID["global_synthetic_artifact"]
