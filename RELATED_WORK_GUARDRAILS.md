# Related-work guardrails for AuralGuard-AASIST++

Use the project title:

> **AuralGuard-AASIST++: A Unified AASIST-Based Forensic Framework for Codec-Aware Partial Speech Deepfake Detection and Explanation**

## What this implementation now takes into account

The novelty should **not** be claimed as simply using AASIST, detecting codec fakes, detecting partial fake audio, or doing generic explainability. Those parts already exist as active research directions.

The defensible claim is the **combination**:

```text
shared AASIST representation
+ utterance-level fake/real detection
+ codec-aware attack-type prediction
+ temporal suspicious-region localization
+ evidence-grounded explanation
```

## Implementation choices added to avoid looking like a copy

1. **AASIST stays the center** of the main system.
2. **Codec recognition is one forensic task**, not a standalone codec-source-tracing paper.
3. **Partial localization is evaluated separately** with IoU and center-error metrics.
4. **Explanation is evidence-grounded**, built from scores, attack type, explanation category, and suspicious windows.
5. **Whisper/WavLM are not part of the main implementation**. They are left as optional ablations only.

## Baselines to report

| Baseline | Purpose |
|---|---|
| Original AASIST | Shows the base detector works |
| AASIST + attack head | Shows whether attack-type learning helps |
| AASIST + sliding-window localization | Shows timestamp benefit |
| Full AuralGuard-AASIST++ | Shows the unified system |
| Optional Whisper/WavLM ablation | Only if time permits; not the main novelty |

## Practical rule for thesis writing

Do not write:

> We propose AASIST for audio deepfake detection.

Write:

> We extend AASIST into a unified forensic framework that jointly predicts fake/real status, codec-aware attack type, suspicious temporal regions, and evidence-grounded explanations.
