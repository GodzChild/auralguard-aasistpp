# Related Work and Novelty Guardrails

This document explains how to describe AuralGuard-AASIST++ carefully and honestly in a thesis, report, paper, or presentation.

---

## 1. Why Guardrails Are Needed

Audio deepfake detection is an active research area.

Many individual ideas already exist in previous work, including AASIST-based audio anti-spoofing, fake/real speech detection, codec-aware detection, partial fake localization, explainable model outputs, out-of-domain detection, and robustness evaluation.

Therefore, AuralGuard-AASIST++ should not claim that every component is completely new.

The project should be framed as a unified and practical pipeline that combines these ideas around a specific problem: reducing false alarms on real accented and interview-style speech.

---

## 2. Safe Main Claim

A safe and accurate claim is:

> AuralGuard-AASIST++ extends an AASIST-based detector into a robust and explainable forensic decision-support pipeline, with a focus on reducing false alarms on real accented and interview-style speech.

This is stronger and more honest than claiming to invent audio deepfake detection.

---

## 3. Claims to Avoid

Avoid saying:

- “This is the first audio deepfake detector.”
- “This is the first explainable deepfake system.”
- “This is the first partial fake localization method.”
- “This completely solves accent bias.”
- “This system proves whether audio is fake.”
- “The model works on every random audio clip.”

These claims are too strong and may be inaccurate.

---

## 4. Better Claims

Use careful claims such as:

- “The project focuses on false-alarm reduction.”
- “The system improves robustness on the tested real-accent datasets.”
- “The demo supports human review by showing scores, explanations, and evidence.”
- “The timestamp evidence is approximate and should not be treated as word-level proof.”
- “The model should be used as decision support, not legal proof.”

---

## 5. How to Explain Novelty

The novelty is best described as a combination of:

1. AASIST-based detection.
2. Real accented and interview-style speech robustness.
3. Balanced training to reduce false fake bias.
4. False-alarm evaluation as a main result.
5. GLOBE adaptation for global-accent robustness.
6. Human-review decision gate.
7. Explanation and evidence-based demo.
8. Audio-quality diagnostics.

A short novelty sentence:

> The project’s contribution is a robustness-focused AASIST-based pipeline that combines fake detection with false-alarm evaluation, real-accent adaptation, explainable output, and human-review support.

---

## 6. Related Work Positioning

| Area | Relationship |
|---|---|
| ASVspoof | Provides benchmark spoofing task and data |
| AASIST | Provides backbone anti-spoofing model idea |
| WaveFake | Provides generated fake speech data |
| EdAcc | Supports accented real-speech evaluation |
| GLOBE | Supports global-accent robustness testing |
| Partial deepfake research | Motivates suspicious timestamp localization |
| Out-of-domain detection | Motivates human-review and uncertainty handling |

---

## 7. Limitations to State Clearly

The project should clearly state these limitations:

- It is not legal proof.
- It does not guarantee correct results on every random clip.
- It is mainly evaluated on selected English speech datasets.
- Some timestamp evidence is approximate.
- The explanation text is derived from model outputs and rules, not from a perfect causal explanation of the neural network.
- New deepfake generation methods may still require additional testing.

---

## 8. Best Thesis Framing

A good thesis or report framing is:

> This work investigates how an AASIST-based audio deepfake detector can be extended into a more robust and explainable decision-support system, especially by reducing false alarms on real accented and interview-style speech.

---

## 9. Summary

The project is strongest when it is framed as responsible improvement rather than complete invention.

The key message is:

```text
not just higher accuracy
but safer behavior on real speech
```
