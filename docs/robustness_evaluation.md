# Robustness Evaluation Pack

This document explains the robustness evaluation tools used in AuralGuard-AASIST++.

---

## 1. Purpose

Standard accuracy is not enough to understand whether an audio deepfake detector is reliable.

A model can have high accuracy but still fail in specific real-world situations.

The robustness evaluation pack tests questions such as:

- Does the model falsely flag real accented speech?
- Does the model behave differently for male and female speakers?
- Does audio length affect the prediction?
- Are the model probabilities believable?
- Can the model approximately locate partial fake regions?

---

## 2. Main Evaluations

### False-Alarm Evaluation

Checks how often real speech is wrongly predicted as fake. This is the most important evaluation for the project.

### Gender Bias Diagnosis

Checks whether false-fake rates differ across gender groups when metadata is available. This is reported as a diagnostic, not as a complete fairness solution.

### Audio Length Robustness

Checks whether predictions change when the same type of audio is evaluated with different durations.

### Calibration

Checks whether the model’s confidence scores are meaningful.

### Partial Fake Localization

Checks whether sliding-window inference can point near manipulated regions in simulated partial-fake examples.

---

## 3. Main Scripts

| Script | Purpose |
|---|---|
| `evaluate_false_alarms.py` | Tests false fake rates on real speech |
| `evaluate_gender_bias.py` | Compares subgroup false-alarm behavior |
| `evaluate_length_robustness.py` | Tests different input lengths |
| `evaluate_calibration.py` | Measures probability calibration |
| `create_partial_fake_set.py` | Creates simulated partial fake examples |
| `evaluate_partial_localization.py` | Tests timestamp localization quality |
| `build_final_research_tables.py` | Builds summary result tables |

---

## 4. Important Results

| Test | Result |
|---|---:|
| EdAcc false fake rate improved | 71.33% to 1.67% |
| GLOBE false fake rate improved | 89.90% to 1.01% |
| Calibration ECE | 0.0249 |
| Partial fake hit rate | 71.67% |
| Mean center error | 0.477 seconds |

---

## 5. Why This Matters

These tests make the project stronger because they show more than simple training performance.

They show how the system behaves under realistic conditions and whether the model output should be trusted, reviewed, or treated carefully.

---

## 6. Summary

The robustness evaluation pack supports the central claim of the project: AuralGuard-AASIST++ is not only a detector. It is a robustness-focused forensic decision-support pipeline.
