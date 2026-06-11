# Professional Demo

This document explains the final professional Gradio demo for AuralGuard-AASIST++.

---

## 1. Purpose

The professional demo is the main user-facing interface for the project.

It is designed to show the final model result in a clear and understandable way.

The demo is meant for project presentation, supervisor demonstration, model inspection, qualitative testing, and comparison with a simple fake/real baseline.

---

## 2. Why a Professional Demo Was Needed

A command-line output is useful for development, but it is difficult to present to non-technical users.

The professional demo makes the system easier to understand by showing final decision, fake probability, attack-type clue, simple explanation, suspicious timestamp evidence, audio-quality diagnostics, and evidence packet.

This turns the model output into a readable forensic report.

---

## 3. Final Decision Logic

The demo uses a safer decision gate.

Instead of only:

```text
real / fake
```

it can output:

```text
likely real
human review
likely fake
```

This is important because uncertain predictions should not be presented as confident accusations.

---

## 4. Demo Command

```cmd
python -m src.demo_professional --checkpoint "results\final_accent_globe_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --share
```

---

## 5. Output Sections

### Final Decision

Shows whether the audio is likely real, should be reviewed, or is likely fake.

### Fake Probability

Shows the model’s suspicion score.

### Attack-Type Clue

Gives a clue about the type of suspicious pattern, such as bonafide or TTS/voice conversion.

### Beginner-Friendly Explanation

Explains the result in simple language.

### Suspicious Timestamp Evidence

Shows approximate regions where fake-like evidence appears.

### Audio-Quality Diagnostics

Warns if the audio may be too short, quiet, clipped, or otherwise unreliable.

### Evidence Packet

Provides a structured technical summary for review.

---

## 6. How to Explain the Demo

The demo is not just showing a label. It shows the model score, explains the decision, checks the audio quality, and recommends human review when the model is uncertain.

---

## 7. Summary

The professional demo is the final presentation interface for AuralGuard-AASIST++. It communicates model evidence clearly and supports responsible interpretation.
