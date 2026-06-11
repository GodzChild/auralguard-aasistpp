# Demo Upgrade

This document explains the demo upgrade from a simple model output to a more complete forensic interface.

---

## 1. Original Demo Idea

The first demo idea was simple:

```text
upload audio -> model predicts real or fake
```

This is useful for testing, but it does not explain enough for a realistic forensic scenario.

A user may ask why the model said fake, whether the model was confident, which part of the audio was suspicious, whether the audio quality is good enough, and whether a human should review the result.

---

## 2. What Was Added

The upgraded demo adds final decision, fake probability, attack-type clue, suspicious timestamp evidence, beginner-friendly explanation, technical evidence summary, audio-quality diagnostics, and human-review recommendation.

---

## 3. Why This Is Important

A fake/real label alone can be misleading.

If the model is uncertain, forcing it to say fake or real may create a false sense of certainty.

The upgraded demo uses safer output logic:

```text
likely real
human review
likely fake
```

---

## 4. Relationship to the Project Goal

The demo upgrade directly supports the main research goal: build a detector that is useful for real-world review, not just benchmark classification.

The interface shows both the prediction and the evidence behind the prediction.

---

## 5. Example Command

```cmd
python -m src.demo_professional --checkpoint "results\final_accent_globe_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --share
```

---

## 6. Summary

The demo upgrade turns the project from a simple classifier into a more understandable forensic decision-support system.
