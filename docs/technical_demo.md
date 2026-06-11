# Technical Demo

This document explains the technical demo version.

---

## 1. Purpose

The technical demo was created for development and debugging.

Unlike the professional demo, which is designed for presentation, the technical demo exposes more internal information.

It is useful when checking whether model outputs, scores, and evidence fields are working correctly.

---

## 2. Intended User

The technical demo is intended for the developer, technical reviewers, debugging sessions, and model-output inspection.

It is not the best demo for a non-technical audience because it may show too much information.

---

## 3. Typical Output

The technical demo may show fake probability, attack-type prediction, explanation category, suspicious segments, evidence fields, and raw or semi-raw model output.

---

## 4. Difference from Professional Demo

| Technical Demo | Professional Demo |
|---|---|
| More internal details | Cleaner presentation |
| Useful for debugging | Useful for professor/demo |
| May be harder to read | Beginner-friendly |
| Shows technical evidence directly | Organizes evidence into readable sections |

---

## 5. Example Command

```cmd
python -m src.demo_technical --checkpoint "results\final_accent_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --share
```

---

## 6. Summary

The technical demo is useful for development, but the professional demo should be used for the final presentation.
