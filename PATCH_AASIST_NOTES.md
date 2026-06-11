# Patch Notes: AASIST Integration

This document explains how the original AASIST model is used inside AuralGuard-AASIST++ and why the project includes wrapper code instead of directly modifying the official repository.

---

## 1. Purpose of the AASIST Patch

The official AASIST implementation provides a strong audio anti-spoofing model for fake/real speech detection. AuralGuard-AASIST++ uses this as the backbone, but the project needs additional functionality for a broader forensic workflow.

The wrapper code allows the project to load the original AASIST model structure, connect it to the project training scripts, run inference from a unified interface, and attach additional components such as decision gates, explanations, and demo outputs.

---

## 2. Why AASIST Is Kept in `external/aasist`

The original AASIST repository is kept inside:

```text
external/aasist/
```

This separates third-party code from the project code:

```text
external/aasist/  -> original third-party model code
src/              -> AuralGuard-AASIST++ implementation
scripts/          -> data preparation and evaluation tools
```

The external AASIST repository is not uploaded to this GitHub repository. Users should clone it separately.

---

## 3. How AASIST Is Extended

The project does not claim to invent AASIST. Instead, it builds a practical forensic pipeline around the AASIST idea.

The extensions include false-alarm robustness evaluation, real accented speech training and testing, balanced dataset preparation, human-review decisions, suspicious timestamp evidence, explanation generation, audio-quality diagnostics, and a professional demo interface.

---

## 4. Correct Research Framing

The project should be described as:

> an AASIST-based forensic framework for robust and explainable audio deepfake detection.

It should not be described as:

> a completely new audio deepfake architecture.

The contribution is mainly in the unified pipeline, the evaluation design, the false-alarm reduction focus, and the real-world robustness improvements.

---

## 5. Common Setup Issue

If this command:

```cmd
git clone https://github.com/clovaai/aasist.git external/aasist
```

returns:

```text
fatal: destination path 'external/aasist' already exists
```

that usually means AASIST has already been downloaded. This is not a problem unless the folder is incomplete.

---

## 6. Summary

AASIST is used as the backbone. AuralGuard-AASIST++ contributes the practical forensic system around it: robustness testing, false-alarm reduction, explanations, human-review logic, and demo presentation.
