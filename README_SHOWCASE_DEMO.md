# Showcase Demo

This document explains the showcase demo version.

---

## 1. Purpose

The showcase demo was created to make the project visually presentable during development.

It focused on making the interface look more polished and easier to explain to an audience.

---

## 2. Difference from Other Demos

| Demo | Main purpose |
|---|---|
| Simple ASVspoof demo | Basic fake/real baseline |
| Clear demo | Simple readable result |
| Technical demo | Detailed technical output |
| Showcase demo | Presentation-friendly visual layout |
| Professional demo | Final polished decision-support interface |

The showcase demo was an intermediate step before the professional demo.

---

## 3. Features

The showcase demo may include improved layout, clearer result blocks, stronger visual hierarchy, model score display, explanation display, and evidence sections.

---

## 4. When to Use It

Use the showcase demo only if you want to show an earlier presentation-style interface.

For final project presentation, use:

```cmd
python -m src.demo_professional --checkpoint "results\final_accent_globe_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --share
```

---

## 5. Summary

The showcase demo was useful during interface development, but the professional demo is the recommended final version.
