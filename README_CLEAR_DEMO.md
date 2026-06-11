# Clear Demo

This document explains the clear demo version used during development.

---

## 1. Purpose

The clear demo was created to make the output easier to understand for non-technical viewers.

Earlier demo versions contained too many technical sections at once. This made the interface harder to explain during a short presentation.

The clear demo places the most important information first:

```text
final decision
fake probability
simple explanation
supporting evidence
```

---

## 2. Intended User

The clear demo is designed for supervisors, project reviewers, non-specialist audiences, and beginners learning about audio deepfake detection.

It avoids overwhelming the viewer with too much technical detail at the top of the page.

---

## 3. Main Output

The clear demo focuses on the final decision, fake probability, attack-type clue, explanation, and selected supporting evidence.

It is a middle step between the basic fake/real demo and the final professional demo.

---

## 4. Relationship to Other Demos

| Demo | Purpose |
|---|---|
| Simple ASVspoof demo | Show basic fake/real baseline |
| Clear demo | Show readable result with fewer distractions |
| Technical demo | Show more internal details |
| Professional demo | Final polished presentation version |

---

## 5. Example Command

```cmd
python -m src.demo_clear --checkpoint "results\final_accent_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --share
```

---

## 6. Summary

The clear demo was a readability-focused development version. It helped simplify the user experience before the final professional demo was created.
