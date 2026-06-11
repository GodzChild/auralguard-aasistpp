
# AuralGuard-AASIST++ Improved Demo Patch

This patch adds:

```text
src/demo_compare.py
src/decision_gate.py
src/audio_quality.py
src/explain_plus.py
```

It does not delete your old `src/demo.py`.

## Run with baseline and final model

From the project root:

```cmd
python -m src.demo_compare --baseline-checkpoint "results\asvspoof_full_clean_gpu\best.pt" --final-checkpoint "results\final_accent_wavefake_balanced_full\best.pt" --baseline-name "ASVspoof-only baseline" --final-name "Final balanced AuralGuard" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --share
```

If your ASVspoof baseline is in another folder, use:

```text
results\asvspoof_full_clean\best.pt
```

or whichever `best.pt` exists.

## What it shows

- Baseline vs final model result
- Fake probability comparison
- Safer decision gate: likely real / suspicious human review / likely fake
- Beginner-friendly explanation
- Audio quality warnings
- Suspicious timestamp regions
- Evidence packet
- Exported JSON report: `results\demo_last_report.json`

## Why this is useful for your thesis

It visibly shows what makes your project different from ordinary AASIST:

```text
Ordinary AASIST: audio → fake/real
AuralGuard-AASIST++: audio → fake/real/human-review + explanation + evidence + robustness warning
```
