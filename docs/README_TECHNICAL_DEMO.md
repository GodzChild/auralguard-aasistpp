
# AuralGuard-AASIST++ Technical Demo Design Patch

This patch adds a visually polished forensic dashboard:

```text
src/demo_technical.py
src/decision_gate.py
src/audio_quality.py
src/explain_plus.py
```

It keeps your old demo files safe.

## Run

```cmd
python -m src.demo_technical --baseline-checkpoint "results\asvspoof_full_clean_gpu\best.pt" --final-checkpoint "results\final_accent_wavefake_balanced_full\best.pt" --baseline-name "Baseline AASIST" --final-name "AuralGuard-AASIST++" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --share
```

If your baseline checkpoint is elsewhere, replace the baseline path.

## Design features

- Dark forensic dashboard style
- Blue/cyan technical gradient background
- Traffic-light decision cards
- Fake probability progress bars
- Baseline vs final model comparison
- Beginner explanation
- Audio-quality diagnostics
- Suspicious timestamp evidence
- Exported JSON report

The demo avoids informal notes and keeps the wording professional.
