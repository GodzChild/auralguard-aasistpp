
# AuralGuard-AASIST++ Showcase Demo

This patch adds a visually polished demo:

```text
src/demo_showcase.py
```

It also includes helper modules if they do not already exist:

```text
src/decision_gate.py
src/audio_quality.py
src/explain_plus.py
```

## What it shows

- Beautiful dashboard-style layout
- Baseline vs final model comparison
- Traffic-light style decisions
- Fake probability progress bars
- Beginner-friendly explanation
- Suspicious timestamp region tab
- Audio quality metrics and warnings
- Exported JSON report at `results/demo_showcase_last_report.json`

## Run

From the project root:

```cmd
python -m src.demo_showcase --baseline-checkpoint "results\asvspoof_full_clean_gpu\best.pt" --final-checkpoint "results\final_accent_wavefake_balanced_full\best.pt" --baseline-name "ASVspoof-only baseline" --final-name "Final balanced AuralGuard" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --share
```

If the baseline path does not exist, use:

```text
results\asvspoof_full_clean\best.pt
```

## For your presentation

Use the same DECTE or FRED clip that was wrongly predicted as fake before.
The demo will show whether the final balanced model reduces the fake probability.
