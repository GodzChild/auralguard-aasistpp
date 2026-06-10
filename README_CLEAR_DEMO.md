
# AuralGuard-AASIST++ Clear Demo Patch

This adds:

```text
src/demo_clear.py
```

The demo makes the wording clearer:

- The final AuralGuard result appears first.
- The baseline is marked as "COMPARISON ONLY".
- The final model is marked as "USE THIS RESULT".
- The headline banner shows the final result only.
- The wording is cleaner and less confusing.

## Run

```cmd
python -m src.demo_clear --baseline-checkpoint "results\asvspoof_full_clean_gpu\best.pt" --final-checkpoint "results\final_accent_wavefake_balanced_full\best.pt" --baseline-name "Baseline AASIST" --final-name "AuralGuard-AASIST++" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --share
```

If your baseline checkpoint is elsewhere, replace the baseline path.
