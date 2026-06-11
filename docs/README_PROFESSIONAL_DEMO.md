
# AuralGuard-AASIST++ Professional Demo Patch

This adds:

```text
src/demo_professional.py
```

It removes the model comparison section completely.

## Run

```cmd
python -m src.demo_professional --checkpoint "results\final_accent_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --share
```

## What changed

- No baseline/model comparison section
- Only final AuralGuard-AASIST++ result is shown
- Professional forensic dashboard design
- White readable text
- Clear final decision
- Evidence summary
- Audio quality diagnostics
- Suspicious timestamp evidence
- Exported JSON report
