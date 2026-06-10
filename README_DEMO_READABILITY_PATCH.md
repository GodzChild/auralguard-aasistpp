# AuralGuard Demo Readability Patch

This patch only changes the readability of:

- Evidence packet JSON
- Suspicious timestamp regions JSON
- Simple explanation textbox

It does not change the model, inference, decision logic, or the rest of the dashboard.

## Run from project root

```cmd
cd /d "C:\Users\AYO\Desktop\JKU\Extra Semester\THESIS AND PRACTICAL\auralguard-aasistpp"
conda activate auralguard2
python scripts\patch_demo_readability.py
```

Then run:

```cmd
python -m src.demo_professional --checkpoint "results\final_accent_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --share
```
