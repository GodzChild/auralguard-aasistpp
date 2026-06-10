
# AuralGuard Metadata Balancing Patch

This patch fixes the problem where WaveFake dominates the final training CSV.

Your previous merged CSV looked like:
- Fake: 130,211
- Real: 16,603

That is about 89% fake, so the model may learn to predict "fake" too often.

## Files added

```text
scripts/balance_metadata.py
scripts/show_metadata_balance.py
```

## Recommended commands

From the AuralGuard project root:

```cmd
python scripts\balance_metadata.py --input "data\metadata\train_final_accent_wavefake.csv" --out "data\metadata\train_final_accent_wavefake_balanced.csv" --fake-multiplier 1.0 --cap-wavefake 15000

python scripts\balance_metadata.py --input "data\metadata\val_final_accent_wavefake.csv" --out "data\metadata\val_final_accent_wavefake_balanced.csv" --fake-multiplier 1.0 --cap-wavefake 3000
```

Then check:

```cmd
python scripts\show_metadata_balance.py --csv "data\metadata\train_final_accent_wavefake_balanced.csv"
python scripts\show_metadata_balance.py --csv "data\metadata\val_final_accent_wavefake_balanced.csv"
```

Train with:

```cmd
python -m src.train --train-csv "data\metadata\train_final_accent_wavefake_balanced.csv" --val-csv "data\metadata\val_final_accent_wavefake_balanced.csv" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --epochs 10 --batch-size 4 --out-dir "results\final_accent_wavefake_balanced_full"
```
