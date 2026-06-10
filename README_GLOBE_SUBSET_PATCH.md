
# GLOBE Small Subset Patch for AuralGuard-AASIST++

This patch adds:

```text
scripts/make_globe_subset_metadata.py
```

It creates small real/bonafide metadata CSVs from a local GLOBE audio folder.

## 1. Create GLOBE metadata

From your project root:

```cmd
python scripts\make_globe_subset_metadata.py --audio-root "PASTE_PATH_TO_GLOBE_AUDIO_FOLDER" --max-train 2000 --max-val 300 --max-test 300
```

This creates:

```text
data\metadata\train_globe.csv
data\metadata\val_globe.csv
data\metadata\globe_test.csv
```

GLOBE is labelled as real/bonafide:

```text
binary_label = 0
attack_type = bonafide
dataset = GLOBE
```

## 2. Check the metadata

```cmd
python scripts\check_metadata.py --csv "data\metadata\train_globe.csv"
python scripts\check_metadata.py --csv "data\metadata\val_globe.csv"
python scripts\check_metadata.py --csv "data\metadata\globe_test.csv"
```

## 3. Merge GLOBE with the final full CSVs

Use the original final CSVs, not the already-balanced CSVs, then balance again.

```cmd
python scripts\merge_metadata.py --inputs "data\metadata\train_final_accent_wavefake.csv" "data\metadata\train_globe.csv" --out "data\metadata\train_final_accent_globe_wavefake.csv"
```

```cmd
python scripts\merge_metadata.py --inputs "data\metadata\val_final_accent_wavefake.csv" "data\metadata\val_globe.csv" --out "data\metadata\val_final_accent_globe_wavefake.csv"
```

## 4. Balance again

```cmd
python scripts\balance_metadata.py --input "data\metadata\train_final_accent_globe_wavefake.csv" --out "data\metadata\train_final_accent_globe_wavefake_balanced.csv" --fake-multiplier 1.0 --cap-wavefake 15000
```

```cmd
python scripts\balance_metadata.py --input "data\metadata\val_final_accent_globe_wavefake.csv" --out "data\metadata\val_final_accent_globe_wavefake_balanced.csv" --fake-multiplier 1.0 --cap-wavefake 3000
```

## 5. Debug train first

```cmd
python -m src.train --train-csv "data\metadata\train_final_accent_globe_wavefake_balanced.csv" --val-csv "data\metadata\val_final_accent_globe_wavefake_balanced.csv" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --epochs 2 --batch-size 4 --out-dir "results\debug_final_accent_globe_wavefake_balanced"
```

## 6. Full train

```cmd
python -m src.train --train-csv "data\metadata\train_final_accent_globe_wavefake_balanced.csv" --val-csv "data\metadata\val_final_accent_globe_wavefake_balanced.csv" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --epochs 10 --batch-size 4 --out-dir "results\final_accent_globe_wavefake_balanced_full"
```

## 7. Evaluate GLOBE false alarms

Before training with GLOBE, test the old final model:

```cmd
python scripts\evaluate_false_alarms.py --csv "data\metadata\globe_test.csv" --checkpoint "results\final_accent_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --limit 300 --out-csv "results\final_accent_wavefake_balanced_full\globe_false_alarms_before.csv"
```

After training with GLOBE, test the new model:

```cmd
python scripts\evaluate_false_alarms.py --csv "data\metadata\globe_test.csv" --checkpoint "results\final_accent_globe_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --limit 300 --out-csv "results\final_accent_globe_wavefake_balanced_full\globe_false_alarms_after.csv"
```
