
# AuralGuard-AASIST++ Full Improvements Patch

This patch implements the 8 requested improvements:

1. False-alarm evaluation script
2. Human-review / uncertainty gate
3. Out-of-domain/quality warning support
4. Better beginner explanation helper
5. Robustness augmentation
6. Balanced sampling
7. Partial fake localization test
8. Final comparison tables

## Files added

```text
src/decision_gate.py
src/audio_quality.py
src/explain_plus.py

scripts/evaluate_false_alarms.py
scripts/compare_models_on_audio.py
scripts/balance_metadata.py
scripts/show_metadata_balance.py
scripts/make_augmented_metadata.py
scripts/create_partial_fake_set.py
scripts/evaluate_partial_localization.py
scripts/build_final_comparison_table.py
```

## 1. Balance your final metadata

```cmd
python scripts\balance_metadata.py --input "data\metadata\train_final_accent_wavefake.csv" --out "data\metadata\train_final_accent_wavefake_balanced.csv" --fake-multiplier 1.0 --cap-wavefake 15000

python scripts\balance_metadata.py --input "data\metadata\val_final_accent_wavefake.csv" --out "data\metadata\val_final_accent_wavefake_balanced.csv" --fake-multiplier 1.0 --cap-wavefake 3000
```

Check:

```cmd
python scripts\show_metadata_balance.py --csv "data\metadata\train_final_accent_wavefake_balanced.csv"
```

## 2. Train balanced final model

```cmd
python -m src.train --train-csv "data\metadata\train_final_accent_wavefake_balanced.csv" --val-csv "data\metadata\val_final_accent_wavefake_balanced.csv" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --epochs 10 --batch-size 4 --out-dir "results\final_accent_wavefake_balanced_full"
```

## 3. False-alarm evaluation

```cmd
python scripts\evaluate_false_alarms.py --csv "data\metadata\decte_val.csv" --checkpoint "results\final_accent_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --limit 300 --out-csv "results\final_accent_wavefake_balanced_full\decte_false_alarms.csv"

python scripts\evaluate_false_alarms.py --csv "data\metadata\val_english_dialects.csv" --checkpoint "results\final_accent_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --limit 300 --out-csv "results\final_accent_wavefake_balanced_full\english_dialects_false_alarms.csv"

python scripts\evaluate_false_alarms.py --csv "data\metadata\val_edacc.csv" --checkpoint "results\final_accent_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --limit 300 --out-csv "results\final_accent_wavefake_balanced_full\edacc_false_alarms.csv"
```

## 4. Compare models on the same audio clip

```cmd
python scripts\compare_models_on_audio.py --audio "PASTE_AUDIO_PATH_HERE" --names "ASVspoof_only" "DECTE_WaveFake" "Final_Balanced" --checkpoints "results\asvspoof_full_clean\best.pt" "results\asvspoof_decte_wavefake_full\best.pt" "results\final_accent_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --out-csv "results\same_clip_comparison.csv"
```

## 5. Robustness augmentation

Create augmented audio from a small subset:

```cmd
python scripts\make_augmented_metadata.py --csv "data\metadata\train_final_accent_wavefake_balanced.csv" --out-audio-dir "data\augmentations\final_balanced" --out-csv "data\metadata\train_final_accent_wavefake_augmented.csv" --limit 2000
```

Then merge your balanced CSV with augmented CSV:

```cmd
python scripts\merge_metadata.py --inputs "data\metadata\train_final_accent_wavefake_balanced.csv" "data\metadata\train_final_accent_wavefake_augmented.csv" --out "data\metadata\train_final_accent_wavefake_balanced_augmented.csv"
```

Train only if you have time:

```cmd
python -m src.train --train-csv "data\metadata\train_final_accent_wavefake_balanced_augmented.csv" --val-csv "data\metadata\val_final_accent_wavefake_balanced.csv" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --epochs 10 --batch-size 4 --out-dir "results\final_balanced_augmented_full"
```

## 6. Partial fake localization test

Create a synthetic partial-fake test set:

```cmd
python scripts\create_partial_fake_set.py --real-csv "data\metadata\val_final_accent_wavefake_balanced.csv" --fake-csv "data\metadata\wavefake_val.csv" --out-audio-dir "data\partial_sim" --out-csv "data\metadata\partial_sim_test.csv" --num-samples 300
```

Evaluate localization:

```cmd
python scripts\evaluate_partial_localization.py --csv "data\metadata\partial_sim_test.csv" --checkpoint "results\final_accent_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --out-csv "results\final_accent_wavefake_balanced_full\partial_localization_eval.csv"
```

## 7. Final comparison table

```cmd
python scripts\build_final_comparison_table.py --false-alarm-csvs "results\final_accent_wavefake_balanced_full\decte_false_alarms.csv" "results\final_accent_wavefake_balanced_full\english_dialects_false_alarms.csv" "results\final_accent_wavefake_balanced_full\edacc_false_alarms.csv" --labels "DECTE" "EnglishDialects" "EdAcc" --out-csv "results\final_false_alarm_summary.csv" --out-md "results\final_false_alarm_summary.md"
```

## Recommended order

1. Balance metadata
2. Train balanced final model
3. Run false-alarm evaluation
4. Compare models on same clips
5. Run final demo
6. Add augmentation only if time
7. Add partial localization test only if time
