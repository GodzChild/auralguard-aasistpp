
# AuralGuard-AASIST++ Robustness Evaluation Pack

Implements:
1. Gender bias diagnosis using EdAcc gender labels
2. Audio length robustness test
3. Calibration / confidence reliability
4. Partial fake localization test
5. Final research tables

## 1. Regenerate EdAcc with gender
```cmd
python scripts\make_edacc_metadata_with_gender.py --max-train 3000 --max-val 500
```

## 2. Gender bias diagnosis
```cmd
python scripts\evaluate_gender_bias.py --csv "data\metadata\val_edacc_gender.csv" --checkpoint "results\final_accent_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --limit 500 --out-csv "results\robustness\gender_predictions.csv" --summary-csv "results\robustness\gender_summary.csv"
```

## 3. Audio length robustness
```cmd
python scripts\evaluate_length_robustness.py --csv "data\metadata\val_final_accent_wavefake_balanced.csv" --checkpoint "results\final_accent_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --durations 4 8 12 20 --limit 300 --out-csv "results\robustness\length_predictions.csv" --summary-csv "results\robustness\length_summary.csv"
```

## 4. Calibration / confidence reliability
```cmd
python scripts\evaluate_calibration.py --csv "data\metadata\val_final_accent_wavefake_balanced.csv" --checkpoint "results\final_accent_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --limit 1000 --out-pred-csv "results\robustness\calibration_predictions.csv" --out-calibration-csv "results\robustness\calibration_bins.csv" --out-summary-csv "results\robustness\calibration_summary.csv"
```

## 5. Partial fake localization test
```cmd
python scripts\create_partial_fake_set.py --real-csv "data\metadata\val_final_accent_wavefake_balanced.csv" --fake-csv "data\metadata\wavefake_val.csv" --out-audio-dir "data\partial_sim" --out-csv "data\metadata\partial_sim_test.csv" --num-samples 300

python scripts\evaluate_partial_localization.py --csv "data\metadata\partial_sim_test.csv" --checkpoint "results\final_accent_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --out-csv "results\robustness\partial_predictions.csv" --summary-csv "results\robustness\partial_summary.csv"
```

## 6. Build final research tables
```cmd
python scripts\build_final_research_tables.py --gender-summary "results\robustness\gender_summary.csv" --length-summary "results\robustness\length_summary.csv" --calibration-summary "results\robustness\calibration_summary.csv" --partial-summary "results\robustness\partial_summary.csv" --out-md "results\robustness\final_research_tables.md" --out-csv-prefix "results\robustness\final_table"
```
