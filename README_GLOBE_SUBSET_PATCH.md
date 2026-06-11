# GLOBE Subset Patch

This document explains the GLOBE dataset subset workflow used in AuralGuard-AASIST++.

---

## 1. Why GLOBE Was Added

After training the balanced accent model, the system performed well on DECTE, EdAcc, and English Dialects.

However, testing on GLOBE exposed a new weakness: the model falsely flagged many real global-accent clips as fake.

This showed that even after adding some accent data, the model still needed broader global-accent exposure.

---

## 2. Problem

GLOBE is a large dataset. Downloading the full dataset is unnecessary for a small project experiment and may take too much storage and time.

The solution was to stream a small subset instead of downloading everything.

---

## 3. Goal of the Patch

The GLOBE subset patch creates a small local subset:

```text
1000 training clips
200 validation clips
about 198 test clips
```

This allows the model to learn from real global-accent English speech without downloading the full dataset.

---

## 4. Main Script

```text
scripts/make_globe_streaming_subset.py
```

Example command:

```cmd
python scripts\make_globe_streaming_subset.py --max-train 1000 --max-val 200 --max-test 200
```

Expected outputs:

```text
data\metadata\globe_train.csv
data\metadata\globe_val.csv
data\metadata\globe_test.csv
```

---

## 5. Important Metadata Check

After creating the CSV files, confirm that the GLOBE test set contains only GLOBE real speech and not accidentally mixed fake files.

```cmd
python -c "import pandas as pd; df=pd.read_csv('data\\metadata\\globe_test.csv'); print('WaveFake paths:', df.file_path.str.contains('wavefake|generated_audio', case=False).sum()); print('ASVspoof paths:', df.file_path.str.contains('asvspoof', case=False).sum())"
```

Expected result:

```text
WaveFake paths: 0
ASVspoof paths: 0
```

---

## 6. Result

Before GLOBE adaptation:

| Threshold | False fake rate |
|---|---:|
| 0.65 | 89.90% |
| 0.85 | 82.32% |

After GLOBE adaptation:

| Threshold | False fake rate |
|---|---:|
| 0.65 | 1.01% |
| 0.85 | 0.00% |

This was one of the strongest results in the project.

---

## 7. Beginner Explanation

The model was still too suspicious of some global English accents. By adding a small amount of correctly labelled GLOBE real speech, the model learned that these accents are real, not fake.

---

## 8. Summary

The GLOBE subset patch improved global-accent robustness without requiring the full dataset download.
