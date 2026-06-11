# Dataset Balancing Patch

This document explains the balancing tools used in AuralGuard-AASIST++.

---

## 1. Why Balancing Was Needed

During early experiments, some dataset combinations became too fake-heavy. This happened because fake datasets such as WaveFake can contain many generated samples, while some real-speech datasets contain fewer usable clips.

A fake-heavy dataset can create a serious problem:

```text
too many fake examples -> model learns to predict fake too often
```

This is dangerous because the project is not only trying to detect fake speech. It is also trying to avoid falsely accusing real accented speech.

---

## 2. Goal of the Patch

The balancing patch creates a cleaner real/fake training set.

The target is:

```text
real speech samples = fake speech samples
```

The final balanced training set contained:

| Class | Samples |
|---|---:|
| Real speech | 16,603 |
| Fake speech | 16,603 |
| Total | 33,206 |

This makes the model less biased toward predicting one class.

---

## 3. Main Scripts

### `scripts/balance_metadata.py`

Balances a metadata CSV by sampling real and fake examples.

```cmd
python scripts\balance_metadata.py --input "data\metadata\train_final_accent_wavefake.csv" --out "data\metadata\train_final_accent_wavefake_balanced.csv" --fake-multiplier 1.0 --cap-wavefake 15000
```

### `scripts/show_metadata_balance.py`

Prints class and dataset counts so the result can be checked.

```cmd
python scripts\show_metadata_balance.py --csv "data\metadata\train_final_accent_wavefake_balanced.csv"
```

---

## 4. Why It Matters

Balancing directly supports the research goal.

The project asks:

> How often does the model wrongly call real speech fake?

A balanced dataset helps reduce false alarms because the model sees enough real examples during training.

---

## 5. Beginner Explanation

If the model sees too many fake examples, it may become too suspicious. Balancing gives the model equal real and fake examples, so it learns a fairer boundary between the two classes.

---

## 6. Summary

The balancing patch improves training data quality and supports the main contribution of AuralGuard-AASIST++: reducing false fake alarms on real accented and interview-style speech.
