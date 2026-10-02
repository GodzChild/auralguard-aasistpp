# Reproducing the project

This note gives the main order I used for training and evaluation. The repository does not contain the datasets or trained checkpoints.

## 1. Environment

Create the main environment and install the Python requirements:

```bash
conda create -n auralguard2 python=3.9 -y
conda activate auralguard2
pip install -r requirements.txt
```

Install PyTorch/torchaudio with the CUDA build appropriate for the machine if GPU training is required.

Clone AASIST into:

```text
external/aasist/
```

The training and evaluation scripts use that location by default.

## 2. Prepare metadata

The repository contains dataset-specific scripts under `scripts/` for ASVspoof, EdAcc, English Dialects, GLOBE and other local data sources.

Useful checks include:

```bash
python scripts/check_metadata.py --help
python scripts/show_metadata_balance.py --help
python scripts/balance_metadata.py --help
```

The exact local audio paths depend on where the datasets are stored.

## 3. Train

```bash
python -m src.train \
  --train-csv path/to/train.csv \
  --val-csv path/to/val.csv \
  --aasist-root external/aasist \
  --aasist-config external/aasist/config/AASIST.conf \
  --out-dir results/run1
```

The default training script supports optional backbone freezing through `--freeze-backbone`.

## 4. Evaluate the detector

```bash
python -m src.evaluate \
  --csv path/to/test.csv \
  --checkpoint results/run1/best.pt \
  --out-csv results/predictions.csv
```

## 5. False-alarm testing

For real-speech datasets:

```bash
python scripts/evaluate_false_alarms.py \
  --csv path/to/real_speech.csv \
  --checkpoint results/run1/best.pt \
  --out-csv results/false_alarms.csv
```

This was one of the main evaluations because the project focuses on avoiding false fake predictions on real accented speech.

## 6. Other robustness checks

The main scripts are:

```text
scripts/evaluate_gender_bias.py
scripts/evaluate_calibration.py
scripts/evaluate_length_robustness.py
scripts/create_partial_fake_set.py
scripts/evaluate_partial_localization.py
```

Each script has CLI arguments for local metadata/checkpoint paths.

## 7. Demo

```bash
python -m src.demo_professional \
  --checkpoint results/run1/best.pt
```

The demo adds the review gate, quality checks, explanations, localization and prosody diagnostics around the detector output.

## 8. Reproducibility limits

The original datasets are not redistributable from this repository, and trained checkpoints are also excluded.

Some experiments were run at different stages of the project with different metadata combinations. For that reason, the exact result numbers should be matched to the corresponding local metadata and checkpoint rather than assumed to come from every possible training configuration.

The README reports the final results used in the project summary.
