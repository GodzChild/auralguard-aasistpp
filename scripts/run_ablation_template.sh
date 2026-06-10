#!/usr/bin/env bash
set -euo pipefail

# Edit these paths before running.
TRAIN_CSV="data/metadata/train.csv"
VAL_CSV="data/metadata/val.csv"
TEST_CSV="data/metadata/test.csv"
AASIST_ROOT="external/aasist"
AASIST_CONFIG="external/aasist/config/AASIST.conf"

# 1) Full unified model. The current starter trains all heads by default.
python -m src.train \
  --train-csv "$TRAIN_CSV" \
  --val-csv "$VAL_CSV" \
  --aasist-root "$AASIST_ROOT" \
  --aasist-config "$AASIST_CONFIG" \
  --epochs 10 \
  --batch-size 8 \
  --out-dir results/full_unified

python -m src.evaluate \
  --csv "$TEST_CSV" \
  --checkpoint results/full_unified/best.pt \
  --aasist-root "$AASIST_ROOT" \
  --aasist-config "$AASIST_CONFIG" \
  --out-csv results/full_unified/test_predictions.csv

# 2) Partial/localization evaluation if your CSV has start_fake and end_fake.
python -m src.evaluate_localization \
  --csv "$TEST_CSV" \
  --checkpoint results/full_unified/best.pt \
  --aasist-root "$AASIST_ROOT" \
  --aasist-config "$AASIST_CONFIG" \
  --out-csv results/full_unified/localization_predictions.csv
