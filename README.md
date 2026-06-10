# AuralGuard-AASIST++

Robust and explainable audio deepfake detection for synthetic speech and real-world accented speech.

## Project Goal

AuralGuard-AASIST++ is an AASIST-based audio deepfake detection system.

The goal is not only to detect fake speech, but also to reduce false alarms on real accented, dialectal, and interview-style speech.

A normal detector may say:

```text
audio -> real / fake
```

This project improves that into:

```text
audio -> likely real / human review / likely fake
      -> fake probability
      -> attack-type clue
      -> suspicious timestamp evidence
      -> simple explanation
      -> audio-quality diagnostics
```

## Main Contribution

This project focuses on a realistic problem:

> Can an audio deepfake detector detect fake speech without wrongly accusing real accented speech?

The system improves the original AASIST-style detector by adding:

* balanced real/fake training
* real accented and interview-style speech datasets
* false-alarm evaluation
* human-review decision gate
* explainable demo output
* suspicious timestamp evidence
* audio-quality diagnostics

## Datasets Used

Fake / spoofed speech:

* ASVspoof 2019 LA
* WaveFake

Real speech:

* DECTE
* EdAcc
* English Dialects
* GLOBE

Datasets and audio files are not included in this repository because of size and licensing.

## Main Results

The final model reduced false alarms on real accented speech.

Important results:

* EdAcc false fake rate dropped from 71.33% to 1.67%
* GLOBE false fake rate dropped from 89.90% to 1.01%
* DECTE false fake rate: 0.00%
* EdAcc false fake rate after final GLOBE adaptation: 0.00%
* English Dialects false fake rate: 0.00%

The final balanced model also remained strong on normal detection metrics:

* Accuracy: 97.38%
* F1 score: 97.44%
* EER: 1.67%
* AUC: 99.84%

## Run the Final Demo

```cmd
python -m src.demo_professional --checkpoint "results\final_accent_globe_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --share
```

## Run Simple ASVspoof Baseline Demo

```cmd
python -m src.demo_asvspoof_simple --checkpoint "results\asvspoof_full_clean_gpu\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --share
```

## Note

This system is not legal proof.

It is a forensic decision-support tool.
If the system is unsure, it recommends human review instead of forcing a fake/real decision.
