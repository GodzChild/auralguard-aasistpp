# AuralGuard-AASIST++

**Robust and explainable audio deepfake detection for synthetic speech and real-world accented speech.**

AuralGuard-AASIST++ is an AASIST-based audio deepfake detection project.
The goal is not only to detect fake or synthetic speech, but also to reduce false alarms on real accented, dialectal, and interview-style speech.

---

## 1. Project Motivation

Most audio deepfake detectors are evaluated mainly with fake/real accuracy.

A basic detector usually works like this:

```text
audio -> model -> real / fake
```

This is useful, but it is too simple for realistic forensic use.

Real-world speech can contain:

* accents
* dialects
* background noise
* interview pauses
* different microphones
* compression
* natural speaking variation

These features are not fake. However, a model that has not seen enough real-world speech may wrongly classify unfamiliar real speech as fake.

This project focuses on the question:

> Can an audio deepfake detector detect synthetic speech while avoiding false accusations on real accented speech?

---

## 2. Main Contribution

AuralGuard-AASIST++ extends an AASIST-style detector into a more complete forensic decision-support pipeline.

Instead of only producing:

```text
real / fake
```

the system produces:

```text
likely real / human review / likely fake
fake probability
attack-type clue
suspicious timestamp evidence
simple explanation
audio-quality diagnostics
```

The main improvements are:

* balanced real/fake training
* inclusion of real accented and interview-style speech
* false-alarm evaluation on real speech
* human-review decision gate
* explainable demo output
* suspicious timestamp localization
* audio-quality warnings

The system is designed as decision support, not legal proof.

---

## 3. Model Backbone

The project is based on AASIST, an audio anti-spoofing model originally designed for detecting fake or spoofed speech.

AASIST learns patterns from speech audio that may help separate real human speech from synthetic or manipulated speech.

This project uses the AASIST idea as the backbone, then adds a broader evaluation and demo pipeline focused on real-world robustness.

---

## 4. Datasets

The project uses both fake-speech datasets and real-speech datasets.

### Fake / spoofed speech

* ASVspoof 2019 LA
* WaveFake

### Real speech

* DECTE
* EdAcc
* English Dialects
* GLOBE

The real-speech datasets are important because they teach the model that accents, dialects, and interview-style recordings should not automatically be treated as fake.

Datasets and audio files are not included in this repository because of size and licensing restrictions.

---

## 5. Balanced Training Data

The final balanced training set contained:

| Class       | Samples |
| ----------- | ------: |
| Real speech |  16,603 |
| Fake speech |  16,603 |
| Total       |  33,206 |

This 50/50 balance was important because earlier data combinations were too fake-heavy.

A fake-heavy dataset can make the model more likely to wrongly classify unfamiliar real speech as fake.

---

## 6. Main Results

The final balanced model remained strong on standard detection metrics:

| Metric   | Result |
| -------- | -----: |
| Accuracy | 97.38% |
| F1 score | 97.44% |
| EER      |  1.67% |
| AUC      | 99.84% |

However, the main focus of this project is false-alarm reduction on real speech.

Important false-alarm results:

| Dataset / Test                           | Result |
| ---------------------------------------- | -----: |
| EdAcc false fake rate before improvement | 71.33% |
| EdAcc false fake rate after improvement  |  1.67% |
| GLOBE false fake rate before adaptation  | 89.90% |
| GLOBE false fake rate after adaptation   |  1.01% |
| DECTE false fake rate                    |  0.00% |
| English Dialects false fake rate         |  0.00% |

These results show that the model became much more careful with real accented and interview-style speech.

---

## 7. Demo

The project includes two demo versions.

### Simple ASVspoof Baseline Demo

This demo only shows a basic fake/real result.

```cmd
python -m src.demo_asvspoof_simple --checkpoint "results\asvspoof_full_clean_gpu\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --share
```

### Final AuralGuard-AASIST++ Demo

This demo shows the improved system with explanation and human-review support.

```cmd
python -m src.demo_professional --checkpoint "results\final_accent_globe_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --share
```

The final demo outputs:

* likely real / human review / likely fake
* fake probability
* attack-type clue
* suspicious timestamp evidence
* beginner-friendly explanation
* audio-quality diagnostics
* evidence packet

---

## 8. Repository Structure

```text
auralguard-aasistpp/
├── src/                  # model, inference, demo, training code
├── scripts/              # dataset preparation and evaluation scripts
├── configs/              # experiment configuration files
├── docs/                 # patch notes and additional documentation
├── requirements.txt      # Python dependencies
├── .gitignore            # ignored datasets, checkpoints, and large files
└── README.md             # main project documentation
```

---

## 9. Setup

Create and activate the environment:

```cmd
conda create -n auralguard2 python=3.9 -y
conda activate auralguard2
pip install -r requirements.txt
```

Clone the original AASIST repository into the external folder:

```cmd
mkdir external
git clone https://github.com/clovaai/aasist.git external/aasist
```

Expected structure:

```text
external/aasist/
├── models/
└── config/
```

---

## 10. Important Note

This repository does not include:

* datasets
* audio files
* trained checkpoints
* large result folders
* external AASIST source code

These are excluded using `.gitignore`.

To run the full demo, the trained checkpoint must be placed locally in the correct `results/` folder.

---

## 11. Limitations

AuralGuard-AASIST++ is not perfect and should not be used as legal proof.

The system may still fail on:

* very short clips
* noisy recordings
* phone-call audio
* compressed audio
* music-heavy audio
* unseen languages
* new deepfake generation methods

For uncertain or high-stakes cases, the system recommends human review.

---

## 12. Final Summary

AuralGuard-AASIST++ is an AASIST-based audio deepfake detection pipeline that detects synthetic speech while reducing false alarms on real accented and interview-style speech.

The main contribution is not only high fake/real accuracy, but a more responsible and explainable system that supports human review.
