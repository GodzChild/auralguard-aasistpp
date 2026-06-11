# AuralGuard-AASIST++

**Robust and explainable audio deepfake detection for synthetic speech and real-world accented speech.**

AuralGuard-AASIST++ is an AASIST-based audio deepfake detection project. The system is designed to detect synthetic or manipulated speech while also reducing false alarms on real accented, dialectal, and interview-style speech.

The project started from a simple audio anti-spoofing setup:

```text
audio -> model -> real / fake
```

It was extended into a more complete forensic decision-support pipeline:

```text
audio
  -> likely real / human review / likely fake
  -> fake probability
  -> attack-type clue
  -> suspicious timestamp evidence
  -> beginner-friendly explanation
  -> audio-quality diagnostics
  -> evidence packet for review
```

The main motivation is that high benchmark accuracy is not enough for real-world forensic use. A detector can perform well on clean benchmark data but still wrongly flag real speakers when the audio contains accents, dialects, interview pauses, microphone differences, compression, or background noise.

---

## 1. Project Goal

The goal of this project is to answer a practical question:

> Can an audio deepfake detector detect fake speech while avoiding false accusations on real accented and interview-style speech?

This is important because a false fake accusation can be harmful. If a real speaker with an accent, dialect, or unusual recording condition is wrongly flagged as fake, the detector is not reliable enough for responsible use.

AuralGuard-AASIST++ therefore focuses on two goals:

1. **Detection**: identify synthetic or manipulated speech.
2. **Restraint**: avoid wrongly classifying real accented speech as fake.

---

## 2. Main Contribution

AuralGuard-AASIST++ extends an AASIST-style detector into a broader research and demo system.

The main contributions are:

- AASIST-based fake/real detection using a strong audio anti-spoofing backbone.
- Balanced real/fake training to reduce bias toward predicting fake.
- Real-speech robustness testing using real accented, dialectal, and interview-style datasets.
- False-alarm evaluation as a central metric, not only standard accuracy.
- Human-review decision gate with three safer outputs: likely real, human review, and likely fake.
- Suspicious timestamp evidence using sliding-window inference.
- Beginner-friendly explanations for non-technical users.
- Audio-quality diagnostics to warn when input conditions may reduce reliability.
- Professional Gradio demo for presenting model predictions and evidence.

The project should be understood as a forensic decision-support system, not as an automatic legal judgement system.

---

## 3. Model Background

The project is based on the AASIST audio anti-spoofing architecture.

AASIST was designed to detect spoofed or fake speech by learning audio patterns that distinguish bonafide human speech from synthetic or manipulated speech. In simple terms, it looks for fake-like clues in the sound signal.

AuralGuard-AASIST++ uses AASIST as the core model idea, then adds a practical pipeline around it:

```text
AASIST representation
+ fake/real prediction
+ attack-type clue
+ sliding-window suspicious region detection
+ decision gate
+ explanation layer
+ demo interface
```

---

## 4. Datasets

The project uses both fake-speech datasets and real-speech datasets.

### Fake / spoofed speech

| Dataset | Role |
|---|---|
| ASVspoof 2019 LA | Benchmark real and spoofed speech for anti-spoofing training and evaluation |
| WaveFake | Additional generated fake speech examples |

### Real speech

| Dataset | Role |
|---|---|
| DECTE | Real interview and dialect speech |
| EdAcc | Real accented English speech |
| English Dialects | British Isles accent coverage |
| GLOBE | Global English accent coverage |

The real-speech datasets are important because they teach the model that accent, dialect, and interview-style speech are still real speech.

Datasets and audio files are **not included** in this repository because of size and licensing restrictions.

---

## 5. Balanced Training Data

A key improvement was balancing the training data.

Earlier dataset combinations were too fake-heavy. This can make a model more likely to classify unfamiliar real speech as fake.

The final balanced training set contained:

| Class | Samples |
|---|---:|
| Real speech | 16,603 |
| Fake speech | 16,603 |
| Total | 33,206 |

This 50/50 split helped the model learn both sides of the task more fairly.

---

## 6. Main Results

The final balanced model remained strong on standard detection metrics:

| Metric | Result |
|---|---:|
| Accuracy | 97.38% |
| F1 score | 97.44% |
| Equal Error Rate, EER | 1.67% |
| AUC | 99.84% |

However, the main result of this project is not only standard accuracy. The most important result is the reduction of false alarms on real accented speech.

| Test | False fake rate |
|---|---:|
| EdAcc before improvement | 71.33% |
| EdAcc after improvement | 1.67% |
| GLOBE before adaptation | 89.90% |
| GLOBE after adaptation | 1.01% |
| DECTE after final evaluation | 0.00% |
| EdAcc after final GLOBE adaptation | 0.00% |
| English Dialects after final evaluation | 0.00% |

These results show that the system became much more careful with real accented and interview-style speech while still keeping strong fake-speech detection performance.

---

## 7. Demo Versions

The repository includes multiple demo versions used during development. The two most important ones are:

### 7.1 Simple ASVspoof Baseline Demo

This demo is intentionally basic. It is useful for showing the difference between a simple fake/real detector and the improved AuralGuard-AASIST++ demo.

```cmd
python -m src.demo_asvspoof_simple --checkpoint "results\asvspoof_full_clean_gpu\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --share
```

Output:

```text
REAL or FAKE
fake probability
```

### 7.2 Final Professional Demo

This is the final user-facing demo.

```cmd
python -m src.demo_professional --checkpoint "results\final_accent_globe_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --share
```

Output:

```text
likely real / human review / likely fake
fake probability
attack-type clue
suspicious timestamp evidence
beginner-friendly explanation
audio-quality diagnostics
evidence packet
```

---

## 8. Repository Structure

```text
auralguard-aasistpp/
├── src/                  # model, training, inference, evaluation, and demos
├── scripts/              # dataset preparation, balancing, robustness tests
├── configs/              # experiment configuration files
├── docs/                 # extra documentation and patch notes
├── requirements.txt      # Python package requirements
├── .gitignore            # excludes datasets, checkpoints, results, and audio
└── README.md             # main project documentation
```

---

## 9. Setup

Create the environment:

```cmd
conda create -n auralguard2 python=3.9 -y
conda activate auralguard2
pip install -r requirements.txt
```

Clone the original AASIST repository into the expected external folder:

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

## 10. Important Files Not Included

This repository does not include:

- datasets
- audio files
- trained checkpoints
- large result folders
- downloaded external repositories
- zipped datasets

These are intentionally excluded using `.gitignore`.

To run the full demo, trained checkpoints must be placed locally in the expected `results/` folder.

---

## 11. Limitations

AuralGuard-AASIST++ is not perfect.

The system may still fail on:

- very short audio clips
- noisy recordings
- compressed phone-call audio
- music-heavy clips
- languages not represented in training
- new deepfake generation methods
- domains very different from the training data

For uncertain or high-stakes cases, the system should recommend human review.

---

## 12. Final Summary

AuralGuard-AASIST++ is an AASIST-based audio deepfake detection pipeline that detects synthetic speech while reducing false alarms on real accented and interview-style speech.

The main improvement is not only higher model performance, but a more responsible and explainable forensic workflow:

```text
detect fake speech
+ reduce false accusations
+ explain the decision
+ support human review
```
