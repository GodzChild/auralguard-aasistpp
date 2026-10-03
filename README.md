# AuralGuard-AASIST++

**Live demo:** https://huggingface.co/spaces/AyoPrince/AuralGuard

AuralGuard-AASIST++ is my audio deepfake detection project built around the AASIST anti-spoofing model.

The main goal was not only to detect synthetic speech, but also to study a practical problem I kept seeing during testing: a detector can score well on benchmark data and still wrongly flag real accented or interview-style speech as fake.

I therefore extended the basic detector with extra evaluation, explanation and review tools, and tested it on a mix of benchmark, generated and real-accent speech.

## What the project does

The core model predicts whether an audio clip is real or fake. Around that, the project adds:

- false-alarm evaluation on real speech,
- a three-way decision gate: likely real / review / likely fake,
- sliding-window localization for suspicious regions,
- simple explanation text and evidence output,
- audio-quality checks,
- prosody/voice diagnostics,
- subgroup and robustness evaluation,
- a Gradio demo.

The system is intended as research and decision support. It should not be treated as proof that a recording is genuine or fake.

## Data used

The project uses several datasets for different parts of the work:

| Dataset | Use |
| --- | --- |
| ASVspoof 2019 LA | main anti-spoofing benchmark |
| WaveFake | additional synthetic speech |
| DECTE | real interview/dialect speech |
| EdAcc | real accented English |
| English Dialects | additional accent coverage |
| GLOBE | global English accent coverage |

The datasets are not included in this repository because of size and licensing restrictions.

## Training setup

One issue in the earlier experiments was class imbalance. The final training metadata used an equal number of real and fake examples:

| Class | Samples |
| --- | ---: |
| Real | 16,603 |
| Fake | 16,603 |
| Total | 33,206 |

The model code is in `src/`, while dataset preparation and evaluation utilities are in `scripts/`.

## Selected results

The final balanced model reported:

| Metric | Result |
| --- | ---: |
| Accuracy | 97.38% |
| F1 | 97.44% |
| EER | 1.67% |
| AUC | 99.84% |

I also tracked false-fake rates on real accented speech because this was one of the main practical concerns of the project.

| Evaluation | False-fake rate |
| --- | ---: |
| EdAcc before adaptation | 71.33% |
| EdAcc after adaptation | 1.67% |
| GLOBE before adaptation | 89.90% |
| GLOBE after adaptation | 1.01% |

These numbers come from the experiments in this project and should be read in the context of the tested datasets and checkpoints. They are not a claim of general performance on arbitrary real-world audio.

More details are in `docs/robustness_evaluation.md`.

## Repository structure

```text
auralguard-aasistpp/
├── src/                  model, training, inference, evaluation and demos
├── scripts/              data preparation and robustness experiments
├── configs/              experiment configurations
├── docs/                 implementation and research notes
├── requirements.txt
└── README.md
```

The repository developed iteratively during the project, so some scripts are experiment-specific rather than part of a polished Python package.

## Setup

I used Python 3.9 and a Conda environment called `auralguard2`.

```bash
conda create -n auralguard2 python=3.9 -y
conda activate auralguard2
pip install -r requirements.txt
```

The project expects the original AASIST repository under `external/aasist`:

```bash
mkdir external
git clone https://github.com/clovaai/aasist.git external/aasist
```

For CUDA, install a PyTorch build that matches the CUDA version on the machine.

## Training

A basic training command is:

```bash
python -m src.train \
  --train-csv path/to/train.csv \
  --val-csv path/to/val.csv \
  --out-dir results/run1
```

## Evaluation

```bash
python -m src.evaluate \
  --csv path/to/test.csv \
  --checkpoint results/run1/best.pt \
  --out-csv results/predictions.csv
```

## Single-file inference

```bash
python -m src.infer \
  --audio path/to/audio.wav \
  --checkpoint results/run1/best.pt
```

## Hugging Face Space

The repository now includes a root `app.py` for permanent Gradio deployment on Hugging Face Spaces. The hosted launcher keeps the checkpoint outside GitHub and downloads it from a separate Hugging Face model repository.

The model repository can remain private while the Space is public. In that setup, store a read-capable Hugging Face token as the Space secret `HF_TOKEN` and configure the model repository ID with the Space variable `AURALGUARD_MODEL_REPO`.

The Space also retrieves the required AASIST model/config files from the official upstream repository at a pinned commit instead of committing the external repository here.

See `docs/HUGGINGFACE_SPACE.md` for the complete deployment steps, local hosted-style test command, CPU/GPU notes, and checkpoint licensing cautions.

## Demo

The main demo I used at the end of the project is:

```bash
python -m src.demo_professional \
  --checkpoint results/run1/best.pt
```

Add `--share` if you want Gradio to create a temporary public link.

Other demo files in `src/` are earlier versions kept for comparison during development.

## Robustness and diagnostics

The repository includes scripts for:

- false-alarm testing,
- gender subgroup diagnostics,
- calibration,
- audio-length robustness,
- partial-fake localization,
- metadata balancing,
- comparison tables.

See `docs/REPRODUCIBILITY.md` for the main workflow and `docs/robustness_evaluation.md` for the evaluation summary.

## Files not included

The repository does not include:

- datasets,
- audio files,
- trained checkpoints,
- generated result folders,
- external model repositories.

These are ignored by Git.

## Limitations

This project is still a research prototype. Results depend on the datasets, checkpoints and preprocessing used in the experiments.

The detector can be affected by recording quality, compression, noise, very short clips, unseen languages and newer generation methods. The localization and explanation outputs are supporting evidence rather than exact causal explanations of the model.

For high-stakes use, uncertain predictions should be reviewed by a person.
