# Hugging Face Spaces deployment

This project can be hosted as a Gradio Space without committing the trained checkpoint or the AASIST repository to GitHub.

The hosted entry point is `app.py`. It reuses the existing professional interface from `src/demo_professional.py`.

## How startup works

When the Space starts:

1. `app.py` downloads the small set of original AASIST files needed for inference from the official `clovaai/aasist` repository.
2. The AASIST source is pinned to commit `a04c9863f63d44471dde8a6abcb3b082b07cd1d1`.
3. The trained AuralGuard checkpoint is downloaded with `huggingface_hub.hf_hub_download` from the model repository configured in `AURALGUARD_MODEL_REPO`.
4. `src.demo_professional.create_demo(...)` builds the same professional Gradio interface used locally.
5. The model runs on CUDA when available, otherwise on CPU.

Downloaded files are cached by the Space runtime. They are not committed back to this repository.

## Space variables and secrets

Set these in the Hugging Face Space under **Settings → Variables and secrets**.

Variables:

- `AURALGUARD_MODEL_REPO` — required. Full model repository ID, for example `<hf-username>/auralguard-aasistpp-model`.
- `AURALGUARD_CHECKPOINT_FILE` — optional. Defaults to `best.pt`.
- `AURALGUARD_MODEL_REVISION` — optional. Defaults to `main`.
- `AURALGUARD_AASIST_COMMIT` — optional. Normally leave this unset so deployment stays pinned to the tested AASIST commit.

Secret:

- `HF_TOKEN` — required only while the model repository is private. Use a Hugging Face user access token that can read that private model repository.

Do not put the token in `README.md`, `app.py`, `.env`, Git history, screenshots, or model cards.

## Model repository

Create a Hugging Face **Model** repository named `auralguard-aasistpp-model`. Keep it private initially.

Upload only the deployment checkpoint that corresponds to the documented AuralGuard model. The default filename expected by the Space is:

```text
best.pt
```

Using the current Hugging Face CLI:

```bash
hf auth login
hf upload <hf-username>/auralguard-aasistpp-model path/to/best.pt best.pt
```

The Hub handles large model files. Do not add `best.pt` to the GitHub source repository.

When replacing the deployed model later, either upload a new `best.pt` revision or use a different filename/revision and update `AURALGUARD_CHECKPOINT_FILE` or `AURALGUARD_MODEL_REVISION` in the Space settings. Restart the Space after changing the deployment configuration.

## Run the hosted-style app locally

The normal command-line demo remains supported:

```bash
python -m src.demo_professional --checkpoint results/run1/best.pt
```

To test the hosted entry point locally, set the same environment values used by the Space and run:

Windows PowerShell:

```powershell
$env:AURALGUARD_MODEL_REPO="<hf-username>/auralguard-aasistpp-model"
$env:HF_TOKEN="<your-read-token>"
python app.py
```

Linux/macOS:

```bash
export AURALGUARD_MODEL_REPO="<hf-username>/auralguard-aasistpp-model"
export HF_TOKEN="<your-read-token>"
python app.py
```

If the model repository is public, omit `HF_TOKEN`.

## Create the Space

Create a new Hugging Face **Space** using the Gradio SDK. A practical name is `AuralGuard`.

The Space repository needs the files from this deployment branch. One simple workflow is to clone this GitHub repository, check out `deployment/huggingface-space`, then push that checked-out source to the Space repository. The root `README.md` contains the Space metadata and `app.py` is the startup file.

After the source is present in the Space repository:

1. Add `AURALGUARD_MODEL_REPO` as a Space variable.
2. If the model repo is private, add `HF_TOKEN` as a Space secret.
3. Keep the Space itself public if you want a portfolio link.
4. Let the Space build on the default CPU hardware first.
5. Check the build/runtime logs if startup fails.

A Space normally has a public page at:

```text
https://huggingface.co/spaces/<hf-username>/AuralGuard
```

Its app is also exposed through a `.hf.space` host generated from the owner and Space name.

## CPU and GPU notes

AuralGuard can execute on CPU because the code does not require CUDA. CPU inference will be slower, especially because one analysis also runs sliding-window localization over the full recording and computes acoustic/prosody diagnostics.

Start with CPU hardware so the public demo is inexpensive and easy to maintain. If response time is too slow, change the Space hardware in its settings to a GPU option. The launcher automatically selects CUDA when `torch.cuda.is_available()` becomes true; no model-code redesign is required.

If changing to GPU hardware, verify that the installed PyTorch build in that Space image exposes CUDA. Keep the trained checkpoint and model architecture unchanged.

## AASIST dependency and license

The deployment does not copy the full AASIST repository into this GitHub project. At runtime it retrieves only:

- `models/AASIST.py`
- `config/AASIST.conf`
- `LICENSE`

from the official repository at the pinned commit above.

The upstream AASIST code is MIT licensed. The downloaded upstream license is kept alongside the downloaded source in `external/aasist/LICENSE`.

## Checkpoint licensing and dataset restrictions

The code license of AASIST does not automatically determine whether a trained AuralGuard checkpoint can be redistributed publicly.

This checkpoint was trained using several third-party speech datasets. Their data licenses and access terms differ, and some corpora place restrictions on redistribution or research use. A derived neural-network checkpoint may or may not be treated explicitly by each dataset's terms.

For that reason, keep `auralguard-aasistpp-model` private until the training-data permissions have been reviewed dataset by dataset. In particular, verify the terms that applied to the exact copies of ASVspoof 2019 LA, WaveFake, DECTE, EdAcc, English Dialects and GLOBE used during training, and record the result before publishing the weights.

If a dataset license is unclear about redistribution of derived model weights, the conservative deployment is:

- public GitHub source,
- public Space,
- private model repository,
- `HF_TOKEN` stored only as a Space secret.

This lets visitors use the demo without giving them direct public access to the checkpoint.

## Security and privacy

The application does not require API keys other than the optional private-model read token.

Uploaded audio is processed by the Space runtime. Do not describe the service as providing confidential or legally privileged processing unless the hosting setup has been independently reviewed for that purpose.

The interface deliberately states that AuralGuard is research/decision-support software and not legal proof that a recording is genuine or fake.
