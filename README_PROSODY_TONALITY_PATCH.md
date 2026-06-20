# Prosody / Tonality Patch

This patch adds a new diagnostic layer to the AuralGuard-AASIST++ professional demo.

## Added files

```text
src/prosody_diagnostics.py
src/demo_professional.py
docs/prosody_tonality_diagnostics.md
```

## What it adds

The demo now includes a new tab:

```text
Prosody / Tonality Diagnostics
```

It measures supporting voice-pattern clues such as:

- pitch variation,
- pitch range,
- abrupt pitch movement,
- energy/loudness variation,
- silence ratio,
- pause-like gap count,
- voiced fraction.

## Important

This module does not decide whether the audio is fake.

It supports the explanation only. The final fake/real decision still comes from the AASIST-based model and the human-review decision gate.

## Run command

```cmd
python -m src.demo_professional --checkpoint "results\final_accent_globe_wavefake_balanced_full\best.pt" --aasist-root "external\aasist" --aasist-config "external\aasist\config\AASIST.conf" --share
```

## How to explain it

> I added a prosody and tonality diagnostics layer to the demo. It measures pitch movement, voice energy, pause-like gaps, and tonal variation. It does not replace the AASIST model, but it gives extra voice-pattern evidence that supports the explanation and human review process.
