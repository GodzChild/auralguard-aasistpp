# Prosody / Tonality Diagnostics

This document explains the added prosody and tonality diagnostics in AuralGuard-AASIST++.

## Purpose

The original AASIST-based model analyzes the audio signal and produces a fake probability. The prosody module does not replace this model. It adds supporting voice-pattern evidence that can help a human reviewer understand the audio better.

Prosody means how speech sounds beyond the words. It includes pitch movement, rhythm, pauses, loudness changes, and tonal variation.

## What the module measures

The module extracts lightweight diagnostic features such as:

- pitch range,
- pitch variation,
- abrupt pitch movement,
- voice energy variation,
- silence or pause ratio,
- pause-like gap count,
- voiced fraction,
- spectral flatness,
- zero crossing rate.

These features help describe whether a voice sounds flat, highly variable, very smooth, strongly paused, or difficult to measure.

## How to interpret it

The prosody module should be treated as explanation support, not proof.

For example:

- low pitch variation may suggest a flat or monotone voice pattern,
- very low energy variation may suggest overly smooth speech,
- many silence gaps may suggest pauses or low-energy sections,
- weak pitch tracking may mean the audio is noisy, quiet, or not speech-like.

The final fake/real decision still comes from the AASIST-based model and decision gate.

## Why this improves the project

This addition connects the model output with human voice knowledge. Linguists, singers, vocal coaches, and audio engineers often listen for pitch, rhythm, breathing, pauses, tone quality, and emotional naturalness when judging whether a voice sounds human.

Adding these diagnostics makes the demo more explainable and creates a clear future research direction.

## Paper wording

A suggested paper paragraph:

> To provide additional voice-pattern evidence, the final demo was extended with a lightweight prosody and tonality diagnostics module. This module measures pitch variation, pitch movement, energy variation, pause-like gaps, silence ratio, and voiced speech proportion. The module does not replace the AASIST-based fake probability and is not used as standalone proof. Instead, it supports human interpretation by describing whether the uploaded speech contains unusually flat tone, abrupt tonal movement, very stable loudness, or pause-heavy structure. This makes the system more explainable and creates a bridge between neural audio deepfake detection and human voice expertise.

## Future work wording

A suggested future work paragraph:

> Future work should improve this prosody module through expert-informed analysis. Linguists, singers, vocal coaches, and audio engineers could be interviewed to identify human-perceived signs of artificial speech, such as unnatural pitch movement, missing breathing patterns, overly smooth voice texture, unnatural rhythm, and mismatch between emotion and tone. These expert insights could be converted into additional measurable features and combined with the AASIST representation in a multimodal or multi-branch model.
