# R6B.2 — SNN encoder reconnect causal test

## Question

Test whether the previously observed image-learning failure was caused primarily by the old image front-end.

Old historical route (already executed; **do not rerun**):

`28x28 -> raw 4x4 patches -> KMeans64 -> 7x7 tokens -> exact P3 SNN-Transform/MSDL`

New route:

`28x28 -> sensory LIF -> spiking conv -> 7x7 latent LIF populations -> VQ64 -> exact P3 SNN-Transform/MSDL`

The frozen R6B.1 tokenizer checkpoint is `SNN_TRANSFORM_R6B1_TUNE_a075_t8` from Actions run `39139040624`.

## What is held fixed

The downstream P3 image system is kept at the historical P3I-2 `plain_s05` configuration:

- resident H/g linear attention;
- resident Xphi V eligibility;
- collective K eligibility with R=7;
- persistent Izhikevich-derived E/M populations;
- local `dW ~ M E^T` for WQ/WK/WV/WO/W1/W2/WR;
- adaptive reciprocal BR/BO/B2 populations;
- plain startup MSDL scaffold;
- plain online mismatch weighting (no class balancing / CB);
- uniform internal forward-learning-rate scale = 0.5;
- 30,000 plastic events;
- same MNIST 0/1/2 split and same class conditioning as the historical image gate.

The R6B.1 SNN tokenizer is **frozen** during R6B.2. It was trained previously with surrogate gradients only to establish the image representation, but R6B.2 itself introduces no tokenizer backpropagation and no optimizer into the P3 core.

## Historical comparator

Use the already-existing P3I-2 `plain_s05` result from Actions run `39130766259`, artifact `SNN_TRANSFORM_P3I2_plain_s05`.

Historical values embedded in the R6B.2 script:

- dominant token fraction: 0.6464326
- scaffold CE50: 1.8275121
- final CE50: 1.7748955
- scaffold generation condition match: 0.5833333
- final generation condition match: 0.4722222

No KMeans baseline is rerun.

## Evaluation logic

Primary cross-route comparison is **image-space generation condition match**, because the old and new tokenizers define different discrete vocabularies. Absolute cross-entropy values across the two tokenizers are therefore not treated as directly comparable.

Within the new SNN-tokenizer route, report masked-token CE and accuracy at 25%, 50%, 90%, and 100% masking before and after plain P3 MSDL learning. Also report token usage, dominant-token fraction, and sensory/latent firing rate.

Pre-registered evidence for the input-bottleneck hypothesis requires both:

1. final generation condition match >= 0.60; and
2. improvement over the historical KMeans route >= 0.10 absolute.

A positive result supports the statement that the old image input/tokenization interface was a major bottleneck. A negative result means that replacing the KMeans front-end with this frozen discrete SNN-VQ front-end is not sufficient; it does **not** rule out a later experiment in which continuous spike/membrane states are coupled directly into SNN-Transform instead of being discretized by VQ.

## MaskGIT comparison boundary

No official MaskGIT run is performed in R6B.2. MaskGIT is used only as the architectural reference that separates an image tokenizer from the downstream masked-token model. R6B.2 tests the analogous modular replacement inside this project.

## Files

- `r6b2_snn_encoder_p3_msdl.py` — experiment code.
- `.github/workflows/snn_transform_r6b2_snn_encoder_reconnect.yml` — GitHub Actions runner.

The workflow is intentionally `workflow_dispatch` only so preparing these files does not start the experiment automatically.
