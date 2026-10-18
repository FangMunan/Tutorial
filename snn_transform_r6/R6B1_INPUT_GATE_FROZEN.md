# R6B.1 — Spiking sensory tokenizer gate (FROZEN)

Date: 2026-10-18

## Purpose

Before any further SNN-Transform/MSDL image learning or generation, replace the historical non-neural `4x4 raw patch -> KMeans -> 7x7 token` front end with a neural sensory path and verify that it preserves image information at a level comparable to a matched learned ANN-VQ tokenizer.

## Literature-guided architecture

The candidate follows the VQ-SVAE direction used in Spiking-Diffusion: spiking convolutional downsampling to a `7x7` latent map, followed by vector quantization.  We add an explicit `28x28` sensory LIF sheet so raw pixels first drive neurons.

Frozen candidate:

```text
28x28 raw pixels
  -> 28x28 sensory LIF sheet
  -> 32 x 14x14 LIF (3x3 stride-2 convolution)
  -> 64 x 7x7 LIF (3x3 stride-2 convolution)
  -> 16 x 7x7 latent LIF (1x1 convolution)
  -> 0.75 spike-rate + 0.25 membrane-state latent readout
  -> VQ codebook K=64
  -> 7x7 discrete latent grid
```

Simulation/training settings used for the frozen gate:

```text
steps = 8
beta = 0.85
sensory_gain = 1.6
alpha = 0.75
latent_dim = 16
codebook = 64
epochs = 12
training subset = MNIST 0/1/2, 1200 examples/class before validation split
```

Surrogate-gradient training is used only for this tokenizer/input-information gate.  This is not the final MSDL learning rule.

## Matched comparison

All learned tokenizer candidates use `28x28 -> 7x7`, latent dimension 16 and VQ64.  Evaluation is tokenizer-only; there is no SNN-Transform, MSDL or free generation.

| metric | historical KMeans | matched ANN-VQ | frozen SNN-VQ (a=0.75,T=8) |
|---|---:|---:|---:|
| MSE | 0.008433 | 0.005995 | 0.006919 |
| PSNR (dB) | 20.740 | 22.222 | 21.600 |
| SSIM | 0.9116 | 0.8650 | 0.8713 |
| MNIST-feature rFID | 1.5610 | 2.3715 | **0.9799** |
| reconstruction classifier accuracy | 98.75% | 98.67% | **98.75%** |
| classifier consistency | 99.42% | 99.50% | 99.42% |
| codebook usage | 100% | 73.44% | 73.44% |
| active codes | 64/64 | 47/64 | 47/64 |
| code perplexity | 8.15 | 12.06 | **15.27** |
| largest-token fraction | 64.12% | 40.28% | **27.15%** |

SNN firing fractions:

```text
sensory      0.1447
14x14 layer  0.1957
7x7 layer    0.2106
latent       0.1615
```

The SNN candidate is within 0.62 dB of the ANN-VQ PSNR, slightly exceeds its SSIM, improves the same-dataset feature-rFID, matches codebook usage, and greatly reduces dominant-token concentration while retaining sparse latent firing.

## Tuning audit

GitHub Actions run `39139040624` tested:

- alpha 0.25, T=8
- alpha 0.50, T=8
- alpha 0.75, T=8
- alpha 0.50, T=16

The alpha=0.75, T=8 variant is frozen because it provided the best combined semantic/feature reconstruction quality and token balance without increasing simulation length.

## Decision

**R6B.1 INPUT GATE: PASS.**

This establishes only that a raw-image-driven spiking neural front end can produce a `7x7` VQ latent representation with same-class reconstruction quality comparable to a matched learned ANN tokenizer, while avoiding the severe single-token dominance of the old KMeans front end.

It does **not** yet establish:

- MSDL learning of the sensory encoder;
- SNN-Transform learning from these latents;
- MaskGIT masked-token learning on the new codebook;
- autonomous generation;
- neuromorphic energy/SynOp advantage.

Per project plan, downstream dynamical learning/generation must start only from this frozen input gate.
