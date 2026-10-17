# R6A — All-four-block H/g joint recovery

R6A is the first full-stack recovery gate after R5C. It does **not** yet test from-scratch learning.

## Fixed vs plastic

Plastic:

- blocks 0–3: WQ/WK/WV/WO, FFN W1/W2, LayerNorm affine parameters and associated biases.

Fixed:

- visual token embedding,
- position embedding,
- class embedding,
- prediction dense/LN/head and tied token readout,
- random-feature maps.

Class conditioning is intentionally retained as a causal control. R6A is therefore not the final label-free experiment.

## Credit rule

For blocks 0–2, every local perturbation uses per-image masked CE modulation because downstream attention can redistribute one perturbed token across the sequence.

For final block 3:

- Q/O/FFN/LN: token-local masked CE modulation;
- K/V: per-image masked CE modulation.

All updates are antithetic forward perturbation estimates passed through the same fast/slow learning states used in R5C. There is no backward pass, torch optimizer, learned reciprocal matrix, or auxiliary trainable credit network.

## Damage protocol

Each block receives the same controlled lesion used in R5C:

- attention matrices/biases × 0.60,
- FFN matrices/biases × 0.50,
- LayerNorm gamma × 0.75,
- LayerNorm beta → 0.

A cumulative lesion ladder measures 50%-mask CE after damaging:

1. block 3,
2. blocks 2–3,
3. blocks 1–3,
4. blocks 0–3.

This verifies that adding earlier damaged blocks produces a real functional deficit.

## Depth-aware learning rates

Default multiplicative scales are:

- block 0: 0.35,
- block 1: 0.50,
- block 2: 0.70,
- block 3: 1.00.

The scaling is a variance-control device, not a learned feedback mechanism.

## Pre-registered quantitative gate

R6A `pass=true` requires:

- full-stack damage increases CE50 by > 0.20;
- adding block 1 damage after blocks 2–3 increases CE50 by > 0.005;
- adding block 0 damage after blocks 1–3 increases CE50 by > 0.005;
- CE50 recovery fraction > 0.50;
- recovered CE is better than damaged CE at 25%, 50%, and 90% masking;
- reverting recovered block 0 alone to its damaged state worsens CE50 by > 0.002;
- reverting recovered block 1 alone to its damaged state worsens CE50 by > 0.002;
- no autograd gradients exist;
- learning remains isolated to blocks 0–3.

Generation metrics and galleries are saved as a separate advancement guard. Numerical `pass=true` alone will not be treated as sufficient if generation topology collapses.

## Interpretation boundary

A successful R6A would establish that forward-perturbative credit remains useful across the full four-block stack when starting from a controlled lesion of a pretrained solution. It would **not** establish random-initialized training, autonomous concept discovery, class-free learning, or equivalence to the original exact P3 Izhikevich E/M dynamics.
