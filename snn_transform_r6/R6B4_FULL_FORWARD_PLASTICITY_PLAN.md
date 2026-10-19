# R6B.4 — restore complete forward-parameter plasticity

Status: IMPLEMENTATION BRANCH / R6B.3 BASELINE PRESERVED.

This branch exists because R6B.3 intentionally froze several forward parameters that had previously been part of the agreed "complete forward-parameter plasticity" target. The restoration does **not** change the MaskGIT objective, label boundary, tokenizer, or the basic local MSDL update principle.

## Forward parameters that must be plastic

The label-free image model has no class embedding. The forward parameter set to be plastic is therefore:

- visual token embedding table, initialized from the frozen SNN-VQ codebook geometry;
- 7x7 positional embedding table, initialized from the deterministic 2-D sinusoidal representation;
- Q/K/V projection weights and biases: `WQ, WK, WV, bq, bk, bv`;
- attention output projection and bias: `WO, bo`;
- normalization affine parameters: `gamma2, beta2`;
- FFN weights and biases: `W1, b1, W2, b2`;
- neuronal FFN threshold `theta` (kept in the forward parameter audit because it directly changes the forward map);
- categorical output head `WR` (its last constant feature already acts as the readout bias).

Structural/non-forward-learning quantities remain fixed: sparse masks, positive-RF basis `omega`, normalization epsilon, learning-rate scalars, and the frozen SNN-VQ encoder/decoder itself.

## What stays unchanged

- MaskGIT training masks: `r~U[0,1)`, `gamma(r)=cos(pi*r/2)`, all masked targets use one immutable corrupted context.
- No class labels enter input, learning, or generation.
- No backpropagation/BPTT/autograd optimizer.
- Forward learning remains local MSDL `dW ~ M_post E_pre^T` with persistent Izhikevich-derived E/M state.
- Existing P3 sufficient states `H/g`, collective K route, and the seven previously plastic forward matrices remain intact.

## Credit coverage restoration

R6B.3's local credit machinery already reaches `WR -> W2/W1/WO -> Q/K/V`, but it stopped before the normalization affine and input embedding tables. R6B.4 extends the **same existing reciprocal-mirror mechanism** to the missing local interfaces so that these already-existing forward parameters can receive an M state. This is a coverage extension of the current P3/MSDL implementation, not a new objective or conventional optimizer.

Extra reciprocal states are learning machinery, not forward inference parameters. They are used only to deliver local modulatory state to the newly unfrozen forward parameters.

## First gate

Before any long 2->3 epoch continuation, run a short upgrade/smoke gate from the exact 2.0-epoch R6B.3 checkpoint and require:

1. all old and newly restored forward parameter groups are finite;
2. every forward parameter group moves under learning;
3. visual-token rows (not only MASK) and position rows show nonzero movement;
4. no label enters the neural path;
5. no BP/autograd optimizer is introduced;
6. the original 2.0-epoch checkpoint remains untouched on the R6B.3 branch.

Only after this gate passes should generation training continue under the restored parameter set.
