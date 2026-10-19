# R6B.4 — restore complete forward-parameter plasticity

Status: IMPLEMENTATION BRANCH / R6B.3 BASELINE PRESERVED.

R6B.3 unintentionally left several ordinary forward parameters fixed relative to the previously agreed complete-forward-plasticity target. R6B.4 restores those parameters without changing the MaskGIT objective or replacing the existing P3/MSDL learning rule.

## Plastic forward parameters

The label-free model has no class embedding. The forward parameter set is:

- visual token embedding table, initialized from the frozen SNN-VQ codebook geometry;
- 7x7 positional embedding table, initialized from the deterministic 2-D sinusoidal representation;
- `WQ, WK, WV` and `bq, bk, bv`;
- `WO` and `bo`;
- normalization affine parameters `gamma2, beta2`;
- `W1, b1, W2, b2`;
- categorical output head `WR` (its constant feature supplies the readout bias degree of freedom).

`theta` remains a fixed neuronal threshold rather than a standard MaskGIT affine parameter. Structural quantities also remain fixed: sparse masks, positive-RF basis `omega`, normalization epsilon, learning-rate scalars, and the frozen SNN-VQ encoder/decoder.

## Learning structure is not changed

- MaskGIT masking remains `r~U[0,1)`, `gamma(r)=cos(pi*r/2)`, with all masked targets evaluated from one immutable corrupted context.
- No label enters neural input, MSDL learning, or generation.
- No BP/BPTT/autograd optimizer is introduced.
- No new learned feedback/inverse matrix is introduced.
- No new finite-difference or perturbation-based credit route is introduced.
- The already-developed P3 local rule remains `dW ~ M_post E_pre^T` with persistent Izhikevich-derived E/M states.

The restoration therefore extends parameter coverage using local E/M signals already present in R6B.3/P3. Biases are ordinary constant-input local synapses. The normalization affine parameters reuse the existing W1-side local M state through the fixed sparse fan-out topology. Token and position embeddings reuse the already-existing residual, Q, collective-K and V local M signals together with the resident forward attention coefficients. These operations add no learned inverse path.

## First gate

Before any long continuation, the exact R6B.3 2.0-epoch checkpoint is upgraded only in memory and a short disposable smoke run must establish:

1. all pre-existing forward values are unchanged at upgrade time;
2. no new learned `B*` feedback/inverse matrix appears;
3. with all forward learning rates disabled, every forward parameter remains fixed;
4. under MSDL learning, every restored forward parameter group moves and remains finite;
5. visible token-embedding rows encountered by the smoke stream move, and all 49 position rows receive plastic updates;
6. labels remain absent and no BP/autograd optimizer is introduced.

The original R6B.3 branch and exact 2.0-epoch checkpoint remain untouched. A formal 2->3 epoch experiment should be launched only after this restoration gate passes.
