# Gate D5A — pathway-specific local-credit divergence audit

## Frozen scientific question

D3/D4 established that the same F_EM boundary intervention produces terminal forward changes with opposite functional valence: W2^F is beneficial, whereas WO^F and bo^F are harmful and approximately additive. D5A asks why the same training episode drives these local plastic pathways in different functional directions.

D5A is **diagnostic only**. It does not modify the learner, learning rates, target order, sparse masks, reciprocal microsteps, E/M equations, or update boundaries. The only causal intervention retained is the already-frozen F_EM context-boundary reset of E2/M2 from D3E.

## Compared trajectories

Starting from the exact R6B3 2.0-epoch checkpoint, upgrade once to R6B4 aux_lr_scale=1.0 and replay the exact next 1024 contexts under two paired arms:

- `N`: native R6B4 continuation.
- `F_EM`: at every context boundary reset `(ve2,ue2,ze2)` and `(vm2,um2,zm2)` to canonical fresh initialization; all other states/rules remain native.

Image order, mask RNG, target order, mirror seeds, and external RNG must be identical between arms. Same-run plain/instrumented parity is checked at contexts 256, 512, and 1024 for both N and F_EM.

## Exact instantaneous descent oracles

For each target event, let L be the masked-token cross-entropy and let `g_y = -dL/dy`.

### W2

`G_W2 = (g_y h^T) * mask2`

Native local contribution:

`U_W2 = (M2 E2^T) * mask2`

### O projection

The O pathway must use the exact negative gradient with respect to the pre-normalization residual state z, not dy.

Forward:

`z = xraw[t] + bo + WO a`

`zhat = LayerNorm_no_affine(z)`

`u = gamma2 * zhat + beta2`

`h = ReLU(W1 u + b1 - theta)`

`y = z + W2 h + b2`

The exact `g_z = -dL/dz` includes both the direct residual route `z -> y` and the indirect route `z -> norm -> W1 -> h -> W2 -> y`.

`G_WO = (g_z a^T) * maskO`

Native local contribution:

`U_WO = (MO EO^T) * maskO`

### bo

`G_bo = g_z`

Native local contribution:

`U_bo = MO * Eb`

where `Eb` is the persistent constant-one eligibility state used by the native bias rule.

Before any trajectory result is interpreted, the z-gradient, WO-gradient, bo-gradient, and existing W2-gradient analytic formulas must pass finite-difference directional-derivative tests with relative error < 2e-7.

## M/E decomposition

Per event record:

- W2: `A_M = cos(M2, g_y)` and `A_E = cos(E2, h)`.
- WO: `A_M = cos(MO, g_z)` and `A_E = cos(EO, a)`.
- bo: shared `A_M = cos(MO, g_z)` plus constant-eligibility sign/amplitude audit for `Eb`.

The purpose is to distinguish a postsynaptic-credit mismatch from a presynaptic-eligibility mismatch. The shared-MO hypothesis is considered supported only if O-side M alignment is selectively poor/shifted while O-side E alignment remains comparatively intact and bo shows the same credit-direction problem.

## Context/trajectory endpoints

For each pathway `W2`, `WO`, `bo`, and each arm `N`, `F_EM`, report:

- context cosine alignment `A = cos(delta_theta, G)`;
- signed descent `delta_theta . G`;
- efficiency `(delta_theta . G) / ||delta_theta||^2`;
- `P(A<0)`;
- event-level M-side and E-side alignment;
- native path length, net displacement, and tortuosity;
- norm-matched oracle path/net/tortuosity;
- excess tortuosity native/oracle;
- orthogonal path fraction;
- `C_ug(lag)` for lags 1, 2, 4, 8.

Primary causal comparisons are paired pathway shifts between F_EM and N, especially:

`Delta A_i = A_i^(F_EM) - A_i^N`

with paired bootstrap confidence intervals for W2, WO, and bo.

## Interpretive outcomes

1. **O credit-direction failure** — WO/bo exact-oracle alignment is materially worse than W2 and worsens under F_EM.
2. **Shared MO failure** — O-side M alignment is selectively poor while EO eligibility remains comparatively intact, with bo showing the same M-linked direction problem.
3. **Local-objective/deployment mismatch** — WO/bo local exact-gradient alignment remains good, despite D4B proving their terminal displacement harms autonomous rollout.
4. **Temporal-organization divergence** — instantaneous O alignment remains good, but O-path excess tortuosity and/or lagged credit coupling is selectively abnormal.

D5A does not change lr_O, freeze O parameters, reset MO, or introduce any repair. Repair experiments are deferred until the diagnostic outcome is known.

## Provenance boundary

D4B remains formally classified by its preregistered classifier as `MIXED_OR_NONMONOTONIC_REQUIRES_INTERPRETATION`; `approximately additive dual-component O adaptation` is retained only as the post-hoc mechanistic descriptor. D5A tests the learning-dynamics mechanism underlying that terminal attribution.