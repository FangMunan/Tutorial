# Gate D5D — emergence of negative deployment utility along the exact-WO trajectory

## Frozen scientific question

D5B established that norm-matched exact local descent in `WO` realizes stronger finite-step masked-token descent yet produces a terminal `WO` state that is deployment-harmful in fixed native and W2-beneficial backgrounds. D5C then showed that this mismatch is not localized to an immediate first-order self-conditioning gradient gap: a small exact-WO descent perturbation around the terminal native state improves both teacher-conditioned and self-conditioned prediction, and full-CRN branch switching does not reverse that sign.

D5D therefore asks a trajectory-level question:

> When, and by what geometric mode, does negative deployment utility emerge along the exact-WO trajectory?

D5D is **replay + checkpoint + readout only**. It introduces no new learner rule, no learning-rate change, no state reset beyond the already frozen F_EM boundary reset, no optimizer, no truncation, and no parameter freezing.

## Trajectories and paired stream

Replay exactly two arms from the same exact R6B3 2.0-context-epoch checkpoint, upgraded once to the frozen R6B4 aux_lr_scale=1.0 system:

- `F_native`: ordinary F_EM continuation.
- `F_WO_oracle`: the exact D5B WO-only oracle arm. O-side E/M dynamics remain native; only the context-boundary WO update direction is replaced by the arm-specific norm-matched exact negative gradient, exactly as in D5B.

Both arms replay the exact same 1024 contexts, masks, target order, mirror seeds, and external RNG. Save full checkpoints at

`T = {0, 64, 128, 256, 512, 768, 1024}`.

The terminal Gate-A outputs for both arms must reproduce the successful D5B stored arrays with exact proposal IDs and finite metric error < 1e-7 before D5D is interpreted.

## D5D-A — deployment onset curve

At every checkpoint t, evaluate both paired arms with the frozen Gate-A coordinate system. Primary endpoints remain:

- `F61`
- `D_non61`

Supportive endpoints remain:

- `C61`
- `delta_P61`

`CE` is retained as a diagnostic endpoint.

The primary causal quantity is time-matched:

`Delta J_m(t) = J_m[WO-oracle(t)] - J_m[native(t)]`.

All endpoint differences are stored as per-image arrays and summarized with paired 10,000-bootstrap confidence intervals.

Also record at every checkpoint:

- cumulative and recent online masked NLL from the replayed training stream;
- fixed-probe masked CE;
- ordinary masked-token evaluation at ratios 0.25, 0.5, 0.9, and 1.0;
- autonomous generation global, spatial, and pair JS;
- final-round token-marginal JS to the frozen Gate-A ground-truth token distribution;
- mean position-wise token entropy;
- final proposal P(token=61);
- round-wise sampled confidence and proposal-61 fraction;
- oracle-vs-native proposal-switch fraction by round.

The purpose of the additional deployment coordinates is to identify whether distribution drift precedes the visible Gate-A phenotype. They are descriptive unless explicitly covered by a frozen causal contrast.

## D5D-B — fixed-probe gradient-field rotation

Gradient rotation must not be confounded by changing training examples. D5D therefore reuses the D5C fixed held-out gradient-probe construction: 256 validation-token images disjoint from both the training split and frozen Gate-A images, with one deterministic masked context per image.

At every checkpoint and for both arms recompute

`g_t = - d L_masked(B_probe; theta_t) / d WO`

and normalize

`g_hat_t = g_t / ||g_t||`.

Store:

- `R0(t) = cos(g_hat_t, g_hat_0)`;
- `Rprev(t) = cos(g_hat_t, g_hat_previous_checkpoint)`;
- `Rcross(t) = cos(g_hat_t^oracle, g_hat_t^native)`;
- checkpoint-sampled cumulative rotation
  `Omega(t) = sum_k arccos(clip(Rprev(k), -1, 1))`.

This is a model-state-induced gradient-field measure because the probe inputs and masks are fixed.

## D5D-C — endpoint displacement versus path dependence

At each checkpoint retain the exact oracle WO state `WO_oracle(t)`.

### Time-matched native background

Construct

`native(t) + WO_oracle(t)`

by replacing only WO in the time-matched native checkpoint. Compare this transplant with the unmodified native(t) Gate-A result.

### Fixed final native background

Use the final native checkpoint `native(1024)` as one common forward background. Into it transplant each `WO_oracle(t)`. This produces

`J_path_fixed(t)`.

### Straight chord control

Let

`W0 = WO_oracle(0)` and `WT = WO_oracle(1024)`.

Define

`W_chord(lambda) = W0 + lambda (WT - W0)`

at

`lambda = {0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1}`,

matching the seven checkpoint fractions in time. Evaluate each chord point in the same final native background, producing `J_chord(lambda)`.

For every real path checkpoint also store geometric quantities:

- `||WO_oracle(t)-W0||`;
- path length accumulated from actual WO context steps;
- projection coordinate onto the endpoint chord;
- perpendicular distance from the endpoint chord.

The primary path-dependence contrast is paired per-image:

`J_path_fixed(t) - J_chord(t/1024)`.

A significant path-versus-chord difference is required before attributing functional importance to path geometry; gradient rotation alone is not causal evidence.

## Preregistered trajectory classes

D5D reports all component flags even if the final exclusive label is unresolved.

### D1 — smooth cumulative displacement

Supported when:

1. terminal `Delta J(t)` is significantly harmful on at least one primary endpoint;
2. primary endpoint point estimates evolve monotonically or near-monotonically with WO displacement;
3. no intermediate fixed-background path-versus-chord primary contrast is significant;
4. the chord control reproduces the overall deployment deterioration.

Interpretation: harmful utility is mainly a smooth consequence of cumulative endpoint displacement; path curvature is not required.

### D2 — rotation/path-dependent emergence

Supported only when both conditions hold:

1. at least one intermediate fixed-background path-versus-chord contrast is significant on a primary endpoint;
2. substantial fixed-probe gradient rotation co-occurs before or by deployment-harm onset (`R0 < 0.90` or checkpoint-sampled cumulative rotation > 45 degrees).

Interpretation: nonstationary gradient-field rotation/path geometry is implicated, but the wording remains `supported` rather than `proved causal`.

### D3 — threshold-like transition

Supported when a primary endpoint is non-harmful at all preceding saved checkpoints, becomes significantly harmful at a first checkpoint t*, and the paired adjacent increment at t* is significantly positive and more than twice the median absolute size of earlier point-estimate increments for that endpoint.

The chord curve is then used to distinguish a parameter-space boundary from a path-specific transition:

- similar chord onset: parameter-space threshold-like boundary;
- no corresponding chord onset: path/history dependence remains implicated.

If none of D1-D3 is satisfied cleanly, or more than one is simultaneously supported without a clear hierarchy, classify as `D_MIXED_OR_UNRESOLVED` and preserve all raw component flags.

## Interpretation boundary

D5D does not test or repair a new learner. It localizes how a deployment-harmful WO state emerges under the already certified D5B exact-WO rule. A correlation between gradient rotation and harm is insufficient by itself; path-versus-chord functional divergence is required before attributing functional importance to trajectory geometry.
