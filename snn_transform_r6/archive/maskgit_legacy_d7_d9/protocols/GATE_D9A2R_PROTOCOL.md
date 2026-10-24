# Gate D9A.2-R — matched trajectory-relative functional displacement audit

## Status

This gate is **diagnostic only**. It performs no learning-rule change, no activation intervention, no plastic-velocity projection, no controller, no parameter search, and no new functional-mode discovery.

Prerequisites are frozen:

- D8A = `I1_SHARED_LOW_DIMENSIONAL_FUNCTIONAL_CANDIDATE`, selected `k=1`.
- D8B = `B1_BIDIRECTIONAL_CAUSAL_HARMFUL_FUNCTIONAL_MODE`.
- D9A = `R0_NO_HARD_FUNCTIONAL_TRAJECTORY_RESCUE`.
- D9A.1 = `V0_NO_HARMFUL_PLASTIC_VELOCITY_EXCESS`.

The rank-1 y-space basis `v` is the exact D8A/D8B frozen basis. It is not refit, rotated, sign-flipped, or re-estimated here.

## Scientific question

D8B established terminal causality for the relative displacement between the D7A safe-W2 trajectory state `C` and the rescued terminal hybrid `R`. D9A and D9A.1 showed that neither absolute occupancy `v^T y` nor simple context-independent positive single-update plastic velocity is a valid trajectory controller.

D9A.2-R asks only:

> Is the terminal causal mode already present as a stable time-matched `C_t - R_t` functional coordinate during training, or is it created/consolidated by the trajectory itself?

## Frozen trajectories and time-matched reference

Replay the exact D7A synchronized 1024-context trajectories:

- `F_t`: F_EM reference.
- `B_t`: combined failure learner.
- `C_t`: combined learner with W2 clamped after every context to time-matched `F_t`.

At checkpoints

`64, 128, 256, 512, 768, 1024`

construct an analysis-only hybrid

`R_t = B_t[W2 <- W2^{F_t}]`.

Because D7A enforces `W2^{C_t} = W2^{F_t}` after every context, D9A.2-R requires bit-exact

`W2^{C_t} = W2^{R_t}`

at every checkpoint. `R_t` never participates in learning and cannot affect F/B/C state or RNG.

## Probe separation

Functional geometry uses exactly the 128-image D9A.1 diagnostic set, including its frozen masks and provenance hash. This set is already certified disjoint from:

- the historical 256-image Gate-A set, and
- the 256-image D8A discovery set.

No new probe selection is allowed.

Deployment-gap measurement uses only the frozen historical Gate-A/CRN arrays. Geometry and deployment evidence therefore remain on separate data.

## Primary functional geometry

For every checkpoint, on the frozen D9A.1 diagnostic contexts, form

`DeltaY_t = Y_{C_t} - Y_{R_t}`

in y-space.

Report:

1. Signed frozen-mode amplitude

`a_t = mean_rows(DeltaY_t v)`.

2. Frozen-mode energy capture

`rho_v(t) = ||DeltaY_t v v^T||_F^2 / ||DeltaY_t||_F^2`.

3. Total difference energy

`E_t = ||DeltaY_t||_F^2` and `||DeltaY_t||_F`.

4. Checkpoint-specific dominant direction

Let `v_t` be the top right-singular vector of uncentered `DeltaY_t`. Report

`c_t = |v_t^T v|`

and `EV1_t = sigma_1^2 / sum_j sigma_j^2`.

5. Mean-displacement energy fraction, as a supportive offset-like diagnostic.

No centering is used for the primary geometry because D8A established that mean displacement can itself be part of the causal mode.

## Mode-presence threshold

Do not invent new geometric scales. Reuse the frozen D8A engineering thresholds:

- `rho_v(t) >= 0.50`, and
- `c_t >= 0.80`.

A checkpoint has `frozen_mode_present = true` only if both hold.

These are engineering classification thresholds, not biological constants.

## Deployment-gap trajectory

At every checkpoint, evaluate `C_t` and `R_t` with the exact historical Gate-A/CRN protocol, regulator/intervention OFF.

For `m in {F61, D_non61}` define

`L_m(t) = J_m(C_t) - J_m(R_t)`.

Report paired-bootstrap mean and 95% CI at every checkpoint. Positive `L_m` is the historical harmful C-over-R direction.

The deployment gap is descriptive for trajectory timing; it does not redefine the frozen mode.

## Formal trajectory classes

The six-checkpoint binary sequence of `frozen_mode_present` is the primary classifier. Total energy and deployment gaps are required context but do not permit post-hoc threshold changes.

### T1_FIXED_RELATIVE_MODE_ACCUMULATION

- mode present at both checkpoints 64 and 128;
- mode present at least 5 of 6 checkpoints; and
- mode present at 1024.

Interpretation: the terminal causal mode is already a stable relative coordinate early in training.

### T2_LATE_MODE_CONSOLIDATION

- mode absent at both 64 and 128;
- mode present at both 768 and 1024; and
- at least one pre-768 checkpoint has nonzero/material functional difference energy (`E_t > numerical tolerance`).

Interpretation: distributed earlier differences later consolidate onto the terminal frozen mode.

### T4_TERMINAL_ONLY_CAUSAL_GEOMETRY

- mode present at 1024;
- mode absent at every checkpoint before 1024.

Interpretation: the D8B direction is supported as a terminal causal geometry but not as a resolved training-state coordinate on these checkpoints.

### T3_ROTATING_OR_PATH_DEPENDENT_RELATIVE_GEOMETRY

Any remaining pattern, including nonmonotonic appearance/disappearance or persistent mismatch between checkpoint-specific dominant directions and the terminal basis.

Interpretation: relative functional geometry is path-dependent/rotating rather than a single fixed trajectory coordinate.

## A0 provenance gates

Before scientific interpretation, require:

1. D8A/D8B frozen basis identity and unit norm.
2. D9A.1 diagnostic IDs/tokens/hash reproduced exactly.
3. F/B/C synchronized stream and external RNG exactness.
4. D7A post-context W2 clamp exact at every context.
5. Terminal F/B/C model replay to frozen D7A within the existing D7A tolerance.
6. Terminal checkpoint `R_1024` and `C_1024` Gate-A arrays reproduce frozen D8B ordinary R/C arrays within `1e-7`, with proposals exact.
7. Stateless y extractor parity below the existing D9A.1 tolerance.
8. No training state may be mutated by geometry or Gate-A diagnostic evaluation; all diagnostics operate on clones/snapshots.

Failure of any A0 gate invalidates T1-T4 classification.

## Outputs

Save at minimum:

- `D9A2R_SUMMARY.json`
- `D9A2R_CHECKPOINT_GEOMETRY.csv`
- `D9A2R_DEPLOYMENT_GAPS.csv`
- `D9A2R_DIAGNOSTIC_IDS.csv`
- `D9A2R_ARRAYS.npz`
- `D9A2R_MANIFEST.json`

## Decision boundary

D9A.2-R is not a controller gate. No learning-rule implementation is authorized automatically by T1-T4. The result determines what kind of next causal test is scientifically justified:

- T1 motivates cumulative relative-displacement regulation.
- T2 motivates mode-formation/consolidation analysis before any clamp.
- T3 motivates explicitly evolving/path-dependent functional geometry.
- T4 argues against directly turning the terminal D8B vector into an online training coordinate.
