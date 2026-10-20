# Gate D3 — temporal credit-memory matching audit

Gate D3 is a diagnostic-only extension of the verified Gate D instrumentation. It does **not** change the candidate learner, learning rates, reciprocal dynamics, sparse masks, target stream, E/M persistence of the native trajectory, or the context update boundary.

The scientific question is whether persistent W2 credit-state memory outlives the correlation time of useful task credit, producing stale/cancelling motion in the observed low-dimensional W2 plastic channel.

## D3A — temporal credit kernel

Measure native W2 update `u_t` against exact instantaneous oracle `g_t` at lags `0,1,2,4,8`, plus oracle/oracle and native/native lag correlations. In a fixed 256-D CountSketch dominant mode, fit a descriptive FIR model using lags 0..8 and report held-out R² for the full model versus lag-0 only. Lag coefficients are descriptive, not causal proof.

## D3B — target-order sensitivity

At deterministic probe contexts, freeze the pre-context state, reciprocal result, forward cache, target set and exact context oracle. Replay only the direct W2 `E2/M2` pathway under native, reverse, and deterministic random target permutations. Report pairwise update cosine, normalized order variance, and alignment to the exact oracle. This measures W2-direct order sensitivity, not whole-network order sensitivity.

## D3C — persistence decomposition

On clones only, compare:

- native
- no_E_within
- no_M_within
- no_EM_within
- fresh_E_context
- fresh_M_context
- fresh_EM_context

The first set removes within-context carry while preserving inherited pre-context state; the second resets inherited cross-context state to equilibrium but allows normal within-context persistence. These are counterfactual diagnostics, not candidate learners.

## D3D — frozen future-context oracle

At each probe pre-context state, freeze candidate parameters and construct `G_H = -d/dW2 sum_{k=0}^{H-1} L_{t+k}(W_t)` for `H=1,2,4,8` over the native future GT-visible mask stream, with no intermediate learning. Decompose the native update relative to `G_1` and measure alignment of its instantaneous-orthogonal component with `G_H`. This is a frozen-parameter future-context oracle, not BPTT.

## Hard parity gates

The instrumented native trajectory runs in lockstep with an uninstrumented reference and must be bitwise identical at contexts 256, 512 and 1024 (or final context if shorter), with equal external RNG state. At every probe context:

- direct W2 E/M replay must match the actual native W2 step within `2e-14` max absolute error;
- independent `G_1` reconstruction must match the native audit oracle within `2e-14`.

Any failure aborts the workflow and invalidates scientific interpretation.

## Default run

`contexts=1024`, `probe_contexts=64`, `order_permutations=8`.

Artifact: `SNN_TRANSFORM_GATE_D3_TEMPORAL_CREDIT_MEMORY`.
