# Gate D7A — causal upper bound for preventing W2 re-adaptation

## Status

Pre-registered trajectory-level synthesis gate. D6 diagnosis is closed. D7A introduces no candidate final local learning rule; it is a privileged causal upper bound.

## Scientific question

D6 established that:

- F_EM boundary cleaning improves the W2 trajectory/state;
- after the coupled WO-zero + exact-bo repair changes the learning environment, W2 re-adapts;
- the resulting W2^B terminal state is the dominant suppressor of the otherwise beneficial bo^B state;
- b2 is a smaller causal contributor that largely overlaps the same W2-dominant FFN-output suppressive mode;
- no additional W2×FF1 terminal transplant is required for accounting closure.

D7A asks one question only:

> If W2 is prevented from leaving the already-established beneficial time-matched F_EM trajectory throughout the 1024-context combined-learning episode, is that sufficient to rescue the combined learner?

This is stronger than a terminal W2 transplant. It tests trajectory-level causal sufficiency.

## Frozen start and stream

All arms start from the exact frozen 2.0-context-epoch R6B3 checkpoint used by D6A. They use the exact same 1024 image indices, context masks, target order, mirror seeds, and external RNG stream.

The historical Gate-A coordinate system is unchanged:

- 256 frozen validation images;
- frozen historical MaskGIT schedule;
- frozen shared CRN uniforms;
- identical proposal logic and per-image effects;
- 10,000 paired bootstrap samples.

## Core arms

Only three arms are permitted.

1. **F = F_EM shadow/safe baseline**
   - D3E/D6A F_EM boundary cleaning;
   - native forward plasticity otherwise unchanged.

2. **B = combined failure baseline**
   - exactly the D6A `F_EM_WO_zero_bo_oracle` learner;
   - F_EM boundary cleaning;
   - WO parameter update suppressed while O-side dynamical state evolves;
   - exact norm-matched bo oracle direction;
   - W2 learns normally in the changed coupled environment.

3. **C = B_W2_safe_clamp**
   - identical to B in every respect except the privileged W2 intervention below.

No W2×FF1 arm, no broad block search, and no projector/admissibility rule are allowed in D7A.

## Time-matched safe-W2 clamp

Maintain F and C synchronously on paired copies of the same external stream.

At the **start** of every context t, C must already satisfy

\[
W_2^C(t-1)=W_2^F(t-1).
\]

F then executes its ordinary F_EM context update. C executes the combined B context from the same pre-context safe W2 state while all of its other parameters follow the combined trajectory. After C completes the context update, record the unconstrained post-update W2 displacement and then enforce

\[
\boxed{W_2^C(t)\leftarrow W_2^F(t)}.
\]

Only the W2 parameter is overwritten. The following remain C-specific and evolve normally:

- E2/M2 and all other dynamical state;
- bo oracle state and bo parameter trajectory;
- WO-side states, with the same WO-zero parameter intervention as B;
- Q/K/V, FF1, b2, readout, norm, embeddings, and all other parameters.

This preserves beneficial time-matched W2 adaptation instead of setting \(\Delta W_2=0\).

The clamp must be exact after every context. Record at least:

- pre-clamp \(\|W_2^{C,post-native}(t)-W_2^F(t)\|\);
- post-clamp maximum absolute error;
- cumulative clamp correction path length;
- baseline \(\|W_2^B(t)-W_2^F(t)\|\) at checkpoints 64, 128, 256, 512, 768, 1024.

## Provenance gates

Before scientific interpretation:

1. D6A source classification must remain `R0_NO_REPAIR_SUFFICIENCY` with valid bo oracle.
2. D6A.1 must remain `I3_NO_INTRINSIC_BO_REPAIR_DEGRADATION`.
3. D6A.2 must remain `S1_DOMINANT_SINGLE_SUPPRESSOR`.
4. D6A.3 must remain `P4_MIXED_BACKGROUND_DEPENDENT`.
5. Exact 2.0-epoch checkpoint identity must pass.
6. F and B replay streams and external RNG must be identical to each other and to C.
7. Replayed F and B terminal model fields must reproduce the successful D6A terminals tightly (dtype/shape exact; numeric max-abs < 1e-8).
8. Replayed F and B Gate-A arrays must reproduce D6A with <1e-7 numeric error and exact proposal arrays.
9. C must satisfy exact post-context W2 clamp at every context.
10. WO-zero implementation must remain exact in B and C.
11. Exact bo oracle finite-step interpretability must pass in both B and C. If C loses this property, deployment results are descriptive only and the formal result is `TX_ORACLE_INTERPRETATION_INVALID`.

## Primary deployment endpoints

Unchanged from D6:

\[
\boxed{F_{61},\qquad D_{non61}}
\]

Lower is better.

The primary causal contrast is

\[
\Delta J_m^{clamp}=J_m(C)-J_m(B).
\]

A negative paired-bootstrap CI entirely below zero means that preventing W2 re-adaptation improves that endpoint.

Also report

\[
J_m(C)-J_m(F)
\]

and the descriptive gap-closure fraction, when \(J_m(B)-J_m(F)>0\):

\[
\boxed{R_m^{close}=\frac{J_m(B)-J_m(C)}{J_m(B)-J_m(F)}}.
\]

Interpretation:

- 0 = no B-to-F gap closed;
- 1 = mean effect reaches the F mean;
- >1 = mean effect exceeds F in the beneficial direction.

This ratio is descriptive and is not an equivalence test.

Supportive Gate-A endpoints remain:

- C61;
- delta_P61;
- CE.

## Multi-objective guardrails

D7A keeps the frozen three-part deployment view:

\[
U^{deploy}=(U_{causal\ rollout},U_{distributional},U_{predictive}).
\]

### Ordinary masked prediction

Evaluate mask ratios 0.25, 0.5, 0.9, 1.0 under the same deterministic mask seed for F, B, and C.

For every ratio require

\[
CE_C\le \min(CE_F,CE_B)\times1.01+10^{-8}.
\]

### Autonomous distribution

Using the same generation seed and 256 samples, evaluate global, spatial, and pair JS. Require for each

\[
JS_C\le \min(JS_F,JS_B)\times1.02+10^{-6}.
\]

The tolerances are diagnostic Monte-Carlo guardrails, not equivalence claims.

## Pre-registered classifications

### T1_STRONG_W2_TRAJECTORY_RESCUE

Requirements:

1. bo oracle interpretation valid in B and C;
2. both primary contrasts C-B have 95% paired-bootstrap CIs entirely below zero;
3. all masked-CE guardrails pass;
4. all rollout-JS guardrails pass.

Interpretation:

> Preventing W2 re-adaptation along the privileged time-matched safe trajectory is sufficient to rescue both primary deployment endpoints without sacrificing ordinary masked prediction or the frozen distributional guardrails.

This is a causal upper bound, not a final learning rule.

### T2_PRIMARY_RESCUE_WITH_GUARDRAIL_TENSION

Both primary C-B CIs are entirely negative, but at least one masked-CE or JS guardrail fails.

### T3_PARTIAL_W2_TRAJECTORY_RESCUE

Exactly one primary C-B CI is entirely negative.

### T0_NO_W2_TRAJECTORY_RESCUE

Neither primary C-B CI is entirely negative.

### TX_ORACLE_INTERPRETATION_INVALID

The exact bo oracle fails its finite-step interpretability gate in B or C. Deployment results may still be reported descriptively, but no trajectory-clamp mechanism conclusion is allowed.

## Interpretation boundaries

A T1/T2 result supports trajectory-level causal sufficiency of preventing W2 re-adaptation under a privileged reference trajectory. It does **not** establish:

- a local or biological implementation;
- a static safe subspace;
- a fixed projector;
- that the final admissibility signal is parameter-space distance;
- that W2 is the only state affected by future local rules.

If D7A succeeds, the next problem is to replace the privileged shadow trajectory by a local/dynamical admissibility mechanism, preferably defined in functional FFN-output coordinates rather than raw W2 distance.

If D7A fails, terminal W2^B remains a causal suppressor but is not sufficient as the unique trajectory-level explanation of the combined learning failure; only then should mediated pathways such as FF1/Q be revisited.
