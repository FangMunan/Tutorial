# Gate D6A — mechanism-derived pathway-specific repair sufficiency test

## Status

Diagnostic synthesis experiment only. This gate does **not** claim a final biologically/local implementable learner. It asks whether the mechanisms isolated in D3–D5 are jointly sufficient to improve autonomous deployment while preserving ordinary masked-token competence.

No changes are permitted on `r6b4-full-forward-plasticity`; this protocol runs only on `gate-d-plasticity-spectrum-audit`.

## Scientific question

Starting from the exact same 2.0-context-epoch R6B3 checkpoint and the same 1024-context continuation stream, does pathway-specific plasticity organization improve the learner?

The evidence motivating the repair is frozen as:

- `W2`: D3/D3E — inherited cross-context `E2/M2` state increases stale credit and trajectory cancellation; boundary cleaning improves W2 credit geometry and terminal W2 utility.
- `bO`: D5B — exact local direction produces a beneficial terminal `bO` state in common backgrounds; native MSDL directional approximation remains causally limiting.
- `WO`: D5B/D5D — unrestricted exact local descent is deployment-harmful and its utility is trajectory-dependent; D6A therefore uses **WO-zero only as a causal repair control**, not as the claimed final rule.

## Frozen arms

All arms use the exact same 1024 training contexts, same image order, same mask sets, same target order, same mirror seeds, and same external RNG state.

1. `N`
   - Original aux1.0 learner.
   - No `F_EM` boundary reset.

2. `F_EM`
   - Native learner with the D3E boundary intervention: reset inherited `E2/M2` state at every context boundary.
   - No O-parameter intervention.

3. `F_EM_WO_zero`
   - Same `F_EM` boundary cleaning.
   - O-side eligibility/modulatory dynamics evolve normally.
   - The native `WO` proposal is computed but the **WO parameter update is not applied**.
   - `bo` remains native MSDL.

4. `F_EM_bo_oracle`
   - Same `F_EM` boundary cleaning.
   - `WO` remains native MSDL.
   - At each context boundary, compute the arm-specific native `bo` proposal norm and replace only its direction by the exact support-matched negative masked-token gradient:

     `u_bo^oracle = ||u_bo^native|| * G_bo / (||G_bo|| + eps)`.

   - O-side E/M dynamics remain native and are never reset/replaced.
   - This arm is an **upper-bound diagnostic**, not a final local learner.

5. `F_EM_WO_zero_bo_oracle`
   - Combined mechanism-derived repair.
   - `F_EM` boundary cleaning.
   - `WO` parameter update suppressed, while its O-side states still evolve.
   - `bo` uses the arm-specific norm-matched exact local descent direction.

No learning-rate sweep, no optimizer change, no new state reset, no truncation, and no other parameter update is modified.

## Context-boundary exact `bO` oracle

The exact local objective for context `c` is

`L_c = (1/n_c) sum_j -log p(y_j | x_c)`.

The exact descent direction is

`G_bo,c = - d L_c / d bo`.

The D5A finite-difference-validated oracle implementation is reused. The oracle is accumulated over the same cached pre-update context and applied only at the native `apply_update()` boundary.

The norm is **arm-specific**: the native MSDL accumulator of the current oracle arm at the current context supplies the plastic budget. No norm is borrowed from another trajectory.

## Finite-step oracle interpretability gate

For both `bO`-oracle arms, record on every context:

- `Delta L_c^native-bo`
- `Delta L_c^oracle-bo`

with all non-`bo` parameters fixed for the hypothetical local probe.

The oracle arm is interpretable only if:

- `P(Delta L_c^oracle-bo < 0) > 0.95`,
- `mean(Delta L_c^oracle-bo) < 0`, and
- the paired-bootstrap 95% CI of `Delta L_c^oracle-bo - Delta L_c^native-bo` is entirely below zero.

Failure of this condition is diagnostic, not workflow failure, and blocks a claim that `bO` credit fidelity was cleanly improved.

## Provenance and parity

Hard requirements:

- exact frozen 2.0-epoch checkpoint identity;
- same stream and external RNG across all five arms;
- `N` plain/instrumented bitwise parity at contexts 256/512/1024;
- `F_EM` plain/instrumented bitwise parity at contexts 256/512/1024;
- fresh `N` Gate-A calibration to frozen D3E `N` arrays;
- fresh `F_EM` Gate-A calibration to frozen D3E `F_EM` arrays;
- D5A oracle self-test passes;
- successful D5B artifact is present and its `bO` oracle finite-step result is certified.

## Evaluation coordinate system

Gate-A remains frozen:

- 256 validation images;
- 3 rounds;
- historical schedule;
- common random numbers;
- paired per-image bootstrap, 10,000 resamples.

### Primary deployment endpoints

- `F61`
- `D_non61`

For each repair arm, report paired contrasts against both `N` and `F_EM`.

The combined arm has strong primary repair only if **both primary endpoints** have 95% CIs entirely below zero versus **both** `N` and `F_EM` in the `J_arm - J_reference` convention.

### Supportive Gate-A endpoints

- `C61`
- `delta_P61`
- ordinary Gate-A CE

### Generative distribution guardrail

Record with the same frozen generation seed and sample count:

- global JS;
- spatial JS;
- pair JS.

For the combined arm, define a conservative non-degradation guardrail relative to the *better* of `N` and `F_EM` for each metric:

`JS_combined <= JS_best_baseline * 1.02 + 1e-6`.

This 2% tolerance is a diagnostic Monte-Carlo guardrail, not a claim of equivalence.

### Ordinary prediction guardrail

Evaluate masked CE at mask ratios:

- 0.25
- 0.5
- 0.9
- 1.0

For each ratio, the combined arm must satisfy:

`CE_combined <= CE_best_baseline * 1.01 + 1e-8`.

Thus D6A requires deployment repair without a material loss of ordinary masked prediction.

## Component attribution

For `F_EM_WO_zero`, `F_EM_bo_oracle`, and the combined arm, report contrasts versus `F_EM`.

For every Gate-A metric, also report the descriptive interaction

`I = J_combined - J_WOzero - J_bOoracle + J_FEM`

with paired bootstrap. `I=0` corresponds to additive repair effects in the frozen Gate-A coordinate system.

## Pre-registered classification

### R1 — strong pathway-specific repair sufficiency

Requires:

1. both `bO` oracle arms pass the finite-step interpretability gate;
2. the combined arm improves both `F61` and `D_non61` significantly versus both `N` and `F_EM`;
3. all three JS guardrails pass;
4. all four masked-CE guardrails pass.

Interpretation:

> Mechanism-derived pathway-specific plasticity organization is sufficient to produce a better autonomous learner while preserving ordinary masked-token competence.

This **does not** establish WO-freeze or exact-`bO` gradient as the final implementable learning rule.

### R2 — primary repair with competence/quality tension

Primary deployment criterion passes, but one or more JS or masked-CE guardrails fail.

### R3 — partial pathway repair

At least one primary endpoint improves significantly versus both references, but the full R1 primary criterion is not met.

### R0 — no repair sufficiency

No primary endpoint shows robust improvement versus both references.

### RX — oracle interpretation invalid

A `bO` oracle arm fails its finite-step interpretability gate. Deployment numbers remain descriptive only.

## Interpretation boundary

D6A is a synthesis/upper-bound experiment. A successful combined arm would justify the principle of pathway-specific plasticity organization. It would then motivate:

- D6B: replace exact `bO` credit by an allowed local/dynamical approximation;
- D6C: replace `WO` freeze by a state/history-dependent plasticity admissibility operator `A_WO(t)`.

No `P_safe`, low-rank projector, or final biological learning rule is assumed in D6A.