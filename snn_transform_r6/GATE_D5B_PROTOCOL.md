# Gate D5B — Norm-matched exact-local-descent versus rollout-utility test

## Frozen scientific question

D5A showed that the F_EM boundary intervention strongly improves W2 local-credit alignment and temporal organization, while the O pathway (WO and bo) retains positive exact-gradient alignment and does not develop worse O-side temporal organization even though D4B established that the terminal WO^F and bo^F displacements are deployment-harmful. D5B asks the narrower causal question:

> If the O pathway is forced to follow the exact negative gradient of the same masked-token objective, while preserving its arm-specific native plastic budget and all O-side E/M dynamics, does autonomous rollout improve, or does the O pathway remain deployment-harmful?

D5B is a **diagnostic counterfactual**, not a candidate learner. It does not alter the masked-token objective, context stream, F_EM boundary reset, O-side E/M dynamics, reciprocal dynamics, sparse supports, target order, mirror seeds, or any non-O learning rule.

## Starting state and paired stream

All arms start from the exact R6B3 2.0-context-epoch checkpoint, upgraded once to the frozen R6B4 aux_lr_scale=1.0 system. Every arm replays the exact next 1024 training contexts with identical image indices, masks, target order, mirror seeds, and external RNG. Before every context, every arm receives the already-frozen F_EM boundary intervention: E2/M2 are reset to canonical fresh initialization. O-side E/M states are never reset or replaced.

The five arms are:

- `F_native`: ordinary F_EM continuation; all parameter updates remain native MSDL.
- `F_WO_oracle`: only the final WO context update direction is replaced by exact local descent.
- `F_bo_oracle`: only the final bo context update direction is replaced by exact local descent.
- `F_O_oracle`: WO and bo are both replaced component-wise by exact local descent.
- `F_O_zero`: WO and bo parameter updates are suppressed, while their E/M states continue to evolve normally.

`F_native` must pass same-run plain/instrumented bitwise parity at contexts 256, 512, and 1024, and its Gate-A terminal phenotype must reproduce the frozen provenance-correct D3E F_EM result within 1e-7 with exact proposal IDs.

## Context-boundary exact oracle

For each immutable masked context c, define

`L_c = (1/n_c) sum_j -log p(y_j | x_c)`.

All target predictions are cached before any forward parameter update. D5B therefore sums the exact per-target negative gradients from D5A on the same pre-update model state:

`G_WO,c = -(d L_c / d WO) * maskO`

`G_bo,c = -(d L_c / d bo)`.

The native O proposals are computed from the arm's own current MSDL accumulators at that same context boundary:

`U_WO,c^native = lrO * ao / n_c`

`U_bo,c^native = aux_lr_scale * lrO * abo / n_c`.

Norm matching is **arm-specific**. No norm is borrowed from F_native after trajectories diverge.

For WO:

`Uhat_WO,c = ||U_WO,c^native||_F * G_WO,c / (||G_WO,c||_F + eps)`.

For bo:

`Uhat_bo,c = ||U_bo,c^native||_2 * G_bo,c / (||G_bo,c||_2 + eps)`.

The joint O oracle uses component-wise norm matching, preserving the two native plastic budgets separately. It does not concatenate WO and bo into one global norm.

The replacement occurs only at the existing `apply_update()` boundary after the entire masked context has contributed. All O-side E/M populations have already followed their native dynamics for the full context.

## Finite-step realized-descent gate

A gradient direction is not automatically a realized finite-step descent. For every context and every arm, D5B evaluates the fixed-context masked-token CE before and after hypothetical native-O and oracle-O updates while holding all other parameters fixed at their pre-update values.

For each relevant component x in {WO, bo, O}, record:

`delta L_c^native(x) = L_c(theta + U_x^native) - L_c(theta)`

`delta L_c^oracle(x) = L_c(theta + Uhat_x) - L_c(theta)`.

An oracle arm is interpretable as a realized local-descent trajectory only if all of the following hold for its active component:

1. `P(delta L_c^oracle < 0) > 0.95` across the 1024 contexts;
2. `mean(delta L_c^oracle) < 0`;
3. the paired-bootstrap 95% CI of `delta L_c^oracle - delta L_c^native` lies entirely below zero.

If the joint O oracle fails this gate, the formal D5B classification is `L4_FINITE_STEP_ORACLE_INCONCLUSIVE`; rollout results are not used to claim objective mismatch.

The recomputed pre-update fixed-context CE must reproduce the cached pre-update CE to <2e-10, and applied oracle/zero O-step norms must match their declared budgets to <2e-12 absolute error.

## Local-objective endpoints

For each arm, save per-context mean NLL, sum of context losses, event-weighted NLL, the native and oracle fixed-context delta-L probes, exact-gradient norms, native proposal norms, and norm-matching audits.

The primary local causal comparison is not cosine. It is whether the exact-gradient proposal realizes a larger reduction in the same local masked-token objective than the arm-specific native MSDL proposal.

## Frozen deployment endpoints

Gate-A remains unchanged from D3E/D4:

Primary endpoints:

- `F61`
- `D_non61`

Supportive endpoints:

- `C61`
- `delta_P61`

`CE` is retained in the causal arrays. The evaluator uses the same frozen 256 validation images, three-round historical schedule, shared CRN field, self-versus-teacher construction, and paired per-image bootstrap.

Secondary endpoints are ordinary masked-token evaluation at mask ratios 0.25, 0.5, 0.9, and 1.0, plus autonomous generation global/spatial/pair JS metrics. These remain secondary to the primary Gate-A causal endpoints.

## O-zero comparator

`F_O_zero` suppresses only WO/bo parameter updates. It does not freeze or reset O-side E/M dynamics. This supplies the deployment counterfactual "no O parameter adaptation over these 1024 contexts" and prevents an oracle arm from being called deployment-aligned merely because it improves relative to a harmful native O trajectory while remaining worse than no O adaptation.

## Common-background terminal utility

Full oracle-training arms can induce adaptation elsewhere through subsequent contexts. Therefore D5B also evaluates terminal O states in fixed backgrounds.

Using the provenance-correct D3E terminal systems, construct two fixed backgrounds:

- `N` terminal background;
- `W2^F` beneficial background: N terminal background with only W2 replaced by the D3E F_EM W2.

Into each background transplant only WO/bo from:

- the D3E F_EM O state (`OF`) as a known-harm sanity control;
- the D5B joint exact-gradient O terminal state (`Ooracle`);
- the D5B O-zero terminal state (`Ozero`).

Component WO-oracle and bo-oracle donor states are also evaluated diagnostically. These common-background tests estimate terminal O deployment utility without allowing the rest of the oracle-training trajectory to co-adapt.

## Preregistered decision matrix

The formal classification is centered on `F_O_oracle`.

### L1 — Strong local-objective/deployment mismatch

The joint oracle passes the finite-step realized-descent gate, yet it remains significantly worse than O-zero on at least one primary deployment endpoint and its O terminal state is significantly harmful relative to native O in both N and W2^F common backgrounds.

Interpretation: optimizing the masked-token local objective more exactly in the O subspace is itself deployment-misaligned for autonomous rollout under this system/protocol.

### L2 — Residual MSDL credit error

The joint oracle passes the local gate, significantly rescues at least one primary endpoint versus F_native, and shows neither significant harm versus O-zero nor significant common-background O harm.

Interpretation: positive average D5A alignment was insufficient; residual MSDL directional error mattered causally.

### L3 — Mixed mechanism

The joint oracle passes the local gate and significantly rescues F_native, but a significant residual deployment penalty remains versus O-zero and/or in both common backgrounds.

Interpretation: both residual credit error and local-objective/deployment mismatch contribute.

### L4 — Finite-step oracle inconclusive

The joint same-norm exact-gradient proposal fails the realized-descent gate. No objective-mismatch conclusion is allowed. A later smaller-step directional probe would be required.

Any pattern not satisfying L1-L4 exactly is retained as `L_UNRESOLVED_REQUIRES_INTERPRETATION` rather than forced into a mechanism class.

## Interpretation boundary

Even an L1 result would **not** show that gradient descent is generally unsuitable for generation. It would show the narrower claim that, in this learner and deployment protocol, the same masked-token local objective is not uniformly deployment-aligned across plastic parameter subspaces. This motivates separating local-credit alignment from deployment utility in the plasticity spectrum.
