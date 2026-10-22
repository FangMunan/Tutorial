# Gate D6A.1 — state-dependent repair interaction audit

## Scientific question

D6A established that the first additive pathway-specific repair

`F_EM + WO_zero + bo_oracle`

does not satisfy repair sufficiency (`R0_NO_REPAIR_SUFFICIENCY`), even though `F_EM + bo_oracle` shows a beneficial deployment signal and the combined arm preserves masked prediction and improves aggregate rollout JS.

D6A.1 does **not** introduce a new learner. It replays the already defined D6A trajectories and asks why the benefit of `bo_oracle` is attenuated when `WO` plasticity is suppressed.

The two competing explanations are:

- **H_A — intrinsic bo-repair degradation:** `WO_zero` changes the future exact `bo` credit field, causing the `bo_oracle` parameter trajectory itself to acquire worse deployment utility.
- **H_B — external co-adaptive suppression:** the `bo_oracle` trajectory remains intrinsically beneficial, but other simultaneously adapted pathways in the combined system suppress that benefit.

The primary mechanistic question is therefore:

> Does repairing one pathway reshape another pathway's plasticity field in a way that changes the intrinsic deployment utility of the second pathway?

## Frozen prerequisite

The successful D6A run must be present and must certify:

- classification `R0_NO_REPAIR_SUFFICIENCY`;
- exact stream/RNG pairing;
- valid norm-matched `bo_oracle` finite-step descent;
- successful control calibration.

No D6A learning rule, norm, learning rate, boundary reset, target order, RNG stream, or Gate-A evaluator may be changed.

## Replayed systems

Two focal trajectories are replayed from the exact 2.0-context-epoch checkpoint with the exact D6A stream:

- **A = `F_EM_bo_oracle`**
- **B = `F_EM_WO_zero_bo_oracle`**

A third replay, **`F_EM`**, is used only as the time-matched common background for transplants. It is a reference trajectory, not a new intervention arm.

All three use the exact same 1024 contexts, image order, masks, mirror seeds, target order, and external RNG state.

## Checkpoints

Save systems after:

`0, 64, 128, 256, 512, 768, 1024` contexts.

At every checkpoint save `bo`, `WO`, and the complete system object needed for deterministic readout.

## A. Common-background bo utility test — primary gate

For each checkpoint `t`, use the same time-matched reference background `theta_F_EM(t)` and construct:

- `T_A_b(t) = theta_F_EM(t)[bo <- bo_A(t)]`
- `T_B_b(t) = theta_F_EM(t)[bo <- bo_B(t)]`

No other field is transplanted.

Run the frozen Gate-A evaluator with the same schedules and CRN field and record:

- `F61`
- `D_non61`
- `C61`
- `delta_P61`
- `CE`

Primary paired contrasts:

- `B_b - A_b`: positive means the `bo` state learned under `WO_zero` has worse intrinsic deployment utility;
- `A_b - F_EM` and `B_b - F_EM`: negative means the transplanted `bo` state is beneficial relative to the common background.

Bootstrap is paired over the same Gate-A images, 10,000 replicates.

The formal terminal intrinsic-degradation flag requires:

1. both primary point estimates (`F61`, `D_non61`) for `B_b - A_b` are non-negative; and
2. at least one primary 95% CI is entirely positive.

The formal retained-benefit flag requires:

1. no primary `B_b - A_b` CI is entirely positive; and
2. at least one primary `B_b - F_EM` CI is entirely negative.

This utility test is evaluated before interpreting any gradient-field divergence.

## B. Fixed-probe bo gradient-field divergence

Use a deterministic fixed validation probe batch disjoint from the frozen Gate-A 256 images. Validation and training sets are already disjoint; the probe is additionally selected from the complement of the Gate-A validation IDs.

Use the same deterministic probe construction as D5C:

- 256 probe images;
- ID seed `551901`;
- mask seed `552003`.

At every checkpoint compute the exact mean masked-token negative gradient

`g_A(t) = -dL_probe / dbo | theta_A(t)`

`g_B(t) = -dL_probe / dbo | theta_B(t)`.

Record:

- `C_AB(t) = cos(g_A, g_B)`;
- norm ratio `rho_AB(t) = ||g_B|| / ||g_A||`;
- `R_A(t) = cos(g_A(t), g_A(0))`;
- `R_B(t) = cos(g_B(t), g_B(0))`;
- successive-step cosine for A and B;
- cumulative checkpoint rotation `Omega_A`, `Omega_B`.

A formal terminal field-divergence flag is `C_AB(1024) < 0.95`. This threshold is preregistered as a descriptive effect-size gate, not a statistical p-value.

## C. Hybrid WO-swap test — direct versus mediated coupling

Own-state `g_A != g_B` is confounded by all co-adapted parameter differences. Therefore, at each checkpoint construct:

- `H_A<-B_WO = theta_A(t)[WO <- WO_B(t)]`
- `H_B<-A_WO = theta_B(t)[WO <- WO_A(t)]`

All non-`WO` fields remain in the recipient system.

Compute exact fixed-probe `bo` gradients:

- `g_A<-B_WO`
- `g_B<-A_WO`.

Define direction-capture scores using unit gradients:

`capture_A = 1 - ||hat(g_A<-B_WO) - hat(g_B)|| / ||hat(g_A) - hat(g_B)||`

`capture_B = 1 - ||hat(g_B<-A_WO) - hat(g_A)|| / ||hat(g_B) - hat(g_A)||`.

A value near 1 means swapping only `WO` reproduces most of the opposite trajectory's gradient direction; near 0 means it reproduces little; negative means the swap moves farther away.

For formal classification use the terminal mean of the two capture scores:

- direct-coupling support: mean capture >= 0.50;
- mediated-coupling support: mean capture <= 0.25;
- values in between are mixed/insufficient.

Also save the corresponding cosine-to-target and gradient-norm ratios; classification is not based on cosine alone.

## D. bo parameter-trajectory divergence

Relative to the common starting state `bo_0`, record:

- `Delta bo_A(t) = bo_A(t) - bo_0`
- `Delta bo_B(t) = bo_B(t) - bo_0`
- `C_b(t) = cos(Delta bo_A, Delta bo_B)`
- `D_b(t) = ||bo_A(t) - bo_B(t)||`
- path lengths of the two checkpoint-sampled trajectories.

These quantities connect:

`credit-field divergence -> parameter-path divergence -> deployment-utility divergence`.

They are mechanistic readouts, not independent evidence of causality.

## E. Full-system alignment

At the same checkpoints, evaluate the complete A and B systems with the frozen Gate-A evaluator. These results are used only after sections A-D, to locate where intrinsic `bo` utility and whole-network deployment diverge or decouple.

At terminal, the replayed A and B Gate-A arrays must reproduce the corresponding successful D6A artifact within `1e-7`, with proposal arrays exactly equal.

## Formal classification

### I1 — DIRECT_PATHWAY_COUPLING

Requires:

- terminal intrinsic `bo_B` degradation is present;
- terminal fixed-probe field divergence is present;
- terminal mean hybrid `WO`-swap direction capture >= 0.50.

Interpretation: `WO` state directly reorganizes the exact `bo` plasticity field strongly enough to account for much of the field divergence associated with worse intrinsic `bo` utility.

### I2 — MEDIATED_NETWORK_COUPLING

Requires:

- terminal intrinsic `bo_B` degradation is present;
- terminal fixed-probe field divergence is present;
- terminal mean hybrid direction capture <= 0.25.

Interpretation: the `WO_zero` intervention changes the `bo` credit field mainly through co-adapted network state rather than the instantaneous `WO` state alone.

### I3 — NO_INTRINSIC_BO_REPAIR_DEGRADATION

Requires:

- terminal intrinsic-degradation flag is false; and
- terminal retained-benefit flag is true.

Interpretation: the combined D6A R0 is not explained by destruction of the intrinsic `bo_oracle` terminal benefit; suppression lies elsewhere in the co-adapted system.

### I4 — MIXED_OR_INSUFFICIENT

All other valid outcomes, including strong gradient divergence without stable intrinsic utility divergence.

Technical/provenance failure is not I1-I4.

## Multi-objective deployment boundary

D6A's preregistered R0 verdict is unchanged. D6A.1 does not redefine it.

For later synthesis work, deployment utility should be represented as a vector rather than a single Gate-A scalar:

`U_deploy = (U_causal_rollout, U_distributional, U_predictive)`.

D6A.1 remains focused on the causal rollout component because the immediate question is the mechanism of the missing Gate-A repair benefit. Aggregate JS and masked CE are not used to reclassify D6A.

## Scientific boundary

D6A.1 is a replay/checkpoint/transplant/readout audit only. It must not:

- alter any learner update;
- tune learning rates or update norms;
- introduce a new optimizer;
- truncate training;
- add new resets;
- change `WO_zero` or `bo_oracle` definitions;
- change D6A's R0 verdict.

The strongest supported claim from I1/I2 would be **state-dependent coupled plasticity organization**, not a final implementable learning rule.