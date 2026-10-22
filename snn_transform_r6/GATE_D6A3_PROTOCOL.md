# Gate D6A.3 — W2–b2 suppressor interaction test

## Status and scope

D6A.2 is frozen as `S1_DOMINANT_SINGLE_SUPPRESSOR`, scientifically refined as: W2 is the dominant bidirectional suppressor; b2 is a secondary bidirectional suppressor; Q/Norm/FF1 are necessity-only co-suppressors; WO is protective. D6A.3 is an **eval-only terminal causal interaction test**. No training, learner change, E/M intervention, learning-rate change, or expanded block search is permitted.

The sole question is:

> Are W2 and b2 approximately additive, redundant/sub-additive, or synergistic suppressors of the beneficial bo_B state?

This pair is mechanistically motivated because the current learner updates W2 and b2 through the same M2 postsynaptic credit population, while both act on the same FFN output coordinate `y = z + W2 @ h + b2`.

## Frozen source states

Use the successful D6A terminal states:

- `F = F_EM`
- `A = F_EM_bo_oracle`
- `B = F_EM_WO_zero_bo_oracle`

and define:

- `H = F[bo <- bo_B]`
- `A_star = A[bo <- bo_B]`

`bo` remains fixed to `bo_B` in every D6A.3 causal hybrid.

## Frozen Gate-A coordinate system

Use the exact historical 256-image Gate-A coordinate system, schedules, shared CRN, proposal logic, and per-image effects used in D4–D6A.2. Primary endpoints remain:

- `F61`
- `D_non61`

Supportive endpoints:

- `C61`
- `delta_P61`

Secondary readouts are masked CE and rollout `global_js_nats`, `spatial_js_nats`, `pair_js_nats`.

Bootstrap inference uses paired 10,000-resample bootstrap on per-image effects.

## A0 provenance / closure gate

Before interpreting interactions:

1. D6A prerequisite must remain `R0_NO_REPAIR_SUFFICIENCY`.
2. D6A.1 prerequisite must remain `I3_NO_INTRINSIC_BO_REPAIR_DEGRADATION`.
3. D6A.2 prerequisite must remain `S1_DOMINANT_SINGLE_SUPPRESSOR`, with `W2` and `b2` both in `bidirectional_primary_blocks` and no pair search previously performed.
4. Reconstruct `H`, `H_W2`, `H_b2`, `B_-W2`, `B_-b2`, `A_star`, `A_star_W2`, and `A_star_b2` from frozen D6A terminal states and require their Gate-A per-image effects to reproduce the frozen D6A.2 arrays within `1e-7`.
5. Exact forward-field construction checks must verify that the pair hybrids change only `W2` and `b2`, with `bo == bo_B` exactly.

Any failure stops scientific interpretation.

## Sufficiency interaction

Construct:

- `H`
- `H_W2 = H[W2 <- W2_B]`
- `H_b2 = H[b2 <- b2_B]`
- `H_W2_b2 = H[W2 <- W2_B, b2 <- b2_B]`

For metric `m`, define single and joint suppression:

`S_W2,m = J_m(H_W2) - J_m(H)`

`S_b2,m = J_m(H_b2) - J_m(H)`

`S_12,m = J_m(H_W2_b2) - J_m(H)`

and the paired interaction:

`I^S_m = J_m(H_W2_b2) - J_m(H_W2) - J_m(H_b2) + J_m(H)`.

Interpretation:

- `I^S > 0`: supra-additive / synergistic suppression.
- `I^S = 0`: additive-compatible.
- `I^S < 0`: overlap / redundancy / sub-additivity.

Descriptive overlap coefficient, only when `S_W2 + S_b2 > 0`:

`R_S = S_12 / (S_W2 + S_b2)`.

`R_S` is descriptive and does **not** determine significance.

## Necessity interaction

Construct:

- `B`
- `B_-W2 = B[W2 <- W2_F]`
- `B_-b2 = B[b2 <- b2_F]`
- `B_-(W2+b2) = B[W2 <- W2_F, b2 <- b2_F]`

Define:

`N_W2,m = J_m(B) - J_m(B_-W2)`

`N_b2,m = J_m(B) - J_m(B_-b2)`

`N_12,m = J_m(B) - J_m(B_-(W2+b2))`

and:

`I^N_m = N_12,m - N_W2,m - N_b2,m`

which is algebraically:

`I^N_m = J_m(B_-W2) + J_m(B_-b2) - J_m(B_-(W2+b2)) - J_m(B)`.

Interpretation:

- `I^N > 0`: joint reversion releases more benefit than the sum of single reversions.
- `I^N = 0`: additive-compatible recovery.
- `I^N < 0`: overlapping / redundant recovery.

Descriptive coefficient, only when `N_W2 + N_b2 > 0`:

`R_N = N_12 / (N_W2 + N_b2)`.

## Cross-background pair confirmation

Only the pair endpoint is newly tested in the supportive background:

`A_star_W2_b2 = A_star[W2 <- W2_B, b2 <- b2_B]`.

Compare it with frozen `A_star`, and interpret alongside the already frozen D6A.2 single-block A-star effects. This test asks whether b2 becomes harmful specifically in the presence of W2_B, despite b2 alone being non-harmful in A-star.

## Formal classification

Classification uses only the four primary paired-interaction bootstrap CIs:

- sufficiency: `I^S_F61`, `I^S_D_non61`
- necessity: `I^N_F61`, `I^N_D_non61`

### `P1_REDUNDANT_W2_DOMINANT`

All four primary interaction CIs are entirely below zero. D6A.2 already establishes W2 as the dominant single suppressor; D6A.3 then establishes that b2 substantially overlaps that W2-side suppressive mode rather than adding an independent load.

### `P2_ADDITIVE`

All four primary interaction CIs include zero. This is an additive-compatible result. `R_S` and `R_N` are reported descriptively but do not alter the classification.

### `P3_SYNERGISTIC_FFN_SUPPRESSOR`

All four primary interaction CIs are entirely above zero. W2_B and b2_B then form a coordinated supra-additive FFN-output suppressive state.

### `P4_MIXED_BACKGROUND_DEPENDENT`

Any other pattern: endpoint disagreement, sufficiency/necessity disagreement, or mixed-sign/partially significant interaction. No forced unification is permitted.

## Interpretation boundary

D6A.3 does not establish a final learning rule. It only determines whether the terminal W2/b2 suppressive effects behave as one overlapping W2-dominant functional mode, two approximately additive harmful displacements, or a synergistic coordinated FFN-output subsystem. No FF1 pair test or broader factorial search is permitted in this gate.
