# Gate D6A.2 — terminal suppressor mapping of the beneficial `bo_B` state

## Scientific question

D6A classified the first additive repair as `R0_NO_REPAIR_SUFFICIENCY`. D6A.1 then classified the interaction audit as `I3_NO_INTRINSIC_BO_REPAIR_DEGRADATION`: the `bo` state learned in the combined arm remains intrinsically beneficial when transplanted onto the common `F_EM` background, even though the full combined system expresses little of that benefit.

D6A.2 asks one question only:

> Which terminal co-adapted forward-parameter block(s) prevent the beneficial `bo_B` displacement from being expressed in the full combined system?

D6A.2 is **eval-only**. It performs no training, no plastic update, no E/M manipulation, no learner repair, and no new optimization.

## Frozen terminal systems

Use the successful D6A terminal systems:

- `F = F_EM`
- `A = F_EM_bo_oracle`
- `B = F_EM_WO_zero_bo_oracle`

Define the primary good common-background reference

`H = F[bo <- bo_B]`.

`bo` is fixed to `bo_B` in every D6A.2 causal hybrid. It is never included in a tested suppressor block.

D6A.1 must be present and classified `I3_NO_INTRINSIC_BO_REPAIR_DEGRADATION` before D6A.2 is interpreted.

## A0 — forward-partition closure gate

The frozen forward parameter set must remain exactly:

`{tok,pos,WQ,WK,WV,WO,W1,W2,WR,bq,bk,bv,bo,b1,b2,gamma2,beta2}`.

With `bo` excluded because it is fixed to `bo_B`, the remaining 16 fields must be covered exactly once by the following ten blocks:

- `Q = (WQ,bq)`
- `K = (WK,bk)`
- `V = (WV,bv)`
- `O_W = (WO)`
- `Norm = (gamma2,beta2)`
- `FF1 = (W1,b1)`
- `W2 = (W2)`
- `b2 = (b2)`
- `R = (WR)`
- `Emb = (tok,pos)`

Before any scientific interpretation, verify exact forward-state closure:

1. `H[all non-bo blocks <- B] == B` for every forward parameter;
2. `B[all non-bo blocks <- F] == H` for every forward parameter.

Any failure is `S4_PARTITION_FAILURE` and terminates interpretation.

## Primary causal map

The primary background is `H`, because D6A.1 already established that `bo_B` is beneficial there.

For each block `i`:

### Suppressor sufficiency

Construct

`H_i+ = H[block_i <- B_i]`

and define for metric `m`

`S_i,m = J_m(H_i+) - J_m(H)`.

Positive values mean that the B-state of block `i`, by itself, suppresses part of the otherwise expressed `bo_B` benefit.

### Suppressor necessity

Construct

`B_i- = B[block_i <- F_i]`

and define

`N_i,m = J_m(B) - J_m(B_i-)`.

Positive values mean that reverting block `i` toward its F-state releases benefit in the actual combined background.

All comparisons are paired over the frozen Gate-A images with 10,000 bootstrap replicates.

## Frozen deployment coordinate system

Use the exact historical Gate-A coordinate system:

- same 256 validation images;
- same historical MaskGIT schedules;
- same common random numbers;
- same proposal logic;
- same per-image effect arrays.

Primary endpoints:

- `F61`
- `D_non61`

Supportive endpoints:

- `C61`
- `delta_P61`

`CE` is also recorded in the causal map.

Secondary, non-classifying readouts:

- ordinary masked CE at mask ratios 0.25, 0.5, 0.9, 1.0;
- autonomous rollout `global_js_nats`, `spatial_js_nats`, and `pair_js_nats`.

These secondary readouts cannot override the primary Gate-A classification.

## Suppression load

The total observed terminal suppression load is measured directly as

`L_m = J_m(B) - J_m(H)`.

This is the quantity the single-block map attempts to localize.

For each block report `S_i,m / L_m` and `N_i,m / L_m` when `L_m` is nonzero, but do not interpret these ratios as additive variance explained.

## Formal block labels

For a block `i`:

### Bidirectional dominant-suppressor evidence

For **both** primary metrics, require:

- `S_i,m > 0` with paired-bootstrap 95% CI entirely positive; and
- `N_i,m > 0` with paired-bootstrap 95% CI entirely positive.

A block satisfying this receives `bidirectional_primary = true`.

### Context-dependent co-suppressor evidence

If necessity is significant on both primary metrics but sufficiency is not significant on both primary metrics, label `necessity_only_primary = true`.

These block-level labels are reported independently of the overall S1-S4 classifier.

## Overall classification

### S1 — `S1_DOMINANT_SINGLE_SUPPRESSOR`

Requires A0 closure plus at least one block with `bidirectional_primary = true`.

If multiple blocks satisfy the criterion, report all of them and rank by the mean normalized primary suppression load; do not force uniqueness.

Interpretation: at least one terminal B-state block is individually sufficient to impose suppression on `H` and necessary to maintain suppression in `B`.

### S2 — `S2_CONTEXT_DEPENDENT_SUPPRESSOR`

Requires A0 closure, no S1 block, and at least one block with `necessity_only_primary = true`.

Interpretation: suppression is causally localized to a block only in the co-adapted combined background; its B-state is not independently harmful on `H`.

### S3 — `S3_DISTRIBUTED_HIGHER_ORDER_SUPPRESSION`

Requires A0 closure, no S1 or S2 block, and a positive total suppression load on both primary endpoints (`B-H > 0`).

Interpretation: the missing `bo_B` benefit is not localized to any single tested forward block under the preregistered bidirectional/necessity criteria. Pair or top-k interaction testing may follow, but is not part of D6A.2.

### S4 — `S4_PARTITION_FAILURE`

Any A0 closure failure. Stop scientific interpretation.

### `SX_NO_POSITIVE_SUPPRESSION_LOAD`

If A0 passes but the full `B-H` load is not positive on both primary endpoints, do not force S1-S3. Report that the premise of a positive dual-primary suppression load is not supported by the exact frozen evaluator.

## Optional top-block cross-background check

The terminal `A = F_EM_bo_oracle` system is not part of the primary map. D6A.2 may construct

`A_star = A[bo <- bo_B]`

and, **only after the primary map identifies the top one or two blocks**, evaluate those B-state blocks on `A_star` as an off-manifold robustness check.

This supportive check cannot create an S1/S2 classification that is absent in the primary `H/B` map.

## No pair search in D6A.2

D6A.2 does not execute a broad pair/factorial search. If S3 occurs, a later preregistered extension may test only top-ranked pairs using

`I_ij = J(H+B_i+B_j) - J(H+B_i) - J(H+B_j) + J(H)`.

## Scientific boundary

D6A.2 tests terminal forward-state suppression of an already demonstrated beneficial `bo_B` state. It does **not** test whether credit-field coupling is harmful, does not modify plasticity, and does not establish a final learning rule.

The relevant conceptual distinction is:

`local credit -> integrated plastic trajectory -> terminal parameter state -> background-dependent deployment utility`.

D6A.2 acts only on the final arrow.
