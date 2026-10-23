# Gate D7A.1 — compensation mapping around the safe-W2 constraint

## Status

Pre-registered terminal causal map. D7A is frozen as `T0_NO_W2_TRAJECTORY_RESCUE`.
D7A.1 is eval-only: no training, no learner modification, no gradient audit, no trajectory replay, and no pair search.

## Scientific question

D7A proved only that exact time-matched safe-W2 clamping is insufficient for deployment rescue. The hypothesis that suppression is redistributed into other plastic coordinates remains an inference until D7A.1.

Define:

- `B`: D6A combined terminal, `F_EM_WO_zero_bo_oracle`;
- `F`: D6A `F_EM` terminal;
- `R = B[W2 <- W2^F]`: the D6A.2 rescued terminal hybrid;
- `C`: D7A `B_W2_safe_clamp` terminal.

The key identity is

\[
\boxed{W_2^R=W_2^C=W_2^F}.
\]

Therefore the deployment difference `C - R` cannot be attributed to W2 and must be carried by non-W2 terminal forward adaptation, provided closure passes.

## A0 provenance and closure gate

Scientific interpretation is forbidden unless all of the following pass.

1. D6A remains `R0_NO_REPAIR_SUFFICIENCY` with valid oracle.
2. D6A.2 remains `S1_DOMINANT_SINGLE_SUPPRESSOR`.
3. D6A.3 remains `P4_MIXED_BACKGROUND_DEPENDENT`.
4. D7A remains `T0_NO_W2_TRAJECTORY_RESCUE` with valid oracle.
5. Reconstruct `R = B[W2 <- W2^F]` exactly in forward parameter space.
6. `R` reproduces the frozen D6A.2 `B_minus_W2` per-image Gate-A effect arrays to <1e-7. The historical D6A.2 artifact did not store proposal IDs or self/teacher metric arrays, so those cannot be retrospectively calibrated and must not be claimed.
7. `C` reproduces D7A self/teacher metrics and effects to <1e-7 and proposal arrays exactly.
8. W2 identity is exact: `max_abs(W2_R - W2_C) == 0` and both equal `W2_F` exactly.
9. WO is invariant between R and C exactly and is not a candidate block.
10. The candidate partition covers every remaining forward field exactly once.
11. Forward closure:

\[
R[\theta_{\neg W2,\neg WO}\leftarrow\theta^C]=C,
\]

\[
C[\theta_{\neg W2,\neg WO}\leftarrow\theta^R]=R
\]

must hold exactly over all forward parameter fields.
12. Functional closure: Gate-A evaluation of the two all-block closure hybrids must reproduce C/R respectively: proposal arrays exact and all per-image effects <1e-7.

If parameter closure passes but functional/Gate-A closure fails, formal result is `C4_FUNCTIONAL_STATE_CLOSURE_FAILURE` and no causal carrier interpretation is allowed.

## Candidate blocks

W2 is fixed-identical and excluded. WO is exact-identical and excluded.

The only candidate blocks are:

- `Q = (WQ, bq)`
- `K = (WK, bk)`
- `V = (WV, bv)`
- `bO = (bo,)`
- `Norm = (gamma2, beta2)`
- `FF1 = (W1, b1)`
- `b2 = (b2,)`
- `Readout = (WR,)`
- `Embedding = (tok, pos)`

These nine blocks must partition all forward fields other than W2 and WO with no omissions or duplicates.

`bO` is explicitly included even though its exact local oracle remained finite-step valid in D7A: local-oracle validity does not imply that its terminal state is deployment-neutral in the altered trajectory.

## Total compensation load

For metric `m`, define

\[
\boxed{L_m^{comp}=J_m(C)-J_m(R)}.
\]

Primary endpoints are unchanged:

\[
\boxed{m\in\{F_{61},D_{non61}\}}.
\]

A positive load means the D7A-clamped terminal system lost rescue relative to R.

Supportive Gate-A endpoints remain `C61`, `delta_P61`, and `CE`.

## Bidirectional terminal compensation map

For candidate block `i`:

### Compensation sufficiency

\[
R_i^C=R[\theta_i\leftarrow\theta_i^C]
\]

\[
\boxed{S_{i,m}^{comp}=J_m(R_i^C)-J_m(R)}.
\]

Positive means the C-state of block i is harmful when inserted alone into the rescued R background.

### Compensation necessity

\[
C_i^R=C[\theta_i\leftarrow\theta_i^R]
\]

\[
\boxed{N_{i,m}^{comp}=J_m(C)-J_m(C_i^R)}.
\]

Positive means restoring the R-state of block i releases rescue within the actual C background.

All inference uses per-image paired effects and 10,000 paired bootstrap samples. For each block report both absolute effects and descriptive fractions `S/L` and `N/L` when the corresponding load is positive. Fractions are not additive attribution scores.

## Formal result classes

### C1_SINGLE_COMPENSATORY_CARRIER

At least one block has, for both primary endpoints:

- sufficiency CI entirely > 0; and
- necessity CI entirely > 0.

Such blocks are direct bidirectional terminal compensation carriers. Rank multiple C1 blocks descriptively by mean fraction-of-load score; do not claim uniqueness unless only one qualifies.

### C2_CONTEXT_DEPENDENT_COMPENSATION

No C1 block exists, but at least one block has necessity CIs entirely >0 on both primary endpoints while failing the two-endpoint sufficiency requirement.

Interpretation: compensation is maintained by a co-adapted configuration and is not sufficient as an isolated displacement.

### C3_DISTRIBUTED_COMPENSATION

Both primary total compensation-load 95% CIs are entirely >0 and no C1/C2 block exists.

Interpretation: the safe-W2 constraint is bypassed through a distributed terminal configuration. Pair/top-k interaction testing, if needed, requires a separate preregistered gate.

### C4_FUNCTIONAL_STATE_CLOSURE_FAILURE

Parameter partition/closure passes but all-block Gate-A closure does not reproduce C/R. Stop causal interpretation and repair the audit/evaluator state definition.

### CX_INSUFFICIENT_DUAL_COMPENSATION_LOAD

No C1/C2 block exists and the two primary compensation loads do not both have 95% CIs entirely >0. This includes non-positive or statistically unresolved total load. Descriptive block results may still be reported, but distributed compensation is not established.

A parameter-partition or exact-forward closure failure is a technical failure and is not classified as C1-C4.

## Interpretation boundaries

D7A.1 localizes terminal causal carriers of the difference between a rescued terminal W2 hybrid and the unrescued trajectory-clamped learner. It does not establish when a carrier emerged or prove that a functional mode literally migrated during learning.

If `b2` is a carrier, the precise interpretation is that the W2-dominant FFN-output suppressive mode is re-expressed in part through b2 when W2 is constrained, not that b2 was previously non-causal.

If `FF1` is a carrier, the natural functional hypothesis is that changing h reconstitutes a harmful output through fixed `W2^F h`.

If upstream Q/K/V/Norm/Embedding blocks dominate, the hypothesis shifts toward input-manifold/routing reorganization around the safe FFN map.

Only after a top carrier is identified should its trajectory emergence be studied. D7A.1 itself performs no trajectory analysis.

<!-- launch marker: formal D7A.1 compensation audit -->
