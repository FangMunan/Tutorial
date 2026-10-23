# Gate D8A — parameterization-invariant functional-mode identification

## Status

Pre-registered functional analysis after D7 is closed. D7A.2 is frozen as `A1_COMPACT_TWO_BLOCK_COMPENSATION`.

D8A is **identification only**: no training, no learner modification, no parameter/block transplant search beyond reconstructing the already frozen R/C/P/Q states, and no activation intervention. D8B is a separate future gate and is allowed only if D8A identifies a stable low-dimensional shared functional candidate.

## Scientific question

D7 established:

1. a safe W2 trajectory can be bypassed by co-adaptation elsewhere in the plastic network;
2. the resulting single-block compensation is distributed;
3. most of that compensation substantially concentrates in the compact `(b2,bO)` pair.

D8A asks whether the compact parameter subsystem and the full R→C difference converge onto the **same low-dimensional functional displacement**.

The four frozen terminal states are:

- `R`: rescued reference, D6A combined terminal with W2 replaced by the D7A same-run safe W2;
- `C`: unrescued D7A safe-W2-clamped terminal;
- `P = R[b2 <- b2_C, bO <- bO_C]`: pair insertion state from D7A.2;
- `Q = C[b2 <- b2_R, bO <- bO_R]`: pair reversion state from D7A.2.

No new parameter candidate is introduced.

Define functional displacements at layer/activity q:

\[
\Delta Q_{full}=Q_C-Q_R,
\]

\[
\Delta Q_{pair}=Q_P-Q_R,
\]

\[
\Delta Q_{release}=Q_C-Q_Q,
\]

\[
\Delta Q_{residual}=Q_C-Q_P=\Delta Q_{full}-\Delta Q_{pair}.
\]

## A0 provenance gate

Scientific interpretation is forbidden unless all of the following pass.

1. D7A.2 remains `A1_COMPACT_TWO_BLOCK_COMPENSATION`.
2. Reconstruct R/C/P/Q exactly from the frozen D6A and D7A terminals.
3. R/C/P/Q reproduce the frozen D7A.2 Gate-A arrays for `R`, `C`, `R_plus_C_b2_bO`, and `C_minus_b2_bO` to `<1e-7`, with proposal arrays exact.
4. `W2_R == W2_C` and `WO_R == WO_C` bit-exact, preserving the D7 causal contrast.
5. Historical Gate-A validation tokens are reproduced from the frozen MNIST/tokenizer coordinate system before any discovery probes are selected.
6. Discovery image IDs are disjoint from all 256 historical Gate-A validation IDs exactly.
7. Activation extraction is self-tested against the model's frozen `_tail()` probability path to `<1e-10` before analysis.

Any A0 failure is technical and stops D8A interpretation.

## Independent discovery set

Use the same frozen MNIST 0/1/2 validation split and frozen R6B1 tokenizer as the D-series, but select a new **256-image discovery set** that is exactly disjoint from historical Gate-A 256.

Selection is deterministic and approximately class-balanced across digits 0/1/2 using a fixed discovery seed. Discovery IDs, labels, token IDs and a SHA256 provenance hash are saved.

The discovery set is split deterministically into two disjoint 128-image halves for internal mode-stability checks. Neither half overlaps Gate-A.

## Common-input probe contexts

D8A must not let R/C/P/Q generate different next-round inputs. All systems receive exactly the same discrete token array and mask pattern.

For each discovery image, construct three nested teacher-forced contexts from the ground-truth token sequence:

- depth 0: 0 visible positions, all N positions masked;
- depth 1: floor(N/3) visible ground-truth positions;
- depth 2: floor(2N/3) visible ground-truth positions.

The visible-position permutation is deterministic per image under a frozen mask seed. At each depth, activations are recorded only at currently masked positions. This yields identical input/mask coordinates for R/C/P/Q and avoids contamination from system-specific sampled proposals.

## Recorded functional coordinates

For every probed masked position, record the exact frozen forward quantities:

\[
z=x_{raw}+W_Oa+b_O,
\]

\[
u=\hat z\gamma_2+\beta_2,
\]

\[
h=\mathrm{ReLU}(W_1u+b_1-\theta),
\]

\[
y=z+W_2h+b_2,
\]

and readout logits

\[
\ell=W_R r_f(y).
\]

The analyzed layers are therefore `z`, `u`, `h`, `y`, and `logits`.

## Primary spectral analysis: uncentered displacement

The primary SVD/eigendecomposition is **uncentered**. A constant offset is part of the hypothesized compensatory functional mode and must not be removed by centering.

For a displacement matrix `DeltaQ` with one row per probe-position coordinate:

\[
\Delta Q=U\Sigma V^T.
\]

Report for `k in {1,2,3,5,10}` where dimensions permit:

\[
EV_k=\frac{\sum_{j=1}^k\sigma_j^2}{\sum_j\sigma_j^2}.
\]

Also report effective rank, participation ratio, total Frobenius energy, and the fraction of energy carried by the mean displacement vector.

Centered spectra are reported supportively only.

## Pair-to-full reconstruction metrics

For the top-k right-singular feature subspace of `DeltaQ_pair`, `V_pair,k`, define:

\[
\Gamma_k=
\frac{\|\Delta Q_{full}V_{pair,k}V_{pair,k}^T\|_F^2}
{\|\Delta Q_{full}\|_F^2}.
\]

This is the fraction of the full functional displacement captured by the pair-derived feature subspace.

Also report:

- principal-angle cosines between `span(V_full,k)` and `span(V_pair,k)`;
- principal-angle cosines between `span(V_pair,k)` and `span(V_release,k)`;
- Frobenius cosine between `DeltaQ_full` and `DeltaQ_pair`;
- raw pair/full and residual/full energy ratios.

The reciprocal `release = C-Q` analysis tests whether pair insertion and pair reversion express the same functional directions in different backgrounds.

## Stability analysis

For the pooled discovery set and separately for the two 128-image discovery halves, compute full and pair subspaces.

For the selected k, report split-half principal-angle cosines for the full displacement and pair displacement.

Mask-depth-specific spectra/alignment for the three fixed depths are supportive diagnostics. D8A does not require the mode to be identical at every mask depth, but strong depth dependence must be reported.

## Primary layer and preregistered identification gate

The **primary identification layer is y-space**, because this is where the downstream effect of bO and direct b2 shift first coexist in the same affine FFN output coordinate. Logits are an independent downstream confirmation. z/u/h are mechanistic localization layers.

Search only `k in {1,2,3}` and choose the smallest k satisfying all pooled y-space conditions:

1. `EV_full(k) >= 0.70`;
2. `EV_pair(k) >= 0.80`;
3. `Gamma_k >= 0.50`;
4. minimum principal-angle cosine between full and pair k-dimensional subspaces `>= 0.80`;
5. minimum principal-angle cosine between pair and reciprocal-release k-dimensional subspaces `>= 0.80`.

These numerical thresholds are an explicit engineering gate for deciding whether a low-dimensional candidate is strong enough to justify D8B. They are not universal biological constants.

A pooled candidate is considered **stable** only if, at the selected k:

- full-displacement split-half minimum principal cosine `>= 0.70`; and
- pair-displacement split-half minimum principal cosine `>= 0.70`.

## D8A result classes

### I1_SHARED_LOW_DIMENSIONAL_FUNCTIONAL_CANDIDATE

A k<=3 satisfies all pooled y-space criteria and the split-half stability criteria.

Interpretation: D8A identifies a stable low-dimensional y-space candidate shared by full R→C compensation and the compact `(b2,bO)` subsystem. Freeze the **pair-derived** `V_pair,k` from `DeltaY_pair = Y_P - Y_R` and proceed to a separate D8B bidirectional causal activation intervention on the historical Gate-A 256 only. The full `C-R` displacement is used for capture/alignment tests and is not allowed to redefine the D8B basis after classification.

### I4_UNSTABLE_SHARED_MODE

A pooled k<=3 satisfies all shared-mode criteria, but split-half stability fails.

Interpretation: a low-dimensional pooled description exists but is not stable enough for a causal mode claim. Do not proceed directly to D8B without resolving generalization.

### I2_LOW_DIMENSIONAL_BUT_NOT_SHARED

No shared-mode k passes, but at k=3 both `EV_full >= 0.70` and `EV_pair >= 0.80`.

Interpretation: both displacements are individually low-dimensional, but the pair subsystem does not span the dominant full compensation direction strongly enough to claim a common macroscopic mode.

### I3_HIGH_DIMENSIONAL_FUNCTIONAL_DISPLACEMENT

All remaining valid outcomes.

Interpretation: the terminal compensation cannot be reduced to a <=3-dimensional shared y-space mode under this frozen probe definition.

## D8B boundary

D8A performs no activation intervention and makes no causal harmful-mode claim.

Only `I1_SHARED_LOW_DIMENSIONAL_FUNCTIONAL_CANDIDATE` authorizes D8B. D8B must use the frozen discovery-set **pair-derived y-space top-k basis** from `P-R` before evaluating the historical Gate-A 256, which were never used for discovery. No re-fitting or rotation of the basis on Gate-A is permitted.

The intended D8B test is bidirectional:

\[
y'_C=y_C-P_k(y_C-y_R)
\]

and

\[
y'_R=y_R+P_k(y_C-y_R),
\]

but those interventions are explicitly outside D8A.

## Interpretation boundary

D8A can establish a low-dimensional, cross-parameter functional candidate. It cannot by itself establish negative deployment utility, trajectory emergence, or the final local/dynamical admissibility rule. Those require D8B and later synthesis gates.
