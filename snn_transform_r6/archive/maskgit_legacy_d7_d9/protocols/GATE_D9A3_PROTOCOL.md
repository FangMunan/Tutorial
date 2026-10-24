# Gate D9A.3 — pair-versus-residual mode-consolidation decomposition

## Status

D9A.3 is **diagnostic only**. It performs no learning-rule change, no activation intervention, no plastic-velocity projection, no controller, no new functional-mode discovery, and no additional block/pair search.

Frozen prerequisites:

- D7A.2 = `A1_COMPACT_TWO_BLOCK_COMPENSATION`, with the only pair `(b2,bO)`.
- D8A = `I1_SHARED_LOW_DIMENSIONAL_FUNCTIONAL_CANDIDATE`, selected `k=1`.
- D8B = `B1_BIDIRECTIONAL_CAUSAL_HARMFUL_FUNCTIONAL_MODE`.
- D9A.1 = `V0_NO_HARMFUL_PLASTIC_VELOCITY_EXCESS`.
- D9A.2-R = `T2_LATE_MODE_CONSOLIDATION`.

Scientific interpretation of T2 is frozen as:

`early partial representation -> progressive geometric consolidation -> causal terminal mode`.

The rank-1 y-space vector `v` is the exact frozen D8A/D8B basis. It is not refit, rotated, sign-flipped, or re-estimated here.

## Scientific question

D9A.2-R showed that the time-matched relative geometry `C_t-R_t` is already partially aligned to the terminal causal direction early in training, but becomes progressively more dominant/aligned, with the first formal mode-presence transition between checkpoints 128 and 256.

D9A.3 asks one binary mechanism question only:

> Is the 128->256 consolidation mainly generated within the already-known `(b2,bO)` subsystem, produced by release/reorganization of residual-network cancellation, jointly organized by pair and residual network, or not explained by the pair trajectory at all?

No individual residual blocks may be searched.

## Frozen trajectories and checkpoint states

Replay the exact D7A synchronized 1024-context trajectories:

- `F_t`: F_EM reference.
- `B_t`: combined failure learner.
- `C_t`: combined learner with W2 clamped after every context to time-matched `F_t`.

At checkpoints

`64, 128, 256, 512, 768, 1024`

construct the same D9A.2-R analysis-only reference

`R_t = B_t[W2 <- W2^{F_t}]`.

Because D7A enforces `W2^{C_t}=W2^{F_t}`, require exact

`W2^{C_t}=W2^{R_t}`.

Then construct eval-only pair transplants:

`P_t = R_t[b2 <- b2^{C_t}, bo <- bo^{C_t}]`

and reciprocal supportive state

`Q_t = C_t[b2 <- b2^{R_t}, bo <- bo^{R_t}]`.

`P_t` and `Q_t` never participate in learning and cannot alter F/B/C state or RNG.

## Probe separation

Use exactly the frozen 128-image D9A.1 diagnostic set and masks already used by successful D9A.2-R. Preserve its exact provenance schema:

- diagnostic IDs: int64,
- labels: int8,
- tokens: int16,
- original D9A.1 seeds.

The set remains disjoint from both historical Gate-A and D8A discovery sets. No new probe selection is allowed.

## Exact functional decomposition

For every checkpoint and every frozen diagnostic row, compute in y-space:

`DeltaY_full = Y_C - Y_R`

`DeltaY_pair = Y_P - Y_R`

`DeltaY_res = Y_C - Y_P`.

Therefore require the arithmetic identity

`DeltaY_full = DeltaY_pair + DeltaY_res`

with maximum absolute numerical error <= `1e-12`.

Supportive reciprocal release is

`DeltaY_release = Y_C - Y_Q`.

The reciprocal state is not used to redefine pair/residual additivity.

## Geometry measured for full, pair, residual and supportive release

For each component `X in {full,pair,res,release}` report:

1. Frozen-mode signed mean amplitude

`a_X(t) = mean_rows(DeltaY_X v)`.

2. Frozen-mode projected energy capture

`rho_v,X(t) = ||DeltaY_X v||^2 / ||DeltaY_X||_F^2`.

3. Total functional difference energy and Frobenius norm.

4. Uncentered top right-singular direction `v_X,t`, its explained-energy fraction `EV1_X(t)`, and

`c_X(t) = |cos(v_X,t, v_terminal)|`.

5. Mean-displacement energy fraction as supportive offset-like geometry.

Primary SVD remains uncentered because D8A established that the mean/affine displacement is part of the candidate functional mode.

## Pair-residual interaction geometry

At each checkpoint report the vectorized matrix cosine

`chi_t = <DeltaY_pair,DeltaY_res> / (||DeltaY_pair||_F ||DeltaY_res||_F)`.

Negative `chi_t` means pair and residual matrices are globally opposing; positive means reinforcing.

Also report:

- scalar sign product `a_pair(t) * a_res(t)`;
- scale-free opposing terminal-mode ratio

`opposing_ratio(t) = max(0, -a_res(t)/a_pair(t))`

when `a_pair(t)` is nonzero;
- supportive frozen-v projected-energy ratio

`E_v,pair(t) / E_v,full(t)`;
- Frobenius cosine between `DeltaY_pair` and supportive `DeltaY_release`.

These are descriptive geometry quantities, not additive causal attribution scores.

## Frozen transition window

The formal mechanism classifier uses **only 128 -> 256**, because D9A.2-R preregistered this as the first absent-to-present transition of the frozen terminal mode.

Define signed frozen-v amplitude growth:

`g_full = a_full(256) - a_full(128)`

`g_pair = a_pair(256) - a_pair(128)`

`g_res = a_res(256) - a_res(128)`.

The exact decomposition implies

`g_full = g_pair + g_res`.

Require this identity within `1e-12`.

When `g_full > 0`, define

`s_pair = g_pair/g_full`, `s_res = g_res/g_full`.

Reuse the frozen D8A terminal-alignment engineering gate

`c >= 0.80`.

Use `0.50` only as the explicit definition of majority signed-amplitude growth and `0.25` only as the explicit nontrivial-share definition for joint growth. These are preregistered engineering thresholds, not biological constants.

## Formal mechanism classifier

Classes are mutually exclusive and evaluated in the following precedence order.

### M2_RESIDUAL_UNMASKING_CANCELLATION_RELEASE

Require all:

- `g_full > 0`;
- pair is already terminal-aligned at checkpoint 128: `c_pair(128) >= 0.80`;
- pair and residual have opposite frozen-v signed amplitudes at 128: `a_pair(128)*a_res(128) < 0`;
- matrix-level pair-residual geometry is opposing at 128: `chi_128 < 0`;
- the scale-free opposing ratio decreases from 128 to 256.

Interpretation: the pair mode is already present/aligned before the full mode consolidates, and residual opposition/cancellation weakens, progressively unmasking it.

### M3_JOINT_CO_CONSOLIDATION

If M2 is false, require all:

- `g_full > 0`;
- `g_pair > 0` and `g_res > 0`;
- `s_pair >= 0.25` and `s_res >= 0.25`;
- both pair and residual top directions are terminal-aligned at checkpoint 256: `c_pair(256) >= 0.80` and `c_res(256) >= 0.80`.

Interpretation: pair and residual network make nontrivial same-sign contributions to the consolidation and both organize toward the terminal direction.

### M1_PAIR_DRIVEN_CONSOLIDATION

If M2 and M3 are false, require all:

- `g_full > 0`;
- `s_pair >= 0.50`;
- `c_pair(256) >= 0.80`.

Interpretation: the known `(b2,bO)` subsystem explains a majority of the signed terminal-mode amplitude growth through the 128->256 consolidation transition.

### M4_PAIR_DOES_NOT_EXPLAIN_TRAJECTORY_CONSOLIDATION

Any other valid outcome.

Interpretation: the pair may remain a strong terminal causal actuator, but its own time-resolved geometry does not explain the consolidation transition. Do not expand into new per-block checkpoint searches.

## A0 provenance and validity gates

Before any M1-M4 interpretation require:

1. D7A.2 classification/pair exactly `A1_COMPACT_TWO_BLOCK_COMPENSATION`, `['b2','bO']`.
2. D9A.2-R classification exactly `T2_LATE_MODE_CONSOLIDATION`.
3. D8A/D8B/D9A.1/D9A.2-R frozen basis identity and unit norm.
4. D9A.1 diagnostic IDs/labels/tokens/hash reproduced exactly with original dtypes.
5. F/B/C synchronized stream and external RNG exactness.
6. D7A post-context W2 clamp exact at every context.
7. At every checkpoint `R_t=B_t[W2<-W2^F]`; P/Q change only `b2` and `bo` relative to their bases.
8. Diagnostics operate only on clones/snapshots and cannot mutate the learned trajectory.
9. Stateless y reconstruction passes the existing D9A.1 parity tolerance.
10. `DeltaY_full = DeltaY_pair + DeltaY_res` rowwise with max abs <= `1e-12`.
11. Full `C_t-R_t` row projections reproduce successful D9A.2-R frozen arrays within `1e-10` at all six checkpoints.
12. Full checkpoint summary geometry reproduces successful D9A.2-R values within `1e-10`.
13. Terminal F/B/C systems replay frozen D7A terminals within the existing D7A tolerance.

Failure of any A0 gate invalidates M1-M4 classification.

## Outputs

Save at minimum:

- `D9A3_SUMMARY.json`
- `D9A3_CHECKPOINT_DECOMPOSITION.csv`
- `D9A3_DIAGNOSTIC_IDS.csv`
- `D9A3_ARRAYS.npz`
- `D9A3_MANIFEST.json`

## Decision boundary

D9A.3 is a mechanism audit, not a controller gate.

- M1 motivates a targeted causal intervention on coordinated `(b2,bO)` trajectory dynamics.
- M2 motivates a targeted causal test of maintaining/releasing protective residual cancellation.
- M3 motivates a collective pair-plus-residual coordination test.
- M4 says terminal pair attribution does not explain the trajectory consolidation mechanism.

No M class automatically authorizes a final learning rule, and no result authorizes reopening an arbitrary block/pair search.
