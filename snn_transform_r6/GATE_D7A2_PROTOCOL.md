# Gate D7A.2 — top-two compensation interaction: b2 + bO

## Status

Pre-registered final parameter-level stopping gate after D7A.1 = `C3_DISTRIBUTED_COMPENSATION`.
D7A.2 is eval-only: no training, no learner modification, no gradient audit, no trajectory replay, and no additional block or pair search.

## Scientific question

D7A.1 established a large, significant terminal compensation load between rescued `R` and unrescued safe-W2-clamped `C`, with no single non-W2 block bidirectionally explaining both primary endpoints. `b2` and `bO` were the top two descriptive candidates.

D7A.2 asks one final parameter-level question:

> Does the top-two pair `(b2,bO)` concentrate the distributed compensation into a compact additive/affine subsystem, or does distributed compensation persist?

If the pair does not meet the compact/subsystem criteria below, parameter-level decomposition stops and the next stage must be functional-mode identification.

## Frozen systems

Use the same terminal systems and Gate-A coordinate system as successful D7A.1:

- `R`: rescued hybrid `B[W2 <- W2^F]`, using the D7A same-run replayed F terminal as the operational safe-W2 reference;
- `C`: D7A `B_W2_safe_clamp` terminal.

The invariants remain:

\[
W_2^R=W_2^C,
\qquad
W_O^R=W_O^C
\]

bit-exact.

D7A.2 must reproduce frozen D7A.1 per-image arrays for `R`, `C`, `R_plus_C_b2`, `R_plus_C_bO`, `C_minus_b2`, and `C_minus_bO` to <1e-7, with proposal arrays exact where available in the D7A.1 artifact.

## Pair systems

Only two new scientific systems are evaluated:

### Sufficiency pair

\[
R_{b2+bO}=R[b_2\leftarrow b_2^C,\;b_O\leftarrow b_O^C].
\]

### Necessity pair

\[
C_{-(b2+bO)}=C[b_2\leftarrow b_2^R,\;b_O\leftarrow b_O^R].
\]

No other pair is permitted in D7A.2.

## Primary endpoints

Unchanged:

\[
\boxed{F_{61},\qquad D_{non61}}
\]

Supportive Gate-A endpoints remain `C61`, `delta_P61`, and `CE`.

All inference uses per-image paired effects with 10,000 paired bootstrap samples.

## Frozen total compensation load

For metric `m`:

\[
L_m^{comp}=J_m(C)-J_m(R).
\]

D7A.2 must reproduce the D7A.1 load from frozen per-image arrays.

## Pair sufficiency and necessity

### Pair sufficiency

\[
S_{12,m}=J_m(R_{b2+bO})-J_m(R).
\]

### Pair necessity

\[
N_{12,m}=J_m(C)-J_m(C_{-(b2+bO)}).
\]

Descriptive concentration fractions are:

\[
E^S_m=\frac{S_{12,m}}{L_m^{comp}},
\qquad
E^N_m=\frac{N_{12,m}}{L_m^{comp}}
\]

when the corresponding total load is positive. Fractions are descriptive and are not additive attribution scores.

## Pair interactions

Reuse frozen D7A.1 singles.

### Sufficiency interaction

\[
I^S_m = J_m(R_{b2+bO})-J_m(R_{b2})-J_m(R_{bO})+J_m(R).
\]

### Necessity interaction

\[
I^N_m = N_{12,m}-N_{b2,m}-N_{bO,m}.
\]

Also report the direct incremental sufficiency effect beyond b2:

\[
\Delta^{S}_{pair-b2,m}=J_m(R_{b2+bO})-J_m(R_{b2}).
\]

This is used only to test the redundancy condition below.

## Formal stopping classifier

The classes are mutually exclusive and evaluated in the following precedence order.

### A3_SYNERGISTIC_PAIR

Require, for both primary endpoints:

- pair sufficiency CI entirely > 0;
- pair necessity CI entirely > 0;
- `I^S` CI entirely > 0; and
- `I^N` CI entirely > 0.

Interpretation: `b2+bO` form a coordinated supra-additive compensatory affine subsystem. This is the only outcome that motivates deeper parameter-subsystem analysis.

### A2_REDUNDANT_OFFSET_MODE

If A3 is false, require for both primary endpoints:

- pair sufficiency CI entirely > 0;
- pair necessity CI entirely > 0;
- `I^S` CI entirely < 0; and
- the 95% CI of `pair - b2` sufficiency contains zero.

Interpretation: `bO` largely overlaps the compensation already expressed by `b2`; the top-two pair does not constitute an independent synergistic subsystem.

### A1_COMPACT_TWO_BLOCK_COMPENSATION

If A3 and A2 are false, require for both primary endpoints:

- pair sufficiency CI entirely > 0;
- pair necessity CI entirely > 0; and
- all four point-estimate concentration fractions `E^S_F61`, `E^S_D`, `E^N_F61`, `E^N_D` are >= 0.50.

The 0.50 threshold is used only as an explicit stopping definition of "majority of total compensation load". Statistical inference remains based on paired-bootstrap CIs.

Interpretation: compensation substantially concentrates into the `(b2,bO)` pair even without resolved synergy/redundancy structure.

### A4_DISTRIBUTED_COMPENSATION_PERSISTS

Any other valid result.

Interpretation: the top-two pair does not compactly account for compensation across both primary endpoints. This is the mandatory stopping condition for parameter-level decomposition. Do not test `b2+V`, `b2+FF1`, `b2+Q`, or any further pair/block combination. Proceed to D8 parameterization-invariant functional plastic-mode identification.

## Interpretation boundaries

D7A.2 is a stopping test, not a search. It does not establish trajectory emergence or a migrating mode during learning.

A2 means overlap in terminal functional effect, not that `bO` is non-causal.
A3 is the only result that justifies continued study of a compact `(b2,bO)` parameter subsystem.
A1 means the pair captures a majority of the terminal compensation load, but does not itself prove a low-dimensional functional mode.
A4 formally ends parameter transplant decomposition and motivates D8.

<!-- launch marker: formal D7A.2 b2+bO stopping audit -->
