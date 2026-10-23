# Gate D8B — frozen rank-1 bidirectional causal functional-mode intervention

## Status

Pre-registered causal test following frozen D8A result:

`I1_SHARED_LOW_DIMENSIONAL_FUNCTIONAL_CANDIDATE`, selected `k=1`.

D8B is eval-only. It performs no training, no learner modification, no parameter/block transplant search, no trajectory replay, and no re-fitting of the functional basis on Gate-A.

## Frozen functional coordinate

Load the exact D8A artifact and use only:

\[
v = \texttt{D8B\_frozen\_y\_pair\_basis}[:,0]\in\mathbb R^{96},\qquad \|v\|_2=1.
\]

The basis was discovered from the independent D8A discovery set and is frozen before the historical Gate-A 256 are evaluated.

The D8A discovery set must remain exactly disjoint from historical Gate-A 256.

## Frozen terminal systems

Reconstruct the same terminal contrast used in D8A:

- `R`: rescued reference;
- `C`: unrescued safe-W2-clamped state.

Require exact frozen invariants:

\[
W_2^R=W_2^C,\qquad W_O^R=W_O^C.
\]

R and C must reproduce frozen D7A.2 Gate-A arrays to `<1e-7`, with proposal arrays exact, before any intervention result is interpreted.

## Matched-input counterfactual definition

For every Gate-A branch, round, masked position and current discrete context `cur`, evaluate target and counterfactual reference on the **same current token array and mask coordinates**.

Do not use R's own sampled trajectory as the counterfactual for C, or vice versa.

Let

\[
y_C(cur,t),\qquad y_R(cur,t)
\]

be the two frozen y-space activations for the identical current context and masked position.

Define the frozen scalar mode amplitude

\[
a(cur,t)=v^T\left(y_C(cur,t)-y_R(cur,t)\right).
\]

Only the scalar coordinate along `v` is changed. All upstream quantities and the frozen readout remain unchanged.

## Primary matched-counterfactual interventions

### Removal from C

\[
C^-:\qquad y_C^- = y_C-a v
= y_C-vv^T(y_C-y_R).
\]

### Addition to R

\[
R^+:\qquad y_R^+ = y_R+a v
= y_R+vv^T(y_C-y_R).
\]

The intervention is applied identically to both self and teacher-forced prediction branches of Gate-A. This prevents an artificial self/teacher asymmetry.

This is a **privileged matched-counterfactual causal upper bound** because the Gate-A-specific amplitude uses the matched R/C counterfactual. It is not a deployable online controller.

## Primary endpoints and causal contrasts

Primary endpoints remain:

\[
F_{61},\qquad D_{non61},
\]

with lower values better.

Define per-image paired contrasts:

\[
\Delta^-_m = J_m(C^-)-J_m(C),
\]

\[
\Delta^+_m = J_m(R^+)-J_m(R).
\]

Use paired bootstrap with 10,000 resamples.

Primary removal succeeds for endpoint `m` iff the 95% CI of `Delta^-_m` is entirely below zero.

Primary addition succeeds for endpoint `m` iff the 95% CI of `Delta^+_m` is entirely above zero.

A complete bidirectional causal result requires all four endpoint-direction tests:

- `Delta^-_F61 < 0`, CI entirely negative;
- `Delta^-_D_non61 < 0`, CI entirely negative;
- `Delta^+_F61 > 0`, CI entirely positive;
- `Delta^+_D_non61 > 0`, CI entirely positive.

## Original compensation load and causal load capture

Define the frozen R-to-C compensation load:

\[
L_m=J_m(C)-J_m(R).
\]

Report:

\[
G^-_m=\frac{J_m(C)-J_m(C^-)}{L_m},
\]

\[
G^+_m=\frac{J_m(R^+)-J_m(R)}{L_m}.
\]

Use paired bootstrap ratio distributions with the same resampled image indices. Report point estimate and 95% CI.

`majority point capture` means `G > 0.50` by point estimate.

`majority certified capture` means the 95% CI lower bound is also `> 0.50`.

These load-capture labels are descriptive/supportive and do not alter B1–B4 classification.

## Directionality control: matched sign flip

Construct:

\[
C^{wrong}=y_C+a v,
\]

\[
R^{wrong}=y_R-a v.
\]

This tests whether the observed effect is oriented rather than caused by arbitrary perturbation along the same axis.

Report:

\[
J(C^{wrong})-J(C)
\]

and

\[
J(R^{wrong})-J(R).
\]

Supportive orientation evidence is:

- `C_wrong` significantly worsens both primary endpoints;
- `R_wrong` does not significantly recreate the same positive-harm direction on either primary endpoint.

This control does not alter B1–B4 classification.

## Discovery-fixed amplitude robustness control

D8A showed that the pair mode is strongly offset-like. Freeze a scalar from the **D8A discovery artifact only**:

\[
\bar a_{disc}=v^T\,\mathbb E_{discover}[y_P-y_R].
\]

Operationally this is computed from frozen `y__pair__mean` in `D8A_MODES.npz`.

Define the fixed shift:

\[
s_{disc}=\bar a_{disc}v.
\]

On Gate-A, without using any matched R/C activation difference to determine magnitude, evaluate:

\[
\widetilde C^-:\quad y_C-s_{disc},
\]

\[
\widetilde R^+:\quad y_R+s_{disc}.
\]

Report the same two primary endpoint contrasts and load-capture fractions.

This is a robustness/generalization control only. It does not alter the formal B1–B4 primary classification.

## Formal outcome classes

### B1_BIDIRECTIONAL_CAUSAL_HARMFUL_FUNCTIONAL_MODE

Matched removal succeeds on both primary endpoints **and** matched addition succeeds on both primary endpoints.

Interpretation: the frozen rank-1 y-space coordinate carries bidirectional causal negative deployment utility under a privileged matched-counterfactual amplitude intervention.

### B2_RESCUE_ONLY_FUNCTIONAL_MODE

Matched removal succeeds on both primary endpoints, but matched addition does not succeed on both.

Interpretation: removing the coordinate is causally useful, but the coordinate is not sufficient by itself to recreate the harmful deployment state.

### B3_ADDITION_ONLY_FUNCTIONAL_MODE

Matched addition succeeds on both primary endpoints, but matched removal does not succeed on both.

Interpretation: the coordinate is sufficient to induce harm but not sufficient to account for the harmful state already present in C.

### B4_DESCRIPTIVE_ONLY_FUNCTIONAL_MODE

All remaining valid outcomes.

Interpretation: D8A's shared rank-1 coordinate remains descriptive rather than a bidirectional causal learning coordinate under the frozen matched intervention.

## A0 technical gates

Scientific interpretation is forbidden unless all pass:

1. D8A classification is exactly `I1_SHARED_LOW_DIMENSIONAL_FUNCTIONAL_CANDIDATE` with selected `k=1`.
2. D8A decision authorizes D8B and basis source is `pair_y_discovery_topk`.
3. Basis shape is `(96,1)`, finite, and unit norm within `1e-10`.
4. Discovery/Gate-A overlap is exactly zero.
5. R/C exact W2 and WO invariants hold.
6. R/C reproduce D7A.2 frozen Gate-A arrays to `<1e-7`, proposals exact.
7. Zero-intervention custom evaluator reproduces the frozen ordinary Gate-A R/C arrays to `<1e-10` for probabilities/metrics and exact proposals.
8. The discovery-fixed scalar is computed only from frozen D8A discovery statistics and never from Gate-A activation differences.

Any A0 failure is technical and produces no B1–B4 interpretation.

## Interpretation boundary

B1 would justify the phrase **causal harmful functional plastic mode** only with the explicit qualifier that D8B uses a privileged matched-counterfactual amplitude upper bound.

The discovery-fixed control can strengthen generalization toward an online scalar controller, but D8B does not establish a final local/MSDL/SNN learning rule, trajectory emergence of the mode, or a deployable controller. Those require later synthesis and local dynamical approximation gates.