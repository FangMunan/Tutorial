# Gate D9A — trajectory-level rank-1 functional regulation upper bound

## Status

Pre-registered trajectory-level bridge following frozen D8B result:

`B1_BIDIRECTIONAL_CAUSAL_HARMFUL_FUNCTIONAL_MODE`.

D8 attribution is closed. D9A does **not** search additional parameter blocks, modes, bases, learning rates, thresholds, or controller time constants. It asks one causal question:

> If the harmful rank-1 functional coordinate itself is constrained throughout the 1024-context learning trajectory, does the terminal learner improve when the regulator is removed at evaluation?

D9A is a privileged functional upper bound, not the final local/SNN/MSDL learning rule.

## Frozen functional coordinate

Load the exact D8A artifact and use only the selected rank-1 y-space pair basis

\[
v=\texttt{D8B\_frozen\_y\_pair\_basis}[:,0]\in\mathbb R^{96},\qquad \|v\|_2=1.
\]

Require D8A=`I1_SHARED_LOW_DIMENSIONAL_FUNCTIONAL_CANDIDATE`, selected `k=1`, and D8B=`B1_BIDIRECTIONAL_CAUSAL_HARMFUL_FUNCTIONAL_MODE`.

The frozen orientation is the D8B orientation: positive displacement along `v` is harmful. Require the D8B discovery-fixed amplitude to be positive and bidirectionally supported.

## Discovery-derived safe reference

No Gate-A image may set the training regulator threshold.

Reconstruct the exact D8A independent discovery IDs and teacher-forced mask contexts. Reconstruct the frozen D8A rescued reference

\[
R=B[W_2\leftarrow W_2^{F,D7A}].
\]

For every discovery row, record

\[
q_R=v^T y_R.
\]

Freeze

\[
\boxed{q_{ref}=\mathbb E_{discover}[q_R]}.
\]

Also report standard deviation and quantiles, but they are descriptive only. No quantile or margin is tuned or selected after seeing D9A outcomes.

This is intentionally a privileged upper bound because `q_ref` comes from a later rescued reference state. It is not claimed to be an online-local quantity.

## Starting state and synchronized training stream

All arms start from the exact frozen R6B3 2.0-epoch checkpoint and exact R6B4 upgrade used by D6A/D7A.

Use exactly 1024 contexts with identical:

- image indices;
- masks;
- target accumulation order;
- mirror seeds;
- external RNG state;
- reset boundary rule.

Four synchronized arms are allowed:

1. `F` — historical `F_EM` reference;
2. `B` — historical combined failure learner `F_EM_WO_zero_bo_oracle`;
3. `Q1` — same B learner plus hard rank-1 functional regulation;
4. `Q2` — same B learner plus one pre-registered slow scalar dynamical regulator.

No other arm is allowed.

Replayed F and B terminal states and Gate-A arrays must reproduce frozen D6A within the historical tolerances before D9A is interpreted.

## Training-only intervention

The regulator is active **only while learning the 1024 contexts**.

All terminal Gate-A, masked-CE and rollout-JS evaluations are performed with the regulator OFF. Therefore a successful D9A result must arise because the learning trajectory was reshaped, not because the D8B inference-time correction was left on.

For every immutable MaskGIT context, native upstream activity is computed first:

\[
y=z+W_2h+b_2,\qquad q=v^Ty.
\]

The regulated `y` is then used to form readout features, probabilities, prediction errors, and all subsequent local E/M plasticity drives in that context. Upstream `z,u,h` remain the native forward values for that context.

## Q1 — hard functional projection

For each masked position independently,

\[
e=[q-q_{ref}]_+,
\]

\[
\boxed{y^{reg}=y-e v.}
\]

Thus

\[
v^Ty^{reg}\le q_{ref}
\]

up to floating-point tolerance. Only the harmful positive side is clipped; the 95-dimensional orthogonal subspace is unchanged.

This is the primary privileged causal upper bound.

### Exact bo-oracle compatibility

The combined B learner contains the frozen D6A norm-matched exact `bo` local oracle. Under active Q1 projection, the derivative of the regulated output with respect to native y is

\[
J_{reg}=I-vv^T.
\]

Therefore the bo-oracle gradient in Q1 must analytically include this Jacobian whenever the projection is active. A finite-difference self-test must certify this corrected oracle before the formal trajectory is run.

WO remains frozen exactly as in B.

## Q2 — fixed slow scalar dynamic regulator

Q2 is a non-tuned dynamical approximation, not a search.

Reuse the already existing slow factor

\[
\boxed{\rho=0.92}
\]

and set `kappa=1` without tuning.

Initialize

\[
m_0=q_{ref}.
\]

At context `t`, use the **previous** persistent scalar state

\[
e_t=[m_{t-1}-q_{ref}]_+
\]

for all masked positions in the current immutable context:

\[
\boxed{y^{reg}_{t,j}=y_{t,j}-e_t v.}
\]

After the context's native unregulated q values have been computed, update

\[
\bar q_t=\frac1{n_t}\sum_j v^Ty_{t,j},
\]

\[
\boxed{m_t=0.92m_{t-1}+0.08\bar q_t.}
\]

The one-context delay is deliberate: the current correction is independent of current-context parameters, preserving an exact local bo-oracle definition while testing a genuinely persistent scalar controller.

No rho/kappa sweep is allowed in D9A.

## Regulator-work / network-pressure audit

For Q1 and Q2 record, per context and cumulatively:

\[
C_q=\sum_{t,j}\|y_{t,j}-y^{reg}_{t,j}\|_2.
\]

Because `v` is unit norm this equals the absolute scalar correction summed over events.

Also record native harmful pressure before regulation:

\[
E_q=\sum_{t,j}[v^Ty_{t,j}-q_{ref}]_+.
\]

Report:

- cumulative `C_q`;
- cumulative native `E_q`;
- active-regulation event fraction;
- mean/max scalar correction;
- Q2 controller state `m_t`;
- checkpoint values at contexts 64,128,256,512,768,1024.

These quantities test whether microscopic co-adaptation continues to push against the functional constraint even when the harmful coordinate is prevented from being expressed.

## Terminal deployment endpoints

Primary deployment endpoints remain

\[
F_{61},\qquad D_{non61},
\]

with lower values better.

For `A in {Q1,Q2}` define paired terminal contrasts against B:

\[
\Delta_{A,m}=J_m(A)-J_m(B).
\]

Use 10,000 paired bootstrap resamples.

A primary endpoint is rescued iff the 95% CI of `Delta` is entirely below zero.

Report gap closure relative to the historical B-vs-F deficit:

\[
G_{A,m}=\frac{J_m(B)-J_m(A)}{J_m(B)-J_m(F)}.
\]

The point estimate and paired-bootstrap ratio CI are descriptive/supportive; primary significance is determined by the paired contrast.

## Synthesis guardrails

Because D9A now evaluates a candidate trajectory-level control principle rather than attribution alone, terminal utility requires multi-objective guardrails.

With regulator OFF, evaluate masked CE at ratios 0.25, 0.5, 0.9, 1.0 and rollout:

- `global_js_nats`;
- `spatial_js_nats`;
- `pair_js_nats`.

For each regulated arm, use the same frozen D7A material-degradation limits relative to the better of F/B:

- masked CE: `<= 1.01 * best(F,B) + 1e-8`;
- each JS: `<= 1.02 * best(F,B) + 1e-6`.

## Formal classification

Classification is determined by Q1 first. Q2 cannot rescue a failed Q1 theory.

### R3_HARD_AND_DYNAMIC_FUNCTIONAL_RESCUE

- Q1 improves both primary endpoints with 95% CIs entirely below zero;
- Q1 masked-CE and rollout-JS guardrails all pass;
- Q2 also improves both primary endpoints with 95% CIs entirely below zero;
- Q2 masked-CE and rollout-JS guardrails all pass.

Interpretation: both the privileged hard functional constraint and the fixed slow scalar approximation reshape learning successfully.

### R2_HARD_FUNCTIONAL_RESCUE_DYNAMIC_APPROX_FAIL

- Q1 improves both primary endpoints;
- Q1 all guardrails pass;
- Q2 fails either dual-primary rescue or its synthesis guardrails.

Interpretation: the causal coordinate survives the trajectory-level test, but this first fixed dynamic controller is not yet an adequate approximation. Do not tune it inside D9A.

### R1_HARD_PRIMARY_RESCUE_WITH_GUARDRAIL_TENSION

- Q1 improves both primary endpoints;
- at least one Q1 synthesis guardrail fails.

Interpretation: direct functional regulation has primary deployment value but creates a multi-objective tradeoff.

### R0_NO_HARD_FUNCTIONAL_TRAJECTORY_RESCUE

Q1 fails significant rescue on one or both primary endpoints.

Interpretation: D8B terminal causal utility does not establish that persistent trajectory-level regulation of this coordinate is sufficient to produce a better terminal learner.

## Interpretation boundary

Even R3 does not establish the final local learning rule. Q1 directly modifies activation space using a discovery-derived privileged reference; Q2 uses a globally available scalar projection `v^T y`. D9B is authorized only if Q1 succeeds. D9B must replace explicit y-space projection with a local/SNN/MSDL-compatible plasticity coordinator driven by the same scalar causal coordinate.
