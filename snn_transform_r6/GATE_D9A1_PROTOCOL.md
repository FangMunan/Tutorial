# Gate D9A.1 — exact same-state plastic functional velocity audit

## Frozen prerequisite chain

This gate begins only after the following results are frozen:

- D7A = `T0_NO_W2_TRAJECTORY_RESCUE`: a parameter-specific safe-W2 trajectory can be bypassed by co-adaptation.
- D8A = `I1_SHARED_LOW_DIMENSIONAL_FUNCTIONAL_CANDIDATE`, selected rank `k=1`.
- D8B = `B1_BIDIRECTIONAL_CAUSAL_HARMFUL_FUNCTIONAL_MODE`: the frozen rank-1 `y`-space direction has bidirectional causal negative deployment utility at the terminal state.
- D9A = `R0_NO_HARD_FUNCTIONAL_TRAJECTORY_RESCUE`: absolute-state regulation of `q=v^T y` during learning is not a valid trajectory controller.

D9A.1 is **diagnostic only**. It does not alter the learner, learning rates, target order, reciprocal dynamics, masks, update boundary, W2-clamp rule, or any forward/plastic state.

## Question

Does the native plastic update itself carry measurable harmful-oriented functional motion along the frozen D8B direction?

The gate does **not** assume that every positive motion is harmful. It asks whether the known harmful trajectories carry an excess positive plastic-functional flux relative to the clean reference.

## Frozen direction

Use the exact D8A/D8B rank-1 `y`-space basis

\[
v\in\mathbb R^{96},\qquad \|v\|_2=1,
\]

with the orientation already causally fixed by D8B: positive displacement along `v` is the harmful orientation in the frozen R/C comparison.

No basis fitting, sign flip, PCA, threshold search, or re-estimation occurs in D9A.1.

## Trajectories

Replay exactly three synchronized 1024-context trajectories from the same frozen 2.0-epoch start:

1. `F = F_EM` — clean reference.
2. `B = F_EM_WO_zero_bo_oracle` — combined harmful learner.
3. `C = B_W2_safe_clamp` — the D7A trajectory in which, after every context, `W2_C` is replaced by the time-matched `W2_F`.

All three arms must use identical image indices, masks, mirror seeds, target order, and external RNG stream. Final F/B/C systems must replay the frozen successful D7A terminal models and Gate-A arrays within the existing frozen tolerances.

## Exact same-state plastic functional velocity

For context `t`, let `Theta_t^-` be the complete forward-parameter snapshot immediately before the context update and `Theta_t^+` the actual forward-parameter snapshot after the context update.

For C, the **primary** `Theta_t^+` is the actual post-clamp state after `W2_C <- W2_F`. The native pre-clamp C proposal is recorded separately as supportive information.

The context input is frozen to the exact training array used for that update:

\[
\xi_t=(\text{same discrete token array, same masked positions}).
\]

No recurrent/Izhikevich/plastic dynamical state is advanced by the diagnostic. The forward computation is reconstructed algebraically from the frozen context and parameter snapshot:

- token/position embedding and row normalization;
- Q/K/V projections;
- exact positive-random-feature linear attention `H/g` for that frozen context;
- `z`, layer norm, `u`, ReLU `h`, and `y`.

This stateless diagnostic consumes no learner RNG and performs no E/M, reciprocal, or Izhikevich update. A parity self-test must verify that the reconstructed pre-update `y` agrees with the learner's ordinary frozen forward computation to numerical tolerance.

For every masked position `r` in context `t`:

\[
\Delta y_{t,r}^{pl}=y(\Theta_t^+;\xi_{t,r})-y(\Theta_t^-;\xi_{t,r}).
\]

The context-level plastic functional displacement is preregistered as the mean over the masked positions:

\[
\overline{\Delta y}_t^{pl}=\frac{1}{N_t}\sum_r\Delta y_{t,r}^{pl},
\]

and

\[
\boxed{\Delta q_t^{pl}=v^T\overline{\Delta y}_t^{pl}}.
\]

This is a finite-step same-context parameter intervention, not `q_{t+1}-q_t` and not a gradient approximation.

## Primary accumulated quantities

For each arm:

\[
Q_{net}(T)=\sum_{t\le T}\Delta q_t^{pl},
\]

\[
Q_+(T)=\sum_{t\le T}[\Delta q_t^{pl}]_+,
\]

\[
Q_-(T)=\sum_{t\le T}[-\Delta q_t^{pl}]_+,
\]

\[
Q_{abs}(T)=\sum_{t\le T}|\Delta q_t^{pl}|.
\]

Cancellation is therefore visible rather than hidden by `Q_net`.

## Normalized quantities

Let

\[
\|\Delta\Theta_t^{fwd}\|_2
\]

be the Euclidean norm of the actual update over the frozen forward-parameter groups used by R6B4 (`tok`, `pos`, Q/K/V/O, FFN, readout, forward biases, and norm parameters).

Define

\[
\boxed{e_t=\frac{[\Delta q_t^{pl}]_+}{\|\Delta\Theta_t^{fwd}\|_2+\epsilon}}.
\]

Also report

\[
r_t=\frac{\Delta q_t^{pl}}{\|\overline{\Delta y}_t^{pl}\|_2+\epsilon}.
\]

`e_t` is the preregistered size-confound control used by the formal classifier. `r_t` is supportive directional geometry.

## C-specific supportive measurement

For C, record both:

- `C_native`: finite-step velocity after C's native B-like update but before the W2 clamp;
- `C_actual`: finite-step velocity after replacing `W2_C` by the time-matched `W2_F`.

Only `C_actual` enters the primary F/B/C classifier. `C_native` quantifies how much the W2 intervention removes or redirects within-context functional motion.

## Independent checkpoint diagnostic probe

Select 128 frozen validation images using a new deterministic seed. The IDs must be disjoint from both:

- the historical Gate-A 256 IDs;
- all D8A discovery IDs.

Save the IDs, labels, token arrays, and a provenance SHA256.

At contexts

\[
64,128,256,512,768,1024,
\]

use three fixed teacher-forced mask depths on this diagnostic set and compute, with no learning and no state advancement,

\[
a_t^B=\mathbb E_x[v^T(y_B(x)-y_F(x))],
\]

\[
a_t^C=\mathbb E_x[v^T(y_C(x)-y_F(x))].
\]

Report these beside cumulative `Q_net`, `Q_+`, `Q_-`, and `Q_abs`. Correlation/concordance is supportive only and never determines the formal class.

## Statistical contrasts

Because F/B/C use the exact same 1024-context stream, use paired context bootstrap (10,000 resamples) for:

- `[dq_B]_+ - [dq_F]_+`;
- `[dq_C]_+ - [dq_F]_+`;
- `e_B - e_F`;
- `e_C - e_F`.

A contrast is positive only when its 95% paired-bootstrap CI lies entirely above zero.

No arbitrary effect-size threshold is used.

## Formal mutually exclusive classification

Precedence is V1 -> V2 -> V3 -> V0.

### V1 — `V1_STRONG_PLASTIC_FUNCTIONAL_VELOCITY_SUPPORT`

Both raw positive-flux contrasts are significantly positive **and** both normalized `e_t` contrasts are significantly positive:

\[
B>F,\qquad C>F
\]

for both raw and normalized harmful-oriented plastic velocity.

Only V1 authorizes a later D9A.2 privileged plastic-velocity projection experiment.

### V2 — `V2_RAW_EXCESS_WITH_SIZE_CONFOUND`

Both raw positive-flux contrasts are significantly positive, but both normalized contrasts do not pass. The apparent velocity excess may therefore be explained by larger overall parameter motion.

D9A.2 is not authorized.

### V3 — `V3_PARTIAL_ARM_SUPPORT`

Exactly one of B or C has a significantly larger raw positive flux than F. This is insufficient to claim that functional plastic velocity tracks the parameter-rerouted compensation seen under the safe-W2 clamp.

D9A.2 is not authorized.

### V0 — `V0_NO_HARMFUL_PLASTIC_VELOCITY_EXCESS`

Neither harmful trajectory has significantly larger raw positive flux than F.

D9A.2 is not authorized.

## Interpretation boundary

Even V1 would establish only that harmful learners carry an excess finite-step plastic-functional velocity along the already frozen D8B causal direction, including after W2-safe clamping. It would **not** prove that every positive `Delta q_pl` update is individually harmful, nor would it establish a final local/MSDL/SNN-compatible rule.

The next intervention, if authorized, must therefore remain a privileged causal upper bound before any local dynamical implementation is attempted.
