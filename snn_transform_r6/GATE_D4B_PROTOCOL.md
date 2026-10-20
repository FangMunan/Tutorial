# Gate D4B — O-block decomposition and dose-response attribution

## Frozen scientific question

D4A identified the output-projection O block `(WO, bo)` as the dominant single terminal antagonist of the beneficial `W2^F_EM` state. D4B asks whether that harmful terminal adaptation is carried by `WO`, `bo`, or their joint change, and whether the effect varies continuously between native and F_EM terminal states.

D4B is **eval-only**. It performs no training, does not modify the learner, and does not intervene on E/M states.

## Frozen source objects

Use the provenance-correct D3E terminal systems from Run `39454173518`:

- `D3E_N_terminal.pkl`
- `D3E_F_EM_terminal.pkl`

Use the successful D4A artifact from Run `39483623999` for calibration and exact causal-coordinate reuse. Gate-A uses the same 256 validation images, historical position schedule, CRN field, and self/teacher intervention protocol.

## Part A — exact O decomposition

Let `N` and `F` denote native and F_EM terminal forward states. Let `WO^F`, `bo^F`, and `W2^F` denote the corresponding donor fields.

Evaluate:

- Native background: `N`, `WO`, `bo`, `O=(WO,bo)`
- W2-beneficial background: `W2`, `W2+WO`, `W2+bo`, `W2+O`
- Full-F reverse background: `F`, `F-WO`, `F-bo`, `F-O`

Primary endpoints remain `F61` and `D_non61`. `C61` and `delta_P61` are supportive; Gate-A CE is retained diagnostically.

For component `x in {WO, bo}` define:

- Sufficiency: `S_x = J(W2+x) - J(W2)`
- Necessity: `N_x = J(F) - J(F-x)`
- W2 interaction: `I_W2,x = J(W2+x)-J(W2)-J(x)+J(N)`

Define within-O interaction:

`I_WO,bo = J(O)-J(WO)-J(bo)+J(N)`.

All contrasts are paired per-image and receive 10,000-replicate paired-bootstrap confidence intervals.

## Part B — dose-response interpolation

For `x in {WO, bo, O_joint}` use

`x(lambda) = x^N + lambda (x^F - x^N)`

at `lambda = {0, 0.25, 0.5, 0.75, 1}` in three backgrounds:

1. Native: `N + x(lambda)`
2. W2-beneficial: `N + W2^F + x(lambda)`
3. Full-F reverse: all other parameters fixed at F, while `x=x(lambda)`

The F-background convention gives `lambda=0 -> F-x` and `lambda=1 -> F`.

For every image and endpoint retain the complete causal-effect vector. For every lambda report paired change from lambda=0 and the normalized aggregate dose response

`R_m(lambda) = [J_m(lambda)-J_m(0)] / [J_m(1)-J_m(0)]`.

Bootstrap the normalized ratio by paired image resampling. Also report adjacent-step paired contrasts; monotonicity alone is not treated as sufficient evidence.

The W2-vs-native dose interaction curve is

`D_x(lambda) = [J_W2,x(lambda)-J_W2,x(0)] - [J_N,x(lambda)-J_N,x(0)]`.

Near-zero `D_x(lambda)` supports approximate background-transferability/additivity; systematic nonzero values support background dependence even if endpoint epistasis at lambda=1 is small.

## Calibration hard gate

Before interpreting D4B, reproduce D4A exactly for:

- `N`
- `F`
- `W2`
- `O`
- `W2+O`
- `F-O`

Require proposal IDs exact and finite metric/effect entries within `1e-7`, with identical NaN/Inf masks. If calibration fails, stop before decomposition/interpolation interpretation.

## Interpretive cases

- **B1 — WO-dominant:** WO alone is bidirectionally harmful on both primary endpoints (positive S and N CIs), bo is small/non-bidirectional, and WO interpolation gives a graded reproducible dose response.
- **B2 — bo-dominant:** symmetric criterion for bo.
- **B3 — joint O-block:** individual components are insufficient but positive within-O interaction plus joint interpolation supports coordinated WO/bo harm.
- **B4 — transplant/off-manifold-like:** the full effect appears only near the endpoint or interpolation/reverse-interpolation is inconsistent; D4A transplant interpretation must be weakened.
- Otherwise report mixed evidence without forcing a case.

No D4B result by itself proves a training-time compensatory mechanism. D4B attributes terminal forward-parameter causality.