# Gate D4A — terminal compensatory co-adaptation mapping

D3 is closed. D4A is an **eval-only terminal causal mapping** stage. It does not retrain the learner, modify E/M states, change the MSDL rule, or introduce metaplasticity.

## Frozen causal question

Which `F_EM`-induced terminal forward-parameter changes suppress the large beneficial effect of `W2^F`?

Let `theta^N` and `theta^F` be the provenance-correct D3E native and `F_EM` terminal forward-parameter states. `W2` is the focal perturbation. For each terminal block `i`, D4A evaluates:

- `i`: native background with only block `i` transplanted from `F_EM`;
- `W2+i`: native background with `W2` and block `i` transplanted from `F_EM`;
- `F-i`: full `F_EM` background with block `i` reverted to native.

No terminal E/M plastic state is transplanted or interpreted in D4A.

## Blocks

- `Q = {WQ,bq}`
- `K = {WK,bk}`
- `V = {WV,bv}`
- `O = {WO,bo}`
- `Norm = {gamma2,beta2}`
- `FF1 = {W1,b1}`
- `Readout = {WR}`
- `Embedding = {tok,pos}`
- `b2 = {b2}`

`W2 = {W2}` remains the focal block and is not merged with `b2`.

## Primary causal endpoints

The primary higher-is-worse phenotype coordinates remain:

- `F61 = P(61|GT!=61)_self - P(61|GT!=61)_teacher`
- `D_non61 = CE_non61,self - CE_non61,teacher`

Supportive: `C61`, `delta_P61`. Gate-A `CE` is retained as a secondary causal coordinate. All evaluations reuse the exact D3E 256 validation images, historical committed-position schedule, common-random-number field, and paired per-image self/teacher protocol.

## Three D4A contrasts

For each metric `m` and block `i`:

`S_i,m = J_m(W2+i) - J_m(W2)`

measures **antagonism sufficiency**. Positive values mean adding the F_EM block to the beneficial W2-only hybrid makes the phenotype worse.

`N_i,m = J_m(F) - J_m(F-i)`

measures **antagonism necessity**. Positive values mean reverting the block to native within the full F_EM background improves the phenotype.

`I_W2,i,m = J_m(W2+i) - J_m(W2) - J_m(i) + J_m(N)`

measures W2-centered non-additivity / epistasis. Positive values support block-specific incompatibility with `W2^F` rather than a purely additive block defect.

All three contrasts are formed per image first, then summarized with paired bootstrap confidence intervals.

## Non-additivity boundary

Single-block `S_i`/`N_i` values and percentages of the D3E antagonistic load are **not additive attributions**. They are ranking/causal-strength diagnostics only. If no single block explains the terminal antagonism bidirectionally, the correct next conclusion is distributed higher-order co-adaptation and D4C pair testing, not forced linear decomposition.

## Hard implementation gates

Before interpreting new blocks, the D4A evaluator must reproduce the D3E artifact for:

- Native `N`;
- `W2`-only transplant;
- `b2`-only transplant;
- `W2+b2` transplant.

For those calibration systems, stored per-image Gate-A effect arrays must match within `1e-7`, and proposal IDs must match exactly. D3E provenance gates in the source artifact must all be true.

Every generated terminal hybrid is checked for transplant purity: selected forward fields must be bitwise equal to the intended donor and every non-selected forward field must remain bitwise equal to the intended background.

## D4A interpretation

A block is strongest evidence for a specific compensatory subsystem when the same primary endpoint shows:

- positive sufficiency `S_i`;
- positive necessity `N_i`;
- positive W2 epistasis `I_W2,i`;

with paired uncertainty reported. D4A establishes terminal antagonism/co-adaptation structure; it does not establish the training-time mechanism that produced that structure.

## Next stages

D4B is allowed only for top D4A blocks and performs interpolation `theta_i(lambda)=theta_i^N+lambda(theta_i^F-theta_i^N)` at `lambda={0,.25,.5,.75,1}` while holding `W2=W2^F` (plus a W2 dose-response). D4C is entered only if single-block mapping is insufficient, using a small preregistered set of structurally motivated pairs rather than exhaustive `2^9` factorial testing.
