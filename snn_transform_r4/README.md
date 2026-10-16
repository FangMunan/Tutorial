# R4 — Single-module recovery gate

R4 begins only after E29 established that the trained B0 solution survives the B0 -> H/g attention replacement.

This gate does **not** attempt full-network learning. It asks a narrower question:

> Can forward-parameter-only MSDL repair one damaged parameter matrix while every other forward parameter remains correct?

The first target is the final block `W2 = blocks[3].f2.weight` because it is the cleanest causal case. After this matrix there is no cross-token mixing. For each masked token i we can inject an antithetic perturbation into the W2 output and pair the resulting token-local loss difference with the local presynaptic hidden state.

For perturbation \(\xi_i\):

\[
M_i = \frac{\ell_i(+\sigma\xi_i)-\ell_i(-\sigma\xi_i)}{2\sigma}
\]

and local eligibility

\[
E_i = \xi_i h_i^T.
\]

The batch estimator is

\[
G = \frac{1}{N_M}\sum_{i\in M}M_iE_i.
\]

Fast and slow non-learned MSDL states low-pass this estimator before the W2 update. There is no backpropagation, optimizer, transpose transport, or learned feedback network.

## Damage

`blocks[3].f2.weight` is set to zero. Its bias and every other model parameter are untouched.

## Pass gate

R4 passes only if:

- damage and learning are isolated to the one target matrix;
- no autograd gradient tensors are created;
- the damage increases 50% mask CE by > 0.03;
- MSDL recovers > 50% of the CE gap to the intact E29 model;
- recovered CE is better than damaged CE at 25%, 50%, and 90% masks.

Gold W2 is retained only for post-hoc distance/cosine diagnostics and is never used by the learning rule or checkpoint selection.
