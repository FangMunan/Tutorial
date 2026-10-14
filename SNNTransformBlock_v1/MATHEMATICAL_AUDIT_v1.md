# SNNTransformBlock-v1 mathematical audit

Date: 2026-10-14

Purpose: compare the frozen block against a standard modern Pre-LN Transformer block, identify exact mathematical redundancies and implementation residue, and separate those from operators that are essential for function or event scaling.

## 1. Reference block

A standard Pre-LN Transformer block can be written as

\[
Z = X + W_O\,\mathrm{Attn}(\mathrm{LN}_1(X)),
\]
\[
Y = Z + W_2\,\sigma(W_1\mathrm{LN}_2(Z)+b_1)+b_2.
\]

The frozen v1 has the same outer residual/normalization scaffold. Its attention operator is replaced by normalized positive-random-feature linear attention, and its output/FFN channel mixers are structurally sparse.

For head \(h\), token \(i\):

\[
q_i=W_Q\mathrm{LN}_1(x_i),\quad k_i=W_K\mathrm{LN}_1(x_i),\quad v_i=W_V\mathrm{LN}_1(x_i),
\]
\[
H=\sum_j \phi(k_j)v_j^T,\qquad g=\sum_j\phi(k_j),
\]
\[
a_i=\frac{\phi(q_i)^T H}{\phi(q_i)^T g},
\]
\[
Z=X+W_O^{\mathrm{sparse}}a,
\]
\[
Y=Z+W_2^{\mathrm{sparse}}\,\mathrm{ReLU}(W_1^{\mathrm{sparse}}\mathrm{LN}_2(Z)+b_1-\theta)+b_2.
\]

Thus the block-level topology is a valid Pre-LN Transformer block. The nonstandard parts are deliberate operator replacements, not missing block components.

## 2. Keep: mathematically/functionally essential

The following are not redundant and should remain in the canonical architecture:

- two residual paths;
- two per-token LayerNorms;
- Q/K/V projections;
- head split/concatenation;
- positive feature map / frozen per-head Izh feature interface;
- persistent \(H\) and \(g\);
- sparse output projection \(W_O\);
- sparse two-layer FFN;
- cached \(\phi(q)\), \(\phi(k)\), and \(v\) in mutable-token event execution.

Why the caches are needed: if a token is replaced, the old K/V contribution must be removed from \(H,g\) without recomputing all tokens. Cached Q features also avoid recomputing Q for unchanged tokens when a full output tensor is requested.

## 3. Exact mathematical redundancies in frozen v1

### 3.1 `theta` is redundant with the first FFN bias

Current hidden activation:

\[
h=\mathrm{ReLU}(W_1u+b_1-\theta).
\]

Define

\[
\tilde b_1=b_1-\theta.
\]

Then

\[
h=\mathrm{ReLU}(W_1u+\tilde b_1)
\]

for every input. Therefore `theta` can be removed exactly by replacing `f1.bias <- f1.bias - theta` when converting a frozen checkpoint. This changes neither function nor asymptotic scaling. A future genuinely dynamical neuron threshold should be introduced as a neuronal state/parameter, not kept as a duplicate static FFN bias.

### 3.2 `weight` state is redundant when v1 retention is fixed at alpha=1

Frozen v1 uses \(\alpha=1\). `weight` is initialized to one, multiplied by one at every step, and reset to one for changed tokens. Hence by induction

\[
\mathrm{weight}_{bhi}=1
\]

for all steps. All uses of `old_weight` therefore reduce to multiplication by one. The state tensor and all generic retention plumbing can be removed from the v1.1 core with exact equivalence.

Selective/Mamba-like retention should return later as an extension, not remain latent inside the frozen fixed-retention core.

## 4. Engineering residue that should move out of the mathematical core

### 4.1 `x_cache` and automatic changed-token inference

`x_cache` is not part of attention mathematics. It exists only to compare current and previous raw input and infer a `[B,N]` change mask.

This convenience path costs \(O(BND)\) comparison work and \(O(BND)\) extra memory, which can erase the point of event-driven execution at scale. The scalable core should receive changed-token events explicitly. Automatic comparison can live in a wrapper for debugging/general PyTorch use.

Minimal core persistent state after removing fixed-retention residue:

\[
\boxed{S=(H,g,p_q,p_k,v)}.
\]

### 4.2 `force_project_all`

This is a validation reference path, not an operator. It should be removed from the production core and retained only in parity tests.

### 4.3 `last_stats`, clamp statistics, and synchronous scalar extraction

Calls such as `float(tensor.mean().detach())` cause host synchronization in accelerator execution and have no role in the block function. Profiling/accounting should be optional external instrumentation.

### 4.4 smoke test and accounting helpers

`_smoke_test()` and `synapse_account()` are useful engineering tools, but should not define the mathematical block. Keep them in tests/profiling utilities.

## 5. Scaling-relevant exact implementation cleanup

These do not change the mathematics, but matter strongly for later scaling/hardware work.

### 5.1 Do not materialize the whole Izh membrane trajectory

The current Izh decoder computes

\[
r=d_0+\sum_{t=1}^{T} d_t v_t.
\]

The full tensor \((v_1,\ldots,v_T)\) is therefore unnecessary. The same result can be accumulated online:

\[
r_0=d_0,\qquad r_t=r_{t-1}+d_t v_t.
\]

This reduces trajectory memory from \(O(BHNmT)\) to \(O(BHNm)\) while remaining exactly equivalent. It also maps directly to on-board resident decoding.

### 5.2 Sparse matrices should eventually be stored sparsely

Current `SparseLinear` stores a dense weight matrix plus a dense binary mask and calls dense `F.linear`. It represents the correct sparse graph, but it does not realize sparse memory/compute in PyTorch.

The exact sparse operator is

\[
y_o=b_o+\sum_{j\in S_o}w_{oj}x_j.
\]

A scalable implementation should store only index sets \(S_o\) and corresponding weights. This preserves the function and makes the 25% connectivity claim true at storage/execution level rather than only at graph level. SpiNNaker should naturally use this representation as synapse lists.

### 5.3 Python loops are backend residue

The per-batch loop over changed token indices is mathematically harmless but not a scalable execution strategy. A cloud backend can vectorize it; a SpiNNaker backend should consume sparse event packets directly.

## 6. Important non-redundancies / scaling caveats

### 6.1 Event-cached Q/K/V does not imply a fully event-sparse block

When any K/V token changes, \(H,g\) change globally, so every query readout can change:

\[
a_i'=\frac{\phi(q_i)^TH'}{\phi(q_i)^Tg'}.
\]

Therefore the current block still performs attention readout and the tail for all queried tokens. Event caching exactly reduces Q/K/V reprojection work, but does not by itself guarantee sparse activity through deep stacks.

This must be measured during scaling rather than assumed.

### 6.2 Mutable-token caches are not redundant in the current workload

For iterative masked/replacement workloads, cached \(p_k,v\) are required to subtract old contributions. For append-only causal workloads a specialized backend could avoid full per-token K/V cache and keep only running \(H,g\). That is an execution specialization, not a reason to delete the mutable-token state from the generic block.

### 6.3 `m`, event density, and hidden activity are scaling variables

Linear attention is favorable while feature count \(m\ll N\), but later scaling must measure whether required \(m\) grows with sequence/model size. Likewise measure layerwise event density and hidden activity rather than assuming the lightweight values remain constant.

## 7. Standard-block comparison verdict

### Structurally aligned

- Pre-LN residual scaffold: YES.
- Multi-head Q/K/V + output projection role: YES.
- Two-layer FFN role: YES.
- Positional encoding inside the block: not required; correctly external.
- Dropout: absent, but not a mathematical requirement of the block and irrelevant to frozen inference structure.

### Deliberate replacements

- softmax attention -> normalized positive-RF linear attention / Izh feature interface;
- dense output/FFN mixers -> structurally sparse mixers;
- standard stateless attention execution -> persistent event-cached H/g execution.

These are the scientific content of the architecture and should not be removed as "redundancy."

## 8. Recommended v1.1 clean-core boundary

Production mathematical core:

```text
Input X + explicit changed-token events
  -> LN1
  -> event Q/K/V projection updates
  -> per-head feature dynamics
  -> minimal state S=(H,g,pq,pk,v)
  -> normalized H/g read
  -> sparse WO
  -> residual
  -> LN2
  -> sparse W1 + single bias
  -> exact-zero ReLU population
  -> sparse W2
  -> residual
  -> Y + new S
```

Move outside core:

- automatic `x_cache` event inference;
- generic retention/`weight` state;
- `force_project_all`;
- profiling/statistics;
- clamp diagnostics;
- smoke tests;
- SynOp accounting.

Exact-clean transformations before v1.1 freeze:

1. absorb `theta` into `f1.bias`;
2. remove fixed-retention `weight` and retention plumbing;
3. separate `x_cache` into a convenience wrapper;
4. stream the Izh linear decoder rather than storing the trajectory;
5. keep sparse topology semantically identical, then implement compact sparse storage/backend separately.

## 9. Audit conclusion

No redundant component was found in the core Transformer scaffold that would invalidate later scaling. The frozen v1 does, however, contain historical/general-purpose execution residue. Two items are exactly mathematically redundant (`theta`, fixed-alpha `weight`); several others are validation/convenience code that can create scaling overhead if left in the hot path (`x_cache` inference, runtime stats, full trajectory materialization, dense-mask sparse storage, Python event loops).

The appropriate next version is therefore a cleanup, not an architectural redesign: **Block-v1.1 should preserve the accepted function while reducing the core to the minimal state and operator graph above.**
