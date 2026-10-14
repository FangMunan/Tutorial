# SNN-Transform unified project status — U4

Canonical development line: `snn-transform-u4-sparse-tail-20261014`

## Frozen / accepted results

### U0/U1 — lightweight linear-attention block
PASS. Positive-RF normalized linear attention with persistent H/g is the reference computation. Izh feature replacement is treated as a neuronal-dynamics interface, not yet as a strict spike-only claim.

### U2 — event-cached Q/K/V
PASS as an algebraic/event-execution gate. On the lightweight trajectory, token projection evaluations fall from 384 to 144 (62.5% reduction) with zero numerical error in the cache test. Integration caveat: the current main `LinearAttentionCore.step` still evaluates Q globally; the Q-cache proof remains to be folded into the canonical implementation.

### U3.1 — trained-distribution Izh interface
PASS on the dense-tail trained retrieval block. Held-out Q+K replacement reached ~0.446% prediction relative-L2 with unchanged 100% task sign accuracy.

### Mamba-like selective retention
PARKED, not rejected. The mechanism learned different retention for reset/non-reset inputs, but U3B did not establish a functional task advantage because the gate task itself remained at chance. Revisit only when a long-context/context-switch workload actually requires selective forgetting.

## U4 — sparse Output Projection + FFN

### U4A — frozen dense random teacher screen
FAIL, as expected for the wrong equivalence question. A sparse network is not expected to reproduce an arbitrary frozen dense random mapping exactly. This result is retained only as a negative control.

### U4B — task-trained sparse architecture
PASS.

Strongest tested point:
- output projection connectivity density = 25%
- FFN input connectivity density = 25%
- FFN output connectivity density = 25%
- task sign accuracy = 100%
- shuffled key/value binding control sign accuracy ~54%
- static tail synapses = 1280 vs 5120 dense (25%)
- hidden activity ~64.6%
- effective tail SynOp proxy ~21.5% of dense (~78.5% reduction)

Scientific conclusion: the lightweight retrieval function can be learned by a structurally sparse Output+FFN realization. This establishes U4 sparse-tail feasibility; it is not a claim that an arbitrary trained dense tail can be losslessly pruned.

### U4C/U4D — composition with post-hoc Izh feature replacement
Full strict composition gate is not frozen yet.

With 25%-25%-25% sparse tail, exact positive-RF task remains excellent, but post-hoc Izh Q+K replacement gives ~2.87% prediction relative-L2 while preserving 100% task sign accuracy.

U4D localization shows:
- Q-only prediction rel-L2 ~0.72%
- K-only prediction rel-L2 ~2.49%
- Q+K prediction rel-L2 ~2.87%
- H/g state rel-L2 only ~0.13–0.15%
- attention-output rel-L2 ~12.3%
- sparse FFN/tail does **not** amplify the error; downstream stages reduce it to ~2.87%
- hidden gate flips are ~1%

Therefore the remaining strict-parity issue is the K-side Izh / normalized H-g attention interface under the sparse-trained upstream distribution, not the sparse Output Projection or sparse FFN.

## Current architecture status

```text
Input
  -> Q/K/V projections
  -> positive-RF / Izh feature dynamics
  -> persistent H/g linear attention
  -> 25% structurally sparse output projection       [U4 PASS]
  -> 25% structurally sparse FFN input connections   [U4 PASS]
  -> thresholded hidden population (~65% active)
  -> 25% structurally sparse FFN output connections  [U4 PASS]
  -> residual / normalization
```

## Next gate

Do not reopen sparse-tail topology unless robustness/scaling later requires it. The next integration task is to make the K-side neuronal feature interface robust to the sparse-trained attention distribution (or train the block with that interface in the loop), while preserving the U4 sparse connectivity and task performance.

No scaling or SpiNNaker timing claim is made from U4 cloud SynOp proxies.
