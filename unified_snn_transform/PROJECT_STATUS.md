# SNN-Transform unified project status — U5 / Block v1

Canonical frozen line: `snn-transform-block-v1-20261014`

## Frozen / accepted results

### U0/U1 — lightweight linear-attention block
PASS. Positive-RF normalized linear attention with persistent H/g is the reference computation. Izh feature replacement is treated as a neuronal-dynamics interface, not as a strict spike-only claim.

### U2 — event-cached Q/K/V
PASS and now integrated into the canonical block, including Q. In the U5 6-step, 25%-change trajectory, Q/K/V token projections each fall from 768 dense evaluations to 288 event evaluations (62.5% reduction) with zero numerical difference from a dense-projection reference under the same state-update semantics.

### U3.1 — trained-distribution Izh interface
PASS on the dense-tail trained retrieval block. Held-out Q+K replacement reached ~0.446% prediction relative-L2 with unchanged 100% task sign accuracy.

### Mamba-like selective retention
PARKED, not rejected. It is deliberately excluded from Block v1. The canonical v1 uses fixed retention alpha=1 and exposes a retention argument for later experiments. Revisit only when a long-context/context-switch task actually requires selective forgetting.

## U4 — sparse Output Projection + FFN

### U4B — task-trained sparse architecture
PASS and frozen in Block v1.

Strongest structure:
- output projection density = 25%
- FFN input density = 25%
- FFN output density = 25%
- thresholded exact-zero hidden population
- task sign accuracy = 100%
- static tail synapses = 1280 vs 5120 dense (25%)
- U5 hidden activity ~63.8%
- U5 effective tail SynOp proxy ~21.38% of dense (~78.62% reduction)

This is a task-trained sparse realization, not a claim that an arbitrary dense tail can be pruned losslessly.

### U4G — stable per-head Izh operating-point calibration
PASS and frozen as the neuronal feature interface.

Each attention head uses the same Izh RS -> membrane trajectory -> fixed linear decoder principle but may have its own fixed operating point and decoder.

## U5 — canonical full block

### API / implementation
PASS.

The accepted operators are now packaged in `snn_transform_block.py` behind:

```python
y, state = SNNTransformBlock(x, state)
```

The state contains persistent H/g plus cached Q-feature, K-feature, V and per-token contribution weights. If an explicit changed-token mask is not supplied, the block infers token changes from the cached previous input. Reset is deterministic.

Canonical structure:

```text
Input
  -> LayerNorm
  -> event-cached Q/K/V projections
  -> per-head calibrated Izh Q/K feature dynamics
  -> persistent normalized H/g linear attention
  -> 25% structurally sparse output projection
  -> residual
  -> LayerNorm
  -> 25% structurally sparse FFN input
  -> thresholded exact-zero hidden population
  -> 25% structurally sparse FFN output
  -> residual
  -> Output + persistent state
```

### U5 integration results

Canonical packaging of the exact sparse reference is bit-equivalent in this gate:
- block relative-L2 = 0
- prediction relative-L2 = 0

Per-head Izh integrated candidate on held-out retrieval data:
- task prediction relative-L2 = 0.009714 (~0.971%) — PASS <=1%
- task prediction cosine = 0.9999647
- reference sign accuracy = 100%
- candidate sign accuracy = 100%
- H relative-L2 ~0.252%
- g relative-L2 ~0.360%

Event execution:
- Q evaluations: 768 -> 288 (-62.5%)
- K evaluations: 768 -> 288 (-62.5%)
- V evaluations: 768 -> 288 (-62.5%)
- event-cache vs dense-projection reference: zero output/state error in the tested trajectory

State API:
- repeated fresh execution: exact match
- reset vs fresh: exact match
- inferred changed tokens: 32/32 correct in the gate
- carry updates persistent H/g as expected

Sparse tail:
- static density = 25%
- effective fraction ~21.38%
- effective reduction ~78.62% under the tested hidden activity

### Important numerical caveat
The Izh candidate does **not** have <=1% parity for the entire hidden block representation. Held-out full block-output relative-L2 is ~5.29%, although block-output cosine remains ~0.99863. The <=1% gate is satisfied at the task/readout output and at H/g state, not at every internal channel.

Therefore the correct claim is:

> `SNNTransformBlock-v1` is a complete reusable functional/architectural block with accepted event Q/K/V, neuronal Q/K feature dynamics, persistent H/g attention, sparse output projection and sparse FFN. It is not yet a proof of <=1% equivalence for every internal hidden representation.

### Artifact
The U5 workflow emits:
- `SNNTransformBlock_v1.pt`
- `U5_RESULTS.json`
- `run.log`

Artifact name: `SNN_TRANSFORM_BLOCK_V1`.

## Next stage

The architecture itself should now be treated as frozen for the next experiment. Do not reopen operator design unless a downstream task exposes a specific failure.

Next: plug `SNNTransformBlock-v1` into a small independent workload (associative/content retrieval plus tiny iterative masked generation) without changing the block internals. This tests whether the block is genuinely reusable rather than a one-task construction. Scaling and SpiNNaker mapping come after that downstream plug-in gate.
