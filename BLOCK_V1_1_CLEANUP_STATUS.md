# SNNTransformBlock-v1.1 cleanup status

Branch: `snn-transform-block-v1.1-clean-20261014`

Purpose: perform the mathematical-audit cleanup of frozen Block-v1 without changing the accepted architecture or scientific meaning.

## Changes applied

- absorbed historical static FFN `theta` exactly into `f1.bias` during v1 checkpoint migration;
- removed fixed-alpha retention plumbing and `weight` state from the core;
- removed raw `x_cache` and automatic changed-token inference from the core;
- removed validation/profiling paths from the production operator;
- reduced persistent mutable-token state to `S=(H,g,pq,pk,v)`;
- replaced full Izh membrane-trajectory materialization by online linear decoding;
- retained the accepted Q/K/V, H/g attention, sparse output projection, sparse FFN, LayerNorm and residual structure unchanged;
- retained dense-mask `SparseLinear` only as the portable PyTorch reference representation; compact sparse storage is deferred to backend compilation.

## Parity gate

GitHub Actions workflow: `SNN Transform Block v1.1 Clean Parity`.

Latest code gate result: **PASS**.

Measured v1 -> v1.1 relative-L2 maxima over a six-step mutable-token trajectory:

### Exact PositiveRF path

- output: `1.6738979979891155e-08`
- H: `0`
- g: `0`
- pq: `0`
- pk: `0`
- v: `0`

### Izh neuronal feature path

- output: `2.6534499042440984e-08`
- H: `1.632184876143583e-07`
- g: `1.5525006347161252e-07`
- pq: `5.149065884779702e-08`
- pk: `4.382476248565581e-08`
- v: `0`

Native v1.1 checkpoint round-trip relative-L2: `0`.

Residue gates all passed:

- `theta` absent from v1.1 parameters;
- state fields exactly `{H,g,pq,pk,v}`;
- no `x_cache` in core state;
- no `weight` in core state;
- persistent calls require explicit changed-token events;
- native checkpoint format is `SNNTransformBlock-v1.1`.

## Decision

**Freeze the clean v1.1 mathematical core for the next engineering stage.**

The next SpiNNaker work should change execution/backend representation only. It should not reintroduce removed convenience state or alter the accepted mathematical operator unless a hardware constraint exposes a specific equivalence problem.
