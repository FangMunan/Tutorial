# H1.1 — frozen checkpoint -> SpiNNaker HIR status

Branch: `snn-transform-spinnaker-h0-h1-20261014`

## Scope

H1.1 replaces the representative H1 parameters with the actual frozen U5
`SNNTransformBlock_v1.pt` checkpoint and produces the hardware-neutral HIR used
by H2.  H1.1 is a CPU/CI fixed-point backend result; it is **not** a SpiNNaker
board result.

Source checkpoint:

- U5 workflow run: `38538624735`
- artifact: `SNN_TRANSFORM_BLOCK_V1`
- checkpoint SHA256: `e86fdbcef1ed32652cc3547642d2890a6b40166dbb0242fda966e4a1fbc2dcc5`

## Frozen numerical contract

- main activations/weights: 20 fractional bits;
- Q feature cache: 24 fractional bits;
- K feature cache / g: 16 fractional bits;
- H resident state: 20 fractional bits;
- Izh membrane: 22 fractional bits;
- decoder coefficients: 20 fractional bits.

Q/K/V event execution is exact for the validation trajectory: `768 -> 288`
evaluations per projection, a `62.5%` reduction.  The sparse tail contains
exactly `1280/5120 = 25%` static connections.  H/g/pq/pk/v remain signed-int32
safe; required products and accumulators use signed int64.

## Batch-isolation bug — CLOSED

H1.1 exposed a Python reference bug in the persistent update path: an H/g delta
computed for batch item `b` was broadcast over the complete batch.

The canonical v1.1 implementation now updates only:

```text
H[b:b+1]
g[b:b+1]
```

A regression test changes only one item of a B=3 batch and verifies:

- unaffected samples keep H and g bitwise unchanged;
- the changed sample is bitwise identical to an isolated B=1 execution.

CI result: **PASS**.  The previously measured batch residue is now exactly zero
for output, H and g.

## Functional H1.1 backend contract — FROZEN

The chosen hardware handoff is explicitly **functional**, not the stricter
full-hidden-representation criterion.

Contract version: `H1.1-functional-v9`.

Post-fix actual-checkpoint result:

- max block-output relative-L2: `0.02612186796` (~2.612%);
- H relative-L2: `0.0001704371`;
- g relative-L2: `0.0002981541`;
- pq relative-L2: `0.0002389603`;
- pk relative-L2: `0.0001897077`;
- v relative-L2: `1.83149e-06`;
- frozen task-readout prediction relative-L2: `0.0024357253` (~0.244%);
- task sign accuracy: `1.0 -> 1.0`;
- event reduction: `62.5%`;
- sparse-tail static fraction: `25%`.

Functional acceptance therefore **PASSes**.  The deliberately stricter
full-block representation gate remains:

```text
required <= 1%
observed  = 2.612%
strict representation PASS = false
```

This failure is retained explicitly and must not be reported as passed.

## Reproducibility record

H1.1 functional workflow run: `38575839809`.

Artifact:

- name: `SNN_TRANSFORM_SPINNAKER_H1_1_FUNCTIONAL_V9`
- artifact ID: `11900355732`
- ZIP SHA256: `35ba4dacde216ec1802274c4d415dd007d8ed0dc80e767d0075b9fb75a61bb78`

Frozen HIR inside the artifact:

`SNNTransformBlock_v1_1_HIR_H1_1_FUNCTIONAL_V9.npz`

## Decision

**H1.1 is closed for the functional hardware route.**

H2 must consume this frozen contract without changing the scientific operator.
The remaining ~2.6% internal block-representation error is a documented H1.1
limitation, not a reason to silently modify H2 mathematics.
