# H1.1 — frozen checkpoint -> SpiNNaker HIR status

Branch: `snn-transform-spinnaker-h0-h1-20261014`

## Scope

H1.1 replaces the representative H1 parameters with the actual frozen U5
`SNNTransformBlock_v1.pt` checkpoint.  It remains a CPU/CI fixed-point backend
proof; it is **not** a SpiNNaker board result.

Source checkpoint:

- U5 workflow run: `38538624735`
- artifact: `SNN_TRANSFORM_BLOCK_V1`
- checkpoint SHA256: `e86fdbcef1ed32652cc3547642d2890a6b40166dbb0242fda966e4a1fbc2dcc5`

## Verified backend facts

Across actual-checkpoint H1.1 experiments:

- Q/K/V event execution count is exact: `768 -> 288` evaluations per projection,
  a `62.5%` reduction for the six-step 25%-change trajectory;
- sparse tail compiles to exactly `1280/5120 = 25%` static connections;
- trained K/V/H/g resident fixed-point states are numerically stable and fit in
  signed 32-bit storage under the selected range-aware contract;
- integer products/accumulators use signed 64-bit where required;
- no task-label training is used by the hardware-interface calibration attempts.

The stable range-aware contract used by the best run is:

- main activations/weights: 20 fractional bits;
- Q feature cache: 24 fractional bits;
- K feature cache / g: 16 fractional bits;
- H resident state: 20 fractional bits;
- Izh membrane: 22 fractional bits;
- decoder coefficients: 20 fractional bits.

## Best current result: v9

v9 uses label-free, attention-aware selection of the existing per-head Izh
operating point / linear trajectory decoder.  The scientific operator is not
changed.

Best actual-checkpoint six-step result:

- max block-output relative-L2: `0.0301634381` (~3.02%);
- H relative-L2: `0.000170184`;
- g relative-L2: `0.000297349`;
- pq relative-L2: `0.000243679`;
- pk relative-L2: `0.000189614`;
- v relative-L2: `1.83149e-06`;
- frozen task-readout prediction relative-L2: `0.00252115` (~0.252%);
- task sign accuracy: `1.0 -> 1.0`.

Thus the **functional task/readout parity is below 1%**, but the deliberately
strict internal block-output <=1% representation gate is not met.

The remaining bottleneck is localized to the conditioning of the normalized
query read

`A(q;H,g) = q^T H / (q^T g)`.

The trained K/H/g state is strongly anisotropic.  Tiny unweighted Q-feature
errors can therefore produce percent-level attention errors.  In v9 the best
held-out internal attention error for head 0 within the current shared-decoder
Izh family was ~2.48%; head 1 reached ~0.872%.

## Negative / diagnostic experiments

- simple higher fractional precision did not solve the problem;
- hardware-aware linear decoder refitting improved but did not close strict
  block parity;
- unweighted/projective Q calibration could look excellent while producing poor
  end-to-end attention, confirming the anisotropic conditioning issue;
- sensitivity-weighted and direct nonlinear decoder experiments did not improve
  the v9 result and are retained as negative controls, not promoted.

## Discovered reference implementation residue

H1.1 exposed a batch-isolation bug in the v1/v1.1 Python persistent update path:
inside a loop over batch index `b`, H/g deltas were broadcast over the full batch
instead of being applied to `H[b:b+1]` and `g[b:b+1]` only.

Measured one-update discrepancy vs the intended independent-batch equations:

- output rel-L2: `0.000406198`;
- H rel-L2: `3.90791e-05`;
- g rel-L2: `5.25212e-05`.

This is a software indexing residue, not a scientific operator change.  It must
be patched before H2 and accompanied by a batch-isolation invariant test.

## Current decision

H1.1 is **functionally successful but not formally closed under the strict
<=1% full-block representation gate**.

Do not silently relax that gate.  Before H2, choose one of two explicit paths:

1. **Strict representation path:** continue interface work until block-output
   rel-L2 <=1%; or
2. **Functional hardware path:** accept the v9 HIR because the frozen task
   readout is 0.252% rel-L2 with 100% sign accuracy, while all resident state,
   event-count, sparsity and integer-range gates pass.

Regardless of that choice, patch the batch-isolation bug before generating APLX.
