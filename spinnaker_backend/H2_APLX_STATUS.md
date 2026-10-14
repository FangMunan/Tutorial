# H2 — APLX resident backend status

Branch: `snn-transform-spinnaker-h0-h1-20261014`

## Frozen input contract

H2 consumes `H1.1-functional-v9`.  H1.1 functional parity is accepted; strict
full-block representation parity remains separately reported as failed at
~2.612% max block-output rel-L2.

H2 is not permitted to change the v1.1 scientific operator to improve this
number.

## H2.0 / H2A — fixed-point C kernel bring-up

Implemented in `spinnaker_backend/h2_aplx/`:

- `h2_kernels.h/.c`
  - Q20 dense linear kernel;
  - compact sparse-row Q20 linear kernel;
  - batch-free resident H/g subtract-old/add-new token update;
  - normalized H/g read;
  - overflow-safe scaled integer division that avoids materializing
    `num << 16`;
- `test_h2_kernels.py`
  - host-native C compilation;
  - bit-exact comparison with Python integer reference;
- `snn_transform_h2.c`
  - first SpiNNaker/SARK arithmetic self-test application source;
- `Makefile`
  - `spinnaker_tools` `app.make` build convention.

### Verified H2A result

GitHub Actions run: `38576713519`.

Host C gate: **PASS**.

Measured gate output:

```text
scaled_div_cases: 4997
attention_read_max_abs_diff: 0
sparse_test_nnz: 512
contract: H1.1-functional-v9
```

The C source also passes `gcc -std=c11 -O2 -Wall -Wextra -Werror`.

This establishes bit-exact host parity for the currently implemented attention
state/read and sparse-linear primitives.  It does **not** establish APLX
cross-compilation or board execution.

## H2 stages

- **H2A — PASS:** H/g token delta, normalized read, sparse linear primitives,
  overflow-safe fixed-point division.
- **H2B — next:** integer LayerNorm + event Q/K/V + `stable_a` C kernels and
  bit-exact parity.
- **H2C:** fixed-point Izh RS substeps + online trajectory decoder C parity.
- **H2D:** compose the complete resident C block and compare against the frozen
  H1.1 functional HIR trajectory.
- **H2E:** cross-compile `snn_transform_h2.aplx` in the actual SpiNNaker
  toolchain and run the arithmetic self-test on a board.
- **H2F:** SDRAM HIR loader + changed-token event ingress + full resident board
  execution and end-of-run output/state readback.

## Claim boundary

Current strongest valid statement:

> H2 has entered APLX-oriented implementation.  The first fixed-point C kernels
> are bit-exact on a host compiler and an APLX/SARK self-test source has been
> prepared.

Do not yet state that the full block has been compiled to APLX or executed on a
SpiNNaker board.
