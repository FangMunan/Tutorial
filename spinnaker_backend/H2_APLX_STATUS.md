# H2 — APLX resident backend status

Branch: `snn-transform-spinnaker-h0-h1-20261014`

## Frozen input contract

H2 consumes `H1.1-functional-v9`. H1.1 functional parity is accepted; strict
full-block representation parity remains separately reported as failed at
~2.612% max block-output rel-L2.

H2 is not permitted to change the v1.1 scientific operator to improve this
number.

## Implemented APLX-oriented resident kernels

`spinnaker_backend/h2_aplx/` now contains the fixed-point implementation of the
complete frozen block arithmetic required for resident execution:

- Q20 dense Q/K/V projection;
- Q20 compact sparse-row WO/W1/W2 projection;
- integer LayerNorm with integer mean/variance/sqrt;
- fixed-point `stable_a` random-feature projection;
- Q22 Izhikevich RS two-half-step integration and online decoder;
- initial H/g construction;
- batch-free changed-token subtract-old/add-new H/g update;
- normalized H/g read with overflow-safe scaled integer division;
- residual, ReLU and sparse FFN composition in the H2D host harness.

The directory also contains:

- `snn_transform_h2.c`: first SARK/APLX arithmetic self-test source;
- `Makefile`: `spinnaker_tools` `app.make` build convention;
- `test_h2_kernels.py`: primitive bit-exact host C gate;
- `test_h2_actual_hir.py`: full-step actual frozen-HIR host C gate.

## H2A/H2B/H2C primitive gate — PASS

The strict host C build uses:

```text
gcc -std=c11 -O2 -Wall -Wextra -Werror
```

Verified primitive result:

```text
scaled_div_cases: 4997
layernorm_max_abs_diff: 0
stable_a_max_abs_diff: 0
izh_scalar_features_checked: 771
izh_max_abs_diff: 0
attention_read_max_abs_diff: 0
sparse_test_nnz: 512
contract: H1.1-functional-v9
```

Thus the implemented LayerNorm, stable-a, Izh decoder, H/g read/update and sparse
linear primitives are bit-exact against their independent Python integer
references on the host compiler.

## H2D actual frozen-HIR full-step gate — PASS

GitHub Actions run: `38578938548`.

Source HIR is downloaded directly from the frozen H1.1 functional artifact:

- H1.1 run `38575839809`;
- artifact `SNN_TRANSFORM_SPINNAKER_H1_1_FUNCTIONAL_V9`;
- HIR `SNNTransformBlock_v1_1_HIR_H1_1_FUNCTIONAL_V9.npz`;
- source checkpoint SHA256
  `e86fdbcef1ed32652cc3547642d2890a6b40166dbb0242fda966e4a1fbc2dcc5`.

The H2D harness composes the full resident integer path over six steps:

```text
LN1
 -> event Q/K/V
 -> stable_a
 -> Izh + online decoder
 -> H/g init or changed-token delta update
 -> normalized H/g read
 -> sparse WO
 -> residual
 -> LN2
 -> sparse W1
 -> ReLU
 -> sparse W2
 -> residual
```

Trajectory:

```text
step 0: changed tokens 16, output max-abs diff 0
step 1: changed tokens  4, output max-abs diff 0
step 2: changed tokens  4, output max-abs diff 0
step 3: changed tokens  4, output max-abs diff 0
step 4: changed tokens  4, output max-abs diff 0
step 5: changed tokens  4, output max-abs diff 0
final resident state: bit-exact
```

Therefore:

**H2D PASS: the composed host C execution of the actual frozen H1.1 HIR is
bit-exact to the independent integer reference for the tested persistent/event
trajectory.**

## Remaining H2 stages

- **H2A — PASS:** H/g token delta, normalized read, sparse linear primitives,
  overflow-safe fixed-point division.
- **H2B — PASS:** integer LayerNorm, dense event Q/K/V and `stable_a` C kernels.
- **H2C — PASS:** fixed-point Izh RS substeps + online trajectory decoder.
- **H2D — PASS:** complete host C resident block on the actual frozen H1.1 HIR,
  including persistent changed-token execution.
- **H2E — next:** cross-compile `snn_transform_h2.aplx` with the actual
  SpiNNaker toolchain and execute the arithmetic self-test on a board.
- **H2F:** SDRAM HIR loader + changed-token event ingress + full resident board
  execution and end-of-run output/state readback.

## Claim boundary

Current strongest valid statement:

> H2 has a complete APLX-oriented fixed-point C arithmetic path whose composed
> execution is bit-exact to the frozen H1.1 integer contract on the tested host
> trajectory. A SARK/APLX application source and Makefile are prepared.

This is not yet an APLX cross-compilation result and not a SpiNNaker board
execution result. Those claims require H2E/H2F.
